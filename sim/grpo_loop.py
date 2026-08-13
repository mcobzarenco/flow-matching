"""Token-GRPO synchronous training loop on the molmoact2 discrete
surface (phase-2 instrument item 4, design memo
posts/2026-08-13-token-grpo-phase2-design.md §2/§8; rollout + replay
halves landed as item 3, ``bijou/molmoact2/replay.py``).

One RL step (memo §2, πRL-style synchronous — rollouts and gradient
steps alternate on one GPU):

1. **Rollout wave** — S fresh spawn seeds from the dedicated training
   stream (``train_seed_base + step*S``, disjoint from sim100's 0–99,
   the probe's 0–14 and the held-out 200–219), G grammar-masked SAMPLED
   draws each at the frozen temperature, training rows captured via the
   driver's ``TrainingRowWriter`` (frames + sampled bins + per-token
   chosen logprobs + MODEL-unit state).
2. **Score** — the §3 composite reward per episode (dense
   ``progress_final_cm`` + 10·success − 2·tipped − 5·strike).
3. **Z-filter** — advantages are within-group z-scores (per spawn
   seed, ddof=0); zero-variance groups (std < 0.05 cm, the probe's
   non-degeneracy line) are dropped whole.
4. **Gradient step** — chunked backward over the step's rows through
   ``molmoact2_grpo_sums`` (each chunk's objective sum divided by the
   FULL-batch token count, so chunking never changes the gradient),
   grad-clip 1.0 over the trainable set, non-finite grad norm skips
   the step loudly. Trainable surface = memo §4 option B retargeted:
   the trunk's TEXT stack (embeddings + transformer + lm_head),
   vision frozen.
5. **Telemetry** — k3 KL to the frozen step-0 anchor measured on a row
   subsample via a parameter-swap reference forward (memo §2: recorded
   every step, the hacking early-warning; judged at boundaries, §7),
   plus per-step reward/guard/ratio facts appended to
   ``<out>/train.jsonl`` — babysit's ``train-jsonl`` schema (``step``,
   ``loss``, ``s_per_step``, ``vram_gib``; probe rows carry
   ``eval_reward_mean``).
6. **Held-out eval** — seeds ``eval_seed_base..+count`` greedy
   (grammar-masked argmax, the registered serving mode) at step 0
   (BEFORE any update — the pairing baseline) and every
   ``eval_every`` steps + once at completion; paired Δ composite
   reward vs step 0 with the seeded 10k-bootstrap CI95.

Tripwires (§7) stop the loop — a ``tripwire`` heartbeat row, a
checkpoint, exit code 3 — never silently continue: any reset strike in
training rollouts; non-finite loss; median group std < 0.05 cm for 3
consecutive steps (spread collapse); knock-away rate > 2× the probe's
10/120 baseline for 3 consecutive steps (violence explosion); held-out
paired CI entirely below the −1.0 cm competence floor. The
KL-to-anchor curve is recorded only — its numeric line is set at
finalization from R0's measured scale.

Numerics contract: the loop casts the trunk's text stack to fp32 at
load and BOTH the rollout decode and the replay forward run those same
weights — the ratio π_new/π_old is only meaningful under the same
masked softmax (bf16 rollout vs fp32 replay would eat the 1e-5
reproduction bound; AdamW at lr 5e-6 on bf16 master weights would
round most updates away). Vision stays bf16 (frozen).

The loop core (:func:`run_grpo_loop`) takes the wave/eval functions as
parameters — production wires the sim worker fan-out
(:func:`make_sim_wave_fns`, reusing the parallel driver's lockation
machinery verbatim); the CPU oracles in tests/test_grpo_loop.py drive
the identical loop end-to-end on the tiny-real-trunk fixture with
in-process waves.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from bijou.molmoact2.replay import (
    ReplayRow,
    load_training_rows,
    molmoact2_grpo_sums,
    replay_logprobs,
)
from bijou.train_grpo import GRPOConfig

from . import OUTPUT_DIR
from .rollout_sim import STATS_REPO_ID, TASK, EpisodeResult, resolve_replans
from .rollout_sim_parallel import (
    MOLMOACT2_NORM_TAG,
    MOLMOACT2_OFFICIAL_OFFSETS_DEG,
    MOLMOACT2_OFFICIAL_SIGNS,
    TrainingRowWriter,
    WorkerConfig,
    _worker_main,
    molmoact2_discrete_chunks,
    serve,
)

# Memo §3 frozen candidate constants (cm-commensurate).
SUCCESS_BONUS = 10.0
TIP_PENALTY = 2.0
TIP_UPRIGHT = 0.9
STRIKE_PENALTY = 5.0
# The probe's knock-away read (read_grpo_signal_probe.py): an episode
# that ENDS at least 1 cm worse than its spawn.
KNOCKAWAY_CM = -1.0
# §7 competence floor: held-out paired delta CI entirely below this.
COMPETENCE_FLOOR_CM = -1.0

WaveFn = Callable[[int, list[int]], tuple[list[EpisodeResult], Path]]
EvalFn = Callable[[list[int]], list[EpisodeResult]]


def composite_reward(episode: EpisodeResult) -> float:
    """Memo §3: dense progress base + the SimpleVLA binary success
    grafted on top; tips and hard faults penalized (knock-aways need no
    extra term — they ARE negative progress)."""
    reward = episode.progress_final_cm
    if episode.success_tick is not None:
        reward += SUCCESS_BONUS
    if episode.final_upright < TIP_UPRIGHT:
        reward -= TIP_PENALTY
    if episode.reset_strikes > 0:
        reward -= STRIKE_PENALTY
    return reward


@dataclass(frozen=True, slots=True)
class GroupFacts:
    """Per-wave grouping telemetry (§6 record-only reads)."""

    total: int
    kept: int
    median_std: float
    per_seed_std: dict[int, float]


def group_advantages(
    episodes: list[EpisodeResult],
    *,
    min_std: float,
) -> tuple[dict[tuple[int, int], float], GroupFacts]:
    """((seed, draw) -> advantage for episodes of KEPT groups,
    grouping facts). Advantages are within-group z-scores of the
    composite reward (ddof=0 — the probe's statistic); a group is
    dropped whole when its reward std falls below ``min_std`` OR is
    exactly zero (a z-score needs spread; the dynamic-sampling filter,
    memo §2 step 2). Loud on duplicate (seed, draw) identities."""
    groups: dict[int, list[EpisodeResult]] = {}
    seen: set[tuple[int, int]] = set()
    for episode in episodes:
        key = (episode.seed, episode.draw)
        if key in seen:
            raise ValueError(f"duplicate episode identity (seed, draw) {key}")
        seen.add(key)
        groups.setdefault(episode.seed, []).append(episode)
    advantages: dict[tuple[int, int], float] = {}
    stds: dict[int, float] = {}
    kept = 0
    for seed, members in groups.items():
        rewards = np.array([composite_reward(m) for m in members], dtype=np.float64)
        std = float(rewards.std(ddof=0))
        stds[seed] = std
        if std < min_std or std == 0.0:
            continue
        kept += 1
        mean = float(rewards.mean())
        for member, reward in zip(members, rewards, strict=True):
            advantages[(member.seed, member.draw)] = (float(reward) - mean) / std
    facts = GroupFacts(
        total=len(groups),
        kept=kept,
        median_std=float(np.median(list(stds.values()))),
        per_seed_std=stds,
    )
    return advantages, facts


def apply_option_b_freeze(predictor: Any) -> list[tuple[str, nn.Parameter]]:
    """Memo §4 option B retargeted to the molmoact2 trunk: the TEXT
    stack (embeddings + transformer + lm_head) trains, vision stays
    frozen. Loud if the split degenerates (no trainable params, or
    everything trainable — no vision to freeze would mean the wrong
    module layout)."""
    trunk = predictor.trunk
    trunk.requires_grad_(False)
    trunk.text.requires_grad_(True)
    named = [
        (f"text.{name}", param)
        for name, param in trunk.text.named_parameters()
        if param.requires_grad
    ]
    total = len(list(trunk.parameters()))
    if not named or len(named) == total:
        raise ValueError(
            f"option-B freeze degenerated: {len(named)} trainable of "
            f"{total} trunk params",
        )
    return named


class AnchorSnapshot:
    """CPU clones of the trainable tensors at construction — the
    frozen anchor for the §2 KL telemetry. ``swapped()`` installs the
    anchor values for a reference forward and restores the LIVE
    tensors on exit (restoration is by object identity: bit-exact)."""

    def __init__(self, named: list[tuple[str, nn.Parameter]]) -> None:
        self.named = list(named)
        self.values = [param.detach().to("cpu", copy=True) for _, param in self.named]

    def swapped(self) -> _AnchorSwap:
        return _AnchorSwap(self)


class _AnchorSwap:
    def __init__(self, snapshot: AnchorSnapshot) -> None:
        self.snapshot = snapshot
        self.live: list[Tensor] = []

    def __enter__(self) -> None:
        self.live = [param.data for _, param in self.snapshot.named]
        for (_, param), value in zip(
            self.snapshot.named,
            self.snapshot.values,
            strict=True,
        ):
            param.data = value.to(param.device)

    def __exit__(self, *exc: object) -> None:
        for (_, param), data in zip(self.snapshot.named, self.live, strict=True):
            param.data = data


def anchor_kl(
    predictor: Any,
    anchor: AnchorSnapshot,
    rows: list[ReplayRow],
    *,
    task: str,
    temperature: float,
) -> float:
    """k3 estimate of KL(π_rollout ‖ π_anchor) over the rows' decision
    tokens: the rows' RECORDED chosen logprobs are π_rollout, so the
    measurement costs exactly one reference forward (memo §2) — the
    anchor's, via parameter swap. Reduction-shape noise only when the
    rollout policy IS the anchor; §7's runaway input (recorded every
    step, judged at boundaries)."""
    with anchor.swapped(), torch.no_grad():
        anchor_logprobs, decisions = replay_logprobs(
            predictor,
            rows,
            task=task,
            temperature=temperature,
        )
    rollout_logprobs = torch.zeros_like(anchor_logprobs)
    for index, row in enumerate(rows):
        rollout_logprobs[index, : row.logprobs.shape[0]] = torch.from_numpy(
            row.logprobs,
        )
    log_ratio = anchor_logprobs - rollout_logprobs
    k3 = torch.exp(log_ratio) - 1.0 - log_ratio
    trained = decisions.to(k3.dtype)
    return float((k3 * trained).sum() / trained.sum().clamp(min=1))


@dataclass(frozen=True, slots=True)
class TrainStepFacts:
    loss: float
    tokens: int
    mean_ratio: float
    min_ratio: float
    max_ratio: float
    clip_fraction: float
    approx_kl: float
    grad_norm: float
    skipped: bool


def accumulate_grpo_grads(
    predictor: Any,
    rows: list[ReplayRow],
    advantages: Tensor,
    *,
    task: str,
    config: GRPOConfig,
    microbatch_rows: int,
) -> tuple[float, int, float, float, float, float, float]:
    """Chunked backward accumulation: (loss, tokens, mean_ratio,
    min_ratio, max_ratio, clip_fraction, approx_kl). Each chunk's
    objective sum is divided by the FULL-batch token count before
    ``backward()`` — the accumulated gradient equals the single-batch
    token-weighted mean's (the sum-form discipline; oracle in
    tests/test_grpo_loop.py). Stats aggregate token-weighted."""
    if not rows:
        raise ValueError("accumulate_grpo_grads on an empty row batch")
    full_count = sum(len(row.ids) for row in rows)
    loss_total = 0.0
    tokens = 0
    ratio_sum = 0.0
    kl_sum = 0.0
    clip_sum = 0.0
    min_ratio = math.inf
    max_ratio = -math.inf
    for start in range(0, len(rows), microbatch_rows):
        chunk = rows[start : start + microbatch_rows]
        objective_sum, _, stats = molmoact2_grpo_sums(
            predictor,
            chunk,
            task=task,
            advantages=advantages[start : start + microbatch_rows],
            config=config,
        )
        loss = -objective_sum / full_count
        loss.backward()
        loss_total += float(loss.detach())
        tokens += stats.tokens
        ratio_sum += stats.mean_ratio * stats.tokens
        kl_sum += stats.approx_kl * stats.tokens
        clip_sum += stats.clip_fraction * stats.tokens
        min_ratio = min(min_ratio, stats.min_ratio)
        max_ratio = max(max_ratio, stats.max_ratio)
    return (
        loss_total,
        tokens,
        ratio_sum / max(tokens, 1),
        min_ratio,
        max_ratio,
        clip_sum / max(tokens, 1),
        kl_sum / max(tokens, 1),
    )


def grpo_train_step(
    predictor: Any,
    rows: list[ReplayRow],
    advantages: Tensor,
    *,
    task: str,
    config: GRPOConfig,
    parameters: list[tuple[str, nn.Parameter]],
    optimizer: torch.optim.Optimizer,
    microbatch_rows: int,
    grad_clip: float,
) -> TrainStepFacts:
    """One optimizer step over a wave's retained rows: chunked
    accumulation, grad-clip over the trainable set, non-finite grad
    norm SKIPS the update (loudly, in the returned facts — the §7
    NaN tripwire reads them)."""
    optimizer.zero_grad(set_to_none=True)
    loss, tokens, mean_ratio, min_ratio, max_ratio, clip_fraction, kl = (
        accumulate_grpo_grads(
            predictor,
            rows,
            advantages,
            task=task,
            config=config,
            microbatch_rows=microbatch_rows,
        )
    )
    grad_norm = float(
        torch.nn.utils.clip_grad_norm_(
            [param for _, param in parameters],
            grad_clip,
        ),
    )
    skipped = not math.isfinite(grad_norm)
    if not skipped:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return TrainStepFacts(
        loss=loss,
        tokens=tokens,
        mean_ratio=mean_ratio,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        clip_fraction=clip_fraction,
        approx_kl=kl,
        grad_norm=grad_norm,
        skipped=skipped,
    )


def paired_bootstrap_ci(
    deltas: np.ndarray,
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Seeded percentile CI95 of the mean paired delta — the probe
    read scripts' exact procedure (default_rng(0), 10k resamples)."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), size=(resamples, len(deltas)))
    boot = deltas[indices].mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


@dataclass(frozen=True, slots=True)
class EvalFacts:
    step: int
    per_seed: dict[int, float]
    mean_reward: float
    successes: int
    delta_mean: float | None
    ci_lo: float | None
    ci_hi: float | None

    @property
    def competence_crash(self) -> bool:
        """§7: held-out greedy worse than the −1.0 cm floor with the
        paired CI entirely below it."""
        return self.ci_hi is not None and self.ci_hi < COMPETENCE_FLOOR_CM


def eval_facts(
    step: int,
    episodes: list[EpisodeResult],
    baseline: dict[int, float] | None,
) -> EvalFacts:
    """Score a held-out greedy eval and pair it against the step-0
    baseline (None baseline = this IS the baseline). Pairing is by
    seed and loud on a mismatch — a partial eval must never silently
    shrink the read."""
    per_seed = {e.seed: composite_reward(e) for e in episodes}
    if len(per_seed) != len(episodes):
        raise ValueError("duplicate seeds in a held-out eval wave")
    delta_mean = ci_lo = ci_hi = None
    if baseline is not None:
        if set(baseline) != set(per_seed):
            raise ValueError(
                f"eval seeds {sorted(per_seed)} != baseline seeds "
                f"{sorted(baseline)} — the paired read needs identical sets",
            )
        deltas = np.array(
            [per_seed[s] - baseline[s] for s in sorted(per_seed)],
            dtype=np.float64,
        )
        delta_mean = float(deltas.mean())
        ci_lo, ci_hi = paired_bootstrap_ci(deltas)
    return EvalFacts(
        step=step,
        per_seed=per_seed,
        mean_reward=float(np.mean(list(per_seed.values()))),
        successes=sum(e.success_tick is not None for e in episodes),
        delta_mean=delta_mean,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
    )


@dataclass
class TripwireState:
    collapse_streak: int = 0
    violence_streak: int = 0


def update_tripwires(
    state: TripwireState,
    *,
    strikes: int,
    loss_finite: bool,
    median_group_std: float,
    knockaway_frac: float,
    config: GRPOLoopConfig,
) -> list[str]:
    """§7 wave-level tripwires; mutates the streak state and returns
    the fired descriptions (empty = continue)."""
    fired: list[str] = []
    if strikes > 0:
        fired.append(
            f"reset strike(s) in training rollouts: {strikes} (probe baseline 0/360)",
        )
    if not loss_finite:
        fired.append("non-finite loss")
    if median_group_std < config.min_group_std:
        state.collapse_streak += 1
    else:
        state.collapse_streak = 0
    if state.collapse_streak >= config.tripwire_streak:
        fired.append(
            f"spread collapse: median group std < {config.min_group_std} "
            f"for {state.collapse_streak} consecutive steps",
        )
    if knockaway_frac > 2.0 * config.knockaway_baseline:
        state.violence_streak += 1
    else:
        state.violence_streak = 0
    if state.violence_streak >= config.tripwire_streak:
        fired.append(
            f"violence explosion: knock-away rate > 2x the "
            f"{config.knockaway_baseline:.3f} baseline for "
            f"{state.violence_streak} consecutive steps",
        )
    return fired


@dataclass(frozen=True, slots=True)
class GRPOLoopConfig:
    """Frozen loop constants (memo §2/§5/§6/§7 candidates; the
    finalized pre-reg pins the actual run's values)."""

    out_dir: Path
    total_steps: int
    task: str = TASK
    run_seed: int = 0
    seeds_per_step: int = 8
    draws: int = 8
    temperature: float = 1.0
    train_seed_base: int = 1000
    eval_every: int = 5
    eval_seed_base: int = 200
    eval_seed_count: int = 20
    min_group_std: float = 0.05
    knockaway_baseline: float = 10.0 / 120.0
    tripwire_streak: int = 3
    lr: float = 5e-6
    microbatch_rows: int = 1
    grad_clip: float = 1.0
    kl_rows: int = 32
    save_every: int = 5
    keep_checkpoints: int = 2


@dataclass(slots=True)
class LoopResult:
    steps_done: int
    stopped_reason: str | None
    baseline: dict[int, float] | None
    last_eval: EvalFacts | None


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def save_checkpoint(
    config: GRPOLoopConfig,
    step: int,
    named: list[tuple[str, nn.Parameter]],
    optimizer: torch.optim.Optimizer,
    baseline: dict[int, float] | None,
) -> Path:
    """Trainable tensors + optimizer + pairing baseline as
    ``step_NNNN.pt``; prunes to the newest ``keep_checkpoints`` (the
    text stack is GBs per save — memo §5's budget prices GPU-h, not
    disk, but 15 steps of full saves would be ~50 GB)."""
    path = config.out_dir / f"step_{step:04d}.pt"
    torch.save(
        {
            "step": step,
            "trainable": {name: param.detach().cpu() for name, param in named},
            "optimizer": optimizer.state_dict(),
            "baseline": baseline,
        },
        path,
    )
    saved = sorted(config.out_dir.glob("step_*.pt"))
    for stale in saved[: -config.keep_checkpoints]:
        stale.unlink()
    return path


def load_checkpoint(
    path: Path,
    named: list[tuple[str, nn.Parameter]],
    optimizer: torch.optim.Optimizer,
) -> tuple[int, dict[int, float] | None]:
    """(step, baseline) — restores trainable tensors + optimizer in
    place; loud on a name-set mismatch (a resume must be the same
    trainable surface)."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    stored: dict[str, Tensor] = payload["trainable"]
    if set(stored) != {name for name, _ in named}:
        raise ValueError(
            f"checkpoint {path} trainable set does not match the live "
            "surface — resume must use the same option",
        )
    with torch.no_grad():
        for name, param in named:
            param.copy_(stored[name].to(param.device, param.dtype))
    optimizer.load_state_dict(payload["optimizer"])
    baseline = payload["baseline"]
    return int(payload["step"]), (
        {int(k): float(v) for k, v in baseline.items()} if baseline else None
    )


def run_grpo_loop(
    predictor: Any,
    config: GRPOLoopConfig,
    *,
    wave_fn: WaveFn,
    eval_fn: EvalFn,
    parameters: list[tuple[str, nn.Parameter]] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    start_step: int = 0,
    baseline: dict[int, float] | None = None,
) -> LoopResult:
    """The synchronous loop (module docstring). ``wave_fn(step,
    seeds) -> (episodes, rows_dir)`` runs one sampled training wave
    with row capture; ``eval_fn(seeds) -> episodes`` one greedy
    held-out wave. Injected so the CPU oracles drive the identical
    loop; production wires :func:`make_sim_wave_fns`."""
    config.out_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = config.out_dir / "train.jsonl"
    grpo = GRPOConfig(temperature=config.temperature)
    named = parameters if parameters is not None else apply_option_b_freeze(predictor)
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            [param for _, param in named],
            lr=config.lr,
            betas=(0.9, 0.95),
            eps=1e-6,
            weight_decay=0.0,
        )
    anchor = AnchorSnapshot(named)
    tripwires = TripwireState()
    eval_seeds = list(
        range(config.eval_seed_base, config.eval_seed_base + config.eval_seed_count),
    )
    last_eval: EvalFacts | None = None
    stopped: str | None = None
    completed = start_step

    def run_eval(at_step: int) -> str | None:
        nonlocal baseline, last_eval
        episodes = eval_fn(eval_seeds)
        facts = eval_facts(at_step, episodes, baseline)
        last_eval = facts
        append_jsonl(
            heartbeat,
            {
                "step": at_step,
                "eval_reward_mean": round(facts.mean_reward, 4),
                "eval_successes": facts.successes,
                "eval_delta_mean": (
                    None if facts.delta_mean is None else round(facts.delta_mean, 4)
                ),
                "eval_delta_ci_lo": (
                    None if facts.ci_lo is None else round(facts.ci_lo, 4)
                ),
                "eval_delta_ci_hi": (
                    None if facts.ci_hi is None else round(facts.ci_hi, 4)
                ),
            },
        )
        if baseline is None:
            baseline = facts.per_seed
        if facts.competence_crash:
            return (
                f"competence crash: held-out paired CI "
                f"[{facts.ci_lo}, {facts.ci_hi}] entirely below "
                f"the {COMPETENCE_FLOOR_CM} cm floor"
            )
        return None

    for step in range(start_step, config.total_steps):
        if step % config.eval_every == 0:
            stopped = run_eval(step)
            if stopped:
                break
        started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        seeds = [
            config.train_seed_base + step * config.seeds_per_step + i
            for i in range(config.seeds_per_step)
        ]
        episodes, rows_root = wave_fn(step, seeds)
        rollout_s = time.perf_counter() - started
        strikes = sum(e.reset_strikes for e in episodes)
        knockaway_frac = float(
            np.mean([e.progress_final_cm <= KNOCKAWAY_CM for e in episodes]),
        )
        rewards = [composite_reward(e) for e in episodes]
        advantages_map, groups = group_advantages(
            episodes,
            min_std=config.min_group_std,
        )

        facts: TrainStepFacts | None = None
        kl_anchor = None
        chosen_nll = None
        if advantages_map and strikes == 0:
            _, rows = load_training_rows(rows_root)
            rows = [r for r in rows if (r.seed, r.draw) in advantages_map]
            rows.sort(key=lambda r: (r.seed, r.draw, r.replan))
            advantages = torch.tensor(
                [advantages_map[(r.seed, r.draw)] for r in rows],
                dtype=torch.float32,
            )
            chosen_nll = float(
                np.mean([-r.logprobs.mean() for r in rows]),
            )
            facts = grpo_train_step(
                predictor,
                rows,
                advantages,
                task=config.task,
                config=grpo,
                parameters=named,
                optimizer=optimizer,
                microbatch_rows=config.microbatch_rows,
                grad_clip=config.grad_clip,
            )
            kl_anchor = anchor_kl(
                predictor,
                anchor,
                rows[: config.kl_rows],
                task=config.task,
                temperature=config.temperature,
            )
        vram = (
            round(torch.cuda.max_memory_allocated() / 2**30, 2)
            if torch.cuda.is_available()
            else None
        )
        step_row: dict[str, Any] = {
            "step": step + 1,
            "loss": None if facts is None else round(facts.loss, 6),
            "s_per_step": round(time.perf_counter() - started, 1),
            "rollout_s": round(rollout_s, 1),
            "vram_gib": vram,
            "episodes": len(episodes),
            "reward_mean": round(float(np.mean(rewards)), 4),
            "median_group_std": round(groups.median_std, 4),
            "groups_kept": groups.kept,
            "groups_total": groups.total,
            "knockaway_frac": round(knockaway_frac, 4),
            "strikes": strikes,
            "successes": sum(e.success_tick is not None for e in episodes),
            "chosen_nll": None if chosen_nll is None else round(chosen_nll, 4),
            "tokens": None if facts is None else facts.tokens,
            "mean_ratio": None if facts is None else round(facts.mean_ratio, 5),
            "clip_fraction": None if facts is None else round(facts.clip_fraction, 5),
            "approx_kl": None if facts is None else round(facts.approx_kl, 8),
            "anchor_kl": None if kl_anchor is None else round(kl_anchor, 8),
            "grad_norm": None if facts is None else round(facts.grad_norm, 4),
            "step_skipped": None if facts is None else facts.skipped,
        }
        append_jsonl(heartbeat, step_row)
        fired = update_tripwires(
            tripwires,
            strikes=strikes,
            loss_finite=facts is None or math.isfinite(facts.loss),
            median_group_std=groups.median_std,
            knockaway_frac=knockaway_frac,
            config=config,
        )
        if fired:
            stopped = "; ".join(fired)
            append_jsonl(heartbeat, {"step": step + 1, "tripwire": fired})
            break
        completed = step + 1
        shutil.rmtree(rows_root, ignore_errors=True)
        if completed % config.save_every == 0:
            save_checkpoint(config, completed, named, optimizer, baseline)
    if stopped is None and (last_eval is None or last_eval.step != config.total_steps):
        # The endpoint read — the §6 primary is endpoint vs step 0.
        stopped = run_eval(config.total_steps)

    save_checkpoint(config, completed, named, optimizer, baseline)
    return LoopResult(
        steps_done=completed,
        stopped_reason=stopped,
        baseline=baseline,
        last_eval=last_eval,
    )


# ---------------------------------------------------------------- GPU wiring


def run_units(
    predictor: Any,
    shim: Any,
    units: list[tuple[int, int]],
    *,
    workers: int,
    replans: int,
    horizon: int,
    task: str,
    post_backend: str,
    temperature: float | None,
    run_seed: int,
    writer: TrainingRowWriter | None,
) -> list[EpisodeResult]:
    """One worker-wave over (seed, draw) units — the parallel driver's
    lockstep machinery verbatim (spawned sim workers, batched
    parent-side predicts through ``molmoact2_discrete_chunks``), with
    the driver's RNG keying (``stable_sample_rng(run_seed,
    repo_id(draw), seed, replan, 0)``). ``temperature=None`` decodes
    greedy under the grammar mask (the held-out eval mode);
    ``writer`` captures training rows (sampled waves only)."""
    from bijou.eval.policies import stable_sample_rng

    def rng_for(seed: int, replan: int, draw: int) -> Any:
        repo = "sim/eval100" if draw == 0 else f"sim/eval100/draw{draw:02d}"
        return stable_sample_rng(run_seed, repo, seed, replan, 0)

    def predict_batch(requests: list[tuple[Any, ...]]) -> list[np.ndarray]:
        token_rows: list[Any] | None = [] if writer is not None else None
        states: list[np.ndarray] | None = [] if writer is not None else None
        chunks, _ = molmoact2_discrete_chunks(
            predictor,
            shim,
            requests,
            task=task,
            grammar_masked=True,
            temperature=temperature,
            rng_for=rng_for if temperature is not None else None,
            token_rows=token_rows,
            model_states=states,
        )
        if writer is not None:
            assert token_rows is not None and states is not None
            for message, row, model_state in zip(
                requests,
                token_rows,
                states,
                strict=True,
            ):
                _, _, seed, replan, draw, top, wrist, _ = message
                writer.write(
                    seed=seed,
                    replan=replan,
                    draw=draw,
                    top=top,
                    wrist=wrist,
                    state=model_state,
                    row=row,
                )
        return chunks

    count = max(1, min(workers, len(units)))
    context = mp.get_context("spawn")
    processes: list[Any] = []
    conns: list[Any] = []
    for worker_id in range(count):
        parent_conn, child_conn = context.Pipe()
        worker = WorkerConfig(
            worker_id=worker_id,
            units=tuple(units[worker_id::count]),
            replans=replans,
            horizon=horizon,
            hold=False,
            out_dir=None,
            post_backend=post_backend,
        )
        process = context.Process(
            target=_worker_main,
            args=(worker, child_conn),
            daemon=True,
        )
        process.start()
        child_conn.close()
        processes.append(process)
        conns.append(parent_conn)
    episodes: list[EpisodeResult] = []
    try:
        serve(conns, predict_batch, episodes.append)
    finally:
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
    episodes.sort(key=lambda e: (e.seed, e.draw))
    return episodes


def make_sim_wave_fns(
    predictor: Any,
    shim: Any,
    config: GRPOLoopConfig,
    *,
    workers: int,
    replans: int,
    horizon: int,
    post_backend: str,
    checkpoint: str,
    commit: str,
) -> tuple[WaveFn, EvalFn]:
    """(training wave, held-out eval wave) over :func:`run_units`."""

    def wave(step: int, seeds: list[int]) -> tuple[list[EpisodeResult], Path]:
        rows_dir = config.out_dir / "rows" / f"step_{step:04d}"
        writer = TrainingRowWriter(
            rows_dir,
            {
                "checkpoint": checkpoint,
                "run_seed": config.run_seed,
                "decode": "molmoact2_grammar_masked",
                "temperature": config.temperature,
                "task": config.task,
                "state_units": "model (official shim applied)",
                "norm_tag": MOLMOACT2_NORM_TAG,
                "stats_repo_id": STATS_REPO_ID,
                "commit": commit,
                "step": step,
                "rng_key": (
                    "stable_sample_rng(run_seed, repo_id(draw), seed, replan, 0)"
                ),
            },
        )
        units = [(seed, draw) for seed in seeds for draw in range(config.draws)]
        episodes = run_units(
            predictor,
            shim,
            units,
            workers=workers,
            replans=replans,
            horizon=horizon,
            task=config.task,
            post_backend=post_backend,
            temperature=config.temperature,
            run_seed=config.run_seed,
            writer=writer,
        )
        return episodes, rows_dir

    def held_out(seeds: list[int]) -> list[EpisodeResult]:
        return run_units(
            predictor,
            shim,
            [(seed, 0) for seed in seeds],
            workers=workers,
            replans=replans,
            horizon=horizon,
            task=config.task,
            post_backend=post_backend,
            temperature=None,
            run_seed=config.run_seed,
            writer=None,
        )

    return wave, held_out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="HF layout dir or hub id")
    parser.add_argument(
        "--fast-tokenizer",
        default="allenai/MolmoAct2-FAST-Tokenizer",
    )
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR / "grpo_loop")
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--seeds-per-step", type=int, default=8)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--episode-seconds", type=float, default=30.0)
    parser.add_argument("--execute-horizon", type=int, default=30)
    parser.add_argument("--post-backend", default="auto")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--microbatch-rows", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--kl-rows", type=int, default=32)
    parser.add_argument("--run-seed", type=int, default=0)
    parser.add_argument("--train-seed-base", type=int, default=1000)
    parser.add_argument("--eval-seed-base", type=int, default=200)
    parser.add_argument("--eval-seed-count", type=int, default=20)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="step_NNNN.pt from a prior run of the SAME config",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import torch as _torch  # noqa: F401 — parity with the driver's parent-only import

    from bijou.eval.molmo_norm import AffineMap
    from bijou.molmoact2 import MolmoAct2Predictor

    args = parse_args(argv)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    fast_source = args.fast_tokenizer
    if not Path(fast_source).exists():
        from huggingface_hub import snapshot_download

        fast_source = snapshot_download(fast_source)
    predictor = MolmoAct2Predictor.load(
        args.checkpoint,
        MOLMOACT2_NORM_TAG,
        device=device,
        dtype=torch.bfloat16,
        fast_tokenizer=fast_source,
    )
    # Training dtype (module docstring): the text stack — the option-B
    # trainable surface — runs fp32 for BOTH rollout and replay; vision
    # stays frozen bf16.
    predictor.trunk.text.float()
    shim = AffineMap(
        scale=torch.tensor(MOLMOACT2_OFFICIAL_SIGNS),
        offset=torch.tensor(MOLMOACT2_OFFICIAL_OFFSETS_DEG),
    )
    tag_horizon = int(predictor.metadata.get("action_horizon") or 0)
    horizon = min(args.execute_horizon, tag_horizon or args.execute_horizon)
    replans = resolve_replans(None, args.episode_seconds, horizon)

    config = GRPOLoopConfig(
        out_dir=args.out_dir,
        total_steps=args.total_steps,
        run_seed=args.run_seed,
        seeds_per_step=args.seeds_per_step,
        draws=args.draws,
        temperature=args.temperature,
        train_seed_base=args.train_seed_base,
        eval_every=args.eval_every,
        eval_seed_base=args.eval_seed_base,
        eval_seed_count=args.eval_seed_count,
        lr=args.lr,
        microbatch_rows=args.microbatch_rows,
        kl_rows=args.kl_rows,
        save_every=args.save_every,
    )
    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "meta.json").write_text(
        json.dumps(
            {
                **{
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in asdict(config).items()
                },
                "checkpoint": str(args.checkpoint),
                "workers": args.workers,
                "replans": replans,
                "horizon": horizon,
                "commit": commit,
            },
            indent=1,
        ),
    )
    print(
        f"grpo loop: {args.total_steps} steps x {args.seeds_per_step} seeds "
        f"x {args.draws} draws at T={args.temperature}; {replans} replans "
        f"x {horizon} ticks; heartbeat {config.out_dir / 'train.jsonl'}",
        flush=True,
    )
    named = apply_option_b_freeze(predictor)
    optimizer = torch.optim.AdamW(
        [param for _, param in named],
        lr=config.lr,
        betas=(0.9, 0.95),
        eps=1e-6,
        weight_decay=0.0,
    )
    start_step = 0
    baseline: dict[int, float] | None = None
    if args.resume is not None:
        start_step, baseline = load_checkpoint(args.resume, named, optimizer)
        print(f"resumed {args.resume} at step {start_step}", flush=True)

    wave_fn, eval_fn = make_sim_wave_fns(
        predictor,
        shim,
        config,
        workers=args.workers,
        replans=replans,
        horizon=horizon,
        post_backend=args.post_backend,
        checkpoint=str(args.checkpoint),
        commit=commit,
    )
    result = run_grpo_loop(
        predictor,
        config,
        wave_fn=wave_fn,
        eval_fn=eval_fn,
        parameters=named,
        optimizer=optimizer,
        start_step=start_step,
        baseline=baseline,
    )
    if result.stopped_reason is not None:
        print(f"TRIPWIRE STOP at step {result.steps_done}: {result.stopped_reason}")
        return 3
    print(f"complete: {result.steps_done} steps")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
