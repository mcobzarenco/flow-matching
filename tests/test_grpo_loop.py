"""GRPO loop-harness oracles (phase-2 instrument item 4), pure CPU on
the tiny-real-trunk fixture — the SAME loop production runs, driven by
in-process wave/eval functions.

What these pin:

1. the §3 composite reward constants and the §2 z-filter (ddof=0
   z-scores, zero-variance groups dropped whole, duplicate identities
   loud);
2. the option-B freeze splits the trunk (text trains, vision frozen)
   and the anchor snapshot swaps in/out with bit-exact restoration;
3. the chunked backward accumulates EXACTLY the single-batch
   token-weighted-mean gradient (the sum-form discipline), and zero
   advantage yields exactly-zero gradients;
4. anchor KL is exactly 0 at an unchanged policy and positive after a
   perturbation;
5. the paired-eval arithmetic (seeded bootstrap CI, competence-crash
   line, loud seed-set mismatch) and the §7 tripwire streaks;
6. the loop end to end: heartbeat rows in babysit's train-jsonl
   schema, baseline pairing at step 0 + the endpoint read, parameters
   actually move, rows pruned after the gradient pass, checkpoints
   saved/restored;
7. tripwires stop the loop (strike immediately; collapse and violence
   after the registered streak) with rows kept for diagnosis.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from bijou.eval.policies import stable_sample_rng, token_rows_from_capture
from bijou.grpo_replay import (
    MolmoAct2DiscreteStack,
    ReplayRow,
    molmoact2_grpo_loss,
)
from bijou.modelling.interface import ActionCaptureStep
from bijou.testing import write_tiny_molmoact2_release
from bijou.train_grpo import GRPOConfig, grpo_objective_sums

ACTION_TOKEN_START = 151_934  # the release block base (fixture-mirrored)
from sim.grpo_loop import (
    AnchorSnapshot,
    GRPOLoopConfig,
    TripwireState,
    WaveFn,
    accumulate_grpo_grads,
    anchor_chunk_logprobs,
    anchor_kl,
    apply_option_a_freeze,
    apply_option_b_freeze,
    build_optimizer,
    composite_reward,
    composite_reward_v2,
    eval_facts,
    group_advantages,
    grpo_train_step,
    load_checkpoint,
    mixed_group_fraction,
    option_a_row_span,
    paired_bootstrap_ci,
    parse_args,
    run_grpo_loop,
    save_checkpoint,
    update_tripwires,
)
from sim.rollout_sim import EpisodeResult
from sim.rollout_sim_parallel import TrainingRowWriter

TASK = "Pick up the cube."


def _observation() -> dict[str, Any]:
    """One deterministic observation at the tiny fixture's geometry
    (6-dim state, two [H, W, 3] uint8 frames)."""
    generator = np.random.default_rng(7)
    return {
        "images": [
            generator.integers(0, 256, size=(48, 64, 3), dtype=np.uint8),
            generator.integers(0, 256, size=(48, 64, 3), dtype=np.uint8),
        ],
        "task": TASK,
        "state": torch.tensor([0.5, -0.25, 1.75, 10.0, -4.0, 2.0]),
    }


@pytest.fixture(scope="module")
def subject(tmp_path_factory: pytest.TempPathFactory) -> MolmoAct2DiscreteStack:
    """The suite's subject: the discrete stack over a tiny
    release-class bijou checkpoint, loaded through the family registry
    (the builder's VLA-format conversion; fp32 CPU)."""
    root = tmp_path_factory.mktemp("grpo-loop") / "tiny"
    write_tiny_molmoact2_release(root)
    return MolmoAct2DiscreteStack.load(
        root / "checkpoint_vla",
        device="cpu",
        dtype=torch.float32,
        fast_tokenizer=str(
            Path(__file__).parent / "fixtures" / "molmoact2_fast_tokenizer",
        ),
    )


def episode(
    seed: int,
    draw: int,
    *,
    progress: float = 0.0,
    success: bool = False,
    upright: float = 1.0,
    strikes: int = 0,
) -> EpisodeResult:
    return EpisodeResult(
        seed=seed,
        initial_cm=10.0,
        min_cm=10.0 - max(progress, 0.0),
        final_cm=10.0 - progress,
        success_tick=100 if success else None,
        spawn_xy=(0.0, 0.0),
        reset_strikes=strikes,
        final_z_mm=0.0,
        final_upright=upright,
        ticks=900,
        draw=draw,
    )


def rng(seed: int, draw: int) -> np.random.Generator:
    repo = "sim/eval100" if draw == 0 else f"sim/eval100/draw{draw:02d}"
    return stable_sample_rng(0, repo, seed, 0, 0)


def sampled_row(subject: MolmoAct2DiscreteStack, seed: int, draw: int) -> Any:
    """One grammar-masked sampled decode at T=1.0 -> TokenRow."""
    obs = _observation()
    capture: list[ActionCaptureStep] = []
    subject.predict_action_discrete(
        images=obs["images"],
        task=obs["task"],
        state=obs["state"],
        grammar_masked=True,
        temperature=1.0,
        sample_rng=rng(seed, draw),
        action_capture=capture,
    )
    (row,) = token_rows_from_capture(
        capture,
        block_base=ACTION_TOKEN_START,
        temperature=1.0,
    )
    return row


def memory_row(subject: MolmoAct2DiscreteStack, seed: int, draw: int) -> ReplayRow:
    obs = _observation()
    row = sampled_row(subject, seed, draw)
    return ReplayRow(
        top=obs["images"][0],
        wrist=obs["images"][1],
        state=obs["state"],
        ids=row.ids,
        logprobs=row.logprobs,
        allowed_packed=row.allowed_packed,
        vocab_total=row.vocab_total,
        temperature=row.temperature,
        seed=seed,
        draw=draw,
        replan=0,
    )


def restore(snapshot: AnchorSnapshot) -> None:
    with torch.no_grad():
        for (_, param), value in zip(snapshot.named, snapshot.values, strict=True):
            param.copy_(value.to(param.device, param.dtype))


# ------------------------------------------------------------ pure arithmetic


def test_composite_reward_cases() -> None:
    assert composite_reward(episode(0, 0, progress=2.5)) == 2.5
    assert composite_reward(episode(0, 0, progress=1.0, success=True)) == 11.0
    assert composite_reward(episode(0, 0, progress=1.0, upright=0.5)) == -1.0
    assert composite_reward(episode(0, 0, progress=0.0, strikes=1)) == -5.0
    assert (
        composite_reward(episode(0, 0, progress=-2.0, upright=0.5, strikes=2)) == -9.0
    )


def test_composite_reward_v2_pays_only_grasped_progress() -> None:
    # 4 cm endpoint progress: 3 earned under pinch, 2 shoved closer +
    # 1 knocked away ungrasped -> v1 pays 4.0, v2 pays 3 - 0.5*3 = 1.5
    row = episode(0, 0, progress=4.0)
    row = replace(
        row,
        distance_cm=[10.0, 8.0, 5.0, 6.0],
        grip=[0, 1, 3, 0],
    )
    assert composite_reward(row) == 4.0
    assert composite_reward_v2(row) == pytest.approx(1.5)
    # pure shove to the SAME endpoint: v1 cannot tell them apart, v2
    # makes it strictly unprofitable
    shove = replace(row, distance_cm=[10.0, 8.0, 6.0, 6.0], grip=[0, 0, 0, 0])
    assert composite_reward(shove) == 4.0
    assert composite_reward_v2(shove) == pytest.approx(-2.0)
    # bonuses/penalties unchanged
    win = replace(
        episode(0, 0, progress=4.0, success=True),
        distance_cm=[10.0, 6.0],
        grip=[0, 3],
    )
    assert composite_reward_v2(win) == pytest.approx(4.0 + 10.0)
    # pre-instrument rows refuse loudly instead of silently reverting
    with pytest.raises(ValueError, match="grip trace"):
        composite_reward_v2(episode(0, 0, progress=1.0))


def test_group_advantages_takes_reward_fn() -> None:
    # same endpoint, different mechanism: under v2 the earned episode
    # must carry the positive advantage
    earned = replace(
        episode(5, 0, progress=4.0),
        distance_cm=[10.0, 6.0],
        grip=[0, 3],
    )
    shoved = replace(
        episode(5, 1, progress=4.0),
        distance_cm=[10.0, 6.0],
        grip=[0, 0],
    )
    advantages, facts = group_advantages(
        [earned, shoved],
        min_std=0.05,
        reward_fn=composite_reward_v2,
    )
    assert facts.kept == 1
    assert advantages[(5, 0)] > 0 > advantages[(5, 1)]
    # v1 sees identical rewards -> zero spread -> group dropped
    advantages_v1, facts_v1 = group_advantages([earned, shoved], min_std=0.05)
    assert facts_v1.kept == 0 and not advantages_v1


def test_group_advantages_zscores_and_dead_group_drop() -> None:
    episodes = [
        episode(1, 0, progress=0.0),
        episode(1, 1, progress=1.0),
        episode(2, 0, progress=3.0),
        episode(2, 1, progress=3.0),  # zero variance -> dropped whole
    ]
    advantages, facts = group_advantages(episodes, min_std=0.05)
    assert facts.total == 2 and facts.kept == 1
    assert facts.per_seed_std == {1: 0.5, 2: 0.0}
    assert facts.median_std == 0.25
    assert set(advantages) == {(1, 0), (1, 1)}
    assert advantages[(1, 0)] == -1.0 and advantages[(1, 1)] == 1.0
    below_bar = [episode(3, 0, progress=0.0), episode(3, 1, progress=0.04)]
    none_kept, bar_facts = group_advantages(below_bar, min_std=0.05)
    assert none_kept == {} and bar_facts.kept == 0
    with pytest.raises(ValueError, match="duplicate"):
        group_advantages([episode(1, 0), episode(1, 0)], min_std=0.05)


def test_group_advantages_clip_tempers_outliers() -> None:
    """R0-A lever: a lone big reward in a group of 8 puts z ≈ +2.65 on
    one row (R0's collapse driver); clip=2.0 clamps it and leaves
    sub-threshold z-scores untouched."""
    episodes = [episode(1, draw, progress=0.0) for draw in range(7)]
    episodes.append(episode(1, 7, progress=10.0))
    raw, _ = group_advantages(episodes, min_std=0.05)
    assert raw[(1, 7)] == pytest.approx(2.6458, abs=1e-3)
    clipped, _ = group_advantages(episodes, min_std=0.05, clip=2.0)
    assert clipped[(1, 7)] == 2.0
    assert clipped[(1, 0)] == pytest.approx(raw[(1, 0)])
    assert abs(clipped[(1, 0)]) < 2.0
    with pytest.raises(ValueError, match="clip"):
        group_advantages(episodes, min_std=0.05, clip=0.0)


def test_paired_bootstrap_and_eval_facts() -> None:
    lo, hi = paired_bootstrap_ci(np.full(20, 1.0))
    assert lo == hi == 1.0
    baseline = dict.fromkeys(range(200, 210), 2.0)
    flat = eval_facts(0, [episode(s, 0, progress=2.0) for s in range(200, 210)], None)
    assert flat.delta_mean is None and not flat.competence_crash
    assert flat.per_seed == baseline
    crashed = eval_facts(
        5,
        [episode(s, 0, progress=0.0) for s in range(200, 210)],
        baseline,
    )
    assert crashed.delta_mean == -2.0
    assert crashed.ci_lo == crashed.ci_hi == -2.0
    assert crashed.competence_crash
    improved = eval_facts(
        5,
        [episode(s, 0, progress=3.0) for s in range(200, 210)],
        baseline,
    )
    assert improved.delta_mean == 1.0 and not improved.competence_crash
    with pytest.raises(ValueError, match="identical sets"):
        eval_facts(5, [episode(0, 0)], baseline)


def test_update_tripwires_streaks() -> None:
    config = GRPOLoopConfig(out_dir=Path("unused"), total_steps=1)

    def wires(
        state: TripwireState,
        *,
        strikes: int = 0,
        loss_finite: bool = True,
        median_group_std: float = 0.5,
        knockaway_frac: float = 0.0,
    ) -> list[str]:
        return update_tripwires(
            state,
            strikes=strikes,
            loss_finite=loss_finite,
            median_group_std=median_group_std,
            knockaway_frac=knockaway_frac,
            config=config,
        )

    state = TripwireState()
    assert wires(state) == []
    fired = wires(state, strikes=2)
    assert fired and "strike" in fired[0]
    fired = wires(state, loss_finite=False)
    assert fired and "non-finite" in fired[0]
    state = TripwireState()
    for expected in (0, 0, 1):
        fired = wires(state, median_group_std=0.01)
        assert len(fired) == expected or (expected and "collapse" in fired[0])
    assert wires(state) == []
    assert state.collapse_streak == 0
    state = TripwireState()
    for _ in range(2):
        assert wires(state, knockaway_frac=0.5) == []
    fired = wires(state, knockaway_frac=0.5)
    assert fired and "violence" in fired[0]


def test_update_tripwires_wave0_self_baseline() -> None:
    """A3.4's ``--knockaway-baseline wave0``: config ``None`` captures
    the FIRST wave's measured rate as the violence-wire baseline (that
    wave itself exempt), and the 2x line runs against the capture, not
    the R0-era 10/120 default."""
    config = GRPOLoopConfig(
        out_dir=Path("unused"),
        total_steps=1,
        knockaway_baseline=None,
    )

    def wires(state: TripwireState, frac: float) -> list[str]:
        return update_tripwires(
            state,
            strikes=0,
            loss_finite=True,
            median_group_std=0.5,
            knockaway_frac=frac,
            config=config,
        )

    state = TripwireState()
    # Capture wave: 0.25 (>> 2x the retired 10/120 default) must NOT
    # start a streak — it IS the baseline.
    assert wires(state, 0.25) == []
    assert state.knockaway_baseline == 0.25
    # 0.3 < 2x 0.25: quiet against the capture (would have streaked
    # against the old default 0.167).
    for _ in range(3):
        assert wires(state, 0.3) == []
    # > 2x the capture streaks and fires with the captured value named.
    for _ in range(2):
        assert wires(state, 0.6) == []
    fired = wires(state, 0.6)
    assert fired and "violence" in fired[0] and "0.250" in fired[0]
    # Other wires still fire ON the capture wave (early return keeps
    # only the violence check exempt).
    fresh = TripwireState()
    fired = update_tripwires(
        fresh,
        strikes=1,
        loss_finite=True,
        median_group_std=0.5,
        knockaway_frac=0.1,
        config=config,
    )
    assert fired and "strike" in fired[0]
    assert fresh.knockaway_baseline == 0.1


def test_mixed_group_fraction() -> None:
    """The A3.3 calibration read: a group is mixed only when it carries
    BOTH outcomes — all-success and all-failure groups are equally
    uninformative for the ±10 contrast."""
    mixed = [episode(1, 0), episode(1, 1, progress=1.0, success=True)]
    all_fail = [episode(2, 0), episode(2, 1, progress=1.0)]
    all_win = [
        episode(3, 0, success=True),
        episode(3, 1, progress=1.0, success=True),
    ]
    assert mixed_group_fraction(mixed) == 1.0
    assert mixed_group_fraction(all_fail) == 0.0
    assert mixed_group_fraction(all_win) == 0.0
    assert mixed_group_fraction(mixed + all_fail + all_win) == pytest.approx(1 / 3)


def test_grpo_objective_kl_penalty_math() -> None:
    """The differentiable anchor penalty, pinned analytically: k3 per
    token is exp(a−n) − (a−n) − 1 (0 with zero gradient at a == n),
    the objective subtracts β·k3_sum, and the loss gradient w.r.t. the
    live logprobs is −β·(exp(a−n) − 1). Loud when the penalty is on
    without anchors, on shape mismatch, and on non-finite anchors."""
    config = GRPOConfig()
    new = torch.tensor([[-1.0, -2.0, -0.5]], requires_grad=True)
    old = new.detach().clone()
    advantages = torch.zeros(1)  # zero advantage isolates the penalty
    decisions = torch.tensor([[True, True, False]])
    anchor = torch.tensor([[-1.5, -2.0, 7.7]])  # pad position ignored
    beta = 0.5
    objective, count, stats = grpo_objective_sums(
        new,
        old,
        advantages,
        decisions,
        config,
        anchor_logprobs=anchor,
        kl_beta=beta,
    )
    delta = (anchor - new.detach())[decisions]
    k3 = (delta.exp() - delta - 1.0).sum()
    assert float(objective.detach()) == pytest.approx(-beta * float(k3), rel=1e-6)
    assert stats.anchor_k3 == pytest.approx(float(k3) / 2, rel=1e-6)
    (-objective / count).backward()
    grad = new.grad
    assert grad is not None
    expected = -(beta / 2) * ((anchor - old).exp() - 1.0)
    assert torch.allclose(grad[decisions], expected[decisions], atol=1e-6)
    assert float(grad[~decisions].abs().max()) == 0.0
    # a == n: penalty exactly 0 with exactly-zero gradient.
    fresh = torch.tensor([[-1.0, -2.0, -0.5]], requires_grad=True)
    objective, count, stats = grpo_objective_sums(
        fresh,
        old,
        advantages,
        decisions,
        config,
        anchor_logprobs=old.clone(),
        kl_beta=beta,
    )
    assert float(objective.detach()) == 0.0 and stats.anchor_k3 == 0.0
    (-objective / count).backward()
    assert fresh.grad is not None and float(fresh.grad.abs().max()) == 0.0
    with pytest.raises(ValueError, match="needs anchor_logprobs"):
        grpo_objective_sums(new, old, advantages, decisions, config, kl_beta=beta)
    with pytest.raises(ValueError, match="must match"):
        grpo_objective_sums(
            new,
            old,
            advantages,
            decisions,
            config,
            anchor_logprobs=anchor[:, :2],
            kl_beta=beta,
        )
    with pytest.raises(ValueError, match="non-finite anchor"):
        grpo_objective_sums(
            new,
            old,
            advantages,
            decisions,
            config,
            anchor_logprobs=torch.tensor([[-1.5, -math.inf, 0.0]]),
            kl_beta=beta,
        )


# ------------------------------------------------------ surface + step oracle


def test_option_b_freeze_and_anchor_swap(subject: MolmoAct2DiscreteStack) -> None:
    named = apply_option_b_freeze(subject)
    assert named and all(name.startswith("text.") for name, _ in named)
    assert all(not p.requires_grad for p in subject.trunk.vision.parameters())
    assert all(p.requires_grad for p in subject.trunk.text.parameters())
    anchor = AnchorSnapshot(named)
    target = named[0][1]
    original = target.detach().clone()
    with torch.no_grad():
        target.add_(1.0)
    with anchor.swapped():
        assert torch.equal(target, original)
    assert torch.equal(target, original + 1.0)
    restore(anchor)
    assert torch.equal(target, original)


def test_option_a_freeze_trains_only_block_rows(
    subject: MolmoAct2DiscreteStack,
) -> None:
    """The R0-A surface: exactly the two untied matrices are trainable,
    and a full gradient step moves ONLY their FAST-block rows — every
    row outside [base, end) stays bit-identical (the embedding sees
    real prompt/scaffold-token gradients; the span masking is what
    keeps them frozen)."""
    named = apply_option_a_freeze(subject)
    assert {name for name, _ in named} == {
        "text.transformer.wte.embedding",
        "text.lm_head.weight",
    }
    assert all(not p.requires_grad for p in subject.trunk.vision.parameters())
    frozen = sum(1 for p in subject.trunk.text.parameters() if not p.requires_grad)
    assert frozen > 0, "the rest of the text stack must stay frozen"
    base, end = option_a_row_span(subject)
    assert base == ACTION_TOKEN_START and end == base + 2048
    snapshot = AnchorSnapshot(named)
    try:
        rows = [memory_row(subject, 11, 0), memory_row(subject, 11, 1)]
        before = {name: p.detach().clone() for name, p in named}
        facts = grpo_train_step(
            subject,
            rows,
            torch.tensor([1.0, -1.0]),
            task=TASK,
            config=GRPOConfig(),
            parameters=named,
            optimizer=build_optimizer(named, lr=1e-2),
            microbatch_rows=1,
            grad_clip=1.0,
            grad_row_span=(base, end),
        )
        assert not facts.skipped and facts.tokens > 0
        moved = 0
        for name, param in named:
            after = param.detach()
            assert torch.equal(after[:base], before[name][:base]), name
            assert torch.equal(after[end:], before[name][end:]), name
            if not torch.equal(after[base:end], before[name][base:end]):
                moved += 1
        assert moved == 2, "both matrices' block rows should move"
    finally:
        restore(snapshot)


def test_chunked_grads_match_single_batch(subject: MolmoAct2DiscreteStack) -> None:
    named = apply_option_b_freeze(subject)
    snapshot = AnchorSnapshot(named)
    try:
        rows = [
            memory_row(subject, 3, 0),
            memory_row(subject, 3, 1),
            memory_row(subject, 4, 0),
        ]
        advantages = torch.tensor([0.5, -0.5, 1.0])
        config = GRPOConfig()
        loss_chunked, tokens, mean_ratio, *_ = accumulate_grpo_grads(
            subject,
            rows,
            advantages,
            task=TASK,
            config=config,
            microbatch_rows=2,
        )
        assert tokens == sum(len(r.ids) for r in rows)
        assert abs(mean_ratio - 1.0) < 1e-4
        chunked = {
            name: p.grad.detach().clone() for name, p in named if p.grad is not None
        }
        for _, p in named:
            p.grad = None
        loss_full, _ = molmoact2_grpo_loss(
            subject,
            rows,
            task=TASK,
            advantages=advantages,
            config=config,
        )
        loss_full.backward()
        assert abs(loss_chunked - float(loss_full.detach())) < 1e-6
        checked = 0
        for name, p in named:
            if p.grad is None:
                assert name not in chunked
                continue
            assert torch.allclose(chunked[name], p.grad, rtol=1e-5, atol=1e-7), name
            checked += 1
        assert checked > 0
        # Zero advantage -> exactly-zero gradient on every parameter.
        for _, p in named:
            p.grad = None
        accumulate_grpo_grads(
            subject,
            rows,
            torch.zeros(3),
            task=TASK,
            config=config,
            microbatch_rows=2,
        )
        for name, p in named:
            if p.grad is not None:
                assert float(p.grad.abs().max()) == 0.0, name
        for _, p in named:
            p.grad = None
    finally:
        restore(snapshot)


def test_anchor_kl_zero_then_positive(subject: MolmoAct2DiscreteStack) -> None:
    named = apply_option_b_freeze(subject)
    anchor = AnchorSnapshot(named)
    try:
        rows = [memory_row(subject, 5, 0)]
        # Rollout policy == anchor: reduction-shape noise only (the
        # recorded logprobs came from the incremental decode, the
        # anchor forward is one-shot — k3 is quadratic in that delta).
        fresh = anchor_kl(subject, anchor, rows, task=TASK, temperature=1.0)
        assert 0.0 <= fresh < 1e-9
        with torch.no_grad():
            named[-1][1].add_(0.01 * torch.randn_like(named[-1][1]))
        # Rows recorded under the MOVED policy drift from the anchor.
        drifted = [memory_row(subject, 5, 1)]
        moved = anchor_kl(subject, anchor, drifted, task=TASK, temperature=1.0)
        assert moved > 1e-9 and math.isfinite(moved)
    finally:
        restore(anchor)


def test_checkpoint_roundtrip(
    subject: MolmoAct2DiscreteStack,
    tmp_path: Path,
) -> None:
    named = apply_option_b_freeze(subject)
    snapshot = AnchorSnapshot(named)
    try:
        config = GRPOLoopConfig(out_dir=tmp_path, total_steps=10, keep_checkpoints=2)
        optimizer = torch.optim.AdamW([p for _, p in named], lr=1e-3)
        baseline = {200: 1.5, 201: -0.5}
        path = save_checkpoint(config, 7, named, optimizer, baseline)
        assert path.name == "step_0007.pt"
        with torch.no_grad():
            named[0][1].add_(1.0)
        step, restored_baseline = load_checkpoint(path, named, optimizer)
        assert step == 7 and restored_baseline == baseline
        assert torch.equal(named[0][1], snapshot.values[0].to(named[0][1].dtype))
        with pytest.raises(ValueError, match="trainable set"):
            load_checkpoint(path, named[1:], optimizer)
    finally:
        restore(snapshot)


# ------------------------------------------------------------ loop end to end


def loop_config(tmp_path: Path, **overrides: Any) -> GRPOLoopConfig:
    defaults: dict[str, Any] = {
        "out_dir": tmp_path / "loop",
        "total_steps": 2,
        "task": TASK,
        "seeds_per_step": 2,
        "draws": 2,
        "temperature": 1.0,
        "train_seed_base": 1000,
        "eval_every": 5,
        "eval_seed_base": 200,
        "eval_seed_count": 2,
        "lr": 1e-3,
        "microbatch_rows": 3,
        "kl_rows": 2,
        "save_every": 1,
    }
    defaults.update(overrides)
    return GRPOLoopConfig(**defaults)


def make_wave(
    subject: MolmoAct2DiscreteStack,
    tmp_path: Path,
    *,
    progress_by_draw: tuple[float, ...],
    strikes: int = 0,
    write_rows: bool = True,
) -> WaveFn:
    obs = _observation()

    def wave(step: int, seeds: list[int]) -> tuple[list[EpisodeResult], Path]:
        writer = TrainingRowWriter(
            tmp_path / "rows" / f"step_{step:04d}",
            {"run_seed": 0, "task": obs["task"]},
        )
        episodes: list[EpisodeResult] = []
        for seed in seeds:
            for draw, progress in enumerate(progress_by_draw):
                if write_rows:
                    row = sampled_row(subject, seed, draw)
                    writer.write(
                        seed=seed,
                        replan=0,
                        draw=draw,
                        top=obs["images"][0],
                        wrist=obs["images"][1],
                        state=obs["state"].numpy(),
                        row=row,
                    )
                episodes.append(
                    episode(seed, draw, progress=progress, strikes=strikes),
                )
        return episodes, writer.root

    return wave


def eval_wave(seeds: list[int]) -> list[EpisodeResult]:
    return [episode(seed, 0, progress=2.0) for seed in seeds]


def heartbeat_rows(config: GRPOLoopConfig) -> list[dict[str, Any]]:
    lines = (config.out_dir / "train.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_loop_end_to_end(subject: MolmoAct2DiscreteStack, tmp_path: Path) -> None:
    named = apply_option_b_freeze(subject)
    snapshot = AnchorSnapshot(named)
    try:
        config = loop_config(tmp_path)
        before = named[0][1].detach().clone()
        result = run_grpo_loop(
            subject,
            config,
            wave_fn=make_wave(subject, tmp_path, progress_by_draw=(0.0, 1.0)),
            eval_fn=eval_wave,
            parameters=named,
        )
        assert result.stopped_reason is None
        assert result.steps_done == 2
        assert result.baseline == {200: 2.0, 201: 2.0}
        assert result.last_eval is not None and result.last_eval.step == 2
        rows = heartbeat_rows(config)
        assert [r["step"] for r in rows] == [0, 1, 2, 2]
        baseline_row, step1, step2, endpoint = rows
        assert baseline_row["eval_delta_mean"] is None
        assert baseline_row["eval_reward_mean"] == 2.0
        for row in (step1, step2):
            assert row["loss"] is not None and math.isfinite(row["loss"])
            assert row["tokens"] > 0
            assert row["groups_kept"] == 2 and row["groups_total"] == 2
            assert row["median_group_std"] == 0.5
            assert row["strikes"] == 0
            assert row["anchor_kl"] is not None
            assert row["s_per_step"] >= 0
            assert row["step_skipped"] is False
        # Step 1 trains the fresh policy, but off DISK rows: the replay
        # consumes JPEG-decoded frames while the rollout consumed raw —
        # the memo §8 registered lossy budget. On the random-init
        # fixture that budget is ~1% of ratio (a real checkpoint is
        # smoother); the bit-level ratio/KL contracts are pinned by the
        # in-memory-row oracles above.
        assert abs(step1["mean_ratio"] - 1.0) < 0.05
        assert step1["clip_fraction"] < 0.5
        assert step1["anchor_kl"] >= 0.0 and step2["anchor_kl"] >= 0.0
        assert endpoint["eval_delta_mean"] == 0.0
        assert not torch.equal(named[0][1], before), "parameters never moved"
        assert not list((tmp_path / "rows").iterdir()), "rows must prune"
        saved = sorted(p.name for p in config.out_dir.glob("step_*.pt"))
        assert saved == ["step_0001.pt", "step_0002.pt"]
        step, loaded_baseline = load_checkpoint(
            config.out_dir / "step_0002.pt",
            named,
            torch.optim.AdamW([p for _, p in named], lr=1e-3),
        )
        assert step == 2 and loaded_baseline == result.baseline
    finally:
        restore(snapshot)


def test_loop_end_to_end_option_a_with_mitigation(
    subject: MolmoAct2DiscreteStack,
    tmp_path: Path,
) -> None:
    """The R0-A configuration through the production loop: surface A
    selected by config (the loop builds the freeze + row span itself),
    KL penalty on (anchor_k3_pre in every trained step's heartbeat
    row), advantage clip threaded — and only block rows move."""
    named = apply_option_a_freeze(subject)
    snapshot = AnchorSnapshot(named)
    base, end = option_a_row_span(subject)
    try:
        config = loop_config(
            tmp_path,
            surface="a",
            kl_beta=0.5,
            advantage_clip=2.0,
            eval_every=1,
        )
        before = {name: p.detach().clone() for name, p in named}
        result = run_grpo_loop(
            subject,
            config,
            wave_fn=make_wave(subject, tmp_path, progress_by_draw=(0.0, 1.0)),
            eval_fn=eval_wave,
        )
        assert result.stopped_reason is None and result.steps_done == 2
        trained = [row for row in heartbeat_rows(config) if row.get("loss") is not None]
        assert len(trained) == 2
        for row in trained:
            assert row["anchor_k3_pre"] is not None
            assert row["anchor_k3_pre"] >= 0.0
            assert math.isfinite(row["loss"])
        moved = 0
        for name, param in named:
            after = param.detach()
            assert torch.equal(after[:base], before[name][:base]), name
            assert torch.equal(after[end:], before[name][end:]), name
            moved += int(not torch.equal(after[base:end], before[name][base:end]))
        assert moved == 2, "the option-A block rows should move"
        # The anchor-chunk widths must align with the accumulation
        # chunking (mixed chunk sizes: 4 rows at microbatch 3).
        rows = [memory_row(subject, 21, d) for d in range(4)]
        anchor = AnchorSnapshot(named)
        chunks = anchor_chunk_logprobs(
            subject,
            anchor,
            rows,
            task=TASK,
            temperature=1.0,
            microbatch_rows=3,
        )
        assert [c.shape[0] for c in chunks] == [3, 1]
        with pytest.raises(ValueError, match="one anchor tensor per chunk"):
            accumulate_grpo_grads(
                subject,
                rows,
                torch.tensor([1.0, -1.0, 0.5, -0.5]),
                task=TASK,
                config=GRPOConfig(),
                microbatch_rows=3,
                anchor_chunks=chunks[:1],
                kl_beta=0.5,
            )
        for _, p in named:
            p.grad = None
    finally:
        restore(snapshot)


def test_resume_uses_step0_anchor_and_completes(
    subject: MolmoAct2DiscreteStack,
    tmp_path: Path,
) -> None:
    """A resumed loop keeps the step-0 policy as the KL anchor: main()
    snapshots BEFORE load_checkpoint restores the live tensors —
    snapshotting after the restore would silently rebase anchor_kl
    onto the resumed weights (the reference R1 leans on)."""
    named = apply_option_b_freeze(subject)
    snapshot = AnchorSnapshot(named)
    try:
        config = loop_config(tmp_path)
        result = run_grpo_loop(
            subject,
            config,
            wave_fn=make_wave(subject, tmp_path, progress_by_draw=(0.0, 1.0)),
            eval_fn=eval_wave,
            parameters=named,
        )
        assert result.steps_done == 2
        # Fresh-process resume in main()'s order: pristine step-0
        # weights, anchor captured, THEN the restore overwrites live.
        restore(snapshot)
        anchor = AnchorSnapshot(named)
        optimizer = build_optimizer(named, lr=config.lr)
        start_step, baseline = load_checkpoint(
            config.out_dir / "step_0001.pt",
            named,
            optimizer,
        )
        assert start_step == 1 and baseline == result.baseline
        moved = any(
            not torch.equal(value, param.detach().cpu())
            for value, (_, param) in zip(anchor.values, anchor.named, strict=True)
        )
        assert moved, "the restored step-1 weights should differ from the anchor"
        resumed = run_grpo_loop(
            subject,
            config,
            wave_fn=make_wave(subject, tmp_path, progress_by_draw=(0.0, 1.0)),
            eval_fn=eval_wave,
            parameters=named,
            optimizer=optimizer,
            start_step=start_step,
            baseline=baseline,
            anchor=anchor,
        )
        assert resumed.stopped_reason is None and resumed.steps_done == 2
        rows = heartbeat_rows(config)
        # First run wrote [0, 1, 2, 2]; the resume appends only its
        # step-2 row + the endpoint eval.
        assert [r["step"] for r in rows] == [0, 1, 2, 2, 2, 2]
        assert rows[-2]["anchor_kl"] is not None and rows[-2]["anchor_kl"] >= 0.0
        assert rows[-1]["eval_delta_mean"] == 0.0
        assert all(
            torch.equal(current, original)
            for current, original in zip(
                anchor.values,
                snapshot.values,
                strict=True,
            )
        ), "anchor must still hold the step-0 values bit-exactly"
    finally:
        restore(snapshot)


def test_loop_strike_tripwire_stops_before_update(
    subject: MolmoAct2DiscreteStack,
    tmp_path: Path,
) -> None:
    named = apply_option_b_freeze(subject)
    snapshot = AnchorSnapshot(named)
    try:
        config = loop_config(tmp_path, total_steps=3)
        before = named[0][1].detach().clone()
        result = run_grpo_loop(
            subject,
            config,
            wave_fn=make_wave(
                subject,
                tmp_path,
                progress_by_draw=(0.0, 1.0),
                strikes=1,
                write_rows=False,
            ),
            eval_fn=eval_wave,
            parameters=named,
        )
        assert result.stopped_reason is not None
        assert "strike" in result.stopped_reason
        assert result.steps_done == 0
        assert torch.equal(named[0][1], before), "strike wave must never train"
        rows = heartbeat_rows(config)
        assert rows[-1] == {"step": 1, "tripwire": [result.stopped_reason]}
        assert rows[-2]["loss"] is None
        assert (tmp_path / "rows" / "step_0000").exists(), "rows kept for diagnosis"
    finally:
        restore(snapshot)


def test_loop_collapse_tripwire(
    subject: MolmoAct2DiscreteStack,
    tmp_path: Path,
) -> None:
    named = apply_option_b_freeze(subject)
    config = loop_config(tmp_path, total_steps=5)
    result = run_grpo_loop(
        subject,
        config,
        wave_fn=make_wave(
            subject,
            tmp_path,
            progress_by_draw=(1.0, 1.0),  # zero spread every group
            write_rows=False,
        ),
        eval_fn=eval_wave,
        parameters=named,
    )
    assert result.stopped_reason is not None
    assert "collapse" in result.stopped_reason
    assert result.steps_done == 2  # streak of 3 fires on the third wave
    step_rows = [r for r in heartbeat_rows(config) if "loss" in r]
    assert all(r["loss"] is None and r["groups_kept"] == 0 for r in step_rows)


def test_loop_kl_stop_tripwire(
    subject: MolmoAct2DiscreteStack,
    tmp_path: Path,
) -> None:
    """The §7 KL numeric line (set at the R0 boundary): one anchor_kl
    reading over ``kl_stop`` stops the loop — no streak. A line at the
    noise floor fires on the very first trained step (disk-row JPEG
    reduction noise is nonzero even at the anchor itself)."""
    named = apply_option_b_freeze(subject)
    snapshot = AnchorSnapshot(named)
    try:
        config = loop_config(tmp_path, total_steps=3, kl_stop=1e-12)
        result = run_grpo_loop(
            subject,
            config,
            wave_fn=make_wave(subject, tmp_path, progress_by_draw=(0.0, 1.0)),
            eval_fn=eval_wave,
            parameters=named,
        )
        assert result.stopped_reason is not None
        assert "anchor-KL runaway" in result.stopped_reason
        assert result.steps_done < 3
        rows = heartbeat_rows(config)
        assert rows[-1]["tripwire"] == [result.stopped_reason]
    finally:
        restore(snapshot)


def test_loop_violence_tripwire(
    subject: MolmoAct2DiscreteStack,
    tmp_path: Path,
) -> None:
    named = apply_option_b_freeze(subject)
    config = loop_config(tmp_path, total_steps=5, min_group_std=0.0)
    result = run_grpo_loop(
        subject,
        config,
        wave_fn=make_wave(
            subject,
            tmp_path,
            progress_by_draw=(-2.0, -2.0),  # every episode a knock-away
            write_rows=False,
        ),
        eval_fn=eval_wave,
        parameters=named,
    )
    assert result.stopped_reason is not None
    assert "violence" in result.stopped_reason
    assert result.steps_done == 2
    step_rows = [r for r in heartbeat_rows(config) if "knockaway_frac" in r]
    assert all(r["knockaway_frac"] == 1.0 for r in step_rows)


def make_success_mix_wave(tmp_path: Path, *, mixed: bool) -> WaveFn:
    """Waves whose groups carry (or lack) the success/failure contrast
    at ZERO reward spread — the z-filter drops every group, so no
    training rows are needed and the wave-0 calibration read is
    isolated from the gradient path. Equal rewards by construction:
    failure at 11 cm progress == success at 1 cm + the +10 bonus."""

    def wave(step: int, seeds: list[int]) -> tuple[list[EpisodeResult], Path]:
        root = tmp_path / "rows" / f"step_{step:04d}"
        root.mkdir(parents=True, exist_ok=True)
        episodes: list[EpisodeResult] = []
        for seed in seeds:
            episodes.append(
                episode(seed, 0, progress=11.0 if mixed else 1.0),
            )
            episodes.append(
                episode(seed, 1, progress=1.0, success=mixed),
            )
        return episodes, root

    return wave


def test_loop_wave0_mixed_abort(
    subject: MolmoAct2DiscreteStack,
    tmp_path: Path,
) -> None:
    """The A3.3 wave-0 calibration gate: mixed-group fraction below the
    bar at wave 0 stops the run; at-or-above it never fires again (the
    gate is wave-0-only), and the fraction lands in the heartbeat."""
    named = apply_option_b_freeze(subject)
    config = loop_config(tmp_path, total_steps=2, wave0_mixed_abort=0.2)
    result = run_grpo_loop(
        subject,
        config,
        wave_fn=make_success_mix_wave(tmp_path, mixed=False),
        eval_fn=eval_wave,
        parameters=named,
    )
    assert result.stopped_reason is not None
    assert "wave-0 calibration abort" in result.stopped_reason
    assert result.steps_done == 0
    rows = heartbeat_rows(config)
    step_rows = [r for r in rows if "mixed_groups_frac" in r]
    assert step_rows and step_rows[0]["mixed_groups_frac"] == 0.0
    assert rows[-1]["tripwire"] == [result.stopped_reason]

    config = loop_config(
        tmp_path,
        out_dir=tmp_path / "loop-pass",
        total_steps=2,
        wave0_mixed_abort=0.2,
    )
    result = run_grpo_loop(
        subject,
        config,
        wave_fn=make_success_mix_wave(tmp_path, mixed=True),
        eval_fn=eval_wave,
        parameters=named,
    )
    assert result.stopped_reason is None and result.steps_done == 2
    step_rows = [r for r in heartbeat_rows(config) if "mixed_groups_frac" in r]
    assert [r["mixed_groups_frac"] for r in step_rows] == [1.0, 1.0]


def test_parse_args_knockaway_baseline_normalization() -> None:
    """--knockaway-baseline dies at parse time on a typo (before any
    checkpoint load) and normalizes to float | 'wave0' | None."""
    base = ["--checkpoint", "ckpt", "--total-steps", "1"]
    assert parse_args(base).knockaway_baseline is None
    assert parse_args([*base, "--knockaway-baseline", "wave0"]).knockaway_baseline == (
        "wave0"
    )
    parsed = parse_args([*base, "--knockaway-baseline", "0.25"])
    assert parsed.knockaway_baseline == 0.25
    with pytest.raises(SystemExit):
        parse_args([*base, "--knockaway-baseline", "wave1"])
    with pytest.raises(SystemExit):
        parse_args([*base, "--knockaway-baseline", "0.0"])


def test_parse_args_clutter_appearance() -> None:
    """--clutter-appearance (R2 wave-0 postmortem): default stays the
    production 'patched' (zero behavior change for other runs), the
    substrate-pinned run passes 'standins', a typo dies at parse time."""
    base = ["--checkpoint", "ckpt", "--total-steps", "1"]
    assert parse_args(base).clutter_appearance == "patched"
    assert (
        parse_args([*base, "--clutter-appearance", "standins"]).clutter_appearance
        == "standins"
    )
    with pytest.raises(SystemExit):
        parse_args([*base, "--clutter-appearance", "patchd"])


def test_wave_fns_forward_clutter_appearance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both sim-wave closures (training wave AND in-loop eval) forward
    the config substrate into run_units — the wave-0 postmortem's exact
    failure seam (the loop silently rode WorkerConfig's 'patched'
    default while the anchors were standins)."""
    from sim import grpo_loop

    captured: list[str] = []

    def fake_run_units(*args: Any, **kwargs: Any) -> list[Any]:
        captured.append(kwargs["clutter_appearance"])
        return []

    monkeypatch.setattr(grpo_loop, "run_units", fake_run_units)
    config = loop_config(tmp_path, clutter_appearance="standins")
    wave_fn, eval_fn = grpo_loop.make_sim_wave_fns(
        None,
        None,
        config,
        workers=1,
        replans=1,
        horizon=1,
        post_backend="none",
        checkpoint="ckpt",
        commit="deadbee",
    )
    wave_fn(0, [2000])
    eval_fn([200])
    assert captured == ["standins", "standins"]
