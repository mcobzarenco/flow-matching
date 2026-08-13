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
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from test_molmoact2_discrete import (
    ACTION_TOKEN_START,
    codec,
    discrete_predictor,
)
from test_molmoact2_predictor import _observation, build_predictor

from bijou.decoders.ar_backbone import ActionCaptureStep
from bijou.eval.policies import stable_sample_rng, token_rows_from_capture
from bijou.molmoact2 import MolmoAct2Predictor
from bijou.molmoact2.replay import ReplayRow, molmoact2_grpo_loss
from bijou.train_grpo import GRPOConfig
from sim.grpo_loop import (
    AnchorSnapshot,
    GRPOLoopConfig,
    TripwireState,
    WaveFn,
    accumulate_grpo_grads,
    anchor_kl,
    apply_option_b_freeze,
    composite_reward,
    eval_facts,
    group_advantages,
    load_checkpoint,
    paired_bootstrap_ci,
    run_grpo_loop,
    save_checkpoint,
    update_tripwires,
)
from sim.rollout_sim import EpisodeResult
from sim.rollout_sim_parallel import TrainingRowWriter

TASK = _observation()["task"]


@pytest.fixture(scope="module")
def predictor() -> MolmoAct2Predictor:
    return build_predictor(vocab_size=156_032)


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


def sampled_row(subject: MolmoAct2Predictor, seed: int, draw: int) -> Any:
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


def memory_row(subject: MolmoAct2Predictor, seed: int, draw: int) -> ReplayRow:
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


# ------------------------------------------------------ surface + step oracle


def test_option_b_freeze_and_anchor_swap(predictor: MolmoAct2Predictor) -> None:
    subject = discrete_predictor(predictor, codec())
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


def test_chunked_grads_match_single_batch(predictor: MolmoAct2Predictor) -> None:
    subject = discrete_predictor(predictor, codec())
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


def test_anchor_kl_zero_then_positive(predictor: MolmoAct2Predictor) -> None:
    subject = discrete_predictor(predictor, codec())
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
    predictor: MolmoAct2Predictor,
    tmp_path: Path,
) -> None:
    subject = discrete_predictor(predictor, codec())
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
    subject: MolmoAct2Predictor,
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


def test_loop_end_to_end(predictor: MolmoAct2Predictor, tmp_path: Path) -> None:
    subject = discrete_predictor(predictor, codec())
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


def test_loop_strike_tripwire_stops_before_update(
    predictor: MolmoAct2Predictor,
    tmp_path: Path,
) -> None:
    subject = discrete_predictor(predictor, codec())
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
    predictor: MolmoAct2Predictor,
    tmp_path: Path,
) -> None:
    subject = discrete_predictor(predictor, codec())
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


def test_loop_violence_tripwire(
    predictor: MolmoAct2Predictor,
    tmp_path: Path,
) -> None:
    subject = discrete_predictor(predictor, codec())
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
