"""Resume hardening (deep-dive finding 2) — the three quiet traps.

(a) Nothing restores the data-stream position on --resume: the loop
restarts at epoch 0 with the --seed shuffle, so a same-seed resume
replays exactly the batches and τ/ε draws the checkpoint already
trained on. The team's fresh-seed convention is now enforced by
``check_resume_seed`` with ``--allow-same-seed-resume`` as the
reproduction-only escape hatch. (c) A changed lr/weight-decay on
resume is silently ignored by ``optimizer.load_state_dict`` — the
historical advisory note checked param group 0 only, so a changed
``--backbone-*-lr`` slipped through; ``resume_hyperparameter_notes``
covers every group. ((b), the bf16-snap warning, is a print at the
resume load site — no logic to test.)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from bijou.data import DatasetStats
from bijou.decoders.flow import SelfAttentionMode, TimeConditioning
from bijou.loading import CheckpointInfo, CheckpointTrainArgs
from bijou.train import (
    TrainArgs,
    _build_parser,
    check_resume_seed,
    resume_hyperparameter_notes,
)


def checkpoint_dir(tmp_path: Path, train_args: dict | None) -> Path:
    ckpt = tmp_path / "step_000100"
    ckpt.mkdir()
    meta: dict = {"step": 100}
    if train_args is not None:
        meta["train_args"] = train_args
    (ckpt / "bijou_config.json").write_text(json.dumps(meta))
    return ckpt


# ---- (a) fresh-seed enforcement ------------------------------------


def test_same_seed_resume_refused(tmp_path: Path) -> None:
    ckpt = checkpoint_dir(tmp_path, {"seed": 3})
    with pytest.raises(SystemExit, match="replays"):
        check_resume_seed(ckpt, 3, allow_same_seed=False)


def test_fresh_seed_resume_passes(tmp_path: Path) -> None:
    ckpt = checkpoint_dir(tmp_path, {"seed": 3})
    note = check_resume_seed(ckpt, 4, allow_same_seed=False)
    assert "fresh --seed 4" in note
    assert "3" in note


def test_same_seed_escape_hatch_warns(tmp_path: Path) -> None:
    ckpt = checkpoint_dir(tmp_path, {"seed": 3})
    note = check_resume_seed(ckpt, 3, allow_same_seed=True)
    assert note.startswith("WARNING")
    assert "REPLAY" in note


def test_pre_recording_checkpoint_warns_not_dies(tmp_path: Path) -> None:
    # Checkpoints from before train_args recording: enforcement cannot
    # verify, but a warm start of an old run must not become impossible.
    for train_args in (None, {}):
        base = tmp_path / str(train_args is None)
        base.mkdir()
        ckpt = checkpoint_dir(base, train_args)
        note = check_resume_seed(ckpt, 0, allow_same_seed=False)
        assert note.startswith("WARNING")
        assert "skipped" in note


def test_not_a_checkpoint_dies_cleanly(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="not a checkpoint"):
        check_resume_seed(tmp_path, 0, allow_same_seed=False)


# ---- the CLI flag only applies to --resume -------------------------


def _flow_checkpoint_info() -> CheckpointInfo:
    """A fabricated flow checkpoint's metadata — parse-time resume tests
    go through TrainArgs.from_namespace so no fixture checkpoint (and no
    file I/O) is needed (the from_json/from_dict split)."""
    stats = DatasetStats.from_state_dict(
        {
            "action": {"mean": [0.0], "std": [1.0]},
            "observation.state": {"mean": [0.0], "std": [1.0]},
        },
    )
    return CheckpointInfo(
        backbone="google/gemma-4-e2b-it",
        train_args=CheckpointTrainArgs(
            decoder="flow",
            decoder_hidden=768,
            decoder_heads=6,
            decoder_intermediate=3072,
            decoder_cross_heads=4,
            stream_counts=(4, 4, 7),
            self_attention_mode=SelfAttentionMode.CAUSAL_ACTIONS,
            chunk_size=50,
            max_soft_tokens=140,
            max_crops=1,
            time_conditioning=TimeConditioning.ADDITIVE,
            target_time_embed=False,
            fast_tokenizer=None,
        ),
        step=100,
        normalization=stats,
        per_dataset_normalization={},
        condition_fields=(),
        generate_bracket=False,
    )


def _parse(*extra: str) -> TrainArgs:
    parser = _build_parser()
    raw = parser.parse_args(["--train-data", "corpus", *extra])
    checkpoint = _flow_checkpoint_info() if raw.resume is not None else None
    return TrainArgs.from_namespace(raw, parser, checkpoint=checkpoint)


def test_escape_hatch_without_resume_refused() -> None:
    with pytest.raises(SystemExit):
        _parse("--allow-same-seed-resume")


def test_escape_hatch_with_resume_parses() -> None:
    args = _parse(
        "--resume",
        "ckpt/step_000100",
        "--allow-same-seed-resume",
    )
    assert args.allow_same_seed_resume is True


# ---- (c) hyperparameter notes cover every param group --------------


def _optimizer() -> torch.optim.AdamW:
    # The production shape: decoder + a backbone (decayed, no-decay) pair.
    params = [torch.nn.Parameter(torch.zeros(2)) for _ in range(3)]
    return torch.optim.AdamW(
        [
            {"params": [params[0]], "lr": 1e-4},
            {"params": [params[1]], "lr": 5e-6},
            {"params": [params[2]], "lr": 5e-6, "weight_decay": 0.0},
        ],
        lr=1e-4,
        weight_decay=0.1,
    )


CLI_GROUPS = [
    ("decoder", 1e-4, 0.1),
    ("backbone_text (decayed)", 5e-6, 0.1),
    ("backbone_text (no decay)", 5e-6, 0.0),
]


def test_matching_hyperparameters_are_silent() -> None:
    assert resume_hyperparameter_notes(_optimizer(), CLI_GROUPS) == []


def test_changed_backbone_lr_is_surfaced() -> None:
    # The trap the group-0-only note missed: the checkpoint's backbone
    # lr differs from the CLI's, decoder group untouched. A scheduler
    # restore leaves the schedule-decayed lr in "lr" and the base in
    # "initial_lr" — the note must read the base.
    optimizer = _optimizer()
    for group in optimizer.param_groups[1:]:
        group["initial_lr"] = group["lr"]
        group["lr"] = group["lr"] * 0.5  # schedule-decayed current lr
    cli = [CLI_GROUPS[0]] + [(name, 1e-5, decay) for name, _, decay in CLI_GROUPS[1:]]
    notes = resume_hyperparameter_notes(optimizer, cli)
    assert len(notes) == 2
    assert all("backbone_text" in note for note in notes)
    assert all("5.00e-06" in note and "1.00e-05" in note for note in notes)


def test_changed_weight_decay_is_surfaced() -> None:
    cli = [("decoder", 1e-4, 0.05), *CLI_GROUPS[1:]]
    notes = resume_hyperparameter_notes(_optimizer(), cli)
    assert len(notes) == 1
    assert "weight decay" in notes[0]


def test_group_count_mismatch_asserts() -> None:
    with pytest.raises(AssertionError):
        resume_hyperparameter_notes(_optimizer(), CLI_GROUPS[:2])
