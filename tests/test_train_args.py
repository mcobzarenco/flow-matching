"""Oracles for the checkpoint-inferred-flag rule (molmo_flow plan step 1,
architecture.md §8.13 decision 9) and the TrainArgs validation split.

- from_namespace owns explicitness: --resume refuses every ARCH_FLAGS
  flag (they resolve from the checkpoint); --init-from refuses
  inherited-section flags, admits DECODER-section flags only behind the
  --decoder replacement declarator (the stage-2 path), and always admits
  EXTENSION flags (zero-init structure additions).
- __post_init__ owns the value invariants of the RESOLVED config, once —
  from_namespace translates its ValueError to parser.error, so direct
  construction and the CLI share one encoding.
- ARCH_FLAGS (write side) and loading.CheckpointTrainArgs (read side)
  both encode "what rebuilds the model"; the sync test pins them so the
  two cannot drift.

All tests drive the REAL parser (_build_parser) with argv lists and a
fabricated CheckpointInfo — no fixture checkpoints, no file I/O (the
from_json/from_dict split: parse_args is the I/O shell).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from bijou.data import DatasetStats
from bijou.decoders.flow import SelfAttentionMode, TimeConditioning
from bijou.loading import CheckpointInfo, CheckpointTrainArgs
from bijou.train import (
    ARCH_DEFAULTS,
    ARCH_FLAGS,
    DEFAULT_BACKBONE,
    ArchSection,
    TrainArgs,
    _build_parser,
)

_STATS = DatasetStats.from_state_dict(
    {
        "action": {"mean": [0.0] * 6, "std": [1.0] * 6},
        "observation.state": {"mean": [0.0] * 6, "std": [1.0] * 6},
    },
)


def _checkpoint_info(**overrides: object) -> CheckpointInfo:
    """A fabricated flow-decoder checkpoint's metadata (values chosen to
    differ from ARCH_DEFAULTS wherever a resolution test reads them)."""
    train_args = CheckpointTrainArgs(
        decoder="flow",
        decoder_hidden=1024,
        decoder_heads=8,
        decoder_intermediate=4096,
        decoder_cross_heads=8,
        stream_counts=(4, 4, 4, 4, 6),
        conditioning_streams="kv",
        self_attention_mode=SelfAttentionMode.BIDIRECTIONAL,
        chunk_size=50,
        max_soft_tokens=140,
        max_crops=1,
        time_conditioning=TimeConditioning.ADARMS,
        target_time_embed=False,
        fast_tokenizer=None,
        joint_ce=False,
    )
    fields: dict[str, object] = {
        "backbone": "google/gemma-4-e2b-it",
        "train_args": train_args,
        "step": 1000,
        "normalization": _STATS,
        "per_dataset_normalization": {},
        "condition_fields": (),
        "generate_bracket": True,
    }
    train_args_overrides = {
        k: v
        for k, v in overrides.items()
        if k in {f.name for f in dataclasses.fields(CheckpointTrainArgs)}
    }
    if train_args_overrides:
        fields["train_args"] = dataclasses.replace(
            train_args,
            **train_args_overrides,  # type: ignore[arg-type] — test override table
        )
    fields.update(
        {k: v for k, v in overrides.items() if k not in train_args_overrides},
    )
    return CheckpointInfo(**fields)  # type: ignore[arg-type] — kwargs built above


def _parse(argv: list[str], checkpoint: CheckpointInfo | None = None) -> TrainArgs:
    parser = _build_parser()
    raw = parser.parse_args(["--train-data", "data", *argv])
    return TrainArgs.from_namespace(raw, parser, checkpoint=checkpoint)


def test_arch_partition_matches_checkpoint_train_args() -> None:
    """Write side (ARCH_FLAGS) == read side (CheckpointTrainArgs) — the
    two encodings of "what rebuilds the model" cannot drift. backbone and
    prompt_generate_bracket resolve from CheckpointInfo's section fields
    instead; every other arch flag resolves from CheckpointTrainArgs."""
    section_resolved = {"backbone", "prompt_generate_bracket"}
    from_train_args = set(ARCH_FLAGS) - section_resolved
    recorded = {f.name for f in dataclasses.fields(CheckpointTrainArgs)}
    assert from_train_args == recorded
    assert set(ARCH_DEFAULTS) == set(ARCH_FLAGS)
    # PROMPT/BACKBONE sections are never replaceable via CLI flags.
    for field in section_resolved:
        assert ARCH_FLAGS[field][1] in (ArchSection.BACKBONE, ArchSection.PROMPT)


def test_fresh_run_resolves_defaults() -> None:
    args = _parse([])
    assert args.backbone == DEFAULT_BACKBONE
    assert args.decoder == "flow"
    assert args.decoder_hidden == 768
    assert args.stream_counts == (4, 4, 7)
    assert args.chunk_size == 50
    assert args.prompt_generate_bracket is False
    assert args.target_time_embed is False
    assert args.joint_ce is False


def test_fresh_run_explicit_arch_flags_respected() -> None:
    args = _parse(["--decoder-hidden", "1024", "--chunk-size", "30"])
    assert args.decoder_hidden == 1024
    assert args.chunk_size == 30


def test_resume_refuses_every_arch_flag() -> None:
    info = _checkpoint_info()
    for field, (flag, _section) in ARCH_FLAGS.items():
        argv = ["--resume", "ckpt", flag]
        # Value-taking flags need a value; store_true flags do not.
        if field not in ("prompt_generate_bracket", "target_time_embed", "joint_ce"):
            argv.append(
                {
                    "backbone": "x",
                    "decoder": "flow",
                    "conditioning_streams": "kv",
                    "self_attention_mode": "bidirectional",
                    "time_conditioning": "adarms",
                    "fast_tokenizer": "x",
                    "stream_counts": "4",
                }.get(field, "8"),
            )
        with pytest.raises(SystemExit):
            _parse(argv, info)


def test_resume_resolves_recorded_architecture() -> None:
    info = _checkpoint_info()
    args = _parse(["--resume", "ckpt", "--seed", "7"], info)
    assert args.backbone == info.backbone
    assert args.decoder == "flow"
    assert args.decoder_hidden == 1024
    assert args.decoder_heads == 8
    assert args.stream_counts == (4, 4, 4, 4, 6)
    assert args.self_attention_mode == "bidirectional"
    assert args.time_conditioning == "adarms"
    assert args.prompt_generate_bracket is True  # recorded section value
    assert args.seed == 7  # run policy stays CLI-owned


def test_resume_ar_backbone_infers_implied_bracket() -> None:
    """ar_backbone sources refused the bracket flag and rendered it
    implicitly — resolution must infer True, not the recorded False."""
    info = _checkpoint_info(
        decoder="ar_backbone",
        fast_tokenizer="user/repo/tok_v2",
        # Real ar_backbone records carry the flow-only knobs at their
        # defaults (the old parser refused anything else for non-flow).
        time_conditioning=TimeConditioning.ADDITIVE,
        generate_bracket=False,
    )
    args = _parse(["--resume", "ckpt"], info)
    assert args.decoder == "ar_backbone"
    assert args.fast_tokenizer == "user/repo/tok_v2"
    assert args.prompt_generate_bracket is True


def test_init_from_refuses_inherited_sections() -> None:
    info = _checkpoint_info()
    for argv in (
        ["--backbone", "x"],
        ["--max-soft-tokens", "200"],
        ["--max-crops", "2"],
        ["--prompt-generate-bracket"],
        # DECODER-section flag without the --decoder declarator:
        ["--decoder-hidden", "512"],
        ["--chunk-size", "30"],
    ):
        with pytest.raises(SystemExit):
            _parse(["--init-from", "ckpt", *argv], info)


def test_init_from_inherits_decoder_without_declarator() -> None:
    info = _checkpoint_info()
    args = _parse(["--init-from", "ckpt"], info)
    assert args.decoder_hidden == 1024
    assert args.time_conditioning == "adarms"


def test_init_from_decoder_declarator_opens_the_section() -> None:
    """The stage-2 path: --decoder declares a fresh decoder section, its
    shape flags become legal, unset ones take FRESH defaults (not the
    source checkpoint's), and inherited sections stay checkpoint-owned."""
    info = _checkpoint_info(
        decoder="ar_backbone",
        fast_tokenizer="t",
        time_conditioning=TimeConditioning.ADDITIVE,
    )
    args = _parse(
        ["--init-from", "ckpt", "--decoder", "flow", "--decoder-hidden", "512"],
        info,
    )
    assert args.decoder == "flow"
    assert args.decoder_hidden == 512
    assert args.decoder_heads == 6  # fresh default, not the source's
    assert args.time_conditioning == "additive"  # fresh default
    assert args.fast_tokenizer is None  # fresh decoder section
    assert args.backbone == info.backbone  # inherited
    assert args.prompt_generate_bracket is True  # inherited (implied OR)


def test_init_from_admits_extensions() -> None:
    info = _checkpoint_info()
    args = _parse(["--init-from", "ckpt", "--target-time-embed"], info)
    assert args.target_time_embed is True


def test_snapflow_implies_phi_s_where_mutable() -> None:
    fresh = _parse(["--distill", "snapflow", "--time-conditioning", "adarms"])
    assert fresh.target_time_embed is True
    # A resumed checkpoint cannot grow structure: refused, not implied.
    info = _checkpoint_info(target_time_embed=False)
    with pytest.raises(SystemExit):
        _parse(["--resume", "ckpt", "--distill", "snapflow"], info)
    extended = _checkpoint_info(target_time_embed=True)
    args = _parse(["--resume", "ckpt", "--distill", "snapflow"], extended)
    assert args.target_time_embed is True


def test_ar_backbone_shape_flags_refused_fresh() -> None:
    with pytest.raises(SystemExit):
        _parse(
            [
                "--decoder",
                "ar_backbone",
                "--fast-tokenizer",
                "t",
                "--decoder-hidden",
                "1024",
            ],
        )


def test_residual_stream_counts_refused() -> None:
    with pytest.raises(SystemExit):
        _parse(
            [
                "--conditioning-streams",
                "residual",
                "--stream-counts",
                "4",
                "4",
            ],
        )


def test_post_init_validates_direct_construction() -> None:
    """The single-encoding property: building an invalid TrainArgs
    directly raises the same message the CLI would print."""
    args = _parse([])
    with pytest.raises(ValueError, match="--holdout-episodes must be in"):
        dataclasses.replace(args, holdout_episodes=1.5)
    with pytest.raises(ValueError, match="--decoder ar_fast requires --fast-tokenizer"):
        dataclasses.replace(args, decoder="ar_fast")
    with pytest.raises(ValueError, match="--rewarmup-steps anchors at the resume"):
        dataclasses.replace(args, rewarmup_steps=100)
    with pytest.raises(ValueError, match="template order"):
        dataclasses.replace(
            args,
            decoder="ar_backbone",
            fast_tokenizer="t",
            aux_fields=("holding", "subgoal"),
        )


def test_value_invariants_reach_cli_as_parser_error() -> None:
    with pytest.raises(SystemExit):  # parser.error(str(ValueError))
        _parse(["--holdout-episodes", "1.5"])
    with pytest.raises(SystemExit):
        _parse(["--decoder", "ar_fast"])  # no --fast-tokenizer


def test_resume_and_init_from_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        _parse(["--resume", "a", "--init-from", "b"], _checkpoint_info())


def test_dropout_requires_enabler_pre_resolution() -> None:
    with pytest.raises(SystemExit):
        _parse(["--field-dropout", "0.2"])  # no --aux-fields
    args = _parse([])
    assert args.field_dropout == 0.0  # resolved conditional default


def test_train_args_serializes_resolved_values() -> None:
    """Checkpoint metadata records what the run actually trained with:
    a resumed run's asdict carries the checkpoint's architecture, so the
    NEXT resume resolves identically (Path fields stringify at the
    metadata edge as before)."""
    info = _checkpoint_info()
    args = _parse(["--resume", "ckpt"], info)
    record = dataclasses.asdict(args)
    assert record["decoder_hidden"] == 1024
    assert record["stream_counts"] == (4, 4, 4, 4, 6)
    reread = CheckpointTrainArgs.from_dict(
        {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in record.items()
            if v is None or isinstance(v, (str, int, float, bool, tuple))
        },
    )
    assert reread == info.train_args


def test_molmo_flow_is_inherit_only() -> None:
    """§8.13 step 5: molmo_flow never appears in --decoder choices (the
    declarable-fresh kinds); it resolves from a checkpoint. The
    composition rules ride the resolved value."""
    with pytest.raises(SystemExit):  # not a choice — argparse refuses
        _parse(["--decoder", "molmo_flow"])
    info = _checkpoint_info(
        decoder="molmo_flow",
        conditioning_streams="kv_cache",
        chunk_size=30,
        # Converted checkpoints record the placeholder defaults for the
        # flow-only knobs (the converter's synthesized train_args).
        time_conditioning=TimeConditioning.ADDITIVE,
        self_attention_mode=SelfAttentionMode.BIDIRECTIONAL,
    )
    args = _parse(["--resume", "ckpt"], info)
    assert args.decoder == "molmo_flow"
    assert args.chunk_size == 30
    assert args.insulate_expert is False
    insulated = _parse(["--resume", "ckpt", "--insulate-expert"], info)
    assert insulated.insulate_expert is True
    # Frozen-trunk insulation is their post-train; an unfrozen trunk
    # under insulation trains on nothing (no CE rider until step 6).
    with pytest.raises(SystemExit):
        _parse(
            ["--resume", "ckpt", "--insulate-expert", "--backbone-text-lr", "1e-5"],
            info,
        )
    with pytest.raises(SystemExit):  # bijou bracket surfaces have no bytes
        _parse(
            ["--resume", "ckpt", "--condition-fields", "outcome"],
            info,
        )


def test_insulate_expert_is_molmo_flow_only() -> None:
    with pytest.raises(SystemExit):
        _parse(["--insulate-expert"])  # fresh flow run
    args = _parse([])
    with pytest.raises(ValueError, match="molmo_flow KV seam"):
        dataclasses.replace(args, insulate_expert=True)
    with pytest.raises(ValueError, match="trains from a checkpoint only"):
        dataclasses.replace(args, decoder="molmo_flow")


def test_paths_and_policy_flags_stay_cli_owned() -> None:
    info = _checkpoint_info()
    args = _parse(
        [
            "--resume",
            "ckpt",
            "--steps",
            "80000",
            "--decoder-lr",
            "5e-5",
            "--rewarmup-steps",
            "1000",
            "--train-data",
            "other_box_path",
        ],
        info,
    )
    assert args.steps == 80000
    assert args.rewarmup_steps == 1000
    assert args.train_data == (Path("other_box_path"),)
