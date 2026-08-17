"""Oracles for the family CLI and the checkpoint-inferred-flag rule.

- Fresh runs DECLARE ``--family``; under --resume/--init-from the family
  is checkpoint-inferred and the flag is refused (the ARCH-flag
  discipline, generalized).
- from_namespace owns explicitness: --resume refuses every ARCH_FLAGS
  flag (they resolve from the checkpoint) and --distill (the recorded
  objective is locked); --init-from refuses inherited-section flags and
  admits EXTENSION flags only (φ_s, and --objective — the molmoact2
  pathway matrix, which transforms the family).
- __post_init__ owns the value invariants of the RESOLVED config, once —
  from_namespace translates its ValueError to parser.error, so direct
  construction and the CLI share one encoding.
- ARCH_FLAGS (write side) and the checkpoint read side (metadata family
  + objective kind + loading.CheckpointTrainArgs) both encode "what
  rebuilds the model"; the sync test pins them so the two cannot drift.

All tests drive the REAL parser (_build_parser) with argv lists and a
fabricated CheckpointResolution — no fixture checkpoints, no file I/O
(the from_json/from_dict split: parse_args is the I/O shell).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from bijou.loading import CheckpointTrainArgs
from bijou.modelling.decoders.flow import SelfAttentionMode, TimeConditioning
from bijou.train import (
    ARCH_DEFAULTS,
    ARCH_FLAGS,
    DEFAULT_BACKBONE,
    ArchSection,
    CheckpointResolution,
    TrainArgs,
    _build_parser,
)
from bijou.vla import VLAFamily


def _checkpoint(**overrides: object) -> CheckpointResolution:
    """A fabricated gemma_flow checkpoint's resolution facts (values
    chosen to differ from ARCH_DEFAULTS wherever a resolution test
    reads them)."""
    train_args = CheckpointTrainArgs(
        decoder="flow",
        decoder_hidden=1024,
        decoder_heads=8,
        decoder_intermediate=4096,
        decoder_cross_heads=8,
        stream_counts=(4, 4, 4, 4, 6),
        self_attention_mode=SelfAttentionMode.BIDIRECTIONAL,
        chunk_size=50,
        max_soft_tokens=140,
        max_crops=1,
        time_conditioning=TimeConditioning.ADARMS,
        target_time_embed=False,
        fast_tokenizer=None,
    )
    fields: dict[str, object] = {
        "family": VLAFamily.GEMMA_FLOW,
        "backbone": "google/gemma-4-e2b-it",
        "step": 1000,
        "objective": {"kind": "flow"},
        "train_args": train_args,
        "condition_fields": (),
        "generate_bracket": True,
    }
    # "objective" is ambiguous between the two records: here it names
    # the RESOLUTION's metadata dict (the tagged objective payload),
    # never the legacy train_args string (CheckpointTrainArgs.objective
    # — convert-time family inference, not a resolution input).
    train_args_overrides = {
        k: v
        for k, v in overrides.items()
        if k != "objective"
        and k in {f.name for f in dataclasses.fields(CheckpointTrainArgs)}
    }
    if train_args_overrides:
        fields["train_args"] = dataclasses.replace(train_args, **train_args_overrides)
    fields.update(
        {k: v for k, v in overrides.items() if k not in train_args_overrides},
    )
    return CheckpointResolution(**fields)  # type: ignore[arg-type] — kwargs built above


def _parse(
    argv: list[str],
    checkpoint: CheckpointResolution | None = None,
) -> TrainArgs:
    parser = _build_parser()
    raw = parser.parse_args(["--train-data", "data", *argv])
    return TrainArgs.from_namespace(raw, parser, checkpoint=checkpoint)


def test_arch_partition_matches_checkpoint_read_side() -> None:
    """Write side (ARCH_FLAGS) == read side — the encodings of "what
    rebuilds the model" cannot drift. backbone and
    prompt_generate_bracket resolve from the metadata/prompt section;
    objective resolves into the FAMILY (metadata.family + the
    --init-from transition); every other arch flag resolves from the
    recorded CheckpointTrainArgs."""
    metadata_resolved = {"backbone", "prompt_generate_bracket", "objective"}
    from_train_args = set(ARCH_FLAGS) - metadata_resolved
    recorded = {f.name for f in dataclasses.fields(CheckpointTrainArgs)}
    # 'decoder' is the retired section declarator: it survives on the
    # READ side only (legacy train_args parse), never as a flag.
    assert from_train_args == recorded - {"decoder", "objective"}
    assert set(ARCH_DEFAULTS) == set(ARCH_FLAGS)
    for field in ("backbone", "prompt_generate_bracket"):
        assert ARCH_FLAGS[field][1] in (ArchSection.BACKBONE, ArchSection.PROMPT)
    assert ARCH_FLAGS["objective"][1] is ArchSection.EXTENSION


def test_fresh_run_requires_family() -> None:
    with pytest.raises(SystemExit):
        _parse([])  # no --family: fresh runs must declare one


def test_fresh_run_resolves_defaults() -> None:
    args = _parse(["--family", "gemma_flow"])
    assert args.family == "gemma_flow"
    assert args.backbone == DEFAULT_BACKBONE
    assert args.decoder_hidden == 768
    assert args.stream_counts == (4, 4, 7)
    assert args.chunk_size == 50
    assert args.prompt_generate_bracket is False
    assert args.target_time_embed is False
    assert args.decoder_lr == 1e-4  # the sentinel resolves to the default


def test_fresh_run_explicit_arch_flags_respected() -> None:
    args = _parse(
        ["--family", "gemma_flow", "--decoder-hidden", "1024", "--chunk-size", "30"],
    )
    assert args.decoder_hidden == 1024
    assert args.chunk_size == 30


def test_family_refused_under_resume_and_init_from() -> None:
    """The generalized ARCH discipline: the family is checkpoint-
    inferred — declaring it against a checkpoint is a contradiction."""
    info = _checkpoint()
    for lead in (["--resume", "ckpt"], ["--init-from", "ckpt"]):
        with pytest.raises(SystemExit):
            _parse([*lead, "--family", "gemma_flow"], info)


def test_resume_infers_family() -> None:
    info = _checkpoint()
    args = _parse(["--resume", "ckpt", "--seed", "7"], info)
    assert args.family == "gemma_flow"
    assert args.seed == 7  # run policy stays CLI-owned


def test_resume_refuses_every_arch_flag() -> None:
    info = _checkpoint()
    for field, (flag, _section) in ARCH_FLAGS.items():
        argv = ["--resume", "ckpt", flag]
        # Value-taking flags need a value; store_true flags do not.
        if field not in ("prompt_generate_bracket", "target_time_embed"):
            argv.append(
                {
                    "backbone": "x",
                    "objective": "flow",
                    "self_attention_mode": "bidirectional",
                    "time_conditioning": "adarms",
                    "fast_tokenizer": "x",
                    "stream_counts": "4",
                }.get(field, "8"),
            )
        with pytest.raises(SystemExit):
            _parse(argv, info)


def test_resume_refuses_distill() -> None:
    """The objective is recorded and locked under --resume; a recorded
    snapflow objective restores WITHOUT the flags — mix knobs included,
    from the recorded payload."""
    info = _checkpoint()
    with pytest.raises(SystemExit):
        _parse(["--resume", "ckpt", "--distill", "snapflow"], info)
    recorded = _checkpoint(
        objective={"kind": "snapflow", "alpha": 0.5, "shortcut_weight": 0.1},
        target_time_embed=True,
    )
    with pytest.raises(SystemExit):
        _parse(["--resume", "ckpt", "--snapflow-alpha", "0.5"], recorded)
    args = _parse(["--resume", "ckpt"], recorded)
    assert args.distill == "snapflow"
    assert args.snapflow_alpha == 0.5
    assert args.snapflow_shortcut_weight == 0.1
    assert args.target_time_embed is True


def test_snapflow_mix_flags_are_required_and_scoped() -> None:
    """--distill snapflow declares its mix explicitly (no silent
    defaults) and the mix flags are refused without it."""
    snapflow = ["--family", "gemma_flow", "--distill", "snapflow"]
    with pytest.raises(SystemExit):
        _parse(snapflow)  # no mix flags
    with pytest.raises(SystemExit):
        _parse([*snapflow, "--snapflow-alpha", "0.5"])  # half a mix
    with pytest.raises(SystemExit):
        # Flags without the objective they parameterize.
        _parse(["--family", "gemma_flow", "--snapflow-alpha", "0.5"])
    with pytest.raises(SystemExit):
        # Value invariants live on the payload: alpha=1 is FlowObjective.
        _parse(
            [
                *snapflow,
                "--snapflow-alpha",
                "1.0",
                "--snapflow-shortcut-weight",
                "0.1",
            ],
        )
    args = _parse(
        [
            *snapflow,
            "--snapflow-alpha",
            "0.5",
            "--snapflow-shortcut-weight",
            "0.1",
        ],
    )
    assert args.distill == "snapflow"
    assert args.snapflow_alpha == 0.5
    assert args.snapflow_shortcut_weight == 0.1
    assert args.target_time_embed is True  # implied where mutable


def test_resume_resolves_recorded_architecture() -> None:
    info = _checkpoint()
    args = _parse(["--resume", "ckpt", "--seed", "7"], info)
    assert args.backbone == info.backbone
    assert args.decoder_hidden == 1024
    assert args.decoder_heads == 8
    assert args.stream_counts == (4, 4, 4, 4, 6)
    assert args.self_attention_mode == "bidirectional"
    assert args.time_conditioning == "adarms"
    assert args.prompt_generate_bracket is True  # recorded section value


def test_resume_ar_family_infers_implied_bracket() -> None:
    """AR-suffix sources refused the bracket flag and rendered it
    implicitly — resolution must infer True, not the recorded False."""
    info = _checkpoint(
        family=VLAFamily.GEMMA_AR,
        objective={"kind": "ar", "narration_weight": 1.0},
        fast_tokenizer="user/repo/tok_v2",
        # Real AR records carry the flow-only knobs at their defaults
        # (the parser refused anything else for non-flow).
        time_conditioning=TimeConditioning.ADDITIVE,
        generate_bracket=False,
    )
    args = _parse(["--resume", "ckpt"], info)
    assert args.family == "gemma_ar"
    assert args.fast_tokenizer == "user/repo/tok_v2"
    assert args.prompt_generate_bracket is True


def test_init_from_refuses_inherited_sections() -> None:
    """Every non-EXTENSION section is inherited: a fresh decoder on an
    inherited trunk is --backbone-init-from + --family (the stage-2
    path), never a section-replacement flag."""
    info = _checkpoint()
    for argv in (
        ["--backbone", "x"],
        ["--max-soft-tokens", "200"],
        ["--max-crops", "2"],
        ["--prompt-generate-bracket"],
        ["--decoder-hidden", "512"],
        ["--chunk-size", "30"],
        ["--fast-tokenizer", "t"],
    ):
        with pytest.raises(SystemExit):
            _parse(["--init-from", "ckpt", *argv], info)


def test_init_from_inherits_decoder_section() -> None:
    info = _checkpoint()
    args = _parse(["--init-from", "ckpt"], info)
    assert args.decoder_hidden == 1024
    assert args.time_conditioning == "adarms"


def test_init_from_admits_extensions() -> None:
    info = _checkpoint()
    args = _parse(["--init-from", "ckpt", "--target-time-embed"], info)
    assert args.target_time_embed is True


def test_objective_transitions_molmoact2_family() -> None:
    """The pathway matrix under --init-from: --objective transforms a
    molmoact2 source's family; omitted, the recorded family carries."""
    source = _checkpoint(
        family=VLAFamily.MOLMOACT2_FLOW,
        backbone="allenai/MolmoAct2-SO100_101",
        decoder="molmo_flow",
        chunk_size=30,
        time_conditioning=TimeConditioning.ADDITIVE,
        self_attention_mode=SelfAttentionMode.BIDIRECTIONAL,
    )
    inherited = _parse(["--init-from", "ckpt"], source)
    assert inherited.family == "molmoact2_flow"
    ar = _parse(
        ["--init-from", "ckpt", "--objective", "ar", "--backbone-text-lr", "1e-5"],
        source,
    )
    assert ar.family == "molmoact2_ar"
    joint = _parse(
        [
            "--init-from",
            "ckpt",
            "--objective",
            "joint",
            "--joint-ce-weight",
            "0.5",
            "--backbone-text-lr",
            "1e-5",
        ],
        source,
    )
    assert joint.family == "molmoact2_joint"
    assert joint.joint_ce_weight == 0.5


def test_objective_refused_off_family_and_fresh() -> None:
    with pytest.raises(SystemExit):  # gemma source has one objective
        _parse(["--init-from", "ckpt", "--objective", "ar"], _checkpoint())
    with pytest.raises(SystemExit):  # fresh runs declare --family
        _parse(["--family", "gemma_flow", "--objective", "flow"])
    info = _checkpoint(family=VLAFamily.MOLMOACT2_FLOW)
    with pytest.raises(SystemExit):  # locked under --resume
        _parse(["--resume", "ckpt", "--objective", "joint"], info)


def test_snapflow_implies_phi_s_where_mutable() -> None:
    mix = ["--snapflow-alpha", "0.5", "--snapflow-shortcut-weight", "0.1"]
    fresh = _parse(
        [
            "--family",
            "gemma_flow",
            "--distill",
            "snapflow",
            *mix,
            "--time-conditioning",
            "adarms",
        ],
    )
    assert fresh.target_time_embed is True
    # --init-from may declare a new objective (stage-2 flows); φ_s is
    # implied there too (the sanctioned zero-init extension), and the
    # mix is re-declared explicitly (a NEW objective, not the record).
    extended = _parse(
        ["--init-from", "ckpt", "--distill", "snapflow", *mix],
        _checkpoint(target_time_embed=False),
    )
    assert extended.target_time_embed is True
    assert extended.snapflow_alpha == 0.5


def test_ar_family_shape_flags_refused_fresh() -> None:
    with pytest.raises(SystemExit):
        _parse(
            [
                "--family",
                "gemma_ar",
                "--fast-tokenizer",
                "t",
                "--decoder-hidden",
                "1024",
            ],
        )
    with pytest.raises(SystemExit):  # the implied bracket has one spelling
        _parse(
            [
                "--family",
                "gemma_ar",
                "--fast-tokenizer",
                "t",
                "--prompt-generate-bracket",
            ],
        )


def test_molmoact2_ar_refuses_explicit_decoder_lr() -> None:
    """LR-vs-offer reconciliation, the parse-time half: molmoact2_ar's
    'decoder' param group is structurally empty — an explicit
    --decoder-lr contradicts the offer."""
    source = _checkpoint(
        family=VLAFamily.MOLMOACT2_FLOW,
        decoder="molmo_flow",
        chunk_size=30,
        time_conditioning=TimeConditioning.ADDITIVE,
        self_attention_mode=SelfAttentionMode.BIDIRECTIONAL,
    )
    with pytest.raises(SystemExit):
        _parse(
            [
                "--init-from",
                "ckpt",
                "--objective",
                "ar",
                "--backbone-text-lr",
                "1e-5",
                "--decoder-lr",
                "1e-4",
            ],
            source,
        )
    # The default-resolved decoder lr is fine (nothing trains at it).
    args = _parse(
        ["--init-from", "ckpt", "--objective", "ar", "--backbone-text-lr", "1e-5"],
        source,
    )
    assert args.decoder_lr == 1e-4


def test_post_init_validates_direct_construction() -> None:
    """The single-encoding property: building an invalid TrainArgs
    directly raises the same message the CLI would print."""
    args = _parse(["--family", "gemma_flow"])
    with pytest.raises(ValueError, match="--holdout-episodes must be in"):
        dataclasses.replace(args, holdout_episodes=1.5)
    with pytest.raises(ValueError, match="unknown family 'ar_fast'"):
        dataclasses.replace(args, family="ar_fast")
    with pytest.raises(ValueError, match="requires --fast-tokenizer"):
        dataclasses.replace(args, family="gemma_ar")
    with pytest.raises(ValueError, match="--rewarmup-steps anchors at the resume"):
        dataclasses.replace(args, rewarmup_steps=100)
    with pytest.raises(ValueError, match="template order"):
        dataclasses.replace(
            args,
            family="gemma_ar",
            fast_tokenizer="t",
            aux_fields=("holding", "subgoal"),
        )
    with pytest.raises(ValueError, match="molmo_flow KV seam"):
        dataclasses.replace(args, insulate_flow=True)
    with pytest.raises(ValueError, match="trains from a checkpoint only"):
        dataclasses.replace(args, family="molmoact2_flow")
    with pytest.raises(ValueError, match="builds no such"):
        dataclasses.replace(args, flow_decoder_init="fresh")
    with pytest.raises(ValueError, match="requires the molmoact2_joint family"):
        dataclasses.replace(args, joint_ce_weight=0.5)


def test_value_invariants_reach_cli_as_parser_error() -> None:
    with pytest.raises(SystemExit):  # parser.error(str(ValueError))
        _parse(["--family", "gemma_flow", "--holdout-episodes", "1.5"])
    with pytest.raises(SystemExit):  # retired kind: argparse choices
        _parse(["--family", "ar_fast"])


def test_resume_and_init_from_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        _parse(["--resume", "a", "--init-from", "b"], _checkpoint())


def test_dropout_requires_enabler_pre_resolution() -> None:
    with pytest.raises(SystemExit):
        _parse(["--family", "gemma_flow", "--field-dropout", "0.2"])  # no --aux-fields
    args = _parse(["--family", "gemma_flow"])
    assert args.field_dropout == 0.0  # resolved conditional default


def test_train_args_serializes_resolved_values() -> None:
    """Checkpoint metadata records what the run actually trained with:
    a resumed run's asdict carries the checkpoint's architecture, so the
    NEXT resume resolves identically (the recorded train_args parse
    through the same CheckpointTrainArgs reader either way)."""
    info = _checkpoint()
    args = _parse(["--resume", "ckpt"], info)
    record = dataclasses.asdict(args)
    assert record["family"] == "gemma_flow"
    assert record["decoder_hidden"] == 1024
    assert record["stream_counts"] == (4, 4, 4, 4, 6)
    reread = CheckpointTrainArgs.from_dict(
        {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in record.items()
            if v is None or isinstance(v, (str, int, float, bool, tuple))
        },
    )
    assert reread == dataclasses.replace(info.train_args, decoder="flow")


def test_molmoact2_families_are_inherit_only() -> None:
    """The molmoact2 families never construct fresh; they resolve from a
    converted checkpoint. The composition rules ride the resolved
    family."""
    with pytest.raises(SystemExit):  # fresh molmoact2 refused
        _parse(["--family", "molmoact2_flow"])
    info = _checkpoint(
        family=VLAFamily.MOLMOACT2_FLOW,
        decoder="molmo_flow",
        chunk_size=30,
        # Converted checkpoints record the placeholder defaults for the
        # flow-only knobs (the converter's synthesized train_args).
        time_conditioning=TimeConditioning.ADDITIVE,
        self_attention_mode=SelfAttentionMode.BIDIRECTIONAL,
    )
    args = _parse(["--resume", "ckpt"], info)
    assert args.family == "molmoact2_flow"
    assert args.chunk_size == 30
    assert args.insulate_flow is False
    insulated = _parse(["--resume", "ckpt", "--insulate-flow"], info)
    assert insulated.insulate_flow is True
    # Frozen-trunk insulation is their post-train; an unfrozen trunk
    # under flow-only insulation trains on nothing.
    with pytest.raises(SystemExit):
        _parse(
            ["--resume", "ckpt", "--insulate-flow", "--backbone-text-lr", "1e-5"],
            info,
        )
    with pytest.raises(SystemExit):  # bijou bracket surfaces have no bytes
        _parse(
            ["--resume", "ckpt", "--condition-fields", "outcome"],
            info,
        )
    with pytest.raises(SystemExit):  # the trunk IS the trainable surface
        _parse(["--init-from", "ckpt", "--objective", "ar"], info)


def test_flow_decoder_init_requires_init_from() -> None:
    source = _checkpoint(
        family=VLAFamily.MOLMOACT2_FLOW,
        decoder="molmo_flow",
        chunk_size=30,
        time_conditioning=TimeConditioning.ADDITIVE,
        self_attention_mode=SelfAttentionMode.BIDIRECTIONAL,
    )
    with pytest.raises(SystemExit):  # resume: weights come from the checkpoint
        _parse(["--resume", "ckpt", "--flow-decoder-init", "fresh"], source)
    with pytest.raises(SystemExit):  # fresh runs have nothing to warm-start
        _parse(["--family", "gemma_flow", "--flow-decoder-init", "fresh"])
    args = _parse(["--init-from", "ckpt", "--flow-decoder-init", "fresh"], source)
    assert args.flow_decoder_init == "fresh"


def test_paths_and_policy_flags_stay_cli_owned() -> None:
    info = _checkpoint()
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
    assert args.decoder_lr == 5e-5
    assert args.train_data == (Path("other_box_path"),)


def test_recompute_stats_scoping() -> None:
    """--recompute-stats: molmoact2 families with --init-from only —
    refused under --resume (locked run fact), on fresh runs (no source
    table), and on per-dataset-normalizing families."""
    molmoact2 = _checkpoint(
        family=VLAFamily.MOLMOACT2_JOINT,
        backbone="allenai/MolmoAct2-SO100_101",
        decoder="molmo_flow",
        chunk_size=30,
        time_conditioning=TimeConditioning.ADDITIVE,
        self_attention_mode=SelfAttentionMode.BIDIRECTIONAL,
        objective={"kind": "joint", "ce_weight": 1.0, "insulate_flow": False},
    )
    args = _parse(
        [
            "--init-from",
            "ckpt",
            "--recompute-stats",
            "--backbone-text-lr",
            "1e-5",
            "--save-dir",
            "out",
        ],
        molmoact2,
    )
    assert args.recompute_stats is True
    with pytest.raises(SystemExit):
        _parse(
            [
                "--resume",
                "ckpt",
                "--recompute-stats",
                "--backbone-text-lr",
                "1e-5",
                "--save-dir",
                "out",
            ],
            molmoact2,
        )
    with pytest.raises(SystemExit):
        _parse(
            ["--init-from", "ckpt", "--recompute-stats", "--save-dir", "out"],
            _checkpoint(),  # gemma_flow: per-dataset normalization
        )


def test_per_dataset_flow_norm_scoping() -> None:
    """--per-dataset-flow-norm: molmoact2 flow/joint only — refused on
    molmoact2_ar (no flow decoder), on gemma_flow (already per-dataset
    at collate time), and under --resume (the scheme is a recorded
    section fact, inherited)."""
    joint = _checkpoint(
        family=VLAFamily.MOLMOACT2_JOINT,
        backbone="allenai/MolmoAct2-SO100_101",
        decoder="molmo_flow",
        chunk_size=30,
        time_conditioning=TimeConditioning.ADDITIVE,
        self_attention_mode=SelfAttentionMode.BIDIRECTIONAL,
        objective={"kind": "joint", "ce_weight": 1.0, "insulate_flow": False},
    )
    args = _parse(
        [
            "--init-from",
            "ckpt",
            "--per-dataset-flow-norm",
            "--backbone-text-lr",
            "1e-5",
            "--save-dir",
            "out",
        ],
        joint,
    )
    assert args.per_dataset_flow_norm is True
    # Composes with --recompute-stats (CE/state stay merged-recomputed).
    args = _parse(
        [
            "--init-from",
            "ckpt",
            "--per-dataset-flow-norm",
            "--recompute-stats",
            "--backbone-text-lr",
            "1e-5",
            "--save-dir",
            "out",
        ],
        joint,
    )
    assert args.per_dataset_flow_norm is True and args.recompute_stats is True
    with pytest.raises(SystemExit):
        _parse(
            [
                "--resume",
                "ckpt",
                "--per-dataset-flow-norm",
                "--backbone-text-lr",
                "1e-5",
                "--save-dir",
                "out",
            ],
            joint,
        )
    with pytest.raises(SystemExit):
        _parse(
            [
                "--init-from",
                "ckpt",
                "--per-dataset-flow-norm",
                "--objective",
                "ar",
                "--backbone-text-lr",
                "1e-5",
                "--save-dir",
                "out",
            ],
            joint,
        )
    with pytest.raises(SystemExit):
        _parse(
            ["--init-from", "ckpt", "--per-dataset-flow-norm", "--save-dir", "out"],
            _checkpoint(),  # gemma_flow: already per-dataset at collate
        )
