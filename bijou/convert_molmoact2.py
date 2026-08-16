"""Convert a MolmoAct2 HF checkpoint into a bijou VLA checkpoint.

Conversion-first loading (architecture.md §8.13): runtime
never reads their HF layout — this CLI materializes a normal VLA
checkpoint directory (``bijou/checkpoint.py``) once, and everything
downstream (`--init-from`, eval, rollout) consumes it like any other
checkpoint:

- ``metadata.json`` (schema 2): family per ``--family`` (flow, ar or
  joint — the release trained BOTH heads; rig-ft 'continuous' exports
  refuse ar/joint), prompt
  component (``MolmoAct2PromptConfig``: their template facts, discrete
  state, the ``action_mode`` mask flavor — load-bearing for the expert
  weights; parameterless — ``weights: false``), flow_decoder component
  (``MolmoFlowDecoderConfig``: expert geometry + flow parameters + the
  real action geometry of the norm tag), objective ``flow``, serving
  ``euler`` at the recorded step count, their ``config.json`` contents
  VERBATIM in the backbone section (parsed at load with
  ``Molmo2Config.from_dict``), and the tag's merged stats table
  verbatim (their mean/std/q01/q99 rows, unfloored — the q01/q99
  fields ARE the decoder-owned clamp table, stored once).
- ``backbone_text.safetensors`` + ``backbone_vision.safetensors``: the
  trunk IMPORTED wholesale (our key names, bytes/dtypes verbatim —
  ``modelling.molmo2.loading.import_backbone_state``, whose audit
  proves text + vision + expert + known-skipped exactly cover every
  source shard key). Both trained flags False: their trunk IS this
  checkpoint's pristine reference.
- ``tokenizer/``: their ``tokenizer.json``, hard-linked (all the Molmo
  prompt path reads).
- ``flow_decoder.safetensors``: the 588 ``model.action_expert.*``
  tensors, prefix-stripped, bytes verbatim (the parity oracle
  byte-compares all three artifacts: their export, the port, the
  decoder). The compat tensors their loader injects (identity
  ``state_encoder``, zero ``kv_proj``) are deliberately NOT
  materialized — injection is the loader's job, matching their own
  export convention.
- ``train_args.converted_from``: provenance (source ref, file digests,
  tensor census) inside the free-form provenance record. No timestamps
  — conversion is deterministic (same source → content-identical
  output; an existing destination is refused, checkpoints being
  immutable once published).

The P2 guards run at convert time, loudly: ``n_obs_steps`` present and
1, every dropout-like expert key 0.0, a supported ``action_mode``, the
setup/control prompt strings non-empty (via ``load_norm_stats``).

The recorded backbone id is provenance only — loading is
self-contained through the imported part files and ``tokenizer/``.
``--backbone-ref`` records a hub id when converting from a local
export directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from safetensors import safe_open
from torch import Tensor

from .checkpoint import (
    VLAMetadata,
    read_metadata,
    tokenizer_manifest,
    validate_checkpoint,
)
from .checkpoint import write_checkpoint as write_vla_checkpoint
from .data import DatasetStats
from .loading import (
    BackboneDepth,
    MolmoAct2PromptConfig,
    MolmoFlowDecoderConfig,
)
from .modelling.encoders.molmoact2 import MOLMOACT2_PROMPT_FORMAT
from .modelling.encoders.molmoact2_processing import (
    load_norm_stats,
    require_single_obs,
    validate_inference_config,
)
from .modelling.gemma4.loading import resolve_checkpoint_dir
from .modelling.molmo2.loading import import_backbone_state
from .sections import (
    MOLMOACT2_FAST_TOKENIZER_REF,
    ar_backbone_config_to_dict,
    molmoact2_ar_config_from_flow_section,
)
from .vla import VLAFamily


def _validate_source_config(config: dict[str, Any]) -> None:
    """The P2 guards, at convert time: everything the port refuses at
    load is refused here, before any artifact exists."""
    validate_inference_config(config)
    require_single_obs(config)
    for key, value in config["action_expert_config"].items():
        if "dropout" in key and float(value) != 0.0:
            raise SystemExit(
                f"source action_expert_config.{key}={value}: nonzero expert "
                "dropout is not wired (released/rig checkpoints ship 0.0)",
            )
    if not (config.get("add_setup_tokens") and config.get("add_control_tokens")):
        raise SystemExit(
            "add_setup_tokens/add_control_tokens=false prompts are not wired "
            "(the only shipped configuration wraps both)",
        )
    if str(config.get("state_format")) != "discrete":
        raise SystemExit(
            f"state_format={config.get('state_format')!r} is not wired "
            "(their serving stack embeds state as discrete prompt tokens)",
        )
    cutoff = float(config.get("flow_matching_cutoff", 1.0))
    if cutoff != 1.0:
        raise SystemExit(
            f"flow_matching_cutoff={cutoff} is not wired (released/rig "
            "checkpoints ship 1.0; a truncated t-law would change the "
            "training objective silently)",
        )


def _stats_vector(stats: dict[str, Any], field: str, *, what: str) -> tuple[float, ...]:
    values = stats.get(field)
    if values is None:
        raise SystemExit(
            f"norm_stats {what} carries no {field!r} row — the converted "
            "normalization table stores their rows verbatim; refusing to "
            "synthesize",
        )
    return tuple(float(v) for v in values)


def _dataset_stats(tag_metadata: dict[str, Any], action_dim: int) -> DatasetStats:
    """The tag's merged table as a DatasetStats row, VERBATIM (no std
    flooring — this is their table, not a data-path fit; its q01/q99
    fields are the molmo_flow clamp table). Their vectors
    are max_action_dim-padded; the real ``action_dim`` prefix is the
    recorded geometry."""
    action = tag_metadata["action_stats"]
    state = tag_metadata["state_stats"]
    return DatasetStats(
        action_mean=_stats_vector(action, "mean", what="action_stats")[:action_dim],
        action_std=_stats_vector(action, "std", what="action_stats")[:action_dim],
        state_mean=_stats_vector(state, "mean", what="state_stats"),
        state_std=_stats_vector(state, "std", what="state_stats"),
        action_q01=_stats_vector(action, "q01", what="action_stats")[:action_dim],
        action_q99=_stats_vector(action, "q99", what="action_stats")[:action_dim],
        state_q01=_stats_vector(state, "q01", what="state_stats"),
        state_q99=_stats_vector(state, "q99", what="state_stats"),
    )


def _tensor_digest(tensors: dict[str, Tensor]) -> str:
    """Order-independent digest over (name, bytes) — the byte-verification
    anchor the write side and the re-read side must both reproduce."""
    digest = hashlib.sha256()
    for name in sorted(tensors):
        digest.update(name.encode())
        tensor = tensors[name]
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.contiguous().view(-1).numpy().tobytes())
    return digest.hexdigest()


def _synthesized_train_args(
    decoder_config: MolmoFlowDecoderConfig,
) -> dict[str, Any]:
    """A CheckpointTrainArgs-parseable record for --init-from resolution
    (no run produced this checkpoint, so run-policy fields are absent —
    check_resume_seed warns-not-dies on the missing seed, and a resume
    of a conversion is meaningless anyway: there is no optimizer.pt).
    Placeholder fields follow the ar_backbone precedent (CLI defaults
    for knobs this decoder kind has no dial for)."""
    return {
        "decoder": "molmo_flow",
        "decoder_hidden": decoder_config.hidden_size,
        "decoder_heads": decoder_config.num_heads,
        "decoder_intermediate": _round_up(
            int(decoder_config.hidden_size * decoder_config.mlp_ratio),
            decoder_config.ffn_multiple_of,
        ),
        "decoder_cross_heads": decoder_config.num_heads,
        "stream_counts": [],
        "self_attention_mode": "bidirectional",
        "chunk_size": decoder_config.action_horizon,
        "max_soft_tokens": 140,
        "max_crops": 1,
        "time_conditioning": "additive",
        "target_time_embed": False,
        "fast_tokenizer": None,
        "joint_ce": False,
    }


def _round_up(value: int, multiple_of: int) -> int:
    return int(math.ceil(value / multiple_of) * multiple_of)


def convert(
    source: str,
    out: Path,
    *,
    norm_tag: str,
    backbone_ref: str | None,
    norm_stats_from: str | None = None,
    family: VLAFamily = VLAFamily.MOLMOACT2_FLOW,
    fast_tokenizer: str = MOLMOACT2_FAST_TOKENIZER_REF,
) -> Path:
    """Materialize the VLA checkpoint; returns ``out``. Deterministic
    (same source -> content-identical output; an existing ``out`` is
    refused — checkpoints are immutable once published).

    ``norm_stats_from`` loads the q01/q99 tables (and the tag's
    geometry/setup fields) from ANOTHER artifact's ``norm_stats.json``
    instead of the source's — their fine-tune recipe's semantics, where
    the table is RECOMPUTED on the target-domain data at fine-tune
    start (e.g. released weights under a rig table, exactly a rig
    fine-tune's starting point). Weights still come from ``source``;
    the substitution is recorded in ``converted_from`` and
    ``stats_note``."""
    source_dir = resolve_checkpoint_dir(source)
    config = json.loads((source_dir / "config.json").read_text())
    _validate_source_config(config)
    if not (source_dir / "tokenizer.json").exists():
        raise SystemExit(f"{source_dir} has no tokenizer.json")

    stats_dir = (
        resolve_checkpoint_dir(norm_stats_from)
        if norm_stats_from is not None
        else source_dir
    )
    if not (stats_dir / "norm_stats.json").exists():
        raise SystemExit(f"{stats_dir} has no norm_stats.json")
    _action_stats, _state_stats, tag = load_norm_stats(stats_dir, norm_tag)
    action_dim = int(tag.get("action_dim", 32))
    real_action_dim = len(tag["action_stats"]["q01"])
    ae_cfg = config["action_expert_config"]
    text_cfg = config["text_config"]
    decoder_config = MolmoFlowDecoderConfig(
        max_horizon=int(config["max_action_horizon"]),
        max_action_dim=int(config["max_action_dim"]),
        hidden_size=int(ae_cfg["hidden_size"]),
        num_layers=int(ae_cfg["num_layers"]),
        num_heads=int(ae_cfg["num_heads"]),
        mlp_ratio=float(ae_cfg["mlp_ratio"]),
        ffn_multiple_of=int(ae_cfg["ffn_multiple_of"]),
        timestep_embed_dim=int(ae_cfg["timestep_embed_dim"]),
        context_layer_norm=bool(ae_cfg["context_layer_norm"]),
        qk_norm=bool(ae_cfg["qk_norm"]),
        qk_norm_eps=float(ae_cfg["qk_norm_eps"]),
        rope=bool(ae_cfg["rope"]),
        causal_attn=bool(ae_cfg["causal_attn"]),
        llm_kv_dim=int(text_cfg["head_dim"]) * int(text_cfg["num_key_value_heads"]),
        num_flow_steps=int(config["flow_matching_num_steps"]),
        mask_action_dim_padding=bool(config["mask_action_dim_padding"]),
        action_dim=real_action_dim,
        action_horizon=int(tag["action_horizon"]),
        n_action_steps=int(tag["n_action_steps"]),
        normalization="q01q99",
        time_offset=float(config["flow_matching_time_offset"]),
        time_scale=float(config["flow_matching_time_scale"]),
        beta_alpha=float(config["flow_matching_beta_alpha"]),
        beta_beta=float(config["flow_matching_beta_beta"]),
    )
    if action_dim not in (real_action_dim, decoder_config.max_action_dim):
        raise SystemExit(
            f"norm tag action_dim {action_dim} matches neither the stats "
            f"width {real_action_dim} nor max_action_dim "
            f"{decoder_config.max_action_dim} — unrecognized layout",
        )
    prompt_config = MolmoAct2PromptConfig(
        format=MOLMOACT2_PROMPT_FORMAT,
        norm_tag=norm_tag,
        setup_type=str(tag["setup_type"]),
        control_mode=str(tag["control_mode"]),
        num_state_tokens=int(config["num_state_tokens"]),
        state_dim=len(tag["state_stats"]["q01"]),
        action_mode=str(config.get("action_mode", "continuous")),
        n_obs_steps=int(config["n_obs_steps"]),
        camera_keys=tuple(str(key) for key in tag.get("camera_keys", [])),
        # Their checkpoints never narrate; narration-on is a bijou
        # training choice recorded when a run makes it.
        narration=False,
    )

    recorded_backbone = backbone_ref if backbone_ref is not None else source

    # The full import: text + vision + expert + known-skipped must
    # exactly cover every source shard key (the importer's audit).
    imported = import_backbone_state(source_dir)
    expert = imported.expert
    if not expert:
        raise SystemExit(f"{source_dir} has no model.action_expert.* tensors")
    print(
        f"[convert] imported trunk: text ({len(imported.text)}) + vision "
        f"({len(imported.vision)}) tensors, expert ({len(expert)}), "
        f"skipped {list(imported.skipped)}",
    )
    expert_sha256 = _tensor_digest(expert)
    stats = _dataset_stats(tag, real_action_dim)
    train_args = _synthesized_train_args(decoder_config)
    train_args["converted_from"] = {
        "source": str(source),
        "source_config_sha256": hashlib.sha256(
            (source_dir / "config.json").read_bytes(),
        ).hexdigest(),
        "source_norm_stats_sha256": hashlib.sha256(
            (stats_dir / "norm_stats.json").read_bytes(),
        ).hexdigest(),
        **(
            {"norm_stats_from": str(norm_stats_from)}
            if norm_stats_from is not None
            else {}
        ),
        "expert_tensors": len(expert),
        "expert_sha256": expert_sha256,
        "converter": "bijou.convert_molmoact2",
    }
    if family not in (
        VLAFamily.MOLMOACT2_FLOW,
        VLAFamily.MOLMOACT2_AR,
        VLAFamily.MOLMOACT2_JOINT,
    ):
        raise SystemExit(f"--family {family.value} is not a MolmoAct2 family")
    if family is not VLAFamily.MOLMOACT2_FLOW and prompt_config.action_mode != "both":
        raise SystemExit(
            f"--family {family.value} needs a checkpoint whose discrete head "
            f"trained (action_mode='both'); this export records "
            f"{prompt_config.action_mode!r} — its <action_*> rows were never "
            "trained (the rig-ft 'continuous' class), so an AR/joint import "
            "would serve an untrained head",
        )
    components: dict[str, Any] = {
        # The prompt side owns zero parameters — config-only.
        "prompt": {"config": prompt_config.to_dict(), "weights": False},
        # The expert weights are release-trained; every family carries
        # them (the AR family keeps them as inherited provenance — its
        # own decoder is parameterless, trunk-native rows).
        "flow_decoder": {"config": decoder_config.to_dict(), "weights": True},
    }
    if family is not VLAFamily.MOLMOACT2_FLOW:
        # The anchor verification inside this helper reads the REAL
        # source tokenizer (token_to_id on <action_*>), so it takes the
        # source directory, never the provenance ref.
        ar_config = molmoact2_ar_config_from_flow_section(
            decoder_config,
            prompt_config,
            str(source_dir),
            fast_tokenizer=fast_tokenizer,
        )
        components["ar_decoder"] = {
            "config": ar_backbone_config_to_dict(ar_config),
            "weights": False,
        }
    match family:
        case VLAFamily.MOLMOACT2_AR:
            objective: dict[str, Any] = {"kind": "ar"}
        case VLAFamily.MOLMOACT2_JOINT:
            # An IMPORT records the default continuation plan, not the
            # release's (unknown) training mixture — printed so nobody
            # mistakes it for a measured fact.
            objective = {"kind": "joint", "ce_weight": 1.0, "insulate_flow": False}
            print(
                "[convert] joint import: objective recorded as ce_weight=1.0, "
                "insulate_flow=False (the DEFAULT continuation plan; the "
                "release's own training mixture is not public)",
            )
        case _:
            objective = {"kind": "flow"}
    metadata = VLAMetadata(
        family=family,
        chunk_size=decoder_config.action_horizon,
        action_dim=real_action_dim,
        backbone_id=str(recorded_backbone),
        backbone_depth=BackboneDepth.FULL.value,
        backbone_config=config,
        backbone_text_trained=False,
        backbone_vision_trained=False,
        objective=objective,
        serving={
            "kind": "flow",
            "num_steps": decoder_config.num_flow_steps,
            "method": "euler",
        },
        components=components,
        artifacts={},
        stats=stats,
        per_dataset_stats={},
        train_args=train_args,
        step=0,
        stats_note=(
            f"norm table loaded from {norm_stats_from} (their fine-tune "
            "table-recompute semantics)"
            if norm_stats_from is not None
            else None
        ),
    )
    write_vla_checkpoint(
        out,
        metadata=metadata,
        components={"flow_decoder": expert},
        backbone_text=imported.text,
        backbone_vision=imported.vision,
        tokenizer_files={
            name: source_dir / name for name in tokenizer_manifest(family)
        },
    )

    # Self-verification (run on every conversion): the directory
    # validates as self-contained, the metadata round-trips, and the
    # written expert bytes reproduce the source digest.
    validate_checkpoint(out)
    reread = read_metadata(out)
    if reread.family is not family:
        raise SystemExit("round-trip failed: recorded family mismatch")
    with safe_open(
        out / "flow_decoder.safetensors",
        framework="pt",
        device="cpu",
    ) as f:
        written = {key: f.get_tensor(key) for key in f.keys()}  # noqa: SIM118
    if _tensor_digest(written) != expert_sha256:
        raise SystemExit("round-trip failed: written expert bytes differ from source")
    print(
        f"converted {source} [{norm_tag}] -> {out} as {family.value}: "
        f"{len(expert)} expert tensors "
        f"(sha256 {expert_sha256[:16]}...), "
        f"decoder {decoder_config.num_layers}x{decoder_config.hidden_size} "
        f"horizon {decoder_config.action_horizon}/{decoder_config.max_horizon} "
        f"action_dim {decoder_config.action_dim}/{decoder_config.max_action_dim}, "
        f"prompt action_mode={prompt_config.action_mode!r}, "
        f"backbone ref {recorded_backbone!r}",
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m bijou.convert_molmoact2",
        description="Convert a MolmoAct2 HF checkpoint (released or rig-ft "
        "export) into a bijou VLA checkpoint directory (architecture.md "
        "§8.13). The trunk imports wholesale into per-part files "
        "(backbone_text/backbone_vision, our key names) and tokenizer.json "
        "links into tokenizer/; the recorded backbone id stays the "
        "provenance reference.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="MolmoAct2 checkpoint: HF repo id (e.g. allenai/MolmoAct2-"
        "SO100_101) or a local export directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output bijou checkpoint directory (created if absent)",
    )
    parser.add_argument(
        "--family",
        choices=[
            VLAFamily.MOLMOACT2_FLOW.value,
            VLAFamily.MOLMOACT2_AR.value,
            VLAFamily.MOLMOACT2_JOINT.value,
        ],
        default=VLAFamily.MOLMOACT2_FLOW.value,
        help="which trained surface(s) the import serves: the release "
        "trained BOTH heads (action_mode='both'), so flow, ar and joint "
        "are all faithful; rig-ft 'continuous' exports refuse ar/joint "
        "(their discrete head never trained)",
    )
    parser.add_argument(
        "--fast-tokenizer",
        default=MOLMOACT2_FAST_TOKENIZER_REF,
        help="FAST tokenizer artifact ref recorded in the ar_decoder "
        "config (ar/joint families only)",
    )
    parser.add_argument(
        "--norm-tag",
        default="so100_so101_molmoact2",
        help="norm_stats.json tag to convert (their per-embodiment key)",
    )
    parser.add_argument(
        "--backbone-ref",
        default=None,
        help="backbone id recorded in the checkpoint as PROVENANCE "
        "(default: --source as given; loading is self-contained and "
        "never resolves it)",
    )
    parser.add_argument(
        "--norm-stats-from",
        default=None,
        help="load the q01/q99 tables (norm_stats.json, selected by "
        "--norm-tag) from ANOTHER artifact instead of --source — their "
        "fine-tune recipe's table-recompute semantics (e.g. the released "
        "weights under a rig-ft export's rig table = exactly their "
        "fine-tune starting point; docs/molmoact2-retirement.md decision "
        "7). Weights still come from --source; recorded in "
        "converted_from.norm_stats_from",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert(
        args.source,
        args.out,
        norm_tag=args.norm_tag,
        backbone_ref=args.backbone_ref,
        norm_stats_from=args.norm_stats_from,
        family=VLAFamily(args.family),
        fast_tokenizer=args.fast_tokenizer,
    )


if __name__ == "__main__":
    main()
