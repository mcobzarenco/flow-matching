"""Convert a MolmoAct2 HF checkpoint into a bijou checkpoint (§8.13 step 2).

Conversion-first loading (architecture.md §8.13 decision 4): runtime
never reads their HF layout — this CLI materializes a normal bijou
checkpoint directory once, and everything downstream (`--init-from`,
eval, rollout) consumes it like any other checkpoint:

- ``bijou_config.json``: format-3 sections — backbone (a REFERENCE to
  the source artifact: their trunk IS this checkpoint's pristine
  backbone, so no 20 GB copy and no ``backbone.safetensors``; the same
  convention as every checkpoint's ``google/gemma-4-e2b-it`` ref),
  prompt (``MolmoAct2PromptConfig``: their template facts, discrete
  state, the ``action_mode`` mask flavor — load-bearing for the expert
  weights), decoder (``MolmoFlowDecoderConfig``: expert geometry +
  flow parameters + the real action geometry of the norm tag).
- ``normalization``: the tag's merged stats table verbatim (their
  mean/std/q01/q99 rows, unfloored — the q01/q99 fields ARE the
  decoder-owned clamp table, stored once; decision 6).
- ``expert.safetensors``: the 588 ``model.action_expert.*`` tensors,
  prefix-stripped, bytes verbatim (the parity oracle byte-compares all
  three artifacts: their export, the port, the step-3 decoder). The
  compat tensors their loader injects (identity ``state_encoder``,
  zero ``kv_proj``) are deliberately NOT materialized — injection is
  the loader's job, matching their own export convention.
- ``converted_from``: provenance (source ref, file digests, tensor
  census). No timestamps — conversion is deterministic and idempotent.

The P2 guards run at convert time, loudly: ``n_obs_steps`` present and
1, every dropout-like expert key 0.0, a supported ``action_mode``, the
setup/control prompt strings non-empty (via ``load_norm_stats``).

The tokenizer stays in the source artifact (the backbone ref carries
it — their tokenizer.json re-homes the image specials and holds the
state-token block). A LOCAL source path makes the converted checkpoint
non-portable; ``--backbone-ref`` records a hub id instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from .data import DatasetStats
from .encoders.molmoact2 import MOLMOACT2_PROMPT_FORMAT
from .gemma4.loading import resolve_checkpoint_dir
from .loading import (
    BackboneConfig,
    BackboneDepth,
    CheckpointMetadata,
    MolmoAct2PromptConfig,
    MolmoFlowDecoderConfig,
    read_checkpoint_info,
)
from .molmoact2.predictor import require_single_obs
from .molmoact2.processing import load_norm_stats
from .molmoact2.wiring import validate_inference_config

_EXPERT_PREFIX = "model.action_expert."


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
    fields are the molmo_flow clamp table, decision 6). Their vectors
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


def _extract_expert_tensors(source_dir: Path) -> dict[str, Tensor]:
    """The ``model.action_expert.*`` tensors, prefix-stripped, bytes and
    dtypes verbatim.

    Shapes:
    - returns: name -> tensor, exactly the source's 588 (released size)
    """
    tensors: dict[str, Tensor] = {}
    weight_files = sorted(source_dir.glob("*.safetensors"))
    if not weight_files:
        raise SystemExit(f"no *.safetensors files in {source_dir}")
    for weight_file in weight_files:
        with safe_open(weight_file, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118 — safetensors handle, not a dict
                if key.startswith(_EXPERT_PREFIX):
                    tensors[key[len(_EXPERT_PREFIX) :]] = f.get_tensor(key)
    if not tensors:
        raise SystemExit(f"{source_dir} has no {_EXPERT_PREFIX}* tensors")
    return tensors


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
) -> Path:
    """Materialize the bijou checkpoint; returns ``out``. Deterministic
    and idempotent (same source -> byte-identical output)."""
    source_dir = resolve_checkpoint_dir(source)
    config = json.loads((source_dir / "config.json").read_text())
    _validate_source_config(config)
    if not (source_dir / "tokenizer.json").exists():
        raise SystemExit(f"{source_dir} has no tokenizer.json")

    _action_stats, _state_stats, tag = load_norm_stats(source_dir, norm_tag)
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
        # training choice (§8.13 decision 7) recorded when a run makes it.
        narration=False,
    )

    recorded_backbone = backbone_ref if backbone_ref is not None else source
    if Path(recorded_backbone).is_dir():
        print(
            f"NOTE: recorded backbone ref {recorded_backbone!r} is a local "
            "path — the converted checkpoint is only loadable where that "
            "path exists (pass --backbone-ref with a hub id for a portable "
            "artifact)",
        )

    expert = _extract_expert_tensors(source_dir)
    expert_sha256 = _tensor_digest(expert)
    metadata = CheckpointMetadata(
        backbone=BackboneConfig(id=str(recorded_backbone), depth=BackboneDepth.FULL),
        prompt=prompt_config,
        decoder=decoder_config.to_dict(),
        normalization=_dataset_stats(tag, real_action_dim),
        per_dataset_normalization={},
        train_args=_synthesized_train_args(decoder_config),
        step=0,
    )
    payload = metadata.to_json_dict()
    payload["converted_from"] = {
        "source": str(source),
        "source_config_sha256": hashlib.sha256(
            (source_dir / "config.json").read_bytes(),
        ).hexdigest(),
        "source_norm_stats_sha256": hashlib.sha256(
            (source_dir / "norm_stats.json").read_bytes(),
        ).hexdigest(),
        "expert_tensors": len(expert),
        "expert_sha256": expert_sha256,
        "converter": "bijou.convert_molmoact2",
    }

    out.mkdir(parents=True, exist_ok=True)
    save_file(expert, str(out / "expert.safetensors"))
    (out / "bijou_config.json").write_text(json.dumps(payload, indent=2) + "\n")

    # Self-verification (the step-2 gates, run on every conversion):
    # the metadata round-trips through bijou.loading, and the written
    # expert bytes reproduce the source digest.
    info = read_checkpoint_info(out)
    if info.train_args.decoder != "molmo_flow":
        raise SystemExit("round-trip failed: train_args decoder kind mismatch")
    with safe_open(out / "expert.safetensors", framework="pt", device="cpu") as f:
        written = {key: f.get_tensor(key) for key in f.keys()}  # noqa: SIM118
    if _tensor_digest(written) != expert_sha256:
        raise SystemExit("round-trip failed: written expert bytes differ from source")
    print(
        f"converted {source} [{norm_tag}] -> {out}: "
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
        "export) into a bijou checkpoint directory (architecture.md §8.13 "
        "step 2). The trunk/tokenizer stay in the source artifact — the "
        "converted checkpoint records it as its backbone.",
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
        "--norm-tag",
        default="so100_so101_molmoact2",
        help="norm_stats.json tag to convert (their per-embodiment key)",
    )
    parser.add_argument(
        "--backbone-ref",
        default=None,
        help="backbone id recorded in the checkpoint (default: --source "
        "as given; pass a hub id when converting from a local dir to "
        "keep the artifact portable)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert(
        args.source,
        args.out,
        norm_tag=args.norm_tag,
        backbone_ref=args.backbone_ref,
    )


if __name__ == "__main__":
    main()
