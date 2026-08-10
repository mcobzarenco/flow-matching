"""G1 parity — our action-expert port vs their HF remote-code module.

Pre-reg posts/2026-08-10-prereg-molmoact2-firstclass-port.md, gate G1:
load the SAME real checkpoint weights into (a) their HF ``ActionExpert``
(the remote-code class shipped inside the converted checkpoint dir) and
(b) ``bijou.molmoact2.ActionExpert``, drive both with identical random
inputs, and compare outputs. CPU/fp32 is the deterministic rung (kernel
noise only); ``--device cuda --dtype bfloat16`` measures the deployed-
dtype gap the pre-reg budgets at <= 1e-2.

Weights come from the fontaine-checkpoints delta artifact (the full
``model.action_expert.*`` tensor set) or any converted HF dir.

Usage:
    uv run python fontaine/scripts/molmoact2_ae_parity.py \
        [--weights ~/checkpoints/upload_molmoact2-so101-rig-r1-step2000/model-delta.safetensors] \
        [--hf-dir ~/checkpoints/molmoact2-so101-rig-r1-step2000-hf] \
        [--device cpu] [--dtype float32] [--tolerance 1e-4]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

PREFIX = "model.action_expert."


def load_ae_tensors(weights: Path) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    out: dict[str, torch.Tensor] = {}
    with safe_open(weights, framework="pt") as sf:
        for key in sf.keys():  # noqa: SIM118 — safetensors handle, not a dict
            if key.startswith(PREFIX):
                out[key] = sf.get_tensor(key)
    if not out:
        raise SystemExit(f"no {PREFIX}* tensors in {weights}")
    return out


def build_theirs(hf_dir: Path, ae_state: dict[str, torch.Tensor]):  # noqa: ANN201
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cls = get_class_from_dynamic_module("modeling_molmoact2.ActionExpert", str(hf_dir))
    cfg_cls = get_class_from_dynamic_module(
        "configuration_molmoact2.MolmoAct2ActionExpertConfig",
        str(hf_dir),
    )
    top = json.loads((hf_dir / "config.json").read_text())
    ae_cfg = dict(top["action_expert_config"])
    ae_cfg.pop("model_type", None)
    text_cfg = top["text_config"]
    llm_dim = text_cfg["hidden_size"]
    llm_kv_dim = text_cfg["head_dim"] * text_cfg["num_key_value_heads"]
    cfg = cfg_cls(
        **ae_cfg,
        max_action_dim=top["max_action_dim"],
        max_action_horizon=top["max_action_horizon"],
    )
    theirs = cls(
        cfg,
        llm_dim=llm_dim,
        llm_kv_dim=llm_kv_dim,
        llm_num_layers=ae_cfg["num_layers"],
    )
    stripped = {k[len(PREFIX) :]: v for k, v in ae_state.items()}
    missing, unexpected = theirs.load_state_dict(stripped, strict=False)
    if unexpected:
        raise SystemExit(f"their module rejected keys: {unexpected[:5]}")
    if missing:
        raise SystemExit(f"their module missing keys: {missing[:5]}")
    return theirs, llm_kv_dim


def build_ours(hf_dir: Path, ae_state: dict[str, torch.Tensor], llm_kv_dim: int):  # noqa: ANN201
    from bijou.molmoact2 import ActionExpertConfig, load_action_expert_state

    top = json.loads((hf_dir / "config.json").read_text())
    ae_cfg = top["action_expert_config"]
    cfg = ActionExpertConfig(
        max_horizon=top["max_action_horizon"],
        max_action_dim=top["max_action_dim"],
        hidden_size=ae_cfg["hidden_size"],
        num_layers=ae_cfg["num_layers"],
        num_heads=ae_cfg["num_heads"],
        mlp_ratio=ae_cfg["mlp_ratio"],
        ffn_multiple_of=ae_cfg["ffn_multiple_of"],
        timestep_embed_dim=ae_cfg["timestep_embed_dim"],
        context_layer_norm=ae_cfg["context_layer_norm"],
        qk_norm=ae_cfg["qk_norm"],
        qk_norm_eps=ae_cfg["qk_norm_eps"],
        rope=ae_cfg["rope"],
        causal_attn=ae_cfg["causal_attn"],
    )
    ours = cfg.build(llm_kv_dim=llm_kv_dim)
    load_action_expert_state(ours, ae_state)
    return ours


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default=str(
            Path(
                "~/checkpoints/upload_molmoact2-so101-rig-r1-step2000/model-delta.safetensors",
            ).expanduser(),
        ),
    )
    parser.add_argument(
        "--hf-dir",
        default=str(
            Path("~/checkpoints/molmoact2-so101-rig-r1-step2000-hf").expanduser(),
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--ctx-len", type=int, default=17)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="max-abs output gap allowed (CPU/fp32 rung; the bf16 GPU "
        "rung uses the pre-reg 1e-2 budget)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    hf_dir = Path(args.hf_dir)
    ae_state = load_ae_tensors(Path(args.weights))

    theirs, llm_kv_dim = build_theirs(hf_dir, ae_state)
    ours = build_ours(hf_dir, ae_state, llm_kv_dim)
    theirs = theirs.to(device=device, dtype=dtype).eval()
    ours = ours.to(device=device, dtype=dtype).eval()
    horizon = json.loads((hf_dir / "config.json").read_text())["max_action_horizon"]
    action_dim = json.loads((hf_dir / "config.json").read_text())["max_action_dim"]
    n_layers = len(ours.blocks)

    worst = 0.0
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        actions = torch.randn(
            args.batch,
            horizon,
            action_dim,
            device=device,
            dtype=dtype,
        )
        timesteps = torch.rand(args.batch, device=device, dtype=dtype)
        kv_states = [
            (
                torch.randn(
                    args.batch,
                    args.ctx_len,
                    llm_kv_dim,
                    device=device,
                    dtype=dtype,
                ),
                torch.randn(
                    args.batch,
                    args.ctx_len,
                    llm_kv_dim,
                    device=device,
                    dtype=dtype,
                ),
            )
            for _ in range(n_layers)
        ]
        enc_mask = torch.ones(args.batch, args.ctx_len, device=device)
        enc_mask[:, -3:] = 0
        with torch.no_grad():
            out_theirs = theirs(
                actions,
                timesteps,
                encoder_kv_states=kv_states,
                encoder_attention_mask=enc_mask,
            )
            out_ours = ours(
                actions,
                timesteps,
                kv_states,
                encoder_attention_mask=enc_mask,
            )
        gap = (out_theirs.float() - out_ours.float()).abs().max().item()
        scale = out_theirs.float().abs().max().item()
        worst = max(worst, gap)
        print(f"seed {seed}: max|Δ| {gap:.3e} (out scale {scale:.3f})", flush=True)

    verdict = "PASS" if worst <= args.tolerance else "FAIL"
    print(
        f"G1 parity {verdict}: worst max|Δ| {worst:.3e} vs tolerance "
        f"{args.tolerance:g} ({args.device}/{args.dtype}, {args.seeds} seeds, "
        f"real weights {Path(args.weights).name})",
    )
    if verdict == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
