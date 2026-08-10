"""Wiring parity — our backbone↔expert glue vs their executing HF code.

Pre-reg posts/2026-08-10-prereg-molmoact2-firstclass-port.md, item 1
remainder (the G1 companion for the wiring): drive THEIR shipped
``MolmoAct2ForConditionalGeneration`` action path — ``_extract_kv_states``,
``_get_encoder_attention_mask`` and the full ``generate_actions_from_inputs``
flow loop — unbound on a stub carrying their real remote-code
``ActionExpert`` (same checkpoint weights as ours), and byte-compare
against ``bijou.molmoact2.wiring`` on identical inputs. No 4B trunk is
loaded: the trunk enters both sides as the same random per-layer KV,
which is exactly the wiring seam this gate isolates.

Rungs (same convention as molmoact2_ae_parity.py): CPU/fp32
deterministic, ``--device cuda --dtype bfloat16`` for the deployed
dtype (pre-reg budget <= 1e-2 there).

Usage:
    uv run python fontaine/scripts/molmoact2_wiring_parity.py \
        [--weights .../model-delta.safetensors] [--hf-dir .../step2000-hf] \
        [--device cpu] [--dtype float32] [--seeds 3] [--tolerance 1e-4]
"""

from __future__ import annotations

import argparse
import inspect
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from molmoact2_ae_parity import build_ours, build_theirs, load_ae_tensors

# Methods their action path calls through ``self``; bound onto the stub
# so the unbound class methods execute against it.
_STUB_METHODS = (
    "_require_action_expert",
    "_get_encoder_attention_mask",
    "_depth_gate_from_condition",
    "_get_depth_token_mask",
    "_apply_depth_gate_to_layer_kv_states",
    "_resolve_action_horizon",
    "_run_action_flow_loop",
    "_extract_kv_states",
    "_cache_to_sequence",
)


def load_their_model_cls(hf_dir: Path):  # noqa: ANN201
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    # The action path (_extract_kv_states / generate_actions_from_inputs)
    # lives on the inner MolmoAct2Model, not the CausalLM wrapper.
    return get_class_from_dynamic_module(
        "modeling_molmoact2.MolmoAct2Model",
        str(hf_dir),
    )


def make_stub(model_cls, cfg, expert) -> SimpleNamespace:  # noqa: ANN001
    stub = SimpleNamespace(
        config=cfg,
        action_expert=expert,
        action_expert_depth_gate=None,
        action_cuda_graph_manager=None,
        _depth_gate_token_ids=[],
    )
    for name in _STUB_METHODS:
        attr = inspect.getattr_static(model_cls, name)
        if isinstance(attr, (classmethod, staticmethod)):
            setattr(stub, name, getattr(model_cls, name))
        else:
            setattr(stub, name, types.MethodType(getattr(model_cls, name), stub))
    stub._mask_action_dim_tensor = model_cls._mask_action_dim_tensor
    return stub


def check(label: str, gap: float, worst: dict[str, float]) -> None:
    worst[label] = max(worst.get(label, 0.0), gap)
    print(f"  {label}: max|Δ| {gap:.3e}", flush=True)


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
        help="max-abs gap allowed on the sampled chunk (CPU/fp32 rung; "
        "the bf16 GPU rung uses the pre-reg 1e-2 budget)",
    )
    args = parser.parse_args()

    from transformers import AutoConfig
    from transformers.cache_utils import DynamicCache

    from bijou.molmo2.cache import Molmo2KVCache
    from bijou.molmoact2 import (
        encoder_attention_mask,
        extract_kv_states,
        generate_actions,
        validate_inference_config,
    )

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    hf_dir = Path(args.hf_dir)

    cfg = AutoConfig.from_pretrained(hf_dir, trust_remote_code=True)
    validate_inference_config(cfg.to_dict())
    model_cls = load_their_model_cls(hf_dir)
    ae_state = load_ae_tensors(Path(args.weights))
    theirs_ae, llm_kv_dim = build_theirs(hf_dir, ae_state)
    ours_ae = build_ours(hf_dir, ae_state, llm_kv_dim)
    theirs_ae = theirs_ae.to(device=device, dtype=dtype).eval()
    ours_ae = ours_ae.to(device=device, dtype=dtype).eval()
    stub = make_stub(model_cls, cfg, theirs_ae)

    n_layers = cfg.action_expert_config.num_layers
    n_heads = cfg.text_config.num_attention_heads
    n_kv_heads = cfg.text_config.num_key_value_heads
    head_dim = cfg.text_config.head_dim
    steps = int(cfg.flow_matching_num_steps)
    action_dim = int(cfg.max_action_dim)
    worst: dict[str, float] = {}

    for seed in range(args.seeds):
        print(f"seed {seed}:", flush=True)
        torch.manual_seed(seed)

        # --- KV extraction: their HF-Cache path vs ours off Molmo2KVCache.
        layer_kv = [
            (
                torch.randn(
                    args.batch,
                    n_kv_heads,
                    args.ctx_len,
                    head_dim,
                    device=device,
                    dtype=dtype,
                ),
                torch.randn(
                    args.batch,
                    n_kv_heads,
                    args.ctx_len,
                    head_dim,
                    device=device,
                    dtype=dtype,
                ),
            )
            for _ in range(n_layers)
        ]
        hf_cache = DynamicCache()
        our_cache = Molmo2KVCache(num_layers=n_layers)
        for idx, (k, v) in enumerate(layer_kv):
            hf_cache.update(k, v, idx)
            our_cache.update(idx, k, v)
        our_cache.advance(args.ctx_len)
        kv_theirs = stub._extract_kv_states(hf_cache)
        kv_ours = extract_kv_states(
            our_cache,
            num_expert_blocks=n_layers,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
        )
        gap = max(
            max(
                (kt - ko).abs().max().item(),
                (vt - vo).abs().max().item(),
            )
            for (kt, vt), (ko, vo) in zip(kv_theirs, kv_ours, strict=True)
        )
        check("kv_extraction", gap, worst)

        # --- Encoder mask: attention_mask, input_ids-fallback branches.
        input_ids = torch.randint(0, 1000, (args.batch, args.ctx_len), device=device)
        input_ids[:, -2:] = -1
        attn = torch.ones(args.batch, args.ctx_len, device=device)
        attn[:, -3:] = 0
        for label, ids_arg, mask_arg in (
            ("enc_mask_attn", input_ids, attn),
            ("enc_mask_ids", input_ids, None),
        ):
            mask_theirs = stub._get_encoder_attention_mask(ids_arg, mask_arg)
            mask_ours = encoder_attention_mask(ids_arg, mask_arg)
            same = torch.equal(mask_theirs, mask_ours)
            check(label, 0.0 if same else 1.0, worst)

        # --- Full flow loop on the real expert weights: their shipped
        # generate_actions_from_inputs vs our generate_actions, same
        # noise (fresh same-seeded generator each side).
        enc_mask = encoder_attention_mask(input_ids, attn)
        pad = torch.zeros(action_dim, dtype=torch.bool, device=device)
        pad[6:] = True  # SO-101: 6 real joints of 32
        chunk_theirs = model_cls.generate_actions_from_inputs(
            stub,
            input_ids=None,
            encoder_kv_states=[(k.clone(), v.clone()) for k, v in kv_theirs],
            encoder_attention_mask=enc_mask.clone(),
            action_dim_is_pad=pad,
            generator=torch.Generator(device=device).manual_seed(seed + 100),
        )
        chunk_ours = generate_actions(
            ours_ae,
            encoder_kv_states=kv_ours,
            encoder_attention_mask=enc_mask,
            action_dim_is_pad=pad,
            num_steps=steps,
            generator=torch.Generator(device=device).manual_seed(seed + 100),
        )
        check(
            "flow_loop",
            (chunk_theirs.float() - chunk_ours.float()).abs().max().item(),
            worst,
        )

    overall = max(worst.values())
    verdict = "PASS" if overall <= args.tolerance else "FAIL"
    detail = ", ".join(f"{k} {v:.3e}" for k, v in worst.items())
    print(
        f"wiring parity {verdict}: worst max|Δ| {overall:.3e} vs tolerance "
        f"{args.tolerance:g} ({args.device}/{args.dtype}, {args.seeds} seeds, "
        f"{steps}-step loop, real weights) [{detail}]",
    )
    if verdict == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
