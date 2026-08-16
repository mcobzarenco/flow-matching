"""Grad-flow verification for the unfreeze flags (tiny backbone, CPU).

Asserts, for --text-lr > 0 (vision frozen), with BOTH decoders:
- gradients reach a full text layer, the stop layer's K/V projections,
  the PLE *projection*, and the multimodal projector;
- the stop layer's never-run submodules (q_proj) stay frozen;
- token embeddings, PLE *tables* and the vision tower get NO gradients;
- EVERY requires_grad=True parameter receives a finite gradient (the
  exactness of the partition is what makes DDP static_graph safe).

Then for --vision-lr > 0 (flow decoder): tower gradients appear, and the
all-trainable-params-have-grads invariant still holds (this is the mode
where gradients must traverse the frozen embedding scatter).

Flags-on loss oracles (seed 0, 2 items of the oracle corpus) are
ASSERTED in main() — the asserts are the single source of truth for
the anchors (numbers duplicated into prose rot: this file's own
docstring and architecture.md §5 both carried stale values at the
2026-08-05 re-run). Corpus/format changes re-baseline them loudly.
Re-anchored 2026-08-15 on the family classes (GemmaFlowVLA /
GemmaARVLA) when the old composition root retired: the family forward
composes the same kernels in the same op order, and both anchors
reproduced at the swap.

AR differences from flow, asserted here: the AR decoder has NO zero-init
output head (lm_head is normal-init), so backbone gradients flow from step 1
without the warm-start hack the flow arm needs; token_embedding/lm_head
must receive gradients (DDP participation).

Run from the repo root: uv run python -m probes.probe_unfreeze_gradflow
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers

from bijou.data import EpisodeSplit, select_datasets
from bijou.loading import (
    BackboneDepth,
    build_gemma_encoder,
    build_gemma_flow_parts,
    default_expert_config,
)
from bijou.modelling.aux_text import SUFFIX_FORMAT
from bijou.modelling.codecs import FastActionCodec
from bijou.modelling.decoders.ar_gemma import GemmaARDecoder
from bijou.modelling.decoders.ar_suffix import ARDecoderConfig
from bijou.modelling.encoders.gemma4 import GemmaInputsCollator
from bijou.modelling.gemma4.loading import load_config
from bijou.modelling.gemma4.text import DecoderLayer
from bijou.modelling.interface import CollatedBatch, Collator, SamplingMethod
from bijou.models.gemma_ar import GemmaARVLA
from bijou.models.gemma_flow import GemmaFlowVLA
from bijou.models.objectives import ARObjective, FlowObjective
from bijou.models.serving import ARServing, FlowServing
from bijou.vla import VLA

TINY = "outputs/tiny-gemma4"
# The oracle corpus (2026-08-05, PR #1): rig v2 at its standard mirror
# path — present on every box AND the laptop, unlike the retired
# laptop-only community_dataset_v1_v3.
DATA = Path.home() / "datasets/mcobzarenco/so101_pick_place_v2"
FIXTURE_TOKENIZER = Path("tests/fixtures/tiny_fast_tokenizer")
EXPORTS = (1, 3, 5)  # tiny backbone's global layers


@dataclass(frozen=True, slots=True)
class ProbeArgs:
    """The slice of the historical train config this probe consumes."""

    seed: int
    stream_counts: tuple[int, ...]
    decoder_hidden: int
    decoder_heads: int
    decoder_intermediate: int
    decoder_cross_heads: int
    chunk_size: int
    max_soft_tokens: int
    backbone_text_lr: float | None
    backbone_vision_lr: float | None

    @property
    def backbone_trained(self) -> bool:
        return self.backbone_text_lr is not None or self.backbone_vision_lr is not None


def probe_unfreeze(model: VLA[Any], args: ProbeArgs) -> None:
    """The historical unfreeze: flip requires_grad on the requested
    backbone subsets off the model's named groups."""
    groups = model.param_groups()
    if args.backbone_text_lr is not None:
        for parameter in groups["backbone_text"]:
            parameter.requires_grad_(True)
    if args.backbone_vision_lr is not None:
        for parameter in groups["backbone_vision"]:
            parameter.requires_grad_(True)


def family_loss(model: VLA[Any], batch: CollatedBatch) -> torch.Tensor:
    """One single-process forward through the family's own precision
    policy: counts pass through un-reduced, the objective is the graph
    scalar (family.forward decides live-trunk encode from requires_grad,
    exactly the flags this probe just flipped)."""
    counts = model.loss_counts(batch)
    return model(batch, counts=counts).objective


def make_args(
    backbone_text_lr: float | None,
    backbone_vision_lr: float | None,
) -> ProbeArgs:
    return ProbeArgs(
        seed=0,
        stream_counts=(1, 1, 2),
        decoder_hidden=64,
        decoder_heads=2,
        decoder_intermediate=128,
        decoder_cross_heads=2,
        chunk_size=50,
        max_soft_tokens=140,
        backbone_text_lr=backbone_text_lr,
        backbone_vision_lr=backbone_vision_lr,
    )


def build_flow(args: ProbeArgs) -> GemmaFlowVLA:
    torch.manual_seed(args.seed)
    config = load_config(Path(TINY))
    expert_config = default_expert_config(
        config,
        action_dim=6,
        state_dim=6,
        stream_counts=args.stream_counts,
        hidden_size=args.decoder_hidden,
        num_attention_heads=args.decoder_heads,
        intermediate_size=args.decoder_intermediate,
        cross_attention_heads=args.decoder_cross_heads,
        chunk_size=args.chunk_size,
    )
    backbone, encoder, decoder = build_gemma_flow_parts(
        Path(TINY),
        config,
        expert_config,
        max_soft_tokens=args.max_soft_tokens,
        device="cpu",
        dtype=torch.float32 if args.backbone_trained else None,
        expert_dtype=torch.float32,
    )
    # A fresh flow expert's action_out_proj is ZERO-initialized, which
    # blocks all upstream gradients on the very first step (dL/dhidden =
    # W^T d = 0 - grads populate as zeros, so DDP participation is still
    # fine, and step 2 onward flows). The real unfreeze path warm-starts
    # from a trained checkpoint; simulate that here (seeded).
    torch.nn.init.normal_(decoder.action_out_proj.weight, std=0.02)
    model = GemmaFlowVLA(
        backbone,
        encoder,
        decoder,
        objective=FlowObjective(),
        serving=FlowServing(num_steps=5, method=SamplingMethod.HEUN),
    )
    probe_unfreeze(model, args)
    return model


def build_ar_backbone(args: ProbeArgs) -> GemmaARVLA:
    torch.manual_seed(args.seed)
    backbone_config = load_config(Path(TINY))
    codec = FastActionCodec.load(FIXTURE_TOKENIZER)
    stop = backbone_config.text.first_kv_shared_layer_idx - 1
    backbone, encoder = build_gemma_encoder(
        Path(TINY),
        backbone_config,
        exports=(stop,),
        max_soft_tokens=args.max_soft_tokens,
        state_dim=6,
        device="cpu",
        dtype=torch.float32 if args.backbone_trained else None,
        depth=BackboneDepth.FULL,
    )
    decoder = GemmaARDecoder(
        ARDecoderConfig(
            tokenizer=str(FIXTURE_TOKENIZER),
            vocab_total=codec.vocab_total,
            block_base=backbone_config.text.vocab_size - codec.vocab_total,
            chunk_size=args.chunk_size,
            action_dim=6,
            suffix_format=SUFFIX_FORMAT,
            aux=None,
        ),
        backbone_config.text,
        codec,
        tokenizer=transformers.AutoTokenizer.from_pretrained(TINY),
        device="cpu",
        dtype=torch.float32,
    )
    decoder.init_tables_from_backbone(backbone)
    model = GemmaARVLA(
        backbone,
        encoder,
        decoder,
        objective=ARObjective(narration_weight=1.0),
        serving=ARServing(),
    )
    probe_unfreeze(model, args)
    return model


def grad_state(parameter: torch.Tensor) -> str:
    # Accepts Tensor: torch's Linear.weight stubs widen Parameter to
    # Tensor, and .grad lives on Tensor anyway.
    if parameter.grad is None:
        return "none"
    if not bool(parameter.grad.isfinite().all()):
        return "nonfinite"
    if bool((parameter.grad == 0).all()):
        return "zero"
    return "nonzero"


def check_text_partition(model: GemmaFlowVLA | GemmaARVLA, label: str) -> None:
    """The shared text-only assertions (both families)."""
    backbone = model.backbone
    # The tiny trunk is multimodal — both vision halves exist.
    assert backbone.vision_tower is not None
    assert backbone.embed_vision is not None
    text = backbone.language_model
    stop = max(EXPORTS)
    layers = list(text.layers)
    stop_layer = layers[stop]
    assert isinstance(stop_layer, DecoderLayer)
    assert stop_layer.self_attn.k_proj is not None
    full_layer = layers[0]
    assert isinstance(full_layer, DecoderLayer)
    assert full_layer.self_attn.k_proj is not None

    checks = {
        "layer0 k_proj": grad_state(full_layer.self_attn.k_proj.weight) == "nonzero",
        "layer0 mlp": grad_state(full_layer.mlp.down_proj.weight) == "nonzero",
        f"stop({stop}) k_proj": grad_state(stop_layer.self_attn.k_proj.weight)
        == "nonzero",
        f"stop({stop}) q_proj frozen": stop_layer.self_attn.q_proj.weight.grad is None,
        "ple projection": grad_state(text.per_layer_model_projection.weight)
        == "nonzero",
        "embed_vision": grad_state(
            next(iter(backbone.embed_vision.parameters())),
        )
        == "nonzero",
        "embed_tokens frozen": text.embed_tokens.weight.grad is None,
        "ple tables frozen": text.embed_tokens_per_layer.weight.grad is None,
        "vision tower frozen": next(
            iter(backbone.vision_tower.parameters()),
        ).grad
        is None,
    }
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    bad = [(n, grad_state(p)) for n, p in trainable if grad_state(p) != "nonzero"]
    checks["ALL trainable have nonzero finite grads"] = not bad
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    if bad:
        print(f"  offending params: {bad[:10]}")
    assert all(checks.values()), f"{label} grad-flow FAILED"


def flow_batch(items: list) -> CollatedBatch:
    collator = Collator(
        inputs=GemmaInputsCollator(TINY, 140),
        instruction=None,
        camera_filter=None,
        max_cameras=None,
        action_codec=None,
        aux=None,
        generate_bracket=False,
        generate_override=None,
        camera_kind_dropout=0.0,
        instruction_augment=0.0,
        condition_fields=(),
        condition_dropout=0.0,
        subgoal_condition_dropout=0.0,
    )
    return collator(items)


def ar_batch(items: list, *, generate_bracket: bool = False) -> CollatedBatch:
    collator = Collator(
        inputs=GemmaInputsCollator(TINY, 140),
        instruction=None,
        camera_filter=None,
        max_cameras=None,
        action_codec=FastActionCodec.load(FIXTURE_TOKENIZER),
        aux=None,
        generate_bracket=generate_bracket,
        generate_override=None,
        camera_kind_dropout=0.0,
        instruction_augment=0.0,
        condition_fields=(),
        condition_dropout=0.0,
        subgoal_condition_dropout=0.0,
    )
    return collator(items)


def main() -> None:
    selection = select_datasets((DATA,), (), 50, episode_split=EpisodeSplit.ALL)
    dataset = selection.concat()
    items = [dataset[0], dataset[1000]]

    # --- flow, text-only unfreeze -------------------------------------------
    model = build_flow(make_args(backbone_text_lr=1e-5, backbone_vision_lr=None))
    torch.manual_seed(0)
    loss = family_loss(model, flow_batch(items))
    loss.backward()
    print(f"FLAGS-ON ORACLE (flow, text-only, seed 0, 2 dev items): {loss.item():.4f}")
    print(
        "  expected 1.6948 (re-recorded 2026-08-05: rig-v2 oracle corpus) - MUST match exactly",
    )
    assert f"{loss.item():.4f}" == "1.6948", "flow flags-on oracle DRIFTED"
    check_text_partition(model, "flow text-only")

    # --- flow, text+vision unfreeze -------------------------------------------
    model = build_flow(make_args(backbone_text_lr=1e-5, backbone_vision_lr=1e-5))
    torch.manual_seed(0)
    loss = family_loss(model, flow_batch(items))
    loss.backward()
    print(f"FLAGS-ON ORACLE (flow, text+vision, seed 0): {loss.item():.4f}")
    backbone = model.backbone
    assert backbone.vision_tower is not None
    tower = grad_state(backbone.vision_tower.patch_embedder.input_proj.weight)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    bad = [(n, grad_state(p)) for n, p in trainable if grad_state(p) != "nonzero"]
    print(f"  {'PASS' if tower == 'nonzero' else 'FAIL'}  tower input_proj")
    print(f"  {'PASS' if not bad else 'FAIL'}  ALL trainable have nonzero finite grads")
    if bad:
        print(f"  offending params: {bad[:10]}")
    assert tower == "nonzero" and not bad, "text+vision grad-flow FAILED"

    # (The ar_fast text-only arm retired with the decoder, 2026-08-13 —
    # tag pre-decoder-simplify; its 4.8395 anchor retired with it.)

    # --- ar_backbone, text-only unfreeze (FULL depth: all layers train) ------
    model = build_ar_backbone(
        make_args(
            backbone_text_lr=2.5e-5,
            backbone_vision_lr=None,
        ),
    )
    torch.manual_seed(0)
    loss = family_loss(model, ar_batch(items, generate_bracket=True))
    loss.backward()
    print(f"FLAGS-ON ORACLE (ar_backbone, text-only, seed 0): {loss.item():.4f}")
    print(
        "  expected 27.8546 (re-recorded 2026-08-05: rig-v2 oracle corpus) - MUST match exactly",
    )
    assert f"{loss.item():.4f}" == "27.8546", "ar_backbone flags-on oracle DRIFTED"
    backbone = model.backbone
    text = backbone.language_model
    decoder = model.ar_decoder
    encoder = model.encoder
    last = text.layers[len(text.layers) - 1]
    assert isinstance(last, DecoderLayer)
    checks = {
        "patch fast_embed grads": grad_state(decoder.fast_embed.weight) == "nonzero",
        "patch fast_ple grads": grad_state(decoder.fast_ple.weight) == "nonzero",
        # state_proj moved PROMPT-side with format 3 (GemmaEncoder owns
        # it now); same invariant — zero-init yet grad-reachable through
        # its K/V use — tested at its new home.
        "prompt state_proj grads (zero-init, K/V-reachable)": grad_state(
            encoder.state_proj.weight,
        )
        == "nonzero",
        # FULL-depth partition: the KV-shared deep half trains (it runs
        # on the suffix), and so does the final norm (prefix never ran it).
        "deep (KV-shared) layer mlp": grad_state(
            last.mlp.down_proj.weight,
        )
        == "nonzero",
        # RMSNorm.weight stubs as Parameter | None; ours always exists.
        "final norm": text.norm.weight is not None
        and grad_state(text.norm.weight) == "nonzero",
        "embed_tokens frozen": text.embed_tokens.weight.grad is None,
        "ple tables frozen": text.embed_tokens_per_layer.weight.grad is None,
        "tied lm_head frozen": backbone.lm_head.weight.grad is None,
    }
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    bad = [(n, grad_state(p)) for n, p in trainable if grad_state(p) != "nonzero"]
    print(f"  {'PASS' if not bad else 'FAIL'}  ALL trainable have nonzero finite grads")
    if bad:
        print(f"  offending params: {bad[:10]}")
    assert all(checks.values()) and not bad, "ar_backbone grad-flow FAILED"

    print("GRADFLOW CHECKS PASSED")


if __name__ == "__main__":
    main()
