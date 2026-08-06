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

AR differences from flow, asserted here: the AR decoder has NO zero-init
output head (lm_head is normal-init), so backbone gradients flow from step 1
without the warm-start hack the flow arm needs; token_embedding/lm_head
must receive gradients (DDP participation).

Run from the repo root: uv run python -m probes.probe_unfreeze_gradflow
"""

from __future__ import annotations

from pathlib import Path

import torch
import transformers

from bijou.aux_text import SUFFIX_FORMAT
from bijou.data import EpisodeSplit, select_datasets
from bijou.decoders.ar_backbone import ARBackboneConfig, ARBackboneDecoder
from bijou.decoders.ar_fast import ARFastConfig, ARFastDecoder
from bijou.decoders.flow import FlowDecoder
from bijou.encoders.gemma4 import GemmaEncoder, GemmaInputsCollator
from bijou.fast.codec import ActionCodec
from bijou.gemma4.loading import load_config
from bijou.gemma4.text import DecoderLayer
from bijou.interface import CollatedBatch, Collator
from bijou.loading import (
    BackboneDepth,
    build_gemma_encoder,
    default_expert_config,
    from_backbone,
)
from bijou.model import BijouModel
from bijou.train import BijouTrainStep, TrainArgs, unfreeze_backbone

TINY = "outputs/tiny-gemma4"
# The oracle corpus (2026-08-05, PR #1): rig v2 at its standard mirror
# path — present on every box AND the laptop, unlike the retired
# laptop-only community_dataset_v1_v3.
DATA = Path.home() / "datasets/mcobzarenco/so101_pick_place_v2"
FIXTURE_TOKENIZER = Path("tests/fixtures/tiny_fast_tokenizer")
EXPORTS = (1, 3, 5)  # tiny backbone's global layers
AR_SCHEDULE = ("kv1", "kv3", "kv5", "kv5")


def make_args(
    backbone_text_lr: float | None,
    backbone_vision_lr: float | None,
    *,
    decoder: str = "flow",
) -> TrainArgs:
    return TrainArgs(
        train_data=(DATA,),
        exclude=(),
        fps=None,
        camera_counts=None,
        holdout_episodes=0.0,
        split_seed=0,
        backbone=TINY,
        save_dir=Path("outputs/train/unused"),
        init_from=None,
        resume=None,
        allow_same_seed_resume=False,
        backbone_init_from=None,
        prompt_generate_bracket=False,
        instruction=None,
        cameras=None,
        max_cameras=None,
        max_crops=1,
        max_soft_tokens=140,
        stream_counts=(1, 1, 2),
        conditioning_streams="kv",
        self_attention_mode="causal_actions",
        time_conditioning="additive",
        target_time_embed=False,
        distill=None,
        decoder=decoder,
        fast_tokenizer=(str(FIXTURE_TOKENIZER) if decoder != "flow" else None),
        aux_fields=None,
        aux_loss_weight=0.5,
        aux_dropout=0.0,
        field_dropout=0.0,
        aux_prompt_hash=None,
        camera_kind_dropout=0.0,
        instruction_augment=0.0,
        condition_fields=None,
        condition_dropout=0.0,
        subgoal_dropout=0.0,
        state_dropout=0.0,
        decoder_hidden=64,
        decoder_heads=2,
        decoder_intermediate=128,
        decoder_cross_heads=2,
        chunk_size=50,
        batch_size=2,
        bucket_by_length=False,
        backward_chunks=1,
        zero1=False,
        steps=2,
        decoder_lr=1e-4,
        backbone_text_lr=backbone_text_lr,
        backbone_vision_lr=backbone_vision_lr,
        warmup_steps=1,
        rewarmup_steps=0,
        weight_decay=1e-5,
        grad_clip=1.0,
        log_every=1,
        eval_every=5,
        save_every=1000,
        num_workers=0,
        prefetch_factor=4,
        video_decoder_cache=4,
        device="cpu",
        seed=0,
        eval_samples=None,
        eval_seed=0,
        wandb_project=None,
        wandb_run_name=None,
    )


def build_flow(args: TrainArgs) -> BijouTrainStep:
    torch.manual_seed(args.seed)
    expert_config = default_expert_config(
        load_config(Path(TINY)),
        action_dim=6,
        state_dim=6,
        stream_counts=args.stream_counts,
        hidden_size=args.decoder_hidden,
        num_attention_heads=args.decoder_heads,
        intermediate_size=args.decoder_intermediate,
        cross_attention_heads=args.decoder_cross_heads,
        chunk_size=args.chunk_size,
    )
    model = from_backbone(
        TINY,
        expert_config,
        device="cpu",
        dtype=torch.float32 if args.backbone_trained else None,
        expert_dtype=torch.float32,
    )
    # A fresh flow expert's action_out_proj is ZERO-initialized, which
    # blocks all upstream gradients on the very first step (dL/dhidden =
    # W^T d = 0 - grads populate as zeros, so DDP participation is still
    # fine, and step 2 onward flows). The real unfreeze path warm-starts
    # from a trained checkpoint; simulate that here (seeded).
    assert isinstance(model.decoder, FlowDecoder)
    torch.nn.init.normal_(model.decoder.action_out_proj.weight, std=0.02)
    unfreeze_backbone(model, args)
    return BijouTrainStep(model, backbone_trained=args.backbone_trained)


def build_ar(args: TrainArgs) -> BijouTrainStep:
    torch.manual_seed(args.seed)
    backbone_config = load_config(Path(TINY))
    codec = ActionCodec.load(FIXTURE_TOKENIZER)
    backbone, encoder = build_gemma_encoder(
        Path(TINY),
        backbone_config,
        exports=EXPORTS,
        max_soft_tokens=args.max_soft_tokens,
        state_dim=6,
        device="cpu",
        dtype=torch.float32 if args.backbone_trained else None,
    )
    decoder = ARFastDecoder(
        ARFastConfig(
            hidden_size=args.decoder_hidden,
            num_attention_heads=args.decoder_heads,
            intermediate_size=args.decoder_intermediate,
            hidden_activation=backbone_config.text.hidden_activation,
            rms_norm_eps=backbone_config.text.rms_norm_eps,
            self_attention_rope_theta=10_000.0,
            cross_attention_heads=args.decoder_cross_heads,
            schedule=AR_SCHEDULE,
            tokenizer=str(FIXTURE_TOKENIZER),
            vocab_total=codec.vocab_total,
            state_dim=6,
            chunk_size=args.chunk_size,
            action_dim=6,
        ),
        encoder.stream_geometries(),
        codec,
        device="cpu",
        dtype=torch.float32,
    )
    # NO warm-start hack: lm_head is normal-init, backbone grads must flow
    # from step 1 (asserted below) - a real difference from the flow arm.
    model = BijouModel(backbone=backbone, encoder=encoder, decoder=decoder)
    unfreeze_backbone(model, args)
    return BijouTrainStep(model, backbone_trained=args.backbone_trained)


def build_ar_backbone(args: TrainArgs) -> BijouTrainStep:
    torch.manual_seed(args.seed)
    backbone_config = load_config(Path(TINY))
    codec = ActionCodec.load(FIXTURE_TOKENIZER)
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
    decoder = ARBackboneDecoder(
        ARBackboneConfig(
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
    model = BijouModel(backbone=backbone, encoder=encoder, decoder=decoder)
    unfreeze_backbone(model, args)
    return BijouTrainStep(model, backbone_trained=args.backbone_trained)


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


def check_text_partition(step: BijouTrainStep, label: str) -> None:
    """The shared text-only assertions (both decoders)."""
    backbone = step.model.backbone
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
    trainable = [(name, p) for name, p in step.named_parameters() if p.requires_grad]
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
        action_codec=ActionCodec.load(FIXTURE_TOKENIZER),
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
    step = build_flow(make_args(backbone_text_lr=1e-5, backbone_vision_lr=None))
    torch.manual_seed(0)
    loss, _, _, _ = step(flow_batch(items))
    loss.backward()
    print(f"FLAGS-ON ORACLE (flow, text-only, seed 0, 2 dev items): {loss.item():.4f}")
    print(
        "  expected 1.6948 (re-recorded 2026-08-05: rig-v2 oracle corpus) - MUST match exactly",
    )
    assert f"{loss.item():.4f}" == "1.6948", "flow flags-on oracle DRIFTED"
    check_text_partition(step, "flow text-only")

    # --- flow, text+vision unfreeze -------------------------------------------
    step = build_flow(make_args(backbone_text_lr=1e-5, backbone_vision_lr=1e-5))
    torch.manual_seed(0)
    loss, _, _, _ = step(flow_batch(items))
    loss.backward()
    print(f"FLAGS-ON ORACLE (flow, text+vision, seed 0): {loss.item():.4f}")
    backbone = step.model.backbone
    assert backbone.vision_tower is not None
    tower = grad_state(backbone.vision_tower.patch_embedder.input_proj.weight)
    trainable = [(n, p) for n, p in step.named_parameters() if p.requires_grad]
    bad = [(n, grad_state(p)) for n, p in trainable if grad_state(p) != "nonzero"]
    print(f"  {'PASS' if tower == 'nonzero' else 'FAIL'}  tower input_proj")
    print(f"  {'PASS' if not bad else 'FAIL'}  ALL trainable have nonzero finite grads")
    if bad:
        print(f"  offending params: {bad[:10]}")
    assert tower == "nonzero" and not bad, "text+vision grad-flow FAILED"

    # --- ar_fast, text-only unfreeze (the 192.222.54.70 run's config) --------
    step = build_ar(
        make_args(backbone_text_lr=3e-4, backbone_vision_lr=None, decoder="ar_fast"),
    )
    torch.manual_seed(0)
    loss, _, _, _ = step(ar_batch(items))
    loss.backward()
    print(f"FLAGS-ON ORACLE (ar_fast, text-only, seed 0): {loss.item():.4f}")
    print(
        "  expected 4.8395 (re-recorded 2026-08-05: rig-v2 oracle corpus) - MUST match exactly",
    )
    assert f"{loss.item():.4f}" == "4.8395", "ar flags-on oracle DRIFTED"
    decoder = step.model.decoder
    assert isinstance(decoder, ARFastDecoder)
    extra = {
        "ar token_embedding grads": grad_state(decoder.token_embedding.weight)
        == "nonzero",
        "ar lm_head grads": grad_state(decoder.lm_head.weight) == "nonzero",
    }
    for name, passed in extra.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    assert all(extra.values()), "ar decoder head/embedding grads FAILED"
    # Backbone grads from step 1, cold decoder, no warm-start hack:
    check_text_partition(step, "ar_fast text-only")

    # --- ar_backbone, text-only unfreeze (FULL depth: all layers train) ------
    step = build_ar_backbone(
        make_args(
            backbone_text_lr=2.5e-5,
            backbone_vision_lr=None,
            decoder="ar_backbone",
        ),
    )
    torch.manual_seed(0)
    loss, _, _, _ = step(ar_batch(items, generate_bracket=True))
    loss.backward()
    print(f"FLAGS-ON ORACLE (ar_backbone, text-only, seed 0): {loss.item():.4f}")
    print(
        "  expected 27.8546 (re-recorded 2026-08-05: rig-v2 oracle corpus) - MUST match exactly",
    )
    assert f"{loss.item():.4f}" == "27.8546", "ar_backbone flags-on oracle DRIFTED"
    backbone = step.model.backbone
    text = backbone.language_model
    decoder = step.model.decoder
    assert isinstance(decoder, ARBackboneDecoder)
    encoder = step.model.encoder
    assert isinstance(encoder, GemmaEncoder)  # narrow the seam type
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
    trainable = [(n, p) for n, p in step.named_parameters() if p.requires_grad]
    bad = [(n, grad_state(p)) for n, p in trainable if grad_state(p) != "nonzero"]
    print(f"  {'PASS' if not bad else 'FAIL'}  ALL trainable have nonzero finite grads")
    if bad:
        print(f"  offending params: {bad[:10]}")
    assert all(checks.values()) and not bad, "ar_backbone grad-flow FAILED"

    print("GRADFLOW CHECKS PASSED")


if __name__ == "__main__":
    main()
