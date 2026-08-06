"""save_checkpoint's backbone-snapshot invariant.

``backbone.safetensors`` must be present iff the checkpoint's backbone
differs from pristine HF — trained in-run OR inherited frozen from an
adapted --init-from/--resume checkpoint. Conditioning on
``args.backbone_trained`` alone shipped flags-off fine-tunes whose decoders
were trained against adapted features but whose checkpoints silently
loaded the pristine backbone (found 2026-07-31, ft-rig arm F).

The backbone-trained branch itself (live fp32 masters -> bf16 snapshot) is
exercised by real unfreeze runs and stays out of scope here: it needs
materialized backbone weights, which is probe territory.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from test_backbone_continuation import tiny_text_config

from bijou.decoders.flow import (
    ExpertConfig,
    FlowDecoder,
    SelfAttentionMode,
    TimeConditioning,
)
from bijou.encoders.gemma4 import GemmaEncoder
from bijou.gemma4.config import Gemma4Config, e2b_config
from bijou.gemma4.loading import truncated_config
from bijou.gemma4.model import Gemma4Model
from bijou.loading import BackboneDepth, checkpoint_sections, load_backbone_init
from bijou.model import BijouModel
from bijou.nn import RopeParameters, RopeType
from bijou.train import (
    Normalizer,
    Normalizers,
    TrainArgs,
    link_or_copy,
    lr_lambda,
    save_checkpoint,
)

DIM = 6


def tiny_decoder(hidden_size: int = 64) -> FlowDecoder:
    config = ExpertConfig(
        hidden_size=hidden_size,
        num_attention_heads=2,
        intermediate_size=128,
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        self_attention_mode=SelfAttentionMode.CAUSAL_ACTIONS,
        self_attention_rope_theta=10_000.0,
        cross_attention_heads=2,
        cross_attention_head_dim=32,
        cross_attention_rope=RopeParameters(
            rope_type=RopeType.DEFAULT,
            rope_theta=10_000.0,
            factor=1.0,
            partial_rotary_factor=1.0,
        ),
        cross_attention_schedule=(4, 9, 14),
        action_dim=DIM,
        state_dim=DIM,
        chunk_size=50,
        time_embed_dim=256,
        time_conditioning=TimeConditioning.ADDITIVE,
    )
    return FlowDecoder(config, device="cpu", dtype=torch.float32)


def tiny_model() -> BijouModel:
    # The backbone stays on the meta device: the flags-off save paths under
    # test never touch its weights (only encoder.exports for metadata).
    # Truncated exactly as the real build (depth derivation reads the
    # backbone's config: no KV-shared layers <=> prefix depth).
    config = truncated_config(e2b_config(), 15)
    encoder = GemmaEncoder(
        config,
        exports=(4, 9, 14),
        processor_dir="unused",
        max_soft_tokens=140,
        state_dim=6,
    )
    return BijouModel(
        backbone=Gemma4Model(config, device="meta"),
        encoder=encoder,
        decoder=tiny_decoder(),
    )


def test_frozen_state_proj_leaves_the_decoder_group() -> None:
    """Frozen-backbone runs freeze state_proj (no gradient path through
    a no-grad prefix encode) — param_groups must then EXCLUDE it, or
    DDP's every-trainable-gets-a-grad contract breaks on the first
    backward; live runs keep it in the decoder group."""
    model = tiny_model()
    encoder_params = set(model.encoder.parameters())
    live = set(model.param_groups()["decoder"])
    assert encoder_params <= live  # trainable by default (live runs)
    model.encoder.state_proj.requires_grad_(False)  # train.py, frozen runs
    frozen = set(model.param_groups()["decoder"])
    assert not (encoder_params & frozen)
    assert frozen == live - encoder_params


def make_args(save_dir: Path) -> TrainArgs:
    return TrainArgs(
        train_data=(Path("/unused"),),
        exclude=(),
        fps=None,
        camera_counts=None,
        holdout_episodes=0.0,
        split_seed=0,
        backbone="google/gemma-4-e2b-it",
        save_dir=save_dir,
        init_from=None,
        resume=None,
        allow_same_seed_resume=False,
        backbone_init_from=None,
        prompt_generate_bracket=False,
        instruction=None,
        cameras=None,
        max_cameras=None,
        max_soft_tokens=140,
        max_crops=1,
        stream_counts=(4, 4, 8),
        conditioning_streams="kv",
        self_attention_mode="causal_actions",
        time_conditioning="additive",
        target_time_embed=False,
        distill=None,
        decoder="flow",
        fast_tokenizer=None,
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
        chunk_grad_allreduce=False,
        steps=10,
        decoder_lr=1e-4,
        backbone_text_lr=None,  # flags OFF: the paths under test
        backbone_vision_lr=None,
        warmup_steps=1,
        rewarmup_steps=0,
        weight_decay=1e-5,
        grad_clip=10.0,
        log_every=1,
        eval_every=5,
        save_every=5,
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


def run_save(save_dir: Path, adapted_backbone_source: Path | None) -> Path:
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return save_checkpoint(
        model,
        args=make_args(save_dir),
        normalizers=Normalizers(
            action=Normalizer(mean=torch.zeros(DIM), std=torch.ones(DIM)),
            state=Normalizer(mean=torch.zeros(DIM), std=torch.ones(DIM)),
        ),
        per_dataset_stats={},
        optimizer=optimizer,
        scheduler=scheduler,
        step=5,
        adapted_backbone_source=adapted_backbone_source,
    )


def test_link_or_copy_links_and_overwrites(tmp_path: Path) -> None:
    source = tmp_path / "source.safetensors"
    save_file({"w": torch.arange(4.0)}, str(source))
    destination = tmp_path / "dest.safetensors"
    destination.write_bytes(b"stale")
    link_or_copy(source, destination)
    assert destination.stat().st_ino == source.stat().st_ino  # hardlinked
    torch.testing.assert_close(load_file(str(destination))["w"], torch.arange(4.0))


def test_frozen_pristine_run_writes_no_backbone(tmp_path: Path) -> None:
    """Historical layout: flags off, no adapted init => no backbone file."""
    checkpoint = run_save(tmp_path, adapted_backbone_source=None)
    assert (checkpoint / "expert.safetensors").exists()
    assert (checkpoint / "bijou_config.json").exists()
    assert not (checkpoint / "backbone.safetensors").exists()


def test_frozen_inherited_backbone_rides_along(tmp_path: Path) -> None:
    """Flags off but initialized from an adapted checkpoint: the inherited
    snapshot must ride in every checkpoint, byte-identical to the source
    (from_checkpoint detects adapted backbones by file presence)."""
    source_dir = tmp_path / "init_checkpoint"
    source_dir.mkdir()
    source = source_dir / "backbone.safetensors"
    adapted = {"language_model.layers.0.mlp.down_proj.weight": torch.randn(8, 8)}
    save_file(adapted, str(source))

    checkpoint = run_save(tmp_path, adapted_backbone_source=source)
    carried = checkpoint / "backbone.safetensors"
    assert carried.exists()
    torch.testing.assert_close(
        load_file(str(carried))["language_model.layers.0.mlp.down_proj.weight"],
        adapted["language_model.layers.0.mlp.down_proj.weight"],
    )
    # Metadata still records the pristine id (resolution base) — the
    # adapted diff is exactly the carried file — and the depth the model
    # was built at.
    sections = checkpoint_sections(
        json.loads((checkpoint / "bijou_config.json").read_text()),
    )
    assert sections.backbone.id == "google/gemma-4-e2b-it"
    assert sections.backbone.depth is BackboneDepth.PREFIX


def test_inherited_source_must_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_save(tmp_path, adapted_backbone_source=tmp_path / "missing.safetensors")


def tiny_cpu_model(
    decoder_hidden: int,
    seed: int,
    truncate: int | None = None,
) -> BijouModel:
    """Real-tensor (non-meta) tiny model: --backbone-init-from actually
    copies weights, so the backbone must exist on CPU. ``truncate``
    builds the prefix-encoder shape (the flow arms' backbone)."""
    torch.manual_seed(seed)
    config = Gemma4Config(
        text=tiny_text_config(),
        vision=None,
        image_token_id=999,
        video_token_id=998,
        audio_token_id=997,
        boi_token_id=996,
        eoi_token_id=995,
        dtype=torch.float32,
    )
    if truncate is not None:
        config = truncated_config(config, truncate)
    encoder = GemmaEncoder(
        config,
        exports=(4,),
        processor_dir="unused",
        max_soft_tokens=4,
        state_dim=DIM,
    )
    return BijouModel(
        backbone=Gemma4Model(config, device="cpu"),
        encoder=encoder,
        decoder=tiny_decoder(hidden_size=decoder_hidden),
    )


def test_backbone_init_from_loads_trunk_and_prompt_only(tmp_path: Path) -> None:
    """Stage-2 inheritance: backbone + state_proj come from the source
    checkpoint (bf16 snapshot semantics), the decoder keeps its fresh
    build — across DIFFERENT decoder configs, which --init-from refuses."""
    source = tiny_cpu_model(decoder_hidden=64, seed=0)
    source_encoder = source.encoder
    assert isinstance(source_encoder, GemmaEncoder)  # narrow the seam type
    # A trained-looking prompt projection (zero init would make the
    # copy assertion vacuous).
    torch.nn.init.normal_(source_encoder.state_proj.weight, std=0.1)
    optimizer = torch.optim.AdamW(source.decoder.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    checkpoint = save_checkpoint(
        source,
        # backbone_text_lr set => the writer snapshots the live backbone.
        args=dataclasses.replace(make_args(tmp_path), backbone_text_lr=2e-5),
        normalizers=Normalizers(
            action=Normalizer(mean=torch.zeros(DIM), std=torch.ones(DIM)),
            state=Normalizer(mean=torch.zeros(DIM), std=torch.ones(DIM)),
        ),
        per_dataset_stats={},
        optimizer=optimizer,
        scheduler=scheduler,
        step=5,
        adapted_backbone_source=None,
    )

    target = tiny_cpu_model(decoder_hidden=32, seed=1)  # ≠ seed, ≠ decoder
    decoder_before = {k: v.clone() for k, v in target.decoder.state_dict().items()}
    canary = "language_model.layers.0.mlp.down_proj.weight"
    assert not torch.equal(
        target.backbone.state_dict()[canary],
        source.backbone.state_dict()[canary],
    )

    load_backbone_init(target, checkpoint)

    # Backbone: the snapshot's bf16 values, exactly (bf16 -> fp32 is lossless).
    snapshot = load_file(str(checkpoint / "backbone.safetensors"), device="cpu")
    for key, value in snapshot.items():
        assert torch.equal(
            target.backbone.state_dict()[key],
            value.to(torch.float32),
        ), key
    # Prompt: exact copy of the trained projection.
    target_encoder = target.encoder
    assert isinstance(target_encoder, GemmaEncoder)
    assert torch.equal(
        target_encoder.state_proj.weight,
        source_encoder.state_proj.weight,
    )
    # Decoder: untouched fresh build.
    for key, value in target.decoder.state_dict().items():
        assert torch.equal(value, decoder_before[key]), key


def test_backbone_init_from_full_depth_into_truncated(tmp_path: Path) -> None:
    """Stage-2's real shape: the source trunk is FULL depth (ar_backbone
    trains all layers), the target is the truncated prefix encoder —
    deeper layers drop and the fused per-layer-embedding tensors slice
    to the kept layers' width (the [262144, 35x256] vs [262144, 15x256]
    mismatch that killed ablation arm B on 2026-08-04). Values must be
    the source's own leading slices — packing is layer-major."""
    source = tiny_cpu_model(decoder_hidden=64, seed=0)  # full 8 layers
    optimizer = torch.optim.AdamW(source.decoder.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    checkpoint = save_checkpoint(
        source,
        args=dataclasses.replace(make_args(tmp_path), backbone_text_lr=2e-5),
        normalizers=Normalizers(
            action=Normalizer(mean=torch.zeros(DIM), std=torch.ones(DIM)),
            state=Normalizer(mean=torch.zeros(DIM), std=torch.ones(DIM)),
        ),
        per_dataset_stats={},
        optimizer=optimizer,
        scheduler=scheduler,
        step=5,
        adapted_backbone_source=None,
    )

    # Tiny config: 8 layers, KV-shared last 2 => prefix depth 6 (ends on
    # a full-attention layer), mirroring E2B's 35 -> 15.
    target = tiny_cpu_model(decoder_hidden=32, seed=1, truncate=6)
    load_backbone_init(target, checkpoint)

    snapshot = load_file(str(checkpoint / "backbone.safetensors"), device="cpu")
    target_state = target.backbone.state_dict()
    kept = "language_model.layers.5.mlp.down_proj.weight"
    assert torch.equal(target_state[kept], snapshot[kept].to(torch.float32))
    ple_width = 6 * 4  # kept layers x hidden_size_per_layer_input
    assert torch.equal(
        target_state["language_model.embed_tokens_per_layer.weight"],
        snapshot["language_model.embed_tokens_per_layer.weight"][:, :ple_width].to(
            torch.float32,
        ),
    )
    assert torch.equal(
        target_state["language_model.per_layer_model_projection.weight"],
        snapshot["language_model.per_layer_model_projection.weight"][
            :ple_width,
            :,
        ].to(torch.float32),
    )


def test_backbone_init_from_refuses_pristine_checkpoints(tmp_path: Path) -> None:
    """A checkpoint without backbone.safetensors has nothing to inherit —
    silently proceeding would run an ablation's stock arm twice."""
    checkpoint = run_save(tmp_path, adapted_backbone_source=None)
    with pytest.raises(SystemExit, match=r"no backbone\.safetensors"):
        load_backbone_init(tiny_cpu_model(decoder_hidden=32, seed=1), checkpoint)


def test_lr_lambda_rewarmup_anchors_at_resume_step() -> None:
    """Extension runs: the re-warmup ramp starts ~0 AT the resume step,
    recovers the plain schedule after ``rewarmup_steps``, and never
    touches the schedule when off — the initial warmup anchors at step
    0 and is long past by any resume."""
    args = dataclasses.replace(
        make_args(Path("/unused")),
        steps=80_000,
        warmup_steps=500,
        rewarmup_steps=1_000,
    )
    plain = dataclasses.replace(args, rewarmup_steps=0)
    resume = 40_000

    # Ramp start: ~0 at the seam, exactly base/1000 (step-resume+1 = 1).
    assert lr_lambda(resume, args, resume) == pytest.approx(
        lr_lambda(resume, plain, resume) / 1_000,
    )
    # Mid-ramp: half the base.
    assert lr_lambda(resume + 499, args, resume) == pytest.approx(
        lr_lambda(resume + 499, plain, resume) / 2,
    )
    # Ramp done: identical to the plain schedule from there on.
    assert lr_lambda(resume + 1_000, args, resume) == pytest.approx(
        lr_lambda(resume + 1_000, plain, resume),
    )
    assert lr_lambda(79_999, args, resume) == pytest.approx(
        lr_lambda(79_999, plain, resume),
    )
    # Off = bit-identical old behavior, including the floor at 10%.
    assert lr_lambda(79_999, plain, resume) == pytest.approx(0.1, abs=1e-6)
