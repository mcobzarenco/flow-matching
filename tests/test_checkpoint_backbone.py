"""save_checkpoint's backbone invariant (the VLA writer).

The backbone is ALWAYS materialized (D9): a bf16 ``backbone.safetensors``
snapshot when the trunk state differs from pristine — trained in-run OR
inherited frozen from an adapted --init-from/--resume checkpoint — else
a hard-linked ``backbone/`` mirror of the pristine snapshot, with
``metadata.backbone.trained`` recording the fact explicitly (presence is
not a signal). Conditioning on ``args.backbone_trained`` alone shipped
flags-off fine-tunes whose decoders were trained against adapted
features but whose checkpoints silently loaded the pristine backbone
(found 2026-07-31, ft-rig arm F).

The backbone-trained branch itself (live fp32 masters -> bf16 snapshot)
is exercised by the stage-2 tests below on real tensors; big-trunk
behavior is probe territory.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from test_backbone_continuation import tiny_text_config

from bijou.checkpoint import link_or_copy, validate_checkpoint
from bijou.loading import BackboneDepth
from bijou.modelling.decoders.flow import (
    FlowDecoder,
    FlowDecoderConfig,
    SelfAttentionMode,
    TimeConditioning,
)
from bijou.modelling.encoders.gemma4 import GemmaEncoder
from bijou.modelling.gemma4.config import Gemma4Config
from bijou.modelling.gemma4.loading import truncated_config
from bijou.modelling.gemma4.model import Gemma4Model
from bijou.modelling.interface import SamplingMethod
from bijou.modelling.nn import RopeParameters, RopeType
from bijou.models.gemma_flow import GemmaFlowVLA
from bijou.models.objectives import FlowObjective
from bijou.models.serving import FlowServing
from bijou.train import (
    Normalizer,
    Normalizers,
    TrainArgs,
    lr_lambda,
    save_checkpoint,
    stage2_backbone_init,
)

DIM = 6


def tiny_decoder(hidden_size: int = 64) -> FlowDecoder:
    config = FlowDecoderConfig(
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


def family_of(
    backbone: Gemma4Model,
    encoder: GemmaEncoder,
    decoder: FlowDecoder,
) -> GemmaFlowVLA:
    return GemmaFlowVLA(
        backbone,
        encoder,
        decoder,
        objective=FlowObjective(),
        serving=FlowServing(num_steps=5, method=SamplingMethod.HEUN),
    )


def tiny_model() -> GemmaFlowVLA:
    # The backbone stays on the meta device: the flags-off save paths under
    # test never touch its weights (only encoder.exports for metadata).
    # Truncated exactly as the real build (depth derivation reads the
    # backbone's config: no KV-shared layers <=> prefix depth).
    config = truncated_config(Gemma4Config.e2b(), 15)
    encoder = GemmaEncoder(
        config,
        exports=(4, 9, 14),
        processor_dir="unused",
        max_soft_tokens=140,
        state_dim=6,
    )
    return family_of(Gemma4Model(config, device="meta"), encoder, tiny_decoder())


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
        family="gemma_flow",
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
        insulate_flow=False,
        self_attention_mode="causal_actions",
        time_conditioning="additive",
        target_time_embed=False,
        distill=None,
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
        image_augment=0.0,
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
        activation_checkpointing=False,
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


def pristine_dir(tmp_path: Path) -> Path:
    """A stand-in pristine trunk snapshot to mirror (two files — the
    hard-link rule cares about files, not their contents)."""
    snapshot = tmp_path / "pristine_trunk"
    if not snapshot.exists():
        snapshot.mkdir()
        (snapshot / "config.json").write_text("{}")
        save_file({"w": torch.arange(4.0)}, str(snapshot / "model.safetensors"))
    return snapshot


def run_save(
    save_dir: Path,
    adapted_backbone_source: Path | None,
    *,
    model: GemmaFlowVLA | None = None,
    args: TrainArgs | None = None,
) -> Path:
    model = model if model is not None else tiny_model()
    optimizer = torch.optim.AdamW(model.flow_decoder.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return save_checkpoint(
        model,
        model.backbone,
        args=args if args is not None else make_args(save_dir),
        normalizers=Normalizers(
            action=Normalizer(mean=torch.zeros(DIM), std=torch.ones(DIM)),
            state=Normalizer(mean=torch.zeros(DIM), std=torch.ones(DIM)),
        ),
        per_dataset_stats={},
        optimizer=optimizer,
        scheduler=scheduler,
        step=5,
        adapted_backbone_source=adapted_backbone_source,
        pristine_trunk_dir=pristine_dir(save_dir),
    )


def test_link_or_copy_links_fresh_destinations(tmp_path: Path) -> None:
    source = tmp_path / "source.safetensors"
    save_file({"w": torch.arange(4.0)}, str(source))
    destination = tmp_path / "dest.safetensors"
    link_or_copy(source, destination)
    assert destination.stat().st_ino == source.stat().st_ino  # hardlinked
    torch.testing.assert_close(load_file(str(destination))["w"], torch.arange(4.0))


def test_frozen_pristine_run_mirrors_the_snapshot(tmp_path: Path) -> None:
    """Self-containment without disk cost: flags off, no adapted init =>
    a hard-linked backbone/ mirror, no backbone.safetensors, and the
    metadata records trained=False explicitly."""
    checkpoint = run_save(tmp_path, adapted_backbone_source=None)
    assert (checkpoint / "flow_decoder.safetensors").exists()
    assert (checkpoint / "prompt.safetensors").exists()
    assert (checkpoint / "optimizer.pt").exists()
    assert not (checkpoint / "backbone.safetensors").exists()
    mirror = checkpoint / "backbone" / "model.safetensors"
    assert (
        mirror.stat().st_ino
        == (pristine_dir(tmp_path) / "model.safetensors").stat().st_ino
    )
    metadata = validate_checkpoint(checkpoint)
    assert metadata.backbone_trained is False
    assert metadata.family.value == "gemma_flow"
    assert metadata.backbone_depth == BackboneDepth.PREFIX.value


def test_frozen_inherited_backbone_rides_along(tmp_path: Path) -> None:
    """Flags off but initialized from an adapted checkpoint: the inherited
    snapshot must ride in every checkpoint, byte-identical to the source,
    with trained=True recorded (the loader reloads it over the mount)."""
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
    # adapted diff is exactly the carried file — plus the built depth
    # and the explicit trained fact.
    metadata = validate_checkpoint(checkpoint)
    assert metadata.backbone_id == "google/gemma-4-e2b-it"
    assert metadata.backbone_depth == BackboneDepth.PREFIX.value
    assert metadata.backbone_trained is True


def test_inherited_source_must_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_save(tmp_path, adapted_backbone_source=tmp_path / "missing.safetensors")


def tiny_cpu_model(
    decoder_hidden: int,
    seed: int,
    truncate: int | None = None,
) -> GemmaFlowVLA:
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
    return family_of(
        Gemma4Model(config, device="cpu"),
        encoder,
        tiny_decoder(hidden_size=decoder_hidden),
    )


def test_backbone_init_from_loads_trunk_and_prompt_only(tmp_path: Path) -> None:
    """Stage-2 inheritance: backbone + state_proj come from the source
    checkpoint (bf16 snapshot semantics), the decoder keeps its fresh
    build — across DIFFERENT decoder configs (the point of the path)."""
    source = tiny_cpu_model(decoder_hidden=64, seed=0)
    source_encoder = source.encoder
    # A trained-looking prompt projection (zero init would make the
    # copy assertion vacuous).
    torch.nn.init.normal_(source_encoder.state_proj.weight, std=0.1)
    # backbone_text_lr set => the writer snapshots the live backbone.
    checkpoint = run_save(
        tmp_path,
        adapted_backbone_source=None,
        model=source,
        args=dataclasses.replace(make_args(tmp_path), backbone_text_lr=2e-5),
    )

    target = tiny_cpu_model(decoder_hidden=32, seed=1)  # ≠ seed, ≠ decoder
    decoder_before = {k: v.clone() for k, v in target.flow_decoder.state_dict().items()}
    canary = "language_model.layers.0.mlp.down_proj.weight"
    assert not torch.equal(
        target.backbone.state_dict()[canary],
        source.backbone.state_dict()[canary],
    )

    stage2_backbone_init(target, target.backbone, checkpoint)

    # Backbone: the snapshot's bf16 values, exactly (bf16 -> fp32 is lossless).
    snapshot = load_file(str(checkpoint / "backbone.safetensors"), device="cpu")
    for key, value in snapshot.items():
        assert torch.equal(
            target.backbone.state_dict()[key],
            value.to(torch.float32),
        ), key
    # Prompt: exact copy of the trained projection.
    assert torch.equal(
        target.encoder.state_proj.weight,
        source_encoder.state_proj.weight,
    )
    # Decoder: untouched fresh build.
    for key, value in target.flow_decoder.state_dict().items():
        assert torch.equal(value, decoder_before[key]), key


def test_backbone_init_from_full_depth_into_truncated(tmp_path: Path) -> None:
    """Stage-2's real shape: the source trunk is FULL depth (the AR
    families train all layers), the target is the truncated prefix
    encoder — deeper layers drop and the fused per-layer-embedding
    tensors slice to the kept layers' width (the [262144, 35x256] vs
    [262144, 15x256] mismatch that killed ablation arm B on 2026-08-04).
    Values must be the source's own leading slices — packing is
    layer-major."""
    source = tiny_cpu_model(decoder_hidden=64, seed=0)  # full 8 layers
    checkpoint = run_save(
        tmp_path,
        adapted_backbone_source=None,
        model=source,
        args=dataclasses.replace(make_args(tmp_path), backbone_text_lr=2e-5),
    )

    # Tiny config: 8 layers, KV-shared last 2 => prefix depth 6 (ends on
    # a full-attention layer), mirroring E2B's 35 -> 15.
    target = tiny_cpu_model(decoder_hidden=32, seed=1, truncate=6)
    stage2_backbone_init(target, target.backbone, checkpoint)

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
    target = tiny_cpu_model(decoder_hidden=32, seed=1)
    with pytest.raises(SystemExit, match=r"no backbone\.safetensors"):
        stage2_backbone_init(target, target.backbone, checkpoint)


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
        # __post_init__ enforces the rewarmup⇔resume coupling the CLI
        # always had; the fabricated extension run must carry its resume.
        resume=Path("ckpt/step_040000"),
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
