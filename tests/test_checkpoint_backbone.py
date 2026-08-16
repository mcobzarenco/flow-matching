"""save_checkpoint's backbone invariant (the VLA writer, schema 2).

Both backbone PART files are ALWAYS materialized (D9, per part): a part
trained this run snapshots bf16 at every boundary; a FROZEN part
hard-links its source — the init checkpoint's file, or the previous
save's after a fresh run's first boundary serialized the mount once —
with ``metadata.backbone.text_trained``/``vision_trained`` recording
the facts explicitly (presence is not a signal). Conditioning on
``args.backbone_trained`` alone shipped flags-off fine-tunes whose
decoders were trained against adapted features but whose checkpoints
silently loaded the pristine backbone (found 2026-07-31, ft-rig arm F);
the per-part flags carry the same invariant part-wise.

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

from bijou.checkpoint import (
    BACKBONE_TEXT_FILENAME,
    BACKBONE_VISION_FILENAME,
    GEMMA_TOKENIZER_FILES,
    link_or_copy,
    validate_checkpoint,
)
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
    BackbonePartSources,
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


def tiny_meta_model() -> GemmaFlowVLA:
    """Meta-device E2B-shaped model for group-partition inspection only
    (weights never touched; the real E2B config carries the vision
    tower the encoder's param_groups contract asserts)."""
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
    model = tiny_meta_model()
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
        snapflow_alpha=None,
        snapflow_shortcut_weight=None,
        fast_tokenizer=None,
        aux_fields=None,
        narration_weight=0.5,
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


def tokenizer_files_fixture(tmp_path: Path) -> dict[str, Path]:
    """A gemma-manifest tokenizer payload for the writer (stub contents —
    the writer links files; only their presence is validated here)."""
    source = tmp_path / "tokenizer_source"
    if not source.exists():
        source.mkdir()
        for name in GEMMA_TOKENIZER_FILES:
            (source / name).write_text("{}")
    return {name: source / name for name in GEMMA_TOKENIZER_FILES}


def run_save(
    save_dir: Path,
    *,
    model: GemmaFlowVLA,
    args: TrainArgs | None = None,
    part_sources: BackbonePartSources | None = None,
    inherited_text_trained: bool = False,
    inherited_vision_trained: bool = False,
    step: int = 5,
) -> Path:
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
        step=step,
        backbone_config={"model_type": "gemma4"},
        part_sources=(
            part_sources
            if part_sources is not None
            else BackbonePartSources(text=None, vision=None)
        ),
        inherited_text_trained=inherited_text_trained,
        inherited_vision_trained=inherited_vision_trained,
        tokenizer_files=tokenizer_files_fixture(save_dir),
    )


def test_link_or_copy_links_fresh_destinations(tmp_path: Path) -> None:
    source = tmp_path / "source.safetensors"
    save_file({"w": torch.arange(4.0)}, str(source))
    destination = tmp_path / "dest.safetensors"
    link_or_copy(source, destination)
    assert destination.stat().st_ino == source.stat().st_ino  # hardlinked
    torch.testing.assert_close(load_file(str(destination))["w"], torch.arange(4.0))


def test_fresh_frozen_run_serializes_once_then_links(tmp_path: Path) -> None:
    """Flags off, no source checkpoint: the FIRST save serializes both
    frozen parts from the mount (once), the next save hard-links the
    first's files (one inode per frozen lineage), and the metadata
    records both parts untrained explicitly."""
    model = tiny_cpu_model(decoder_hidden=64, seed=0)
    first = run_save(tmp_path, model=model, step=5)
    assert (first / "flow_decoder.safetensors").exists()
    assert (first / "prompt.safetensors").exists()
    assert (first / "optimizer.pt").exists()
    text_file = first / BACKBONE_TEXT_FILENAME
    vision_file = first / BACKBONE_VISION_FILENAME
    assert text_file.is_file()
    assert vision_file.is_file()  # vision-less config: a valid empty file
    assert (first / "tokenizer" / "tokenizer.json").is_file()
    metadata = validate_checkpoint(first)
    assert metadata.backbone_text_trained is False
    assert metadata.backbone_vision_trained is False
    assert metadata.family.value == "gemma_flow"
    assert metadata.backbone_depth == BackboneDepth.FULL.value
    # The serialized part is the mount, bf16 params.
    saved = load_file(str(text_file), device="cpu")
    canary = "language_model.layers.0.mlp.down_proj.weight"
    assert saved[canary].dtype == torch.bfloat16
    assert torch.equal(
        saved[canary],
        model.backbone.state_dict()[canary].to(torch.bfloat16),
    )
    assert "lm_head.weight" not in saved  # the tied alias never serializes

    second = run_save(
        tmp_path,
        model=model,
        part_sources=BackbonePartSources(text=text_file, vision=vision_file),
        step=10,
    )
    assert (second / BACKBONE_TEXT_FILENAME).stat().st_ino == text_file.stat().st_ino
    assert (
        second / BACKBONE_VISION_FILENAME
    ).stat().st_ino == vision_file.stat().st_ino
    validate_checkpoint(second)


def test_frozen_inherited_parts_link_and_stay_trained(tmp_path: Path) -> None:
    """Flags off but initialized from an adapted checkpoint: both part
    files hard-link the source's and the metadata keeps the inherited
    trained facts (a frozen adapted part is still adapted)."""
    source_dir = tmp_path / "init_checkpoint"
    source_dir.mkdir()
    text_source = source_dir / BACKBONE_TEXT_FILENAME
    vision_source = source_dir / BACKBONE_VISION_FILENAME
    save_file(
        {"language_model.layers.0.mlp.down_proj.weight": torch.randn(8, 8)},
        str(text_source),
    )
    save_file({}, str(vision_source))

    checkpoint = run_save(
        tmp_path,
        model=tiny_cpu_model(decoder_hidden=64, seed=0),
        part_sources=BackbonePartSources(text=text_source, vision=vision_source),
        inherited_text_trained=True,
        inherited_vision_trained=False,
    )
    carried = checkpoint / BACKBONE_TEXT_FILENAME
    assert carried.stat().st_ino == text_source.stat().st_ino
    metadata = validate_checkpoint(checkpoint)
    assert metadata.backbone_id == "google/gemma-4-e2b-it"
    assert metadata.backbone_text_trained is True
    assert metadata.backbone_vision_trained is False


def test_trained_text_snapshots_while_frozen_vision_links(tmp_path: Path) -> None:
    """The mixed regime: --backbone-text-lr set, vision frozen — the
    text part snapshots fresh (never links) while the vision part links
    its source; flags follow the split."""
    vision_dir = tmp_path / "prev"
    vision_dir.mkdir()
    vision_source = vision_dir / BACKBONE_VISION_FILENAME
    save_file({}, str(vision_source))
    model = tiny_cpu_model(decoder_hidden=64, seed=0)
    checkpoint = run_save(
        tmp_path,
        model=model,
        args=dataclasses.replace(make_args(tmp_path), backbone_text_lr=2e-5),
        part_sources=BackbonePartSources(
            text=tmp_path / "never_used.safetensors",
            vision=vision_source,
        ),
    )
    metadata = validate_checkpoint(checkpoint)
    assert metadata.backbone_text_trained is True
    assert metadata.backbone_vision_trained is False
    assert (
        checkpoint / BACKBONE_VISION_FILENAME
    ).stat().st_ino == vision_source.stat().st_ino
    # The text file is a fresh serialization (the fabricated "source"
    # was never touched — it does not even exist).
    assert not (tmp_path / "never_used.safetensors").exists()
    saved = load_file(str(checkpoint / BACKBONE_TEXT_FILENAME), device="cpu")
    canary = "language_model.layers.0.mlp.down_proj.weight"
    assert torch.equal(
        saved[canary],
        model.backbone.state_dict()[canary].to(torch.bfloat16),
    )


def tiny_cpu_model(
    decoder_hidden: int,
    seed: int,
    truncate: int | None = None,
) -> GemmaFlowVLA:
    """Real-tensor (non-meta) tiny model: the v2 writer serializes parts
    from the mount, so the backbone must exist on CPU. ``truncate``
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
    checkpoint's part files (bf16 snapshot semantics), the decoder keeps
    its fresh build — across DIFFERENT decoder configs (the point of the
    path)."""
    source = tiny_cpu_model(decoder_hidden=64, seed=0)
    source_encoder = source.encoder
    # A trained-looking prompt projection (zero init would make the
    # copy assertion vacuous).
    torch.nn.init.normal_(source_encoder.state_proj.weight, std=0.1)
    # backbone_text_lr set => the writer snapshots the live backbone.
    checkpoint = run_save(
        tmp_path,
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

    # Backbone: the part file's bf16 values, exactly (bf16 -> fp32 is
    # lossless).
    snapshot = load_file(str(checkpoint / BACKBONE_TEXT_FILENAME), device="cpu")
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
        model=source,
        args=dataclasses.replace(make_args(tmp_path), backbone_text_lr=2e-5),
    )

    # Tiny config: 8 layers, KV-shared last 2 => prefix depth 6 (ends on
    # a full-attention layer), mirroring E2B's 35 -> 15.
    target = tiny_cpu_model(decoder_hidden=32, seed=1, truncate=6)
    stage2_backbone_init(target, target.backbone, checkpoint)

    snapshot = load_file(str(checkpoint / BACKBONE_TEXT_FILENAME), device="cpu")
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
    """A checkpoint whose metadata records both parts pristine has
    nothing to inherit — silently proceeding would run an ablation's
    stock arm twice."""
    checkpoint = run_save(tmp_path, model=tiny_cpu_model(decoder_hidden=64, seed=0))
    target = tiny_cpu_model(decoder_hidden=32, seed=1)
    with pytest.raises(SystemExit, match="both backbone parts are pristine"):
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
