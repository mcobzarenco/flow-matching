"""save_checkpoint's backbone-snapshot invariant.

``backbone.safetensors`` must be present iff the checkpoint's backbone
differs from pristine HF — trained in-run OR inherited frozen from an
adapted --init-from/--resume checkpoint. Conditioning on
``args.trunk_trained`` alone shipped flags-off fine-tunes whose decoders
were trained against adapted features but whose checkpoints silently
loaded the pristine trunk (found 2026-07-31, ft-rig arm F).

The trunk-trained branch itself (live fp32 masters -> bf16 snapshot) is
exercised by real unfreeze runs and stays out of scope here: it needs
materialized backbone weights, which is probe territory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from bijou.decoders.flow import (
    ExpertConfig,
    FlowDecoder,
    SelfAttentionMode,
    TimeConditioning,
)
from bijou.encoders.gemma4 import GemmaEncoder
from bijou.gemma4.config import e2b_config
from bijou.gemma4.model import Gemma4Model
from bijou.model import BijouModel
from bijou.nn import RopeParameters, RopeType
from bijou.train import (
    Normalizer,
    Normalizers,
    TrainArgs,
    link_or_copy,
    save_checkpoint,
)

DIM = 6


def tiny_decoder() -> FlowDecoder:
    config = ExpertConfig(
        hidden_size=64,
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
    encoder = GemmaEncoder(
        Gemma4Model(e2b_config(), device="meta"),
        exports=(4, 9, 14),
        processor_dir="unused",
        max_soft_tokens=140,
    )
    return BijouModel(encoder=encoder, decoder=tiny_decoder())


def make_args(save_dir: Path) -> TrainArgs:
    return TrainArgs(
        train_data=(Path("/unused"),),
        exclude=(),
        fps=None,
        holdout_episodes=0.0,
        split_seed=0,
        backbone="google/gemma-4-e2b-it",
        save_dir=save_dir,
        init_from=None,
        resume=None,
        instruction=None,
        cameras=None,
        max_cameras=None,
        max_soft_tokens=140,
        stream_counts=(4, 4, 8),
        self_attention_mode="causal_actions",
        time_conditioning="additive",
        decoder="flow",
        fast_tokenizer=None,
        expert_hidden=64,
        expert_heads=2,
        expert_intermediate=128,
        expert_cross_heads=2,
        chunk_size=50,
        batch_size=2,
        steps=10,
        expert_lr=1e-4,
        text_lr=None,  # flags OFF: the paths under test
        vision_lr=None,
        warmup_steps=1,
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


def test_frozen_inherited_trunk_rides_along(tmp_path: Path) -> None:
    """Flags off but initialized from an adapted checkpoint: the inherited
    snapshot must ride in every checkpoint, byte-identical to the source
    (from_checkpoint detects adapted trunks by file presence)."""
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
    # adapted diff is exactly the carried file.
    meta = json.loads((checkpoint / "bijou_config.json").read_text())
    backbone_id = meta["encoder"]["backbone"] if "encoder" in meta else meta["backbone"]
    assert backbone_id == "google/gemma-4-e2b-it"


def test_inherited_source_must_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_save(tmp_path, adapted_backbone_source=tmp_path / "missing.safetensors")
