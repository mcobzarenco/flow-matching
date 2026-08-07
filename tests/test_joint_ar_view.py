"""AR-view materializer oracles (attach-screen launch prep, 2026-08-07).

Read 4 of the attach-screen pre-reg (K trunk-drift: greedy AR panel of
K's trunk vs the 40k endpoint number) needs the joint checkpoint's CE
rider as a plain ``ar_backbone``-view checkpoint that ``bijou.eval``
loads unchanged. The queue-item oracle, run against the REAL write side:

(i)   a joint checkpoint written by the real ``save_checkpoint`` (flow
      decoder + joint_ce rider + adapted trunk) materializes into a view
      that loads via ``from_checkpoint`` — decoder is the rider BITWISE,
      trunk is K's adapted snapshot, taps are stripped — and
      greedy-decodes a valid chunk on the tiny fixture;
(ii)  the view is eval-only (no optimizer.pt) and its weight files are
      the joint checkpoint's bytes;
(iii) non-joint checkpoints (the F arm) are refused loudly.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest
import torch
from test_checkpoint_backbone import make_args
from test_molmo2_ar import FIXTURE, batch, codec, text_config, tiny_inputs
from test_molmo2_residual import build_flow_model, build_joint_model, fresh_model

from bijou.decoders.ar_molmo2 import Molmo2ARDecoder
from bijou.loading import from_checkpoint
from bijou.model import BijouModel
from bijou.train import Normalizer, Normalizers, save_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fontaine.scripts.materialize_joint_ar_view import (
    ar_view_meta,
    materialize,
)

BATCH = 2
DIM = codec().action_dim


def write_char_tokenizer(path: Path) -> None:
    """A ``tokenizer.json`` whose ids equal ``ord(char)`` — byte-for-byte
    the CharTokenizer id scheme the tiny AR fixtures train with, so the
    real ``Molmo2TextTokenizer`` path inside ``from_checkpoint``
    (generation opener, newline carriers) sees the trained ids."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

    vocab = {chr(i): i for i in range(text_config().text.total_vocab_size + 64)}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token=chr(0)))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), "isolated")
    tokenizer.decoder = decoders.Fuse()
    tokenizer.save(str(path))


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from bijou.molmo2.testing import write_tiny_text_checkpoint

    checkpoint = write_tiny_text_checkpoint(
        tmp_path_factory.mktemp("joint-ar-view") / "tiny-molmo2",
    )
    write_char_tokenizer(checkpoint / "tokenizer.json")
    return checkpoint


def save_joint_checkpoint(
    model: BijouModel,
    tiny_checkpoint: Path,
    save_dir: Path,
    *,
    joint: bool = True,
) -> Path:
    """The REAL write side: what a K-arm (or, with ``joint=False``, an
    F-arm) run at these flags serializes is exactly what the materializer
    will meet at the screen's 10k step."""
    args = dataclasses.replace(
        make_args(save_dir),
        backbone=str(tiny_checkpoint),
        decoder="flow",
        conditioning_streams="residual",
        prompt_generate_bracket=True,
        max_crops=1,
        seam_stop_grad=joint,
        joint_ce=joint,
        fast_tokenizer=str(FIXTURE) if joint else None,
        # K trains the trunk -> backbone_trained -> the adapted snapshot
        # rides in backbone.safetensors (the drift read's subject).
        backbone_text_lr=2e-5 if joint else None,
    )
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return save_checkpoint(
        model,
        args=args,
        normalizers=Normalizers(
            action=Normalizer(mean=torch.zeros(DIM), std=torch.ones(DIM)),
            state=Normalizer(mean=torch.zeros(DIM), std=torch.ones(DIM)),
        ),
        per_dataset_stats={},
        optimizer=optimizer,
        scheduler=scheduler,
        step=10,
        adapted_backbone_source=None,
    )


def test_ar_view_loads_and_greedy_decodes(
    tiny_checkpoint: Path,
    tmp_path: Path,
) -> None:
    model = build_joint_model(tiny_checkpoint)
    assert model.joint_ce is not None
    checkpoint = save_joint_checkpoint(model, tiny_checkpoint, tmp_path / "K")
    view = materialize(checkpoint)

    view_meta = json.loads((view / "bijou_config.json").read_text())
    joint_meta = json.loads((checkpoint / "bijou_config.json").read_text())
    assert view_meta["decoder"] == joint_meta["joint_ce"]
    assert "joint_ce" not in view_meta
    assert view_meta["prompt"]["residual_exports"] == []
    assert not (view / "optimizer.pt").exists()  # eval-only view
    assert (view / "expert.safetensors").read_bytes() == (
        checkpoint / "joint_ce.safetensors"
    ).read_bytes()

    # Explicit fp32 mount (the CPU-oracle convention — the default bf16
    # is the GPU eval mix); the snapshot still round-trips the bf16 grid.
    loaded, _info = from_checkpoint(view, dtype=torch.float32)
    assert isinstance(loaded.decoder, Molmo2ARDecoder)
    assert loaded.joint_ce is None
    # The view's decoder is the rider BITWISE (both sides fp32).
    rider_state = model.joint_ce.state_dict()
    loaded_state = loaded.decoder.state_dict()
    assert loaded_state.keys() == rider_state.keys()
    for key, tensor in rider_state.items():
        assert torch.equal(loaded_state[key], tensor.cpu()), key
    # The trunk is K's ADAPTED snapshot (bf16 grid — the checkpoint's
    # serialization precision), not the pristine tiny checkpoint.
    trained_trunk = model.backbone.state_dict()
    loaded_trunk = loaded.backbone.state_dict()
    trainable = {name for name, _ in model.backbone.named_parameters()}
    assert loaded_trunk.keys() == trained_trunk.keys()
    for key, tensor in trained_trunk.items():
        expected = (
            tensor.to(torch.bfloat16).to(torch.float32) if key in trainable else tensor
        )
        assert torch.equal(loaded_trunk[key].cpu(), expected.cpu()), key
    # Taps stripped: the view encodes like a phase-1 AR checkpoint.
    assert loaded.encoder.residual_exports == ()

    sample = batch(codec(), tiny_inputs())
    memory = loaded.encoder.encode(
        loaded.backbone,
        sample.encoder_inputs,
        with_grad=False,
        retain_cache=True,
    )
    prediction = loaded.decoder.predict_chunk(loaded.backbone, memory, sample)
    horizon = codec().time_horizon
    assert prediction.actions.shape == (BATCH, horizon, DIM)
    assert torch.isfinite(prediction.actions).all()
    assert prediction.generations is not None and len(prediction.generations) == BATCH


def test_non_joint_checkpoint_is_refused(
    tiny_checkpoint: Path,
    tmp_path: Path,
) -> None:
    model = build_flow_model(tiny_checkpoint, fresh_model(tiny_checkpoint))
    checkpoint = save_joint_checkpoint(
        model,
        tiny_checkpoint,
        tmp_path / "F",
        joint=False,
    )
    with pytest.raises(SystemExit, match="no joint_ce section"):
        materialize(checkpoint)
    with pytest.raises(SystemExit, match="no joint_ce section"):
        ar_view_meta(json.loads((checkpoint / "bijou_config.json").read_text()))
