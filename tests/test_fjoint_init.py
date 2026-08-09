"""F-then-joint composite warm-start oracles (fjoint-rung pre-reg
2026-08-09, Instrument §1–§3) — CPU, tiny molmo2 fixture, run against
the REAL write and load sides:

(i)   a composite materialized from an F-arm checkpoint (real
      ``save_checkpoint``, no rider) and a phase-1 ``ar_backbone``
      checkpoint carries F's expert/prompt/trunk BYTES plus the phase-1
      tables as ``joint_ce.safetensors``, under the metadata a real
      joint run would record (``decoder`` = F's flow section,
      ``joint_ce`` = phase-1's decoder section, taps preserved, no
      optimizer.pt);
(ii)  the ``--init-from --joint-ce`` load contract round-trips STRICTLY
      on the composite — the exact sequence train.py's init block runs —
      landing every tensor bitwise: expert ≡ F's, rider ≡ the phase-1
      tables, trunk ≡ the shared snapshot (bf16 grid);
(iii) incoherent inputs are refused loudly: a joint checkpoint in the
      flow slot, a flow checkpoint in the phase-1 slot, a missing trunk
      snapshot, byte-differing trunks (the wrong-phase-1 trap);
(iv)  a J-written checkpoint (``--joint-ce`` with the seam OPEN — the
      ``--joint-unfrozen-seam`` run mode) has K's file shape exactly:
      the drift read's ``materialize_joint_ar_view`` consumes it
      unchanged and the view loads via ``from_checkpoint``;
(v)   the naive-joint guard escape's parse contract: the refusal stays
      verbatim without the flag; the flag admits the combination only
      warm-started (``--init-from``), and refuses fresh runs, an
      explicit ``--seam-stop-grad``, and use without ``--joint-ce``.

The gradient half of the escape's contract — flow-loss gradients into
the trunk NONZERO with the seam open — is test_molmo2_residual (iv),
the former negative control now this run mode's positive contract.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from test_checkpoint_backbone import make_args
from test_joint_ar_view import write_char_tokenizer
from test_molmo2_ar import FIXTURE, build_decoder, build_encoder, codec
from test_molmo2_residual import build_flow_model, build_joint_model, fresh_model

from bijou.decoders.ar_molmo2 import Molmo2ARDecoder
from bijou.loading import from_checkpoint, load_adapted_backbone
from bijou.model import BijouModel
from bijou.train import (
    Normalizer,
    Normalizers,
    TrainArgs,
    ensure_matching_decoder_config,
    parse_args,
    save_checkpoint,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fontaine.scripts.materialize_fjoint_init import (
    composite_meta,
    materialize,
)
from fontaine.scripts.materialize_joint_ar_view import (
    materialize as materialize_ar_view,
)

DIM = codec().action_dim


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from bijou.molmo2.testing import write_tiny_text_checkpoint

    checkpoint = write_tiny_text_checkpoint(
        tmp_path_factory.mktemp("fjoint-init") / "tiny-molmo2",
    )
    write_char_tokenizer(checkpoint / "tokenizer.json")
    return checkpoint


def nudge(module: torch.nn.Module, seed: int) -> None:
    """Make a fixture module's bytes distinguishable from a same-seed
    rebuild — the round-trip asserts must not pass vacuously on two
    identically-initialized modules."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(
                0.01 * torch.randn(parameter.shape, generator=generator),
            )


def save_ckpt(
    model: BijouModel,
    args: TrainArgs,
    *,
    adapted_backbone_source: Path | None = None,
) -> Path:
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
        adapted_backbone_source=adapted_backbone_source,
    )


def save_phase1_checkpoint(tiny_checkpoint: Path, save_dir: Path) -> Path:
    """The REAL phase-1 write side: an ar_backbone run with a live trunk
    (its ``expert.safetensors`` IS the FAST-table decoder the rider
    continues; ``backbone.safetensors`` is the trained-trunk snapshot)."""
    backbone = fresh_model(tiny_checkpoint)
    decoder, _ = build_decoder(backbone)
    nudge(decoder, seed=13)
    model = BijouModel(
        backbone=backbone,
        encoder=build_encoder(tiny_checkpoint),
        decoder=decoder,
    )
    args = dataclasses.replace(
        make_args(save_dir),
        backbone=str(tiny_checkpoint),
        decoder="ar_backbone",
        prompt_generate_bracket=True,
        max_crops=1,
        fast_tokenizer=str(FIXTURE),
        backbone_text_lr=2e-5,  # phase 1 trains the trunk -> snapshot rides
    )
    return save_ckpt(model, args)


def save_f_checkpoint(
    tiny_checkpoint: Path,
    save_dir: Path,
    *,
    trunk_source: Path | None,
) -> tuple[Path, BijouModel]:
    """The REAL F-arm write side: a frozen-trunk flow run whose inherited
    trunk snapshot rides as a hardlink of ``trunk_source`` — on the box,
    the 60k endpoint's file (``trunk_source=None`` reproduces a
    trunk-less save for the refusal oracle)."""
    model = build_flow_model(tiny_checkpoint, fresh_model(tiny_checkpoint))
    nudge(model.decoder, seed=11)
    nudge(model.encoder, seed=12)
    args = dataclasses.replace(
        make_args(save_dir),
        backbone=str(tiny_checkpoint),
        decoder="flow",
        conditioning_streams="residual",
        max_crops=1,
    )
    return save_ckpt(model, args, adapted_backbone_source=trunk_source), model


@pytest.fixture(scope="module")
def fixture_family(
    tiny_checkpoint: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict:
    base = tmp_path_factory.mktemp("fjoint-family")
    phase1 = save_phase1_checkpoint(tiny_checkpoint, base / "phase1")
    f_ckpt, f_model = save_f_checkpoint(
        tiny_checkpoint,
        base / "F",
        trunk_source=phase1 / "backbone.safetensors",
    )
    composite = materialize(f_ckpt, phase1)
    return {
        "phase1": phase1,
        "f_ckpt": f_ckpt,
        "f_model": f_model,
        "composite": composite,
    }


# -- (i) composite bytes + metadata -----------------------------------


def test_composite_bytes_and_metadata(fixture_family: dict) -> None:
    composite = fixture_family["composite"]
    f_ckpt, phase1 = fixture_family["f_ckpt"], fixture_family["phase1"]
    for name, source in (
        ("expert.safetensors", f_ckpt / "expert.safetensors"),
        ("prompt.safetensors", f_ckpt / "prompt.safetensors"),
        ("backbone.safetensors", f_ckpt / "backbone.safetensors"),
        ("joint_ce.safetensors", phase1 / "expert.safetensors"),
    ):
        assert (composite / name).read_bytes() == source.read_bytes(), name
    meta = json.loads((composite / "bijou_config.json").read_text())
    f_meta = json.loads((f_ckpt / "bijou_config.json").read_text())
    phase1_meta = json.loads((phase1 / "bijou_config.json").read_text())
    assert meta["decoder"] == f_meta["decoder"]
    assert meta["joint_ce"] == phase1_meta["decoder"]
    # Taps preserved: J conditions exactly as F did.
    assert meta["prompt"] == f_meta["prompt"]
    assert meta["prompt"]["residual_exports"]
    # Weights-only warm start: --init-from's contract, made structural.
    assert not (composite / "optimizer.pt").exists()


# -- (ii) the --init-from --joint-ce round-trip ------------------------


def test_init_from_joint_ce_round_trips_strictly(
    tiny_checkpoint: Path,
    fixture_family: dict,
) -> None:
    composite = fixture_family["composite"]
    # The J run mode: rider mounted, seam OPEN (--joint-unfrozen-seam).
    model = build_joint_model(tiny_checkpoint, stop_grad=False)
    assert model.joint_ce is not None

    # The exact sequence train.py's --init-from block runs, strict at
    # every load — a key mismatch anywhere is the oracle failing.
    ensure_matching_decoder_config(model.decoder, composite)
    model.decoder.load_state_dict(
        load_file(str(composite / "expert.safetensors"), device="cpu"),
        strict=True,
    )
    model.encoder.load_state_dict(
        load_file(str(composite / "prompt.safetensors"), device="cpu"),
        strict=True,
    )
    rider_path = composite / "joint_ce.safetensors"
    assert rider_path.exists()  # the abort train.py fires on F@10k directly
    model.joint_ce.load_state_dict(
        load_file(str(rider_path), device="cpu"),
        strict=True,
    )
    load_adapted_backbone(model, composite)

    # Expert + prompt land F's bytes (both sides fp32 — the expert file
    # serializes at native precision).
    f_model = fixture_family["f_model"]
    for loaded_module, source_module in (
        (model.decoder, f_model.decoder),
        (model.encoder, f_model.encoder),
    ):
        loaded_state = loaded_module.state_dict()
        source_state = source_module.state_dict()
        assert loaded_state.keys() == source_state.keys()
        for key, tensor in source_state.items():
            assert torch.equal(loaded_state[key], tensor), key
    # The rider lands the phase-1 tables (continuing, not restarting).
    phase1_tables = load_file(
        str(fixture_family["phase1"] / "expert.safetensors"),
        device="cpu",
    )
    rider_state = model.joint_ce.state_dict()
    assert rider_state.keys() == phase1_tables.keys()
    for key, tensor in phase1_tables.items():
        assert torch.equal(rider_state[key], tensor), key
    # The trunk lands the shared snapshot (parameters on the bf16 grid —
    # the checkpoint's serialization precision; buffers native).
    pristine = fresh_model(tiny_checkpoint).state_dict()
    trainable = {name for name, _ in model.backbone.named_parameters()}
    loaded_trunk = model.backbone.state_dict()
    assert loaded_trunk.keys() == pristine.keys()
    for key, tensor in pristine.items():
        expected = (
            tensor.to(torch.bfloat16).to(torch.float32) if key in trainable else tensor
        )
        assert torch.equal(loaded_trunk[key], expected), key


# -- (iii) refusals ----------------------------------------------------


def test_joint_checkpoint_in_flow_slot_refused(fixture_family: dict) -> None:
    with pytest.raises(SystemExit, match="already carries a joint_ce"):
        materialize(fixture_family["composite"], fixture_family["phase1"])
    with pytest.raises(SystemExit, match="already carries a joint_ce"):
        composite_meta(
            json.loads(
                (fixture_family["composite"] / "bijou_config.json").read_text(),
            ),
            json.loads((fixture_family["phase1"] / "bijou_config.json").read_text()),
        )


def test_flow_checkpoint_in_phase1_slot_refused(fixture_family: dict) -> None:
    with pytest.raises(SystemExit, match="not an ar_backbone checkpoint"):
        materialize(fixture_family["f_ckpt"], fixture_family["f_ckpt"])


def test_trunkless_flow_checkpoint_refused(
    tiny_checkpoint: Path,
    fixture_family: dict,
    tmp_path: Path,
) -> None:
    bare, _ = save_f_checkpoint(tiny_checkpoint, tmp_path / "F", trunk_source=None)
    with pytest.raises(SystemExit, match="frozen trunk"):
        materialize(bare, fixture_family["phase1"])


def test_mismatched_trunks_refused(fixture_family: dict, tmp_path: Path) -> None:
    # The wrong-phase-1 trap: same shape, different trunk bytes (a 40k
    # checkpoint passed where the 60k endpoint belongs). Same-size byte
    # flip => the digest branch decides, not the size shortcut.
    wrong = tmp_path / "phase1-wrong"
    shutil.copytree(fixture_family["phase1"], wrong)
    trunk = wrong / "backbone.safetensors"
    data = bytearray(trunk.read_bytes())
    data[-1] ^= 1
    trunk.write_bytes(bytes(data))
    with pytest.raises(SystemExit, match="trunk mismatch"):
        materialize(fixture_family["f_ckpt"], wrong)


# -- (iv) J-written checkpoints feed the drift read unchanged ----------


def test_j_checkpoint_materializes_ar_view(
    tiny_checkpoint: Path,
    tmp_path: Path,
) -> None:
    model = build_joint_model(tiny_checkpoint, stop_grad=False)
    assert model.joint_ce is not None
    args = dataclasses.replace(
        make_args(tmp_path / "J"),
        backbone=str(tiny_checkpoint),
        decoder="flow",
        conditioning_streams="residual",
        prompt_generate_bracket=True,
        max_crops=1,
        seam_stop_grad=False,
        joint_ce=True,
        joint_unfrozen_seam=True,
        fast_tokenizer=str(FIXTURE),
        backbone_text_lr=2e-5,
    )
    checkpoint = save_ckpt(model, args)
    view = materialize_ar_view(checkpoint)
    view_meta = json.loads((view / "bijou_config.json").read_text())
    joint_meta = json.loads((checkpoint / "bijou_config.json").read_text())
    assert view_meta["decoder"] == joint_meta["joint_ce"]
    assert (view / "expert.safetensors").read_bytes() == (
        checkpoint / "joint_ce.safetensors"
    ).read_bytes()
    loaded, _info = from_checkpoint(view, dtype=torch.float32)
    assert isinstance(loaded.decoder, Molmo2ARDecoder)
    assert loaded.joint_ce is None


# -- (v) the guard escape's parse contract -----------------------------

J_ARGV = [
    "--train-data",
    "corpus",
    "--decoder",
    "flow",
    "--conditioning-streams",
    "residual",
    "--joint-ce",
    "--fast-tokenizer",
    "tok",
    "--backbone-text-lr",
    "2e-5",
    "--prompt-generate-bracket",
    "--init-from",
    "ckpt/step_010000",
]

NAIVE_JOINT_REFUSAL = (
    "--joint-ce without --seam-stop-grad is the naive-joint arm — a "
    "published collapse (KI), refused as a run"
)


def _parse(monkeypatch: pytest.MonkeyPatch, *argv: str) -> TrainArgs:
    monkeypatch.setattr("sys.argv", ["bijou.train", *argv])
    return parse_args()


def test_naive_joint_refusal_is_verbatim_without_the_flag(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        _parse(monkeypatch, *J_ARGV)
    assert NAIVE_JOINT_REFUSAL in capsys.readouterr().err


def test_escape_admits_the_warm_started_combination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parse(monkeypatch, *J_ARGV, "--joint-unfrozen-seam")
    assert args.joint_ce is True
    assert args.seam_stop_grad is False
    assert args.joint_unfrozen_seam is True


def test_escape_without_joint_ce_refused(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        _parse(
            monkeypatch,
            "--train-data",
            "corpus",
            "--init-from",
            "ckpt",
            "--joint-unfrozen-seam",
        )
    assert "does nothing without it" in capsys.readouterr().err


def test_escape_contradicting_stop_grad_refused(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        _parse(monkeypatch, *J_ARGV, "--seam-stop-grad", "--joint-unfrozen-seam")
    assert "contradicts --seam-stop-grad" in capsys.readouterr().err


def test_escape_for_a_fresh_run_refused(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fresh = [flag for flag in J_ARGV if flag not in ("--init-from", "ckpt/step_010000")]
    with pytest.raises(SystemExit):
        _parse(monkeypatch, *fresh, "--joint-unfrozen-seam")
    assert "requires --init-from" in capsys.readouterr().err
