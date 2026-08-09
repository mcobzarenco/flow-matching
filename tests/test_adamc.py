"""AdamC (arXiv 2506.02285) — corrected weight decay via stock AdamW.

The implementation's whole claim is: AdamC == AdamW with a per-group,
time-varying ``weight_decay`` (λ̂_t = λ·γ_t/γ_max on "normalized" =
hidden layers; standard decay on the output head; nothing on 1-D
parameters), written into the groups immediately before each step so
the stock (fused) kernel applies it bit-exactly. These tests pin the
three legs of that claim:

1. PARTITION (the tied/shared-layer care the run spec demanded): on the
   molmo2 AR composition, ``fast_head`` routes to standard decay,
   ``fast_embed`` and every hidden matrix to corrected decay, 1-D to no
   decay; the groups disjointly and exactly cover the trainable set; a
   parameter smuggled into two groups (the tied-lm_head failure mode)
   and a head parameter missing from the decoder group both die loudly;
   an unaudited decoder type dies loudly. The adamw partition is
   byte-identical to the historical construction.
2. TRAJECTORY: ``apply_adamc_weight_decay`` writes exactly
   λ·γ_t/γ_max per corrected group as the scheduler moves, warmup
   included, and the write reaches a ZeRO-1 wrapper's local optimizer.
3. EQUIVALENCE: at γ_t == γ_max an adamc step is bitwise an adamw
   step; under a decayed lr it is bitwise AdamW with the corrected
   coefficient set by hand.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_molmo2_ar import build_decoder, build_encoder
from torch import nn

from bijou.decoders.ar_molmo2 import Molmo2ARDecoder
from bijou.model import BijouModel
from bijou.molmo2.model import load_model
from bijou.molmo2.testing import write_tiny_text_checkpoint
from bijou.train import (
    adamc_output_head_parameters,
    apply_adamc_weight_decay,
    build_optimizer_param_groups,
    decay_split,
    resume_hyperparameter_notes,
)


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_tiny_text_checkpoint(
        tmp_path_factory.mktemp("adamc") / "tiny-molmo2",
    )


def bijou_model(tiny_checkpoint: Path) -> BijouModel:
    """The molmo2 AR composition with train.py's freezing applied:
    backbone fully frozen, then the text + vision subsets unfrozen the
    way unfreeze_backbone does for the run spec's flags."""
    backbone = load_model(str(tiny_checkpoint), dtype=torch.float32)
    decoder, _ = build_decoder(backbone)
    encoder = build_encoder(tiny_checkpoint)
    model = BijouModel(backbone=backbone, encoder=encoder, decoder=decoder)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    subsets = encoder.param_groups(backbone)
    for name in ("text", "vision"):
        for parameter in subsets[name]:
            parameter.requires_grad_(True)
    return model


def groups_of(
    model: BijouModel,
    optimizer_name: str,
) -> tuple[list[dict], list[tuple[str, float, float]], list[bool]]:
    return build_optimizer_param_groups(
        model,
        optimizer_name=optimizer_name,
        decoder_lr=1e-4,
        backbone_text_lr=2e-5,
        backbone_vision_lr=2e-5,
        weight_decay=0.1,
    )


def test_adamc_partition_molmo2_ar(tiny_checkpoint: Path) -> None:
    model = bijou_model(tiny_checkpoint)
    param_groups, cli_groups, corrected = groups_of(model, "adamc")
    assert [name for name, _, _ in cli_groups] == [
        "decoder (corrected decay)",
        "decoder head (standard decay)",
        "decoder (no decay)",
        "backbone_text (decayed)",
        "backbone_text (no decay)",
        "backbone_vision (decayed)",
        "backbone_vision (no decay)",
    ]
    assert corrected == [True, False, False, True, False, True, False]
    ids = [{id(p) for p in group["params"]} for group in param_groups]
    decoder = model.decoder
    assert isinstance(decoder, Molmo2ARDecoder)
    # The untied output head is the ONLY standard-decay decoder param;
    # the untied input table stays on the corrected side.
    assert ids[1] == {id(decoder.fast_head.weight)}
    assert id(decoder.fast_embed.weight) in ids[0]
    # 1-D parameters (norm scales, biases) never decay, anywhere.
    for index, group in enumerate(param_groups):
        expected_decayed = index not in (2, 4, 6)
        for parameter in group["params"]:
            assert (parameter.dim() >= 2) == expected_decayed
    # Disjoint, exact cover of the trainable set (frozen wte/lm_head out).
    flat = [p for group in param_groups for p in group["params"]]
    assert len(flat) == len({id(p) for p in flat})
    assert {id(p) for p in flat} == {
        id(p) for p in model.parameters() if p.requires_grad
    }
    frozen_text = model.backbone.text
    assert not frozen_text.lm_head.weight.requires_grad
    assert id(frozen_text.lm_head.weight) not in {id(p) for p in flat}


def test_adamw_partition_unchanged(tiny_checkpoint: Path) -> None:
    """The historical construction, byte-identical: one decoder group in
    model.param_groups() order, then a decayed/undecayed split per
    unfrozen backbone subset."""
    model = bijou_model(tiny_checkpoint)
    param_groups, cli_groups, corrected = groups_of(model, "adamw")
    assert not any(corrected)
    named = model.param_groups()
    assert [name for name, _, _ in cli_groups] == [
        "decoder",
        "backbone_text (decayed)",
        "backbone_text (no decay)",
        "backbone_vision (decayed)",
        "backbone_vision (no decay)",
    ]
    assert param_groups[0]["params"] == named["decoder"]
    text_decayed, text_undecayed = decay_split(named["backbone_text"])
    assert param_groups[1]["params"] == text_decayed
    assert param_groups[2]["params"] == text_undecayed


def test_tied_parameter_in_two_groups_dies(tiny_checkpoint: Path) -> None:
    model = bijou_model(tiny_checkpoint)
    decoder = model.decoder
    assert isinstance(decoder, Molmo2ARDecoder)
    shared = next(iter(decoder.fast_embed.parameters()))
    original = model.param_groups()
    original["backbone_text"] = [*original["backbone_text"], shared]
    model.param_groups = lambda: original
    with pytest.raises(SystemExit, match="decayed twice"):
        groups_of(model, "adamw")


def test_head_missing_from_decoder_group_dies(tiny_checkpoint: Path) -> None:
    model = bijou_model(tiny_checkpoint)
    decoder = model.decoder
    assert isinstance(decoder, Molmo2ARDecoder)
    head = decoder.fast_head.weight
    original = model.param_groups()
    original["decoder"] = [p for p in original["decoder"] if p is not head]
    model.param_groups = lambda: original
    with pytest.raises(SystemExit, match="output-head parameter"):
        groups_of(model, "adamc")


def test_unaudited_decoder_dies(tiny_checkpoint: Path) -> None:
    model = bijou_model(tiny_checkpoint)
    model.decoder = cast(Any, nn.Linear(4, 4))
    with pytest.raises(SystemExit, match="not audited"):
        adamc_output_head_parameters(model)


def tiny_setup(
    lr_lambda: Callable[[int], float],
    *,
    corrected_weight_decay: float = 0.1,
) -> tuple[nn.Sequential, torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
    torch.manual_seed(7)
    net = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 4))
    optimizer = torch.optim.AdamW(
        [
            {"params": net[0].parameters(), "lr": 1e-3},
            {"params": net[1].parameters(), "lr": 1e-4},
        ],
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=corrected_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return net, optimizer, scheduler


def loss_backward(net: nn.Sequential) -> None:
    torch.manual_seed(11)
    net.zero_grad()
    net(torch.randn(3, 4)).square().mean().backward()


def test_adamc_decay_tracks_schedule_exactly() -> None:
    """λ̂_t = λ·γ_t/γ_max per group, exact floats, through warmup (<1),
    peak (=1) and decay (<1) multipliers — group 1 (uncorrected) never
    touched."""
    multipliers = [0.25, 1.0, 0.5, 0.1]
    net, optimizer, scheduler = tiny_setup(
        lambda step: multipliers[min(step, len(multipliers) - 1)],
    )
    for _ in multipliers:
        loss_backward(net)
        apply_adamc_weight_decay(optimizer, [0], 0.1)
        expected = 0.1 * float(optimizer.param_groups[0]["lr"]) / 1e-3
        assert optimizer.param_groups[0]["weight_decay"] == expected
        assert optimizer.param_groups[1]["weight_decay"] == 0.1
        optimizer.step()
        scheduler.step()


def test_adamc_equals_adamw_at_peak_lr() -> None:
    """γ_t == γ_max ⇒ λ̂ == λ ⇒ the corrected step IS the AdamW step,
    bitwise."""
    net_a, opt_a, sched_a = tiny_setup(lambda step: 1.0)
    net_b, opt_b, sched_b = tiny_setup(lambda step: 1.0)
    for _ in range(3):
        loss_backward(net_a)
        opt_a.step()
        sched_a.step()
        loss_backward(net_b)
        apply_adamc_weight_decay(opt_b, [0, 1], 0.1)
        opt_b.step()
        sched_b.step()
    for parameter_a, parameter_b in zip(
        net_a.parameters(),
        net_b.parameters(),
        strict=True,
    ):
        assert torch.equal(parameter_a, parameter_b)


def test_adamc_equals_manual_corrected_adamw() -> None:
    """Under a decaying schedule, the adamc path is bitwise AdamW with
    the corrected coefficient computed BY HAND (independent arithmetic,
    same kernel) and written into the reference's group each step."""
    multipliers = [1.0, 0.6, 0.3]
    clamped = lambda step: multipliers[min(step, len(multipliers) - 1)]
    net_a, opt_a, sched_a = tiny_setup(clamped)
    net_b, opt_b, sched_b = tiny_setup(clamped)
    for multiplier in multipliers:
        loss_backward(net_a)
        opt_a.param_groups[0]["weight_decay"] = 0.1 * (1e-3 * multiplier) / 1e-3
        opt_a.step()
        sched_a.step()
        loss_backward(net_b)
        apply_adamc_weight_decay(opt_b, [0], 0.1)
        assert (
            opt_b.param_groups[0]["weight_decay"]
            == opt_a.param_groups[0]["weight_decay"]
        )
        opt_b.step()
        sched_b.step()
    for parameter_a, parameter_b in zip(
        net_a.parameters(),
        net_b.parameters(),
        strict=True,
    ):
        assert torch.equal(parameter_a, parameter_b)


def test_adamc_decay_reaches_zero1_local_optimizer() -> None:
    """The training loop mutates the ZeRO-1 WRAPPER's groups; torch's
    step() copies group attributes wrapper → local optimizer before the
    sharded step. Pin that contract — if a torch upgrade drops it, AdamC
    under --zero1 silently reverts to constant decay."""
    from torch.distributed.optim import ZeroRedundancyOptimizer

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    torch.distributed.init_process_group("gloo", rank=0, world_size=1)
    try:
        torch.manual_seed(7)
        net = nn.Linear(4, 4)
        optimizer = ZeroRedundancyOptimizer(
            [{"params": list(net.parameters()), "lr": 1e-3}],
            optimizer_class=torch.optim.AdamW,
            lr=1e-3,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5)
        scheduler.step()  # move off the peak: lr = 5e-4
        net(torch.randn(2, 4)).square().mean().backward()
        apply_adamc_weight_decay(optimizer, [0], 0.1)
        optimizer.step()
        assert optimizer.param_groups[0]["weight_decay"] == 0.05
        assert optimizer.optim.param_groups[0]["weight_decay"] == 0.05
    finally:
        torch.distributed.destroy_process_group()


def test_resume_notes_flag_adamc_governed_decay() -> None:
    """On --resume the checkpoint's saved λ̂ is transient state for
    corrected groups: the notes say the CLI λ governs there instead of
    reporting a spurious 'CLI ignored' diff."""
    _net, optimizer, _ = tiny_setup(lambda step: 1.0)
    optimizer.param_groups[0]["weight_decay"] = 0.0123  # a restored λ̂
    notes = resume_hyperparameter_notes(
        optimizer,
        [("corrected", 1e-3, 0.1), ("head", 1e-4, 0.1)],
        [True, False],
    )
    assert any("schedule-managed" in note and "GOVERNS" in note for note in notes)
    assert not any("0.0123" in note for note in notes)
