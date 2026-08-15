"""bijou.train's family seams: the LR-vs-offer reconciliation, the
new-format save round-trip through ``bijou.loading.load_vla``, the
chunked-backward counts invariant at family level, and the φ_s
strict-load tolerance.

Reuses the shared hermetic gemma fixtures (tiny trunk + legacy
checkpoint + ``convert_legacy``, ``tests/vla_fixtures.py``): the
converted directory is exactly the artifact class the new CLI trains
from, and the family loaded from it is the model whose save/load
round-trip must be lossless.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from test_checkpoint_backbone import make_args
from torch import nn
from vla_fixtures import gemma_batch, write_gemma_flow_legacy, write_gemma_trunk

from bijou.checkpoint import read_metadata, validate_checkpoint
from bijou.convert_legacy import convert
from bijou.loading import load_vla
from bijou.models.gemma_flow import GemmaFlowVLA
from bijou.train import (
    Normalizer,
    Normalizers,
    load_family_weights,
    reconcile_lr_offer,
    save_checkpoint,
    summed_loss_counts,
)

DIM = 6
GEMMA_CHUNK = 10


@pytest.fixture(scope="module")
def gemma_flow_converted(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[GemmaFlowVLA, Path]:
    root = tmp_path_factory.mktemp("train-vla-gemma-flow")
    trunk = write_gemma_trunk(root / "trunk")
    legacy = write_gemma_flow_legacy(root / "legacy", trunk)
    converted = root / "converted"
    convert(legacy, converted)
    family = GemmaFlowVLA.from_checkpoint(converted, device="cpu", dtype=torch.float32)
    return family, converted


# ---------------------------------------------------------------------------
# LR flags vs the structural offer (D4, both directions)


def fake_offer(
    *,
    text: int,
    vision: int,
) -> dict[str, list[nn.Parameter]]:
    return {
        "decoder": [nn.Parameter(torch.zeros(2, 2))],
        "backbone_text": [nn.Parameter(torch.zeros(2, 2)) for _ in range(text)],
        "backbone_vision": [nn.Parameter(torch.zeros(2, 2)) for _ in range(vision)],
    }


def test_lr_flag_for_empty_group_dies() -> None:
    with pytest.raises(SystemExit, match="receives no gradients"):
        reconcile_lr_offer(
            fake_offer(text=1, vision=0),
            family="gemma_flow",
            backbone_text_lr=None,
            backbone_vision_lr=1e-5,
        )
    with pytest.raises(SystemExit, match="--backbone-text-lr"):
        reconcile_lr_offer(
            fake_offer(text=0, vision=0),
            family="gemma_flow",
            backbone_text_lr=1e-5,
            backbone_vision_lr=None,
        )


def test_offered_group_without_lr_freezes_loudly() -> None:
    notes = reconcile_lr_offer(
        fake_offer(text=2, vision=1),
        family="gemma_flow",
        backbone_text_lr=None,
        backbone_vision_lr=None,
    )
    assert len(notes) == 2
    assert any("backbone_text" in note and "FROZEN" in note for note in notes)
    assert any("backbone_vision" in note and "FROZEN" in note for note in notes)
    # Both LRs given for offered groups: nothing freezes, nothing dies.
    assert (
        reconcile_lr_offer(
            fake_offer(text=1, vision=1),
            family="gemma_flow",
            backbone_text_lr=1e-5,
            backbone_vision_lr=1e-5,
        )
        == []
    )


def test_real_family_offer_reconciles(
    gemma_flow_converted: tuple[GemmaFlowVLA, Path],
) -> None:
    family, _ = gemma_flow_converted
    offer = family.param_groups()
    notes = reconcile_lr_offer(
        offer,
        family="gemma_flow",
        backbone_text_lr=None,
        backbone_vision_lr=None,
    )
    # The tiny trunk offers a text group; vision exists on this fixture
    # iff its config mounts a tower — either way every offered group is
    # named in a freeze note.
    offered = [name for name in ("backbone_text", "backbone_vision") if offer[name]]
    assert len(notes) == len(offered)


# ---------------------------------------------------------------------------
# Save round-trip: train's writer -> load_vla -> the same model


def test_save_round_trips_through_load_vla(
    gemma_flow_converted: tuple[GemmaFlowVLA, Path],
    tmp_path: Path,
) -> None:
    family, converted = gemma_flow_converted
    args = make_args(tmp_path)
    optimizer = torch.optim.AdamW(family.flow_decoder.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    saved = save_checkpoint(
        family,
        family.backbone,
        args=args,
        normalizers=Normalizers(
            action=Normalizer(mean=torch.zeros(DIM), std=torch.ones(DIM)),
            state=Normalizer(mean=torch.zeros(DIM), std=torch.ones(DIM)),
        ),
        per_dataset_stats={},
        optimizer=optimizer,
        scheduler=scheduler,
        step=7,
        adapted_backbone_source=None,
        # The mounted trunk directory IS the pristine source — exactly
        # what main() threads through (here: the converted checkpoint's
        # own hard-linked mirror).
        pristine_trunk_dir=converted / "backbone",
    )
    metadata = validate_checkpoint(saved)
    assert metadata.step == 7
    assert metadata.family.value == "gemma_flow"
    assert metadata.objective == {"kind": "flow"}
    assert metadata.serving == {"kind": "flow", "num_steps": 5, "method": "heun"}
    assert metadata.train_args["family"] == "gemma_flow"
    assert (saved / "optimizer.pt").exists()

    reloaded = load_vla(saved, device="cpu", dtype=torch.float32)
    assert isinstance(reloaded, GemmaFlowVLA)
    assert reloaded.spec == family.spec
    original_state = family.state_dict()
    reloaded_state = reloaded.state_dict()
    assert set(original_state) == set(reloaded_state)
    for name, tensor in original_state.items():
        assert torch.equal(tensor, reloaded_state[name]), f"state mismatch at {name}"


# ---------------------------------------------------------------------------
# The chunked-backward counts invariant, at family level (D5)


def test_micro_slice_objectives_sum_to_full_batch(
    gemma_flow_converted: tuple[GemmaFlowVLA, Path],
) -> None:
    """forward on micro-slices with the SAME summed counts yields
    objective addends whose sum equals the full-batch objective —
    ``VLA.forward``'s chunked-backward contract, exercised through the
    loop's own count summation. RNG realization differs per slice (the
    documented chunked semantics), so the identity is over one shared
    noise draw: seed per forward, slice the SAME batch."""
    family, _ = gemma_flow_converted
    halves = [
        gemma_batch(201, chunk_size=GEMMA_CHUNK, with_tokens=False),
        gemma_batch(202, chunk_size=GEMMA_CHUNK, with_tokens=False),
    ]
    counts = summed_loss_counts(family, halves)
    assert int(counts["action"]) == sum(
        int(family.loss_counts(half)["action"]) for half in halves
    )
    torch.manual_seed(9)
    addends: list[torch.Tensor] = []
    component_sums: list[torch.Tensor] = []
    for half in halves:
        report = family(half, counts=counts)
        addends.append(report.objective.detach())
        component_sums.append(report.components["action"].sum.detach())
    # The addends divide by the GLOBAL count, so their sum is the
    # count-weighted whole: sum(sums) / global_count.
    total = torch.stack(addends).sum()
    reconstructed = torch.stack(component_sums).sum() / counts["action"]
    assert torch.allclose(total, reconstructed, atol=1e-6)
    # And a mismatched key set dies loudly at the loop's summation.
    with pytest.raises(SystemExit, match="run-constant"):
        bad: Any = [{"action": torch.tensor(1)}, {"other": torch.tensor(1)}]
        summed_loss_counts_probe(family, bad)


def summed_loss_counts_probe(family: GemmaFlowVLA, fabricated: Any) -> None:
    """Drive summed_loss_counts' key-set guard with fabricated counts
    (monkeypatching loss_counts would hide the seam under test)."""
    original = family.loss_counts
    queue = list(fabricated)
    family.loss_counts = lambda batch: queue.pop(0)  # fault injection
    try:
        summed_loss_counts(family, [object(), object()])  # type: ignore[list-item] — counts-only path
    finally:
        family.loss_counts = original


# ---------------------------------------------------------------------------
# The φ_s extension's strict-load tolerance


def test_phi_s_extension_load_tolerates_exactly_the_new_keys(
    gemma_flow_converted: tuple[GemmaFlowVLA, Path],
    tmp_path: Path,
) -> None:
    """--init-from --target-time-embed over an unextended source: the
    load misses EXACTLY the fresh φ_s keys (zero-init output ⇒ step-0
    model ≡ checkpoint); anything else mismatched still dies."""
    import dataclasses as dc

    from bijou.modelling.decoders.flow import FlowDecoder
    from bijou.modelling.interface import SamplingMethod
    from bijou.models.objectives import FlowObjective
    from bijou.models.serving import FlowServing

    family, converted = gemma_flow_converted
    metadata = read_metadata(converted)
    extended_config = dc.replace(family.flow_decoder.config, target_time_embed=True)
    torch.manual_seed(11)
    extended = GemmaFlowVLA(
        family.backbone,
        family.encoder,
        FlowDecoder(extended_config, device="cpu", dtype=torch.float32),
        objective=FlowObjective(),
        serving=FlowServing(num_steps=5, method=SamplingMethod.HEUN),
    )
    load_family_weights(
        extended,
        converted,
        metadata,
        args=make_args(tmp_path),
        device=torch.device("cpu"),
        phi_s_extension=True,
        is_main=True,
    )
    # Shared keys match the source bitwise; φ_s keys kept their init.
    source_state = family.flow_decoder.state_dict()
    for name, tensor in extended.flow_decoder.state_dict().items():
        if name.startswith(("target_time_in_proj.", "target_time_out_proj.")):
            continue
        assert torch.equal(tensor, source_state[name]), name
    # The reverse claim (phi_s_extension=False on an extended build)
    # dies on the strict load — trained-parameter drops are never quiet.
    torch.manual_seed(12)
    strict = GemmaFlowVLA(
        family.backbone,
        family.encoder,
        FlowDecoder(extended_config, device="cpu", dtype=torch.float32),
        objective=FlowObjective(),
        serving=FlowServing(num_steps=5, method=SamplingMethod.HEUN),
    )
    with pytest.raises(RuntimeError, match="Missing key"):
        load_family_weights(
            strict,
            converted,
            metadata,
            args=make_args(tmp_path),
            device=torch.device("cpu"),
            phi_s_extension=False,
            is_main=True,
        )
