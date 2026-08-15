"""Phase-2 gates for the VLA trait scaffolding: currency types,
objective payload validation, and the trait lattice's shape (abstract,
stateless, capability membership by inheritance)."""

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import cast, override

import pytest
import torch
from torch import Tensor, nn

from bijou.modelling.interface import CollatedBatch, InputsCollator
from bijou.models.objectives import ARObjective, FlowObjective, SnapflowObjective
from bijou.vla import (
    ARVLA,
    VLA,
    ARPrediction,
    FlowPrediction,
    FlowVLA,
    Loss,
    LossReport,
    NarratedPrediction,
    NarratingVLA,
    VLAFamily,
    VLASpec,
)


def test_family_values_are_registry_strings() -> None:
    assert {f.value for f in VLAFamily} == {
        "gemma_flow",
        "gemma_ar",
        "molmo2_ar",
        "molmoact2_flow",
        "molmoact2_ar",
        "molmoact2_joint",
    }
    # round-trip: metadata strings parse back to the enum
    for family in VLAFamily:
        assert VLAFamily(family.value) is family


def test_currency_types_are_frozen() -> None:
    loss = Loss(sum=torch.tensor(2.0), count=torch.tensor(4.0))
    with pytest.raises(AttributeError):
        loss.sum = torch.tensor(0.0)  # type: ignore[misc]  # frozen: the assignment IS the test
    spec = VLASpec(family=VLAFamily.GEMMA_FLOW, chunk_size=50, action_dim=6)
    with pytest.raises(AttributeError):
        spec.chunk_size = 1  # type: ignore[misc]  # frozen: the assignment IS the test


def test_prediction_structs_carry_their_trait_fields() -> None:
    actions = torch.zeros(2, 5, 6)
    assert ARPrediction(actions=actions).actions.shape == (2, 5, 6)
    flow = FlowPrediction(actions=actions, noise=torch.zeros(2, 5, 6))
    assert flow.noise.shape == (2, 5, 6)
    narrated = NarratedPrediction(actions=actions, generations=[])
    assert narrated.generations == []


def test_loss_report_shape() -> None:
    action = Loss(sum=torch.tensor(3.0), count=torch.tensor(6.0))
    report = LossReport(objective=torch.tensor(0.5), components={"action": action})
    assert report.components["action"].sum.item() == 3.0


def test_snapflow_objective_validates() -> None:
    SnapflowObjective(alpha=0.75, shortcut_weight=1.0)
    with pytest.raises(ValueError, match="FM share"):
        SnapflowObjective(alpha=1.0, shortcut_weight=1.0)
    with pytest.raises(ValueError, match="FM share"):
        SnapflowObjective(alpha=0.0, shortcut_weight=1.0)
    with pytest.raises(ValueError, match="shortcut_weight"):
        SnapflowObjective(alpha=0.5, shortcut_weight=0.0)


def test_ar_objective_validates() -> None:
    ARObjective(aux_loss_weight=1.0)
    with pytest.raises(ValueError, match="aux_loss_weight"):
        ARObjective(aux_loss_weight=0.0)


def test_flow_objective_is_a_unit_variant() -> None:
    assert FlowObjective() == FlowObjective()


def test_traits_are_abstract() -> None:
    for trait in (VLA, ARVLA, FlowVLA, NarratingVLA):
        assert isinstance(trait, abc.ABCMeta)
        with pytest.raises(TypeError, match="abstract"):
            trait()  # pyright accepts the call (generic class object); runtime must not


def test_traits_are_stateless() -> None:
    """The styleguide's trait rule, pinned: no fields, no __init__ of
    their own — nn.Module's is the only one in the MRO below object."""
    for trait in (VLA, ARVLA, FlowVLA, NarratingVLA):
        assert "__init__" not in trait.__dict__


@dataclass(frozen=True, slots=True)
class _StubInputs:
    """Minimal BatchInputs conformer for the stub family."""

    def pin_memory(self) -> "_StubInputs":
        return self

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "_StubInputs":
        return self

    def tensors(self) -> dict[str, Tensor]:
        return {}


class _StubVLA(FlowVLA[_StubInputs]):
    """Minimal concrete family: proves the lattice is implementable and
    that capability membership is inheritance (isinstance narrowing)."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2)

    @property
    @override
    def spec(self) -> VLASpec:
        return VLASpec(family=VLAFamily.GEMMA_FLOW, chunk_size=1, action_dim=2)

    @override
    def collator(self) -> InputsCollator[_StubInputs]:
        raise NotImplementedError

    @override
    def loss_counts(self, batch: "CollatedBatch[_StubInputs]") -> dict[str, Tensor]:
        return {"action": torch.tensor(4.0)}

    @override
    def forward(
        self,
        batch: "CollatedBatch[_StubInputs]",
        *,
        counts: dict[str, Tensor],
    ) -> LossReport:
        total = self.proj.weight.sum()
        return LossReport(
            objective=total / counts["action"],
            components={"action": Loss(sum=total, count=torch.tensor(4.0))},
        )

    @override
    def predict(self, batch: "CollatedBatch[_StubInputs]") -> Tensor:
        return torch.zeros(1, 1, 2)

    @override
    def predict_flow(
        self,
        batch: "CollatedBatch[_StubInputs]",
        **kwargs: object,
    ) -> FlowPrediction:
        actions = torch.zeros(1, 1, 2)
        return FlowPrediction(actions=actions, noise=actions)

    @override
    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        return {"decoder": list(self.proj.parameters())}

    @override
    def output_head_parameters(self) -> list[nn.Parameter]:
        return []

    @override
    def checkpoint_components(self) -> dict[str, nn.Module]:
        return {"flow_decoder": self.proj}

    @classmethod
    @override
    def from_checkpoint(
        cls,
        checkpoint: Path,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "_StubVLA":
        raise NotImplementedError


def test_stub_family_narrowing_and_forward() -> None:
    model = _StubVLA()
    assert isinstance(model, VLA)
    assert isinstance(model, FlowVLA)
    assert not isinstance(model, ARVLA)
    assert not isinstance(model, NarratingVLA)
    batch = cast("CollatedBatch[_StubInputs]", object())  # the stub never reads it
    counts = model.loss_counts(batch)
    report = model(batch, counts=counts)
    assert report.components.keys() == counts.keys()
    report.objective.backward()
    grads = [p.grad for p in model.param_groups()["decoder"]]
    assert grads[0] is not None
