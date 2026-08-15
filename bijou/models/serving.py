"""Recorded serving operating points — the typed values behind
:meth:`~bijou.vla.VLA.predict`'s "no knobs" contract.

A checkpoint's metadata records how the model is SERVED (its ``serving``
tagged dict); families parse it into one of these payloads at
construction and ``predict`` runs exactly that point, so cross-family
paired evals compare like with like and no family carries silent
serving defaults. Knobbed inference lives on the capability traits
(:meth:`~bijou.vla.FlowVLA.predict_flow`,
:meth:`~bijou.vla.ARVLA.predict_ar`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..modelling.interface import SamplingMethod


@dataclass(frozen=True, slots=True)
class FlowServing:
    """A flow family's recorded operating point: solver step count and
    method for the deployment integration."""

    num_steps: int
    method: SamplingMethod

    def __post_init__(self) -> None:
        if self.num_steps < 1:
            raise ValueError(
                f"serving num_steps must be >= 1, got {self.num_steps}",
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlowServing:
        kind = data.get("kind")
        if kind != "flow":
            raise SystemExit(
                f"serving kind {kind!r} is not 'flow' — this checkpoint "
                "does not serve through a flow decoder",
            )
        return cls(
            num_steps=int(data["num_steps"]),
            method=SamplingMethod(data["method"]),
        )


@dataclass(frozen=True, slots=True)
class ARServing:
    """A discrete family's recorded operating point: the deterministic
    greedy block decode (the deployment and paired-eval path) — a unit
    payload; sampled decodes are the :class:`~bijou.vla.ARVLA` trait's
    explicit knobs."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ARServing:
        kind = data.get("kind")
        if kind != "ar":
            raise SystemExit(
                f"serving kind {kind!r} is not 'ar' — this checkpoint "
                "does not serve through a discrete action decoder",
            )
        return cls()
