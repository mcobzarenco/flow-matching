"""Shared objective payloads — the typed values a family's constructor
takes to select what receives gradients in a run.

An objective is a closed union of frozen dataclasses per family (Rust
enum-with-payload): each variant carries exactly the knobs that
parameterize its term composition, so a knob without its term is
unrepresentable. Objectives are graph facts — loss terms, their
weights, gradient gates (insulation) — never optimizer policy: LRs,
weight decay, adamc, and schedules live in train.py, reconciled
against :meth:`~bijou.vla.VLA.param_groups`' structural offer.

Objectives are recorded in ``train_args`` and serialized into
checkpoint metadata (a loaded model reconstructs as-it-was-trained);
they are never model state. Family-unique payloads (e.g. the joint
objective) co-locate with their family module; the payloads here are
shared by more than one family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FlowObjective:
    """Plain flow matching over the action chunk (the unit variant —
    no knobs)."""


@dataclass(frozen=True, slots=True)
class SnapflowObjective:
    """Self-distillation mix over a φ_s-extended flow decoder:
    α·mean(fm) + (1−α)·shortcut_weight·mean(shortcut). FM runs s=t
    (φ_s trained, not bypassed); the shortcut term regresses the
    one-step field at pure noise onto the model's own multi-step
    integration (stop-grad teacher).

    Admissible on any flow family whose decoder config has
    ``target_time_embed`` — the family's constructor validates and
    names the remedy (extend the checkpoint at init; the φ_s MLP is
    zero-initialized, so extension is function-preserving)."""

    alpha: float
    shortcut_weight: float

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(
                f"alpha is the FM share of the mix and must sit in (0, 1), "
                f"got {self.alpha} — alpha=1 is the flow objective, alpha=0 "
                "trains no flow matching at all",
            )
        if not self.shortcut_weight > 0:
            raise ValueError(
                f"shortcut_weight must be > 0, got {self.shortcut_weight} — "
                "a zero-weight shortcut term is the flow objective; "
                "construct with FlowObjective instead",
            )


@dataclass(frozen=True, slots=True)
class ARObjective:
    """Next-token CE over the suffix (value lines where the family
    narrates, then the action block). ``aux_loss_weight`` mixes the
    value-line CE against the action-block CE; families without a text
    surface validate that their construction carries no aux fields for
    it to weight."""

    aux_loss_weight: float

    def __post_init__(self) -> None:
        if not self.aux_loss_weight > 0:
            raise ValueError(
                f"aux_loss_weight must be > 0, got {self.aux_loss_weight} — "
                "training aux fields at weight 0 is not 'no aux'; drop the "
                "aux fields from the run instead",
            )


def parse_ar_objective(data: dict[str, Any]) -> ARObjective:
    """The AR families' payload from the metadata's tagged dict.
    ``aux_loss_weight`` defaults to 1.0 when unrecorded (families
    without a text surface have nothing for it to weight)."""
    kind = data.get("kind")
    if kind != "ar":
        raise SystemExit(
            f"objective kind {kind!r} is not the suffix-CE objective ('ar')",
        )
    return ARObjective(aux_loss_weight=float(data.get("aux_loss_weight", 1.0)))
