"""Off-contract normalization modes for molmo_flow checkpoints
(``--molmo-norm``; §8.13 follow-up to the 2026-08-12 curated-v0 read).

A molmo_flow checkpoint's contract is ONE global q01/q99 table baked in
at conversion (decision 6): state clamps/bins through it, decoded
actions unnormalize through it. The 256-frame curated-v0 read showed
what happens when the eval corpus violates the table's homogeneity
assumption: 55% of panel frames sat on datasets whose calibration
convention puts truth OUTSIDE the box — unreachable by construction
(predictions clamp into the box) and invisible (state bins saturate).

Both modes here re-express the model's I/O through a per-dataset,
per-joint AFFINE map ``A: dataset units -> checkpoint-table units``:
state is rewritten ``A(state)`` on the way in (the standard
checkpoint-table path then normalizes/bins it), and decoded raw chunks
are pulled back ``A⁻¹(chunk)`` on the way out. The decoder itself runs
byte-identical to the contract path — the map wraps it.

- **per-dataset** (``_pdnorm``): quantile equating — ``A`` maps the
  dataset's own q01/q99 onto the table's, exactly what evaluating with
  per-dataset statistics means under their clamp semantics. Absorbs
  convention offsets AND rescales spans: a narrow task workspace is
  stretched across the model's whole normalized range (a state
  marginal it never trained on) and decode gain shifts by
  span_dataset/span_table per joint. State and action maps fit their
  own modalities (mirroring the checkpoint's dual-table structure).
- **convention-map** (``_convmap``): the physical arms are identical;
  only the calibration frame differs, and that family is small and
  discrete — per-joint sign ∈ {±1}, offset ∈ {0, ±90, ±180} degrees.
  Fit by snapping the dataset's ACTION box onto the table's action box
  (reachability is what the box governs), tie-broken by mapped-mean
  distance; the SAME map applies to state (one physical convention per
  rig). Joints the discrete family cannot bring inside the box within
  tolerance (e.g. tick-scale units) keep the IDENTITY map and are
  reported — the arm stays a pure translation, never a silent
  quantile swap. Scale is untouched (gain 1), so covered datasets get
  the exact identity and stay bitwise on-contract.

Both are UNCONSTRAINED-class reads: the policy name gains ``_pdnorm``
/ ``_convmap`` so they can never pool with contract reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor

from ..data import DatasetStats

# The discrete convention family (degrees) and the decision
# thresholds. Pre-registered 2026-08-12: the probe measures adequacy;
# widening the family or loosening thresholds after seeing panel
# residuals would be fishing.
CONVENTION_OFFSETS = (0.0, 90.0, -90.0, 180.0, -180.0)
CONVENTION_SIGNS = (1.0, -1.0)
# A candidate FITS a joint when at most this fraction of the mapped
# dataset interval falls outside the table box (uncovered fraction —
# NOT floor distance: the real rig table touches the release box on a
# 3.4° sliver, so disjointness alone would call a 97%-outside joint
# "covered"). The identity is preferred whenever IT fits, so covered
# joints are never translated.
COVERAGE_SLACK = 0.10
# When nothing fits cleanly, translate anyway iff the best member
# uncovers at least this much LESS of the interval than the identity
# (a decisive improvement — e.g. a workspace wider than the table box,
# where the right offset still clips tails); otherwise the joint keeps
# identity and is flagged needs_affine.
DECISIVE_IMPROVEMENT = 0.50
# Quantile equating with a (near-)zero dataset span would decode the
# model's whole normalized range into nothing / explode the state
# stretch; joints narrower than this fall back to a pure offset map
# (scale 1, midpoints matched). The convention fit treats them as
# points (inside the box or not).
MIN_SPAN_DEG = 1.0


class MolmoNorm(StrEnum):
    """``--molmo-norm``: how molmo_flow normalization meets the data."""

    CHECKPOINT = "checkpoint"
    PER_DATASET = "per-dataset"
    CONVENTION_MAP = "convention-map"


@dataclass(frozen=True, slots=True)
class AffineMap:
    """Per-joint affine map, dataset units -> checkpoint-table units.

    Shapes:
    - ``scale``/``offset``: [D] fp32 CPU
    - ``apply``/``invert`` operate on [..., D] tensors (broadcast over
      leading dims), returning fp32
    """

    scale: Tensor
    offset: Tensor

    def __post_init__(self) -> None:
        if self.scale.shape != self.offset.shape or self.scale.ndim != 1:
            raise ValueError(
                f"expected matching 1-D scale/offset, got "
                f"{tuple(self.scale.shape)} / {tuple(self.offset.shape)}",
            )
        if bool((self.scale == 0).any()):
            raise ValueError("zero scale is not invertible")

    def apply(self, values: Tensor) -> Tensor:
        return values.to(torch.float32) * self.scale + self.offset

    def invert(self, values: Tensor) -> Tensor:
        return (values.to(torch.float32) - self.offset) / self.scale

    @property
    def is_identity(self) -> bool:
        return bool((self.scale == 1.0).all()) and bool((self.offset == 0.0).all())


@dataclass(frozen=True, slots=True)
class ConventionFit:
    """The snapped discrete fit for one dataset, with the diagnostics
    that make it reviewable (the probe and the eval print these).

    Shapes: every tensor is [D] fp32 CPU (per joint, degrees).
    """

    map: AffineMap
    # Uncovered fraction of the dataset's action interval vs the table
    # box with NO translation — the share of the joint's range the
    # contract read cannot reach.
    identity_uncovered: Tensor
    # Same, after the chosen map: what the translation leaves outside.
    snapped_uncovered: Tensor
    # Floor distance (deg, interval-to-box) before/after — the
    # unreachable-error magnitude the floors analysis reports.
    identity_floor: Tensor
    snapped_floor: Tensor
    # Per joint: a non-identity member was chosen (and accepted).
    translated: tuple[bool, ...]
    # Per joint: nothing fit and no member improved decisively — the
    # joint keeps the identity map (pure-translation discipline) and
    # only the affine per-dataset mode can rescue it.
    needs_affine: tuple[bool, ...]
    # Per joint: how many family members (incl. identity) FIT cleanly
    # (uncovered ≤ COVERAGE_SLACK) — >1 means the stats under-determine
    # the map and the mean tiebreak decided; 0 with translated=True
    # marks a decisive-improvement (partial-coverage) choice.
    clean_fits: tuple[int, ...]


def _quantile_tensors(stats: DatasetStats, *, modality: str) -> tuple[Tensor, Tensor]:
    """(q01, q99) [D] fp32 for one modality; loud when the stats predate
    the quantile backfill (both modes are meaningless without them)."""
    q01 = getattr(stats, f"{modality}_q01")
    q99 = getattr(stats, f"{modality}_q99")
    if q01 is None or q99 is None:
        raise SystemExit(
            f"--molmo-norm needs {modality} q01/q99 stats — this dataset's "
            "stats predate the quantile backfill",
        )
    return (
        torch.tensor(q01, dtype=torch.float32),
        torch.tensor(q99, dtype=torch.float32),
    )


def quantile_equating_map(
    stats: DatasetStats,
    table: DatasetStats,
    *,
    modality: str,
) -> AffineMap:
    """The per-dataset mode's map for one modality: A(dataset q01) =
    table q01 and A(dataset q99) = table q99, so the standard
    checkpoint-table normalization of A(x) equals the DATASET-quantile
    normalization of x exactly (the algebraic identity the tests pin).
    Joints narrower than MIN_SPAN_DEG get a pure offset map (scale 1,
    midpoints matched) instead of a degenerate gain."""
    d01, d99 = _quantile_tensors(stats, modality=modality)
    t01, t99 = _quantile_tensors(table, modality=modality)
    span_d = d99 - d01
    span_t = t99 - t01
    narrow = span_d < MIN_SPAN_DEG
    scale = torch.where(
        narrow,
        torch.ones_like(span_d),
        span_t / span_d.clamp(min=1e-8),
    )
    offset = torch.where(
        narrow,
        (t01 + t99) / 2 - (d01 + d99) / 2,
        t01 - scale * d01,
    )
    return AffineMap(scale=scale, offset=offset)


def _floor(low: float, high: float, box_low: float, box_high: float) -> float:
    """Distance (deg) from interval [low, high] to box [box_low,
    box_high]: 0 when they overlap — the unreachable-error floor of an
    in-box policy against truth drawn from the interval's far side."""
    return max(0.0, box_low - high) + max(0.0, low - box_high)


def _uncovered(low: float, high: float, box_low: float, box_high: float) -> float:
    """Fraction of [low, high] falling outside [box_low, box_high] —
    the share of the joint's range the box cannot represent (intervals
    narrower than MIN_SPAN_DEG score as points: 0.0 inside, 1.0 out)."""
    span = high - low
    if span < MIN_SPAN_DEG:
        mid = (low + high) / 2
        return 0.0 if box_low <= mid <= box_high else 1.0
    covered = max(0.0, min(high, box_high) - max(low, box_low))
    return 1.0 - covered / span


def fit_convention_map(stats: DatasetStats, table: DatasetStats) -> ConventionFit:
    """Snap one dataset onto the checkpoint table's convention.

    Per joint, over sign ∈ {±1} x offset ∈ {0, ±90, ±180}: score each
    candidate by (uncovered fraction of the mapped action interval,
    floor distance, |mapped action mean − table mean|), lexicographic.
    Decision per joint:

    1. identity FITS (uncovered ≤ COVERAGE_SLACK) → identity, always —
       covered joints are never translated, even when a shifted member
       also fits;
    2. else the best candidate FITS → translate (ambiguity between
       clean fits is broken by the mean distance and recorded in
       ``clean_fits``);
    3. else the best candidate uncovers DECISIVE_IMPROVEMENT less of
       the interval than identity → translate (workspaces genuinely
       wider than the table box);
    4. else → identity + ``needs_affine`` (tick-scale units, exotic
       conventions) — the pure-translation discipline: never a silent
       quantile swap.

    Fit on ACTION stats (the box governs reachability); the caller
    applies the same map to state — one physical convention per rig.
    """
    d01, d99 = _quantile_tensors(stats, modality="action")
    t01, t99 = _quantile_tensors(table, modality="action")
    mean_d = torch.tensor(stats.action_mean, dtype=torch.float32)
    mean_t = torch.tensor(table.action_mean, dtype=torch.float32)

    dims = d01.shape[0]
    scale = torch.ones(dims)
    offset = torch.zeros(dims)
    identity_uncovered = torch.zeros(dims)
    snapped_uncovered = torch.zeros(dims)
    identity_floor = torch.zeros(dims)
    snapped_floor = torch.zeros(dims)
    translated: list[bool] = []
    needs_affine: list[bool] = []
    clean_fits: list[int] = []

    for j in range(dims):
        box = (float(t01[j]), float(t99[j]))
        ends_id = (float(d01[j]), float(d99[j]))
        identity_uncovered[j] = _uncovered(*ends_id, *box)
        identity_floor[j] = _floor(*ends_id, *box)

        best: tuple[float, float, float] | None = None
        best_member = (1.0, 0.0)
        fits = 0
        for sign in CONVENTION_SIGNS:
            for shift in CONVENTION_OFFSETS:
                ends = (
                    sign * float(d01[j]) + shift,
                    sign * float(d99[j]) + shift,
                )
                low, high = min(ends), max(ends)
                score = (
                    _uncovered(low, high, *box),
                    _floor(low, high, *box),
                    abs(sign * float(mean_d[j]) + shift - float(mean_t[j])),
                )
                if score[0] <= COVERAGE_SLACK:
                    fits += 1
                if best is None or score < best:
                    best = score
                    best_member = (sign, shift)
        assert best is not None  # the family is non-empty
        clean_fits.append(fits)

        identity_fits = float(identity_uncovered[j]) <= COVERAGE_SLACK
        decisive = float(identity_uncovered[j]) - best[0] >= DECISIVE_IMPROVEMENT
        if identity_fits or not (best[0] <= COVERAGE_SLACK or decisive):
            translated.append(False)
            needs_affine.append(not identity_fits)
            snapped_uncovered[j] = identity_uncovered[j]
            snapped_floor[j] = identity_floor[j]
        else:
            sign, shift = best_member
            scale[j] = sign
            offset[j] = shift
            snapped_uncovered[j] = best[0]
            snapped_floor[j] = best[1]
            translated.append(True)
            needs_affine.append(False)

    return ConventionFit(
        map=AffineMap(scale=scale, offset=offset),
        identity_uncovered=identity_uncovered,
        snapped_uncovered=snapped_uncovered,
        identity_floor=identity_floor,
        snapped_floor=snapped_floor,
        translated=tuple(translated),
        needs_affine=tuple(needs_affine),
        clean_fits=tuple(clean_fits),
    )


@dataclass(frozen=True, slots=True)
class ItemMaps:
    """The fitted per-dataset transforms one item needs at inference:
    ``state`` rewrites the raw state on the way in, ``action`` pulls
    the decoded raw chunk back into dataset units (both dataset ->
    checkpoint units; predict INVERTS ``action``)."""

    state: AffineMap
    action: AffineMap


def fit_item_maps(
    stats: DatasetStats,
    table: DatasetStats,
    mode: MolmoNorm,
) -> tuple[ItemMaps, ConventionFit | None]:
    """One dataset's maps under the chosen mode (+ the convention fit
    diagnostics when that mode did the fitting). CHECKPOINT mode has no
    maps by definition — calling this with it is a wiring bug."""
    if mode is MolmoNorm.PER_DATASET:
        return (
            ItemMaps(
                state=quantile_equating_map(stats, table, modality="state"),
                action=quantile_equating_map(stats, table, modality="action"),
            ),
            None,
        )
    if mode is MolmoNorm.CONVENTION_MAP:
        fit = fit_convention_map(stats, table)
        # One physical convention per rig: the action-fit map applies
        # to the state too.
        return ItemMaps(state=fit.map, action=fit.map), fit
    raise ValueError(f"no item maps under --molmo-norm {mode.value}")
