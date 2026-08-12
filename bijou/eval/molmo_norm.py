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
# thresholds. The FAMILY is pre-registered (2026-08-12); the decision
# rule was re-gated the same day after the first panel probe failed
# the cross-instrument check (1% "covered" vs the floors analysis's
# ~45–55% — a coverage-fraction gate cannot tell a convention SHIFT
# from a workspace TAIL sticking a few degrees past a 1st/99th
# percentile box, and near-symmetric intervals let spurious mirrors
# win). The rule keys on MIDPOINT displacement instead: the family's
# members move midpoints by >= 90 deg, tails do not.
CONVENTION_OFFSETS = (0.0, 90.0, -90.0, 180.0, -180.0)
CONVENTION_SIGNS = (1.0, -1.0)
# A joint is IN CONVENTION (and a candidate QUALIFIES) when its
# (mapped) interval midpoint lies within the table box padded by this
# many degrees per side — percentile boxes clip legitimate tails, so
# midpoints near the edge are normal; midpoints far outside are not.
BOX_PAD_DEG = 10.0
# A mirror (sign -1) candidate must beat the best sign +1 qualifier by
# at least this much uncovered-fraction to be chosen: mirrored joints
# are physically rare across lerobot calibration eras, and stats alone
# cannot distinguish a mirror from an offset on near-symmetric
# intervals — demand decisive evidence.
MIRROR_MARGIN = 0.25
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
    # Per joint: out of convention (midpoint outside the padded box)
    # and NO family member brings it in — the joint keeps the identity
    # map (pure-translation discipline) and only the affine per-dataset
    # mode can rescue it.
    needs_affine: tuple[bool, ...]
    # Per joint: how many family members QUALIFIED (mapped midpoint
    # inside the padded box) — >1 means the stats under-determine the
    # map and the (floor, uncovered, mean-distance) ranking plus the
    # mirror margin decided.
    qualified: tuple[int, ...]


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

    The gate is MIDPOINT DISPLACEMENT, not interval coverage: the
    family's members move a joint's midpoint by >= 90 deg, while
    legitimate workspace tails past a 1st/99th-percentile box move it
    by a few. Decision per joint:

    1. the interval midpoint lies inside the table box padded by
       BOX_PAD_DEG → IN CONVENTION → identity, always — tails are
       normal, and a shifted member that also happens to fit must
       never flip a whole joint's convention;
    2. else, over sign ∈ {±1} x offset ∈ {0, ±90, ±180}: candidates
       whose mapped midpoint enters the padded box QUALIFY; they rank
       by (floor distance, uncovered fraction, |mapped action mean −
       table mean|), and a mirror (sign −1) wins only by beating the
       best sign +1 qualifier by MIRROR_MARGIN uncovered — mirrors are
       physically rare and near-symmetric intervals fake them;
    3. no qualifier → identity + ``needs_affine`` (tick-scale units,
       exotic conventions) — the pure-translation discipline: never a
       silent quantile swap.

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
    qualified_counts: list[int] = []

    for j in range(dims):
        box = (float(t01[j]), float(t99[j]))
        padded = (box[0] - BOX_PAD_DEG, box[1] + BOX_PAD_DEG)
        ends_id = (float(d01[j]), float(d99[j]))
        identity_uncovered[j] = _uncovered(*ends_id, *box)
        identity_floor[j] = _floor(*ends_id, *box)
        mid_id = (ends_id[0] + ends_id[1]) / 2

        if padded[0] <= mid_id <= padded[1]:
            # In convention: tails past the percentile box are normal.
            translated.append(False)
            needs_affine.append(False)
            qualified_counts.append(1)  # identity
            snapped_uncovered[j] = identity_uncovered[j]
            snapped_floor[j] = identity_floor[j]
            continue

        best_plus: tuple[tuple[float, float, float], tuple[float, float]] | None = None
        best_mirror: tuple[tuple[float, float, float], tuple[float, float]] | None = (
            None
        )
        qualifying = 0
        for sign in CONVENTION_SIGNS:
            for shift in CONVENTION_OFFSETS:
                if sign == 1.0 and shift == 0.0:
                    continue  # the identity was already rejected above
                ends = (
                    sign * float(d01[j]) + shift,
                    sign * float(d99[j]) + shift,
                )
                low, high = min(ends), max(ends)
                mid = (low + high) / 2
                if not padded[0] <= mid <= padded[1]:
                    continue
                qualifying += 1
                score = (
                    _floor(low, high, *box),
                    _uncovered(low, high, *box),
                    abs(sign * float(mean_d[j]) + shift - float(mean_t[j])),
                )
                if sign == 1.0:
                    if best_plus is None or score < best_plus[0]:
                        best_plus = (score, (sign, shift))
                elif best_mirror is None or score < best_mirror[0]:
                    best_mirror = (score, (sign, shift))
        qualified_counts.append(qualifying)

        chosen = best_plus
        if best_mirror is not None and (
            best_plus is None or best_mirror[0][1] <= best_plus[0][1] - MIRROR_MARGIN
        ):
            chosen = best_mirror
        if chosen is None:
            translated.append(False)
            needs_affine.append(True)
            snapped_uncovered[j] = identity_uncovered[j]
            snapped_floor[j] = identity_floor[j]
        else:
            (floor_score, uncovered_score, _), (sign, shift) = chosen
            scale[j] = sign
            offset[j] = shift
            snapped_uncovered[j] = uncovered_score
            snapped_floor[j] = floor_score
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
        qualified=tuple(qualified_counts),
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
