"""Off-contract convention-map seam for release-checkpoint-in-sim reads
(case 3 of the owner-forwarded box note,
fontaine/notes/molmoact2-unit-contracts-box-note.md).

The sim seam speaks the rig's controller-native units (the ftrig table's
convention); a released molmoact2 checkpoint's global q01/q99 table is a
different unit contract. This module produces the per-joint affine
``A: seam units -> checkpoint-table units`` that the drivers wrap around
the policy: state passes through ``A`` on the way in, decoded chunks pull
back through ``A⁻¹`` on the way out (both directions ride
``bijou.eval.molmo_norm.AffineMap`` — the box's own machinery, so the
fitted map is directly comparable with its panel snaps).

The fit is ``fit_convention_map`` on the seam table vs the model table.
Its midpoint gate was designed for panel datasets and can under-translate
a joint whose seam midpoint sits just inside the padded box while most of
its range hangs below the floor (the rig elbow does exactly this vs the
SO100_101 release: identity leaves ~56% of the range unreachable, +90
leaves ~10%). ``--convmap-override`` exists for that case: an explicit
per-joint offset from the same discrete family, applied only after the
tripwire script (fontaine/scripts/convmap_tripwires.py) shows the gated
choice failing workspace coverage and the override passing the
first-action-vs-state check. Overrides are provenance: they ride the
rows JSON verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bijou.data import DatasetStats
from bijou.eval.molmo_norm import AffineMap, ConventionFit, ItemMaps, fit_convention_map
from bijou.loading import read_checkpoint_info
from bijou.rollout import SO_MOTORS


@dataclass(frozen=True, slots=True)
class SeamConventionMap:
    """The resolved seam: stats in seam units, the gated fit, and the
    final map (fit + overrides) the drivers apply to state AND action —
    one physical convention per rig, exactly the box's convmap rule."""

    seam_stats: DatasetStats
    fit: ConventionFit
    map: AffineMap
    overrides: dict[str, tuple[float, float]]

    @property
    def item_maps(self) -> ItemMaps:
        return ItemMaps(state=self.map, action=self.map)


def parse_overrides(specs: list[str]) -> dict[str, tuple[float, float]]:
    """``joint=offset`` or ``joint=sign,offset`` specs (degrees). The
    bare form keeps sign +1; the two-part form carries a mirror — the
    official LeRobot v3.0->v2.1 conversion sign-flips shoulder_lift
    ((−1,+90) = 90−arm), which the bare syntax could not express and the
    fit's MIRROR_MARGIN gate rejected despite qualifying. Sign must be
    exactly +1 or −1 (the discrete convention family has no other
    members)."""
    overrides: dict[str, tuple[float, float]] = {}
    for spec in specs:
        joint, _, value = spec.partition("=")
        if joint not in SO_MOTORS:
            raise SystemExit(
                f"--convmap-override joint {joint!r} not in {SO_MOTORS}",
            )
        sign_str, comma, offset_str = value.partition(",")
        if not comma:
            sign_str, offset_str = "1", value
        try:
            sign, offset = float(sign_str), float(offset_str)
        except ValueError:
            raise SystemExit(
                f"--convmap-override value {value!r} is not OFFSET or SIGN,OFFSET",
            ) from None
        if sign not in (1.0, -1.0):
            raise SystemExit(
                f"--convmap-override sign {sign_str!r} must be 1 or -1",
            )
        overrides[joint] = (sign, offset)
    return overrides


def seam_convention_map(
    seam_checkpoint: Path,
    model_table: DatasetStats,
    override_specs: list[str] | None = None,
) -> SeamConventionMap:
    """Fit the seam -> model-table convention map.

    ``seam_checkpoint`` is a checkpoint whose normalization table states
    the sim seam's units (the ftrig rig-recomputed table is exactly
    that); ``model_table`` is the release checkpoint's global table
    (``policy.info.normalization``).
    """
    seam_stats = read_checkpoint_info(seam_checkpoint).normalization
    if seam_stats.action_q01 is None or model_table.action_q01 is None:
        raise SystemExit(
            "convention map needs q01/q99 on both tables — one of the "
            "checkpoints predates the quantile backfill",
        )
    fit = fit_convention_map(seam_stats, model_table)
    overrides = parse_overrides(override_specs or [])
    return SeamConventionMap(
        seam_stats=seam_stats,
        fit=fit,
        map=resolve_map(fit.map, overrides),
        overrides=overrides,
    )


def resolve_map(
    fitted: AffineMap,
    overrides: dict[str, tuple[float, float]],
) -> AffineMap:
    """The final seam map: the gated fit with any overridden joints
    replaced by the explicit (sign, offset). Non-overridden joints
    keep the fit's choice bit-exactly."""
    scale = fitted.scale.clone()
    offset = fitted.offset.clone()
    for joint, (sign, value) in overrides.items():
        index = SO_MOTORS.index(joint)
        scale[index] = sign
        offset[index] = value
    return AffineMap(scale=scale, offset=offset)


def coverage_report(
    seam: SeamConventionMap,
    model_table: DatasetStats,
    *,
    max_uncovered: float = 0.5,
) -> tuple[list[str], list[str]]:
    """Tripwire (a) of the pre-reg: the mapped seam workspace
    (action q01/q99 through A) must land inside the model's box — the
    clamp travels with the model, so workspace outside the box is
    unreachable and state outside it is invisible.

    Returns (report lines, failures). A joint fails when the mapped
    interval is disjoint from the box (floor > 0) or when more than
    ``max_uncovered`` of it falls outside (percentile tails always leave
    a few percent uncovered; a majority uncovered means the model is
    blind/clamped for most of the task)."""
    lines: list[str] = []
    failures: list[str] = []
    assert seam.seam_stats.action_q01 is not None  # checked at fit time
    assert seam.seam_stats.action_q99 is not None
    assert model_table.action_q01 is not None
    assert model_table.action_q99 is not None
    for j, joint in enumerate(SO_MOTORS):
        low, high = seam.seam_stats.action_q01[j], seam.seam_stats.action_q99[j]
        scale, offset = float(seam.map.scale[j]), float(seam.map.offset[j])
        ends = (low * scale + offset, high * scale + offset)
        m_low, m_high = min(ends), max(ends)
        box = (model_table.action_q01[j], model_table.action_q99[j])
        overlap = max(0.0, min(m_high, box[1]) - max(m_low, box[0]))
        span = m_high - m_low
        uncovered = 1.0 - overlap / span if span > 0 else 0.0
        floor = max(0.0, box[0] - m_high) + max(0.0, m_low - box[1])
        status = "ok"
        if floor > 0:
            status = "FAIL disjoint"
            failures.append(joint)
        elif uncovered > max_uncovered:
            status = f"FAIL uncovered {uncovered:.0%}"
            failures.append(joint)
        lines.append(
            f"{joint:13s} seam [{low:8.2f},{high:8.2f}] "
            f"-> A [{m_low:8.2f},{m_high:8.2f}] vs box "
            f"[{box[0]:8.2f},{box[1]:8.2f}] "
            f"uncovered {uncovered:5.1%} floor {floor:6.2f}  {status}",
        )
    return lines, failures
