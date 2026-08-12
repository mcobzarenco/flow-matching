"""Oracles for the release-in-sim convention-map seam (sim.convmap).

The affine machinery itself (AffineMap, fit_convention_map) is pinned by
bijou's own tests; what this file pins is the seam's new surface: the
override resolution (an override replaces exactly its joint, scale reset
to 1, everything else bit-exact) and the coverage tripwire's verdicts
(disjoint fails, majority-uncovered fails, tail-clipped passes).
"""

import pytest
import torch

from bijou.data import DatasetStats
from bijou.eval.molmo_norm import AffineMap
from sim.convmap import (
    SeamConventionMap,
    coverage_report,
    parse_overrides,
    resolve_map,
)


def stats(q01: list[float], q99: list[float]) -> DatasetStats:
    zeros = tuple([0.0] * 6)
    return DatasetStats(
        action_mean=zeros,
        action_std=tuple([1.0] * 6),
        state_mean=zeros,
        state_std=tuple([1.0] * 6),
        action_q01=tuple(q01),
        action_q99=tuple(q99),
        state_q01=tuple(q01),
        state_q99=tuple(q99),
    )


def test_parse_overrides_rejects_unknown_joint() -> None:
    with pytest.raises(SystemExit, match="not in"):
        parse_overrides(["elbow=90"])  # sim name; the seam speaks SO_MOTORS
    with pytest.raises(SystemExit, match="is not"):
        parse_overrides(["elbow_flex=ninety"])
    assert parse_overrides(["elbow_flex=90", "wrist_roll=-90"]) == {
        "elbow_flex": (1.0, 90.0),
        "wrist_roll": (1.0, -90.0),
    }


def test_parse_overrides_sign_carrying() -> None:
    # The official v3.0->v2.1 lift conversion: model = 90 - seam.
    assert parse_overrides(["shoulder_lift=-1,90"]) == {
        "shoulder_lift": (-1.0, 90.0),
    }
    assert parse_overrides(["shoulder_pan=1,0"]) == {"shoulder_pan": (1.0, 0.0)}
    with pytest.raises(SystemExit, match="must be 1 or -1"):
        parse_overrides(["shoulder_lift=-2,90"])
    with pytest.raises(SystemExit, match="is not"):
        parse_overrides(["shoulder_lift=minus,90"])


def test_resolve_map_touches_only_overridden_joints() -> None:
    fitted = AffineMap(
        scale=torch.tensor([1.0, -1.0, 1.0, 1.0, 1.0, 1.0]),
        offset=torch.tensor([0.0, 180.0, 0.0, 0.0, 90.0, 0.0]),
    )
    resolved = resolve_map(fitted, {"elbow_flex": (1.0, 90.0)})
    assert resolved.offset.tolist() == [0.0, 180.0, 90.0, 0.0, 90.0, 0.0]
    assert resolved.scale.tolist() == [1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    # And the round trip holds where it matters: apply then invert is
    # the identity (the state-in/action-out contract).
    values = torch.tensor([[4.6, -102.7, 97.0, 78.7, 77.6, 3.5]])
    assert torch.allclose(resolved.invert(resolved.apply(values)), values)


def test_resolve_map_sign_override_mirrors() -> None:
    fitted = AffineMap(
        scale=torch.ones(6),
        offset=torch.tensor([0.0, 180.0, 0.0, 0.0, 0.0, 0.0]),
    )
    resolved = resolve_map(fitted, {"shoulder_lift": (-1.0, 90.0)})
    assert resolved.scale.tolist() == [1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    assert resolved.offset.tolist() == [0.0, 90.0, 0.0, 0.0, 0.0, 0.0]
    # model = 90 - seam on lift; round trip still exact under the mirror.
    values = torch.tensor([[4.6, -102.7, 97.0, 78.7, 77.6, 3.5]])
    mapped = resolved.apply(values)
    assert mapped[0, 1].item() == pytest.approx(90.0 - (-102.7))
    assert torch.allclose(resolved.invert(mapped), values)


def test_coverage_report_verdicts() -> None:
    # Joint 0: identity map, seam == box -> covered. Joint 1: mapped
    # interval entirely below the box -> disjoint FAIL. Joint 2: only a
    # sliver of the mapped interval reaches the box -> majority
    # uncovered FAIL. Joint 3: a tail pokes past the box -> pass.
    seam_stats = stats(
        q01=[-10.0, -100.0, -40.0, -10.0, 0.0, 0.0],
        q99=[10.0, -50.0, 100.0, 12.0, 1.0, 1.0],
    )
    table = stats(
        q01=[-10.0, 45.0, 90.0, -10.0, 0.0, 0.0],
        q99=[10.0, 186.0, 174.0, 10.0, 1.0, 1.0],
    )
    seam = SeamConventionMap(
        seam_stats=seam_stats,
        fit=None,  # type: ignore[arg-type] — coverage never reads the fit
        map=AffineMap(scale=torch.ones(6), offset=torch.zeros(6)),
        overrides={},
    )
    lines, failures = coverage_report(seam, table)
    assert len(lines) == 6
    assert failures == ["shoulder_lift", "elbow_flex"]
    assert "FAIL disjoint" in lines[1]
    assert "FAIL uncovered" in lines[2]
