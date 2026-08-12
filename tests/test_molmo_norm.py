"""Oracles for --molmo-norm's estimator (bijou/eval/molmo_norm.py).

Pure CPU, synthetic stats. The estimator is the load-bearing half of
the off-contract modes: a wrong map silently re-labels a whole
dataset's joints, so every branch gets a pin — identity preference on
covered joints, offset/sign recovery on the convention families we
measured in curated-v0 (±90/±180 shifts, sign flips), the
pure-translation refusal on scale-family (tick-unit) datasets, the
quantile-equating algebraic identity, and the degenerate-span guard.
"""

from __future__ import annotations

import pytest
import torch

from bijou.data import DatasetStats
from bijou.eval.molmo_norm import (
    AffineMap,
    MolmoNorm,
    fit_convention_map,
    fit_item_maps,
    quantile_equating_map,
)

# A checkpoint-table stand-in shaped like the release's: joint 1 spans
# the +45..+186 (old-convention) band, the rest are zero-centered.
TABLE = DatasetStats(
    action_mean=(0.0, 120.0, 0.0, 0.0, 0.0, 20.0),
    action_std=(1.0,) * 6,
    state_mean=(0.0, 120.0, 0.0, 0.0, 0.0, 20.0),
    state_std=(1.0,) * 6,
    action_q01=(-40.0, 45.0, -60.0, -60.0, -60.0, 0.0),
    action_q99=(40.0, 186.0, 60.0, 60.0, 60.0, 44.0),
    state_q01=(-40.0, 45.0, -60.0, -60.0, -60.0, 0.0),
    state_q99=(40.0, 186.0, 60.0, 60.0, 60.0, 44.0),
)


def stats(
    q01: tuple[float, ...],
    q99: tuple[float, ...],
    mean: tuple[float, ...] | None = None,
) -> DatasetStats:
    if mean is None:
        mean = tuple((a + b) / 2 for a, b in zip(q01, q99, strict=True))
    return DatasetStats(
        action_mean=mean,
        action_std=(1.0,) * len(q01),
        state_mean=mean,
        state_std=(1.0,) * len(q01),
        action_q01=q01,
        action_q99=q99,
        state_q01=q01,
        state_q99=q99,
    )


def test_covered_dataset_is_identity() -> None:
    """Boxes inside the table box: no translation, no diagnostics —
    the covered 45% of the panel must stay bitwise on-contract."""
    fit = fit_convention_map(
        stats(
            (-30.0, 90.0, -50.0, -10.0, -5.0, 2.0),
            (30.0, 180.0, 50.0, 10.0, 5.0, 40.0),
        ),
        TABLE,
    )
    assert fit.map.is_identity
    assert not any(fit.translated)
    assert not any(fit.needs_affine)
    assert float(fit.snapped_floor.max()) == 0.0


def test_offset_180_recovery() -> None:
    """The dopaul signature: new-convention lift box [-100, -80] vs the
    table's [45, 186] — only +180 brings it inside (+90 maps to
    [-10, 10], still floored by 35)."""
    q01 = (-30.0, -100.0, -50.0, -10.0, -5.0, 2.0)
    q99 = (30.0, -80.0, 50.0, 10.0, 5.0, 40.0)
    fit = fit_convention_map(stats(q01, q99), TABLE)
    assert fit.translated[1]
    assert float(fit.map.offset[1]) == 180.0
    assert float(fit.map.scale[1]) == 1.0
    # Every other joint untouched.
    assert [float(o) for o in fit.map.offset] == [0.0, 180.0, 0.0, 0.0, 0.0, 0.0]
    assert float(fit.snapped_floor[1]) == 0.0
    assert float(fit.identity_floor[1]) == pytest.approx(125.0)


def test_sign_flip_recovery() -> None:
    """A mirrored joint: box [-186, -45] maps onto [45, 186] under
    sign −1, offset 0."""
    q01 = (-30.0, -186.0, -50.0, -10.0, -5.0, 2.0)
    q99 = (30.0, -45.0, 50.0, 10.0, 5.0, 40.0)
    fit = fit_convention_map(stats(q01, q99), TABLE)
    assert fit.translated[1]
    assert float(fit.map.scale[1]) == -1.0
    assert float(fit.map.offset[1]) == 0.0
    assert float(fit.snapped_floor[1]) == 0.0


def test_wide_box_stats_prefer_the_mirror_fit() -> None:
    """The identifiability limit, pinned: the rig-table-shaped lift box
    [-104, 49] fits the table band BEST under sign -1, offset +90
    ([41, 194], 92% covered) — cleanly — while the offset-only members
    cover 72% (+180) and 61% (+90). Interval+mean stats cannot tell a
    mirrored joint from an offset one (both exist in lerobot's
    calibration history); the estimator takes the stats-optimal member,
    the printed per-dataset map makes the choice reviewable, and the
    GPU arm adjudicates the physics. The mapped mean agrees here too
    (-(-32)+90 = 122 vs table 120)."""
    q01 = (-30.0, -104.0, -50.0, -10.0, -5.0, 2.0)
    q99 = (30.0, 49.0, 50.0, 10.0, 5.0, 40.0)
    fit = fit_convention_map(
        stats(q01, q99, mean=(0.0, -32.0, 0.0, 0.0, 0.0, 20.0)),
        TABLE,
    )
    assert fit.translated[1]
    assert float(fit.map.scale[1]) == -1.0
    assert float(fit.map.offset[1]) == 90.0
    assert fit.clean_fits[1] == 1
    assert float(fit.snapped_uncovered[1]) < 0.10
    assert float(fit.identity_uncovered[1]) > 0.9


def test_decisive_improvement_translates_without_clean_fit() -> None:
    """A box wider than any clean fit allows ([-160, 5] lift): +180
    still leaves ~15% uncovered (no member is clean) but beats
    identity's 100% by far more than DECISIVE_IMPROVEMENT — translate,
    with clean_fits = 0 marking the partial-coverage choice."""
    q01 = (-30.0, -160.0, -50.0, -10.0, -5.0, 2.0)
    q99 = (30.0, 5.0, 50.0, 10.0, 5.0, 40.0)
    fit = fit_convention_map(
        stats(q01, q99, mean=(0.0, -60.0, 0.0, 0.0, 0.0, 20.0)),
        TABLE,
    )
    assert fit.translated[1]
    assert float(fit.map.scale[1]) == 1.0
    assert float(fit.map.offset[1]) == 180.0
    assert fit.clean_fits[1] == 0
    assert 0.10 < float(fit.snapped_uncovered[1]) < 0.20
    assert float(fit.identity_uncovered[1]) == 1.0


def test_scale_family_keeps_identity_loudly() -> None:
    """Tick-unit datasets (the willnorris signature): no discrete
    member reaches tolerance — the joint keeps IDENTITY and is flagged
    needs_affine; only the per-dataset mode may rescue it."""
    q01 = tuple(float(v) for v in (1500, 1500, 1500, 1500, 1500, 1500))
    q99 = tuple(float(v) for v in (2500, 2500, 2500, 2500, 2500, 2500))
    fit = fit_convention_map(stats(q01, q99), TABLE)
    assert fit.map.is_identity
    assert all(fit.needs_affine)
    assert not any(fit.translated)
    assert float(fit.snapped_floor.min()) > 100.0
    assert float(fit.snapped_uncovered.min()) == 1.0


def test_identity_preferred_over_qualifying_alternative() -> None:
    """A box overlapping the table band under identity must NOT flip
    convention even if a shifted member also fits (the 3.4° sliver
    lesson: near-boundary datasets stay untranslated)."""
    q01 = (-30.0, 46.0, -50.0, -10.0, -5.0, 2.0)
    q99 = (30.0, 130.0, 50.0, 10.0, 5.0, 40.0)
    fit = fit_convention_map(stats(q01, q99), TABLE)
    assert fit.map.is_identity
    assert not fit.translated[1]


def test_quantile_equating_identity() -> None:
    """THE per-dataset-mode contract: checkpoint-table normalization of
    A(x) == dataset-quantile normalization of x, for arbitrary x."""
    dataset = stats(
        (-100.0, -104.0, -40.0, -90.0, -110.0, 2.0),
        (44.0, 49.0, 97.0, 95.0, 165.0, 39.0),
    )
    mapping = quantile_equating_map(dataset, TABLE, modality="state")
    x = torch.tensor([[-50.0, -30.0, 20.0, 0.0, 100.0, 10.0]])
    t01 = torch.tensor(TABLE.state_q01)
    t99 = torch.tensor(TABLE.state_q99)
    d01 = torch.tensor(dataset.state_q01)
    d99 = torch.tensor(dataset.state_q99)
    via_table = 2 * (mapping.apply(x) - t01) / (t99 - t01) - 1
    via_dataset = 2 * (x - d01) / (d99 - d01) - 1
    torch.testing.assert_close(via_table, via_dataset, atol=1e-5, rtol=1e-5)


def test_quantile_equating_narrow_span_guard() -> None:
    """A (near-)static joint gets a pure offset map (scale 1, midpoints
    matched) instead of a degenerate gain."""
    dataset = stats(
        (-100.0, 100.0, -40.0, -90.0, -110.0, 20.0),
        (44.0, 100.5, 97.0, 95.0, 165.0, 20.5),
    )
    mapping = quantile_equating_map(dataset, TABLE, modality="action")
    assert float(mapping.scale[1]) == 1.0
    assert float(mapping.offset[1]) == pytest.approx(115.5 - 100.25)
    assert float(mapping.scale[5]) == 1.0
    assert float(mapping.offset[5]) == pytest.approx(22.0 - 20.25)


def test_affine_map_round_trip_and_guards() -> None:
    mapping = AffineMap(
        scale=torch.tensor([2.0, -1.0]),
        offset=torch.tensor([5.0, 180.0]),
    )
    x = torch.randn(3, 4, 2)
    torch.testing.assert_close(mapping.invert(mapping.apply(x)), x)
    with pytest.raises(ValueError, match="zero scale"):
        AffineMap(scale=torch.tensor([0.0]), offset=torch.tensor([1.0]))
    with pytest.raises(ValueError, match="1-D"):
        AffineMap(scale=torch.ones(2, 2), offset=torch.ones(2, 2))


def test_fit_item_maps_modes() -> None:
    """convention-map shares ONE map across state and action; the
    per-dataset mode fits each modality on its own table; checkpoint
    mode is a wiring bug here."""
    dataset = stats(
        (-30.0, -100.0, -50.0, -10.0, -5.0, 2.0),
        (30.0, -80.0, 50.0, 10.0, 5.0, 40.0),
    )
    conv, fit = fit_item_maps(dataset, TABLE, MolmoNorm.CONVENTION_MAP)
    assert fit is not None
    assert conv.state is conv.action
    pd, none_fit = fit_item_maps(dataset, TABLE, MolmoNorm.PER_DATASET)
    assert none_fit is None
    assert not torch.equal(pd.state.scale, conv.state.scale)
    with pytest.raises(ValueError, match="checkpoint"):
        fit_item_maps(dataset, TABLE, MolmoNorm.CHECKPOINT)


def test_missing_quantiles_is_loud() -> None:
    bare = DatasetStats(
        action_mean=(0.0,) * 6,
        action_std=(1.0,) * 6,
        state_mean=(0.0,) * 6,
        state_std=(1.0,) * 6,
        action_q01=None,
        action_q99=None,
        state_q01=None,
        state_q99=None,
    )
    with pytest.raises(SystemExit, match="q01/q99"):
        fit_item_maps(bare, TABLE, MolmoNorm.PER_DATASET)
