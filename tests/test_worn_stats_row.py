"""The sim drivers' worn-row resolution (``--stats-repo-id``).

Under the per-dataset flow scheme the sampled chunks denormalize under
the row the item WEARS, so the resolver's contract is load-bearing:
the default lookup (rig row else merged) must stay bit-identical to
the banked sim100 reads, and an explicit row must resolve loudly or
refuse — a silent fallback would serve a mixed-corpus checkpoint
through the wrong window (the 288%-overflow class, isolation post
2026-08-17).
"""

import pytest

pytest.importorskip("mujoco")

from bijou.data import DatasetStats, PolicyInfo
from sim.rollout_sim import STATS_REPO_ID, resolve_worn_stats, worn_stats_key

DIM = 6


def row(width: float) -> DatasetStats:
    return DatasetStats(
        action_mean=(0.0,) * DIM,
        action_std=(1.0,) * DIM,
        state_mean=(0.0,) * DIM,
        state_std=(1.0,) * DIM,
        action_q01=(-width,) * DIM,
        action_q99=(width,) * DIM,
        state_q01=(-width,) * DIM,
        state_q99=(width,) * DIM,
    )


def info(per_dataset: dict[str, DatasetStats]) -> PolicyInfo:
    return PolicyInfo(
        chunk_size=30,
        normalization=row(1.0),
        per_dataset_normalization=per_dataset,
        condition_fields=(),
        generate_bracket=False,
    )


def test_default_wears_rig_row_when_present() -> None:
    rig = row(2.0)
    resolved = resolve_worn_stats(info({STATS_REPO_ID: rig}), None)
    assert resolved is rig


def test_default_falls_back_to_merged_table() -> None:
    merged_info = info({"grasp_demos_v2/merged": row(3.0)})
    resolved = resolve_worn_stats(merged_info, None)
    assert resolved is merged_info.normalization


def test_explicit_row_resolves_exactly() -> None:
    demos = row(3.0)
    per_dataset = {STATS_REPO_ID: row(2.0), "grasp_demos_v2/merged": demos}
    resolved = resolve_worn_stats(info(per_dataset), "grasp_demos_v2/merged")
    assert resolved is demos


def test_explicit_row_never_falls_back() -> None:
    with pytest.raises(SystemExit, match="grasp_demos_v2/merged"):
        resolve_worn_stats(info({STATS_REPO_ID: row(2.0)}), "grasp_demos_v2/merged")


def test_worn_key_records_the_resolved_row_not_the_lookup_key() -> None:
    """The out-json record must name the row actually worn: the default
    lookup's merged-table fallback is NOT the rig key it looked up
    (recording the key there mislabeled the demosonly sim100 legs)."""
    assert worn_stats_key(info({STATS_REPO_ID: row(2.0)}), None) == STATS_REPO_ID
    assert worn_stats_key(info({"grasp_demos_v2/merged": row(3.0)}), None) == (
        "<merged-table>"
    )
    assert (
        worn_stats_key(
            info({STATS_REPO_ID: row(2.0), "grasp_demos_v2/merged": row(3.0)}),
            "grasp_demos_v2/merged",
        )
        == "grasp_demos_v2/merged"
    )
