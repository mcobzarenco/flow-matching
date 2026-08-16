"""--recompute-stats gates: the exact pooled-quantile math, the
derived-checkpoint materialization, and the args refusals."""

from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from bijou.checkpoint import (
    VLAMetadata,
    derive_with_stats,
    read_metadata,
    validate_checkpoint,
    write_checkpoint,
)
from bijou.data import DatasetStats, pooled_quantile_stats
from bijou.vla import VLAFamily


def test_pooled_quantile_stats_exact() -> None:
    rng = np.random.default_rng(0)
    action = rng.normal(size=(10_000, 3)) * np.array([1.0, 10.0, 100.0])
    state = rng.normal(size=(10_000, 2))
    stats = pooled_quantile_stats(action, state)
    assert stats.action_q01 is not None and stats.action_q99 is not None
    expected = np.quantile(action, [0.01, 0.99], axis=0)
    np.testing.assert_allclose(stats.action_q01, expected[0], rtol=1e-12)
    np.testing.assert_allclose(stats.action_q99, expected[1], rtol=1e-12)
    np.testing.assert_allclose(stats.action_mean, action.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(stats.action_std, action.std(axis=0), rtol=1e-12)


def test_pooled_quantile_stats_is_pooling_not_averaging() -> None:
    """The mixture quantile of two disjoint boxes is NOT the average of
    their per-dataset quantiles — the composition bug the aggregate
    fallback's docstring memorializes, pinned here as the reason this
    function exists."""
    low = np.full((1_000, 1), -100.0)
    high = np.full((9_000, 1), 50.0)
    pooled = np.concatenate([low, high])
    stats = pooled_quantile_stats(pooled, pooled)
    assert stats.action_q01 is not None
    # the pooled 1% sits inside the low mass, nowhere near a count-
    # weighted average of per-dataset q01s (which would be ≈ -115/…)
    assert stats.action_q01[0] == pytest.approx(-100.0)


def test_pooled_quantile_stats_refuses_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        pooled_quantile_stats(np.zeros((0, 6)), np.zeros((0, 6)))


def test_oriented_like_preserves_source_orientation() -> None:
    """A remapped source's descending (sign-flipped) pair must stay
    descending in the recomputed table — magnitudes pooled, direction
    the model's — or the recompute re-inverts the axis the remap
    fixed. Ascending joints and reference-less rows pass through."""
    from bijou.data import oriented_like

    pooled = stats(0.0)  # ascending everywhere: q01=-1, q99=+1
    reference = DatasetStats.from_state_dict(
        {
            "action": {
                "mean": [0.0] * 6,
                "std": [1.0] * 6,
                # joint 1 descending (the remapped lift), others ascending
                "q01": [-5.0, 44.8, -5.0, -5.0, -5.0, -5.0],
                "q99": [5.0, -96.1, 5.0, 5.0, 5.0, 5.0],
            },
            "observation.state": {
                "mean": [0.0] * 6,
                "std": [1.0] * 6,
            },
        },
    )
    oriented = oriented_like(pooled, reference)
    assert oriented.action_q01 is not None and oriented.action_q99 is not None
    # flipped joint: pooled magnitudes, swapped to descending
    assert oriented.action_q01[1] == pytest.approx(1.0)
    assert oriented.action_q99[1] == pytest.approx(-1.0)
    # ascending joints untouched
    assert oriented.action_q01[0] == pytest.approx(-1.0)
    assert oriented.action_q99[0] == pytest.approx(1.0)
    # reference state row has no quantiles: pooled passes through
    assert oriented.state_q01 == pooled.state_q01
    # orientation-preserving normalization identity: for the flipped
    # joint, normalize under the oriented pooled pair maps larger raw
    # to SMALLER normalized, matching the reference's direction
    lo, hi = oriented.action_q01[1], oriented.action_q99[1]
    n = 2.0 * (0.5 - lo) / (hi - lo) - 1.0
    assert n < 0.0  # raw +0.5 sits on the negative normalized side


def stats(value: float) -> DatasetStats:
    return DatasetStats.from_state_dict(
        {
            "action": {
                "mean": [value] * 6,
                "std": [1.0] * 6,
                "q01": [value - 1.0] * 6,
                "q99": [value + 1.0] * 6,
            },
            "observation.state": {
                "mean": [value] * 6,
                "std": [1.0] * 6,
                "q01": [value - 1.0] * 6,
                "q99": [value + 1.0] * 6,
            },
        },
    )


def checkpoint_fixture(tmp_path: Path) -> Path:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    save_file({"w": torch.zeros(2)}, str(snapshot / "model.safetensors"))
    metadata = VLAMetadata(
        family=VLAFamily.MOLMOACT2_FLOW,
        chunk_size=4,
        action_dim=6,
        backbone_id="user/tiny",
        backbone_depth="full",
        backbone_config={},
        backbone_text_trained=False,
        backbone_vision_trained=False,
        objective={"kind": "flow"},
        serving={"kind": "flow", "num_steps": 2, "method": "euler"},
        components={
            "prompt": {"config": {"kind": "molmoact2"}, "weights": False},
            "flow_decoder": {"config": {"kind": "molmo_flow"}, "weights": True},
        },
        artifacts={},
        stats=stats(0.0),
        per_dataset_stats={},
        train_args={"seed": 0},
        step=0,
        stats_note=None,
    )
    target = tmp_path / "source_ckpt"
    write_checkpoint(
        target,
        metadata=metadata,
        components={"flow_decoder": {"proj.weight": torch.ones(2, 2)}},
        backbone_text={"transformer.w": torch.zeros(3)},
        backbone_vision={"tower.w": torch.zeros(3)},
        tokenizer_files={"tokenizer.json": _tokenizer_file(tmp_path)},
    )
    return target


def _tokenizer_file(tmp_path: Path) -> Path:
    path = tmp_path / "tokenizer.json"
    if not path.exists():
        path.write_text("{}")
    return path


def test_derive_with_stats(tmp_path: Path) -> None:
    source = checkpoint_fixture(tmp_path)
    (source / "optimizer.pt").write_bytes(b"opt")
    derived_dir = tmp_path / "derived"
    new_stats = stats(42.0)
    derived = derive_with_stats(
        source,
        derived_dir,
        stats=new_stats,
        stats_note="recomputed for the test",
    )
    assert derived.stats.action_mean == (42.0,) * 6
    reread = validate_checkpoint(derived_dir)
    assert reread.stats_note == "recomputed for the test"
    # weight files are links to the source, never rewrites
    assert (derived_dir / "flow_decoder.safetensors").samefile(
        source / "flow_decoder.safetensors",
    )
    assert (derived_dir / "backbone_text.safetensors").samefile(
        source / "backbone_text.safetensors",
    )
    # optimizer is deliberately not carried; the source is untouched
    assert not (derived_dir / "optimizer.pt").exists()
    assert read_metadata(source).stats.action_mean == (0.0,) * 6
    # immutable once published
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        derive_with_stats(
            source,
            derived_dir,
            stats=new_stats,
            stats_note="again",
        )
