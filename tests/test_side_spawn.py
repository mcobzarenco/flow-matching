"""Side-spawn reset extension oracles (probe_side_spawn feasibility
work, 2026-08-16): ``reset(boat_start="side")`` must produce a stable
side-lying boat, and must leave the historical upright path — spawn
stream included — bit-identical."""

import numpy as np
import pytest

from sim.so101_sim import SO101Sim


@pytest.fixture(scope="module")
def sim() -> SO101Sim:
    return SO101Sim(spawn_version="v2.1", width=64, height=48, render_style="v0")


def test_side_reset_rests_on_side(sim: SO101Sim) -> None:
    for seed in (1000, 1007, 1023):
        sim.reset(seed, boat_start="side")
        pos, upright = sim.benchy_pose()
        assert abs(upright) < 0.5, f"seed {seed}: upright {upright}"
        # Body origin rides at ~half the beam when lying on the hull
        # side — well above the upright rest band's top (0.012).
        assert pos[2] > 0.013
        assert not sim.success()


def test_upright_stream_bit_identical(sim: SO101Sim) -> None:
    # The side-mode roll draw comes AFTER every upright-mode draw on
    # the spawn stream, and an interleaved side reset must not leak
    # state into a later upright reset.
    sim.reset(1000)
    baseline = sim.data.qpos.copy()
    sim.reset(1001, boat_start="side")
    sim.reset(1000, boat_start="upright")
    assert np.array_equal(sim.data.qpos, baseline)


def test_unknown_boat_start_raises(sim: SO101Sim) -> None:
    with pytest.raises(ValueError, match="boat_start"):
        sim.reset(1000, boat_start="capsized")
