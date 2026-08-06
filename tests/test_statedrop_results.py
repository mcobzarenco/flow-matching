"""Oracles for fontaine/scripts/statedrop_results.py (arm C frozen reads).

The script's --oracle mode runs the full machinery against the banked
A-s0 panel npz (gitignored data); these tests keep the decision logic and
the paired-read math under check.py on synthetic fixtures only.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "fontaine"
    / "scripts"
    / "statedrop_results.py"
)
spec = importlib.util.spec_from_file_location("statedrop_results", SCRIPT)
assert spec is not None and spec.loader is not None
sdr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sdr)


def test_capability_boundaries() -> None:
    assert sdr.classify_capability(2.43) == "qualitative-first"  # < 2.4316
    assert sdr.classify_capability(2.44) == "strong"
    assert sdr.classify_capability(5.99) == "strong"
    assert sdr.classify_capability(6.0) == "partial"
    assert sdr.classify_capability(14.99) == "partial"
    assert sdr.classify_capability(15.0) == "failed"


def test_verdict_all_preregistered_branches() -> None:
    v = sdr.verdict(-0.2, sanity_pass=True, capability="strong", probe_final=None)
    assert v["primary_verdict"].startswith("HELPS")
    assert v["adoption"].startswith("ADOPT as recipe default")

    v = sdr.verdict(0.0, sanity_pass=True, capability="strong", probe_final=None)
    assert v["primary_verdict"].startswith("NEUTRAL")
    assert v["adoption"].startswith("ADOPT as free hardening lever")

    v = sdr.verdict(0.0, sanity_pass=False, capability="failed", probe_final=None)
    assert v["adoption"] == "NO ADOPTION"
    assert "MECHANISM INERT" in v["branch"]

    v = sdr.verdict(0.0, sanity_pass=True, capability="partial", probe_final=None)
    assert "no pre-registered adoption path" in v["branch"]

    # sanity gate failed blocks the neutral-adopt path even with capability
    v = sdr.verdict(0.0, sanity_pass=False, capability="strong", probe_final=None)
    assert not v["adoption"].startswith("ADOPT")

    v = sdr.verdict(0.2, sanity_pass=True, capability="strong", probe_final=None)
    assert v["primary_verdict"].startswith("COSTS")
    assert "p=0.3 screen" in v["branch"]

    v = sdr.verdict(0.2, sanity_pass=False, capability="failed", probe_final=None)
    assert "FALSIFIED" in v["branch"]

    # E3 formal final gate overrides any adoption, reads stand
    v = sdr.verdict(-0.2, sanity_pass=True, capability="strong", probe_final=10.0)
    assert "NO ADOPTION PATH" in v["adoption"]
    v = sdr.verdict(-0.2, sanity_pass=True, capability="strong", probe_final=9.9)
    assert v["adoption"].startswith("ADOPT as recipe default")

    # band edges are inclusive-neutral (the rule is strict inequality outside)
    for m in (-0.15, 0.15):
        v = sdr.verdict(m, sanity_pass=True, capability="strong", probe_final=None)
        assert v["primary_verdict"].startswith("NEUTRAL")


def _synthetic_pair(scale_c: float) -> tuple:
    """A-s0 panel + C panel with C's abs errors scaled by exactly scale_c."""
    rng = np.random.default_rng(0)
    n, k, dims = 32, 3, 4
    truth = rng.normal(size=(n, k, dims))
    valid = np.ones((n, k), dtype=bool)
    core = np.ones(n, dtype=bool)
    a_pred = truth + rng.normal(size=truth.shape)
    d = {
        "truth": truth,
        "valid": valid,
        "core": core,
        "index": np.arange(n),
        "repo_id": np.array([f"r{i % 4}" for i in range(n)]),
    }
    a = sdr.spr._DictNpz({**d, "pred:bijou@40000": a_pred})
    c = sdr.spr._DictNpz(
        {**d, "pred:bijou@40000": truth + scale_c * (a_pred - truth)},
    )
    return a, c, truth, a_pred


def test_analyze_synthetic_known_delta() -> None:
    a, c, truth, a_pred = _synthetic_pair(scale_c=1.5)
    key = "pred:bijou@40000"
    # masked side: C's own subset rows unmodified -> read 2 deltas exactly 0
    pos = np.arange(0, 32, 4)
    m_key = key + "_state-masked"
    masked = sdr.spr._DictNpz(
        {
            "truth": truth[pos],
            "valid": a["valid"][pos],
            "core": a["core"][pos],
            "index": a["index"][pos],
            "repo_id": a["repo_id"][pos],
            m_key: c[key][pos],
        },
    )
    res = sdr.analyze(c, key, a, key, masked, m_key, pos, None, None)

    # read 1: per-frame delta = 0.5 * A's per-frame MAE, exact up to the
    # report rounding (5 dp on the mean, 4 dp on the pooled values)
    err_a = np.abs(a_pred - truth)
    want = 0.5 * float(err_a.mean())
    got = res["read1_primary_C_minus_As0"]["mean"]
    assert abs(got - want) < 1e-5
    assert res["read2_reliance"]["delta_first"]["mean"] == 0.0
    assert res["read2_reliance"]["delta_first"]["ci95"] == [0.0, 0.0]
    assert res["read2_reliance"]["sanity_gate"]["passed"]
    # pooled chunk delta agrees with the frame-level delta on this
    # fully-valid fixture (identical weighting)
    assert abs(res["read1_primary_C_minus_As0"]["pooled_dchunk"] - want) < 2e-4


def test_analyze_degenerate_same_arm() -> None:
    a, _c, truth, _a_pred = _synthetic_pair(scale_c=1.0)
    key = "pred:bijou@40000"
    pos = np.arange(0, 32, 4)
    m_key = key + "_state-masked"
    masked = sdr.spr._DictNpz(
        {
            "truth": truth[pos],
            "valid": a["valid"][pos],
            "core": a["core"][pos],
            "index": a["index"][pos],
            "repo_id": a["repo_id"][pos],
            m_key: a[key][pos],
        },
    )
    res = sdr.analyze(a, key, a, key, masked, m_key, pos, None, None)
    r1 = res["read1_primary_C_minus_As0"]
    assert r1["mean"] == 0.0
    assert r1["ci95"] == [0.0, 0.0]
    # the fixture's first_mae (~0.8) beats the banked A-s0 constant and the
    # zero-delta masked side passes the sanity gate -> the joint read must
    # take the collapsed-reliance attribution path
    assert res["read3_grounding"]["joint_interpretation"].startswith("vision did it")
