"""Oracles for fontaine/scripts/snapflow_results.py (SnapFlow endpoint reads).

The script's --oracle mode runs the full machinery against the banked
teacher JSONs/npz (gitignored data); these tests keep the decision
logic, the semantics guards, and the per-step curve math under check.py
on synthetic fixtures only.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "fontaine" / "scripts" / "snapflow_results.py"
)
spec = importlib.util.spec_from_file_location("snapflow_results", SCRIPT)
assert spec is not None and spec.loader is not None
sfr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sfr)


def test_primary_classification_band_edges() -> None:
    assert sfr.classify_primary(6.60) == "parity-adopt"
    assert sfr.classify_primary(sfr.ADOPT_LINE) == "parity-adopt"  # inclusive
    assert sfr.classify_primary(sfr.ADOPT_LINE + 1e-9) == "intermediate-miss"
    assert sfr.classify_primary(sfr.FALSIFY_LINE) == "intermediate-miss"  # <=
    assert sfr.classify_primary(sfr.FALSIFY_LINE + 1e-9) == "falsified"


def test_registered_lines_are_the_preregistered_arithmetic() -> None:
    assert pytest.approx(sfr.TEACHER_CHUNK + 0.15) == sfr.ADOPT_LINE
    assert pytest.approx(sfr.TEACHER_CHUNK + 0.5) == sfr.FALSIFY_LINE
    assert pytest.approx(sfr.TEACHER_FIRST + 0.05) == sfr.EDGE_LINE
    assert sfr.DEPLOY_LINE == sfr.AR_CHUNK
    assert pytest.approx(sfr.TEACHER_PROBE_CHUNK + 3.0) == sfr.KILL_LINE


def test_verdict_all_branches() -> None:
    v = sfr.verdict("parity-adopt", edge_ok=True, deploy_ok=True, probe_killed=None)
    assert v["primary_verdict"].startswith("PARITY")
    assert "DEPLOYMENT HEADLINE" in v["adoption"]
    v = sfr.verdict("parity-adopt", edge_ok=True, deploy_ok=False, probe_killed=False)
    assert "did NOT survive" in v["adoption"]
    v = sfr.verdict(
        "intermediate-miss",
        edge_ok=False,
        deploy_ok=False,
        probe_killed=None,
    )
    assert v["primary_verdict"].startswith("MISS")
    assert v["adoption"] == "NO ADOPTION"
    v = sfr.verdict("falsified", edge_ok=True, deploy_ok=True, probe_killed=False)
    assert v["primary_verdict"].startswith("FALSIFIED")
    assert v["adoption"] == "NO ADOPTION"
    v = sfr.verdict("parity-adopt", edge_ok=True, deploy_ok=True, probe_killed=True)
    assert "KILL LINE FIRED" in v["adoption"]


def test_endpoint_loader_guards() -> None:
    good = sfr.make_endpoint_json(6.70, 1.95, 1)
    row = sfr.load_endpoint(good, 1, "t")
    assert row["chunk_mae"] == 6.70 and row["first_mae"] == 1.95
    for k, v in [
        ("sample_steps", 30),
        ("sample_method", "heun"),
        ("target_time", "t"),
        ("noise_key", "stable"),
        ("sample_plan", "plans/other.json"),
        ("core_frames", 2458),
    ]:
        with pytest.raises(SystemExit):
            sfr.load_endpoint(sfr.make_endpoint_json(6.70, 1.95, 1, **{k: v}), 1, "t")
    with pytest.raises(SystemExit):  # draws mismatch
        sfr.load_endpoint(sfr.make_endpoint_json(5.5, 1.4, 10), 5, "t")
    with pytest.raises(SystemExit):  # missing _drawsN suffix
        bad = sfr.make_endpoint_json(5.5, 1.4, 10)
        bad["summaries"][1]["policy"] = "bijou@30000"
        sfr.load_endpoint(bad, 10, "t")


def test_probe_loader_and_kill_line() -> None:
    probe = sfr.load_probe(
        sfr.make_endpoint_json(
            9.0,
            3.0,
            1,
            sample_plan=sfr.PROBE_PLAN,
            core_frames=sfr.PROBE_CORE,
        ),
        "t",
    )
    e1 = sfr.load_endpoint(sfr.make_endpoint_json(6.70, 1.95, 1), 1, "t")
    e10 = sfr.load_endpoint(sfr.make_endpoint_json(5.55, 1.42, 10), 10, "t")
    res = sfr.analyze(e1, e10, None, probe, None, None)
    assert res["decision"]["probe_killed"] is False
    at_line = dict(probe, chunk_mae=sfr.KILL_LINE)
    res = sfr.analyze(e1, e10, None, at_line, None, None)
    assert res["decision"]["probe_killed"] is False  # strictly >
    over = dict(probe, chunk_mae=sfr.KILL_LINE + 0.01)
    res = sfr.analyze(e1, e10, None, over, None, None)
    assert res["decision"]["probe_killed"] is True
    with pytest.raises(SystemExit):  # panel JSON can't pose as the probe
        sfr.load_probe(sfr.make_endpoint_json(9.0, 3.0, 1), "t")


def _tiny_pair(n: int = 4, steps: int = 3, motors: int = 2) -> tuple[dict, dict, str]:
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(n, steps, motors))
    valid = np.ones((n, steps), dtype=bool)
    core = np.array([True, True, True, False])
    base = {
        "truth": truth,
        "valid": valid,
        "core": core,
        "index": np.arange(n),
        "repo_id": np.array(["r"] * n),
    }
    teacher = dict(base, files=[*list(base), "pred:bijou@80000"])
    teacher["pred:bijou@80000"] = truth + 1.0
    student = dict(base, files=[*list(base), "pred:bijou@80000"])
    return student, teacher, "pred:bijou@80000"


def test_perstep_curve_math_exact() -> None:
    student, teacher, key = _tiny_pair()
    # student: error 0.5 on steps 0-1, 2.6 on step 2 (teacher flat 1.0)
    err = np.array([0.5, 0.5, 2.6])
    student[key] = teacher["truth"] + err[None, :, None]
    ps = sfr.perstep_read(student, key, teacher, key, {}, None)
    assert ps["step_curve"]["teacher"] == [1.0, 1.0, 1.0]
    assert ps["step_curve"]["student"] == [0.5, 0.5, 2.6]
    assert ps["step_delta"] == [-0.5, -0.5, 1.6]
    assert ps["crossover_step"] == 2
    # first-k is the cumulative element-weighted mean
    assert ps["firstk_curve"]["student"] == [0.5, 0.5, 1.2]
    assert ps["firstk_crossover"] == 2
    # pooled over core rows only: mean of (0.5, 0.5, 2.6) = 1.2
    assert ps["npz_pooled_student"]["chunk_mae"] == 1.2
    assert ps["npz_pooled_student"]["first_mae"] == 0.5


def test_perstep_pairing_break_aborts() -> None:
    student, teacher, key = _tiny_pair()
    student[key] = teacher["truth"]
    student["index"] = np.array([1, 0, 2, 3])
    with pytest.raises(SystemExit):
        sfr.perstep_read(student, key, teacher, key, {}, None)


def test_v2_keep_mask() -> None:
    plan = {
        "exclusions": {
            "leaked_episodes": ["repoA::3"],
            "corrupt_repos": ["repoB"],
        },
    }
    join = np.zeros(4, dtype=[("repo", "U16"), ("episode", "i8"), ("frame", "i8")])
    join["repo"] = ["repoA", "repoA", "repoB", "repoC"]
    join["episode"] = [3, 4, 1, 3]
    keep = sfr.v2_keep_mask(plan, join)
    assert keep.tolist() == [False, True, False, True]
