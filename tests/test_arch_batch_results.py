"""Oracles for fontaine/scripts/arch_batch_results.py (arch batch #1 reads).

The script's --oracle mode runs the full machinery against the banked
teacher npz (gitignored data); these tests keep the decision logic, the
paired-read math, the strict-semantics guard, and the K1 gate under
check.py on synthetic fixtures only.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "fontaine"
    / "scripts"
    / "arch_batch_results.py"
)
spec = importlib.util.spec_from_file_location("arch_batch_results", SCRIPT)
assert spec is not None and spec.loader is not None
abr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(abr)


def _read(mean: float, ci: list) -> dict:
    return {"mean": mean, "ci95": ci}


def test_classification_boundaries() -> None:
    null = _read(0.0, [-0.1, 0.1])
    # adopt is inclusive at the -0.15 edge (pre-reg: "<= -0.15")
    c = abr.classify_arm(_read(-0.15, [-0.2, -0.1]), null)
    assert c["adopt_lever"] and c["chunk_class"] == "adopt-lever"
    # CI touching 0 blocks adoption regardless of the mean
    c = abr.classify_arm(_read(-0.3, [-0.6, 0.0]), null)
    assert not c["adopt_lever"] and c["chunk_class"] == "null"
    # actively worse beyond the band: falsified even with CI spanning 0
    c = abr.classify_arm(_read(0.16, [-0.05, 0.4]), null)
    assert c["chunk_class"] == "falsified-worse"
    # measurable but inside the band: sub-band, no adoption
    c = abr.classify_arm(_read(-0.05, [-0.08, -0.02]), null)
    assert c["chunk_class"] == "sub-band" and not c["adopt_lever"]
    # grounding is inclusive at the -0.10 edge and needs CI exclusion
    c = abr.classify_arm(null, _read(-0.10, [-0.15, -0.05]))
    assert c["grounding_moved"]
    c = abr.classify_arm(null, _read(-0.10, [-0.2, 0.0]))
    assert not c["grounding_moved"]
    c = abr.classify_arm(null, _read(-0.09, [-0.15, -0.05]))
    assert not c["grounding_moved"]


def _cls(*, adopt: bool = False, ground: bool = False) -> dict:
    return {
        "adopt_lever": adopt,
        "grounding_moved": ground,
        "chunk_class": "adopt-lever" if adopt else "null",
        "rules": {},
    }


def test_assembly_all_preregistered_branches() -> None:
    # arm A adopt: follow-on + 560 rung, no Molmo, no upstream
    d = abr.assemble_verdict({"A": _cls(adopt=True), "B": _cls()})
    text = " ".join(d["assembly"])
    assert "ADOPT-LEVER" in text and "560 rung" in text
    assert "Molmo2-4B" not in text and "upstream" not in text

    # arm B adopt: upstream offer, no 560 line
    d = abr.assemble_verdict({"A": _cls(), "B": _cls(adopt=True)})
    text = " ".join(d["assembly"])
    assert "upstream" in text and "560 rung" not in text

    # both null: Molmo promotion fires
    d = abr.assemble_verdict({"A": _cls(), "B": _cls()})
    assert any("Molmo2-4B" in line for line in d["assembly"])

    # grounding-only on A: 560 justified AND blocks the Molmo promotion
    d = abr.assemble_verdict({"A": _cls(ground=True), "B": _cls()})
    text = " ".join(d["assembly"])
    assert "560 rung" in text and "Molmo2-4B" not in text

    # grounding-only on B: no Molmo, no 560, conditioning front stays alive
    d = abr.assemble_verdict({"A": _cls(), "B": _cls(ground=True)})
    text = " ".join(d["assembly"])
    assert "Molmo2-4B" not in text and "560 rung" not in text
    assert "conditioning front alive" in text

    # partial (one arm only): flagged, never promotes Molmo
    d = abr.assemble_verdict({"A": _cls()})
    text = " ".join(d["assembly"])
    assert "PARTIAL" in text and "Molmo2-4B" not in text

    # control-only (no arm data): band-check-only line, never promotes Molmo
    d = abr.assemble_verdict({})
    text = " ".join(d["assembly"])
    assert "CONTROL-ONLY" in text and "Molmo2-4B" not in text


def _fixture(scale_a: float) -> tuple:
    rng = np.random.default_rng(0)
    n, k, dims = 48, 5, 4
    truth = rng.normal(size=(n, k, dims))
    valid = np.ones((n, k), dtype=bool)
    core = np.ones(n, dtype=bool)
    c_pred = truth + rng.normal(size=truth.shape)
    d = {
        "truth": truth,
        "valid": valid,
        "core": core,
        "index": np.arange(n),
        "repo_id": np.array([f"r{i % 4}" for i in range(n)]),
    }
    key = "pred:bijou@40000"
    ctrl = abr._DictNpz({**d, key: c_pred})
    arm = abr._DictNpz({**d, key: truth + scale_a * (c_pred - truth)})
    return ctrl, arm, key


def test_analyze_degenerate_exact_zero() -> None:
    ctrl, _, key = _fixture(1.0)
    res = abr.analyze({"A": ctrl, "B": ctrl}, {"A": key, "B": key}, ctrl, key, None)
    for name in ("A", "B"):
        r1 = res["arms"][name]["read1_primary_dchunk"]
        r2 = res["arms"][name]["read2_grounding_dfirst"]
        assert r1["mean"] == 0.0 and r1["ci95"] == [0.0, 0.0]
        assert r2["mean"] == 0.0 and r2["ci95"] == [0.0, 0.0]
    assert any("Molmo2-4B" in line for line in res["decision"]["assembly"])


def test_analyze_synthetic_known_delta() -> None:
    ctrl, arm, key = _fixture(1.30)
    truth, _valid, _core, w = abr.bbr.masks(ctrl)
    err_c = np.abs(ctrl[key] - truth)
    fr, _ = abr.bbr.frame_mae(err_c, w)
    want_chunk = 0.30 * float(fr.mean())
    want_first = 0.30 * float(abr.first_rows(err_c).mean())
    res = abr.analyze({"A": arm}, {"A": key}, ctrl, key, None)
    r1 = res["arms"]["A"]["read1_primary_dchunk"]
    r2 = res["arms"]["A"]["read2_grounding_dfirst"]
    assert abs(r1["mean"] - want_chunk) < 1e-5  # 5-dp rounding in paired_read
    assert abs(r2["mean"] - want_first) < 1e-5
    assert r1["ci95"][0] > 0  # uniform inflation: CI cleanly above zero
    assert res["arms"]["A"]["classification"]["chunk_class"] == "falsified-worse"


def test_analyze_pairing_abort() -> None:
    ctrl, arm, key = _fixture(1.0)
    broken = abr._DictNpz(dict(arm))
    broken["index"] = arm["index"][::-1].copy()
    with pytest.raises(SystemExit):
        abr.analyze({"A": broken}, {"A": key}, ctrl, key, None)


def test_strict_endpoint_json_guard() -> None:
    good = {
        "sample_steps": 30,
        "sample_method": "heun",
        "sample_draws": 1,
        "target_time": "t",
        "noise_key": "stable",
        "mask_state": False,
        "sample_plan": abr.V2_PLAN,
        "core_frames": abr.V2_CORE,
        "labeled_frames": abr.V2_LABELED,
    }
    abr.load_endpoint_json(dict(good), "good")
    for k, bad in [
        ("sample_steps", 1),
        ("sample_method", "euler"),
        ("sample_draws", 10),
        ("noise_key", "index"),
        ("mask_state", True),
        ("sample_plan", abr.V1_PLAN),
        ("core_frames", 17204),
    ]:
        doctored = {**good, k: bad}
        with pytest.raises(SystemExit):
            abr.load_endpoint_json(doctored, f"bad-{k}")


def test_k1_gate(tmp_path: Path) -> None:
    probe = tmp_path / "teacher_probe.json"
    probe.write_text(
        json.dumps({"probe": {"4500": 10.0, "5000": 9.0, "6000": 8.0}}),
    )
    log = tmp_path / "train_log.jsonl"
    rows = [
        {"step": 4500, "eval_chunk_mae": 19.5},  # below the 5k gate: ignored
        {"step": 5000, "eval_chunk_mae": 11.9},  # +2.9: clean
        {"step": 5500, "eval_chunk_mae": 99.0},  # no teacher point: skipped
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows))
    res = abr.k1_check(str(log), str(probe), "t")
    assert not res["kill"] and res["n_evals_checked"] == 1
    rows.append({"step": 6000, "eval_chunk_mae": 11.1})  # +3.1: kill
    log.write_text("\n".join(json.dumps(r) for r in rows))
    res = abr.k1_check(str(log), str(probe), "t")
    assert res["kill"]
    assert res["violations"] == [
        {"step": 6000, "arm_probe": 11.1, "teacher_probe": 8.0, "excess": 3.1},
    ]
