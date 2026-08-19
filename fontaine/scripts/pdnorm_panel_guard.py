"""Pdnorm endpoint panel guard — the registered tertiary read.

Registered consumer: pre-reg
posts/2026-08-18-prereg-grasp-sft-v2-joint-pdnorm.md, "Tertiary —
panel guard, paired at endpoint": the endpoint's k4l2 panel npz paired
vs the discriminator's banked step-1000 npz on the shared frames.
Frozen rule (house guard convention): **FAIL iff the endpoint is worse
by > +0.05 pooled chunk MAE with the paired bootstrap CI95 excluding
0**; anything else passes. Per-motor chunk-MAE deltas are recorded
alongside — wrist_flex and wrist_roll are the channels the pdnorm
mechanism predicts should move.

The guard reads npz-vs-npz AS SCORED: the endpoint wears native
per-dataset rows, disc-1000 wore the demos-recomputed global table —
the wear asymmetry is interpreted through the ladder anchors
(27.40 / 27.14 / 25.15 / 8.37) and the truthfit-rewear cross-check
(`pdnorm_endpoint_truthfit_rewear.py`), never by moving this rule.

Execution oracles (each failure a hard abort): identity columns
byte-equal across the pair; state-copy rows byte-match; each npz
re-pools to its own report json (5e-3).

``--oracle``: planted-delta fixtures (−1.0 exact Δ + degenerate CI +
PASS; 0.0 CI-SPANS-0 PASS; +1.0 FAIL; +0.03 significant-but-small
PASS; +0.05 boundary PASS) + the abort branches.

Pure CPU, read-only on inputs, deterministic (seeded bootstrap).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent


def _sibling(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bbr = _sibling("box_batch_results")

MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
STATE_KEYS = ("pred:state-copy", "pred:state-copy-norm")
SUMMARY_TOL = 5e-3
GUARD_THRESHOLD = 0.05
PREREG = "posts/2026-08-18-prereg-grasp-sft-v2-joint-pdnorm.md"


def _load_npz(path: str | Path) -> dict:
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def _check_report(npz: dict, key: str, report: dict, run_dir: str, label: str) -> None:
    ckpt = str(report.get("checkpoint", ""))
    if run_dir not in ckpt:
        sys.exit(f"{label}: report checkpoint {ckpt!r} is not {run_dir} — stop")
    truth, valid, core, w = bbr.masks(npz)
    err = np.abs(npz[key] - truth)
    gc = bbr.pooled_chunk(err, core, w)
    gf = bbr.pooled_first(err, valid, core)
    policy = key.removeprefix("pred:")
    summ = [s for s in report.get("summaries", []) if s.get("policy") == policy]
    if len(summ) != 1:
        sys.exit(f"{label}: report has {len(summ)} summaries for {policy!r} — stop")
    wc, wf = summ[0]["chunk_mae"], summ[0]["first_mae"]
    if abs(gc - wc) >= SUMMARY_TOL or abs(gf - wf) >= SUMMARY_TOL:
        sys.exit(
            f"{label}: npz re-pool {gc:.4f}/{gf:.4f} does not reproduce the "
            f"report's {wc:.4f}/{wf:.4f} — plan/scoring drift, stop",
        )
    print(f"{label}: report cross-check OK ({policy} chunk {gc:.4f} first {gf:.4f})")


def _per_motor(npz: dict, key: str) -> list[float]:
    truth, valid, core, _w = bbr.masks(npz)
    mask = (valid & np.isfinite(truth).all(-1)) & core[:, None]
    return [float(v) for v in np.abs(npz[key] - truth)[mask].mean(axis=0)]


def analyze(
    cand_npz: dict,
    cand_rep: dict,
    base_npz: dict,
    base_rep: dict,
    out_path: str | None,
    cand_key: str,
    cand_run: str,
    base_key: str,
    base_run: str,
) -> dict:
    # ---- execution oracles gate every number below ----
    for key in bbr.PAIR_KEYS:
        if not np.array_equal(cand_npz[key], base_npz[key]):
            sys.exit(f"panel pairing broken on {key} — stop")
    for key in STATE_KEYS:
        a, b = cand_npz[key], base_npz[key]
        if not (
            a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
        ):
            sys.exit(f"{key} rows do NOT byte-match across the pair — stop")
    _check_report(cand_npz, cand_key, cand_rep, cand_run, "endpoint arm")
    _check_report(base_npz, base_key, base_rep, base_run, "baseline arm")

    truth, valid, core, w = bbr.masks(cand_npz)
    err_c = np.abs(cand_npz[cand_key] - truth)
    err_b = np.abs(base_npz[base_key] - truth)
    cc = bbr.pooled_chunk(err_c, core, w)
    cf = bbr.pooled_first(err_c, valid, core)
    bc = bbr.pooled_chunk(err_b, core, w)
    bf = bbr.pooled_first(err_b, valid, core)
    frame_c, nvalid = bbr.frame_mae(err_c, w)
    frame_b, _ = bbr.frame_mae(err_b, w)
    keep = (nvalid > 0) & core
    deltas = (frame_c - frame_b)[keep]
    lo, hi = bbr.bootstrap_ci(deltas)
    delta_pooled = cc - bc
    ci_excludes_zero = lo > 0 or hi < 0
    fail = delta_pooled > GUARD_THRESHOLD and lo > 0
    verdict = "FAIL" if fail else "PASS"

    motors_c = _per_motor(cand_npz, cand_key)
    motors_b = _per_motor(base_npz, base_key)
    per_motor = {
        name: {
            "endpoint": round(mc, 4),
            "baseline": round(mb, 4),
            "delta": round(mc - mb, 4),
        }
        for name, mc, mb in zip(MOTORS, motors_c, motors_b, strict=True)
    }

    out: dict[str, Any] = {
        "prereg": PREREG,
        "rule": (
            f"FAIL iff endpoint worse than baseline by > +{GUARD_THRESHOLD} "
            "pooled chunk MAE with paired bootstrap CI95 excluding 0"
        ),
        "guard": verdict,
        "pooled": {
            "endpoint": {"chunk_mae": round(cc, 5), "first_mae": round(cf, 5)},
            "baseline": {"chunk_mae": round(bc, 5), "first_mae": round(bf, 5)},
        },
        "paired": {
            "delta_pooled": round(delta_pooled, 5),
            "delta_frame_mean": round(float(deltas.mean()), 5),
            "ci95": [round(lo, 5), round(hi, 5)],
            "ci_excludes_zero": bool(ci_excludes_zero),
            "n_frames": int(keep.sum()),
        },
        "per_motor_chunk_mae": per_motor,
        "state_copy": "byte-match",
    }
    print(
        f"guard {verdict}: endpoint {cc:.4f} vs baseline {bc:.4f} "
        f"(Δ {delta_pooled:+.4f}, frame-mean {deltas.mean():+.4f}, "
        f"CI95 [{lo:+.5f}, {hi:+.5f}], n={int(keep.sum())})",
    )
    for name in ("wrist_flex", "wrist_roll"):
        m = per_motor[name]
        print(
            f"  mechanism channel {name}: {m['endpoint']:.3f} vs "
            f"{m['baseline']:.3f} (Δ {m['delta']:+.3f})",
        )
    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=2))
        print(f"wrote {out_path}")
    return out


# ------------------------------------------------------------------ oracle

ORACLE_CAND_KEY = "pred:bijou@3000"
ORACLE_CAND_RUN = "cand_run"
ORACLE_BASE_KEY = "pred:bijou@1000"
ORACLE_BASE_RUN = "base_run"


def _fixture(delta: float) -> tuple:
    n, chunk, dims = 10, 8, len(MOTORS)
    truth = np.zeros((n, chunk, dims), dtype=np.float32)
    state = np.full((n, chunk, dims), 7.0, dtype=np.float32)
    base_cols = {
        "index": np.arange(n, dtype=np.int64),
        "truth": truth,
        "valid": np.ones((n, chunk), dtype=bool),
        "repo_id": np.array([f"repo{i % 2}" for i in range(n)]),
        "core": np.array([True] * 8 + [False] * 2),
        "pred:state-copy": state,
        "pred:state-copy-norm": state + 1,
    }

    def rep(npz: dict, key: str, run: str) -> dict:
        truth_, valid_, core_, w_ = bbr.masks(npz)
        err = np.abs(npz[key] - truth_)
        return {
            "checkpoint": f"outputs/train/{run}/step_0",
            "summaries": [
                {
                    "policy": key.removeprefix("pred:"),
                    "chunk_mae": bbr.pooled_chunk(err, core_, w_),
                    "first_mae": bbr.pooled_first(err, valid_, core_),
                },
            ],
        }

    cand = dict(
        {k: v.copy() for k, v in base_cols.items()},
        **{ORACLE_CAND_KEY: np.full((n, chunk, dims), 3.0 + delta, dtype=np.float32)},
    )
    base = dict(
        {k: v.copy() for k, v in base_cols.items()},
        **{ORACLE_BASE_KEY: np.full((n, chunk, dims), 3.0, dtype=np.float32)},
    )
    return (
        cand,
        rep(cand, ORACLE_CAND_KEY, ORACLE_CAND_RUN),
        base,
        rep(base, ORACLE_BASE_KEY, ORACLE_BASE_RUN),
    )


def _oracle_analyze(fix: tuple) -> dict:
    cand, crep, base, brep = fix
    return analyze(
        cand,
        crep,
        base,
        brep,
        None,
        ORACLE_CAND_KEY,
        ORACLE_CAND_RUN,
        ORACLE_BASE_KEY,
        ORACLE_BASE_RUN,
    )


def oracle() -> None:
    def expect_exit(fn: Callable[[], object], needle: str, label: str) -> None:
        try:
            fn()
        except SystemExit as err:
            if needle not in str(err):
                raise AssertionError(
                    f"{label}: aborted with {err!r}, wanted {needle!r}",
                ) from None
            print(f"  abort branch OK: {label}")
            return
        raise AssertionError(f"{label}: did not abort")

    out = _oracle_analyze(_fixture(-1.0))
    assert out["paired"]["delta_frame_mean"] == -1.0, out
    assert out["paired"]["ci95"] == [-1.0, -1.0], out
    assert out["guard"] == "PASS", out
    assert out["per_motor_chunk_mae"]["shoulder_pan"]["delta"] == -1.0, out
    print("  planted −1.0 OK (exact Δ, degenerate CI, PASS)")

    out = _oracle_analyze(_fixture(0.0))
    assert not out["paired"]["ci_excludes_zero"], out
    assert out["guard"] == "PASS", out
    print("  planted 0 OK (CI-SPANS-0, PASS)")

    out = _oracle_analyze(_fixture(1.0))
    assert out["guard"] == "FAIL", out
    print("  planted +1.0 OK (FAIL)")

    out = _oracle_analyze(_fixture(0.03))
    assert out["paired"]["ci_excludes_zero"], out
    assert out["guard"] == "PASS", out
    print("  planted +0.03 OK (significant but ≤ threshold, PASS)")

    out = _oracle_analyze(_fixture(GUARD_THRESHOLD))
    assert out["guard"] == "PASS", out
    print("  planted +0.05 OK (boundary is not > threshold, PASS)")

    cand, crep, base, brep = _fixture(-1.0)
    mut = {k: v.copy() for k, v in base.items()}
    mut["truth"][0] += 1.0
    expect_exit(
        lambda: analyze(
            cand,
            crep,
            mut,
            brep,
            None,
            ORACLE_CAND_KEY,
            ORACLE_CAND_RUN,
            ORACLE_BASE_KEY,
            ORACLE_BASE_RUN,
        ),
        "pairing broken",
        "identity-column mismatch",
    )

    cand, crep, base, brep = _fixture(-1.0)
    mut = {k: v.copy() for k, v in base.items()}
    mut["pred:state-copy"][0] += 1.0
    expect_exit(
        lambda: analyze(
            cand,
            crep,
            mut,
            brep,
            None,
            ORACLE_CAND_KEY,
            ORACLE_CAND_RUN,
            ORACLE_BASE_KEY,
            ORACLE_BASE_RUN,
        ),
        "byte-match",
        "state-copy mismatch",
    )

    cand, crep, base, brep = _fixture(-1.0)
    crep["summaries"][0]["chunk_mae"] += 1.0
    expect_exit(
        lambda: analyze(
            cand,
            crep,
            base,
            brep,
            None,
            ORACLE_CAND_KEY,
            ORACLE_CAND_RUN,
            ORACLE_BASE_KEY,
            ORACLE_BASE_RUN,
        ),
        "does not reproduce",
        "report re-pool mismatch",
    )

    cand, crep, base, brep = _fixture(-1.0)
    brep["checkpoint"] = "outputs/train/wrong_run/step_0"
    expect_exit(
        lambda: analyze(
            cand,
            crep,
            base,
            brep,
            None,
            ORACLE_CAND_KEY,
            ORACLE_CAND_RUN,
            ORACLE_BASE_KEY,
            ORACLE_BASE_RUN,
        ),
        "is not",
        "wrong checkpoint run",
    )
    print("oracle: ALL GREEN")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cand-npz")
    ap.add_argument("--cand-json")
    ap.add_argument("--cand-key", default="pred:bijou@3000")
    ap.add_argument("--cand-run", default="grasp_sft_v2_joint_1gpu_pdnorm")
    ap.add_argument("--base-npz")
    ap.add_argument("--base-json")
    ap.add_argument("--base-key", default="pred:bijou@1000")
    ap.add_argument("--base-run", default="grasp_sft_v2_demosonly_1gpu_disc")
    ap.add_argument("--out")
    ap.add_argument("--oracle", action="store_true")
    args = ap.parse_args()
    if args.oracle:
        oracle()
        return
    required = ("cand_npz", "cand_json", "base_npz", "base_json", "out")
    missing = [k for k in required if not getattr(args, k)]
    if missing:
        ap.error(f"missing required arguments: {missing}")
    analyze(
        _load_npz(args.cand_npz),
        json.loads(Path(args.cand_json).read_text()),
        _load_npz(args.base_npz),
        json.loads(Path(args.base_json).read_text()),
        args.out,
        args.cand_key,
        args.cand_run,
        args.base_key,
        args.base_run,
    )


if __name__ == "__main__":
    main()
