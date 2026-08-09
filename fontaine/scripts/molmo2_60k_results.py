"""Molmo2 60k continuation frozen reads — §"Frozen reads" of the
pre-reg (posts/2026-08-08-prereg-molmo2-ar-60k-continuation.md), no
other numbers:

  1. PRIMARY  paired per-frame Δ (60k − 40k) vs the banked 40k
     endpoint npz (identical plan/rows), core rows, seeded bootstrap
     CI95 (seed 0, 10,000). IMPROVED iff CI95 entirely below 0;
     PARITY if it spans 0; DAMAGED if entirely above.
  2. OWNER BAR pooled chunk/first vs AR-100k 5.8026/2.1431
     (cross-trunk, unpaired — quoted with that caveat). PASSES-100k
     iff pooled chunk < 5.8026.
  3. INTEGRITY state-copy / state-copy-norm rows byte-match between
     the two npzs (both = the banked panel values) — hard abort.
  4. PROBE trajectory (record-only): the rewarmed segment's low vs
     5.91@26500, passed in via --probe-low/--probe-low-step (read
     off the box jsonl, quoted verbatim).
  5. DECISION, frozen: IMPROVED + PASSES-100k ⇒ repoint the attach
     warm-start to step_060000 (amendment + K-smoke re-run);
     IMPROVED only ⇒ same repoint, bar noted honestly;
     PARITY/DAMAGED ⇒ 40k endpoint stands.

Execution oracles (each failure a hard abort): identity columns
byte-match across the two npzs; each npz re-pools to its own report
summary (chunk + first, 5e-3); policy keys carry the expected steps;
checkpoints in the reports name the expected run dirs.

``--oracle``: planted-delta fixtures (−1.0 ⇒ exact Δ, degenerate CI,
IMPROVED; 0 ⇒ PARITY; +1.0 ⇒ DAMAGED) + every abort branch.

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

STEM_40K = (
    "reports/eval__fontaine_molmo2_ar_40k_ddp4__step_040000__panel_curated_v0_k4l2"
)
STEM_60K = (
    "reports/eval__fontaine_molmo2_ar_60k_ddp4__step_060000__panel_curated_v0_k4l2"
)
OUT_DEFAULT = "reports/analysis__molmo2_60k_vs_40k_k4l2.json"
KEY_40K = "pred:bijou@40000"
KEY_60K = "pred:bijou@60000"
AR100K_BAR = bbr.ANCHORS["ar"]  # (5.8026, 2.1431)
PROBE_PRIOR_LOW = (5.91, 26500)
STATE_KEYS = ("pred:state-copy", "pred:state-copy-norm")
SUMMARY_TOL = 5e-3


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


def analyze(
    npz40: dict,
    rep40: dict,
    npz60: dict,
    rep60: dict,
    probe_low: tuple[float, int] | None,
    out_path: str | None,
) -> dict:
    # ---- execution oracles gate every number below ----
    for key in bbr.PAIR_KEYS:
        if not np.array_equal(npz40[key], npz60[key]):
            sys.exit(f"panel pairing broken on {key} between 40k and 60k — stop")
    for key in STATE_KEYS:
        a, b = npz40[key], npz60[key]
        if not (
            a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
        ):
            sys.exit(f"{key} rows do NOT byte-match between the endpoints — stop")
    _check_report(npz40, KEY_40K, rep40, "fontaine_molmo2_ar_40k_ddp4", "40k arm")
    _check_report(npz60, KEY_60K, rep60, "fontaine_molmo2_ar_60k_ddp4", "60k arm")

    truth, valid, core, w = bbr.masks(npz40)
    err40 = np.abs(npz40[KEY_40K] - truth)
    err60 = np.abs(npz60[KEY_60K] - truth)
    c40 = bbr.pooled_chunk(err40, core, w)
    f40 = bbr.pooled_first(err40, valid, core)
    c60 = bbr.pooled_chunk(err60, core, w)
    f60 = bbr.pooled_first(err60, valid, core)

    # ---- read 1: paired per-frame Δ, core rows ----
    frame40, nvalid = bbr.frame_mae(err40, w)
    frame60, _ = bbr.frame_mae(err60, w)
    keep = (nvalid > 0) & core
    deltas = (frame60 - frame40)[keep]
    lo, hi = bbr.bootstrap_ci(deltas)
    if hi < 0:
        classification = "IMPROVED"
    elif lo > 0:
        classification = "DAMAGED"
    else:
        classification = "PARITY"

    # ---- read 2: the owner bar (cross-trunk, unpaired) ----
    passes_100k = c60 < AR100K_BAR[0]

    # ---- read 5: frozen decision ----
    if classification == "IMPROVED":
        decision = (
            "REPOINT: the 60k endpoint replaces the 40k endpoint as the "
            "phase-2 flow-trunk candidate; the attach screen warm-starts "
            "from step_060000 (checkpoint-repoint amendment + K-smoke "
            "re-run before any arm launches)"
            + (
                ""
                if passes_100k
                else " — the AR-100k bar is NOT passed, noted honestly"
            )
        )
    else:
        decision = (
            f"{classification}: the 40k endpoint stands; the continuation "
            "is banked as the longer-training answer at this scale and the "
            "attach screen proceeds from step_040000 unchanged"
        )

    out: dict[str, Any] = {
        "pooled": {
            "40k": {"chunk_mae": round(c40, 5), "first_mae": round(f40, 5)},
            "60k": {"chunk_mae": round(c60, 5), "first_mae": round(f60, 5)},
        },
        "read1_paired": {
            "delta_frame_mean": round(float(deltas.mean()), 5),
            "delta_pooled": round(c60 - c40, 5),
            "ci95": [round(lo, 5), round(hi, 5)],
            "n_frames": int(keep.sum()),
            "classification": classification,
        },
        "read2_ar100k_bar": {
            "bar": AR100K_BAR,
            "delta_chunk_vs_bar": round(c60 - AR100K_BAR[0], 5),
            "passes_100k": passes_100k,
            "caveat": "cross-trunk, unpaired",
        },
        "read3_state_copy": "byte-match (both endpoints)",
        "read4_probe_segment_low": (
            {
                "low": probe_low[0],
                "step": probe_low[1],
                "prior_low": PROBE_PRIOR_LOW,
                "new_low": probe_low[0] < PROBE_PRIOR_LOW[0],
            }
            if probe_low
            else None
        ),
        "decision": decision,
    }
    print(
        f"\nread 1 (primary): Δ(60k−40k) frame-mean "
        f"{out['read1_paired']['delta_frame_mean']:+.4f}  CI95 "
        f"[{lo:+.5f}, {hi:+.5f}]  ({classification}, n={int(keep.sum())})",
    )
    print(
        f"read 2 (owner bar): 60k pooled {c60:.4f}/{f60:.4f} vs AR-100k "
        f"{AR100K_BAR[0]}/{AR100K_BAR[1]} — "
        f"{'PASSES-100k' if passes_100k else 'NOT passed'} "
        f"({c60 - AR100K_BAR[0]:+.4f}, cross-trunk unpaired)",
    )
    if probe_low:
        print(
            f"read 4 (record): segment probe low {probe_low[0]}@{probe_low[1]} "
            f"vs prior {PROBE_PRIOR_LOW[0]}@{PROBE_PRIOR_LOW[1]} — "
            f"{'NEW LOW' if probe_low[0] < PROBE_PRIOR_LOW[0] else 'no new low'}",
        )
    print(f"decision: {decision}")
    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=2))
        print(f"wrote {out_path}")
    return out


# ------------------------------------------------------------------ oracle


def _fixture(delta: float) -> tuple:
    n, chunk, dims = 10, 8, 2
    truth = np.zeros((n, chunk, dims), dtype=np.float32)
    state = np.full((n, chunk, dims), 7.0, dtype=np.float32)
    base = {
        "index": np.arange(n, dtype=np.int64),
        "truth": truth,
        "valid": np.ones((n, chunk), dtype=bool),
        "repo_id": np.array([f"repo{i % 2}" for i in range(n)]),
        "core": np.array([True] * 8 + [False] * 2),
        "pred:state-copy": state,
        "pred:state-copy-norm": state + 1,
    }
    npz40 = dict(base, **{KEY_40K: np.full((n, chunk, dims), 3.0, dtype=np.float32)})
    npz60 = dict(
        {k: v.copy() for k, v in base.items()},
        **{KEY_60K: np.full((n, chunk, dims), 3.0 + delta, dtype=np.float32)},
    )

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

    return (
        npz40,
        rep(npz40, KEY_40K, "fontaine_molmo2_ar_40k_ddp4"),
        npz60,
        rep(npz60, KEY_60K, "fontaine_molmo2_ar_60k_ddp4"),
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

    npz40, rep40, npz60, rep60 = _fixture(-1.0)
    out = analyze(npz40, rep40, npz60, rep60, (5.0, 41000), None)
    r1 = out["read1_paired"]
    assert r1["delta_frame_mean"] == -1.0 and r1["ci95"] == [-1.0, -1.0], out
    assert r1["classification"] == "IMPROVED", out
    assert out["read2_ar100k_bar"]["passes_100k"] is True, out  # planted 2.0
    assert out["read4_probe_segment_low"]["new_low"] is True, out
    assert "REPOINT" in out["decision"], out
    print("  planted −1.0 OK (exact Δ, degenerate CI, IMPROVED ⇒ repoint)")

    npz40, rep40, npz60, rep60 = _fixture(0.0)
    out = analyze(npz40, rep40, npz60, rep60, None, None)
    assert out["read1_paired"]["classification"] == "PARITY", out
    assert "40k endpoint stands" in out["decision"], out
    print("  planted 0 OK (PARITY ⇒ 40k stands)")

    npz40, rep40, npz60, rep60 = _fixture(1.0)
    out = analyze(npz40, rep40, npz60, rep60, (6.5, 55000), None)
    assert out["read1_paired"]["classification"] == "DAMAGED", out
    assert out["read4_probe_segment_low"]["new_low"] is False, out
    print("  planted +1.0 OK (DAMAGED)")

    npz40, rep40, npz60, rep60 = _fixture(-1.0)
    mut = {k: v.copy() for k, v in npz60.items()}
    mut["truth"][0] += 1.0
    expect_exit(
        lambda: analyze(npz40, rep40, mut, rep60, None, None),
        "pairing broken",
        "identity drift",
    )
    mut = {k: v.copy() for k, v in npz60.items()}
    mut["pred:state-copy"][1] += 1.0
    expect_exit(
        lambda: analyze(npz40, rep40, mut, rep60, None, None),
        "byte-match",
        "state-copy drift",
    )
    bad = dict(rep60, summaries=[dict(rep60["summaries"][0], chunk_mae=9.9)])
    expect_exit(
        lambda: analyze(npz40, rep40, npz60, bad, None, None),
        "plan/scoring drift",
        "report drift",
    )
    bad = dict(rep60, checkpoint="outputs/train/other_run/step_0")
    expect_exit(
        lambda: analyze(npz40, rep40, npz60, bad, None, None),
        "is not",
        "checkpoint mismatch",
    )
    print("oracle: ALL branches OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--stem-40k", default=STEM_40K)
    parser.add_argument("--stem-60k", default=STEM_60K)
    parser.add_argument("--probe-low", type=float, default=None)
    parser.add_argument("--probe-low-step", type=int, default=None)
    parser.add_argument("--out", default=OUT_DEFAULT)
    args = parser.parse_args()
    if args.oracle:
        oracle()
        return
    probe = (
        (args.probe_low, args.probe_low_step)
        if args.probe_low is not None and args.probe_low_step is not None
        else None
    )
    analyze(
        _load_npz(f"{args.stem_40k}.npz"),
        json.loads(Path(f"{args.stem_40k}.json").read_text()),
        _load_npz(f"{args.stem_60k}.npz"),
        json.loads(Path(f"{args.stem_60k}.json").read_text()),
        probe,
        args.out,
    )


if __name__ == "__main__":
    main()
