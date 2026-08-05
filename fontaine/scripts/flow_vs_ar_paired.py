"""Paired per-frame flow-vs-AR analysis on the k4l2 panel npzs.

Inputs (pulled from the box 2026-08-05, owner's 12:20Z evals):
  reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2.npz
  reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_k4l2_heun30.npz

Anchors (must reproduce before any delta is trusted):
  AR-100k  chunk_mae 5.8026, first_mae 2.1431
  flow-80k chunk_mae 6.6232, first_mae 1.9331

Output: JSON summary to reports/analysis__flow_vs_ar_paired_k4l2.json
and a printed report. Pure CPU, read-only on the npzs.
"""

import json
import sys
from pathlib import Path

import numpy as np

AR_NPZ = "reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2.npz"
FLOW_NPZ = "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_k4l2_heun30.npz"
ANCHORS = {"ar": (5.8026, 2.1431), "flow": (6.6232, 1.9331)}


def main() -> None:
    ar = np.load(AR_NPZ, allow_pickle=True)
    fl = np.load(FLOW_NPZ, allow_pickle=True)

    # 1. Pairing must be exact.
    for k in ["truth", "valid", "index", "repo_id", "core"]:
        assert np.array_equal(ar[k], fl[k]), f"pairing broken on {k}"
    truth, valid = ar["truth"], ar["valid"]
    repo = ar["repo_id"]
    core = ar["core"]

    pred_ar = ar["pred:bijou@100000"]
    pred_fl = fl["pred:bijou@80000"]
    pred_sc = ar["pred:state-copy"]

    # 2. Element-wise abs error, masked. Pooled = mean over valid elements
    #    (matches the report's chunk_mae weighting, per sealed_v2 notes).
    m3 = valid[:, :, None] & np.isfinite(truth).all(-1, keepdims=True)
    err = {
        n: np.abs(p - truth)
        for n, p in [("ar", pred_ar), ("flow", pred_fl), ("sc", pred_sc)]
    }

    # Report summaries pool over CORE frames only (17,204 of 25,800);
    # the rest are the labeled/aux rows.
    def pooled(e: np.ndarray) -> float:
        return float(e[core][m3.repeat(6, axis=2)[core]].mean())

    def first(e: np.ndarray) -> float:
        v0 = valid[:, 0] & core
        return float(e[v0, 0, :].mean())

    got = {n: (pooled(e), first(e)) for n, e in err.items() if n != "sc"}
    for n, (want_c, want_f) in ANCHORS.items():
        gc, gf = got[n]
        ok = abs(gc - want_c) < 5e-3 and abs(gf - want_f) < 5e-3
        print(
            f"anchor {n}: chunk {gc:.4f} (want {want_c}) first {gf:.4f} "
            f"(want {want_f}) {'OK' if ok else 'FAIL'}",
        )
        if not ok:
            sys.exit(f"anchor mismatch on {n} — metric semantics wrong, stop")

    # 3. Per-frame chunk MAE (mean over that frame's valid elements) + deltas.
    w = m3.repeat(6, axis=2).astype(np.float64)
    nvalid = w.sum(axis=(1, 2))
    frame = {
        n: (e * w).sum(axis=(1, 2)) / np.maximum(nvalid, 1) for n, e in err.items()
    }
    keep = (nvalid > 0) & core  # headline cuts are panel-consistent: core only
    d = frame["flow"] - frame["ar"]  # >0 = flow worse
    dk = d[keep]

    # 4. Per-step-in-horizon MAE curve (pooled over frames/dims at each step).
    def step_curve(e: np.ndarray) -> list[float]:
        wv = (valid & core[:, None]).astype(np.float64)
        num = (e.sum(axis=2) * wv).sum(axis=0)
        den = wv.sum(axis=0) * 6
        return (num / np.maximum(den, 1)).tolist()

    curves = {n: step_curve(e) for n, e in err.items()}
    cross = next(
        (
            i
            for i, (a, f) in enumerate(zip(curves["ar"], curves["flow"], strict=True))
            if f > a
        ),
        None,
    )

    # Deployment view: MAE pooled over horizon steps 0..k-1 (execute-k
    # -then-replan). Element-weighted like the headline metric.
    def firstk_curve(e: np.ndarray) -> list[float]:
        wv = (valid & core[:, None]).astype(np.float64)
        num = np.cumsum((e.sum(axis=2) * wv).sum(axis=0))
        den = np.cumsum(wv.sum(axis=0) * 6)
        return (num / np.maximum(den, 1)).tolist()

    fk = {n: firstk_curve(e) for n, e in err.items() if n != "sc"}
    fk_cross = next(
        (k for k, (a, f) in enumerate(zip(fk["ar"], fk["flow"], strict=True)) if f > a),
        None,
    )

    # 5. Per-repo paired deltas.
    repos = {}
    for r in np.unique(repo):
        s = keep & (repo == r)
        if s.sum() < 20:
            continue
        repos[str(r)] = {
            "n": int(s.sum()),
            "ar": round(float(frame["ar"][s].mean()), 4),
            "flow": round(float(frame["flow"][s].mean()), 4),
            "delta": round(float(d[s].mean()), 4),
            "flow_win_rate": round(float((d[s] < 0).mean()), 4),
        }

    # 6. Motion cut: state-copy per-frame MAE as the motion proxy.
    sc = frame["sc"][keep]
    qs = np.quantile(sc, [0.25, 0.5, 0.75])
    motion_cut = {}
    labels = ["q1_still", "q2", "q3", "q4_motion"]
    bins = np.digitize(sc, qs)
    for b, lab in enumerate(labels):
        s = bins == b
        motion_cut[lab] = {
            "n": int(s.sum()),
            "sc_mae": round(float(sc[s].mean()), 3),
            "ar": round(float(frame["ar"][keep][s].mean()), 4),
            "flow": round(float(frame["flow"][keep][s].mean()), 4),
            "delta": round(float(dk[s].mean()), 4),
            "flow_win_rate": round(float((dk[s] < 0).mean()), 4),
        }

    out = {
        "n_frames": int(keep.sum()),
        "pooled": {
            n: {"chunk": round(c, 4), "first": round(f, 4)} for n, (c, f) in got.items()
        },
        "paired_delta": {
            "mean": round(float(dk.mean()), 4),
            "median": round(float(np.median(dk)), 4),
            "flow_win_rate": round(float((dk < 0).mean()), 4),
            "p10": round(float(np.quantile(dk, 0.1)), 4),
            "p90": round(float(np.quantile(dk, 0.9)), 4),
        },
        "core_vs_rest": {
            lab: {
                "n": int(s.sum()),
                "delta": round(float(d[s].mean()), 4),
                "flow_win_rate": round(float((d[s] < 0).mean()), 4),
            }
            for lab, s in [("core", keep), ("labeled", (nvalid > 0) & ~core)]
        },
        "step_curve": {n: [round(v, 4) for v in c] for n, c in curves.items()},
        "crossover_step": cross,
        "firstk_curve": {n: [round(v, 4) for v in c] for n, c in fk.items()},
        "firstk_crossover": fk_cross,
        "per_repo": dict(sorted(repos.items(), key=lambda kv: kv[1]["delta"])),
        "motion_quartiles_by_statecopy_mae": motion_cut,
    }
    path = Path("reports/analysis__flow_vs_ar_paired_k4l2.json")
    with path.open("w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}")
    print(
        f"frames {out['n_frames']}; paired delta mean "
        f"{out['paired_delta']['mean']} median {out['paired_delta']['median']} "
        f"flow win rate {out['paired_delta']['flow_win_rate']}",
    )
    print(f"crossover step (flow first worse than AR): {cross}")
    for lab, v in motion_cut.items():
        print(f"  {lab}: delta {v['delta']} win {v['flow_win_rate']}")
    best = list(out["per_repo"].items())
    print(
        "flow-friendliest repos:",
        [(k.split("/")[-1], v["delta"]) for k, v in best[:3]],
    )
    print("flow-worst repos:", [(k.split("/")[-1], v["delta"]) for k, v in best[-3:]])


if __name__ == "__main__":
    main()
