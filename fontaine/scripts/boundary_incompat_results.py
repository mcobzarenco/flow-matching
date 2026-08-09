"""Boundary-incompatibility read — cross-chunk seam disagreement on banked panel npz.

EXPLORATORY, NOT PRE-REGISTERED (ideas #1/#22, SEAM lit rung banked
2026-08-09, papers/seam-boundary-steering.md): SEAM (2607.04609) targets
*cross-chunk mode incompatibility* — adjacent chunks generated from
independent noise picking different valid modes, so the executed
trajectory jerks at the boundary. Our SDN read measured only
*within-chunk* smoothness (null); this read measures the cross-chunk
term directly from banked stacks. Record-only — no decision rule; a
null closes the #22 bridging direction for our stack at zero GPU, a
real signal is design input for any SEAM/PAINT-class arm at the #16
rig bench. Escalation needs its own pre-registration.

The geometry: panel plans place ~6 frames per episode; whenever two
frames of the same episode sit dt < 50 apart, their predicted chunks
overlap on 50-dt steps — the earlier chunk's tail and the later
chunk's head are predictions FOR THE SAME ACTIONS from observations
dt ticks apart. Truth chunks agree byte-exactly on every overlap
(asserted, hard abort — this also proves chunk steps == frame steps).

Reads (all record-only, banked pooling conventions):
  * seam disagreement D — valid-masked mean |early_tail - late_head|
    over the overlap, per pair; pooled mean + bootstrap CI95 + LORO
    (box_batch_results conventions).
  * anchors — (1) model error on the same overlap (tail-vs-truth,
    head-vs-truth, same mask): D/err says whether chunks disagree
    within their own error budget or beyond it; (2) within-chunk
    smoothness W (mean per-step |a[t+1]-a[t]| of the later chunk, and
    of truth) — the SDN-read anchor the queue item names.
  * boundary jump J — |late_head[0] - early_tail[0]|, the executed-
    trajectory switch cost at the seam; J/W_truth = boundary jerk in
    units of typical motion (the SEAM paper's jerk framing).
  * dt curve — D pooled in 7 bins of 7 over dt in [1,49], plus a
    near-zero intercept (dt <= 5): at dt -> 0 the observations are
    nearly identical, so any residual D is the pure noise/mode term
    (deterministic policies must go to ~0 there; fresh-noise flow
    keeps its mode-flip floor).
  * state-copy reference — same stats on pred:state-copy, whose D is
    exactly |state(f1) - state(f2)|: how far the scene moved in dt.

Execution guards (hard aborts): episode/frame identity columns
present; truth byte-agreement on every valid overlap (tol 1e-5,
observed exactly 0 on all 13,693 pairs of the k4l2 panels); at least
one qualifying pair; no NaN/inf in predictions on valid steps;
exactly one non-state-copy pred key unless --pred names it.

Oracle mode (--oracle, synthetic pre-data): planted compatible pair
-> D exactly 0; planted constant-offset pair -> D == offset exactly;
degenerate same-frame overlap (dt=0) -> D exactly 0; NaN poison on
invalid steps changes nothing; hand fixture (dt=48, 2-step overlap)
reproduces D/J/W/err to float64; dt-binning recovers planted
per-bin offsets; abort battery (misaligned truth, missing identity
columns, NaN in valid region, missing pred key, no qualifying pairs).

Defaults = the five banked full-panel stacks (flow-80k stablekey /
ticket33 / draws10-seating mean-of-draws; molmo2 AR 40k / 60k
greedy). Pure CPU, read-only on inputs.
"""

import argparse
import importlib.util
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPORTS = _HERE.parent.parent / "reports"


def _sibling(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bbr = _sibling("box_batch_results")

DEFAULT_STACKS = {
    "flow80k_stablekey": "eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_curated_v0_k4l2_stablekey_heun30.npz",
    "flow80k_ticket33": "eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_curated_v0_k4l2_ticket33_heun30.npz",
    "flow80k_draws10mean": "eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_curated_v0_k4l2_draws10_seating_heun30.npz",
    "molmo2_ar40k_greedy": "eval__fontaine_molmo2_ar_40k_ddp4__step_040000__panel_curated_v0_k4l2.npz",
    "molmo2_ar60k_greedy": "eval__fontaine_molmo2_ar_60k_ddp4__step_060000__panel_curated_v0_k4l2.npz",
}
TRUTH_TOL = 1e-5
DT_BINS = [(lo, lo + 6) for lo in range(1, 50, 7)]  # 7 bins covering 1..49
INTERCEPT_DT_MAX = 5


def mine_pairs(
    repo: np.ndarray,
    episode: np.ndarray,
    frame: np.ndarray,
    chunk_len: int,
    min_dt: int = 1,
) -> list[tuple[int, int, int]]:
    """(row_early, row_late, dt) for all same-episode pairs with
    min_dt <= dt < chunk_len. Deterministic order."""
    groups: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for i in range(len(repo)):
        groups[(str(repo[i]), int(episode[i]))].append((int(frame[i]), i))
    pairs = []
    for key in sorted(groups):
        rows = sorted(groups[key])
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                dt = rows[b][0] - rows[a][0]
                if min_dt <= dt < chunk_len:
                    pairs.append((rows[a][1], rows[b][1], dt))
    return pairs


def _pooled(vals: np.ndarray, repo: np.ndarray) -> dict:
    ci = bbr.bootstrap_ci(vals)
    lo = bbr.loro(vals, repo)
    influential = sorted(
        lo.items(),
        key=lambda kv: abs(kv[1]["mean_without"] - float(vals.mean())),
        reverse=True,
    )[:3]
    return {
        "mean": round(float(vals.mean()), 5),
        "median": round(float(np.median(vals)), 5),
        "ci95": [round(ci[0], 5), round(ci[1], 5)],
        "n_pairs": len(vals),
        "most_influential_repos": [{"repo": k, **v} for k, v in influential],
    }


def seam_read(npz: dict, pred_key: str, min_dt: int = 1) -> dict:
    """The full per-stack read for one prediction key. npz is a dict of
    arrays (np.load result or fixture)."""
    for col in ("episode_index", "frame_index"):
        if col not in npz:
            raise SystemExit(
                f"ABORT: '{col}' missing — this stack predates the identity "
                "columns; re-dump or reconstruct before reading",
            )
    truth, valid = npz["truth"], npz["valid"].astype(bool)
    repo = npz["repo_id"]
    if pred_key not in npz:
        raise SystemExit(f"ABORT: pred key '{pred_key}' not in stack")
    pred = npz[pred_key]
    chunk_len = truth.shape[1]
    if pred.shape != truth.shape:
        raise SystemExit(
            f"ABORT: pred shape {pred.shape} != truth shape {truth.shape}",
        )
    if not np.isfinite(pred[valid]).all():
        raise SystemExit(f"ABORT: NaN/inf in '{pred_key}' on valid steps")

    pairs = mine_pairs(
        repo,
        npz["episode_index"],
        npz["frame_index"],
        chunk_len,
        min_dt,
    )
    if not pairs:
        raise SystemExit("ABORT: no qualifying same-episode pairs under chunk overlap")

    rows = {
        "D": [],
        "err_tail": [],
        "err_head": [],
        "jump": [],
        "W_pred": [],
        "W_truth": [],
        "dt": [],
        "repo": [],
    }
    for i1, i2, dt in pairs:
        ov = chunk_len - dt
        m = valid[i1][dt:] & valid[i2][:ov]
        if not m.any():
            continue
        t1, t2 = truth[i1][dt:][m], truth[i2][:ov][m]
        worst = float(np.abs(t1 - t2).max())
        if worst > TRUTH_TOL:
            raise SystemExit(
                f"ABORT: truth overlap mismatch {worst:.3e} > {TRUTH_TOL} "
                f"(rows {i1},{i2}, dt {dt}) — chunk/frame alignment broken",
            )
        tail = pred[i1][dt:][m].astype(np.float64)
        head = pred[i2][:ov][m].astype(np.float64)
        t_ov = t1.astype(np.float64)
        rows["D"].append(float(np.abs(tail - head).mean()))
        rows["err_tail"].append(float(np.abs(tail - t_ov).mean()))
        rows["err_head"].append(float(np.abs(head - t_ov).mean()))
        # Boundary jump: the executed-trajectory switch at the seam —
        # only defined when the seam step itself is valid on both sides.
        if m[0]:
            rows["jump"].append(
                float(
                    np.abs(
                        pred[i2][0].astype(np.float64)
                        - pred[i1][dt].astype(np.float64),
                    ).mean(),
                ),
            )
        else:
            rows["jump"].append(np.nan)
        # Within-chunk smoothness of the later chunk (and truth), over
        # consecutive valid steps of the full chunk.
        vc = valid[i2]
        cons = vc[1:] & vc[:-1]
        if cons.any():
            rows["W_pred"].append(
                float(
                    np.abs(np.diff(pred[i2].astype(np.float64), axis=0))[cons].mean(),
                ),
            )
            rows["W_truth"].append(
                float(
                    np.abs(np.diff(truth[i2].astype(np.float64), axis=0))[cons].mean(),
                ),
            )
        else:
            rows["W_pred"].append(np.nan)
            rows["W_truth"].append(np.nan)
        rows["dt"].append(dt)
        rows["repo"].append(str(repo[i1]))

    arr = {k: np.array(v) for k, v in rows.items()}
    rp = np.array(arr["repo"])
    out = {
        "pred_key": pred_key,
        "n_pairs": len(arr["D"]),
        "seam_disagreement": _pooled(arr["D"], rp),
        "err_tail_overlap": round(float(arr["err_tail"].mean()), 5),
        "err_head_overlap": round(float(arr["err_head"].mean()), 5),
    }
    err_mean = (arr["err_tail"].mean() + arr["err_head"].mean()) / 2
    out["D_over_err"] = round(float(arr["D"].mean() / err_mean), 4)
    jm = arr["jump"][~np.isnan(arr["jump"])]
    wp = arr["W_pred"][~np.isnan(arr["W_pred"])]
    wt = arr["W_truth"][~np.isnan(arr["W_truth"])]
    out["boundary_jump"] = round(float(jm.mean()), 5)
    out["within_chunk_step_pred"] = round(float(wp.mean()), 5)
    out["within_chunk_step_truth"] = round(float(wt.mean()), 5)
    out["jump_over_truth_step"] = round(float(jm.mean() / wt.mean()), 4)
    # state-copy's constant chunk has zero within-chunk motion — the
    # ratio is undefined there, not infinite.
    out["jump_over_pred_step"] = (
        round(float(jm.mean() / wp.mean()), 4) if wp.mean() > 0 else None
    )
    bins = []
    for lo, hi in DT_BINS:
        s = (arr["dt"] >= lo) & (arr["dt"] <= hi)
        if s.any():
            ci = bbr.bootstrap_ci(arr["D"][s])
            bins.append(
                {
                    "dt": [lo, hi],
                    "D_mean": round(float(arr["D"][s].mean()), 5),
                    "ci95": [round(ci[0], 5), round(ci[1], 5)],
                    "n": int(s.sum()),
                },
            )
    out["dt_bins"] = bins
    s = arr["dt"] <= INTERCEPT_DT_MAX
    if s.any():
        ci = bbr.bootstrap_ci(arr["D"][s])
        out["near_zero_intercept"] = {
            "dt_max": INTERCEPT_DT_MAX,
            "D_mean": round(float(arr["D"][s].mean()), 5),
            "ci95": [round(ci[0], 5), round(ci[1], 5)],
            "n": int(s.sum()),
        }
    return out


def headline_pred(npz: dict) -> str:
    keys = [
        k for k in npz if k.startswith("pred:") and not k.startswith("pred:state-copy")
    ]
    if len(keys) != 1:
        raise SystemExit(
            f"ABORT: expected exactly one model pred key, found {keys} — pass --pred",
        )
    return keys[0]


# ---------------------------------------------------------------- oracle


def _fixture(rows: list[dict], chunk_len: int = 50, dim: int = 2) -> dict:
    n = len(rows)
    out = {
        "truth": np.zeros((n, chunk_len, dim), np.float32),
        "valid": np.ones((n, chunk_len), bool),
        "repo_id": np.array([r.get("repo", "fix/repo") for r in rows]),
        "episode_index": np.array([r.get("ep", 0) for r in rows]),
        "frame_index": np.array([r["frame"] for r in rows]),
        "index": np.arange(n),
    }
    for i, r in enumerate(rows):
        out["truth"][i] = r.get("truth", 0.0)
        if "valid" in r:
            out["valid"][i] = r["valid"]
        for k, v in r.items():
            if k.startswith("pred:"):
                out.setdefault(k, np.zeros((n, chunk_len, dim), np.float32))
                out[k][i] = v
    return out


def _expect_abort(fn: Callable[[], object], needle: str, label: str) -> None:
    try:
        fn()
    except SystemExit as e:
        assert needle in str(e), f"{label}: wrong abort message: {e}"
        print(f"oracle abort OK: {label}")
        return
    raise AssertionError(f"{label}: expected abort not raised")


def oracle() -> None:
    step_n, pk = 50, "pred:fix"

    # A ramp truth shared by an episode: action at absolute time t is
    # (t, -t); a frame at f stores truth[f + i] = (f+i, -(f+i)).
    def ramp(f: int) -> np.ndarray:
        t = np.arange(f, f + step_n, dtype=np.float32)
        return np.stack([t, -t], axis=1)

    # (a) planted compatible pair -> D exactly 0; offset pair -> D == c.
    fx = _fixture(
        [
            {"frame": 0, "truth": ramp(0), pk: ramp(0)},
            {"frame": 10, "truth": ramp(10), pk: ramp(10)},
            {"frame": 100, "truth": ramp(100), pk: ramp(100) + 0.75},
            {"frame": 110, "truth": ramp(110), pk: ramp(110) - 0.75},
        ],
    )
    r = seam_read(fx, pk)
    assert r["n_pairs"] == 2
    assert r["dt_bins"] == [
        {"dt": [8, 14], "D_mean": 0.75, "ci95": r["dt_bins"][0]["ci95"], "n": 2},
    ]
    # pair 1 compatible (D=0), pair 2 disagrees by exactly 1.5 per element
    assert r["seam_disagreement"]["mean"] == round((0.0 + 1.5) / 2, 5), r
    print("oracle OK: compatible pair -> 0; planted offset recovered exactly")

    # (b) degenerate same-frame overlap (dt=0 mining) -> D exactly 0.
    fx0 = _fixture(
        [
            {"frame": 5, "truth": ramp(5), pk: ramp(5) + 0.3},
            {"frame": 5, "truth": ramp(5), pk: ramp(5) + 0.3},
        ],
    )
    fx0["frame_index"] = np.array([5, 5])
    r0 = seam_read(fx0, pk, min_dt=0)
    assert r0["seam_disagreement"]["mean"] == 0.0
    assert r0["boundary_jump"] == 0.0
    print("oracle OK: degenerate same-frame overlap reads exactly 0")

    # (c) NaN poison on INVALID steps changes nothing.
    v = np.ones(step_n, bool)
    v[40:] = False
    fx_clean = _fixture(
        [
            {"frame": 0, "truth": ramp(0), pk: ramp(0), "valid": v},
            {"frame": 10, "truth": ramp(10), pk: ramp(10) + 1.0, "valid": v},
        ],
    )
    fx_poison = {k: np.copy(a) for k, a in fx_clean.items()}
    fx_poison[pk][:, 40:, :] = np.nan
    fx_poison["truth"][:, 40:, :] = 123.0
    rc, rp = seam_read(fx_clean, pk), seam_read(fx_poison, pk)
    assert rc["seam_disagreement"] == rp["seam_disagreement"]
    assert rc["boundary_jump"] == rp["boundary_jump"]
    assert rc["within_chunk_step_pred"] == rp["within_chunk_step_pred"]
    print("oracle OK: NaN/value poison on invalid steps leaks nowhere")

    # (d) hand fixture, dt=48 -> 2-step overlap, hand-computed exactly.
    #     early pred: ramp(0)+0.5, late pred: ramp(48)-0.25
    #     overlap tail = [(48.5,-47.5),(49.5,-48.5)], head = [(47.75,-48.25),(48.75,-49.25)]
    #     D = mean(|0.75|) = 0.75; jump = same first step = 0.75
    #     err_tail = 0.5, err_head = 0.25; W (ramp step) = 1.0
    fxh = _fixture(
        [
            {"frame": 0, "truth": ramp(0), pk: ramp(0) + 0.5},
            {"frame": 48, "truth": ramp(48), pk: ramp(48) - 0.25},
        ],
    )
    rh = seam_read(fxh, pk)
    assert rh["seam_disagreement"]["mean"] == 0.75
    assert rh["boundary_jump"] == 0.75
    assert rh["err_tail_overlap"] == 0.5 and rh["err_head_overlap"] == 0.25
    assert rh["within_chunk_step_truth"] == 1.0
    assert rh["within_chunk_step_pred"] == 1.0
    assert rh["jump_over_truth_step"] == 0.75
    assert rh["D_over_err"] == round(0.75 / 0.375, 4)
    print("oracle OK: dt=48 hand fixture reproduces D/J/W/err exactly")

    # (e) dt-binning recovers planted per-bin offsets.
    fxb = _fixture(
        [
            {"frame": 0, "truth": ramp(0), pk: ramp(0)},
            {"frame": 3, "truth": ramp(3), pk: ramp(3) + 0.25, "ep": 0},
            {"frame": 200, "truth": ramp(200), pk: ramp(200), "ep": 1},
            {"frame": 230, "truth": ramp(230), pk: ramp(230) + 0.625, "ep": 1},
        ],
    )
    fxb["episode_index"] = np.array([0, 0, 1, 1])
    rb = seam_read(fxb, pk)
    got = {tuple(b["dt"]): b["D_mean"] for b in rb["dt_bins"]}
    assert got[(1, 7)] == 0.25 and got[(29, 35)] == 0.625, got
    assert rb["near_zero_intercept"]["D_mean"] == 0.25
    print("oracle OK: dt bins + near-zero intercept recover planted offsets")

    # (f) abort battery.
    bad = {k: np.copy(a) for k, a in fxh.items()}
    bad["truth"][1] += 1.0
    _expect_abort(
        lambda: seam_read(bad, pk),
        "truth overlap mismatch",
        "misaligned truth",
    )
    noid = {k: a for k, a in fxh.items() if k != "episode_index"}
    _expect_abort(
        lambda: seam_read(noid, pk),
        "episode_index",
        "missing identity column",
    )
    nan = {k: np.copy(a) for k, a in fxh.items()}
    nan[pk][0, 0, 0] = np.nan
    _expect_abort(lambda: seam_read(nan, pk), "NaN/inf", "NaN in valid region")
    _expect_abort(
        lambda: seam_read(fxh, "pred:absent"),
        "not in stack",
        "missing pred key",
    )
    far = _fixture(
        [
            {"frame": 0, "truth": ramp(0), pk: ramp(0)},
            {"frame": 60, "truth": ramp(60), pk: ramp(60)},
        ],
    )
    _expect_abort(lambda: seam_read(far, pk), "no qualifying", "no qualifying pairs")
    twopred = {k: np.copy(a) for k, a in fxh.items()}
    twopred["pred:other"] = twopred[pk]
    _expect_abort(
        lambda: headline_pred(twopred),
        "exactly one",
        "ambiguous headline pred",
    )
    print("ORACLE PASS: boundary-incompatibility read verified pre-data")


# ------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument(
        "--npz",
        action="append",
        default=None,
        help="label=path (repeatable); default = the 5 banked stacks",
    )
    ap.add_argument(
        "--pred",
        default=None,
        help="explicit pred key (default: the unique non-state-copy key)",
    )
    ap.add_argument(
        "--out",
        default=str(_REPORTS / "analysis__boundary_incompat_panels.json"),
    )
    args = ap.parse_args()
    if args.oracle:
        oracle()
        return

    stacks = (
        {s.split("=", 1)[0]: s.split("=", 1)[1] for s in args.npz}
        if args.npz
        else {k: str(_REPORTS / v) for k, v in DEFAULT_STACKS.items()}
    )
    results: dict = {"read": "boundary_incompat", "stacks": {}}
    for label, path in stacks.items():
        npz = dict(np.load(path, allow_pickle=True))
        pred_key = args.pred or headline_pred(npz)
        entry = {
            "path": str(path),
            "model": seam_read(npz, pred_key),
            "state_copy_reference": seam_read(npz, "pred:state-copy"),
        }
        results["stacks"][label] = entry
        m, s = entry["model"], entry["state_copy_reference"]
        print(f"\n=== {label} ({pred_key}) — {m['n_pairs']} pairs ===")
        print(
            f"  seam D           {m['seam_disagreement']['mean']:.4f} "
            f"CI95 {m['seam_disagreement']['ci95']}",
        )
        print(
            f"  overlap err      tail {m['err_tail_overlap']:.4f} / "
            f"head {m['err_head_overlap']:.4f}  ->  D/err {m['D_over_err']:.3f}",
        )
        print(
            f"  boundary jump    {m['boundary_jump']:.4f}  "
            f"(within-chunk step: pred {m['within_chunk_step_pred']:.4f}, "
            f"truth {m['within_chunk_step_truth']:.4f}; "
            f"J/W_truth {m['jump_over_truth_step']:.3f})",
        )
        iz = m.get("near_zero_intercept")
        if iz:
            print(
                f"  D at dt<={iz['dt_max']}      {iz['D_mean']:.4f} "
                f"CI95 {iz['ci95']} (n={iz['n']})",
            )
        print(
            "  dt curve         "
            + "  ".join(
                f"{b['dt'][0]}-{b['dt'][1]}:{b['D_mean']:.3f}" for b in m["dt_bins"]
            ),
        )
        print(
            f"  state-copy D     {s['seam_disagreement']['mean']:.4f} "
            f"(scene motion over dt)",
        )
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
