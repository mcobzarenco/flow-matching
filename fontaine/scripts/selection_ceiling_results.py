"""Selection-ceiling read — oracle best-of-K bound over a --dump-draws stack.

EXPLORATORY, NOT PRE-REGISTERED (ideas #19, lit rung banked 2026-08-07):
bounds what ANY selector over sampled draws (MG-Select / VLA-ATTC /
CoVer / TapSampling flavors) could buy on our panel BEFORE building
one. Record-only — no decision rule; escalation to an actual selector
needs its own pre-registration.

Reads (all record-only, banked pooling conventions):
  * ceiling ladder — E[best-of-K] chunk MAE for K = 1..N via the exact
    order-statistic expectation over random K-subsets of the per-frame
    draw MAEs (no Monte Carlo): with the frame's draw MAEs sorted
    ascending m_(1..N), P(rank i is the min of a K-subset) =
    C(N-i, K-1)/C(N, K). K=1 is the average single draw, K=N the hard
    per-frame oracle. Frames pooled valid-element-weighted — identical
    to the banked pooled chunk_mae (composite best-of-N prediction
    pooled through box_batch_results reproduces ladder K=N exactly;
    asserted on every run).
  * headroom — greedy minus best-of-N and ensemble(mean-of-draws)
    minus best-of-N, with a paired per-frame read (bootstrap CI95,
    LORO — arch_batch_results conventions) on the oracle gain.
  * first_mae mirrors — same ladder on first-step MAE, best draw
    selected independently (the draws_fairness read-2 convention).
  * selector diagnostics — argmin-draw histogram (exchangeability
    sanity), per-frame gain quantiles, dispersion-conditioned gain
    quartiles + Spearman (where would a selector buy most?).

Execution guards (hard aborts): sample_draws metadata matches the
stack; the draws policy name extends the greedy policy exactly
(checkpoint pairing at npz level); identity byte-match or the
pre-registered q4 subset join (draws10_t1_results.join_rows verbatim);
greedy npz-recomputed summaries reproduce its report (|d| < 5e-3);
mean over the draw axis reproduces the paired pooled npz prediction
column (float32 rounding; wrong rows/plan show up at O(1)).

Oracle mode (--oracle, pre-data on the banked AR-100k greedy npz):
  (a) ladder formula vs brute-force enumeration over ALL K-subsets on
      a small fixture; monotone nonincreasing in K;
  (b) degenerate draws=1 (greedy as a 1-draw stack) reproduces the
      5.8026/2.1431 anchor through the ceiling code path;
  (c) planted best-draw pattern (draw f mod 10 scaled x0.5 per frame)
      recovered exactly — argmin pattern + histogram + K=N and K=1
      pooled magnitudes;
  (d) abort battery: sample_draws mismatch, non-extending / non-draws
      policy, draws=1 in real mode, misaligned equal-length index,
      pooled-npz mean drift;
  (e) subset-shaped slice joins and reproduces the direct computation.

Defaults = the molmo2 endpoint draws launcher's exact output stems
(eval_box_molmo2_endpoint_draws10_t1.sh; the q4 fallback and other
arms pass explicit paths). Pure CPU, read-only on inputs.
"""

import argparse
import importlib.util
import itertools
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np

_HERE = Path(__file__).resolve().parent


def _sibling(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bbr = _sibling("box_batch_results")
abr = _sibling("arch_batch_results")
dfair = _sibling("draws_fairness")
d10 = _sibling("draws10_t1_results")

RUN_STEM = "reports/eval__fontaine_molmo2_ar_40k_ddp4__step_040000"
GREEDY_STEM = f"{RUN_STEM}__panel_curated_v0_k4l2"
DRAWS_STEM = f"{RUN_STEM}__panel_k4l2_draws10_t1"
OUT_DEFAULT = "reports/analysis__selection_ceiling_molmo2_40k_k4l2.json"

AR100K_STEM = "reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2"
AR_ANCHOR = bbr.ANCHORS["ar"]  # (5.8026, 2.1431) — oracle mode only
POOLED_TOL = 1e-2  # raw-degree float32 rounding headroom; drift is O(1)
NOTE = (
    "EXPLORATORY, NOT PRE-REGISTERED — record-only ceiling bound; "
    "building any actual selector needs its own pre-registration"
)


def _meta(d: dict, key: str, label: str):  # noqa: ANN202 — npz scalars
    if key not in d.files:
        sys.exit(f"{label}: npz lacks the {key!r} metadata key — not a draws dump?")
    return np.asarray(d[key]).item()


def ladder_weights(n: int, k: int) -> np.ndarray:
    """P(sorted-rank i is the min of a uniformly random K-subset)."""
    total = math.comb(n, k)
    return np.array(
        [math.comb(n - 1 - i, k - 1) / total for i in range(n)],
        dtype=np.float64,
    )


def orderstat_ladder(per_draw_frame: np.ndarray) -> dict[int, np.ndarray]:
    """K -> per-frame E[min over a random K-subset] for a [N, F] array."""
    n = per_draw_frame.shape[0]
    s = np.sort(per_draw_frame, axis=0)
    return {k: ladder_weights(n, k) @ s for k in range(1, n + 1)}


def pool_frames(vals: np.ndarray, nvalid: np.ndarray, keep: np.ndarray) -> float:
    """Valid-element-weighted frame pooling == the banked pooled_chunk."""
    return float((vals * nvalid)[keep].sum() / nvalid[keep].sum())


def ceiling_core(
    draws: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    core: np.ndarray,
) -> dict:
    """The ceiling reads on a [F, N, chunk, dim] stack (float64)."""
    n_frames, n_draws = draws.shape[:2]
    _t, _v, _c, w = bbr.masks({"truth": truth, "valid": valid, "core": core})
    nvalid = w.sum(axis=(1, 2))
    keep_chunk = (nvalid > 0) & core
    keep_first = valid[:, 0] & core

    per_draw_chunk = np.stack(
        [bbr.frame_mae(np.abs(draws[:, j] - truth), w)[0] for j in range(n_draws)],
    )  # [N, F]
    per_draw_first = np.stack(
        [
            np.abs(draws[:, j, 0, :] - truth[:, 0, :]).mean(axis=1)
            for j in range(n_draws)
        ],
    )

    ladder_chunk = orderstat_ladder(per_draw_chunk)
    ladder_first = orderstat_ladder(per_draw_first)

    # Composite best-of-N prediction pooled through the BANKED path must
    # reproduce ladder K=N — ties the expectation ladder to pooled_chunk.
    best_idx = per_draw_chunk.argmin(axis=0)
    composite = draws[np.arange(n_frames), best_idx]
    pooled_best = bbr.pooled_chunk(np.abs(composite - truth), core, w)
    ladder_best = pool_frames(ladder_chunk[n_draws], nvalid, keep_chunk)
    assert abs(pooled_best - ladder_best) < 1e-8, (
        f"ladder K={n_draws} {ladder_best} != composite pooled {pooled_best}"
    )

    mean_pred = draws.mean(axis=1)
    std = draws.std(axis=1)
    dispersion = (std * w).sum(axis=(1, 2)) / np.maximum(nvalid, 1)

    return {
        "n_frames": int(n_frames),
        "n_draws": int(n_draws),
        "keep_chunk": keep_chunk,
        "keep_first": keep_first,
        "nvalid": nvalid,
        "w": w,
        "per_draw_chunk": per_draw_chunk,
        "best_idx": best_idx,
        "dispersion": dispersion,
        "ladder_chunk_pooled": {
            k: round(pool_frames(v, nvalid, keep_chunk), 4)
            for k, v in ladder_chunk.items()
        },
        "ladder_first_pooled": {
            k: round(float(v[keep_first].mean()), 4) for k, v in ladder_first.items()
        },
        "best_frame_chunk": ladder_chunk[n_draws],  # per-frame oracle min
        "best_first_pooled": round(
            float(per_draw_first.min(axis=0)[keep_first].mean()),
            4,
        ),
        "ensemble": {
            "chunk_mae": round(bbr.pooled_chunk(np.abs(mean_pred - truth), core, w), 4),
            "first_mae": round(
                bbr.pooled_first(np.abs(mean_pred - truth), valid, core),
                4,
            ),
        },
        "mean_pred": mean_pred,
    }


def analyze(
    g_npz: dict,
    g_key: str,
    g_rep: dict,
    d_npz: dict,
    out_path: str | None,
    pooled_pred: np.ndarray | None = None,
) -> dict:
    # ---- execution guards (each failure is a hard abort) ----
    policy = str(_meta(d_npz, "policy", "draws"))
    sample_draws = int(_meta(d_npz, "sample_draws", "draws"))
    if d_npz["draws"].shape[1] != sample_draws:
        sys.exit(
            f"draws stack has {d_npz['draws'].shape[1]} draws but sample_draws "
            f"metadata says {sample_draws} — dump is inconsistent, stop",
        )
    if sample_draws < 2:
        sys.exit("draws dump needs >= 2 draws for a selection ceiling — stop")
    policy_g = g_key.removeprefix("pred:")
    if not policy.startswith(f"{policy_g}_draws"):
        sys.exit(
            f"draws policy {policy!r} does not extend greedy policy "
            f"{policy_g!r} — arms are not the same checkpoint's reads",
        )
    rows, subset = d10.join_rows(g_npz, d_npz)

    truth = d_npz["truth"].astype(np.float64)
    valid, core = d_npz["valid"], d_npz["core"]
    draws = d_npz["draws"].astype(np.float64)
    res = ceiling_core(draws, truth, valid, core)
    n_draws = res["n_draws"]

    if pooled_pred is not None:
        drift = float(
            np.abs(res["mean_pred"] - pooled_pred.astype(np.float64)).max(),
        )
        if drift >= POOLED_TOL:
            sys.exit(
                f"mean over the draw axis drifts {drift:.4g} from the pooled "
                f"npz prediction (tol {POOLED_TOL}) — not the same eval, stop",
            )

    # ---- greedy arm on the paired rows ----
    w = res["w"]
    keep_chunk = res["keep_chunk"]
    err_g = np.abs(g_npz[g_key][rows].astype(np.float64) - truth)
    fr_g, _ = bbr.frame_mae(err_g, w)
    greedy = {
        "chunk_mae": round(bbr.pooled_chunk(err_g, core, w), 4),
        "first_mae": round(bbr.pooled_first(err_g, valid, core), 4),
        "pred_key": g_key,
    }

    # ---- headroom: what a perfect selector buys ----
    gain = res["best_frame_chunk"] - fr_g  # negative = ceiling beats greedy
    paired_gain = abr.paired_read(
        res["best_frame_chunk"],
        fr_g,
        keep_chunk,
        d_npz["repo_id"],
    )
    paired_gain["definition"] = (
        "per-frame best-of-N chunk_mae minus greedy (negative = oracle "
        "selection beats the deployment decode)"
    )
    best_chunk = res["ladder_chunk_pooled"][n_draws]
    headroom = {
        "greedy_minus_bestN_chunk": round(greedy["chunk_mae"] - best_chunk, 4),
        "ensemble_minus_bestN_chunk": round(
            res["ensemble"]["chunk_mae"] - best_chunk,
            4,
        ),
        "greedy_minus_bestN_first": round(
            greedy["first_mae"] - res["best_first_pooled"],
            4,
        ),
        "frames_bestN_beats_greedy": round(float((gain[keep_chunk] < 0).mean()), 4),
        "per_frame_gain_quantiles": {
            f"p{int(q * 100)}": round(float(np.quantile(gain[keep_chunk], q)), 4)
            for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        },
        "paired_gain": paired_gain,
    }

    # ---- selector diagnostics ----
    hist = np.bincount(res["best_idx"][keep_chunk], minlength=n_draws)
    dispersion = res["dispersion"]
    qs = np.quantile(dispersion[keep_chunk], [0.25, 0.5, 0.75])
    bins = np.digitize(dispersion, qs)
    quartiles = {}
    for b, label in enumerate(["q1_tight", "q2", "q3", "q4_dispersed"]):
        s = (bins == b) & keep_chunk
        quartiles[label] = {
            "n": int(s.sum()),
            "dispersion": round(float(dispersion[s].mean()), 4),
            "oracle_gain": round(float(gain[s].mean()), 4),
        }
    diagnostics = {
        "argmin_draw_histogram": hist.tolist(),
        "dispersion_gain_quartiles": quartiles,
        "spearman_dispersion_vs_gain": round(
            dfair.spearman(dispersion[keep_chunk], gain[keep_chunk]),
            4,
        ),
    }

    out = {
        "note": NOTE,
        "provenance": {
            "draws_policy": policy,
            "sample_draws": sample_draws,
            "seed": int(_meta(d_npz, "seed", "draws")),
            "noise_key": str(_meta(d_npz, "noise_key", "draws")),
            "checkpoint": "/".join(d10._ckpt_id(g_rep)),
            "row_pairing": "subset-join on index (q4 fallback)"
            if subset
            else "full byte-match",
            "n_rows_paired": len(rows),
            "pooled_npz_check": "skipped" if pooled_pred is None else "reproduced",
        },
        "pooled": {"greedy": greedy, "ensemble_mean_of_draws": res["ensemble"]},
        "ceiling_ladder_chunk": {
            f"K={k}": v for k, v in res["ladder_chunk_pooled"].items()
        },
        "ceiling_ladder_first": {
            f"K={k}": v for k, v in res["ladder_first_pooled"].items()
        },
        "headroom": headroom,
        "selector_diagnostics": diagnostics,
    }
    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=1))
        print(f"wrote {out_path}")
    print(json.dumps(out, indent=1))
    print(
        f"greedy {greedy['chunk_mae']} | ensemble {res['ensemble']['chunk_mae']} | "
        f"best-of-{n_draws} ceiling {best_chunk} | headroom vs greedy "
        f"{headroom['greedy_minus_bestN_chunk']} ({NOTE.split(' — ')[0]})",
    )
    return out


# ---- oracle -----------------------------------------------------------------


def _stack_fixture(
    g_npz: dict,
    g_key: str,
    rows: np.ndarray,
    factors: np.ndarray,
    tag: str = "_draws%d_t1",
) -> dict:
    """A draws npz slice fixture: draw j's error = factors[f, j] x greedy's."""
    truth = g_npz["truth"][rows].astype(np.float64)
    pred = g_npz[g_key][rows].astype(np.float64)
    resid = np.where(np.isfinite(truth), pred - truth, 0.0)
    base = np.where(np.isfinite(truth), truth, pred)
    draws = base[:, None] + resid[:, None] * factors[:, :, None, None]
    n = factors.shape[1]
    out = abr._DictNpz({k: g_npz[k][rows] for k in bbr.PAIR_KEYS})
    out["draws"] = draws
    out["policy"] = np.array(f"{g_key.removeprefix('pred:')}{tag % n}")
    out["sample_draws"] = np.array(n)
    out["seed"] = np.array(0)
    out["noise_key"] = np.array("stable")
    return out


def oracle() -> None:
    # (a) ladder formula vs brute-force enumeration + monotonicity
    rng = np.random.default_rng(0)
    per_draw = rng.uniform(1.0, 5.0, size=(5, 7))
    ladder = orderstat_ladder(per_draw)
    for k in range(1, 6):
        brute = np.mean(
            [
                per_draw[list(sub)].min(axis=0)
                for sub in itertools.combinations(range(5), k)
            ],
            axis=0,
        )
        assert np.allclose(ladder[k], brute, atol=1e-12), f"ladder K={k} != brute force"
    for k in range(2, 6):
        assert (ladder[k] <= ladder[k - 1] + 1e-12).all(), "ladder not monotone"
    print("oracle (a) OK: order-stat ladder == exhaustive subsets, monotone in K")

    g_npz = np.load(f"{AR100K_STEM}.npz", allow_pickle=True)
    g_rep = json.loads(Path(f"{AR100K_STEM}.json").read_text())
    g_key = d10.bare_key(g_npz, "greedy")
    d10.report_crosscheck(g_npz, g_key, g_rep, "oracle-greedy")
    truth, valid, core, _w = bbr.masks(g_npz)

    # (b) degenerate draws=1 reproduces the greedy anchor through the
    # ceiling code path
    res = ceiling_core(
        g_npz[g_key].astype(np.float64)[:, None],
        truth.astype(np.float64),
        valid,
        core,
    )
    got = res["ladder_chunk_pooled"][1]
    assert abs(got - AR_ANCHOR[0]) < 5e-3, f"degenerate chunk {got} != {AR_ANCHOR[0]}"
    got_f = res["ladder_first_pooled"][1]
    assert abs(got_f - AR_ANCHOR[1]) < 5e-3, f"degenerate first {got_f}"
    print(f"oracle (b) OK: draws=1 ceiling reproduces the {AR_ANCHOR} anchor")

    # (c) planted best-draw pattern recovered exactly
    rows = np.flatnonzero(core)[:2400]
    n = 10
    factors = 1.0 + 0.02 * np.arange(n)[None, :].repeat(len(rows), axis=0)
    planted = np.arange(len(rows)) % n
    factors[np.arange(len(rows)), planted] = 0.5
    fx = _stack_fixture(g_npz, g_key, rows, factors)
    t_s, v_s, c_s, w_s = bbr.masks(fx)
    fr_g, _ = bbr.frame_mae(np.abs(g_npz[g_key][rows] - t_s), w_s)
    nv_s = w_s.sum(axis=(1, 2))
    keep = (nv_s > 0) & c_s
    assert fr_g[keep].min() > 0, "fixture needs strictly positive greedy error"
    res = ceiling_core(fx["draws"], t_s.astype(np.float64), v_s, c_s)
    assert (res["best_idx"][keep] == planted[keep]).all(), "planted pattern lost"
    assert res["ladder_chunk_pooled"][n] == round(
        pool_frames(0.5 * fr_g, nv_s, keep),
        4,
    ), "K=N magnitude off"
    assert res["ladder_chunk_pooled"][1] == round(
        pool_frames(factors.mean(axis=1) * fr_g, nv_s, keep),
        4,
    ), "K=1 magnitude off"
    want_hist = np.bincount(planted[keep], minlength=n).tolist()
    got_hist = np.bincount(res["best_idx"][keep], minlength=n).tolist()
    assert got_hist == want_hist, "argmin histogram off"
    print("oracle (c) OK: planted best-draw pattern in == out, magnitudes checked")

    # (d) abort battery
    def _expect(fn: Callable[[], object], needle: str, label: str) -> None:
        try:
            fn()
            raise AssertionError(f"{label}: expected abort not raised")
        except SystemExit as e:
            assert needle in str(e), f"{label}: wrong abort message: {e}"
        print(f"oracle abort OK: {label}")

    sub_rows = np.arange(0, len(g_npz["index"]), 3)
    fx3 = _stack_fixture(
        g_npz,
        g_key,
        sub_rows,
        np.tile(np.array([0.9, 1.0, 1.1]), (len(sub_rows), 1)),
    )
    bad = abr._DictNpz(dict(fx3))
    bad["sample_draws"] = np.array(5)
    _expect(
        lambda: analyze(g_npz, g_key, g_rep, bad, None),
        "inconsistent",
        "sample_draws metadata mismatch",
    )
    bad = abr._DictNpz(dict(fx3))
    bad["policy"] = np.array("bijou@999_draws3_t1")
    _expect(
        lambda: analyze(g_npz, g_key, g_rep, bad, None),
        "does not extend",
        "policy not extending greedy",
    )
    one = _stack_fixture(g_npz, g_key, sub_rows, np.ones((len(sub_rows), 1)))
    _expect(
        lambda: analyze(g_npz, g_key, g_rep, one, None),
        ">= 2 draws",
        "draws=1 in real mode",
    )
    full = _stack_fixture(
        g_npz,
        g_key,
        np.arange(len(g_npz["index"])),
        np.tile(np.array([0.9, 1.1]), (len(g_npz["index"]), 1)),
    )
    mis = abr._DictNpz(dict(full))
    idx = np.array(mis["index"])
    idx[:2] = idx[:2][::-1]
    mis["index"] = idx
    _expect(
        lambda: analyze(g_npz, g_key, g_rep, mis, None),
        "pairing broken",
        "misaligned equal-length index",
    )
    drifted = fx3["draws"].mean(axis=1) + 0.5
    _expect(
        lambda: analyze(g_npz, g_key, g_rep, fx3, None, pooled_pred=drifted),
        "not the same eval",
        "pooled-npz mean drift",
    )

    # (e) subset-shaped slice joins and reproduces the direct computation
    res_join = analyze(g_npz, g_key, g_rep, fx3, None)
    assert res_join["provenance"]["row_pairing"].startswith("subset-join")
    direct = ceiling_core(
        fx3["draws"],
        fx3["truth"].astype(np.float64),
        fx3["valid"],
        fx3["core"],
    )
    assert (
        res_join["ceiling_ladder_chunk"]["K=3"] == direct["ladder_chunk_pooled"][3]
    ), "subset join drifts from the direct read"
    assert (
        res_join["ceiling_ladder_chunk"]["K=1"] == direct["ladder_chunk_pooled"][1]
    ), "subset join drifts from the direct read (K=1)"
    print("oracle (e) OK: subset slice joins + reproduces the direct read")
    print("ORACLE PASS: selection-ceiling reads verified pre-data")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--draws-npz", default=f"{DRAWS_STEM}_draws.npz")
    p.add_argument(
        "--pooled-npz",
        default=f"{DRAWS_STEM}.npz",
        help="the paired --dump-predictions npz ('none' skips the check)",
    )
    p.add_argument(
        "--greedy",
        nargs=2,
        metavar=("NPZ", "JSON"),
        default=[f"{GREEDY_STEM}.npz", f"{GREEDY_STEM}.json"],
    )
    p.add_argument("--out", default=OUT_DEFAULT)
    p.add_argument("--oracle", action="store_true")
    a = p.parse_args()
    if a.oracle:
        oracle()
        return
    needed = [a.draws_npz, *a.greedy]
    if a.pooled_npz != "none":
        needed.append(a.pooled_npz)
    for path in needed:
        if not Path(path).exists():
            sys.exit(f"missing input {path} — eval not finished / not rsynced?")
    d_npz = np.load(a.draws_npz, allow_pickle=True)
    g_npz = np.load(a.greedy[0], allow_pickle=True)
    g_rep = json.loads(Path(a.greedy[1]).read_text())
    g_key = d10.bare_key(g_npz, "greedy")
    d10.report_crosscheck(g_npz, g_key, g_rep, "greedy")
    pooled_pred = None
    if a.pooled_npz != "none":
        p_npz = np.load(a.pooled_npz, allow_pickle=True)
        if not np.array_equal(p_npz["index"], d_npz["index"]):
            sys.exit("pooled npz rows differ from the draws npz — not the same eval")
        p_key = d10.bare_key(p_npz, "pooled")
        want = f"pred:{_meta(d_npz, 'policy', 'draws')}"
        if p_key != want:
            sys.exit(f"pooled npz pred key {p_key!r} != draws policy {want!r}")
        pooled_pred = p_npz[p_key]
    analyze(g_npz, g_key, g_rep, d_npz, a.out, pooled_pred=pooled_pred)


if __name__ == "__main__":
    main()
