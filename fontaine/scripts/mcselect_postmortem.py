"""#6 rung-(c) post-mortem — record-only 2-axis map of the closed family.

NOT pre-registered; no decision rides on any number here (queue item
``idea6-mcselect-postmortem``, the selection-ceiling read precedent).
The zero-training scorer family is CLOSED (SC +0.210, MC +0.313, both
CI95 entirely > 0); this read maps HOW it failed on the banked
rung-(c) dump before anyone prices a learned verifier:

  1. per-candidate KL-vs-frame-error correlation — pooled (raw +
     row-centered Pearson) and per-row Spearman;
  2. the oracle-best candidate's rank histogram on the KL
     (informativeness) axis — where do the good candidates sit;
  3. the same two reads on SC's self-certainty axis (recomputed from
     the banked candidates file via the bijou scorer, the exact
     function the rung-(a) bon picks were verified against);
  4. cross-axis per-row Spearman(KL, SC) — is the family one axis
     twice, or two failures;
  5. records: where each scorer's PICK sits on the error axis.

Argmax/tie/eligibility conventions are NOT re-implemented — this file
reuses ``mcselect_results`` (eligible list, contract keys),
``box_batch_results`` (masks, frame MAE, bootstrap), and
``bijou.eval.subgoal_scoring.self_certainty``. It extends only the
correlation delta. Output: one analysis json, no deployment claim.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bijou.eval.subgoal_scoring import self_certainty

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
mcres = __import__("mcselect_results")
bbr = mcres.bbr
sdr = mcres.sdr

OUT_DEFAULT = "reports/analysis__subgoal_mcselect_postmortem_q4_ar100k_k4l2.json"
RAW_DEFAULT = "reports/analysis__subgoal_mcselect_postmortem_q4_ar100k_k4l2_raw.npz"


# ------------------------------------------------------------- rank math


def avg_rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based, ties share the mean rank) — scipy-free."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """Spearman rho via Pearson on average ranks; None if either side
    is constant (correlation undefined)."""
    if np.unique(a).size < 2 or np.unique(b).size < 2:
        return None
    ra, rb = avg_rank(a), avg_rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / math.sqrt((ra**2).sum() * (rb**2).sum()))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / math.sqrt((a**2).sum() * (b**2).sum()))


def desc_rank_of(scores: np.ndarray, pos: int) -> int:
    """Competition rank of ``pos`` on the DESCENDING axis: the number
    of entries strictly greater (0 = the axis's top pick region)."""
    return int((scores > scores[pos]).sum())


# --------------------------------------------------------------- analyze


def _rho_block(rhos: list[float]) -> dict:
    arr = np.array(rhos, dtype=np.float64)
    lo, hi = bbr.bootstrap_ci(arr)
    return {
        "mean": round(float(arr.mean()), 5),
        "ci95": [round(lo, 5), round(hi, 5)],
        "median": round(float(np.median(arr)), 5),
        "frac_positive": round(float((arr > 0).mean()), 5),
        "n_rows": len(rhos),
    }


def _rank_block(ranks: list[int], n_eligs: list[int]) -> dict:
    """Descending-axis rank stats for a distinguished candidate, with
    the uniform-pick null alongside (mean normalized rank 0.5, top-1
    rate mean(1/n_elig))."""
    arr = np.array(ranks, dtype=np.int64)
    ne = np.array(n_eligs, dtype=np.int64)
    norm = arr / np.maximum(ne - 1, 1)
    lo, hi = bbr.bootstrap_ci(norm.astype(np.float64))
    width = int(ne.max())
    return {
        "histogram_rank0_is_axis_top": [int((arr == r).sum()) for r in range(width)],
        "mean_normalized_rank": round(float(norm.mean()), 5),
        "ci95": [round(lo, 5), round(hi, 5)],
        "uniform_null": 0.5,
        "top1_rate": round(float((arr == 0).mean()), 5),
        "top1_uniform_null": round(float((1.0 / ne).mean()), 5),
        "bottom1_rate": round(float((arr == ne - 1).mean()), 5),
    }


def analyze(
    scores_npz: dict,
    candidates: dict,
    out_path: str | None,
    raw_out: str | None = None,
) -> dict:
    for key in (mcres.KL_KEY, mcres.CAND_PRED_KEY):
        if key not in scores_npz:
            sys.exit(f"scores npz missing {key} — not a mcselect dump, stop")

    # ---- row alignment vs the candidates file (the mcres convention) --
    by_index = {int(r["index"]): r for r in candidates["rows"]}
    idx = scores_npz["index"]
    missing = [int(ix) for ix in idx if int(ix) not in by_index]
    if missing:
        sys.exit(
            f"{len(missing)} scores rows absent from the candidates file "
            f"(first: {missing[:3]}) — not the banked width, stop",
        )
    if len(idx) != len(by_index):
        sys.exit(
            f"scores npz has {len(idx)} rows, candidates file "
            f"{len(by_index)} — partial dump, stop",
        )
    rows_in_order = [by_index[int(ix)] for ix in idx]

    kl = scores_npz[mcres.KL_KEY]
    cand_pred = scores_npz[mcres.CAND_PRED_KEY]
    truth, _valid, core, w = bbr.masks(scores_npz)
    n, n_cand = kl.shape

    # ---- eligibility + contract guards, and the SC axis recompute ----
    eligibles: list[list[int]] = []
    sc = np.full((n, n_cand), np.nan)
    for i, row in enumerate(rows_in_order):
        cands = row["candidates"]
        elig = mcres.eligible_list(cands)
        if len(elig) < 2:
            sys.exit(
                f"row index {int(idx[i])} has {len(elig)} eligible "
                "candidate(s) — a rank read over < 2 candidates is "
                "degenerate, this dump does not carry the rung-(b') "
                "width, stop",
            )
        if not np.isfinite(kl[i, elig]).all():
            sys.exit(
                f"non-finite KL at an ELIGIBLE candidate (row index "
                f"{int(idx[i])}) — broken dump, stop",
            )
        inelig = [j for j in range(len(cands)) if j not in elig]
        if inelig and np.isfinite(kl[i, inelig]).any():
            sys.exit(
                f"finite KL at an INELIGIBLE candidate (row index "
                f"{int(idx[i])}) — scored outside the clean filter, stop",
            )
        vocabs = {c["allowed_vocab"] for c in cands}
        if len(vocabs) != 1:
            sys.exit(
                f"row index {int(idx[i])}: mixed allowed_vocab {sorted(vocabs)} — stop",
            )
        vocab = cands[0]["allowed_vocab"]
        for j in elig:
            sc[i, j] = self_certainty(cands[j]["mean_logprob"], vocab)
        eligibles.append(elig)

    # ---- per-candidate frame error (the panel frame-MAE convention) --
    err = np.full((n, n_cand), np.nan)
    nvalid = None
    for c in range(n_cand):
        e, nvalid = bbr.frame_mae(np.abs(cand_pred[:, c] - truth), w)
        err[:, c] = e
    keep = (nvalid > 0) & core
    dropped = int((~keep).sum())

    # ---- reads ----
    rho_kl_err: list[float] = []
    rho_sc_err: list[float] = []
    rho_kl_sc: list[float] = []
    kl_const_rows = 0
    best_rank_kl: list[int] = []
    best_rank_sc: list[int] = []
    mc_pick_err_rank: list[int] = []
    sc_pick_err_rank: list[int] = []
    n_eligs: list[int] = []
    kl_centered: list[np.ndarray] = []
    err_centered: list[np.ndarray] = []
    kl_pooled: list[np.ndarray] = []
    err_pooled: list[np.ndarray] = []
    for i in range(n):
        if not keep[i]:
            continue
        elig = eligibles[i]
        k, s, e = kl[i, elig], sc[i, elig], err[i, elig]
        r_ke = spearman(k, e)
        if r_ke is None:
            kl_const_rows += 1
        else:
            rho_kl_err.append(r_ke)
        r_se = spearman(s, e)
        if r_se is not None:
            rho_sc_err.append(r_se)
        r_ks = spearman(k, s)
        if r_ks is not None:
            rho_kl_sc.append(r_ks)
        # oracle-best on the error axis; ties -> lowest index (the
        # family's tie convention, applied to the min)
        best = int(np.flatnonzero(e == e.min())[0])
        best_rank_kl.append(desc_rank_of(k, best))
        best_rank_sc.append(desc_rank_of(s, best))
        # where each scorer's pick sits on the error axis (ascending:
        # 0 = the pick IS the best candidate)
        mc_pick = int(np.flatnonzero(k == k.max())[0])
        sc_pick = int(np.flatnonzero(s == s.max())[0])
        mc_pick_err_rank.append(int((e < e[mc_pick]).sum()))
        sc_pick_err_rank.append(int((e < e[sc_pick]).sum()))
        n_eligs.append(len(elig))
        kl_pooled.append(k)
        err_pooled.append(e)
        kl_centered.append(k - k.mean())
        err_centered.append(e - e.mean())

    kp, ep = np.concatenate(kl_pooled), np.concatenate(err_pooled)
    kc, ec = np.concatenate(kl_centered), np.concatenate(err_centered)
    ne = np.array(n_eligs, dtype=np.float64)
    mc_norm = np.array(mc_pick_err_rank) / np.maximum(ne - 1, 1)
    sc_norm = np.array(sc_pick_err_rank) / np.maximum(ne - 1, 1)
    out = {
        "note": (
            "record-only post-mortem, NOT pre-registered, no decision "
            "rides here; family already CLOSED by the pre-registered reads"
        ),
        "n_rows": int(keep.sum()),
        "n_rows_dropped_no_valid_frames": dropped,
        "eligible_count_histogram": {
            str(c): int((ne == c).sum()) for c in range(2, int(ne.max()) + 1)
        },
        "kl_vs_error": {
            "per_row_spearman": _rho_block(rho_kl_err),
            "kl_constant_rows_excluded": kl_const_rows,
            "pooled_pearson_row_centered": round(pearson(kc, ec), 5),
            "pooled_pearson_raw_confounded_by_row_difficulty": round(
                pearson(kp, ep),
                5,
            ),
        },
        "sc_vs_error": {"per_row_spearman": _rho_block(rho_sc_err)},
        "kl_vs_sc": {"per_row_spearman": _rho_block(rho_kl_sc)},
        "oracle_best_on_kl_axis": _rank_block(best_rank_kl, n_eligs),
        "oracle_best_on_sc_axis": _rank_block(best_rank_sc, n_eligs),
        "picks_on_error_axis": {
            "note": "ascending error rank, 0 = the pick is the oracle-best",
            "mc_pick_mean_normalized_rank": round(float(mc_norm.mean()), 5),
            "mc_pick_is_best_rate": round(float((mc_norm == 0).mean()), 5),
            "mc_pick_is_worst_rate": round(float((mc_norm == 1).mean()), 5),
            "sc_pick_mean_normalized_rank": round(float(sc_norm.mean()), 5),
            "sc_pick_is_best_rate": round(float((sc_norm == 0).mean()), 5),
            "sc_pick_is_worst_rate": round(float((sc_norm == 1).mean()), 5),
            "uniform_null_mean": 0.5,
        },
    }
    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=1))
        print(f"wrote {out_path}")
    if raw_out:
        np.savez_compressed(
            raw_out,
            rho_kl_err=np.array(rho_kl_err),
            rho_sc_err=np.array(rho_sc_err),
            rho_kl_sc=np.array(rho_kl_sc),
            best_rank_kl=np.array(best_rank_kl),
            best_rank_sc=np.array(best_rank_sc),
            n_eligs=np.array(n_eligs),
        )
        print(f"wrote {raw_out}")
    kve = out["kl_vs_error"]["per_row_spearman"]
    print(
        f"per-row Spearman(KL, err): {kve['mean']:+.5f} CI95 "
        f"[{kve['ci95'][0]:+.5f}, {kve['ci95'][1]:+.5f}] over "
        f"{kve['n_rows']} rows",
    )
    ob = out["oracle_best_on_kl_axis"]
    print(
        f"oracle-best mean normalized KL-rank: "
        f"{ob['mean_normalized_rank']:.5f} (uniform null 0.5), top-1 "
        f"rate {ob['top1_rate']:.5f} vs null {ob['top1_uniform_null']:.5f}",
    )
    return out


# ---------------------------------------------------------------- oracle


def _fixture() -> tuple[dict, dict]:
    """Six-row planted fixture, exact arithmetic throughout. Candidate
    offsets from truth are 0.1/0.2/0.3/0.4 while KL is 1/2/3/4 in the
    same candidate order -> per-row Spearman(KL, err) = +1 exactly and
    the row-centered pooled Pearson = +1 (linear). SC is planted as the
    REVERSE order (single-step mean_logprob descending in candidate
    index) -> Spearman(SC, err) = -1, Spearman(KL, SC) = -1. The
    oracle-best (offset 0.1) sits at KL-rank 3 (bottom) and SC-rank 0
    (top) on full rows; row 5 truncates candidate 3 -> 3 eligibles,
    ranks 2 and 0. Row 4 plants a KL TIE at the top (2.0, 2.0, 4.0,
    4.0): strictly-greater rank of the best candidate = 2."""
    rng = np.random.default_rng(11)
    n, c, s, d = 6, 4, 5, 2
    truth = rng.normal(size=(n, s, d)).astype(np.float64)
    offsets = np.array([0.1, 0.2, 0.3, 0.4])
    cand_pred = np.stack([truth + o for o in offsets], axis=1)
    kl = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (n, 1))
    kl[4] = [2.0, 2.0, 4.0, 4.0]
    kl[5, 3] = np.nan
    cand_pred[5, 3] = np.nan
    rows = []
    vocab = 100
    for i in range(n):
        cands = [
            {
                "text": f"cand{j}",
                "truncated": bool(i == 5 and j == 3),
                # single-step: sc = -mean_logprob - log(vocab),
                # descending in j -> reverse of the KL order
                "mean_logprob": [float(j)],
                "allowed_vocab": vocab,
            }
            for j in range(c)
        ]
        rows.append({"index": 10 + i, "candidates": cands})
    scores = {
        "index": np.arange(10, 10 + n),
        "truth": truth,
        "valid": np.ones((n, s), dtype=bool),
        "core": np.ones(n, dtype=bool),
        mcres.KL_KEY: kl,
        mcres.CAND_PRED_KEY: cand_pred,
    }
    return scores, {"rows": rows}


def oracle() -> None:
    def expect_exit(fn: Callable[[], object], needle: str, label: str) -> None:
        try:
            fn()
        except SystemExit as e:
            assert needle in str(e), f"{label}: wrong abort: {e}"
            print(f"abort branch OK: {label}")
            return
        raise AssertionError(f"{label}: abort did not fire")

    # rank helpers first: ties share average ranks, exact
    assert np.array_equal(
        avg_rank(np.array([3.0, 1.0, 2.0, 2.0])),
        np.array([4.0, 1.0, 2.5, 2.5]),
    )
    assert spearman(np.array([1.0, 1.0]), np.array([1.0, 2.0])) is None
    assert spearman(np.array([1.0, 2.0, 3.0]), np.array([5.0, 6.0, 9.0])) == 1.0

    scores, candidates = _fixture()
    out = analyze(scores, candidates, None)
    assert out["n_rows"] == 6 and out["n_rows_dropped_no_valid_frames"] == 0
    assert out["eligible_count_histogram"] == {"2": 0, "3": 1, "4": 5}
    # planted monotone orders, exact hand arithmetic: rows 0-3 and 5
    # are strictly monotone (rho +1); row 4's KL tie gives rank vectors
    # [1.5, 1.5, 3.5, 3.5] vs [1, 2, 3, 4] -> rho = 4/sqrt(20) = 2/sqrt(5)
    exp_mean = round((5 + 2 / math.sqrt(5)) / 6, 5)
    assert out["kl_vs_error"]["per_row_spearman"]["mean"] == exp_mean
    assert out["kl_vs_error"]["per_row_spearman"]["n_rows"] == 6
    assert out["sc_vs_error"]["per_row_spearman"]["mean"] == -1.0
    assert out["kl_vs_sc"]["per_row_spearman"]["mean"] == -exp_mean
    # pooled row-centered Pearson by hand: num 2.6, sum kc^2 26,
    # sum ec^2 0.27 -> 2.6/sqrt(7.02)
    assert out["kl_vs_error"]["pooled_pearson_row_centered"] == round(
        2.6 / math.sqrt(26 * 0.27),
        5,
    )
    # oracle-best ranks: rows 0-3 rank 3 of 4, row 4 tie-top rank 2,
    # row 5 rank 2 of 3 -> histogram [0, 0, 2, 4]
    ob = out["oracle_best_on_kl_axis"]
    assert ob["histogram_rank0_is_axis_top"] == [0, 0, 2, 4], ob
    exp_norm = round((4 * 1.0 + 2 / 3 + 1.0) / 6, 5)  # rows 0-3 + row 4 + row 5
    assert ob["mean_normalized_rank"] == exp_norm
    assert ob["top1_rate"] == 0.0
    assert ob["bottom1_rate"] == round(5 / 6, 5)
    assert out["oracle_best_on_sc_axis"]["histogram_rank0_is_axis_top"] == [6, 0, 0, 0]
    assert out["oracle_best_on_sc_axis"]["top1_rate"] == 1.0
    # picks on the error axis: mc pick = highest KL = worst candidate
    # on the strict rows (row 4's tie -> lowest index of the tied max,
    # candidate 2, err rank 2 of 3); sc pick = candidate 0 = oracle-best
    pk = out["picks_on_error_axis"]
    assert pk["mc_pick_is_worst_rate"] == round(5 / 6, 5)
    assert pk["mc_pick_is_best_rate"] == 0.0
    assert pk["mc_pick_mean_normalized_rank"] == exp_norm
    assert pk["sc_pick_is_best_rate"] == 1.0
    print("planted fixture OK (spearman exact, rank histograms, picks map)")

    # constant-KL row: excluded from the rho read, counted, no crash
    const_scores, const_cands = _fixture()
    const_scores[mcres.KL_KEY][0] = [1.0, 1.0, 1.0, 1.0]
    out2 = analyze(const_scores, const_cands, None)
    assert out2["kl_vs_error"]["kl_constant_rows_excluded"] == 1
    assert out2["kl_vs_error"]["per_row_spearman"]["n_rows"] == 5
    print("constant-KL row exclusion OK")

    # ---- abort branches ----
    scores, candidates = _fixture()
    deg = json.loads(json.dumps(candidates))
    for cand in deg["rows"][2]["candidates"][1:]:
        cand["truncated"] = True
    deg_scores = dict(scores)
    deg_kl = scores[mcres.KL_KEY].copy()
    deg_kl[2, 1:] = np.nan
    deg_pred = scores[mcres.CAND_PRED_KEY].copy()
    deg_pred[2, 1:] = np.nan
    deg_scores[mcres.KL_KEY] = deg_kl
    deg_scores[mcres.CAND_PRED_KEY] = deg_pred
    expect_exit(
        lambda: analyze(deg_scores, deg, None),
        "degenerate",
        "single-eligible row",
    )
    expect_exit(
        lambda: analyze(
            {k: v for k, v in scores.items() if k != mcres.KL_KEY},
            candidates,
            None,
        ),
        "missing mcselect:kl",
        "missing kl key",
    )
    bad = scores[mcres.KL_KEY].copy()
    bad[5, 3] = 0.5
    expect_exit(
        lambda: analyze({**scores, mcres.KL_KEY: bad}, candidates, None),
        "INELIGIBLE",
        "finite KL at truncated candidate",
    )
    nan_kl = scores[mcres.KL_KEY].copy()
    nan_kl[1, 0] = np.nan
    expect_exit(
        lambda: analyze({**scores, mcres.KL_KEY: nan_kl}, candidates, None),
        "ELIGIBLE",
        "non-finite KL at eligible candidate",
    )
    short = {k: (v[:4] if isinstance(v, np.ndarray) else v) for k, v in scores.items()}
    expect_exit(
        lambda: analyze(short, candidates, None),
        "partial dump",
        "row-count mismatch",
    )
    mixed = json.loads(json.dumps(candidates))
    mixed["rows"][0]["candidates"][1]["allowed_vocab"] = 7
    expect_exit(
        lambda: analyze(scores, mixed, None),
        "mixed allowed_vocab",
        "mixed vocab",
    )
    print("oracle: ALL branches OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--scores-stem", default=mcres.SCORES_STEM)
    parser.add_argument("--candidates", default=mcres.CAND_DEFAULT)
    parser.add_argument("--out", default=OUT_DEFAULT)
    parser.add_argument("--raw-out", default=RAW_DEFAULT)
    args = parser.parse_args()
    if args.oracle:
        oracle()
        return
    analyze(
        sdr._load_npz(f"{args.scores_stem}.npz"),
        json.loads(Path(args.candidates).read_text()),
        args.out,
        args.raw_out,
    )


if __name__ == "__main__":
    main()
