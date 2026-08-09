"""#6 rung (c) frozen-read script — masked-contrast (MG-Select) selection.

Mechanizes the reads of the rung-(c) pre-reg draft
(2026-08-09-prereg-subgoal-mcselect.md), landed PRE-DATA per the house
convention (draws10_t1 / selection-ceiling / energy-score precedent):
this file is the producer's dump CONTRACT. The future scorer run must
emit, alongside the standard identity/state columns:

  ``mcselect:kl``          [N, C]        KL(p_cond(c) || p_masked^(1/tau))
                                         per candidate; **NaN at
                                         ineligible (truncated)
                                         candidates** — a finite score
                                         there is a contract violation.
  ``mcselect:cand_pred``   [N, C, S, D]  teacher-forced predictions per
                                         candidate.
  ``mcselect:pred_masked`` [N, S, D]     the subgoal-masked reference
                                         pass (record-only diagnostic).

plus report fields ``mcselect_tau`` (must be 4.0) and
``candidates_sha256`` (must match the banked rung-(b') candidates file
— the scorer re-ranks EXACTLY the width whose ceiling/floor are on the
board). The ARGMAX LIVES HERE, not in the producer: the producer
measures, this script adjudicates (single home for the tie rule).

Reads (frozen in the draft): primary falsifier = paired (mc - self)
CI95 vs the banked rung-(a) self arm (PASS iff entirely below 0;
entirely above 0 = anti-selection second strike, the zero-training
scorer family closes); Delta_mc vs bare; capture fraction vs the
banked ceil-self -0.181; late-horizon signature; agreement records.
Inert-scorer abort: mc pick text == greedy on > 95% of rows.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
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
sdr = _sibling("subgoal_draws_results")

SCORES_STEM = (
    "reports/eval__bijou_arb_rcond_100k_ddp4__step_100000"
    "__stateprobe_q4_subgoalmcselect"
)
CAND_DEFAULT = (
    "reports/eval__bijou_arb_rcond_100k_ddp4__step_100000"
    "__stateprobe_q4_subgoalcleandraws_candidates.json"
)
BANKED_DEFAULT = "reports/analysis__subgoal_draws_cleanlist_q4_ar100k_k4l2.json"
OUT_DEFAULT = "reports/analysis__subgoal_mcselect_q4_ar100k_k4l2.json"

KL_KEY = "mcselect:kl"
CAND_PRED_KEY = "mcselect:cand_pred"
MASKED_PRED_KEY = "mcselect:pred_masked"
TAU_REQUIRED = 4.0
INERT_AGREE_BAR = 0.95


def eligible_list(cands: list[dict]) -> list[int]:
    """The rung-(b') clean filter, verbatim: non-truncated candidates."""
    return [i for i, c in enumerate(cands) if not c["truncated"]]


def mc_picks(
    kl: np.ndarray,
    eligibles: list[list[int]],
) -> tuple[np.ndarray, int]:
    """Argmax KL over the eligible list per row; ties -> lowest index."""
    picks = np.zeros(len(eligibles), dtype=np.int64)
    ties = 0
    for i, elig in enumerate(eligibles):
        scores = kl[i, elig]
        best = float(scores.max())
        winners = [e for e, s in zip(elig, scores, strict=True) if float(s) == best]
        ties += len(winners) > 1
        picks[i] = winners[0]
    return picks, ties


def _horizon_delta(arm_curve: list[float], base_curve: list[float]) -> dict:
    n10 = max(1, len(base_curve) // 10)
    d = np.array(arm_curve) - np.array(base_curve)
    return {
        "first10": round(float(d[:n10].mean()), 5),
        "last10": round(float(d[-n10:].mean()), 5),
    }


def analyze(
    scores_npz: dict,
    scores_rep: dict,
    base: dict,
    self_npz: dict,
    candidates: dict,
    candidates_sha: str,
    banked: dict,
    out_path: str | None,
) -> dict:
    # ---- contract guards first: every number below rides on them ----
    tau = scores_rep.get("mcselect_tau")
    if tau != TAU_REQUIRED:
        sys.exit(
            f"scorer report mcselect_tau={tau!r} != the pre-registered "
            f"{TAU_REQUIRED} — wrong scorer configuration, stop",
        )
    rep_sha = scores_rep.get("candidates_sha256")
    if rep_sha != candidates_sha:
        sys.exit(
            "scorer run's candidates_sha256 does not match the banked "
            "rung-(b') candidates file — the run re-ranked a different "
            "width, every banked comparator is void, stop",
        )
    for key in (KL_KEY, CAND_PRED_KEY, MASKED_PRED_KEY):
        if key not in scores_npz:
            sys.exit(f"scores npz missing {key} — not a mcselect dump, stop")
    if sdr.BASELINE_KEY in scores_npz:
        sys.exit(
            "scores npz carries a bare bijou column — the baseline must "
            "never re-run, stop",
        )
    for key in ("head_to_head_bon_minus_self", "adjudication_ceil_minus_self_labeled"):
        if key not in banked:
            sys.exit(f"banked analysis missing {key} — wrong analysis json, stop")

    # ---- row alignment: scores rows == candidates rows, then panel join
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
            f"{len(by_index)} — partial scorer run, stop",
        )
    rows_in_order = [by_index[int(ix)] for ix in idx]
    kl = scores_npz[KL_KEY]
    n_cand = max(len(r["candidates"]) for r in rows_in_order)
    if kl.ndim != 2 or kl.shape[1] != n_cand:
        sys.exit(
            f"{KL_KEY} shape {kl.shape} does not match the candidate "
            f"width {n_cand} — contract violation, stop",
        )
    eligibles: list[list[int]] = []
    for i, row in enumerate(rows_in_order):
        cands = row["candidates"]
        elig = eligible_list(cands)
        if not elig:
            sys.exit(
                f"row index {int(idx[i])} has an all-truncated candidate "
                "list — rung (b') banked 0 fallback rows, this run does "
                "not mirror the clean convention, stop",
            )
        if not np.isfinite(kl[i, elig]).all():
            sys.exit(
                f"non-finite KL at an ELIGIBLE candidate (row index "
                f"{int(idx[i])}) — scorer numerics broken, stop",
            )
        inelig = [j for j in range(len(cands)) if j not in elig]
        if inelig and np.isfinite(kl[i, inelig]).any():
            sys.exit(
                f"finite KL at an INELIGIBLE (truncated) candidate (row "
                f"index {int(idx[i])}) — the producer scored outside the "
                "clean filter, stop",
            )
        eligibles.append(elig)

    # ---- anchor + panel join (the sdr conventions, reused verbatim) ----
    truth, valid, core, w = bbr.masks(base)
    base_err = np.abs(base[sdr.BASELINE_KEY] - truth)
    bc = bbr.pooled_chunk(base_err, core, w)
    bf = bbr.pooled_first(base_err, valid, core)
    anchor = bbr.ANCHORS["ar"]
    if abs(bc - anchor[0]) >= sdr.SUMMARY_TOL or abs(bf - anchor[1]) >= sdr.SUMMARY_TOL:
        sys.exit(
            f"baseline re-pool {bc:.4f}/{bf:.4f} does not reproduce the "
            f"banked anchor {anchor[0]}/{anchor[1]} — wrong baseline npz, stop",
        )
    sdr.check_pairing(base, self_npz, "banked self arm")
    rows, subset = sdr.join_rows(base, scores_npz, "mcselect run")
    if subset:
        base = {k: v[rows] for k, v in base.items()}
        self_npz = {k: v[rows] for k, v in self_npz.items()}
        truth, valid, core, w = bbr.masks(base)
        base_err = np.abs(base[sdr.BASELINE_KEY] - truth)
    sdr.check_state_rows(base, scores_npz, "mcselect run")

    # ---- the pick (argmax here, tie rule here) + inert guard ----
    picks, ties = mc_picks(kl, eligibles)
    greedy_agree = float(
        np.mean(
            [
                row["candidates"][int(p)]["text"] == row["candidates"][0]["text"]
                for row, p in zip(rows_in_order, picks, strict=True)
            ],
        ),
    )
    if greedy_agree > INERT_AGREE_BAR:
        sys.exit(
            f"inert scorer: mc pick text == greedy on {greedy_agree:.1%} "
            f"of rows (> {INERT_AGREE_BAR:.0%}) — no verdict either way, stop",
        )

    # ---- reads ----
    n = len(picks)
    err_mc = np.abs(scores_npz[CAND_PRED_KEY][np.arange(n), picks] - truth)
    err_masked = np.abs(scores_npz[MASKED_PRED_KEY] - truth)
    self_err = np.abs(self_npz[sdr.SELF_KEY] - truth)
    base_frame, nvalid = bbr.frame_mae(base_err, w)
    mc_frame, _ = bbr.frame_mae(err_mc, w)
    self_frame, _ = bbr.frame_mae(self_err, w)
    keep = (nvalid > 0) & core

    def ci_block(deltas: np.ndarray) -> dict:
        lo, hi = bbr.bootstrap_ci(deltas)
        return {
            "delta_frame_mean": round(float(deltas.mean()), 5),
            "ci95": [round(lo, 5), round(hi, 5)],
            "n_frames": int(keep.sum()),
        }

    primary = ci_block((mc_frame - self_frame)[keep])
    delta_mc = ci_block((mc_frame - base_frame)[keep])
    banked_ceil_self = banked["adjudication_ceil_minus_self_labeled"]
    capture = (
        round(primary["delta_frame_mean"] / banked_ceil_self["delta_frame_mean"], 5)
        if banked_ceil_self["delta_frame_mean"]
        else None
    )
    base_curve = sdr.step_curve(base_err, valid, core)
    mc_curve = sdr.step_curve(err_mc, valid, core)
    lo, hi = primary["ci95"]
    if hi < 0:
        verdict = "PASS — mc pick beats the greedy self subgoal (CI95 < 0)"
    elif lo > 0:
        verdict = (
            "ANTI-SELECT — second strike after SC; the zero-training "
            "scorer family CLOSES for this trunk"
        )
    else:
        verdict = "FALSIFIED — CI spans 0; record-only, family stays closed"
    out = {
        "primary_mc_minus_self": primary,
        "verdict": verdict,
        "delta_mc_vs_bare": delta_mc,
        "capture_fraction_of_ceiling": capture,
        "banked_comparators": {
            "ceil_minus_self_labeled": banked_ceil_self,
            "bon_minus_self": banked["head_to_head_bon_minus_self"],
        },
        "mc_pooled": {
            "chunk_mae": round(bbr.pooled_chunk(err_mc, core, w), 5),
            "first_mae": round(bbr.pooled_first(err_mc, valid, core), 5),
        },
        "masked_reference_pooled": {
            "chunk_mae": round(bbr.pooled_chunk(err_masked, core, w), 5),
        },
        "horizon_delta": _horizon_delta(mc_curve, base_curve),
        "curve": [round(v, 5) for v in mc_curve],
        "agreement": {
            "n_frames": n,
            "pick_text_differs_from_greedy": round(1.0 - greedy_agree, 5),
            "tie_rows": ties,
            "mc_agrees_with_sc_pick": round(
                float(
                    np.mean(
                        [
                            int(p) == row["picks"]["bon"]
                            for row, p in zip(rows_in_order, picks, strict=True)
                        ],
                    ),
                ),
                5,
            ),
            "mc_agrees_with_ceil_on_labeled": _ceil_agreement(rows_in_order, picks),
        },
    }
    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=1))
        print(f"wrote {out_path}")
    print(
        f"PRIMARY (mc - self): {primary['delta_frame_mean']:+.5f} "
        f"CI95 [{lo:+.5f}, {hi:+.5f}] -> {verdict}",
    )
    return out


def _ceil_agreement(rows_in_order: list[dict], picks: np.ndarray) -> float | None:
    hits, labeled = 0, 0
    for row, p in zip(rows_in_order, picks, strict=True):
        ceil = row["picks"].get("ceil")
        if ceil is not None:
            labeled += 1
            hits += int(p) == ceil
    return round(hits / labeled, 5) if labeled else None


# ---------------------------------------------------------------- oracle


def _fixture() -> tuple:
    """Six-row planted fixture: candidate 2 wins the KL argmax on rows
    0-3 (its predictions are strictly better than self), row 4 plants a
    TIE between candidates 0 and 1 (rule: lowest index), row 5's argmax
    is candidate 1 with truncated candidate 2 (NaN KL respected)."""
    rng = np.random.default_rng(7)
    n, c, s, d = 6, 3, 10, 2
    truth = rng.normal(size=(n, s, d))
    valid = np.ones((n, s), dtype=bool)
    core = np.ones(n, dtype=bool)
    index = np.arange(10, 10 + n)
    state = rng.normal(size=(n, s, d))
    base_pred = truth + 0.8
    self_pred = truth + 0.6
    base = {
        "index": index,
        "truth": truth,
        "valid": valid,
        "core": core,
        "repo_id": np.array([f"r{i}" for i in range(n)]),
        sdr.BASELINE_KEY: base_pred,
        "pred:state-copy": state,
        "pred:state-copy-norm": state + 1,
    }
    self_npz = dict(base)
    del self_npz[sdr.BASELINE_KEY]
    self_npz[sdr.SELF_KEY] = self_pred
    cand_pred = np.stack(
        [truth + 0.7, truth + 0.5, truth + 0.2],
        axis=1,
    )  # candidate 2 best, then 1, then 0
    kl = np.tile(np.array([0.1, 0.2, 0.9]), (n, 1))
    kl[4] = [0.5, 0.5, 0.1]  # tie on 0/1 -> pick 0
    kl[5] = [0.1, 0.8, np.nan]  # candidate 2 truncated on row 5
    expected_picks = np.array([2, 2, 2, 2, 0, 1])
    rows = []
    for i in range(n):
        cands = [
            {"text": f"cand{j}row{i}" if j else "greedy", "truncated": False}
            for j in range(c)
        ]
        if i == 5:
            cands[2]["truncated"] = True
        rows.append(
            {
                "index": int(index[i]),
                "candidates": cands,
                "picks": {"bon": 1, "ceil": 2 if i % 2 == 0 else None},
            },
        )
    candidates = {"rows": rows}
    scores = {
        "index": index,
        "truth": truth,
        "valid": valid,
        "core": core,
        "repo_id": base["repo_id"],
        "pred:state-copy": state,
        "pred:state-copy-norm": state + 1,
        KL_KEY: kl,
        CAND_PRED_KEY: cand_pred,
        MASKED_PRED_KEY: truth + 0.9,
    }
    rep = {"mcselect_tau": 4.0, "candidates_sha256": "cafe"}
    banked = {
        "head_to_head_bon_minus_self": {"delta_frame_mean": 0.21043},
        "adjudication_ceil_minus_self_labeled": {"delta_frame_mean": -0.18103},
    }
    return scores, rep, base, self_npz, candidates, banked, expected_picks


def oracle() -> None:
    def expect_exit(fn: Callable[[], object], needle: str, label: str) -> None:
        try:
            fn()
        except SystemExit as e:
            assert needle in str(e), f"{label}: wrong abort: {e}"
            print(f"abort branch OK: {label}")
            return
        raise AssertionError(f"{label}: abort did not fire")

    scores, rep, base, self_npz, candidates, banked, expected_picks = _fixture()
    anchors_saved = bbr.ANCHORS["ar"]
    truth, valid, core, w = bbr.masks(base)
    base_err = np.abs(base[sdr.BASELINE_KEY] - truth)
    bbr.ANCHORS["ar"] = (
        bbr.pooled_chunk(base_err, core, w),
        bbr.pooled_first(base_err, valid, core),
    )
    try:
        out = analyze(scores, rep, base, self_npz, candidates, "cafe", banked, None)
        # planted argmax: independent recompute, exact
        picks = np.array(
            [
                int(p)
                for p in mc_picks(
                    scores[KL_KEY],
                    [eligible_list(r["candidates"]) for r in candidates["rows"]],
                )[0]
            ],
        )
        assert np.array_equal(picks, expected_picks), picks
        assert out["agreement"]["tie_rows"] == 1
        # exact paired arithmetic: |cand_pred - truth| is constant offset
        # per candidate, so frame MAE deltas are exact rationals
        exp = {2: 0.2 - 0.6, 0: 0.7 - 0.6, 1: 0.5 - 0.6}
        expected_mean = float(np.mean([exp[int(p)] for p in expected_picks]))
        got = out["primary_mc_minus_self"]["delta_frame_mean"]
        assert got == round(expected_mean, 5), (got, expected_mean)
        assert out["verdict"].startswith("PASS"), out["verdict"]
        exp_capture = round(round(expected_mean, 5) / -0.18103, 5)
        assert out["capture_fraction_of_ceiling"] == exp_capture
        exp_bare = float(np.mean([{2: -0.6, 0: -0.1, 1: -0.3}[int(p)] for p in picks]))
        assert out["delta_mc_vs_bare"]["delta_frame_mean"] == round(exp_bare, 5)
        # horizon: constant offsets -> first10 == last10 == pooled delta
        hd = out["horizon_delta"]
        assert abs(hd["first10"] - round(exp_bare, 5)) < 1e-4, hd
        assert abs(hd["first10"] - hd["last10"]) < 1e-9
        print("planted fixture OK (picks, tie rule, exact deltas, capture)")

        # ---- abort branches ----
        expect_exit(
            lambda: analyze(
                scores,
                {**rep, "mcselect_tau": 1.0},
                base,
                self_npz,
                candidates,
                "cafe",
                banked,
                None,
            ),
            "mcselect_tau",
            "tau mismatch",
        )
        expect_exit(
            lambda: analyze(
                scores,
                rep,
                base,
                self_npz,
                candidates,
                "beef",
                banked,
                None,
            ),
            "candidates_sha256",
            "candidates sha mismatch",
        )
        expect_exit(
            lambda: analyze(
                {k: v for k, v in scores.items() if k != KL_KEY},
                rep,
                base,
                self_npz,
                candidates,
                "cafe",
                banked,
                None,
            ),
            "missing mcselect:kl",
            "missing kl key",
        )
        expect_exit(
            lambda: analyze(
                {**scores, sdr.BASELINE_KEY: base[sdr.BASELINE_KEY]},
                rep,
                base,
                self_npz,
                candidates,
                "cafe",
                banked,
                None,
            ),
            "bare bijou column",
            "baseline re-run",
        )
        expect_exit(
            lambda: analyze(
                scores,
                rep,
                base,
                self_npz,
                candidates,
                "cafe",
                {"head_to_head_bon_minus_self": {}},
                None,
            ),
            "banked analysis missing",
            "wrong banked analysis",
        )
        bad_kl = scores[KL_KEY].copy()
        bad_kl[5, 2] = 0.99  # finite at the truncated candidate
        expect_exit(
            lambda: analyze(
                {**scores, KL_KEY: bad_kl},
                rep,
                base,
                self_npz,
                candidates,
                "cafe",
                banked,
                None,
            ),
            "INELIGIBLE",
            "finite KL at truncated candidate",
        )
        nan_kl = scores[KL_KEY].copy()
        nan_kl[0, 1] = np.nan
        expect_exit(
            lambda: analyze(
                {**scores, KL_KEY: nan_kl},
                rep,
                base,
                self_npz,
                candidates,
                "cafe",
                banked,
                None,
            ),
            "ELIGIBLE",
            "non-finite KL at eligible candidate",
        )
        all_trunc = json.loads(json.dumps(candidates))
        for cand in all_trunc["rows"][3]["candidates"]:
            cand["truncated"] = True
        expect_exit(
            lambda: analyze(
                scores,
                rep,
                base,
                self_npz,
                all_trunc,
                "cafe",
                banked,
                None,
            ),
            "all-truncated",
            "all-truncated row",
        )
        short = {
            k: (v[:4] if isinstance(v, np.ndarray) else v) for k, v in scores.items()
        }
        expect_exit(
            lambda: analyze(
                short,
                rep,
                base,
                self_npz,
                candidates,
                "cafe",
                banked,
                None,
            ),
            "partial scorer run",
            "row-count mismatch vs candidates",
        )
        inert = json.loads(json.dumps(candidates))
        for row in inert["rows"]:
            for cand in row["candidates"]:
                cand["text"] = "greedy"
        expect_exit(
            lambda: analyze(
                scores,
                rep,
                base,
                self_npz,
                inert,
                "cafe",
                banked,
                None,
            ),
            "inert scorer",
            "inert-scorer bar",
        )
        print("oracle: ALL branches OK")
    finally:
        bbr.ANCHORS["ar"] = anchors_saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--scores-stem", default=SCORES_STEM)
    parser.add_argument("--baseline-stem", default=sdr.BASE_STEM)
    parser.add_argument("--self-stem", default=sdr.SELF_STEM)
    parser.add_argument("--candidates", default=CAND_DEFAULT)
    parser.add_argument("--banked-analysis", default=BANKED_DEFAULT)
    parser.add_argument("--out", default=OUT_DEFAULT)
    args = parser.parse_args()
    if args.oracle:
        oracle()
        return
    cand_bytes = Path(args.candidates).read_bytes()
    analyze(
        sdr._load_npz(f"{args.scores_stem}.npz"),
        json.loads(Path(f"{args.scores_stem}.json").read_text()),
        sdr._load_npz(f"{args.baseline_stem}.npz"),
        sdr._load_npz(f"{args.self_stem}.npz"),
        json.loads(cand_bytes.decode()),
        hashlib.sha256(cand_bytes).hexdigest(),
        json.loads(Path(args.banked_analysis).read_text()),
        args.out,
    )


if __name__ == "__main__":
    main()
