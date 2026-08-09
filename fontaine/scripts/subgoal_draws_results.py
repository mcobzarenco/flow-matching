"""Frozen reads for subgoal-draws selection (#6 rung (b)).

One command over the rung-(b) run's dumps (pre-reg
2026-08-08-prereg-subgoal-draws.md, "Frozen reads" — semantics frozen
there, this file only mechanizes them):

  1. PRIMARY  Δ_bon = chunk_mae(bon) − banked 5.8026 on all core
     frames, paired per-frame seeded bootstrap CI — with the
     HEAD-TO-HEAD paired (bon − self) read per-frame vs the BANKED
     rung-(a) self npz beside it: the rung's pass/fail number
     (falsifier: CI95 not entirely below 0 ⇒ FALSIFIED);
  2. BOUND    Δ_ceil on the labeled subset, with paired (ceil − self)
     adjudicating WHY on a null: CI including 0 ⇒ no-diversity (the
     selection family closes at this width), clearly below ⇒ no-scorer
     (heavier signals may earn a pre-reg);
  3. AGREEMENT (record-only, from the candidates dump): pick ≠ greedy
     rate, likelihood/medoid alternates' agreement with the primary and
     ceil picks, per-frame unique-candidate counts;
  4. HORIZON  per-step-in-horizon MAE curves + early/late tail means
     (the slot's rung-(a) gain was 6x late-horizon);
  5. MIRRORS  first_mae versions of 1–2;
  6. EXECUTION ORACLES (each failure a hard abort): banked baseline
     anchor reproduced; identity columns byte-match across all npzs;
     state-copy rows byte-match the banked panel; policy keys carry the
     mode suffixes and no bare bijou column exists; reports record mode
     "draws" + the width/temperature and reproduce npz-recomputed
     pooled values; the candidates dump covers the panel and its LIVE
     picks byte-match an offline scorer recompute (bon from the
     distribution stats, ceil from token-F1 vs the true label,
     label-less rows None). The pass-1 narr column vs the banked
     rung-(a) narr column is RECORDED, not adjudicated (composition/
     device kernel noise, the amendment-1 class).

The baseline and the rung-(a) self arm are BANKED npzs — never re-run.
Small deltas are quoted beside the decode-noise floor (±0.016 per-frame
CI at matched composition, amendment 1).

``--candidate-filter clean`` is the rung-(b') read (pre-reg
2026-08-08-prereg-subgoal-draws-cleanlist): reads 1–6 verbatim with
"the 9 candidates" read as "the eligible list" — arm keys carry the
filter (``_boncleansubgoal``/``_ceilcleansubgoal``), every offline
pick recompute (arms AND record-only alternates) runs over the
eligible (non-truncated) list with the greedy fallback rule, the
dumped eligible/fallback fields must byte-agree with a recompute from
the truncated flags, and the agreement block gains the per-row
eligible-list size distribution + fallback-row count.

Row pairing: a full-panel draws npz must byte-match the baseline on
the identity columns; a shorter draws npz (the pre-registered q4
cost-gate fallback, stateprobe_q4 plan) is joined on the corpus concat
``index`` (the draws10/energy-score join convention) and the baseline +
banked-self comparators re-pooled onto those rows — recorded as
subset_mode, never silent.

``--oracle`` runs the pre-data selftest: exact-arithmetic fixtures,
degenerate arm ⇒ delta exactly 0 with CI [0, 0] and the falsifier
firing, and every abort branch exercised — in BOTH conventions,
including the planted filter-binds world (full-list argmax truncated ⇒
filtered pick differs, both scorers), the all-truncated fallback, and
the q4-shaped slice fixture (joins, re-pools, reproduces the sliced
direct computation; duplicate/foreign-index and identity-drift aborts).
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bijou.eval.subgoal_scoring import (
    ceiling_pick,
    eligible_indices,
    likelihood_pick,
    medoid_pick,
    self_certainty_pick,
)

_HERE = Path(__file__).resolve().parent


def _sibling(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bbr = _sibling("box_batch_results")

BASE_STEM = "reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2"
SELF_STEM = f"{BASE_STEM}_selfsubgoal"
DRAWS_STEM = f"{BASE_STEM}_subgoaldraws"
OUT_DEFAULT = "reports/analysis__subgoal_draws_ar100k_k4l2.json"
# Rung (b') clean-list run (pre-reg …-cleanlist): distinct stems and
# arm keys carry the filter so a filtered read can never pool as (b).
CLEAN_DRAWS_STEM = f"{BASE_STEM}_subgoalcleandraws"
CLEAN_OUT_DEFAULT = "reports/analysis__subgoal_draws_cleanlist_ar100k_k4l2.json"

BASELINE_KEY = "pred:bijou@100000"
NARR_KEY = "pred:bijou@100000_narrsubgoal"
SELF_KEY = "pred:bijou@100000_selfsubgoal"
BON_KEY = "pred:bijou@100000_bonsubgoal"
CEIL_KEY = "pred:bijou@100000_ceilsubgoal"
CLEAN_BON_KEY = "pred:bijou@100000_boncleansubgoal"
CLEAN_CEIL_KEY = "pred:bijou@100000_ceilcleansubgoal"


def _arm_keys(candidate_filter: str | None) -> tuple[str, str]:
    if candidate_filter == "clean":
        return CLEAN_BON_KEY, CLEAN_CEIL_KEY
    return BON_KEY, CEIL_KEY


def _eligible(cands: list[dict], candidate_filter: str | None) -> list[int]:
    if candidate_filter == "clean":
        return eligible_indices([c["truncated"] for c in cands])
    return list(range(len(cands)))


IDENTITY_KEYS = ("index", "truth", "valid", "repo_id", "core")
STATE_KEYS = ("pred:state-copy", "pred:state-copy-norm")
SUMMARY_TOL = 5e-3
# Matched-composition decode-noise floor (amendment 1, rung (a)):
# per-frame paired CI ±0.016 — quoted beside every small delta.
NOISE_FLOOR = 0.016


def _load_npz(path: str | Path) -> dict:
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def _bytes_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def check_pairing(base: dict, probe: dict, label: str) -> None:
    for key in IDENTITY_KEYS:
        if not np.array_equal(base[key], probe[key]):
            sys.exit(f"{label}: panel pairing broken on {key} — stop")


def join_rows(base: dict, probe: dict, label: str) -> tuple[np.ndarray, bool]:
    """Rows of the baseline npz pairing the probe npz, in probe order.

    Equal length ⇒ the identity columns must byte-match (no silent
    reordering); shorter probe ⇒ the pre-registered q4 fallback, joined
    on `index` with identity equality re-checked on the joined rows
    (the draws10_t1/energy-score convention). Anything else aborts.
    """
    bi, pi = base["index"], probe["index"]
    if len(pi) == len(bi):
        check_pairing(base, probe, label)
        return np.arange(len(bi)), False
    if len(pi) > len(bi):
        sys.exit(
            f"{label}: probe npz has MORE rows ({len(pi)}) than the "
            f"baseline panel ({len(bi)}) — stop",
        )
    if len(np.unique(pi)) != len(pi):
        sys.exit(f"{label}: probe npz has duplicate index values — refusing the join")
    pos = {int(ix): i for i, ix in enumerate(bi)}
    missing = [int(ix) for ix in pi if int(ix) not in pos]
    if missing:
        sys.exit(
            f"{label}: {len(missing)} probe rows absent from the baseline "
            f"panel (first: {missing[:3]}) — subset join broken",
        )
    rows = np.array([pos[int(ix)] for ix in pi])
    for key in ("truth", "valid", "repo_id", "core"):
        if not np.array_equal(base[key][rows], probe[key]):
            sys.exit(f"{label}: subset rows disagree with the baseline panel on {key}")
    return rows, True


def check_state_rows(base: dict, probe: dict, label: str) -> None:
    for key in STATE_KEYS:
        if key not in probe:
            sys.exit(f"{label}: {key} missing from the npz")
        if not _bytes_equal(base[key], probe[key]):
            sys.exit(
                f"{label}: {key} rows do NOT byte-match the banked panel — "
                "execution drift, stop",
            )


def check_report(
    npz: dict,
    key: str,
    report: dict,
    label: str,
    candidate_filter: str | None = None,
) -> None:
    if report.get("subgoal_mode") != "draws":
        sys.exit(
            f"{label}: report subgoal_mode {report.get('subgoal_mode')!r} != "
            "'draws' — wrong run, stop",
        )
    if report.get("subgoal_candidate_filter") != candidate_filter:
        sys.exit(
            f"{label}: report subgoal_candidate_filter "
            f"{report.get('subgoal_candidate_filter')!r} != expected "
            f"{candidate_filter!r} — candidate-filter provenance broken, stop",
        )
    if report.get("selfsubgoal_force_empty"):
        sys.exit(f"{label}: forced-empty run is never an arm read — stop")
    truth, valid, core, w = bbr.masks(npz)
    err = np.abs(npz[key] - truth)
    gc = bbr.pooled_chunk(err, core, w)
    gf = bbr.pooled_first(err, valid, core)
    policy = key.removeprefix("pred:")
    summ = [s for s in report.get("summaries", []) if s.get("policy") == policy]
    if len(summ) != 1:
        sys.exit(
            f"{label}: report has {len(summ)} summaries for {policy!r} "
            f"(have: {[s.get('policy') for s in report.get('summaries', [])]})",
        )
    wc, wf = summ[0]["chunk_mae"], summ[0]["first_mae"]
    if abs(gc - wc) >= SUMMARY_TOL or abs(gf - wf) >= SUMMARY_TOL:
        sys.exit(
            f"{label}: npz-recomputed chunk/first {gc:.4f}/{gf:.4f} do not "
            f"reproduce the report's {wc:.4f}/{wf:.4f} for {policy} — "
            "plan/scoring drift, stop",
        )
    print(f"{label}: report cross-check OK ({policy} chunk {gc:.4f} first {gf:.4f})")


def check_candidates(
    candidates: dict,
    base: dict,
    candidate_filter: str | None = None,
) -> dict[int, dict]:
    """Panel coverage + LIVE-pick equality vs an offline scorer
    recompute (abort-grade: the dumped picks are what the arms actually
    conditioned on, so a mismatch means the scorer path drifted). Under
    the clean-list filter the recompute runs over the eligible list and
    the dumped eligible/fallback fields must byte-agree with a
    recompute from the truncated flags. Returns rows keyed by global
    index."""
    rows = candidates.get("rows")
    if not isinstance(rows, list) or not rows:
        sys.exit("candidates json has no 'rows' — not a --dump-subgoal-candidates file")
    if candidates.get("subgoal_mode") != "draws":
        sys.exit(
            f"candidates json subgoal_mode {candidates.get('subgoal_mode')!r} "
            "!= 'draws' — wrong dump, stop",
        )
    if candidates.get("subgoal_candidate_filter") != candidate_filter:
        sys.exit(
            f"candidates json subgoal_candidate_filter "
            f"{candidates.get('subgoal_candidate_filter')!r} != expected "
            f"{candidate_filter!r} — candidate-filter provenance broken, stop",
        )
    if candidates.get("selfsubgoal_force_empty"):
        sys.exit("candidates json from a forced-empty run — never an arm read, stop")
    by_index = {int(r["index"]): r for r in rows}
    missing = [int(ix) for ix in base["index"] if int(ix) not in by_index]
    if missing:
        sys.exit(
            f"{len(missing)} panel rows have no candidate record (first "
            f"{missing[:3]}) — the dump does not cover the panel, stop",
        )
    for index, row in by_index.items():
        cands = row["candidates"]
        vocabs = {c["allowed_vocab"] for c in cands}
        if len(vocabs) != 1:
            sys.exit(f"row {index}: mixed allowed_vocab {sorted(vocabs)} — stop")
        elig = _eligible(cands, candidate_filter)
        if candidate_filter == "clean":
            members = set(elig)
            flags = [i in members for i in range(len(cands))]
            if row.get("eligible") != flags:
                sys.exit(
                    f"row {index}: dumped eligible flags {row.get('eligible')} "
                    f"!= recompute from truncated flags {flags} — eligible-"
                    "list drift, stop",
                )
            fallback = all(c["truncated"] for c in cands)
            if bool(row.get("fallback")) != fallback:
                sys.exit(
                    f"row {index}: dumped fallback {row.get('fallback')!r} != "
                    f"recompute {fallback} — fallback recording drift, stop",
                )
        bon = elig[
            self_certainty_pick(
                [cands[i]["mean_logprob"] for i in elig],
                cands[0]["allowed_vocab"],
            )
        ]
        if row["picks"]["bon"] != bon:
            sys.exit(
                f"row {index}: dumped bon pick {row['picks']['bon']} != "
                f"offline self-certainty recompute {bon} — scorer drift, stop",
            )
        label = row.get("true_subgoal")
        ceil = (
            None
            if label is None
            else elig[
                ceiling_pick(
                    [cands[i]["text"] for i in elig],
                    label,
                )
            ]
        )
        if row["picks"]["ceil"] != ceil:
            sys.exit(
                f"row {index}: dumped ceil pick {row['picks']['ceil']} != "
                f"offline token-F1 recompute {ceil} — scorer drift, stop",
            )
    return by_index


def labeled_mask(by_index: dict[int, dict], base: dict) -> np.ndarray:
    return np.array(
        [by_index[int(ix)].get("true_subgoal") is not None for ix in base["index"]],
    )


def paired_read(
    base_frame: np.ndarray,
    arm_frame: np.ndarray,
    keep: np.ndarray,
    pooled_delta: float,
) -> dict:
    deltas = (arm_frame - base_frame)[keep]
    lo, hi = bbr.bootstrap_ci(deltas)
    return {
        "delta_pooled": round(pooled_delta, 5),
        "delta_frame_mean": round(float(deltas.mean()), 5),
        "ci95": [round(lo, 5), round(hi, 5)],
        "n_frames": int(keep.sum()),
    }


def step_curve(err: np.ndarray, valid: np.ndarray, core: np.ndarray) -> list[float]:
    dims = err.shape[2]
    wv = (valid & core[:, None]).astype(np.float64)
    num = (err.sum(axis=2) * wv).sum(axis=0)
    den = wv.sum(axis=0) * dims
    return (num / np.maximum(den, 1)).tolist()


def agreement_records(
    by_index: dict[int, dict],
    candidate_filter: str | None = None,
) -> dict:
    """Read 3 — record-only, straight off the candidates dump. Every
    alternate operates on the same eligible list as the arms (the b'
    filter applies to every scorer); under the filter the per-row
    eligible-list size distribution and the fallback-row count ride
    along (pre-reg …-cleanlist, "scorer-agreement records")."""
    n = len(by_index)
    pick_ne_greedy = 0
    likelihood_eq_bon = 0
    medoid_eq_bon = 0
    likelihood_eq_ceil = 0
    medoid_eq_ceil = 0
    labeled = 0
    fallback_rows = 0
    unique_counts: list[int] = []
    eligible_sizes: list[int] = []
    for row in by_index.values():
        cands = row["candidates"]
        elig = _eligible(cands, candidate_filter)
        eligible_sizes.append(len(elig))
        if candidate_filter == "clean" and all(c["truncated"] for c in cands):
            fallback_rows += 1
        texts = [c["text"] for c in cands]
        unique_counts.append(len({texts[i] for i in elig}))
        bon = row["picks"]["bon"]
        if texts[bon] != texts[0]:
            pick_ne_greedy += 1
        lik = elig[likelihood_pick([cands[i]["chosen_logprob"] for i in elig])]
        med = elig[medoid_pick([texts[i] for i in elig])]
        likelihood_eq_bon += lik == bon
        medoid_eq_bon += med == bon
        ceil = row["picks"]["ceil"]
        if ceil is not None:
            labeled += 1
            likelihood_eq_ceil += lik == ceil
            medoid_eq_ceil += med == ceil
    counts = np.array(unique_counts)
    out = {
        "n_frames": n,
        "pick_text_differs_from_greedy": round(pick_ne_greedy / n, 5),
        "likelihood_agrees_with_bon": round(likelihood_eq_bon / n, 5),
        "medoid_agrees_with_bon": round(medoid_eq_bon / n, 5),
        "likelihood_agrees_with_ceil_on_labeled": (
            round(likelihood_eq_ceil / labeled, 5) if labeled else None
        ),
        "medoid_agrees_with_ceil_on_labeled": (
            round(medoid_eq_ceil / labeled, 5) if labeled else None
        ),
        "unique_candidates": {
            "mean": round(float(counts.mean()), 5),
            "min": int(counts.min()),
            "max": int(counts.max()),
            "frac_ge_2": round(float((counts >= 2).mean()), 5),
        },
    }
    if candidate_filter == "clean":
        sizes = np.array(eligible_sizes)
        out["eligible_list_size"] = {
            "mean": round(float(sizes.mean()), 5),
            "min": int(sizes.min()),
            "max": int(sizes.max()),
            "hist": {
                str(k): int((sizes == k).sum())
                for k in range(int(sizes.min()), int(sizes.max()) + 1)
                if int((sizes == k).sum())
            },
        }
        out["fallback_rows"] = fallback_rows
    return out


def analyze(
    base: dict,
    self_npz: dict,
    draws_npz: dict,
    draws_rep: dict,
    candidates: dict,
    anchor: tuple[float, float],
    out_path: str | None,
    candidate_filter: str | None = None,
) -> dict:
    bon_key, ceil_key = _arm_keys(candidate_filter)
    # ---- read 6 first: execution oracles gate every number below ----
    truth, valid, core, w = bbr.masks(base)
    base_err = np.abs(base[BASELINE_KEY] - truth)
    bc = bbr.pooled_chunk(base_err, core, w)
    bf = bbr.pooled_first(base_err, valid, core)
    if abs(bc - anchor[0]) >= SUMMARY_TOL or abs(bf - anchor[1]) >= SUMMARY_TOL:
        sys.exit(
            f"baseline re-pool {bc:.4f}/{bf:.4f} does not reproduce the "
            f"banked anchor {anchor[0]}/{anchor[1]} — wrong baseline npz, stop",
        )
    print(f"anchor OK: baseline re-pool {bc:.4f}/{bf:.4f}")
    for key, npz, label in (
        (SELF_KEY, self_npz, "banked self arm"),
        (bon_key, draws_npz, "bon arm"),
        (ceil_key, draws_npz, "ceil arm"),
        (NARR_KEY, draws_npz, "draws pass 1"),
    ):
        if key not in npz:
            sys.exit(
                f"{label}: {key} missing — policy name does not carry the mode, stop",
            )
    if BASELINE_KEY in draws_npz:
        sys.exit(
            "draws npz carries a bare bijou column — the baseline must "
            "never re-run, stop",
        )
    # A filtered read must never consume an unfiltered run's arms (and
    # vice versa): the OTHER convention's keys may not appear.
    for stray in set(_arm_keys("clean" if candidate_filter is None else None)):
        if stray in draws_npz:
            sys.exit(
                f"draws npz carries {stray} — wrong candidate-filter "
                "convention for this read, stop",
            )
    check_pairing(base, self_npz, "banked self arm")
    rows, subset = join_rows(base, draws_npz, "draws run")
    if subset:
        n_full = len(base["index"])
        base = {k: v[rows] for k, v in base.items()}
        self_npz = {k: v[rows] for k, v in self_npz.items()}
        truth, valid, core, w = bbr.masks(base)
        base_err = np.abs(base[BASELINE_KEY] - truth)
        bc = bbr.pooled_chunk(base_err, core, w)
        bf = bbr.pooled_first(base_err, valid, core)
        print(
            f"subset join: {len(rows)}/{n_full} panel rows (pre-registered "
            f"q4 fallback); baseline + banked self re-pooled onto the "
            f"subset ({bc:.4f}/{bf:.4f})",
        )
    check_state_rows(base, draws_npz, "draws run")
    check_report(draws_npz, bon_key, draws_rep, "bon arm", candidate_filter)
    check_report(draws_npz, ceil_key, draws_rep, "ceil arm", candidate_filter)
    by_index = check_candidates(candidates, base, candidate_filter)
    labeled = labeled_mask(by_index, base)
    # Descriptive (amendment-1 class): the draws run's pass-1 narr
    # column vs the banked rung-(a) narr column — bit-equality holds
    # only at matched composition/device/world-size.
    narr_matches = None
    if NARR_KEY in self_npz:
        differ = (
            (draws_npz[NARR_KEY].view(np.int32) != self_npz[NARR_KEY].view(np.int32))
            .any(axis=(1, 2))
            .sum()
        )
        narr_matches = {"rows_differ": int(differ), "rows": len(core)}
    print(
        f"execution oracles GREEN ({int(labeled.sum())} labeled / "
        f"{int((~labeled).sum())} label-less rows; candidate picks "
        "byte-match offline recompute"
        + (
            f"; narr column vs banked: {narr_matches['rows_differ']} rows "
            "differ — recorded, not adjudicated"
            if narr_matches is not None
            else ""
        )
        + ")",
    )

    # ---- per-frame machinery ----
    arms = {
        "bon": np.abs(draws_npz[bon_key] - truth),
        "ceil": np.abs(draws_npz[ceil_key] - truth),
        "narr": np.abs(draws_npz[NARR_KEY] - truth),
    }
    self_err = np.abs(self_npz[SELF_KEY] - truth)
    base_frame, nvalid = bbr.frame_mae(base_err, w)
    self_frame, _ = bbr.frame_mae(self_err, w)
    keep_core = (nvalid > 0) & core
    keep_labeled = keep_core & labeled
    first_valid = valid[:, 0] & core

    def reads(arm_err: np.ndarray) -> dict:
        arm_frame, _ = bbr.frame_mae(arm_err, w)
        pooled = bbr.pooled_chunk(arm_err, core, w)
        first = bbr.pooled_first(arm_err, valid, core)
        first_delta = arm_err[first_valid, 0, :].mean(axis=1) - base_err[
            first_valid,
            0,
            :,
        ].mean(axis=1)
        flo, fhi = bbr.bootstrap_ci(first_delta)
        labeled_pool = (
            bbr.pooled_chunk(arm_err, keep_labeled, w) if keep_labeled.any() else None
        )
        return {
            "chunk_mae": round(pooled, 5),
            "first_mae": round(first, 5),
            "core": paired_read(base_frame, arm_frame, keep_core, pooled - bc),
            "labeled_subset": (
                paired_read(
                    base_frame,
                    arm_frame,
                    keep_labeled,
                    (labeled_pool - bbr.pooled_chunk(base_err, keep_labeled, w))
                    if labeled_pool is not None
                    else 0.0,
                )
                | {
                    "chunk_mae": round(labeled_pool, 5)
                    if labeled_pool is not None
                    else None,
                }
            ),
            "first_mirror": {
                "delta": round(first - bf, 5),
                "ci95": [round(flo, 5), round(fhi, 5)],
            },
            "curve": [round(v, 5) for v in step_curve(arm_err, valid, core)],
        }

    out: dict[str, Any] = {
        "candidate_filter": candidate_filter,
        "subset_mode": subset,
        "row_pairing": (
            f"subset-join on index (q4 fallback, {len(rows)} rows; baseline "
            "+ banked self re-pooled onto the subset)"
            if subset
            else "full-panel identity byte-match"
        ),
        "baseline": {"chunk_mae": round(bc, 5), "first_mae": round(bf, 5)},
        "baseline_curve": [round(v, 5) for v in step_curve(base_err, valid, core)],
        "banked_self": {
            "chunk_mae": round(bbr.pooled_chunk(self_err, core, w), 5),
            "delta_pooled": round(bbr.pooled_chunk(self_err, core, w) - bc, 5),
        },
        "narr_vs_banked_rows_differ": narr_matches,
        "noise_floor_per_frame_ci": NOISE_FLOOR,
        "arms": {name: reads(err) for name, err in arms.items()},
        "agreement": agreement_records(by_index, candidate_filter),
    }

    # read 1 head-to-head: paired (bon − self) per-frame vs the BANKED
    # self npz — the rung's pass/fail number.
    bon_frame, _ = bbr.frame_mae(arms["bon"], w)
    hh = (bon_frame - self_frame)[keep_core]
    hlo, hhi = bbr.bootstrap_ci(hh)
    out["head_to_head_bon_minus_self"] = {
        "delta_frame_mean": round(float(hh.mean()), 5),
        "ci95": [round(hlo, 5), round(hhi, 5)],
    }
    # read 2 adjudication: paired (ceil − self) on the labeled subset.
    ceil_frame, _ = bbr.frame_mae(arms["ceil"], w)
    cs = (ceil_frame - self_frame)[keep_labeled]
    clo, chi = bbr.bootstrap_ci(cs)
    out["adjudication_ceil_minus_self_labeled"] = {
        "delta_frame_mean": round(float(cs.mean()), 5),
        "ci95": [round(clo, 5), round(chi, 5)],
    }

    # read 4 summary: early/late tail means of the horizon curves.
    for name in arms:
        curve = np.array(out["arms"][name]["curve"])
        base_curve = np.array(out["baseline_curve"])
        d = curve - base_curve
        n10 = max(1, len(d) // 5)
        out["arms"][name]["horizon_delta"] = {
            "first10": round(float(d[:n10].mean()), 5),
            "last10": round(float(d[-n10:].mean()), 5),
        }

    # Expectation dispositions (pre-reg "Numbered expectations").
    d_bon = out["arms"]["bon"]["core"]["delta_pooled"]
    d_ceil = out["arms"]["ceil"]["labeled_subset"]["delta_pooled"]
    d_self = out["banked_self"]["delta_pooled"]
    falsified = not hhi < 0
    ceil_beats_self = chi < 0
    out["expectations"] = {
        "e2_ceil_below_self_ci_clear": ceil_beats_self,
        "e3_bon_between_self_and_ceil": (
            min(d_self, d_ceil) <= d_bon <= max(d_self, d_ceil)
        ),
        "e5_pick_differs_ge_20pct": (
            out["agreement"]["pick_text_differs_from_greedy"] >= 0.20
        ),
        "e6_falsified_bon_minus_self_ci_not_below_zero": falsified,
    }
    if not falsified:
        verdict = (
            "LIVE: paired (bon − self) CI95 entirely below 0 — verifier-free "
            "selection beats greedy self-conditioning at this width"
        )
    elif ceil_beats_self:
        verdict = (
            "FALSIFIED (E6), adjudication NO-SCORER: the width contains "
            "better-phase texts (ceil < self, CI clear) but self-certainty "
            "does not find them — scorer-side escalations may earn a pre-reg"
        )
    else:
        verdict = (
            "FALSIFIED (E6), adjudication NO-DIVERSITY: even the oracle "
            "ceiling does not beat self at this width — the selection "
            "family closes at N/T (a result, not a failure)"
        )
    out["verdict"] = verdict

    print(
        f"\nΔ_bon   (core, primary) {d_bon:+.4f}  "
        f"CI {out['arms']['bon']['core']['ci95']}",
    )
    print(
        f"bon − self (head-to-head) {out['head_to_head_bon_minus_self']['delta_frame_mean']:+.4f}  "
        f"CI {out['head_to_head_bon_minus_self']['ci95']}  "
        f"(noise floor ±{NOISE_FLOOR})",
    )
    print(
        f"Δ_ceil  (labeled bound)  {d_ceil:+.4f}  "
        f"CI {out['arms']['ceil']['labeled_subset']['ci95']}",
    )
    print(
        f"ceil − self (adjudication) {out['adjudication_ceil_minus_self_labeled']['delta_frame_mean']:+.4f}  "
        f"CI {out['adjudication_ceil_minus_self_labeled']['ci95']}",
    )
    print(f"verdict: {verdict}")
    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=2))
        print(f"wrote {out_path}")
    return out


# ------------------------------------------------------------------ oracle


def _fixture() -> tuple:
    """Exact-arithmetic fixture: truth 0, constant preds per region —
    every pooled delta hand-computable. 12 frames, 2 non-core, 4
    labeled; baseline err 1.0, self 0.98 (Δ −0.02), narr 1.02, bon
    0.95 (Δ −0.05; bon − self = −0.03 exactly), ceil 0.90 on labeled
    rows (else = bon)."""
    n, chunk, dims = 12, 10, 2
    truth = np.zeros((n, chunk, dims), dtype=np.float32)
    valid = np.ones((n, chunk), dtype=bool)
    core = np.ones(n, dtype=bool)
    core[10:] = False
    labeled = np.zeros(n, dtype=bool)
    labeled[:4] = True
    base_pred = np.full((n, chunk, dims), 1.0, dtype=np.float32)
    state = np.full((n, chunk, dims), 7.0, dtype=np.float32)
    base = {
        "index": np.arange(100, 100 + n, dtype=np.int64),
        "truth": truth,
        "valid": valid,
        "repo_id": np.array([f"repo{i % 3}" for i in range(n)]),
        "core": core,
        BASELINE_KEY: base_pred,
        "pred:state-copy": state,
        "pred:state-copy-norm": state + 1,
    }
    self_npz = {k: v.copy() for k, v in base.items() if k != BASELINE_KEY}
    self_npz[SELF_KEY] = np.full_like(base_pred, 0.98)
    self_npz[NARR_KEY] = np.full_like(base_pred, 1.02)
    draws_npz = {k: v.copy() for k, v in base.items() if k != BASELINE_KEY}
    draws_npz[BON_KEY] = np.full_like(base_pred, 0.95)
    ceil_pred = np.full_like(base_pred, 0.95)
    ceil_pred[labeled] = 0.90
    draws_npz[CEIL_KEY] = ceil_pred
    draws_npz[NARR_KEY] = self_npz[NARR_KEY].copy()

    # Candidates: 0 greedy (mean −2), 1 peaked (mean −4 → bon pick), 2
    # flat (mean −1). Candidate 1 also matches the true label → ceil
    # pick 1 on labeled rows.
    def row(i: int, ix: int) -> dict:
        cands = [
            {
                "text": text,
                "truncated": False,
                "chosen_logprob": [m + 0.5, m + 0.5],
                "mean_logprob": [m, m],
                "allowed_vocab": 16,
            }
            for text, m in (
                ("lower the gripper", -2.0),
                ("reach the handle", -4.0),
                ("spin in place", -1.0),
            )
        ]
        label = "reach the handle" if labeled[i] else None
        return {
            "index": ix,
            "repo_id": f"repo{i % 3}",
            "episode_index": i,
            "frame_index": 10 * i,
            "instruction": "pick up the cube",
            "true_subgoal": label,
            "greedy_subgoal": "lower the gripper",
            "candidates": cands,
            "picks": {
                "bon": 1,
                "ceil": 1 if label is not None else None,
                "likelihood": 1,
                "medoid": 0,
            },
        }

    candidates = {
        "subgoal_mode": "draws",
        "subgoal_draws": 2,
        "subgoal_temperature": 1.0,
        "selfsubgoal_force_empty": False,
        "rows": [row(i, int(ix)) for i, ix in enumerate(base["index"])],
    }

    def summaries(npz: dict, keys: tuple[str, ...]) -> list[dict]:
        truth_, valid_, core_, w_ = bbr.masks(npz)
        out = []
        for key in keys:
            err = np.abs(npz[key] - truth_)
            out.append(
                {
                    "policy": key.removeprefix("pred:"),
                    "chunk_mae": bbr.pooled_chunk(err, core_, w_),
                    "first_mae": bbr.pooled_first(err, valid_, core_),
                },
            )
        return out

    draws_rep = {
        "subgoal_mode": "draws",
        "subgoal_draws": 2,
        "subgoal_temperature": 1.0,
        "selfsubgoal_force_empty": False,
        "summaries": summaries(draws_npz, (BON_KEY, CEIL_KEY, NARR_KEY)),
    }
    return base, self_npz, draws_npz, draws_rep, candidates


def _clean_fixture() -> tuple:
    """The clean-list (rung b') transform of the base fixture: the same
    planted arrays under the filtered arm keys, filter provenance in
    dump + report, and two planted worlds — row 0: the full-list SC AND
    ceil argmax (candidate 1) is truncated, so both filtered picks move
    to candidate 0 (pre-reg oracle viii); row 1: all three candidates
    truncated → greedy fallback, recorded (oracle ix). Picks and
    eligible/fallback fields are recomputed through the frozen rule."""
    base, self_npz, draws_npz, draws_rep, candidates = _fixture()
    draws_npz[CLEAN_BON_KEY] = draws_npz.pop(BON_KEY)
    draws_npz[CLEAN_CEIL_KEY] = draws_npz.pop(CEIL_KEY)
    draws_rep = dict(draws_rep, subgoal_candidate_filter="clean")
    draws_rep["summaries"] = [
        dict(
            s,
            policy=s["policy"]
            .replace("bonsubgoal", "boncleansubgoal")
            .replace("ceilsubgoal", "ceilcleansubgoal"),
        )
        for s in draws_rep["summaries"]
    ]
    candidates = json.loads(json.dumps(candidates))
    candidates["subgoal_candidate_filter"] = "clean"
    for i, row in enumerate(candidates["rows"]):
        cands = row["candidates"]
        if i == 0:
            cands[1]["truncated"] = True
        if i == 1:
            for c in cands:
                c["truncated"] = True
        elig = eligible_indices([c["truncated"] for c in cands])
        members = set(elig)
        row["eligible"] = [j in members for j in range(len(cands))]
        row["fallback"] = all(c["truncated"] for c in cands)
        row["picks"]["bon"] = elig[
            self_certainty_pick(
                [cands[j]["mean_logprob"] for j in elig],
                cands[0]["allowed_vocab"],
            )
        ]
        label = row["true_subgoal"]
        row["picks"]["ceil"] = (
            None
            if label is None
            else elig[ceiling_pick([cands[j]["text"] for j in elig], label)]
        )
        row["picks"]["likelihood"] = elig[
            likelihood_pick([cands[j]["chosen_logprob"] for j in elig])
        ]
        row["picks"]["medoid"] = elig[medoid_pick([cands[j]["text"] for j in elig])]
    return base, self_npz, draws_npz, draws_rep, candidates


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

    base, self_npz, draws_npz, draws_rep, candidates = _fixture()
    anchor = (1.0, 1.0)

    out = analyze(base, self_npz, draws_npz, draws_rep, candidates, anchor, None)
    assert out["arms"]["bon"]["core"]["delta_pooled"] == -0.05, out
    assert out["banked_self"]["delta_pooled"] == -0.02, out
    hh = out["head_to_head_bon_minus_self"]
    assert hh["delta_frame_mean"] == -0.03 and hh["ci95"] == [-0.03, -0.03], out
    assert out["arms"]["ceil"]["labeled_subset"]["delta_pooled"] == -0.1, out
    adj = out["adjudication_ceil_minus_self_labeled"]
    assert adj["delta_frame_mean"] == -0.08 and adj["ci95"] == [-0.08, -0.08], out
    assert not out["expectations"]["e6_falsified_bon_minus_self_ci_not_below_zero"]
    assert out["expectations"]["e2_ceil_below_self_ci_clear"], out
    assert out["expectations"]["e3_bon_between_self_and_ceil"], out
    # pick text ("reach the handle") differs from greedy on every row.
    assert out["agreement"]["pick_text_differs_from_greedy"] == 1.0, out
    assert out["agreement"]["unique_candidates"]["frac_ge_2"] == 1.0, out
    assert "LIVE" in out["verdict"], out
    assert out["arms"]["bon"]["horizon_delta"] == {"first10": -0.05, "last10": -0.05}
    print("  exact-arithmetic reads OK (planted deltas reproduced)")

    # Degenerate: bon byte-equals the banked self arm → head-to-head
    # exactly 0 with CI [0, 0] → the falsifier fires; ceil == self on
    # labeled rows → NO-DIVERSITY adjudication.
    degen = {k: v.copy() for k, v in draws_npz.items()}
    degen[BON_KEY] = self_npz[SELF_KEY].copy()
    degen[CEIL_KEY] = self_npz[SELF_KEY].copy()
    degen_rep = dict(
        draws_rep,
        summaries=[
            {
                "policy": key.removeprefix("pred:"),
                "chunk_mae": 0.98,
                "first_mae": 0.98,
            }
            for key in (BON_KEY, CEIL_KEY)
        ],
    )
    out2 = analyze(base, self_npz, degen, degen_rep, candidates, anchor, None)
    hh2 = out2["head_to_head_bon_minus_self"]
    assert hh2["delta_frame_mean"] == 0.0 and hh2["ci95"] == [0.0, 0.0], out2
    assert out2["expectations"]["e6_falsified_bon_minus_self_ci_not_below_zero"]
    assert "NO-DIVERSITY" in out2["verdict"], out2
    print("  degenerate arm OK (delta 0, CI [0,0], E6 fires, no-diversity routes)")

    # No-scorer routing: bon == self but ceil clearly better.
    noscorer = {k: v.copy() for k, v in draws_npz.items()}
    noscorer[BON_KEY] = self_npz[SELF_KEY].copy()
    noscorer_rep = dict(
        draws_rep,
        summaries=[
            {
                "policy": BON_KEY.removeprefix("pred:"),
                "chunk_mae": 0.98,
                "first_mae": 0.98,
            },
            draws_rep["summaries"][1],
        ],
    )
    out3 = analyze(base, self_npz, noscorer, noscorer_rep, candidates, anchor, None)
    assert out3["expectations"]["e6_falsified_bon_minus_self_ci_not_below_zero"]
    assert "NO-SCORER" in out3["verdict"], out3
    print("  no-scorer branch OK (ceil clear of self, scorer-side routing)")

    expect_exit(
        lambda: analyze(
            base,
            self_npz,
            draws_npz,
            draws_rep,
            candidates,
            (0.5, 1.0),
            None,
        ),
        "banked anchor",
        "anchor mismatch",
    )
    mut = {k: v.copy() for k, v in draws_npz.items()}
    mut["truth"][0] += 1.0
    expect_exit(
        lambda: analyze(base, self_npz, mut, draws_rep, candidates, anchor, None),
        "pairing broken",
        "identity drift",
    )
    mut = {k: v.copy() for k, v in draws_npz.items()}
    mut["pred:state-copy"][1] += 1.0
    expect_exit(
        lambda: analyze(base, self_npz, mut, draws_rep, candidates, anchor, None),
        "byte-match the banked panel",
        "state-copy drift",
    )
    mut = {k: v.copy() for k, v in draws_npz.items()}
    del mut[CEIL_KEY]
    expect_exit(
        lambda: analyze(base, self_npz, mut, draws_rep, candidates, anchor, None),
        "does not carry the mode",
        "missing arm key",
    )
    mut = {k: v.copy() for k, v in draws_npz.items()}
    mut[BASELINE_KEY] = base[BASELINE_KEY].copy()
    expect_exit(
        lambda: analyze(base, self_npz, mut, draws_rep, candidates, anchor, None),
        "never re-run",
        "bare-column refusal",
    )
    bad_rep = dict(draws_rep)
    bad_rep["summaries"] = [
        dict(draws_rep["summaries"][0], chunk_mae=2.0),
        draws_rep["summaries"][1],
    ]
    expect_exit(
        lambda: analyze(base, self_npz, draws_npz, bad_rep, candidates, anchor, None),
        "plan/scoring drift",
        "report drift",
    )
    expect_exit(
        lambda: analyze(
            base,
            self_npz,
            draws_npz,
            dict(draws_rep, subgoal_mode="self"),
            candidates,
            anchor,
            None,
        ),
        "wrong run",
        "mode provenance",
    )
    expect_exit(
        lambda: analyze(
            base,
            self_npz,
            draws_npz,
            dict(draws_rep, selfsubgoal_force_empty=True),
            candidates,
            anchor,
            None,
        ),
        "never an arm read",
        "force-empty refusal",
    )
    expect_exit(
        lambda: analyze(
            base,
            self_npz,
            draws_npz,
            draws_rep,
            dict(candidates, rows=candidates["rows"][:5]),
            anchor,
            None,
        ),
        "does not cover the panel",
        "dump coverage",
    )
    bad_pick = json.loads(json.dumps(candidates))
    bad_pick["rows"][0]["picks"]["bon"] = 2
    expect_exit(
        lambda: analyze(base, self_npz, draws_npz, draws_rep, bad_pick, anchor, None),
        "scorer drift",
        "live-pick mismatch",
    )

    # ---- rung (b') clean-list mode (pre-reg …-cleanlist oracles) ----
    cbase, cself, cdraws, crep, ccands = _clean_fixture()
    row0 = ccands["rows"][0]["candidates"]
    # Oracle viii planted world holds: the full-list argmax IS truncated
    # and the filtered pick differs — on BOTH scorers.
    full_bon = self_certainty_pick(
        [c["mean_logprob"] for c in row0],
        row0[0]["allowed_vocab"],
    )
    full_ceil = ceiling_pick(
        [c["text"] for c in row0],
        ccands["rows"][0]["true_subgoal"],
    )
    assert full_bon == 1 and row0[1]["truncated"], ccands["rows"][0]
    assert ccands["rows"][0]["picks"]["bon"] == 0 != full_bon
    assert full_ceil == 1 and ccands["rows"][0]["picks"]["ceil"] == 0 != full_ceil
    # Oracle ix planted world holds: all-truncated → greedy fallback.
    assert ccands["rows"][1]["fallback"] is True
    assert ccands["rows"][1]["picks"]["bon"] == 0
    assert ccands["rows"][1]["picks"]["ceil"] == 0
    print("  clean fixture planted worlds OK (filter binds both scorers; fallback)")

    cout = analyze(
        cbase,
        cself,
        cdraws,
        crep,
        ccands,
        anchor,
        None,
        candidate_filter="clean",
    )
    assert cout["candidate_filter"] == "clean", cout
    # Same planted arrays → the exact-arithmetic reads carry verbatim.
    assert cout["arms"]["bon"]["core"]["delta_pooled"] == -0.05, cout
    chh = cout["head_to_head_bon_minus_self"]
    assert chh["delta_frame_mean"] == -0.03 and chh["ci95"] == [-0.03, -0.03], cout
    agree = cout["agreement"]
    assert agree["fallback_rows"] == 1, agree
    assert agree["eligible_list_size"]["hist"] == {"1": 1, "2": 1, "3": 10}, agree
    # Rows 0/1 pick candidate 0 (greedy text) under the filter; the
    # other 10 keep pick 1 → 10/12 differ from greedy.
    assert agree["pick_text_differs_from_greedy"] == round(10 / 12, 5), agree
    assert agree["unique_candidates"]["frac_ge_2"] == round(11 / 12, 5), agree
    print("  clean-mode reads OK (filtered agreement records, deltas carried)")

    expect_exit(
        lambda: analyze(
            cbase,
            cself,
            cdraws,
            dict(crep, subgoal_candidate_filter=None),
            ccands,
            anchor,
            None,
            candidate_filter="clean",
        ),
        "candidate-filter provenance",
        "report filter provenance",
    )
    unfiltered_dump = json.loads(json.dumps(ccands))
    del unfiltered_dump["subgoal_candidate_filter"]
    expect_exit(
        lambda: analyze(
            cbase,
            cself,
            cdraws,
            crep,
            unfiltered_dump,
            anchor,
            None,
            candidate_filter="clean",
        ),
        "candidate-filter provenance",
        "dump filter provenance",
    )
    stray = {k: v.copy() for k, v in cdraws.items()}
    stray[BON_KEY] = stray[CLEAN_BON_KEY].copy()
    expect_exit(
        lambda: analyze(
            cbase,
            cself,
            stray,
            crep,
            ccands,
            anchor,
            None,
            candidate_filter="clean",
        ),
        "wrong candidate-filter convention",
        "cross-convention arm key",
    )
    mut_c = json.loads(json.dumps(ccands))
    mut_c["rows"][0]["eligible"] = [True, True, True]
    expect_exit(
        lambda: analyze(
            cbase,
            cself,
            cdraws,
            crep,
            mut_c,
            anchor,
            None,
            candidate_filter="clean",
        ),
        "eligible-list drift",
        "eligible flags drift",
    )
    mut_c = json.loads(json.dumps(ccands))
    mut_c["rows"][1]["fallback"] = False
    expect_exit(
        lambda: analyze(
            cbase,
            cself,
            cdraws,
            crep,
            mut_c,
            anchor,
            None,
            candidate_filter="clean",
        ),
        "fallback recording drift",
        "fallback flag drift",
    )
    # The planted-world regression: a scorer path that FORGOT the filter
    # would dump the full-list argmax on row 0 — must abort.
    mut_c = json.loads(json.dumps(ccands))
    mut_c["rows"][0]["picks"]["bon"] = full_bon
    expect_exit(
        lambda: analyze(
            cbase,
            cself,
            cdraws,
            crep,
            mut_c,
            anchor,
            None,
            candidate_filter="clean",
        ),
        "scorer drift",
        "unfiltered-pick regression (filter binds)",
    )

    # ---- q4-shaped subset slice (the draws10/energy oracle-(f) style) ----
    # Slice the clean fixture's draws artifacts to 8 of 12 rows (keeps
    # both planted worlds, all 4 labeled rows and a non-core row); the
    # full-panel base/self join + re-pool must reproduce the sliced
    # direct computation — planted constants make every delta carry.
    keep = np.array([0, 1, 2, 3, 5, 7, 8, 10])
    sdraws = {k: v[keep] for k, v in cdraws.items()}
    scands = json.loads(json.dumps(ccands))
    scands["rows"] = [scands["rows"][i] for i in keep]

    def _summaries(npz: dict, keys: tuple[str, ...]) -> list[dict]:
        t, v, c, w_ = bbr.masks(npz)
        return [
            {
                "policy": key.removeprefix("pred:"),
                "chunk_mae": bbr.pooled_chunk(np.abs(npz[key] - t), c, w_),
                "first_mae": bbr.pooled_first(np.abs(npz[key] - t), v, c),
            }
            for key in keys
        ]

    srep = dict(
        crep,
        summaries=_summaries(sdraws, (CLEAN_BON_KEY, CLEAN_CEIL_KEY, NARR_KEY)),
    )
    sout = analyze(
        cbase,
        cself,
        sdraws,
        srep,
        scands,
        anchor,
        None,
        candidate_filter="clean",
    )
    assert sout["subset_mode"] is True, sout
    assert "subset-join" in sout["row_pairing"], sout
    # Constant planted errors ⇒ the exact-arithmetic reads carry onto
    # the joined subset verbatim.
    assert sout["arms"]["bon"]["core"]["delta_pooled"] == -0.05, sout
    shh = sout["head_to_head_bon_minus_self"]
    assert shh["delta_frame_mean"] == -0.03 and shh["ci95"] == [-0.03, -0.03], sout
    assert sout["arms"]["ceil"]["labeled_subset"]["delta_pooled"] == -0.1, sout
    sadj = sout["adjudication_ceil_minus_self_labeled"]
    assert sadj["delta_frame_mean"] == -0.08 and sadj["ci95"] == [-0.08, -0.08], sout
    sagree = sout["agreement"]
    assert sagree["n_frames"] == 8, sagree
    assert sagree["fallback_rows"] == 1, sagree
    assert sagree["eligible_list_size"]["hist"] == {"1": 1, "2": 1, "3": 6}, sagree
    # Rows 0/1 pick the greedy text under the filter ⇒ 6/8 differ.
    assert sagree["pick_text_differs_from_greedy"] == 0.75, sagree
    print("  q4-style subset join OK (re-pooled reads reproduce the slice)")

    dup = {k: v.copy() for k, v in sdraws.items()}
    dup["index"][1] = dup["index"][0]
    dup["truth"][1] = dup["truth"][0]
    dup["valid"][1] = dup["valid"][0]
    dup["repo_id"][1] = dup["repo_id"][0]
    dup["core"][1] = dup["core"][0]
    expect_exit(
        lambda: analyze(
            cbase,
            cself,
            dup,
            srep,
            scands,
            anchor,
            None,
            candidate_filter="clean",
        ),
        "duplicate index",
        "subset duplicate-index refusal",
    )
    foreign = {k: v.copy() for k, v in sdraws.items()}
    foreign["index"][0] = 9999
    expect_exit(
        lambda: analyze(
            cbase,
            cself,
            foreign,
            srep,
            scands,
            anchor,
            None,
            candidate_filter="clean",
        ),
        "absent from the baseline panel",
        "subset foreign-index refusal",
    )
    drift = {k: v.copy() for k, v in sdraws.items()}
    drift["truth"][0] += 1.0
    expect_exit(
        lambda: analyze(
            cbase,
            cself,
            drift,
            srep,
            scands,
            anchor,
            None,
            candidate_filter="clean",
        ),
        "subset rows disagree",
        "subset identity-drift refusal",
    )
    print("oracle: ALL branches OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--baseline-stem", default=BASE_STEM)
    parser.add_argument("--self-stem", default=SELF_STEM)
    parser.add_argument("--draws-stem", default=DRAWS_STEM)
    parser.add_argument(
        "--candidate-filter",
        choices=["clean"],
        default=None,
        help="rung (b') clean-list read: expect the _boncleansubgoal/"
        "_ceilcleansubgoal arms, recompute every pick over the eligible "
        "(non-truncated) list, and require filter provenance in the "
        "dump + report; default stems/out switch to the cleanlist names",
    )
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--out", default=OUT_DEFAULT)
    args = parser.parse_args()
    if args.oracle:
        oracle()
        return
    if args.candidate_filter == "clean":
        if args.draws_stem == DRAWS_STEM:
            args.draws_stem = CLEAN_DRAWS_STEM
        if args.out == OUT_DEFAULT:
            args.out = CLEAN_OUT_DEFAULT
    candidates_path = args.candidates or f"{args.draws_stem}_candidates.json"
    analyze(
        _load_npz(f"{args.baseline_stem}.npz"),
        _load_npz(f"{args.self_stem}.npz"),
        _load_npz(f"{args.draws_stem}.npz"),
        json.loads(Path(f"{args.draws_stem}.json").read_text()),
        json.loads(Path(candidates_path).read_text()),
        bbr.ANCHORS["ar"],
        args.out,
        candidate_filter=args.candidate_filter,
    )


if __name__ == "__main__":
    main()
