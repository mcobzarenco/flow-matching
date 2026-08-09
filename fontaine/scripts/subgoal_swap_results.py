"""Frozen reads for the #6 subgoal-swap content read.

One command over the swap-arm artifacts (pre-reg
2026-08-09-prereg-subgoal-swap.md, "Frozen reads" — semantics frozen
there, this file only mechanizes them):

  1. PRIMARY   Δ_swap = chunk_mae(swap) − banked baseline, paired
     per-frame seeded bootstrap CI (seed 0, 10,000 resamples) on core
     frames — the rung-(a) machinery verbatim; labeled-subset value
     quoted beside it (labeled = panel rows carrying a swap record);
  2. CONTRAST  swap vs oracle on the same frames, paired per-frame
     (the banked oracle-arm npz — never re-run);
  3. HORIZON   mirror (record-only): per-step curves + first-10/last-10
     delta means — the −0.464-shaped late-horizon signature is the
     content-read's fingerprint, a format effect should be flat;
     first_mae mirrors of 1–2 ride along.
  4. EXECUTION ORACLES (each failure a hard abort): banked baseline
     anchor 5.8026/2.1431 reproduced; identity columns (incl. the
     episode/frame triple) match across all npzs; state-copy rows
     byte-match; the swap report records subgoal_mode=oracle +
     subgoal_swap_seed=0 + identity=False (an identity-mode npz is
     NEVER a read); npz-recomputed pooled values reproduce the report;
     no bare baseline column (the baseline must never re-run); the
     swap dump covers the panel exactly and its swapped/empty counts
     recompute.

Interpretation is the pre-reg's frozen 3-row table, adjudicated
mechanically from the CIs (CI-contains-0 = "≈"); anything not matching
a row verbatim is reported as MIXED/record-only, no decision.

``--oracle`` runs the pre-data selftest: exact-arithmetic fixtures and
every abort branch fired.
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

BASE_STEM = "reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2"
ORACLE_STEM = f"{BASE_STEM}_oraclesubgoal"
SWAP_STEM = f"{BASE_STEM}_swapsubgoal"
OUT_DEFAULT = "reports/analysis__subgoal_swap_ar100k_k4l2.json"

BASELINE_KEY = "pred:bijou@100000"
ORACLE_KEY = "pred:bijou@100000_oraclesubgoal"
SWAP_KEY = "pred:bijou@100000_swapsubgoal"
# the banked baseline npz predates the episode/frame columns, so the
# base pairing runs on its own identity surface; the triple is paired
# oracle-vs-swap (both carry it) and keys the dump join
IDENTITY_KEYS = ("index", "truth", "valid", "repo_id", "core")
TRIPLE_KEYS = ("repo_id", "episode_index", "frame_index")
STATE_KEYS = ("pred:state-copy", "pred:state-copy-norm")
SUMMARY_TOL = 5e-3
# banked rung-(a) context, printed beside the reads (labeled subset):
ORACLE_BANKED = {"delta": -0.290, "ci95": [-0.331, -0.225]}


def _load_npz(path: str | Path) -> dict:
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def _bytes_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def check_pairing(
    base: dict,
    probe: dict,
    label: str,
    keys: tuple[str, ...] = IDENTITY_KEYS,
) -> None:
    """Full-panel runs share the one plan: identity columns match."""
    for key in keys:
        if key not in probe:
            sys.exit(f"{label}: identity column {key} missing — stop")
        if not np.array_equal(base[key], probe[key]):
            sys.exit(f"{label}: panel pairing broken on {key} — stop")


def check_state_rows(base: dict, probe: dict, label: str) -> None:
    for key in STATE_KEYS:
        if key not in probe:
            sys.exit(f"{label}: {key} missing from the npz")
        if not _bytes_equal(base[key], probe[key]):
            sys.exit(
                f"{label}: {key} rows do NOT byte-match the banked panel — "
                "execution drift, stop",
            )


def check_swap_report(npz: dict, report: dict) -> None:
    if report.get("subgoal_mode") != "oracle":
        sys.exit(
            f"swap arm: report subgoal_mode {report.get('subgoal_mode')!r} != "
            "'oracle' — the swap arm rides the oracle path, wrong run, stop",
        )
    if report.get("subgoal_swap_seed") != 0:
        sys.exit(
            f"swap arm: subgoal_swap_seed {report.get('subgoal_swap_seed')!r} "
            "!= 0 — not the pre-registered arm, stop",
        )
    if report.get("subgoal_swap_identity"):
        sys.exit(
            "swap arm: subgoal_swap_identity is set — an identity-mode run "
            "is the oracle-(ii) check, never a read — stop",
        )
    truth, valid, core, w = bbr.masks(npz)
    err = np.abs(npz[SWAP_KEY] - truth)
    gc = bbr.pooled_chunk(err, core, w)
    gf = bbr.pooled_first(err, valid, core)
    policy = SWAP_KEY.removeprefix("pred:")
    summ = [s for s in report.get("summaries", []) if s.get("policy") == policy]
    if len(summ) != 1:
        sys.exit(
            f"swap arm: report has {len(summ)} summaries for {policy!r} "
            f"(have: {[s.get('policy') for s in report.get('summaries', [])]})",
        )
    wc, wf = summ[0]["chunk_mae"], summ[0]["first_mae"]
    if abs(gc - wc) >= SUMMARY_TOL or abs(gf - wf) >= SUMMARY_TOL:
        sys.exit(
            f"swap arm: npz-recomputed chunk/first {gc:.4f}/{gf:.4f} do not "
            f"reproduce the report's {wc:.4f}/{wf:.4f} — plan/scoring "
            "drift, stop",
        )
    print(f"swap arm: report cross-check OK (chunk {gc:.4f} first {gf:.4f})")


def swap_masks(dump: dict, panel: dict) -> tuple[np.ndarray, np.ndarray]:
    """(labeled, swapped) panel masks from the swap dump, abort-grade.

    labeled = rows with a swap record (the slot rendered SOMETHING —
    donor text or the by-design empty slot); swapped = donor assigned.
    Every dump record must land on exactly one panel row.
    """
    if dump.get("subgoal_swap_identity"):
        sys.exit("swap dump: identity-mode dump is never a read — stop")
    rows = dump.get("rows")
    if not isinstance(rows, list) or not rows:
        sys.exit("swap dump has no 'rows' — not a --dump-subgoal-swaps file")
    by_triple: dict[tuple[str, int, int], dict] = {}
    for r in rows:
        triple = (str(r["repo_id"]), int(r["episode_index"]), int(r["frame_index"]))
        if triple in by_triple:
            sys.exit(f"swap dump: duplicate record for {triple} — stop")
        by_triple[triple] = r
    labeled = np.zeros(len(panel["index"]), dtype=bool)
    swapped = np.zeros(len(panel["index"]), dtype=bool)
    hit = 0
    for i in range(len(panel["index"])):
        triple = (
            str(panel["repo_id"][i]),
            int(panel["episode_index"][i]),
            int(panel["frame_index"][i]),
        )
        rec = by_triple.get(triple)
        if rec is None:
            continue
        hit += 1
        labeled[i] = True
        swapped[i] = rec["donor_episode_index"] is not None
    if hit != len(rows):
        sys.exit(
            f"swap dump: {len(rows) - hit} record(s) match no panel row — "
            "the dump does not cover the panel, stop",
        )
    n_swap, n_empty = int(swapped.sum()), int(labeled.sum() - swapped.sum())
    if dump.get("swapped") != n_swap or dump.get("empty_rendered") != n_empty:
        sys.exit(
            f"swap dump: header counts swapped={dump.get('swapped')}/"
            f"empty_rendered={dump.get('empty_rendered')} do not recompute "
            f"({n_swap}/{n_empty}) — stop",
        )
    return labeled, swapped


def paired_read(
    base_frame: np.ndarray,
    arm_frame: np.ndarray,
    keep: np.ndarray,
    pooled_delta: float | None,
) -> dict:
    deltas = (arm_frame - base_frame)[keep]
    lo, hi = bbr.bootstrap_ci(deltas)
    return {
        "delta_pooled": round(pooled_delta, 5) if pooled_delta is not None else None,
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


def adjudicate(swap_core: dict, swap_vs_oracle: dict) -> tuple[str, str]:
    """The frozen 3-row table, mechanical: CI-contains-0 = '≈'.

    swap_vs_oracle deltas are swap − oracle in MAE, so 'oracle ≪ swap'
    (oracle much better) = that CI entirely ABOVE 0.
    """
    lo, hi = swap_core["ci95"]
    olo, ohi = swap_vs_oracle["ci95"]
    swap_zero = lo <= 0 <= hi
    swap_neg = hi < 0
    swap_pos = lo > 0
    oracle_better = olo > 0
    same_as_oracle = olo <= 0 <= ohi
    if swap_pos:
        return (
            "row3",
            (
                "Δ_swap > 0 (hurts): content is consumed and TRUSTED — "
                "strongest pro-scorer case"
            ),
        )
    if swap_zero and oracle_better:
        return (
            "row1",
            (
                "Δ_swap ≈ 0 and oracle ≪ swap: content is consumed; wrong "
                "content is ignored/neutral — learned-scorer escalations "
                "stay coherent"
            ),
        )
    if swap_neg and same_as_oracle:
        return (
            "row2",
            (
                "Δ_swap ≈ Δ_oracle < 0: format/prior effect — any plausible "
                "words help; the scorer ladder is chasing a mirage, "
                "deprioritize #6 escalations toward the future-latent family"
            ),
        )
    return (
        "mixed",
        (
            "MIXED/intermediate outcome — reported against the table "
            "without a decision (record-only fallback, pre-reg)"
        ),
    )


def analyze(
    base: dict,
    oracle_npz: dict,
    swap_npz: dict,
    swap_rep: dict,
    dump: dict,
    anchor: tuple[float, float],
    out_path: str | None,
) -> dict:
    # ---- read 4 first: execution oracles gate every number below ----
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
        (ORACLE_KEY, oracle_npz, "oracle arm"),
        (SWAP_KEY, swap_npz, "swap arm"),
    ):
        if key not in npz:
            sys.exit(
                f"{label}: {key} missing — policy name does not carry the mode, stop",
            )
    if BASELINE_KEY in swap_npz:
        sys.exit(
            "swap arm npz carries a bare bijou column — the baseline must "
            "never re-run, stop",
        )
    check_pairing(base, oracle_npz, "oracle arm")
    check_pairing(base, swap_npz, "swap arm")
    # both arm npzs carry the episode/frame triple — pair them on it
    check_pairing(oracle_npz, swap_npz, "swap-vs-oracle triple", TRIPLE_KEYS)
    check_state_rows(base, oracle_npz, "oracle arm")
    check_state_rows(base, swap_npz, "swap arm")
    check_swap_report(swap_npz, swap_rep)
    labeled, swapped = swap_masks(dump, swap_npz)
    print(
        f"execution oracles GREEN ({int(labeled.sum())} labeled rows: "
        f"{int(swapped.sum())} swapped + "
        f"{int(labeled.sum() - swapped.sum())} empty-rendered; "
        f"{int((~labeled).sum())} label-less)",
    )

    # ---- per-frame machinery shared by the reads ----
    swap_err = np.abs(swap_npz[SWAP_KEY] - truth)
    oracle_err = np.abs(oracle_npz[ORACLE_KEY] - truth)
    base_frame, nvalid = bbr.frame_mae(base_err, w)
    swap_frame, _ = bbr.frame_mae(swap_err, w)
    oracle_frame, _ = bbr.frame_mae(oracle_err, w)
    keep_core = (nvalid > 0) & core
    keep_labeled = keep_core & labeled
    first_valid = valid[:, 0] & core

    sc = bbr.pooled_chunk(swap_err, core, w)
    sf = bbr.pooled_first(swap_err, valid, core)
    out: dict[str, Any] = {
        "baseline": {"chunk_mae": round(bc, 5), "first_mae": round(bf, 5)},
        "swap": {"chunk_mae": round(sc, 5), "first_mae": round(sf, 5)},
        "oracle_banked_context": ORACLE_BANKED,
        "n_rows": {
            "labeled": int(labeled.sum()),
            "swapped": int(swapped.sum()),
            "empty_rendered": int(labeled.sum() - swapped.sum()),
        },
    }

    # read 1: Δ_swap vs banked baseline
    out["delta_swap"] = {
        "core": paired_read(base_frame, swap_frame, keep_core, sc - bc),
        "labeled_subset": paired_read(
            base_frame,
            swap_frame,
            keep_labeled,
            bbr.pooled_chunk(swap_err, keep_labeled, w)
            - bbr.pooled_chunk(base_err, keep_labeled, w),
        ),
    }
    first_delta = swap_err[first_valid, 0, :].mean(axis=1) - base_err[
        first_valid,
        0,
        :,
    ].mean(axis=1)
    flo, fhi = bbr.bootstrap_ci(first_delta)
    out["delta_swap"]["first_mirror"] = {
        "delta": round(sf - bf, 5),
        "ci95": [round(flo, 5), round(fhi, 5)],
    }

    # read 2: swap vs oracle, paired on the same frames
    out["swap_vs_oracle"] = {
        "core": paired_read(oracle_frame, swap_frame, keep_core, None),
        "labeled_subset": paired_read(oracle_frame, swap_frame, keep_labeled, None),
    }

    # read 3: horizon mirror (record-only)
    base_curve = np.array(step_curve(base_err, valid, core))
    out["curves"] = {
        "baseline": [round(v, 5) for v in base_curve.tolist()],
        "swap": [round(v, 5) for v in step_curve(swap_err, valid, core)],
        "oracle": [round(v, 5) for v in step_curve(oracle_err, valid, core)],
    }
    n10 = max(1, len(base_curve) // 5)
    for name, err in (("swap", swap_err), ("oracle", oracle_err)):
        d = np.array(step_curve(err, valid, core)) - base_curve
        out[f"horizon_delta_{name}"] = {
            "first10": round(float(d[:n10].mean()), 5),
            "last10": round(float(d[-n10:].mean()), 5),
        }

    # frozen 3-row table, adjudicated on the labeled subset (the rows
    # where the slot content actually differs between the three arms)
    row, reading = adjudicate(
        out["delta_swap"]["labeled_subset"],
        out["swap_vs_oracle"]["labeled_subset"],
    )
    out["table"] = {"row": row, "reading": reading}

    d = out["delta_swap"]
    v = out["swap_vs_oracle"]
    print(
        f"\nΔ_swap (core, primary)     {d['core']['delta_pooled']:+.4f}"
        f"  CI {d['core']['ci95']}",
    )
    print(
        f"Δ_swap (labeled subset)    {d['labeled_subset']['delta_pooled']:+.4f}"
        f"  CI {d['labeled_subset']['ci95']}",
    )
    print(
        f"swap − oracle (labeled)    {v['labeled_subset']['delta_frame_mean']:+.4f}"
        f"  CI {v['labeled_subset']['ci95']}",
    )
    print(
        f"Δ_oracle banked context    {ORACLE_BANKED['delta']:+.4f}"
        f"  CI {ORACLE_BANKED['ci95']}",
    )
    print(
        f"horizon (swap)  first10 {out['horizon_delta_swap']['first10']:+.4f}"
        f"  last10 {out['horizon_delta_swap']['last10']:+.4f}",
    )
    print(f"table: {row} — {reading}")
    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=2))
        print(f"wrote {out_path}")
    return out


# ------------------------------------------------------------------ oracle


def _fixture() -> tuple:
    """Exact-arithmetic fixture: truth 0 everywhere, constant preds per
    region, so every pooled delta is hand-computable."""
    n, chunk, dims = 12, 10, 2
    truth = np.zeros((n, chunk, dims), dtype=np.float32)
    valid = np.ones((n, chunk), dtype=bool)
    core = np.ones(n, dtype=bool)
    core[10:] = False  # 2 non-core rows must not move any number
    labeled = np.zeros(n, dtype=bool)
    labeled[:4] = True
    base_pred = np.full((n, chunk, dims), 1.0, dtype=np.float32)
    state = np.full((n, chunk, dims), 7.0, dtype=np.float32)
    base = {
        "index": np.arange(100, 100 + n, dtype=np.int64),
        "truth": truth,
        "valid": valid,
        "repo_id": np.array([f"repo{i % 3}" for i in range(n)]),
        "episode_index": np.arange(n, dtype=np.int64) // 2,
        "frame_index": np.arange(n, dtype=np.int64) * 5,
        "core": core,
        BASELINE_KEY: base_pred,
        "pred:state-copy": state,
        "pred:state-copy-norm": state + 1,
    }
    # oracle arm: labeled rows err 0.8 (-0.2); swap arm: labeled rows
    # err 0.9 (-0.1, so swap-vs-oracle = +0.1 on labeled)
    oracle_npz = {k: v.copy() for k, v in base.items() if k != BASELINE_KEY}
    opred = base_pred.copy()
    opred[labeled] = 0.8
    oracle_npz[ORACLE_KEY] = opred
    swap_npz = {k: v.copy() for k, v in base.items() if k != BASELINE_KEY}
    spred = base_pred.copy()
    spred[labeled] = 0.9
    swap_npz[SWAP_KEY] = spred
    # dump: 4 labeled rows, first 3 swapped, 1 empty-rendered
    dump = {
        "subgoal_swap_seed": 0,
        "subgoal_swap_identity": False,
        "swapped": 3,
        "empty_rendered": 1,
        "rows": [
            {
                "repo_id": str(base["repo_id"][i]),
                "episode_index": int(base["episode_index"][i]),
                "frame_index": int(base["frame_index"][i]),
                "true_subgoal": "reach the handle",
                "donor_episode_index": (int(base["episode_index"][i]) + 1)
                if i < 3
                else None,
                "rendered_subgoal": "open the drawer" if i < 3 else "",
            }
            for i in range(4)
        ],
    }
    truth_, valid_, core_, w_ = bbr.masks(swap_npz)
    err = np.abs(swap_npz[SWAP_KEY] - truth_)
    rep = {
        "subgoal_mode": "oracle",
        "subgoal_swap_seed": 0,
        "subgoal_swap_identity": False,
        "summaries": [
            {
                "policy": SWAP_KEY.removeprefix("pred:"),
                "chunk_mae": bbr.pooled_chunk(err, core_, w_),
                "first_mae": bbr.pooled_first(err, valid_, core_),
            },
        ],
    }
    return base, oracle_npz, swap_npz, rep, dump


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

    base, oracle_npz, swap_npz, rep, dump = _fixture()
    anchor = (1.0, 1.0)  # constant preds: pooled == first == 1.0 exactly

    out = analyze(base, oracle_npz, swap_npz, rep, dump, anchor, None)
    assert out["delta_swap"]["labeled_subset"]["delta_pooled"] == -0.1, out
    # core-wide: 4 labeled of 10 core rows at -0.1 -> -0.04
    assert out["delta_swap"]["core"]["delta_pooled"] == -0.04, out
    assert out["swap_vs_oracle"]["labeled_subset"]["delta_frame_mean"] == 0.1, out
    assert out["n_rows"] == {"labeled": 4, "swapped": 3, "empty_rendered": 1}, out
    # constant errors: horizon delta flat at the pooled value
    assert out["horizon_delta_swap"] == {"first10": -0.04, "last10": -0.04}, out
    # planted geometry: swap < 0 with swap-vs-oracle CI entirely > 0 -> mixed
    assert out["table"]["row"] == "mixed", out
    print("  exact-arithmetic reads OK (planted deltas reproduced)")

    # degenerate swap == baseline -> delta 0 CI [0,0]; oracle better -> row1
    degen = {k: v.copy() for k, v in swap_npz.items()}
    degen[SWAP_KEY] = base[BASELINE_KEY].copy()
    rep_degen = dict(
        rep,
        summaries=[
            dict(rep["summaries"][0], chunk_mae=1.0, first_mae=1.0),
        ],
    )
    out2 = analyze(base, oracle_npz, degen, rep_degen, dump, anchor, None)
    assert out2["delta_swap"]["core"]["delta_pooled"] == 0.0, out2
    assert out2["delta_swap"]["core"]["ci95"] == [0.0, 0.0], out2
    assert out2["table"]["row"] == "row1", out2
    print("  degenerate arm OK (delta 0, CI [0,0], table row1)")

    # adjudicator geometry, direct: row2 and row3
    assert (
        adjudicate(
            {"ci95": [-0.3, -0.1]},
            {"ci95": [-0.05, 0.05]},
        )[0]
        == "row2"
    )
    assert adjudicate({"ci95": [0.1, 0.3]}, {"ci95": [0.1, 0.3]})[0] == "row3"
    print("  adjudicator geometry OK (row2/row3 fire on planted CIs)")

    expect_exit(
        lambda: analyze(base, oracle_npz, swap_npz, rep, dump, (0.5, 1.0), None),
        "banked anchor",
        "anchor mismatch",
    )
    mut = {k: v.copy() for k, v in swap_npz.items()}
    mut["truth"][0] += 1.0
    expect_exit(
        lambda: analyze(base, oracle_npz, mut, rep, dump, anchor, None),
        "pairing broken",
        "identity drift",
    )
    mut = {k: v.copy() for k, v in swap_npz.items()}
    mut["pred:state-copy"][1] += 1.0
    expect_exit(
        lambda: analyze(base, oracle_npz, mut, rep, dump, anchor, None),
        "byte-match the banked panel",
        "state-copy drift",
    )
    expect_exit(
        lambda: analyze(
            base,
            oracle_npz,
            swap_npz,
            dict(rep, subgoal_swap_identity=True),
            dump,
            anchor,
            None,
        ),
        "never a read",
        "identity-run refusal",
    )
    expect_exit(
        lambda: analyze(
            base,
            oracle_npz,
            swap_npz,
            dict(rep, subgoal_swap_seed=7),
            dump,
            anchor,
            None,
        ),
        "not the pre-registered arm",
        "wrong seed",
    )
    bad_rep = dict(rep, summaries=[dict(rep["summaries"][0], chunk_mae=2.0)])
    expect_exit(
        lambda: analyze(base, oracle_npz, swap_npz, bad_rep, dump, anchor, None),
        "plan/scoring drift",
        "report drift",
    )
    mut = {k: v.copy() for k, v in swap_npz.items()}
    mut[BASELINE_KEY] = base[BASELINE_KEY].copy()
    expect_exit(
        lambda: analyze(base, oracle_npz, mut, rep, dump, anchor, None),
        "never re-run",
        "bare-column refusal",
    )
    bad_dump = dict(dump, rows=[*dump["rows"], dict(dump["rows"][0], repo_id="ghost")])
    expect_exit(
        lambda: analyze(base, oracle_npz, swap_npz, rep, bad_dump, anchor, None),
        "does not cover the panel",
        "dump coverage",
    )
    bad_dump = dict(dump, swapped=99)
    expect_exit(
        lambda: analyze(base, oracle_npz, swap_npz, rep, bad_dump, anchor, None),
        "do not recompute",
        "dump count drift",
    )
    expect_exit(
        lambda: analyze(
            base,
            oracle_npz,
            swap_npz,
            rep,
            dict(dump, subgoal_swap_identity=True),
            anchor,
            None,
        ),
        "never a read",
        "identity-dump refusal",
    )
    print("oracle: ALL branches OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--baseline-stem", default=BASE_STEM)
    parser.add_argument("--oracle-stem", default=ORACLE_STEM)
    parser.add_argument("--swap-stem", default=SWAP_STEM)
    parser.add_argument("--out", default=OUT_DEFAULT)
    args = parser.parse_args()
    if args.oracle:
        oracle()
        return
    base = _load_npz(f"{args.baseline_stem}.npz")
    analyze(
        base,
        _load_npz(f"{args.oracle_stem}.npz"),
        _load_npz(f"{args.swap_stem}.npz"),
        json.loads(Path(f"{args.swap_stem}.json").read_text()),
        json.loads(Path(f"{args.swap_stem}_swaps.json").read_text()),
        bbr.ANCHORS["ar"],
        args.out,
    )


if __name__ == "__main__":
    main()
