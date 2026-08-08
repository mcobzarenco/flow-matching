"""Frozen reads for the self-subgoal probe (#6 rung (a)).

One command over the stage-2 arm dumps (pre-reg
2026-08-07-prereg-selfsubgoal-probe.md, "Frozen reads" — semantics
frozen there, this file only mechanizes them):

  1. PRIMARY  Δ_self = chunk_mae(self-subgoal) − banked 5.8026 on all
     core frames, paired per-frame seeded bootstrap CI (seed 0, 10,000
     resamples), labeled-subset value quoted beside it;
  2. BOUND    Δ_oracle on the labeled core subset (labeled = frames
     whose TRUE ``subgoal_text`` is non-None, mask from the self run's
     ``--dump-subgoals`` file);
  3. CHANNEL  Δ_narr (suffix voice, pass 1) vs Δ_self (prompt slot),
     plus the paired narr−self per-frame CI;
  4. HORIZON  per-step-in-horizon MAE curves per arm (the
     flow_vs_ar_paired conventions) + early/late tail means;
  5. MIRRORS  first_mae versions of 1–3;
  6. EXECUTION ORACLES (each failure a hard abort): banked baseline
     anchor 5.8026/2.1431 reproduced; identity columns byte-match
     across all npzs; state-copy / state-copy-norm rows byte-match the
     banked panel; policy keys carry the mode suffixes; report JSONs
     record the mode and npz-recomputed pooled values reproduce each
     report's summaries. Label-less oracle-row decode deltas are
     RECORDED, not adjudicated (amendment 1: label-bearing batchmates
     change the batch composition, so equality vs the banked baseline
     holds only up to kernel composition noise).

The baseline is the BANKED panel npz — never re-run. Expectation 5
(falsifier): Δ_self ≥ 0 ⇒ rung (a) gives nothing at panel granularity;
the verdict line is printed but every escalation needs a new pre-reg.

``--oracle`` runs the pre-data selftest: exact-arithmetic fixtures
(constructed truths/preds so deltas and curves are hand-computable),
degenerate arm == baseline ⇒ delta exactly 0 with CI [0, 0], and every
abort branch fired.
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
SELF_STEM = f"{BASE_STEM}_selfsubgoal"
OUT_DEFAULT = "reports/analysis__selfsubgoal_ar100k_k4l2.json"

BASELINE_KEY = "pred:bijou@100000"
ORACLE_KEY = "pred:bijou@100000_oraclesubgoal"
NARR_KEY = "pred:bijou@100000_narrsubgoal"
SELF_KEY = "pred:bijou@100000_selfsubgoal"
IDENTITY_KEYS = ("index", "truth", "valid", "repo_id", "core")
STATE_KEYS = ("pred:state-copy", "pred:state-copy-norm")
SUMMARY_TOL = 5e-3
NARR_CONTEXT = +0.054  # banked: all-fields narrated arm 5.8565 vs 5.8026


def _load_npz(path: str | Path) -> dict:
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def _bytes_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def check_pairing(base: dict, probe: dict, label: str) -> None:
    """Full-panel runs share the one plan: identity columns byte-match."""
    for key in IDENTITY_KEYS:
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


def check_report(
    npz: dict,
    key: str,
    report: dict,
    mode: str,
    label: str,
) -> None:
    if report.get("subgoal_mode") != mode:
        sys.exit(
            f"{label}: report subgoal_mode {report.get('subgoal_mode')!r} != "
            f"{mode!r} — wrong run, stop",
        )
    if report.get("selfsubgoal_force_empty"):
        sys.exit(f"{label}: selfsubgoal_force_empty run is never an arm read — stop")
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


def labeled_mask(subgoals: dict, base: dict) -> np.ndarray:
    rows = subgoals.get("rows")
    if not isinstance(rows, list) or not rows:
        sys.exit("subgoals json has no 'rows' — not a --dump-subgoals file")
    by_index = {int(r["index"]): r for r in rows}
    missing = [int(ix) for ix in base["index"] if int(ix) not in by_index]
    if missing:
        sys.exit(
            f"{len(missing)} panel rows have no subgoal record (first "
            f"{missing[:3]}) — the label mask does not cover the panel, stop",
        )
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


def analyze(
    base: dict,
    base_rep: dict,
    oracle_npz: dict,
    oracle_rep: dict,
    self_npz: dict,
    self_rep: dict,
    subgoals: dict,
    anchor: tuple[float, float],
    out_path: str | None,
) -> dict:
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
        (ORACLE_KEY, oracle_npz, "oracle arm"),
        (SELF_KEY, self_npz, "self arm"),
    ):
        if key not in npz:
            sys.exit(
                f"{label}: {key} missing — policy name does not carry the mode, stop",
            )
    if NARR_KEY not in self_npz:
        sys.exit("self arm: narrated pass-1 column missing — stop")
    if BASELINE_KEY in self_npz:
        sys.exit(
            "self arm npz carries a bare bijou column — the baseline must never re-run, stop",
        )
    check_pairing(base, oracle_npz, "oracle arm")
    check_pairing(base, self_npz, "self arm")
    check_state_rows(base, oracle_npz, "oracle arm")
    check_state_rows(base, self_npz, "self arm")
    check_report(oracle_npz, ORACLE_KEY, oracle_rep, "oracle", "oracle arm")
    check_report(self_npz, SELF_KEY, self_rep, "self", "self arm")
    labeled = labeled_mask(subgoals, base)
    unlabeled = ~labeled
    # Amendment 1 (posted before the stage-2 launch): label-less rows in
    # the oracle arm decode alongside label-BEARING batchmates, so their
    # equality vs the banked baseline holds only up to batch-composition
    # kernel noise — recorded descriptively, never an abort. Abort-grade
    # byte checks stay on the composition-independent surfaces above
    # (identity columns, state-copy rows, provenance/report fields).
    diff_labelless = unlabeled & (
        (
            base[BASELINE_KEY].view(np.int32) != oracle_npz[ORACLE_KEY].view(np.int32)
        ).any(axis=(1, 2))
    )
    labelless_decode: dict[str, Any] = {
        "rows_differ": int(diff_labelless.sum()),
        "rows_labelless": int(unlabeled.sum()),
    }
    if diff_labelless.any():
        pool_mask = diff_labelless & core
        if pool_mask.any():
            oerr_all = np.abs(oracle_npz[ORACLE_KEY] - truth)
            labelless_decode["pooled_delta_on_differing_core_rows"] = round(
                bbr.pooled_chunk(oerr_all, pool_mask, w)
                - bbr.pooled_chunk(base_err, pool_mask, w),
                5,
            )
    print(
        f"execution oracles GREEN ({int(labeled.sum())} labeled / "
        f"{int(unlabeled.sum())} label-less rows; label-less decode vs "
        f"banked: {labelless_decode['rows_differ']} rows differ — "
        "amendment-1 composition noise, recorded)",
    )

    # ---- per-frame machinery shared by reads 1-3 + 5 ----
    arms = {
        "oracle": np.abs(oracle_npz[ORACLE_KEY] - truth),
        "narr": np.abs(self_npz[NARR_KEY] - truth),
        "self": np.abs(self_npz[SELF_KEY] - truth),
    }
    base_frame, nvalid = bbr.frame_mae(base_err, w)
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
        "baseline": {"chunk_mae": round(bc, 5), "first_mae": round(bf, 5)},
        "baseline_curve": [round(v, 5) for v in step_curve(base_err, valid, core)],
        "labelless_decode": labelless_decode,
        "arms": {name: reads(err) for name, err in arms.items()},
    }

    # read 3: channel comparison narr vs self, paired per-frame
    narr_frame, _ = bbr.frame_mae(arms["narr"], w)
    self_frame, _ = bbr.frame_mae(arms["self"], w)
    ch = (narr_frame - self_frame)[keep_core]
    clo, chi = bbr.bootstrap_ci(ch)
    out["channel_narr_minus_self"] = {
        "delta_frame_mean": round(float(ch.mean()), 5),
        "ci95": [round(clo, 5), round(chi, 5)],
    }

    # read 4 summary: early/late tail means of the horizon curves
    for name in ("oracle", "narr", "self"):
        curve = np.array(out["arms"][name]["curve"])
        base_curve = np.array(out["baseline_curve"])
        d = curve - base_curve
        n10 = max(1, len(d) // 5)
        out["arms"][name]["horizon_delta"] = {
            "first10": round(float(d[:n10].mean()), 5),
            "last10": round(float(d[-n10:].mean()), 5),
        }

    # expectation dispositions (pre-reg "Numbered expectations")
    d_self = out["arms"]["self"]["core"]["delta_pooled"]
    d_oracle = out["arms"]["oracle"]["labeled_subset"]["delta_pooled"]
    out["expectations"] = {
        "e1_oracle_negative_on_labeled": d_oracle < 0,
        "e2_self_negative_smaller_than_oracle": d_self < 0
        and abs(d_self) < abs(d_oracle),
        "e4_narr_context_banked": NARR_CONTEXT,
        "e5_falsified_delta_self_nonneg": d_self >= 0,
    }
    verdict = (
        "FALSIFIED (E5): Δ_self >= 0 — explicit self-hierarchy gives nothing "
        "at panel granularity on this body; escalations need a NEW pre-reg"
        if d_self >= 0
        else "Δ_self < 0 — self-conditioning helps at panel granularity"
    )
    out["verdict"] = verdict

    print(
        f"\nΔ_self  (core, primary) {d_self:+.4f}  CI {out['arms']['self']['core']['ci95']}",
    )
    print(
        f"Δ_oracle (labeled bound) {d_oracle:+.4f}  "
        f"CI {out['arms']['oracle']['labeled_subset']['ci95']}",
    )
    print(
        f"Δ_narr  (suffix voice)  {out['arms']['narr']['core']['delta_pooled']:+.4f}  "
        f"CI {out['arms']['narr']['core']['ci95']}",
    )
    print(f"verdict: {verdict}")
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
        "core": core,
        BASELINE_KEY: base_pred,
        "pred:state-copy": state,
        "pred:state-copy-norm": state + 1,
    }
    # oracle arm: labeled rows err 0.8 (-0.2), label-less byte-equal
    opred = base_pred.copy()
    opred[labeled] = 0.8
    oracle_npz = {k: v.copy() for k, v in base.items() if k != BASELINE_KEY}
    oracle_npz[ORACLE_KEY] = opred
    # self arm: err 0.9 core-wide (-0.1); narr: err 1.05 (+0.05)
    self_npz = {k: v.copy() for k, v in base.items() if k != BASELINE_KEY}
    self_npz[SELF_KEY] = np.full_like(base_pred, 0.9)
    self_npz[NARR_KEY] = np.full_like(base_pred, 1.05)
    subgoals = {
        "rows": [
            {
                "index": int(ix),
                "true_subgoal": "reach the handle" if labeled[i] else None,
            }
            for i, ix in enumerate(base["index"])
        ],
    }

    def rep(mode: str, key: str, npz: dict) -> dict:
        truth_, valid_, core_, w_ = bbr.masks(npz)
        err = np.abs(npz[key] - truth_)
        return {
            "subgoal_mode": mode,
            "selfsubgoal_force_empty": False,
            "summaries": [
                {
                    "policy": key.removeprefix("pred:"),
                    "chunk_mae": bbr.pooled_chunk(err, core_, w_),
                    "first_mae": bbr.pooled_first(err, valid_, core_),
                },
            ],
        }

    reps = {
        "base": {"subgoal_mode": None, "selfsubgoal_force_empty": False},
        "oracle": rep("oracle", ORACLE_KEY, oracle_npz),
        "self": rep("self", SELF_KEY, self_npz),
    }
    return base, oracle_npz, self_npz, subgoals, reps


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

    base, oracle_npz, self_npz, subgoals, reps = _fixture()
    anchor = (1.0, 1.0)  # constant preds: pooled == first == 1.0 exactly

    out = analyze(
        base,
        reps["base"],
        oracle_npz,
        reps["oracle"],
        self_npz,
        reps["self"],
        subgoals,
        anchor,
        None,
    )
    assert out["arms"]["self"]["core"]["delta_pooled"] == -0.1, out
    assert out["arms"]["narr"]["core"]["delta_pooled"] == 0.05, out
    assert out["arms"]["oracle"]["labeled_subset"]["delta_pooled"] == -0.2, out
    # oracle arm pooled over ALL core rows: 4 labeled of 10 -> -0.08
    assert out["arms"]["oracle"]["core"]["delta_pooled"] == -0.08, out
    assert out["channel_narr_minus_self"]["delta_frame_mean"] == 0.15, out
    assert not out["expectations"]["e5_falsified_delta_self_nonneg"], out
    # constant-error curves: delta constant at every horizon step
    assert out["arms"]["self"]["horizon_delta"] == {"first10": -0.1, "last10": -0.1}, (
        out
    )
    print("  exact-arithmetic reads OK (planted deltas reproduced)")

    # degenerate arm == baseline -> delta exactly 0, CI [0, 0]
    degen = {k: v.copy() for k, v in self_npz.items() if k not in (SELF_KEY, NARR_KEY)}
    degen[SELF_KEY] = base[BASELINE_KEY].copy()
    degen[NARR_KEY] = base[BASELINE_KEY].copy()
    rep_degen = {
        "subgoal_mode": "self",
        "selfsubgoal_force_empty": False,
        "summaries": [
            {
                "policy": SELF_KEY.removeprefix("pred:"),
                "chunk_mae": 1.0,
                "first_mae": 1.0,
            },
        ],
    }
    out2 = analyze(
        base,
        reps["base"],
        oracle_npz,
        reps["oracle"],
        degen,
        rep_degen,
        subgoals,
        anchor,
        None,
    )
    assert out2["arms"]["self"]["core"]["delta_pooled"] == 0.0, out2
    assert out2["arms"]["self"]["core"]["ci95"] == [0.0, 0.0], out2
    assert out2["expectations"]["e5_falsified_delta_self_nonneg"], out2
    assert "FALSIFIED" in out2["verdict"], out2
    print("  degenerate arm OK (delta 0, CI [0,0], E5 fires)")

    expect_exit(
        lambda: analyze(
            base,
            reps["base"],
            oracle_npz,
            reps["oracle"],
            self_npz,
            reps["self"],
            subgoals,
            (0.5, 1.0),
            None,
        ),
        "banked anchor",
        "anchor mismatch",
    )
    mut = {k: v.copy() for k, v in oracle_npz.items()}
    mut["truth"][0] += 1.0
    expect_exit(
        lambda: analyze(
            base,
            reps["base"],
            mut,
            reps["oracle"],
            self_npz,
            reps["self"],
            subgoals,
            anchor,
            None,
        ),
        "pairing broken",
        "identity drift",
    )
    mut = {k: v.copy() for k, v in oracle_npz.items()}
    mut["pred:state-copy"][1] += 1.0
    expect_exit(
        lambda: analyze(
            base,
            reps["base"],
            mut,
            reps["oracle"],
            self_npz,
            reps["self"],
            subgoals,
            anchor,
            None,
        ),
        "byte-match the banked panel",
        "state-copy drift",
    )
    mut = {k: v.copy() for k, v in oracle_npz.items()}
    mut[ORACLE_KEY] = mut[ORACLE_KEY].copy()
    mut[ORACLE_KEY][5] = 0.99  # a label-less row drifts (composition class)
    drifted = analyze(
        base,
        reps["base"],
        mut,
        reps["oracle"],
        self_npz,
        reps["self"],
        subgoals,
        anchor,
        None,
    )
    assert drifted["labelless_decode"]["rows_differ"] == 1, drifted["labelless_decode"]
    print(
        "  descriptive branch OK: label-less drift recorded, not adjudicated "
        "(amendment 1)",
    )
    bad_rep = dict(reps["self"])
    bad_rep["summaries"] = [dict(reps["self"]["summaries"][0], chunk_mae=2.0)]
    expect_exit(
        lambda: analyze(
            base,
            reps["base"],
            oracle_npz,
            reps["oracle"],
            self_npz,
            bad_rep,
            subgoals,
            anchor,
            None,
        ),
        "plan/scoring drift",
        "report drift",
    )
    expect_exit(
        lambda: analyze(
            base,
            reps["base"],
            oracle_npz,
            dict(reps["oracle"], subgoal_mode="self"),
            self_npz,
            reps["self"],
            subgoals,
            anchor,
            None,
        ),
        "wrong run",
        "mode provenance",
    )
    expect_exit(
        lambda: analyze(
            base,
            reps["base"],
            oracle_npz,
            reps["oracle"],
            self_npz,
            reps["self"],
            {"rows": subgoals["rows"][:5]},
            anchor,
            None,
        ),
        "does not cover the panel",
        "mask coverage",
    )
    mut = {k: v.copy() for k, v in self_npz.items()}
    mut[BASELINE_KEY] = base[BASELINE_KEY].copy()
    expect_exit(
        lambda: analyze(
            base,
            reps["base"],
            oracle_npz,
            reps["oracle"],
            mut,
            reps["self"],
            subgoals,
            anchor,
            None,
        ),
        "never re-run",
        "bare-column refusal",
    )
    forced = dict(reps["self"], selfsubgoal_force_empty=True)
    expect_exit(
        lambda: analyze(
            base,
            reps["base"],
            oracle_npz,
            reps["oracle"],
            self_npz,
            forced,
            subgoals,
            anchor,
            None,
        ),
        "never an arm read",
        "force-empty refusal",
    )
    print("oracle: ALL branches OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--baseline-stem", default=BASE_STEM)
    parser.add_argument("--oracle-stem", default=ORACLE_STEM)
    parser.add_argument("--self-stem", default=SELF_STEM)
    parser.add_argument("--out", default=OUT_DEFAULT)
    args = parser.parse_args()
    if args.oracle:
        oracle()
        return
    base = _load_npz(f"{args.baseline_stem}.npz")
    analyze(
        base,
        json.loads(Path(f"{args.baseline_stem}.json").read_text()),
        _load_npz(f"{args.oracle_stem}.npz"),
        json.loads(Path(f"{args.oracle_stem}.json").read_text()),
        _load_npz(f"{args.self_stem}.npz"),
        json.loads(Path(f"{args.self_stem}.json").read_text()),
        json.loads(Path(f"{args.self_stem}_subgoals.json").read_text()),
        bbr.ANCHORS["ar"],
        args.out,
    )


if __name__ == "__main__":
    main()
