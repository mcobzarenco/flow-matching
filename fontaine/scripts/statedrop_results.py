"""State-dropout arm C results — the pre-registered reads, ready before the data.

Implements exactly the frozen reads of the state-dropout pre-reg
(posts/2026-08-06-prereg-state-dropout-40k.md, ideas #9):

  * READ 1 (primary): paired per-frame panel chunk_mae C - A-s0 over the
    core frames, seeded bootstrap CI (the box_batch_results paired-read
    path, re-oracled on this pair). Band = max(3*sigma_seed, 0.15) = 0.15:
      < -0.15  HELPS  -> adopt as recipe default (own follow-up pre-reg)
      inside   NEUTRAL-> decision moves to reads 2-3 (free hardening lever
                         adopted iff the mechanism reads clean)
      > +0.15  COSTS  -> adopt nothing; branch on read 2 (p=0.3 screen iff
                         the mechanism worked, else the dropout leg is
                         falsified at this scale)
  * READ 2 (reliance): C's masked q4-subset eval vs C's OWN full-panel npz
    pooled on the subset rows (the state-probe instrument's pattern,
    including its byte-match execution oracles).
      sanity gate: Delta_first(C) = masked - intact < 5.0 (vs A-s0's
        +19.950) — proves the regularizer trained the intended condition
      capability:  masked first_mae vs banked comparators on the identical
        rows — intact state-copy 2.4316 (< it = qualitative first),
        < 6.0 strong vision signal, >= 15.0 dropout failed to build one
        (A-s0 masked 23.8154, B masked 24.0783)
  * READ 3 (grounding): C's intact panel first_mae vs A-s0's 3.9422
    (state-copy floor 2.6202). Pre-declared trap: only meaningful jointly
    with read 2 (better first + collapsed reliance = vision did it;
    better first + intact reliance = shortcut did it).
  * E3 formal final gate (--probe-final, from the train log): in-run probe
    < 10 @40k — above it, reads still run (a negative is a deliverable)
    but NO adoption path opens from this arm.
  * E4 descriptive: C pooled chunk_mae vs the honest prior band 7.65-8.30.

Execution oracles inherited from the sibling instruments, abort on failure:
report-JSON reproduction (|d| < 5e-3) for both panels and the masked run;
state-copy/-norm byte-match between C's masked npz and C's panel npz on
the subset rows; ``mask_state: true`` + ``_state-masked`` policy naming;
frozen q4 plan sha256; PAIR_KEYS equality between the C and A-s0 panels.

Oracle mode (--oracle, run before any arm-C data existed):
  (a) A-s0 panel npz through this file's pooling reproduces the banked
      7.7966/3.9422 + state-copy 11.7848/2.6202, and the q4 subset rows
      reproduce the probe's intact state-copy first 2.4316;
  (b) degenerate C := A-s0 (panel) + unmodified subset rows (masked) ->
      read 1 exactly 0 / CI [0,0] -> NEUTRAL, sanity gate trivially green,
      capability = intact level (strong) -> the neutral-adopt path;
  (c) synthetic known effects: 1.05x error inflation on C's panel ->
      +~0.29 > band -> COSTS (with the p=0.3 branch when the mechanism
      read is clean); 0.95x deflation -> HELPS; 6.2x masked inflation ->
      capability ~24 >= 15 -> mechanism inert -> dropout leg killed;
      1.5x masked inflation -> strong capability + sanity pass -> the
      neutral hardening-lever adoption; probe-final 10.5 blocks adoption;
  (d) misaligned masked index -> hard abort (pair_banked).

Pooling semantics are byte-identical to box_batch_results.py /
state_probe_results.py (both anchored). Pure CPU, read-only on inputs,
deterministic (seeded bootstrap).

Usage (defaults = the chained eval's output names, rsynced local):
  python fontaine/scripts/statedrop_results.py \
      [--c-panel NPZ JSON] [--a-panel NPZ JSON] [--c-masked NPZ JSON] \
      [--probe-final FLOAT] [--plan ...] [--out ...]
"""

import argparse
import importlib.util
import json
import sys
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
spr = _sibling("state_probe_results")

C_STEM = "reports/eval__fontaine_arb_rcond_statedrop80_40k_1xh100__step_040000"
C_PANEL_NPZ = f"{C_STEM}__panel_curated_v0_k4l2.npz"
C_PANEL_JSON = f"{C_STEM}__panel_curated_v0_k4l2.json"
C_MASKED_NPZ = f"{C_STEM}__stateprobe_q4_state-masked.npz"
C_MASKED_JSON = f"{C_STEM}__stateprobe_q4_state-masked.json"
A_STEM = "reports/eval__fontaine_arb_rcond_40k_1xh100__step_040000"
A_PANEL_NPZ = f"{A_STEM}__panel_curated_v0_k4l2.npz"
A_PANEL_JSON = f"{A_STEM}__panel_curated_v0_k4l2.json"
OUT_DEFAULT = "reports/analysis__statedrop_40k_k4l2.json"

# Banked comparators (box-batch results + state-probe results, v1 keying)
AS0_CHUNK, AS0_FIRST = 7.7966, 3.9422
PANEL_STATE_COPY = (11.7848, 2.6202)
SUBSET_STATE_COPY_FIRST = 2.4316
AS0_MASKED_FIRST = 23.8154
B_MASKED_FIRST = 24.0783
AS0_DELTA_FIRST = 19.950

BAND = 0.15  # read 1: max(3*sigma_seed 0.038, 0.15) — floor binds
SANITY_MAX = 5.0  # read 2 sanity gate on Delta_first(C)
STRONG_MAX = 6.0  # read 2 capability: strong vision signal below this
FAILED_MIN = 15.0  # read 2 capability: dropout failed to build one at/above
PROBE_GATE = 10.0  # E3 formal final gate on the in-run probe @40k
E4_BAND = (7.65, 8.30)  # honest prior on C's pooled chunk_mae (descriptive)
SUMMARY_TOL = 5e-3


def classify_capability(masked_first: float) -> str:
    if masked_first < SUBSET_STATE_COPY_FIRST:
        return "qualitative-first"  # beats the proprioceptive-extrapolation floor
    if masked_first < STRONG_MAX:
        return "strong"
    if masked_first < FAILED_MIN:
        return "partial"
    return "failed"


def verdict(
    read1_mean: float,
    *,
    sanity_pass: bool,
    capability: str,
    probe_final: float | None,
) -> dict:
    """The frozen decision assembly. Every branch is pre-registered."""
    probe_ok = probe_final is None or probe_final < PROBE_GATE
    if read1_mean < -BAND:
        primary = "HELPS: state dropout improves actions in-distribution"
        adoption = (
            "ADOPT as recipe default (own follow-up pre-reg for the next lineage run)"
        )
        branch = None
    elif read1_mean <= BAND:
        primary = (
            "NEUTRAL: |C - A-s0| within the 0.15 band — decision moves to reads 2-3"
        )
        if capability in ("qualitative-first", "strong") and sanity_pass:
            adoption = "ADOPT as free hardening lever (mechanism reads clean: vision capability built at no panel cost)"
            branch = None
        elif capability == "failed":
            adoption = "NO ADOPTION"
            branch = (
                "MECHANISM INERT (pre-declared): neutral panel + capability near the "
                "20s = trained-in mask tolerance without vision grounding — the "
                "dropout leg is killed at this scale"
            )
        else:
            adoption = "NO ADOPTION from this arm"
            branch = (
                "mechanism partial (capability in [6, 15) or sanity gate failed) — "
                "owner-facing discussion material, no pre-registered adoption path"
            )
    else:
        primary = "COSTS: p=0.8 costs actions beyond the band"
        adoption = "ADOPT NOTHING"
        branch = (
            "p=0.3 screen is the one sanctioned follow-up (mechanism worked: reliance collapsed)"
            if sanity_pass
            else "#9's dropout leg is FALSIFIED at this scale (band miss + mechanism failed)"
        )
    if not probe_ok:
        adoption = (
            f"NO ADOPTION PATH (E3 formal final gate failed: probe {probe_final} "
            f">= {PROBE_GATE} @40k — p=0.8 too aggressive at this scale); reads stand as deliverables"
        )
    return {
        "read1_mean": round(read1_mean, 5),
        "band": BAND,
        "sanity_gate_passed": bool(sanity_pass),
        "capability_class": capability,
        "probe_final": probe_final,
        "probe_gate_ok": bool(probe_ok),
        "primary_verdict": primary,
        "adoption": adoption,
        "branch": branch,
    }


def analyze(
    c_panel: dict,
    c_key: str,
    a_panel: dict,
    a_key: str,
    c_masked: dict,
    m_key: str,
    pos: np.ndarray,
    probe_final: float | None,
    out_path: str | None,
) -> dict:
    # READ 1 — paired per-frame panel chunk_mae, C - A-s0
    for k in bbr.PAIR_KEYS:
        if not np.array_equal(a_panel[k], c_panel[k]):
            sys.exit(f"panel pairing broken on {k} between C and A-s0")
    truth, valid, core, w = bbr.masks(a_panel)
    err_c = np.abs(c_panel[c_key] - truth)
    err_a = np.abs(a_panel[a_key] - truth)
    f_c, _ = bbr.frame_mae(err_c, w)
    f_a, _ = bbr.frame_mae(err_a, w)
    keep = (w.sum(axis=(1, 2)) > 0) & core
    d_rows = (f_c - f_a)[keep]
    ci = bbr.bootstrap_ci(d_rows)
    pooled = {
        "C": {
            "chunk_mae": round(bbr.pooled_chunk(err_c, core, w), 4),
            "first_mae": round(bbr.pooled_first(err_c, valid, core), 4),
            "pred_key": c_key,
        },
        "A-s0": {
            "chunk_mae": round(bbr.pooled_chunk(err_a, core, w), 4),
            "first_mae": round(bbr.pooled_first(err_a, valid, core), 4),
            "pred_key": a_key,
        },
    }
    lo = bbr.loro(d_rows, a_panel["repo_id"][keep])
    influential = sorted(
        lo.items(),
        key=lambda kv: abs(kv[1]["mean_without"] - float(d_rows.mean())),
        reverse=True,
    )[:5]
    read1 = {
        "definition": "paired per-frame chunk_mae, C - A-s0, core frames",
        "mean": round(float(d_rows.mean()), 5),
        "median": round(float(np.median(d_rows)), 5),
        "ci95": [round(ci[0], 5), round(ci[1], 5)],
        "c_win_rate": round(float((d_rows < 0).mean()), 4),
        "n_frames": int(keep.sum()),
        "pooled_dchunk": round(
            pooled["C"]["chunk_mae"] - pooled["A-s0"]["chunk_mae"],
            5,
        ),
        "most_influential_repos": [{"repo": k, **v} for k, v in influential],
    }

    # READ 2 — reliance: C masked subset vs C's own panel pooled on the rows
    s_truth, s_valid, s_w = spr.masks(c_masked)
    m_err = np.abs(c_masked[m_key] - s_truth)
    i_err = np.abs(c_panel[c_key][pos] - s_truth)
    d_first_rows = (spr.row_first_mae(m_err) - spr.row_first_mae(i_err))[s_valid[:, 0]]
    d_chunk_rows = (spr.row_chunk_mae(m_err, s_w) - spr.row_chunk_mae(i_err, s_w))[
        s_w.sum(axis=(1, 2)) > 0
    ]
    f_ci = spr.bootstrap_ci(d_first_rows)
    c_ci = spr.bootstrap_ci(d_chunk_rows)
    masked_first = spr.pooled_first(m_err, s_valid)
    delta_first = float(d_first_rows.mean())
    sanity_pass = delta_first < SANITY_MAX
    capability = classify_capability(masked_first)
    read2 = {
        "subset_rows": int(s_truth.shape[0]),
        "masked_first": round(masked_first, 4),
        "masked_chunk": round(spr.pooled_chunk(m_err, s_w), 4),
        "intact_first_on_subset": round(spr.pooled_first(i_err, s_valid), 4),
        "intact_chunk_on_subset": round(spr.pooled_chunk(i_err, s_w), 4),
        "delta_first": {
            "mean": round(delta_first, 5),
            "ci95": [round(f_ci[0], 5), round(f_ci[1], 5)],
        },
        "delta_chunk": {
            "mean": round(float(d_chunk_rows.mean()), 5),
            "ci95": [round(c_ci[0], 5), round(c_ci[1], 5)],
        },
        "sanity_gate": {
            "rule": f"delta_first < {SANITY_MAX} (vs A-s0's +{AS0_DELTA_FIRST})",
            "passed": bool(sanity_pass),
        },
        "capability": {
            "masked_first": round(masked_first, 4),
            "class": capability,
            "comparators_same_rows": {
                "intact_state_copy_first": SUBSET_STATE_COPY_FIRST,
                "As0_masked_first": AS0_MASKED_FIRST,
                "B_masked_first": B_MASKED_FIRST,
            },
            "thresholds": {
                "qualitative_first_lt": SUBSET_STATE_COPY_FIRST,
                "strong_lt": STRONG_MAX,
                "failed_ge": FAILED_MIN,
            },
        },
    }

    # READ 3 — grounding: intact panel first_mae vs A-s0, joint with read 2
    c_first = pooled["C"]["first_mae"]
    better_first = c_first < AS0_FIRST
    if better_first and sanity_pass:
        joint = "vision did it: better first_mae WITH collapsed reliance"
    elif better_first:
        joint = "shortcut did it: better first_mae with INTACT reliance (the B trap)"
    else:
        joint = "no first_mae gain to attribute"
    read3 = {
        "c_first_mae": c_first,
        "as0_first_mae": AS0_FIRST,
        "delta_first_vs_as0": round(c_first - AS0_FIRST, 5),
        "state_copy_floor": PANEL_STATE_COPY[1],
        "joint_interpretation": joint,
    }

    e4 = {
        "band": list(E4_BAND),
        "c_chunk_mae": pooled["C"]["chunk_mae"],
        "inside": bool(E4_BAND[0] <= pooled["C"]["chunk_mae"] <= E4_BAND[1]),
    }
    dec = verdict(
        read1["mean"],
        sanity_pass=sanity_pass,
        capability=capability,
        probe_final=probe_final,
    )

    out = {
        "pooled": pooled,
        "read1_primary_C_minus_As0": read1,
        "read2_reliance": read2,
        "read3_grounding": read3,
        "E4_prior_band": e4,
        "decision": dec,
    }
    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=1))
        print(f"wrote {out_path}")
    print(json.dumps(out, indent=1))
    print(f"\nDECISION: {dec['primary_verdict']}")
    print(f"  adoption: {dec['adoption']}")
    if dec["branch"]:
        print(f"  branch: {dec['branch']}")
    return out


def subset_positions(masked: dict, panel: dict, label: str) -> np.ndarray:
    """pair_banked's byte-match oracles, C's masked npz vs C's OWN panel."""
    return spr.pair_banked(masked, panel, label)


def oracle() -> None:
    a = np.load(A_PANEL_NPZ, allow_pickle=True)
    a_key = spr.banked_bare_key(a, "oracle-as0")

    # (a) anchor reproduction through THIS file's pooling
    truth, valid, core, w = bbr.masks(a)
    err = np.abs(a[a_key] - truth)
    gc, gf = bbr.pooled_chunk(err, core, w), bbr.pooled_first(err, valid, core)
    assert abs(gc - AS0_CHUNK) < SUMMARY_TOL and abs(gf - AS0_FIRST) < SUMMARY_TOL, (
        f"A-s0 anchor FAIL: {gc:.4f}/{gf:.4f} vs banked {AS0_CHUNK}/{AS0_FIRST}"
    )
    sc_err = np.abs(a["pred:state-copy"] - truth)
    sc = (
        bbr.pooled_chunk(sc_err, core, w),
        bbr.pooled_first(sc_err, valid, core),
    )
    assert abs(sc[0] - PANEL_STATE_COPY[0]) < SUMMARY_TOL
    assert abs(sc[1] - PANEL_STATE_COPY[1]) < SUMMARY_TOL
    pos = np.where(a["core"])[0][::4]
    assert len(pos) == spr.SUBSET_ROWS
    syn = spr.make_synthetic_masked(a, pos, a_key)
    s_truth, s_valid, _s_w = spr.masks(syn)
    sub_sc = spr.pooled_first(np.abs(syn["pred:state-copy"] - s_truth), s_valid)
    assert abs(sub_sc - SUBSET_STATE_COPY_FIRST) < SUMMARY_TOL, (
        f"subset state-copy first {sub_sc:.4f} vs banked {SUBSET_STATE_COPY_FIRST}"
    )
    print(
        f"oracle (a) anchors: A-s0 {gc:.4f}/{gf:.4f}, state-copy {sc[0]:.4f}/{sc[1]:.4f}, "
        f"subset state-copy first {sub_sc:.4f} OK",
    )

    # (b) degenerate: C := A-s0, masked := unmodified subset rows
    m_key = a_key + "_state-masked"
    p2 = subset_positions(syn, a, "oracle-degenerate")
    assert np.array_equal(p2, pos)
    res = analyze(a, a_key, a, a_key, syn, m_key, pos, None, None)
    r1 = res["read1_primary_C_minus_As0"]
    assert r1["mean"] == 0.0 and r1["ci95"] == [0.0, 0.0]
    assert res["read2_reliance"]["delta_first"]["mean"] == 0.0
    assert res["read2_reliance"]["sanity_gate"]["passed"]
    assert res["decision"]["capability_class"] == "strong"  # intact level ~3.87
    assert res["decision"]["adoption"].startswith("ADOPT as free hardening lever")
    print("\noracle (b) degenerate: zero deltas, neutral-adopt path OK\n")

    # (c) synthetic known effects
    # 1.05x error inflation scales every valid element's abs error by exactly
    # 1.05, so the paired frame-delta mean is 0.05 * A-s0's mean frame MAE
    # and the pooled chunk_mae is exactly 1.05 * the banked anchor.
    f_a, _ = bbr.frame_mae(err, w)
    keep = (w.sum(axis=(1, 2)) > 0) & a["core"]
    want_m = 0.05 * float(f_a[keep].mean())
    infl = {k: a[k] for k in a.files}
    infl[a_key] = a["truth"] + 1.05 * (a[a_key] - a["truth"])
    res = analyze(infl, a_key, a, a_key, syn, m_key, pos, None, None)
    m = res["read1_primary_C_minus_As0"]["mean"]
    assert abs(m - want_m) < 5e-3, (
        f"synthetic delta {m} != 0.05*frame-mean {want_m:.5f}"
    )
    assert abs(res["pooled"]["C"]["chunk_mae"] - 1.05 * AS0_CHUNK) < 5e-3
    assert m > BAND and res["decision"]["primary_verdict"].startswith("COSTS")
    assert "p=0.3 screen" in res["decision"]["branch"]
    defl = {k: a[k] for k in a.files}
    defl[a_key] = a["truth"] + 0.95 * (a[a_key] - a["truth"])
    res = analyze(defl, a_key, a, a_key, syn, m_key, pos, None, None)
    assert res["read1_primary_C_minus_As0"]["mean"] < -BAND
    assert res["decision"]["primary_verdict"].startswith("HELPS")
    dead = spr.make_synthetic_masked(a, pos, a_key, scale=6.2)
    res = analyze(a, a_key, a, a_key, dead, m_key, pos, None, None)
    assert res["decision"]["capability_class"] == "failed"
    assert not res["read2_reliance"]["sanity_gate"]["passed"]
    assert "MECHANISM INERT" in res["decision"]["branch"]
    mid = spr.make_synthetic_masked(a, pos, a_key, scale=1.5)
    res = analyze(a, a_key, a, a_key, mid, m_key, pos, None, None)
    assert res["decision"]["capability_class"] == "strong"
    assert res["read2_reliance"]["sanity_gate"]["passed"]
    assert res["decision"]["adoption"].startswith("ADOPT as free hardening lever")
    res = analyze(a, a_key, a, a_key, mid, m_key, pos, 10.5, None)
    assert "NO ADOPTION PATH" in res["decision"]["adoption"]
    print(
        "\noracle (c) synthetics: COSTS/HELPS/inert-kill/strong-adopt/probe-gate OK\n",
    )

    # (d) misaligned masked index -> hard abort
    perm = np.random.default_rng(1).permutation(spr.SUBSET_ROWS)
    broken = spr._DictNpz({k: syn[k] for k in syn.files})
    broken["index"] = syn["index"][perm]
    try:
        subset_positions(broken, a, "oracle-misaligned")
    except SystemExit:
        print("oracle (d) pairing break: misaligned index aborted OK")
    else:
        sys.exit("oracle (d) FAILED: misaligned index was not caught")
    print("\nORACLE: all four checks PASSED")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--c-panel",
        nargs=2,
        metavar=("NPZ", "JSON"),
        default=[C_PANEL_NPZ, C_PANEL_JSON],
    )
    p.add_argument(
        "--a-panel",
        nargs=2,
        metavar=("NPZ", "JSON"),
        default=[A_PANEL_NPZ, A_PANEL_JSON],
    )
    p.add_argument(
        "--c-masked",
        nargs=2,
        metavar=("NPZ", "JSON"),
        default=[C_MASKED_NPZ, C_MASKED_JSON],
    )
    p.add_argument("--probe-final", type=float, default=None)
    p.add_argument("--plan", default=spr.PLAN_DEFAULT)
    p.add_argument("--out", default=OUT_DEFAULT)
    p.add_argument("--oracle", action="store_true")
    a = p.parse_args()
    if a.oracle:
        oracle()
        return
    spr.assert_plan(a.plan)
    c_panel = np.load(a.c_panel[0], allow_pickle=True)
    a_panel = np.load(a.a_panel[0], allow_pickle=True)
    truth, valid, core, w = bbr.masks(c_panel)
    pred_keys = [k for k in c_panel.files if k.startswith("pred:bijou@")]
    _scores, c_key = bbr.pick_headline(
        c_panel,
        pred_keys,
        a.c_panel[1],
        truth,
        valid,
        core,
        w,
        "C-panel",
    )
    truth, valid, core, w = bbr.masks(a_panel)
    pred_keys = [k for k in a_panel.files if k.startswith("pred:bijou@")]
    _scores, a_key = bbr.pick_headline(
        a_panel,
        pred_keys,
        a.a_panel[1],
        truth,
        valid,
        core,
        w,
        "A-s0-panel",
    )
    c_masked, m_key = spr.load_masked(a.c_masked[0], a.c_masked[1], "C-masked")
    pos = subset_positions(c_masked, c_panel, "C-masked")
    analyze(c_panel, c_key, a_panel, a_key, c_masked, m_key, pos, a.probe_final, a.out)


if __name__ == "__main__":
    main()
