"""Energy-score read over a --dump-draws stack — the strictly-proper
scoring-rule AR-vs-flow comparison from banked data.

EXPLORATORY, NOT PRE-REGISTERED (ideas #19, queue item
idea19-endpoint-fairness-es-read): the draws_fairness read-4 energy
score was pre-declared for the FLOW probe; applying it to an AR
endpoint draws dump is a diagnostic, not a registered claim.
Record-only — no decision rule.

Audit note: selection_ceiling_results.py already covers mean-of-draws /
best-of-N / dispersion on the same stack; this file extends ONLY the
energy-score delta (Amendment-2 definitions, code reused verbatim from
draws_fairness.py) plus the flow-side comparison.

Reads (all record-only):
  * endpoint ES — per frame ES = (1/N)·Σ d(a_i, y) − (1/2N²)·Σ d(a_i, a_j)
    with d = valid-element L2 / sqrt(m) (draws_fairness.energy_score
    verbatim), pooled valid-element-weighted (pooled_by_valid). The
    paired greedy arm is the degenerate N=1 baseline (interaction term
    zero by definition): ES gain = greedy ES − draws ES, with a paired
    per-frame read (bootstrap CI95 + LORO, arch_batch_results
    conventions; note: the paired read is frame-weighted, the pooled
    numbers valid-element-weighted).
  * flow comparison — intersect-join on `index` to the banked flow
    probe draws stack (drawsprobe_s7, 2458 rows x 10 draws); on the
    identical frames both families get the SAME instrument: N-draw ES
    both sides, truth terms as the matched single-draw-average read,
    and the paired per-frame ES delta (AR − flow; positive = flow
    better under the proper rule).

Execution guards (hard aborts): sample_draws metadata matches the
stack; >= 2 draws; the draws policy extends the greedy policy
(checkpoint pairing at npz level); greedy npz-recomputed summaries
reproduce its report (draws10_t1_results.report_crosscheck); identity
byte-match or the subset join (draws10_t1_results.join_rows verbatim,
q4-fallback compatible); flow-join rows must byte-match on
truth/valid.

Oracle mode (--oracle, pre-data on banked npzs only):
  (a) degenerate draws=1 (the banked AR-100k greedy as a 1-draw
      stack): interaction term EXACTLY 0 and ES == the directly
      computed RMS-L2 (< 1e-12) — the draws_fairness --validate
      pattern, per the queue item;
  (b) banked-anchor reproduction: the flow probe stack + AR-100k
      greedy joined through THIS file's path must reproduce the banked
      analysis__draws_fairness_k4l2.json read-4 numbers exactly
      (flow_es 5.930763 / single 9.882476 / interaction 3.951713 /
      ar_es 8.769585);
  (c) hand-computable N=2 fixture: truth term (d1+d2)/2, interaction
      d12/4, exact to 1e-12; residual scaling x c scales every term
      x c (homogeneity of d);
  (d) abort battery: sample_draws mismatch, non-extending policy,
      draws=1 in real mode, misaligned equal-length index, flow-join
      truth drift.

Defaults = the molmo2 endpoint draws launcher's exact output stems
(shared with selection_ceiling_results.py; the q4 fallback and the
T-sensitivity rung dumps pass explicit paths — the extending-policy
guard is T-agnostic). Pure CPU, read-only on inputs.
"""

import argparse
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
abr = _sibling("arch_batch_results")
dfair = _sibling("draws_fairness")
d10 = _sibling("draws10_t1_results")
scr = _sibling("selection_ceiling_results")

GREEDY_STEM = scr.GREEDY_STEM
DRAWS_STEM = scr.DRAWS_STEM
OUT_DEFAULT = "reports/analysis__energy_score_molmo2_40k_k4l2.json"

FLOW_DRAWS_NPZ = (
    "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000"
    "__panel_curated_v0_k4l2_drawsprobe_s7_draws10_heun30.npz"
)
AR100K_STEM = scr.AR100K_STEM
# Banked read-4 numbers (analysis__draws_fairness_k4l2.json, landed
# 2026-08-06) — the oracle must reproduce them through this file's path.
BANKED_READ4 = {
    "flow_es": 5.930763,
    "flow_single_draw_es": 9.882476,
    "interaction_term": 3.951713,
    "ar_es": 8.769585,
}
NOTE = (
    "EXPLORATORY, NOT PRE-REGISTERED — record-only energy-score "
    "diagnostic; the pre-declared read-4 was registered for the flow "
    "probe, not this dump"
)


def es_block(draws: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict:
    """Pooled ES decomposition + the per-frame arrays, dfair verbatim."""
    es, truth_term, interaction = dfair.energy_score(draws, truth, mask)
    return {
        "es": round(dfair.pooled_by_valid(es, mask), 6),
        "truth_term_single_draw_es": round(
            dfair.pooled_by_valid(truth_term, mask),
            6,
        ),
        "interaction_term": round(dfair.pooled_by_valid(interaction, mask), 6),
        "_es_frames": es,
    }


def flow_join(d_index: np.ndarray, f_npz: dict) -> tuple[np.ndarray, np.ndarray]:
    """Intersect the draws rows with the flow probe rows on `index`,
    returning (draws-side positions, flow-side positions)."""
    f_core = f_npz["core"]
    f_index = f_npz["index"][f_core]
    pos = {int(ix): i for i, ix in enumerate(d_index)}
    pairs = [(pos[int(ix)], j) for j, ix in enumerate(f_index) if int(ix) in pos]
    if not pairs:
        sys.exit("flow probe and draws dump share no rows — join is empty, stop")
    di, fi = (np.array(p) for p in zip(*pairs, strict=True))
    return di, np.flatnonzero(f_core)[fi]


def analyze(
    g_npz: dict,
    g_key: str,
    g_rep: dict,
    d_npz: dict,
    out_path: str | None,
    f_npz: dict | None = None,
) -> dict:
    # ---- execution guards (each failure is a hard abort) ----
    policy = str(scr._meta(d_npz, "policy", "draws"))
    sample_draws = int(scr._meta(d_npz, "sample_draws", "draws"))
    if d_npz["draws"].shape[1] != sample_draws:
        sys.exit(
            f"draws stack has {d_npz['draws'].shape[1]} draws but sample_draws "
            f"metadata says {sample_draws} — dump is inconsistent, stop",
        )
    if sample_draws < 2:
        sys.exit("draws dump needs >= 2 draws for an energy-score read — stop")
    policy_g = g_key.removeprefix("pred:")
    if not policy.startswith(f"{policy_g}_draws"):
        sys.exit(
            f"draws policy {policy!r} does not extend greedy policy "
            f"{policy_g!r} — arms are not the same checkpoint's reads",
        )
    rows, subset = d10.join_rows(g_npz, d_npz)

    core = d_npz["core"]
    truth, valid = d_npz["truth"][core], d_npz["valid"][core]
    draws = d_npz["draws"][core].astype(np.float64)
    g_pred = g_npz[g_key][rows][core]
    mask = dfair.element_mask(truth, valid)
    keep = mask.sum(axis=(1, 2)) > 0

    # ---- endpoint ES: draws vs the degenerate N=1 greedy baseline ----
    ar = es_block(draws, truth, mask)
    es_frames = ar.pop("_es_frames")
    greedy_frames = dfair.frame_rms_dist(g_pred, truth, mask)
    paired = abr.paired_read(
        es_frames,
        greedy_frames,
        keep,
        d_npz["repo_id"][core],
    )
    paired["definition"] = (
        "per-frame draws ES minus degenerate greedy ES (negative = "
        "sampled draws beat the deployment decode under the proper rule)"
    )
    greedy_es = round(dfair.pooled_by_valid(greedy_frames, mask), 6)
    endpoint = {
        "ar_draws": ar,
        "ar_greedy_degenerate_n1_es": greedy_es,
        "es_gain_greedy_minus_draws": round(greedy_es - ar["es"], 6),
        "paired_es_delta": paired,
    }

    # ---- flow comparison on the identical joined frames ----
    flow_cmp: dict | str = "skipped (no flow draws npz)"
    if f_npz is not None:
        di, fi = flow_join(d_npz["index"][core], f_npz)
        for k, arr in (("truth", truth), ("valid", valid)):
            if not np.array_equal(arr[di], f_npz[k][fi]):
                sys.exit(f"flow-join rows disagree on {k} — corpus drift, stop")
        j_truth, j_valid = truth[di], valid[di]
        j_mask = dfair.element_mask(j_truth, j_valid)
        j_keep = j_mask.sum(axis=(1, 2)) > 0
        ar_j = es_block(draws[di], j_truth, j_mask)
        ar_j_frames = ar_j.pop("_es_frames")
        flow_j = es_block(
            f_npz["draws"][fi].astype(np.float64),
            j_truth,
            j_mask,
        )
        flow_j_frames = flow_j.pop("_es_frames")
        paired_af = abr.paired_read(
            ar_j_frames,
            flow_j_frames,
            j_keep,
            d_npz["repo_id"][core][di],
        )
        paired_af["definition"] = (
            "per-frame AR draws ES minus flow draws ES on the joined "
            "frames (positive = flow better under the proper rule)"
        )
        flow_cmp = {
            "n_joined": len(di),
            "flow_policy": str(scr._meta(f_npz, "policy", "flow draws")),
            "flow_sample_draws": int(scr._meta(f_npz, "sample_draws", "flow draws")),
            "ar_draws": ar_j,
            "flow_draws": flow_j,
            "ar_greedy_degenerate_n1_es": round(
                dfair.pooled_by_valid(
                    dfair.frame_rms_dist(g_pred[di], j_truth, j_mask),
                    j_mask,
                ),
                6,
            ),
            "es_delta_ar_minus_flow": round(ar_j["es"] - flow_j["es"], 6),
            "paired_es_delta": paired_af,
        }

    out = {
        "note": NOTE,
        "provenance": {
            "draws_policy": policy,
            "sample_draws": sample_draws,
            "seed": int(scr._meta(d_npz, "seed", "draws")),
            "noise_key": str(scr._meta(d_npz, "noise_key", "draws")),
            "checkpoint": "/".join(d10._ckpt_id(g_rep)),
            "row_pairing": "subset-join on index (q4 fallback)"
            if subset
            else "full byte-match",
            "n_rows_core": int(core.sum()),
        },
        "endpoint_es": endpoint,
        "flow_comparison": flow_cmp,
    }
    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=1))
        print(f"wrote {out_path}")
    print(json.dumps(out, indent=1))
    print(
        f"draws ES {ar['es']} vs greedy degenerate {greedy_es} "
        f"(gain {endpoint['es_gain_greedy_minus_draws']}) | flow cmp "
        f"{flow_cmp if isinstance(flow_cmp, str) else flow_cmp['es_delta_ar_minus_flow']} "
        f"({NOTE.split(' — ')[0]})",
    )
    return out


# ---- oracle -----------------------------------------------------------------


def oracle() -> None:
    g_npz = np.load(f"{AR100K_STEM}.npz", allow_pickle=True)
    g_key = d10.bare_key(g_npz, "greedy")
    truth, valid, core, _w = bbr.masks(g_npz)
    t_c, v_c = truth[core], valid[core]
    mask = dfair.element_mask(t_c, v_c)

    # (a) degenerate draws=1: interaction EXACTLY 0, ES == direct RMS-L2
    one = g_npz[g_key][core].astype(np.float64)[:, None]
    res = es_block(one, t_c, mask)
    es_frames = res.pop("_es_frames")
    _es, _tt, inter = dfair.energy_score(one, t_c, mask)
    assert float(np.abs(inter).max()) == 0.0, "degenerate interaction not exactly 0"
    direct = dfair.pooled_by_valid(
        dfair.frame_rms_dist(one[:, 0], t_c, mask),
        mask,
    )
    assert abs(res["es"] - round(direct, 6)) < 1e-12, "degenerate ES != direct RMS-L2"
    assert res["interaction_term"] == 0.0
    assert np.abs(es_frames - dfair.frame_rms_dist(one[:, 0], t_c, mask)).max() == 0.0
    print("oracle (a) OK: draws=1 -> interaction exactly 0, ES == direct RMS-L2")

    # (b) banked-anchor reproduction through THIS file's join + pooling
    f_npz = np.load(FLOW_DRAWS_NPZ, allow_pickle=True)
    di, fi = flow_join(g_npz["index"][core], f_npz)
    assert len(di) == int(f_npz["core"].sum()), "probe rows not all joined"
    j_truth, j_valid = t_c[di], v_c[di]
    assert np.array_equal(j_truth, f_npz["truth"][fi]), "join truth drift"
    j_mask = dfair.element_mask(j_truth, j_valid)
    flow_j = es_block(f_npz["draws"][fi].astype(np.float64), j_truth, j_mask)
    flow_j.pop("_es_frames")
    ar_es = round(
        dfair.pooled_by_valid(
            dfair.frame_rms_dist(g_npz[g_key][core][di], j_truth, j_mask),
            j_mask,
        ),
        6,
    )
    got = {
        "flow_es": flow_j["es"],
        "flow_single_draw_es": flow_j["truth_term_single_draw_es"],
        "interaction_term": flow_j["interaction_term"],
        "ar_es": ar_es,
    }
    for k, want in BANKED_READ4.items():
        assert abs(got[k] - want) < 2e-6, f"banked {k}: got {got[k]}, want {want}"
    print(f"oracle (b) OK: banked read-4 numbers reproduced ({got})")

    # (c) hand-computable N=2 fixture + residual-scaling homogeneity
    t2 = np.zeros((1, 2, 3))
    v2 = np.ones((1, 2), dtype=bool)
    m2 = dfair.element_mask(t2, v2)
    a = np.ones((1, 2, 3))
    dr2 = np.stack([a, -a], axis=1)  # d(a_i, y) = 1, d(a_1, a_2) = 2
    es2, tt2, in2 = dfair.energy_score(dr2, t2, m2)
    assert abs(tt2[0] - 1.0) < 1e-12, "N=2 truth term != (d1+d2)/2"
    assert abs(in2[0] - 0.5) < 1e-12, "N=2 interaction != d12/4"
    assert abs(es2[0] - 0.5) < 1e-12, "N=2 ES != truth - interaction"
    rng = np.random.default_rng(0)
    dr = rng.normal(size=(4, 3, 5, 6))
    tr = rng.normal(size=(4, 5, 6))
    vr = np.ones((4, 5), dtype=bool)
    mr = dfair.element_mask(tr, vr)
    es1, *_ = dfair.energy_score(tr[:, None] + (dr - tr[:, None]), tr, mr)
    es3, *_ = dfair.energy_score(tr[:, None] + 3.0 * (dr - tr[:, None]), tr, mr)
    assert np.allclose(es3, 3.0 * es1, atol=1e-12), "ES not homogeneous in residuals"
    print("oracle (c) OK: N=2 hand values exact, residual scaling x3 -> ES x3")

    # (d) abort battery
    g_rep = json.loads(Path(f"{AR100K_STEM}.json").read_text())

    def _expect(fn: Callable[[], object], needle: str, label: str) -> None:
        try:
            fn()
            raise AssertionError(f"{label}: expected abort not raised")
        except SystemExit as e:
            assert needle in str(e), f"{label}: wrong abort message: {e}"
        print(f"oracle abort OK: {label}")

    sub_rows = np.arange(0, len(g_npz["index"]), 3)
    fx3 = scr._stack_fixture(
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
    one_fx = scr._stack_fixture(g_npz, g_key, sub_rows, np.ones((len(sub_rows), 1)))
    _expect(
        lambda: analyze(g_npz, g_key, g_rep, one_fx, None),
        ">= 2 draws",
        "draws=1 in real mode",
    )
    full = scr._stack_fixture(
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
    bad_flow = abr._DictNpz({k: np.array(f_npz[k]) for k in f_npz.files})
    bad_flow["truth"] = np.array(bad_flow["truth"])
    bad_flow["truth"][0] += 1.0
    _expect(
        lambda: analyze(g_npz, g_key, g_rep, full, None, f_npz=bad_flow),
        "corpus drift",
        "flow-join truth drift",
    )

    # real-mode smoke on the fixture: runs end-to-end with the flow join
    res_full = analyze(g_npz, g_key, g_rep, full, None, f_npz=f_npz)
    assert res_full["flow_comparison"]["n_joined"] == len(di), "smoke join count"
    assert res_full["flow_comparison"]["flow_draws"]["es"] == BANKED_READ4["flow_es"], (
        "smoke flow ES != banked"
    )
    print("ORACLE PASS: energy-score reads verified pre-data")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--draws-npz", default=f"{DRAWS_STEM}_draws.npz")
    p.add_argument(
        "--greedy",
        nargs=2,
        metavar=("NPZ", "JSON"),
        default=[f"{GREEDY_STEM}.npz", f"{GREEDY_STEM}.json"],
    )
    p.add_argument(
        "--flow-draws",
        default=FLOW_DRAWS_NPZ,
        help="banked flow probe draws npz ('none' skips the comparison)",
    )
    p.add_argument("--out", default=OUT_DEFAULT)
    p.add_argument("--oracle", action="store_true")
    a = p.parse_args()
    if a.oracle:
        oracle()
        return
    needed = [a.draws_npz, *a.greedy]
    if a.flow_draws != "none":
        needed.append(a.flow_draws)
    for path in needed:
        if not Path(path).exists():
            sys.exit(f"missing input {path} — eval not finished / not rsynced?")
    d_npz = np.load(a.draws_npz, allow_pickle=True)
    g_npz = np.load(a.greedy[0], allow_pickle=True)
    g_rep = json.loads(Path(a.greedy[1]).read_text())
    g_key = d10.bare_key(g_npz, "greedy")
    d10.report_crosscheck(g_npz, g_key, g_rep, "greedy")
    f_npz = None
    if a.flow_draws != "none":
        f_npz = np.load(a.flow_draws, allow_pickle=True)
    analyze(g_npz, g_key, g_rep, d_npz, a.out, f_npz=f_npz)


if __name__ == "__main__":
    main()
