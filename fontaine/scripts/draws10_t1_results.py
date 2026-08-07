"""AR sampled-draws frozen reads — Delta_AR + fairness/family, ready before the data.

Implements exactly the frozen reads of the AR sampled-draws pre-reg
(posts/2026-08-06-prereg-ar-sampled-draws.md), reads 1-5, for one arm
(default: AR-100k, the deployment-anchor checkpoint; the molmo2 arm at
its endpoint passes explicit paths, same semantics):

  * READ 1 (primary): Delta_AR = chunk_mae(_draws10_t1) - chunk_mae
    (greedy), paired per-frame on identical rows, seeded frame bootstrap
    CI95 (seed 0, 10,000 resamples — the draws-fairness assembly
    conventions, pooling verbatim from box_batch_results.py).
  * READ 2 (fairness comparison): Delta_AR vs the flow teacher's -1.258
    (its draws1 6.6232 -> draws10 5.365 gain on the same panel) — does
    the AR family ensemble like flow, or is greedy already the mean?
  * READ 3 (family read): does _draws10_t1 reach the flow draws10 band
    (5.365)? Both families then hold mean-of-10 reads under their own
    stochasticity — the first symmetric-instrument comparison. Quoted
    beside: teacher draws10-heun30 5.3645/1.4242, SnapFlow student
    mean-of-10 5.3675/1.5927 (banked, record-only).
  * READ 4: first_mae mirrors of reads 1-3 (flow first gain -0.5089 =
    1.4242 - 1.9331, descriptive). The T-sensitivity rung is a separate
    record-only eval, not assembled here.
  * READ 5 (execution oracles, abort on failure): state-copy and
    state-copy-norm columns byte-match the banked greedy npz on the
    paired rows; report JSON carries ar_temperature 1.0 + sample_draws
    10; the draws policy name carries _draws10_t1 and extends the
    greedy policy name exactly; both reports name the same checkpoint;
    npz-recomputed pooled chunk/first reproduce each report's summaries
    entry (|d| < 5e-3 — plan/scoring drift stops the read).

Expectations (banked before data; E-numbers from the pre-reg):
  E1 Delta_AR < 0 (conf medium); E2 |Delta_AR| < 1.258 (conf med-high,
  the informative read either way); E3 _draws10_t1 does NOT overtake
  5.365 (conf medium); E4 FALSIFIED IF Delta_AR > +0.1 — the
  mean-of-samples premise fails and the instrument retires to
  diagnostic use (no temperature fishing beyond the registered rung).

Row pairing: full-panel npzs must byte-match on the identity columns;
a q4-fallback draws npz (the pre-registered cost-gate subset,
stateprobe_q4 plan) is joined on the corpus concat `index` (the
draws-fairness join convention) and the greedy comparison re-pooled
onto those rows — recorded as subset_mode, never silent.

Oracle mode (--oracle, run before any draws data exists):
  (a) the banked AR-100k greedy npz pooled through THIS file reproduces
      the 5.8026/2.1431 anchor; the quoted flow constants reproduce
      from the banked draws10-heun30 report JSON;
  (b) degenerate draws := greedy -> Delta_AR exactly 0, CI [0, 0], not
      falsified, within-noise branch;
  (c) synthetic known effects (err scaled x0.95 / x1.005 / x1.05 /
      x0.90 / x0.75): E1+E2 branch with magnitude check, null branch,
      the E4 falsifier line, the E3 overtake line, the E2-not-met
      (ensembles-like-flow) branch;
  (d) misaligned equal-length index -> hard abort; subset join with a
      duplicated / missing index -> hard abort;
  (e) execution-oracle aborts: corrupted state-copy column, wrong
      ar_temperature / sample_draws / plan / counts, policy missing
      _draws10_t1, checkpoint mismatch, report-value drift;
  (f) subset path: a q4-shaped slice fixture joins, re-pools, and
      reproduces the sliced direct computation.

Pure CPU, read-only on inputs, deterministic (seeded bootstrap).

Usage (defaults = the AR-100k launcher's exact output stems):
  python fontaine/scripts/draws10_t1_results.py \
      [--draws NPZ JSON] [--greedy NPZ JSON] [--out ...] [--oracle]
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

GREEDY_STEM = "reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2"
DRAWS_STEM = f"{GREEDY_STEM}_draws10_t1"
OUT_DEFAULT = "reports/analysis__draws10_t1_ar100k_k4l2.json"

V1_PLAN = "plans/holdout_curated_v0_k4l2.json"
Q4_PLAN = "plans/holdout_curated_v0_k4l2_stateprobe_q4.json"
PLAN_COUNTS = {V1_PLAN: (17204, 8596), Q4_PLAN: (4301, 0)}

AR_ANCHOR = bbr.ANCHORS["ar"]  # (5.8026, 2.1431) — the deployment anchor
FLOW_GAIN = -1.258  # pre-reg read 2: teacher draws1 6.6232 -> draws10 5.365
FLOW_BAND = 5.365  # pre-reg read 3: the flow draws10 band
FLOW_HEUN30 = (5.3645, 1.4242)  # teacher draws10-heun30, banked
SNAPFLOW_MEAN10 = (5.3675, 1.5927)  # student mean-of-10, banked, record-only
FLOW_FIRST_GAIN = -0.5089  # 1.4242 - 1.9331, first_mae mirror, descriptive
FALSIFIER = 0.1  # E4: falsified if Delta_AR > +0.1
SUMMARY_TOL = 5e-3


def load_draws_report(src: str | Path | dict, label: str) -> dict:
    """Strict loader for the _draws10_t1 report JSON — every sampling
    semantic the pre-reg fixed must be recorded, or we stop (read 5)."""
    d = src if isinstance(src, dict) else json.loads(Path(src).read_text())
    plan = d.get("sample_plan")
    if plan not in PLAN_COUNTS:
        sys.exit(
            f"{label}: sample_plan = {plan!r}, registered plans are "
            f"{sorted(PLAN_COUNTS)} — refusing the read",
        )
    want = {
        "core_frames": PLAN_COUNTS[plan][0],
        "labeled_frames": PLAN_COUNTS[plan][1],
        "ar_temperature": 1.0,
        "sample_draws": 10,
    }
    for k, v in want.items():
        got = d.get(k)
        if got != v:
            sys.exit(f"{label}: {k} = {got!r}, registered {v!r} — refusing the read")
    return d


def bare_key(d: dict, label: str) -> str:
    keys = [
        k for k in d.files if k.startswith("pred:bijou@") and not k.endswith("+fields")
    ]
    if len(keys) != 1:
        sys.exit(f"{label}: expected exactly one bare pred:bijou@ column, got {keys}")
    return keys[0]


def report_crosscheck(d: dict, key: str, rep: dict, label: str) -> None:
    """The pick_headline oracle on a loaded report dict: npz-recomputed
    pooled chunk/first must reproduce the report's summaries entry."""
    policy = key.removeprefix("pred:")
    summ = [s for s in rep.get("summaries", []) if s.get("policy") == policy]
    if len(summ) != 1:
        sys.exit(
            f"{label}: report has {len(summ)} summaries for policy {policy!r} "
            f"(have: {[s.get('policy') for s in rep.get('summaries', [])]})",
        )
    truth, valid, core, w = bbr.masks(d)
    err = np.abs(d[key] - truth)
    gc = bbr.pooled_chunk(err, core, w)
    gf = bbr.pooled_first(err, valid, core)
    wc, wf = summ[0]["chunk_mae"], summ[0]["first_mae"]
    if abs(gc - wc) >= SUMMARY_TOL or abs(gf - wf) >= SUMMARY_TOL:
        sys.exit(
            f"{label}: npz-recomputed chunk/first {gc:.4f}/{gf:.4f} do not "
            f"reproduce the report's {wc:.4f}/{wf:.4f} for {policy} — "
            f"plan/scoring drift, stop",
        )
    print(f"{label}: report cross-check OK ({policy} chunk {gc:.4f} first {gf:.4f})")


def _ckpt_id(rep: dict) -> tuple:
    return Path(str(rep.get("checkpoint", ""))).parts[-2:]


def join_rows(g_npz: dict, d_npz: dict) -> tuple[np.ndarray, bool]:
    """Rows of the greedy npz pairing the draws npz, in draws order.

    Equal length => the identity columns must byte-match (no silent
    reordering); shorter draws npz => the pre-registered q4 fallback,
    joined on `index` with identity equality re-checked on the joined
    rows. Anything else is a hard abort.
    """
    gi, di = g_npz["index"], d_npz["index"]
    if len(di) == len(gi):
        for k in bbr.PAIR_KEYS:
            if not np.array_equal(g_npz[k], d_npz[k]):
                sys.exit(f"panel pairing broken on {k} between greedy and draws")
        return np.arange(len(gi)), False
    if len(di) > len(gi):
        sys.exit(f"draws npz has MORE rows ({len(di)}) than greedy ({len(gi)})")
    if len(np.unique(di)) != len(di):
        sys.exit("draws npz has duplicate index values — refusing the join")
    pos = {int(ix): i for i, ix in enumerate(gi)}
    missing = [int(ix) for ix in di if int(ix) not in pos]
    if missing:
        sys.exit(
            f"{len(missing)} draws rows absent from the greedy panel "
            f"(first: {missing[:3]}) — subset join broken",
        )
    rows = np.array([pos[int(ix)] for ix in di])
    for k in ["truth", "valid", "repo_id", "core"]:
        if not np.array_equal(g_npz[k][rows], d_npz[k]):
            sys.exit(f"subset rows disagree with the greedy panel on {k}")
    return rows, True


def analyze(
    g_npz: dict,
    g_key: str,
    g_rep: dict,
    d_npz: dict,
    d_key: str,
    d_rep: dict,
    out_path: str | None,
) -> dict:
    # ---- read 5: execution oracles (each failure is a hard abort) ----
    policy_d = d_key.removeprefix("pred:")
    policy_g = g_key.removeprefix("pred:")
    if not policy_d.endswith("_draws10_t1"):
        sys.exit(f"draws policy {policy_d!r} does not carry _draws10_t1 — stop")
    if policy_d != f"{policy_g}_draws10_t1":
        sys.exit(
            f"draws policy {policy_d!r} does not extend greedy policy "
            f"{policy_g!r} — arms are not the same checkpoint's reads",
        )
    if _ckpt_id(g_rep) != _ckpt_id(d_rep):
        sys.exit(
            f"checkpoint mismatch: greedy {_ckpt_id(g_rep)} vs draws "
            f"{_ckpt_id(d_rep)} — stop",
        )
    rows, subset = join_rows(g_npz, d_npz)
    for k in ["pred:state-copy", "pred:state-copy-norm"]:
        if not np.array_equal(g_npz[k][rows], d_npz[k], equal_nan=True):
            sys.exit(
                f"{k} columns differ between greedy and draws — execution drift, stop",
            )
    oracles = {
        "row_pairing": "subset-join on index (q4 fallback)"
        if subset
        else "full byte-match",
        "state_copy_byte_match": True,
        "state_copy_norm_byte_match": True,
        "ar_temperature": 1.0,
        "sample_draws": 10,
        "draws_policy": policy_d,
        "checkpoint": "/".join(_ckpt_id(d_rep)),
        "report_crosscheck": "both arms reproduced (|d| < 5e-3)",
    }

    # ---- reads 1-4 on the paired rows (draws-npz masks) ----
    truth, valid, core, w = bbr.masks(d_npz)
    repo = d_npz["repo_id"]
    err_g = np.abs(g_npz[g_key][rows] - truth)
    err_d = np.abs(d_npz[d_key] - truth)
    keep_chunk = (w.sum(axis=(1, 2)) > 0) & core
    keep_first = valid[:, 0] & core

    fr_g, _ = bbr.frame_mae(err_g, w)
    fr_d, _ = bbr.frame_mae(err_d, w)
    read1 = abr.paired_read(fr_d, fr_g, keep_chunk, repo)
    read1["definition"] = (
        "Delta_AR: paired per-frame chunk_mae, _draws10_t1 - greedy, core "
        "frames (negative = ensembling helps)"
    )
    mean = read1["mean"]
    ci_real = abr.ci_excludes_zero(read1["ci95"])

    pooled = {
        "greedy": {
            "chunk_mae": round(bbr.pooled_chunk(err_g, core, w), 4),
            "first_mae": round(bbr.pooled_first(err_g, valid, core), 4),
            "pred_key": g_key,
        },
        "draws10_t1": {
            "chunk_mae": round(bbr.pooled_chunk(err_d, core, w), 4),
            "first_mae": round(bbr.pooled_first(err_d, valid, core), 4),
            "pred_key": d_key,
        },
    }
    d_chunk = pooled["draws10_t1"]["chunk_mae"]

    read2 = {
        "delta_ar_mean": mean,
        "flow_teacher_gain": FLOW_GAIN,
        "ar_gain_smaller_than_flow": bool(abs(mean) < abs(FLOW_GAIN)),
        "definition": (
            "Delta_AR vs the flow teacher's draws1 6.6232 -> draws10 "
            f"{FLOW_BAND} gain ({FLOW_GAIN}) on the same panel — does AR "
            "ensemble like flow, or is greedy already the mean?"
        ),
    }
    read3 = {
        "draws10_t1_chunk_mae": d_chunk,
        "flow_draws10_band": FLOW_BAND,
        "reaches_band": bool(d_chunk <= FLOW_BAND),
        "anchors_record_only": {
            "teacher_draws10_heun30": FLOW_HEUN30,
            "snapflow_student_mean10": SNAPFLOW_MEAN10,
        },
        "subset_mode": subset,
        "note": (
            "q4-subset re-pool — the band is a full-panel number, quoted directionally"
            if subset
            else "full panel"
        ),
    }
    read4_paired = abr.paired_read(
        abr.first_rows(err_d),
        abr.first_rows(err_g),
        keep_first,
        repo,
    )
    read4_paired["definition"] = "first_mae mirror of read 1 (record-only)"
    read4 = {
        "delta_ar_first": read4_paired,
        "flow_first_gain_descriptive": FLOW_FIRST_GAIN,
        "flow_first_band_descriptive": FLOW_HEUN30[1],
        "draws10_t1_first_mae": pooled["draws10_t1"]["first_mae"],
    }

    # ---- expectations E1-E4 (frozen) + assembly ----
    falsified = mean > FALSIFIER
    expectations = {
        "e1_delta_negative": {"met": bool(mean < 0), "ci95_excludes_zero": ci_real},
        "e2_gain_smaller_than_flow": {"met": bool(abs(mean) < abs(FLOW_GAIN))},
        "e3_no_overtake_of_flow_band": {"met": bool(d_chunk > FLOW_BAND)},
        "e4_falsifier": {
            "rule": f"falsified if Delta_AR > +{FALSIFIER}",
            "falsified": bool(falsified),
        },
    }
    lines = []
    if falsified:
        lines.append(
            f"FALSIFIED (E4): Delta_AR {mean} > +{FALSIFIER} — sampling + "
            "averaging actively hurts the AR family at T=1.0; the "
            "mean-of-samples premise fails; the instrument retires to "
            "diagnostic use (no temperature fishing beyond the "
            "pre-registered sensitivity rung)",
        )
    elif mean < 0 and ci_real:
        if abs(mean) < abs(FLOW_GAIN):
            lines.append(
                "E1 MET + E2 MET: Delta_AR < 0 with CI95 excluding 0 and "
                f"|Delta_AR| < {abs(FLOW_GAIN)} — the AR gain is real but "
                "SMALLER than the flow teacher's; greedy decode already "
                "sits near the predictive mean (the mean-collapse shape)",
            )
        else:
            lines.append(
                f"E1 MET, E2 NOT MET: |Delta_AR| >= {abs(FLOW_GAIN)} — the "
                "AR family ensembles like flow; draw diversity survives; "
                f"the {AR_ANCHOR[0]} deployment anchor understates the family",
            )
    elif mean < 0:
        lines.append(
            "E1 direction only: Delta_AR < 0 but CI95 contains 0 — the "
            "ensembling gain is within pairing noise",
        )
    else:
        lines.append(
            f"E1 NOT MET: Delta_AR >= 0 (inside the +{FALSIFIER} falsifier "
            "band) — no ensembling gain; greedy is already the "
            "mean-of-samples",
        )
    if d_chunk <= FLOW_BAND:
        lines.append(
            f"E3 VIOLATED: _draws10_t1 {d_chunk} reaches the flow draws10 "
            f"band {FLOW_BAND} — both families hold mean-of-10 reads and "
            "the symmetric comparison flips"
            + (" (q4 subset — directional)" if subset else ""),
        )
    else:
        lines.append(
            f"E3 as expected: _draws10_t1 {d_chunk} does not overtake the "
            f"flow draws10 band {FLOW_BAND}"
            + (" (q4 subset — directional)" if subset else ""),
        )

    out = {
        "arms_pooled": pooled,
        "subset_mode": subset,
        "n_rows_paired": len(rows),
        "read1_delta_ar": read1,
        "read2_fairness_vs_flow": read2,
        "read3_family_band": read3,
        "read4_first_mirrors": read4,
        "read5_execution_oracles": oracles,
        "expectations": expectations,
        "decision": {"falsified": bool(falsified), "assembly": lines},
    }
    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=1))
        print(f"wrote {out_path}")
    print(json.dumps(out, indent=1))
    print(
        f"Delta_AR {mean} CI {read1['ci95']} | flow gain {FLOW_GAIN} | "
        f"draws10_t1 {d_chunk} vs band {FLOW_BAND} | falsifier +{FALSIFIER}",
    )
    for line in lines:
        print(f"VERDICT: {line}")
    return out


# ---- oracle -----------------------------------------------------------------


def _fixture_report(d: dict, key: str, plan: str, base_rep: dict) -> dict:
    """A report dict for a fabricated npz, self-consistent with THIS
    file's pooling (the abort path is exercised with a drifted copy)."""
    truth, valid, core, w = bbr.masks(d)
    err = np.abs(d[key] - truth)
    return {
        "sample_plan": plan,
        "core_frames": PLAN_COUNTS[plan][0],
        "labeled_frames": PLAN_COUNTS[plan][1],
        "ar_temperature": 1.0,
        "sample_draws": 10,
        "checkpoint": base_rep.get("checkpoint"),
        "summaries": [
            {
                "policy": key.removeprefix("pred:"),
                "chunk_mae": bbr.pooled_chunk(err, core, w),
                "first_mae": bbr.pooled_first(err, valid, core),
            },
        ],
    }


def _synth(g_npz: dict, g_key: str, factor: float) -> tuple[dict, str]:
    """Draws fixture: err exactly factor x greedy's err, renamed key."""
    truth, pred_g = g_npz["truth"], g_npz[g_key]
    d_key = f"{g_key}_draws10_t1"
    out = abr._DictNpz({k: g_npz[k] for k in g_npz.files})
    del out[g_key]
    if f"{g_key}+fields" in out:
        del out[f"{g_key}+fields"]
    out[d_key] = np.where(
        np.isfinite(truth),
        truth + (pred_g - truth) * factor,
        pred_g,
    )
    return out, d_key


def _expect_exit(fn: Callable[[], object], needle: str, label: str) -> None:
    try:
        fn()
        raise AssertionError(f"{label}: expected abort not raised")
    except SystemExit as e:
        assert needle in str(e), f"{label}: wrong abort message: {e}"
    print(f"oracle abort OK: {label}")


def oracle() -> None:
    g_npz = np.load(f"{GREEDY_STEM}.npz", allow_pickle=True)
    g_rep = json.loads(Path(f"{GREEDY_STEM}.json").read_text())
    g_key = bare_key(g_npz, "greedy")

    # (a) anchors through THIS file's pooling + quoted-constant sources
    truth, valid, core, w = bbr.masks(g_npz)
    err = np.abs(g_npz[g_key] - truth)
    gc = bbr.pooled_chunk(err, core, w)
    gf = bbr.pooled_first(err, valid, core)
    assert abs(gc - AR_ANCHOR[0]) < SUMMARY_TOL, f"AR anchor FAIL {gc:.4f}"
    assert abs(gf - AR_ANCHOR[1]) < SUMMARY_TOL, f"AR first anchor FAIL {gf:.4f}"
    report_crosscheck(g_npz, g_key, g_rep, "oracle-greedy")
    heun30 = json.loads(
        Path(
            "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000"
            "__panel_curated_v0_k4l2_draws10_heun30.json",
        ).read_text(),
    )
    row = [s for s in heun30["summaries"] if s["policy"] == "bijou@80000_draws10"]
    assert len(row) == 1
    assert abs(row[0]["chunk_mae"] - FLOW_HEUN30[0]) < SUMMARY_TOL
    assert abs(row[0]["first_mae"] - FLOW_HEUN30[1]) < SUMMARY_TOL
    print(f"oracle (a) OK: AR anchor {gc:.4f}/{gf:.4f}; flow heun30 constants banked")

    fr_g, _ = bbr.frame_mae(err, w)
    keep = (w.sum(axis=(1, 2)) > 0) & core
    base_frame_mae = float(fr_g[keep].mean())

    # (b) degenerate draws := greedy -> exact zero, within-noise branch
    d_npz, d_key = _synth(g_npz, g_key, 1.0)
    d_rep = _fixture_report(d_npz, d_key, V1_PLAN, g_rep)
    res = analyze(g_npz, g_key, g_rep, d_npz, d_key, d_rep, None)
    r1 = res["read1_delta_ar"]
    assert r1["mean"] == 0.0 and r1["ci95"] == [0.0, 0.0]
    assert not res["decision"]["falsified"]
    assert "E1 NOT MET" in res["decision"]["assembly"][0]
    assert res["expectations"]["e3_no_overtake_of_flow_band"]["met"]
    print("oracle (b) OK: degenerate draws:=greedy -> zero delta, within noise")

    # (c) synthetic known effects
    for factor, want in [
        (0.95, "E1 MET + E2 MET"),  # real gain, smaller than flow's
        (1.005, "E1 NOT MET"),  # tiny positive, under the falsifier
        (1.05, "FALSIFIED (E4)"),  # the falsifier line
        (0.75, "E2 NOT MET"),  # ensembles like flow
    ]:
        d_npz, d_key = _synth(g_npz, g_key, factor)
        d_rep = _fixture_report(d_npz, d_key, V1_PLAN, g_rep)
        res = analyze(g_npz, g_key, g_rep, d_npz, d_key, d_rep, None)
        assert want in res["decision"]["assembly"][0], f"x{factor}: wrong branch"
        want_delta = (factor - 1.0) * base_frame_mae
        got = res["read1_delta_ar"]["mean"]
        assert abs(got - want_delta) < SUMMARY_TOL, f"x{factor} magnitude off: {got}"
    d_npz, d_key = _synth(g_npz, g_key, 0.90)  # pooled 0.9*5.8026 < 5.365
    d_rep = _fixture_report(d_npz, d_key, V1_PLAN, g_rep)
    res = analyze(g_npz, g_key, g_rep, d_npz, d_key, d_rep, None)
    assert not res["expectations"]["e3_no_overtake_of_flow_band"]["met"]
    assert any("E3 VIOLATED" in ln for ln in res["decision"]["assembly"])
    print("oracle (c) OK: x0.95/x1.005/x1.05/x0.75 branches + x0.90 E3 overtake")

    # (d) pairing aborts
    d_npz, d_key = _synth(g_npz, g_key, 0.95)
    d_rep = _fixture_report(d_npz, d_key, V1_PLAN, g_rep)
    bad = abr._DictNpz(dict(d_npz))
    idx = np.array(bad["index"])
    idx[:2] = idx[:2][::-1]
    bad["index"] = idx
    _expect_exit(
        lambda: analyze(g_npz, g_key, g_rep, bad, d_key, d_rep, None),
        "pairing broken",
        "misaligned equal-length index",
    )
    sub = abr._DictNpz({k: d_npz[k][::3] for k in d_npz.files})
    dup = abr._DictNpz(dict(sub))
    idx = np.array(dup["index"])
    idx[1] = idx[0]
    dup["index"] = idx
    _expect_exit(
        lambda: analyze(g_npz, g_key, g_rep, dup, d_key, d_rep, None),
        "duplicate index",
        "duplicated subset index",
    )
    alien = abr._DictNpz(dict(sub))
    idx = np.array(alien["index"])
    idx[0] = int(g_npz["index"].max()) + 1
    alien["index"] = idx
    _expect_exit(
        lambda: analyze(g_npz, g_key, g_rep, alien, d_key, d_rep, None),
        "absent from the greedy panel",
        "missing subset index",
    )

    # (e) execution-oracle aborts
    drift = abr._DictNpz(dict(d_npz))
    sc = np.array(drift["pred:state-copy"])
    sc[np.isfinite(sc)] += 1e-3
    drift["pred:state-copy"] = sc
    _expect_exit(
        lambda: analyze(g_npz, g_key, g_rep, drift, d_key, d_rep, None),
        "execution drift",
        "corrupted state-copy column",
    )
    for field, value in [
        ("ar_temperature", 0.7),
        ("sample_draws", 5),
        ("core_frames", 17203),
    ]:
        wrong = dict(d_rep)
        wrong[field] = value
        _expect_exit(
            lambda wrong=wrong: load_draws_report(wrong, "bad report"),
            "refusing the read",
            f"wrong {field}",
        )
    wrong = dict(d_rep)
    wrong["sample_plan"] = "plans/other.json"
    _expect_exit(
        lambda: load_draws_report(wrong, "bad report"),
        "refusing the read",
        "unregistered plan",
    )
    greedy_as_draws = abr._DictNpz(dict(g_npz))
    _expect_exit(
        lambda: analyze(g_npz, g_key, g_rep, greedy_as_draws, g_key, d_rep, None),
        "_draws10_t1",
        "policy missing _draws10_t1",
    )
    wrong = dict(d_rep)
    wrong["checkpoint"] = "outputs/train/other_run/step_000001"
    _expect_exit(
        lambda: analyze(g_npz, g_key, g_rep, d_npz, d_key, wrong, None),
        "checkpoint mismatch",
        "checkpoint mismatch",
    )
    drifted_rep = json.loads(json.dumps(d_rep))
    drifted_rep["summaries"][0]["chunk_mae"] += 0.1
    _expect_exit(
        lambda: report_crosscheck(d_npz, d_key, drifted_rep, "drifted"),
        "plan/scoring drift",
        "report-value drift",
    )

    # (f) subset path: q4-shaped slice joins, re-pools, reproduces direct
    core_rows = np.flatnonzero(g_npz["core"])[: PLAN_COUNTS[Q4_PLAN][0]]
    d_npz, d_key = _synth(g_npz, g_key, 0.95)
    sub = abr._DictNpz({k: d_npz[k][core_rows] for k in d_npz.files})
    sub_rep = _fixture_report(sub, d_key, Q4_PLAN, g_rep)
    res = analyze(g_npz, g_key, g_rep, sub, d_key, sub_rep, None)
    assert res["subset_mode"] and res["n_rows_paired"] == len(core_rows)
    t_s, _v_s, c_s, w_s = bbr.masks(sub)
    err_gs = np.abs(g_npz[g_key][core_rows] - t_s)
    fr_gs, _ = bbr.frame_mae(err_gs, w_s)
    keep_s = (w_s.sum(axis=(1, 2)) > 0) & c_s
    want_delta = -0.05 * float(fr_gs[keep_s].mean())
    got = res["read1_delta_ar"]["mean"]
    assert abs(got - want_delta) < SUMMARY_TOL, f"subset magnitude off: {got}"
    assert "directional" in res["decision"]["assembly"][-1]
    print("oracle (f) OK: q4-shaped subset joins + reproduces the direct read")
    print("ORACLE PASS: all draws10_t1 read branches verified pre-data")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--draws",
        nargs=2,
        metavar=("NPZ", "JSON"),
        default=[f"{DRAWS_STEM}.npz", f"{DRAWS_STEM}.json"],
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
    for path in [*a.draws, *a.greedy]:
        if not Path(path).exists():
            sys.exit(f"missing input {path} — eval not finished / not rsynced?")
    g_npz = np.load(a.greedy[0], allow_pickle=True)
    g_rep = json.loads(Path(a.greedy[1]).read_text())
    d_npz = np.load(a.draws[0], allow_pickle=True)
    d_rep = load_draws_report(a.draws[1], "draws report")
    g_key = bare_key(g_npz, "greedy")
    d_key = bare_key(d_npz, "draws")
    report_crosscheck(g_npz, g_key, g_rep, "greedy")
    report_crosscheck(d_npz, d_key, d_rep, "draws")
    if d_key.removeprefix("pred:") == "bijou@100000_draws10_t1":
        # the AR-100k arm: the greedy side must be the anchor itself
        truth, _valid, core, w = bbr.masks(g_npz)
        err = np.abs(g_npz[g_key] - truth)
        gc = bbr.pooled_chunk(err, core, w)
        if abs(gc - AR_ANCHOR[0]) >= SUMMARY_TOL:
            sys.exit(f"greedy pooled {gc:.4f} != AR anchor {AR_ANCHOR[0]} — stop")
    analyze(g_npz, g_key, g_rep, d_npz, d_key, d_rep, a.out)


if __name__ == "__main__":
    main()
