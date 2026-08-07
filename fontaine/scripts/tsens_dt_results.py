"""#19 dT diagnostic table — T-sensitivity rungs vs the T=1.0 primary. RECORD-ONLY.

The pre-reg (posts/2026-08-06-prereg-ar-sampled-draws.md) registers one
sensitivity rung — T in {0.5, 0.7, 1.3}, draws 10, on the frozen q4
subset — as RECORD-ONLY: "quoted as a dT diagnostic, never a headline,
and never a license to re-pick T post hoc". This file is that quote:
ONE table of pooled chunk/first per T on the SAME q4 rows, the T=1.0
row re-pooled from the full-panel primary npz via the subset join.
There are NO decision branches here by design — no expectations, no
verdicts, no falsifiers; the primary read lives in
draws10_t1_results.py and stays there.

Audit (queue item idea19-tsens-dt-read): draws10_t1_results.py's
join_rows subset machinery + box_batch_results pooling are the
reusable core, but its loaders hard-pin ar_temperature 1.0 and the
_draws10_t1 policy suffix. The delta is a T-parameterized sibling
loader over the registered T set {0.5, 0.7, 1.0, 1.3} ONLY — an
unregistered T is a hard abort, not a parameter.

Guards (each failure is a hard abort):
  * unregistered T (anything outside {0.5, 0.7, 1.0, 1.3});
  * wrong plan (rungs must carry the frozen q4 plan, the primary the
    full v1 plan, with the registered frame counts);
  * wrong draws (sample_draws != 10) / wrong ar_temperature for the
    declared T;
  * policy/stem tag mismatch (npz policy key and file stem must carry
    _draws10_t{T:%g} for the declared T);
  * rung rows disagreeing with each other or joining the primary
    panel wrongly (join_rows: identity byte-match / duplicate /
    missing index);
  * state-copy execution drift vs the primary rows; checkpoint
    mismatch; report summaries not reproduced from the npz
    (|d| < 5e-3).

Oracle mode (--oracle, runs before any tsens data exists, on the
banked AR-100k greedy panel npz):
  (a) a synthetic T=1.0 rung fixture (the primary strictly row-sliced
      to q4-shaped rows) reproduces the primary's q4 re-pool EXACTLY
      (float-equal, delta exactly 0.0);
  (b) rung fixtures with known error scalings (x0.93 / x0.98 / x1.07)
      land in the table at exactly factor x the re-pooled primary,
      deltas matching;
  (c) every guard above fires on a purpose-built bad input.

Pure CPU, read-only on inputs, deterministic. Usage (defaults = the
eval_ar100k_tsens_q4_draws10.sh launcher's exact stems):
  python fontaine/scripts/tsens_dt_results.py \
      [--primary NPZ JSON] [--rung T NPZ JSON ...] [--out ...] [--oracle]
"""

import argparse
import importlib.util
import json
import re
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
abr = _sibling("arch_batch_results")
d10 = _sibling("draws10_t1_results")

RUN = "bijou_arb_rcond_100k_ddp4"
PRIMARY_STEM = f"reports/eval__{RUN}__step_100000__panel_k4l2_draws10_t1"
RUNG_STEM = f"reports/eval__{RUN}__step_100000__stateprobe_q4_draws10_t{{tag}}"
OUT_DEFAULT = "reports/analysis__tsens_dt_ar100k_q4.json"

REGISTERED_T = (0.5, 0.7, 1.0, 1.3)  # the pre-reg's full set; 1.0 = primary
RUNG_T = (0.5, 0.7, 1.3)  # the launcher's three rungs
PRIMARY_T = 1.0
V1_PLAN = d10.V1_PLAN
Q4_PLAN = d10.Q4_PLAN
SUMMARY_TOL = d10.SUMMARY_TOL


def t_tag(t: float) -> str:
    return f"{t:g}"  # 1.0 -> "1", 0.5 -> "0.5" — the policy-suffix format


def t_suffix(t: float) -> str:
    return f"_draws10_t{t_tag(t)}"


def check_registered_t(t: float, label: str) -> None:
    if t not in REGISTERED_T:
        sys.exit(
            f"{label}: T = {t} is not in the registered set "
            f"{list(REGISTERED_T)} — the pre-reg licenses no other "
            "temperature, refusing the read",
        )


def check_stem_tag(path: str | Path, t: float, label: str) -> None:
    """The file stem's _draws10_t<tag> must carry the declared T."""
    m = re.search(r"_draws10_t([0-9.]+)\.", Path(path).name)
    if not m or m.group(1) != t_tag(t):
        sys.exit(
            f"{label}: stem {Path(path).name!r} does not carry "
            f"{t_suffix(t)!r} for declared T = {t} — tag mismatch, stop",
        )


def load_report(src: str | Path | dict, t: float, plan: str, label: str) -> dict:
    """T-parameterized sibling of draws10_t1_results.load_draws_report:
    same registered-semantics pinning, ar_temperature pinned to the
    declared (registered) T instead of 1.0."""
    check_registered_t(t, label)
    d = src if isinstance(src, dict) else json.loads(Path(src).read_text())
    got_plan = d.get("sample_plan")
    if got_plan != plan:
        sys.exit(
            f"{label}: sample_plan = {got_plan!r}, this arm is registered "
            f"on {plan!r} — refusing the read",
        )
    want = {
        "core_frames": d10.PLAN_COUNTS[plan][0],
        "labeled_frames": d10.PLAN_COUNTS[plan][1],
        "ar_temperature": t,
        "sample_draws": 10,
    }
    for k, v in want.items():
        got = d.get(k)
        if got != v:
            sys.exit(f"{label}: {k} = {got!r}, registered {v!r} — refusing the read")
    return d


def _pool(npz: dict, key: str) -> tuple[float, float]:
    truth, valid, core, w = bbr.masks(npz)
    err = np.abs(npz[key] - truth)
    return bbr.pooled_chunk(err, core, w), bbr.pooled_first(err, valid, core)


def _slice(npz: dict, rows: np.ndarray) -> dict:
    return abr._DictNpz({k: npz[k][rows] for k in npz.files})


def analyze(
    p_npz: dict,
    p_key: str,
    p_rep: dict,
    rungs: list[tuple[float, dict, str, dict]],
    out_path: str | None,
) -> dict:
    # ---- guards (hard aborts, no decision branches beyond them) ----
    policy_p = p_key.removeprefix("pred:")
    if not policy_p.endswith(t_suffix(PRIMARY_T)):
        sys.exit(f"primary policy {policy_p!r} does not carry _draws10_t1 — stop")
    rows_ref: np.ndarray | None = None
    for t, r_npz, r_key, r_rep in rungs:
        check_registered_t(t, f"rung t{t_tag(t)}")
        policy_r = r_key.removeprefix("pred:")
        if not policy_r.endswith(t_suffix(t)):
            sys.exit(
                f"rung t{t_tag(t)}: policy {policy_r!r} does not carry "
                f"{t_suffix(t)!r} — tag mismatch, stop",
            )
        if d10._ckpt_id(r_rep) != d10._ckpt_id(p_rep):
            sys.exit(
                f"checkpoint mismatch: rung t{t_tag(t)} {d10._ckpt_id(r_rep)} "
                f"vs primary {d10._ckpt_id(p_rep)} — stop",
            )
        rows, subset = d10.join_rows(p_npz, r_npz)
        if not subset:
            sys.exit(
                f"rung t{t_tag(t)} is not a strict subset of the primary "
                "panel — the q4 rung has the wrong shape, stop",
            )
        if rows_ref is None:
            rows_ref = rows
        elif not np.array_equal(rows, rows_ref):
            sys.exit(
                f"rung t{t_tag(t)} disagrees with the other rungs on the "
                "q4 rows — the rungs are not the same frozen subset, stop",
            )
        for k in ["pred:state-copy", "pred:state-copy-norm"]:
            if not np.array_equal(p_npz[k][rows], r_npz[k], equal_nan=True):
                sys.exit(
                    f"{k} differs between primary and rung t{t_tag(t)} — "
                    "execution drift, stop",
                )
    assert rows_ref is not None

    # ---- the one table: pooled chunk/first per T on the same q4 rows ----
    sub_p = _slice(p_npz, rows_ref)
    ref_chunk, ref_first = _pool(sub_p, p_key)
    table = [
        {
            "t": PRIMARY_T,
            "chunk_mae": ref_chunk,
            "first_mae": ref_first,
            "delta_chunk_vs_t1": 0.0,
            "delta_first_vs_t1": 0.0,
            "source": "primary re-pool (full-panel npz, subset-joined onto q4)",
        },
    ]
    for t, r_npz, r_key, _r_rep in rungs:
        chunk, first = _pool(r_npz, r_key)
        table.append(
            {
                "t": t,
                "chunk_mae": chunk,
                "first_mae": first,
                "delta_chunk_vs_t1": chunk - ref_chunk,
                "delta_first_vs_t1": first - ref_first,
                "source": "q4 rung",
            },
        )
    table.sort(key=lambda r: r["t"])

    out = {
        "record_only": True,
        "note": (
            "pre-registered dT diagnostic (sensitivity clause): never a "
            "headline, never a license to re-pick T; the primary read is "
            "draws10_t1_results.py"
        ),
        "n_rows": len(rows_ref),
        "checkpoint": "/".join(d10._ckpt_id(p_rep)),
        "registered_t": list(REGISTERED_T),
        "dt_table": [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
            for row in table
        ],
    }
    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=1))
        print(f"wrote {out_path}")
    print(json.dumps(out, indent=1))
    print(f"dT table (record-only), {len(rows_ref)} q4 rows:")
    print("     T   chunk_mae  first_mae   dChunk   dFirst  source")
    for row in table:
        print(
            f"  {row['t']:4g}  {row['chunk_mae']:9.4f}  {row['first_mae']:9.4f}"
            f"  {row['delta_chunk_vs_t1']:+7.4f}  {row['delta_first_vs_t1']:+7.4f}"
            f"  {row['source']}",
        )
    return {**out, "dt_table_raw": table}


# ---- oracle -----------------------------------------------------------------


def _rung_fixture_report(npz: dict, key: str, t: float, base_rep: dict) -> dict:
    """Self-consistent q4 rung report for a fabricated npz (the drift
    abort is exercised with a corrupted copy)."""
    truth, valid, core, w = bbr.masks(npz)
    err = np.abs(npz[key] - truth)
    return {
        "sample_plan": Q4_PLAN,
        "core_frames": d10.PLAN_COUNTS[Q4_PLAN][0],
        "labeled_frames": d10.PLAN_COUNTS[Q4_PLAN][1],
        "ar_temperature": t,
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


def _rung_fixture(
    p_npz: dict,
    p_key: str,
    rows: np.ndarray,
    t: float,
    factor: float,
) -> tuple[dict, str]:
    """Rung fixture: the primary row-sliced to q4-shaped rows, err
    exactly factor x the primary's, policy re-tagged to the rung's T."""
    sub = _slice(p_npz, rows)
    truth, pred = sub["truth"], sub[p_key]
    key = p_key.removesuffix(t_suffix(PRIMARY_T)) + t_suffix(t)
    del sub[p_key]
    sub[key] = np.where(
        np.isfinite(truth),
        truth + (pred - truth) * factor,
        pred,
    )
    return sub, key


def oracle() -> None:
    g_npz = np.load(f"{d10.GREEDY_STEM}.npz", allow_pickle=True)
    g_rep = json.loads(Path(f"{d10.GREEDY_STEM}.json").read_text())
    g_key = d10.bare_key(g_npz, "greedy")
    p_npz, p_key = d10._synth(g_npz, g_key, 1.0)  # synthetic T=1.0 primary
    p_rep = d10._fixture_report(p_npz, p_key, V1_PLAN, g_rep)
    q4_rows = np.flatnonzero(g_npz["core"])[: d10.PLAN_COUNTS[Q4_PLAN][0]]

    # (a) a synthetic T=1.0 rung fixture reproduces the primary's q4
    # re-pool EXACTLY — the rung path and the subset-join re-pool path
    # are the same numbers, bit for bit.
    fx, fx_key = _rung_fixture(p_npz, p_key, q4_rows, PRIMARY_T, 1.0)
    fx_rep = _rung_fixture_report(fx, fx_key, PRIMARY_T, p_rep)
    res = analyze(p_npz, p_key, p_rep, [(PRIMARY_T, fx, fx_key, fx_rep)], None)
    ref, rung = res["dt_table_raw"][0], res["dt_table_raw"][1]
    assert ref["source"].startswith("primary re-pool") and rung["source"] == "q4 rung"
    assert rung["chunk_mae"] == ref["chunk_mae"], "rung path != re-pool path"
    assert rung["first_mae"] == ref["first_mae"], "rung first != re-pool first"
    assert rung["delta_chunk_vs_t1"] == 0.0 and rung["delta_first_vs_t1"] == 0.0
    assert res["n_rows"] == len(q4_rows)
    print("oracle (a) OK: T=1.0 rung fixture == primary q4 re-pool exactly")

    # (b) known error scalings land at exactly factor x the re-pool
    factors = {0.5: 0.93, 0.7: 0.98, 1.3: 1.07}
    rungs = []
    for t in RUNG_T:
        r_npz, r_key = _rung_fixture(p_npz, p_key, q4_rows, t, factors[t])
        rungs.append((t, r_npz, r_key, _rung_fixture_report(r_npz, r_key, t, p_rep)))
    res = analyze(p_npz, p_key, p_rep, rungs, None)
    by_t = {row["t"]: row for row in res["dt_table_raw"]}
    ref_chunk = by_t[PRIMARY_T]["chunk_mae"]
    ref_first = by_t[PRIMARY_T]["first_mae"]
    # fp32 npz arrays: the analytic factor x re-pool expectation carries
    # ~1e-6 rounding from truth + (pred - truth) * factor; 5e-5 is far
    # below any real pooling drift and far above that noise.
    for t, factor in factors.items():
        for field, ref_val in [("chunk_mae", ref_chunk), ("first_mae", ref_first)]:
            got, want = by_t[t][field], factor * ref_val
            assert abs(got - want) < 5e-5, f"t{t} {field}: {got} != {want}"
        d_got = by_t[t]["delta_chunk_vs_t1"]
        assert abs(d_got - (factor - 1.0) * ref_chunk) < 5e-5, f"t{t} delta off"
    assert [row["t"] for row in res["dt_table_raw"]] == [0.5, 0.7, 1.0, 1.3]
    print("oracle (b) OK: x0.93/x0.98/x1.07 rungs land at exactly factor x re-pool")

    # (c) guards
    good = rungs[0]  # (0.5, npz, key, rep)
    d10._expect_exit(
        lambda: check_registered_t(0.9, "rung t0.9"),
        "not in the registered set",
        "unregistered T",
    )
    d10._expect_exit(
        lambda: load_report(dict(good[3], sample_plan=V1_PLAN), 0.5, Q4_PLAN, "bad"),
        "refusing the read",
        "wrong plan (rung carrying the v1 plan)",
    )
    d10._expect_exit(
        lambda: load_report(dict(good[3], sample_draws=5), 0.5, Q4_PLAN, "bad"),
        "refusing the read",
        "wrong draws",
    )
    d10._expect_exit(
        lambda: load_report(dict(good[3]), 0.7, Q4_PLAN, "bad"),
        "refusing the read",
        "ar_temperature not matching the declared T",
    )
    d10._expect_exit(
        lambda: analyze(p_npz, p_key, p_rep, [(0.7, good[1], good[2], good[3])], None),
        "tag mismatch",
        "policy tag not matching the declared T",
    )
    d10._expect_exit(
        lambda: check_stem_tag("reports/x_draws10_t0.5.npz", 0.7, "stem"),
        "tag mismatch",
        "stem tag not matching the declared T",
    )
    other_rows = np.flatnonzero(g_npz["core"])[1 : len(q4_rows) + 1]
    r2_npz, r2_key = _rung_fixture(p_npz, p_key, other_rows, 0.7, 0.98)
    r2 = (0.7, r2_npz, r2_key, _rung_fixture_report(r2_npz, r2_key, 0.7, p_rep))
    d10._expect_exit(
        lambda: analyze(p_npz, p_key, p_rep, [good, r2], None),
        "not the same frozen subset",
        "rungs disagreeing on the q4 rows",
    )
    full, full_key = _rung_fixture(
        p_npz,
        p_key,
        np.arange(len(p_npz["index"])),
        0.5,
        1.0,
    )
    full_rep = _rung_fixture_report(full, full_key, 0.5, p_rep)
    d10._expect_exit(
        lambda: analyze(p_npz, p_key, p_rep, [(0.5, full, full_key, full_rep)], None),
        "strict subset",
        "full-panel npz passed as a q4 rung",
    )
    drift = abr._DictNpz(dict(good[1]))
    sc = np.array(drift["pred:state-copy"])
    sc[np.isfinite(sc)] += 1e-3
    drift["pred:state-copy"] = sc
    d10._expect_exit(
        lambda: analyze(p_npz, p_key, p_rep, [(0.5, drift, good[2], good[3])], None),
        "execution drift",
        "corrupted state-copy column",
    )
    wrong_ck = dict(good[3], checkpoint="outputs/train/other_run/step_000001")
    d10._expect_exit(
        lambda: analyze(p_npz, p_key, p_rep, [(0.5, good[1], good[2], wrong_ck)], None),
        "checkpoint mismatch",
        "checkpoint mismatch",
    )
    drifted_rep = json.loads(json.dumps(good[3]))
    drifted_rep["summaries"][0]["chunk_mae"] += 0.1
    d10._expect_exit(
        lambda: d10.report_crosscheck(good[1], good[2], drifted_rep, "drifted"),
        "plan/scoring drift",
        "report-value drift",
    )
    print("ORACLE PASS: dT-table rung path, deltas, and all guards verified pre-data")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--primary",
        nargs=2,
        metavar=("NPZ", "JSON"),
        default=[f"{PRIMARY_STEM}.npz", f"{PRIMARY_STEM}.json"],
    )
    p.add_argument(
        "--rung",
        nargs=3,
        metavar=("T", "NPZ", "JSON"),
        action="append",
        default=None,
        help="repeatable; default = the launcher's three q4 rung stems",
    )
    p.add_argument("--out", default=OUT_DEFAULT)
    p.add_argument("--oracle", action="store_true")
    a = p.parse_args()
    if a.oracle:
        oracle()
        return
    rung_args = a.rung or [
        [
            t_tag(t),
            RUNG_STEM.format(tag=t_tag(t)) + ".npz",
            RUNG_STEM.format(tag=t_tag(t)) + ".json",
        ]
        for t in RUNG_T
    ]
    paths = [*a.primary, *[q for r in rung_args for q in r[1:]]]
    for path in paths:
        if not Path(path).exists():
            sys.exit(f"missing input {path} — rung not finished / not rsynced?")
    check_stem_tag(a.primary[0], PRIMARY_T, "primary npz")
    check_stem_tag(a.primary[1], PRIMARY_T, "primary json")
    p_npz = np.load(a.primary[0], allow_pickle=True)
    p_rep = load_report(a.primary[1], PRIMARY_T, V1_PLAN, "primary report")
    p_key = d10.bare_key(p_npz, "primary")
    d10.report_crosscheck(p_npz, p_key, p_rep, "primary")
    rungs = []
    for t_str, npz_path, json_path in rung_args:
        t = float(t_str)
        label = f"rung t{t_tag(t)}"
        check_registered_t(t, label)
        if t == PRIMARY_T:
            sys.exit("T = 1.0 is the primary re-pool, not a rung — stop")
        check_stem_tag(npz_path, t, f"{label} npz")
        check_stem_tag(json_path, t, f"{label} json")
        r_npz = np.load(npz_path, allow_pickle=True)
        r_rep = load_report(json_path, t, Q4_PLAN, f"{label} report")
        r_key = d10.bare_key(r_npz, label)
        d10.report_crosscheck(r_npz, r_key, r_rep, label)
        rungs.append((t, r_npz, r_key, r_rep))
    analyze(p_npz, p_key, p_rep, rungs, a.out)


if __name__ == "__main__":
    main()
