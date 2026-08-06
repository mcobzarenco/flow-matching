"""SnapFlow distill endpoint results — the pre-registered reads, ready before the data.

Implements exactly the frozen reads of the SnapFlow pre-reg
(posts/2026-08-06-prereg-snapflow-distill.md + Amendment 1, ideas #12):

  * PROBE (@10k, record-only): 1-NFE stride-7 probe chunk_mae vs the
    kill line = teacher Heun-30 probe read 6.6755 + 3.0 = 9.6755
    (catastrophic non-convergence only; > is strict).
  * READ 1 (primary): endpoint 1-NFE single-draw panel chunk_mae.
      adopt-signal iff <= 6.7732        (6.6232 + max(3*sigma_draw, 0.15);
                                         Amendment 1: the 0.15 floor binds)
      falsified   iff >  7.1232         (expectation 2: 6.6232 + 0.5 —
                                         SnapFlow does not transfer)
      between the two: miss-but-not-falsified, banked as a partial
      negative with the probe curve + per-step read explaining why.
  * READ 2 (grounding edge): endpoint 1-NFE first_mae <= 1.9831
    (teacher 1.9331 + 0.05, expectation 4).
  * READ 3 (deployment headline): mean-of-10 @1-NFE chunk_mae <= 5.8026
    (the AR-100k anchor; expectation 3, modal band 5.4-5.6 descriptive).
    draws5 co-read descriptive (teacher Heun-30: draws10 5.3645/1.4242,
    draws5 5.5235/1.4985).
  * PER-STEP HORIZON READ (paired-analysis protocol, ships with the
    results post): student endpoint npz vs the banked teacher panel npz
    (index-keyed v1), per-step + first-k curves with crossover steps —
    a distill that fixes only late-horizon must not be misread at
    pooled chunk_mae alone. Requires the npz-dump addendum eval
    (eval_snapdistill_endpoint_1nfe_npz.sh) — the chained stage-4 evals
    dump JSON only. The v2 column (panel-v2 keep-mask re-pool) rides on
    the same npz, quoted alongside per the transition convention.

All registered comparators are INDEX-keyed (the pre-reg predates the
#18.2 stable-key adoption; in-flight pre-registered reads finish as
registered) — every endpoint JSON must record noise_key == 'index',
and the stable-key teacher anchor 6.5997/1.9355 is quoted descriptively
only. Semantics guards on every new JSON: sample_steps == 1,
sample_method == 'euler', target_time == 'zero', sample_draws == N,
the registered plan path, and the full core frame count.

Oracle mode (--oracle, run before any endpoint data existed):
  (a) banked-JSON extraction reproduces teacher panel 6.6232/1.9331,
      draws10 5.3645/1.4242, draws5 5.5235/1.4985, AR 5.8026/2.1431,
      teacher probe 6.6755 exactly;
  (b) teacher npz through this file's pooling reproduces 6.6232/1.9331,
      the step curve byte-matches the banked flow-vs-AR analysis JSON,
      the v2 keep-mask re-pool reproduces the banked v2 anchor
      6.7151/1.9453, and the degenerate self-pair gives all-zero
      per-step deltas with no crossover;
  (c) synthetic endpoint JSONs drive every verdict branch, both band
      edges inclusive (6.7732 adopts, 7.1232 is not falsified), the
      probe kill boundary (9.6755 survives, above it kills);
  (d) semantics guards die loud (heun/30-step/wrong-draws/stable-key/
      subset-frames JSONs are all refused);
  (e) synthetic late-horizon student (errors inflated from step 25 on)
      -> early per-step deltas 0, crossover at exactly 25.

Pooling semantics are byte-identical to box_batch_results.py (sibling
import); curve semantics byte-identical to flow_vs_ar_paired.py
(oracled against its banked output). Pure CPU, read-only on inputs.

Usage (defaults = the chained stage-4 output names + the addendum npz):
  python fontaine/scripts/snapflow_results.py \
      [--endpoint1 JSON] [--endpoint10 JSON] [--endpoint5 JSON] \
      [--probe JSON] [--npz NPZ] [--out ...]
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
dai = _sibling("dup_census_anchor_impact")

RUN = "fontaine_flow_snapdistill_h1024_30k_1xh100"
STEM = f"reports/eval__{RUN}__step_030000__panel_curated_v0_k4l2"
E1_JSON = f"{STEM}_1nfe_euler1.json"
E10_JSON = f"{STEM}_1nfe_euler1_draws10.json"
E5_JSON = f"{STEM}_1nfe_euler1_draws5.json"
PROBE_JSON = "reports/eval__snapdistill__step_010000__probe_s7_1nfe_euler1.json"
NPZ_DEFAULT = f"{STEM}_1nfe_euler1_npz.npz"
NPZ_JSON_DEFAULT = f"{STEM}_1nfe_euler1_npz.json"
TEACHER_NPZ = (
    "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000"
    "__panel_k4l2_heun30.npz"
)
V1_PLAN = "plans/holdout_curated_v0_k4l2.json"
V2_PLAN = "plans/holdout_curated_v0_k4l2_panel_v2.json"
PROBE_PLAN = "plans/holdout_curated_v0_k4l2_drawsprobe_s7.json"
FVAP_JSON = "reports/analysis__flow_vs_ar_paired_k4l2.json"
OUT_DEFAULT = "reports/analysis__snapflow_distill_30k_k4l2.json"

# Banked comparators — ALL index-keyed v1 (the registered keying)
TEACHER_CHUNK, TEACHER_FIRST = 6.6232, 1.9331
TEACHER_PROBE_CHUNK = 6.6755  # step-0 extended s=t, drift-gated == teacher
TEACHER_D10 = (5.3645, 1.4242)
TEACHER_D5 = (5.5235, 1.4985)
AR_CHUNK, AR_FIRST = 5.8026, 2.1431
STABLE_TEACHER = (6.5997, 1.9355)  # descriptive only: DIFFERENT keying
V2_TEACHER = (6.7151, 1.9453)  # banked v2 re-pool of the teacher npz

# Frozen decision lines (pre-reg + Amendment 1)
ADOPT_LINE = 6.7732  # R1: adopt-signal iff chunk <= (0.15 floor binds)
FALSIFY_LINE = 7.1232  # R1: falsified iff chunk >  (teacher + 0.5)
EDGE_LINE = 1.9831  # R2: grounding edge iff first <= (teacher + 0.05)
DEPLOY_LINE = 5.8026  # R3: deployment iff mean-of-10 <= (the AR anchor)
KILL_LINE = TEACHER_PROBE_CHUNK + 3.0  # probe: killed iff strictly >
MODAL_D10 = (5.4, 5.6)  # expectation 3 modal band, descriptive
PANEL_CORE = 17204
PROBE_CORE = 2458
SUMMARY_TOL = 5e-3


def _as_dict(src: dict | str | Path) -> dict:
    if isinstance(src, dict):
        return src
    return json.loads(Path(src).read_text())


def bijou_row(d: dict, label: str) -> dict:
    """The bare headline row — the sibling instruments' banked_bare_key
    rule mapped to report JSONs (the narrated +fields pass and any
    masked diagnostic must never be picked up as the headline)."""
    rows = [
        s
        for s in d["summaries"]
        if s["policy"].startswith("bijou@")
        and "+" not in s["policy"]
        and "masked" not in s["policy"]
    ]
    if len(rows) != 1:
        sys.exit(
            f"{label}: expected exactly one bare bijou@ policy row, got "
            f"{[s['policy'] for s in rows]}",
        )
    return rows[0]


def load_banked(src: dict | str | Path, label: str) -> tuple[float, float]:
    """Lenient extraction for BANKED comparator JSONs (pre-hardening files
    may lack the scoring-semantics fields)."""
    row = bijou_row(_as_dict(src), label)
    return float(row["chunk_mae"]), float(row["first_mae"])


def load_endpoint(src: dict | str | Path, draws: int, label: str) -> dict:
    """Strict loader for NEW SnapFlow endpoint JSONs: every scoring
    semantic the registration fixed must be recorded, or we stop."""
    d = _as_dict(src)
    want = {
        "sample_steps": 1,
        "sample_method": "euler",
        "target_time": "zero",
        "sample_draws": draws,
        "noise_key": "index",
        "sample_plan": V1_PLAN,
    }
    for k, v in want.items():
        got = d.get(k)
        if got != v:
            sys.exit(f"{label}: {k} = {got!r}, registered {v!r} — refusing the read")
    if d.get("core_frames") != PANEL_CORE:
        sys.exit(
            f"{label}: core_frames {d.get('core_frames')} != panel {PANEL_CORE} "
            "— not the registered full-panel eval",
        )
    row = bijou_row(d, label)
    if draws > 1 and not row["policy"].endswith(f"_draws{draws}"):
        sys.exit(f"{label}: policy {row['policy']!r} lacks _draws{draws} suffix")
    return {
        "policy": row["policy"],
        "chunk_mae": float(row["chunk_mae"]),
        "first_mae": float(row["first_mae"]),
    }


def load_probe(src: dict | str | Path, label: str) -> dict:
    d = _as_dict(src)
    want = {
        "sample_steps": 1,
        "sample_method": "euler",
        "target_time": "zero",
        "sample_draws": 1,
        "noise_key": "index",
        "sample_plan": PROBE_PLAN,
    }
    for k, v in want.items():
        got = d.get(k)
        if got != v:
            sys.exit(f"{label}: {k} = {got!r}, registered {v!r} — refusing the read")
    if d.get("core_frames") != PROBE_CORE:
        sys.exit(f"{label}: core_frames {d.get('core_frames')} != {PROBE_CORE}")
    row = bijou_row(d, label)
    return {"chunk_mae": float(row["chunk_mae"]), "first_mae": float(row["first_mae"])}


def classify_primary(chunk: float) -> str:
    if chunk <= ADOPT_LINE:
        return "parity-adopt"
    if chunk <= FALSIFY_LINE:
        return "intermediate-miss"
    return "falsified"


def verdict(
    primary_class: str,
    *,
    edge_ok: bool,
    deploy_ok: bool,
    probe_killed: bool | None,
) -> dict:
    """The frozen decision assembly. Every line is pre-registered; the
    combinations only compose the registered reads."""
    if primary_class == "falsified":
        primary = (
            f"FALSIFIED: 1-NFE chunk_mae > {FALSIFY_LINE} (teacher + 0.5) — "
            "SnapFlow does not transfer to this lineage; banked as a negative "
            "with the @10k/endpoint probe curve + per-step read explaining why"
        )
        adoption = "NO ADOPTION"
    elif primary_class == "intermediate-miss":
        primary = (
            f"MISS (not falsified): chunk_mae in ({ADOPT_LINE}, {FALSIFY_LINE}] — "
            "no adopt-signal; partial negative, probe curve + per-step read "
            "carry the diagnosis"
        )
        adoption = "NO ADOPTION"
    else:
        primary = (
            f"PARITY: 1-NFE chunk_mae <= {ADOPT_LINE} — the pre-registered "
            "adopt-signal fires"
        )
        adoption = (
            "ADOPT-SIGNAL + DEPLOYMENT HEADLINE: mean-of-10 @1-NFE beats the AR "
            "anchor at ~10-expert-eval cost — the charter §2 cost caveat on the "
            "draws win closes (results post + owner adoption decision)"
            if deploy_ok
            else "ADOPT-SIGNAL (single-draw parity) but mean-of-10 misses the AR "
            "line — the draws win did NOT survive distillation at N=10; the "
            "deployment claim stays with the Heun-30 teacher"
        )
    if probe_killed:
        # Record-only line, but if it fired the run should already be dead;
        # reads over a post-kill checkpoint are descriptive only.
        adoption = (
            f"KILL LINE FIRED @10k (probe > {KILL_LINE}): catastrophic "
            "non-convergence — endpoint reads are descriptive, no adoption path"
        )
    return {
        "primary_class": primary_class,
        "primary_verdict": primary,
        "grounding_edge_survives": bool(edge_ok),
        "deployment_headline_ok": bool(deploy_ok),
        "probe_killed": probe_killed,
        "adoption": adoption,
    }


# ---- per-step horizon read (flow_vs_ar_paired.py protocol, verbatim) ----


def step_curve(e: np.ndarray, valid: np.ndarray, core: np.ndarray) -> list[float]:
    wv = (valid & core[:, None]).astype(np.float64)
    num = (e.sum(axis=2) * wv).sum(axis=0)
    den = wv.sum(axis=0) * e.shape[2]
    return (num / np.maximum(den, 1)).tolist()


def firstk_curve(e: np.ndarray, valid: np.ndarray, core: np.ndarray) -> list[float]:
    wv = (valid & core[:, None]).astype(np.float64)
    num = np.cumsum((e.sum(axis=2) * wv).sum(axis=0))
    den = np.cumsum(wv.sum(axis=0) * e.shape[2])
    return (num / np.maximum(den, 1)).tolist()


def crossover(student: list[float], teacher: list[float]) -> int | None:
    return next(
        (i for i, (s, t) in enumerate(zip(student, teacher, strict=True)) if s > t),
        None,
    )


def v2_keep_mask(v2_plan: dict, join: np.ndarray) -> np.ndarray:
    ex = v2_plan["exclusions"]
    leaked = {
        (k.rsplit("::", 1)[0], int(k.rsplit("::", 1)[1])) for k in ex["leaked_episodes"]
    }
    corrupt = set(ex["corrupt_repos"])
    return np.fromiter(
        (
            (r, int(e)) not in leaked and r not in corrupt
            for r, e in zip(join["repo"], join["episode"], strict=True)
        ),
        dtype=bool,
        count=len(join),
    )


def perstep_read(
    student: dict,
    s_key: str,
    teacher: dict,
    t_key: str,
    v1_plan: dict,
    v2_plan: dict | None,
) -> dict:
    for k in bbr.PAIR_KEYS:
        if not np.array_equal(student[k], teacher[k]):
            sys.exit(f"per-step pairing broken on {k} (student npz vs teacher npz)")
    truth, valid, core, w = bbr.masks(teacher)
    err_s = np.abs(student[s_key] - truth)
    err_t = np.abs(teacher[t_key] - truth)
    npz_pooled = {
        "chunk_mae": round(bbr.pooled_chunk(err_s, core, w), 4),
        "first_mae": round(bbr.pooled_first(err_s, valid, core), 4),
    }
    curves = {
        "student": step_curve(err_s, valid, core),
        "teacher": step_curve(err_t, valid, core),
    }
    fk = {
        "student": firstk_curve(err_s, valid, core),
        "teacher": firstk_curve(err_t, valid, core),
    }
    out = {
        "npz_pooled_student": npz_pooled,
        "step_curve": {n: [round(v, 4) for v in c] for n, c in curves.items()},
        "step_delta": [
            round(s - t, 4)
            for s, t in zip(curves["student"], curves["teacher"], strict=True)
        ],
        "crossover_step": crossover(curves["student"], curves["teacher"]),
        "firstk_curve": {n: [round(v, 4) for v in c] for n, c in fk.items()},
        "firstk_crossover": crossover(fk["student"], fk["teacher"]),
        "note": (
            "student = SnapFlow 1-NFE, teacher = Heun-30 (banked v1, "
            "index-keyed); crossover = first horizon step where the "
            "student's pooled MAE exceeds the teacher's"
        ),
    }
    if v2_plan is not None:
        join = dai.build_join(v1_plan, teacher)
        keep = v2_keep_mask(v2_plan, join)
        sel = core & keep
        out["v2_column"] = {
            "student": {
                "chunk_mae": round(bbr.pooled_chunk(err_s, sel, w), 4),
                "first_mae": round(bbr.pooled_first(err_s, valid, sel), 4),
            },
            "teacher_banked": {"chunk_mae": V2_TEACHER[0], "first_mae": V2_TEACHER[1]},
            "note": "descriptive per the panel-v2 transition convention "
            "(v1 stays the registered read)",
        }
    return out


def analyze(
    e1: dict,
    e10: dict,
    e5: dict | None,
    probe: dict | None,
    perstep: dict | None,
    out_path: str | None,
) -> dict:
    primary_class = classify_primary(e1["chunk_mae"])
    edge_ok = e1["first_mae"] <= EDGE_LINE
    deploy_ok = e10["chunk_mae"] <= DEPLOY_LINE
    probe_killed = None if probe is None else probe["chunk_mae"] > KILL_LINE
    dec = verdict(
        primary_class,
        edge_ok=edge_ok,
        deploy_ok=deploy_ok,
        probe_killed=probe_killed,
    )
    if perstep is not None:
        drift = abs(perstep["npz_pooled_student"]["chunk_mae"] - e1["chunk_mae"])
        if drift > SUMMARY_TOL:
            print(
                f"WARNING: npz re-run pooled chunk_mae drifts {drift:.4f} from the "
                "chained primary JSON — nondeterminism between identical evals; "
                "the chained JSON stays the registered read",
            )
        perstep["npz_vs_chain_drift"] = round(drift, 5)
    out = {
        "read1_primary_1nfe": {
            "chunk_mae": e1["chunk_mae"],
            "adopt_line": ADOPT_LINE,
            "falsify_line": FALSIFY_LINE,
            "teacher_chunk_index_key": TEACHER_CHUNK,
            "teacher_chunk_stable_key_descriptive": STABLE_TEACHER[0],
            "delta_vs_teacher": round(e1["chunk_mae"] - TEACHER_CHUNK, 4),
        },
        "read2_grounding_edge": {
            "first_mae": e1["first_mae"],
            "edge_line": EDGE_LINE,
            "teacher_first": TEACHER_FIRST,
            "survives": bool(edge_ok),
        },
        "read3_deployment": {
            "mean10_chunk_mae": e10["chunk_mae"],
            "mean10_first_mae": e10["first_mae"],
            "deploy_line_ar": DEPLOY_LINE,
            "modal_band": list(MODAL_D10),
            "inside_modal_band": bool(
                MODAL_D10[0] <= e10["chunk_mae"] <= MODAL_D10[1],
            ),
            "teacher_draws10_heun30": list(TEACHER_D10),
            "draws5": None
            if e5 is None
            else {
                "chunk_mae": e5["chunk_mae"],
                "first_mae": e5["first_mae"],
                "teacher_draws5_heun30": list(TEACHER_D5),
            },
        },
        "probe_10k_record_only": None
        if probe is None
        else {
            "chunk_mae": probe["chunk_mae"],
            "kill_line": KILL_LINE,
            "killed": bool(probe_killed),
        },
        "perstep_horizon": perstep,
        "decision": dec,
    }
    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=1))
        print(f"wrote {out_path}")
    print(
        json.dumps({k: v for k, v in out.items() if k != "perstep_horizon"}, indent=1),
    )
    print(f"\nDECISION: {dec['primary_verdict']}")
    print(f"  adoption: {dec['adoption']}")
    return out


# ---- oracle ----


def make_endpoint_json(chunk: float, first: float, draws: int, **over: object) -> dict:
    suffix = f"_draws{draws}" if draws > 1 else ""
    d = {
        "sample_steps": 1,
        "sample_method": "euler",
        "target_time": "zero",
        "sample_draws": draws,
        "noise_key": "index",
        "sample_plan": V1_PLAN,
        "core_frames": PANEL_CORE,
        "summaries": [
            {"policy": "state-copy", "chunk_mae": 11.78, "first_mae": 2.62},
            {"policy": f"bijou@30000{suffix}", "chunk_mae": chunk, "first_mae": first},
        ],
    }
    d.update(over)
    return d


def _expect_exit(fn: Callable[[], object], label: str) -> None:
    try:
        fn()
    except SystemExit:
        print(f"oracle guard: {label} refused OK")
    else:
        sys.exit(f"oracle guard FAILED: {label} was accepted")


def oracle() -> None:
    # (a) banked-JSON extraction reproduces every registered comparator
    banked = [
        (
            (
                "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000"
                "__panel_k4l2_heun30.json"
            ),
            (TEACHER_CHUNK, TEACHER_FIRST),
        ),
        (
            (
                "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000"
                "__panel_curated_v0_k4l2_draws10_heun30.json"
            ),
            TEACHER_D10,
        ),
        (
            (
                "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000"
                "__panel_curated_v0_k4l2_draws5_heun30.json"
            ),
            TEACHER_D5,
        ),
        (
            "reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2.json",
            (AR_CHUNK, AR_FIRST),
        ),
        ("reports/eval__snapflow_step0__probe_s7_heun30.json", None),
    ]
    for path, want in banked:
        c, f = load_banked(path, path)
        if want is None:
            assert abs(c - TEACHER_PROBE_CHUNK) < SUMMARY_TOL, (path, c)
        else:
            assert abs(c - want[0]) < SUMMARY_TOL and abs(f - want[1]) < SUMMARY_TOL, (
                path,
                c,
                f,
                want,
            )
        print(f"oracle (a) {Path(path).name}: {c:.4f}/{f:.4f} OK")

    # (b) teacher npz: pooling anchors, banked step-curve byte-match,
    # v2 re-pool, degenerate self-pair
    t = np.load(TEACHER_NPZ, allow_pickle=True)
    t_key = "pred:bijou@80000"
    v1_plan = json.loads(Path(V1_PLAN).read_text())
    v2_plan = json.loads(Path(V2_PLAN).read_text())
    ps = perstep_read(t, t_key, t, t_key, v1_plan, v2_plan)
    got = ps["npz_pooled_student"]
    assert abs(got["chunk_mae"] - TEACHER_CHUNK) < SUMMARY_TOL
    assert abs(got["first_mae"] - TEACHER_FIRST) < SUMMARY_TOL
    assert all(d == 0.0 for d in ps["step_delta"])
    assert ps["crossover_step"] is None and ps["firstk_crossover"] is None
    banked_curve = json.loads(Path(FVAP_JSON).read_text())["step_curve"]["flow"]
    assert ps["step_curve"]["teacher"] == banked_curve, (
        "step-curve protocol drifted from the banked flow_vs_ar analysis"
    )
    v2 = ps["v2_column"]["student"]
    assert abs(v2["chunk_mae"] - V2_TEACHER[0]) <= 1e-4 + 5e-5
    assert abs(v2["first_mae"] - V2_TEACHER[1]) <= 1e-4 + 5e-5
    print(
        f"oracle (b) teacher npz: pooled {got['chunk_mae']}/{got['first_mae']}, "
        f"step-curve banked-match, v2 {v2['chunk_mae']}/{v2['first_mae']}, "
        "self-pair deltas all zero OK",
    )

    # (c) synthetic endpoints: every verdict branch + inclusive edges
    e10_pass = load_endpoint(make_endpoint_json(5.55, 1.42, 10), 10, "syn10")
    e10_miss = load_endpoint(make_endpoint_json(5.90, 1.55, 10), 10, "syn10m")
    cases = [
        (6.70, 1.95, e10_pass, "parity-adopt", True, "DEPLOYMENT HEADLINE"),
        (6.70, 1.95, e10_miss, "parity-adopt", True, "did NOT survive"),
        (ADOPT_LINE, EDGE_LINE, e10_pass, "parity-adopt", True, "DEPLOYMENT"),
        (6.95, 2.05, e10_pass, "intermediate-miss", False, "NO ADOPTION"),
        (FALSIFY_LINE, 1.95, e10_pass, "intermediate-miss", True, "NO ADOPTION"),
        (7.20, 1.95, e10_pass, "falsified", True, "NO ADOPTION"),
    ]
    for chunk, first, e10, want_class, want_edge, want_frag in cases:
        e1 = load_endpoint(make_endpoint_json(chunk, first, 1), 1, "syn1")
        res = analyze(e1, e10, None, None, None, None)
        d = res["decision"]
        assert d["primary_class"] == want_class, (chunk, d["primary_class"])
        assert d["grounding_edge_survives"] == want_edge, (first, want_edge)
        assert want_frag in d["adoption"], (chunk, d["adoption"], want_frag)
    probe_ok = load_probe(
        make_endpoint_json(
            KILL_LINE,
            3.0,
            1,
            sample_plan=PROBE_PLAN,
            core_frames=PROBE_CORE,
        ),
        "syn-probe",
    )
    e1 = load_endpoint(make_endpoint_json(6.70, 1.95, 1), 1, "syn1")
    res = analyze(e1, e10_pass, None, probe_ok, None, None)
    assert res["decision"]["probe_killed"] is False  # kill is strictly >
    probe_bad = dict(probe_ok, chunk_mae=KILL_LINE + 0.01)
    res = analyze(e1, e10_pass, None, probe_bad, None, None)
    assert res["decision"]["probe_killed"] is True
    assert "KILL LINE FIRED" in res["decision"]["adoption"]
    print("\noracle (c) synthetic verdict branches + band edges + kill line OK\n")

    # (d) semantics guards refuse doctored JSONs
    _expect_exit(
        lambda: load_endpoint(
            make_endpoint_json(6.7, 1.9, 1, sample_steps=30),
            1,
            "g",
        ),
        "sample_steps=30",
    )
    _expect_exit(
        lambda: load_endpoint(
            make_endpoint_json(6.7, 1.9, 1, sample_method="heun"),
            1,
            "g",
        ),
        "method=heun",
    )
    _expect_exit(
        lambda: load_endpoint(
            make_endpoint_json(6.7, 1.9, 1, target_time="t"),
            1,
            "g",
        ),
        "target_time=t",
    )
    _expect_exit(
        lambda: load_endpoint(
            make_endpoint_json(6.7, 1.9, 1, noise_key="stable"),
            1,
            "g",
        ),
        "noise_key=stable",
    )
    _expect_exit(
        lambda: load_endpoint(make_endpoint_json(6.7, 1.9, 10), 5, "g"),
        "draws mismatch",
    )
    _expect_exit(
        lambda: load_endpoint(
            make_endpoint_json(6.7, 1.9, 1, core_frames=2458),
            1,
            "g",
        ),
        "subset-as-panel",
    )

    # (e) synthetic late-horizon student: crossover detected at step 25
    syn = {k: t[k] for k in bbr.PAIR_KEYS}
    truth = t["truth"]
    pred = t[t_key].copy()
    late = np.zeros(pred.shape[1], dtype=bool)
    late[25:] = True
    pred[:, late, :] = truth[:, late, :] + 1.5 * (pred[:, late, :] - truth[:, late, :])
    syn[t_key] = pred
    syn.update({k: t[k] for k in t.files if k not in syn})
    ps = perstep_read(syn, t_key, t, t_key, v1_plan, None)
    assert all(d == 0.0 for d in ps["step_delta"][:25])
    assert all(d > 0.0 for d in ps["step_delta"][25:])
    assert ps["crossover_step"] == 25
    assert ps["firstk_crossover"] == 25
    print("oracle (e) late-horizon synthetic: crossover at 25, early deltas 0 OK")
    print("\nORACLE: all five checks PASSED")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint1", default=E1_JSON)
    p.add_argument("--endpoint10", default=E10_JSON)
    p.add_argument("--endpoint5", default=E5_JSON)
    p.add_argument("--probe", default=None, help=f"e.g. {PROBE_JSON}")
    p.add_argument(
        "--npz",
        nargs=2,
        metavar=("NPZ", "JSON"),
        default=None,
        help=f"addendum npz-dump primary, e.g. {NPZ_DEFAULT} {NPZ_JSON_DEFAULT}",
    )
    p.add_argument("--teacher-npz", default=TEACHER_NPZ)
    p.add_argument("--v1-plan", default=V1_PLAN)
    p.add_argument("--v2-plan", default=V2_PLAN)
    p.add_argument("--out", default=OUT_DEFAULT)
    p.add_argument("--oracle", action="store_true")
    a = p.parse_args()
    if a.oracle:
        oracle()
        return
    e1 = load_endpoint(a.endpoint1, 1, "endpoint-1nfe")
    e10 = load_endpoint(a.endpoint10, 10, "endpoint-draws10")
    e5 = (
        load_endpoint(a.endpoint5, 5, "endpoint-draws5")
        if Path(a.endpoint5).exists()
        else None
    )
    probe = load_probe(a.probe, "probe-10k") if a.probe else None
    perstep = None
    if a.npz:
        # The addendum npz eval must carry the same registered semantics.
        load_endpoint(a.npz[1], 1, "npz-addendum")
        student = np.load(a.npz[0], allow_pickle=True)
        s_keys = [k for k in student.files if k.startswith("pred:bijou@")]
        if len(s_keys) != 1:
            sys.exit(f"{a.npz[0]}: expected one pred:bijou@ key, got {s_keys}")
        teacher = np.load(a.teacher_npz, allow_pickle=True)
        perstep = perstep_read(
            student,
            s_keys[0],
            teacher,
            "pred:bijou@80000",
            json.loads(Path(a.v1_plan).read_text()),
            json.loads(Path(a.v2_plan).read_text()),
        )
    analyze(e1, e10, e5, probe, perstep, a.out)


if __name__ == "__main__":
    main()
