"""Architecture batch #1 results — the pre-registered reads, ready before the data.

Implements exactly the frozen reads of the arch-batch pre-reg
(posts/2026-08-06-prereg-arch-batch-1.md + Amendments 1-2, ideas #11):

  * Per arm X in {A (img280), B (fullresid)}, paired per-frame vs the
    CONTROL (teacher's own step_040000, Amendment 1) on the panel-v2
    core rows, CI95 by seeded frame bootstrap:
      READ 1 (primary):   Dchunk = chunk_mae(X) - chunk_mae(ctrl).
                          ADOPT-LEVER iff Dchunk <= -0.15 AND CI95 excludes 0.
      READ 2 (grounding): Dfirst <= -0.10 AND CI95 excludes 0 =>
                          the lever moved grounding specifically
                          (v2 state-copy first 2.5851 = ignore-images floor).
      READ 3 (falsified): CI95 contains 0 (null) or Dchunk > +0.15
                          (actively worse => lever killed at this scale).
                          A CI-excluding-0 effect inside the band is
                          classed sub-band: measurable, below the
                          adoption floor, no adoption path.
  * READ 4 (verdict assembly, both arms):
      any adopt-lever  => follow-on pre-reg (combine winners / 80k
                          extension; winner = preferred SnapFlow teacher, #12)
      arm B adopt      => additionally offer upstream (mainline #4)
      arm A adopt OR grounding => the 560 rung is a justified follow-on
                          (Amendment 2 dose-response branch)
      all arms null/falsified AND no grounding move =>
                          conditioning-side levers dead at this scale =>
                          Molmo2-4B trunk swap PROMOTED to the next
                          multi-GPU pre-reg (decision-relevant outcome).
  * Control expectation band (descriptive, Amendment 1): ctrl chunk in
    [6.7, 7.9], first in [1.90, 2.35]; outside => surprise-log entry,
    paired reads unaffected.
  * K1 helper (--k1-train-log): arm's in-run 256-frame probe vs the
    teacher's banked curve at matched steps — kill iff arm >
    teacher@step + 3.0 at any eval >= 5k (usable mid-run at babysits;
    curve banked from the box: reports/teacher_artrunk40k_probe_curve.json).

Execution guards (abort on failure): strict endpoint-JSON semantics
(steps 30 / heun / draws 1 / target_time t / noise_key stable /
sample_plan panel_v2 / core 15,056 / labeled 7,522 / mask_state off);
report-JSON reproduction (|d| < 5e-3) through pick_headline; PAIR_KEYS
byte-equality between every arm npz and the control npz.

Oracle mode (--oracle, run before any arm or control data existed):
  (a) the banked teacher@80k v1 npz re-pooled through THIS file's
      v2 keep-mask reproduces the banked v2 anchors 6.7151/1.9453 and
      state-copy 11.7639/2.5851, with exactly 15,056 core / 7,522
      labeled rows kept;
  (b) degenerate arm := ctrl (native-v2 fabrication by row-slice) ->
      reads exactly 0 / CI [0,0] -> both-null assembly -> the
      Molmo2-4B promotion branch fires;
  (c) synthetic known effects: 1.05x error inflation -> Dchunk =
      +0.05 * ctrl mean frame-MAE (falsified-worse), 0.95x -> adopt
      (arm A => 560-justified; arm B => upstream offer); first-step-only
      0.9x pull-in -> grounding fires without chunk adoption
      (sub-band class) and blocks the Molmo promotion;
  (d) misaligned index -> hard abort;
  (e) K1: teacher+3.1 @6000 kills, +2.9 does not, +9 @4500 is ignored
      (below the 5k gate).

Pooling semantics are byte-identical to box_batch_results.py (anchored);
v2 selection semantics are byte-identical to snapflow_results.py /
panel_v2.py (anchored in oracle (a)). Pure CPU, read-only on inputs,
deterministic (seeded bootstrap).

Usage (defaults = the chained evals' output names, rsynced local):
  python fontaine/scripts/arch_batch_results.py \
      [--arm-a NPZ JSON] [--arm-b NPZ JSON] [--ctrl NPZ JSON] \
      [--k1-train-log PATH --k1-arm NAME] [--out ...] [--oracle]
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
snr = _sibling("snapflow_results")
dai = _sibling("dup_census_anchor_impact")

CTRL_STEM = (
    "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_040000"
    "__panel_v2_ctrl_heun30_draws1_stable"
)
ARM_A_STEM = (
    "reports/eval__fontaine_flow_archA_img280_40k_ddp3__step_040000"
    "__panel_v2_heun30_draws1_stable"
)
ARM_B_STEM = (
    "reports/eval__fontaine_flow_archB_fullresid_40k_ddp3__step_040000"
    "__panel_v2_heun30_draws1_stable"
)
OUT_DEFAULT = "reports/analysis__arch_batch_1_panel_v2.json"
TEACHER_PROBE_JSON = "reports/teacher_artrunk40k_probe_curve.json"

V1_PLAN = "plans/holdout_curated_v0_k4l2.json"
V2_PLAN = "plans/holdout_curated_v0_k4l2_panel_v2.json"
TEACHER_NPZ = (
    "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000"
    "__panel_k4l2_heun30.npz"
)

# Banked v2 anchors (panel_v2.py derivation, index-keyed teacher@80k)
V2_TEACHER = (6.7151, 1.9453)
V2_STATE_COPY = (11.7639, 2.5851)
V2_CORE, V2_LABELED = 15056, 7522

ADOPT_BAND = 0.15  # read 1: max(3*sigma_seed 0.114, 0.15) — floor binds
GROUND_BAND = 0.10  # read 2
CTRL_EXPECT_CHUNK = (6.7, 7.9)  # Amendment 1 expectation band (descriptive)
CTRL_EXPECT_FIRST = (1.90, 2.35)
K1_MARGIN = 3.0
K1_MIN_STEP = 5000
SUMMARY_TOL = 5e-3
ANCHOR_TOL = 1e-4 + 5e-5  # banked at 4 dp


def load_endpoint_json(src: str | Path | dict, label: str) -> dict:
    """Strict loader for NEW panel-v2 endpoint JSONs: every scoring
    semantic the registration fixed must be recorded, or we stop."""
    d = src if isinstance(src, dict) else json.loads(Path(src).read_text())
    want = {
        "sample_steps": 30,
        "sample_method": "heun",
        "sample_draws": 1,
        "target_time": "t",
        "noise_key": "stable",
        "mask_state": False,
        "sample_plan": V2_PLAN,
        "core_frames": V2_CORE,
        "labeled_frames": V2_LABELED,
    }
    for k, v in want.items():
        got = d.get(k)
        if got != v:
            sys.exit(f"{label}: {k} = {got!r}, registered {v!r} — refusing the read")
    return d


def first_rows(err: np.ndarray) -> np.ndarray:
    return err[:, 0, :].mean(axis=1)


def paired_read(
    rows_x: np.ndarray,
    rows_c: np.ndarray,
    keep: np.ndarray,
    repo: np.ndarray,
) -> dict:
    d = (rows_x - rows_c)[keep]
    ci = bbr.bootstrap_ci(d)
    lo = bbr.loro(d, repo[keep])
    influential = sorted(
        lo.items(),
        key=lambda kv: abs(kv[1]["mean_without"] - float(d.mean())),
        reverse=True,
    )[:5]
    return {
        "mean": round(float(d.mean()), 5),
        "median": round(float(np.median(d)), 5),
        "ci95": [round(ci[0], 5), round(ci[1], 5)],
        "arm_win_rate": round(float((d < 0).mean()), 4),
        "n_frames": int(keep.sum()),
        "most_influential_repos": [{"repo": k, **v} for k, v in influential],
    }


def ci_excludes_zero(ci: list) -> bool:
    return ci[0] > 0 or ci[1] < 0


def classify_arm(read1: dict, read2: dict) -> dict:
    """The frozen per-arm decision. Every branch is pre-registered."""
    dchunk, chunk_ci = read1["mean"], read1["ci95"]
    dfirst, first_ci = read2["mean"], read2["ci95"]
    adopt = dchunk <= -ADOPT_BAND and ci_excludes_zero(chunk_ci)
    grounding = dfirst <= -GROUND_BAND and ci_excludes_zero(first_ci)
    if adopt:
        chunk_class = "adopt-lever"
    elif dchunk > ADOPT_BAND:
        chunk_class = "falsified-worse"
    elif ci_excludes_zero(chunk_ci):
        chunk_class = "sub-band"  # measurable, below the adoption floor
    else:
        chunk_class = "null"
    return {
        "adopt_lever": bool(adopt),
        "grounding_moved": bool(grounding),
        "chunk_class": chunk_class,
        "rules": {
            "adopt": f"Dchunk <= -{ADOPT_BAND} AND CI95 excludes 0",
            "grounding": f"Dfirst <= -{GROUND_BAND} AND CI95 excludes 0",
            "falsified": f"CI95 contains 0 or Dchunk > +{ADOPT_BAND}",
        },
    }


def assemble_verdict(arms: dict) -> dict:
    """READ 4. `arms` maps 'A'/'B' -> classify_arm dict (absent = no data yet)."""
    lines = []
    any_adopt = any(a["adopt_lever"] for a in arms.values())
    any_ground = any(a["grounding_moved"] for a in arms.values())
    if any_adopt:
        winners = [k for k, a in arms.items() if a["adopt_lever"]]
        lines.append(
            f"ADOPT-LEVER fired for arm(s) {winners}: follow-on pre-reg "
            "(combine winning levers and/or 80k extension of the winner; "
            "winner becomes the preferred SnapFlow teacher config, #12)",
        )
    if arms.get("B", {}).get("adopt_lever"):
        lines.append("arm B adopt => offer upstream (mainline #4 stream question)")
    a_cls = arms.get("A")
    if a_cls and (a_cls["adopt_lever"] or a_cls["grounding_moved"]):
        lines.append(
            "arm A moved (Amendment 2 dose-response): the 560 rung is a "
            "justified follow-on on this same instrument",
        )
    if len(arms) == 2 and not any_adopt and not any_ground:
        lines.append(
            "BOTH ARMS NULL/FALSIFIED: conditioning-side levers are dead at "
            "this scale => the Molmo2-4B trunk swap is PROMOTED to the next "
            "multi-GPU pre-reg (decision-relevant, not a failure)",
        )
    if not arms:
        lines.append(
            "CONTROL-ONLY: no arm data yet — control pooled numbers + "
            "expectation-band check only (K1 re-anchor ready)",
        )
    elif len(arms) < 2:
        lines.append(
            f"PARTIAL: only arm(s) {sorted(arms)} read — assembly final "
            "when both arms have data",
        )
    if not lines:
        lines.append(
            "no adoption path fired; a grounding-only move elsewhere keeps "
            "the conditioning front alive (no Molmo promotion from this read)",
        )
    return {"per_arm": arms, "assembly": lines}


def analyze(
    arm_npzs: dict,
    arm_keys: dict,
    ctrl: dict,
    ctrl_key: str,
    out_path: str | None,
) -> dict:
    truth, valid, core, w = bbr.masks(ctrl)
    err_c = np.abs(ctrl[ctrl_key] - truth)
    fr_c, _ = bbr.frame_mae(err_c, w)
    fi_c = first_rows(err_c)
    keep_chunk = (w.sum(axis=(1, 2)) > 0) & core
    keep_first = valid[:, 0] & core
    ctrl_pooled = {
        "chunk_mae": round(bbr.pooled_chunk(err_c, core, w), 4),
        "first_mae": round(bbr.pooled_first(err_c, valid, core), 4),
        "pred_key": ctrl_key,
    }
    band = {
        "expect_chunk": list(CTRL_EXPECT_CHUNK),
        "expect_first": list(CTRL_EXPECT_FIRST),
        "chunk_inside": bool(
            CTRL_EXPECT_CHUNK[0] <= ctrl_pooled["chunk_mae"] <= CTRL_EXPECT_CHUNK[1],
        ),
        "first_inside": bool(
            CTRL_EXPECT_FIRST[0] <= ctrl_pooled["first_mae"] <= CTRL_EXPECT_FIRST[1],
        ),
    }
    band["note"] = (
        "inside the Amendment 1 expectation band"
        if band["chunk_inside"] and band["first_inside"]
        else "SURPRISE-LOG: control outside the banked expectation band — "
        "paired reads unaffected, log the surprise with the results post"
    )

    arms_out, arm_classes = {}, {}
    for name, d in arm_npzs.items():
        for k in bbr.PAIR_KEYS:
            if not np.array_equal(d[k], ctrl[k]):
                sys.exit(f"panel pairing broken on {k} between arm {name} and control")
        err_x = np.abs(d[arm_keys[name]] - truth)
        fr_x, _ = bbr.frame_mae(err_x, w)
        read1 = paired_read(fr_x, fr_c, keep_chunk, ctrl["repo_id"])
        read1["definition"] = (
            f"paired per-frame chunk_mae, arm {name} - control, v2 core frames"
        )
        read2 = paired_read(first_rows(err_x), fi_c, keep_first, ctrl["repo_id"])
        read2["definition"] = (
            f"paired per-frame first-step MAE, arm {name} - control, v2 core frames"
        )
        cls = classify_arm(read1, read2)
        arm_classes[name] = cls
        arms_out[name] = {
            "pooled": {
                "chunk_mae": round(bbr.pooled_chunk(err_x, core, w), 4),
                "first_mae": round(bbr.pooled_first(err_x, valid, core), 4),
                "pred_key": arm_keys[name],
            },
            "read1_primary_dchunk": read1,
            "read2_grounding_dfirst": read2,
            "classification": cls,
        }

    out = {
        "control_pooled": ctrl_pooled,
        "control_expectation_band": band,
        "v2_state_copy_context": {
            "chunk_mae": V2_STATE_COPY[0],
            "first_mae": V2_STATE_COPY[1],
            "note": "ignore-images floor (banked; context, no frozen read consumes it)",
        },
        "arms": arms_out,
        "decision": assemble_verdict(arm_classes),
    }
    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=1))
        print(f"wrote {out_path}")
    print(json.dumps({k: v for k, v in out.items() if k != "arms"}, indent=1))
    for name, a in arms_out.items():
        r1, r2 = a["read1_primary_dchunk"], a["read2_grounding_dfirst"]
        print(
            f"arm {name}: Dchunk {r1['mean']} CI {r1['ci95']} | "
            f"Dfirst {r2['mean']} CI {r2['ci95']} | {a['classification']['chunk_class']}"
            f"{' +grounding' if a['classification']['grounding_moved'] else ''}",
        )
    for line in out["decision"]["assembly"]:
        print(f"DECISION: {line}")
    return out


def k1_check(train_log: str, teacher_probe: str, arm_name: str) -> dict:
    """K1 in-run kill gate: arm probe > teacher@matched-step + 3.0 at any
    eval >= 5k => kill at the next save boundary. Run at every babysit."""
    curve = json.loads(Path(teacher_probe).read_text())["probe"]
    rows = [json.loads(line) for line in Path(train_log).read_text().splitlines()]
    evals = {r["step"]: r["eval_chunk_mae"] for r in rows if "eval_chunk_mae" in r}
    violations = [
        {
            "step": s,
            "arm_probe": v,
            "teacher_probe": curve[str(s)],
            "excess": round(v - curve[str(s)], 4),
        }
        for s, v in sorted(evals.items())
        if s >= K1_MIN_STEP and str(s) in curve and v > curve[str(s)] + K1_MARGIN
    ]
    out = {
        "arm": arm_name,
        "rule": f"kill iff arm probe > teacher@step + {K1_MARGIN} at any eval >= {K1_MIN_STEP}",
        "n_evals_checked": len(
            [s for s in evals if s >= K1_MIN_STEP and str(s) in curve],
        ),
        "last_eval": max(evals) if evals else None,
        "violations": violations,
        "kill": bool(violations),
    }
    print(json.dumps(out, indent=1))
    if violations:
        print(f"K1 KILL: arm {arm_name} — kill at the next save boundary")
    return out


# ---- oracle -----------------------------------------------------------------


def _v2_slice(npz: dict, keep: np.ndarray) -> dict:
    """Fabricate a native-v2 npz from a banked v1 npz by strict row-slice
    (exactly the panel-v2 definition: row subset, original order)."""
    return {k: npz[k][keep] for k in npz.files}


class _DictNpz(dict):
    @property
    def files(self) -> list:
        return list(self.keys())


def oracle() -> None:
    t = np.load(TEACHER_NPZ, allow_pickle=True)
    bare = [
        k for k in t.files if k.startswith("pred:bijou@") and not k.endswith("+fields")
    ]
    assert len(bare) == 1
    t_key = bare[0]

    # (a) v2 anchor reproduction through THIS file's selection + pooling
    v1_plan = json.loads(Path(V1_PLAN).read_text())
    v2_plan = json.loads(Path(V2_PLAN).read_text())
    assert len(v2_plan["core"]) == V2_CORE and len(v2_plan["labeled"]) == V2_LABELED
    join = dai.build_join(v1_plan, t)
    keep = snr.v2_keep_mask(v2_plan, join)
    truth, valid, core, w = bbr.masks(t)
    assert int((core & keep).sum()) == V2_CORE, "v2 core row count mismatch"
    assert int((~core & keep).sum()) == V2_LABELED, "v2 labeled row count mismatch"
    err = np.abs(t[t_key] - truth)
    sel = core & keep
    gc, gf = bbr.pooled_chunk(err, sel, w), bbr.pooled_first(err, valid, sel)
    assert (
        abs(gc - V2_TEACHER[0]) < ANCHOR_TOL and abs(gf - V2_TEACHER[1]) < ANCHOR_TOL
    ), f"v2 teacher anchor FAIL: {gc:.4f}/{gf:.4f} vs banked {V2_TEACHER}"
    sc_err = np.abs(t["pred:state-copy"] - truth)
    sc = (bbr.pooled_chunk(sc_err, sel, w), bbr.pooled_first(sc_err, valid, sel))
    assert abs(sc[0] - V2_STATE_COPY[0]) < ANCHOR_TOL
    assert abs(sc[1] - V2_STATE_COPY[1]) < ANCHOR_TOL
    print(
        f"oracle (a) v2 anchors: teacher {gc:.4f}/{gf:.4f}, "
        f"state-copy {sc[0]:.4f}/{sc[1]:.4f}, rows {V2_CORE}/{V2_LABELED} OK",
    )

    # (b) degenerate: arm := ctrl (native-v2 fabrication)
    tn = _DictNpz({k: t[k] for k in t.files})
    ctrl = _DictNpz(_v2_slice(tn, keep))
    res = analyze({"A": ctrl, "B": ctrl}, {"A": t_key, "B": t_key}, ctrl, t_key, None)
    for name in ("A", "B"):
        r1 = res["arms"][name]["read1_primary_dchunk"]
        r2 = res["arms"][name]["read2_grounding_dfirst"]
        assert r1["mean"] == 0.0 and r1["ci95"] == [0.0, 0.0]
        assert r2["mean"] == 0.0 and r2["ci95"] == [0.0, 0.0]
        assert res["arms"][name]["classification"]["chunk_class"] == "null"
    assert any("Molmo2-4B" in line for line in res["decision"]["assembly"])
    assert abs(res["control_pooled"]["chunk_mae"] - V2_TEACHER[0]) < 1e-3
    assert res["control_expectation_band"]["chunk_inside"]  # 6.7151 in [6.7, 7.9]
    print("\noracle (b) degenerate: zero deltas, both-null => Molmo promotion OK\n")

    # (c) synthetic known effects
    truth2, _valid2, core2, w2 = bbr.masks(ctrl)
    err2 = np.abs(ctrl[t_key] - truth2)
    fr, _ = bbr.frame_mae(err2, w2)
    kc = (w2.sum(axis=(1, 2)) > 0) & core2
    want = 0.05 * float(fr[kc].mean())
    infl = _DictNpz(dict(ctrl))
    infl[t_key] = ctrl["truth"] + 1.05 * (ctrl[t_key] - ctrl["truth"])
    res = analyze({"A": infl}, {"A": t_key}, ctrl, t_key, None)
    m = res["arms"]["A"]["read1_primary_dchunk"]["mean"]
    assert abs(m - want) < 5e-3, f"synthetic delta {m} != 0.05*frame-mean {want:.5f}"
    assert res["arms"]["A"]["classification"]["chunk_class"] == "falsified-worse"
    defl = _DictNpz(dict(ctrl))
    defl[t_key] = ctrl["truth"] + 0.95 * (ctrl[t_key] - ctrl["truth"])
    res = analyze({"A": defl}, {"A": t_key}, ctrl, t_key, None)
    assert res["arms"]["A"]["classification"]["adopt_lever"]
    assert any("560 rung" in line for line in res["decision"]["assembly"])
    res = analyze({"B": defl}, {"B": t_key}, ctrl, t_key, None)
    assert res["arms"]["B"]["classification"]["adopt_lever"]
    assert any("upstream" in line for line in res["decision"]["assembly"])
    first_only = _DictNpz(dict(ctrl))
    pred = ctrl[t_key].copy()
    pred[:, 0, :] = ctrl["truth"][:, 0, :] + 0.9 * (pred - ctrl["truth"])[:, 0, :]
    first_only[t_key] = pred
    res = analyze(
        {"A": first_only, "B": ctrl},
        {"A": t_key, "B": t_key},
        ctrl,
        t_key,
        None,
    )
    a = res["arms"]["A"]["classification"]
    assert a["grounding_moved"] and not a["adopt_lever"]
    assert a["chunk_class"] in ("sub-band", "null")
    assert not any("Molmo2-4B" in line for line in res["decision"]["assembly"])
    assert any("560 rung" in line for line in res["decision"]["assembly"])
    print("\noracle (c) synthetics: worse-kill/adopt-A/adopt-B/grounding-only OK\n")

    # (d) misaligned index -> hard abort
    rng = np.random.default_rng(1)
    broken = _DictNpz(dict(ctrl))
    broken["index"] = ctrl["index"][rng.permutation(len(ctrl["index"]))]
    try:
        analyze({"A": broken}, {"A": t_key}, ctrl, t_key, None)
    except SystemExit:
        print("oracle (d) pairing break: misaligned index aborted OK")
    else:
        sys.exit("oracle (d) FAILED: misaligned index was not caught")

    # (e) K1 gate on synthetic curves
    import tempfile

    curve = json.loads(Path(TEACHER_PROBE_JSON).read_text())["probe"]
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "train_log.jsonl"
        rows = [
            {"step": 4500, "eval_chunk_mae": curve["4500"] + 9.0},
            {"step": 6000, "eval_chunk_mae": curve["6000"] + 2.9},
        ]
        log.write_text("\n".join(json.dumps(r) for r in rows))
        assert not k1_check(str(log), TEACHER_PROBE_JSON, "oracle")["kill"]
        rows.append({"step": 6500, "eval_chunk_mae": curve["6500"] + 3.1})
        log.write_text("\n".join(json.dumps(r) for r in rows))
        res = k1_check(str(log), TEACHER_PROBE_JSON, "oracle")
        assert res["kill"] and res["violations"][0]["step"] == 6500
    print("oracle (e) K1: 4500 ignored, +2.9 clean, +3.1 kills OK")
    print("\nORACLE: all five checks PASSED")


def _load_pair(npz_path: str, json_path: str, label: str) -> tuple:
    load_endpoint_json(json_path, label)
    d = np.load(npz_path, allow_pickle=True)
    truth, valid, core, w = bbr.masks(d)
    pred_keys = [k for k in d.files if k.startswith("pred:bijou@")]
    _scores, key = bbr.pick_headline(
        d,
        pred_keys,
        json_path,
        truth,
        valid,
        core,
        w,
        label,
    )
    return d, key


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ctrl",
        nargs=2,
        metavar=("NPZ", "JSON"),
        default=[f"{CTRL_STEM}.npz", f"{CTRL_STEM}.json"],
    )
    p.add_argument("--arm-a", nargs=2, metavar=("NPZ", "JSON"), default=None)
    p.add_argument("--arm-b", nargs=2, metavar=("NPZ", "JSON"), default=None)
    p.add_argument("--no-default-arms", action="store_true")
    p.add_argument(
        "--ctrl-only",
        action="store_true",
        help="run the control pooled + expectation-band read with no arm data",
    )
    p.add_argument("--k1-train-log", default=None)
    p.add_argument("--k1-arm", default="?")
    p.add_argument("--teacher-probe", default=TEACHER_PROBE_JSON)
    p.add_argument("--out", default=OUT_DEFAULT)
    p.add_argument("--oracle", action="store_true")
    a = p.parse_args()
    if a.oracle:
        oracle()
        return
    if a.k1_train_log:
        res = k1_check(a.k1_train_log, a.teacher_probe, a.k1_arm)
        sys.exit(2 if res["kill"] else 0)
    arm_args = {}
    if a.arm_a:
        arm_args["A"] = a.arm_a
    if a.arm_b:
        arm_args["B"] = a.arm_b
    if not arm_args and not a.no_default_arms:
        for name, stem in (("A", ARM_A_STEM), ("B", ARM_B_STEM)):
            if Path(f"{stem}.npz").exists():
                arm_args[name] = [f"{stem}.npz", f"{stem}.json"]
    if not arm_args and not a.ctrl_only:
        sys.exit("no arm data found (defaults absent) — nothing to read")
    ctrl, ctrl_key = _load_pair(a.ctrl[0], a.ctrl[1], "control")
    arm_npzs, arm_keys = {}, {}
    for name, (npz_path, json_path) in arm_args.items():
        arm_npzs[name], arm_keys[name] = _load_pair(npz_path, json_path, f"arm-{name}")
    analyze(arm_npzs, arm_keys, ctrl, ctrl_key, a.out)


if __name__ == "__main__":
    main()
