"""Attach-screen frozen reads — Delta_seam + trunk-drift, ready before the data.

Implements exactly the frozen reads of the attachment seam screen pre-reg
(posts/2026-08-07-prereg-molmo2-attach-screen.md), reads 1-5:

  * READ 1 (primary): Delta_seam = chunk_mae(K) - chunk_mae(F), paired
    per-frame on the panel-v2 core rows at matched steps, seeded frame
    bootstrap CI95 (seed 0, 10,000 resamples — the arch-batch
    conventions). This one number is the screen.
  * READ 2 (decision rule, frozen in the pre-reg):
      Delta_seam < 0 with CI95 excluding 0 AND read 4's drift band
      respected  => KI-JOINT IS THE ATTACHMENT RECIPE (the full-length
                    attachment run pre-registers citing this).
      Delta_seam >= 0 or CI95 contains 0
                   => THE FROZEN DEFAULT STANDS (ties go to cheaper +
                    simpler); the KI-joint direction closes for this
                    trunk class; the Wall-OSS reading — phase-1 CE
                    already routed the action gradients — is recorded.
      K wins but breaks the band => K WINS WITH A NAMED COST; the AEGIS
                    orthogonal-projection repair is the named escalation
                    (banked, NOT built); adoption waits for owner steer.
  * READ 3 (context anchors, quoted beside, never the decision): the
    molmo2 40k endpoint greedy AR panel number (pulled from the drift
    read's endpoint JSON), gemma flow lineage 6.5997 @80k stable-key
    (cross-trunk, directional only), and state-copy as EXECUTION ORACLE:
    both arms must beat it decisively or the screen is VOID, not merely
    negative. "Decisively" is pinned here, before data, as chunk_mae at
    least ``VOID_MARGIN`` (1.0) below the same-npz state-copy pooled
    number — a healthy flow arm clears it by ~4-5.
  * READ 4 (trunk-drift diagnostic, K only): greedy AR panel eval of
    K's materialized ar_view at the screen end vs the 40k endpoint AR
    number, same k4l2 plan family. Band, frozen: |Delta_AR| <= 0.3.
  * READ 5 (record-only): first_mae mirrors of read 1 and per-step-in-
    horizon MAE curves for both arms from the npzs.

Execution guards (abort on failure): panel-v2 endpoint-JSON semantics +
report reproduction through arch_batch_results.load_endpoint_json /
box_batch_results.pick_headline; PAIR_KEYS byte-equality F vs K
(including the shared state-copy column); k4l2 plan + frame-count
semantics on both drift JSONs.

Oracle mode (--oracle, run before any arm data existed):
  (a) the banked teacher@80k v1 npz, sliced to native panel-v2 rows,
      pooled through THIS file reproduces the banked v2 anchors
      6.7151/1.9453 and state-copy 11.7639/2.5851;
  (b) degenerate K := F -> Delta_seam exactly 0, CI [0, 0], drift 0
      -> the frozen-default branch fires;
  (c) synthetic known effects: K error x0.95 -> Delta_seam < 0 with CI
      excluding 0; with drift 0.12 (and the 0.30 band edge, inclusive)
      the KI-joint branch fires; with drift -0.35 the named-cost /
      owner-steer branch fires; K error x1.05 -> falsified -> frozen
      default; K error x3.0 -> the state-copy VOID branch fires and
      outranks the seam read;
  (d) misaligned index between F and K -> hard abort;
  (e) drift-JSON semantics guard: wrong plan or frame counts -> abort.

Pooling semantics are byte-identical to box_batch_results.py (anchored
at AR-100k 5.8026 / flow-80k 6.6232); v2 selection semantics byte-
identical to arch_batch_results.py oracle (a). Pure CPU, read-only on
inputs, deterministic (seeded bootstrap).

Usage (defaults = the attach launchers' chained-eval output names,
rsynced local; --steps 5000 if the matched 5k downshift fired):
  python fontaine/scripts/attach_seam_results.py \
      [--steps 10000] [--f NPZ JSON] [--k NPZ JSON] \
      [--k-view-json PATH] [--endpoint-json PATH] [--out ...] [--oracle]
"""

import argparse
import importlib.util
import json
import sys
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
abr = _sibling("arch_batch_results")

F_RUN = "fontaine_molmo2_flow_frozen_{k}k_ddp4"
K_RUN = "fontaine_molmo2_flow_kijoint_{k}k_ddp4"
ENDPOINT_JSON = (
    "reports/eval__fontaine_molmo2_ar_40k_ddp4__step_040000__panel_curated_v0_k4l2.json"
)
OUT_DEFAULT = "reports/analysis__attach_seam_panel_v2.json"

V1_PLAN = "plans/holdout_curated_v0_k4l2.json"
V1_CORE, V1_LABELED = 17204, 8596

DRIFT_BAND = 0.3  # read 4, frozen: |Delta_AR| <= 0.3
VOID_MARGIN = 1.0  # read 3 "decisively": >= 1.0 below state-copy, pinned pre-data
GEMMA_FLOW_LINEAGE = 6.5997  # @80k stable-key — cross-trunk, directional only


def _stem(run: str, steps: int) -> str:
    return f"reports/eval__{run}__step_{steps:06d}__panel_v2_heun30_draws1_stable"


def default_paths(steps: int) -> dict:
    k = steps // 1000
    f_run, k_run = F_RUN.format(k=k), K_RUN.format(k=k)
    return {
        "f": (f"{_stem(f_run, steps)}.npz", f"{_stem(f_run, steps)}.json"),
        "k": (f"{_stem(k_run, steps)}.npz", f"{_stem(k_run, steps)}.json"),
        "k_view_json": (
            f"reports/eval__{k_run}_ar_view__step_{steps:06d}"
            "__panel_curated_v0_k4l2.json"
        ),
        "endpoint_json": ENDPOINT_JSON,
    }


def load_drift_json(src: str | Path | dict, label: str) -> dict:
    """Strict loader for the k4l2 greedy-AR JSONs the drift read consumes
    (K ar_view at screen end; molmo2 40k endpoint). Plan + frame-count
    semantics must match the anchored k4l2 family, or we stop."""
    d = src if isinstance(src, dict) else json.loads(Path(src).read_text())
    want = {
        "sample_plan": V1_PLAN,
        "core_frames": V1_CORE,
        "labeled_frames": V1_LABELED,
    }
    for k, v in want.items():
        got = d.get(k)
        if got != v:
            sys.exit(f"{label}: {k} = {got!r}, registered {v!r} — refusing the read")
    bare = [
        s
        for s in d.get("summaries", [])
        if str(s.get("policy", "")).startswith("bijou@")
        and not str(s.get("policy", "")).endswith("+fields")
    ]
    if len(bare) != 1:
        sys.exit(
            f"{label}: expected exactly one bare bijou@ policy summary, "
            f"got {[s.get('policy') for s in d.get('summaries', [])]}",
        )
    return {
        "policy": bare[0]["policy"],
        "chunk_mae": float(bare[0]["chunk_mae"]),
        "first_mae": float(bare[0]["first_mae"]),
    }


def step_curve(err: np.ndarray, valid: np.ndarray, core: np.ndarray) -> list:
    """Per-step-in-horizon pooled MAE over core frames (read 5, record-only)."""
    out = []
    for s in range(err.shape[1]):
        sel = valid[:, s] & core
        out.append(round(float(err[sel, s, :].mean()), 4) if sel.any() else None)
    return out


def void_check(
    err_arm: np.ndarray,
    err_sc: np.ndarray,
    core: np.ndarray,
    w: np.ndarray,
    label: str,
) -> dict:
    arm = bbr.pooled_chunk(err_arm, core, w)
    sc = bbr.pooled_chunk(err_sc, core, w)
    margin = sc - arm
    return {
        "arm": label,
        "chunk_mae": round(arm, 4),
        "state_copy_chunk_mae": round(sc, 4),
        "margin_below_state_copy": round(margin, 4),
        "beats_decisively": bool(margin >= VOID_MARGIN),
        "rule": (
            f"screen VOID unless both arms sit >= {VOID_MARGIN} below the "
            "same-npz state-copy pooled chunk_mae (pinned pre-data; the "
            "pre-reg word is 'decisively')"
        ),
    }


def drift_read(view: dict, endpoint: dict) -> dict:
    d = view["chunk_mae"] - endpoint["chunk_mae"]
    return {
        "k_ar_view": view,
        "endpoint_ar": endpoint,
        "delta_ar": round(d, 4),
        "band": DRIFT_BAND,
        "inside_band": bool(abs(d) <= DRIFT_BAND),
        "definition": (
            "greedy AR panel chunk_mae of K's materialized ar_view at the "
            "screen end minus the molmo2 40k endpoint AR number, same k4l2 "
            f"plan family; frozen band |Delta_AR| <= {DRIFT_BAND}"
        ),
        "first_mae_delta_descriptive": round(
            view["first_mae"] - endpoint["first_mae"],
            4,
        ),
    }


def decide(read1: dict, drift: dict | None, voids: list) -> dict:
    """READ 2 — every branch is the pre-reg's frozen decision rule."""
    lines = []
    screen_void = any(not v["beats_decisively"] for v in voids)
    k_wins = read1["mean"] < 0 and abr.ci_excludes_zero(read1["ci95"])
    if screen_void:
        bad = [v["arm"] for v in voids if not v["beats_decisively"]]
        lines.append(
            f"SCREEN VOID: arm(s) {bad} fail the state-copy execution "
            "oracle — the screen is void, not merely negative; no seam "
            "verdict is recorded",
        )
    elif k_wins:
        if drift is None:
            lines.append(
                "PARTIAL: Delta_seam < 0 with CI excluding 0, but read 4 "
                "(trunk drift) has no data — the decision rule needs the "
                "drift band before KI-joint can be adopted",
            )
        elif drift["inside_band"]:
            lines.append(
                "KI-JOINT IS THE ATTACHMENT RECIPE: Delta_seam < 0, CI "
                "excludes 0, drift inside the band — the full-length "
                "attachment run pre-registers citing this screen",
            )
        else:
            lines.append(
                "K WINS WITH A NAMED COST: Delta_seam < 0 with CI "
                f"excluding 0 but |Delta_AR| > {DRIFT_BAND} — the AEGIS "
                "orthogonal-projection repair is the named escalation "
                "(banked, NOT built); adoption waits for owner steer",
            )
    else:
        lines.append(
            "THE FROZEN DEFAULT STANDS: Delta_seam >= 0 or CI contains 0 "
            "— ties go to cheaper + simpler; the KI-joint direction "
            "closes for this trunk class; recorded interpretation: the "
            "Wall-OSS reading (phase-1 CE already routed the action "
            "gradients)",
        )
    return {
        "screen_void": bool(screen_void),
        "k_wins_seam": bool(k_wins),
        "assembly": lines,
    }


def analyze(
    f_npz: dict,
    f_key: str,
    k_npz: dict,
    k_key: str,
    view_src: str | dict | None,
    endpoint_src: str | dict | None,
    out_path: str | None,
) -> dict:
    for k in bbr.PAIR_KEYS:
        if not np.array_equal(f_npz[k], k_npz[k]):
            sys.exit(f"panel pairing broken on {k} between F and K")
    if not np.array_equal(
        f_npz["pred:state-copy"],
        k_npz["pred:state-copy"],
        equal_nan=True,
    ):
        sys.exit("state-copy columns differ between F and K — execution drift, stop")
    truth, valid, core, w = bbr.masks(f_npz)
    repo = f_npz["repo_id"]
    err_f = np.abs(f_npz[f_key] - truth)
    err_k = np.abs(k_npz[k_key] - truth)
    err_sc = np.abs(f_npz["pred:state-copy"] - truth)
    keep_chunk = (w.sum(axis=(1, 2)) > 0) & core
    keep_first = valid[:, 0] & core

    fr_f, _ = bbr.frame_mae(err_f, w)
    fr_k, _ = bbr.frame_mae(err_k, w)
    read1 = abr.paired_read(fr_k, fr_f, keep_chunk, repo)
    read1["definition"] = (
        "Delta_seam: paired per-frame chunk_mae, K - F, v2 core frames "
        "(negative = KI-joint better)"
    )
    read5_first = abr.paired_read(
        abr.first_rows(err_k),
        abr.first_rows(err_f),
        keep_first,
        repo,
    )
    read5_first["definition"] = (
        "first_mae mirror of read 1, K - F, v2 core frames (record-only)"
    )

    voids = [
        void_check(err_f, err_sc, core, w, "F"),
        void_check(err_k, err_sc, core, w, "K"),
    ]
    drift = None
    if view_src is not None and endpoint_src is not None:
        drift = drift_read(
            load_drift_json(view_src, "K ar_view"),
            load_drift_json(endpoint_src, "endpoint AR"),
        )
    decision = decide(read1, drift, voids)

    out = {
        "arms_pooled": {
            "F": {
                "chunk_mae": round(bbr.pooled_chunk(err_f, core, w), 4),
                "first_mae": round(bbr.pooled_first(err_f, valid, core), 4),
                "pred_key": f_key,
            },
            "K": {
                "chunk_mae": round(bbr.pooled_chunk(err_k, core, w), 4),
                "first_mae": round(bbr.pooled_first(err_k, valid, core), 4),
                "pred_key": k_key,
            },
        },
        "read1_delta_seam": read1,
        "read3_context_anchors": {
            "endpoint_ar_chunk_mae": (
                drift["endpoint_ar"]["chunk_mae"] if drift else None
            ),
            "gemma_flow_lineage_chunk_mae": GEMMA_FLOW_LINEAGE,
            "note": (
                "quoted beside, never the decision (own-baseline rule; "
                "gemma is cross-trunk, directional only)"
            ),
            "state_copy_execution_oracle": voids,
        },
        "read4_trunk_drift": drift
        or {"note": "no drift data yet — read 4 pending (K ar_view + endpoint)"},
        "read5_record_only": {
            "first_mae_mirror": read5_first,
            "step_curve_F": step_curve(err_f, valid, core),
            "step_curve_K": step_curve(err_k, valid, core),
        },
        "decision": decision,
    }
    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=1))
        print(f"wrote {out_path}")
    headline = {k: v for k, v in out.items() if k != "read5_record_only"}
    print(json.dumps(headline, indent=1))
    print(
        f"Delta_seam {read1['mean']} CI {read1['ci95']} | "
        f"drift {drift['delta_ar'] if drift else 'pending'} "
        f"(band {DRIFT_BAND}) | void={decision['screen_void']}",
    )
    for line in decision["assembly"]:
        print(f"DECISION: {line}")
    return out


# ---- oracle -----------------------------------------------------------------


def _drift_fixture(delta: float, base: float = 5.0) -> tuple[dict, dict]:
    def j(chunk: float, policy: str) -> dict:
        return {
            "sample_plan": V1_PLAN,
            "core_frames": V1_CORE,
            "labeled_frames": V1_LABELED,
            "summaries": [
                {"policy": policy, "chunk_mae": chunk, "first_mae": 2.0},
                {"policy": "state-copy", "chunk_mae": 11.78, "first_mae": 2.62},
            ],
        }

    return j(base + delta, "bijou@10000"), j(base, "bijou@40000")


def _synth(f_npz: Any, key: str, factor: float) -> dict:
    """K fixture with err exactly factor x F's err (same signs)."""
    truth, pred_f = f_npz["truth"], f_npz[key]
    pred = np.where(np.isfinite(truth), truth + (pred_f - truth) * factor, pred_f)
    out = abr._DictNpz({k: f_npz[k] for k in f_npz.files})
    out[key] = pred
    return out


def oracle() -> None:
    t = np.load(abr.TEACHER_NPZ, allow_pickle=True)
    bare = [
        k for k in t.files if k.startswith("pred:bijou@") and not k.endswith("+fields")
    ]
    assert len(bare) == 1
    key = bare[0]

    # (a) native-v2 fabrication by strict row-slice + anchor reproduction
    v1_plan = json.loads(Path(V1_PLAN).read_text())
    v2_plan = json.loads(Path(abr.V2_PLAN).read_text())
    join = abr.dai.build_join(v1_plan, t)
    keep = abr.snr.v2_keep_mask(v2_plan, join)
    f = abr._DictNpz(abr._v2_slice(t, keep))
    truth, valid, core, w = bbr.masks(f)
    assert int(core.sum()) == abr.V2_CORE and int((~core).sum()) == abr.V2_LABELED
    err = np.abs(f[key] - truth)
    gc, gf = bbr.pooled_chunk(err, core, w), bbr.pooled_first(err, valid, core)
    assert abs(gc - abr.V2_TEACHER[0]) < abr.ANCHOR_TOL, f"anchor FAIL {gc:.4f}"
    assert abs(gf - abr.V2_TEACHER[1]) < abr.ANCHOR_TOL, f"anchor FAIL {gf:.4f}"
    err_sc = np.abs(f["pred:state-copy"] - truth)
    sc = bbr.pooled_chunk(err_sc, core, w)
    assert abs(sc - abr.V2_STATE_COPY[0]) < abr.ANCHOR_TOL, f"sc anchor FAIL {sc:.4f}"
    print(f"oracle (a) OK: v2 anchors reproduced ({gc:.4f}/{gf:.4f}, sc {sc:.4f})")

    # (b) degenerate K := F -> exact zero, frozen default stands
    view, endpoint = _drift_fixture(0.0)
    res = analyze(f, key, abr._DictNpz(dict(f)), key, view, endpoint, None)
    r1 = res["read1_delta_seam"]
    assert r1["mean"] == 0.0 and r1["ci95"] == [0.0, 0.0]
    assert not res["decision"]["k_wins_seam"] and not res["decision"]["screen_void"]
    assert "FROZEN DEFAULT STANDS" in res["decision"]["assembly"][0]
    print("oracle (b) OK: degenerate K:=F -> zero delta, frozen default stands")

    # (c) synthetic known effects
    k_better = _synth(f, key, 0.95)
    view, endpoint = _drift_fixture(0.12)
    res = analyze(f, key, k_better, key, view, endpoint, None)
    r1 = res["read1_delta_seam"]
    assert r1["mean"] < 0 and abr.ci_excludes_zero(r1["ci95"])
    fr_f, _ = bbr.frame_mae(err, w)
    keep = (w.sum(axis=(1, 2)) > 0) & core
    want_delta = -0.05 * float(fr_f[keep].mean())
    assert abs(r1["mean"] - want_delta) < 5e-3, "synthetic delta magnitude off"
    assert "KI-JOINT IS THE ATTACHMENT RECIPE" in res["decision"]["assembly"][0]
    view, endpoint = _drift_fixture(DRIFT_BAND)  # band edge is inclusive
    res = analyze(f, key, k_better, key, view, endpoint, None)
    assert res["read4_trunk_drift"]["inside_band"]
    assert "KI-JOINT IS THE ATTACHMENT RECIPE" in res["decision"]["assembly"][0]
    view, endpoint = _drift_fixture(-0.35)
    res = analyze(f, key, k_better, key, view, endpoint, None)
    assert not res["read4_trunk_drift"]["inside_band"]
    assert "NAMED COST" in res["decision"]["assembly"][0]
    res = analyze(f, key, k_better, key, None, None, None)
    assert "PARTIAL" in res["decision"]["assembly"][0]
    res = analyze(f, key, _synth(f, key, 1.05), key, *_drift_fixture(0.0), None)
    assert res["read1_delta_seam"]["mean"] > 0
    assert "FROZEN DEFAULT STANDS" in res["decision"]["assembly"][0]
    res = analyze(f, key, _synth(f, key, 3.0), key, *_drift_fixture(0.0), None)
    assert res["decision"]["screen_void"]
    assert "SCREEN VOID" in res["decision"]["assembly"][0]
    print(
        "oracle (c) OK: known effects -> adopt / band-edge / named-cost / "
        "partial / falsified / void branches all fire",
    )

    # (d) misaligned index -> hard abort
    bad = abr._DictNpz(dict(f))
    idx = np.array(bad["index"])
    idx[:2] = idx[:2][::-1]
    bad["index"] = idx
    try:
        analyze(f, key, bad, key, None, None, None)
        raise AssertionError("misaligned index not caught")
    except SystemExit as e:
        assert "pairing broken" in str(e)
    print("oracle (d) OK: misaligned index -> hard abort")

    # (e) drift-JSON semantics guard
    view, _ = _drift_fixture(0.0)
    view["sample_plan"] = "plans/other.json"
    try:
        load_drift_json(view, "bad view")
        raise AssertionError("wrong-plan drift JSON not caught")
    except SystemExit as e:
        assert "refusing the read" in str(e)
    print("oracle (e) OK: drift-JSON semantics guard aborts on wrong plan")
    print("ORACLE PASS: all attach-seam read branches verified pre-data")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=10000, choices=[10000, 5000])
    p.add_argument("--f", nargs=2, metavar=("NPZ", "JSON"), default=None)
    p.add_argument("--k", nargs=2, metavar=("NPZ", "JSON"), default=None)
    p.add_argument("--k-view-json", default=None)
    p.add_argument("--endpoint-json", default=None)
    p.add_argument("--out", default=OUT_DEFAULT)
    p.add_argument("--oracle", action="store_true")
    a = p.parse_args()
    if a.oracle:
        oracle()
        return
    d = default_paths(a.steps)
    f_npz_path, f_json = a.f or d["f"]
    k_npz_path, k_json = a.k or d["k"]
    view = a.k_view_json or d["k_view_json"]
    endpoint = a.endpoint_json or d["endpoint_json"]
    for path in [f_npz_path, f_json, k_npz_path, k_json]:
        if not Path(path).exists():
            sys.exit(f"missing input {path} — arms not rsynced yet?")
    drift_srcs = [view, endpoint]
    if not all(Path(x).exists() for x in drift_srcs):
        missing = [x for x in drift_srcs if not Path(x).exists()]
        print(f"NOTE: drift inputs missing ({missing}) — read 4 pending")
        view = endpoint = None

    abr.load_endpoint_json(f_json, "F")
    abr.load_endpoint_json(k_json, "K")
    f_npz, f_keys, _ = bbr.load_arm(f_npz_path, f_json)
    k_npz, k_keys, _ = bbr.load_arm(k_npz_path, k_json)
    truth, valid, core, w = bbr.masks(f_npz)
    _, f_key = bbr.pick_headline(f_npz, f_keys, f_json, truth, valid, core, w, "F")
    truth_k, valid_k, core_k, w_k = bbr.masks(k_npz)
    _, k_key = bbr.pick_headline(
        k_npz,
        k_keys,
        k_json,
        truth_k,
        valid_k,
        core_k,
        w_k,
        "K",
    )
    analyze(f_npz, f_key, k_npz, k_key, view, endpoint, a.out)


if __name__ == "__main__":
    main()
