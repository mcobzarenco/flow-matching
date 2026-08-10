"""MolmoAct2 out-of-band panel — matched-window frozen reads, record-only.

Owner steering 2026-08-10 10:50Z/11:06Z + GO 11:59Z; pre-reg
posts/2026-08-10-prereg-molmoact2-oob-panel.md. Scores the released
``allenai/MolmoAct2-SO100_101`` predictions (molmoact2_panel_predict.py
npz, steps 0..29 filled / 30..49 NaN) against our banked panel arms on
THEIR native horizon: **all pooling restricted to chunk steps 0..29
(= 1.0 s at 30 fps)**, both sides, same frames, paired per-frame.

Reads (each x {pooled, clean, contaminated} repo splits, core rows):

  1. matched-window chunk MAE per arm (molmoact2, snapflow-80k
     top-10-tickets + stable-key, ar_40k endpoint, ar_60k continuation,
     er_60k@15000, state-copy floor) + step-0 first MAE;
  2. paired per-frame Δ (molmoact2 − arm), seeded bootstrap CI95
     (seed 0, 10,000), classified MOLMOACT2-BETTER / MOLMOACT2-WORSE /
     CI-SPANS-0;
  3. contamination split derived live from AllenAI's own mixture list
     (``SO100_SO101_MOLMOACT2`` in their repo), pinned to the measured
     245 repos / 7,996 frames / 5,332 core frames — drift is a hard
     abort (their list changed ⇒ re-approve the split);
  4. our arms' full-50 pooled numbers recorded as secondary anchors
     (never quoted against the 30-step side).

Execution oracles (each failure a hard abort): identity columns
byte-match across every npz; state-copy rows byte-match; the molmoact2
pred is all-finite on the window and all-NaN after it; every banked
arm's full-50 re-pool reproduces its own report json (5e-3).

``--oracle``: planted-delta fixtures (exact Δ, degenerate CI, both
signs + CI-SPANS-0, split arithmetic) + every abort branch.

Pure CPU, read-only on inputs, deterministic. RECORD-ONLY: nothing
gates or repoints our runs.
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

WINDOW = 30  # their SO-100/101 horizon: 30 steps at 30 fps = 1.0 s
CAND_STEM = "reports/eval__molmoact2_so100_release__panel_curated_v0_k4l2_oob"
CAND_KEY = "pred:molmoact2-so100@release"
BASELINES = [
    {
        "label": "snapflow80k top10tickets",
        "stem": (
            "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000"
            "__panel_curated_v0_k4l2_top10tickets_heun30"
        ),
        "key": "pred:bijou@80000_draws10_ticket",
        "run": "bijou_flow_artrunk_h1024_40k_ddp2",
    },
    {
        "label": "snapflow80k stablekey",
        "stem": (
            "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000"
            "__panel_curated_v0_k4l2_stablekey_heun30"
        ),
        "key": "pred:bijou@80000",
        "run": "bijou_flow_artrunk_h1024_40k_ddp2",
    },
    {
        # owner ask 14:33Z (skipped 15:01Z on a wrong not-banked claim,
        # corrected 15:5xZ): mean-of-10 draws, seating noise assignment —
        # full-panel per-frame npz banked by the noise-ladder seating stage
        "label": "flow80k draws10 seating",
        "stem": (
            "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000"
            "__panel_curated_v0_k4l2_draws10_seating_heun30"
        ),
        "key": "pred:bijou@80000_draws10",
        "run": "bijou_flow_artrunk_h1024_40k_ddp2",
    },
    {
        # owner add 15:22Z 08-10: the SnapFlow 1-NFE distilled student
        # (only the single-draw bank kept a per-frame npz; draws5/10
        # exist as summary json+html only)
        "label": "snapflow student 30k 1nfe",
        "stem": (
            "reports/eval__fontaine_flow_snapdistill_h1024_30k_1xh100"
            "__step_030000__panel_curated_v0_k4l2_1nfe_euler1_npz"
        ),
        "key": "pred:bijou@30000",
        "run": "fontaine_flow_snapdistill_h1024_30k_1xh100",
    },
    {
        # owner add 14:33Z 08-10: the original heun-30 single-draw bank
        # (the 6.6232 anchor eval; identity byte-pairs with the curated
        # stems — verified before adding)
        "label": "snapflow80k heun30",
        "stem": (
            "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000"
            "__panel_k4l2_heun30"
        ),
        "key": "pred:bijou@80000",
        "run": "bijou_flow_artrunk_h1024_40k_ddp2",
    },
    {
        "label": "ar_40k endpoint",
        "stem": (
            "reports/eval__fontaine_molmo2_ar_40k_ddp4__step_040000"
            "__panel_curated_v0_k4l2"
        ),
        "key": "pred:bijou@40000",
        "run": "fontaine_molmo2_ar_40k_ddp4",
    },
    {
        "label": "ar_60k continuation",
        "stem": (
            "reports/eval__fontaine_molmo2_ar_60k_ddp4__step_060000"
            "__panel_curated_v0_k4l2"
        ),
        "key": "pred:bijou@60000",
        "run": "fontaine_molmo2_ar_60k_ddp4",
    },
    {
        "label": "er_60k@15000",
        "stem": (
            "reports/eval__fontaine_molmo2_er_60k_ddp4__step_015000"
            "__panel_curated_v0_k4l2"
        ),
        "key": "pred:bijou@15000",
        "run": "fontaine_molmo2_er_60k_ddp4",
    },
]
STATE_KEYS = ("pred:state-copy", "pred:state-copy-norm")
OUT_DEFAULT = "reports/analysis__molmoact2_oob_panel_k4l2.json"
CONTAM_OUT_DEFAULT = "reports/analysis__molmoact2_contamination_repos.json"
MIXTURE_CONSTANTS = Path(
    "~/molmoact2/experiments/launch_scripts/data_constants.py",
).expanduser()
# Pinned at measurement time (2026-08-10 ~11:3xZ session, plan post §5).
CONTAM_EXPECT = {"repos": 245, "frames": 7996, "core_frames": 5332}
# AMENDMENT (owner 2026-08-10 13:14:54Z, logged in-channel before any
# real read ran): wraparound-unit repo excluded from every statistic —
# truth |max| ~3141 on 24 panel frames (16 core) makes any policy
# comparison there meaningless and one frame alone moved the smoke
# pooled mean ~+4. The contamination PIN above stays full-panel (it
# verifies their mixture list, not our row selection); exclusion is
# applied downstream to the row mask.
EXCLUDED_REPOS = ("willnorris/bbox-2",)


def _load_npz(path: str | Path) -> dict:
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def mixture_repo_set(constants_path: Path = MIXTURE_CONSTANTS) -> set[str]:
    """AllenAI's SO-100/101 fine-tune mixture, from their own repo file
    (pure literal lists; ``lerobot:`` prefixes stripped)."""
    spec = importlib.util.spec_from_file_location(
        "molmoact2_data_constants",
        constants_path,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    repos = mod.SO100_SO101_MOLMOACT2
    return {str(r).removeprefix("lerobot:") for r in repos}


def contamination_masks(
    repo_id: np.ndarray,
    core: np.ndarray,
    contam_set: set[str],
    expect: dict | None,
) -> tuple[np.ndarray, dict]:
    contam = np.isin(repo_id, sorted(contam_set))
    stats = {
        "repos": len(set(repo_id[contam])),
        "frames": int(contam.sum()),
        "core_frames": int((contam & core).sum()),
        "clean_repos": len(set(repo_id[~contam])),
    }
    if expect is not None:
        got = {k: stats[k] for k in expect}
        if got != expect:
            sys.exit(
                f"contamination split drifted: measured {got} vs pinned "
                f"{expect} — their mixture list changed, re-approve the split",
            )
    return contam, stats


def window_view(npz: dict, key: str) -> dict:
    """Slice the npz views to the matched window so bbr's pooling
    machinery applies unchanged (its functions are step-axis generic)."""
    return {
        "truth": npz["truth"][:, :WINDOW],
        "valid": npz["valid"][:, :WINDOW],
        "core": npz["core"],
        key: npz[key][:, :WINDOW],
    }


def _check_report(npz: dict, key: str, report: dict, run_dir: str, label: str) -> None:
    """Full-50 re-pool must reproduce the arm's own report json."""
    ckpt = str(report.get("checkpoint", ""))
    if run_dir not in ckpt:
        sys.exit(f"{label}: report checkpoint {ckpt!r} is not {run_dir} — stop")
    truth, valid, core, w = bbr.masks(npz)
    err = np.abs(npz[key] - truth)
    gc = bbr.pooled_chunk(err, core, w)
    gf = bbr.pooled_first(err, valid, core)
    policy = key.removeprefix("pred:")
    summ = [s for s in report.get("summaries", []) if s.get("policy") == policy]
    if len(summ) != 1:
        sys.exit(f"{label}: report has {len(summ)} summaries for {policy!r} — stop")
    wc, wf = summ[0]["chunk_mae"], summ[0]["first_mae"]
    if abs(gc - wc) >= 5e-3 or abs(gf - wf) >= 5e-3:
        sys.exit(
            f"{label}: npz re-pool {gc:.4f}/{gf:.4f} does not reproduce the "
            f"report's {wc:.4f}/{wf:.4f} — plan/scoring drift, stop",
        )
    print(f"{label}: report cross-check OK ({policy} full-50 {gc:.4f}/{gf:.4f})")


def _classify(lo: float, hi: float) -> str:
    if hi < 0:
        return "MOLMOACT2-BETTER"
    if lo > 0:
        return "MOLMOACT2-WORSE"
    return "CI-SPANS-0"


def analyze(
    cand_npz: dict,
    baselines: list[tuple[dict, dict | None, dict]],
    contam_set: set[str],
    contam_expect: dict | None,
    out_path: str | None,
    contam_out_path: str | None = None,
) -> dict:
    # ---- execution oracles gate every number below ----
    for npz, _rep, spec in baselines:
        for key in bbr.PAIR_KEYS:
            if not np.array_equal(cand_npz[key], npz[key]):
                sys.exit(
                    f"panel pairing broken on {key} between molmoact2 and "
                    f"{spec['label']} — stop",
                )
        for key in STATE_KEYS:
            a, b = cand_npz[key], npz[key]
            if not (
                a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
            ):
                sys.exit(
                    f"{key} rows do NOT byte-match between molmoact2 and "
                    f"{spec['label']} — stop",
                )
    cand = cand_npz[CAND_KEY]
    if not np.isfinite(cand[:, :WINDOW]).all():
        sys.exit("molmoact2 pred has non-finite values inside the window — stop")
    if not np.isnan(cand[:, WINDOW:]).all():
        sys.exit(
            "molmoact2 pred has non-NaN values after the 30-step window — "
            "horizon contract violated, stop",
        )
    for npz, rep, spec in baselines:
        if rep is not None:
            _check_report(npz, spec["key"], rep, spec["run"], spec["label"])

    contam, contam_stats = contamination_masks(
        cand_npz["repo_id"],
        cand_npz["core"],
        contam_set,
        contam_expect,
    )
    print(f"contamination split: {contam_stats}")
    if contam_out_path:
        Path(contam_out_path).write_text(
            json.dumps(
                {
                    "source": "SO100_SO101_MOLMOACT2 (their data_constants.py)",
                    "stats": contam_stats,
                    "contaminated_repos": sorted(set(cand_npz["repo_id"][contam])),
                },
                indent=1,
            ),
        )
        print(f"wrote {contam_out_path}")

    wv_cand = window_view(cand_npz, CAND_KEY)
    truth, valid, core, w = bbr.masks(wv_cand)
    err_cand = np.abs(wv_cand[CAND_KEY] - truth)
    frame_cand, nvalid = bbr.frame_mae(err_cand, w)
    excluded = np.isin(cand_npz["repo_id"], list(EXCLUDED_REPOS))
    excluded_stats = {
        "repos": list(EXCLUDED_REPOS),
        "frames": int(excluded.sum()),
        "core_frames": int((excluded & cand_npz["core"]).sum()),
        "reason": "owner amendment 2026-08-10 13:14Z: wraparound-unit truth",
    }
    print(f"excluded rows: {excluded_stats}")
    keep = (nvalid > 0) & core & ~excluded
    splits = {
        "pooled": keep,
        "clean": keep & ~contam,
        "contaminated": keep & contam,
    }

    def pooled_pair(err: np.ndarray, sel: np.ndarray) -> dict:
        return {
            "chunk_mae": round(bbr.pooled_chunk(err, sel, w), 5),
            "first_mae": round(bbr.pooled_first(err, valid, sel), 5),
        }

    out: dict[str, Any] = {
        "note": (
            "RECORD-ONLY out-of-band reference: released MolmoAct2-SO100_101 "
            "vs banked arms, ALL numbers pooled over chunk steps 0..29 "
            "(their native 1.0 s horizon) unless marked full-50"
        ),
        "window_steps": WINDOW,
        "contamination": contam_stats,
        "excluded": excluded_stats,
        "matched_window": {},
        "paired_reads": {},
        "secondary_full50": {},
        "state_copy": "byte-match (all arms)",
    }
    for name, sel in splits.items():
        out["matched_window"].setdefault("molmoact2", {})[name] = pooled_pair(
            err_cand,
            sel,
        )
    # state-copy floor on the window (copied rows live in the cand npz)
    err_copy = np.abs(cand_npz["pred:state-copy"][:, :WINDOW] - truth)
    for name, sel in splits.items():
        out["matched_window"].setdefault("state-copy", {})[name] = pooled_pair(
            err_copy,
            sel,
        )

    for npz, _rep, spec in baselines:
        wv = window_view(npz, spec["key"])
        err_b = np.abs(wv[spec["key"]] - truth)
        frame_b, _ = bbr.frame_mae(err_b, w)
        for name, sel in splits.items():
            out["matched_window"].setdefault(spec["label"], {})[name] = pooled_pair(
                err_b,
                sel,
            )
            deltas = (frame_cand - frame_b)[sel]
            lo, hi = bbr.bootstrap_ci(deltas)
            cls = _classify(lo, hi)
            out["paired_reads"].setdefault(spec["label"], {})[name] = {
                "delta_frame_mean": round(float(deltas.mean()), 5),
                "ci95": [round(lo, 5), round(hi, 5)],
                "n_frames": int(sel.sum()),
                "classification": cls,
            }
            print(
                f"read (molmoact2 − {spec['label']}) [{name}]: "
                f"{deltas.mean():+.4f}  CI95 [{lo:+.5f}, {hi:+.5f}]  "
                f"({cls}, n={int(sel.sum())})",
            )
        # full-50 secondary anchor, ours only, never paired vs theirs;
        # banked (as-reported, incl. excluded repos) AND with the owner
        # exclusion applied, so the bbox-2 effect is visible (owner 14:33Z)
        truth50, valid50, core50, w50 = bbr.masks(npz)
        err50 = np.abs(npz[spec["key"]] - truth50)
        excl50 = core50 & ~excluded
        out["secondary_full50"][spec["label"]] = {
            "chunk_mae": round(bbr.pooled_chunk(err50, core50, w50), 5),
            "first_mae": round(bbr.pooled_first(err50, valid50, core50), 5),
            "chunk_mae_excl": round(bbr.pooled_chunk(err50, excl50, w50), 5),
            "first_mae_excl": round(bbr.pooled_first(err50, valid50, excl50), 5),
        }

    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=1))
        print(f"wrote {out_path}")
    return out


# ------------------------------------------------------------------ oracle


def _fixture(delta: float, chunk: int = 8, window: int = 4) -> tuple:
    n, dims = 12, 2
    truth = np.zeros((n, chunk, dims), dtype=np.float32)
    state = np.full((n, chunk, dims), 7.0, dtype=np.float32)
    base = {
        "index": np.arange(n, dtype=np.int64),
        "truth": truth,
        "valid": np.ones((n, chunk), dtype=bool),
        "repo_id": np.array([f"repo{i % 3}" for i in range(n)]),
        "core": np.array([True] * 9 + [False] * 3),
        "pred:state-copy": state,
        "pred:state-copy-norm": state + 1,
    }
    cand_pred = np.full((n, chunk, dims), np.nan, dtype=np.float32)
    cand_pred[:, :window] = 3.0 + delta
    cand = dict({k: v.copy() for k, v in base.items()}, **{CAND_KEY: cand_pred})

    def rep(npz: dict, key: str, run: str) -> dict:
        truth_, valid_, core_, w_ = bbr.masks(npz)
        err = np.abs(npz[key] - truth_)
        return {
            "checkpoint": f"outputs/train/{run}/step_0",
            "summaries": [
                {
                    "policy": key.removeprefix("pred:"),
                    "chunk_mae": bbr.pooled_chunk(err, core_, w_),
                    "first_mae": bbr.pooled_first(err, valid_, core_),
                },
            ],
        }

    baselines = []
    for spec in BASELINES:
        npz = dict(
            {k: v.copy() for k, v in base.items()},
            **{spec["key"]: np.full((n, chunk, dims), 3.0, dtype=np.float32)},
        )
        baselines.append((npz, rep(npz, spec["key"], spec["run"]), spec))
    return cand, baselines


def oracle() -> None:
    global WINDOW  # noqa: PLW0603 — fixtures use a small window
    window_saved = WINDOW
    WINDOW = 4
    contam_set = {"repo1"}
    # repo pattern repo0/1/2 over 12 rows, core = first 9: repo1 rows are
    # 1,4,7,10 -> frames 4, core 3
    expect = {"repos": 1, "frames": 4, "core_frames": 3}

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

    try:
        cand, baselines = _fixture(-1.0)
        out = analyze(cand, baselines, contam_set, expect, None)
        for spec in BASELINES:
            for split in ("pooled", "clean", "contaminated"):
                r = out["paired_reads"][spec["label"]][split]
                assert r["delta_frame_mean"] == -1.0, out
                assert r["ci95"] == [-1.0, -1.0], out
                assert r["classification"] == "MOLMOACT2-BETTER", out
        assert out["paired_reads"][BASELINES[0]["label"]]["pooled"]["n_frames"] == 9
        assert out["paired_reads"][BASELINES[0]["label"]]["clean"]["n_frames"] == 6
        assert (
            out["paired_reads"][BASELINES[0]["label"]]["contaminated"]["n_frames"] == 3
        )
        assert out["contamination"]["repos"] == 1
        assert out["excluded"]["frames"] == 0  # no fixture repo matches
        print("  planted −1.0 OK (exact Δ, degenerate CI, splits 9/6/3)")

        # owner-amendment exclusion: dropping repo2 (core rows 2, 5, 8)
        # shrinks pooled 9 -> 6 and clean 6 -> 3; contaminated untouched.
        global EXCLUDED_REPOS  # noqa: PLW0603
        excl_saved = EXCLUDED_REPOS
        EXCLUDED_REPOS = ("repo2",)
        try:
            out = analyze(cand, baselines, contam_set, expect, None)
        finally:
            EXCLUDED_REPOS = excl_saved
        pr = out["paired_reads"][BASELINES[0]["label"]]
        assert pr["pooled"]["n_frames"] == 6, out
        assert pr["clean"]["n_frames"] == 3, out
        assert pr["contaminated"]["n_frames"] == 3, out
        assert out["excluded"] == {
            "repos": ["repo2"],
            "frames": 4,
            "core_frames": 3,
            "reason": "owner amendment 2026-08-10 13:14Z: wraparound-unit truth",
        }, out
        print("  exclusion OK (repo2 dropped: splits 6/3/3, stats recorded)")

        cand, baselines = _fixture(0.0)
        out = analyze(cand, baselines, contam_set, expect, None)
        for spec in BASELINES:
            assert (
                out["paired_reads"][spec["label"]]["pooled"]["classification"]
                == "CI-SPANS-0"
            ), out
        print("  planted 0 OK (CI-SPANS-0)")

        cand, baselines = _fixture(1.0)
        out = analyze(cand, baselines, contam_set, expect, None)
        for spec in BASELINES:
            assert (
                out["paired_reads"][spec["label"]]["pooled"]["classification"]
                == "MOLMOACT2-WORSE"
            ), out
        # state-copy floor: |7-0|=7 on the window, both splits populated
        assert out["matched_window"]["state-copy"]["pooled"]["chunk_mae"] == 7.0
        print("  planted +1.0 OK (MOLMOACT2-WORSE + state-copy floor 7.0)")

        cand, baselines = _fixture(-1.0)
        mut = {k: v.copy() for k, v in baselines[0][0].items()}
        mut["truth"][0] += 1.0
        expect_exit(
            lambda: analyze(
                cand,
                [(mut, baselines[0][1], baselines[0][2]), *baselines[1:]],
                contam_set,
                expect,
                None,
            ),
            "pairing broken",
            "identity drift",
        )
        mut = {k: v.copy() for k, v in baselines[1][0].items()}
        mut["pred:state-copy"][1] += 1.0
        expect_exit(
            lambda: analyze(
                cand,
                [baselines[0], (mut, baselines[1][1], baselines[1][2]), *baselines[2:]],
                contam_set,
                expect,
                None,
            ),
            "byte-match",
            "state-copy drift",
        )
        bad_cand = {k: v.copy() for k, v in cand.items()}
        bad_cand[CAND_KEY][0, WINDOW] = 5.0
        expect_exit(
            lambda: analyze(bad_cand, baselines, contam_set, expect, None),
            "horizon contract",
            "NaN-tail violation",
        )
        bad_cand = {k: v.copy() for k, v in cand.items()}
        bad_cand[CAND_KEY][0, 0] = np.nan
        expect_exit(
            lambda: analyze(bad_cand, baselines, contam_set, expect, None),
            "non-finite",
            "window hole",
        )
        expect_exit(
            lambda: analyze(
                cand,
                baselines,
                contam_set,
                {"repos": 2, "frames": 4, "core_frames": 3},
                None,
            ),
            "drifted",
            "contamination drift",
        )
        bad_rep = dict(
            baselines[0][1],
            summaries=[dict(baselines[0][1]["summaries"][0], chunk_mae=9.9)],
        )
        expect_exit(
            lambda: analyze(
                cand,
                [(baselines[0][0], bad_rep, baselines[0][2]), *baselines[1:]],
                contam_set,
                expect,
                None,
            ),
            "plan/scoring drift",
            "report drift",
        )
        print("oracle: ALL branches OK")
    finally:
        WINDOW = window_saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--stem-cand", default=CAND_STEM)
    parser.add_argument("--out", default=OUT_DEFAULT)
    parser.add_argument("--contam-out", default=CONTAM_OUT_DEFAULT)
    args = parser.parse_args()
    if args.oracle:
        oracle()
        return
    baselines = [
        (
            _load_npz(f"{spec['stem']}.npz"),
            json.loads(Path(f"{spec['stem']}.json").read_text()),
            spec,
        )
        for spec in BASELINES
    ]
    analyze(
        _load_npz(f"{args.stem_cand}.npz"),
        baselines,
        mixture_repo_set(),
        CONTAM_EXPECT,
        args.out,
        args.contam_out,
    )


if __name__ == "__main__":
    main()
