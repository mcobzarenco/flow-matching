"""er_60k step_015000 owner-requested panel — record-only frozen reads.

Owner steering 2026-08-10 08:29Z: hub-copy the @15000 checkpoint, run
the eval panel locally, post the HTML report link. This instrument
produces the paired numbers quoted next to that link — RECORD-ONLY:
the er_60k decision point stays the pre-registered endpoint panel
(~08-11 ~12:00Z); nothing here gates or repoints anything.

  1. paired per-frame Δ (er15k − ar_40k endpoint, banked 6.0079),
     core rows, seeded bootstrap CI95 (seed 0, 10,000). Sign quoted
     as BELOW-BASELINE / ABOVE-BASELINE / CI-SPANS-0.
  2. same vs the ar_60k continuation (banked 5.8602).
  3. INTEGRITY state-copy / state-copy-norm rows byte-match across
     all three npzs — hard abort.

Execution oracles (each failure a hard abort): identity columns
byte-match across all npzs; each npz re-pools to its own report
summary (chunk + first, 5e-3); checkpoints in the reports name the
expected run dirs.

``--oracle``: planted-delta fixtures (−1.0 ⇒ exact Δ, degenerate CI,
BELOW-BASELINE on both legs; 0 ⇒ CI-SPANS-0; +1.0 ⇒ ABOVE-BASELINE)
+ every abort branch.

Pure CPU, read-only on inputs, deterministic (seeded bootstrap).
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

STEM_CAND = (
    "reports/eval__fontaine_molmo2_er_60k_ddp4__step_015000__panel_curated_v0_k4l2"
)
KEY_CAND = "pred:bijou@15000"
RUN_CAND = "fontaine_molmo2_er_60k_ddp4"
BASELINES = [
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
]
OUT_DEFAULT = "reports/analysis__er15k_panel_vs_banked_k4l2.json"
STATE_KEYS = ("pred:state-copy", "pred:state-copy-norm")
SUMMARY_TOL = 5e-3


def _load_npz(path: str | Path) -> dict:
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def _check_report(npz: dict, key: str, report: dict, run_dir: str, label: str) -> None:
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
    if abs(gc - wc) >= SUMMARY_TOL or abs(gf - wf) >= SUMMARY_TOL:
        sys.exit(
            f"{label}: npz re-pool {gc:.4f}/{gf:.4f} does not reproduce the "
            f"report's {wc:.4f}/{wf:.4f} — plan/scoring drift, stop",
        )
    print(f"{label}: report cross-check OK ({policy} chunk {gc:.4f} first {gf:.4f})")


def _classify(lo: float, hi: float) -> str:
    if hi < 0:
        return "BELOW-BASELINE"
    if lo > 0:
        return "ABOVE-BASELINE"
    return "CI-SPANS-0"


def analyze(
    cand_npz: dict,
    cand_rep: dict,
    baselines: list[tuple[dict, dict, dict]],
    out_path: str | None,
    key_cand: str = KEY_CAND,
) -> dict:
    # ---- execution oracles gate every number below ----
    for npz, _rep, spec in baselines:
        for key in bbr.PAIR_KEYS:
            if not np.array_equal(cand_npz[key], npz[key]):
                sys.exit(
                    f"panel pairing broken on {key} between er15k and "
                    f"{spec['label']} — stop",
                )
        for key in STATE_KEYS:
            a, b = cand_npz[key], npz[key]
            if not (
                a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
            ):
                sys.exit(
                    f"{key} rows do NOT byte-match between er15k and "
                    f"{spec['label']} — stop",
                )
    _check_report(cand_npz, key_cand, cand_rep, RUN_CAND, "candidate arm")
    for npz, rep, spec in baselines:
        _check_report(npz, spec["key"], rep, spec["run"], spec["label"])

    truth, valid, core, w = bbr.masks(cand_npz)
    err_c = np.abs(cand_npz[key_cand] - truth)
    cc = bbr.pooled_chunk(err_c, core, w)
    cf = bbr.pooled_first(err_c, valid, core)
    frame_c, nvalid = bbr.frame_mae(err_c, w)
    keep = (nvalid > 0) & core

    pooled: dict[str, Any] = {
        "er15k": {"chunk_mae": round(cc, 5), "first_mae": round(cf, 5)},
    }
    reads: dict[str, Any] = {}
    for npz, _rep, spec in baselines:
        err_b = np.abs(npz[spec["key"]] - truth)
        bc = bbr.pooled_chunk(err_b, core, w)
        bf = bbr.pooled_first(err_b, valid, core)
        frame_b, _ = bbr.frame_mae(err_b, w)
        deltas = (frame_c - frame_b)[keep]
        lo, hi = bbr.bootstrap_ci(deltas)
        cls = _classify(lo, hi)
        pooled[spec["label"]] = {
            "chunk_mae": round(bc, 5),
            "first_mae": round(bf, 5),
        }
        reads[spec["label"]] = {
            "delta_frame_mean": round(float(deltas.mean()), 5),
            "delta_pooled": round(cc - bc, 5),
            "ci95": [round(lo, 5), round(hi, 5)],
            "n_frames": int(keep.sum()),
            "classification": cls,
        }
        print(
            f"read (er15k − {spec['label']}): frame-mean "
            f"{deltas.mean():+.4f}  pooled {cc - bc:+.4f}  CI95 "
            f"[{lo:+.5f}, {hi:+.5f}]  ({cls}, n={int(keep.sum())})",
        )

    out: dict[str, Any] = {
        "note": (
            "RECORD-ONLY owner-requested mid-run peek at er_60k step_015000; "
            "the pre-registered decision point is the endpoint panel"
        ),
        "pooled": pooled,
        "paired_reads": reads,
        "state_copy": "byte-match (all arms)",
    }
    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=2))
        print(f"wrote {out_path}")
    return out


# ------------------------------------------------------------------ oracle


def _fixture(delta: float) -> tuple:
    n, chunk, dims = 10, 8, 2
    truth = np.zeros((n, chunk, dims), dtype=np.float32)
    state = np.full((n, chunk, dims), 7.0, dtype=np.float32)
    base = {
        "index": np.arange(n, dtype=np.int64),
        "truth": truth,
        "valid": np.ones((n, chunk), dtype=bool),
        "repo_id": np.array([f"repo{i % 2}" for i in range(n)]),
        "core": np.array([True] * 8 + [False] * 2),
        "pred:state-copy": state,
        "pred:state-copy-norm": state + 1,
    }
    cand = dict(
        {k: v.copy() for k, v in base.items()},
        **{KEY_CAND: np.full((n, chunk, dims), 3.0 + delta, dtype=np.float32)},
    )

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
    return cand, rep(cand, KEY_CAND, RUN_CAND), baselines


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

    cand, crep, baselines = _fixture(-1.0)
    out = analyze(cand, crep, baselines, None)
    for spec in BASELINES:
        r = out["paired_reads"][spec["label"]]
        assert r["delta_frame_mean"] == -1.0 and r["ci95"] == [-1.0, -1.0], out
        assert r["classification"] == "BELOW-BASELINE", out
    print("  planted −1.0 OK (exact Δ, degenerate CI, BELOW-BASELINE x2)")

    cand, crep, baselines = _fixture(0.0)
    out = analyze(cand, crep, baselines, None)
    for spec in BASELINES:
        assert out["paired_reads"][spec["label"]]["classification"] == "CI-SPANS-0", out
    print("  planted 0 OK (CI-SPANS-0 x2)")

    cand, crep, baselines = _fixture(1.0)
    out = analyze(cand, crep, baselines, None)
    for spec in BASELINES:
        r = out["paired_reads"][spec["label"]]
        assert r["classification"] == "ABOVE-BASELINE", out
    print("  planted +1.0 OK (ABOVE-BASELINE x2)")

    cand, crep, baselines = _fixture(-1.0)
    mut_npz = {k: v.copy() for k, v in baselines[0][0].items()}
    mut_npz["truth"][0] += 1.0
    mut = [(mut_npz, baselines[0][1], baselines[0][2]), baselines[1]]
    expect_exit(
        lambda: analyze(cand, crep, mut, None),
        "pairing broken",
        "identity drift",
    )
    mut_npz = {k: v.copy() for k, v in baselines[1][0].items()}
    mut_npz["pred:state-copy"][1] += 1.0
    mut = [baselines[0], (mut_npz, baselines[1][1], baselines[1][2])]
    expect_exit(
        lambda: analyze(cand, crep, mut, None),
        "byte-match",
        "state-copy drift",
    )
    bad = dict(crep, summaries=[dict(crep["summaries"][0], chunk_mae=9.9)])
    expect_exit(
        lambda: analyze(cand, bad, baselines, None),
        "plan/scoring drift",
        "report drift",
    )
    bad = dict(crep, checkpoint="outputs/train/other_run/step_0")
    expect_exit(
        lambda: analyze(cand, bad, baselines, None),
        "is not",
        "checkpoint mismatch",
    )
    print("oracle: ALL branches OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--stem-cand", default=STEM_CAND)
    parser.add_argument(
        "--key-cand",
        default=None,
        help="prediction key in the candidate npz (default: derived "
        "from --stem-cand's step_NNNNNN as pred:bijou@N)",
    )
    parser.add_argument("--out", default=OUT_DEFAULT)
    args = parser.parse_args()
    if args.oracle:
        oracle()
        return
    key_cand = args.key_cand
    if key_cand is None:
        import re

        m = re.search(r"step_0*([0-9]+)", args.stem_cand)
        key_cand = f"pred:bijou@{m.group(1)}" if m else KEY_CAND
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
        json.loads(Path(f"{args.stem_cand}.json").read_text()),
        baselines,
        args.out,
        key_cand=key_cand,
    )


if __name__ == "__main__":
    main()
