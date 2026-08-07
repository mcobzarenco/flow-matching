"""Live pre-launch oracles for the self-subgoal probe (#6 rung (a)).

The four pre-registered oracles' CPU halves are pinned in
``tests/test_selfsubgoal.py`` (prompt bytes, one rendering path,
request-set exclusion). This script is the REAL-CHECKPOINT half, run on
the q4-subset live runs BEFORE the stage-2 arms launch (pre-reg
2026-08-07-prereg-selfsubgoal-probe.md, "oracles (abort-on-red before
launch)"):

  (i)  the self arm with generation forced EMPTY
       (``--selfsubgoal-force-empty``, policy ``…_selfsubgoal_emptyhint``)
       reproduces the banked planner-less baseline decode BIT-EXACT on
       the joined rows;
  (ii) the oracle arm's LABEL-LESS rows decode bit-exact to baseline
       (label-less frames collate the baseline context byte-exact) —
       and at least one LABELED row differs (different prompt bytes
       must reach the decode: all-equal means the conditioning path is
       not wired);
  plus the execution oracle shared with every probe read: state-copy /
  state-copy-norm rows byte-match the banked panel npz.

NO scalar (MAE or otherwise) is computed or printed — stage 2's
scalars stay behind the stage-1 go/no-go. Every failure is a hard
abort (SystemExit). ``--selftest`` exercises the pass path and every
abort branch on synthetic fixtures, no data or GPU needed.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

BASELINE_KEY = "pred:bijou@100000"
EMPTYHINT_KEY = "pred:bijou@100000_selfsubgoal_emptyhint"
ORACLE_KEY = "pred:bijou@100000_oraclesubgoal"
IDENTITY_KEYS = ("truth", "valid", "repo_id", "core")
STATE_KEYS = ("pred:state-copy", "pred:state-copy-norm")


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def subset_rows(base: dict, probe: dict, label: str) -> np.ndarray:
    """Baseline row positions pairing the probe npz, probe order.

    The draws10_t1_results.py join contract: join on ``index``,
    identity columns re-checked on the joined rows; duplicates,
    missing rows or identity drift are hard aborts."""
    bi, pi = base["index"], probe["index"]
    if len(pi) > len(bi):
        sys.exit(
            f"{label}: probe npz has MORE rows ({len(pi)}) than baseline ({len(bi)})",
        )
    if len(np.unique(pi)) != len(pi):
        sys.exit(f"{label}: probe npz has duplicate index values — refusing the join")
    pos = {int(ix): i for i, ix in enumerate(bi)}
    missing = [int(ix) for ix in pi if int(ix) not in pos]
    if missing:
        sys.exit(
            f"{label}: {len(missing)} probe rows absent from the baseline "
            f"panel (first: {missing[:3]}) — subset join broken",
        )
    rows = np.array([pos[int(ix)] for ix in pi])
    for key in IDENTITY_KEYS:
        if not np.array_equal(base[key][rows], probe[key]):
            sys.exit(f"{label}: joined rows disagree with the baseline panel on {key}")
    return rows


def _bytes_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def check_state_rows(base: dict, probe: dict, rows: np.ndarray, label: str) -> None:
    for key in STATE_KEYS:
        if key not in probe:
            sys.exit(f"{label}: {key} missing from the probe npz")
        if not _bytes_equal(np.ascontiguousarray(base[key][rows]), probe[key]):
            sys.exit(
                f"{label}: {key} rows do NOT byte-match the banked panel "
                "values — execution drift, stop",
            )
    print(f"{label}: state-copy / state-copy-norm byte-match banked panel rows OK")


def check_provenance(report: dict, mode: str, *, force_empty: bool, label: str) -> None:
    if report.get("subgoal_mode") != mode:
        sys.exit(
            f"{label}: report subgoal_mode is {report.get('subgoal_mode')!r}, "
            f"expected {mode!r} — wrong run for this check",
        )
    if bool(report.get("selfsubgoal_force_empty")) != force_empty:
        sys.exit(
            f"{label}: report selfsubgoal_force_empty is "
            f"{report.get('selfsubgoal_force_empty')!r}, expected {force_empty} — "
            "wrong run for this check",
        )


def check_emptyhint(base: dict, probe: dict, report: dict) -> None:
    """Oracle (i): forced-empty pass 2 == banked baseline, bit-exact."""
    label = "oracle-i (emptyhint)"
    check_provenance(report, "self", force_empty=True, label=label)
    if EMPTYHINT_KEY not in probe:
        sys.exit(f"{label}: {EMPTYHINT_KEY} missing from the probe npz")
    rows = subset_rows(base, probe, label)
    if not _bytes_equal(
        np.ascontiguousarray(base[BASELINE_KEY][rows]),
        probe[EMPTYHINT_KEY],
    ):
        diff = np.flatnonzero(
            (base[BASELINE_KEY][rows] != probe[EMPTYHINT_KEY])
            .reshape(len(rows), -1)
            .any(axis=1),
        )
        sys.exit(
            f"{label}: forced-empty decode is NOT bit-exact to the banked "
            f"baseline ({len(diff)}/{len(rows)} rows differ, first "
            f"{diff[:3].tolist()}) — the no-hint limit does not reproduce "
            "the historical path, stop",
        )
    check_state_rows(base, probe, rows, label)
    print(f"{label}: PASS — {len(rows)} rows bit-exact vs {BASELINE_KEY}")


def check_oracle_arm(base: dict, probe: dict, report: dict, subgoals: dict) -> None:
    """Oracle (ii): label-less rows == baseline bit-exact; labeled rows
    must reach the decode (>=1 row differs)."""
    label = "oracle-ii (label-less)"
    check_provenance(report, "oracle", force_empty=False, label=label)
    if ORACLE_KEY not in probe:
        sys.exit(f"{label}: {ORACLE_KEY} missing from the probe npz")
    rows = subset_rows(base, probe, label)
    missing = [int(ix) for ix in probe["index"] if str(int(ix)) not in subgoals]
    if missing:
        sys.exit(
            f"{label}: {len(missing)} oracle rows have no subgoal record "
            f"(first: {missing[:3]}) — the label mask does not cover the run",
        )
    labeled = np.array(
        [
            subgoals[str(int(ix))].get("true_subgoal") is not None
            for ix in probe["index"]
        ],
    )
    if not labeled.any():
        sys.exit(f"{label}: zero labeled rows in the probe run — mask broken?")
    if labeled.all():
        sys.exit(f"{label}: zero label-less rows in the probe run — nothing to check")
    base_rows = base[BASELINE_KEY][rows]
    unlabeled = np.flatnonzero(~labeled)
    if not _bytes_equal(
        np.ascontiguousarray(base_rows[unlabeled]),
        np.ascontiguousarray(probe[ORACLE_KEY][unlabeled]),
    ):
        bad = unlabeled[
            (base_rows[unlabeled] != probe[ORACLE_KEY][unlabeled])
            .reshape(len(unlabeled), -1)
            .any(axis=1)
        ]
        sys.exit(
            f"{label}: {len(bad)} label-less rows do NOT decode bit-exact "
            f"to baseline (first {bad[:3].tolist()}) — the label-less "
            "context is not the baseline context, stop",
        )
    labeled_idx = np.flatnonzero(labeled)
    changed = (
        (base_rows[labeled_idx] != probe[ORACLE_KEY][labeled_idx])
        .reshape(len(labeled_idx), -1)
        .any(axis=1)
    )
    if not changed.any():
        sys.exit(
            f"{label}: ALL {len(labeled_idx)} labeled rows are bit-exact to "
            "baseline — different prompt bytes never reached the decode; "
            "the conditioning path is not wired, stop",
        )
    check_state_rows(base, probe, rows, label)
    print(
        f"{label}: PASS — {len(unlabeled)} label-less rows bit-exact; "
        f"{int(changed.sum())}/{len(labeled_idx)} labeled rows moved "
        "(wiring live; magnitudes stay behind the stage-1 gate)",
    )


# ---------------------------------------------------------------- selftest


def _fixture(root: Path) -> dict[str, Path]:
    rng = np.random.default_rng(0)
    n, sub = 8, 6
    base = {
        "index": np.arange(100, 100 + n, dtype=np.int64),
        "truth": rng.normal(size=(n, 4, 2)).astype(np.float32),
        "valid": np.ones((n, 4), dtype=bool),
        "repo_id": np.array([f"repo{i % 2}" for i in range(n)]),
        "core": np.ones(n, dtype=bool),
        BASELINE_KEY: rng.normal(size=(n, 4, 2)).astype(np.float32),
        "pred:state-copy": rng.normal(size=(n, 4, 2)).astype(np.float32),
        "pred:state-copy-norm": rng.normal(size=(n, 4, 2)).astype(np.float32),
    }
    rows = np.arange(sub)
    keep = {k: np.ascontiguousarray(v[rows]) for k, v in base.items()}
    empty = dict(keep)
    empty[EMPTYHINT_KEY] = keep.pop(BASELINE_KEY)
    oracle = dict(empty)
    del oracle[EMPTYHINT_KEY]
    pred = np.ascontiguousarray(base[BASELINE_KEY][rows]).copy()
    pred[0] += 1.0  # the one labeled row moves
    oracle[ORACLE_KEY] = pred
    subgoals = {
        str(int(ix)): {"true_subgoal": "pick up the block" if i == 0 else None}
        for i, ix in enumerate(empty["index"])
    }
    paths = {
        "base": root / "base.npz",
        "empty": root / "empty.npz",
        "oracle": root / "oracle.npz",
        "subgoals": root / "subgoals.json",
    }
    np.savez(paths["base"], **base)
    np.savez(paths["empty"], **empty)
    np.savez(paths["oracle"], **oracle)
    paths["subgoals"].write_text(json.dumps(subgoals))
    return paths


def selftest() -> int:
    def expect_abort(fn: Callable[[], None], fragment: str, label: str) -> None:
        try:
            fn()
        except SystemExit as err:
            if fragment not in str(err):
                raise AssertionError(
                    f"{label}: aborted with {err!r}, wanted {fragment!r}",
                ) from None
            print(f"  abort branch OK: {label}")
            return
        raise AssertionError(f"{label}: did not abort")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = _fixture(root)
        base = _load_npz(paths["base"])
        empty = _load_npz(paths["empty"])
        oracle = _load_npz(paths["oracle"])
        subgoals = json.loads(paths["subgoals"].read_text())
        rep_self = {"subgoal_mode": "self", "selfsubgoal_force_empty": True}
        rep_oracle = {"subgoal_mode": "oracle", "selfsubgoal_force_empty": False}

        check_emptyhint(base, empty, rep_self)
        check_oracle_arm(base, oracle, rep_oracle, subgoals)
        print("  pass path OK (both live oracles green on the fixture)")

        expect_abort(
            lambda: check_emptyhint(base, empty, rep_oracle),
            "wrong run for this check",
            "provenance mode",
        )
        expect_abort(
            lambda: check_emptyhint(base, empty, {"subgoal_mode": "self"}),
            "selfsubgoal_force_empty",
            "provenance force-empty",
        )
        mut = {k: v.copy() for k, v in empty.items()}
        mut[EMPTYHINT_KEY] = mut[EMPTYHINT_KEY].copy()
        mut[EMPTYHINT_KEY][2, 0, 0] += 1e-3
        expect_abort(
            lambda: check_emptyhint(base, mut, rep_self),
            "NOT bit-exact",
            "emptyhint drift",
        )
        mut = {k: v.copy() for k, v in empty.items()}
        mut["truth"][0] += 1.0
        expect_abort(
            lambda: check_emptyhint(base, mut, rep_self),
            "disagree with the baseline panel on truth",
            "identity drift",
        )
        mut = {k: v.copy() for k, v in empty.items()}
        mut["pred:state-copy"][1] += 1.0
        expect_abort(
            lambda: check_emptyhint(base, mut, rep_self),
            "do NOT byte-match the banked panel",
            "state-copy drift",
        )
        mut = {k: v.copy() for k, v in empty.items()}
        mut["index"] = mut["index"].copy()
        mut["index"][1] = mut["index"][0]
        expect_abort(
            lambda: check_emptyhint(base, mut, rep_self),
            "duplicate index",
            "duplicate join",
        )
        mut = {k: v.copy() for k, v in oracle.items()}
        mut[ORACLE_KEY] = mut[ORACLE_KEY].copy()
        mut[ORACLE_KEY][3, 0, 0] += 1e-3  # row 3 is label-less
        expect_abort(
            lambda: check_oracle_arm(base, mut, rep_oracle, subgoals),
            "label-less rows do NOT decode bit-exact",
            "label-less drift",
        )
        mut = {k: v.copy() for k, v in oracle.items()}
        mut[ORACLE_KEY] = np.ascontiguousarray(base[BASELINE_KEY][:6])  # nothing moves
        expect_abort(
            lambda: check_oracle_arm(base, mut, rep_oracle, subgoals),
            "not wired",
            "inert wiring",
        )
        expect_abort(
            lambda: check_oracle_arm(base, oracle, rep_oracle, {}),
            "no subgoal record",
            "mask coverage",
        )
    print("selftest: ALL branches OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--baseline-npz", type=Path)
    parser.add_argument("--emptyhint-npz", type=Path)
    parser.add_argument("--emptyhint-json", type=Path)
    parser.add_argument("--oracle-npz", type=Path)
    parser.add_argument("--oracle-json", type=Path)
    parser.add_argument("--subgoals-json", type=Path)
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    required = [
        args.baseline_npz,
        args.emptyhint_npz,
        args.emptyhint_json,
        args.oracle_npz,
        args.oracle_json,
        args.subgoals_json,
    ]
    if any(p is None for p in required):
        sys.exit("all six input paths are required (or --selftest)")
    base = _load_npz(args.baseline_npz)
    if BASELINE_KEY not in base:
        sys.exit(f"{BASELINE_KEY} missing from {args.baseline_npz} — wrong baseline")
    check_emptyhint(
        base,
        _load_npz(args.emptyhint_npz),
        json.loads(args.emptyhint_json.read_text()),
    )
    dump = json.loads(args.subgoals_json.read_text())
    rows = dump.get("rows")
    if not isinstance(rows, list) or not rows:
        sys.exit(f"{args.subgoals_json} has no 'rows' — not a --dump-subgoals file")
    records = {str(r["index"]): r for r in rows}
    check_oracle_arm(
        base,
        _load_npz(args.oracle_npz),
        json.loads(args.oracle_json.read_text()),
        records,
    )
    print(
        "LIVE ORACLES: ALL GREEN — stage-2 launch is unblocked (stage-1 gate still applies)",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
