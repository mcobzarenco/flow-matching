"""Live pre-launch oracles for the self-subgoal probe (#6 rung (a)).

The four pre-registered oracles' CPU halves are pinned in
``tests/test_selfsubgoal.py`` (prompt bytes, one rendering path,
request-set exclusion). This script is the REAL-CHECKPOINT half, run on
the q4-subset live runs BEFORE the stage-2 arms launch (pre-reg
2026-08-07-prereg-selfsubgoal-probe.md + amendment 1, "oracles
(abort-on-red before launch)").

AMENDMENT 1 (2026-08-08, posted before launch): the first adjudication
run falsified "bit-exact vs the BANKED full-panel npz" as the oracle-(i)
comparator — greedy AR decode is batch-composition-sensitive at the
kernel level (padding/shape-dependent reduction order flips near-tie
argmaxes; measured 1207/4301 rows on the q4 subset with mean-zero
pooled effect −0.0008, and reproduced by a PLAIN baseline eval at q4
composition with zero instrument code involved). Quantiles are
per-item, so bins are composition-independent; only kernel numerics
move. The amended live oracles therefore compare at MATCHED batch
composition:

  (i)  emptyhint decode BIT-EXACT vs a plain-baseline decode of the
       same plan at the same batching (``--matched-baseline-npz``) —
       abort-grade;
  (ii) oracle-arm wiring live: >=1 labeled row differs from the matched
       baseline — abort-grade; the label-less byte-equality half is
       DESCRIPTIVE at decode level (label-bearing batchmates change
       padding, so composition noise is unavoidable there) and stands
       on the pinned CPU prompt-byte oracle;
  plus abort-grade execution oracles vs the BANKED panel npz, which are
  composition-independent: identity columns and state-copy /
  state-copy-norm rows byte-match, and the matched baseline itself must
  satisfy them too. The banked-vs-matched decode flip count is printed
  as the recorded composition-noise diagnostic.

NO scalar (MAE or otherwise) is computed or printed — stage 2's scalars
stay behind the stage-1 go/no-go. Every failure is a hard abort
(SystemExit). ``--selftest`` exercises the pass path and every abort
branch on synthetic fixtures, no data or GPU needed.
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


def check_provenance(
    report: dict,
    mode: str | None,
    *,
    force_empty: bool,
    label: str,
) -> None:
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


def check_matched_baseline(banked: dict, matched: dict, report: dict) -> None:
    """The matched-composition plain run: same subset, banked-grade
    execution oracles, decode flip count recorded (composition noise)."""
    label = "matched-baseline"
    check_provenance(report, None, force_empty=False, label=label)
    if BASELINE_KEY not in matched:
        sys.exit(f"{label}: {BASELINE_KEY} missing — not a plain baseline run")
    rows = subset_rows(banked, matched, label)
    check_state_rows(banked, matched, rows, label)
    flips = (
        (banked[BASELINE_KEY][rows] != matched[BASELINE_KEY])
        .reshape(len(rows), -1)
        .any(axis=1)
    )
    print(
        f"{label}: decode flips vs banked full-panel rows: "
        f"{int(flips.sum())}/{len(rows)} (recorded composition-noise "
        "diagnostic — amendment 1; prompts identical, kernel numerics only)",
    )


def check_emptyhint(banked: dict, matched: dict, probe: dict, report: dict) -> None:
    """Oracle (i), amendment-1 form: forced-empty pass 2 == the
    matched-composition plain baseline decode, bit-exact."""
    label = "oracle-i (emptyhint)"
    check_provenance(report, "self", force_empty=True, label=label)
    if EMPTYHINT_KEY not in probe:
        sys.exit(f"{label}: {EMPTYHINT_KEY} missing from the probe npz")
    rows = subset_rows(banked, probe, label)
    if not np.array_equal(matched["index"], probe["index"]):
        sys.exit(f"{label}: matched baseline and probe row order disagree — stop")
    if not _bytes_equal(matched[BASELINE_KEY], probe[EMPTYHINT_KEY]):
        diff = np.flatnonzero(
            (matched[BASELINE_KEY] != probe[EMPTYHINT_KEY])
            .reshape(len(rows), -1)
            .any(axis=1),
        )
        sys.exit(
            f"{label}: forced-empty decode is NOT bit-exact to the "
            f"matched-composition baseline ({len(diff)}/{len(rows)} rows "
            f"differ, first {diff[:3].tolist()}) — the no-hint limit does "
            "not reproduce the plain path, stop",
        )
    check_state_rows(banked, probe, rows, label)
    print(
        f"{label}: PASS — {len(rows)} rows bit-exact vs the "
        "matched-composition baseline decode",
    )


def check_oracle_arm(
    banked: dict,
    matched: dict,
    probe: dict,
    report: dict,
    subgoals: dict,
) -> None:
    """Oracle (ii), amendment-1 form: conditioning wiring live
    (>=1 labeled row differs, abort-grade); label-less decode equality
    descriptive (composition-confounded; CPU prompt half is the pin)."""
    label = "oracle-ii (wiring)"
    check_provenance(report, "oracle", force_empty=False, label=label)
    if ORACLE_KEY not in probe:
        sys.exit(f"{label}: {ORACLE_KEY} missing from the probe npz")
    rows = subset_rows(banked, probe, label)
    if not np.array_equal(matched["index"], probe["index"]):
        sys.exit(f"{label}: matched baseline and probe row order disagree — stop")
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
    labeled_idx = np.flatnonzero(labeled)
    changed = (
        (matched[BASELINE_KEY][labeled_idx] != probe[ORACLE_KEY][labeled_idx])
        .reshape(len(labeled_idx), -1)
        .any(axis=1)
    )
    if not changed.any():
        sys.exit(
            f"{label}: ALL {len(labeled_idx)} labeled rows are bit-exact to "
            "the matched baseline — different prompt bytes never reached the "
            "decode; the conditioning path is not wired, stop",
        )
    unlabeled = np.flatnonzero(~labeled)
    if len(unlabeled):
        ul_diff = (
            (matched[BASELINE_KEY][unlabeled] != probe[ORACLE_KEY][unlabeled])
            .reshape(len(unlabeled), -1)
            .any(axis=1)
        )
        print(
            f"{label}: label-less rows differing vs matched baseline: "
            f"{int(ul_diff.sum())}/{len(unlabeled)} (DESCRIPTIVE — "
            "composition-confounded by label-bearing batchmates; the "
            "prompt-byte equality pin is the CPU oracle)",
        )
    check_state_rows(banked, probe, rows, label)
    print(
        f"{label}: PASS — {int(changed.sum())}/{len(labeled_idx)} labeled "
        "rows moved (wiring live; magnitudes stay behind the stage-1 gate)",
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
    # matched baseline: same subset; one decode flip vs banked (row 4)
    # models the measured composition noise.
    matched = {k: v.copy() for k, v in keep.items()}
    matched[BASELINE_KEY] = matched[BASELINE_KEY].copy()
    matched[BASELINE_KEY][4] += 0.25
    empty = {k: v.copy() for k, v in matched.items()}
    empty[EMPTYHINT_KEY] = empty.pop(BASELINE_KEY)
    oracle = {k: v.copy() for k, v in matched.items()}
    del oracle[BASELINE_KEY]
    pred = matched[BASELINE_KEY].copy()
    pred[0] += 1.0  # the one labeled row moves
    oracle[ORACLE_KEY] = pred
    subgoals = {
        str(int(ix)): {"true_subgoal": "pick up the block" if i == 0 else None}
        for i, ix in enumerate(empty["index"])
    }
    paths = {
        "base": root / "base.npz",
        "matched": root / "matched.npz",
        "empty": root / "empty.npz",
        "oracle": root / "oracle.npz",
        "subgoals": root / "subgoals.json",
    }
    np.savez(paths["base"], **base)
    np.savez(paths["matched"], **matched)
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
        matched = _load_npz(paths["matched"])
        empty = _load_npz(paths["empty"])
        oracle = _load_npz(paths["oracle"])
        subgoals = json.loads(paths["subgoals"].read_text())
        rep_plain = {"subgoal_mode": None, "selfsubgoal_force_empty": False}
        rep_self = {"subgoal_mode": "self", "selfsubgoal_force_empty": True}
        rep_oracle = {"subgoal_mode": "oracle", "selfsubgoal_force_empty": False}

        check_matched_baseline(base, matched, rep_plain)
        check_emptyhint(base, matched, empty, rep_self)
        check_oracle_arm(base, matched, oracle, rep_oracle, subgoals)
        print("  pass path OK (matched-composition comparator, amendment 1)")

        expect_abort(
            lambda: check_matched_baseline(base, matched, rep_self),
            "wrong run for this check",
            "matched-baseline provenance",
        )
        expect_abort(
            lambda: check_emptyhint(base, matched, empty, rep_oracle),
            "wrong run for this check",
            "provenance mode",
        )
        expect_abort(
            lambda: check_emptyhint(base, matched, empty, {"subgoal_mode": "self"}),
            "selfsubgoal_force_empty",
            "provenance force-empty",
        )
        mut = {k: v.copy() for k, v in empty.items()}
        mut[EMPTYHINT_KEY] = mut[EMPTYHINT_KEY].copy()
        mut[EMPTYHINT_KEY][2, 0, 0] += 1e-3
        expect_abort(
            lambda: check_emptyhint(base, matched, mut, rep_self),
            "NOT bit-exact to the matched-composition baseline",
            "emptyhint drift vs matched",
        )
        mut = {k: v.copy() for k, v in empty.items()}
        mut["truth"][0] += 1.0
        expect_abort(
            lambda: check_emptyhint(base, matched, mut, rep_self),
            "disagree with the baseline panel on truth",
            "identity drift",
        )
        mut = {k: v.copy() for k, v in empty.items()}
        mut["pred:state-copy"][1] += 1.0
        expect_abort(
            lambda: check_emptyhint(base, matched, mut, rep_self),
            "do NOT byte-match the banked panel",
            "state-copy drift",
        )
        mut = {k: v.copy() for k, v in matched.items()}
        mut["pred:state-copy"][1] += 1.0
        expect_abort(
            lambda: check_matched_baseline(base, mut, rep_plain),
            "do NOT byte-match the banked panel",
            "matched-baseline state-copy drift",
        )
        mut = {k: v.copy() for k, v in empty.items()}
        mut["index"] = mut["index"].copy()
        mut["index"][1] = mut["index"][0]
        expect_abort(
            lambda: check_emptyhint(base, matched, mut, rep_self),
            "duplicate index",
            "duplicate join",
        )
        mut = {k: v.copy() for k, v in oracle.items()}
        mut[ORACLE_KEY] = matched[BASELINE_KEY].copy()  # nothing moves
        expect_abort(
            lambda: check_oracle_arm(base, matched, mut, rep_oracle, subgoals),
            "not wired",
            "inert wiring",
        )
        expect_abort(
            lambda: check_oracle_arm(base, matched, oracle, rep_oracle, {}),
            "no subgoal record",
            "mask coverage",
        )
    print("selftest: ALL branches OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--baseline-npz", type=Path)
    parser.add_argument("--matched-baseline-npz", type=Path)
    parser.add_argument("--matched-baseline-json", type=Path)
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
        args.matched_baseline_npz,
        args.matched_baseline_json,
        args.emptyhint_npz,
        args.emptyhint_json,
        args.oracle_npz,
        args.oracle_json,
        args.subgoals_json,
    ]
    if any(p is None for p in required):
        sys.exit("all eight input paths are required (or --selftest)")
    banked = _load_npz(args.baseline_npz)
    if BASELINE_KEY not in banked:
        sys.exit(f"{BASELINE_KEY} missing from {args.baseline_npz} — wrong baseline")
    matched = _load_npz(args.matched_baseline_npz)
    check_matched_baseline(
        banked,
        matched,
        json.loads(args.matched_baseline_json.read_text()),
    )
    check_emptyhint(
        banked,
        matched,
        _load_npz(args.emptyhint_npz),
        json.loads(args.emptyhint_json.read_text()),
    )
    dump = json.loads(args.subgoals_json.read_text())
    rows = dump.get("rows")
    if not isinstance(rows, list) or not rows:
        sys.exit(f"{args.subgoals_json} has no 'rows' — not a --dump-subgoals file")
    records = {str(r["index"]): r for r in rows}
    check_oracle_arm(
        banked,
        matched,
        _load_npz(args.oracle_npz),
        json.loads(args.oracle_json.read_text()),
        records,
    )
    print(
        "LIVE ORACLES: ALL GREEN (amendment-1 semantics) — stage-2 launch "
        "is unblocked (stage-1 gate still applies)",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
