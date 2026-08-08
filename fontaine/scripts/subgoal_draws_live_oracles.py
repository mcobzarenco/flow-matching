"""Live pre-launch oracles for subgoal-draws selection (#6 rung (b)).

The CPU halves are pinned in ``tests/test_subgoal_draws.py`` (scorer
fixtures, provenance separation, request-set exclusion, tiny-model
decode loop). This script is the REAL-CHECKPOINT half, run on the
q4-subset live runs BEFORE the stage-2 arms launch (pre-reg
2026-08-08-prereg-subgoal-draws.md, "Instrument … oracles
(abort-on-red before launch)"):

  (i)  the draws-0 limit (greedy candidate only) reproduces the
       rung-(a) self-arm decode BIT-EXACT — adjudicated against a
       FRESH q4 self-mode run from the same launcher, so both sides
       share batch composition by construction (the amendment-1
       lesson: banked full-panel npzs are composition-mismatched
       comparators; kernel numerics flip near-tie argmaxes). Pass-1
       narr columns must match too, and the candidates dump must show
       exactly one candidate per row, bon pick 0 everywhere, and the
       greedy candidate text byte-equal to the self run's dumped
       pass-1 subgoal;
  (ii) forced-empty reproduces the plain path: both selection arms of
       a draws-0 --selfsubgoal-force-empty run BIT-EXACT vs the BANKED
       rung-(a) q4 emptyhint decode (itself adjudicated bit-exact to a
       plain baseline at this composition) — same plan, same batching,
       so the comparison is composition-matched;
  plus execution guards: identity columns byte-agree across all four
  npzs, state-copy / state-copy-norm rows byte-match the banked run,
  and every report carries the mode/width/force-empty/plan it claims.

NO scalar (MAE or otherwise) is computed or printed — stage-2 scalars
stay behind the stage-1 go/no-go. Every failure is a hard abort
(SystemExit). On green, writes a machine-readable summary JSON (the
arms launcher refuses to run without it). ``--selftest`` exercises the
pass path and every abort branch on synthetic fixtures, no GPU needed.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

SELF_KEY = "pred:bijou@100000_selfsubgoal"
NARR_KEY = "pred:bijou@100000_narrsubgoal"
BON_KEY = "pred:bijou@100000_bonsubgoal"
CEIL_KEY = "pred:bijou@100000_ceilsubgoal"
BON_EMPTY_KEY = "pred:bijou@100000_bonsubgoal_emptyhint"
CEIL_EMPTY_KEY = "pred:bijou@100000_ceilsubgoal_emptyhint"
BANKED_EMPTY_KEY = "pred:bijou@100000_selfsubgoal_emptyhint"
IDENTITY_KEYS = ("index", "truth", "valid", "repo_id", "core")
STATE_KEYS = ("pred:state-copy", "pred:state-copy-norm")
Q4_PLAN = "plans/holdout_curated_v0_k4l2_stateprobe_q4.json"


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def _bytes_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _diff_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.flatnonzero((a != b).reshape(len(a), -1).any(axis=1))


def check_identity(ref: dict, probe: dict, label: str) -> None:
    for key in IDENTITY_KEYS:
        if key not in probe:
            sys.exit(f"{label}: identity key {key} missing from the npz")
        if not np.array_equal(ref[key], probe[key]):
            sys.exit(
                f"{label}: identity column {key} disagrees with the reference "
                "run — not the same plan/order, composition match broken, stop",
            )


def check_state_rows(ref: dict, probe: dict, label: str) -> None:
    for key in STATE_KEYS:
        if key not in probe:
            sys.exit(f"{label}: {key} missing from the npz")
        if not _bytes_equal(ref[key], probe[key]):
            sys.exit(
                f"{label}: {key} rows do NOT byte-match the banked run — "
                "execution drift, stop",
            )


def check_report(
    report: dict,
    *,
    mode: str,
    draws: int | None,
    force_empty: bool,
    label: str,
) -> None:
    if report.get("subgoal_mode") != mode:
        sys.exit(
            f"{label}: report subgoal_mode is {report.get('subgoal_mode')!r}, "
            f"expected {mode!r} — wrong run for this check",
        )
    if mode == "draws" and report.get("subgoal_draws") != draws:
        sys.exit(
            f"{label}: report subgoal_draws is {report.get('subgoal_draws')!r}, "
            f"expected {draws} — wrong run for this check",
        )
    if bool(report.get("selfsubgoal_force_empty")) != force_empty:
        sys.exit(
            f"{label}: report selfsubgoal_force_empty is "
            f"{report.get('selfsubgoal_force_empty')!r}, expected {force_empty}",
        )
    plan = str(report.get("sample_plan") or "")
    if Path(plan).name != Path(Q4_PLAN).name:
        sys.exit(
            f"{label}: report sample_plan is {plan!r}, expected the q4 subset "
            f"plan {Q4_PLAN!r} — wrong rows",
        )
    if report.get("seed") != 0:
        sys.exit(f"{label}: report seed is {report.get('seed')!r}, expected 0")


def check_draws0_vs_self(
    selfrun: dict,
    self_report: dict,
    self_subgoals: dict,
    draws0: dict,
    draws0_report: dict,
    candidates: dict,
) -> None:
    """Oracle (i): the draws-0 limit == the rung-(a) self decode,
    bit-exact at matched composition, decode AND text level."""
    label = "oracle-i (draws-0 limit)"
    check_report(
        self_report,
        mode="self",
        draws=None,
        force_empty=False,
        label=f"{label}/self",
    )
    check_report(
        draws0_report,
        mode="draws",
        draws=0,
        force_empty=False,
        label=f"{label}/draws0",
    )
    for key, side in ((SELF_KEY, selfrun), (NARR_KEY, selfrun)):
        if key not in side:
            sys.exit(f"{label}: {key} missing from the fresh self run npz")
    for key in (BON_KEY, CEIL_KEY, NARR_KEY):
        if key not in draws0:
            sys.exit(f"{label}: {key} missing from the draws-0 npz")
    check_identity(selfrun, draws0, label)
    if not _bytes_equal(selfrun[NARR_KEY], draws0[NARR_KEY]):
        diff = _diff_rows(selfrun[NARR_KEY], draws0[NARR_KEY])
        sys.exit(
            f"{label}: pass-1 narr decode is NOT bit-exact between self and "
            f"draws-0 modes ({len(diff)} rows differ, first {diff[:3].tolist()}) "
            "— the shared pass-1 path diverged, stop",
        )
    if not _bytes_equal(selfrun[SELF_KEY], draws0[BON_KEY]):
        diff = _diff_rows(selfrun[SELF_KEY], draws0[BON_KEY])
        sys.exit(
            f"{label}: bon arm at draws-0 is NOT bit-exact to the self arm "
            f"({len(diff)} rows differ, first {diff[:3].tolist()}) — the "
            "greedy-only limit does not reproduce rung (a), stop",
        )
    rows = candidates.get("rows")
    if not isinstance(rows, list) or not rows:
        sys.exit(f"{label}: candidates dump has no 'rows' — wrong file")
    if len(rows) != len(draws0["index"]):
        sys.exit(
            f"{label}: candidates dump covers {len(rows)} rows, npz has "
            f"{len(draws0['index'])} — pass-1 capture incomplete, stop",
        )
    self_by_index = {r["index"]: r for r in self_subgoals["rows"]}
    for row in rows:
        cands = row.get("candidates") or []
        if len(cands) != 1:
            sys.exit(
                f"{label}: frame {row['index']} has {len(cands)} candidates "
                "in the draws-0 limit — expected exactly the greedy one, stop",
            )
        if row["picks"]["bon"] != 0:
            sys.exit(
                f"{label}: frame {row['index']} bon pick is "
                f"{row['picks']['bon']!r} — expected 0 (single candidate), stop",
            )
        ref = self_by_index.get(row["index"])
        if ref is None:
            sys.exit(
                f"{label}: frame {row['index']} missing from the self run's "
                "subgoal dump — coverage broken, stop",
            )
        if row["greedy_subgoal"] != ref["generated_subgoal"]:
            sys.exit(
                f"{label}: frame {row['index']} greedy candidate text "
                f"{row['greedy_subgoal']!r} != self-mode pass-1 text "
                f"{ref['generated_subgoal']!r} — text-level drift, stop",
            )
    print(
        f"{label}: PASS — {len(rows)} rows bit-exact (narr + bon decodes) and "
        "text-exact (greedy candidate == pass-1 subgoal), bon pick 0 everywhere",
    )


def check_forced_empty(
    banked_empty: dict,
    empty: dict,
    empty_report: dict,
) -> None:
    """Oracle (ii): forced-empty selection arms == the plain path,
    via the BANKED rung-(a) q4 emptyhint decode (matched composition)."""
    label = "oracle-ii (forced-empty)"
    check_report(
        empty_report,
        mode="draws",
        draws=0,
        force_empty=True,
        label=label,
    )
    if BANKED_EMPTY_KEY not in banked_empty:
        sys.exit(f"{label}: {BANKED_EMPTY_KEY} missing — wrong banked npz")
    check_identity(banked_empty, empty, label)
    for key in (BON_EMPTY_KEY, CEIL_EMPTY_KEY):
        if key not in empty:
            sys.exit(f"{label}: {key} missing from the forced-empty npz")
        if not _bytes_equal(banked_empty[BANKED_EMPTY_KEY], empty[key]):
            diff = _diff_rows(banked_empty[BANKED_EMPTY_KEY], empty[key])
            sys.exit(
                f"{label}: {key} is NOT bit-exact to the banked emptyhint "
                f"decode ({len(diff)} rows differ, first {diff[:3].tolist()}) "
                "— forced-empty does not reproduce the plain path, stop",
            )
    check_state_rows(banked_empty, empty, label)
    print(
        f"{label}: PASS — both selection arms bit-exact vs the banked "
        f"emptyhint decode on {len(empty['index'])} rows",
    )


def adjudicate(
    selfrun: dict,
    self_report: dict,
    self_subgoals: dict,
    draws0: dict,
    draws0_report: dict,
    candidates: dict,
    empty: dict,
    empty_report: dict,
    banked_empty: dict,
    out: Path | None,
) -> None:
    check_identity(banked_empty, selfrun, "identity (self vs banked)")
    check_state_rows(banked_empty, selfrun, "state rows (self vs banked)")
    check_state_rows(banked_empty, draws0, "state rows (draws0 vs banked)")
    check_draws0_vs_self(
        selfrun,
        self_report,
        self_subgoals,
        draws0,
        draws0_report,
        candidates,
    )
    check_forced_empty(banked_empty, empty, empty_report)
    if out is not None:
        out.write_text(
            json.dumps(
                {
                    "verdict": "GREEN",
                    "adjudicated_utc": datetime.now(UTC).isoformat(),
                    "rows": len(selfrun["index"]),
                    "oracle_i": "draws-0 bon+narr bit-exact vs fresh self run; "
                    "candidate texts exact; bon pick 0 everywhere",
                    "oracle_ii": "forced-empty bon+ceil bit-exact vs banked "
                    "q4 emptyhint decode",
                    "composition": "all comparisons at matched q4 composition "
                    "(amendment-1 semantics by construction)",
                },
                indent=2,
            ),
        )
        print(f"wrote green summary to {out}")
    print(
        "LIVE ORACLES: ALL GREEN — stage-2 arms launch is unblocked "
        "(stage-1 go/no-go still applies)",
    )


# ---------------------------------------------------------------- selftest


def _fixture(root: Path) -> dict[str, Path]:
    rng = np.random.default_rng(0)
    n = 6
    ident = {
        "index": np.arange(100, 100 + n, dtype=np.int64),
        "truth": rng.normal(size=(n, 4, 2)).astype(np.float32),
        "valid": np.ones((n, 4), dtype=bool),
        "repo_id": np.array([f"repo{i % 2}" for i in range(n)]),
        "core": np.ones(n, dtype=bool),
        "pred:state-copy": rng.normal(size=(n, 4, 2)).astype(np.float32),
        "pred:state-copy-norm": rng.normal(size=(n, 4, 2)).astype(np.float32),
    }
    plain = rng.normal(size=(n, 4, 2)).astype(np.float32)
    narr = rng.normal(size=(n, 4, 2)).astype(np.float32)
    conditioned = rng.normal(size=(n, 4, 2)).astype(np.float32)

    banked_empty = dict(ident)
    banked_empty[BANKED_EMPTY_KEY] = plain.copy()
    selfrun = dict(ident)
    selfrun[NARR_KEY] = narr.copy()
    selfrun[SELF_KEY] = conditioned.copy()
    draws0 = dict(ident)
    draws0[NARR_KEY] = narr.copy()
    draws0[BON_KEY] = conditioned.copy()
    draws0[CEIL_KEY] = conditioned.copy()
    empty = dict(ident)
    empty[NARR_KEY] = narr.copy()
    empty[BON_EMPTY_KEY] = plain.copy()
    empty[CEIL_EMPTY_KEY] = plain.copy()

    texts = [f"align above part {i}" for i in range(n)]
    self_subgoals = {
        "rows": [
            {"index": int(ix), "generated_subgoal": texts[i], "true_subgoal": None}
            for i, ix in enumerate(ident["index"])
        ],
    }
    candidates = {
        "subgoal_draws": 0,
        "rows": [
            {
                "index": int(ix),
                "greedy_subgoal": texts[i],
                "candidates": [{"text": texts[i]}],
                "picks": {"bon": 0, "ceil": 0},
            }
            for i, ix in enumerate(ident["index"])
        ],
    }
    paths = {
        "banked_empty": root / "banked_empty.npz",
        "self": root / "self.npz",
        "draws0": root / "draws0.npz",
        "empty": root / "empty.npz",
        "self_subgoals": root / "self_subgoals.json",
        "candidates": root / "candidates.json",
    }
    np.savez(paths["banked_empty"], **banked_empty)
    np.savez(paths["self"], **selfrun)
    np.savez(paths["draws0"], **draws0)
    np.savez(paths["empty"], **empty)
    paths["self_subgoals"].write_text(json.dumps(self_subgoals))
    paths["candidates"].write_text(json.dumps(candidates))
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

    rep_self = {
        "subgoal_mode": "self",
        "selfsubgoal_force_empty": False,
        "sample_plan": Q4_PLAN,
        "seed": 0,
    }
    rep_draws0 = {
        "subgoal_mode": "draws",
        "subgoal_draws": 0,
        "selfsubgoal_force_empty": False,
        "sample_plan": Q4_PLAN,
        "seed": 0,
    }
    rep_empty = dict(rep_draws0, selfsubgoal_force_empty=True)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = _fixture(root)
        banked = _load_npz(paths["banked_empty"])
        selfrun = _load_npz(paths["self"])
        draws0 = _load_npz(paths["draws0"])
        empty = _load_npz(paths["empty"])
        self_subgoals = json.loads(paths["self_subgoals"].read_text())
        candidates = json.loads(paths["candidates"].read_text())
        out = root / "green.json"

        adjudicate(
            selfrun,
            rep_self,
            self_subgoals,
            draws0,
            rep_draws0,
            candidates,
            empty,
            rep_empty,
            banked,
            out,
        )
        assert json.loads(out.read_text())["verdict"] == "GREEN"
        print("  pass path OK (green summary written)")

        defaults = {
            "selfrun": selfrun,
            "self_report": rep_self,
            "self_subgoals": self_subgoals,
            "draws0": draws0,
            "draws0_report": rep_draws0,
            "candidates": candidates,
            "empty": empty,
            "empty_report": rep_empty,
            "banked_empty": banked,
            "out": None,
        }

        def run(**kw: object) -> None:
            adjudicate(**{**defaults, **kw})  # type: ignore[arg-type]

        expect_abort(
            lambda: run(self_report=rep_draws0),
            "wrong run for this check",
            "self-report provenance",
        )
        expect_abort(
            lambda: run(draws0_report=dict(rep_draws0, subgoal_draws=8)),
            "subgoal_draws",
            "draws-0 width guard",
        )
        expect_abort(
            lambda: run(empty_report=dict(rep_empty, selfsubgoal_force_empty=False)),
            "selfsubgoal_force_empty",
            "force-empty flag guard",
        )
        expect_abort(
            lambda: run(draws0_report=dict(rep_draws0, sample_plan="plans/other.json")),
            "expected the q4 subset plan",
            "wrong-plan guard",
        )
        mut = {k: v.copy() for k, v in draws0.items()}
        mut[BON_KEY][2, 0, 0] += 1e-3
        expect_abort(
            lambda: run(draws0=mut),
            "does not reproduce rung (a)",
            "bon-vs-self drift",
        )
        mut = {k: v.copy() for k, v in draws0.items()}
        mut[NARR_KEY][1, 0, 0] += 1e-3
        expect_abort(
            lambda: run(draws0=mut),
            "shared pass-1 path diverged",
            "narr drift",
        )
        mut = {k: v.copy() for k, v in draws0.items()}
        mut["index"] = mut["index"].copy()
        mut["index"][0] += 1
        expect_abort(
            lambda: run(draws0=mut),
            "composition match broken",
            "identity drift",
        )
        mut = {k: v.copy() for k, v in selfrun.items()}
        mut["pred:state-copy"][1] += 1.0
        expect_abort(
            lambda: run(selfrun=mut),
            "do NOT byte-match the banked run",
            "state-copy drift",
        )
        mut_c = json.loads(json.dumps(candidates))
        mut_c["rows"][3]["candidates"].append({"text": "extra"})
        expect_abort(
            lambda: run(candidates=mut_c),
            "expected exactly the greedy one",
            "candidate-count guard",
        )
        mut_c = json.loads(json.dumps(candidates))
        mut_c["rows"][2]["picks"]["bon"] = 1
        expect_abort(
            lambda: run(candidates=mut_c),
            "expected 0 (single candidate)",
            "bon-pick guard",
        )
        mut_c = json.loads(json.dumps(candidates))
        mut_c["rows"][4]["greedy_subgoal"] = "something else"
        expect_abort(
            lambda: run(candidates=mut_c),
            "text-level drift",
            "greedy-text guard",
        )
        mut = {k: v.copy() for k, v in empty.items()}
        mut[CEIL_EMPTY_KEY][0, 0, 0] += 1e-3
        expect_abort(
            lambda: run(empty=mut),
            "does not reproduce the plain path",
            "forced-empty drift",
        )
        mut = {k: v.copy() for k, v in empty.items()}
        del mut[BON_EMPTY_KEY]
        expect_abort(
            lambda: run(empty=mut),
            "missing from the forced-empty npz",
            "missing-arm guard",
        )
    print("selftest: ALL branches OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--self-npz", type=Path)
    parser.add_argument("--self-json", type=Path)
    parser.add_argument("--self-subgoals", type=Path)
    parser.add_argument("--draws0-npz", type=Path)
    parser.add_argument("--draws0-json", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--empty-npz", type=Path)
    parser.add_argument("--empty-json", type=Path)
    parser.add_argument("--banked-empty-npz", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    required = [
        args.self_npz,
        args.self_json,
        args.self_subgoals,
        args.draws0_npz,
        args.draws0_json,
        args.candidates,
        args.empty_npz,
        args.empty_json,
        args.banked_empty_npz,
        args.out,
    ]
    if any(p is None for p in required):
        sys.exit("all ten input paths are required (or --selftest)")
    adjudicate(
        _load_npz(args.self_npz),
        json.loads(args.self_json.read_text()),
        json.loads(args.self_subgoals.read_text()),
        _load_npz(args.draws0_npz),
        json.loads(args.draws0_json.read_text()),
        json.loads(args.candidates.read_text()),
        _load_npz(args.empty_npz),
        json.loads(args.empty_json.read_text()),
        _load_npz(args.banked_empty_npz),
        args.out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
