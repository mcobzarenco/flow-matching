"""Live post-run oracles for the #6 rung-(c) mcselect run — the
REAL-DATA half of the instrument gates (pre-reg
2026-08-09-prereg-subgoal-mcselect.md; the CPU halves are pinned in
tests/test_mcselect.py: planted-informative KL fixture, tau-degeneracy,
decode-vs-teacher-forced identity on the real tiny decoder,
capture-off byte-equality).

Runs BEFORE mcselect_results.py in the launcher chain. Abort-grade
(every check the read script also makes is duplicated here cheaply —
defense in depth, and a red stops the chain before any number is
looked at):

  1. report ``mcselect_tau`` == 4.0 and ``candidates_sha256`` ==
     sha256(banked candidates file bytes);
  2. the npz carries the three ``mcselect:*`` keys and NEVER the bare
     baseline column;
  3. rows == the banked candidates width (all indices present, counts
     equal);
  4. KL finiteness == eligibility, elementwise (finite iff
     non-truncated);
  5. identity + state-copy rows byte-match the banked full-panel
     baseline npz on the joined rows (composition-INdependent — the
     rung-(a) amendment-1 convention).

Diagnostics (printed, never abort-grade, NO pooled scalar — stage-2
numbers stay behind the frozen read): candidate-0 conditioned decode
vs the banked rung-(a) self arm on joined rows, and the planner-less
reference vs the banked baseline — per-row flip counts + max |Δ|. The
amendment-1 finding stands: greedy AR decode is batch-composition-
sensitive at the kernel level, so byte-exactness vs npzs banked at
OTHER compositions is a falsified bar; the op-identity half of the
draft's oracle 3 is pinned at matched composition in the unit tests.

``--selftest`` exercises the pass path and every abort branch on
synthetic fixtures — no data, no GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

SCORES_STEM = (
    "reports/eval__bijou_arb_rcond_100k_ddp4__step_100000"
    "__stateprobe_q4_subgoalmcselect"
)
CAND_DEFAULT = (
    "reports/eval__bijou_arb_rcond_100k_ddp4__step_100000"
    "__stateprobe_q4_subgoalcleandraws_candidates.json"
)
BASE_STEM = "reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2"
SELF_STEM = f"{BASE_STEM}_selfsubgoal"

BASELINE_KEY = "pred:bijou@100000"
SELF_KEY = "pred:bijou@100000_selfsubgoal"
KL_KEY = "mcselect:kl"
CAND_PRED_KEY = "mcselect:cand_pred"
MASKED_PRED_KEY = "mcselect:pred_masked"
STATE_KEYS = ("pred:state-copy", "pred:state-copy-norm")
TAU_REQUIRED = 4.0


def _load_npz(path: str | Path) -> dict:
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def _bytes_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def subset_rows(base: dict, probe: dict, label: str) -> np.ndarray:
    """Positions of probe rows inside the banked panel npz (the q4
    subset join — identity triple keyed, order-preserving)."""
    key = {}
    for position, (repo, episode, frame) in enumerate(
        zip(base["repo_id"], base["episode_index"], base["frame_index"], strict=True),
    ):
        key[str(repo), int(episode), int(frame)] = position
    rows = []
    for repo, episode, frame in zip(
        probe["repo_id"],
        probe["episode_index"],
        probe["frame_index"],
        strict=True,
    ):
        position = key.get((str(repo), int(episode), int(frame)))
        if position is None:
            sys.exit(
                f"{label}: probe row ({repo}, {episode}, {frame}) absent "
                "from the banked panel npz — wrong plan or corpus, stop",
            )
        rows.append(position)
    return np.array(rows, dtype=np.int64)


def check_report(report: dict, candidates_sha: str) -> None:
    if report.get("mcselect_tau") != TAU_REQUIRED:
        sys.exit(
            f"report mcselect_tau={report.get('mcselect_tau')!r} != "
            f"{TAU_REQUIRED} — wrong scorer configuration, stop",
        )
    if report.get("candidates_sha256") != candidates_sha:
        sys.exit(
            "report candidates_sha256 does not match the banked rung-(b') "
            "candidates file — the run re-ranked a different width, stop",
        )


def check_npz_contract(scores: dict, candidates: dict) -> None:
    for key in (KL_KEY, CAND_PRED_KEY, MASKED_PRED_KEY):
        if key not in scores:
            sys.exit(f"scores npz missing {key} — not a mcselect dump, stop")
    if BASELINE_KEY in scores:
        sys.exit(
            "scores npz carries the bare bijou column — the baseline "
            "must never re-run, stop",
        )
    by_index = {int(row["index"]): row for row in candidates["rows"]}
    idx = scores["index"]
    missing = [int(i) for i in idx if int(i) not in by_index]
    if missing:
        sys.exit(
            f"{len(missing)} scores rows absent from the candidates file "
            f"(first: {missing[:3]}) — not the banked width, stop",
        )
    if len(idx) != len(by_index):
        sys.exit(
            f"scores npz has {len(idx)} rows, candidates file "
            f"{len(by_index)} — partial scorer run, stop",
        )
    kl = scores[KL_KEY]
    for position, index in enumerate(idx):
        cands = by_index[int(index)]["candidates"]
        for slot, cand in enumerate(cands):
            finite = bool(np.isfinite(kl[position, slot]))
            if cand["truncated"] and finite:
                sys.exit(
                    f"finite KL at a TRUNCATED candidate (row index "
                    f"{int(index)}, slot {slot}) — the producer scored "
                    "outside the clean filter, stop",
                )
            if not cand["truncated"] and not finite:
                sys.exit(
                    f"non-finite KL at an ELIGIBLE candidate (row index "
                    f"{int(index)}, slot {slot}) — scorer numerics "
                    "broken, stop",
                )


def check_state_rows(base: dict, scores: dict, rows: np.ndarray) -> None:
    for key in STATE_KEYS:
        if key not in scores:
            sys.exit(f"scores npz missing {key} — state baselines absent, stop")
        if not _bytes_equal(np.ascontiguousarray(base[key][rows]), scores[key]):
            sys.exit(
                f"{key} differs from the banked panel rows — the runs "
                "scored different frames or corpus state drifted, stop",
            )
    for key in ("truth", "valid"):
        if not _bytes_equal(np.ascontiguousarray(base[key][rows]), scores[key]):
            sys.exit(f"{key} differs from the banked panel rows — stop")


def decode_diagnostics(
    base: dict,
    self_npz: dict,
    scores: dict,
    rows: np.ndarray,
) -> None:
    """Composition-noise diagnostics, printed only (amendment-1
    precedent: byte-exactness vs other-composition npzs is a falsified
    bar; no pooled scalar before the frozen read)."""
    cand0 = scores[CAND_PRED_KEY][:, 0]
    banked_self = self_npz[SELF_KEY][rows]
    flips = int((~np.isclose(cand0, banked_self, atol=0.0).all(axis=(1, 2))).sum())
    delta = float(np.abs(cand0 - banked_self).max())
    print(
        f"diagnostic cand0-vs-banked-self: {flips}/{len(rows)} rows differ "
        f"anywhere, max |d| {delta:.4f} (batch-composition noise expected; "
        "record, not a gate)",
    )
    masked = scores[MASKED_PRED_KEY]
    banked_base = base[BASELINE_KEY][rows]
    flips = int((~np.isclose(masked, banked_base, atol=0.0).all(axis=(1, 2))).sum())
    delta = float(np.abs(masked - banked_base).max())
    print(
        f"diagnostic pred_masked-vs-banked-baseline: {flips}/{len(rows)} "
        f"rows differ anywhere, max |d| {delta:.4f} (same caveat)",
    )


def run(
    scores_stem: str,
    candidates_path: str,
    base_stem: str,
    self_stem: str,
) -> int:
    cand_bytes = Path(candidates_path).read_bytes()
    candidates = json.loads(cand_bytes.decode())
    candidates_sha = hashlib.sha256(cand_bytes).hexdigest()
    scores = _load_npz(f"{scores_stem}.npz")
    report = json.loads(Path(f"{scores_stem}.json").read_text())
    base = _load_npz(f"{base_stem}.npz")
    self_npz = _load_npz(f"{self_stem}.npz")
    check_report(report, candidates_sha)
    check_npz_contract(scores, candidates)
    rows = subset_rows(base, scores, "mcselect run")
    check_state_rows(base, scores, rows)
    if not np.array_equal(base["index"], self_npz["index"]):
        sys.exit(
            "banked self arm is not row-paired with the banked baseline "
            "— wrong self npz, stop",
        )
    decode_diagnostics(base, self_npz, scores, rows)
    print("mcselect live oracles: ALL ABORT-GRADE CHECKS GREEN")
    return 0


# ---------------------------------------------------------- selftest


def _fixture(root: Path) -> dict[str, str]:
    rng = np.random.default_rng(3)
    n, c, s, d = 5, 3, 4, 2
    idx = np.arange(10, 10 + n)
    repo = np.array([f"user/ds{i % 2}" for i in range(n)])
    episode = np.arange(n)
    frame = np.arange(n) * 7
    truth = rng.normal(size=(n, s, d)).astype(np.float32)
    valid = np.ones((n, s), dtype=bool)
    base_pred = truth + 0.1
    self_pred = truth + 0.2
    state = rng.normal(size=(n, s, d)).astype(np.float32)
    base = {
        "index": idx,
        "repo_id": repo,
        "episode_index": episode,
        "frame_index": frame,
        "truth": truth,
        "valid": valid,
        "core": np.ones(n, dtype=bool),
        BASELINE_KEY: base_pred,
        SELF_KEY: self_pred,
        "pred:state-copy": state,
        "pred:state-copy-norm": state + 1,
    }
    rows = []
    for i in range(n):
        cands = [
            {"text": "greedy", "truncated": False},
            {"text": "alt", "truncated": False},
            {"text": "cut", "truncated": True},
        ]
        rows.append({"index": int(idx[i]), "candidates": cands})
    candidates = {"rows": rows}
    kl = np.full((n, c), np.nan)
    kl[:, 0] = 0.1
    kl[:, 1] = 0.3
    scores = {
        "index": idx,
        "repo_id": repo,
        "episode_index": episode,
        "frame_index": frame,
        "truth": truth,
        "valid": valid,
        "core": np.ones(n, dtype=bool),
        "pred:state-copy": state,
        "pred:state-copy-norm": state + 1,
        KL_KEY: kl,
        CAND_PRED_KEY: np.stack(
            [
                np.stack(
                    [truth[i] + 0.2, truth[i] + 0.3, np.full_like(truth[i], np.nan)],
                )
                for i in range(n)
            ],
        ),
        MASKED_PRED_KEY: base_pred,
    }
    cand_path = root / "cands.json"
    cand_path.write_text(json.dumps(candidates))
    np.savez(root / "scores.npz", **scores)
    np.savez(root / "base.npz", **base)
    np.savez(
        root / "self.npz",
        **{
            k: base[k]
            for k in (
                "index",
                "repo_id",
                "episode_index",
                "frame_index",
                "truth",
                "valid",
                "core",
            )
        },
        **{SELF_KEY: self_pred},
    )
    report = {
        "mcselect_tau": 4.0,
        "candidates_sha256": hashlib.sha256(cand_path.read_bytes()).hexdigest(),
    }
    (root / "scores.json").write_text(json.dumps(report))
    return {
        "scores_stem": str(root / "scores"),
        "candidates": str(cand_path),
        "base_stem": str(root / "base"),
        "self_stem": str(root / "self"),
    }


def selftest() -> int:
    def expect_abort(fn: Callable[[], int], fragment: str, label: str) -> None:
        try:
            fn()
        except SystemExit as stop:
            message = str(stop)
            if fragment not in message:
                raise AssertionError(
                    f"{label}: aborted with {message!r}, expected {fragment!r}",
                ) from None
            print(f"  abort branch OK: {label}")
            return
        raise AssertionError(f"{label}: did not abort")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = _fixture(root)

        def go(**overrides: str) -> int:
            merged = {**paths, **overrides}
            return run(
                merged["scores_stem"],
                merged["candidates"],
                merged["base_stem"],
                merged["self_stem"],
            )

        assert go() == 0
        print("  pass path OK")

        # tau drift
        report_path = root / "scores.json"
        good = report_path.read_text()
        bad = json.loads(good)
        bad["mcselect_tau"] = 1.0
        report_path.write_text(json.dumps(bad))
        expect_abort(go, "mcselect_tau", "tau drift")
        report_path.write_text(good)

        # sha drift
        bad = json.loads(good)
        bad["candidates_sha256"] = "cafe"
        report_path.write_text(json.dumps(bad))
        expect_abort(go, "candidates_sha256", "sha drift")
        report_path.write_text(good)

        # finite KL at truncated slot
        scores = _load_npz(f"{paths['scores_stem']}.npz")
        broken = dict(scores)
        kl = scores[KL_KEY].copy()
        kl[0, 2] = 0.5
        broken[KL_KEY] = kl
        np.savez(root / "broken.npz", **broken)
        (root / "broken.json").write_text(good)
        expect_abort(
            lambda: go(scores_stem=str(root / "broken")),
            "TRUNCATED",
            "finite-at-truncated",
        )

        # NaN KL at eligible slot
        broken = dict(scores)
        kl = scores[KL_KEY].copy()
        kl[1, 0] = np.nan
        broken[KL_KEY] = kl
        np.savez(root / "broken2.npz", **broken)
        (root / "broken2.json").write_text(good)
        expect_abort(
            lambda: go(scores_stem=str(root / "broken2")),
            "ELIGIBLE",
            "nan-at-eligible",
        )

        # bare baseline column present
        broken = dict(scores)
        broken[BASELINE_KEY] = scores[MASKED_PRED_KEY]
        np.savez(root / "broken3.npz", **broken)
        (root / "broken3.json").write_text(good)
        expect_abort(
            lambda: go(scores_stem=str(root / "broken3")),
            "bare bijou column",
            "baseline re-run",
        )

        # partial run (row dropped)
        broken = {
            k: (v[:-1] if isinstance(v, np.ndarray) and v.shape[:1] == (5,) else v)
            for k, v in scores.items()
        }
        np.savez(root / "broken4.npz", **broken)
        (root / "broken4.json").write_text(good)
        expect_abort(
            lambda: go(scores_stem=str(root / "broken4")),
            "partial scorer run",
            "partial run",
        )

        # state-copy drift
        broken = dict(scores)
        broken["pred:state-copy"] = scores["pred:state-copy"] + 1e-3
        np.savez(root / "broken5.npz", **broken)
        (root / "broken5.json").write_text(good)
        expect_abort(
            lambda: go(scores_stem=str(root / "broken5")),
            "pred:state-copy",
            "state-copy drift",
        )

        # foreign row (identity not in the banked panel)
        broken = dict(scores)
        broken["frame_index"] = scores["frame_index"].copy()
        broken["frame_index"][0] = 9999
        np.savez(root / "broken6.npz", **broken)
        (root / "broken6.json").write_text(good)
        expect_abort(
            lambda: go(scores_stem=str(root / "broken6")),
            "absent from the banked panel",
            "foreign row",
        )

    print("selftest: ALL branches OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--scores-stem", default=SCORES_STEM)
    parser.add_argument("--candidates", default=CAND_DEFAULT)
    parser.add_argument("--baseline-stem", default=BASE_STEM)
    parser.add_argument("--self-stem", default=SELF_STEM)
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    return run(args.scores_stem, args.candidates, args.baseline_stem, args.self_stem)


if __name__ == "__main__":
    sys.exit(main())
