"""#16 critical-frame re-pooling screen — RECORD-ONLY robustness read.

Pre-reg: posts/2026-08-07-prereg-critical-frame-repooling.md (lands
with this file, before any critical-pool number is read). The CI-MSE
transfer (papers/offline-validation.md): raw chunk MAE's metric class
measured Spearman -0.61 vs rollout success with a sign-flip case;
scoring only task-critical frames recovers -0.87. Our judge labels
already mark those frames, every leaderboard row dumped per-frame npz
— so re-pool the existing dumps over critical frames only and check
whether any published ranking reorders.

Frame-selection rule (frozen in the pre-reg, no tuned parameters): a
scored frame at 0-based within-episode index f0 has prediction window
W = [f0+1, f0+50] in the judge's 1-based coordinates; critical iff
its episode's blessed judgment has (1) a subgoal boundary in W (the
last until_frame, the episode end by contract, excluded), or (2) a
holding transition between consecutive judge-annotated frames whose
bracket intersects W, or (3) a judge-annotated frame with non-empty
events inside W. Blessed judgment = records matching the dataset's
meta/judge_annotations.json stamp, last judged_at per episode wins —
the bijou.data training-side selection rule, reused not re-derived.
Episodes without a valid blessed judgment are UNCOVERED: their frames
enter neither pool and are reported as coverage.

Reorder criterion (frozen): for scoreboard rows adjacent in published
rank, the paired per-frame MAE delta on the critical core pool with a
seeded frame-level bootstrap CI95 (n=10,000, seed 0 — the
box_batch_results machinery). REORDER = mean delta sign opposite to
the published gap AND CI95 excluding 0. All 10 pairs scanned as a
secondary. The A-s0/s1/s2 seed trio's internal spread is the
empirical null scale, never a verdict.

Guards (hard abort): missing dump / missing pred key; PAIR_KEYS
(truth/valid/index/repo_id/core) not byte-identical across dumps;
episode_index/frame_index mismatch where a dump carries them; overall
pooled chunk MAE not reproducing the published number to 5e-4;
critical + complement + uncovered not recombining to the overall
pooled value to 1e-6. Coverage gates (core-frame coverage >= 80%,
critical core pool >= 500 frames) DOWNGRADE to descriptive-only —
numbers print, verdicts are suppressed — per the pre-reg.

Oracle mode (--selftest, pure synthetic, runs before any real read):
  (a) rule fixtures — planted boundaries / holding brackets / event
      frames produce exactly the expected critical flags, window
      edges inclusive on both ends;
  (b) blessed-selection fixtures — wrong-stamp records ignored, last
      judged_at wins, unparseable judgment -> uncovered;
  (c) recombination identity exact on synthetic arrays;
  (d) reorder detection — a planted sign flip with a clean CI fires
      REORDER; sub-noise jitter does not;
  (e) guards — pairing mismatch, missing pred key, published-number
      mismatch, and the coverage downgrade each fire.

Pure CPU, read-only on inputs, deterministic. Usage:
  uv run python fontaine/scripts/critical_frame_repooling.py \
      [--reports-dir reports] [--datasets-root ~/datasets/...] \
      [--out reports/analysis__critical_frame_repooling.json] [--selftest]
"""

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bijou.annotations import load_sidecar
from bijou.data import annotation_stamp

PAIR_KEYS = ["truth", "valid", "index", "repo_id", "core"]
CHUNK = 50  # prediction window length == the scored action chunk
BOOT_N = 10_000
BOOT_SEED = 0
COVERAGE_GATE = 0.80  # min fraction of core frames in covered episodes
CRITICAL_GATE = 500  # min critical core frames
REPRO_TOL = 5e-4  # published numbers are rounded to 4 dp
RECOMBINE_TOL = 1e-6

# Scoreboard rows in published rank order (leaderboard.md #4..#8) +
# descriptive extras. (file, pred_key, published_chunk_mae).
SCOREBOARD = [
    (
        "student_1nfe_draw1",
        "eval__fontaine_flow_snapdistill_h1024_30k_1xh100__step_030000__panel_curated_v0_k4l2_1nfe_euler1_npz.npz",
        "pred:bijou@30000",
        5.6036,
    ),
    (
        "ar100k_draws10_t1",
        "eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2_draws10_t1.npz",
        "pred:bijou@100000_draws10_t1",
        5.6515,
    ),
    (
        "ar100k_greedy",
        "eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2.npz",
        "pred:bijou@100000",
        5.8026,
    ),
    (
        "teacher_heun30_stablekey",
        "eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_curated_v0_k4l2_stablekey_heun30.npz",
        "pred:bijou@80000",
        6.5997,
    ),
    (
        "state_copy",
        "eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2.npz",
        "pred:state-copy",
        11.7847,
    ),
]
DESCRIPTIVE = [
    (
        "state_copy_norm",
        "eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2.npz",
        "pred:state-copy-norm",
        11.7357,
    ),
    (
        "teacher_heun30_oldkey",
        "eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_k4l2_heun30.npz",
        "pred:bijou@80000",
        6.6232,
    ),
    (
        "A_s0",
        "eval__fontaine_arb_rcond_40k_1xh100__step_040000__panel_curated_v0_k4l2.npz",
        "pred:bijou@40000",
        7.7966,
    ),
    (
        "A_s1",
        "eval__fontaine_arb_rcond_40k_1xh100_s1__step_040000__panel_curated_v0_k4l2.npz",
        "pred:bijou@40000",
        7.8052,
    ),
    (
        "A_s2",
        "eval__fontaine_arb_rcond_40k_1xh100_s2__step_040000__panel_curated_v0_k4l2.npz",
        "pred:bijou@40000",
        7.7355,
    ),
    (
        "statedrop80",
        "eval__fontaine_arb_rcond_statedrop80_40k_1xh100__step_040000__panel_curated_v0_k4l2.npz",
        "pred:bijou@40000",
        10.5024,
    ),
]
SEED_TRIO = ["A_s0", "A_s1", "A_s2"]


# ---------------------------------------------------------------- rule


class EpisodeLabels:
    """One episode's critical-frame evidence in judge (1-based) coords."""

    def __init__(
        self,
        boundaries: list[int],
        hold_brackets: list[tuple[int, int]],
        event_frames: list[int],
    ) -> None:
        self.boundaries = boundaries
        self.hold_brackets = hold_brackets
        self.event_frames = event_frames

    def critical(self, f0: int) -> bool:
        lo, hi = f0 + 1, f0 + CHUNK
        if any(lo <= b <= hi for b in self.boundaries):
            return True
        if any(not (b_hi < lo or b_lo > hi) for b_lo, b_hi in self.hold_brackets):
            return True
        return any(lo <= g <= hi for g in self.event_frames)


def labels_from_judgment(judgment: Any) -> EpisodeLabels:
    """EpisodeJudgment -> the rule's evidence. Last until_frame excluded
    (episode end by the check_subgoals contract, not a transition)."""
    boundaries = [s.until_frame for s in judgment.subgoals[:-1]]
    anns = sorted(judgment.frame_annotations, key=lambda a: a.frame)
    hold_brackets = [
        (a.frame, b.frame)
        for a, b in itertools.pairwise(anns)
        if a.holding != b.holding
    ]
    events = [a.frame for a in anns if a.events]
    return EpisodeLabels(boundaries, hold_brackets, events)


def blessed_labels(dataset_dir: Path, repo_id: str) -> dict[int, EpisodeLabels]:
    """episode_index -> labels under the training-side selection rule:
    stamp-matched records, last judged_at wins; parse failures skipped
    (that episode stays uncovered)."""
    stamp = annotation_stamp(dataset_dir, repo_id, None)
    if stamp is None:
        return {}
    records = [
        r
        for r in load_sidecar(dataset_dir)
        if r.prompt_hash == stamp.prompt_hash and r.model == stamp.judge_model
    ]
    records.sort(key=lambda r: (r.episode_index, r.judged_at))
    out: dict[int, EpisodeLabels] = {}
    for record in records:
        try:
            out[record.episode_index] = labels_from_judgment(record.parsed_judgment())
        except (KeyError, TypeError, ValueError):
            out.pop(record.episode_index, None)
            continue
    return out


# ------------------------------------------------------------- pooling


def masks(d: Any) -> np.ndarray:
    truth, valid = d["truth"], d["valid"]
    m3 = valid[:, :, None] & np.isfinite(truth).all(-1, keepdims=True)
    return m3.repeat(truth.shape[2], axis=2)


def pooled_chunk(err: np.ndarray, sel: np.ndarray, w: np.ndarray) -> tuple[float, int]:
    cells = w[sel]
    n = int(cells.sum())
    return (float(err[sel][cells].mean()) if n else float("nan")), n


def frame_mae(err: np.ndarray, w: np.ndarray) -> np.ndarray:
    nvalid = w.sum(axis=(1, 2))
    return (err * w).sum(axis=(1, 2)) / np.maximum(nvalid, 1)


def bootstrap_ci(deltas: np.ndarray, n: int = BOOT_N, seed: int = BOOT_SEED) -> tuple:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(deltas), size=(n, len(deltas)))
    means = deltas[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def pair_verdict(mean: float, ci: tuple[float, float], published_gap: float) -> str:
    """REORDER / holds / within-noise for one paired critical-pool delta
    against the published gap's sign."""
    if mean * published_gap < 0 and (ci[0] > 0 or ci[1] < 0):
        return "REORDER"
    if ci[0] > 0 or ci[1] < 0:
        return "holds (CI excludes 0)"
    return "holds (within noise)"


# ---------------------------------------------------------------- main


def load_rows(reports_dir: Path, rows: list) -> dict:
    arms = {}
    for label, fname, key, published in rows:
        path = reports_dir / fname
        if not path.exists():
            sys.exit(f"{label}: missing dump {path}")
        d = np.load(path, allow_pickle=True)
        if key not in d.files:
            sys.exit(f"{label}: pred key {key!r} not in {path.name}")
        arms[label] = (d, key, published)
    return arms


def check_pairing(arms: dict) -> tuple:
    """All dumps must be the identical frame set; the reference dump
    (first with episode_index/frame_index) supplies the episode join."""
    base_label = next(iter(arms))
    base = arms[base_label][0]
    ref = None
    for label, (d, _, _) in arms.items():
        keys = set(d.files) if hasattr(d, "files") else set(d)
        for k in PAIR_KEYS:
            if not np.array_equal(base[k], d[k]):
                sys.exit(f"pairing broken on {k} for {label} — refusing the read")
        if "episode_index" in keys and "frame_index" in keys:
            if ref is None:
                ref = (d["episode_index"], d["frame_index"], label)
            elif not (
                np.array_equal(ref[0], d["episode_index"])
                and np.array_equal(ref[1], d["frame_index"])
            ):
                sys.exit(f"episode/frame_index mismatch: {label} vs {ref[2]}")
    if ref is None:
        sys.exit("no dump carries episode_index/frame_index — cannot join labels")
    return base, ref[0], ref[1]


def critical_masks(
    base: Any,
    episode_index: np.ndarray,
    frame_index: np.ndarray,
    datasets_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    repo_ids = base["repo_id"]
    covered = np.zeros(len(repo_ids), dtype=bool)
    critical = np.zeros(len(repo_ids), dtype=bool)
    cache: dict[str, dict[int, EpisodeLabels]] = {}
    for i, repo in enumerate(repo_ids):
        repo = str(repo)
        if repo not in cache:
            cache[repo] = blessed_labels(datasets_root / repo, repo)
        labels = cache[repo].get(int(episode_index[i]))
        if labels is None:
            continue
        covered[i] = True
        critical[i] = labels.critical(int(frame_index[i]))
    return covered, critical


def analyze(
    arms: dict,
    base: Any,
    covered: np.ndarray,
    critical: np.ndarray,
    out_path: Path | None,
) -> dict:
    truth, core = base["truth"], base["core"]
    w = masks(base)
    crit = core & covered & critical
    comp = core & covered & ~critical
    uncov = core & ~covered

    n_core = int(core.sum())
    coverage = float((core & covered).sum() / n_core)
    downgraded = coverage < COVERAGE_GATE or int(crit.sum()) < CRITICAL_GATE

    table, frames = {}, {}
    for label, (d, key, published) in arms.items():
        err = np.abs(np.asarray(d[key], dtype=np.float64) - truth)
        overall, n_all = pooled_chunk(err, core, w)
        if abs(overall - published) > REPRO_TOL:
            sys.exit(
                f"{label}: overall pooled {overall:.4f} != published "
                f"{published:.4f} (tol {REPRO_TOL}) — identity broken, stop",
            )
        c_mae, n_c = pooled_chunk(err, crit, w)
        p_mae, n_p = pooled_chunk(err, comp, w)
        u_mae, n_u = pooled_chunk(err, uncov, w)
        recombined = (c_mae * n_c + p_mae * n_p + (u_mae * n_u if n_u else 0.0)) / n_all
        if abs(recombined - overall) > RECOMBINE_TOL:
            sys.exit(f"{label}: recombination {recombined} != {overall} — stop")
        frames[label] = frame_mae(err, w)
        table[label] = {
            "published": published,
            "overall": round(overall, 4),
            "critical": round(c_mae, 4),
            "complement": round(p_mae, 4),
            "n_cells": {"critical": n_c, "complement": n_p, "uncovered": n_u},
        }

    keep = crit & (w.sum(axis=(1, 2)) > 0)
    ranked = [r[0] for r in SCOREBOARD if r[0] in arms]
    pairs = []
    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            a, b = ranked[i], ranked[j]
            pub_gap = arms[b][2] - arms[a][2]
            deltas = (frames[b] - frames[a])[keep]
            if deltas.size == 0:
                ci, mean = (float("nan"), float("nan")), float("nan")
            else:
                ci = bootstrap_ci(deltas)
                mean = float(deltas.mean())
            pairs.append(
                {
                    "pair": f"{a} vs {b}",
                    "adjacent": j == i + 1,
                    "published_gap": round(pub_gap, 4),
                    "critical_mean_delta": round(mean, 5),
                    "ci95": [round(ci[0], 5), round(ci[1], 5)],
                    "verdict": (
                        "descriptive-only (coverage gate)"
                        if downgraded
                        else pair_verdict(mean, ci, pub_gap)
                    ),
                },
            )

    trio = [t for t in SEED_TRIO if t in table]
    null_scale = (
        round(
            max(
                abs(table[a]["critical"] - table[b]["critical"])
                for i, a in enumerate(trio)
                for b in trio[i + 1 :]
            ),
            4,
        )
        if len(trio) == 3
        else None
    )

    result = {
        "prereg": "posts/2026-08-07-prereg-critical-frame-repooling.md",
        "rule": "chunk window [f0+1, f0+50] hits subgoal boundary | holding bracket | event frame",
        "coverage": {
            "core_frames": n_core,
            "covered_frac": round(coverage, 4),
            "critical_core_frames": int(crit.sum()),
            "complement_core_frames": int(comp.sum()),
            "uncovered_core_frames": int(uncov.sum()),
            "downgraded": downgraded,
        },
        "table": table,
        "pairs": pairs,
        "seed_trio_critical_null_scale": null_scale,
    }
    if out_path is not None:
        out_path.write_text(json.dumps(result, indent=1))
        print(f"wrote {out_path}")
    return result


def print_report(result: dict) -> None:
    cov = result["coverage"]
    print(
        f"coverage: {cov['covered_frac']:.1%} of {cov['core_frames']} core frames; "
        f"critical {cov['critical_core_frames']} / complement "
        f"{cov['complement_core_frames']} / uncovered {cov['uncovered_core_frames']}"
        + ("  ** DOWNGRADED: descriptive-only **" if cov["downgraded"] else ""),
    )
    print(
        f"{'row':28s} {'published':>9s} {'overall':>8s} {'critical':>8s} {'complement':>10s}",
    )
    for label, t in result["table"].items():
        print(
            f"{label:28s} {t['published']:9.4f} {t['overall']:8.4f} "
            f"{t['critical']:8.4f} {t['complement']:10.4f}",
        )
    print(
        f"seed-trio critical null scale (max pairwise |d|): {result['seed_trio_critical_null_scale']}",
    )
    for p in result["pairs"]:
        tag = "adj" if p["adjacent"] else "   "
        print(
            f"{tag} {p['pair']:52s} pub {p['published_gap']:+8.4f}  "
            f"crit {p['critical_mean_delta']:+9.5f} CI {p['ci95']}  -> {p['verdict']}",
        )


# ------------------------------------------------------------ selftest


class _FakeSubgoal:
    def __init__(self, until_frame: int) -> None:
        self.until_frame = until_frame


class _FakeAnn:
    def __init__(self, frame: int, *, holding: bool, events: tuple = ()) -> None:
        self.frame, self.holding, self.events = frame, holding, tuple(events)


class _FakeJudgment:
    def __init__(self, subgoals: list, anns: list) -> None:
        self.subgoals = [_FakeSubgoal(u) for u in subgoals]
        self.frame_annotations = anns


def selftest() -> None:
    # (a) rule fixtures — boundary 100 (554 = end, excluded), holding
    # bracket [62, 120], event at 300.
    labels = labels_from_judgment(
        _FakeJudgment(
            [100, 554],
            [
                _FakeAnn(1, holding=False),
                _FakeAnn(62, holding=False),
                _FakeAnn(120, holding=True),
                _FakeAnn(200, holding=True),
                _FakeAnn(300, holding=True, events=("battery slips",)),
            ],
        ),
    )
    assert labels.boundaries == [100]
    assert labels.hold_brackets == [(62, 120)]
    assert labels.event_frames == [300]
    cases = {
        49: True,  # W=[50,99] overlaps bracket [62,120]
        50: True,  # W=[51,100] contains boundary 100 (inclusive hi edge)
        100: True,  # W=[101,150] still overlaps bracket (hi 120 >= 101)
        120: True,  # W=[121,170]... bracket hi=120 < 121 -> no; boundary no; -> see below
        150: False,  # W=[151,200] hits nothing
        250: True,  # W=[251,300] contains event 300 (inclusive hi edge)
        300: False,  # W=[301,350] past everything
        99: True,  # W=[100,149] boundary 100 at inclusive lo edge
    }
    cases[120] = False  # bracket ends exactly at 120; window starts 121
    for f0, want in cases.items():
        got = labels.critical(f0)
        assert got == want, f"rule fixture f0={f0}: got {got}, want {want}"
    # unannotated episode -> nothing critical
    empty = labels_from_judgment(
        _FakeJudgment(
            [554],
            [_FakeAnn(1, holding=False), _FakeAnn(554, holding=False)],
        ),
    )
    assert not empty.critical(0) and not empty.boundaries and not empty.hold_brackets
    print("selftest (a) rule fixtures: PASS")

    # (b) blessed selection — stamp filter + last judged_at wins,
    # exercised through the real loaders on a synthetic dataset dir.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        meta = Path(tmp) / "meta"
        meta.mkdir()
        (meta / "judge_annotations.json").write_text(
            json.dumps({"prompt_hash": "aaa", "models": ["judge-m"]}),
        )
        base_j = {
            "overall_score": 5,
            "verdict": "keep",
            "task_completion_visible": "yes",
            "scores": {
                "visual_quality": 5,
                "smoothness": 5,
                "efficiency": 5,
                "camera_framing": 5,
            },
            "instruction_quality": "good",
            "camera_kinds": {"image": "front"},
            "observed_task": "x",
            "suggested_instructions": ["a", "b", "c"],
            "subgoals": [
                {"until_frame": 100, "subgoal": "reach"},
                {"until_frame": 200, "subgoal": "done"},
            ],
            "frame_annotations": [
                {
                    "frame": f,
                    "progress": 0.0,
                    "holding": h,
                    "visible": {"image": {"task_object": True, "gripper": True}},
                    "events": [],
                }
                for f, h in [(1, False), (150, True)]
            ],
        }
        records = []
        for judged_at, hold, phash in [
            ("2026-08-01 00:00:00", True, "aaa"),  # older: transition
            ("2026-08-02 00:00:00", False, "aaa"),  # newer: no transition -> wins
            ("2026-08-03 00:00:00", True, "bbb"),  # wrong stamp: ignored
        ]:
            j = json.loads(json.dumps(base_j))
            j["frame_annotations"][1]["holding"] = hold
            records.append(
                {
                    "episode_index": 0,
                    "model": "judge-m",
                    "prompt_hash": phash,
                    "judged_at": judged_at,
                    "num_timesteps": 2,
                    "max_image_dim": 512,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "judgment": j,
                },
            )
        (meta / "judgments.json").write_text(json.dumps({"judgments": records}))
        out = blessed_labels(Path(tmp), "fixture/repo")
        assert set(out) == {0}, f"expected episode 0 covered, got {set(out)}"
        assert out[0].hold_brackets == [], "last judged_at must win (no transition)"
        assert out[0].boundaries == [100]
        # unparseable newest record -> episode uncovered
        records.append(
            {
                "episode_index": 0,
                "model": "judge-m",
                "prompt_hash": "aaa",
                "judged_at": "2026-08-04 00:00:00",
                "num_timesteps": 2,
                "max_image_dim": 512,
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "judgment": {"broken": True},
            },
        )
        (meta / "judgments.json").write_text(json.dumps({"judgments": records}))
        assert blessed_labels(Path(tmp), "fixture/repo") == {}
    print("selftest (b) blessed selection: PASS")

    # (c)+(d)+(e) on a synthetic 6-frame panel: 2 dims, 2 steps.
    rng = np.random.default_rng(7)
    n, steps, dims = 2000, 2, 2
    truth = rng.normal(size=(n, steps, dims)).astype(np.float32)
    valid = np.ones((n, steps), dtype=bool)
    valid[0, 1] = False  # exercise partial-validity weighting
    core = np.ones(n, dtype=bool)
    core[:20] = False
    covered = np.ones(n, dtype=bool)
    covered[-40:] = False  # uncovered tail
    critical = np.zeros(n, dtype=bool)
    critical[: n // 2] = True
    # arm A: uniform error 1.0; arm B: 0.8 on critical, 1.3 elsewhere
    # -> planted REORDER on the critical pool (published says B > A).
    pred_a = truth + 1.0
    off_b = np.where(critical, 0.8, 1.3)[:, None, None]
    pred_b = truth + off_b
    base = {
        "truth": truth,
        "valid": valid,
        "core": core,
        "index": np.arange(n),
        "repo_id": np.array(["r"] * n),
    }
    pub_a = _pooled_of(base, pred_a, core)
    pub_b = _pooled_of(base, pred_b, core)
    arms = {
        "student_1nfe_draw1": ({**base, "p": pred_a}, "p", round(pub_a, 4)),
        "ar100k_draws10_t1": ({**base, "p": pred_b}, "p", round(pub_b, 4)),
    }
    res = analyze(arms, base, covered, critical, None)
    assert not res["coverage"]["downgraded"]
    (pair,) = res["pairs"]
    assert pair["published_gap"] > 0 and pair["critical_mean_delta"] < 0
    assert pair["verdict"] == "REORDER", pair
    # sub-noise jitter -> no reorder
    pred_c = (
        truth
        + 1.0
        + rng.normal(scale=0.005, size=pred_a.shape)
        * np.where(critical, -1.0, 1.0)[:, None, None]
    )
    pub_c = _pooled_of(base, pred_c, core)
    arms2 = {
        "student_1nfe_draw1": ({**base, "p": pred_a}, "p", round(pub_a, 4)),
        "ar100k_draws10_t1": ({**base, "p": pred_c}, "p", round(pub_c, 4)),
    }
    res2 = analyze(arms2, base, covered, critical, None)
    assert res2["pairs"][0]["verdict"] != "REORDER"
    print("selftest (c) recombination + (d) reorder detection: PASS")

    # (e) guards
    import contextlib

    def must_exit(fn: Any, tag: str) -> None:
        with contextlib.suppress(SystemExit):
            fn()
            raise AssertionError(f"guard {tag} did not fire")

    bad = dict(arms)
    bad["ar100k_draws10_t1"] = (
        bad["ar100k_draws10_t1"][0],
        "p",
        round(pub_b, 4) + 0.01,
    )
    must_exit(lambda: analyze(bad, base, covered, critical, None), "published mismatch")
    b2 = {**base, "truth": truth + 1e-3}
    must_exit(
        lambda: check_pairing({"x": (base, "p", 0.0), "y": (b2, "p", 0.0)}),
        "pairing mismatch",
    )
    sparse = np.zeros(n, dtype=bool)  # coverage below the gate
    res3 = analyze(arms, base, sparse, critical, None)
    assert res3["coverage"]["downgraded"]
    assert all("descriptive-only" in p["verdict"] for p in res3["pairs"])
    print("selftest (e) guards: PASS")
    print("selftest: ALL PASS")


def _pooled_of(base: Any, pred: np.ndarray, core: np.ndarray) -> float:
    err = np.abs(np.asarray(pred, dtype=np.float64) - base["truth"])
    val, _ = pooled_chunk(err, core, masks(base))
    return val


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reports-dir", default="reports")
    p.add_argument(
        "--datasets-root",
        default="/home/ubuntu/datasets/mcobzarenco/community_curated_v0",
    )
    p.add_argument("--out", default="reports/analysis__critical_frame_repooling.json")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        selftest()
        return
    reports_dir = Path(args.reports_dir)
    arms = load_rows(reports_dir, SCOREBOARD + DESCRIPTIVE)
    base, episode_index, frame_index = check_pairing(arms)
    covered, critical = critical_masks(
        base,
        episode_index,
        frame_index,
        Path(args.datasets_root),
    )
    result = analyze(arms, base, covered, critical, Path(args.out))
    print_report(result)


if __name__ == "__main__":
    main()
