"""Kinematic-continuity screen over community_curated_v0 — ideas #9, record-only.

Hook: VISTA (2606.04708, papers/vista-umi-validation.md) validates
human-collected trajectories with a per-tick continuity score (full
marks below a small displacement, penalties up to an exponential regime
past 9x that bar) and shows low-score data trains policies that fail
outright. Our corpus was curated by a VLM judge (semantic quality); a
kinematic-corruption screen — sensor dropouts, teleport jumps in the
recorded actions — is an orthogonal, zero-GPU dimension nobody has run
on it. Our episodes were executed by real SO-100/101 arms, so gross
infeasibility is impossible by construction; what this can catch is
recording corruption. Exploratory read on our own data (SDN-read
precedent): NO pre-registration, and any curation change motivated by
it needs its own pre-reg. A null closes the hook at zero cost.

SCORING (frozen here, before any data is read; VISTA's three regimes
recalibrated to so100 joint space from the rig repos' own displacement
distribution):

  * Displacement d[t, j] = |a[t+1, j] - a[t, j]| over the stored action
    rows of an episode (the training signal is actions, so actions —
    not observation.state).
  * Calibration: lo[j] = per-dim p99.9 of per-tick displacements pooled
    over the two rig calibration anchors (mcobzarenco/
    so101_pick_place_clean + so101_pick_place_v2 — real SO-101 teleop,
    30 fps, trusted clean recordings; abort if any lo[j] == 0). The
    exponential knee sits at 9 * lo, keeping VISTA's 5 mm -> 45 mm
    (1 deg -> 9 deg) ratio. Per-dim calibration lets the gripper (a
    0-100 command that legitimately jumps) carry its own bar.
  * fps scaling: a repo at fps f is scored against lo_eff = lo * 30/f
    (same physical speed => per-tick displacement scales as 1/f).
  * Tick ratio r[t] = max_j d[t, j] / lo_eff[j]; tick score
        r <= 1        -> 1.0                      (full marks)
        1 < r <= 9    -> 1 - 0.5 * (r - 1) / 8    (linear, 1.0 -> 0.5)
        r > 9         -> 0.5 * exp(-(r - 9) / 9)  (exponential)
    (continuous at both knees). Episode score = min over ticks;
    episodes with < 2 frames score 1.0 and are counted separately.
  * Regimes by episode score: FULL (== 1.0), LINEAR ([0.5, 1)),
    EXP tail (< 0.5, i.e. at least one tick beyond 9x the rig bar =
    teleport-class).

READS (frozen):

  R1  Score distribution: episode counts by regime, score quantiles,
      and per-episode max-r quantiles — corpus vs the two rig anchors
      under identical thresholds. (Rig episodes land in the linear
      regime at the expected ~0.1%-of-ticks rate by construction; the
      anchor claim is about the EXP tail, which should be ~empty.)
  R2  EXP tail list: (repo, episode, score, max r, worst dim, tick,
      n_frames) for every episode with score < 0.5, plus an extreme
      cut r >= 50. Counts of episodes / frames / repos involved.
  R3  Repo-level: repos ranked by EXP-tail fraction (>= 3 scored
      episodes), cross-checked against the banked LORO
      influential-repo list (analysis__box_batch_40k_k4l2.json
      decision/most_influential_repos; the arch-batch analysis banked
      no repo-level list — stated, not silently skipped).
  R4  Panel exposure: EXP-tail episodes that are rows of
      plans/holdout_curated_v0_k4l2.json (core or labeled), and tail
      episodes on the train side of panel-selected repos (selection
      mirror: fps 30, camera counts {1,2}, dims == anchor).

DECISION RELEVANCE (record-only, not a decision rule): an ~empty EXP
tail closes the #9 hook at zero cost; a material tail is a curation
lever whose adoption needs its own pre-reg; any R4 panel-row hit gets
flagged as an eval-integrity caveat wherever those anchors are quoted.

VALIDATION (--oracle, synthetic fixtures through the same scoring code
paths; all must pass before real reads are printed):
  * clean smooth episode (every tick at 0.5x lo) -> score exactly 1.0;
  * planted teleport (one tick, one dim, 20x lo) -> EXP regime with
    the exact closed-form score, worst tick/dim recovered;
  * freeze-and-jump-back (sensor dropout shape) -> both jump ticks EXP;
  * knee exactness: r == 1 -> 1.0, r == 9 -> 0.5;
  * fps invariance: displacements x2 at fps 15 == the fps-30 original;
  * calibration: planted pool recovers np.quantile(..., 0.999) exactly;
    an all-constant dim (lo == 0) aborts;
  * single-frame episode -> 1.0, counted as too-short.

Usage:
  uv run python fontaine/scripts/corpus_continuity_screen.py --oracle
  uv run python fontaine/scripts/corpus_continuity_screen.py \
      --corpus ~/datasets/mcobzarenco/community_curated_v0 \
      --rig ~/datasets/mcobzarenco/so101_pick_place_clean \
      --rig ~/datasets/mcobzarenco/so101_pick_place_v2 \
      --plan plans/holdout_curated_v0_k4l2.json \
      --loro-json reports/analysis__box_batch_40k_k4l2.json \
      --output-json reports/analysis__corpus_continuity_screen.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bijou.data import DatasetInfo, discover_datasets, holdout_episodes, repo_id_of

CALIBRATION_QUANTILE = 0.999
KNEE_RATIO = 9.0  # VISTA's 45 mm / 5 mm (9 deg / 1 deg) kept verbatim
CALIBRATION_FPS = 30.0
EXTREME_RATIO = 50.0
PANEL_FPS = (30.0,)
PANEL_CAMERA_COUNTS = (1, 2)
PANEL_HOLDOUT_FRACTION = 0.1
PANEL_SPLIT_SEED = 0


def read_repo_actions(
    repo_dir: Path,
) -> tuple[list[tuple[int, np.ndarray]], list[str]]:
    """[(episode_id, action array)] in stored row order, plus structural
    warnings. Same contract as dup_content_census.read_repo_episodes (the
    audited loader) but action-only and handling BOTH parquet layouts:
    the census corpus stores actions as variable-size lists, the v3.0 rig
    repos as FixedSizeListArray (no offsets attribute). Fails loud on
    ungrouped episode rows or ragged widths."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted(repo_dir.glob("data/chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"{repo_dir}: no data parquet")
    table = pa.concat_tables(
        [pq.read_table(f, columns=["action", "episode_index"]) for f in files],
    )
    episode_column = np.asarray(table["episode_index"])
    ids, starts = np.unique(episode_column, return_index=True)
    if not bool(np.all(np.diff(starts) > 0)):
        raise AssertionError(
            f"{repo_id_of(repo_dir)}: episode rows not grouped ascending",
        )
    col = table["action"].combine_chunks()
    flat = col.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
    if isinstance(col, pa.FixedSizeListArray):
        width = int(col.type.list_size)
    else:
        widths = np.diff(col.offsets.to_numpy())
        if len(np.unique(widths)) != 1:
            raise AssertionError(f"{repo_id_of(repo_dir)}: ragged action column")
        width = int(widths[0])
    actions = flat.reshape(len(col), width)
    bounds = [*starts.tolist(), len(episode_column)]
    episodes = [
        (int(ids[i]), actions[bounds[i] : bounds[i + 1]]) for i in range(len(ids))
    ]
    warnings = []
    if not np.array_equal(ids, np.arange(len(ids))):
        warnings.append(f"{repo_id_of(repo_dir)}: episode ids not contiguous 0..n-1")
    return episodes, warnings


def calibrate(rig_action_arrays: list[np.ndarray]) -> np.ndarray:
    """Per-dim p99.9 of pooled per-tick displacements. Aborts on a zero bar."""
    pools = [np.abs(np.diff(a.astype(np.float64), axis=0)) for a in rig_action_arrays]
    pooled = np.concatenate([p for p in pools if len(p)], axis=0)
    lo = np.quantile(pooled, CALIBRATION_QUANTILE, axis=0)
    if not bool(np.all(lo > 0)):
        raise AssertionError(f"calibration produced a zero bar: lo={lo.tolist()}")
    return lo


def tick_scores(ratios: np.ndarray) -> np.ndarray:
    linear = 1.0 - 0.5 * (ratios - 1.0) / (KNEE_RATIO - 1.0)
    expo = 0.5 * np.exp(-(ratios - KNEE_RATIO) / KNEE_RATIO)
    return np.where(ratios <= 1.0, 1.0, np.where(ratios <= KNEE_RATIO, linear, expo))


def score_episode(actions: np.ndarray, lo_eff: np.ndarray) -> dict:
    """Episode continuity summary; T < 2 scores 1.0 with too_short set."""
    if len(actions) < 2:
        return {
            "score": 1.0,
            "max_ratio": 0.0,
            "worst_tick": -1,
            "worst_dim": -1,
            "too_short": True,
        }
    d = np.abs(np.diff(actions.astype(np.float64), axis=0)) / lo_eff
    per_tick_max = d.max(axis=1)
    scores = tick_scores(per_tick_max)
    worst_tick = int(scores.argmin())
    return {
        "score": float(scores[worst_tick]),
        "max_ratio": float(per_tick_max.max()),
        "worst_tick": worst_tick,
        "worst_dim": int(d[int(per_tick_max.argmax())].argmax()),
        "too_short": False,
    }


def regime(score: float) -> str:
    if score == 1.0:
        return "full"
    return "linear" if score >= 0.5 else "exp"


def quantiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    qs = {"min": float(arr.min()), "max": float(arr.max())}
    for q in (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999):
        qs[f"p{q * 100:g}"] = float(np.quantile(arr, q))
    return qs


def run_oracle() -> None:
    rng = np.random.default_rng(0)
    lo = np.array([0.8, 1.2, 0.5, 0.6, 1.0, 4.0])

    # Clean episode: every per-tick displacement at exactly 0.5x its bar.
    steps = np.tile(0.5 * lo, (99, 1)) * rng.choice([-1.0, 1.0], size=(99, 6))
    clean = np.concatenate([np.zeros((1, 6)), np.cumsum(steps, axis=0)])
    res = score_episode(clean, lo)
    assert res["score"] == 1.0 and not res["too_short"], res

    # Planted teleport: tick 40, dim 2, exactly 20x the bar.
    jump_steps = steps.copy()
    jump_steps[40, 2] = 20.0 * lo[2]
    jump = np.concatenate([np.zeros((1, 6)), np.cumsum(jump_steps, axis=0)])
    res = score_episode(jump, lo)
    expected = 0.5 * math.exp(-(20.0 - KNEE_RATIO) / KNEE_RATIO)
    assert abs(res["score"] - expected) < 1e-12, (res, expected)
    assert res["worst_tick"] == 40 and res["worst_dim"] == 2, res
    assert res["max_ratio"] == 20.0 and regime(res["score"]) == "exp", res

    # Freeze-and-jump-back dropout: both edges land in the exp regime.
    frozen = np.tile(clean[10], (30, 1))
    dropout = np.concatenate([clean[:10], frozen, clean[10:]])
    dropout[15:20, 0] += 12.0 * lo[0]
    d = np.abs(np.diff(dropout, axis=0)) / lo
    exp_ticks = np.where(d.max(axis=1) > KNEE_RATIO)[0]
    assert len(exp_ticks) == 2, exp_ticks
    res = score_episode(dropout, lo)
    assert abs(res["score"] - 0.5 * math.exp(-(12.0 - 9.0) / 9.0)) < 1e-12, res

    # Knee exactness: r == 1 -> 1.0, r == 9 -> 0.5.
    two = np.stack([np.zeros(6), lo.copy()])
    assert score_episode(two, lo)["score"] == 1.0
    two[1] = KNEE_RATIO * lo
    assert score_episode(two, lo)["score"] == 0.5

    # fps invariance: x2 displacements at fps 15 == the fps-30 original.
    res_30 = score_episode(jump, lo * (CALIBRATION_FPS / 30.0))
    res_15 = score_episode(
        clean[0] + 2.0 * (jump - clean[0]),
        lo * (CALIBRATION_FPS / 15.0),
    )
    assert res_30["score"] == res_15["score"], (res_30, res_15)

    # Calibration: recovers np.quantile exactly; a constant dim aborts.
    pool_steps = rng.normal(0.0, 1.0, size=(5000, 6)) * lo
    walk = np.concatenate([np.zeros((1, 6)), np.cumsum(pool_steps, axis=0)])
    got = calibrate([walk])
    want = np.quantile(
        np.abs(np.diff(walk, axis=0)),
        CALIBRATION_QUANTILE,
        axis=0,
    )
    assert np.array_equal(got, want), (got, want)
    flat = walk.copy()
    flat[:, 3] = 7.0
    try:
        calibrate([flat])
        raise SystemExit("oracle FAILED: zero-bar calibration did not abort")
    except AssertionError:
        pass

    # Single-frame episode: 1.0, flagged too short.
    res = score_episode(clean[:1], lo)
    assert res["score"] == 1.0 and res["too_short"], res

    print("[continuity] oracle PASS (7 fixture families)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--rig", type=Path, action="append", default=[])
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--loro-json", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if args.oracle:
        run_oracle()
        return
    if not (args.corpus and len(args.rig) == 2 and args.plan and args.output_json):
        parser.error("--corpus, exactly two --rig, --plan, --output-json required")

    # --- calibration from the rig anchors (structural asserts first) ---
    rig_actions: list[np.ndarray] = []
    rig_repos: list[tuple[str, list[tuple[int, np.ndarray]]]] = []
    anchor_dims: int | None = None
    for rig_dir in args.rig:
        info = DatasetInfo.from_json(rig_dir / "meta" / "info.json")
        assert info.fps == CALIBRATION_FPS, (rig_dir, info.fps)
        assert info.action_state_dims is not None, rig_dir
        anchor_dims = info.action_state_dims[0]
        episodes, warnings = read_repo_actions(rig_dir)
        for w in warnings:
            print(f"[continuity] WARN {w}")
        assert len(episodes) == info.total_episodes, (rig_dir, len(episodes))
        rig_actions.extend(a for _, a in episodes)
        rig_repos.append((repo_id_of(rig_dir), episodes))
    assert anchor_dims == 6, anchor_dims
    lo = calibrate(rig_actions)
    print(
        f"[continuity] calibration: {sum(len(e) for _, e in rig_repos)} rig episodes, "
        f"{sum(len(a) - 1 for a in rig_actions)} ticks; per-dim p99.9 bar "
        f"lo={np.round(lo, 4).tolist()} (exp knee at 9x)",
    )

    # --- selection mirror for R4 train-side exposure ---
    dataset_dirs = discover_datasets((args.corpus,), ())
    infos = [DatasetInfo.from_json(d / "meta" / "info.json") for d in dataset_dirs]
    selected: set[str] = set()
    holdout_map: dict[str, set[int]] = {}
    for d, info in zip(dataset_dirs, infos, strict=True):
        repo = repo_id_of(d)
        if (
            repo not in selected
            and info.action_state_dims == (6, 6)
            and info.fps in PANEL_FPS
            and len(info.cameras) in PANEL_CAMERA_COUNTS
        ):
            selected.add(repo)
            holdout_map[repo] = set(
                holdout_episodes(
                    repo,
                    info.total_episodes,
                    PANEL_HOLDOUT_FRACTION,
                    PANEL_SPLIT_SEED,
                ),
            )

    plan = json.loads(args.plan.read_text())
    plan_rows = {
        (row[0], int(row[1])) for split in ("core", "labeled") for row in plan[split]
    }

    # --- score every scoreable corpus episode, repo by repo ---
    ep_rows: list[dict] = []
    skipped_dims: list[str] = []
    read_failures: list[str] = []
    warnings_all: list[str] = []
    n_too_short = 0
    for d, info in zip(dataset_dirs, infos, strict=True):
        repo = repo_id_of(d)
        if info.action_state_dims is None or info.action_state_dims[0] != 6:
            skipped_dims.append(repo)
            continue
        try:
            episodes, warnings = read_repo_actions(d)
        except (AssertionError, FileNotFoundError, OSError) as exc:
            read_failures.append(f"{repo}: {exc}")
            continue
        warnings_all.extend(warnings)
        lo_eff = lo * (CALIBRATION_FPS / info.fps)
        for episode, actions in episodes:
            res = score_episode(actions, lo_eff)
            if res["too_short"]:
                n_too_short += 1
            if repo in selected:
                exposure = (
                    "panel-row"
                    if (repo, episode) in plan_rows
                    else "holdout"
                    if episode in holdout_map[repo]
                    else "train-side"
                )
            else:
                exposure = "unselected"
            ep_rows.append(
                {
                    "repo": repo,
                    "episode": episode,
                    "n_frames": len(actions),
                    "fps": info.fps,
                    "exposure": exposure,
                    **{k: v for k, v in res.items() if k != "too_short"},
                },
            )
    print(
        f"[continuity] corpus: {len(dataset_dirs)} datasets discovered; scored "
        f"{len(ep_rows)} episodes across {len(dataset_dirs) - len(skipped_dims) - len(read_failures)} "
        f"repos ({len(skipped_dims)} skipped non-6-dim, {len(read_failures)} read "
        f"failures, {n_too_short} single-frame episodes scored 1.0)",
    )
    for f in read_failures:
        print(f"[continuity] READ FAIL {f}")
    for w in warnings_all[:10]:
        print(f"[continuity] WARN {w}")

    # --- rig anchors under the same thresholds ---
    rig_rows = [
        {"repo": repo, "episode": e, **score_episode(a, lo)}
        for repo, eps in rig_repos
        for e, a in eps
    ]

    # R1 — distribution.
    def regime_counts(rows: list[dict]) -> dict[str, int]:
        counts = {"full": 0, "linear": 0, "exp": 0}
        for r in rows:
            counts[regime(r["score"])] += 1
        return counts

    r1 = {
        "corpus": {
            "n_episodes": len(ep_rows),
            "regimes": regime_counts(ep_rows),
            "score_quantiles": quantiles([r["score"] for r in ep_rows]),
            "max_ratio_quantiles": quantiles([r["max_ratio"] for r in ep_rows]),
        },
        "rig_anchors": {
            "n_episodes": len(rig_rows),
            "regimes": regime_counts(rig_rows),
            "max_ratio_max": max(r["max_ratio"] for r in rig_rows),
        },
    }

    # R2 — exponential tail.
    tail = sorted((r for r in ep_rows if r["score"] < 0.5), key=lambda r: r["score"])
    r2 = {
        "n_tail_episodes": len(tail),
        "n_tail_frames": sum(r["n_frames"] for r in tail),
        "n_tail_repos": len({r["repo"] for r in tail}),
        "n_extreme": sum(1 for r in tail if r["max_ratio"] >= EXTREME_RATIO),
        "episodes": tail,
    }

    # R3 — repo ranking + LORO cross-check.
    by_repo: dict[str, list[dict]] = {}
    for r in ep_rows:
        by_repo.setdefault(r["repo"], []).append(r)
    ranked = sorted(
        (
            {
                "repo": repo,
                "n_episodes": len(rows),
                "n_exp": sum(1 for r in rows if r["score"] < 0.5),
                "exp_fraction": sum(1 for r in rows if r["score"] < 0.5) / len(rows),
            }
            for repo, rows in by_repo.items()
            if len(rows) >= 3
        ),
        key=lambda r: r["exp_fraction"],
        reverse=True,
    )
    loro_overlap: dict = {"available": False, "note": "no --loro-json given"}
    if args.loro_json:
        loro = json.loads(args.loro_json.read_text())
        influential = [
            r["repo"]
            for r in loro.get("decision", {}).get("most_influential_repos", [])
        ]
        tail_repos = {r["repo"] for r in tail}
        loro_overlap = {
            "available": True,
            "source": str(args.loro_json),
            "influential_repos": influential,
            "overlap_with_exp_tail": sorted(tail_repos & set(influential)),
            "note": "arch-batch analysis banked no repo-level LORO list; "
            "box-batch top-5 is the only banked list",
        }

    # R4 — panel exposure of the tail.
    r4 = {
        "panel_rows_in_tail": [r for r in tail if r["exposure"] == "panel-row"],
        "holdout_side_in_tail": sum(1 for r in tail if r["exposure"] == "holdout"),
        "train_side_in_tail": sum(1 for r in tail if r["exposure"] == "train-side"),
        "unselected_in_tail": sum(1 for r in tail if r["exposure"] == "unselected"),
    }

    report = {
        "params": {
            "calibration_quantile": CALIBRATION_QUANTILE,
            "knee_ratio": KNEE_RATIO,
            "calibration_fps": CALIBRATION_FPS,
            "extreme_ratio": EXTREME_RATIO,
            "lo_per_dim": lo.tolist(),
            "rig_repos": [repo for repo, _ in rig_repos],
        },
        "coverage": {
            "datasets_discovered": len(dataset_dirs),
            "episodes_scored": len(ep_rows),
            "skipped_non_6dim": sorted(skipped_dims),
            "read_failures": read_failures,
            "single_frame_episodes": n_too_short,
        },
        "R1_distribution": r1,
        "R2_exp_tail": r2,
        "R3_repo_ranking": {"top15": ranked[:15], "loro_cross_check": loro_overlap},
        "R4_panel_exposure": r4,
    }
    args.output_json.write_text(json.dumps(report, indent=1))
    np.save(
        args.output_json.with_suffix(".max_ratios.npy"),
        np.array([r["max_ratio"] for r in ep_rows]),
    )

    # --- printed summary ---
    c = r1["corpus"]
    print(
        f"[continuity] R1 corpus: {c['regimes']['full']} full / "
        f"{c['regimes']['linear']} linear / {c['regimes']['exp']} EXP of "
        f"{c['n_episodes']}; rig anchors {r1['rig_anchors']['regimes']} "
        f"(rig max r {r1['rig_anchors']['max_ratio_max']:.2f})",
    )
    print(
        f"[continuity] R2 tail: {r2['n_tail_episodes']} episodes / "
        f"{r2['n_tail_frames']} frames / {r2['n_tail_repos']} repos; "
        f"{r2['n_extreme']} extreme (r >= {EXTREME_RATIO:g})",
    )
    for r in tail[:20]:
        print(
            f"    {r['repo']}#{r['episode']} score {r['score']:.4f} "
            f"max_r {r['max_ratio']:.1f} dim {r['worst_dim']} tick "
            f"{r['worst_tick']} ({r['exposure']})",
        )
    if len(tail) > 20:
        print(f"    ... {len(tail) - 20} more in the json")
    print("[continuity] R3 top repos by EXP fraction (>=3 eps):")
    for r in ranked[:8]:
        if r["n_exp"] == 0:
            break
        print(
            f"    {r['repo']}: {r['n_exp']}/{r['n_episodes']} "
            f"({r['exp_fraction']:.1%})",
        )
    print(
        f"[continuity] R3 LORO cross-check: {loro_overlap.get('overlap_with_exp_tail', 'n/a')}",
    )
    print(
        f"[continuity] R4 exposure of tail: {len(r4['panel_rows_in_tail'])} PANEL ROWS, "
        f"{r4['holdout_side_in_tail']} holdout-side, {r4['train_side_in_tail']} "
        f"train-side, {r4['unselected_in_tail']} unselected",
    )
    print(f"[continuity] report -> {args.output_json}")


if __name__ == "__main__":
    main()
