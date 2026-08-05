"""±180-degree wraparound census over panel eval + training corpus (CPU).

Idea #14, spawned by the sign-convention stage-1 surprise (the kevin510
standout was a wraparound artifact, not a mirror): SO101 wrist_roll
calibration in lerobot releases ~Jun 2025 - Mar 2026 could zero the
encoder mid-range, so trajectories crossing +-180 deg wrap by ~360 deg
(lerobot#1255, PR#777 removed the software wrap guards, #3193/#1296 the
'Magnitude exceeds 2047' family; properly fixed in release 0.6.0). Wrap
frames poison BOTH raw-degree training targets and raw-degree MAE.

Two censuses, one verdict:

Part A (panel npz, official pooled chunk_mae convention -
abs_error_sum / (valid_steps * dims), bitwise-matched to the eval
instrument's 5.8026 on the reference dump): count frames whose truth
chunk spans > 300 deg per dim, bound their MAE contribution, and
re-score the panel under shortest-arc error (min(|e|, 360-|e|)).

Part B (training corpus): per-repo per-dim count of consecutive-frame
|delta| > 300 deg within an episode in `action` and `observation.state`
- the load-time discontinuity an unwrap-at-load transform would repair -
over all 878 selected repos, joined with codebase_version/robot_type
from meta/info.json. NOTE: the mirror re-serializes every repo to
codebase_version v3.0, so the recording-era lerobot version is not
recoverable from local metadata; the version-correlation the causal
story predicts cannot be tested here.

Anchors (asserted in main() when run with all defaults; first run
2026-08-05 ~16:30Z, scratch parity confirmed): panel - 16/17,204 core
frames wrap (0.093%), pooled chunk_mae 5.8026 all / 5.7306 without wrap
frames (excess 0.0720, 1.2% of the panel number from 0.09% of frames;
wrap frames average 78.27), shortest-arc re-score 5.7498. Corpus - 23
of 878 repos have >=1 wrap jump, 81 of 42,872 episodes (0.19%);
kevin510/lerobot-cat-toy-placement is systematically corrupted (40/40
episodes, 193 action wrist_roll jumps + 78 shoulder_lift) and every
other repo has 1-4 affected episodes; wrist_roll dominates total jumps
(204 action / 184 state).

Verdict vs the pre-registered gate (ideas.md #14: "<0.1% of panel and
negligible excess MAE -> curiosity"): training-side wraps are RARE
(0.19% of episodes, half of it one repo) - an unwrap-at-load training
arm cannot pay for an H100 run and is dropped per the 2026-08-05 16:13Z
steering. The eval-side excess (0.0720 on the panel, larger than the
+-0.05 re-score gate) is a metric artifact concentrated in 16 frames;
shortest-arc scoring is the principled fix but moves every anchor, so
it needs owner sign-off before any instrument change.

Run from the repo root: uv run python -m probes.probe_wrap_census
Non-default inputs skip the anchor asserts (screening use).
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REFERENCE_NPZ = (
    Path.home()
    / "previous-reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2.npz"
)
REFERENCE_POLICY = "pred:bijou@100000"
DEFAULT_CORPUS = Path.home() / "datasets/mcobzarenco/community_curated_v0"
DEFAULT_JUMP_DEG = 300.0
# Reference-run anchors (see docstring).
ANCHOR_PANEL_WRAP_FRAMES = 16
ANCHOR_POOLED_ALL = "5.8026"
ANCHOR_POOLED_NONWRAP = "5.7306"
ANCHOR_POOLED_SHORTEST_ARC = "5.7498"
ANCHOR_CORPUS_REPOS_AFFECTED = 23
ANCHOR_CORPUS_EPISODES_AFFECTED = 81
ANCHOR_TOP_REPO = "kevin510/lerobot-cat-toy-placement"
ANCHOR_TOP_REPO_EPISODES = 40
ANCHOR_WRIST_ROLL_ACTION_JUMPS = 204


def panel_census(npz: Path, policy: str, jump_deg: float) -> dict:
    """Part A: wrap frames, their pooled-MAE contribution, shortest-arc."""
    motor_names = json.loads(npz.with_suffix(".json").read_text())["motor_names"]
    dump = np.load(npz, allow_pickle=True)
    core = dump["core"]
    truth = dump["truth"][core]  # (frames, chunk, dims)
    pred = dump[policy][core]
    valid = dump["valid"][core]  # (frames, chunk)
    repo_ids = dump["repo_id"][core]
    dims = truth.shape[-1]

    big = np.where(valid[..., None], truth, -np.inf).max(1)
    small = np.where(valid[..., None], truth, np.inf).min(1)
    wrap = (big - small) > jump_deg  # (frames, dims)
    wrap_any = wrap.any(1)

    err = np.abs(pred - truth) * valid[..., None]
    err_sa = np.minimum(err, np.where(valid[..., None], 360.0 - err, 0.0))

    def pooled(e: np.ndarray, mask: np.ndarray) -> float:
        return float(e[mask].sum() / max(valid[mask].sum() * dims, 1))

    every = np.ones_like(wrap_any)
    return {
        "core_frames": int(truth.shape[0]),
        "wrap_frames": int(wrap_any.sum()),
        "wrap_pct": float(100 * wrap_any.mean()),
        "per_dim": {
            name: {
                "wrap_frames": int(wrap[:, d].sum()),
                "repos": sorted({str(r) for r in np.unique(repo_ids[wrap[:, d]])}),
            }
            for d, name in enumerate(motor_names)
        },
        "pooled_chunk_mae": pooled(err, every),
        "pooled_nonwrap": pooled(err, ~wrap_any),
        "pooled_wrap_frames_only": pooled(err, wrap_any) if wrap_any.any() else None,
        "pooled_shortest_arc": pooled(err_sa, every),
        "excess_from_wrap_frames": pooled(err, every) - pooled(err, ~wrap_any),
    }


def census_one_repo(job: tuple[str, float]) -> dict:
    """Part B worker: wrap-jump counts for one repo directory."""
    import pyarrow.parquet as pq

    repo_dir_str, jump_deg = job
    repo_dir = Path(repo_dir_str)
    info = json.loads((repo_dir / "meta/info.json").read_text())
    jumps_a: np.ndarray | None = None
    jumps_s: np.ndarray | None = None
    ep_affected_a: set[int] = set()
    ep_affected_s: set[int] = set()
    n_frames = 0
    episodes: set[int] = set()
    for f in sorted(repo_dir.glob("data/*/*.parquet")):
        t = pq.read_table(f, columns=["action", "observation.state", "episode_index"])
        act = np.asarray([np.asarray(x) for x in t["action"].to_pylist()])
        state = np.asarray([np.asarray(x) for x in t["observation.state"].to_pylist()])
        ep = np.asarray(t["episode_index"].to_pylist())
        n_frames += len(ep)
        episodes.update(np.unique(ep).tolist())
        if jumps_a is None:
            jumps_a = np.zeros(act.shape[1], dtype=int)
            jumps_s = np.zeros(state.shape[1], dtype=int)
        same_ep = ep[1:] == ep[:-1]
        for arr, jumps, affected in (
            (act, jumps_a, ep_affected_a),
            (state, jumps_s, ep_affected_s),
        ):
            d = np.abs(np.diff(arr, axis=0)) > jump_deg
            d &= same_ep[:, None]
            jumps += d.sum(0)
            affected.update(ep[np.where(d.any(1))[0]].tolist())
    return {
        "repo": f"{repo_dir.parent.name}/{repo_dir.name}",
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "frames": n_frames,
        "episodes": len(episodes),
        "action_jumps": jumps_a.tolist() if jumps_a is not None else [],
        "state_jumps": jumps_s.tolist() if jumps_s is not None else [],
        "action_eps_affected": len(ep_affected_a),
        "state_eps_affected": len(ep_affected_s),
    }


def corpus_census(
    corpus: Path,
    repos: list[str],
    jump_deg: float,
    workers: int,
) -> list[dict]:
    """Part B: fan the per-repo census over a process pool."""
    jobs = []
    for r in repos:
        d = corpus / r
        if (d / "meta/info.json").exists():
            jobs.append((str(d), jump_deg))
        else:
            print(f"MISSING on disk: {r}")
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(census_one_repo, jobs, chunksize=4)):
            rows.append(row)
            if (i + 1) % 200 == 0:
                print(f"  corpus census: {i + 1}/{len(jobs)} repos")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=REFERENCE_NPZ)
    parser.add_argument("--policy", default=REFERENCE_POLICY)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--jump-deg", type=float, default=DEFAULT_JUMP_DEG)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--skip-corpus", action="store_true")
    args = parser.parse_args()

    a = panel_census(args.npz, args.policy, args.jump_deg)
    print(
        f"panel: {a['wrap_frames']}/{a['core_frames']} wrap frames "
        f"({a['wrap_pct']:.3f}%); pooled chunk_mae {a['pooled_chunk_mae']:.4f} "
        f"-> {a['pooled_nonwrap']:.4f} without them "
        f"(excess {a['excess_from_wrap_frames']:.4f}); "
        f"shortest-arc {a['pooled_shortest_arc']:.4f}",
    )
    for name, row in a["per_dim"].items():
        if row["wrap_frames"]:
            print(f"  {name:22s} wrap_frames={row['wrap_frames']:4d} {row['repos']}")

    rows: list[dict] = []
    if not args.skip_corpus:
        repos = sorted(
            {
                str(r)
                for r in np.unique(np.load(args.npz, allow_pickle=True)["repo_id"])
            },
        )
        rows = corpus_census(args.corpus, repos, args.jump_deg, args.workers)
        affected = [
            r for r in rows if sum(r["action_jumps"]) + sum(r["state_jumps"]) > 0
        ]
        affected.sort(
            key=lambda r: -(sum(r["action_jumps"]) + sum(r["state_jumps"])),
        )
        eps_affected = sum(
            max(r["action_eps_affected"], r["state_eps_affected"]) for r in affected
        )
        total_eps = sum(r["episodes"] for r in rows)
        print(
            f"corpus: {len(affected)}/{len(rows)} repos with >=1 wrap jump; "
            f"{eps_affected}/{total_eps} episodes affected "
            f"({100 * eps_affected / max(total_eps, 1):.2f}%)",
        )
        for r in affected[:10]:
            print(
                f"  {r['repo'][:50]:50s} eps={r['episodes']:4d} "
                f"aff_a={r['action_eps_affected']:3d} aff_s={r['state_eps_affected']:3d} "
                f"jumps_a={r['action_jumps']} jumps_s={r['state_jumps']}",
            )

    is_reference_run = (
        args.npz == REFERENCE_NPZ
        and args.policy == REFERENCE_POLICY
        and args.corpus == DEFAULT_CORPUS
        and args.jump_deg == DEFAULT_JUMP_DEG
        and not args.skip_corpus
    )
    if is_reference_run:
        assert a["wrap_frames"] == ANCHOR_PANEL_WRAP_FRAMES, "panel wrap count DRIFTED"
        assert f"{a['pooled_chunk_mae']:.4f}" == ANCHOR_POOLED_ALL, "pooled MAE DRIFTED"
        assert f"{a['pooled_nonwrap']:.4f}" == ANCHOR_POOLED_NONWRAP, (
            "nonwrap MAE DRIFTED"
        )
        assert f"{a['pooled_shortest_arc']:.4f}" == ANCHOR_POOLED_SHORTEST_ARC, (
            "shortest-arc MAE DRIFTED"
        )
        affected = [
            r for r in rows if sum(r["action_jumps"]) + sum(r["state_jumps"]) > 0
        ]
        assert len(affected) == ANCHOR_CORPUS_REPOS_AFFECTED, "corpus repos DRIFTED"
        eps_affected = sum(
            max(r["action_eps_affected"], r["state_eps_affected"]) for r in affected
        )
        assert eps_affected == ANCHOR_CORPUS_EPISODES_AFFECTED, (
            "corpus episodes DRIFTED"
        )
        top = {r["repo"]: r for r in affected}[ANCHOR_TOP_REPO]
        assert (
            max(top["action_eps_affected"], top["state_eps_affected"])
            == ANCHOR_TOP_REPO_EPISODES
        ), "kevin510 episode count DRIFTED"
        motor_names = json.loads(
            args.npz.with_suffix(".json").read_text(),
        )["motor_names"]
        roll = motor_names.index("main_wrist_roll")
        total_roll = sum(r["action_jumps"][roll] for r in rows)
        assert total_roll == ANCHOR_WRIST_ROLL_ACTION_JUMPS, "wrist_roll jumps DRIFTED"
        print("WRAP CENSUS ANCHORS PASSED")
    else:
        print("(non-default inputs: anchor asserts skipped)")


if __name__ == "__main__":
    main()
