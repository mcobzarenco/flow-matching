"""Stage-B demo collector for the grasp-rich SFT bootstrap
(posts/2026-08-14-prereg-grasp-sft-bootstrap.md §2/§6, frozen).

Runs the privileged scripted expert on demo seeds ascending from
``--seed-start`` (≥ ``DEMO_SEED_BASE`` — the eval holdout 0–99 is
refused), rendered under the production visual config (``SO101Sim()``
defaults — the same pixels stage D evaluates on), and writes ONLY the
successful episodes as a LeRobot v3 dataset in the shape the molmoact2
``train_lerobot.py`` recipe class consumes (the owner rig repos'
schema: ``action`` + ``observation.state`` float32[6] rig-order
degrees, ``observation.images.front`` ← the sim TOP camera (the rig
dataset's historical key), ``observation.images.wrist``).

Convention seam (§6 item 4, frozen): rows are written in the
controller-native rig frame, IDENTITY — no shim anywhere in stages
B–D; stage C recomputes its q01/q99 table from this dataset. The
choice rides ``meta/demo_provenance.json`` as provenance, along with
the kept-seed list and the expert HEAD.

Frame pairing is the BC contract: frame t carries the observation the
expert planned FROM and the absolute joint target (degrees) it
commanded at that tick.

Resume: every ``save_episode`` also banks ``collect_state.json``
(next seed, kept count) next to the dataset; a rerun with the same
``--out`` picks up from there via ``LeRobotDataset.resume``.

Usage:
  MUJOCO_GL=egl uv run python -m sim.collect_demos \
      --out ~/datasets/fontaine/grasp_sft_demos_v0 \
      --target-kept 400 [--seed-start 1000] [--max-wall-hours 4]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .scripted_expert import DEMO_SEED_BASE, ScriptedExpert
from .so101_sim import SO101Sim

REPO_ID = "fontaine/grasp_sft_demos_v0"
FPS = 30
TASK = "Pick up the toy boat and place it on the wooden disk."
MOTOR_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
STATE_UNITS = "rig (identity — recomputed dataset table)"

FEATURES = {
    "action": {"dtype": "float32", "shape": [6], "names": MOTOR_NAMES},
    "observation.state": {"dtype": "float32", "shape": [6], "names": MOTOR_NAMES},
    "observation.images.front": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
    },
    "observation.images.wrist": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
    },
}


@dataclass
class DemoFrame:
    """One BC pair: the observation the expert planned from and the
    absolute joint target (degrees, rig order) it commanded."""

    top: np.ndarray
    wrist: np.ndarray
    state: np.ndarray
    action: np.ndarray


@dataclass
class DemoEpisode:
    seed: int
    success: bool
    frames: list[DemoFrame]
    ticks: int
    final_disk_cm: float


#: (seed) -> DemoEpisode. Production = ``expert_episode_source``; the
#: CPU oracles inject synthetic sources (no GL, no physics).
EpisodeSource = Callable[[int], DemoEpisode]


def expert_episode_source(sim: SO101Sim, max_ticks: int = 600) -> EpisodeSource:
    def run(seed: int) -> DemoEpisode:
        obs = sim.reset(seed)
        expert = ScriptedExpert(sim)
        frames: list[DemoFrame] = []
        tick = 0
        for tick in range(max_ticks):  # noqa: B007 — returned count
            action = expert.action(sim)
            frames.append(
                DemoFrame(
                    top=obs.top,
                    wrist=obs.wrist,
                    state=np.asarray(obs.state, dtype=np.float32),
                    action=np.asarray(action, dtype=np.float32),
                ),
            )
            obs = sim.step(action)
            if sim.success():
                break
        return DemoEpisode(
            seed=seed,
            success=bool(sim.success()),
            frames=frames,
            ticks=tick + 1,
            final_disk_cm=sim.benchy_disk_distance() * 100,
        )

    return run


def _state_path(root: Path) -> Path:
    return root / "collect_state.json"


def _provenance_path(root: Path) -> Path:
    return root / "meta" / "demo_provenance.json"


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def rewrite_quantile_stats(root: Path) -> dict[str, list[float]]:
    """Recompute meta/stats.json quantile rows for the vector features
    from the raw frames, in place.

    LeRobot's ``aggregate_feature_stats`` merges per-episode quantiles
    as a count-weighted MEAN of quantiles — wrong whenever episodes are
    heterogeneous, and catastrophic on cross-episode-bimodal channels
    (measured 2026-08-15 on grasp_sft_demos_v0: wrist_roll true action
    q01/q99 ±157° with 17% of episodes on the π-flipped branch, banked
    table said [35.5, 94.4] → the flipped branch was clamped out of
    both state and action space through training AND serving). Mean /
    std / min / max merge through proper pooled formulas and are left
    untouched; every ``qNN`` key is recomputed exactly over all frames.
    """
    import pandas as pd

    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    frames = pd.concat(
        [pd.read_parquet(p) for p in sorted(root.glob("data/*/*.parquet"))],
    )
    fixed: dict[str, list[float]] = {}
    for feature in ("action", "observation.state"):
        values = np.stack(list(frames[feature].to_numpy())).astype(np.float64)
        for key in list(stats[feature]):
            if key.startswith("q") and key[1:].isdigit():
                q = int(key[1:]) / 100.0
                exact = np.quantile(values, q, axis=0)
                stats[feature][key] = [float(x) for x in exact]
                fixed[f"{feature}/{key}"] = stats[feature][key]
    stats_path.write_text(json.dumps(stats, indent=4) + "\n")
    return fixed


def open_dataset(root: Path, repo_id: str = REPO_ID):  # noqa: ANN201 — lerobot type
    """Create the dataset, or resume an existing collection at root."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if _state_path(root).exists():
        return LeRobotDataset.resume(repo_id, root=root)
    if root.exists() and any(root.iterdir()):
        raise SystemExit(
            f"ABORT: {root} exists without collect_state.json — not a "
            "resumable collection; refusing to overwrite",
        )
    return LeRobotDataset.create(repo_id, fps=FPS, features=FEATURES, root=root)


def collect(
    root: Path,
    source: EpisodeSource,
    *,
    target_kept: int,
    seed_start: int = DEMO_SEED_BASE,
    max_seeds: int = 2000,
    max_wall_s: float = 4 * 3600,
    repo_id: str = REPO_ID,
    extra_provenance: dict[str, Any] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run episodes until ``target_kept`` successes are written (or a
    budget trips). Successes only — misses are discarded whole."""
    if seed_start < DEMO_SEED_BASE:
        raise ValueError(
            f"seed_start {seed_start} is inside the frozen eval holdout — "
            f"demo seeds begin at {DEMO_SEED_BASE} (pre-reg §3)",
        )
    dataset = open_dataset(root, repo_id)
    kept_seeds: list[int] = []
    next_seed = seed_start
    attempted = 0
    if _state_path(root).exists():
        state = json.loads(_state_path(root).read_text())
        next_seed = int(state["next_seed"])
        kept_seeds = [int(s) for s in state["kept_seeds"]]
        attempted = int(state["attempted"])
        log(f"[collect] RESUME at seed {next_seed}, {len(kept_seeds)} kept")

    t0 = time.time()
    stop_reason = "target"
    while len(kept_seeds) < target_kept:
        if next_seed >= seed_start + max_seeds:
            stop_reason = "seed budget"
            break
        if time.time() - t0 > max_wall_s:
            stop_reason = "wall budget"
            break
        episode = source(next_seed)
        attempted += 1
        if episode.success:
            for frame in episode.frames:
                dataset.add_frame(
                    {
                        "action": frame.action,
                        "observation.state": frame.state,
                        "observation.images.front": frame.top,
                        "observation.images.wrist": frame.wrist,
                        "task": TASK,
                    },
                )
            dataset.save_episode()
            kept_seeds.append(episode.seed)
        log(
            f"[collect] seed {episode.seed}: "
            f"{'KEPT' if episode.success else 'miss'} "
            f"({episode.ticks} ticks, {episode.final_disk_cm:.1f} cm) — "
            f"{len(kept_seeds)}/{target_kept} kept, {attempted} attempted",
        )
        next_seed += 1
        _state_path(root).write_text(
            json.dumps(
                {
                    "next_seed": next_seed,
                    "kept_seeds": kept_seeds,
                    "attempted": attempted,
                },
            )
            + "\n",
        )

    dataset.finalize()
    fixed = rewrite_quantile_stats(root)
    log(f"[collect] stats.json quantiles recomputed from raw frames: {sorted(fixed)}")
    summary = {
        "kept": len(kept_seeds),
        "attempted": attempted,
        "kept_seeds": kept_seeds,
        "stop_reason": stop_reason,
        "wall_s": round(time.time() - t0, 1),
        "state_units": STATE_UNITS,
        "expert_head": _git_head(),
        "prereg": "posts/2026-08-14-prereg-grasp-sft-bootstrap.md",
        "substrate": "SO101Sim() production defaults; front <- sim top camera",
        "success_definition": "sim100 harness success_tick machinery (sim.success)",
        **(extra_provenance or {}),
    }
    _provenance_path(root).parent.mkdir(parents=True, exist_ok=True)
    _provenance_path(root).write_text(json.dumps(summary, indent=2) + "\n")
    log(
        f"[collect] DONE ({stop_reason}): {len(kept_seeds)}/{target_kept} kept "
        f"of {attempted} attempted, provenance banked",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-kept", type=int, default=400)
    parser.add_argument("--seed-start", type=int, default=DEMO_SEED_BASE)
    parser.add_argument("--max-seeds", type=int, default=2000)
    parser.add_argument("--max-wall-hours", type=float, default=4.0)
    parser.add_argument("--max-ticks", type=int, default=600)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--spawn-version", choices=("v1", "v2"), default="v1")
    parser.add_argument(
        "--tint-band",
        choices=("rig_gray", "wide", "mix70"),
        default="rig_gray",
    )
    args = parser.parse_args()

    sim = SO101Sim(spawn_version=args.spawn_version, tint_band=args.tint_band)
    collect(
        args.out.expanduser(),
        expert_episode_source(sim, max_ticks=args.max_ticks),
        target_kept=args.target_kept,
        seed_start=args.seed_start,
        max_seeds=args.max_seeds,
        max_wall_s=args.max_wall_hours * 3600,
        repo_id=args.repo_id,
        extra_provenance={
            "spawn_version": args.spawn_version,
            "tint_band": args.tint_band,
            "spawn_v2_prereg": "posts/2026-08-16-prereg-sim-spawn-v2.md"
            if args.spawn_version == "v2"
            else None,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
