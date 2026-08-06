"""Stage-2 sign-convention probe: optical-flow cross-check (CPU).

Pre-registered 2026-08-05 (blog post
`2026-08-05-prereg-sign-stage2.md`) BEFORE this file existed; every
constant below is frozen there. Stage 1 (`probe_sign_convention_stage1`)
found model-vs-truth mirror *signatures* on three (repo, dim) cells but
is structurally blind in both directions. Stage 2 asks the pixels: does
the recorded joint velocity move the world in the same direction as it
does in healthy repos? A flipped-convention repo shows optical flow
*opposite* to what its recorded velocities predict, relative to a
15-repo so100 reference population.

Instrument (per repo, per target dim):
  1. joint velocity from the STATE stream (follower = physical),
     v_d(t) = state_d(t+1) - state_d(t) in deg/frame;
  2. isolated-motion pairs: |v_d| >= 0.5 deg/frame AND |v_d| >= 2x every
     other non-gripper dim (relax once to 1.5x if < 30 pairs; still < 30
     => inconclusive-by-data); capped at 400/repo by uniform stride;
  3. Farneback flow on grayscale 320x240 (params frozen);
  4. flow statistics: omega = image-plane angular velocity about center
     (wrist_roll), t_y = mean vertical flow (wrist_flex, shoulder_lift),
     t_x = mean horizontal flow (shoulder_pan);
  5. ego-cam rule (cams unlabeled): argmax Spearman corr(mean |flow|,
     sum_d |v_d|) with margin >= 0.15, else read both cams and require
     sign agreement (disagree => inconclusive-by-camera);
  6. signed read: Spearman rho between v_d and the dim's statistic;
  7. stream-consistency (no video, classifies mirror type): Spearman
     corr(action_d - state_d, state_d(t+3) - state_d(t)) over pairs with
     |action - state| >= 0.5 deg.

Reference population per dim (deterministic): so100 repos with >= 8
stage-1 panel frames and stage-1 MAE ratio <= 2.0 on the dim, sorted
lexicographically, first 15 yielding >= 30 isolated pairs; validity gate
>= 12/15 sign agreement AND median |rho| >= 0.2. Verdicts per cell:
MIRRORED / NORMAL need |rho| >= 0.3 and >= 90% episode-bootstrap mass
(1000 draws, SeedSequence(13)) on the flipped / matching sign; anything
else is INCONCLUSIVE. Hard validation gate before any candidate cell is
opened: the synthetic-flip oracle (negate the state stream of the
lexicographically-first valid reference repo in-memory) must read
MIRRORED doctored + NORMAL original for each of the three statistics.
Controls (pre-declared): Dongkkka shoulder_pan and kevin510 wrist_roll
must read NORMAL; either MIRRORED => instrument fault, no candidate
verdicts ship. sincostangerines/stack_cubes_p3 gripper has no flow
statistic in the pre-reg => reported as not-runnable, never read.

Phases (the pre-reg allows population + oracle before cells; candidate
verdicts come last, in one shot):
  uv run python -m probes.probe_sign_convention_stage2 --phase population
  uv run python -m probes.probe_sign_convention_stage2 --phase oracle
  uv run python -m probes.probe_sign_convention_stage2 --phase cells

CPU-only; workers are nice'd and cv2/ffmpeg single-threaded so the live
GPU jobs are untouched. Flow statistics are cached per repo under
--cache-dir so re-runs and later phases do not re-decode video; a cached
repo missing a newly-requested dim is recomputed with the dim union.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
import pandas as pd

from probes.probe_sign_convention_stage1 import (
    REFERENCE_NPZ,
    REFERENCE_POLICY,
    repo_stats,
)

CORPUS = Path.home() / "datasets/mcobzarenco/community_curated_v0"
DEFAULT_CACHE_DIR = Path.home() / "sign_stage2_cache"
DEFAULT_REPORT = Path.home() / "sign_stage2_results.json"

# Frozen mapping: target action dim -> flow statistic (pre-reg step 4).
TARGET_STAT = {
    "main_wrist_roll": "omega",
    "main_wrist_flex": "ty",
    "main_shoulder_lift": "ty",
    "main_shoulder_pan": "tx",
}
GRIPPER = "main_gripper"

CANDIDATES = (
    ("kantine/domotic_dishTidyUp_anomaly", "main_wrist_flex"),
    ("kantine/domotic_groceriesSorting_expert", "main_wrist_roll"),
    ("aractingi/push_cube_square_light_reward", "main_shoulder_lift"),
)
# Pre-declared expected outcome NORMAL; MIRRORED => instrument fault.
CONTROLS = (
    ("Dongkkka/koch_arm_gripper_pick_red_pen", "main_shoulder_pan"),
    ("kevin510/lerobot-cat-toy-placement", "main_wrist_roll"),
)
RECORD_ONLY = (
    ("lt-s/so100_train_move_two_blocks_tray_to_matching_dishes", "main_wrist_roll"),
    ("ThomasGossard/grab_box_h2", "main_wrist_flex"),
    ("AntoineA/so100_green_cube_black_circle", "main_wrist_roll"),
)
NOT_RUNNABLE = (("sincostangerines/stack_cubes_p3", GRIPPER),)

MIN_SPEED = 0.5  # deg/frame
DOMINANCE = 2.0
DOMINANCE_RELAXED = 1.5
MIN_PAIRS = 30
MAX_PAIRS = 400
REF_POP = 15
REF_MIN_FRAMES = 8
REF_MAX_RATIO = 2.0
POP_MIN_AGREE = 12  # >= 80% of 15
POP_MIN_MEDIAN_ABS_RHO = 0.2
VERDICT_MIN_ABS_RHO = 0.3
BOOT_DRAWS = 1000
BOOT_MASS = 0.9
BOOT_SEED = 13
EGO_MARGIN = 0.15
STREAM_MIN_DELTA = 0.5  # deg, |action - state| gate for the stream check
STREAM_LOOKAHEAD = 3  # frames
FLOW_W, FLOW_H = 320, 240


class FarnebackParams(TypedDict):
    """Frozen Farneback parameters (pre-reg step 3)."""

    pyr_scale: float
    levels: int
    winsize: int
    iterations: int
    poly_n: int
    poly_sigma: float
    flags: int


FARNEBACK: FarnebackParams = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 21,
    "iterations": 3,
    "poly_n": 7,
    "poly_sigma": 1.5,
    "flags": 0,
}
DECODE_BATCH = 64
STAT_COL = {"omega": 0, "ty": 1, "tx": 2}


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average-tie ranks (scipy-free Spearman support)."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts), dtype=np.float64)
    np.add.at(sums, inverse, ranks)
    return sums[inverse] / counts[inverse]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho; NaN when either signal is (rank-)constant or n < 3."""
    if len(a) < 3:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


@dataclass(frozen=True)
class RepoStreams:
    """State/action streams + video bookkeeping for one repo."""

    repo_id: str
    state: np.ndarray  # (N, dims) float64, degrees
    action: np.ndarray  # (N, dims)
    episode: np.ndarray  # (N,) int64
    frame_index: np.ndarray  # (N,) int64, per-episode
    timestamp: np.ndarray  # (N,) float64, episode-relative seconds
    fps: float
    motor_names: list[str]
    video_keys: list[str]
    episodes_meta: pd.DataFrame  # indexed by episode_index


def load_repo(repo_id: str) -> RepoStreams:
    root = CORPUS / repo_id
    info = json.loads((root / "meta/info.json").read_text())
    video_keys = sorted(
        key.removeprefix("observation.images.")
        for key, feat in info["features"].items()
        if feat.get("dtype") == "video"
    )
    data_files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"], ignore_index=True)
    meta_files = sorted(root.glob("meta/episodes/chunk-*/file-*.parquet"))
    meta = pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True)
    return RepoStreams(
        repo_id=repo_id,
        state=np.stack(df["observation.state"].to_list()).astype(np.float64),
        action=np.stack(df["action"].to_list()).astype(np.float64),
        episode=df["episode_index"].to_numpy().astype(np.int64),
        frame_index=df["frame_index"].to_numpy().astype(np.int64),
        timestamp=df["timestamp"].to_numpy().astype(np.float64),
        fps=float(info["fps"]),
        motor_names=list(info["features"]["observation.state"]["names"]),
        video_keys=video_keys,
        episodes_meta=meta.set_index("episode_index"),
    )


@dataclass(frozen=True)
class PairSet:
    """Isolated-motion pairs for one (repo, dim)."""

    rows: np.ndarray  # (n,) global row index t; the pair is (t, t+1)
    v: np.ndarray  # (n,) signed state velocity on the target dim
    episode: np.ndarray  # (n,)
    dominance: float  # 2.0, or 1.5 after the single pre-registered relax


def isolated_pairs(streams: RepoStreams, dim: int) -> PairSet | None:
    """Frozen selection: speed + dominance gates, one relax, stride cap."""
    consecutive = (streams.episode[:-1] == streams.episode[1:]) & (
        streams.frame_index[:-1] + 1 == streams.frame_index[1:]
    )
    v = streams.state[1:] - streams.state[:-1]
    gripper = streams.motor_names.index(GRIPPER)
    others = [j for j in range(v.shape[1]) if j not in (dim, gripper)]
    for dominance in (DOMINANCE, DOMINANCE_RELAXED):
        mask = consecutive & (np.abs(v[:, dim]) >= MIN_SPEED)
        for j in others:
            mask &= np.abs(v[:, dim]) >= dominance * np.abs(v[:, j])
        rows = np.flatnonzero(mask)
        if len(rows) >= MIN_PAIRS:
            if len(rows) > MAX_PAIRS:
                # Deterministic uniform stride; spacing > 1 guarantees
                # unique indices, no RNG in selection.
                picks = np.round(np.linspace(0, len(rows) - 1, MAX_PAIRS))
                rows = rows[picks.astype(int)]
            return PairSet(
                rows=rows,
                v=v[rows, dim],
                episode=streams.episode[rows],
                dominance=dominance,
            )
    return None


def _make_grid() -> tuple[np.ndarray, np.ndarray, float]:
    ys, xs = np.mgrid[0:FLOW_H, 0:FLOW_W].astype(np.float64)
    rx = xs - (FLOW_W - 1) / 2
    ry = ys - (FLOW_H - 1) / 2
    return rx, ry, float((rx**2 + ry**2).sum())


def _flow_statistics(
    flow: np.ndarray,
    grid: tuple[np.ndarray, np.ndarray, float],
) -> tuple[float, float, float, float]:
    """(omega, t_y, t_x, mean |flow|) for one Farneback field."""
    rx, ry, r2_sum = grid
    cross = rx * flow[..., 1] - ry * flow[..., 0]
    omega = float(cross.sum() / r2_sum)
    ty = float(flow[..., 1].mean())
    tx = float(flow[..., 0].mean())
    mag = float(np.hypot(flow[..., 0], flow[..., 1]).mean())
    return omega, ty, tx, mag


def cache_path(cache_dir: Path, repo_id: str) -> Path:
    return cache_dir / (repo_id.replace("/", "__") + ".npz")


def cached_dims(cache_dir: Path, repo_id: str) -> set[str] | None:
    """Dims already computed for a repo, or None when uncached."""
    path = cache_path(cache_dir, repo_id)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        return {
            "main_" + name.removeprefix("pairs_").removesuffix("_rows")
            for name in data.files
            if name.startswith("pairs_") and name.endswith("_rows")
        }


def compute_repo_flow(task: tuple[str, list[str], str]) -> tuple[str, str]:
    """Worker: flow statistics on every cam over the union of the repo's
    per-dim isolated pairs; caches to npz. Returns (repo_id, "ok"|error)."""
    repo_id, dim_names, cache_dir_s = task
    out = cache_path(Path(cache_dir_s), repo_id)
    try:
        from torchcodec.decoders import VideoDecoder

        cv2.setNumThreads(1)
        os.nice(19)
        streams = load_repo(repo_id)
        dims = {name: streams.motor_names.index(name) for name in dim_names}
        pair_sets = {name: isolated_pairs(streams, d) for name, d in dims.items()}
        row_arrays = [ps.rows for ps in pair_sets.values() if ps is not None]
        union_rows = np.unique(
            np.concatenate(row_arrays or [np.empty(0, dtype=np.int64)]),
        ).astype(np.int64)
        payload: dict[str, np.ndarray] = {"union_rows": union_rows}
        for name, ps in pair_sets.items():
            key = name.removeprefix("main_")
            if ps is None:
                payload[f"pairs_{key}_rows"] = np.empty(0, dtype=np.int64)
                continue
            payload[f"pairs_{key}_rows"] = ps.rows
            payload[f"pairs_{key}_v"] = ps.v
            payload[f"pairs_{key}_episode"] = ps.episode
            payload[f"pairs_{key}_dominance"] = np.array(ps.dominance)
        if len(union_rows) == 0:
            np.savez_compressed(out, **payload)  # pyright: ignore[reportArgumentType]
            return repo_id, "ok"

        v_all = streams.state[1:] - streams.state[:-1]
        gripper = streams.motor_names.index(GRIPPER)
        non_gripper = [j for j in range(v_all.shape[1]) if j != gripper]
        payload["union_sum_abs_v"] = np.abs(v_all[union_rows][:, non_gripper]).sum(1)
        payload["union_episode"] = streams.episode[union_rows]
        payload["cams"] = np.array(streams.video_keys)

        grid = _make_grid()
        half = 0.5 / streams.fps
        meta = streams.episodes_meta
        for cam_i, cam in enumerate(streams.video_keys):
            key = f"observation.images.{cam}"
            stats = np.full((len(union_rows), 4), np.nan)
            groups: dict[tuple[int, int], list[int]] = {}
            for i, t in enumerate(union_rows):
                row = meta.loc[int(streams.episode[t])]
                loc = (
                    int(row[f"videos/{key}/chunk_index"]),
                    int(row[f"videos/{key}/file_index"]),
                )
                groups.setdefault(loc, []).append(i)
            for (chunk, file), members in sorted(groups.items()):
                path = (
                    CORPUS
                    / repo_id
                    / f"videos/{key}/chunk-{chunk:03d}/file-{file:03d}.mp4"
                )
                decoder = VideoDecoder(path, num_ffmpeg_threads=1)
                times: list[float] = []
                for i in members:
                    t = int(union_rows[i])
                    row = meta.loc[int(streams.episode[t])]
                    from_ts = float(row[f"videos/{key}/from_timestamp"])
                    to_ts = float(row[f"videos/{key}/to_timestamp"])
                    t0 = min(from_ts + streams.timestamp[t] + half, to_ts - 1e-4)
                    t1 = min(from_ts + streams.timestamp[t + 1] + half, to_ts - 1e-4)
                    times.extend((t0, t1))
                grays: list[np.ndarray] = []
                for start in range(0, len(times), DECODE_BATCH):
                    batch = decoder.get_frames_played_at(
                        times[start : start + DECODE_BATCH],
                    )
                    for frame in batch.data:
                        rgb = frame.permute(1, 2, 0).numpy()
                        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                        grays.append(
                            cv2.resize(
                                gray,
                                (FLOW_W, FLOW_H),
                                interpolation=cv2.INTER_AREA,
                            ),
                        )
                for k, i in enumerate(members):
                    # cv2 stubs reject flow=None; runtime requires it for
                    # a fresh (non-warm-started) field.
                    flow = cv2.calcOpticalFlowFarneback(  # pyright: ignore[reportCallIssue]
                        grays[2 * k],
                        grays[2 * k + 1],
                        None,  # pyright: ignore[reportArgumentType]
                        **FARNEBACK,
                    )
                    stats[i] = _flow_statistics(flow, grid)
            payload[f"cam{cam_i}_stats"] = stats
        np.savez_compressed(out, **payload)  # pyright: ignore[reportArgumentType]
    except Exception as exc:  # noqa: BLE001 - worker boundary, reported upward
        return repo_id, f"{type(exc).__name__}: {exc}"
    return repo_id, "ok"


class FlowCache:
    """Read-side view of one repo's cached flow statistics."""

    def __init__(self, cache_dir: Path, repo_id: str) -> None:
        self.repo_id = repo_id
        self.data = np.load(cache_path(cache_dir, repo_id), allow_pickle=True)
        self.cams: list[str] = [str(c) for c in self.data["cams"]]
        self.union_rows: np.ndarray = self.data["union_rows"]

    def pairs(self, dim_name: str) -> PairSet | None:
        key = dim_name.removeprefix("main_")
        rows = self.data[f"pairs_{key}_rows"]
        if len(rows) < MIN_PAIRS:
            return None
        return PairSet(
            rows=rows,
            v=self.data[f"pairs_{key}_v"],
            episode=self.data[f"pairs_{key}_episode"],
            dominance=float(self.data[f"pairs_{key}_dominance"]),
        )

    def stat_at(self, cam_i: int, rows: np.ndarray, stat: str) -> np.ndarray:
        idx = np.searchsorted(self.union_rows, rows)
        return self.data[f"cam{cam_i}_stats"][idx, STAT_COL[stat]]

    def ego_cam(self) -> tuple[int, float]:
        """(ego cam index, margin) via corr(mean |flow|, sum_d |v_d|)."""
        sum_abs_v = self.data["union_sum_abs_v"]
        corrs = [
            spearman(self.data[f"cam{i}_stats"][:, 3], sum_abs_v)
            for i in range(len(self.cams))
        ]
        scored = np.nan_to_num(np.array(corrs), nan=-2.0)
        best = int(scored.argmax())
        margin = (
            float(scored[best] - np.delete(scored, best).max())
            if len(self.cams) > 1
            else float("inf")
        )
        return best, margin


@dataclass(frozen=True)
class RepoRead:
    """One (repo, dim) signed read off the cached flow statistics."""

    repo_id: str
    dim_name: str
    n_pairs: int
    dominance: float
    ego_cam: str
    ego_margin: float
    ego_ok: bool  # margin >= 0.15
    rho: float  # ego cam (or best cam when no margin)
    rho_other: float  # the other cam (NaN for 1-cam repos)
    signs_agree: bool  # both-cam sign agreement (True when ego_ok)


def read_repo_dim(cache: FlowCache, dim_name: str) -> RepoRead | None:
    """The signed read; None when the repo is inconclusive-by-data."""
    ps = cache.pairs(dim_name)
    if ps is None:
        return None
    stat = TARGET_STAT[dim_name]
    best, margin = cache.ego_cam()
    rhos = [
        spearman(ps.v, cache.stat_at(i, ps.rows, stat)) for i in range(len(cache.cams))
    ]
    rho = rhos[best]
    other = [r for i, r in enumerate(rhos) if i != best]
    rho_other = other[0] if other else float("nan")
    ego_ok = margin >= EGO_MARGIN
    signs_agree = bool(
        ego_ok
        or (
            np.isfinite(rho)
            and np.isfinite(rho_other)
            and np.sign(rho) == np.sign(rho_other)
        ),
    )
    return RepoRead(
        repo_id=cache.repo_id,
        dim_name=dim_name,
        n_pairs=len(ps.rows),
        dominance=ps.dominance,
        ego_cam=cache.cams[best],
        ego_margin=margin,
        ego_ok=ego_ok,
        rho=rho,
        rho_other=rho_other,
        signs_agree=signs_agree,
    )


def bootstrap_sign_mass(
    ps: PairSet,
    stat_values: np.ndarray,
    target_sign: float,
) -> tuple[float, tuple[float, float]]:
    """Fraction of episode-bootstrap draws whose rho carries target_sign,
    plus the [2.5, 97.5] percentile interval of the draw distribution."""
    rng = np.random.default_rng(np.random.SeedSequence(BOOT_SEED))
    episodes = np.unique(ps.episode)
    by_episode = {int(e): np.flatnonzero(ps.episode == e) for e in episodes}
    hits = 0
    draws: list[float] = []
    for _ in range(BOOT_DRAWS):
        chosen = rng.choice(episodes, size=len(episodes), replace=True)
        idx = np.concatenate([by_episode[int(e)] for e in chosen])
        rho = spearman(ps.v[idx], stat_values[idx])
        if np.isfinite(rho):
            draws.append(rho)
            if np.sign(rho) == target_sign:
                hits += 1
    if not draws:
        return float("nan"), (float("nan"), float("nan"))
    lo, hi = np.percentile(np.array(draws), [2.5, 97.5])
    return hits / len(draws), (float(lo), float(hi))


@dataclass(frozen=True)
class CellVerdict:
    """Decision-rule output for one candidate/control/oracle read."""

    repo_id: str
    dim_name: str
    verdict: str  # MIRRORED / NORMAL / INCONCLUSIVE / inconclusive-by-*
    rho: float
    n_pairs: int
    dominance: float
    ego_cam: str
    ego_margin: float
    ref_sign: float
    boot_mass_flipped: float
    boot_mass_matched: float
    boot_interval: tuple[float, float]


def judge_cell(
    cache: FlowCache,
    dim_name: str,
    ref_sign: float,
    *,
    negate: bool = False,
) -> CellVerdict:
    """Apply the frozen decision rules to one (repo, dim) read.

    negate=True is the synthetic-flip oracle: the state stream's velocity
    on the target dim is negated in-memory (pair selection is invariant,
    |v| is unchanged) and the same rules run on the doctored read.
    """
    read = read_repo_dim(cache, dim_name)
    if read is None:
        return CellVerdict(
            repo_id=cache.repo_id,
            dim_name=dim_name,
            verdict="inconclusive-by-data",
            rho=float("nan"),
            n_pairs=0,
            dominance=float("nan"),
            ego_cam="-",
            ego_margin=float("nan"),
            ref_sign=ref_sign,
            boot_mass_flipped=float("nan"),
            boot_mass_matched=float("nan"),
            boot_interval=(float("nan"), float("nan")),
        )
    base = cache.pairs(dim_name)
    assert base is not None
    ps = PairSet(
        rows=base.rows,
        v=-base.v if negate else base.v,
        episode=base.episode,
        dominance=base.dominance,
    )
    best, _ = cache.ego_cam()
    stat_values = cache.stat_at(best, ps.rows, TARGET_STAT[dim_name])
    rho = spearman(ps.v, stat_values)
    mass_flipped, interval = bootstrap_sign_mass(ps, stat_values, -ref_sign)
    mass_matched = 1.0 - mass_flipped if np.isfinite(mass_flipped) else float("nan")
    # (draws with rho exactly 0 would count as neither sign; with
    # continuous flow statistics that set is empty in practice.)
    if not read.signs_agree:
        verdict = "inconclusive-by-camera"
    elif (
        np.isfinite(rho)
        and np.sign(rho) == -ref_sign
        and abs(rho) >= VERDICT_MIN_ABS_RHO
        and mass_flipped >= BOOT_MASS
    ):
        verdict = "MIRRORED"
    elif (
        np.isfinite(rho)
        and np.sign(rho) == ref_sign
        and abs(rho) >= VERDICT_MIN_ABS_RHO
        and mass_matched >= BOOT_MASS
    ):
        verdict = "NORMAL"
    else:
        verdict = "INCONCLUSIVE"
    return CellVerdict(
        repo_id=cache.repo_id,
        dim_name=dim_name,
        verdict=verdict,
        rho=rho,
        n_pairs=read.n_pairs,
        dominance=read.dominance,
        ego_cam=read.ego_cam,
        ego_margin=read.ego_margin,
        ref_sign=ref_sign,
        boot_mass_flipped=mass_flipped,
        boot_mass_matched=mass_matched,
        boot_interval=interval,
    )


def stream_consistency(repo_id: str, dim_name: str) -> tuple[float, int]:
    """Pre-reg step 7: servo-consistency read (parquet only, no video)."""
    streams = load_repo(repo_id)
    d = streams.motor_names.index(dim_name)
    n = len(streams.state)
    t = np.arange(n - STREAM_LOOKAHEAD)
    same_ep = streams.episode[t] == streams.episode[t + STREAM_LOOKAHEAD]
    contiguous = (
        streams.frame_index[t + STREAM_LOOKAHEAD]
        == streams.frame_index[t] + STREAM_LOOKAHEAD
    )
    delta = streams.action[t, d] - streams.state[t, d]
    mask = same_ep & contiguous & (np.abs(delta) >= STREAM_MIN_DELTA)
    future = streams.state[t + STREAM_LOOKAHEAD, d] - streams.state[t, d]
    return spearman(delta[mask], future[mask]), int(mask.sum())


def stage1_ratio_table() -> tuple[dict[str, np.ndarray], dict[str, int], list[str]]:
    """Per-repo per-dim MAE ratios + panel frame counts off the frozen npz."""
    sidecar = REFERENCE_NPZ.with_suffix(".json")
    motor_names = json.loads(sidecar.read_text())["motor_names"]
    dump = np.load(REFERENCE_NPZ, allow_pickle=True)
    core = dump["core"]
    rows = repo_stats(
        dump["truth"][core],
        dump[REFERENCE_POLICY][core],
        dump["valid"][core],
        dump["repo_id"][core],
    )
    panel_median = np.median(np.stack([r.mae for r in rows]), 0)
    ratios = {r.repo_id: r.mae / panel_median for r in rows}
    frames = {r.repo_id: r.frames for r in rows}
    return ratios, frames, motor_names


def is_so100(repo_id: str) -> bool:
    info_path = CORPUS / repo_id / "meta/info.json"
    if not info_path.exists():
        return False
    return json.loads(info_path.read_text()).get("robot_type") == "so100"


@dataclass(frozen=True)
class PopulationGate:
    """Validity read for one dim's reference population."""

    n: int
    median_rho: float
    healthy_sign: float
    agree: int
    valid: bool


def run_flow(
    repo_dims: dict[str, list[str]],
    cache_dir: Path,
    workers: int,
) -> dict[str, str]:
    """Compute (or reuse) cached flow statistics; returns repo -> status.

    A cached repo missing any requested dim is recomputed over the union
    of cached + requested dims (flow cost repeats, correctness first).
    """
    tasks: list[tuple[str, list[str], str]] = []
    status: dict[str, str] = {}
    for repo_id, dims in sorted(repo_dims.items()):
        have = cached_dims(cache_dir, repo_id)
        if have is not None and set(dims) <= have:
            status[repo_id] = "cached"
            continue
        tasks.append((repo_id, sorted(set(dims) | (have or set())), str(cache_dir)))
    if tasks:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            for repo_id, result in pool.map(compute_repo_flow, tasks):
                status[repo_id] = result
                print(f"  flow {repo_id}: {result}", flush=True)
    return status


def build_populations(
    stage1: tuple[dict[str, np.ndarray], dict[str, int], list[str]],
    cache_dir: Path,
    workers: int,
) -> tuple[dict[str, list[RepoRead]], dict[str, PopulationGate]]:
    """Select + read the 15-repo reference population per target dim."""
    ratios, frames, motor_names = stage1
    shortlists = {}
    for dim in TARGET_STAT:
        d = motor_names.index(dim)
        shortlists[dim] = sorted(
            repo_id
            for repo_id, ratio in ratios.items()
            if frames[repo_id] >= REF_MIN_FRAMES
            and ratio[d] <= REF_MAX_RATIO
            and is_so100(repo_id)
        )
    print("reference shortlists (eligible so100 repos, lexicographic):")
    for dim, repos in shortlists.items():
        print(f"  {dim}: {len(repos)} eligible")

    populations: dict[str, list[RepoRead]] = {dim: [] for dim in TARGET_STAT}
    cursor = dict.fromkeys(TARGET_STAT, 0)
    failed: set[str] = set()
    # Iterate in lexicographic order per dim, computing flow in batches,
    # until each population holds 15 repos with >= 30 pairs (pre-reg).
    while True:
        need: dict[str, list[str]] = {}
        for dim, repos in shortlists.items():
            missing = REF_POP - len(populations[dim])
            if missing <= 0:
                continue
            for repo_id in repos[cursor[dim] : cursor[dim] + missing * 2]:
                if repo_id not in failed:
                    need.setdefault(repo_id, []).append(dim)
        if not need:
            break
        for repo_id, result in run_flow(need, cache_dir, workers).items():
            if result not in ("ok", "cached"):
                failed.add(repo_id)
        for dim in TARGET_STAT:
            repos = shortlists[dim]
            while len(populations[dim]) < REF_POP and cursor[dim] < len(repos):
                repo_id = repos[cursor[dim]]
                if repo_id in failed:
                    cursor[dim] += 1
                    continue
                if cached_dims(cache_dir, repo_id) is None:
                    break  # not yet computed; next batch
                cursor[dim] += 1
                cache = FlowCache(cache_dir, repo_id)
                read = read_repo_dim(cache, dim)
                if read is not None and np.isfinite(read.rho):
                    populations[dim].append(read)
        if all(
            len(populations[dim]) >= REF_POP or cursor[dim] >= len(shortlists[dim])
            for dim in TARGET_STAT
        ):
            break

    gates: dict[str, PopulationGate] = {}
    for dim, reads in populations.items():
        rhos = np.array([r.rho for r in reads])
        median_rho = float(np.median(rhos)) if len(rhos) else float("nan")
        sign = float(np.sign(median_rho)) if np.isfinite(median_rho) else 0.0
        agree = int((np.sign(rhos) == sign).sum()) if len(rhos) else 0
        gates[dim] = PopulationGate(
            n=len(reads),
            median_rho=median_rho,
            healthy_sign=sign,
            agree=agree,
            valid=(
                len(reads) >= REF_POP
                and agree >= POP_MIN_AGREE
                and abs(median_rho) >= POP_MIN_MEDIAN_ABS_RHO
            ),
        )
        flag = "VALID" if gates[dim].valid else "INVALID-BY-POPULATION"
        print(
            f"population {dim}: n={len(reads)} agree={agree}/{len(reads)} "
            f"median rho={median_rho:+.3f} -> {flag}",
        )
        for r in reads:
            no_margin = "" if r.ego_ok else " NO-MARGIN"
            print(
                f"    {r.repo_id[:52]:52s} n={r.n_pairs:4d} dom={r.dominance:.1f} "
                f"ego={r.ego_cam}({r.ego_margin:+.2f}{no_margin}) "
                f"rho={r.rho:+.3f} other={r.rho_other:+.3f}",
            )
    return populations, gates


def run_oracle(
    populations: dict[str, list[RepoRead]],
    gates: dict[str, PopulationGate],
    cache_dir: Path,
) -> dict[str, dict[str, object]]:
    """Synthetic-flip hard gate: one oracle per statistic family."""
    families = (
        ("omega", "main_wrist_roll"),
        ("ty", "main_wrist_flex"),
        ("tx", "main_shoulder_pan"),
    )
    results: dict[str, dict[str, object]] = {}
    for stat, dim in families:
        if not gates[dim].valid:
            results[stat] = {"passed": False, "reason": "population invalid"}
            continue
        reads = populations[dim]
        subject = min(reads, key=lambda r: r.repo_id)
        rest = [r.rho for r in reads if r.repo_id != subject.repo_id]
        ref_sign = float(np.sign(np.median(rest)))
        cache = FlowCache(cache_dir, subject.repo_id)
        original = judge_cell(cache, dim, ref_sign)
        doctored = judge_cell(cache, dim, ref_sign, negate=True)
        passed = original.verdict == "NORMAL" and doctored.verdict == "MIRRORED"
        results[stat] = {
            "subject": subject.repo_id,
            "dim": dim,
            "original": asdict(original),
            "doctored": asdict(doctored),
            "passed": bool(passed),
        }
        print(
            f"oracle {stat} on {subject.repo_id} ({dim}): original "
            f"{original.verdict} (rho {original.rho:+.3f}, mass "
            f"{original.boot_mass_matched:.3f}) / doctored {doctored.verdict} "
            f"(rho {doctored.rho:+.3f}, mass {doctored.boot_mass_flipped:.3f}) "
            f"-> {'PASS' if passed else 'FAIL'}",
        )
    return results


def run_cells(
    gates: dict[str, PopulationGate],
    cache_dir: Path,
    workers: int,
) -> dict[str, object]:
    """Candidate + control + record-only reads, one shot, verdicts last."""
    cells = [(r, d, "control") for r, d in CONTROLS]
    cells += [(r, d, "candidate") for r, d in CANDIDATES]
    cells += [(r, d, "record-only") for r, d in RECORD_ONLY]
    run_flow({repo_id: [dim] for repo_id, dim, _ in cells}, cache_dir, workers)
    out: list[dict[str, object]] = []
    instrument_fault = False
    for repo_id, dim, role in cells:
        gate = gates[dim]
        if not gate.valid:
            out.append(
                {
                    "repo": repo_id,
                    "dim": dim,
                    "role": role,
                    "verdict": "invalid-by-population",
                },
            )
            continue
        cache = FlowCache(cache_dir, repo_id)
        verdict = judge_cell(cache, dim, gate.healthy_sign)
        rho_sc, n_sc = stream_consistency(repo_id, dim)
        if role == "control" and verdict.verdict == "MIRRORED":
            instrument_fault = True
        out.append(
            {
                "repo": repo_id,
                "dim": dim,
                "role": role,
                **asdict(verdict),
                "stream_consistency_rho": rho_sc,
                "stream_consistency_n": n_sc,
            },
        )
        print(
            f"{role:11s} {repo_id[:48]:48s} {dim:18s} -> {verdict.verdict:12s} "
            f"rho={verdict.rho:+.3f} [{verdict.boot_interval[0]:+.3f},"
            f"{verdict.boot_interval[1]:+.3f}] n={verdict.n_pairs} "
            f"mass(flip)={verdict.boot_mass_flipped:.3f} "
            f"stream rho={rho_sc:+.3f} (n={n_sc})",
        )
    for repo_id, dim in NOT_RUNNABLE:
        out.append(
            {
                "repo": repo_id,
                "dim": dim,
                "role": "record-only",
                "verdict": "not-runnable (no flow statistic for gripper)",
            },
        )
    if instrument_fault:
        print(
            "CONTROL READ MIRRORED -> instrument-fault presumption: candidate "
            "verdicts are VOID (pre-reg: a debug post ships, no verdicts).",
        )
    return {"cells": out, "instrument_fault": instrument_fault}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["population", "oracle", "cells", "all"],
        default="all",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    stage1 = stage1_ratio_table()
    report: dict[str, object] = (
        json.loads(args.report.read_text()) if args.report.exists() else {}
    )

    populations, gates = build_populations(stage1, args.cache_dir, args.workers)
    report["populations"] = {
        dim: [asdict(r) for r in reads] for dim, reads in populations.items()
    }
    report["population_gates"] = {dim: asdict(g) for dim, g in gates.items()}

    if args.phase in ("oracle", "cells", "all"):
        oracle = run_oracle(populations, gates, args.cache_dir)
        report["oracle"] = oracle
        oracle_ok = all(bool(res.get("passed")) for res in oracle.values())
        print(f"synthetic-flip hard gate: {'PASSED' if oracle_ok else 'FAILED'}")
        if args.phase in ("cells", "all"):
            if not oracle_ok:
                print("oracle FAILED -> candidate cells stay closed (pre-reg).")
            else:
                report["cells"] = run_cells(gates, args.cache_dir, args.workers)

    args.report.write_text(json.dumps(report, indent=2))
    print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
