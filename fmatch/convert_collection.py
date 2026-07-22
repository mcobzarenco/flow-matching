"""Migrate a collection of LeRobot sub-datasets to dataset format v3.0.

Points at a collection root laid out as ``<root>/<user>/<dataset>`` (e.g. the
HuggingFaceVLA community_dataset_v1 download), prints a census of the
LeRobot format versions found, then converts every selected sub-dataset:

    v2.0 --[synthesize per-episode stats]--> v2.1 --[official converter]--> v3.0

Results land in ``<output>/<user>/<dataset>``. The source is never modified.

Usage:
    uv run python -m fmatch.convert_collection \
        --source /home/marius/w/community_dataset_v1 \
        --output /home/marius/w/community_dataset_v1_v3

    # census only / a subset / redo
    ... --stats-only
    ... --datasets ZGGZZG/so100_drop0 ad330/cubePlace
    ... --force

Properties:
  - idempotent: sub-datasets whose output already exists as valid v3.0 are
    skipped; interrupted work is staged in ``<output>/.staging`` and redone.
  - repairs the known collection quirk where stats keys dropped the
    ``images.`` segment (``observation.image`` vs feature
    ``observation.images.image``) before converting.
  - copies only real dataset files (skips the duplicate flat video trees and
    ``*.bak`` files present in the community collections).
  - appends one JSON line per processed dataset to
    ``<output>/conversion_manifest.jsonl``.
"""

import argparse
import json
import shutil
import subprocess
import tempfile
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

SUPPORTED_SOURCE_VERSIONS = {"v2.0", "v2.1"}


def log(name: str, message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [{name}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Discovery / census
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubDataset:
    name: str  # "<user>/<dataset>"
    path: Path
    version: str
    episodes: int
    frames: int
    size_bytes: int


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def discover(source: Path) -> list[SubDataset]:
    found = []
    for info_path in sorted(source.glob("*/*/meta/info.json")):
        ds_dir = info_path.parent.parent
        info = json.loads(info_path.read_text())
        found.append(
            SubDataset(
                name=f"{ds_dir.parent.name}/{ds_dir.name}",
                path=ds_dir,
                version=str(info.get("codebase_version", "?")),
                episodes=int(info.get("total_episodes", 0)),
                frames=int(info.get("total_frames", 0)),
                size_bytes=dir_size(ds_dir),
            )
        )
    return found


def print_census(datasets: list[SubDataset], output: Path) -> None:
    by_version = Counter(d.version for d in datasets)
    print(f"\n{len(datasets)} sub-datasets found")
    print(f"{'version':<10}{'datasets':>10}{'episodes':>12}{'frames':>14}{'size':>10}")
    for version in sorted(by_version):
        group = [d for d in datasets if d.version == version]
        marker = "" if version in SUPPORTED_SOURCE_VERSIONS else "  (unsupported!)"
        print(
            f"{version:<10}{len(group):>10}{sum(d.episodes for d in group):>12}"
            f"{sum(d.frames for d in group):>14}{sum(d.size_bytes for d in group) / 1e9:>9.1f}G"
            f"{marker}"
        )
    done = sum(1 for d in datasets if is_converted(output, d.name))
    print(f"already converted in {output}: {done}/{len(datasets)}\n")


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def is_converted(output: Path, name: str) -> bool:
    info_path = output / name / "meta" / "info.json"
    if not info_path.is_file():
        return False
    try:
        return json.loads(info_path.read_text()).get("codebase_version") == "v3.0"
    except json.JSONDecodeError:
        return False


# ---------------------------------------------------------------------------
# Staging copy (skips junk: duplicate flat video dirs, *.bak files)
# ---------------------------------------------------------------------------


def stage_copy(src: Path, dst: Path, info: dict) -> None:
    video_keys = {k for k, f in info["features"].items() if f.get("dtype") == "video"}
    dst.mkdir(parents=True, exist_ok=True)

    meta_dst = dst / "meta"
    shutil.copytree(
        src / "meta",
        meta_dst,
        ignore=shutil.ignore_patterns("*.bak"),
        dirs_exist_ok=True,
    )
    shutil.copytree(src / "data", dst / "data", dirs_exist_ok=True)

    # v2.x layout: videos/chunk-XXX/<video_key>/episode_XXXXXX.mp4.
    # Copy only directories matching real video features; the community
    # collections also contain stray duplicates under non-feature names.
    for chunk_dir in sorted((src / "videos").glob("chunk-*")):
        for key_dir in sorted(chunk_dir.iterdir()):
            if key_dir.name in video_keys:
                shutil.copytree(
                    key_dir,
                    dst / "videos" / chunk_dir.name / key_dir.name,
                    dirs_exist_ok=True,
                )


# ---------------------------------------------------------------------------
# Data-column sanitation (drop columns not declared in info features)
# ---------------------------------------------------------------------------


def sanitize_data_columns(root: Path, info: dict) -> set[str]:
    """Drop parquet columns that aren't declared features.

    Some community datasets carry legacy columns (e.g. ``next.done``) in the
    per-episode parquets that are absent from ``info.json`` features; the
    datasets library later refuses to cast the consolidated v3 file against
    the declared schema.
    """
    declared = {k for k, f in info["features"].items() if f.get("dtype") != "video"}
    dropped: set[str] = set()
    for parquet_path in sorted(root.glob("data/chunk-*/episode_*.parquet")):
        names = pq.read_schema(parquet_path).names  # cheap: footer only
        extras = [c for c in names if c not in declared]
        if extras:
            dropped.update(extras)
            table = pq.read_table(parquet_path)
            table = table.select([c for c in names if c in declared])
            pq.write_table(table, parquet_path)
    return dropped


# ---------------------------------------------------------------------------
# Video concat: ffmpeg fallback for streams PyAV refuses to mux
# ---------------------------------------------------------------------------

_original_concatenate = None


def concatenate_with_ffmpeg_fallback(
    input_video_paths: list,
    output_video_path: Path | str,
    *args: object,
    **kwargs: object,
) -> None:
    """lerobot's PyAV concat, falling back to ffmpeg's concat demuxer.

    Some episodes contain duplicate/non-monotonic DTS at file boundaries;
    PyAV's mux raises EINVAL on these while ffmpeg repairs them in stream
    copy mode.
    """
    assert _original_concatenate is not None
    try:
        _original_concatenate(input_video_paths, output_video_path, *args, **kwargs)
        return
    except Exception as error:  # noqa: BLE001 - deliberate fallback
        log("video-concat", f"pyav failed ({error}); retrying with ffmpeg")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        for path in input_video_paths:
            escaped = str(Path(path).resolve()).replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")
        list_path = Path(handle.name)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                str(output_video_path),
            ],
            check=True,
        )
    finally:
        list_path.unlink(missing_ok=True)


def install_concat_fallback() -> None:
    global _original_concatenate
    import lerobot.scripts.convert_dataset_v21_to_v30 as converter_module

    if converter_module.concatenate_video_files is not concatenate_with_ffmpeg_fallback:
        _original_concatenate = converter_module.concatenate_video_files
        converter_module.concatenate_video_files = concatenate_with_ffmpeg_fallback


# ---------------------------------------------------------------------------
# Stats-key repair (observation.X -> observation.images.X)
# ---------------------------------------------------------------------------


def build_stats_key_repairs(features: dict) -> dict[str, str]:
    repairs = {}
    for key in features:
        if key.startswith("observation.images."):
            flat = key.replace("observation.images.", "observation.", 1)
            if flat not in features:
                repairs[flat] = key
    return repairs


def repair_stats_keys(root: Path, features: dict) -> int:
    """Re-key camera stats entries that lost the ``images.`` segment."""
    repairs = build_stats_key_repairs(features)
    fixed = 0

    stats_path = root / "meta" / "stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())
        if any(old in stats for old in repairs):
            stats = {repairs.get(k, k): v for k, v in stats.items()}
            stats_path.write_text(json.dumps(stats, indent=4))
            fixed += 1

    ep_stats_path = root / "meta" / "episodes_stats.jsonl"
    if ep_stats_path.is_file():
        lines = ep_stats_path.read_text().splitlines()
        rewritten, changed = [], False
        for line in lines:
            record = json.loads(line)
            stats = record.get("stats", {})
            if any(old in stats for old in repairs):
                record["stats"] = {repairs.get(k, k): v for k, v in stats.items()}
                changed = True
            rewritten.append(json.dumps(record))
        if changed:
            ep_stats_path.write_text("\n".join(rewritten) + "\n")
            fixed += 1
    return fixed


# ---------------------------------------------------------------------------
# v2.0 -> v2.1: synthesize meta/episodes_stats.jsonl
# ---------------------------------------------------------------------------


def episode_video_stats(video_path: Path) -> dict[str, np.ndarray]:
    """Per-channel stats over sampled frames of one episode video, in [0,1].

    Note: this is the CPU hot spot of the whole pipeline — frame-accurate
    sampling of AV1 video decodes from the nearest keyframe for every sampled
    index. Parallelism is applied at the dataset level (--workers) instead of
    here to avoid oversubscription.
    """
    from lerobot.datasets.compute_stats import get_feature_stats, sample_indices
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(str(video_path))
    num_frames = decoder.metadata.num_frames
    if not num_frames:
        raise RuntimeError(f"no frames in {video_path}")
    indices = sample_indices(int(num_frames))
    frames = decoder.get_frames_at(indices).data.numpy().astype(np.float32)  # (N,C,H,W)
    stats = get_feature_stats(frames, axis=(0, 2, 3), keepdims=True, quantile_list=[])
    return {
        k: v if k == "count" else np.squeeze(v / 255.0, axis=0)
        for k, v in stats.items()
    }


def synthesize_episodes_stats(root: Path, info: dict, name: str) -> None:
    """Write meta/episodes_stats.jsonl for a v2.0 dataset and bump to v2.1."""
    from lerobot.datasets.compute_stats import get_feature_stats

    features = info["features"]
    chunks_size = int(info.get("chunks_size", 1000))
    episodes = [
        json.loads(line)
        for line in (root / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]

    records = []
    for i, episode in enumerate(episodes):
        if i % 10 == 0:
            log(name, f"synthesizing episode stats (v2.0->v2.1): {i}/{len(episodes)}")
        ep_idx = int(episode["episode_index"])
        chunk = ep_idx // chunks_size
        df = pd.read_parquet(
            root / info["data_path"].format(episode_chunk=chunk, episode_index=ep_idx)
        )

        ep_stats: dict[str, dict] = {}
        for key, feature in features.items():
            dtype = feature.get("dtype")
            if dtype in ("string", "language"):
                continue
            if dtype in ("video", "image"):
                video_path = root / info["video_path"].format(
                    episode_chunk=chunk, video_key=key, episode_index=ep_idx
                )
                ep_stats[key] = episode_video_stats(video_path)
            else:
                column = df[key].to_numpy()
                array = np.stack(list(column)) if column.dtype == object else column
                ep_stats[key] = get_feature_stats(
                    array, axis=0, keepdims=array.ndim == 1, quantile_list=[]
                )

        serialized = {
            key: {k: np.asarray(v).tolist() for k, v in stats.items()}
            for key, stats in ep_stats.items()
        }
        records.append(json.dumps({"episode_index": ep_idx, "stats": serialized}))

    (root / "meta" / "episodes_stats.jsonl").write_text("\n".join(records) + "\n")
    info["codebase_version"] = "v2.1"
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=4))


# ---------------------------------------------------------------------------
# Conversion driver
# ---------------------------------------------------------------------------


def validate_v3(root: Path, name: str) -> dict:
    """Load the converted dataset and cross-check its bookkeeping."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    info = json.loads((root / "meta" / "info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        raise RuntimeError(
            f"expected v3.0 after conversion, got {info.get('codebase_version')}"
        )
    dataset = LeRobotDataset(name, root=root)
    if dataset.num_episodes != info["total_episodes"]:
        raise RuntimeError(
            f"episode count mismatch: {dataset.num_episodes} != {info['total_episodes']}"
        )
    if dataset.num_frames != info["total_frames"]:
        raise RuntimeError(
            f"frame count mismatch: {dataset.num_frames} != {info['total_frames']}"
        )
    return {"episodes": dataset.num_episodes, "frames": dataset.num_frames}


def convert_one(ds: SubDataset, output: Path) -> dict:
    from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset

    started = time.time()
    staging_parent = output / ".staging" / ds.name.split("/")[0]
    staging = staging_parent / ds.name.split("/")[1]
    if staging_parent.parent.exists() and staging.exists():
        shutil.rmtree(staging)  # stale partial work from an interrupted run
    old_leftover = staging.parent / f"{staging.name}_old"
    if old_leftover.exists():
        shutil.rmtree(old_leftover)

    info = json.loads((ds.path / "meta" / "info.json").read_text())

    video_keys = [k for k, f in info["features"].items() if f.get("dtype") == "video"]
    if video_keys and not (ds.path / "videos").is_dir():
        raise RuntimeError(
            f"declares video features {video_keys} but has no videos/ directory"
        )

    log(ds.name, f"staging copy ({ds.size_bytes / 1e9:.1f} GB, {ds.version})")
    stage_copy(ds.path, staging, info)
    dropped = sanitize_data_columns(staging, info)
    if dropped:
        log(ds.name, f"dropped undeclared parquet columns: {sorted(dropped)}")
    if repair_stats_keys(staging, info["features"]):
        log(ds.name, "repaired flat camera stats keys")
    if (
        ds.version == "v2.0"
        or not (staging / "meta" / "episodes_stats.jsonl").is_file()
    ):
        synthesize_episodes_stats(staging, info, ds.name)

    log(ds.name, "converting v2.1 -> v3.0")
    install_concat_fallback()
    convert_dataset(repo_id=ds.name, root=staging, push_to_hub=False)

    if old_leftover.exists():
        shutil.rmtree(old_leftover)

    log(ds.name, "validating")
    counts = validate_v3(staging, ds.name)

    final = output / ds.name
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        shutil.rmtree(final)
    shutil.move(str(staging), str(final))
    log(ds.name, f"done in {time.time() - started:.0f}s -> {final}")

    return {
        "status": "converted",
        "source_version": ds.version,
        "seconds": round(time.time() - started, 1),
        **counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Collection root (<root>/<user>/<dataset>).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where converted v3.0 datasets are written.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Subset to process, as '<user>/<dataset>' names (default: all).",
    )
    parser.add_argument(
        "--stats-only", action="store_true", help="Print the census and exit."
    )
    parser.add_argument(
        "--force", action="store_true", help="Reconvert even if output exists."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Convert this many sub-datasets in parallel (processes). The "
        "CPU-heavy phase is v2.0 stats synthesis (AV1 frame sampling); "
        "4-8 workers is a good laptop setting.",
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    datasets = discover(source)
    if not datasets:
        raise SystemExit(
            f"no sub-datasets found under {source} (expected <user>/<ds>/meta/info.json)"
        )

    if args.datasets is not None:
        by_name = {d.name: d for d in datasets}
        unknown = [n for n in args.datasets if n not in by_name]
        if unknown:
            raise SystemExit(f"unknown datasets: {unknown}")
        datasets = [by_name[n] for n in args.datasets]

    print_census(datasets, output)
    if args.stats_only:
        return

    manifest_path = output / "conversion_manifest.jsonl"
    outcomes: Counter = Counter()

    todo: list[SubDataset] = []
    for ds in datasets:
        if not args.force and is_converted(output, ds.name):
            outcomes["skipped"] += 1
        elif ds.version not in SUPPORTED_SOURCE_VERSIONS:
            outcomes["unsupported"] += 1
            append_manifest(
                manifest_path,
                ds.name,
                {"status": "unsupported", "source_version": ds.version},
            )
        else:
            todo.append(ds)
    if outcomes["skipped"]:
        print(f"skipping {outcomes['skipped']} already-converted dataset(s)")

    if args.workers <= 1:
        for ds in tqdm(todo, unit="dataset"):
            result = run_safely(ds, output)
            outcomes[result["status"]] += 1
            append_manifest(manifest_path, ds.name, result)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_safely, ds, output): ds for ds in todo}
            for future in tqdm(
                as_completed(futures), total=len(futures), unit="dataset"
            ):
                ds = futures[future]
                result = future.result()
                outcomes[result["status"]] += 1
                append_manifest(manifest_path, ds.name, result)

    staging_root = output / ".staging"
    if staging_root.exists() and not any(staging_root.rglob("*")):
        shutil.rmtree(staging_root)

    print(f"\ndone: {dict(outcomes)}")
    if outcomes["failed"]:
        print(f"failures are quarantined in the manifest: {manifest_path}")


def run_safely(ds: SubDataset, output: Path) -> dict:
    try:
        return convert_one(ds, output)
    except Exception as error:  # noqa: BLE001 - quarantine and continue the sweep
        traceback.print_exc()
        return {
            "status": "failed",
            "source_version": ds.version,
            "error": str(error),
        }


def append_manifest(manifest_path: Path, name: str, result: dict) -> None:
    with manifest_path.open("a") as f:
        f.write(
            json.dumps({"dataset": name, "time": time.strftime("%F %T"), **result})
            + "\n"
        )


if __name__ == "__main__":
    main()
