"""Materialize the rig few-shot nested training subsets (ideas #16,
pre-reg posts/2026-08-05-prereg-rig-fewshot-benchmark.md).

N10 ⊂ N25 ⊂ N45 episodes from the 45 non-holdout episodes of the two
owner rig repos (holdout = the codebase-native split the frozen plan
records: fraction 0.212, split_seed 16 — see rig_fewshot_plan.py),
one SeedSequence(16) shuffle, materialized as derived LeRobot v3
datasets under ``~/datasets/rig_fewshot_v0/n{10,25,45}/fontaine/``.

Materialization contract (from the loader read-path, mapped 2026-08-05):

- data parquet rows filtered to kept episodes, ``episode_index``
  renumbered contiguously 0..n-1 (lerobot indexes meta/episodes
  POSITIONALLY — a gap silently reads a neighbour's video pointers)
  and ``index`` rewritten 0..rows-1 (must be unique); every other
  column — action/state/timestamps and the materialized judge/language
  aux columns — byte-identical to source.
- meta/episodes parquet filtered + renumbered, ``dataset_from/to_index``
  recomputed, data/meta pointer columns set to the written single-file
  layout; per-episode ``stats/*`` and every ``videos/*`` pointer column
  kept verbatim.
- videos HARDLINKED whole, original chunk/file names: the loader
  reaches frames via the episodes-parquet pointer columns, so unkept
  footage in the files is inert — zero re-encode, bit-identical
  pixels (lerobot's delete_episodes would re-extract streams). Every
  pointer-referenced file must exist or lerobot falls through to a
  network snapshot_download on the derived repo id.
- meta/stats.json recomputed: row-derived features exactly from the
  kept rows with numpy (bijou normalizes from action/observation.state
  mean/std/q01/q99 — missing quantiles is a hard SystemExit); image
  features via lerobot aggregate_stats over the kept episodes'
  per-episode stats (unread by bijou, kept for file completeness).
- meta/judgments.json filtered + episode_index REMAPPED (read by
  training whenever --instruction-augment > 0 — the ft protocol's 0.5
  — and keyed by episode; a verbatim copy attaches wrong episodes'
  records after renumbering). judge_annotations.json (the stamp),
  camera_kinds.json, tasks.parquet copied verbatim.
- meta/info.json: total_episodes/total_frames/splits updated.
- meta/source_provenance.json v1 (the bijou.eval.leakage derived-corpus
  contract): every episode mapped to (source_repo_id, source_episode).

Verification (in-run, loud): per-episode lengths + bitwise action/state
vs source re-read from disk, positional episode contiguity, pointer
targets exist, and a full-set stats oracle per source repo (exact
recompute + aggregate_stats vs the shipped stats.json). The leakage
certification runs separately per subset root:

    uv run python -m bijou.eval.leakage \
        --plan plans/rig_fewshot_v0_k4l2.json \
        --panel-data ~/datasets/mcobzarenco/so101_pick_place_v2 \
                     ~/datasets/mcobzarenco/so101_pick_place_clean \
        --train-data ~/datasets/rig_fewshot_v0/n10

Run from the repo root:
    uv run python fontaine/scripts/rig_fewshot_materialize.py
Refuses to overwrite existing subset dirs without --force.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bijou.data import holdout_episodes, repo_id_of

SOURCE_DIRS = (
    Path("~/datasets/mcobzarenco/so101_pick_place_v2").expanduser(),
    Path("~/datasets/mcobzarenco/so101_pick_place_clean").expanduser(),
)
OUT_ROOT = Path("~/datasets/rig_fewshot_v0").expanduser()
DERIVED_USER = "fontaine"
SUFFIX = {"so101_pick_place_v2": "v2", "so101_pick_place_clean": "clean"}
HOLDOUT_FRACTION = 0.212
SPLIT_SEED = 16
SHUFFLE_SEED = 16  # pre-reg: one SeedSequence(16) shuffle, nested prefixes
SUBSET_SIZES = (10, 25, 45)
PROVENANCE_VERSION = 1
# Row-derived features recomputed exactly; the rest (images) aggregate
# from per-episode stats.
ROW_FEATURES = (
    "action",
    "observation.state",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)
QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def exact_feature_stats(values: np.ndarray) -> dict[str, list]:
    """min/max/mean/std/count + quantiles of one row-derived feature,
    computed exactly (the source used histogram-approximate quantiles;
    exact is at worst one bin width away and strictly better)."""
    values = values.reshape(len(values), -1).astype(np.float64)
    stats: dict[str, list] = {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [len(values)],
    }
    for key, q in QUANTILES.items():
        stats[key] = np.quantile(values, q, axis=0).tolist()
    return {k: (v if isinstance(v, list) else [v]) for k, v in stats.items()}


def per_episode_image_stats(
    episodes: pa.Table,
    camera_keys: list[str],
    indices: list[int],
) -> list[dict[str, dict[str, np.ndarray]]]:
    """Per-episode stats dicts for the image features, unflattened from
    the episodes-parquet ``stats/<feature>/<stat>`` columns."""
    rows = episodes.to_pylist()
    stats_list = []
    for index in indices:
        row = rows[index]
        entry: dict[str, dict[str, np.ndarray]] = {}
        for key in camera_keys:
            entry[key] = {
                stat: np.asarray(row[f"stats/{key}/{stat}"], dtype=np.float64)
                for stat in ("min", "max", "mean", "std", "count", *QUANTILES)
            }
        stats_list.append(entry)
    return stats_list


def replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    field_index = table.schema.get_field_index(name)
    array = pa.array(values, type=table.schema.field(field_index).type)
    return table.set_column(field_index, table.schema.field(field_index), array)


def materialize_dataset(
    source_dir: Path,
    out_dir: Path,
    kept: list[int],
    data: pa.Table,
    episodes: pa.Table,
) -> dict[int, int]:
    """Write one derived dataset; returns {source_episode: length}."""
    source_repo_id = repo_id_of(source_dir)
    new_index_of = {ep: i for i, ep in enumerate(kept)}

    # --- data parquet: filter, renumber episode_index + index ---
    mask = np.isin(np.asarray(data.column("episode_index")), kept)
    subset = data.filter(pa.array(mask))
    old_ep = np.asarray(subset.column("episode_index"))
    subset = replace_column(
        subset,
        "episode_index",
        np.array([new_index_of[e] for e in old_ep]),
    )
    subset = replace_column(subset, "index", np.arange(len(subset)))
    data_dir = out_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    pq.write_table(subset, data_dir / "file-000.parquet")

    # --- episodes parquet: filter, renumber, recompute offsets ---
    ep_column = np.asarray(episodes.column("episode_index"))
    ep_mask = np.isin(ep_column, kept)
    ep_subset = episodes.filter(pa.array(ep_mask))
    # filter preserves source order == ascending episode_index == new order
    lengths = np.asarray(ep_subset.column("length"))
    ends = np.cumsum(lengths)
    ep_subset = replace_column(ep_subset, "episode_index", np.arange(len(kept)))
    ep_subset = replace_column(ep_subset, "dataset_from_index", ends - lengths)
    ep_subset = replace_column(ep_subset, "dataset_to_index", ends)
    for name in (
        "data/chunk_index",
        "data/file_index",
        "meta/episodes/chunk_index",
        "meta/episodes/file_index",
    ):
        ep_subset = replace_column(ep_subset, name, np.zeros(len(kept), dtype=np.int64))
    ep_dir = out_dir / "meta" / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True)
    pq.write_table(ep_subset, ep_dir / "file-000.parquet")

    # --- videos: hardlink whole files, original names ---
    for source_file in sorted(source_dir.glob("videos/**/*.mp4")):
        target = out_dir / source_file.relative_to(source_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source_file, target)
        except OSError:
            shutil.copy2(source_file, target)

    # --- meta: info, stats, tasks, stamps, judgments, provenance ---
    info = json.loads((source_dir / "meta" / "info.json").read_text())
    camera_keys = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    info["total_episodes"] = len(kept)
    info["total_frames"] = len(subset)
    info["splits"] = {"train": f"0:{len(kept)}"}
    (out_dir / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    stats = {
        feature: exact_feature_stats(
            np.asarray(subset.column(feature).to_pylist(), dtype=np.float64),
        )
        for feature in ROW_FEATURES
    }
    from lerobot.datasets.compute_stats import aggregate_stats

    kept_positions = [int(np.flatnonzero(ep_column == ep)[0]) for ep in kept]
    image_stats = aggregate_stats(
        per_episode_image_stats(episodes, camera_keys, kept_positions),
    )
    for key, feature_stats in image_stats.items():
        stats[key] = {s: v.tolist() for s, v in feature_stats.items()}
    (out_dir / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))

    for name in ("tasks.parquet", "camera_kinds.json", "judge_annotations.json"):
        shutil.copy2(source_dir / "meta" / name, out_dir / "meta" / name)

    sidecar = json.loads((source_dir / "meta" / "judgments.json").read_text())
    remapped = sorted(
        (
            {**record, "episode_index": new_index_of[record["episode_index"]]}
            for record in sidecar["judgments"]
            if record["episode_index"] in new_index_of
        ),
        key=lambda r: (r["episode_index"], r.get("judged_at", "")),
    )
    (out_dir / "meta" / "judgments.json").write_text(
        json.dumps({"judgments": remapped}, indent=1),
    )

    provenance = {
        "version": PROVENANCE_VERSION,
        "episodes": [
            {
                "episode_index": i,
                "source_repo_id": source_repo_id,
                "source_episode_index": ep,
            }
            for i, ep in enumerate(kept)
        ],
    }
    (out_dir / "meta" / "source_provenance.json").write_text(
        json.dumps(provenance, indent=1),
    )
    return {ep: int(length) for ep, length in zip(kept, lengths, strict=True)}


def verify_dataset(source_dir: Path, out_dir: Path, kept: list[int]) -> int:
    """Re-read everything from disk and compare against source; returns
    the verified frame count. Loud on any mismatch."""
    source = pq.read_table(min(source_dir.glob("data/*/*.parquet")))
    written = pq.read_table(out_dir / "data" / "chunk-000" / "file-000.parquet")
    episodes = pq.read_table(
        out_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    info = json.loads((out_dir / "meta" / "info.json").read_text())

    new_ep = np.asarray(written.column("episode_index"))
    assert list(np.unique(new_ep)) == list(range(len(kept))), "episodes not contiguous"
    assert np.array_equal(np.asarray(written.column("index")), np.arange(len(written)))
    assert info["total_frames"] == len(written)
    assert info["total_episodes"] == len(kept)

    source_ep = np.asarray(source.column("episode_index"))
    for new, old in enumerate(kept):
        source_rows = source.filter(pa.array(source_ep == old))
        written_rows = written.filter(pa.array(new_ep == new))
        assert len(source_rows) == len(written_rows), f"episode {old}: length mismatch"
        for column in (
            "action",
            "observation.state",
            "timestamp",
            "frame_index",
            "task_index",
            "annotation.progress",
        ):
            a = np.asarray(source_rows.column(column).to_pylist(), dtype=np.float64)
            b = np.asarray(written_rows.column(column).to_pylist(), dtype=np.float64)
            assert np.array_equal(a, b, equal_nan=True), f"ep {old} {column} differs"

    # positional contiguity + offsets + every video pointer target exists
    ep_index = np.asarray(episodes.column("episode_index"))
    assert np.array_equal(ep_index, np.arange(len(kept))), (
        "episodes parquet not positional"
    )
    lengths = np.asarray(episodes.column("length"))
    ends = np.cumsum(lengths)
    assert np.array_equal(np.asarray(episodes.column("dataset_to_index")), ends)
    assert np.array_equal(
        np.asarray(episodes.column("dataset_from_index")),
        ends - lengths,
    )
    camera_keys = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    for row in episodes.to_pylist():
        for key in camera_keys:
            path = out_dir / info["video_path"].format(
                video_key=key,
                chunk_index=row[f"videos/{key}/chunk_index"],
                file_index=row[f"videos/{key}/file_index"],
            )
            assert path.exists(), f"missing video {path}"
    return len(written)


def stats_oracle(source_dir: Path) -> None:
    """Full-set recompute vs the shipped stats.json — certifies the
    stats path before any subset number is trusted. Mean/std/min/max
    must agree tightly; quantiles within the source's histogram
    resolution (range/5000 per bin, a few bins of slack)."""
    shipped = json.loads((source_dir / "meta" / "stats.json").read_text())
    data = pq.read_table(min(source_dir.glob("data/*/*.parquet")))
    worst = 0.0
    for feature in ("action", "observation.state"):
        values = np.asarray(data.column(feature).to_pylist(), dtype=np.float64)
        computed = exact_feature_stats(values)
        span = values.max() - values.min()
        for stat in ("min", "max", "mean", "std"):
            delta = np.max(
                np.abs(np.array(computed[stat]) - np.array(shipped[feature][stat])),
            )
            assert delta < 1e-3, f"{source_dir.name} {feature}/{stat}: Δ{delta}"
            worst = max(worst, float(delta))
        for stat in QUANTILES:
            delta = np.max(
                np.abs(np.array(computed[stat]) - np.array(shipped[feature][stat])),
            )
            assert delta < span / 5000 * 10 + 1e-6, (
                f"{source_dir.name} {feature}/{stat}: Δ{delta} vs bin {span / 5000}"
            )
            worst = max(worst, float(delta))
    print(f"  stats oracle {source_dir.name}: PASSED (worst |Δ| {worst:.2e})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    sources: dict[str, dict] = {}
    pool: list[tuple[str, int]] = []
    for source_dir in SOURCE_DIRS:
        repo_id = repo_id_of(source_dir)
        info = json.loads((source_dir / "meta" / "info.json").read_text())
        held = set(
            holdout_episodes(
                repo_id,
                info["total_episodes"],
                HOLDOUT_FRACTION,
                SPLIT_SEED,
            ),
        )
        train = [e for e in range(info["total_episodes"]) if e not in held]
        sources[repo_id] = {
            "dir": source_dir,
            "data": pq.read_table(min(source_dir.glob("data/*/*.parquet"))),
            "episodes": pq.read_table(
                min(source_dir.glob("meta/episodes/*/*.parquet")),
            ),
        }
        pool.extend((repo_id, e) for e in train)
        print(f"{repo_id}: {len(train)} train episodes (holdout {sorted(held)})")
        stats_oracle(source_dir)
    assert len(pool) == 45, f"train pool {len(pool)} != pre-registered 45"

    order = np.random.default_rng(np.random.SeedSequence(SHUFFLE_SEED)).permutation(
        len(pool),
    )
    manifest: dict[str, dict] = {}
    for size in SUBSET_SIZES:
        chosen = [pool[i] for i in sorted(order[:size])]  # nested by construction
        subset_root = OUT_ROOT / f"n{size}"
        if subset_root.exists():
            if not args.force:
                sys.exit(f"{subset_root} exists (--force to rebuild)")
            shutil.rmtree(subset_root)
        subset_manifest = {}
        for repo_id, source in sources.items():
            kept = sorted(e for r, e in chosen if r == repo_id)
            if not kept:
                continue
            name = f"so101_fewshot_n{size}_{SUFFIX[source['dir'].name]}"
            out_dir = subset_root / DERIVED_USER / name
            lengths = materialize_dataset(
                source["dir"],
                out_dir,
                kept,
                source["data"],
                source["episodes"],
            )
            frames = verify_dataset(source["dir"], out_dir, kept)
            subset_manifest[f"{DERIVED_USER}/{name}"] = {
                "source_repo_id": repo_id,
                "episodes": {str(e): lengths[e] for e in kept},
                "frames": frames,
            }
            print(
                f"n{size}: {DERIVED_USER}/{name} — {len(kept)} eps / {frames} frames "
                f"from {repo_id} — VERIFIED",
            )
        total = sum(d["frames"] for d in subset_manifest.values())
        eps = sum(len(d["episodes"]) for d in subset_manifest.values())
        assert eps == size, f"n{size}: materialized {eps} episodes"
        manifest[f"n{size}"] = {"total_frames": total, "datasets": subset_manifest}
        print(f"n{size}: TOTAL {eps} episodes / {total} frames")

    manifest_path = OUT_ROOT / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "holdout_fraction": HOLDOUT_FRACTION,
                "split_seed": SPLIT_SEED,
                "shuffle_seed": SHUFFLE_SEED,
                "subsets": manifest,
            },
            indent=1,
        ),
    )
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
