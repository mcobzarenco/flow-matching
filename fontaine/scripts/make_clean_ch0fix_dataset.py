#!/usr/bin/env python3
"""Materialize so101_pick_place_clean_ch0fix_n — the shoulder-pan (ch0)
distribution isolation transform (pre-reg 2026-08-22, posts/2026-08-22-
prereg-clean-ch0-affine.md). The ``_n`` suffix is the holdout-draw choice
(gripfix Amendment-1 lesson, pre-applied): the episode holdout keys on
repo_id, and ``_n`` makes the draw ``(2,)`` — the clean-side train split
stays episode-identical to democlean's.

One frozen edit: channel 0 (shoulder_pan.pos) of BOTH the ``action`` and
``observation.state`` columns is mapped through the moment-matched affine

    x' = TARGET + (x - CENTER) * SCALE
       = 0.0923439813196304 + (x - 1.481974338423806) * 2.755193138766973

(constants pinned verbatim to the banked constant-freeze read,
reports/analysis__ch0_shift_constant_read.json — demos' action-column
moments; one transform for both columns; float64 compute, cast back to
float32). Every other channel, every other column, every video and meta
file is carried over unchanged; the ch0-derived summary stats
(meta/stats.json + the per-episode stats columns in meta/episodes) are
mapped through the same affine — order statistics (min/max/quantiles)
and the mean via the affine itself, std via a pure x SCALE (the slope is
positive, so the on-disk metadata stays self-consistent; training
recomputes its own pdnorm row via --recompute-stats regardless).

Oracles (all hard-fail):
  1. no-op guard — refuses if the SOURCE action ch0 std already exceeds
     20 (i.e. the source is not spread-compressed / already transformed;
     the banked clean std is 10.16 vs demos 27.99).
  2. exact transform — dst ch0 == affine(src ch0 as float64) cast back
     to float32, bitwise, for action AND observation.state.
  3. byte-equal elsewhere — every non-target data column equals the
     source array; the non-ch0 channels of action/state are bitwise
     identical; untouched files (videos, tasks, info.json, ...) match
     by sha256.
  4. counts — episode count, frame count, per-episode lengths all
     identical.
  5. support containment — transformed ch0 (action and state) stays
     inside demos' observed ch0 support [-110.0, +79.6] (the edit must
     not manufacture out-of-support joint values; banked expectation
     ~[-72.9, +50.6]).
  6. holdout draw — holdout_episodes(repo_id, n_episodes, 0.1, 0) for
     the dst repo id is exactly (2,), the same 373-frame episode
     democlean holds out.

Usage:
  uv run python fontaine/scripts/make_clean_ch0fix_dataset.py \
      [--src ~/datasets/mcobzarenco/so101_pick_place_clean] \
      [--dst ~/datasets/mcobzarenco/so101_pick_place_clean_ch0fix_n]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import NoReturn

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bijou.data import holdout_episodes, repo_id_of

CENTER = 1.481974338423806  # frozen: clean action ch0 mean (banked read)
SCALE = 2.755193138766973  # frozen: demos/clean action ch0 std ratio
TARGET = 0.0923439813196304  # frozen: demos action ch0 mean
CH = 0  # shoulder_pan.pos
TARGET_COLS = ("action", "observation.state")
AFFINE_KEYS = ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99")
STD_KEYS = ("std",)
NOOP_GUARD_STD = 20.0
SUPPORT = (-110.0, 79.6)  # demos' observed ch0 support (banked read)
HOLDOUT_EXPECT = (2,)
HOLDOUT_FRACTION = 0.1
HOLDOUT_SEED = 0


def fail(msg: str) -> NoReturn:
    print(f"ORACLE FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def col_matrix(table: pa.Table, name: str) -> np.ndarray:
    """A (n_rows, width) float32 view of a fixed_size_list column."""
    col = table.column(name).combine_chunks()
    if col.null_count:
        fail(f"{name} has nulls — transform not specified for nulls")
    width = col.type.list_size
    return np.asarray(col.values).reshape(-1, width)


def affine_ch(src: np.ndarray) -> np.ndarray:
    """The frozen transform: float64 affine, cast back to float32."""
    return (TARGET + (src.astype(np.float64) - CENTER) * SCALE).astype(np.float32)


def columns_equal(a: pa.ChunkedArray, b: pa.ChunkedArray) -> bool:
    """Arrow ``equals`` treats NaN != NaN; float columns compare bitwise."""
    a, b = a.combine_chunks(), b.combine_chunks()
    t = a.type
    if pa.types.is_floating(t):
        if not a.is_null().equals(b.is_null()):
            return False
        view = {4: np.uint32, 8: np.uint64}[t.bit_width // 8]
        an = np.nan_to_num(a.to_numpy(zero_copy_only=False), nan=0.0)
        bn = np.nan_to_num(b.to_numpy(zero_copy_only=False), nan=0.0)
        nan_a = np.isnan(a.to_numpy(zero_copy_only=False))
        nan_b = np.isnan(b.to_numpy(zero_copy_only=False))
        return bool(
            np.array_equal(nan_a, nan_b)
            and np.array_equal(an.view(view), bn.view(view)),
        )
    if pa.types.is_fixed_size_list(t) and pa.types.is_floating(t.value_type):
        if not a.is_null().equals(b.is_null()):
            return False
        view = {4: np.uint32, 8: np.uint64}[t.value_type.bit_width // 8]
        an, bn = np.asarray(a.values), np.asarray(b.values)
        nan_a, nan_b = np.isnan(an), np.isnan(bn)
        return bool(
            np.array_equal(nan_a, nan_b)
            and np.array_equal(
                np.nan_to_num(an, nan=0.0).view(view),
                np.nan_to_num(bn, nan=0.0).view(view),
            ),
        )
    return a.equals(b)


def transform_data_parquet(src_file: Path, dst_file: Path) -> None:
    table = pq.read_table(src_file)
    arrays = {}
    for name in TARGET_COLS:
        mat = col_matrix(table, name).copy()
        mat[:, CH] = affine_ch(mat[:, CH])
        arrays[name] = pa.FixedSizeListArray.from_arrays(
            pa.array(mat.reshape(-1), type=pa.float32()),
            mat.shape[1],
        )
    out = table
    for name, arr in arrays.items():
        out = out.set_column(out.schema.get_field_index(name), out.field(name), arr)
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, dst_file)


def transform_episode_stats(src_file: Path, dst_file: Path) -> None:
    """Per-episode stats are list<double> — transform ch0 in float64, no cast."""
    table = pq.read_table(src_file)
    out = table
    for feat in TARGET_COLS:
        for key in AFFINE_KEYS + STD_KEYS:
            name = f"stats/{feat}/{key}"
            if name not in table.schema.names:
                fail(f"expected episode-stats column {name} missing")
            col = table.column(name).combine_chunks()
            rows = col.to_pylist()
            for row in rows:
                if row is None or len(row) <= CH:
                    fail(f"{name} row too short for ch{CH}")
                if key in STD_KEYS:
                    row[CH] = row[CH] * SCALE
                else:
                    row[CH] = TARGET + (row[CH] - CENTER) * SCALE
            arr = pa.array(rows, type=col.type)
            out = out.set_column(out.schema.get_field_index(name), out.field(name), arr)
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, dst_file)


def transform_stats_json(src_file: Path, dst_file: Path) -> None:
    stats = json.loads(src_file.read_text())
    for feat in TARGET_COLS:
        for key in AFFINE_KEYS:
            vals = stats[feat][key]
            vals[CH] = float(
                np.float32(TARGET + (np.float64(vals[CH]) - CENTER) * SCALE),
            )
        for key in STD_KEYS:
            vals = stats[feat][key]
            vals[CH] = float(np.float32(np.float64(vals[CH]) * SCALE))
    dst_file.write_text(json.dumps(stats, indent=4) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="~/datasets/mcobzarenco/so101_pick_place_clean")
    ap.add_argument(
        "--dst",
        default="~/datasets/mcobzarenco/so101_pick_place_clean_ch0fix_n",
    )
    ap.add_argument("--force", action="store_true", help="overwrite an existing dst")
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    dst = Path(args.dst).expanduser()
    if not (src / "meta" / "info.json").exists():
        fail(f"{src} is not a LeRobot dataset (no meta/info.json)")
    if dst.exists():
        if not args.force:
            fail(f"{dst} exists — pass --force to overwrite")
        shutil.rmtree(dst)

    data_rel = "data/chunk-000/file-000.parquet"
    episodes_rel = "meta/episodes/chunk-000/file-000.parquet"
    stats_rel = "meta/stats.json"
    rewritten = {data_rel, episodes_rel, stats_rel}

    # --- oracle 1: no-op guard, on the SOURCE, before any writes ---
    src_data = pq.read_table(src / data_rel)
    src_action = col_matrix(src_data, "action")
    src_std = float(src_action[:, CH].astype(np.float64).std())
    if src_std > NOOP_GUARD_STD:
        fail(
            f"no-op guard: source action ch{CH} std {src_std:.4f} > "
            f"{NOOP_GUARD_STD} — source not spread-compressed "
            f"(already transformed?), refusing",
        )

    # --- copy everything except .cache, then rewrite the three targets ---
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".cache"))
    transform_data_parquet(src / data_rel, dst / data_rel)
    transform_episode_stats(src / episodes_rel, dst / episodes_rel)
    transform_stats_json(src / stats_rel, dst / stats_rel)

    # --- oracle 2 + 3 (data parquet): exact ch0 affine, bitwise elsewhere ---
    dst_data = pq.read_table(dst / data_rel)
    if dst_data.num_rows != src_data.num_rows:
        fail(f"frame count {dst_data.num_rows} != source {src_data.num_rows}")
    for name in TARGET_COLS:
        s, d = col_matrix(src_data, name), col_matrix(dst_data, name)
        expect = affine_ch(s[:, CH])
        if not np.array_equal(d[:, CH].view(np.uint32), expect.view(np.uint32)):
            fail(f"{name} ch{CH} not bitwise-equal to the frozen affine of source")
        other = [c for c in range(s.shape[1]) if c != CH]
        if not np.array_equal(s[:, other].view(np.uint32), d[:, other].view(np.uint32)):
            fail(f"{name} non-ch{CH} channels differ from source")
    for name in src_data.schema.names:
        if name in TARGET_COLS:
            continue
        if not columns_equal(src_data.column(name), dst_data.column(name)):
            fail(f"data column {name} differs from source")

    # --- oracle 3 (untouched files): sha256 match ---
    src_files = {
        p.relative_to(src).as_posix(): p
        for p in src.rglob("*")
        if p.is_file() and ".cache" not in p.parts
    }
    dst_files = {
        p.relative_to(dst).as_posix(): p for p in dst.rglob("*") if p.is_file()
    }
    if set(src_files) != set(dst_files):
        fail(
            f"file sets differ: only-src {sorted(set(src_files) - set(dst_files))}, "
            f"only-dst {sorted(set(dst_files) - set(src_files))}",
        )
    for rel, sp in sorted(src_files.items()):
        if rel in rewritten:
            continue
        if sha256(sp) != sha256(dst_files[rel]):
            fail(f"untouched file {rel} differs (sha256)")

    # --- oracle 4: episode counts + lengths ---
    src_ep = pq.read_table(src / episodes_rel)
    dst_ep = pq.read_table(dst / episodes_rel)
    if src_ep.num_rows != dst_ep.num_rows:
        fail(f"episode count {dst_ep.num_rows} != source {src_ep.num_rows}")
    for name in ("episode_index", "length"):
        if not src_ep.column(name).equals(dst_ep.column(name)):
            fail(f"episodes column {name} differs from source")

    # --- oracle 5: support containment ---
    ranges = {}
    for name in TARGET_COLS:
        ch = col_matrix(dst_data, name)[:, CH]
        lo, hi = float(ch.min()), float(ch.max())
        ranges[name] = (lo, hi)
        if lo < SUPPORT[0] or hi > SUPPORT[1]:
            fail(
                f"{name} ch{CH} transformed range [{lo:.2f}, {hi:.2f}] leaves "
                f"demos' observed support [{SUPPORT[0]}, {SUPPORT[1]}]",
            )

    # --- oracle 6: holdout draw for the dst repo id ---
    repo_id = repo_id_of(dst)
    draw = holdout_episodes(repo_id, dst_ep.num_rows, HOLDOUT_FRACTION, HOLDOUT_SEED)
    if draw != HOLDOUT_EXPECT:
        fail(
            f"holdout_episodes({repo_id!r}, {dst_ep.num_rows}, "
            f"{HOLDOUT_FRACTION}, {HOLDOUT_SEED}) = {draw} != {HOLDOUT_EXPECT}",
        )

    dst_mean = float(col_matrix(dst_data, "action")[:, CH].astype(np.float64).mean())
    dst_std = float(col_matrix(dst_data, "action")[:, CH].astype(np.float64).std())
    print(
        f"OK: {dst}\n"
        f"  affine            x' = {TARGET!r} + (x - {CENTER!r}) * {SCALE!r}\n"
        f"  frames            {dst_data.num_rows} (source-equal)\n"
        f"  episodes          {dst_ep.num_rows} (source-equal)\n"
        f"  action ch{CH}        mean {dst_mean:.4f} std {dst_std:.4f} "
        f"(targets {TARGET:.4f} / ~{27.99})\n"
        f"  action ch{CH} range  [{ranges['action'][0]:.2f}, {ranges['action'][1]:.2f}]"
        f" within support [{SUPPORT[0]}, {SUPPORT[1]}]\n"
        f"  state  ch{CH} range  [{ranges['observation.state'][0]:.2f}, "
        f"{ranges['observation.state'][1]:.2f}]\n"
        f"  holdout draw      {draw} (episode-identical train split to democlean)\n"
        f"  untouched files   {len(src_files) - len(rewritten)} sha256-verified\n"
        f"  all oracles green",
    )


if __name__ == "__main__":
    main()
