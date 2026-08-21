#!/usr/bin/env python3
"""Materialize so101_pick_place_clean_gripfix_a — the gripper-carrier
isolation transform (pre-reg 2026-08-21, posts/2026-08-21-prereg-clean-
gripper-carrier.md). The ``_a`` suffix is pre-reg Amendment 1: the
episode holdout keys on repo_id, and ``_a`` makes the draw ``(2,)`` —
the clean-side train split stays episode-identical to democlean's.

One frozen edit: channel 5 (gripper.pos) of BOTH the ``action`` and
``observation.state`` columns is scaled by SCALE = 41.69 / 32.3 (the
banked demos open-command / clean raw max — pinned, never re-estimated).
Zero fixed point preserved (scalar multiply). Every other channel, every
other column, every video and meta file is carried over unchanged; the
ch5-derived summary stats (meta/stats.json + the per-episode stats
columns in meta/episodes) are scaled by the same scalar so the on-disk
metadata stays self-consistent (training recomputes its own pdnorm row
via --recompute-stats regardless).

Oracles (all hard-fail):
  1. no-op guard — refuses if the SOURCE action ch5 max already
     exceeds 40 (i.e. the set was already remapped).
  2. exact transform — dst ch5 == (src ch5 as float64 * SCALE) cast
     back to float32, bitwise, for action AND observation.state.
  3. byte-equal elsewhere — every non-target data column equals the
     source array; the non-ch5 channels of action/state are bitwise
     identical; untouched files (videos, tasks, info.json, ...) match
     by sha256.
  4. counts — episode count, frame count, per-episode lengths all
     identical.
  5. landing zone — transformed action ch5 max within 0.05 of 41.69.

Usage:
  uv run python fontaine/scripts/make_clean_gripfix_dataset.py \
      [--src ~/datasets/mcobzarenco/so101_pick_place_clean] \
      [--dst ~/datasets/mcobzarenco/so101_pick_place_clean_gripfix_a]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCALE = 41.69 / 32.3  # frozen: demos open command / clean raw ch5 max
CH = 5  # gripper.pos
TARGET_COLS = ("action", "observation.state")
STAT_KEYS = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")
NOOP_GUARD_MAX = 40.0
LANDING = 41.69
LANDING_TOL = 0.05


def fail(msg: str) -> None:
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


def scaled_ch(src: np.ndarray) -> np.ndarray:
    """The frozen transform: float64 multiply, cast back to float32."""
    return (src.astype(np.float64) * SCALE).astype(np.float32)


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
        mat[:, CH] = scaled_ch(mat[:, CH])
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
    """Per-episode stats are list<double> — scale ch5 in float64, no cast."""
    table = pq.read_table(src_file)
    out = table
    for feat in TARGET_COLS:
        for key in STAT_KEYS:
            name = f"stats/{feat}/{key}"
            if name not in table.schema.names:
                fail(f"expected episode-stats column {name} missing")
            col = table.column(name).combine_chunks()
            rows = col.to_pylist()
            for row in rows:
                if row is None or len(row) <= CH:
                    fail(f"{name} row too short for ch{CH}")
                row[CH] = row[CH] * SCALE
            arr = pa.array(rows, type=col.type)
            out = out.set_column(out.schema.get_field_index(name), out.field(name), arr)
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, dst_file)


def transform_stats_json(src_file: Path, dst_file: Path) -> None:
    stats = json.loads(src_file.read_text())
    for feat in TARGET_COLS:
        for key in STAT_KEYS:
            vals = stats[feat][key]
            vals[CH] = float(np.float32(np.float64(vals[CH]) * SCALE))
    dst_file.write_text(json.dumps(stats, indent=4) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="~/datasets/mcobzarenco/so101_pick_place_clean")
    ap.add_argument(
        "--dst",
        default="~/datasets/mcobzarenco/so101_pick_place_clean_gripfix_a",
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
    src_max = float(src_action[:, CH].max())
    if src_max > NOOP_GUARD_MAX:
        fail(
            f"no-op guard: source action ch{CH} max {src_max:.4f} > "
            f"{NOOP_GUARD_MAX} — source already remapped, refusing",
        )

    # --- copy everything except .cache, then rewrite the three targets ---
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".cache"))
    transform_data_parquet(src / data_rel, dst / data_rel)
    transform_episode_stats(src / episodes_rel, dst / episodes_rel)
    transform_stats_json(src / stats_rel, dst / stats_rel)

    # --- oracle 2 + 3 (data parquet): exact ch5, bitwise elsewhere ---
    dst_data = pq.read_table(dst / data_rel)
    if dst_data.num_rows != src_data.num_rows:
        fail(f"frame count {dst_data.num_rows} != source {src_data.num_rows}")
    for name in TARGET_COLS:
        s, d = col_matrix(src_data, name), col_matrix(dst_data, name)
        expect = scaled_ch(s[:, CH])
        if not np.array_equal(d[:, CH].view(np.uint32), expect.view(np.uint32)):
            fail(f"{name} ch{CH} not bitwise-equal to source x {SCALE!r}")
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

    # --- oracle 5: landing zone ---
    dst_max = float(col_matrix(dst_data, "action")[:, CH].max())
    if abs(dst_max - LANDING) > LANDING_TOL:
        fail(
            f"transformed action ch{CH} max {dst_max:.4f} not within "
            f"{LANDING_TOL} of {LANDING}",
        )

    state_max = float(col_matrix(dst_data, "observation.state")[:, CH].max())
    print(
        f"OK: {dst}\n"
        f"  scale             {SCALE!r} (= 41.69 / 32.3)\n"
        f"  frames            {dst_data.num_rows} (source-equal)\n"
        f"  episodes          {dst_ep.num_rows} (source-equal)\n"
        f"  action  ch{CH} max   {src_max:.6f} -> {dst_max:.6f}\n"
        f"  state   ch{CH} max   -> {state_max:.6f}\n"
        f"  untouched files   {len(src_files) - len(rewritten)} sha256-verified\n"
        f"  all oracles green",
    )


if __name__ == "__main__":
    main()
