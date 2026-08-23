#!/usr/bin/env python3
"""Materialize the carrier-hunt rung-3 branch-B content-bisection subsets
of so101_pick_place_clean (pre-reg
posts/2026-08-22-prereg-carrier-hunt-rung3.md; fires iff the rung-2
ch0fix verdict lands <=10/100 — the named-suspect list exhausted, the
carrier is content-level).

The treatment is SUBSET MEMBERSHIP: every kept frame's every data
column is byte-identical to source (no value edits of any kind).
Episodes are re-indexed 0..n-1 in ascending original order; videos are
hardlinked whole (the loader reaches frames via the episodes-parquet
pointer columns, so unkept footage is inert — bit-identical pixels).
Subset mechanics are IMPORTED from rig_fewshot_materialize.py (the
loader-read-path contract mapped 2026-08-05, verified on the n10/25/45
subsets): data/episodes filter + renumber, offsets, stats.json exact
recompute + image aggregate, judgments remap, provenance.

Pinned cells (dataset names fixed by holdout-draw search, the gripfix
Amendment-1 class pre-applied; ``holdout_episodes`` is a pure function
of the repo name — the DECOY design puts democlean's own never-trained
holdout, original episode 2, in each subset so the mandatory holdout
lands on IT and the cell trains its full intended half):

  ep015_c — originals [0,1,2,5], draw (2,) = original ep 2;
            trains {0,1,5} (1,504 frames). THE first cell: drops
            {3,4,6}, the half holding the two most marginal-anomalous
            episodes (ep 3 KS 0.469, ep 4 mean −7.1).
  ep346_a — originals [2,3,4,6], draw (0,) = original ep 2;
            trains {3,4,6} (1,522 frames). The registered follow-up
            (complement cell) on EITHER verdict.

Oracles (all hard-fail):
  1. episode set — exactly the pinned originals, ascending order,
     re-indexed contiguously (rig_fewshot verify + explicit here).
  2. byte-identity — every kept frame's every data column bitwise
     equal to the source rows (except the renumbered episode_index /
     index bookkeeping columns, checked for exact renumber form).
  3. counts — per-episode frame lengths match the banked basis table
     (reports/analysis__carrier_rung3_basis.json).
  4. holdout draw — holdout_episodes(dst_repo_id, n, 0.1, 0) equals
     the pinned draw AND maps back to original episode 2 (the decoy
     invariant), so the trained set is exactly the pinned half.
  5. repeat glob — the dst repo id matches the launcher's
     ``mcobzarenco/so101_pick_place*=4`` dataset-repeat glob.
  6. source stats oracle — rig_fewshot's full-set recompute vs the
     shipped stats.json (certifies the stats path before any subset
     number is trusted).

Usage (one cell per invocation; the exec session builds the selected
cell only):
  uv run python fontaine/scripts/make_clean_subset_dataset.py \
      --cell ep015_c [--force]
  uv run python fontaine/scripts/make_clean_subset_dataset.py \
      --cell ep346_a [--force]
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bijou.data import holdout_episodes, repo_id_of

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rig_fewshot_materialize import (
    materialize_dataset,
    stats_oracle,
    verify_dataset,
)

SRC = Path("~/datasets/mcobzarenco/so101_pick_place_clean").expanduser()
DST_ROOT = Path("~/datasets/mcobzarenco").expanduser()


@dataclass(frozen=True)
class Cell:
    dst_name: str
    kept: list[int]
    draw_expect: tuple[int, ...]
    trains: set[int]


# Frozen per the pre-reg — episode lists, draws, and the trained halves.
CELLS = {
    "ep015_c": Cell(
        dst_name="so101_pick_place_clean_ep015_c",
        kept=[0, 1, 2, 5],
        draw_expect=(2,),
        trains={0, 1, 5},
    ),
    "ep346_a": Cell(
        dst_name="so101_pick_place_clean_ep346_a",
        kept=[2, 3, 4, 6],
        draw_expect=(0,),
        trains={3, 4, 6},
    ),
}
DECOY = 2  # democlean's own holdout episode — must receive the draw
# Banked basis table (reports/analysis__carrier_rung3_basis.json).
EP_LENGTHS = {0: 511, 1: 509, 2: 373, 3: 380, 4: 694, 5: 484, 6: 448}
REPEAT_GLOB = "mcobzarenco/so101_pick_place*"  # the launcher's ×4 glob
HOLDOUT_FRACTION = 0.1
HOLDOUT_SEED = 0
# Renumbered bookkeeping columns — everything else must be bitwise.
RENUMBERED = ("episode_index", "index")


def fail(msg: str) -> NoReturn:
    print(f"ORACLE FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def bitwise_equal(a: pa.Table, b: pa.Table, name: str) -> bool:
    """Bitwise column compare (NaN-safe) via float bit views."""
    ca, cb = a.column(name).combine_chunks(), b.column(name).combine_chunks()
    t = ca.type
    if pa.types.is_fixed_size_list(t) and pa.types.is_floating(t.value_type):
        view = {4: np.uint32, 8: np.uint64}[t.value_type.bit_width // 8]
        return np.array_equal(
            np.asarray(ca.values).view(view),
            np.asarray(cb.values).view(view),
        )
    if pa.types.is_floating(t):
        view = {4: np.uint32, 8: np.uint64}[t.bit_width // 8]
        return np.array_equal(
            ca.to_numpy(zero_copy_only=False).view(view),
            cb.to_numpy(zero_copy_only=False).view(view),
        )
    return ca.equals(cb)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True, choices=sorted(CELLS))
    ap.add_argument("--force", action="store_true", help="overwrite an existing dst")
    args = ap.parse_args()

    spec = CELLS[args.cell]
    kept = spec.kept
    dst = DST_ROOT / spec.dst_name

    if not (SRC / "meta" / "info.json").exists():
        fail(f"{SRC} is not a LeRobot dataset (no meta/info.json)")
    if dst.exists():
        if not args.force:
            fail(f"{dst} exists — pass --force to overwrite")
        shutil.rmtree(dst)

    if kept != sorted(kept):
        fail(f"pinned episode list {kept} not ascending")
    if DECOY not in kept:
        fail(f"decoy episode {DECOY} missing from {kept}")

    # --- oracle 6 first: certify the stats path on the source ---
    stats_oracle(SRC)

    data = pq.read_table(min(SRC.glob("data/*/*.parquet")))
    episodes = pq.read_table(min(SRC.glob("meta/episodes/*/*.parquet")))

    lengths = materialize_dataset(SRC, dst, kept, data, episodes)
    frames = verify_dataset(SRC, dst, kept)

    # --- oracle 3: per-episode lengths vs the banked basis table ---
    for ep in kept:
        if lengths[ep] != EP_LENGTHS[ep]:
            fail(
                f"episode {ep} length {lengths[ep]} != basis table {EP_LENGTHS[ep]}",
            )
    if frames != sum(EP_LENGTHS[ep] for ep in kept):
        fail(f"total frames {frames} != basis sum")

    # --- oracle 2: EVERY data column bitwise vs source (beyond the
    # rig_fewshot verify's column subset), renumbered columns excepted ---
    written = pq.read_table(dst / "data" / "chunk-000" / "file-000.parquet")
    src_ep = np.asarray(data.column("episode_index"))
    src_subset = data.filter(pa.array(np.isin(src_ep, kept)))
    if set(written.schema.names) != set(data.schema.names):
        fail("data schema names differ from source")
    for name in written.schema.names:
        if name in RENUMBERED:
            continue
        if not bitwise_equal(src_subset, written, name):
            fail(f"data column {name} not bitwise-identical to kept source rows")
    new_ep = np.asarray(written.column("episode_index"))
    expect_ep = np.concatenate(
        [np.full(EP_LENGTHS[ep], i) for i, ep in enumerate(kept)],
    )
    if not np.array_equal(new_ep, expect_ep):
        fail("episode_index renumbering not the ascending-contiguous form")
    if not np.array_equal(np.asarray(written.column("index")), np.arange(frames)):
        fail("index column not 0..frames-1")

    # --- oracle 4: holdout draw + decoy invariant ---
    repo_id = repo_id_of(dst)
    draw = holdout_episodes(repo_id, len(kept), HOLDOUT_FRACTION, HOLDOUT_SEED)
    if draw != spec.draw_expect:
        fail(
            f"holdout_episodes({repo_id!r}, {len(kept)}, {HOLDOUT_FRACTION}, "
            f"{HOLDOUT_SEED}) = {draw} != pinned {spec.draw_expect}",
        )
    if kept[draw[0]] != DECOY:
        fail(f"draw {draw} maps to original ep {kept[draw[0]]} != decoy {DECOY}")
    trained = {ep for i, ep in enumerate(kept) if i not in draw}
    if trained != spec.trains:
        fail(f"trained set {sorted(trained)} != pinned {sorted(spec.trains)}")

    # --- oracle 5: the launcher's dataset-repeat glob must match ---
    if not fnmatch.fnmatch(repo_id, REPEAT_GLOB):
        fail(f"dst repo id {repo_id!r} does not match repeat glob {REPEAT_GLOB!r}")

    print(
        f"OK: {dst}\n"
        f"  cell              {args.cell} (branch B content bisection)\n"
        f"  episodes          originals {kept} -> re-indexed 0..{len(kept) - 1}\n"
        f"  frames            {frames} (per-episode lengths = basis table)\n"
        f"  holdout draw      {draw} = original ep {DECOY} (decoy) -> trains "
        f"{sorted(spec.trains)}\n"
        f"  byte-identity     every data column bitwise vs source "
        f"(episode_index/index renumbered as pinned)\n"
        f"  repeat glob       {REPEAT_GLOB!r} matches {repo_id!r}\n"
        f"  all oracles green",
    )


if __name__ == "__main__":
    main()
