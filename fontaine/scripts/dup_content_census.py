"""Duplicate-content census over curated_v0 — ideas #18.7, declared before the data is read.

The deep-dive finding (posts/2026-08-05-bijou-deep-dive.md, finding 7):
cross-repo duplicate content is never fingerprinted — `data.py` dedups
exact repo ids only (data.py:663-670), while the holdout split is a pure
function of (repo_id, num_episodes, fraction, split_seed)
(data.py:339-358). A community fork (same recordings, different repo id)
therefore gets an UNCORRELATED holdout draw: an episode held out in repo
X can sit in repo Y's train side. Every fine holdout delta (the box-batch
0.15 band, the E4B adopt band) silently assumes this channel is empty.
This census measures it.

READS (frozen here, before any fingerprint is computed):

  R1  Cross-repo duplicate clusters, exact full tier: episodes grouped by
      (n_frames, blake2b(action bytes), blake2b(state bytes)) spanning
      >= 2 repos. Count of clusters / episodes / frames involved.
  R2  PRIMARY — holdout->train leakage under the panel convention
      (fps=30, camera_counts {1,2}, holdout 0.1, split_seed 0, the box
      launchers' and panel's shared filters): holdout episodes whose
      exact-full duplicate sits in the TRAIN side of any selected repo
      (cross-repo, or same-repo across the split). Count + full list.
  R3  Panel impact: rows of plans/holdout_curated_v0_k4l2.json (core and
      labeled separately) whose (repo_id, episode_index) is R2-leaked,
      as a count and a fraction of the 17,204-core / 25,800-row panel.
  R4  Intra-repo duplicate episodes (double-weighting, no split
      crossing): count + list.
  Secondary tiers, same reads: quantized-full (actions+state rounded to
  1e-3 — catches dtype/precision re-encodes) and action-only-exact
  (state ignored — upper bound; constant-action twins can false-positive
  here, so it never feeds R2's headline).

DECISION RELEVANCE (not a decision rule — this is a census): R3 > 0
means leaked panel frames get flagged in the box results post and the
anchors carry a stated caveat; R3 = 0 certifies the channel empty for
the current panel.

VALIDATION (all must pass before the real reads are printed):
  * --oracle: synthetic-array suite through the same fingerprint +
    analyze code paths — planted cross-repo exact dup crossing the
    split (must leak), same dup landing train->train (must NOT leak),
    intra-repo split-crossing dup (must leak, tagged intra), float64
    recast (quantized tier only), +1e-2 noise copy (no tier), constant
    -action/different-state pair (action-only tier only), single
    -episode repo as the train-side donor (must leak).
  * Real-data structural asserts: every plan repo's episode set ==
    holdout_episodes() re-derived here (proves the split mirror on all
    ~800 repos); plan core+labeled totals == 17,204/8,596 == the banked
    npz's 25,800 rows; selected-repo count printed against the
    2026-08-03 measured 878/981 (camera filter alone) as a sanity band;
    info.json total_episodes == parquet distinct episode count; episode
    ids contiguous 0..n-1; episode rows grouped ascending.
  * Hash-collision guard: up to 20 flagged pairs re-loaded and compared
    with np.array_equal — a blake2b collision or a bug fails loud.

Usage:
  uv run python fontaine/scripts/dup_content_census.py --oracle
  uv run python fontaine/scripts/dup_content_census.py \
      --data ~/datasets/mcobzarenco/community_curated_v0 \
      --plan plans/holdout_curated_v0_k4l2.json \
      --npz reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2.npz \
      --output-json ~/dup_census_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bijou.data import (
    DatasetInfo,
    discover_datasets,
    holdout_episodes,
    repo_id_of,
)

PANEL_FPS = (30.0,)
PANEL_CAMERA_COUNTS = (1, 2)
PANEL_HOLDOUT_FRACTION = 0.1
PANEL_SPLIT_SEED = 0
QUANTUM = 1e-3


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    repo_id: str
    episode: int
    n_frames: int
    action_exact: str
    state_exact: str
    action_quant: str
    state_quant: str


def _digest(prefix: bytes, arr: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(prefix)
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def fingerprint_episode(
    action: np.ndarray,
    state: np.ndarray,
) -> tuple[str, str, str, str]:
    """(action_exact, state_exact, action_quant, state_quant).

    Exact tier hashes the float32 bytes as stored. Quantized tier hashes
    int64(round(x / QUANTUM)) — canonical across float dtype/precision
    re-encodes, still exact-match semantics after rounding (a fork that
    resampled or renoised the stream is out of scope for this census
    and is stated as such in the report).
    """
    a32 = np.ascontiguousarray(action, dtype=np.float32)
    s32 = np.ascontiguousarray(state, dtype=np.float32)
    aq = np.round(np.asarray(action, dtype=np.float64) / QUANTUM).astype(np.int64)
    sq = np.round(np.asarray(state, dtype=np.float64) / QUANTUM).astype(np.int64)
    return (
        _digest(b"a32", a32),
        _digest(b"s32", s32),
        _digest(b"aq", aq),
        _digest(b"sq", sq),
    )


def read_repo_episodes(
    repo_dir: Path,
) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], list[str]]:
    """[(episode_id, action array, state array)] in stored row order,
    plus structural warnings. Fails loud on ungrouped episode rows."""
    import pyarrow.parquet as pq

    files = sorted(repo_dir.glob("data/chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"{repo_dir}: no data parquet")
    tables = [
        pq.read_table(f, columns=["action", "observation.state", "episode_index"])
        for f in files
    ]
    import pyarrow as pa

    table = pa.concat_tables(tables)
    episode_column = np.asarray(table["episode_index"])
    ids, starts = np.unique(episode_column, return_index=True)
    if not bool(np.all(np.diff(starts) > 0)):
        raise AssertionError(
            f"{repo_id_of(repo_dir)}: episode rows not grouped ascending",
        )

    def column_matrix(name: str) -> np.ndarray:
        col = table[name].combine_chunks()
        flat = col.values.to_numpy(zero_copy_only=False).astype(np.float32)
        offsets = col.offsets.to_numpy()
        widths = np.diff(offsets)
        if len(np.unique(widths)) != 1:
            raise AssertionError(f"{repo_id_of(repo_dir)}: ragged {name} column")
        return flat.reshape(len(widths), int(widths[0]))

    actions = column_matrix("action")
    states = column_matrix("observation.state")
    bounds = [*starts.tolist(), len(episode_column)]
    episodes = [
        (
            int(ids[i]),
            actions[bounds[i] : bounds[i + 1]],
            states[bounds[i] : bounds[i + 1]],
        )
        for i in range(len(ids))
    ]
    warnings = []
    if not np.array_equal(ids, np.arange(len(ids))):
        warnings.append(f"{repo_id_of(repo_dir)}: episode ids not contiguous 0..n-1")
    return episodes, warnings


def analyze(
    records: list[EpisodeRecord],
    selected: set[str],
    holdout_map: dict[str, set[int]],
    plan_rows: dict[str, list[tuple[str, int]]],
) -> dict:
    """The frozen reads. `holdout_map` covers every selected repo (empty
    set where no episode is held out); train side = episodes - holdout.
    `plan_rows`: {"core": [(repo, episode), ...], "labeled": [...]} —
    one entry PER PANEL ROW (not per episode)."""
    by_key: dict[str, dict[tuple, list[EpisodeRecord]]] = {
        "exact_full": defaultdict(list),
        "quant_full": defaultdict(list),
        "action_only": defaultdict(list),
    }
    frames = {}
    for r in records:
        frames[(r.repo_id, r.episode)] = r.n_frames
        by_key["exact_full"][(r.n_frames, r.action_exact, r.state_exact)].append(r)
        by_key["quant_full"][(r.n_frames, r.action_quant, r.state_quant)].append(r)
        by_key["action_only"][(r.n_frames, r.action_exact)].append(r)

    def is_train(repo: str, episode: int) -> bool:
        return repo in selected and episode not in holdout_map.get(repo, set())

    result: dict = {"tiers": {}}
    for tier, groups in by_key.items():
        clusters = [g for g in groups.values() if len(g) > 1]
        cross = [c for c in clusters if len({r.repo_id for r in c}) > 1]
        intra_pairs = [c for c in clusters if len({r.repo_id for r in c}) < len(c)]
        leaked: dict[tuple[str, int], list[dict]] = {}
        for cluster in clusters:
            members = [(r.repo_id, r.episode) for r in cluster]
            for r in cluster:
                if r.repo_id not in selected:
                    continue
                if r.episode not in holdout_map.get(r.repo_id, set()):
                    continue
                donors = [
                    {"repo_id": r2, "episode": e2, "intra_repo": r2 == r.repo_id}
                    for (r2, e2) in members
                    if (r2, e2) != (r.repo_id, r.episode) and is_train(r2, e2)
                ]
                if donors:
                    leaked[(r.repo_id, r.episode)] = donors
        result["tiers"][tier] = {
            "clusters": len(clusters),
            "cross_repo_clusters": len(cross),
            "cross_repo_episodes": sum(len(c) for c in cross),
            "cross_repo_frames": sum(
                frames[(r.repo_id, r.episode)] for c in cross for r in c
            ),
            "intra_repo_dup_clusters": len(intra_pairs),
            "leaked_holdout_episodes": len(leaked),
            "leaked": {
                f"{repo}::{ep}": donors for (repo, ep), donors in sorted(leaked.items())
            },
            "cluster_members": [
                sorted(f"{r.repo_id}::{r.episode}" for r in c) for c in cross
            ],
        }
        leaked_set = set(leaked)
        for split in ("core", "labeled"):
            rows = plan_rows.get(split, [])
            hit = sum(1 for row in rows if row in leaked_set)
            result["tiers"][tier][f"panel_{split}_rows_leaked"] = hit
            result["tiers"][tier][f"panel_{split}_rows_total"] = len(rows)
    return result


def run_oracle() -> None:
    """Synthetic-array suite through fingerprint_episode + analyze."""
    rng = np.random.default_rng(0)

    def ep(
        n: int = 40,
        d: int = 6,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        g = np.random.default_rng(seed) if seed is not None else rng
        return (
            g.normal(size=(n, d)).astype(np.float32),
            g.normal(size=(n, d)).astype(np.float32),
        )

    base_a, base_s = ep(seed=7)
    donor2_a, donor2_s = ep(seed=8)
    intra_a, intra_s = ep(seed=9)
    const_a = np.zeros((40, 6), dtype=np.float32)

    # Split layout is EXPLICIT (holdout_map below), so leak expectations
    # are constructed, not sampled: X/a holds out {0, 2}; X/b, X/c, X/d
    # are all-train donors; X/a's episode 3 is its own train-side twin
    # of holdout episode 2 (intra-repo split crossing).
    specs = {
        ("X/a", 0): (base_a, base_s),  # holdout, leaked by X/b::1 (cross)
        ("X/a", 1): ep(seed=10),  # train, unique
        ("X/a", 2): (intra_a, intra_s),  # holdout, leaked by X/a::3 (intra)
        ("X/a", 3): (intra_a.copy(), intra_s.copy()),  # train twin of X/a::2
        ("X/a", 4): (donor2_a, donor2_s),  # holdout, leaked by single-ep repo X/c
        ("X/b", 0): ep(seed=11),  # train, unique
        ("X/b", 1): (base_a.copy(), base_s.copy()),  # train twin of X/a::0
        ("X/b", 2): (
            base_a + rng.normal(scale=1e-2, size=base_a.shape).astype(np.float32),
            base_s,
        ),  # noise copy: no tier
        ("X/c", 0): (donor2_a.copy(), donor2_s.copy()),  # single-episode repo donor
        ("X/d", 0): (
            base_a.astype(np.float64).astype(np.float32),
            base_s.copy(),
        ),  # recast survives f64 round-trip -> exact tier too
        ("X/d", 1): (
            (np.round(base_a.astype(np.float64) / QUANTUM) * QUANTUM).astype(
                np.float32,
            ),
            base_s.copy(),
        ),  # re-encoded onto the quantum grid: quant tier only, exact tier differs
        ("X/e", 0): (const_a, ep(seed=12)[1]),  # constant actions, state A
        ("X/f", 0): (
            const_a.copy(),
            ep(seed=13)[1],
        ),  # constant actions, state B -> action_only tier only, train->train
    }
    records = []
    for (repo, episode), (a, s) in specs.items():
        ax, sx, aq, sq = fingerprint_episode(a, s)
        records.append(EpisodeRecord(repo, episode, len(a), ax, sx, aq, sq))
    selected = {"X/a", "X/b", "X/c", "X/d", "X/e", "X/f"}
    holdout_map = {
        "X/a": {0, 2, 4},
        "X/b": set(),
        "X/c": set(),
        "X/d": set(),
        "X/e": set(),
        "X/f": set(),
    }
    plan_rows = {
        "core": [("X/a", 0), ("X/a", 0), ("X/a", 2), ("X/a", 4), ("X/a", 1)],
        "labeled": [("X/a", 0)],
    }
    out = analyze(records, selected, holdout_map, plan_rows)

    exact = out["tiers"]["exact_full"]
    leaked = set(exact["leaked"])
    assert leaked == {"X/a::0", "X/a::2", "X/a::4"}, leaked
    assert exact["leaked"]["X/a::2"][0]["intra_repo"] is True
    assert exact["leaked"]["X/a::0"][0]["intra_repo"] is False
    # X/a::0 has TWO exact train-side twins (X/b::1 and X/d::0 — a f32
    # array survives a f64 round-trip bit-exact, so the "recast" lands
    # in the exact tier as well).
    assert len(exact["leaked"]["X/a::0"]) == 2, exact["leaked"]["X/a::0"]
    assert exact["panel_core_rows_leaked"] == 4  # 2x (X/a,0) + (X/a,2) + (X/a,4)
    assert exact["panel_core_rows_total"] == 5
    assert exact["panel_labeled_rows_leaked"] == 1

    quant = out["tiers"]["quant_full"]
    # quantized tier additionally matches X/d::1 (1e-9 relative perturb
    # rounds away at 1e-3) into the base cluster.
    assert set(quant["leaked"]) == {"X/a::0", "X/a::2", "X/a::4"}
    assert len(quant["leaked"]["X/a::0"]) == 3, quant["leaked"]["X/a::0"]
    # the 1e-2 noise copy X/b::2 must appear in NO tier's clusters.
    for tier in out["tiers"].values():
        assert not any("X/b::2" in m for c in tier["cluster_members"] for m in c)
    # constant-action pair: action_only only, and train->train => no leak.
    action_only = out["tiers"]["action_only"]
    assert any({"X/e::0", "X/f::0"} <= set(c) for c in action_only["cluster_members"])
    assert not any({"X/e::0", "X/f::0"} <= set(c) for c in exact["cluster_members"])
    assert set(action_only["leaked"]) == {"X/a::0", "X/a::2", "X/a::4"}
    print("[oracle] all synthetic assertions passed:")
    print(
        "  cross-repo exact dup across the split -> leaked (2 donors incl. f64 round-trip)",
    )
    print("  intra-repo split-crossing twin -> leaked, tagged intra_repo")
    print("  single-episode donor repo -> leaked")
    print("  1e-2 noise copy -> invisible in every tier")
    print("  quantum-grid re-encode -> quantized tier only")
    print(
        "  constant-action/different-state -> action_only tier only, train->train no leak",
    )
    print("  panel row join: core 4/5, labeled 1/1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--npz", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if args.oracle:
        run_oracle()
        return
    if not (args.data and args.plan and args.npz and args.output_json):
        parser.error("--data/--plan/--npz/--output-json required outside --oracle")

    plan = json.loads(args.plan.read_text())
    assert plan["fps"] == list(PANEL_FPS) and plan["camera_counts"] == list(
        PANEL_CAMERA_COUNTS,
    )
    assert plan["holdout_episodes"] == PANEL_HOLDOUT_FRACTION
    assert plan["split_seed"] == PANEL_SPLIT_SEED
    plan_rows = {
        split: [(row[0], int(row[1])) for row in plan[split]]
        for split in ("core", "labeled")
    }
    npz = np.load(args.npz, allow_pickle=False)
    n_rows = int(npz["index"].shape[0])
    n_core = int(npz["core"].sum())
    assert n_core == len(plan_rows["core"]), (n_core, len(plan_rows["core"]))
    assert n_rows == len(plan_rows["core"]) + len(plan_rows["labeled"]), n_rows
    print(
        f"[census] plan/npz agree: {n_core} core + {n_rows - n_core} labeled = {n_rows} rows",
    )

    # --- selection mirror (data.py make_datasets loop, filters only) ---
    dataset_dirs = discover_datasets((args.data,), ())
    infos = [DatasetInfo.from_json(d / "meta" / "info.json") for d in dataset_dirs]
    anchor = next(i for i in infos if i.action_state_dims is not None)
    selected_dirs: dict[str, Path] = {}
    total_eps: dict[str, int] = {}
    dropped = 0
    for d, info in zip(dataset_dirs, infos, strict=True):
        repo_id = repo_id_of(d)
        if (
            repo_id in selected_dirs
            or info.action_state_dims != anchor.action_state_dims
            or info.fps not in PANEL_FPS
            or len(info.cameras) not in PANEL_CAMERA_COUNTS
        ):
            dropped += 1
            continue
        selected_dirs[repo_id] = d
        total_eps[repo_id] = info.total_episodes
    print(
        f"[census] discovered {len(dataset_dirs)} datasets; selected "
        f"{len(selected_dirs)} under the panel convention ({dropped} dropped; "
        f"2026-08-03 camera-filter-only anchor: 878/981)",
    )

    holdout_map = {
        repo: set(
            holdout_episodes(
                repo,
                total_eps[repo],
                PANEL_HOLDOUT_FRACTION,
                PANEL_SPLIT_SEED,
            ),
        )
        for repo in selected_dirs
    }
    # SPLIT-MIRROR ORACLE: the plan's episode set per repo must equal the
    # holdout re-derived here, for every repo the plan touches, and every
    # selected repo absent from the plan must have exactly one episode.
    plan_eps: dict[str, set[int]] = defaultdict(set)
    for repo, episode in plan_rows["core"]:
        plan_eps[repo].add(episode)
    for repo, eps in plan_eps.items():
        assert repo in selected_dirs, f"plan repo {repo} not in mirrored selection"
        assert eps == holdout_map[repo], (
            f"{repo}: plan holdout {sorted(eps)} != derived {sorted(holdout_map[repo])}"
        )
    for repo in selected_dirs:
        if repo not in plan_eps:
            assert total_eps[repo] < 2, (
                f"{repo}: absent from plan but {total_eps[repo]} episodes"
            )
    print(
        f"[census] split mirror PROVEN: plan == derived holdout on all {len(plan_eps)} plan repos",
    )

    # --- fingerprint sweep (ALL discovered repos, selected or not: an
    # unselected repo cannot leak into train, but its dup content is
    # still R1 census material) ---
    records: list[EpisodeRecord] = []
    warnings: list[str] = []
    ep_counts: dict[str, int] = {}
    arrays_dir: dict[tuple[str, int], Path] = {}
    for n, (d, info) in enumerate(zip(dataset_dirs, infos, strict=True), 1):
        repo_id = repo_id_of(d)
        if repo_id in ep_counts:  # duplicate repo id: same first-wins as data.py
            continue
        try:
            episodes, warn = read_repo_episodes(d)
        except FileNotFoundError as err:
            warnings.append(str(err))
            continue
        warnings.extend(warn)
        ep_counts[repo_id] = len(episodes)
        if info.total_episodes != len(episodes):
            warnings.append(
                f"{repo_id}: info.json total_episodes {info.total_episodes} "
                f"!= parquet distinct {len(episodes)}",
            )
        for episode, action, state in episodes:
            ax, sx, aq, sq = fingerprint_episode(action, state)
            records.append(EpisodeRecord(repo_id, episode, len(action), ax, sx, aq, sq))
            arrays_dir[(repo_id, episode)] = d
        if n % 200 == 0:
            print(
                f"[census] fingerprinted {n}/{len(dataset_dirs)} repos, {len(records)} episodes",
            )
    print(
        f"[census] fingerprinted {len(records)} episodes across {len(ep_counts)} repos",
    )

    out = analyze(records, set(selected_dirs), holdout_map, plan_rows)
    out["warnings"] = warnings
    out["selection"] = {
        "discovered": len(dataset_dirs),
        "selected": len(selected_dirs),
        "episodes_fingerprinted": len(records),
        "convention": {
            "fps": list(PANEL_FPS),
            "camera_counts": list(PANEL_CAMERA_COUNTS),
            "holdout_fraction": PANEL_HOLDOUT_FRACTION,
            "split_seed": PANEL_SPLIT_SEED,
        },
        "quantum": QUANTUM,
    }

    # --- hash-collision guard on up to 20 exact-full flagged pairs ---
    checked = 0
    for cluster in out["tiers"]["exact_full"]["cluster_members"]:
        if checked >= 20:
            break
        (r1, e1), (r2, e2) = (m.rsplit("::", 1) for m in cluster[:2])
        eps1 = {
            e: (a, s) for e, a, s in read_repo_episodes(arrays_dir[(r1, int(e1))])[0]
        }
        eps2 = {
            e: (a, s) for e, a, s in read_repo_episodes(arrays_dir[(r2, int(e2))])[0]
        }
        a1, s1 = eps1[int(e1)]
        a2, s2 = eps2[int(e2)]
        assert np.array_equal(a1, a2) and np.array_equal(s1, s2), cluster[:2]
        checked += 1
    out["collision_guard_pairs_verified"] = checked
    print(f"[census] collision guard: {checked} flagged pairs re-loaded, arrays equal")

    args.output_json.write_text(json.dumps(out, indent=2))
    print(f"[census] report -> {args.output_json}")
    for tier in ("exact_full", "quant_full", "action_only"):
        t = out["tiers"][tier]
        print(
            f"[{tier}] clusters {t['clusters']} (cross-repo {t['cross_repo_clusters']}, "
            f"episodes {t['cross_repo_episodes']}, frames {t['cross_repo_frames']}) | "
            f"leaked holdout episodes {t['leaked_holdout_episodes']} | panel rows leaked "
            f"core {t['panel_core_rows_leaked']}/{t['panel_core_rows_total']} "
            f"labeled {t['panel_labeled_rows_leaked']}/{t['panel_labeled_rows_total']}",
        )
    if warnings:
        print(
            f"[census] {len(warnings)} structural warnings (see report JSON), first 5:",
        )
        for w in warnings[:5]:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
