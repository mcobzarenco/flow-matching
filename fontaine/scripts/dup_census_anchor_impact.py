"""R5 of the duplicate-content census (#18.7): do leaked panel frames bias the anchors?

Declared before computing (this file is written before any subset number
is read): split the banked AR-100k and flow-80k panel core frames into
CLEAN vs LEAKED (leaked = the census's exact_full holdout->train leaked
episodes, ~/dup_census_report.json) and pool chunk_mae / first_mae per
subset with the validated pooling (imported from box_batch_results.py —
the code path the 5.8026/6.6232 anchors reproduce through). The
memorization signature is leaked < clean beyond frame-sampling noise
(seeded frame-level bootstrap on the difference of subset means).

The npz predates the identity-column hardening (#18.1), so rows are
joined to (repo_id, episode_index, frame_index) by per-repo order
alignment: within one repo, concat index = offset + episode_start +
frame_index is strictly monotone in (episode, frame), and the plan's
rows sorted the same way are bijective with the npz's rows for that
repo. VALIDATION, all hard asserts before any read is printed:
  * per repo: npz row count == plan row count AND the npz core-flag
    pattern equals the plan's core/labeled tags in join order (an
    order bug scrambles interleaved core/labeled tags immediately);
  * end-to-end content check on 6 sampled repos incl. >=2 leaked: npz
    truth[i, j, :] must equal the repo parquet's action[frame+j] for
    every valid j (the join is verified against raw data, not just
    structure);
  * partition: clean | leaked == core, clean & leaked == 0, and pooling
    the union reproduces the anchors 5.8026/2.1431 + 6.6232/1.9331.

Usage:
  uv run python fontaine/scripts/dup_census_anchor_impact.py \
      --census ~/dup_census_report.json \
      --plan plans/holdout_curated_v0_k4l2.json \
      --data ~/datasets/mcobzarenco/community_curated_v0 \
      --out ~/dup_census_anchor_impact.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from box_batch_results import masks, pooled_chunk, pooled_first
from dup_content_census import read_repo_episodes

NPZS = {
    "ar_100k": (
        "reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2.npz",
        "pred:bijou@100000",
    ),
    "flow_80k": (
        "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_k4l2_heun30.npz",
        "pred:bijou@80000",
    ),
}
ANCHORS = {"ar_100k": (5.8026, 2.1431), "flow_80k": (6.6232, 1.9331)}
BOOT_N = 10_000
BOOT_SEED = 0


def build_join(plan: dict, d: np.lib.npyio.NpzFile) -> np.ndarray:
    """Structured array (repo_id, episode, frame) aligned to npz rows."""
    rows = [(r, int(e), int(f), True) for r, e, f in plan["core"]]
    rows += [(r, int(e), int(f), False) for r, e, f in plan["labeled"]]
    by_repo: dict[str, list] = defaultdict(list)
    for r, e, f, is_core in rows:
        by_repo[r].append((e, f, is_core))
    for planned_rows in by_repo.values():
        planned_rows.sort()

    repo_ids = d["repo_id"]
    index = d["index"]
    core = d["core"]
    out = np.zeros(
        len(index),
        dtype=[("repo", "U87"), ("episode", "i8"), ("frame", "i8")],
    )
    for repo, planned in by_repo.items():
        sel = np.flatnonzero(repo_ids == repo)
        assert len(sel) == len(planned), (repo, len(sel), len(planned))
        order = sel[np.argsort(index[sel], kind="stable")]
        npz_core_pattern = core[order].tolist()
        plan_core_pattern = [is_core for (_e, _f, is_core) in planned]
        assert npz_core_pattern == plan_core_pattern, (
            f"{repo}: core-flag pattern mismatch — join order is wrong"
        )
        for row, (e, f, _c) in zip(order, planned, strict=True):
            out[row] = (repo, e, f)
    assert not np.any(out["repo"] == ""), "unjoined npz rows"
    return out


def content_check(
    join: np.ndarray,
    d: np.lib.npyio.NpzFile,
    data_root: Path,
    leaked: set,
) -> int:
    truth, valid = d["truth"], d["valid"]
    repos = sorted(set(join["repo"]))
    leaked_repos = sorted({r for (r, _e) in leaked})
    picks = (
        leaked_repos[:2]
        + [r for r in repos[:: max(1, len(repos) // 4)] if r not in leaked_repos][:4]
    )
    checked = 0
    for repo in picks:
        episodes = {e: (a, s) for e, a, s in read_repo_episodes(data_root / repo)[0]}
        for i in np.flatnonzero(join["repo"] == repo)[:8]:
            action = episodes[int(join["episode"][i])][0]
            f = int(join["frame"][i])
            for j in range(truth.shape[1]):
                if not valid[i, j]:
                    continue
                assert np.allclose(truth[i, j], action[f + j], atol=1e-5), (
                    repo,
                    int(join["episode"][i]),
                    f,
                    j,
                )
            checked += 1
    return checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    census = json.loads(args.census.read_text())
    leaked = {
        (k.rsplit("::", 1)[0], int(k.rsplit("::", 1)[1]))
        for k in census["tiers"]["exact_full"]["leaked"]
    }
    plan = json.loads(args.plan.read_text())
    out: dict = {"leaked_episodes": len(leaked), "models": {}}

    for name, (npz_path, pred_key) in NPZS.items():
        d = np.load(npz_path, allow_pickle=False)
        join = build_join(plan, d)
        n_checked = content_check(join, d, args.data, leaked)
        print(
            f"[{name}] join built + content-checked on {n_checked} rows against raw parquet",
        )

        truth, valid, core, w = masks(d)
        err = np.abs(d[pred_key] - truth)
        is_leaked = np.fromiter(
            (
                (r, e) in leaked
                for r, e in zip(join["repo"], join["episode"], strict=True)
            ),
            dtype=bool,
            count=len(join),
        )
        clean_core = core & ~is_leaked
        leak_core = core & is_leaked
        assert np.array_equal(clean_core | leak_core, core)
        assert not np.any(clean_core & leak_core)

        full = (pooled_chunk(err, core, w), pooled_first(err, valid, core))
        anchor = ANCHORS[name]
        assert abs(full[0] - anchor[0]) < 5e-4 and abs(full[1] - anchor[1]) < 5e-4, (
            full,
            anchor,
        )
        clean = (pooled_chunk(err, clean_core, w), pooled_first(err, valid, clean_core))
        leak = (pooled_chunk(err, leak_core, w), pooled_first(err, valid, leak_core))

        # frame-level bootstrap on the leaked-minus-clean difference of
        # frame-MAE means (descriptive noise scale, seeded)
        nvalid = w.sum(axis=(1, 2))
        fmae = (err * w).sum(axis=(1, 2)) / np.maximum(nvalid, 1)
        rng = np.random.default_rng(BOOT_SEED)
        c_idx, l_idx = np.flatnonzero(clean_core), np.flatnonzero(leak_core)
        deltas = np.empty(BOOT_N)
        for b in range(BOOT_N):
            deltas[b] = (
                fmae[rng.choice(l_idx, len(l_idx))].mean()
                - fmae[rng.choice(c_idx, len(c_idx))].mean()
            )
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        out["models"][name] = {
            "full": [round(x, 4) for x in full],
            "clean": [round(x, 4) for x in clean],
            "leaked": [round(x, 4) for x in leak],
            "core_frames": {
                "clean": int(clean_core.sum()),
                "leaked": int(leak_core.sum()),
            },
            "leaked_minus_clean_framemae_ci95": [
                round(float(lo), 4),
                round(float(hi), 4),
            ],
        }
        print(
            f"[{name}] full {full[0]:.4f}/{full[1]:.4f} (anchor OK) | "
            f"clean {clean[0]:.4f}/{clean[1]:.4f} ({int(clean_core.sum())} fr) | "
            f"leaked {leak[0]:.4f}/{leak[1]:.4f} ({int(leak_core.sum())} fr) | "
            f"leaked-clean frame-MAE CI95 [{lo:+.4f}, {hi:+.4f}]",
        )

    args.out.write_text(json.dumps(out, indent=2))
    print(f"[impact] -> {args.out}")


if __name__ == "__main__":
    main()
