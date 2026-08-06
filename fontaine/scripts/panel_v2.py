"""Panel v2 materializer + anchor derivation (#18.7 follow-on) — PROPOSAL instrument.

The dup-content census (posts/2026-08-06-dup-census-results.md) proved
the panel's holdout is breached by the cross-repo fork channel: 524
holdout episodes have byte-exact train twins, carrying 2,096/17,204
core rows (12.2%). Separately, the wrap census flagged 3 repos with
corrupted streams which the SEALED plan v2 dropped but the panel plan
still scores (52 core + 26 labeled rows). This instrument freezes the
proposed re-definition and derives its anchors — it changes NOTHING
until the owner steers the amendment (posts/2026-08-06-panel-v2-amendment.md):

  panel v2 = panel v1 minus (a) every row on a census-leaked
  (repo, episode) and (b) every row of the wrap-census corrupt repos.
  Strict row-subset, original order — no re-draw. Therefore every
  banked per-frame npz re-pools to v2 EXACTLY, with no re-eval.

Outputs:
  * plans/holdout_curated_v0_k4l2_panel_v2.json — self-contained
    (embeds the frozen exclusion lists + provenance counts)
  * a report JSON with v1 + v2 pooled chunk_mae/first_mae for the
    banked AR-100k and flow-80k panels and the state-copy baselines.

Validation (hard asserts, all run before any v2 number is printed):
  * leaked-row exclusion counts equal the census's published
    panel_core_rows_leaked / panel_labeled_rows_leaked (2,096 / 1,048);
  * corrupt-repo rows equal the sealed-v2 amendment's 52 / 26;
  * per npz: the v1 pooling reproduces the banked anchors
    5.8026/2.1431 and 6.6232/1.9331 (<5e-4);
  * the leaked-only exclusion reproduces the census's clean-core
    numbers 5.9761/2.1695 and 6.8137/1.9714 (<=1e-4, they were
    published at 4dp);
  * state-copy is model-free, so both npzs must agree on its pooled
    numbers to 5e-4 (cross-npz join consistency);
  * the materialized v2 plan, re-loaded, re-filters to identical rows
    (idempotence) and is a strict ordered subset of v1.

--oracle runs a synthetic materialization check first: a fabricated
plan with known exclusions must filter to exactly the expected rows.

Usage:
  uv run python fontaine/scripts/panel_v2.py \
      --plan plans/holdout_curated_v0_k4l2.json \
      --census ~/dup_census_report.json \
      --out-plan plans/holdout_curated_v0_k4l2_panel_v2.json \
      --out ~/panel_v2_anchors.json

Pure CPU, read-only on inputs other than --out/--out-plan. Deterministic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from box_batch_results import masks, pooled_chunk, pooled_first
from dup_census_anchor_impact import build_join

# Wrap census (posts/2026-08-05-wrap-census.md): systemic +/-180 wraps /
# state-stream glitch. Same list the sealed plan v2 removed.
CORRUPT_REPOS = [
    "kevin510/lerobot-cat-toy-placement",
    "kevin510/so-100-draw-smiley",
    "willnorris/bbox-2",
]

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
BASELINES = ["pred:state-copy", "pred:state-copy-norm"]
V1_ANCHORS = {"ar_100k": (5.8026, 2.1431), "flow_80k": (6.6232, 1.9331)}
# Census R5 clean-core, published at 4dp (posts/2026-08-06-dup-census-results.md).
CENSUS_CLEAN = {"ar_100k": (5.9761, 2.1695), "flow_80k": (6.8137, 1.9714)}
CENSUS_LEAKED_ROWS = {"core": 2096, "labeled": 1048}
SEALED_V2_CORRUPT_ROWS = {"core": 52, "labeled": 26}


def leaked_set(census: dict) -> set[tuple[str, int]]:
    return {
        (k.rsplit("::", 1)[0], int(k.rsplit("::", 1)[1]))
        for k in census["tiers"]["exact_full"]["leaked"]
    }


def split_rows(
    rows: list,
    leaked: set[tuple[str, int]],
    corrupt: set[str],
) -> tuple[list, int, int, int]:
    """Filter one plan section; returns (kept, n_leaked, n_corrupt, n_overlap)."""
    kept, n_leaked, n_corrupt, n_overlap = [], 0, 0, 0
    for r, e, f in rows:
        is_leaked = (r, int(e)) in leaked
        is_corrupt = r in corrupt
        if is_leaked:
            n_leaked += 1
        if is_corrupt:
            n_corrupt += 1
        if is_leaked and is_corrupt:
            n_overlap += 1
        if not (is_leaked or is_corrupt):
            kept.append([r, e, f])
    return kept, n_leaked, n_corrupt, n_overlap


def materialize(plan: dict, leaked: set, corrupt: set) -> tuple[dict, dict]:
    core, cl, cc, co = split_rows(plan["core"], leaked, corrupt)
    labeled, ll, lc, lo = split_rows(plan["labeled"], leaked, corrupt)
    counts = {
        "core": {
            "v1": len(plan["core"]),
            "leaked": cl,
            "corrupt": cc,
            "overlap": co,
            "v2": len(core),
        },
        "labeled": {
            "v1": len(plan["labeled"]),
            "leaked": ll,
            "corrupt": lc,
            "overlap": lo,
            "v2": len(labeled),
        },
    }
    v2 = {k: v for k, v in plan.items() if k not in ("core", "labeled")}
    v2["version"] = 2
    v2["derived_from"] = "holdout_curated_v0_k4l2.json (v1, byte-identical rows)"
    v2["exclusions"] = {
        "leaked_episodes": sorted(f"{r}::{e}" for r, e in leaked),
        "corrupt_repos": sorted(corrupt),
        "counts": counts,
        "provenance": (
            "leaked: dup-content census exact_full tier "
            "(posts/2026-08-06-dup-census-results.md); corrupt: wrap census "
            "(posts/2026-08-05-wrap-census.md, sealed plan v2 precedent)"
        ),
    }
    v2["core"] = core
    v2["labeled"] = labeled
    return v2, counts


def pool(
    d: np.lib.npyio.NpzFile,
    key: str,
    keep_rows: np.ndarray,
) -> tuple[float, float]:
    truth, valid, core, w = masks(d)
    err = np.abs(d[key] - truth)
    sel = core & keep_rows
    return pooled_chunk(err, sel, w), pooled_first(err, valid, sel)


def synthetic_oracle() -> None:
    plan = {
        "version": 1,
        "core": [["a/x", 0, 0], ["a/x", 1, 0], ["b/y", 0, 3], ["c/z", 2, 1]],
        "labeled": [["a/x", 1, 5], ["c/z", 2, 9]],
    }
    leaked = {("a/x", 1)}
    corrupt = {"c/z"}
    v2, counts = materialize(plan, leaked, corrupt)
    assert v2["core"] == [["a/x", 0, 0], ["b/y", 0, 3]], v2["core"]
    assert v2["labeled"] == [], v2["labeled"]
    assert counts["core"] == {"v1": 4, "leaked": 1, "corrupt": 1, "overlap": 0, "v2": 2}
    assert counts["labeled"] == {
        "v1": 2,
        "leaked": 1,
        "corrupt": 1,
        "overlap": 0,
        "v2": 0,
    }
    # overlap case: an episode both leaked and in a corrupt repo is
    # counted in both tallies and excluded once
    v3, counts3 = materialize(plan, {("c/z", 2)}, corrupt)
    assert counts3["core"]["overlap"] == 1
    assert v3["core"] == [["a/x", 0, 0], ["a/x", 1, 0], ["b/y", 0, 3]]
    print("[oracle] synthetic materialization checks OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--out-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--oracle", action="store_true")
    args = parser.parse_args()

    if args.oracle:
        synthetic_oracle()

    plan = json.loads(args.plan.read_text())
    census = json.loads(args.census.read_text())
    leaked = leaked_set(census)
    corrupt = set(CORRUPT_REPOS)
    v2, counts = materialize(plan, leaked, corrupt)

    # count oracles vs the two published exclusion sources
    for section, want in CENSUS_LEAKED_ROWS.items():
        got = counts[section]["leaked"]
        assert got == want, f"{section}: leaked rows {got} != census {want}"
    for section, want in SEALED_V2_CORRUPT_ROWS.items():
        got = counts[section]["corrupt"]
        assert got == want, f"{section}: corrupt rows {got} != sealed-v2 {want}"

    # idempotence: re-filtering the materialized v2 removes nothing
    v2_again, _ = materialize(v2, leaked, corrupt)
    assert v2_again["core"] == v2["core"] and v2_again["labeled"] == v2["labeled"]
    # strict ordered subset of v1
    it = iter(plan["core"] + plan["labeled"])
    for row in v2["core"] + v2["labeled"]:
        for cand in it:
            if cand == row:
                break
        else:
            raise AssertionError(f"v2 row {row} not in v1 order")

    excluded = leaked | {
        (r, int(e)) for r, e, _f in plan["core"] + plan["labeled"] if r in corrupt
    }
    out: dict = {"counts": counts, "models": {}}
    copy_ref: dict[str, tuple[float, float]] = {}
    for name, (npz_path, pred_key) in NPZS.items():
        d = np.load(npz_path, allow_pickle=False)
        join = build_join(plan, d)
        keep = np.fromiter(
            (
                (r, int(e)) not in excluded
                for r, e in zip(join["repo"], join["episode"], strict=True)
            ),
            dtype=bool,
            count=len(join),
        )
        leaked_only_keep = np.fromiter(
            (
                (r, int(e)) not in leaked
                for r, e in zip(join["repo"], join["episode"], strict=True)
            ),
            dtype=bool,
            count=len(join),
        )
        all_rows = np.ones(len(join), dtype=bool)

        v1_nums = pool(d, pred_key, all_rows)
        a = V1_ANCHORS[name]
        assert abs(v1_nums[0] - a[0]) < 5e-4 and abs(v1_nums[1] - a[1]) < 5e-4, (
            name,
            v1_nums,
        )
        clean = pool(d, pred_key, leaked_only_keep)
        c = CENSUS_CLEAN[name]
        assert (
            abs(round(clean[0], 4) - c[0]) <= 1e-4
            and abs(round(clean[1], 4) - c[1]) <= 1e-4
        ), (
            name,
            clean,
        )
        v2_nums = pool(d, pred_key, keep)
        entry = {
            "v1": [round(x, 4) for x in v1_nums],
            "census_clean_core": [round(x, 4) for x in clean],
            "v2": [round(x, 4) for x in v2_nums],
            "baselines": {},
        }
        for bk in BASELINES:
            b_v1, b_v2 = pool(d, bk, all_rows), pool(d, bk, keep)
            entry["baselines"][bk] = {
                "v1": [round(x, 4) for x in b_v1],
                "v2": [round(x, 4) for x in b_v2],
            }
            if bk in copy_ref:
                pc, pf = copy_ref[bk]
                assert abs(b_v1[0] - pc) < 5e-4 and abs(b_v1[1] - pf) < 5e-4, (
                    f"{bk}: cross-npz state-copy mismatch",
                    (b_v1, (pc, pf)),
                )
            else:
                copy_ref[bk] = b_v1
        out["models"][name] = entry
        print(
            f"[{name}] v1 {v1_nums[0]:.4f}/{v1_nums[1]:.4f} (anchor OK) | "
            f"clean-core {clean[0]:.4f}/{clean[1]:.4f} (census OK) | "
            f"v2 {v2_nums[0]:.4f}/{v2_nums[1]:.4f} "
            f"({int((d['core'] & keep).sum())} core rows)",
        )

    args.out_plan.write_text(json.dumps(v2, indent=1) + "\n")
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"[panel_v2] plan -> {args.out_plan} "
        f"(core {counts['core']['v1']} -> {counts['core']['v2']}, "
        f"labeled {counts['labeled']['v1']} -> {counts['labeled']['v2']}); "
        f"report -> {args.out}",
    )


if __name__ == "__main__":
    main()
