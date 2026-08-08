"""Aliased-frame mining for the field/subgoal-conditioning meta-report.

Owner steering 2026-08-08 13:21Z asked the meta-report to showcase
frames where the right action is ambiguous from the image alone. The
protocol is AliasBench's aliasing diagnostic run in reverse
(papers/observation-aliasing.md, 2605.14712 §experiments): embed panel
frames with a FROZEN vision tower, retrieve nearest neighbors, and flag
frames whose close neighbors carry divergent ground-truth
continuations. AR-100k trained with ``--backbone-text-lr`` only, so the
base Gemma-4 E2B tower IS that policy's frozen eye — "close in
embedding" means close to the perception of the policy being scored.

Two subcommands, both record-only reads on banked data:

``embed`` (GPU, ~30 min, run via run_detached.sh): rebuild the exact
panel selection (args pinned below = the AR-100k eval invocation),
fetch the 17,204 core frames by the banked npz ``index``, hard-abort
unless each item's (repo_id, episode, frame) AND raw action chunk match
the npz row bit-for-bit (the alignment oracle), embed camera 0 (sorted
key order, recorded per row) through the tower, mean-pool the soft
tokens, write float16 embeddings + row metadata.

``mine`` (CPU): within-dataset cosine top-K neighbors (chunk-horizon
exclusion |Δframe| < 50 inside the same episode — overlapping chunks
share their continuation by construction), alias score = mean
per-dataset-std-normalized truth-chunk divergence to the top-K
neighbors; then the concentration read pinned BEFORE execution:

  primary   mean per-frame Δ_oracle (oraclesubgoal − baseline chunk
            MAE) on flagged frames (top decile of alias score over the
            qualifying pool) minus the rest, dataset-clustered
            bootstrap CI95, seed 0, 10k resamples
  secondary Spearman rank corr(alias score, Δ_oracle), same pool
  shape    concentration ⇒ the subgoal channel disambiguates (the
            IntentVLA 9%→45.8% shape); no concentration ⇒ it acts as a
            style/dataset prior — either is a claim for the report

Qualifying pool: datasets with ≥ 16 core rows (top-5 NN needs a
neighborhood; dropped counts are printed, never silent).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# The AR-100k panel eval invocation, pinned (docs/architecture.md
# §bijou_arb_rcond_100k_ddp4 + plans/holdout_curated_v0_k4l2.json).
DATA_ROOT = Path("~/datasets/mcobzarenco/community_curated_v0").expanduser()
FPS = (30.0,)
CAMERA_COUNTS = (1, 2)
HOLDOUT_FRACTION = 0.1
SPLIT_SEED = 0
CHUNK_SIZE = 50
TOWER = "google/gemma-4-e2b-it"  # AR-100k's frozen vision tower (no
# --backbone-vision-lr in the run; docs/architecture.md §setup)

BASELINE_NPZ = (
    REPO_ROOT / "reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2.npz"
)
COND_NPZ = (
    REPO_ROOT
    / "reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2_oraclesubgoal.npz"
)
BASELINE_KEY = "pred:bijou@100000"
COND_KEY = "pred:bijou@100000_oraclesubgoal"

EMBED_OUT = REPO_ROOT / "reports/framemining__ar100k_k4l2__embeddings.npz"
MINE_OUT = REPO_ROOT / "reports/analysis__framemining_ar100k_k4l2.json"
FLAGGED_OUT = REPO_ROOT / "reports/framemining__ar100k_k4l2__flagged.npz"

TOP_K = 5
MIN_ROWS = 16
FLAG_QUANTILE = 0.9
BOOTSTRAP = 10_000
SEED = 0


def load_panel_rows() -> dict[str, np.ndarray]:
    """Core-row metadata + per-frame Δ_oracle from the two banked npz,
    verified against each other (same plan ⇒ identical row order)."""
    base = np.load(BASELINE_NPZ, allow_pickle=True)
    cond = np.load(COND_NPZ, allow_pickle=True)
    if not np.array_equal(base["index"], cond["index"]):
        raise SystemExit("panel npz row order differs between arms")
    if not np.array_equal(base["truth"], cond["truth"]):
        raise SystemExit("panel npz truth differs between arms")
    core = base["core"]
    truth = base["truth"][core]
    valid = base["valid"][core]

    def chunk_mae(z: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
        error = np.abs(z[key][core] - truth).mean(axis=-1)  # [n, chunk]
        masked = np.where(valid, error, 0.0).sum(axis=-1)
        return masked / valid.sum(axis=-1)

    return {
        "index": base["index"][core],
        "repo_id": base["repo_id"][core],
        "episode_index": cond["episode_index"][core],
        "frame_index": cond["frame_index"][core],
        "truth": truth,
        "valid": valid,
        "delta": chunk_mae(cond, COND_KEY) - chunk_mae(base, BASELINE_KEY),
        # Instrument-validation reads (DSSP Prop 4.2 predicts elevated
        # REACTIVE-policy error on aliased frames; state-copy separates
        # "ambiguous" from merely "dynamic"):
        "baseline_mae": chunk_mae(base, BASELINE_KEY),
        "copy_mae": chunk_mae(base, "pred:state-copy"),
    }


def cmd_embed(args: argparse.Namespace) -> None:
    from PIL import Image
    from transformers import AutoProcessor

    from bijou.data import EpisodeSplit, select_datasets
    from bijou.gemma4.loading import load_model

    rows = load_panel_rows()
    n = len(rows["index"]) if args.limit is None else args.limit
    print(f"embedding {n} of {len(rows['index'])} core frames", flush=True)

    selection = select_datasets(
        (DATA_ROOT,),
        (),
        CHUNK_SIZE,
        episode_split=EpisodeSplit.HOLDOUT,
        holdout_fraction=HOLDOUT_FRACTION,
        split_seed=SPLIT_SEED,
        allowed_fps=FPS,
        allowed_camera_counts=CAMERA_COUNTS,
    )
    dataset = selection.concat()
    print(f"selection: {len(selection.datasets)} datasets, {len(dataset)} frames")

    device = torch.device(args.device)
    # truncate_layers=5: the text trunk never runs here — only the
    # vision tower does — but load_model needs a decoder to construct,
    # and truncation must end on a full_attention layer (first at 5).
    model = load_model(TOWER, device=device, truncate_layers=5)
    tower = model.vision_tower
    assert tower is not None
    processor = AutoProcessor.from_pretrained(TOWER)

    order = np.argsort(rows["index"][:n])  # concat order ≈ storage locality
    subset = torch.utils.data.Subset(dataset, rows["index"][:n][order].tolist())
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=lambda items: items,
        prefetch_factor=4 if args.num_workers else None,
    )

    hidden = None
    embeddings = None
    camera_keys = np.empty(n, dtype=object)
    done = 0
    for items in loader:
        pils, positions = [], []
        for offset, item in enumerate(items):
            row = order[done + offset]
            # Alignment oracle: the concat index must resolve to the
            # exact banked frame, actions included.
            if (
                str(item["repo_id"]) != str(rows["repo_id"][row])
                or int(item["episode_index"]) != int(rows["episode_index"][row])
                or int(item["frame_index"]) != int(rows["frame_index"][row])
            ):
                raise SystemExit(
                    f"index misalignment at npz row {row}: item "
                    f"{item['repo_id']}[{item['episode_index']}:"
                    f"{item['frame_index']}] vs npz "
                    f"{rows['repo_id'][row]}[{rows['episode_index'][row]}:"
                    f"{rows['frame_index'][row]}]",
                )
            action = item["action"].float().numpy()
            banked = rows["truth"][row][: action.shape[0]]
            if not np.allclose(action, banked, atol=1e-6):
                raise SystemExit(f"action chunk mismatch at npz row {row}")
            key = min(k for k in item if k.startswith("observation.images."))
            camera_keys[row] = key.removeprefix("observation.images.")
            image = item[key]
            array = (image.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
            pils.append(Image.fromarray(array))
            positions.append(row)

        batch = processor.apply_chat_template(
            [
                [{"role": "user", "content": [{"type": "image", "image": pil}]}]
                for pil in pils
            ],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        pixel_values = batch["pixel_values"].to(device)
        position_ids = batch["image_position_ids"].to(device)
        with torch.inference_mode():
            for i, row in enumerate(positions):
                # One image per forward: VisionModel flattens valid soft
                # tokens across images, so per-image calls keep
                # boundaries trivially correct.
                soft = tower(pixel_values[i : i + 1], position_ids[i : i + 1])
                pooled = soft.float().mean(dim=0)
                if embeddings is None:
                    hidden = pooled.shape[0]
                    embeddings = np.zeros((n, hidden), dtype=np.float16)
                embeddings[row] = pooled.cpu().numpy().astype(np.float16)
        done += len(items)
        if done % (args.batch_size * 20) < args.batch_size:
            print(f"  embedded {done}/{n}", flush=True)

    assert embeddings is not None
    out = (
        EMBED_OUT
        if args.limit is None
        else EMBED_OUT.parent / (EMBED_OUT.name[: -len(".npz")] + "_smoke.npz")
    )
    np.savez_compressed(
        out,
        embedding=embeddings,
        index=rows["index"][:n],
        repo_id=rows["repo_id"][:n],
        episode_index=rows["episode_index"][:n],
        frame_index=rows["frame_index"][:n],
        camera_key=camera_keys.astype(str),
        provenance=np.array(
            json.dumps(
                {
                    "tower": TOWER,
                    "pooling": "mean over VisionModel soft tokens, float32",
                    "camera": "first observation.images.* key, sorted",
                    "source_npz": [BASELINE_NPZ.name, COND_NPZ.name],
                    "alignment": "repo/episode/frame + action chunk, every row",
                },
            ),
        ),
    )
    print(f"wrote {out} ({n} rows, hidden {hidden})", flush=True)


def alias_scores(
    rows: dict[str, np.ndarray],
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Per-frame alias score + qualifying mask + neighbor records."""
    scores = np.full(len(embeddings), np.nan)
    qualifying = np.zeros(len(embeddings), dtype=bool)
    neighbor_records: list[dict[str, object]] = []
    by_repo: dict[str, list[int]] = defaultdict(list)
    for i, repo in enumerate(rows["repo_id"]):
        by_repo[str(repo)].append(i)

    dropped = 0
    for _repo, members in sorted(by_repo.items()):
        if len(members) < MIN_ROWS:
            dropped += len(members)
            continue
        idx = np.array(members)
        emb = embeddings[idx].astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        cosine_dist = 1.0 - emb @ emb.T
        # Continuations in per-dataset-std units: units differ across
        # rigs, and the divergence must be comparable panel-wide.
        truth = rows["truth"][idx]
        valid = rows["valid"][idx]
        std = truth[valid].std() + 1e-8  # valid (frame, step) entries only

        episode = rows["episode_index"][idx]
        frame = rows["frame_index"][idx]
        same_episode = episode[:, None] == episode[None, :]
        near_in_time = np.abs(frame[:, None] - frame[None, :]) < CHUNK_SIZE
        excluded = (same_episode & near_in_time) | np.eye(len(idx), dtype=bool)

        for a in range(len(idx)):
            distances = np.where(excluded[a], np.inf, cosine_dist[a])
            k = min(TOP_K, np.isfinite(distances).sum())
            if k == 0:
                dropped += 1
                continue
            neighbors = np.argpartition(distances, k - 1)[:k]
            both = valid[a][None, :] & valid[neighbors]
            divergence = np.abs(
                truth[neighbors] - truth[a][None],
            ).mean(axis=-1)  # [k, chunk]
            per_neighbor = np.where(both, divergence, 0.0).sum(
                axis=-1,
            ) / np.maximum(both.sum(axis=-1), 1)
            scores[idx[a]] = float(per_neighbor.mean() / std)
            qualifying[idx[a]] = True
            worst = neighbors[int(np.argmax(per_neighbor))]
            neighbor_records.append(
                {
                    "row": int(idx[a]),
                    "neighbor_row": int(idx[worst]),
                    "embed_dist": float(cosine_dist[a, worst]),
                    "divergence_std": float(per_neighbor.max() / std),
                },
            )
    print(
        f"alias scores: {qualifying.sum()} qualifying, {dropped} dropped "
        f"(pool < {MIN_ROWS} rows or no eligible neighbors)",
        flush=True,
    )
    return scores, qualifying, neighbor_records


def cmd_mine(args: argparse.Namespace) -> None:
    rows = load_panel_rows()
    z = np.load(EMBED_OUT, allow_pickle=True)
    if not np.array_equal(z["index"], rows["index"]):
        raise SystemExit("embeddings npz row order differs from panel npz")
    scores, qualifying, neighbors = alias_scores(rows, z["embedding"])

    delta = rows["delta"]
    pool = qualifying & np.isfinite(delta)
    bar = float(np.quantile(scores[pool], FLAG_QUANTILE))
    flagged = pool & (scores >= bar)
    rest = pool & (scores < bar)

    headline = float(delta[flagged].mean() - delta[rest].mean())
    rng = np.random.default_rng(SEED)
    repos = rows["repo_id"][pool]
    unique_repos = np.unique(repos)
    draws = np.empty(BOOTSTRAP)
    pool_idx = np.flatnonzero(pool)
    by_repo = {r: pool_idx[repos == r] for r in unique_repos}
    for b in range(BOOTSTRAP):
        chosen = rng.choice(len(unique_repos), len(unique_repos))
        members = np.concatenate([by_repo[unique_repos[c]] for c in chosen])
        f = flagged[members]
        draws[b] = (
            delta[members][f].mean() - delta[members][~f].mean()
            if f.any() and (~f).any()
            else np.nan
        )
    draws = draws[np.isfinite(draws)]
    ci = [float(np.quantile(draws, q)) for q in (0.025, 0.975)]

    def ranks(values: np.ndarray) -> np.ndarray:
        # Average ranks for ties (no scipy in the project env).
        order = np.argsort(values, kind="stable")
        ranked = np.empty(len(values))
        ranked[order] = np.arange(len(values), dtype=float)
        sorted_values = values[order]
        boundaries = np.flatnonzero(
            np.diff(sorted_values, prepend=np.nan, append=np.nan) != 0,
        )
        for start, stop in itertools.pairwise(boundaries):
            ranked[order[start:stop]] = (start + stop - 1) / 2
        return ranked

    def spearman(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.corrcoef(ranks(a), ranks(b))[0, 1])

    rho = spearman(scores[pool], delta[pool])
    baseline_mae = rows["baseline_mae"]
    copy_mae = rows["copy_mae"]

    deciles = np.quantile(scores[pool], np.linspace(0, 1, 11))
    curve = []
    for d in range(10):
        lo, hi = deciles[d], deciles[d + 1]
        members = pool & (scores >= lo) & (scores <= hi if d == 9 else scores < hi)
        curve.append(
            {
                "decile": d + 1,
                "n": int(members.sum()),
                "alias_score_mid": float(np.median(scores[members])),
                "delta_mean": float(delta[members].mean()),
            },
        )

    result = {
        "instrument": "fontaine/scripts/frame_mining.py mine",
        "protocol": "papers/observation-aliasing.md (2605.14712 diagnostic, reversed)",
        "tower": TOWER,
        "pinned": {
            "top_k": TOP_K,
            "min_rows": MIN_ROWS,
            "flag_quantile": FLAG_QUANTILE,
            "bootstrap": BOOTSTRAP,
            "seed": SEED,
            "primary": "delta_oracle(flagged) - delta_oracle(rest), "
            "dataset-clustered bootstrap CI95",
            "secondary": "spearman(alias_score, delta_oracle)",
        },
        "pool": {
            "qualifying_frames": int(pool.sum()),
            "datasets": len(unique_repos),
            "flagged": int(flagged.sum()),
            "alias_score_bar": bar,
        },
        "delta_oracle": {
            "flagged_mean": float(delta[flagged].mean()),
            "rest_mean": float(delta[rest].mean()),
            "difference": headline,
            "ci95": ci,
            "spearman_rho": rho,
        },
        "decile_curve": curve,
        "validation": {
            "note": "instrument-level: DSSP Prop 4.2 predicts elevated "
            "reactive-policy error on aliased frames; state-copy tracks "
            "intrinsic motion difficulty — the score partly conflates "
            "'ambiguous' with 'dynamic', carried as a caveat",
            "spearman_alias_vs_baseline_mae": spearman(
                scores[pool],
                baseline_mae[pool],
            ),
            "spearman_alias_vs_copy_mae": spearman(scores[pool], copy_mae[pool]),
            "baseline_mae_flagged": float(baseline_mae[flagged].mean()),
            "baseline_mae_rest": float(baseline_mae[rest].mean()),
            "copy_mae_flagged": float(copy_mae[flagged].mean()),
            "copy_mae_rest": float(copy_mae[rest].mean()),
        },
    }
    MINE_OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["delta_oracle"], indent=1))

    neighbors.sort(key=lambda r: -r["divergence_std"])
    seen_pairs: set[frozenset[int]] = set()
    per_repo: dict[str, int] = defaultdict(int)
    top = []
    for record in neighbors:
        if record["embed_dist"] >= 0.05:
            continue
        pair = frozenset((record["row"], record["neighbor_row"]))
        repo = str(rows["repo_id"][record["row"]])
        # Mirrored duplicates (A→B and B→A) and single-dataset floods
        # both waste contact-sheet slots.
        if pair in seen_pairs or per_repo[repo] >= 2:
            continue
        seen_pairs.add(pair)
        per_repo[repo] += 1
        top.append(record)
        if len(top) >= args.top_pairs:
            break
    np.savez_compressed(
        FLAGGED_OUT,
        alias_score=scores,
        qualifying=qualifying,
        flagged=flagged,
        delta=delta,
        index=rows["index"],
        repo_id=rows["repo_id"],
        episode_index=rows["episode_index"],
        frame_index=rows["frame_index"],
        top_pairs=np.array(json.dumps(top)),
    )
    print(f"wrote {MINE_OUT} and {FLAGGED_OUT} ({len(top)} contact-sheet pairs)")


def cmd_sheet(args: argparse.Namespace) -> None:
    """Contact sheet of the top aliased pairs: query frame next to its
    most-divergent close visual neighbor, captioned with the measured
    numbers — the owner's 'start-vs-end indistinguishable' frames,
    found automatically (CPU; refetches only the sheet's frames)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from bijou.data import EpisodeSplit, select_datasets

    z = np.load(FLAGGED_OUT, allow_pickle=True)
    pairs = json.loads(str(z["top_pairs"]))[: args.pairs]
    if not pairs:
        raise SystemExit("no banked pairs — run mine first")
    repo_ids = z["repo_id"]
    episodes = z["episode_index"]
    frames = z["frame_index"]
    indices = z["index"]
    deltas = z["delta"]
    scores = z["alias_score"]

    selection = select_datasets(
        (DATA_ROOT,),
        (),
        CHUNK_SIZE,
        episode_split=EpisodeSplit.HOLDOUT,
        holdout_fraction=HOLDOUT_FRACTION,
        split_seed=SPLIT_SEED,
        allowed_fps=FPS,
        allowed_camera_counts=CAMERA_COUNTS,
    )
    dataset = selection.concat()

    def frame_image(row: int) -> np.ndarray:
        item = dataset[int(indices[row])]
        key = min(k for k in item if k.startswith("observation.images."))
        image = item[key]
        return (image.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()

    columns = 2  # pairs per row
    rows_needed = (len(pairs) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows_needed,
        columns * 2,
        figsize=(columns * 2 * 3.1, rows_needed * 2.9),
        dpi=110,
    )
    fig.patch.set_facecolor("#fcfcfb")
    axes = np.atleast_2d(axes)
    for slot, pair in enumerate(pairs):
        a, b = pair["row"], pair["neighbor_row"]
        grid_row, grid_col = divmod(slot, columns)
        for side, row in enumerate((a, b)):
            ax = axes[grid_row][grid_col * 2 + side]
            ax.imshow(frame_image(row))
            ax.set_xticks(())
            ax.set_yticks(())
            for spine in ax.spines.values():
                spine.set_color("#e5e4e0")
            label = "query" if side == 0 else "neighbor"
            ax.set_title(
                f"{label} · ep {episodes[row]} f {frames[row]} · "
                f"Δ_oracle {deltas[row]:+.2f}",
                fontsize=7,
                color="#52514e",
            )
        axes[grid_row][grid_col * 2].set_xlabel(
            f"{str(repo_ids[a])[:44]}\n"
            f"alias {scores[a]:.2f} · embed dist {pair['embed_dist']:.4f} · "
            f"divergence {pair['divergence_std']:.2f}σ",
            fontsize=7,
            color="#52514e",
        )
    for slot in range(len(pairs), rows_needed * columns):
        grid_row, grid_col = divmod(slot, columns)
        for side in range(2):
            axes[grid_row][grid_col * 2 + side].axis("off")
    fig.suptitle(
        "Aliased frames, mined: visually near-identical, divergent "
        "ground-truth continuations",
        fontsize=11,
        color="#0b0b0b",
        y=1.002,
    )
    fig.tight_layout()
    out = REPO_ROOT / "fontaine/blog/src/img/framemining/contact_sheet.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"wrote {out.relative_to(REPO_ROOT)} ({len(pairs)} pairs)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    embed = sub.add_parser("embed")
    embed.add_argument("--device", default="cuda")
    embed.add_argument("--batch-size", type=int, default=16)
    embed.add_argument("--num-workers", type=int, default=12)
    embed.add_argument("--limit", type=int, default=None)
    embed.set_defaults(func=cmd_embed)
    mine = sub.add_parser("mine")
    mine.add_argument("--top-pairs", type=int, default=24)
    mine.set_defaults(func=cmd_mine)
    sheet = sub.add_parser("sheet")
    sheet.add_argument("--pairs", type=int, default=12)
    sheet.set_defaults(func=cmd_sheet)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
