"""Noise-ladder rung 2, stages 0-1 — the CPU half of the per-dataset
golden-tickets pre-reg (posts/2026-08-08-prereg-noise-ladder-perdataset.md).

Implements exactly the frozen stage-0/1 design; every quantity is a pure
function of the banked stage-1 draws npz + the banked ticket33 full-panel
npz (complement row counts). Zero GPU.

  * Stage 0 — reliability floor F from split-half self-consistency:
    for every dataset with >= 4 probe core frames, split frames by
    frame-index parity (even -> half A, odd -> half B; deterministic,
    no seed knob); argmin ticket on half A; regret on half B against
    half B's own argmin. Regrets pool by exact cell size n; each n-bin's
    observed median is compared against a permutation null (ticket
    labels shuffled within dataset -> the half-A selection becomes a
    uniformly random label, 1,000 permutations, seed 0): the bin PASSES
    iff observed median < the null medians' 5th percentile (strict).
    F = the smallest passing n. No passing bin => the rung CLOSES at
    CPU cost (the pre-reg's first frozen decision rule).
    Datasets whose parity split leaves an empty half cannot enter the
    stage-0 bins; they are counted and listed, and still qualify for
    routing on the (>= F frames, >= 20 complement rows) rule, which
    does not require splittability.
  * Stage 1 — routing map (only if F exists): qualifying datasets
    (>= F probe frames AND >= 20 held-out complement core rows) route
    to the argmin of their full-cell pooled probe MAE restricted to
    the global top-10 ticket set; exact-equality ties break toward the
    global winner 33 if tied, else toward the lowest ticket id
    (deterministic). Non-qualifying datasets route to ticket 33.
    Output: the full dataset -> ticket map + its sha256, the
    qualifying set, and its panel-core / complement-core row weights.

Integrity (abort, never silent): the probe npz must pass the stage-1
provenance gate (m64 bank sha, _ticket policy, 64-draw axis — verbatim
ticket_scores.check_provenance); recomputed per-ticket pooled scores
must match the banked stage-1 json at its own 4 dp; the recomputed
top-10 set must equal the banked top10_tickets; the complement row
count must match the banked stage-2 json.

Oracle mode (--oracle, run before trusting any stage-0/1 number):
  (a) pooling + complement reproduction against the banked artifacts;
  (b) planted split-half worlds: a signal world (one ticket best in
      both halves everywhere) yields zero regret and a PASSING bin; a
      null world (iid ticket scores) yields a bin that does NOT pass;
      an observed median exactly AT the 5th percentile does NOT pass
      (the rule is strict <);
  (c) provenance refusals fire (wrong bank sha via the stage-1 gate);
  (d) routing rules: restriction to top-10 binds even when a non-top-10
      ticket is globally lower; exact ties containing 33 resolve to 33;
      non-qualifying datasets route to 33.

Pure CPU, read-only on inputs, deterministic.

  uv run python fontaine/scripts/noise_ladder_stage01.py \\
      --out reports/analysis__noise_ladder_stage01.json
  uv run python fontaine/scripts/noise_ladder_stage01.py --oracle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
from goldenticket_stage2_results import complement_mask, load_npz
from ticket_scores import check_provenance, per_ticket_scores

RUN_STEM = "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000"
PROBE_NPZ = f"{RUN_STEM}__drawsprobe_s7_ticket_draws64_heun30_draws.npz"
PANEL_NPZ = f"{RUN_STEM}__panel_curated_v0_k4l2_ticket33_heun30.npz"
TICKETS = "plans/tickets_goldenticket_m64.npz"
STAGE1_JSON = "reports/analysis__goldenticket_stage1.json"
STAGE2_JSON = "reports/analysis__goldenticket_stage2.json"

# Frozen constants — from the pre-reg text, never derived at read time.
MIN_SPLIT_FRAMES = 4  # split-half needs >= 4 probe frames
MIN_COMPLEMENT_ROWS = 20  # the confirm needs rows to judge on
N_PERMUTATIONS = 1_000
PERM_SEED = 0
NULL_PCTL = 5.0
GLOBAL_WINNER = 33


def _fail(message: str) -> None:
    raise SystemExit(f"ABORT: {message}")


def dataset_scores(
    draws: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    repo: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-dataset pooled per-ticket chunk MAE (fairness pooling)."""
    out: dict[str, np.ndarray] = {}
    for repo_id in sorted(set(repo.tolist())):
        rows = repo == repo_id
        chunk, _ = per_ticket_scores(draws[rows], truth[rows], valid[rows])
        out[repo_id] = chunk
    return out


def split_half(
    draws: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    frame_index: np.ndarray,
    repo: np.ndarray,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Half-A selection + half-B scores for every splittable dataset
    with >= MIN_SPLIT_FRAMES frames; returns (cells, unsplittable)."""
    cells: dict[str, dict[str, Any]] = {}
    unsplittable: list[str] = []
    for repo_id in sorted(set(repo.tolist())):
        rows = repo == repo_id
        n = int(rows.sum())
        if n < MIN_SPLIT_FRAMES:
            continue
        parity = frame_index[rows] % 2
        half_a, half_b = parity == 0, parity == 1
        if not half_a.any() or not half_b.any():
            unsplittable.append(repo_id)
            continue
        d, t, v = draws[rows], truth[rows], valid[rows]
        scores_a, _ = per_ticket_scores(d[half_a], t[half_a], v[half_a])
        scores_b, _ = per_ticket_scores(d[half_b], t[half_b], v[half_b])
        sel_a = int(np.argmin(scores_a))
        cells[repo_id] = {
            "n": n,
            "sel_a": sel_a,
            "scores_b": scores_b,
            "regret": float(scores_b[sel_a] - scores_b.min()),
        }
    return cells, unsplittable


def stage0_floor(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The frozen n-bin pass rule -> floor F (or None: rung closes)."""
    rng = np.random.default_rng(PERM_SEED)
    ids = sorted(cells)
    m_tickets = len(cells[ids[0]]["scores_b"]) if ids else 0
    # Null: within-dataset label shuffle makes the half-A pick a
    # uniformly random label; draw all selections up front (seeded).
    null_sel = rng.integers(0, m_tickets, size=(N_PERMUTATIONS, len(ids)))
    null_regret = np.empty((N_PERMUTATIONS, len(ids)))
    for j, repo_id in enumerate(ids):
        scores_b = cells[repo_id]["scores_b"]
        null_regret[:, j] = scores_b[null_sel[:, j]] - scores_b.min()
    bins: list[dict[str, Any]] = []
    floor = None
    for n in sorted({cells[i]["n"] for i in ids}):
        idx = [j for j, i in enumerate(ids) if cells[i]["n"] == n]
        observed = float(np.median([cells[ids[j]]["regret"] for j in idx]))
        null_medians = np.median(null_regret[:, idx], axis=1)
        line = float(np.percentile(null_medians, NULL_PCTL))
        passed = observed < line  # strict — AT the line does not pass
        bins.append(
            {
                "n": n,
                "datasets": len(idx),
                "observed_median_regret": round(observed, 4),
                "null_5th_pctl": round(line, 4),
                "null_5th_pctl_exact": line,
                "null_median_of_medians": round(
                    float(np.median(null_medians)),
                    4,
                ),
                "pass": bool(passed),
            },
        )
        if passed and floor is None:
            floor = n
    return {"bins": bins, "floor": floor}


def route_ticket(scores: np.ndarray, top10: list[int]) -> int:
    """Frozen stage-1 selector: argmin over the top-10 set; exact ties
    -> 33 if tied, else lowest ticket id."""
    sub = np.asarray([scores[t] for t in top10])
    tied = [top10[i] for i in np.flatnonzero(sub == sub.min())]
    return GLOBAL_WINNER if GLOBAL_WINNER in tied else min(tied)


def run(out_path: Path | None) -> dict[str, Any]:
    probe = np.load(REPO / PROBE_NPZ, allow_pickle=False)
    bank = np.load(REPO / TICKETS, allow_pickle=False)["tickets"]
    draws_all = probe["draws"]
    sha = check_provenance(
        probe,
        REPO / TICKETS,
        bank.shape[0],
        draws_all.shape,
    )
    core = probe["core"].astype(bool)
    draws, truth, valid = draws_all[core], probe["truth"][core], probe["valid"][core]
    repo = probe["repo_id"][core]
    frame_index = probe["frame_index"][core]

    banked1 = json.loads((REPO / STAGE1_JSON).read_text())
    chunk, _ = per_ticket_scores(draws, truth, valid)
    got = [round(float(s), 4) for s in chunk]
    if got != banked1["per_ticket_chunk_mae"]:
        _fail("recomputed per-ticket pooled scores != banked stage-1 json")
    order = np.argsort(chunk, kind="stable")
    top10 = [int(i) for i in order[:10]]
    if top10 != banked1["top10_tickets"]:
        _fail(f"recomputed top-10 {top10} != banked {banked1['top10_tickets']}")

    panel = load_npz(PANEL_NPZ)
    probe_d = {k: probe[k] for k in probe.files}
    comp = complement_mask(panel, probe_d)
    banked2 = json.loads((REPO / STAGE2_JSON).read_text())
    banked_comp = banked2["rows"]["complement"]
    if int(comp.sum()) != banked_comp:
        _fail(f"complement rows {int(comp.sum())} != banked {banked_comp}")
    comp_repo = panel["repo_id"][comp]
    comp_counts = {
        r: int((comp_repo == r).sum()) for r in sorted(set(comp_repo.tolist()))
    }
    panel_core_repo = panel["repo_id"][panel["core"].astype(bool)]

    # Stage 0.
    cells, unsplittable = split_half(draws, truth, valid, frame_index, repo)
    stage0 = stage0_floor(cells)
    floor = stage0["floor"]

    # Stage 1 (routing exists only if a floor does).
    per_ds = dataset_scores(draws, truth, valid, repo)
    frames = {r: int((repo == r).sum()) for r in per_ds}
    routing: dict[str, int] = {}
    qualifying: list[str] = []
    if floor is not None:
        for repo_id, scores in per_ds.items():
            ok = (
                frames[repo_id] >= floor
                and comp_counts.get(repo_id, 0) >= MIN_COMPLEMENT_ROWS
            )
            if ok:
                qualifying.append(repo_id)
                routing[repo_id] = route_ticket(scores, top10)
            else:
                routing[repo_id] = GLOBAL_WINNER
    map_blob = json.dumps(routing, sort_keys=True).encode()
    q_panel_rows = int(np.isin(panel_core_repo, qualifying).sum())
    q_comp_rows = int(np.isin(comp_repo, qualifying).sum())
    routed_away = [r for r in qualifying if routing[r] != GLOBAL_WINNER]

    result = {
        "inputs": {
            "probe_npz": PROBE_NPZ,
            "panel_npz": PANEL_NPZ,
            "tickets_sha256": sha,
            "core_frames": int(core.sum()),
            "datasets": len(per_ds),
            "complement_rows": int(comp.sum()),
        },
        "frozen": {
            "min_split_frames": MIN_SPLIT_FRAMES,
            "min_complement_rows": MIN_COMPLEMENT_ROWS,
            "permutations": N_PERMUTATIONS,
            "seed": PERM_SEED,
            "null_pctl": NULL_PCTL,
            "top10_tickets": top10,
        },
        "stage0": {
            "splittable_datasets": len(cells),
            "unsplittable_datasets": unsplittable,
            "bins": stage0["bins"],
            "floor": floor,
            "verdict": "FLOOR" if floor is not None else "RUNG CLOSES",
        },
        "stage1": {
            "qualifying_datasets": len(qualifying),
            "qualifying_panel_core_rows": q_panel_rows,
            "qualifying_panel_core_row_frac": round(
                q_panel_rows / max(1, len(panel_core_repo)),
                4,
            ),
            "qualifying_complement_rows": q_comp_rows,
            "routed_away_from_33": len(routed_away),
            "ticket_histogram": {
                str(t): sum(1 for r in qualifying if routing[r] == t)
                for t in top10
                if any(routing[r] == t for r in qualifying)
            },
            "routing_map_sha256": hashlib.sha256(map_blob).hexdigest(),
            "routing_map": {r: routing[r] for r in sorted(routing)},
            "qualifying": sorted(qualifying),
        }
        if floor is not None
        else {"note": "no floor — no routing map exists"},
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=1))
        print(f"wrote {out_path}")
    return result


# ---------------------------------------------------------------- oracles


def _oracle_fail(message: str) -> None:
    raise SystemExit(f"ORACLE RED: {message}")


def _planted_world(
    *,
    signal: bool,
    n_datasets: int = 40,
    n_frames: int = 4,
    m: int = 64,
) -> dict[str, dict[str, Any]]:
    """Synthetic split-half cells: signal worlds make ticket 7 best in
    both halves everywhere; null worlds draw iid ticket scores."""
    rng = np.random.default_rng(1234 if signal else 4321)
    cells: dict[str, dict[str, Any]] = {}
    for i in range(n_datasets):
        scores_a = 6.6 + rng.normal(0, 0.1, size=m)
        scores_b = 6.6 + rng.normal(0, 0.1, size=m)
        if signal:
            scores_a[7] = scores_a.min() - 0.5
            scores_b[7] = scores_b.min() - 0.5
        sel_a = int(np.argmin(scores_a))
        cells[f"ds{i:03d}"] = {
            "n": n_frames,
            "sel_a": sel_a,
            "scores_b": scores_b,
            "regret": float(scores_b[sel_a] - scores_b.min()),
        }
    return cells


def run_oracles() -> None:
    # (a) pooling + complement reproduction on the banked artifacts.
    result = run(out_path=None)
    if result["inputs"]["core_frames"] != 2458 or result["inputs"]["datasets"] != 792:
        _oracle_fail("banked probe shape drifted (expected 2458 frames / 792 datasets)")
    print(
        f"oracle a GREEN: stage-1 scores + top-10 + complement "
        f"{result['inputs']['complement_rows']} all match banked jsons",
    )

    # (b) planted split-half worlds.
    signal = _planted_world(signal=True)
    if any(c["regret"] != 0.0 or c["sel_a"] != 7 for c in signal.values()):
        _oracle_fail("signal world: expected sel_a == 7 and zero regret everywhere")
    s0 = stage0_floor(signal)
    if s0["floor"] != 4 or not s0["bins"][0]["pass"]:
        _oracle_fail(f"signal world did not pass its bin: {s0['bins']}")
    null = _planted_world(signal=False)
    s0n = stage0_floor(null)
    if s0n["floor"] is not None:
        _oracle_fail(f"null world produced a floor: {s0n['bins']}")
    # Strictness: a bin whose observed median sits exactly AT the 5th
    # percentile must not pass.
    at_line = {
        k: {**c, "regret": s0n["bins"][0]["null_5th_pctl_exact"]}
        for k, c in null.items()
    }
    if stage0_floor(at_line)["floor"] is not None:
        _oracle_fail("observed median exactly AT the null line passed (must be <)")
    print("oracle b GREEN: planted signal/null/at-line split-half worlds")

    # (c) provenance refusal (wrong bank => the stage-1 gate aborts).
    probe = np.load(REPO / PROBE_NPZ, allow_pickle=False)
    try:
        check_provenance(
            probe,
            REPO / "plans/holdout_curated_v0_k4l2_drawsprobe_s7.json",
            64,
            probe["draws"].shape,
        )
    except SystemExit:
        pass
    else:
        _oracle_fail("wrong-bank provenance did not abort")
    print("oracle c GREEN: provenance refusal fires")

    # (d) routing rules.
    scores = np.full(64, 7.0)
    scores[5] = 6.0  # not in top10 below — restriction must bind
    top10 = [33, 2, 0, 51, 10, 59, 38, 28, 15, 36]
    scores[2] = 6.5
    if route_ticket(scores, top10) != 2:
        _oracle_fail("restricted argmin did not pick the best top-10 ticket")
    tie = np.full(64, 7.0)
    tie[33] = tie[2] = 6.5
    if route_ticket(tie, top10) != 33:
        _oracle_fail("exact tie containing 33 did not resolve to 33")
    tie2 = np.full(64, 7.0)
    tie2[2] = tie2[51] = 6.5
    if route_ticket(tie2, top10) != 2:
        _oracle_fail("exact tie without 33 did not resolve to the lowest id")
    print("oracle d GREEN: routing restriction + tie-breaks")
    print("ALL ORACLES GREEN")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--oracle", action="store_true")
    args = parser.parse_args()
    if args.oracle:
        run_oracles()
        return
    result = run(args.out)
    s0, s1 = result["stage0"], result["stage1"]
    print(f"stage 0: {s0['verdict']} floor={s0['floor']}")
    for b in s0["bins"]:
        print(
            f"  n={b['n']:>3} datasets={b['datasets']:>3} "
            f"median={b['observed_median_regret']:.4f} "
            f"null5={b['null_5th_pctl']:.4f} pass={b['pass']}",
        )
    if s0["floor"] is not None:
        print(
            f"stage 1: {s1['qualifying_datasets']} qualifying, "
            f"{s1['qualifying_panel_core_rows']} panel core rows "
            f"({s1['qualifying_panel_core_row_frac']:.1%}), "
            f"{s1['qualifying_complement_rows']} complement rows, "
            f"{s1['routed_away_from_33']} routed away from 33, "
            f"map sha {s1['routing_map_sha256'][:16]}…",
        )


if __name__ == "__main__":
    main()
