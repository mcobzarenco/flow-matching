"""Golden-ticket stage-1 scorer + R1 kill line — ready before the data.

Implements exactly the stage-1 read of the golden-ticket pre-reg
(posts/2026-08-07-prereg-golden-ticket-screen.md): per-ticket pooled
core-frame chunk MAE from the search eval's --dump-draws npz (draw m IS
ticket m), through the SAME pooling as the fairness reads
(draws_fairness.element_mask/pooled_mae, verbatim import).

  * Scores: per-ticket pooled core chunk MAE + first-step MAE (the
    tie-break); sample sd (ddof=1), min, mean over the M tickets.
  * R1 (frozen): ticket structure is worth confirming iff
    sd > 0.0785 (upper 95% chi2_63 edge of sigma_probe 0.0669) OR
    min < mean - 0.22 (expected null min mean-0.157, minus 2 sd of the
    min). Otherwise KILL before stage 2 — the screen closes.
  * Winner = argmin (tie-break: lower first_mae); top-10 = the 10
    lowest (stage 3's mean-of-top-10 set).
  * R4a (record-only): per-dataset per-ticket score matrix + each
    dataset's argmin and margin — the task-locality read; selection
    noise caveat applies (each per-dataset winner needs its own
    held-out confirm before anyone trusts it).

Provenance refusals (abort, never silent): the npz must carry the
ticket-mode fields (tickets_sha256 matching the bank file, policy name
carrying _ticket, sample_draws == bank count == draws-axis length) — a
keyed-noise npz must never pool as a ticket read.

Oracle mode (--oracle, run before any ticket data exists — the
pre-reg's oracle 4 plus scorer self-checks):
  (a) pooling reuse: the banked stable-key full-panel npz pooled on
      core rows through THIS file reproduces 6.5997 exactly (4 dp);
      the banked drawsprobe draws10 npz per-draw reproduces the 10
      per-draw pooled MAEs of analysis__sigma_draw_direct.json exactly;
  (b) R1 branches on synthetic score vectors: a null draw (sd at
      sigma_probe, no outlier) KILLs; an injected -0.3 outlier
      CONFIRMs via the min line; an inflated-sd world CONFIRMs via the
      sd line; boundary values sit on the frozen constants;
  (c) provenance refusals fire: missing/mismatched tickets_sha256,
      policy without _ticket, draw-count mismatch;
  (d) winner tie-break: equal chunk scores resolve by lower first_mae.

Pure CPU, read-only on inputs, deterministic.

Usage:
  python fontaine/scripts/ticket_scores.py --oracle
  python fontaine/scripts/ticket_scores.py \
      --npz reports/eval__..._drawsprobe_s7_ticket64_heun30_draws.npz \
      --tickets plans/tickets_goldenticket_m64.npz \
      [--out reports/analysis__goldenticket_stage1.json]
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
sys.path.insert(0, str(_HERE))
from draws_fairness import element_mask, pooled_mae

REPO = _HERE.parents[1]

# R1 constants — frozen in the pre-reg, never derived at read time.
SD_LINE = 0.0785
MIN_MARGIN = 0.22
SIGMA_PROBE = 0.0669
NULL_MIN_SHIFT = 0.157
TOP_K = 10

# Oracle (a) anchors: the banked artifacts and their quoted values.
STABLEKEY_NPZ = (
    "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__"
    "panel_curated_v0_k4l2_stablekey_heun30.npz"
)
STABLEKEY_POLICY = "pred:bijou@80000"
STABLEKEY_CHUNK_MAE = 6.5997
DRAWSPROBE_NPZ = (
    "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__"
    "panel_curated_v0_k4l2_drawsprobe_s7_draws10_heun30.npz"
)
SIGMA_DRAW_JSON = "reports/analysis__sigma_draw_direct.json"


def per_ticket_scores(
    draws: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(pooled chunk MAE, pooled first-step MAE) per ticket for a
    [frames, M, chunk, dim] stack — fairness-read pooling verbatim."""
    mask = element_mask(truth, valid)
    chunk = np.array(
        [pooled_mae(draws[:, m], truth, mask) for m in range(draws.shape[1])],
    )
    first = np.array(
        [
            pooled_mae(draws[:, m, :1], truth[:, :1], mask[:, :1])
            for m in range(draws.shape[1])
        ],
    )
    return chunk, first


def r1_verdict(scores: np.ndarray) -> dict[str, Any]:
    """The frozen R1 decision on the M pooled ticket scores."""
    sd = float(scores.std(ddof=1))
    mean = float(scores.mean())
    minimum = float(scores.min())
    sd_open = sd > SD_LINE
    min_open = minimum < mean - MIN_MARGIN
    return {
        "sd": round(sd, 5),
        "mean": round(mean, 5),
        "min": round(minimum, 5),
        "sd_line": SD_LINE,
        "min_line": round(mean - MIN_MARGIN, 5),
        "sigma_probe_null": SIGMA_PROBE,
        "expected_null_min": round(mean - NULL_MIN_SHIFT, 5),
        "sd_open": sd_open,
        "min_open": min_open,
        "verdict": "CONFIRM" if (sd_open or min_open) else "KILL",
    }


def pick_winner(chunk: np.ndarray, first: np.ndarray) -> int:
    """Argmin of chunk score; ties (exact float equality — the frozen
    tie-break spelling) resolve by lower first_mae."""
    best = np.flatnonzero(chunk == chunk.min())
    return int(best[np.argmin(first[best])])


def check_provenance(
    npz: Any,
    tickets_path: Path,
    bank_count: int,
    draws_shape: tuple[int, ...],
) -> str:
    """Abort unless the npz is a ticket-mode dump against THIS bank;
    returns the sha256 both sides agree on."""
    for field in ("tickets_sha256", "policy", "sample_draws"):
        if field not in npz.files:
            raise SystemExit(
                f"PROVENANCE RED: npz carries no '{field}' — not a "
                "ticket-mode --dump-draws artifact",
            )
    sha = str(npz["tickets_sha256"])
    if not sha:
        raise SystemExit(
            "PROVENANCE RED: empty tickets_sha256 — this npz was dumped "
            "under keyed noise, it must never pool as a ticket read",
        )
    actual = hashlib.sha256(tickets_path.read_bytes()).hexdigest()
    if sha != actual:
        raise SystemExit(
            f"PROVENANCE RED: npz tickets_sha256 {sha[:16]}… != "
            f"{tickets_path} sha256 {actual[:16]}… — different bank",
        )
    policy = str(npz["policy"])
    if "_ticket" not in policy:
        raise SystemExit(
            f"PROVENANCE RED: policy {policy!r} carries no _ticket suffix",
        )
    if int(npz["sample_draws"]) != bank_count or draws_shape[1] != bank_count:
        raise SystemExit(
            f"PROVENANCE RED: sample_draws {int(npz['sample_draws'])} / "
            f"draws axis {draws_shape[1]} != bank count {bank_count}",
        )
    return sha


def score_run(npz_path: Path, tickets_path: Path) -> dict[str, Any]:
    npz = np.load(npz_path, allow_pickle=False)
    bank = np.load(tickets_path, allow_pickle=False)["tickets"]
    draws = npz["draws"]
    sha = check_provenance(npz, tickets_path, bank.shape[0], draws.shape)
    core = npz["core"]
    truth, valid = npz["truth"][core], npz["valid"][core]
    repo = npz["repo_id"][core]
    chunk, first = per_ticket_scores(draws[core], truth, valid)
    winner = pick_winner(chunk, first)
    order = np.argsort(chunk, kind="stable")
    per_dataset: dict[str, Any] = {}
    for repo_id in sorted(set(repo.tolist())):
        rows = repo == repo_id
        ds_chunk, _ = per_ticket_scores(draws[core][rows], truth[rows], valid[rows])
        ds_argmin = int(np.argmin(ds_chunk))
        per_dataset[repo_id] = {
            "frames": int(rows.sum()),
            "argmin_ticket": ds_argmin,
            "argmin_score": round(float(ds_chunk[ds_argmin]), 4),
            "global_winner_score": round(float(ds_chunk[winner]), 4),
            "scores": [round(float(s), 4) for s in ds_chunk],
        }
    return {
        "inputs": {
            "npz": str(npz_path),
            "tickets": str(tickets_path),
            "tickets_sha256": sha,
            "policy": str(npz["policy"]),
            "core_frames": int(core.sum()),
        },
        "per_ticket_chunk_mae": [round(float(s), 4) for s in chunk],
        "per_ticket_first_mae": [round(float(s), 4) for s in first],
        "r1": r1_verdict(chunk),
        "winner": {
            "ticket": winner,
            "chunk_mae": round(float(chunk[winner]), 4),
            "first_mae": round(float(first[winner]), 4),
        },
        "top10_tickets": [int(i) for i in order[:TOP_K]],
        "per_dataset": per_dataset,
    }


# ---------------------------------------------------------------- oracles


def _fail(message: str) -> None:
    raise SystemExit(f"ORACLE RED: {message}")


def _expect_abort(fn: Any, tag: str) -> None:
    try:
        fn()
    except SystemExit:
        return
    _fail(f"{tag}: expected a hard abort, got none")


class _FakeNpz:
    """Minimal stand-in for an np.load handle (files + [] access)."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.files = list(payload)

    def __getitem__(self, key: str) -> Any:
        return self._payload[key]


def run_oracles() -> None:
    # (a) pooling reuse against the banked artifacts — the pre-reg's
    # oracle 4. Exact at the banked values' own precision (4 dp).
    stable = np.load(REPO / STABLEKEY_NPZ, allow_pickle=False)
    core = stable["core"]
    mask = element_mask(stable["truth"][core], stable["valid"][core])
    pooled = pooled_mae(stable[STABLEKEY_POLICY][core], stable["truth"][core], mask)
    if round(pooled, 4) != STABLEKEY_CHUNK_MAE:
        _fail(f"stable-key pooled {pooled:.4f} != banked {STABLEKEY_CHUNK_MAE}")
    probe = np.load(REPO / DRAWSPROBE_NPZ, allow_pickle=False)
    pcore = probe["core"]
    chunk, _ = per_ticket_scores(
        probe["draws"][pcore],
        probe["truth"][pcore],
        probe["valid"][pcore],
    )
    banked = json.loads((REPO / SIGMA_DRAW_JSON).read_text())
    expected = banked["pooled_level_crosscheck"]["per_draw_pooled_mae"]
    got = [round(float(s), 4) for s in chunk]
    if got != expected:
        _fail(f"per-draw pooled {got} != banked {expected}")
    print(f"oracle a GREEN: stable-key core pooled {pooled:.4f}; 10 per-draw match")

    # (b) R1 branches on synthetic score vectors (seeded null draw).
    rng = np.random.default_rng(0)
    null = 6.6 + rng.normal(0.0, SIGMA_PROBE, size=64)
    while null.std(ddof=1) > SD_LINE or null.min() < null.mean() - MIN_MARGIN:
        null = 6.6 + rng.normal(0.0, SIGMA_PROBE, size=64)  # pragma: no cover
    if r1_verdict(null)["verdict"] != "KILL":
        _fail("null world did not KILL")
    outlier = null.copy()
    outlier[17] = outlier.mean() - 0.3
    verdict = r1_verdict(outlier)
    if verdict["verdict"] != "CONFIRM" or not verdict["min_open"]:
        _fail("-0.3 outlier did not CONFIRM via the min line")
    inflated = 6.6 + (null - null.mean()) * (2.5 * SD_LINE / null.std(ddof=1))
    verdict = r1_verdict(inflated)
    if verdict["verdict"] != "CONFIRM" or not verdict["sd_open"]:
        _fail("inflated-sd world did not CONFIRM via the sd line")
    boundary = r1_verdict(6.6 + (null - null.mean()) * (SD_LINE / null.std(ddof=1)))
    if boundary["sd_open"]:
        _fail("sd exactly AT the line opened (line is strict >)")
    print("oracle b GREEN: R1 null/min/sd/boundary branches")

    # (c) provenance refusals on fake handles.
    bank_path = REPO / "plans/tickets_goldenticket_m64.npz"
    sha = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    good = {
        "tickets_sha256": np.array(sha),
        "policy": np.array("bijou@80000_draws64_ticket"),
        "sample_draws": np.array(64),
    }
    shape = (10, 64, 50, 6)
    check_provenance(_FakeNpz(good), bank_path, 64, shape)
    _expect_abort(
        lambda: check_provenance(
            _FakeNpz({**good, "tickets_sha256": np.array("")}),
            bank_path,
            64,
            shape,
        ),
        "empty sha",
    )
    _expect_abort(
        lambda: check_provenance(
            _FakeNpz({**good, "tickets_sha256": np.array("f" * 64)}),
            bank_path,
            64,
            shape,
        ),
        "wrong sha",
    )
    _expect_abort(
        lambda: check_provenance(
            _FakeNpz({**good, "policy": np.array("bijou@80000_draws64")}),
            bank_path,
            64,
            shape,
        ),
        "no _ticket suffix",
    )
    _expect_abort(
        lambda: check_provenance(_FakeNpz(good), bank_path, 64, (10, 10, 50, 6)),
        "draw-count mismatch",
    )
    missing = {k: v for k, v in good.items() if k != "tickets_sha256"}
    _expect_abort(
        lambda: check_provenance(_FakeNpz(missing), bank_path, 64, shape),
        "missing field",
    )
    print("oracle c GREEN: provenance refusals")

    # (d) winner tie-break by first_mae.
    chunk = np.array([6.7, 6.5, 6.5, 6.9])
    first = np.array([1.9, 1.8, 1.7, 2.0])
    if pick_winner(chunk, first) != 2:
        _fail("tie did not resolve by lower first_mae")
    print("oracle d GREEN: winner tie-break")
    print("ALL ORACLES GREEN")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--npz", type=Path, default=None)
    parser.add_argument(
        "--tickets",
        type=Path,
        default=REPO / "plans/tickets_goldenticket_m64.npz",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--oracle", action="store_true")
    args = parser.parse_args()
    if args.oracle:
        run_oracles()
        return
    if args.npz is None:
        raise SystemExit("--npz required (or --oracle)")
    result = score_run(args.npz, args.tickets)
    r1 = result["r1"]
    print(
        f"R1: sd {r1['sd']} (line > {r1['sd_line']}), min {r1['min']} "
        f"(line < {r1['min_line']}) -> {r1['verdict']}",
    )
    print(
        f"winner ticket {result['winner']['ticket']}: chunk "
        f"{result['winner']['chunk_mae']}, first {result['winner']['first_mae']}",
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
