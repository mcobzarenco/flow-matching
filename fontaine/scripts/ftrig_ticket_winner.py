"""ftrig rig-holdout ticket selection + winner export (owner request
2026-08-09 15:25Z).

Consumes the ticket-bank draws dump from
eval_ftrig_rig_ticketbank64.sh (draw d at every frame IS bank ticket
d — the R1 golden-ticket screen contract) and:

  1. scores each ticket: valid-element-weighted pooled chunk MAE over
     the rig holdout (box_batch_results pooling — per-frame mean over
     valid elements, pooled by valid count);
  2. argmin -> the winner, exported byte-verbatim as
     plans/ticket_ftrig4k_rig_winner.npz in the house ticket format
     ({'tickets': float32 [1, chunk, dim]}) — consumable by BOTH the
     eval CLI's --noise-tickets and bijou.rollout --noise-ticket;
  3. quotes ticket 33's rank in the same table (the teacher-panel
     winner lives at bank index 33 — the owner wants both vectors,
     and this read prices the transfer question for free);
  4. banks the full ladder + provenance to
     reports/analysis__ftrig_ticket_selection.json.

Record-only deployment support, not a leaderboard read: the numbers
are rig-holdout offline MAE under the _ticket policy suffix; any
physical claim comes from the owner's rollouts.

Execution guards (hard aborts): bank sha in the dump matches the bank
file (load_tickets' own hashing, byte-level); draws axis == bank
count; policy name carries the _ticket suffix; no NaN/inf in draws on
valid steps; winner file round-trips byte-identical to the bank row.

Oracle mode (--oracle, synthetic pre-data): planted best ticket
recovered at the right index with the exact pooled value; degenerate
all-invalid frames dropped; poisoned invalid steps leak nowhere;
abort battery (sha mismatch, draws/bank count mismatch, non-ticket
policy, NaN in valid region).
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent

sys.path.insert(0, str(_ROOT))

from bijou.eval.policies import load_tickets


def _sibling(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DEFAULT_DRAWS = (
    _ROOT / "reports/eval__fontaine_flow_snapdistill_ftrig_4k_1xh100__step_004000"
    "__rig_holdout_1nfe_euler1_ticketbank64_draws.npz"
)
DEFAULT_BANK = _ROOT / "plans/tickets_goldenticket_m64.npz"
DEFAULT_OUT = _ROOT / "plans/ticket_ftrig4k_rig_winner.npz"
DEFAULT_JSON = _ROOT / "reports/analysis__ftrig_ticket_selection.json"
TEACHER_WINNER_INDEX = 33  # tickets_goldenticket_winner33.npz == bank[33]


def per_ticket_pooled_mae(
    draws: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """[tickets] valid-element-weighted pooled chunk MAE.

    draws [N, T, S, D], truth [N, S, D], valid [N, S] — per frame the
    mean |err| over valid steps x dims, pooled across frames weighted
    by each frame's valid-element count (the banked chunk_mae
    convention). Invalid positions are zero-filled BEFORE any sum —
    NaN poison there must leak nowhere (oracle-pinned)."""
    mask = valid[:, None, :, None]  # broadcasts over tickets and dims
    if not np.isfinite(np.where(mask, draws, 0.0)).all():
        raise SystemExit("ABORT: NaN/inf in draws on valid steps")
    err = np.abs(draws.astype(np.float64) - truth.astype(np.float64)[:, None])
    per_frame_sum = np.where(mask, err, 0.0).sum(axis=(2, 3))  # [N, T]
    nvalid = valid.sum(axis=1) * draws.shape[3]  # [N]
    keep = nvalid > 0
    if not keep.any():
        raise SystemExit("ABORT: no frames with valid steps")
    return per_frame_sum[keep].sum(axis=0) / nvalid[keep].sum()


def analyze(
    draws_path: Path,
    bank_path: Path,
    out_path: Path,
    json_path: Path,
) -> dict:
    d = dict(np.load(draws_path, allow_pickle=True))
    bank, bank_sha = load_tickets(bank_path)
    policy = str(d["policy"])
    if not policy.endswith("_ticket"):
        raise SystemExit(
            f"ABORT: policy {policy!r} lacks the _ticket suffix — this dump "
            "was not produced in ticket mode",
        )
    dump_sha = str(d["tickets_sha256"])
    if dump_sha != bank_sha:
        raise SystemExit(
            f"ABORT: dump tickets_sha256 {dump_sha[:12]}… != bank file sha "
            f"{bank_sha[:12]}… — wrong bank for this dump",
        )
    draws = d["draws"]
    if draws.shape[1] != bank.shape[0]:
        raise SystemExit(
            f"ABORT: {draws.shape[1]} draws != {bank.shape[0]} bank tickets "
            "— draw d IS ticket d only at full-bank draws",
        )
    ladder = per_ticket_pooled_mae(draws, d["truth"], d["valid"].astype(bool))
    order = np.argsort(ladder, kind="stable")
    winner = int(order[0])
    np.savez(out_path, tickets=bank.numpy()[winner : winner + 1])
    reread, out_sha = load_tickets(out_path)
    if not (reread.numpy() == bank.numpy()[winner : winner + 1]).all():
        raise SystemExit("ABORT: winner file does not round-trip byte-identical")
    result = {
        "read": "ftrig_ticket_selection",
        "draws_npz": str(draws_path),
        "bank": {"path": str(bank_path), "sha256": bank_sha},
        "policy": policy,
        "n_frames": int(d["truth"].shape[0]),
        "winner": {
            "index": winner,
            "pooled_mae": round(float(ladder[winner]), 5),
            "file": str(out_path),
            "file_sha256": out_sha,
        },
        "bank_pooled_mae": {
            "min": round(float(ladder.min()), 5),
            "median": round(float(np.median(ladder)), 5),
            "max": round(float(ladder.max()), 5),
            "sd": round(float(ladder.std()), 5),
        },
        # Only meaningful against the real m64 bank (fixtures are small).
        "teacher_winner_ticket33": (
            {
                "index": TEACHER_WINNER_INDEX,
                "pooled_mae": round(float(ladder[TEACHER_WINNER_INDEX]), 5),
                "rank_of_64": int(
                    np.nonzero(order == TEACHER_WINNER_INDEX)[0][0] + 1,
                ),
            }
            if len(ladder) > TEACHER_WINNER_INDEX
            else None
        ),
        "top5": [
            {"index": int(i), "pooled_mae": round(float(ladder[i]), 5)}
            for i in order[:5]
        ],
        "ladder": [round(float(x), 5) for x in ladder],
    }
    json_path.write_text(json.dumps(result, indent=1))
    w = result["winner"]
    t33 = result["teacher_winner_ticket33"]
    print(f"policy {policy}, {result['n_frames']} frames, bank {bank_sha[:12]}…")
    print(
        f"WINNER ticket {w['index']}: pooled MAE {w['pooled_mae']} "
        f"(bank min/median/max {result['bank_pooled_mae']['min']}/"
        f"{result['bank_pooled_mae']['median']}/{result['bank_pooled_mae']['max']})",
    )
    if t33 is not None:
        print(
            f"teacher winner (ticket 33): {t33['pooled_mae']} — "
            f"rank {t33['rank_of_64']}/64 on the rig checkpoint",
        )
    print(f"wrote {w['file']} (sha256 {w['file_sha256']})")
    print(f"wrote {json_path}")
    return result


# ---------------------------------------------------------------- oracle


def _expect_abort(fn, needle: str, label: str) -> None:  # noqa: ANN001
    try:
        fn()
    except SystemExit as e:
        assert needle in str(e), f"{label}: wrong abort message: {e}"
        print(f"oracle abort OK: {label}")
        return
    raise AssertionError(f"{label}: expected abort not raised")


def oracle(tmp: Path) -> None:
    rng = np.random.default_rng(7)
    n_frames, n_tickets, chunk, dim = 6, 5, 4, 2
    bank = rng.standard_normal((n_tickets, chunk, dim)).astype(np.float32)
    bank_path = tmp / "bank.npz"
    np.savez(bank_path, tickets=bank)
    _, bank_sha = load_tickets(bank_path)

    truth = rng.standard_normal((n_frames, chunk, dim)).astype(np.float32)
    valid = np.ones((n_frames, chunk), bool)
    valid[0, 2:] = False  # partial frame
    # Planted best: ticket 3's draws sit exactly 0.125 off truth,
    # every other ticket exactly 1.0 off (valid-masked pooling is then
    # hand-computable: pooled MAE == the planted offset).
    draws = np.ones((n_frames, n_tickets, chunk, dim), np.float32)
    for t in range(n_tickets):
        off = 0.125 if t == 3 else 1.0
        draws[:, t] = truth + off
    # Poison invalid steps: must leak nowhere.
    draws[0, :, 2:, :] = np.nan
    d_path = tmp / "draws.npz"

    def write_dump(**over: object) -> Path:
        payload: dict = {
            "draws": draws,
            "truth": truth,
            "valid": valid,
            "policy": np.array("bijou@4000_ticket"),
            "tickets_sha256": np.array(bank_sha),
        }
        payload.update(over)
        np.savez(d_path, **payload)
        return d_path

    write_dump()
    r = analyze(d_path, bank_path, tmp / "win.npz", tmp / "sel.json")
    assert r["winner"]["index"] == 3
    assert r["winner"]["pooled_mae"] == 0.125
    assert all(row == 1.0 for i, row in enumerate(r["ladder"]) if i != 3), r["ladder"]
    reread, _ = load_tickets(tmp / "win.npz")
    assert (reread.numpy()[0] == bank[3]).all()
    print("oracle OK: planted best ticket recovered exactly, export byte-true")

    write_dump(policy=np.array("bijou@4000"))
    _expect_abort(
        lambda: analyze(d_path, bank_path, tmp / "w2.npz", tmp / "s2.json"),
        "_ticket suffix",
        "non-ticket policy",
    )
    write_dump(tickets_sha256=np.array("deadbeef"))
    _expect_abort(
        lambda: analyze(d_path, bank_path, tmp / "w2.npz", tmp / "s2.json"),
        "wrong bank",
        "sha mismatch",
    )
    write_dump(draws=draws[:, :3])
    _expect_abort(
        lambda: analyze(d_path, bank_path, tmp / "w2.npz", tmp / "s2.json"),
        "bank tickets",
        "draws/bank count mismatch",
    )
    bad = draws.copy()
    bad[1, 0, 0, 0] = np.nan
    write_dump(draws=bad)
    _expect_abort(
        lambda: analyze(d_path, bank_path, tmp / "w2.npz", tmp / "s2.json"),
        "NaN/inf",
        "NaN in valid region",
    )
    print("ORACLE PASS: ftrig ticket selection verified pre-data")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--draws-npz", type=Path, default=DEFAULT_DRAWS)
    ap.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    args = ap.parse_args()
    if args.oracle:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            oracle(Path(tmp))
        return
    analyze(args.draws_npz, args.bank, args.out, args.json)


if __name__ == "__main__":
    main()
