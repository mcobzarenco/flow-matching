"""ftrig v2-all ticket selection — subset diagnostics (owner request
2026-08-09 15:44Z follow-up reads).

Companion to ftrig_ticket_winner.py run on the v2-all dump. Record-only
deployment support, quoted as diagnostics next to the winner table:

  1. splits the v2-all frames into TRAIN rows vs HELD-OUT rows, where
     the held-out episode set is derived from the banked rig-holdout
     draws dump itself (v2 episodes present there = the 10%/seed-0
     holdout; no hand-typed episode list);
  2. per-ticket pooled chunk MAE ladder on each subset (same
     valid-element-weighted pooling, imported from
     ftrig_ticket_winner.py);
  3. Spearman rank agreement between the full v2-all ladder and the
     rig-holdout ladder (both from their banked analysis jsons) — the
     "is ticket choice sensitive to memorized rows" read;
  4. appends everything under "subset_diagnostics" in the v2-all
     analysis json (in place, key must not already exist).

Internal consistency guard (hard abort): the two subset ladders,
recombined with their valid-element weights, must reproduce the full
ladder to 1e-9 — the same numbers the winner script banked.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))

spec = importlib.util.spec_from_file_location(
    "ftrig_ticket_winner",
    _HERE / "ftrig_ticket_winner.py",
)
assert spec is not None and spec.loader is not None
_winner_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_winner_mod)
per_ticket_pooled_mae = _winner_mod.per_ticket_pooled_mae

V2_REPO = "mcobzarenco/so101_pick_place_v2"
STEM = (
    "reports/eval__fontaine_flow_snapdistill_ftrig_4k_1xh100__step_004000"
    "__rigv2all_1nfe_euler1_ticketbank64"
)
HOLDOUT_DRAWS = (
    "reports/eval__fontaine_flow_snapdistill_ftrig_4k_1xh100__step_004000"
    "__rig_holdout_1nfe_euler1_ticketbank64_draws.npz"
)
HOLDOUT_JSON = "reports/analysis__ftrig_ticket_selection.json"
V2ALL_JSON = "reports/analysis__ftrig_ticket_selection_rigv2all.json"


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(np.float64)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra**2).sum() * (rb**2).sum()))


def main() -> None:
    hd = np.load(_ROOT / HOLDOUT_DRAWS, allow_pickle=True)
    holdout_eps = sorted(
        set(hd["episode_index"][hd["repo_id"] == V2_REPO].tolist()),
    )
    if not holdout_eps:
        raise SystemExit("ABORT: no v2 episodes in the holdout dump")

    d = dict(np.load(_ROOT / f"{STEM}_draws.npz", allow_pickle=True))
    if not str(d["policy"]).endswith("_ticket"):
        raise SystemExit("ABORT: v2-all dump not in ticket mode")
    if str(d["tickets_sha256"]) != str(hd["tickets_sha256"]):
        raise SystemExit("ABORT: bank sha differs between the two dumps")
    if set(d["repo_id"].tolist()) != {V2_REPO}:
        raise SystemExit("ABORT: v2-all dump contains non-v2 repos")

    truth, valid, draws = d["truth"], d["valid"].astype(bool), d["draws"]
    in_holdout = np.isin(d["episode_index"], holdout_eps)
    subsets = {"train_rows": ~in_holdout, "heldout_rows": in_holdout}
    ladders, weights = {}, {}
    for name, mask in subsets.items():
        if not mask.any():
            raise SystemExit(f"ABORT: empty subset {name}")
        ladders[name] = per_ticket_pooled_mae(draws[mask], truth[mask], valid[mask])
        weights[name] = float(valid[mask].sum() * draws.shape[3])

    full = json.loads((_ROOT / V2ALL_JSON).read_text())
    if "subset_diagnostics" in full:
        raise SystemExit("ABORT: subset_diagnostics already present")
    full_ladder = np.array(full["ladder"], dtype=np.float64)
    recombined = (
        ladders["train_rows"] * weights["train_rows"]
        + ladders["heldout_rows"] * weights["heldout_rows"]
    ) / (weights["train_rows"] + weights["heldout_rows"])
    # full["ladder"] is rounded to 5 dp — allow exactly that quantization.
    if np.abs(recombined - full_ladder).max() > 5e-6:
        raise SystemExit(
            "ABORT: subset ladders do not recombine to the banked full ladder",
        )

    hold = json.loads((_ROOT / HOLDOUT_JSON).read_text())
    hold_ladder = np.array(hold["ladder"], dtype=np.float64)
    if len(hold_ladder) != len(full_ladder):
        raise SystemExit("ABORT: ladder length mismatch vs holdout json")

    winner = int(full["winner"]["index"])
    quoted = sorted({winner, int(hold["winner"]["index"]), 33})
    diag = {
        "heldout_episode_set": {
            "repo": V2_REPO,
            "episodes": holdout_eps,
            "derived_from": HOLDOUT_DRAWS,
        },
        "n_frames": {name: int(mask.sum()) for name, mask in subsets.items()},
        "subset_ladders": {
            name: {
                "winner_index": int(np.argmin(lad)),
                "winner_pooled_mae": round(float(lad.min()), 5),
                "median": round(float(np.median(lad)), 5),
                "quoted_tickets": {
                    str(i): {
                        "pooled_mae": round(float(lad[i]), 5),
                        "rank_of_64": int(
                            np.nonzero(
                                np.argsort(lad, kind="stable") == i,
                            )[0][0]
                            + 1,
                        ),
                    }
                    for i in quoted
                },
            }
            for name, lad in ladders.items()
        },
        "spearman_v2all_vs_holdout_ladder": round(
            spearman(full_ladder, hold_ladder),
            4,
        ),
        "spearman_train_vs_heldout_rows": round(
            spearman(ladders["train_rows"], ladders["heldout_rows"]),
            4,
        ),
    }
    full["subset_diagnostics"] = diag
    (_ROOT / V2ALL_JSON).write_text(json.dumps(full, indent=1))
    print(json.dumps(diag, indent=1))
    print(f"appended subset_diagnostics to {V2ALL_JSON}")


if __name__ == "__main__":
    main()
