"""Carrier-hunt rung-3 pre-reg measured basis (CPU, record-only).

Queue item `carrier-hunt-rung3-prereg`, drafted DURING the ch0fix ride
(2026-08-22 23:xxZ) so the verdict session executes instead of
drafting. Two reads, both frozen into the rung-3 draft
(posts/2026-08-22-prereg-carrier-hunt-rung3.md):

1. Per-episode ch0/ch5 anomaly table for the 7 clean episodes —
   is the spread compression episode-concentrated (suspects-first
   slicing) or uniform (frame-balanced bisection)?
2. Holdout-draw suffix search (`bijou.data.holdout_episodes`
   reimplemented verbatim) pinning the three candidate dataset names
   so every branch keeps the intended train split — the gripfix
   Amendment-1 class, pre-applied:
   - branch A  (7 eps, want draw (2,)): action-only ch0 affine;
   - branch B  (4 eps [0,1,2,5], want (2,) = decoy ep2): bisection
     cell training exactly {0,1,5};
   - branch B follow-up (4 eps [2,3,4,6], want (0,) = decoy ep2):
     complement cell training exactly {3,4,6}.

Output: reports/analysis__carrier_rung3_basis.json + table on stdout.
Sanity oracles: the search reproduces democlean's and ch0fix_n's
verified (2,) draws; demos ch0 action mean/std reproduce the banked
affine constants (0.0923439813196304 / 27.988688177087553) to 1e-12.
"""

from __future__ import annotations

import json
import random
import string
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "reports/analysis__carrier_rung3_basis.json"

CLEAN = "~/datasets/mcobzarenco/so101_pick_place_clean"
DEMOS = "~/datasets/fontaine/grasp_demos_v2/merged"
# Banked rung-2 affine constants (2026-08-22-prereg-clean-ch0-affine.md).
DEMOS_CH0_MEAN = 0.0923439813196304
DEMOS_CH0_STD = 27.988688177087553


def load_actions(root: str) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(Path(root).expanduser().glob("data/**/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_values("index")
    return np.stack(df["action"].to_numpy()).astype(np.float64), df[
        "episode_index"
    ].to_numpy()


def ks(a: np.ndarray, b_sorted: np.ndarray) -> float:
    a = np.sort(a)
    allv = np.concatenate([a, b_sorted])
    ca = np.searchsorted(a, allv, side="right") / len(a)
    cb = np.searchsorted(b_sorted, allv, side="right") / len(b_sorted)
    return float(np.abs(ca - cb).max())


def holdout_episodes(
    repo_id: str,
    n: int,
    fraction: float = 0.1,
    seed: int = 0,
) -> tuple[int, ...]:
    """bijou/data.py::holdout_episodes, reimplemented verbatim."""
    if fraction <= 0 or n < 2:
        return ()
    count = min(n - 1, max(1, round(fraction * n)))
    rng = random.Random(f"{seed}:{repo_id}")
    return tuple(sorted(rng.sample(range(n), count)))


def suffix_search(base: str, n: int, want: tuple) -> str:
    suffixes = (
        [""]
        + ["_" + c for c in string.ascii_lowercase]
        + ["_" + c + d for c in string.ascii_lowercase for d in string.ascii_lowercase]
    )
    for suf in suffixes:
        if holdout_episodes(base + suf, n) == want:
            return base + suf
    raise RuntimeError(f"no suffix found for {base} n={n} want={want}")


def main() -> None:
    act, ep = load_actions(CLEAN)
    d_act, _ = load_actions(DEMOS)
    d0 = np.sort(d_act[:, 0])

    assert abs(d_act[:, 0].mean() - DEMOS_CH0_MEAN) < 1e-12
    assert abs(d_act[:, 0].std() - DEMOS_CH0_STD) < 1e-12

    per_ep = {}
    print(
        f"{'ep':>3} {'len':>5} {'ch0a_mean':>10} {'ch0a_std':>9} "
        f"{'ch0a_q01':>9} {'ch0a_q99':>9} {'ch5a_max':>9} {'KS_ch0_vs_demos':>15}",
    )
    for e in sorted(set(ep.tolist())):
        m = ep == e
        a0, a5 = act[m, 0], act[m, 5]
        q01, q99 = np.quantile(a0, [0.01, 0.99])
        row = {
            "len": int(m.sum()),
            "ch0a_mean": float(a0.mean()),
            "ch0a_std": float(a0.std()),
            "ch0a_q01": float(q01),
            "ch0a_q99": float(q99),
            "ch5a_max": float(a5.max()),
            "ks_ch0_action_vs_demos": ks(a0, d0),
        }
        per_ep[int(e)] = row
        print(
            f"{e:>3} {row['len']:>5} {row['ch0a_mean']:>10.3f} "
            f"{row['ch0a_std']:>9.3f} {q01:>9.2f} {q99:>9.2f} "
            f"{row['ch5a_max']:>9.2f} {row['ks_ch0_action_vs_demos']:>15.3f}",
        )

    # Sanity: the two names whose draws are already verified in landed specs.
    assert holdout_episodes("mcobzarenco/so101_pick_place_clean", 7) == (2,)
    assert holdout_episodes("mcobzarenco/so101_pick_place_clean_ch0fix_n", 7) == (2,)

    names = {
        "branch_a_action_only": suffix_search(
            "mcobzarenco/so101_pick_place_clean_ch0fix_act",
            7,
            (2,),
        ),
        "branch_b_cell": suffix_search(
            "mcobzarenco/so101_pick_place_clean_ep015",
            4,
            (2,),
        ),
        "branch_b_complement": suffix_search(
            "mcobzarenco/so101_pick_place_clean_ep346",
            4,
            (0,),
        ),
    }
    for k, v in names.items():
        print(k, "->", v)

    REPORT.write_text(json.dumps({"per_episode": per_ep, "names": names}, indent=1))
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
