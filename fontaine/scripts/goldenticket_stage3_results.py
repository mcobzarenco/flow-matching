"""Golden-ticket stage-3 R3 + R4b close-out read — ready before the data.

Implements exactly the stage-3 read of the golden-ticket pre-reg
(posts/2026-08-07-prereg-golden-ticket-screen.md, "Frozen reads" R3 +
the remaining R4 cell):

  * R3 (frozen, RECORD-ONLY either way): pooled Δ = (mean-of-top-10
    tickets core chunk MAE) − (banked mean-of-10 5.3645), pooled level
    only — the banked row's per-frame npz was not retained; both cells'
    pooled draw-noise scales are ≤ σ_draw/√10 ≈ 0.008, so a pooled
    comparison with a ±0.02 tie band is honest. Classified
    INTERESTING iff Δ ≤ −0.02 (the pre-reg's "iff" — the band edge
    itself is interesting), WORSE iff Δ ≥ +0.02 (mirror), else TIE.
    Mean-of-10's board row is NOT displaced by any outcome here.
  * R4b (record-only): dispersion-quartile geometry of the WINNER's
    per-frame gain — per-frame draw dispersion from the stage-3
    top-10-ticket stack (valid-weighted std across draws, the
    selection_ceiling_results convention), quartile-binned; each
    quartile reports the mean per-frame gain of ticket 33 vs the
    banked stable-key single draw, plus Spearman(dispersion, gain).
  * R4c mirror (record-only): per-step horizon curves on core rows for
    mean-of-top-10 / winner / stable-key (the complement-row winner
    horizon is already banked in analysis__goldenticket_stage2.json).
  * first_mae vs the banked mean-of-10 first 1.4242, record-only.

Integrity (abort, never silent): the stage-3 draws npz must carry the
top-10 tickets sha (e537f4cd…), sample_draws == 10 == stack width, and
a _ticket policy; the pooled stage-3 npz must carry _ticket and its
prediction column must equal the draws-stack mean over the draw axis
(tol 2e-3 — ties the pooled read to the ticket stack); the banked npz
must NOT carry _ticket; the winner npz must; all four npzs byte-match
on identity columns (index, repo_id, episode_index, frame_index, core,
truth, valid).

Oracle mode (--oracle, run before any stage-3 data exists):
  (a) pooling reproduction: banked stable-key core rows through THIS
      file → 6.5997; banked stage-2 ticket33 core rows → 5.6468 chunk /
      1.8963 first (the leaderboard row 7 numbers, 4 dp);
  (b) planted R3 boundaries on a synthetic panel (anchor
      parameterized): Δ −0.05 → INTERESTING; 0 → TIE; +0.05 → WORSE;
      exactly −0.02 → INTERESTING and exactly +0.02 → WORSE (the band
      edges bind outward, matching the frozen "iff ≤ −0.02");
  (c) R4b planted geometry: gain proportional to −dispersion recovers
      Spearman −1, monotone quartile means, quartile n's partition the
      kept core rows;
  (d) refusals fire: wrong/missing tickets sha, sample_draws metadata
      vs stack-width mismatch, stage-3 policy without _ticket, banked
      policy carrying _ticket, identity mismatch, pooled-pred vs
      draws-mean drift beyond tol.

Pure CPU, read-only on inputs, deterministic.

  uv run python fontaine/scripts/goldenticket_stage3_results.py \\
      --out reports/analysis__goldenticket_stage3.json
  uv run python fontaine/scripts/goldenticket_stage3_results.py --oracle
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
from draws_fairness import element_mask, frame_mae, pooled_mae, spearman, step_curve

RUN_STEM = "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000"
STAGE3_NPZ = f"{RUN_STEM}__panel_curated_v0_k4l2_top10tickets_heun30.npz"
STAGE3_DRAWS = f"{RUN_STEM}__panel_curated_v0_k4l2_top10tickets_heun30_draws.npz"
STAGE2_NPZ = f"{RUN_STEM}__panel_curated_v0_k4l2_ticket33_heun30.npz"
BANKED_NPZ = f"{RUN_STEM}__panel_curated_v0_k4l2_stablekey_heun30.npz"
TOP10_SHA = "e537f4cde57b7d2b789f1a4e821fee3fc5ea21ee439913f3747d6d0401d36453"
MEAN10_CHUNK = 5.3645  # banked mean-of-10 anchors (pre-reg "Frozen design")
MEAN10_FIRST = 1.4242
STABLEKEY_CORE_CHUNK = 6.5997  # oracle (a) anchors
TICKET33_CORE_CHUNK = 5.6468
TICKET33_CORE_FIRST = 1.8963
R3_BAND = 0.02
N_DRAWS = 10
POOLED_TOL = 2e-3  # draws-mean must reproduce the pooled pred column

IDENTITY_KEYS = ("index", "repo_id", "episode_index", "frame_index", "core")


def _fail(message: str) -> None:
    raise SystemExit(f"ABORT: {message}")


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(REPO / path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def _meta(d: dict[str, np.ndarray], key: str, label: str) -> Any:
    if key not in d:
        _fail(f"{label}: metadata field '{key}' missing")
    return np.asarray(d[key]).ravel()[0]


def policy_key(d: dict[str, np.ndarray], label: str) -> str:
    keys = [k for k in d if k.startswith("pred:bijou")]
    if len(keys) != 1:
        _fail(f"{label}: expected exactly one bijou prediction column, got {keys}")
    return keys[0]


def join_identity(
    a: dict[str, np.ndarray],
    b: dict[str, np.ndarray],
    label: str,
) -> None:
    for k in (*IDENTITY_KEYS, "truth", "valid"):
        if not np.array_equal(a[k], b[k]):
            _fail(f"{label}: column '{k}' differs — not the same rows")


def core_pool(
    d: dict[str, np.ndarray],
    key: str,
    mask: np.ndarray,
    core: np.ndarray,
) -> tuple[float, float]:
    chunk = pooled_mae(d[key][core], d["truth"][core], mask[core])
    err = np.abs(d[key] - d["truth"]) * mask
    first = err[core][:, 0, :].sum(-1) / np.maximum(mask[core][:, 0, :].sum(-1), 1)
    return float(chunk), float(first.mean())


def classify_r3(delta: float) -> str:
    if delta <= -R3_BAND:
        return "INTERESTING"
    if delta >= R3_BAND:
        return "WORSE"
    return "TIE"


def read(
    stage3: dict[str, np.ndarray],
    s3_draws: dict[str, np.ndarray],
    stage2: dict[str, np.ndarray],
    banked: dict[str, np.ndarray],
    anchor_chunk: float = MEAN10_CHUNK,
    anchor_first: float = MEAN10_FIRST,
) -> dict[str, Any]:
    # ---- provenance refusals ----
    s3_key = policy_key(stage3, "stage-3 npz")
    if "_ticket" not in s3_key:
        _fail(f"stage-3 policy '{s3_key}' does not carry _ticket — not a ticket read")
    sha = str(_meta(s3_draws, "tickets_sha256", "stage-3 draws npz"))
    if sha != TOP10_SHA:
        _fail(
            f"stage-3 draws tickets_sha256 {sha[:12]}… != top-10 sha {TOP10_SHA[:12]}…",
        )
    n_meta = int(_meta(s3_draws, "sample_draws", "stage-3 draws npz"))
    stack = s3_draws["draws"]
    if n_meta != N_DRAWS or stack.shape[1] != N_DRAWS:
        _fail(
            f"sample_draws {n_meta} / stack width {stack.shape[1]} != "
            f"the pre-registered {N_DRAWS}",
        )
    d_policy = str(_meta(s3_draws, "policy", "stage-3 draws npz"))
    if "_ticket" not in d_policy:
        _fail(f"stage-3 draws policy '{d_policy}' does not carry _ticket")
    s2_key = policy_key(stage2, "winner npz")
    if "_ticket" not in s2_key:
        _fail(f"winner policy '{s2_key}' does not carry _ticket")
    bk_key = policy_key(banked, "banked npz")
    if "_ticket" in bk_key:
        _fail(f"banked policy '{bk_key}' carries _ticket — stable-key npz required")

    join_identity(stage3, s3_draws, "stage-3 pooled vs draws")
    join_identity(stage3, banked, "stage-3 vs banked")
    join_identity(stage3, stage2, "stage-3 vs winner")

    # The pooled column must BE the mean of the ticket stack.
    drift = float(
        np.abs(stack.astype(np.float64).mean(axis=1) - stage3[s3_key]).max(),
    )
    if drift >= POOLED_TOL:
        _fail(
            f"stage-3 pred drifts {drift:.4g} from the draws-stack mean "
            f"(tol {POOLED_TOL}) — not the same eval",
        )

    mask = element_mask(stage3["truth"], stage3["valid"])
    core = stage3["core"].astype(bool)

    # ---- R3 ----
    s3_chunk, s3_first = core_pool(stage3, s3_key, mask, core)
    delta = s3_chunk - anchor_chunk
    r3 = {
        "mean_of_top10_chunk": round(s3_chunk, 4),
        "banked_mean_of_10_chunk": anchor_chunk,
        "delta_pooled": round(delta, 5),
        "tie_band": R3_BAND,
        "verdict": classify_r3(delta),
        "record_only": True,
        "first_mae": round(s3_first, 4),
        "banked_mean_of_10_first": anchor_first,
        "first_delta": round(s3_first - anchor_first, 5),
    }

    # ---- R4b: dispersion-quartile geometry of the winner's gain ----
    nvalid = mask.sum(axis=(1, 2))
    dispersion = (stack.astype(np.float64).std(axis=1) * mask).sum(
        axis=(1, 2),
    ) / np.maximum(nvalid, 1)
    keep = core & (nvalid > 0)
    f_win = frame_mae(stage2[s2_key], stage2["truth"], mask)
    f_bank = frame_mae(banked[bk_key], banked["truth"], mask)
    gain = f_win - f_bank  # negative = winner beats stable-key
    qs = np.quantile(dispersion[keep], [0.25, 0.5, 0.75])
    bins = np.digitize(dispersion, qs)
    quartiles = {}
    for b, label in enumerate(["q1_tight", "q2", "q3", "q4_dispersed"]):
        s = (bins == b) & keep
        quartiles[label] = {
            "n": int(s.sum()),
            "dispersion": round(float(dispersion[s].mean()), 4),
            "winner_gain": round(float(gain[s].mean()), 4),
        }
    r4b = {
        "dispersion_source": "stage-3 top-10-ticket stack, valid-weighted std",
        "gain_definition": (
            "per-frame chunk MAE, ticket33 minus banked stable-key "
            "(negative = winner wins), full-panel core rows"
        ),
        "quartiles": quartiles,
        "spearman_dispersion_vs_gain": round(
            spearman(dispersion[keep], gain[keep]),
            4,
        ),
    }

    # ---- R4c mirror: core-row horizon curves ----
    err_s3 = np.abs(stage3[s3_key] - stage3["truth"]) * mask
    err_w = np.abs(stage2[s2_key] - stage2["truth"]) * mask
    err_b = np.abs(banked[bk_key] - banked["truth"]) * mask
    horizon = {
        "mean_of_top10": [
            round(v, 4) for v in step_curve(err_s3[core], stage3["valid"][core])
        ],
        "winner": [round(v, 4) for v in step_curve(err_w[core], stage2["valid"][core])],
        "stablekey": [
            round(v, 4) for v in step_curve(err_b[core], banked["valid"][core])
        ],
    }

    return {
        "inputs": {
            "stage3_npz": STAGE3_NPZ,
            "stage3_draws_npz": STAGE3_DRAWS,
            "winner_npz": STAGE2_NPZ,
            "banked_npz": BANKED_NPZ,
            "top10_tickets_sha256": TOP10_SHA,
            "policy": s3_key,
        },
        "rows": {"panel": len(core), "core": int(core.sum()), "kept": int(keep.sum())},
        "r3": r3,
        "r4b": r4b,
        "r4c_core_horizon": horizon,
    }


# ----------------------------------------------------------------- oracle


def _expect_abort(fn: Any, tag: str) -> None:
    try:
        fn()
    except SystemExit as e:
        print(f"  oracle abort-branch '{tag}' fired: {e}")
        return
    raise AssertionError(f"abort branch '{tag}' did NOT fire")


def _tiny(
    n: int = 16,
    planted_err: float = 1.0,
    gain_scale: float = 0.0,
) -> tuple[dict, dict, dict, dict]:
    """Synthetic quad: stage-3 pooled err == planted_err exactly; the
    draws stack means back to the pred column; dispersion grows linearly
    with frame index; winner gain = -gain_scale * dispersion_scale."""
    rng = np.random.default_rng(11)
    truth = rng.normal(size=(n, 5, 3)).astype(np.float64)
    valid = np.ones((n, 5), dtype=bool)
    ident = {
        "index": np.arange(n),
        "repo_id": np.array([f"r{i % 3}" for i in range(n)]),
        "episode_index": np.arange(n) // 2,
        "frame_index": np.arange(n),
        "core": np.ones(n, dtype=bool),
        "truth": truth,
        "valid": valid,
    }
    pred = truth + planted_err
    disp_scale = np.linspace(0.1, 1.0, n)
    offsets = rng.normal(size=(N_DRAWS, 5, 3))
    offsets -= offsets.mean(axis=0, keepdims=True)  # exact zero-mean over draws
    # One shared template scaled per frame: dispersion is then strictly
    # monotone in disp_scale, so the planted Spearman is exactly ±1.
    stack = pred[:, None] + offsets[None] * disp_scale[:, None, None, None]
    stage3 = {**ident, "pred:bijou@80000_draws10_ticket": pred}
    s3_draws = {
        **ident,
        "draws": stack,
        "policy": np.array("bijou@80000_draws10_ticket"),
        "sample_draws": np.array(N_DRAWS),
        "tickets_sha256": np.array(TOP10_SHA),
    }
    win = truth + 1.0 - gain_scale * disp_scale[:, None, None]
    stage2 = {**ident, "pred:bijou@80000_ticket": win}
    banked = {**ident, "pred:bijou@80000": truth + 1.0}
    return stage3, s3_draws, stage2, banked


def run_oracles() -> None:
    print("oracle (a): banked pools reproduce the board anchors")
    banked = load_npz(BANKED_NPZ)
    mask = element_mask(banked["truth"], banked["valid"])
    core = banked["core"].astype(bool)
    got, _ = core_pool(banked, policy_key(banked, "banked"), mask, core)
    assert round(got, 4) == STABLEKEY_CORE_CHUNK, got
    stage2 = load_npz(STAGE2_NPZ)
    c2, f2 = core_pool(stage2, policy_key(stage2, "winner"), mask, core)
    assert round(c2, 4) == TICKET33_CORE_CHUNK, c2
    assert round(f2, 4) == TICKET33_CORE_FIRST, f2
    print(f"  stablekey {round(got, 4)}, ticket33 {round(c2, 4)}/{round(f2, 4)}")

    print("oracle (b): planted R3 boundaries (anchor parameterized)")
    for shift, want in [
        (0.05, "INTERESTING"),
        (0.0, "TIE"),
        (-0.05, "WORSE"),
        (R3_BAND, "INTERESTING"),
        (-R3_BAND, "WORSE"),
    ]:
        s3, dr, s2, bk = _tiny()
        out = read(s3, dr, s2, bk, anchor_chunk=1.0 + shift, anchor_first=0.0)
        got_v = out["r3"]["verdict"]
        assert got_v == want, f"anchor shift {shift}: {got_v} != {want}"
        print(f"  delta {out['r3']['delta_pooled']:+.3f} -> {got_v}")

    print("oracle (c): planted R4b geometry")
    s3, dr, s2, bk = _tiny(gain_scale=0.5)
    out = read(s3, dr, s2, bk, anchor_chunk=1.0, anchor_first=0.0)
    r4b = out["r4b"]
    assert r4b["spearman_dispersion_vs_gain"] == -1.0, r4b
    means = [
        r4b["quartiles"][q]["winner_gain"]
        for q in ("q1_tight", "q2", "q3", "q4_dispersed")
    ]
    assert all(a > b for a, b in itertools.pairwise(means)), means
    assert (
        sum(r4b["quartiles"][q]["n"] for q in r4b["quartiles"]) == out["rows"]["kept"]
    )
    print(f"  spearman {r4b['spearman_dispersion_vs_gain']}, quartile gains {means}")

    print("oracle (d): refusals")
    s3, dr, s2, bk = _tiny()
    bad = dict(dr)
    bad["tickets_sha256"] = np.array("deadbeef")
    _expect_abort(lambda: read(s3, bad, s2, bk), "wrong tickets sha")
    bad = dict(dr)
    del bad["tickets_sha256"]
    _expect_abort(lambda: read(s3, bad, s2, bk), "missing tickets sha")
    bad = dict(dr)
    bad["sample_draws"] = np.array(64)
    _expect_abort(lambda: read(s3, bad, s2, bk), "sample_draws mismatch")
    bad_s3 = dict(s3)
    bad_s3["pred:bijou@80000"] = bad_s3.pop("pred:bijou@80000_draws10_ticket")
    _expect_abort(lambda: read(bad_s3, dr, s2, bk), "stage-3 without _ticket")
    bad_bk = dict(bk)
    bad_bk["pred:bijou@80000_ticket"] = bad_bk.pop("pred:bijou@80000")
    _expect_abort(lambda: read(s3, dr, s2, bad_bk), "banked carries _ticket")
    bad_bk = dict(bk)
    bad_bk["frame_index"] = bad_bk["frame_index"] + 1
    _expect_abort(lambda: read(s3, dr, s2, bad_bk), "identity mismatch")
    bad_s3 = dict(s3)
    bad_s3["pred:bijou@80000_draws10_ticket"] = (
        s3["pred:bijou@80000_draws10_ticket"] + 0.01
    )
    _expect_abort(lambda: read(bad_s3, dr, s2, bk), "pooled-pred drift")

    print("ORACLES GREEN")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "reports/analysis__goldenticket_stage3.json",
    )
    args = parser.parse_args()
    if args.oracle:
        run_oracles()
        return
    out = read(
        load_npz(STAGE3_NPZ),
        load_npz(STAGE3_DRAWS),
        load_npz(STAGE2_NPZ),
        load_npz(BANKED_NPZ),
    )
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    r3 = out["r3"]
    print(
        f"R3: mean-of-top-10 {r3['mean_of_top10_chunk']} vs banked "
        f"{r3['banked_mean_of_10_chunk']} -> delta {r3['delta_pooled']} "
        f"(band ±{r3['tie_band']}) -> {r3['verdict']} [record-only]",
    )
    print(f"    first {r3['first_mae']} vs {r3['banked_mean_of_10_first']}")
    print(f"R4b: {out['r4b']['quartiles']}")
    print(f"     spearman {out['r4b']['spearman_dispersion_vs_gain']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
