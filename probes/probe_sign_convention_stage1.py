"""Stage-1 sign-convention screen over a --dump-predictions npz (CPU).

Slices a panel eval's dumped predictions per-repo per-action-dim looking
for the flipped-sign-convention signature: ONE dim whose MAE is a large
outlier versus the panel-median per-dim MAE while its motion-shape
correlation with truth (chunk-mean removed, valid-masked) drops to ~0 or
negative, with the same repo's OTHER dims staying normal. A repo that is
merely hard is bad on most dims with positive shape correlation; a repo
whose wrist_roll is recorded with the opposite sign is bad on exactly
that dim with ~zero or negative correlation — the model predicts the
mirrored trajectory it learned from the panel majority.

What stage 1 can and cannot see (pre-registered caveats, 2026-08-05):
it screens MODEL-VS-TRUTH disagreement, so it catches repos whose
actions/states disagree with the corpus-majority convention as rendered
by a strong model. It CANNOT catch an internally-consistent mirror-world
repo the model has partially fit, and small per-repo panel samples
(n = 8-16 frames) make individual ratios noisy: candidates are screening
leads for the stage-2 optical-flow probe, never per-repo convictions.

A per-frame classification pass follows the screen (added same day
after LOOKING at the standout's trajectories): for each candidate it
counts wrap frames (truth span > 300 deg — the +-180 boundary makes
raw-degree MAE explode without any convention error), flat predictions
(pred std < 1e-3 deg), anti-correlated frames (corr < -0.5, the actual
mirror signature), and the median per-frame shape corr. This separates
three distinct pathologies the aggregate screen conflates: wraparound
artifacts (kevin510: 5/16 wrap frames), genuine mirror candidates
(kantine dishTidyUp_anomaly wrist_flex: median frame corr -0.75), and
tracked-but-offset dims (Dongkkka shoulder_pan: median corr +0.76 —
shape fine, level off; not a sign issue).

Anchors (asserted in main() when run with all defaults — the asserts are
the single source of truth, per the probe convention): the laptop
reference npz `eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2`
(bijou@100000 greedy, 17,204 core frames, 878 repos) yields exactly
9 isolated-dim candidates at (n>=8, ratio>3, corr<0.1), 4 of them
wrist_roll; standout kevin510/lerobot-cat-toy-placement main_wrist_roll
at 14.85x panel-median MAE with corr -0.02 over n=16 frames while its
other dims sit at 1.64x median. First run 2026-08-05 ~15:27Z (scratch),
formalized same day; results posted to Discord 15:28Z.

Run from the repo root: uv run python -m probes.probe_sign_convention_stage1
Non-default npz/policy/thresholds skip the anchor asserts (screening use).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REFERENCE_NPZ = (
    Path.home()
    / "previous-reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2.npz"
)
REFERENCE_POLICY = "pred:bijou@100000"
DEFAULT_MIN_FRAMES = 8
DEFAULT_RATIO = 3.0
DEFAULT_CORR = 0.1
# Reference-run anchors (see docstring).
ANCHOR_CANDIDATES = 9
ANCHOR_TOP = ("kevin510/lerobot-cat-toy-placement", "main_wrist_roll", "14.85", "-0.02")


@dataclass(frozen=True)
class RepoStats:
    """Per-repo per-dim aggregates over the panel's core frames."""

    repo_id: str
    frames: int
    mae: np.ndarray  # (dims,) mean frame MAE per action dim
    corr: np.ndarray  # (dims,) motion-shape corr (chunk-mean removed); NaN if flat


@dataclass(frozen=True)
class Candidate:
    """One (repo, dim) flagged by the isolated-dim screen."""

    repo_id: str
    frames: int
    dim_name: str
    ratio: float  # this dim's MAE / panel-median MAE for the dim
    corr: float  # NaN renders as +nan and always passes the corr gate
    other_dims_median_ratio: float


@dataclass(frozen=True)
class FrameClassification:
    """Per-frame pathology counts for one candidate (repo, dim)."""

    wrap_frames: int  # truth span > 300 deg: +-180 wraparound artifact
    flat_pred_frames: int  # pred std < 1e-3 deg: model predicts a constant
    anti_frames: int  # per-frame shape corr < -0.5: the mirror signature
    median_frame_corr: float  # NaN when no frame has both signals moving


def repo_stats(
    truth: np.ndarray,
    pred: np.ndarray,
    valid: np.ndarray,
    repo_ids: np.ndarray,
) -> list[RepoStats]:
    """Aggregate (frames, chunk, dims) errors into per-repo rows."""
    dims = truth.shape[-1]
    err = np.abs(pred - truth) * valid[..., None]
    counts = valid.sum(1)[:, None]
    frame_mae = err.sum(1) / np.maximum(counts, 1)

    def centered(x: np.ndarray) -> np.ndarray:
        chunk_mean = (x * valid[..., None]).sum(1, keepdims=True) / np.maximum(
            counts,
            1,
        )[:, None]
        return (x - chunk_mean) * valid[..., None]

    truth_c, pred_c = centered(truth), centered(pred)

    rows = []
    for repo_id in np.unique(repo_ids):
        mask = repo_ids == repo_id
        flat_t = truth_c[mask].reshape(-1, dims)
        flat_p = pred_c[mask].reshape(-1, dims)
        corr = np.full(dims, np.nan)
        for d in range(dims):
            if flat_t[:, d].std() > 1e-6 and flat_p[:, d].std() > 1e-6:
                corr[d] = np.corrcoef(flat_t[:, d], flat_p[:, d])[0, 1]
        rows.append(
            RepoStats(
                repo_id=str(repo_id),
                frames=int(mask.sum()),
                mae=frame_mae[mask].mean(0),
                corr=corr,
            ),
        )
    return rows


def screen(
    rows: list[RepoStats],
    motor_names: list[str],
    *,
    min_frames: int,
    ratio_threshold: float,
    corr_threshold: float,
) -> list[Candidate]:
    """Isolated-dim outliers: high MAE ratio, ~zero shape corr, others normal."""
    panel_median = np.median(np.stack([row.mae for row in rows]), 0)
    candidates = []
    for row in rows:
        if row.frames < min_frames:
            continue
        ratio = row.mae / panel_median
        for d, dim_name in enumerate(motor_names):
            corr_low = bool(np.isnan(row.corr[d])) or row.corr[d] < corr_threshold
            if ratio[d] > ratio_threshold and corr_low:
                candidates.append(
                    Candidate(
                        repo_id=row.repo_id,
                        frames=row.frames,
                        dim_name=dim_name,
                        ratio=float(ratio[d]),
                        corr=float(row.corr[d]),
                        other_dims_median_ratio=float(np.median(np.delete(ratio, d))),
                    ),
                )
    candidates.sort(key=lambda c: -c.ratio)
    return candidates


def classify_frames(
    candidate: Candidate,
    truth: np.ndarray,
    pred: np.ndarray,
    valid: np.ndarray,
    repo_ids: np.ndarray,
    dim: int,
) -> FrameClassification:
    """Separate wraparound / flat-pred / mirror pathologies per frame."""
    mask = repo_ids == candidate.repo_id
    truth_d, pred_d, valid_r = truth[mask][..., dim], pred[mask][..., dim], valid[mask]
    wraps = flats = antis = 0
    corrs = []
    for i in range(truth_d.shape[0]):
        n = int(valid_r[i].sum())
        a, b = truth_d[i][:n], pred_d[i][:n]
        if a.max() - a.min() > 300:
            wraps += 1
        if b.std() < 1e-3:
            flats += 1
        # 1e-3 deg matches the flat threshold: a near-constant float32
        # signal underflows in corrcoef after mean-centering and yields
        # NaN, so "both signals actually move" is the correlation gate.
        if a.std() > 1e-3 and b.std() > 1e-3:
            corr = float(np.corrcoef(a - a.mean(), b - b.mean())[0, 1])
            if np.isfinite(corr):
                corrs.append(corr)
                if corr < -0.5:
                    antis += 1
    return FrameClassification(
        wrap_frames=wraps,
        flat_pred_frames=flats,
        anti_frames=antis,
        median_frame_corr=float(np.median(corrs)) if len(corrs) > 0 else float("nan"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=REFERENCE_NPZ)
    parser.add_argument("--policy", default=REFERENCE_POLICY)
    parser.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES)
    parser.add_argument("--ratio-threshold", type=float, default=DEFAULT_RATIO)
    parser.add_argument("--corr-threshold", type=float, default=DEFAULT_CORR)
    args = parser.parse_args()

    sidecar = args.npz.with_suffix(".json")
    motor_names = json.loads(sidecar.read_text())["motor_names"]
    dump = np.load(args.npz, allow_pickle=True)
    core = dump["core"]
    rows = repo_stats(
        dump["truth"][core],
        dump[args.policy][core],
        dump["valid"][core],
        dump["repo_id"][core],
    )
    candidates = screen(
        rows,
        motor_names,
        min_frames=args.min_frames,
        ratio_threshold=args.ratio_threshold,
        corr_threshold=args.corr_threshold,
    )

    print(f"npz: {args.npz.name}  policy: {args.policy}  repos: {len(rows)}")
    print(
        f"isolated-dim candidates (n>={args.min_frames}, "
        f"ratio>{args.ratio_threshold:g}, corr<{args.corr_threshold:g}): "
        f"{len(candidates)}",
    )
    truth_core = dump["truth"][core]
    pred_core = dump[args.policy][core]
    valid_core = dump["valid"][core]
    repo_core = dump["repo_id"][core]
    classifications = []
    for c in candidates:
        cls = classify_frames(
            c,
            truth_core,
            pred_core,
            valid_core,
            repo_core,
            motor_names.index(c.dim_name),
        )
        classifications.append(cls)
        print(
            f"{c.repo_id[:55]:55s} n={c.frames:4d} dim={c.dim_name:20s} "
            f"ratio={c.ratio:6.2f} corr={c.corr:+.2f} "
            f"other-dims-med-ratio={c.other_dims_median_ratio:.2f} | "
            f"wrap={cls.wrap_frames} flat={cls.flat_pred_frames} "
            f"anti={cls.anti_frames} med-frame-corr={cls.median_frame_corr:+.2f}",
        )

    is_reference_run = (
        args.npz == REFERENCE_NPZ
        and args.policy == REFERENCE_POLICY
        and args.min_frames == DEFAULT_MIN_FRAMES
        and args.ratio_threshold == DEFAULT_RATIO
        and args.corr_threshold == DEFAULT_CORR
    )
    if is_reference_run:
        assert len(candidates) == ANCHOR_CANDIDATES, "candidate count DRIFTED"
        top = candidates[0]
        observed = (top.repo_id, top.dim_name, f"{top.ratio:.2f}", f"{top.corr:.2f}")
        assert observed == ANCHOR_TOP, f"top candidate DRIFTED: {observed}"
        wrist_roll = sum(1 for c in candidates if c.dim_name == "main_wrist_roll")
        assert wrist_roll == 4, "wrist_roll candidate count DRIFTED"
        by_repo = {
            c.repo_id: cls for c, cls in zip(candidates, classifications, strict=True)
        }
        top_cls = by_repo["kevin510/lerobot-cat-toy-placement"]
        assert top_cls.wrap_frames == 5, "kevin510 wrap-frame count DRIFTED"
        mirror = by_repo["kantine/domotic_dishTidyUp_anomaly"]
        assert f"{mirror.median_frame_corr:.2f}" == "-0.75", "mirror anchor DRIFTED"
        offset = by_repo["Dongkkka/koch_arm_gripper_pick_red_pen"]
        assert f"{offset.median_frame_corr:.2f}" == "0.76", "offset anchor DRIFTED"
        print("SIGN-CONVENTION STAGE-1 ANCHORS PASSED")
    else:
        print("(non-default inputs: anchor asserts skipped)")


if __name__ == "__main__":
    main()
