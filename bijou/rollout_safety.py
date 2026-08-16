"""Pre-flight safety gates for the physical rollout (``bijou.rollout``).

Three gates, all cheap and all before the arm moves:

- **Clamp**: ``--max-relative-target`` is mandatory. The vendored
  lerobot degrees branch un-normalizes predicted positions with no
  min/max against calibration, so the per-tick relative clamp is the
  ONLY limiter between a bad chunk (or wrong stats) and arbitrary servo
  ticks at full speed. ``--unclamped`` is the explicit opt-out.
- **First-observation envelope**: the first observation after connect
  must lie inside a widened per-joint plausibility band derived from
  the rig stats — catches a wrong ``--stats-repo-id``, a
  ticks-vs-degrees convention mismatch, and an uncalibrated arm before
  any action is sent.
- **Camera kinds**: ``--camera`` keys ARE semantic kinds
  (wrist/top/front/side/unknown; off-vocabulary or duplicate keys are
  refused). The kind is what the model sees — the prompt's
  (kind, name) image order and the kind-aware formats' text tags — so
  the operator asserts it directly, and the assertion is cross-checked
  LOUDLY against the rig dataset's judged kinds
  (``meta/camera_kinds.json``: the record of what training rendered
  for this rig's views — e.g. a fixed overhead cam the dataset NAMED
  "front" was judged kind "top"). Mismatches and uncovered judged
  kinds warn and proceed with the asserted kinds — explicit operator
  input is never silently rewritten.

Plus the joint-frame remap the rollout applies around the robot
boundary (``--joint-frame``, :class:`JointFrameTransform`): state maps
arm→model before the prompt, chunks map model→arm before
``send_action``. For checkpoints with GLOBAL normalization
(molmo_flow — the joint-angle convention is baked into the checkpoint
table) ``bijou.rollout`` additionally gates the first observation
against the checkpoint's OWN state q01/q99 band in model frame, so a
missing or wrong remap dies before any action is sent.

Pure CPU, no lerobot imports — testable without a robot.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .annotations import CAMERA_KINDS
from .data import DatasetStats, annotation_stamp, camera_kinds_of


def require_clamp(max_relative_target: float | None, *, unclamped: bool) -> None:
    """Refuse to drive the arm without an absolute per-tick motion clamp.

    A set clamp must be a positive finite number of degrees; passing
    both a clamp and ``--unclamped`` is contradictory and dies loud
    rather than guessing which one was meant.
    """
    if max_relative_target is not None:
        if unclamped:
            raise SystemExit(
                "--max-relative-target and --unclamped are contradictory — "
                "pass exactly one",
            )
        if not math.isfinite(max_relative_target) or max_relative_target <= 0:
            raise SystemExit(
                f"--max-relative-target must be a positive number of degrees "
                f"per tick, got {max_relative_target!r}",
            )
        return
    if unclamped:
        print(
            "WARNING: running UNCLAMPED — nothing limits per-tick joint "
            "motion; a single bad chunk commands full-speed arbitrary "
            "positions",
            flush=True,
        )
        return
    raise SystemExit(
        "refusing to drive the arm without --max-relative-target: it is the "
        "ONLY limiter between a bad chunk (or wrong stats) and full-speed "
        "arbitrary servo motion (the lerobot degrees branch applies no "
        "min/max against calibration). Start with --max-relative-target 20; "
        "--unclamped is the explicit opt-out.",
    )


@dataclass(frozen=True, slots=True)
class JointFrameTransform:
    """Per-joint affine map between the arm's CALIBRATION frame and a
    checkpoint's MODEL (training-data) frame, in degrees:

        state_to_model:  model = signs · arm + offsets
        chunk_to_arm:    arm   = signs · (model − offsets)

    (the inverse form holds because every sign is ±1). Needed when a
    checkpoint's action/state distribution was recorded under a
    DIFFERENT calibration convention than the deployed arm: the lerobot
    PR#777 hardware redesign (shipped 0.5.x) moved the SO-100/101 zero
    from arm-extended-horizontal to mid-range and flipped
    shoulder_lift, so a model trained on pre-0.5 data — e.g. a
    converted MolmoAct2 release, whose GLOBAL q01/q99 table bakes the
    old frame in — commands ~90°-off shoulder/elbow poses on a
    post-0.5-calibrated arm (the slam-into-the-table failure).
    Identity for rig-native checkpoints: bijou fine-tunes normalize per
    dataset with stats recorded under the deployment calibration, so
    the convention cancels and there is no remap to get wrong.
    See https://huggingface.co/docs/lerobot/backwardcomp.
    """

    signs: tuple[float, ...]
    offsets: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.signs) != len(self.offsets):
            raise ValueError(
                f"signs ({len(self.signs)}) and offsets ({len(self.offsets)}) "
                "must have one entry per joint",
            )
        if any(sign not in (-1.0, 1.0) for sign in self.signs):
            raise ValueError(
                f"signs must be ±1 (the map must invert exactly), got {self.signs}",
            )

    @staticmethod
    def identity(dim: int) -> JointFrameTransform:
        """No remap: the arm's frame IS the model's frame."""
        return JointFrameTransform(signs=(1.0,) * dim, offsets=(0.0,) * dim)

    @staticmethod
    def lerobot_v30_to_v21() -> JointFrameTransform:
        """The official post-PR#777 ↔ pre-PR#777 SO-100/101 conversion
        (the backwardcomp doc's replay transform, model→arm:
        shoulder_lift' = −(x − 90), elbow_flex' = x − 90, identity
        elsewhere — here stated arm→model, matching the reference
        MolmoAct2 deployment defaults ``--joint-signs 1,-1,1,1,1,1`` /
        ``--joint-offsets 0,90,90,0,0,0``). The literals live HERE
        only; tests pin both directions against the doc's form."""
        return JointFrameTransform(
            signs=(1.0, -1.0, 1.0, 1.0, 1.0, 1.0),
            offsets=(0.0, 90.0, 90.0, 0.0, 0.0, 0.0),
        )

    @property
    def is_identity(self) -> bool:
        return all(sign == 1.0 for sign in self.signs) and all(
            offset == 0.0 for offset in self.offsets
        )

    def state_to_model(self, state: Sequence[float]) -> list[float]:
        """Arm-frame joint positions → model frame, one vector (degrees;
        length must match the per-joint tables, loud otherwise)."""
        return [
            sign * value + offset
            for sign, value, offset in zip(
                self.signs,
                state,
                self.offsets,
                strict=True,
            )
        ]

    def chunk_to_arm(self, chunk: Tensor) -> Tensor:
        """Model-frame action rows → arm frame for ``send_action``.
        Identity passes the tensor through untouched, keeping the
        no-remap deployment path byte-identical.

        Shapes:
        - ``chunk``: [T, dim] absolute joint targets, model frame (degrees)
        - returns: [T, dim] absolute joint targets, arm frame (degrees)
        """
        if self.is_identity:
            return chunk
        signs = torch.tensor(self.signs, dtype=chunk.dtype, device=chunk.device)
        offsets = torch.tensor(self.offsets, dtype=chunk.dtype, device=chunk.device)
        return (chunk - offsets) * signs

    def values_to_arm(self, values: Sequence[float]) -> tuple[float, ...]:
        """The model→arm inverse on a plain per-joint vector — the same
        map :meth:`chunk_to_arm` applies to action rows, for consumers
        that transform TABLES rather than tensors (the conversion-time
        quantile remap: mapped = (recorded − offset) · sign, exact
        because signs are ±1). Length must match the joint count."""
        return tuple(
            (value - offset) * sign
            for value, offset, sign in zip(
                values,
                self.offsets,
                self.signs,
                strict=True,
            )
        )


def state_envelope(
    stats: DatasetStats,
    *,
    expected_dim: int,
    floor: float = 15.0,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Per-joint ``(lo, hi)`` plausibility bounds from the rig stats.

    The q01..q99 band (mean ± 3·std when the stats predate quantiles),
    widened per side by half its own width with an absolute floor of
    ``floor`` degrees — generous enough for an unusual-but-calibrated
    start pose, and orders of magnitude tighter than the failure modes
    it exists to catch (raw servo ticks ~10³, a different rig's table,
    an uncalibrated arm). A stats vector of the wrong dimensionality is
    itself a wrong-stats-source symptom and dies loud.
    """
    if len(stats.state_mean) != expected_dim:
        raise SystemExit(
            f"stats state dimension {len(stats.state_mean)} != the arm's "
            f"{expected_dim} joints — wrong --stats-repo-id/--stats-dataset "
            "(different embodiment?)",
        )
    if stats.state_q01 is not None and stats.state_q99 is not None:
        # Per-joint min/max: a v2.1→v3.0-remapped table stores its
        # sign-flipped joints as DESCENDING q01>q99 pairs (the flip is
        # the table's, not the band's — docs/so101-joint-conventions.md);
        # the plausibility band is orientation-free.
        band_lo = tuple(
            min(a, b) for a, b in zip(stats.state_q01, stats.state_q99, strict=True)
        )
        band_hi = tuple(
            max(a, b) for a, b in zip(stats.state_q01, stats.state_q99, strict=True)
        )
    else:
        band_lo = tuple(
            m - 3.0 * s for m, s in zip(stats.state_mean, stats.state_std, strict=True)
        )
        band_hi = tuple(
            m + 3.0 * s for m, s in zip(stats.state_mean, stats.state_std, strict=True)
        )
    pads = tuple(
        max(0.5 * (hi - lo), floor) for lo, hi in zip(band_lo, band_hi, strict=True)
    )
    return (
        tuple(lo - pad for lo, pad in zip(band_lo, pads, strict=True)),
        tuple(hi + pad for hi, pad in zip(band_hi, pads, strict=True)),
    )


def envelope_violations(
    state: Sequence[float],
    envelope: tuple[tuple[float, ...], tuple[float, ...]],
) -> list[int]:
    """Indices of joints outside the envelope (NaN counts as outside)."""
    lo, hi = envelope
    return [j for j, value in enumerate(state) if not (lo[j] <= value <= hi[j])]


def resolve_camera_kinds(
    names: Iterable[str],
    stats_dataset: Path | None,
) -> dict[str, str]:
    """Validate the rollout's camera kinds and cross-check them against
    the rig dataset's judged kinds.

    ``--camera`` keys ARE semantic kinds (``top=/dev/video6``): the
    kind is the axis the model actually sees — it drives the prompt's
    (kind, name) image order and the tag kind-aware formats render —
    so the operator states it directly. A key outside the fixed
    vocabulary (wrist/top/front/side/unknown) or given twice is a
    SystemExit: explicit operator input is corrected, never silently
    rewritten, and two same-kind cameras have no expressible prompt
    slots on this CLI.

    The asserted kinds are ALWAYS used. When ``--stats-dataset`` points
    at a stamped rig dataset, they are cross-checked against its judged
    kinds (``meta/camera_kinds.json``, the record of what training
    rendered for this rig's views): asserted kinds the dataset never
    judged and judged kinds no camera covers print ONE loud warning —
    the run proceeds with the operator's kinds. An unstamped dataset
    (whose training prompts rendered 'unknown') warns when any
    non-unknown kind is asserted."""
    names = list(names)
    off_vocabulary = sorted(name for name in names if name not in CAMERA_KINDS)
    if off_vocabulary:
        raise SystemExit(
            f"--camera keys are semantic kinds; {off_vocabulary} not in the "
            f"vocabulary {sorted(CAMERA_KINDS)} — key each camera by its "
            "viewpoint, e.g. --camera top=/dev/video6 --camera "
            "wrist=/dev/video4",
        )
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise SystemExit(
            f"--camera kind(s) {duplicates} given more than once — each "
            "kind keys ONE prompt slot; a rig with two same-kind views "
            "cannot be expressed on this CLI",
        )
    kinds = {name: name for name in names}
    if stats_dataset is None:
        return kinds
    repo_id = stats_dataset.name
    stamp = annotation_stamp(stats_dataset, repo_id, None)
    judged = camera_kinds_of(stats_dataset, repo_id, stamp) if stamp is not None else {}
    if not judged:
        if any(kind != "unknown" for kind in kinds):
            print(
                f"WARNING: {repo_id} has no usable stamped kinds file — "
                "its training prompts rendered every camera 'unknown', "
                f"but this rollout asserts {sorted(kinds)}. Proceeding "
                "with the asserted kinds",
                flush=True,
            )
        return kinds
    judged_kinds = set(judged.values())
    unjudged = sorted(set(kinds) - judged_kinds)
    uncovered = sorted(judged_kinds - set(kinds))
    if unjudged or uncovered:
        clauses: list[str] = []
        if unjudged:
            clauses.append(
                f"asserted kind(s) {unjudged} were never judged for this rig's views",
            )
        if uncovered:
            clauses.append(
                f"judged kind(s) {uncovered} are covered by no --camera",
            )
        print(
            f"WARNING: --camera kinds {sorted(kinds)} disagree with "
            f"{repo_id}'s judged kinds {sorted(judged_kinds)} "
            f"({'; '.join(clauses)}). Proceeding with the asserted kinds — "
            "the prompt may carry tags/order the model did not train on "
            "for these views",
            flush=True,
        )
    return kinds


def home_trajectory(
    current: Sequence[float],
    home: Sequence[float],
    *,
    seconds: float,
    fps: float,
) -> list[list[float]]:
    """Tick-by-tick linear interpolation from ``current`` to ``home``
    (inclusive of the final exact-home row): the slow return-to-start
    executed on ctrl-c. Cosine-eased at both ends so the arm neither
    jerks off its stop position nor slams into home — peak per-tick
    step ~1.57x the linear rate, still far under any sane clamp for a
    1-2 s return."""
    ticks = max(2, round(seconds * fps))
    rows: list[list[float]] = []
    for step in range(1, ticks + 1):
        # Cosine ease-in-out: s(u) in [0, 1], s'(0) = s'(1) = 0.
        u = step / ticks
        eased = 0.5 * (1.0 - math.cos(math.pi * u))
        rows.append(
            [c + (h - c) * eased for c, h in zip(current, home, strict=True)],
        )
    return rows
