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
- **Camera kinds**: resolved the way TRAINING resolved them (the rig
  dataset's stamped ``meta/camera_kinds.json``), or by explicit
  ``--camera-kind`` override. Deriving kinds from operator camera
  names when the dataset says otherwise (a fixed overhead cam named
  "front" judged kind "top") is silent conditioning skew on exactly
  the few-shot-transfer surface; the name heuristic survives only when
  no dataset directory is available.

Pure CPU, no lerobot imports — testable without a robot.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from pathlib import Path

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
        band_lo, band_hi = stats.state_q01, stats.state_q99
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


def parse_camera_kind_overrides(
    specs: Iterable[str],
    camera_names: Iterable[str],
) -> dict[str, str]:
    """``--camera-kind NAME=KIND`` overrides, validated: the name must be
    a ``--camera`` name and the kind must be in the vocabulary —
    explicit operator input is corrected, never degraded to unknown."""
    names = set(camera_names)
    overrides: dict[str, str] = {}
    for spec in specs:
        name, sep, kind = spec.partition("=")
        if not sep or not name or not kind:
            raise SystemExit(f"--camera-kind expects NAME=KIND, got {spec!r}")
        if name not in names:
            raise SystemExit(
                f"--camera-kind {name!r} is not a --camera name ({sorted(names)})",
            )
        if kind not in CAMERA_KINDS:
            raise SystemExit(
                f"--camera-kind {name}={kind!r}: kind not in the vocabulary "
                f"{sorted(CAMERA_KINDS)}",
            )
        overrides[name] = kind
    return overrides


def camera_kinds_from_names(names: Iterable[str]) -> dict[str, str]:
    """Per-camera semantic kinds from the operator's own camera names: a
    name inside the judge vocabulary IS its kind; anything else renders
    "unknown" (trained in-distribution via kind dropout) with a LOUD
    warning — name cameras by viewpoint to give the model the signal."""
    kinds: dict[str, str] = {}
    for name in names:
        if name in CAMERA_KINDS:
            kinds[name] = name
        else:
            print(
                f"WARNING: camera name {name!r} is not in the semantic "
                f"kind vocabulary {sorted(CAMERA_KINDS)} — its prompt tag "
                "renders as 'unknown'",
                flush=True,
            )
            kinds[name] = "unknown"
    return kinds


def resolve_camera_kinds(
    names: Iterable[str],
    overrides: dict[str, str],
    stats_dataset: Path | None,
) -> dict[str, str]:
    """Per-camera kinds for the rollout prompt, resolved the way training
    resolved them.

    Priority per camera: explicit override → the rig dataset's stamped
    ``meta/camera_kinds.json`` through training's own resolution path
    (an unstamped or hash-mismatched dataset rendered its cameras
    "unknown" in training prompts, so the rollout mirror does too) →
    the name-is-kind heuristic, only when no dataset directory is
    available."""
    names = list(names)
    if stats_dataset is not None:
        repo_id = stats_dataset.name
        stamp = annotation_stamp(stats_dataset, repo_id, None)
        trained = (
            camera_kinds_of(stats_dataset, repo_id, stamp) if stamp is not None else {}
        )
        if not trained:
            print(
                f"[camera-kinds] {repo_id}: no stamped kinds file — training "
                "rendered these cameras 'unknown'; mirroring that "
                "(--camera-kind overrides)",
                flush=True,
            )
        kinds: dict[str, str] = {}
        for name in names:
            if trained and name not in trained:
                print(
                    f"[camera-kinds] camera {name!r} is not in the dataset's "
                    "kinds file — rendering as 'unknown' "
                    "(--camera-kind overrides)",
                    flush=True,
                )
            kinds[name] = trained.get(name, "unknown")
    else:
        kinds = camera_kinds_from_names(names)
    kinds.update(overrides)
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
