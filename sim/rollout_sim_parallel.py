"""Parallel closed-loop sim rollouts: N env workers, ONE batched policy.

The sequential driver (sim.rollout_sim) is render-bound — the H100 idles
through most of every heun-10 decode at batch 1. This driver splits the
work the way the queue item registered it: each worker process owns a
full ``SO101Sim`` (physics + its own GL context; MuJoCo's EGL display is
per-process global state, so envs must not share a process), while the
parent holds the single checkpoint copy and serves batched
``policy.predict`` calls.

Scheduling is deterministic LOCKSTEP ROUNDS: each round, the parent
collects exactly one message stream per still-active worker in
worker-index order until that worker either requests a predict or
finishes its seed slice, then answers all requests with one batched
forward. Batch membership is therefore a pure function of (seed
partition, worker count, policy outputs) — never of wall-clock timing.
Seeds are partitioned round-robin (worker ``w`` gets ``seeds[w::N]``)
and every row carries the same identity triple as the sequential driver
(``repo_id="sim/eval100"``, ``episode_index=seed``,
``frame_index=replan``), so stable-key flow noise is untouched by
batching or by which worker runs a seed.

Determinism contract (pre-registered): the CPU-tier oracle in
tests/test_sim_parallel_rollouts.py pins harness equivalence — same
per-seed (obs -> chunk -> step) sequence and rows as the sequential
loop, which both drivers share via ``run_episode_loop``. Whether the
BATCHED forward is bit-identical to batch-1 decode is an empirical GPU
question (GEMM reduction order can move with batch shape); the
registered smoke (fontaine/scripts/sim_parallel_oracle.py) answers it
before any registered eval uses this path.

Usage (MUJOCO_GL=egl must be set — spawn workers inherit it):
  MUJOCO_GL=egl uv run python -m sim.rollout_sim_parallel \
      --checkpoint outputs/train/er_60k/step_060000 \
      --seed 0 --num-seeds 100 --workers 8 --out-json out.json
"""

import argparse
import dataclasses
import io
import json
import multiprocessing as mp
import subprocess
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np

from . import OUTPUT_DIR
from .rollout_sim import (
    STATS_REPO_ID,
    TASK,
    EpisodeResult,
    RolloutSim,
    hold_chunk_fn,
    resolve_replans,
    resolve_worn_stats,
    run_episode_loop,
    sim_item,
    worn_stats_key,
)
from .so101_sim import CONTROL_HZ, SimObservation, SO101Sim
from .wrist_transform import (
    TOP_TRANSFORMS,
    WRIST_TRANSFORMS,
    chain_transforms,
    make_top_transform,
    make_wrist_transform,
    print_coverage,
)

# Worker -> parent messages are picklable tagged tuples: "predict"
# carries (worker_id, seed, replan, top, wrist, state); "row" carries
# (worker_id, EpisodeResult); "done" and "error" carry the worker_id
# (plus the traceback text for "error"). The parent answers each
# "predict" with the [horizon, 6] action chunk, nothing else.


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    worker_id: int
    # (seed, draw) work units: the GRPO-probe draws extension makes the
    # unit a stochastic rollout, not a seed — draw 0 keeps the banked
    # identity (and is the only unit that records video), exactly the
    # sequential driver's convention.
    units: tuple[tuple[int, int], ...]
    replans: int
    horizon: int
    hold: bool
    out_dir: Path | None
    post_backend: str
    clutter_appearance: str = "patched"
    flip_camera_mount: bool = True
    # wrist-transfer screen treatment (pre-reg 2026-08-14 §1): rewrites
    # only the wrist frame each predict request carries — worker-local,
    # one fresh transform per (seed, draw) unit.
    wrist_transform: str = "none"
    # T1 positive control (same pre-reg, arm grid): rewrites only the
    # top frame; composes with the wrist hook at the same loop seam.
    top_transform: str = "none"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--hold",
        action="store_true",
        help="no policy: command the settled reset state every tick "
        "(runs fully worker-local, zero predict rounds)",
    )
    parser.add_argument("--seed", type=int, default=0, help="first env + noise seed")
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="episodes: seed .. seed+N-1",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="env worker processes (each owns a SO101Sim + GL context; "
        "capped at num-seeds)",
    )
    parser.add_argument("--replans", type=int, default=None)
    parser.add_argument(
        "--episode-seconds",
        type=float,
        default=None,
        help="episode TIME budget; the replan count derives from the "
        "resolved chunk horizon at 30 Hz, so 30 seconds means 30 "
        "seconds for any checkpoint's chunk length (a fixed --replans "
        "count quietly scales the budget with chunk size: 15 replans "
        "of 1-second molmoact2 chunks was 15 s, not the intended 30)",
    )
    parser.add_argument("--execute-horizon", type=int, default=30)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument(
        "--draws",
        type=int,
        default=1,
        help="stochastic rollouts per seed (GRPO signal probe): draw 0 "
        "is the deterministic banked-identity row and the only one that "
        "records video; draws >= 1 re-key every policy noise/sampling "
        "stream per draw (flow fresh-noise groups need no other flag; "
        "AR groups also want --ar-temperature)",
    )
    parser.add_argument(
        "--ar-temperature",
        type=float,
        default=None,
        help="sample the AR head at this temperature instead of the "
        "greedy deployment decode (BijouPolicy knob; the row/report "
        "name carries _t<T>)",
    )
    parser.add_argument(
        "--sde-noise-level",
        type=float,
        default=None,
        help="decode flow actions with the Euler–Maruyama SDE at this "
        "noise scale a instead of the deterministic ODE (GRPO probe "
        "cell 5; requires --method euler; the row/report name carries "
        "_sde<a>; per-step noise is keyed per (seed, replan, draw) so "
        "rows stay batch-invariant and reproducible)",
    )
    parser.add_argument(
        "--method",
        default="heun",
        choices=["euler", "heun"],
        help="ODE solver: heun for full-flow checkpoints, euler for "
        "1-NFE SnapFlow students (euler-1 IS their training target)",
    )
    parser.add_argument(
        "--flow-decoder-dtype",
        default="bfloat16",
        choices=["float32", "bfloat16"],
        help="post-load cast of the checkpoint's action decoder "
        "(bfloat16 halves flow-decoder memory)",
    )
    parser.add_argument(
        "--post-backend",
        default="auto",
        choices=["auto", "numpy", "torch"],
        help="SO101Sim compositor per worker (auto/torch means one CUDA "
        "context per worker, ~0.5-1 GiB VRAM each; numpy frames differ "
        "from torch by the pinned <=2/255 compositor tolerance)",
    )
    parser.add_argument(
        "--no-mount-flip",
        action="store_true",
        help="run the PRE-flip wrist-bracket physics (mirrored Menagerie "
        "mount) — paired flip-effect reads only; flipped is the "
        "registered geometry",
    )
    parser.add_argument(
        "--convmap-seam-stats",
        type=Path,
        default=None,
        help="OFF-CONTRACT release-in-sim arm (sim.convmap): checkpoint "
        "dir whose normalization table states the sim seam's units (the "
        "ftrig rig-recomputed table); fits the discrete convention map "
        "seam -> checkpoint table and wraps the policy with it (state "
        "in through A, chunks back through A⁻¹). The policy name and "
        "the rows carry _convmap — never pooled with contract reads",
    )
    parser.add_argument(
        "--convmap-override",
        action="append",
        default=[],
        metavar="JOINT=[SIGN,]OFFSET",
        help="explicit per-joint convention override (degrees; "
        "'lift=-1,90' carries a mirror, bare 'elbow_flex=90' keeps sign "
        "+1) replacing the gated fit — only with tripwire-script "
        "evidence (coverage + first-action) or an externally documented "
        "conversion; recorded verbatim in the rows JSON",
    )
    parser.add_argument(
        "--rows-jsonl",
        type=Path,
        default=None,
        help="append one JSON line per episode AS IT COMPLETES "
        "(completion order under workers>1) — the live-progress stream "
        "a watcher can tail; the authoritative rows stay in --out-json",
    )
    parser.add_argument(
        "--emit-training-rows",
        type=Path,
        default=None,
        metavar="DIR",
        help="write one NPZ per (seed, draw, replan) predict — frames "
        "(jpeg), sampled ids, per-token chosen logprobs, state, grammar "
        "masks — the token-GRPO replay surface (AR-suffix checkpoints "
        "only; rollout rows are untouched, capture is observation)",
    )
    parser.add_argument(
        "--molmoact2-discrete",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="OFF-CONTRACT release-in-sim arm, DISCRETE (AR) pathway: "
        "serve the first-class MolmoAct2DiscreteStack (the AR read of a "
        "BIJOU molmoact2-family checkpoint — converted release/rigtable "
        "or an ar/joint descendant; retirement phase 4 re-point) — "
        "grammar-masked, batch-1 per request, state in / chunks back "
        "through the --joint-frame map (identity for v3.0-frame tables; "
        "the official SO-101 shim for unremapped v2.1 releases). Rows "
        "never pool with contract reads",
    )
    parser.add_argument(
        "--molmoact2-fast-tokenizer",
        type=str,
        default="allenai/MolmoAct2-FAST-Tokenizer",
        help="released OpenFAST artifact (dir or hub id) for --molmoact2-discrete",
    )
    parser.add_argument(
        "--molmoact2-grammar-masked",
        action="store_true",
        help="decode with the budget-arithmetic grammar mask (the RL "
        "decode mode — every emission decodable by construction) "
        "instead of the reference's unconstrained greedy + zeros "
        "fallback; per-request violations are counted in the out-json",
    )
    parser.add_argument(
        "--molmoact2-temperature",
        type=float,
        default=None,
        help="sample the masked softmax at this temperature instead of "
        "the masked argmax (the RL rollout draw, keyed per (seed, "
        "replan, draw) via stable_sample_rng) — requires "
        "--molmoact2-grammar-masked",
    )
    parser.add_argument(
        "--joint-frame",
        choices=JOINT_FRAME_CHOICES,
        default="auto",
        help="calibration frame map around the --molmoact2-discrete "
        "predict seam (state in, chunks back out). 'rig' = identity "
        "(v3.0-frame tables: bijou-trained checkpoints, corrected-table "
        "recomputes, conversion-remapped releases). 'v30-to-v21' = the "
        "official lerobot PR#777 shim (unremapped v2.1-table releases "
        "only — the port-era convention). 'auto' fingerprints the "
        "checkpoint's state table (docs/so101-joint-conventions.md §4) "
        "and refuses an unclassifiable one; explicit modes are refused "
        "on a classified mismatch (the R2 wave-0 kill class)",
    )
    parser.add_argument(
        "--wrist-transform",
        default="none",
        choices=WRIST_TRANSFORMS,
        help="wrist-transfer screen treatment (pre-reg 2026-08-14 §1): "
        "rewrite ONLY the wrist frame the policy sees — physics, video "
        "and every state read stay raw; 'freeze' replays the reset "
        "frame, 'arm_blur' is the W3 masked-arm corruption",
    )
    parser.add_argument(
        "--top-transform",
        default="none",
        choices=TOP_TRANSFORMS,
        help="T1 positive control (same pre-reg, arm grid): blackout "
        "the TOP frame the policy sees — same obs-only contract as "
        "--wrist-transform, composable with it",
    )
    parser.add_argument(
        "--clutter-appearance",
        default="patched",
        choices=("patched", "standins"),
        help="v3/v4 clutter appearance (promotion 2026-08-18): "
        "'patched' pastes the mined real crops onto the drawn plate "
        "(production default); 'standins' renders the pre-promotion "
        "stand-in geoms — pin it for reads registered on the old "
        "substrate (e.g. the pdnorm sim100 grid vs the 11/100 "
        "baseline)",
    )
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="write a config header + per-seed rows (incl. per-tick "
        "distance series) for the reads instrument",
    )
    parser.add_argument(
        "--stats-repo-id",
        default=None,
        help="the dataset row sim items wear (a key of the checkpoint's "
        f"per-dataset table; default: {STATS_REPO_ID!r} else the merged "
        "table). Per-dataset-flow-norm checkpoints trained on a mix "
        "denormalize flow chunks under the WORN row — sim episodes must "
        "wear the sim demos' row, not the rig's",
    )
    args = parser.parse_args()
    modes = sum(
        [args.checkpoint is not None, args.hold, args.molmoact2_discrete is not None],
    )
    if modes != 1:
        parser.error(
            "exactly one of --checkpoint / --hold / --molmoact2-discrete is required",
        )
    if args.stats_repo_id is not None and args.checkpoint is None:
        parser.error(
            "--stats-repo-id picks a bijou policy's worn row — only "
            "meaningful with --checkpoint",
        )
    if args.joint_frame != "auto" and args.molmoact2_discrete is None:
        parser.error(
            "--joint-frame maps the --molmoact2-discrete predict seam — "
            "the BijouPolicy path normalizes under the worn stats row "
            "with no frame map to pick",
        )
    if args.stats_repo_id is not None and args.convmap_seam_stats is not None:
        parser.error(
            "--stats-repo-id and --convmap-seam-stats both decide the worn "
            "stats — pick one (the seam already replaces the row wholesale)",
        )
    if args.molmoact2_discrete is not None:
        for flag, name in (
            (args.convmap_seam_stats is not None, "--convmap-seam-stats"),
            (args.ar_temperature is not None, "--ar-temperature"),
            (args.sde_noise_level is not None, "--sde-noise-level"),
        ):
            if flag:
                parser.error(
                    f"{name} is not wired for --molmoact2-discrete "
                    "(bijou-policy flags; the discrete pathway's sampling "
                    "flag is --molmoact2-temperature)",
                )
        if args.emit_training_rows is not None and not args.molmoact2_grammar_masked:
            parser.error(
                "--emit-training-rows on the discrete pathway records the "
                "masked-softmax capture surface — pass "
                "--molmoact2-grammar-masked (the RL rollout decode)",
            )
        if args.draws > 1 and args.molmoact2_temperature is None:
            parser.error(
                "--draws > 1 without --molmoact2-temperature replays the "
                "identical greedy stream per draw — meaningless",
            )
    if args.molmoact2_grammar_masked and args.molmoact2_discrete is None:
        parser.error("--molmoact2-grammar-masked requires --molmoact2-discrete")
    if args.molmoact2_temperature is not None and not args.molmoact2_grammar_masked:
        parser.error(
            "--molmoact2-temperature requires --molmoact2-grammar-masked "
            "— unconstrained sampling would sample the zeros-fallback "
            "class (a measured 6.8% zero-fallback rate on unconstrained "
            "greedy)",
        )
    if args.molmoact2_temperature is not None and args.molmoact2_temperature <= 0:
        parser.error(
            f"--molmoact2-temperature must be > 0, got {args.molmoact2_temperature}",
        )
    if args.convmap_seam_stats is not None and args.hold:
        parser.error("--convmap-seam-stats wraps a policy — meaningless with --hold")
    if args.emit_training_rows is not None and args.hold:
        parser.error(
            "--emit-training-rows records a policy's token stream — "
            "meaningless with --hold",
        )
    if args.convmap_override and args.convmap_seam_stats is None:
        parser.error("--convmap-override requires --convmap-seam-stats")
    if args.replans is not None and args.episode_seconds is not None:
        parser.error(
            "--replans and --episode-seconds state the same budget in "
            "two units — pick one",
        )
    if args.episode_seconds is not None and args.episode_seconds <= 0:
        parser.error(f"--episode-seconds must be > 0, got {args.episode_seconds}")
    if args.draws < 1:
        parser.error(f"--draws must be >= 1, got {args.draws}")
    if args.draws > 1 and args.hold:
        parser.error("--draws > 1 is meaningless for --hold (deterministic)")
    if args.wrist_transform != "none" and args.hold:
        parser.error(
            "--wrist-transform rewrites the policy's wrist input — "
            "meaningless with --hold (which never looks at frames)",
        )
    if args.top_transform != "none" and args.hold:
        parser.error(
            "--top-transform rewrites the policy's top input — "
            "meaningless with --hold (which never looks at frames)",
        )
    if args.sde_noise_level is not None:
        if args.hold:
            parser.error("--sde-noise-level decodes a policy — meaningless with --hold")
        if args.ar_temperature is not None:
            parser.error(
                "--sde-noise-level and --ar-temperature sample "
                "different decoder families — pick one",
            )
        if args.method != "euler":
            parser.error("the SDE decode is Euler-only — pass --method euler")
    return args


# The canonical SO-101 shim the flow-pathway convmap eval validated
# (posts/2026-08-12-prereg-release-eval20-convmap.md amendment 3, the
# 9/100 run): the official LeRobot->MolmoAct2 snippet map exactly —
# shoulder_lift mirrored, lift/elbow offset +90 deg. State goes IN
# through it, chunks come BACK through its inverse. These literals are
# JointFrameTransform.lerobot_v30_to_v21()'s, spelled locally so sim
# workers never import the policy stack (test-pinned equal).
MOLMOACT2_OFFICIAL_SIGNS = (1.0, -1.0, 1.0, 1.0, 1.0, 1.0)
MOLMOACT2_OFFICIAL_OFFSETS_DEG = (0.0, 90.0, 90.0, 0.0, 0.0, 0.0)
MOLMOACT2_NORM_TAG = "so100_so101_molmoact2"

# --joint-frame for the discrete (AR) serving seam. The shim above is
# CORRECT only for a checkpoint whose recorded q01/q99 table is in the
# pre-PR#777 ("v2.1") calibration frame — the unremapped official
# MolmoAct2 releases the port-era predictor served. Every bijou-format
# table today is v3.0-frame (bijou-trained on rig/sim data, the
# corrected-table recomputes, and conversion-remapped releases alike),
# and serving one through the shim is the R2 wave-0 kill's root cause:
# lift/elbow state bins clamp at the table edge and the inverted chunk
# map drives the arm out of range (docs/so101-joint-conventions.md §6:
# a missing OR extra remap each half-breaks the arm, and nothing
# downstream catches it — so this seam refuses mismatches at load).
JOINT_FRAME_CHOICES = ("auto", "rig", "v30-to-v21")


def classify_state_frame(
    state_q01: Sequence[float] | None,
    state_q99: Sequence[float] | None,
) -> str | None:
    """SO-101 calibration fingerprint of a STATE quantile table in
    degrees (docs/so101-joint-conventions.md §4): ``'v21'`` = the
    pre-PR#777 frame official releases trained under; ``'v30'`` = the
    rig/sim frame (bijou-trained tables, and conversion-remapped ones
    — their flipped shoulder_lift lands as a DESCENDING pair); None =
    unclassifiable (the doc's rule: ask, don't guess). Keys on
    shoulder_lift — the sign axis is the expensive one — with
    elbow_flex corroborating the v2.1 read."""
    if state_q01 is None or state_q99 is None or len(state_q01) < 3:
        return None
    lift_q01, lift_q99 = float(state_q01[1]), float(state_q99[1])
    if lift_q01 > lift_q99:
        # Only the conversion-time flip remap writes descending pairs.
        return "v30"
    if lift_q01 >= 30.0:
        # v2.1 lift sits entirely above +30 (fingerprint 45→186);
        # demand the elbow corroborate (35→174) before claiming it.
        elbow_q01, elbow_q99 = float(state_q01[2]), float(state_q99[2])
        return "v21" if elbow_q01 >= 25.0 and elbow_q99 >= 130.0 else None
    if lift_q01 <= -30.0:
        # v3.0 lift reaches far negative (rig range ≈ −103..+29).
        return "v30"
    return None


def resolve_joint_frame(mode: str, stats: Any) -> tuple[str, Any]:
    """(resolved ``'rig'`` | ``'v30-to-v21'``, the AffineMap the
    discrete predict seam applies) for a checkpoint's recorded stats
    table. ``'auto'`` classifies the state table and refuses an
    unclassifiable one; an EXPLICIT mode that contradicts a classified
    table is refused too — a convention mismatch on this seam is never
    a judgment call (module comment above). Parent-side only."""
    import torch

    from bijou.eval.molmo_norm import AffineMap
    from bijou.rollout_safety import JointFrameTransform

    if mode not in JOINT_FRAME_CHOICES:
        raise SystemExit(
            f"--joint-frame must be one of {JOINT_FRAME_CHOICES}, got {mode!r}",
        )
    fingerprint = classify_state_frame(stats.state_q01, stats.state_q99)
    implied = {"v21": "v30-to-v21", "v30": "rig"}.get(fingerprint or "")
    if mode == "auto":
        if implied is None:
            raise SystemExit(
                "--joint-frame auto: the checkpoint's state table matches "
                "no SO-101 calibration fingerprint "
                f"(shoulder_lift q01/q99 = {stats.state_q01[1]}/"
                f"{stats.state_q99[1]}, elbow_flex = {stats.state_q01[2]}/"
                f"{stats.state_q99[2]}) — classify it against "
                "docs/so101-joint-conventions.md §4 and pass --joint-frame "
                "rig or v30-to-v21 explicitly (the doc's rule: ask, don't "
                "guess)",
            )
        resolved = implied
    else:
        if implied is not None and implied != mode:
            raise SystemExit(
                f"--joint-frame {mode} contradicts the checkpoint's state "
                f"table, which fingerprints as the "
                f"{'v2.1' if fingerprint == 'v21' else 'v3.0'} frame "
                f"(shoulder_lift q01/q99 = {stats.state_q01[1]}/"
                f"{stats.state_q99[1]}; docs/so101-joint-conventions.md "
                "§4) — a missing remap clamps lift/elbow state bins at the "
                "table edge and a double remap half-breaks the arm (§6), "
                "so this seam refuses the mismatch instead of serving it",
            )
        resolved = mode
    frame = (
        JointFrameTransform.lerobot_v30_to_v21()
        if resolved == "v30-to-v21"
        else JointFrameTransform.identity(len(stats.state_q01))
    )
    shim = AffineMap(
        scale=torch.tensor(frame.signs, dtype=torch.float32),
        offset=torch.tensor(frame.offsets, dtype=torch.float32),
    )
    return resolved, shim


def molmoact2_discrete_chunks(
    predictor: Any,
    shim: Any,
    requests: list[tuple[Any, ...]],
    *,
    task: str,
    grammar_masked: bool,
    temperature: float | None = None,
    rng_for: Callable[[int, int, int], Any] | None = None,
    token_rows: list[Any] | None = None,
    model_states: list[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], list[bool]]:
    """Parent-side predict round for the discrete (AR) pathway: batch-1
    per request, strictly in request order (the predictor's prompt
    packing is single-observation). Camera order [top, wrist] matches
    the sorted ``observation.images.*`` convention the parity anchors
    packed. Returns (sim-unit chunks, per-request zero-fallback flags —
    always False under the grammar mask, which decodes by
    construction).

    ``temperature`` + ``rng_for`` (the RL rollout draw): sample the
    masked softmax with one keyed generator per request —
    ``rng_for(seed, replan, draw)``. ``token_rows``/``model_states``
    are ``--emit-training-rows`` out-lists: one TokenRow (the
    masked-softmax capture reduced by ``token_rows_from_capture``) and
    one MODEL-unit state vector per request, in request order — the
    replay collator consumes the state the predictor consumed, not the
    sim's."""
    import torch

    from bijou.eval.policies import token_rows_from_capture

    if (temperature is None) != (rng_for is None):
        raise ValueError("temperature and rng_for come together")
    chunks: list[np.ndarray] = []
    fallbacks: list[bool] = []
    for _, _, seed, replan, draw, top, wrist, state in requests:
        model_state = shim.apply(
            torch.from_numpy(np.asarray(state, dtype=np.float32)),
        )
        capture: list[Any] | None = [] if token_rows is not None else None
        result = predictor.predict_action_discrete(
            images=[top, wrist],
            task=task,
            state=model_state,
            grammar_masked=grammar_masked,
            on_undecodable="zeros",
            temperature=temperature,
            sample_rng=None if rng_for is None else rng_for(seed, replan, draw),
            action_capture=capture,
        )
        if token_rows is not None:
            assert capture is not None  # built together above
            rows = token_rows_from_capture(
                capture,
                block_base=predictor.action_token_start_id,
                temperature=temperature,
            )
            token_rows.append(rows[0])
        if model_states is not None:
            model_states.append(model_state.numpy())
        codec = predictor.fast_codec
        horizon = int(predictor.metadata.get("action_horizon") or 0)
        total = horizon * int(predictor.action_stats.q01.numel())
        decodable = (
            bool(result.bins) and int(codec.symbol_lengths[result.bins].sum()) == total
        )
        fallbacks.append(not decodable)
        chunks.append(shim.invert(result.actions[0]).numpy())
    return chunks, fallbacks


def run_worker_episodes(
    sim: RolloutSim,
    config: WorkerConfig,
    send: Callable[[tuple[Any, ...]], None],
    recv: Callable[[], np.ndarray],
) -> None:
    """One worker's (seed, draw) slice, strictly in order.
    ``send``/``recv`` are the predict round-trip transport — a Pipe in
    production, plain queues in the CPU-tier oracle. Timing recorded per
    replan is the round-trip wait (batch forward + lockstep barrier),
    not a solo forward."""
    for seed, draw in config.units:
        latencies: list[float] = []
        wrist_hook = make_wrist_transform(config.wrist_transform, sim)
        transform = chain_transforms(
            wrist_hook,
            make_top_transform(config.top_transform),
        )
        next_chunk: Callable[[SimObservation, int], np.ndarray]

        if config.hold:
            next_chunk = hold_chunk_fn(config.horizon)
        else:

            def remote_chunk(
                obs: SimObservation,
                replan: int,
                _seed: int = seed,
                _draw: int = draw,
                _latencies: list[float] = latencies,
            ) -> np.ndarray:
                start = time.perf_counter()
                send(
                    (
                        "predict",
                        config.worker_id,
                        _seed,
                        replan,
                        _draw,
                        obs.top,
                        obs.wrist,
                        obs.state,
                    ),
                )
                chunk = recv()
                _latencies.append((time.perf_counter() - start) * 1000)
                return chunk

            next_chunk = remote_chunk

        video_path = (
            config.out_dir / f"rollout_seed{seed:03d}.mp4"
            if config.out_dir is not None and draw == 0
            else None
        )
        row = run_episode_loop(
            sim,
            seed,
            next_chunk,
            replans=config.replans,
            horizon=config.horizon,
            video_path=video_path,
            latencies=latencies,
            transform=transform,
        )
        print_coverage(wrist_hook, seed, draw)
        if draw:
            row = dataclasses.replace(row, draw=draw)
        send(("row", config.worker_id, row))
    send(("done", config.worker_id))


def _worker_main(config: WorkerConfig, conn: Connection) -> None:
    try:
        sim = SO101Sim(
            post_backend=config.post_backend,
            clutter_appearance=config.clutter_appearance,
            flip_camera_mount=config.flip_camera_mount,
        )
        run_worker_episodes(sim, config, conn.send, conn.recv)
    except Exception:  # noqa: BLE001 — shipped whole to the parent, which raises
        conn.send(("error", config.worker_id, traceback.format_exc()))


class WorkerDiedError(RuntimeError):
    pass


class TrainingRowWriter:
    """``--emit-training-rows`` sink: one NPZ per (seed, draw, replan)
    predict — the two observation frames as encoded JPEG bytes, the
    state vector, and the policy's TokenRow (sampled codec ids,
    per-token chosen logprobs under the decode's own masked softmax,
    bit-packed grammar masks) — plus ``meta.json`` (run-level RNG-key
    provenance: the sampling stream is fully determined by the recorded
    run seed + each row's (seed, draw, replan) identity through
    ``stable_sample_rng``) and an append-only ``index.jsonl``. Frames
    are stored pre-normalization (the raw uint8 the item wore), so the
    replay collator re-encodes exactly the rollout's prompt inputs;
    JPEG is the registered lossy budget (~0.3 GB/step, pruned after the
    gradient pass)."""

    def __init__(self, root: Path, meta: dict[str, Any]) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        (root / "meta.json").write_text(json.dumps(meta, indent=1))
        self.index_path = root / "index.jsonl"
        self.index_path.write_text("")  # truncate a stale stream
        self.rows_written = 0

    @staticmethod
    def _jpeg(frame: np.ndarray) -> np.ndarray:
        from PIL import Image  # parent-only; workers never construct a writer

        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format="JPEG", quality=92)
        return np.frombuffer(buffer.getvalue(), dtype=np.uint8)

    def write(
        self,
        *,
        seed: int,
        replan: int,
        draw: int,
        top: np.ndarray,
        wrist: np.ndarray,
        state: np.ndarray,
        row: Any,  # bijou.eval.policies.TokenRow (parent-only import)
    ) -> None:
        name = f"seed{seed:03d}_draw{draw:02d}_replan{replan:03d}.npz"
        with (self.root / name).open("wb") as stream:
            np.savez_compressed(
                stream,
                top_jpeg=self._jpeg(top),
                wrist_jpeg=self._jpeg(wrist),
                state=np.asarray(state, dtype=np.float32),
                ids=row.ids,
                logprobs=row.logprobs,
                allowed_packed=row.allowed_packed,
            )
        with self.index_path.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "path": name,
                        "seed": seed,
                        "draw": draw,
                        "replan": replan,
                        "tokens": int(row.ids.shape[0]),
                        "vocab_total": int(row.vocab_total),
                        "temperature": float(row.temperature),
                    },
                )
                + "\n",
            )
        self.rows_written += 1


def serve(
    conns: Sequence[Any],
    predict_batch: Callable[[list[tuple[Any, ...]]], list[np.ndarray]],
    on_row: Callable[[EpisodeResult], None],
) -> list[int]:
    """The lockstep-rounds scheduler. ``conns`` is indexed by worker_id
    (anything with .send/.recv). Returns the per-round batch sizes — a
    deterministic trace the oracle asserts on. Raises WorkerDiedError
    with the worker's traceback if one errors out."""
    active = list(range(len(conns)))
    batch_sizes: list[int] = []
    while active:
        requests: list[tuple[Any, ...]] = []
        still_active: list[int] = []
        for worker_id in active:
            while True:
                try:
                    message = conns[worker_id].recv()
                except EOFError as error:
                    raise WorkerDiedError(
                        f"worker {worker_id} closed its pipe without a "
                        "done/error message — check the worker's stderr",
                    ) from error
                tag = message[0]
                if tag == "row":
                    on_row(message[2])
                elif tag == "done":
                    break
                elif tag == "predict":
                    requests.append(message)
                    still_active.append(worker_id)
                    break
                elif tag == "error":
                    raise WorkerDiedError(f"worker {message[1]} died:\n{message[2]}")
                else:
                    raise WorkerDiedError(f"unknown worker message tag {tag!r}")
        active = still_active
        if requests:
            chunks = predict_batch(requests)
            for message, chunk in zip(requests, chunks, strict=True):
                conns[message[1]].send(chunk)
            batch_sizes.append(len(requests))
    return batch_sizes


def main() -> int:
    # Policy-side imports are parent-only; workers re-import this module
    # under spawn and must not pay for (or touch) the policy stack there.
    import torch

    from bijou.eval.molmo_norm import MolmoNorm
    from bijou.eval.policies import BijouPolicy
    from bijou.modelling.decoders.flow import SamplingMethod

    from .convmap import seam_convention_map

    args = parse_args()
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    predictor = None
    discrete_shim = None
    # Resolved --joint-frame for the out-json record; stays None off
    # the discrete arm.
    joint_frame: str | None = None
    # Resolved worn-row identity for the out-json record; stays None on
    # arms that wear no bijou stats row (hold, molmoact2 shim).
    worn_row: str | None = None
    if args.hold:
        policy = None
        horizon = args.execute_horizon
        print(f"policy: hold (settled reset state, horizon {horizon})")
    elif args.molmoact2_discrete is not None:
        from bijou.checkpoint import read_metadata
        from bijou.grpo_replay import MolmoAct2DiscreteStack

        policy = None
        # The retirement phase-4 re-point: the discrete policy is the
        # first-class AR read of a BIJOU molmoact2-family checkpoint
        # (converted release / rigtable / ar-joint descendants). The
        # port predictor (HF-layout dirs, their norm tags) retired
        # with bijou/molmoact2.
        joint_frame, discrete_shim = resolve_joint_frame(
            args.joint_frame,
            read_metadata(Path(args.molmoact2_discrete)).stats,
        )
        predictor = MolmoAct2DiscreteStack.load(
            args.molmoact2_discrete,
            device=device,
            dtype=torch.bfloat16,
            fast_tokenizer=args.molmoact2_fast_tokenizer,
        )
        tag_horizon = int(predictor.metadata.get("action_horizon") or 0)
        horizon = min(args.execute_horizon, tag_horizon or args.execute_horizon)
        print(
            f"policy: molmoact2-discrete {args.molmoact2_discrete} "
            f"({'grammar-masked' if args.molmoact2_grammar_masked else 'reference greedy'}, "
            f"horizon {horizon}, joint frame {joint_frame} — shim signs "
            f"{discrete_shim.scale.tolist()} offsets "
            f"{discrete_shim.offset.tolist()})",
        )
    else:
        policy = BijouPolicy(
            args.checkpoint,
            device=device,
            seed=args.seed,
            sample_steps=args.sample_steps,
            method=SamplingMethod[args.method.upper()],
            ar_temperature=args.ar_temperature,
            sde_noise_level=args.sde_noise_level,
            flow_decoder_dtype=getattr(torch, args.flow_decoder_dtype),
            molmo_norm=(
                MolmoNorm.CONVENTION_MAP
                if args.convmap_seam_stats is not None
                else MolmoNorm.CHECKPOINT
            ),
        )
        horizon = min(args.execute_horizon, policy.info.chunk_size)
        print(
            f"policy: {policy.name} "
            f"({args.method}-{args.sample_steps}, horizon {horizon})",
        )
    row_writer: TrainingRowWriter | None = None
    if args.emit_training_rows is not None and predictor is not None:
        row_writer = TrainingRowWriter(
            args.emit_training_rows,
            {
                "checkpoint": str(args.molmoact2_discrete),
                "run_seed": args.seed,
                "decode": "molmoact2_grammar_masked",
                "temperature": args.molmoact2_temperature,
                "task": TASK,
                # The stored state is MODEL units — the resolved joint
                # frame applied, exactly what predict_action_discrete
                # consumed (the replay collator feeds it back verbatim).
                "state_units": f"model ({joint_frame} joint frame applied)",
                "joint_frame": joint_frame,
                "norm_tag": MOLMOACT2_NORM_TAG,
                "stats_repo_id": STATS_REPO_ID,
                "commit": commit,
                "rng_key": "stable_sample_rng(run_seed, repo_id(draw), seed, replan, 0)",
            },
        )
        print(f"training rows -> {args.emit_training_rows}")
    elif args.emit_training_rows is not None:
        assert policy is not None  # parse_args refused --hold
        # Capability narrowing (the trait handles, never a concrete
        # class): the rows instrument records the SERVING decode's
        # token stream, so the family must both carry an AR decoder
        # and serve through it (a joint family serves flow — its rows
        # would not be the executed decode's).
        if policy.ar is None or policy.flow is not None:
            raise SystemExit(
                "--emit-training-rows records the AR-serving token "
                f"stream, but {policy.spec.family.value} "
                + (
                    "serves through its flow decoder"
                    if policy.ar is not None
                    else "has no AR action decoder"
                ),
            )
        policy.capture_token_rows = True
        row_writer = TrainingRowWriter(
            args.emit_training_rows,
            {
                "checkpoint": str(args.checkpoint),
                "run_seed": args.seed,
                "ar_temperature": args.ar_temperature,
                "sample_steps": args.sample_steps,
                "method": args.method,
                "flow_decoder_dtype": args.flow_decoder_dtype,
                "stats_repo_id": args.stats_repo_id,
                "worn_row": (
                    "<convmap-seam>"
                    if args.convmap_seam_stats is not None
                    else worn_stats_key(policy.info, args.stats_repo_id)
                ),
                "commit": commit,
                # The RNG-key convention: each row's sampling stream is
                # stable_sample_rng(run_seed, repo_id(draw),
                # episode_index=seed, frame_index=replan, draw=0) with
                # repo_id "sim/eval100" at draw 0 and
                # "sim/eval100/drawNN" otherwise (sim_item's keying).
                "rng_key": "stable_sample_rng(run_seed, repo_id(draw), seed, replan, 0)",
            },
        )
        print(f"training rows -> {args.emit_training_rows}")
    replans = resolve_replans(args.replans, args.episode_seconds, horizon)
    print(
        f"episode budget: {replans} replans x {horizon} ticks = "
        f"{replans * horizon / CONTROL_HZ:.1f} s at {CONTROL_HZ} Hz",
    )

    seeds = list(range(args.seed, args.seed + args.num_seeds))
    # Seed-major, draw-minor — the sequential driver's loop order, so
    # the round-robin partition (and therefore the lockstep batch
    # trace) is a pure function of (seed range, draws, worker count).
    units = [(seed, draw) for seed in seeds for draw in range(args.draws)]
    workers = max(1, min(args.workers, len(units)))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    context = mp.get_context("spawn")
    processes: list[Any] = []
    conns: list[Connection] = []
    for worker_id in range(workers):
        parent_conn, child_conn = context.Pipe()
        config = WorkerConfig(
            worker_id=worker_id,
            units=tuple(units[worker_id::workers]),
            replans=replans,
            horizon=horizon,
            hold=args.hold,
            out_dir=args.out_dir,
            post_backend=args.post_backend,
            clutter_appearance=args.clutter_appearance,
            flip_camera_mount=not args.no_mount_flip,
            wrist_transform=args.wrist_transform,
            top_transform=args.top_transform,
        )
        process = context.Process(
            target=_worker_main,
            args=(config, child_conn),
            daemon=True,
        )
        process.start()
        child_conn.close()
        processes.append(process)
        conns.append(parent_conn)
    print(
        f"spawned {workers} env workers for {len(seeds)} seeds x {args.draws} draw(s)",
        flush=True,
    )

    results: list[EpisodeResult] = []
    predict_ms: list[float] = []
    seam = None
    discrete_fallbacks: list[bool] = []

    if predictor is not None:
        from bijou.eval.policies import stable_sample_rng

        def discrete_rng_for(seed: int, replan: int, draw: int) -> Any:
            repo = "sim/eval100" if draw == 0 else f"sim/eval100/draw{draw:02d}"
            return stable_sample_rng(args.seed, repo, seed, replan, 0)

        def predict_batch(requests: list[tuple[Any, ...]]) -> list[np.ndarray]:
            start = time.perf_counter()
            token_rows: list[Any] | None = [] if row_writer is not None else None
            states: list[np.ndarray] | None = [] if row_writer is not None else None
            chunks, fallbacks = molmoact2_discrete_chunks(
                predictor,
                discrete_shim,
                requests,
                task=TASK,
                grammar_masked=args.molmoact2_grammar_masked,
                temperature=args.molmoact2_temperature,
                rng_for=(
                    discrete_rng_for if args.molmoact2_temperature is not None else None
                ),
                token_rows=token_rows,
                model_states=states,
            )
            predict_ms.append((time.perf_counter() - start) * 1000)
            discrete_fallbacks.extend(fallbacks)
            if row_writer is not None:
                assert token_rows is not None and states is not None
                for message, row, model_state in zip(
                    requests,
                    token_rows,
                    states,
                    strict=True,
                ):
                    _, _, seed, replan, draw, top, wrist, _ = message
                    row_writer.write(
                        seed=seed,
                        replan=replan,
                        draw=draw,
                        top=top,
                        wrist=wrist,
                        state=model_state,
                        row=row,
                    )
            return chunks
    elif policy is None:

        def predict_batch(requests: list[tuple[Any, ...]]) -> list[np.ndarray]:
            raise WorkerDiedError("hold arm workers must not request predicts")
    else:
        chunk_size = policy.info.chunk_size
        if args.convmap_seam_stats is not None:
            # Off-contract seam (sim.convmap): items wear the SEAM's
            # stats (the units the sim actually speaks), and the policy's
            # per-repo map cache is seeded with the resolved fit — so an
            # override rides the exact rewrite path the gated fit would,
            # and the policy never re-fits behind our back.
            seam = seam_convention_map(
                args.convmap_seam_stats,
                policy.info.normalization,
                args.convmap_override,
            )
            policy._molmo_norm_maps["sim/eval100"] = seam.item_maps
            stats = seam.seam_stats
            worn_row = "<convmap-seam>"
            print(
                f"convmap seam {args.convmap_seam_stats.name}: "
                f"scale {seam.map.scale.tolist()} "
                f"offset {seam.map.offset.tolist()} "
                f"(gated fit offset {seam.fit.map.offset.tolist()}, "
                f"overrides {seam.overrides or 'none'})",
                flush=True,
            )
        else:
            # Converted checkpoints (molmoact2 lineage) carry no
            # per-dataset table — their items must wear the checkpoint's
            # MERGED stats (same fallback as the sequential driver).
            stats = resolve_worn_stats(policy.info, args.stats_repo_id)
            worn_row = worn_stats_key(policy.info, args.stats_repo_id)

        def predict_batch(requests: list[tuple[Any, ...]]) -> list[np.ndarray]:
            items = []
            indices = []
            for _, _, seed, replan, draw, top, wrist, state in requests:
                obs = SimObservation(top=top, wrist=wrist, state=state)
                items.append(
                    sim_item(
                        obs,
                        seed,
                        replan,
                        stats=stats,
                        chunk_size=chunk_size,
                        draw=draw,
                    ),
                )
                indices.append(replan)
            start = time.perf_counter()
            chunks = policy.predict(items, indices)
            predict_ms.append((time.perf_counter() - start) * 1000)
            if row_writer is not None:
                rows = policy.last_token_rows
                if rows is None or len(rows) != len(requests):
                    raise WorkerDiedError(
                        "--emit-training-rows: predict retained "
                        f"{0 if rows is None else len(rows)} token rows "
                        f"for {len(requests)} requests — the capture "
                        "surface broke, stop",
                    )
                for message, row in zip(requests, rows, strict=True):
                    _, _, seed, replan, draw, top, wrist, state = message
                    row_writer.write(
                        seed=seed,
                        replan=replan,
                        draw=draw,
                        top=top,
                        wrist=wrist,
                        state=state,
                        row=row,
                    )
            return [chunk.numpy() for chunk in chunks]

    if args.rows_jsonl is not None:
        args.rows_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.rows_jsonl.write_text("")  # truncate a stale stream

    def record_row(row: EpisodeResult) -> None:
        results.append(row)
        if args.rows_jsonl is not None:
            with args.rows_jsonl.open("a") as stream:
                stream.write(
                    json.dumps(
                        {**asdict(row), "progress_final_cm": row.progress_final_cm},
                    )
                    + "\n",
                )

    started = time.perf_counter()
    try:
        batch_sizes = serve(conns, predict_batch, record_row)
    finally:
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
    wall_s = time.perf_counter() - started
    results.sort(key=lambda r: (r.seed, r.draw))

    print("\nseed | draw | init cm | min cm | final cm | progress cm | success")
    for r in sorted(results, key=lambda r: -r.progress_cm):
        success = f"tick {r.success_tick}" if r.success_tick is not None else "-"
        print(
            f"{r.seed:4d} | {r.draw:4d} | {r.initial_cm:7.1f} | {r.min_cm:6.1f} | "
            f"{r.final_cm:8.1f} | {r.progress_cm:11.1f} | {success}",
        )
    mean_batch = sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0.0
    print(
        f"\n{len(results)} episodes in {wall_s / 60:.1f} min | "
        f"{len(batch_sizes)} predict rounds, mean batch {mean_batch:.1f}",
    )
    if row_writer is not None:
        print(
            f"training rows: {row_writer.rows_written} NPZs -> "
            f"{row_writer.root} (index.jsonl + meta.json)",
        )
    if predictor is not None:
        print(
            f"discrete decode: {sum(discrete_fallbacks)} zero-fallback "
            f"emission(s) across {len(discrete_fallbacks)} predicts "
            f"({'grammar-masked' if args.molmoact2_grammar_masked else 'reference greedy'})",
        )

    if args.out_json is not None:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        payload = {
            "config": {
                "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "hold": args.hold,
                "seed": args.seed,
                "num_seeds": args.num_seeds,
                "replans": replans,
                "episode_seconds": args.episode_seconds,
                "execute_horizon": horizon,
                "sample_steps": args.sample_steps,
                "method": args.method,
                "draws": args.draws,
                "ar_temperature": args.ar_temperature,
                "sde_noise_level": args.sde_noise_level,
                "flow_decoder_dtype": args.flow_decoder_dtype,
                "wrist_transform": args.wrist_transform,
                "top_transform": args.top_transform,
                "control_hz": CONTROL_HZ,
                "task": TASK,
                "stats_repo_id": args.stats_repo_id,
                "worn_row": worn_row,
                "mount_flip": not args.no_mount_flip,
                "commit": commit,
                # Off-contract provenance: the resolved seam map (fit +
                # overrides) — None on contract reads. Rows under a
                # non-None convmap must never pool with contract rows.
                "convmap": (
                    None
                    if seam is None or policy is None
                    else {
                        "seam_stats": str(args.convmap_seam_stats),
                        "scale": seam.map.scale.tolist(),
                        "offset": seam.map.offset.tolist(),
                        "fit_offset": seam.fit.map.offset.tolist(),
                        "overrides": seam.overrides,
                        "policy_name": policy.name,
                    }
                ),
                # Off-contract provenance, discrete (AR) pathway: the
                # first-class predictor + the resolved joint-frame map.
                # Rows under a non-None record never pool with contract
                # reads OR with the flow-pathway convmap rows (different
                # serving stacks).
                "molmoact2_discrete": (
                    None
                    if predictor is None or discrete_shim is None
                    else {
                        "checkpoint": str(args.molmoact2_discrete),
                        "fast_tokenizer": str(args.molmoact2_fast_tokenizer),
                        "norm_tag": MOLMOACT2_NORM_TAG,
                        "grammar_masked": args.molmoact2_grammar_masked,
                        "joint_frame": joint_frame,
                        "shim_signs": discrete_shim.scale.tolist(),
                        "shim_offsets_deg": discrete_shim.offset.tolist(),
                        "zero_fallbacks": sum(discrete_fallbacks),
                        "predicts": len(discrete_fallbacks),
                    }
                ),
            },
            "parallel": {
                "workers": workers,
                "scheduler": "lockstep-v1",
                "post_backend": args.post_backend,
                "wall_s": round(wall_s, 1),
                "batch_sizes": batch_sizes,
                "predict_ms": [round(v, 1) for v in predict_ms],
            },
            "episodes": [
                {**asdict(r), "progress_final_cm": r.progress_final_cm} for r in results
            ],
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=1))
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
