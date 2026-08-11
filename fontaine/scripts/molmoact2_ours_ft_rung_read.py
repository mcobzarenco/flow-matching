"""G4 rung reads — our-trainer MolmoAct2 AE fine-tune vs the banked
anchors (port item 4).

Pre-reg posts/2026-08-10-prereg-molmoact2-firstclass-port.md, gate G4:
the ``bijou.molmoact2.train`` run must reproduce the rung-1 result
class — final rung beats BOTH anchors (zero-shot 28.9454, matched
state-copy 9.0824) on the same 240 anchor rows, with a
monotone-or-flat rung curve; the loss curve must sit in the run-1
corridor (frozen in the 2026-08-11 execution note: 5-point rolling
median of ``train/action_flow_loss`` within [0.5x, 2x] of the
reference series at every matched log step >= 100).

Each step dir is AE-only (+config/norm_stats); the trunk and tokenizer
compose from the INIT checkpoint — byte-identical to what the training
run conditioned on. Rows, per-row noise seeds, and the npz identity
oracles are the e2e parity harness's, verbatim.

Usage:
    uv run python fontaine/scripts/molmoact2_ours_ft_rung_read.py \
        --run-dir outputs/train/<run> [--steps 500 1000 1500 2000]
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

HORIZON = 30
NORM_TAG = "so100_so101_molmoact2"
BASE_SEED = 0  # the preflight noise convention: BASE_SEED + concat index
BANKED_STEM = "analysis__molmoact2_rig_preflight"  # rows/truths/states + anchors
ANCHOR_ZERO_SHOT = 28.9454
ANCHOR_STATE_COPY = 9.0824
#: Their-trainer run-1 rung MAEs (reports/, closed 2026-08-10) — the
#: result class G4 compares against, record-only per rung.
REFERENCE_RUNGS = {500: 6.7561, 1000: 4.66, 1500: 3.5871, 2000: 3.2301}
CORRIDOR_LOW, CORRIDOR_HIGH = 0.5, 2.0
CORRIDOR_MIN_STEP = 100
REFERENCE_CORRIDOR = (
    REPO_ROOT / "reports/analysis__molmoact2_rig_ft_r1_loss_corridor.json"
)
OUT_JSON = REPO_ROOT / "reports/analysis__molmoact2_ours_ft_rung_read.json"


def compose_predictor(
    step_dir: Path,
    trunk: Any,
    tokenizer: Any,
    image_token_ids: tuple[int, ...],
    *,
    device: str,
) -> Any:
    """A MolmoAct2Predictor over OUR step dir's expert + norm stats and
    the init checkpoint's trunk/tokenizer (predictor.load's field
    derivation, minus the trunk/tokenizer reload)."""
    from bijou.molmoact2.predictor import MolmoAct2Predictor, load_action_expert
    from bijou.molmoact2.processing import load_norm_stats
    from bijou.molmoact2.wiring import validate_inference_config

    config = json.loads((step_dir / "config.json").read_text())
    validate_inference_config(config)
    action_stats, state_stats, metadata = load_norm_stats(step_dir, NORM_TAG)
    expert = load_action_expert(step_dir, config, device=device, dtype=torch.bfloat16)
    return MolmoAct2Predictor(
        trunk=trunk,
        expert=expert,
        tokenizer=tokenizer,
        action_stats=action_stats,
        state_stats=state_stats,
        metadata=metadata,
        image_token_ids=image_token_ids,
        action_mode=str(config.get("action_mode", "continuous")),
        eos_token_id=(
            None if config.get("eos_token_id") is None else int(config["eos_token_id"])
        ),
        action_start_token_id=(
            None
            if config.get("action_start_token_id") is None
            else int(config["action_start_token_id"])
        ),
        action_end_token_id=(
            None
            if config.get("action_end_token_id") is None
            else int(config["action_end_token_id"])
        ),
        max_action_horizon=int(config["max_action_horizon"]),
        max_action_dim=int(config["max_action_dim"]),
        n_obs_steps=(
            1 if config.get("n_obs_steps") is None else int(config["n_obs_steps"])
        ),
        num_state_tokens=int(config["num_state_tokens"]),
        flow_matching_num_steps=int(config["flow_matching_num_steps"]),
        mask_action_dim_padding=bool(config["mask_action_dim_padding"]),
    )


def corridor_read(run_dir: Path) -> dict[str, Any]:
    """The frozen loss-corridor rule vs the reference series: 5-point
    rolling median ratio in [0.5, 2.0] at every matched step >= 100."""
    ref = json.loads(REFERENCE_CORRIDOR.read_text())["series"]
    ref_by_step = {int(r["step"]): float(r["loss"]) for r in ref}
    ours_by_step: dict[int, float] = {}
    log_path = run_dir / "train_log.jsonl"
    for line in log_path.read_text().splitlines():
        row = json.loads(line)
        ours_by_step[int(row["step"])] = float(row["action_flow_loss"])

    def rolling_median(series: dict[int, float]) -> dict[int, float]:
        steps = sorted(series)
        out = {}
        for i, s in enumerate(steps):
            window = [series[x] for x in steps[max(0, i - 2) : i + 3]]
            out[s] = statistics.median(window)
        return out

    ref_med = rolling_median(ref_by_step)
    ours_med = rolling_median(ours_by_step)
    matched = sorted(set(ref_med) & set(ours_med))
    matched = [s for s in matched if s >= CORRIDOR_MIN_STEP]
    if not matched:
        sys.exit("no matched log steps >= 100 between ours and the reference — stop")
    ratios = {s: ours_med[s] / ref_med[s] for s in matched}
    violations = {
        s: round(r, 3)
        for s, r in ratios.items()
        if not (CORRIDOR_LOW <= r <= CORRIDOR_HIGH)
    }
    return {
        "matched_steps": len(matched),
        "ratio_min": round(min(ratios.values()), 3),
        "ratio_max": round(max(ratios.values()), 3),
        "ratio_final": round(ratios[matched[-1]], 3),
        "violations": violations,
        "pass": not violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--trunk-checkpoint",
        default="allenai/MolmoAct2-SO100_101",
        help="init checkpoint carrying the (frozen) trunk + tokenizer",
    )
    parser.add_argument("--steps", type=int, nargs="+", default=[500, 1000, 1500, 2000])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-corridor", action="store_true")
    args = parser.parse_args()

    from bijou.gemma4.loading import resolve_checkpoint_dir
    from bijou.molmo2.model import load_model as load_trunk
    from bijou.molmo2.tokenizer import Molmo2TextTokenizer
    from bijou.molmoact2.predictor import resolve_image_token_ids

    sys.path.insert(0, str(REPO_ROOT / "fontaine/scripts"))
    from molmoact2_e2e_parity import frame_images, load_dataset

    banked = np.load(REPO_ROOT / f"reports/{BANKED_STEM}.npz")
    rows, truths, states = banked["rows"], banked["truths"], banked["states"]
    if args.limit is not None:
        keep = np.linspace(0, len(rows) - 1, num=args.limit).astype(int)
        rows, truths, states = rows[keep], truths[keep], states[keep]

    dataset = load_dataset()
    trunk_dir = resolve_checkpoint_dir(args.trunk_checkpoint)
    print(f"loading trunk from {trunk_dir} ...", flush=True)
    trunk = load_trunk(trunk_dir, device=args.device, dtype=torch.bfloat16)
    tokenizer = Molmo2TextTokenizer(str(trunk_dir))
    image_token_ids = resolve_image_token_ids(tokenizer)

    rungs: dict[int, dict[str, Any]] = {}
    for step in args.steps:
        step_dir = args.run_dir / f"step_{step:06d}"
        if not step_dir.exists():
            print(f"[step {step}] missing ({step_dir}) — skipped", flush=True)
            continue
        predictor = compose_predictor(
            step_dir,
            trunk,
            tokenizer,
            image_token_ids,
            device=args.device,
        )
        preds = np.full_like(truths, np.nan)
        started = time.monotonic()
        for i, idx in enumerate(rows):
            item = dataset[int(idx)]
            state = np.asarray(item["observation.state"], dtype=np.float32).reshape(-1)
            if not np.allclose(state, states[i], atol=1e-5):
                sys.exit(f"row {i} (concat {idx}): state drifted from the banked npz")
            action = item["action"].float().numpy()
            n = min(action.shape[0], HORIZON)
            if not np.allclose(action[:n], truths[i, :n], atol=1e-5, equal_nan=True):
                sys.exit(f"row {i} (concat {idx}): truth drifted from the banked npz")
            generator = torch.Generator(device=args.device)
            generator.manual_seed(BASE_SEED + int(idx))
            pred = predictor.predict_action(
                images=frame_images(item),
                task=str(item["task"]),
                state=torch.from_numpy(state),
                generator=generator,
            )
            preds[i] = pred[0].numpy()
            if (i + 1) % 80 == 0:
                rate = (i + 1) / max(time.monotonic() - started, 1e-6) * 60
                print(
                    f"[step {step}] {i + 1}/{len(rows)} ({rate:.1f} f/min)",
                    flush=True,
                )
        valid = np.isfinite(truths).all(-1)
        w = valid[:, :, None].repeat(truths.shape[-1], 2)
        mae = float(np.abs(preds - truths)[w].mean())
        # Matched state-copy: current state repeated across the window.
        copy = np.repeat(states[:, None, :], truths.shape[1], axis=1)
        state_copy_mae = float(np.abs(copy - truths)[w].mean())
        rungs[step] = {
            "mae": round(mae, 4),
            "reference_run_mae": REFERENCE_RUNGS.get(step),
            "state_copy_mae_matched": round(state_copy_mae, 4),
        }
        print(f"[step {step}] anchor MAE {mae:.4f}", flush=True)

    if not rungs:
        sys.exit("no rungs read — stop")
    ordered = [rungs[s]["mae"] for s in sorted(rungs)]
    final = ordered[-1]
    monotone_or_flat = all(b <= a * 1.02 for a, b in itertools.pairwise(ordered))
    report: dict[str, Any] = {
        "run_dir": str(args.run_dir),
        "trunk_checkpoint": str(trunk_dir),
        "n_frames": len(rows),
        "rungs": {str(s): rungs[s] for s in sorted(rungs)},
        "anchors": {"zero_shot": ANCHOR_ZERO_SHOT, "state_copy": ANCHOR_STATE_COPY},
        "final_mae": round(final, 4),
        "beats_zero_shot": bool(final < ANCHOR_ZERO_SHOT),
        "beats_state_copy": bool(final < ANCHOR_STATE_COPY),
        "monotone_or_flat": bool(monotone_or_flat),
    }
    if not args.skip_corridor:
        report["loss_corridor"] = corridor_read(args.run_dir)
    read_pass = (
        report["beats_zero_shot"]
        and report["beats_state_copy"]
        and report["monotone_or_flat"]
        and (args.skip_corridor or report["loss_corridor"]["pass"])
    )
    report["g4_pass"] = bool(read_pass)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(f"wrote {OUT_JSON}", flush=True)
    if not read_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
