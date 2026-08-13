"""Discrete (AR) pathway smoke on the REAL release checkpoint —
molmoact2-ar-head-port item (b)/(c) GPU half, first read.

Runs ``predict_action_discrete`` (both modes) on rig anchor rows with
the released ``allenai/MolmoAct2-SO100_101`` + FAST artifact and
reports, per row: emission length + well-formedness (action_start /
action_end / EOS all present), bin count, decodability, masked-mode
violations, masked-vs-unconstrained stream agreement, and the
discrete-vs-continuous action gap (record-only curiosity: the banked
flow-pathway preds on the same frames). This is the cheap "does the
wiring hold on the real model" gate before the formal token-for-token
parity extension (which needs THEIR HF reference executing live —
there are no banked discrete anchors).

Usage:
    uv run python fontaine/scripts/molmoact2_discrete_smoke.py \
        [--rows 4] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from fontaine.scripts.molmoact2_e2e_parity import (
    NORM_TAG,
    frame_images,
    load_dataset,
)

CHECKPOINT = "allenai/MolmoAct2-SO100_101"
FAST_ARTIFACT = "allenai/MolmoAct2-FAST-Tokenizer"
BANKED = REPO_ROOT / "reports/analysis__molmoact2_rig_preflight.npz"
OUT_JSON = REPO_ROOT / "reports/analysis__molmoact2_discrete_smoke.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    from bijou.molmoact2 import MolmoAct2Predictor

    fast_dir = snapshot_download(FAST_ARTIFACT)
    dataset = load_dataset()
    banked = np.load(BANKED)
    rows = banked["rows"]
    flow_preds = banked["preds"]
    keep = np.linspace(0, len(rows) - 1, num=args.rows).astype(int)

    print(f"loading {CHECKPOINT} (+ FAST artifact) ...", flush=True)
    predictor = MolmoAct2Predictor.load(
        CHECKPOINT,
        NORM_TAG,
        device=args.device,
        dtype=torch.bfloat16,
        fast_tokenizer=fast_dir,
    )
    assert predictor.fast_codec is not None
    report: dict = {
        "checkpoint": CHECKPOINT,
        "bpe_vocab": predictor.fast_codec.bpe_vocab,
        "action_token_start_id": predictor.action_token_start_id,
        "rows": [],
    }
    for i in keep:
        idx = int(rows[i])
        item = dataset[idx]
        state = torch.asarray(
            np.asarray(item["observation.state"], dtype=np.float32).reshape(-1),
        )
        images = frame_images(item)
        started = time.monotonic()
        try:
            free = predictor.predict_action_discrete(
                images=images,
                task=str(item["task"]),
                state=state,
                on_undecodable="zeros",
            )
            free_error = None
        except RuntimeError as error:  # no-EOS cap — the reference class
            free = None
            free_error = str(error)
        free_seconds = time.monotonic() - started
        started = time.monotonic()
        masked = predictor.predict_action_discrete(
            images=images,
            task=str(item["task"]),
            state=state,
            grammar_masked=True,
        )
        masked_seconds = time.monotonic() - started
        entry: dict = {
            "concat_index": idx,
            "masked": {
                "bins": len(masked.bins),
                "violations": masked.masked_violations,
                "seconds": round(masked_seconds, 1),
                "action_min": round(float(masked.actions.min()), 3),
                "action_max": round(float(masked.actions.max()), 3),
                "mae_vs_banked_flow_pred": round(
                    float(
                        np.abs(
                            masked.actions[0].numpy()
                            - flow_preds[i][: masked.actions.shape[1]],
                        ).mean(),
                    ),
                    4,
                ),
            },
        }
        if free is None:
            entry["unconstrained"] = {"error": free_error}
        else:
            ids = free.token_ids[0].tolist()
            decodable = (
                int(
                    predictor.fast_codec.symbol_lengths[free.bins].sum(),
                )
                == 180
                if free.bins
                else False
            )
            entry["unconstrained"] = {
                "emission_len": len(ids),
                "has_action_start": predictor.action_start_token_id in ids,
                "has_action_end": predictor.action_end_token_id in ids,
                "hit_eos": ids[-1] == predictor.eos_token_id,
                "bins": len(free.bins),
                "decodable": decodable,
                "seconds": round(free_seconds, 1),
                "same_bins_as_masked": free.bins == masked.bins,
            }
        report["rows"].append(entry)
        print(json.dumps(entry), flush=True)

    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
