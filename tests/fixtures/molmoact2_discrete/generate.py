"""Generate tests/fixtures/molmoact2_discrete/decode_anchors.npz —
the discrete-AR-head parity fixture (docs/molmoact2-retirement.md
phase 0 fixture (b), the phase-2 acceptance target).

Runs FONTAINE'S committed discrete-decode implementation
(bijou/molmoact2/predictor.predict_action_discrete on the fontaine
branch — record the commit in the provenance echo) over BOTH converted
checkpoints on deterministic rig frames, grammar-masked greedy (the
serving/RL mode the first-class MolmoAct2ARDecoder must reproduce) AND
the unconstrained reference mode, recording:

- the prompt input_ids (the packing the decode consumed),
- the full emission token_ids + extracted bins (both modes),
- per-bin-step chosen logprobs under the masked softmax at T=1
  (recomputed from the capture surface: log_softmax over the masked
  block at each step, gathered at the chosen id),
- the executed raw chunk [1, n_action_steps, action_dim] fp32,
- masked_violations per row.

Frame selection is deterministic: the first frame of the first
NUM_ROWS holdout-split episodes of so101_pick_place_v2 (split seed 0,
holdout 0.1 — the standard split machinery, so the rows are stable
identities, not concat-index accidents).
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from bijou.data import EpisodeSplit, select_datasets
from bijou.molmoact2.predictor import MolmoAct2Predictor

NUM_ROWS = 6
# RELEASE only: the discrete pathway exists on action_mode 'both'
# checkpoints; the rig-ft exports are 'continuous' (their fine-tune
# never trained the discrete head) and the decode refuses them by
# design — exactly the checkpoint scope of fontaine's GRPO line.
CHECKPOINTS = {
    "release": "allenai/MolmoAct2-SO100_101",
}
DATA = (
    Path.home() / "datasets/mcobzarenco/so101_pick_place_clean",
    Path.home() / "datasets/mcobzarenco/so101_pick_place_v2",
)
OUT = Path("tests/fixtures/molmoact2_discrete/decode_anchors.npz")

commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    capture_output=True,
    text=True,
    check=True,
).stdout.strip()
print(f"generator running at commit {commit}")

selection = select_datasets(
    DATA,
    (),
    30,
    episode_split=EpisodeSplit.HOLDOUT,
    holdout_fraction=0.1,
    split_seed=0,
)
dataset = selection.concat()
# First frame of each of the first NUM_ROWS holdout episodes: walk the
# concat once, keep frames whose frame_index == 0, stable order.
rows: list[dict] = []
seen: set[tuple[str, int]] = set()
for index in range(len(dataset)):
    item = dataset[index]
    key = (str(item["repo_id"]), int(item["episode_index"]))
    if int(item["frame_index"]) != 0 or key in seen:
        continue
    seen.add(key)
    rows.append(item)
    if len(rows) == NUM_ROWS:
        break
assert len(rows) == NUM_ROWS, f"only {len(rows)} first-frames in the holdout"
print(
    "rows:",
    sorted((r, e) for r, e in seen),
)

payload: dict[str, np.ndarray] = {
    "provenance_commit": np.array(commit),
    "episodes": np.array([int(r["episode_index"]) for r in rows]),
    "repos": np.array([str(r["repo_id"]) for r in rows]),
}

for name, checkpoint in CHECKPOINTS.items():
    predictor = MolmoAct2Predictor.load(
        checkpoint,
        "so100_so101_molmoact2",
        device="cuda",
        dtype=torch.bfloat16,
        # His codec loader takes a LOCAL dir (no hub resolution) — the
        # snapshot path of allenai/MolmoAct2-FAST-Tokenizer on this box.
        fast_tokenizer=str(
            Path.home()
            / ".cache/huggingface/hub/models--allenai--MolmoAct2-FAST-Tokenizer"
            / "snapshots/d45593b4c863d0bc1ca064f8b352fa16b75c38e8",
        ),
    )
    for mode, masked in (("masked", True), ("reference", False)):
        for row_idx, item in enumerate(rows):
            images = [
                item[key]
                for key in sorted(
                    k for k in item if k.startswith("observation.images.")
                )
            ]
            capture: list = []
            result = predictor.predict_action_discrete(
                images=images,
                task=str(item["task"]),
                state=item["observation.state"],
                grammar_masked=masked,
                on_undecodable="zeros" if not masked else "raise",
                action_capture=capture if masked else None,
            )
            key = f"{name}_{mode}_{row_idx}"
            payload[f"{key}_token_ids"] = result.token_ids.numpy()
            payload[f"{key}_bins"] = np.array(result.bins, dtype=np.int64)
            payload[f"{key}_actions"] = result.actions.float().numpy()
            if masked:
                payload[f"{key}_violations"] = np.array(result.masked_violations)
                logprobs = []
                base = predictor.action_token_start_id
                for step in capture:
                    # block_logits/allowed are BLOCK-relative [B, 2048];
                    # chosen carries BACKBONE ids — rebase before gather.
                    bins = step.chosen - base
                    assert bool((bins >= 0).all()) and bool(
                        (bins < step.block_logits.shape[-1]).all(),
                    )
                    logits = step.block_logits.float()
                    logits = logits.masked_fill(~step.allowed, float("-inf"))
                    logprobs.append(
                        logits.log_softmax(-1).gather(-1, bins[..., None]).item(),
                    )
                payload[f"{key}_logprobs"] = np.array(logprobs, dtype=np.float64)
    del predictor
    torch.cuda.empty_cache()
    print(f"{name}: done")

OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT, **payload)
sizes = {k: v.shape for k, v in payload.items() if hasattr(v, "shape")}
print(json.dumps({k: str(v) for k, v in list(sizes.items())[:8]}, indent=2))
print(f"written {OUT} ({OUT.stat().st_size} bytes)")
