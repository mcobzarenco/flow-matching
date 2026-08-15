"""The 240-anchor rig read (docs/molmoact2-retirement.md): score any
molmoact2-family checkpoint on the reference rung's banked anchor rows
(the instrument behind gate-d-lite's verdict, 2026-08-14: step-500
5.556 / step-2000 2.030 vs reference 6.76 / 3.23; promoted from the
gitignored box copy so it cannot rot invisibly). Usage:

    uv run python probes/probe_molmoact2_anchor_read.py <ckpt_dir> ...

Box-resident inputs (fontaine's bank + the rig datasets); phase-3
smokes reuse it for the ar/joint objective reads.

Row identity: the bank's ``rows`` are frame indices into the
concatenated rig corpus (clean + v2, chunk 30, ALL split — verified
here by comparing each row's observation.state against the bank's
``states`` BEFORE any prediction is trusted; a mismatch aborts).
Decode: euler-10 (the recorded serving point), bf16 trunk, fp32
expert, stable keyed noise (BijouPolicy defaults).

Thresholds (pre-registered in the launcher header):
  step_000500: MAE within 6.76 +- 1.0, AND beats state-copy 9.08 and
               zero-shot 28.95.
  step_002000: MAE within 3.23 +- 1.0 (the free full-endpoint read).
"""

import sys
from pathlib import Path

import numpy as np
import torch

from bijou.data import EpisodeSplit, select_datasets
from bijou.eval.policies import BijouPolicy
from bijou.modelling.interface import SamplingMethod

BANK = Path.home() / "flow-matching/reports/analysis__molmoact2_rig_ft_step2000.npz"
DATA = (
    Path.home() / "datasets/mcobzarenco/so101_pick_place_clean",
    Path.home() / "datasets/mcobzarenco/so101_pick_place_v2",
)
BATCH = 16

bank = np.load(BANK)
rows = bank["rows"].tolist()
truths = torch.from_numpy(bank["truths"]).float()
bank_states = torch.from_numpy(bank["states"]).float()

selection = select_datasets(DATA, (), 30, episode_split=EpisodeSplit.ALL)
dataset = selection.concat()
print(f"corpus: {len(dataset)} frames; bank rows {len(rows)} (max {max(rows)})")

items = [dataset[index] for index in rows]
ours_states = torch.stack([item["observation.state"].float() for item in items])
state_delta = float((ours_states - bank_states).abs().max())
print(f"row-identity check: max |state delta| = {state_delta:.2e}")
assert state_delta < 1e-4, "bank row mapping does NOT reproduce — aborting"

copy_mae = float((bank_states[:, None, :] - truths).abs().mean())
print(f"state-copy MAE (recomputed): {copy_mae:.3f}")

for step_dir in sys.argv[1:]:
    policy = BijouPolicy(
        Path(step_dir),
        device=torch.device("cuda"),
        seed=0,
        sample_steps=10,
        method=SamplingMethod.EULER,
        flow_decoder_dtype=torch.float32,
    )
    preds: list[torch.Tensor] = []
    for start in range(0, len(items), BATCH):
        chunk_items = items[start : start + BATCH]
        chunk_rows = rows[start : start + BATCH]
        preds.extend(policy.predict(chunk_items, chunk_rows))
    stacked = torch.stack([p.float() for p in preds])
    mae = float((stacked - truths).abs().mean())
    print(f"{step_dir}: 240-anchor MAE = {mae:.3f}")
    del policy
    torch.cuda.empty_cache()
