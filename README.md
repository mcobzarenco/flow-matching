# Bijou

A vision-language-action model for SO-100/101 robot arms: a frozen,
truncated **Gemma-4 E2B-IT** encodes camera images and a language
instruction once per observation; a 404M-parameter **flow-matching action
expert** cross-attends the exported KV and denoises a 50-step action
chunk. Trained on the LeRobot community corpora (1000+ SO-100 datasets),
fine-tuned and deployed on a physical SO-101.

```
[instruction][cam_1]..[cam_k][instruction]      chat-templated user turn
        │  frozen truncated backbone: layers 0..14 only (E2B, bf16)
        ▼
  KV streams of the GLOBAL prefix layers {4, 9, 14}    PrefixKV, encoded
        │  cross-attention                             once per observation
        ▼
ActionExpert (fp32): 16 narrow layers, each =
  cross-attn(one scheduled stream) → self-attn([state][a_1..a_50]) → MLP
        ▼
velocity at flow time τ   →   Heun integration τ: 1 → 0
```

## Why this shape

- **The backbone's own KV sharing is the hook.** Gemma-4 E-series layers
  ≥15 (E2B) carry no K/V weights — the deep half of the network runs
  query-only against layer 13/14's KV. Bijou truncates exactly there
  (2.55B params instead of 5.2B, exported streams bitwise-identical to a
  full forward) and lets the expert be "more KV-shared layers".
- **Global-attention streams only** ({4, 9, 14}): uniform 512 head_dim
  geometry, p-RoPE trained for arbitrary range, never sliding-window
  truncated. The expert's cross-attention queries adopt the backbone's
  geometry exactly (q-RMSNorm, p-RoPE continuing after each sample's
  real prefix length, scaling 1.0); per-layer stream assignment is an
  explicit `cross_attention_schedule` (default blocks 4-4-8,
  deepest-heavy).
- **Instruction sandwich** prompt: under causal attention this gives
  instruction-conditioned image KV *and* image-conditioned instruction
  KV for a handful of extra tokens.
- **State enters the expert, not the VLM prompt** (π0 layout): the
  frozen backbone only ever sees in-distribution inputs, and slow visual
  context is decoupled from fast state.
- **Per-dataset MEAN_STD normalization** (π0/SmolVLA convention): most
  of the aggregate action variance across community rigs is calibration
  offset that images cannot see; normalizing it away per dataset is what
  makes learning possible. Checkpoints carry the full per-dataset stats
  table; inference normalizes with the deployment rig's stats.
- **Flow matching** with lerobot conventions (`x_τ = τε + (1−τ)a`,
  target `ε − a`, τ ~ Beta(1.5,1)): recipes and eval patterns port
  directly. Sampling defaults to Heun (2nd order; ~2× lower integration
  error than Euler at equal cost).

## Usage

Environment: `uv sync` (Python 3.13, pinned lock; system ffmpeg needed
for video decode).

```sh
# Train (single GPU or torchrun DDP; per-rank batch/workers).
uv run torchrun --standalone --nproc-per-node=4 -m bijou.train \
    --train-data ~/datasets/<collection_or_dataset_dirs>... \
    --holdout-episodes 0.1 --split-seed 0 \
    --steps 40000 --batch-size 64 --eval-samples 256 \
    --save-dir outputs/train/my_run

# Open-loop evaluation (state-copy baselines always included; the
# holdout flags reproduce the training split exactly).
uv run python -m bijou.eval \
    --data ~/datasets/... --episodes holdout --holdout-episodes 0.1 \
    --checkpoint outputs/train/my_run/step_040000 \
    --num-samples 256 --report report.html

# Closed-loop rollout on a physical SO-101 (see docs/rollout_so101.md).
uv run python -m bijou.rollout \
    --checkpoint outputs/train/my_run/step_040000 \
    --stats-repo-id <your_rig_dataset> \
    --camera front=/dev/video6 --camera wrist=/dev/video4 \
    --task "..." --max-relative-target 20 --check
```

Checks: `uv run python check.py` (ruff format + lint + pyright; ends
with an explicit `CHECKS PASSED`/`FAILED` verdict). Conventions:
`docs/code-styleguide.md`.

## Layout

| path | what |
|---|---|
| `bijou/gemma4/` | pure-torch Gemma-4 E-series (text+vision), bit-exact vs HF on greedy text+image generation (`verify_parity.py`); bench + tiny-checkpoint tooling |
| `bijou/expert.py` | the flow-matching action expert |
| `bijou/model.py` | BijouModel: prefix encode (stops at the deepest exported layer's K/V), Heun/Euler sampling |
| `bijou/data.py` | dataset selection guards, deterministic episode holdout, per-dataset stats, prompt collation — shared by train and eval |
| `bijou/loading.py` | model assembly + the checkpoint schema (write/read) |
| `bijou/train.py` | training CLI: DDP, warm start/resume, sharded MAE probes, wandb |
| `bijou/eval/` | eval CLI: seeded frame sampling, baselines, SmolVLA comparison, HTML reports |
| `bijou/rollout.py` | SO-101 closed-loop rollout CLI |
| `docs/` | operational handoff, results, runbooks, styleguide, plans |

## Docs

- `docs/handoff.md` — operational state: results ledger, machines, in-
  flight runs, pitfalls, decision queue. **Start here.**
- `docs/ablation_20k_results.md` — the 4-arm architecture ablation.
- `docs/rollout_so101.md` — physical rollout runbook (measured VRAM and
  latency for an 8 GiB laptop GPU).
- `docs/code-styleguide.md` — how code is written here.
- `docs/architecture.md` — deep reference for the model + training
  system, and the directions under evaluation (trunk unfreezing, adaRMS
  time conditioning, AR FAST-token co-training, and more).
- `docs/so101_recording_tutorial.md`, `docs/community_to_v3_pipeline_plan.md`
  — data recording and corpus conversion.

Checkpoints: [`mcobzarenco/bijou-checkpoints`](https://huggingface.co/mcobzarenco/bijou-checkpoints).
