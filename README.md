# Bijou

Vision-language-action (VLA) models for SO-100/101 robot arms, built
by reusing one pretrained multimodal transformer in every role it can
serve. Two trunk backbones are supported:

- **Molmo2-4B** (SigLIP-so400m + Qwen3-4B decoder, 4.85B) — the
  current training trunk, adopted after a matched-topology screen: the
  full VLM fine-tunes decoder-only on chat-templated observations
  (auxiliary text fields + the action chunk as FAST tokens), beating
  the equivalent Gemma composition decisively at 2.5× fewer steps with
  ~3× cheaper decode. A **flow-matching action expert** attached over
  the *frozen* fine-tuned trunk (reading its residual streams through
  learned adapters) is the trained continuous-action head on this
  trunk.
- **Gemma-4 E2B-IT** — the original trunk and most of the measured
  history: a frozen truncated prefix (E-series KV sharing makes the
  deep half query-only, so a 15-layer prefix exports
  bitwise-identical K/V at half the parameters) feeds the expert via
  cross-attention, or the full 35-layer stack fine-tunes
  decoder-only. The best banked checkpoint overall is still on this
  side: the flow expert trained against the frozen decoder-only
  Gemma trunk, decoded as a 10-draw ensemble.

```
{task}[{kind} camera|<imgs>]..[cond|v][generate|fields actions]{task}⟨state⟩
        │  one chat-templated user turn (per-camera semantic kinds,
        │  [key|value] conditioning, request set, soft state token)
        ▼
  trunk prefill (bf16), encoded once per observation
        │
        ├─ decoder-only: the trunk continues its own prefill as a
        │    model turn — requested auxiliary VALUE lines, then [BOA]
        │    and the action chunk as FAST tokens (full-vocab CE;
        │    fresh untied rows for the FAST block, trunk head frozen)
        │
        └─ flow expert: separate fp32 decoder (trained instances
             ~404M) cross-attends exported K/V streams (or residual-
             tap adapters over a frozen trunk) and denoises the
             50-step chunk (velocity at τ, Heun integration, τ: 1 → 0)
```

## Repository layout: shared codebase vs. research agent

**`fontaine/` is not part of the model codebase.** It is the working
directory of *Fontaine*, an autonomous research agent that develops
this repository on the `fontaine` branch: its operating charter,
work queue, monitoring harness, launch/analysis scripts, and a
research blog (pre-registrations, run reports, paper notes —
rendered at
[mcobzarenco-fontaine-blog.static.hf.space](https://mcobzarenco-fontaine-blog.static.hf.space)).
Read it as lab notebooks, not product code.

Everything else — `bijou/`, `tests/`, `docs/`, `check.py` — is the
shared codebase that humans and local agents develop together. If you
are here to understand or extend the model, start with `docs/` and
`bijou/`; nothing in `bijou/` imports from `fontaine/`.

## Why this shape

- **Reuse the trunk's own machinery.** Decoder-only fine-tuning keeps
  the pretrained VLM intact and adds only the FAST block's
  embedding/head rows (~11M new params on Gemma; fresh untied rows on
  Molmo2, whose shipped embeddings/LM head stay frozen). The flow
  expert adopts the trunk's attention geometry exactly (q-RMSNorm,
  p-RoPE continuation, grouped-query K/V) so exported streams need no
  re-projection.
- **What speaks is commanded by the prompt.** The `[generate|…]`
  request set makes auxiliary text (subgoal / holding / progress /
  event / visibility — weak labels from an LLM judging pipeline)
  conditional on being asked: the model learns p(value | observation,
  asked), and deployment requests exactly `[generate|actions]`.
- **Instruction sandwich** prompt: under causal attention this gives
  instruction-conditioned image KV *and* image-conditioned instruction
  KV for a handful of extra tokens.
- **State enters through a soft token / the expert, not VLM text**
  (π0 layout): the trunk only ever sees in-distribution inputs, and
  slow visual context is decoupled from fast proprioception.
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
# Train (single GPU or torchrun DDP; per-rank batch/workers). Large
# runs add --zero1 (sharded optimizer state), --backward-chunks N
# (gradient accumulation inside a fixed effective batch), and
# --optimizer adamc (AdamW with LR-schedule-corrected weight decay,
# arXiv 2506.02285). Checkpoint saves are asynchronous by default.
uv run torchrun --standalone --nproc-per-node=4 -m bijou.train \
    --train-data ~/datasets/<collection_or_dataset_dirs>... \
    --holdout-episodes 0.1 --split-seed 0 \
    --steps 40000 --batch-size 8 --eval-samples 256 \
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

Checks: `uv run python check.py` (ruff format + lint + pyright +
pytest; ends with an explicit `CHECKS PASSED`/`FAILED` verdict).
Conventions: `docs/code-styleguide.md`.

## Layout

| path | what |
|---|---|
| `bijou/gemma4/` | pure-torch Gemma-4 E-series (text+vision), bit-exact vs HF on greedy text+image generation (`verify_parity.py`); bench + tiny-checkpoint tooling |
| `bijou/molmo2/` | pure-torch Molmo2-4B (SigLIP tower + Qwen3 decoder), same parity discipline; tiny-checkpoint tooling |
| `bijou/interface.py` | the encoder×decoder seam: ObservationMemory (streams and/or prefix KV cache), collation, the decoder ABCs |
| `bijou/encoders/` | per-trunk observation encoding strategies (prompt build, prefill, stream export, unfreeze partition) |
| `bijou/decoders/` | FlowDecoder (flow matching, cross-attention or residual-tap conditioning), ARFastDecoder, ARBackboneDecoder (Gemma decoder-only), Molmo2ARDecoder |
| `bijou/fast/` | owned FAST action tokenizer (DCT + BPE) + corpus-fit CLI |
| `bijou/aux_text.py` | auxiliary text fields: templating, masking, decode |
| `bijou/model.py` | BijouModel composition root: encode once, decode chunks |
| `bijou/data.py` | dataset selection guards, deterministic episode holdout, per-dataset stats — shared by train and eval |
| `bijou/loading.py` | model assembly + the checkpoint schema (write/read) |
| `bijou/train.py` | training CLI: DDP, ZeRO-1, chunked backward, activation checkpointing, component LRs, AdamW/AdamC, warm start/resume, sharded MAE probes, wandb |
| `bijou/async_save.py` | asynchronous checkpoint serialization (background gather/write, byte-identical to sync) |
| `bijou/eval/` | eval CLI: seeded frame sampling, baselines, sampled multi-draw decoding, HTML reports |
| `bijou/judge/` | LLM judging pipeline (episode verdicts, camera kinds, instruction rewrites, frame labels) |
| `bijou/rollout.py` | SO-101 closed-loop rollout CLI (+ `rollout_async.py`) |
| `docs/` | architecture + results, runbooks, styleguides |
| `fontaine/` | the autonomous research agent (see above) |

## Docs

- `docs/architecture.md` — deep reference: model + training system, the
  results ledger that shaped them, and directions under evaluation.
  **Start here.**
- `docs/molmo2.md` / `docs/gemma4.md` — trunk fact sheets (fetched
  primary-source configs; both models post-date common training
  cutoffs, so these beat model memory).
- `docs/code-styleguide.md` — how code is written here.
- `docs/working-together.md` — how work is done here (operating
  conventions, run operations, measurement discipline).
- `docs/init_gpu_machine.md` — fresh GPU box → first training run.
- `docs/rollout_so101.md` — physical rollout runbook (measured VRAM and
  latency for an 8 GiB laptop GPU).
- `docs/ablation_20k_results.md` — the 4-arm architecture ablation
  (historical record).
- `docs/so101_recording_tutorial.md`, `docs/community_to_v3_pipeline_plan.md`
  — data recording and corpus conversion.

Transient state (in-flight runs, machines, queue) lives in wandb,
the HF hub, `reports/`, and the agent's own `fontaine/` state — not
in docs.

Checkpoints: [`mcobzarenco/bijou-checkpoints`](https://huggingface.co/mcobzarenco/bijou-checkpoints)
(shared lineage) and
[`mcobzarenco/fontaine-checkpoints`](https://huggingface.co/mcobzarenco/fontaine-checkpoints)
(agent-trained runs).
