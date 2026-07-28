# Bijou handoff — current state, full context (2026-07-27)

Read this first. Architecture and its design rationale live in
`README.md`; per-module contracts live in code docstrings. This doc is
the operational truth:
what exists, what it scored, what is running, what is queued, and how the
owner likes to work. The owner's chat thread now lives in the main
checkout `/home/marius/w/flow-matching`; everything is committed to
`main` on `github.com:mcobzarenco/flow-matching`, so `git pull` there
first.

## 1. What Bijou is

VLA for SO-100/101 arms: frozen, truncated **Gemma-4 E2B-IT** prefix
encoder (layers 0–14, bf16) exports K/V of its global-attention layers
{4, 9, 14}; a 404M fp32 flow-matching **action expert**
(1024h/8heads/4096ff/8 cross-heads, cross-attention schedule 4-4-8 over
the three streams, causal_actions self-attention, chunk 50) denoises
50-action chunks against them. Per-dataset MEAN_STD normalization (π0
convention; between-rig calibration offsets are why aggregate stats
failed). Prompt = instruction sandwich `[task][cam_1..N][task]` in one
user turn, right-padded batches, 140 soft tokens/camera budget (a cap:
640×480 pools to 130/cam; prompt ≈ 292 tokens for 2 cameras).
Flow convention: `x_τ = τ·ε + (1−τ)·a`, target `u = ε − a`,
τ ~ Beta(1.5,1)→(0.001,1]; sampling integrates τ 1→0, Heun default.
Ultimate goal: beat baselines on the owner's physical SO-101.

Package layout (strict import DAG, enforced by review):
`train`/`eval`/`rollout` → `loading` → `data` → `model` → `expert` →
`gemma4`. gemma4 is a pure-torch reimplementation, bit-exact vs HF on
greedy text+image generation (`bijou/gemma4/verify_parity.py`). Design
decisions and their alternatives (stream choice, schedule, state
placement, prompt shape) are summarized in `README.md`; historical
context beyond that lives in the git log.

## 2. Results ledger (all open-loop chunk MAE, raw degrees, Heun-10, 256 frames, seed 0)

Baselines on the standard frame sets: state-copy = 11.10 (train side) /
10.30 (held-out episodes) / 9.54 (owner rig). state-copy-norm ≈ same.

**Pretrain lineage (mainline)**: v2 20k (1×H100) → v1v2 20k DDP4 →
v1v2v3 40k DDP4 = `bijou_community_v1v2v3_20k_ddp4/step_040000`, ~15.4M
cumulative samples, loss ~0.099. Scores 6.93 on the held-out-episode
frames — **contaminated** (trained on all episodes, no holdout existed
yet) — and 12.76 on the owner rig vs copy 9.54 (loses cross-rig).

**cont45k continuation** (2026-07-27,
`bijou_community_v1v2v3_cont45k_ddp4/step_045000`: init-from 40k +
fresh optimizer + warmup 500, 45k steps × global 256 ≈ +11.5M samples,
~27M cumulative, holdout 0.1/seed 0 ACTIVE, final loss ~0.089):
holdout **6.851** (vs 6.93 — equally contaminated frames, flat within
noise jitter), train side 6.741, owner rig **11.709** (best pretrain
rig score: mainline-40k 12.76, best ablation arm 13.15; copy 9.54
still ahead). Probe curve: warm-restart transient to ~8.9 by 1.5k,
recovered by ~37k, bottomed 6.80 @40k, 7.05 final — episode-level
generalization saturated; the rig side is where the samples went.
Reports `reports/report_mainline_cont45k_{holdout,marius}.html` +
JSONs in `reports/`. On HF with optimizer.pt.

**4-arm ablation** (from scratch, holdout 0.1/seed 0, seed 42, batch 64,
1×H100 each; 20k then lossless resume→40k; doc:
`docs/ablation_20k_results.md`, now 40k-centric):

| arm | holdout 20k→40k | rig 40k |
|---|---|---|
| control (causal, 140, 4-4-8) | 10.52 → **9.60 (Δ−0.56)** | 13.31 |
| bidir | 10.82 → **9.59** | 17.08 (still terrible) |
| streams0016 (0 0 16) | 10.86 → 9.95 | **13.15** (best rig, −1.01) |
| tokens280 (1.9×/step) | 10.39 → 9.75 | 14.84 (got worse) |

Conclusions that stuck: baseline honestly beaten at 40k by all arms;
scale dominates architecture; keep causal (bidir's lower loss is an
easier-objective artifact and it fails cross-rig); tokens280's 20k
holdout edge reversed; streams0016's rig edge is a hint for a future
re-test. Episode-level generalization ≈ free; cross-rig is the wall.

**Fine-tunes** (owner rig, warm start from mainline 40k):
- `bijou_ft_marius_2k` (7-episode `so101_pick_place_clean`, 38 epochs,
  lr 5e-5): eval 1.66 but contaminated (no holdout). First physical
  rollout (step_001000): task structure present (approach/grasp/
  transport/place/retreat, end-state recognition) but weak visual
  grounding, wobbly.
- `bijou_ft_marius_v2_2k` (clean 7 eps + new 50-ep
  `so101_pick_place_v2` = 57 eps/36,078 frames, DDP3, global 192, 2000
  steps ≈ 10.6 epochs, final loss 0.0298): **not yet rolled out
  physically**. No holdout (owner chose same-params); checkpoints every
  250.
- **Init comparison pair** (2026-07-27, `bijou_ft_marius_4k_init45k`
  vs `bijou_ft_marius_4k_init40k`): identical recipes — both rig
  datasets, holdout 0.1/split-seed 0 (3,403 held-out frames: clean ep
  {1} + v2 eps {10,14,16,25,29}), 4k steps DDP2 global 128, lr 5e-5,
  seed 7 — except --init-from (cont45k 45k vs mainline 40k). Offline
  eval on the FULL 3,403-frame holdout (honest; state-copy 11.39 on
  this set), step_004000: **init45k 10.376 vs init40k 10.595**
  (−0.22; p90 17.9 vs 19.0; win-vs-copy 0.452 vs 0.435). Both beat
  copy (−0.92/−0.72) — first rig-side copy wins. Beat copy on mean
  and p90 but lose p50: better on motion frames, slightly worse on
  idle ones. Probe curves flat 2k→4k while train MAE fell to ~2.7
  (memorization without holdout damage). Extra ~11.5M pretrain samples
  → ~2% better downstream MAE + fatter-tail wins at matched budget.
  JSONs/reports in `reports/` (`eval_ft4k_init*_holdout_full.json`,
  `report_ft4k_init*_holdout.html`); checkpoints every 500 on the box.
  **Heun-30 re-score** of init45k @4000 on the same 3,403 frames
  (`*_heun30.*` in `reports/`): 10.376 → **10.097** (−0.28), first_mae
  3.07 → 2.53 (beats copy's 2.63), win-vs-copy 0.477, only +39%
  s/frame (0.057→0.079 — prefix encode dominates) — confirms the
  sampling-analysis finding that fine-tune fields want more integration
  steps; rollout latency budget permitting, try `--sample-steps 30`.

**Camera-swap scare (resolved)**: owner suspected `clean`'s front/wrist
keys were swapped. Frame-level inspection of ALL episodes of both rig
datasets (montages in `outputs/camcheck/` on the old worktree): both
datasets are correctly labeled and mutually consistent. Suspect instead
the ROLLOUT device mapping (`front=/dev/video6 wrist=/dev/video4`) —
device indices move between replugs; verify by covering a lens before
the next rollout.

**Sampling analyses** (v2 fine-tune step_001250, 6 motion frames, report
`reports/sampling_analysis.html` in the main checkout):
- Noise draws (Heun-5): single-draw MAE 5.30°, mean-of-10 2.88°,
  mean-of-100 2.44°; across-draw std ~5.9° > single-draw error; spread
  fans out along the 50-step horizon.
- Heun steps (fixed noise): 5→10→20→40 steps = MAE 5.34→4.13→3.80→3.72;
  integration error at Heun-5 (3.3° vs Heun-50) ≈ model error floor.
  **Fine-tune-specific**: pretrain fields are smoother (old bench:
  Heun-5 fine there). Coarse integration ≈ extra per-draw noise; the
  ensemble mean is step-count-invariant (2.44→2.33).
- Practical: rollout should try `--sample-steps 10`; a `--sample-draws`
  (mean-of-N) rollout option is designed-not-built (draws batch through
  the expert; prefix encodes once; unimodality caveat — inspect charts).

**Re-anchoring probe (2026-07-27, falsifier for delta-actions — no
training; script `outputs/probe_reanchor.py` on box+laptop, JSONs in
`reports/probe_reanchor_*.json`)**: cont45k@45k zero-shot rig
predictions, post-hoc corrected. On clean256 / rig-holdout3403:
raw 11.71/16.35; **anchor-rigid (shift chunk to start at state — the
test-time delta-action equivalent) buys only −0.25/−0.35** — nowhere
near copy (9.54/11.39) → the wall is NOT absolute anchoring; delta
parameterization deprioritized. **Per-frame oracle constant offset:
6.51/9.14** (beats copy, p90 halves) — the predicted chunk SHAPE is
right, the level is wrong. **One offset per dataset recovers nothing**
(11.89/16.37, fitted offsets ±1–6°, zero-mean across frames) → the
level error is frame/scene-dependent, not rig calibration: the model
mis-localizes the mid-chunk working point visually on an unseen rig.
Conclusion: cross-rig gap ≈ spatial grounding, not action-space
parameterization. Redirects budget to backbone adaptation (LoRA arm)
and inference-time ensembling (level uncertainty averages out over
draws — across-draw std 5.9° fits this picture).

**Per-τ / step-count diagnostic (2026-07-27, Phase 0 of the adaLN
investigation; `outputs/probe_tau_diagnostic.py`, JSON in `reports/`)**:
{cont45k, ft-init45k} × {community-holdout 256, rig-holdout 256},
fixed τ grid × 4 draws + fixed-noise Heun {5,10,20,30}. Findings:
(1) **integration gap is real in-domain**: Heun-5→30 = −0.99 MAE
(7.57→6.57, 13%) for the pretrain on community, −1.13 (11.03→9.90,
10%) for the ft on rig — the “pretrain fields are smoother” belief is
wrong at 256-frame scale; ~1 MAE of recoverable integration error
exists on both. (2) **OOD cost lives at HIGH τ**: zero-shot rig
implied-action error at τ=0.9 is 12.6° vs 4.8° in-dist, while its
Heun gap is only 0.4 — initial chunk placement from context, not
refinement: independent confirmation of the grounding story from the
re-anchor probe. (3) **fine-tuning roughens mid-τ**: ft model shows a
vel-MSE bump at τ≈0.2–0.5 on BOTH sides (0.44–0.60 vs pretrain's
0.07 valley) — the 4k-step ft damages mid-τ transport generally.
Also first measured: **ft forgetting on community holdout = +0.85**
(7.71 vs 6.86). Note τ→0 vel-MSE → ~0.8–0.9 is irreducible
(ε unidentifiable), not weakness — read the implied-action row.
Implication for adaLN (Phase 1/2): realistic prize ≈ 0.5–1.0 MAE
in-domain + Heun-5-quality rollouts (latency), NOT the ~5-point OOD
gap. Fidelity: Heun-10 numbers match the eval ledger within 0.01.

**Vision spatial-acuity probe (2026-07-28, laptop;
`outputs/probe_vision_acuity.py`, report `reports/vision_acuity.html`
+ .json, caches in `outputs/acuity/`)**: real boat crop pasted at
controlled positions on 7 rig backgrounds; sensitivity curves +
linear (x,y) readout at every stage of the frozen pipeline. Geometry
confirmed: 640×480 → 624×480 → 39×30 patches (16 px) → 3×3 pool →
130 soft tokens (48×48 px cells). Findings: (1) **the pool is
exonerated** — soft tokens are the BEST linear-position stage (8.4 px
held-out-positions RMSE) beating pre-pool (11.3): averaging denoises;
crop/asymmetric-budget interventions lose their premise. (2) **LM
depth discards position**: soft 8.4 → K4 10.8 → K9 15.4 → K14
17.3 px; K4 is the sharpest stream the expert sees (tension with
streams0016's deepest-heavy rig hint — its edge isn't positional
precision). (3) **Split verdict vs the 24 px pre-registered bar**:
familiar scenes pass (11–17 px ≈ 1–1.5 cm), cross-background misses
(25–32 px ≈ 2–3 cm) — position is present but with thin margin;
expert under-exploitation remains implicated (adaLN/capacity track
live; cross-scene degradation mildly supports LoRA). (4) **Not
object-centric**: task-object motion moves K14 only ~1.9× more than
equal motion of an irrelevant background patch; high nuisance floors
at K/V. Caveats: linear readout = lower bound; pasted sharp edges
read optimistically; front cam, one object.

**E2B-IT prompt/vision validation** (full untruncated model, greedy):
our exact collator prompts elicit coherent instruct behavior; a
describe-the-scene variant grounds correctly (wooden table, arm motors/
cables from wrist cam). Perception is fine; grounding weakness lives in
expert scale/training. (Miscounts "three camera views" for two — small
model, not a wiring bug; soft-token counts are hard-guarded.)

**Training step split** (H100, batch 64, real data, probe): encode_prefix
79.3% / expert fwd 4.6% / bwd 15.4% / optimizer 0.7%, total ~0.91s. So
the frozen backbone forward dominates; expert-bf16-autocast idea is
dead; next perf wins are length-bucketed batching (batch pads 452 vs
~292 typical) and torch.compile of the backbone.

## 3. In flight right now

- **RUNNING: `community-v1v2v3-unftext15k-r2-ddp8`** (launched 19:03
  UTC 2026-07-28, tmux `mainline` on the 8×A100 box, log
  `~/unftext15k_r2_console.log`, launcher
  `~/launch_unfreeze_text15k.sh`): unfrozen-TEXT-trunk continuation r2
  — init-from cont45k step_045000, **--expert-lr 2e-5 --text-lr 2e-5**
  (single LR, π0-style), grad-clip 10.0, warmup 500, **15k steps**
  (owner target ~14h; measured early pace ~2.0 s/step → likely ~9h),
  batch 32×8 = global 256, holdout 0.1/seed 0, probes 256 (cont45k
  eval frames; reference 6.85), save @2500, seed 13, NO --fps filter.
  Input pipeline fixed and VERIFIED (workers 16, prefetch 8, decoder
  cache 8): sustained 8×100% GPU vs r1's constant per-rank 0% stalls;
  host load 35 vs 63.
  **History**: r0 aborted @~170 (grad-clip 1.0 renormalized every step
  ~4×; → 10.0). r1 aborted @~1.8k (expert peak 1e-4 re-heated the warm
  expert: loss 0.104→0.113 rising, probes 8.1–8.6 vs 6.85 — cont45k
  needed ~35k steps to anneal the same transient; unrecoverable in a
  short cosine). Watch item for r2: if train_mae recovers but
  eval_chunk_mae lags, the 5×-slower expert is tracking trunk feature
  drift too loosely.
- **2×H100 box `ssh ubuntu@192.222.54.70`** (owner-provisioned
  2026-07-28): FAST tokenizer fit done (below). Has all 3 community
  collections + both rig sets under `~/datasets/mcobzarenco/` (rig too
  — owner re-downloaded there; the `marius/` copies were removed).
  flow-matching + lerobot-dataset-tools cloned.
- **DONE 2026-07-28 — quantile backfill + FAST tokenizer v1**:
  ldtools.backfill_quantile_stats (exact corpus q01/q10/q50/q90/q99,
  correcting LeRobot's mean-of-episode-quantiles bug) run on all 3
  community collections (v1 129, v2 324, v3 790 datasets; 1 bespoke
  drop) + both rig sets, **re-uploaded to the hub** (community + rig
  dataset repos; local rig copies refreshed). FAST tokenizer
  **`mcobzarenco/bijou-checkpoints/fast_tokenizer_v1`** on HF: fit on
  1040 datasets / 4.9M chunks (~32 min BPE), vocab 1024 (alphabet
  1019), scale 10. Fidelity (fit_report.json in the artifact): p50 52
  tok/chunk, 5.7× compression, recon MAE p50 0.44° / p90 0.58° raw
  (≪ model error), rig 55–57 tok / 0.44–0.68°. Constant-dim guard +
  normalized clip landed during the fit (parked joints / padded dims
  had floored-span division → 7.3e9-symbol alphabet; now map to 0).
- Useful eval pattern to recreate (the old `eval_reports.sh`): 3 sides
  = community holdout/train (`--episodes X --holdout-episodes 0.1
  --split-seed 0`, 256 frames, seed 0) + rig
  (`so101_pick_place_clean`, 256). Full-holdout rig eval:
  `--data .../so101_pick_place_clean .../so101_pick_place_v2
  --episodes holdout --holdout-episodes 0.1 --split-seed 0
  --num-samples 3403`. Ft-pair launchers survive in local gitignored
  `outputs/launch_ft_compare_init{45,40}k.sh` (note distinct rendezvous
  ports — two concurrent torchruns cannot share --standalone's
  default).

## 4. Machines, data, services

- **New machine bring-up**: `docs/init_gpu_machine.md` (init-vm-gpu.sh
  → auth → dataset downloads → from-scratch smoke train run).
- **Box**: `ssh ubuntu@150.136.37.72` (use `-A` for git operations;
  Lambda, provisioned 2026-07-28 via init-vm-gpu.sh). 8×A100-SXM4-80GB,
  240 vCPU, 1.77T RAM, 19T disk. Repo `~/flow-matching`; HF token in
  place (gated backbone verified); wandb login PENDING (owner does
  `wandb login`; launchers read ~/.netrc and export WANDB_API_KEY —
  wandb.init rejects netrc-only). Data:
  `~/datasets/mcobzarenco/community_dataset_v{1,2,3}_v3` complete
  (120/121/687 GB). Rig datasets deliberately NOT transferred (owner:
  not needed yet). cont45k step_045000 (config+expert, no optimizer)
  at `outputs/train/.../step_045000`. Training env:
  `MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072` (+
  `LEROBOT_VIDEO_DECODER_CACHE_SIZE=4` set by train.py) +
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for live-trunk
  runs (69–79 GB/rank measured). The old 4×H100 (68.209.73.71) was
  deleted 2026-07-28; its wandb-key-in-history rotation is still owed.
  **Artifact inventory after deletion** — on HF: mainline lineage
  (v1v2 10k; v1v2v3 15k/40k; cont45k 45k WITH optimizer),
  ft_marius_2k (8 steps), ft_4k_init45k step_004000,
  v2_20k_pdnorm_r2 step_002500. On the laptop: ft_marius_2k,
  ft_marius_v2_2k (never on HF), ft_4k_init45k step_004000, all eval/
  probe JSONs + HTML in `reports/`. LOST with the box: ft_4k_init40k
  checkpoints (scores survive in `reports/`; re-creatable from the HF
  40k mainline + rig data in ~1.2h), the 8 ablation-arm checkpoints
  (abl_*_{20,40}k), cont45k intermediates (5k–40k), community-dataset
  local copies (re-downloadable from HF hub).
- **Laptop**: RTX 3000 Ada, 8 GiB (bf16 backbone 4.77 GiB + bf16 expert
  fits; peak 6.29 GiB; ~233 ms/replan warm, ~2s CUDA warmup first
  replan). Main checkout `/home/marius/w/flow-matching` (owner's chat
  home now; has ft checkpoints under `outputs/train/`, rendered analyses
  in `reports/`). Old worktree `/home/marius/w/worktrees/flow-matching/
  major-yak/flow-matching` (this doc's origin; probes in `outputs/`).
  Dev-sample data `/home/marius/w/community_dataset_v1_v3` (3 datasets,
  44,292 frames). Rig datasets `/home/marius/w/datasets/marius/...`.
  Tiny test backbone `outputs/tiny-gemma4` per worktree (regenerate:
  `uv run python -m bijou.gemma4.testing --output outputs/tiny-gemma4`).
- **HF Hub**: `mcobzarenco/bijou-checkpoints` (public; pretrain lineage
  with optimizer.pt, all fine-tune checkpoints without). Uploader
  pattern: `outputs/hf_upload_ckpt.py` (box `/tmp/hf_upload_ckpt.py`).
  `hf` CLI v1.24 available (`uv run hf download ...`).
- **wandb**: project `bijou-dev`, entity `aristotle1337`. Backlog
  recovery: `wandb sync --legacy <run_dir>`.

## 5. CLI knowledge (semantics that matter)

- **train** (`python -m bijou.train`, DDP via torchrun, per-rank
  batch/workers): `--holdout-episodes F --split-seed S` = deterministic
  per-dataset episode split, pure function of (S, repo_id, count, F);
  train loads TRAIN side; every ≥2-episode dataset contributes ≥1
  holdout. `--eval-samples N` (REQUIRED when holdout>0) sizes two
  probes: eval_chunk_mae (holdout side) + train_mae (train side), frames
  drawn exactly as bijou.eval with `--seed <eval-seed>` would; sharded
  round-robin across ranks, all-reduced; CPU-resident batches (host RAM,
  not VRAM); `--eval-seed` separate from `--seed` on purpose.
  `--init-from` = warm start, config-guarded (`ensure_matching_expert_
  config` diffs the full ExpertConfig; loud SystemExit; NOT guarded:
  `--max-soft-tokens` and `--backbone` — known footguns). `--resume` =
  lossless continuation (optimizer.pt), CLI --expert-lr ignored (printed),
  cosine re-evaluated over new --steps → LR jumps up; owner dislikes
  this, prefers init-from+warmup for extensions.
- **eval** (`python -m bijou.eval`): `--episodes {all,train,holdout}` +
  same holdout flags reproduce training's split; `--checkpoint`,
  `--smolvla`, `--num-samples`, `--seed`; `--output-json`; `--report
  X.html --report-theme dark|light`. Baselines state-copy +
  state-copy-norm always. Heun-10 default.
- **rollout** (`python -m bijou.rollout`, runbook `docs/
  rollout_so101.md` with owner-rig copy-paste command): `--check` dry
  run; `--max-relative-target 20` = lerobot per-tick clamp vs measured
  present position (rate limiter, not a safety system); camera NAMES are
  positional prompt slots (sorted); task string must match the recorded
  instruction; `--expert-dtype bfloat16` on the laptop.
- **checks**: `uv run python check.py` (ruff format+check, pyright;
  `--fix` runs lint fixes BEFORE format — COM812 interplay; prints final
  `CHECKS PASSED/FAILED` verdict line — NEVER trust piped/tailed output
  without it; a buffering artifact once masked 3 pyright errors).
- **oracle** (regression gate, run after ANY change near the math):
  dev-sample 2-step CPU run must reproduce loss **1.8896 / 1.7237**
  exactly:
  `uv run python -m bijou.train --train-data /home/marius/w/community_dataset_v1_v3 --backbone outputs/tiny-gemma4 --expert-hidden 64 --expert-heads 2 --expert-intermediate 128 --expert-cross-heads 2 --stream-counts 1 1 2 --steps 2 --batch-size 2 --num-workers 2 --log-every 1 --eval-every 5 --save-every 1000 --eval-samples 4 --device cpu --seed 0 --save-dir outputs/train/oracle_tmp`
  (values are tied to the current tiny-gemma4; regenerate backbone ⇒
  re-baseline loudly). gemma4 changes additionally gate on
  `verify_parity` (needs a big GPU for the real model).

## 6. Recent code changes a fresh session must know

- **Component-lr flags (2026-07-28, renamed same day from
  --lr/--unfreeze-*-lr)**: `--expert-lr` plus `--text-lr` /
  `--vision-lr` (bijou.train) — omit a component's lr to keep it
  frozen (explicit 0 rejected loudly); they train the backbone per
  `docs/plan_unfreeze_trunk.md`. Text set mirrors kv_stop_layer
  exactly (full layers below the deepest stream; stop layer only
  input_layernorm + k/v proj + k_norm; PLE *projections* yes, PLE
  *tables*/embeddings/final norm/lm_head no; embed_vision in the text
  group). Frozen tower/embeddings feed the decoder grad-free inputs ⇒
  autograd builds no graph for them (no gemma4 changes). Live trunk:
  fp32 masters + bf16-autocast prefix encode inside `BijouTrainStep`,
  wrapped by ONE DDP (static_graph; partition exactness = every
  trainable param gets grads each step — verified). Checkpoints gain
  `backbone.safetensors` (params bf16, buffers native — RoPE tables
  fp32; lm_head excluded: tied storage, safetensors rejects aliases);
  detection by file presence, OLD checkpoints hit a byte-identical
  path. `--init-from` an adapted checkpoint with flags OFF = freeze
  the adapted trunk. Flags-off CPU oracle EXACT; flags-on probe
  `outputs/probe_unfreeze_gradflow.py` (records 1.5528 oracle).
  Known-benign: fresh zero-init out_proj blocks trunk grads at step 1
  only; probes/eval run in the trunk's build dtype (fp32 when live —
  ±jitter-tier vs bf16, comparable within-run). backbone_snapshot
  casts HOST-side (device-side transient ≈4.3 GB would OOM at the
  measured 79/80 GB DDP occupancy).
- **bijou/fast/ (2026-07-28)**: owned FAST action tokenizer (DCT +
  BPE, arXiv:2501.09747) for the planned AR+flow mixture — explicit
  orthonormal DCT matrix (no scipy dep), plain-BPE over a synthetic
  alphabet (no ByteLevel), fixed H/D at fit, FastDecodeError on
  malformed sequences (reference zero-fills silently), clip-and-count
  for out-of-alphabet coefficients, save/load = fast_config.json +
  bpe.json (hub-uploadable dir). Measured on 5 local datasets (7.3k
  chunks, union fit 0.8s, `outputs/probe_fast_local.py`): 21–25
  tokens/chunk (12–14× vs naive 300), reconstruction MAE
  0.37–0.68° raw (p99 ≤3.4°), 0.07 ms/chunk encode. First pytest
  infrastructure landed with it: `tests/` + pytest in check.py's
  verdict (dev-dep; pythonpath ini). Plan: fit ONCE on the full
  corpus (needs per-dataset q01/q99 quantile stats — same pass as the
  shelved delta-stats script), upload the dir to HF, `--fast-tokenizer`
  points at it.
- **--fps filter (2026-07-28)**: train + eval accept `--fps 30 ...` to
  drop datasets at other frame rates; default None = all (historical
  behavior, bit-identical). Any filter changes concatenated frame
  indexing ⇒ eval numbers only comparable between same-filter runs —
  which is why the unfreeze run does NOT use it. Eval JSON gains an
  `fps` field; empty selections now die listing every drop reason.

- **Episode holdout** shared by train/eval (`bijou/data.py`
  `EpisodeSplit`, `holdout_episodes`); metadata-vs-parquet frame guard
  derives claimed counts from per-episode lengths under filtering.
- **Eval probes** (train): batched, CPU-resident, sharded, all-reduced;
  wandb rich tables (≤32 rows, strided across rank-0 shard) under
  `eval/samples` & `train/samples`.
- **Padding position fix** (`expert.py`): cross-attention query
  positions = per-sample REAL prefix length (padding_mask.sum(1));
  padded-batch predictions changed by design (~sub-noise: r1 re-scores
  matched originals within 0.05); batch-1 rollout bitwise unaffected.
  Residual batch sensitivity = bf16 kernel-path jitter (HF-equivalent,
  streams drift 2-3e-2 between batch shapes; accepted).
- **Perf** (measured): `kv_stop_layer` — prefix encode stops after
  caching the deepest exported layer's K/V (its attn/MLP were dead
  compute; −1.7% laptop prefix encode, more when schedule stops lower);
  fused AdamW on CUDA (CPU keeps reference path = oracle stable).
  Queued next: length-bucketed batching, backbone torch.compile.
- **Checkpoint schema dataclasses** (`loading.py`): CheckpointMetadata
  (write) / CheckpointInfo + CheckpointTrainArgs (read);
  `DatasetStats.from_state_dict`/`item_tensors`; `data.DatasetInfo`;
  EvalReport, RepoProcessors, RichRow, TrainState, GenerationDefaults.
  Whole HF lineage verified loadable post-migration.
- **Repo health**: MNIST-era code deleted (fmatch/, eval_smolvla.py,
  etc.); deps pruned (av pin kept deliberately — lerobot leaves it
  unconstrained); ruff 0.16 (MUST match Zed's bundled version — 0.16
  changed default rules; styleguide's "toolchain lockstep" section),
  curated strict pyright (@override everywhere, dead-logic = error),
  COM812 one-arg-per-line, isort `I` rules, `docs/code-styleguide.md`
  ("write Python like it's Rust") — read it before writing code.
- Analyses live as one-off scripts in gitignored `outputs/` of the old
  worktree: probe_effect1_fix.py, probe_kv_stop.py, probe_step_split.py,
  probe_e2b_generations.py, sample_count_analysis.py,
  probe_holdout_split.py, probe_ckpt_compat.py, camcheck/. Rendered
  HTML: `reports/` (gitignored) in the main checkout.

## 7. How the owner works (respect these)

Everything in `docs/code-styleguide.md`, plus operational habits:
design discussion BEFORE architecture code; measured changes only —
before/after numbers, bitwise oracles where possible, no "should be
fine"; loud failures over silent fallbacks; explicit seeds with separate
concerns (--seed/--eval-seed/--split-seed); commit+push freely to main
(rebase if remote moved); update docs so everything reads "as of now"
(never reference deleted files); probes/one-offs in gitignored outputs/;
answer speculative questions with reasoned predictions AND a cheap
falsification plan; the owner interrupts long sleeps — poll in short
increments; check GPU occupancy before using the laptop GPU (owner runs
rollouts/recordings there); don't restart in-flight training runs on new
code mid-experiment.

## 8. Known pitfalls (hard-won; do not re-learn)

- lerobot: spawn (never fork) workers; torchcodec fork-unsafe; default
  decoder cache OOMs hosts (→4); `tolerance_s=0.5/fps` (fp32 pts);
  v3 concatenated files break 1e-4 tolerance ~19min in; corrupt
  community videos strike randomly (StatsAttachedDataset substitutes
  index+9973, loudly, bounded retries); factory `make_pre_post_
  processors` drops dataset_stats in the pretrained branch.
- torch: pickled CPU tensors = 1 shm fd each (keep DatasetStats as
  floats; file_system sharing strategy set WORKER-side for raw items);
  SDPA head_dim>256 needs materialized-KV workaround or silently 3×
  slower; fused SDPA vs additive-mask paths differ at bf16 ULP scale;
  Module.__getattr__ stubs return Tensor|Module (narrow via isinstance
  after ModuleList iteration); torch Linear.bias stub lies (cast).
- process: piped output hides failures unless the tool prints a final
  verdict (check.py does now); Zed tool versions must match CLI (ruff
  0.15 vs 0.16 default-rule skew burned an afternoon); host crashes
  corrupt HF datasets arrow caches (zero-byte dataset_info.json →
  delete the cache dirs); rsync big files with --partial + generous
  timeouts; heredocs with escaped quotes over ssh break — scp scripts
  instead; `$(...)` forbidden in local shell tool calls.
- ML: loss not comparable across normalization schemes, data mixes, or
  self-attention modes; eval MAE definition moves with frames/sampler/
  batching (noise-draw jitter ±0.1-0.3); Beta τ-sampler uses global RNG
  (per-rank seeds); Heun's last corrector evaluates τ=0 (fine);
  sliding-window layers see only trailing 512 prefix tokens (exported
  streams are global layers — unaffected).

## 9. Decision queue / next steps

1. **DONE 2026-07-27**: cont45k scored (§2) — holdout flat (6.85 vs
   6.93), rig −1.05 (11.71): scale still pays cross-rig, episode-level
   generalization saturated. Next budget discussion pending the
   fine-tune comparison (§3); candidates: multi-day continuation, a
   contamination-free from-scratch WITH holdout, streams0016 at scale.
2. **DONE 2026-07-27**: ft init-pair scored on the full rig holdout
   (§2): init45k wins −0.22 MAE, both beat copy. Open follow-ups:
   score intermediate checkpoints (probes say 2k≈4k — verify before
   rollout candidate selection); community-side forgetting check;
   consider HF upload of both arms' step_004000.
3. **DONE 2026-07-27**: re-anchoring probe (§2) — delta-actions
   falsified as the cross-rig fix; error = frame-dependent level
   mis-estimation (visual grounding). Promoted accordingly:
   `--sample-draws N` rollout ensembling (attacks level uncertainty
   directly, zero training) and backbone adaptation (now item 6; the
   LoRA variant was later dropped). Cheap sharpener if wanted:
   copy+oracle-offset baseline (= GT within-chunk dispersion,
   CPU-only) to quantify how much better than copy bijou's chunk
   *shape* is.
4. **Physical**: rollout candidate is now `bijou_ft_marius_4k_init45k/
   step_004000` (first checkpoint to beat copy on held-out rig
   episodes) — or ft_v2 (step_001250 or 1250/2000 comparison)
   after verifying camera device mapping; try `--sample-steps 10`;
   optionally add `--sample-draws N` (mean-of-N; ~20 lines in rollout;
   check unimodality on the sampling report first).
5. **Perf**: length-bucketed batching, then torch.compile spike
   (backbone 79% of step). Profile numbers in §2.
6. **NEXT BIG MOVE (owner-approved direction 2026-07-28)**: unfreeze
   the E2B text trunk and continue from cont45k — updated plan in
   `docs/plan_unfreeze_trunk.md` (`--text-lr` /
   `--vision-lr`, vision expected frozen per the acuity
   probe, embeddings/PLE frozen, fp32 masters + bf16 autocast, old
   checkpoints load unchanged). LoRA arm DROPPED (engineering framing:
   single continuation, not an attribution round). Implement flags →
   validation ladder → A/B on the next box.
7. **Future ablation arms** (matched, holdout recipe): streams0016
   re-test at scale (rig-transfer hint — but see acuity probe: K4 is
   the sharpest positional stream, so a shallow-heavy schedule arm,
   e.g. 8-4-4, is now evidence-backed too), adaLN-Zero time
   conditioning (Phase-0 baselines in §2), E4B backbone (4 streams,
   needs 4-entry --stream-counts), E2B **base vs IT** backbone,
   `--trim-leading-idle` (6.7% idle frames), state-noise augmentation.
   Delta-actions demoted (falsified by the re-anchoring probe §2).
8. **Bigger bets**: lerobot policy plugin
   (`lerobot-rollout --policy.type=bijou`).
9. **Hygiene**: rotate wandb key; guard `--backbone`/
   `--max-soft-tokens` at --init-from; MaskSpec/PrefixKV field defaults
   (styleguide exceptions); consider uploading ft_marius_v2_2k to HF
   (laptop-only since the box deletion; ft_marius_4k_init45k
   step_004000 is on HF weights-only — owner deleted optimizer.pt
   deliberately, not resumable).
