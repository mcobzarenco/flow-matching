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

- **`community-v1v2v3-cont45k-ddp4`** on the box (tmux session
  `mainline`, log `~/mainline_cont45k_console.log`): continuation of
  mainline 40k via **--init-from + fresh optimizer + warmup 500** (owner
  explicitly prefers this over extended-horizon --resume: stale Adam
  second moments + re-heated LR cause a loss transient). 45k steps,
  global 256 ≈ +11.5M samples, holdout 0.1/seed 0 ACTIVE this time
  (train mix −10% vs lineage; eval_chunk_mae probes the exact ablation
  eval frames — contaminated for this lineage but comparable; owner
  accepted). lr peak 1e-4 cosine→1e-5, seed 11, save-every 5000,
  eval-every 500 with 256-frame probes. Started ~23:06 on 2026-07-26,
  ~1.1 s/step → ~13.5h. Save dir
  `outputs/train/bijou_community_v1v2v3_cont45k_ddp4`.
- **Morning task**: score the final/latest checkpoint — bijou.eval on
  holdout (contaminated ref: 6.93) + owner rig (honest; mainline-40k
  got 12.76, arms 13.15–17). Launcher scripts for scoring:
  `~/eval_reports.sh <arm> <gpu> <ckpt> <tag>` (3 sides + HTML reports)
  and `~/eval_ablation2.sh` (JSONs only). Summarizer:
  `outputs/abl_results/summarize_r2.py` (worktree copy).

## 4. Machines, data, services

- **Box**: `ssh ubuntu@68.209.73.71` (use `-A` for git push). 4×H100
  SXM, 104 vCPU, 885GB RAM. Repo `~/flow-matching` (sync: `git fetch &&
  git reset --hard origin/main`; uv at `~/.local/bin/uv`). Env for all
  training: `MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072` (+
  `LEROBOT_VIDEO_DECODER_CACHE_SIZE=4` set by train.py). WANDB key via
  `~/.netrc` (launchers export it — wandb.init rejects netrc-only; key
  appeared in shell history once, owner should rotate). Data:
  `~/datasets/mcobzarenco/community_dataset_v{1,2,3}_v3` (1036 datasets
  selected, 26.98M frames, dims 6/6) and `~/datasets/marius/
  so101_pick_place_clean` (7 eps) + `so101_pick_place_v2` (50 eps).
  Launchers in `~`: launch_ablation{,_r2}.sh, launch_ft{,_v2}.sh,
  launch_mainline_cont45k.sh, eval_ablation2.sh, eval_reports.sh.
  Eval JSONs `~/eval_abl_{r1post,r2}_*.json`; reports `~/reports_out/`.
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
  lossless continuation (optimizer.pt), CLI --lr ignored (printed),
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

1. **Morning**: score cont45k final checkpoint (holdout frames +
   owner rig); compare 6.93/12.76. If the curve is still steep, discuss
   a multi-day run (or a fresh from-scratch WITH holdout at larger
   budget — the contamination-free pretrain we don't have yet).
2. **Physical**: rollout ft_v2 (step_001250 or 1250/2000 comparison)
   after verifying camera device mapping; try `--sample-steps 10`;
   optionally add `--sample-draws N` (mean-of-N; ~20 lines in rollout;
   check unimodality on the sampling report first).
3. **Perf**: length-bucketed batching, then torch.compile spike
   (backbone 79% of step). Profile numbers in §2.
4. **Future ablation arms** (matched, holdout recipe): streams0016
   re-test at scale (rig-transfer hint), E4B backbone (4 streams, needs
   4-entry --stream-counts), E2B **base vs IT** backbone (prediction:
   ±0.2 MAE, IT edge grows only with language-diverse data; verify -pt
   checkpoint ships the vision tower; ablate chat template on/off),
   `--trim-leading-idle` (6.7% idle frames), delta-actions.
5. **Bigger bets**: unfreeze trunk (`docs/plan_unfreeze_trunk.md`);
   lerobot policy plugin (`lerobot-rollout --policy.type=bijou`).
6. **Hygiene**: rotate wandb key; README.md still empty; guard
   `--backbone`/`--max-soft-tokens` at --init-from; MaskSpec/PrefixKV
   field defaults (styleguide exceptions); consider uploading ft_v2 +
   cont45k checkpoints to HF when done.
