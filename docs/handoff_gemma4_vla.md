# Thread handoff: context for the Gemma4-VLA project

Purpose: complete context dump from the previous working thread (July 2026)
so a fresh thread can start implementing **a VLA based on Gemma 4** without
re-discovering the environment. Everything below was verified hands-on unless
marked otherwise. Paths are real; commands are copy-pasteable.

---

## 1. Machines & accounts

### Laptop (primary, where the agent runs)
- Ubuntu, zsh + oh-my-zsh, user `marius`. GPU: RTX 3000 Ada Laptop, 8 GB.
- Repo: `/home/marius/w/flow-matching` (this repo). uv project, Python 3.13
  pinned. Disk: 3.7 TB total, plenty free.
- HF auth: valid login for user **mcobzarenco** (an earlier corrupt cached
  token caused public-repo 401s — fixed by re-login; a guard we had added to
  scripts for this has since been removed).
- gcloud SDK 577 installed at `~/google-cloud-sdk` (user-space, no sudo),
  beta component installed, authed as marius.cobzarenco@gmail.com.

### Lambda H100 box
- `ubuntu@209.20.156.82`, 1× H100 PCIe 80 GB, driver 595.71.05, zsh, ffmpeg.
- Repo at `~/flow-matching` (git clone; laptop pushes, box does
  `git fetch && git reset --hard origin/main`; GitHub auth only via
  **agent forwarding** `ssh -A` — no key on the box).
- Data: `~/community_dataset_v1_v3/ZGGZZG/so100_drop0` (converted v3 sample),
  `~/datasets/marius/so101_pick_place_clean` (own recording, 7 eps).
- Models cached: `lerobot/smolvla_base`, `google/gemma-4-12B-it` (~24 GB).
- Training output: `~/flow-matching/outputs/train/smolvla_pick_place`
  (checkpoints every 100 steps, ran to 2000).
- Provisioned by `init-vm-gpu.sh`.

### GCP (project `kinemind-dev`)
- VM `lerobot-convert`, zone `europe-west4-a`, `n2d-standard-16`,
  **on-demand** (spot was tried and ditched), IP 35.204.223.155,
  ~$0.67/hr + disk ~$5.6/day — **delete when pipeline done**.
- Data disk `lerobot-data`: 3.5 TB **pd-standard** (SSD quota in region is
  only 500 GB), ext4 at `/data`, `auto-delete=no` (survives VM deletion).
- Provisioned by `init-vm-cpu.sh` (mounts the data disk). User did
  `uv run hf auth login` on the box.
- In flight at handoff time: **user downloading community_dataset_v2 to
  `/data/community_dataset_v2`** (resumable `hf download` under tmux).
- GPU quota: `GPUS-ALL-REGIONS-per-project = 1` **granted**;
  `NVIDIA_H100` family in europe-west4 **denied** (at 8 and at 1) by the
  automated gate — appeal via the denial email, or fall back to L4
  (`g2-standard-8`, auto-approves easily) or A100-80GB. Note: H100 shapes
  (`a3-highgpu-1g/2g/4g/8g`) exist only in zones `-b`/`-c`; `-a` has H200
  (`a3-ultragpu-8g`, 8-GPU only). Plan for GPU training on GCP: snapshot
  `lerobot-data` → new disk in the GPU zone.

---

## 2. The repo (`github.com:mcobzarenco/flow-matching`, branch `main`)

Originally a small MNIST flow-matching project (`fmatch/model.py`,
`train.py`, `sampling.py` — untouched); now also hosts the robotics tooling:

| file | what it is |
|---|---|
| `eval_smolvla.py` | Open-loop SmolVLA eval on a local LeRobot v3 dataset. Modular API (`SmolVLAEval.load()`, `predict_chunk`, `evaluate_frame/episode`, frozen result dataclasses) + CLI. **Deterministic**: flow-matching noise from seeded generator, per-frame seed = `seed + dataset_index` (order/stride independent, bit-exact across processes). |
| `lerobot-dataset-tools/ldtools/judge_episode.py` | Claude-as-judge episode QA (Anthropic API). Samples N timesteps × all cameras (512 px JPEG/PNG — token cost is dims-only: ~(w·h)/750), sends task + trajectory stats, strict JSON verdict → frozen `EpisodeJudgment` (+`Verdict`/`TaskCompletion` StrEnums, from/to JSON). `--dry-run` uses the free token-counting endpoint. `--context` for scene clarifications. Usage measured ~2.5–4.3k input tok/episode. |
| `lerobot-dataset-tools/ldtools/judge_episode_gemma.py` | Same concept, fully independent implementation, **local Gemma 4 12B** via transformers. Greedy by default; `--thinking`; `--image-token-budget {70,140,280,560,1120}` (→ `processor_kwargs={"max_soft_tokens": …}`); `--load-in-4bit` (bitsandbytes) for 8 GB GPUs. H100 bf16: ~9.5 s/episode, 4,247 tok @ budget 280. |
| `lerobot-dataset-tools/ldtools/convert_collection.py` | Collection → v3.0 migration pipeline. Census by `codebase_version`, then per sub-dataset: staged copy (skips dup video trees + `*.bak`) → stats-key repair → v2.0→v2.1 hop (synthesizes `episodes_stats.jsonl`; numeric stats exact from parquet, image stats from ~100 torchcodec-sampled frames/episode-camera — the CPU hot spot) → official `convert_dataset_v21_to_v30` (in-process) → validation (`LeRobotDataset` load + count checks) → atomic move. Idempotent (skip if output is valid v3.0), `--workers N` process pool, `--datasets` subset, `--stats-only`, quarantining manifest `conversion_manifest.jsonl`. |
| `init-vm-gpu.sh` | Lambda/Ubuntu24.04 GPU box bootstrap (driver 595, zsh, uv, clone, sync). Non-interactive (needrestart/dpkg guards, oh-my-zsh `--unattended`). |
| `init-vm-cpu.sh` | Same minus driver/reboot; formats+mounts GCE disk `google-lerobot-data` at `/data`. |
| `docs/so101_recording_tutorial.md` | Full SO-101 record→train→deploy tutorial (all lerobot CLIs via uv). |
| `lerobot-dataset-tools/docs/pipeline_plan.md` | The 8-stage plan: inventory → download → normalize → convert → judge → filter → merge-by-feature-signature → QA. |

`pyproject.toml` essentials: `requires-python >=3.13,<3.14`;
`lerobot[dataset,feetech,smolvla,viz]>=0.4` (currently 0.6.0), `av>=15,<16`,
`anthropic`, `pillow`, `bitsandbytes`, torch 2.11+cu130, transformers
**5.14.1** via `[tool.uv] override-dependencies = ["transformers>=5.14"]`
(lerobot caps <5.6 but Gemma 4 Unified needs ≥5.14; SmolVLA verified
bit-exact under 5.14.1 — seeded eval reproduced chunk MAE 11.825181).

Hard-won uv lesson: `uv add` does *inexact* syncs (leaves stale packages);
`uv sync` is exact. A leftover `datasets` package masked a missing
`lerobot[dataset]` extra locally and only failed on a fresh machine.

---

## 3. Datasets

### Community collections (HuggingFaceVLA/…, all **login-gated**)
Disjoint release batches (NOT format versions); v1+v2 = SmolVLA pretraining:

| | v1 | v2 | v3 |
|---|---|---|---|
| sub-datasets | 128 | 340 | 791 |
| episodes | 11,132 | 6,325 | 50,622 |
| frames | 5.11M | 5.03M | 25.97M |
| size | 259 GB | 264 GB | 758 GB |
| format | v2.0+v2.1 | v2.0+v2.1 | per-episode v2.x ("v2.1+") |
| embodiments | SO-100 | SO-100 | 46+ types, 12 action-dim configs |

v1 census (verified locally): 124× v2.1, 4× v2.0.

### Known data quirks (all verified, handled by `convert_collection.py`)
1. **Flat stats keys**: 51/51 checked v1 datasets have stats keyed
   `observation.image*` while features are `observation.images.image*`
   (collection cleaning pipeline bug — its own logs show the mapping, e.g.
   original cams `up`/`left`). Silently breaks normalization (lerobot's
   normalizer no-ops on missing keys) and crashes `lerobot-train`'s
   imagenet-stats injection.
2. **Duplicate video trees**: repos ship each mp4 twice
   (`videos/chunk-000/observation.image/` AND `…/observation.images.image/`).
   Converted output is ~half raw size because we skip the strays.
3. **Corrupt episodes**: `length` ≠ video-span (re-record leftovers; one
   1-frame episode) — crashes `lerobot-edit-dataset` video re-encode with
   "Episode length mismatch". Delete such episodes.
4. AV1 640×480 yuv420p everywhere (v1); already 2× lossy at source.

### Local copies
- Laptop: `/home/marius/w/community_dataset_v1` (complete, byte-verified vs
  hub); `/home/marius/w/community_dataset_v1_v3` (converted so far:
  `ZGGZZG/so100_drop0`, `aimihat/so100_tape`, `ad330/cubePlace`(v2.0 path);
  full-sweep command ready: `uv run python -m ldtools.convert_collection
  --source … --output … --workers 6`).
- Own recording: `/home/marius/w/datasets/marius/so101_pick_place_clean`
  (7 eps, 3399 frames, cams `front`+`wrist`, modern `[-100,100]` units) —
  also on Lambda box.

### Upload plan (decided)
Converted collections → **public HF repos** (per-collection, mirroring
`<user>/<dataset>` layout), via `hf upload-large-folder` (resumable). HF
pricing: $12/TB/mo public with **free egress/CDN** — beats GCS (~$20/TB +
$0.12/GB egress) since the corpus will be re-downloaded to training boxes.
One-time ~$145 GCP egress to upload ~1.3 TB out.

---

## 4. LeRobot knowledge (v0.6.0)

- **v3.0 format**: consolidated `data/chunk-XXX/file-XXX.parquet` (~100 MB),
  `videos/<key>/chunk-XXX/file-XXX.mp4` (~500 MB), `meta/episodes/*.parquet`
  (with `dataset_from_index/to_index` row ranges + per-camera
  `from/to_timestamp`), `meta/tasks.parquet`, `meta/stats.json`.
  Canonical SO-101 features: `action`/`observation.state` float32 (6,) named
  `shoulder_pan.pos, …, gripper.pos`; joints normalized **[-100,100]**
  (degrees only with `--robot.use_degrees=true`); cameras
  `observation.images.<free-form-name>`.
- Converter: `uv run python -m lerobot.scripts.convert_dataset_v21_to_v30
  --repo-id X --root <dataset-dir> --push-to-hub false`. In-place semantics
  (writes `_v30` sibling, swaps, leaves `_old`). v2.0→v2.1 needs
  `episodes_stats.jsonl` (only in lerobot ≤0.3.x, or our synthesizer).
- `--root` in ALL lerobot CLIs = the dataset dir itself (contains `meta/`),
  never a parent tree. Wrong root → silent hub fallback → confusing
  `BackwardCompatibilityError` about the *hub* copy.
- `lerobot-record`: `--dataset.push_to_hub=false` (default true!), keyboard
  → end episode, ← re-record, Esc stop. Policy deployment moved to
  **`lerobot-rollout`** (record refuses policies): `--task`, base strategy
  forbids datasets; `--rename_map` skips the visual-features check.
- `lerobot-train`: needs `--rename_map` when dataset camera names ≠
  checkpoint features; `--policy.push_to_hub=false`; `--dataset.root`;
  accepts dataset lists. Fine-tune SmolVLA: 1.7 GB VRAM @ batch 2 (VLM
  frozen), batch 64 fine on H100.
- `lerobot-dataset-viz` (rerun ≥0.24, `viz` extra), `lerobot-edit-dataset`
  (delete_episodes re-encodes mixed video shards; needs `--new_root` else
  writes to `$HF_LEROBOT_HOME`).
- Web visualizer (`~/w/lerobot-dataset-visualizer`, bun): local datasets via
  `bun scripts/serve-local-datasets.ts <tree-root> 8000` (our hub-style
  file server: Range + CORS + `/resolve/main/` rewrite) +
  `DATASET_URL=http://localhost:8000 bun dev`; browse
  `localhost:3000/{user}/{ds}/episode_0`. v3.0 unlocks stats/3D tabs.

### SmolVLA internals (reference architecture for the new VLA)
- SmolVLM2-500M backbone (16 layers kept) + flow-matching action expert;
  `chunk_size = n_action_steps = 50`; `max_action_dim 32`; images resized
  to 512² internally, `img*2-1` (SigLIP); state via linear proj.
- **Processor pipelines** (v0.6): policies are "naked"; pre/postprocessors
  are serialized artifacts (`policy_preprocessor.json` + safetensors stats)
  loaded via `make_pre_post_processors(policy_cfg, pretrained_path,
  preprocessor_overrides=…)`. Steps flow over `EnvTransition` dicts:
  rename → add-batch-dim → smolvla newline → tokenizer (SmolVLM2, 48 tok) →
  device → normalizer (MEAN_STD state/action, IDENTITY visual).
  Postprocessor: unnormalizer → CPU. Overrides are merged by registry name —
  that's how camera rename maps and dataset-stats injection work.
- **smolvla_base gotcha**: its shipped normalizer stats are keyed
  `so100.buffer.action` etc. (pretraining bookkeeping) — never match runtime
  keys ⇒ silent identity normalization ⇒ garbage zero-shot (63.6 MAE vs 11.7
  with dataset stats injected). Fine-tuned checkpoints save correct keys and
  are self-contained.
- Camera slots: `observation.images.camera1..3` (loose convention: 1=scene,
  2=wrist); lookup is **by key, not position** — missing cameras are dropped
  (`empty_cameras=0`) and it's fine to supply only `camera2`.
- Inference: `predict_action_chunk(batch, noise=)` — noise shape
  `(B, chunk, max_action_dim)`; injecting seeded noise ⇒ fully deterministic
  (verified bit-exact); the rest is deterministic Euler steps.
- Measured: zero-shot base on `ZGGZZG/so100_drop0` chunk-MAE ~11.7–12.8 (deg);
  fine-tune on own 7-ep dataset: chunk-MAE 7.08 (step 100) → 2.08 (step 900,
  [-100,100] units); inference 130–230 ms/frame (laptop), similar H100.

---

## 5. Gemma 4 knowledge (the new VLA's backbone)

Family (June 2026): E2B, E4B (PLE, on-device), **12B "Unified"**, 26B-A4B
MoE, 31B. arXiv 2607.02770. Apache 2.0. transformers ≥5.14
(`gemma4_unified` model type; plain `gemma4` = the encoder variants).

**12B Unified specifics** (all verified against installed source
`transformers/models/gemma4/` + live tensors):
- **Encoder-free**: no ViT. `pixel_values` = raw RGB tiles in [0,1], one row
  per 48×48 px tile (6912 = 3·48²); model scales `2(x−0.5)` and applies a
  single `Linear(6912→hidden)` (`Gemma4VisionPatchEmbedder.input_proj`).
  Learned 2-D pos-emb (x-table + y-table summed) from `image_position_ids`
  ((x,y) per tile, −1 = padding). `Gemma4VisionPooler` spatially avg-pools
  k×k to the **soft-token budget** and scales ×√hidden (fp32).
- Chat template expands each image into `<|image>` + N×`<image_soft_token>`
  (id 258880) + `<image|>` in place; `AutoModelForMultimodalLM` forward
  replaces placeholder ids with pad before embedding, then
  `masked_scatter`s the pooled image features into those slots (count
  check enforced). `mm_token_type_ids` (0=text,1=image,2=other) selects the
  mask AND drives **bidirectional attention within each image block**
  (`use_bidirectional_attention="vision"`); text stays causal.
- **Variable resolution → variable tokens**: 640×480 → 266 soft tokens;
  512×384 → 70. Budget knob `max_soft_tokens ∈ {70,140,280,560,1120}` via
  `processor_kwargs` in `apply_chat_template` (transformers 5.14 wants
  per-call processor kwargs nested).
- Other: hybrid local-sliding/global attention (final layer global), unified
  KV + p-RoPE on global layers, 256K context, native system role, thinking
  mode via `enable_thinking` (output channel `<|channel>thought…<channel|>`;
  `processor.parse_response()` or regex-strip), audio native on 12B.
  Card-recommended sampling 1.0/0.95/64; greedy for determinism.
  Best practice: images before text.
- Bench context: 12B Unified MMMU-Pro 69.1%, MATH-Vision 79.7%.

---

## 6. Judge/curation state (for the later data-filtering phase)

- Both judges emit the same JSON schema (overall 1–10, verdict
  keep/review/discard, task_completion_visible, 4 sub-scores, issues,
  summary) parsed into frozen dataclasses with strict enum validation and
  raw-text fallback.
- Calibration warning: current prompts are **harsh** — every ZGGZZG episode
  judged 3/10 discard; owner disagreed on one ("hole" = duct-tape-roll
  center; fixed via `--context`). Plan requires a ~100-episode hand-labeled
  calibration set before any corpus-scale filtering, plus per-embodiment
  prompt generalization for v3 data.
- Two-tier plan: Gemma first pass (~90–190 GPU-h sequential for 68k eps —
  wants vLLM batching), Claude adjudicates the "review" band (~$100/10% of
  corpus).

---

## 7. Pending / in-flight at handoff

1. GCP VM downloading v2 (user-driven, resumable); then: census → sample →
   full conversion sweep (`--workers 12`) → start v1 download in parallel →
   v3 (758 GB) → `hf upload-large-folder` per collection.
2. Laptop v1 full conversion sweep: ready to run (3/128 done).
3. H100 quota appeal (denial email) or L4/A100 fallback request.
4. Judge calibration set + batch judge mode (`--all-episodes` JSONL) not yet
   built.
5. Teardown reminders: GCP VM ~$16/day while up; Lambda box also bills.

---

## 8. Seeds for the Gemma4-VLA design discussion

- SmolVLA recipe as the template: frozen VLM prefix (images+language+state)
  + flow-matching action expert attending VLM KV; action chunks (50).
  Everything about lerobot integration is known: subclass
  `PreTrainedPolicy`, a `PreTrainedConfig` with input/output features +
  normalization_mapping, processor factory (`make_*_pre_post_processors`),
  registry for `--policy.type`, then `lerobot-train` works for free.
- Why Gemma 4 12B Unified is interesting as a VLA backbone: encoder-free
  patches make image conditioning cheap and *budget-tunable per camera*
  (70–1120 tok — multi-camera robots without token explosion); bidirectional
  image attention; 256K context (long-horizon history); native system role
  for task conditioning; strong vision benchmarks. Risks: 12B ≫ 0.5B
  (SmolVLA) — inference latency on robot hardware (measured ~9.5 s for 165
  gen tokens on H100 in the judge, but a VLA head does a single forward per
  chunk + small expert, not autoregressive text); E2B/E4B are the on-device
  candidates; PLE lookups may complicate KV-sharing with an expert.
- Open design questions: (a) attach a flow-matching expert à la SmolVLA vs
  discrete action tokens vs a diffusion head; (b) which Gemma size;
  (c) reuse the 12B's audio/thinking capabilities or strip; (d) how to feed
  robot state (text vs learned projection into the unified embedding space —
  note the encoder-free design means *anything* linearly projected into
  hidden space is architecturally idiomatic); (e) KV-cache the static prefix
  across chunk replans like SmolVLA's `use_cache`.
- Training data is (or will be) ready: converted v3 corpus on `/data` +
  future filtered version; `so101_pick_place_clean` for smoke tests;
  eval harness pattern (seeded-noise determinism, chunk metrics) ports
  directly.
