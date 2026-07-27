# Initializing a new GPU training machine

Audience: an agent (or human) with **no prior context** tasked with
bringing up a fresh cloud GPU instance (e.g. Lambda, Ubuntu 24.04) until
it can run Bijou training. What Bijou is: see `README.md`. Operational
state (results, machines, conventions): `docs/handoff.md`. This doc is
self-contained for the setup task itself.

**Done means**: `nvidia-smi` shows all GPUs; `uv run` works in
`~/flow-matching`; the three community dataset collections and the
owner's two rig datasets are on disk in the expected layout; HF + wandb
auth work; the smoke eval below reproduces a known score; a short
training run writes a checkpoint.

## 0. What you need before starting

- SSH access to the new instance as `ubuntu` (passwordless sudo —
  standard on Lambda images).
- **GitHub**: the repo (`github.com:mcobzarenco/flow-matching`) is
  private — clone happens over SSH. Connect with agent forwarding
  (`ssh -A`) from a machine whose agent holds a key with repo access
  (the owner's laptop). `git push` from the box also needs `-A`.
- **Hugging Face token** for the `mcobzarenco` account (or any account
  that has accepted the Gemma license — `google/gemma-4-e2b-it` is
  gated; everything else used here is public).
- **wandb API key** (project `bijou-dev`, entity `aristotle1337`).
- **Disk**: ≥ 2 TB. The datasets alone are ~930 GB on disk
  (120G + 121G + 687G community, 1.4G rig); checkpoints are 1.6 GB
  each (4.8 GB with optimizer state); the backbone cache ~10 GB.

## 1. Base system: `init-vm-gpu.sh`

From the machine that has the script (repo root):

```sh
scp init-vm-gpu.sh ubuntu@<ip>:
ssh -A ubuntu@<ip> ./init-vm-gpu.sh     # reboots at the end
```

Idempotent-ish; safe to re-run. It performs: apt dist-upgrade, NVIDIA
driver install (pinned in the script), zsh + oh-my-zsh as default
shell, tmux, **ffmpeg (required: torchcodec links against system
libav\*)**, uv, clone of the repo to `~/flow-matching`, and `uv sync`
(pinned Python + all deps, including lerobot extras). It reboots to
load the driver unless `--no-reboot` is passed — reboot before using
the GPU either way.

After the reboot, verify:

```sh
nvidia-smi        # all GPUs listed, driver loaded
cd ~/flow-matching && uv run python -c \
    "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## 2. Auth

**Hugging Face** (gated backbone + dataset downloads):

```sh
cd ~/flow-matching
uv run hf auth login --token <HF_TOKEN>
uv run hf download google/gemma-4-e2b-it --include config.json  # gate check
```

If the gate check fails with 401/403, the token's account has not
accepted the Gemma license — fix that on the HF website first.

**wandb**: write `~/.netrc` (mode 600) with exactly:

```
machine api.wandb.ai
  login user
  password <WANDB_API_KEY>
```

Note: `wandb.init` does NOT read netrc by itself — every launcher
script exports `WANDB_API_KEY` extracted from netrc:

```sh
WANDB_API_KEY="$(python3 -c "import netrc; print(netrc.netrc().authenticators(\"api.wandb.ai\")[2])")"
export WANDB_API_KEY
```

Keep that pattern (don't paste the key into shell commands — it ends
up in shell history; that has happened once already).

## 3. Datasets

Training consumes **collection roots**: directories whose children are
`<author>/<dataset>` LeRobot-v3 dataset dirs (`data/`, `meta/`,
`videos/`). The repo id — and with it the per-dataset normalization
stats and the episode holdout split — is derived from the last two
path components, so the layout below is load-bearing. Do not flatten
or rename.

### 3a. Community collections (public, HF hub)

Three dataset repos, mirroring the on-disk layout exactly:

| hub repo | on-disk size | author dirs |
|---|---|---|
| `mcobzarenco/community_dataset_v1_v3` | 120 GB | 57 |
| `mcobzarenco/community_dataset_v2_v3` | 121 GB | 111 |
| `mcobzarenco/community_dataset_v3_v3` | 687 GB | 237 |

Download each into `~/datasets/mcobzarenco/<name>` (tmux — this is
hours, not minutes):

```sh
mkdir -p ~/datasets/mcobzarenco
cd ~/flow-matching
for c in community_dataset_v1_v3 community_dataset_v2_v3 community_dataset_v3_v3; do
    uv run hf download "mcobzarenco/$c" --repo-type dataset \
        --local-dir ~/datasets/mcobzarenco/"$c"
done
```

Alternative when another box already holds them (same-DC rsync is
usually faster; always `--partial`, generous timeouts, expect to
re-run after stalls):

```sh
rsync -a --partial --info=progress2 --timeout=120 \
    ubuntu@<old-box-ip>:datasets/mcobzarenco/ ~/datasets/mcobzarenco/
```

### 3b. Owner rig datasets (NOT public)

`marius/so101_pick_place_clean` (7 episodes, 88 MB) and
`marius/so101_pick_place_v2` (50 episodes, 1.3 GB). They exist on the
owner's laptop at `/home/marius/w/datasets/marius/` and on the
current H100 box at `~/datasets/marius/`. Copy preserving the
`marius/<name>` layout:

```sh
rsync -a --partial --info=progress2 \
    ubuntu@<old-box-ip>:datasets/marius/ ~/datasets/marius/
```

### 3c. Verify the data

```sh
ls ~/datasets/mcobzarenco/community_dataset_v1_v3 | wc -l   # 57
ls ~/datasets/mcobzarenco/community_dataset_v2_v3 | wc -l   # 111
ls ~/datasets/mcobzarenco/community_dataset_v3_v3 | wc -l   # 237
ls ~/datasets/marius                                        # both rig datasets
```

The authoritative check is the selection report at the top of any
train/eval run over all three roots: it must say
`1036 datasets, 49533 episodes, 24270322 frames` on the train side of
`--holdout-episodes 0.1 --split-seed 0` (and lists every dropped
dataset with a reason — ~206 dropped for incompatible dims etc. is
expected and correct).

## 4. Checkpoints (optional, for warm starts / eval)

Public model repo `mcobzarenco/bijou-checkpoints`, laid out
`<run_name>/<step_dir>/{bijou_config.json,expert.safetensors[,optimizer.pt]}`
— the same shape training writes under `outputs/train/`. To fetch the
current mainline pretrain (see `docs/handoff.md` §2 for what exists):

```sh
cd ~/flow-matching
uv run hf download mcobzarenco/bijou-checkpoints \
    --include "bijou_community_v1v2v3_cont45k_ddp4/step_045000/*" \
    --local-dir outputs/train
```

`--init-from <dir>` needs only config + weights; `--resume <dir>`
additionally needs `optimizer.pt` (only some hub checkpoints carry it).

## 5. Smoke tests (measure, don't assume)

**Eval** — reproduces a known number end-to-end (backbone download +
gated auth, video decode, expert load, CUDA). With the community data
and the cont45k checkpoint from step 4:

```sh
cd ~/flow-matching
MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072 \
uv run python -m bijou.eval \
    --data ~/datasets/mcobzarenco/community_dataset_v1_v3 \
           ~/datasets/mcobzarenco/community_dataset_v2_v3 \
           ~/datasets/mcobzarenco/community_dataset_v3_v3 \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --checkpoint outputs/train/bijou_community_v1v2v3_cont45k_ddp4/step_045000 \
    --num-samples 256 --num-workers 8 --device cuda
```

Expected: `bijou@45000` chunk MAE **≈ 6.85** (Heun-10, seed 0; ±0.1–0.3
is normal noise-draw/kernel jitter, a whole point is not), state-copy
baseline ≈ 10.30.

**Train** — exercises the write path (DDP, probes, checkpoint save).
A few hundred steps on all GPUs (adjust `--nproc-per-node`):

```sh
cd ~/flow-matching
MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072 \
uv run torchrun --standalone --nproc-per-node=4 -m bijou.train \
    --train-data ~/datasets/mcobzarenco/community_dataset_v1_v3 \
                 ~/datasets/mcobzarenco/community_dataset_v2_v3 \
                 ~/datasets/mcobzarenco/community_dataset_v3_v3 \
    --holdout-episodes 0.1 --split-seed 0 \
    --device cuda \
    --steps 300 --batch-size 64 --num-workers 12 \
    --lr 1e-4 --warmup-steps 100 \
    --log-every 10 --eval-every 200 --eval-samples 64 --save-every 200 \
    --seed 0 --save-dir outputs/train/smoke_test
```

Healthy signs: the selection report from §3c; ~1.1–1.5 s/step on
H100s at batch 64/rank once warm (first step ~12 s — CUDA warmup);
loss starting ≈ 2.0 (fresh expert) and clearly falling within a few
hundred steps; `outputs/train/smoke_test/step_000200/` written with
config + weights + optimizer. Delete `outputs/train/smoke_test` after.

## 6. Conventions for real runs (the short version)

- Always under **tmux**, always `2>&1 | tee ~/<run>_console.log`, via a
  launcher script in `~` (see the `launch_*.sh` examples on the
  current box, described in `docs/handoff.md` §4).
- Always `MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072` (glibc
  arena bloat with many dataloader workers; train.py itself caps the
  lerobot video-decoder cache via `LEROBOT_VIDEO_DECODER_CACHE_SIZE=4`).
- Two concurrent `torchrun`s on one box: give each an explicit
  `--rdzv-backend=c10d --rdzv-endpoint=localhost:<distinct port>`
  instead of `--standalone`.
- Code sync: `git fetch && git reset --hard origin/main` — but **never
  while a training run is in flight** (dataloader workers re-import
  code from disk mid-run).
- CLI semantics that matter (holdout/eval/resume-vs-init-from):
  `docs/handoff.md` §5.

## 7. Failure modes seen in practice

- Driver installed but `nvidia-smi` fails → you skipped the reboot.
- 401/403 fetching the backbone → token's account hasn't accepted the
  Gemma license (§2).
- `wandb.init` permission error despite a correct `~/.netrc` → the
  launcher didn't export `WANDB_API_KEY` (§2).
- Zero-byte `dataset_info.json` / corrupt arrow caches under
  `~/.cache/huggingface/` after a host crash → delete the affected
  cache dirs and re-download; the dataset dirs under `~/datasets` are
  plain files and survive fine.
- Long rsync/hf downloads stall → re-run (`--partial` keeps progress);
  don't baby-sit without timeouts.
- Occasional `[data] <repo>[idx] unfetchable ... substituting` lines
  during training are expected (a few corrupt community videos); the
  substitution is loud, bounded, and per-dataset — a handful per
  multi-hour run is normal, a flood is not.
