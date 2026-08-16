# Initializing a new GPU training machine

> Note (2026-08): the smoke test below predates the curated corpus and current CLI defaults — it exercises the flow path only. Current runs train on `community_curated_v0` with the Molmo2 trunk; see the launchers under `fontaine/scripts/box/` for working full command lines (weights gates, `--zero1`, chunked backward, async saves).

Audience: an agent (or human) with **no prior context** tasked with
bringing up a fresh cloud GPU instance (e.g. Lambda, Ubuntu 24.04) until
it can run Bijou training. What Bijou is: see `README.md` and
`docs/architecture.md`; operating conventions:
`docs/working-together.md`. This doc is self-contained for the setup
task itself.

**Done means**: `nvidia-smi` shows all GPUs; `uv run` works in
`~/flow-matching`; the three community dataset collections and the
owner's two rig datasets are on disk in the expected layout; HF + wandb
auth work; the smoke training run below behaves as described and
writes a checkpoint.

## 0. What you need before starting

- SSH access to the new instance as `ubuntu` (passwordless sudo —
  standard on Lambda images).
- **GitHub**: the repo (`github.com:mcobzarenco/flow-matching`) is
  private — clone happens over SSH. Connect with agent forwarding
  (`ssh -A`) from a machine whose agent holds a key with repo access
  (the owner's laptop). `git push` from the new machine also needs
  `-A`.
- **Hugging Face token** for the `mcobzarenco` account (or any account
  that has accepted the Gemma license — `google/gemma-4-e2b-it` is
  gated; everything else used here is public).
- **wandb API key** (project `bijou-dev`, entity `aristotle1337`).
- **Disk**: ≥ 2 TB. The datasets alone are ~930 GB on disk; training
  checkpoints are a few GB each; the backbone cache ~10 GB.

## 1. Base system: `init-vm-gpu.sh`

From the machine that has the script (repo root):

```sh
scp init-vm-gpu.sh ubuntu@<ip>:
ssh -A ubuntu@<ip> ./init-vm-gpu.sh
```

Idempotent-ish; safe to re-run; never reboots. It dist-upgrades,
installs zsh + oh-my-zsh as default shell, tmux, **ffmpeg (required:
torchcodec links against system libav\*)**, uv, clones the repo to
`~/flow-matching`, and runs `uv sync`.

**NVIDIA drivers are out of the script's scope** (2026-08-05: driver
handling is decided per box, by hand — the script neither installs,
upgrades, holds, nor verifies them). The consequence: a dist-upgrade
CAN replace a preinstalled driver, so verify the GPU right after
running it — and see §6 for the known driver failure modes and the
matched-pair remedy before touching anything by hand.

Afterwards, verify — both lines; `nvidia-smi` alone does NOT prove
CUDA works (§6):

```sh
nvidia-smi        # all GPUs listed, driver loaded
cd ~/flow-matching && uv run python -c \
    "import torch; torch.zeros(1, device='cuda'); print('cuda ok', torch.cuda.device_count())"
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

Alternative when another machine already holds them (machine-to-
machine rsync is often faster; always `--partial`, generous timeouts,
expect to re-run after stalls):

```sh
rsync -a --partial --info=progress2 --timeout=120 \
    <user>@<source-host>:datasets/mcobzarenco/ ~/datasets/mcobzarenco/
```

### 3b. Owner rig datasets (private HF repos)

`mcobzarenco/so101_pick_place_clean` (7 episodes, 88 MB) and
`mcobzarenco/so101_pick_place_v2` (50 episodes, 1.3 GB) are published
as **private** dataset repos on the hub — the download requires the HF
auth from §2 to be for an account with access. Download both into the
same `~/datasets/mcobzarenco/` root as the community collections:

```sh
cd ~/flow-matching
uv run hf download mcobzarenco/so101_pick_place_clean --repo-type dataset \
    --local-dir ~/datasets/mcobzarenco/so101_pick_place_clean
uv run hf download mcobzarenco/so101_pick_place_v2 --repo-type dataset \
    --local-dir ~/datasets/mcobzarenco/so101_pick_place_v2
```

(Small enough to run in the foreground.) The hub copies carry the
exact-quantile `stats.json` backfill that dataset selection requires —
if a download of any dataset fails selection with a "backfill" error,
the copy is stale; re-download rather than patching locally.

### 3c. Verify the data

```sh
ls ~/datasets/mcobzarenco/community_dataset_v1_v3 | wc -l   # 57
ls ~/datasets/mcobzarenco/community_dataset_v2_v3 | wc -l   # 111
ls ~/datasets/mcobzarenco/community_dataset_v3_v3 | wc -l   # 237
ls ~/datasets/mcobzarenco/so101_pick_place_clean/meta       # info.json, stats.json, ...
ls ~/datasets/mcobzarenco/so101_pick_place_v2/meta          # info.json, stats.json, ...
```

The authoritative check is the selection report at the top of any
train/eval run over all three roots: it must say
`1036 datasets, 49533 episodes, 24270322 frames` on the train side of
`--holdout-episodes 0.1 --split-seed 0` (and lists every dropped
dataset with a reason — ~206 dropped for incompatible dims etc. is
expected and correct).

## 4. Smoke test (measure, don't assume)

A short training run from scratch — the expert is freshly (randomly)
initialized, no checkpoint needed — exercises the whole stack: gated
backbone download (~10 GB on first run), video decode, DDP, CUDA,
eval probes, checkpoint write. A few hundred steps on all GPUs
(adjust `--nproc-per-node`):

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
    --decoder-lr 1e-4 --warmup-steps 100 \
    --log-every 10 --eval-every 200 --eval-samples 64 --save-every 200 \
    --seed 0 --save-dir outputs/train/smoke_test
```

Healthy signs: the selection report from §3c; ~1.1–1.5 s/step per
H100 at batch 64/rank once warm (first step ~12 s — CUDA warmup);
loss starting ≈ 2.0 (fresh expert) and clearly falling within a few
hundred steps; `outputs/train/smoke_test/step_000200/` written with
config + weights + optimizer. Delete `outputs/train/smoke_test` after.

## 5. Conventions for real runs (the short version)

- Always under **tmux**, always `2>&1 | tee ~/<run>_console.log`, via a
  launcher script in `~` (local copies of past launchers live in the
  main checkout's gitignored `outputs/`; header conventions in
  `docs/working-together.md`).
- Always `MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072` (glibc
  arena bloat with many dataloader workers; `bijou.train` itself caps
  the lerobot video-decoder cache via
  `LEROBOT_VIDEO_DECODER_CACHE_SIZE=4`).
- Two concurrent `torchrun`s on one box: give each an explicit
  `--rdzv-backend=c10d --rdzv-endpoint=localhost:<distinct port>`
  instead of `--standalone`.
- Code sync: `git fetch && git reset --hard origin/main` — but **never
  while a training run is in flight** (dataloader workers re-import
  code from disk mid-run).
- CLI semantics that matter (holdout/eval/resume-vs-init-from):
  `docs/architecture.md` §5.

## 6. Failure modes seen in practice

- Driver installed but `nvidia-smi` fails → you skipped the reboot.
- `nvidia-smi` worked during setup but fails after ("Driver/library
  version mismatch") → dist-upgrade bumped the preinstalled driver's
  userspace; reboot loads the matching kernel module.
- **CUDA "Error 802: system not yet initialized" while `nvidia-smi`
  answers** (H100 VMs) → a driver newer than Lambda's matched pair
  fabric-probes NVLinked GPUs and gates CUDA on nvidia-fabricmanager,
  which cannot run without an exposed NVSwitch (measured on a 2×H100
  VM after a dist-upgrade pulled 595 over the preinstalled matched
  580 pair). Fix: purge nvidia-*, install a matched driver +
  fabricmanager pair (`nvidia-driver-580-open` +
  `nvidia-fabricmanager=580.173.02` was the known-good pair — bump
  both together, never one), optionally `apt-mark hold` both, reboot;
  if fabric registration never completes, add `NVreg_NvLinkDisable=1`
  (GPUs become PCIe peers; NCCL allreduce verified bit-exact). A dead
  fabricmanager does NOT silently break DDP — collectives complete or
  fail loudly. NOTE: `init-vm-gpu.sh` deliberately does not manage
  the NVIDIA stack (per-box decision), so its dist-upgrade can
  surface exactly this class — always re-verify CUDA after running
  it.
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
