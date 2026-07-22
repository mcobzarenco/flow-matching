#!/usr/bin/env bash
# Initialize a CPU-only processing VM (Ubuntu 24.04) — e.g. the GCP box used
# for the community-dataset v3.0 conversion pipeline. Fully non-interactive
# and safe to re-run. No NVIDIA driver, no reboot required.
#
# Differences from init-vm-gpu.sh:
#   - no GPU driver
#   - formats & mounts an attached data disk (default: the GCE disk named
#     "lerobot-data") at /data, with an fstab entry
#
# Usage:
#   gcloud compute scp init-vm-cpu.sh <vm>:
#   gcloud compute ssh <vm> --ssh-flag="-A" --command="./init-vm-cpu.sh"
#   (agent forwarding -A is needed for the private-repo git clone)
set -euo pipefail

REPO_URL="git@github.com:mcobzarenco/flow-matching.git"
REPO_DIR="$HOME/flow-matching"
DATA_DISK_ID="/dev/disk/by-id/google-lerobot-data"
DATA_MOUNT="/data"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

if [[ "$EUID" -eq 0 ]]; then
    echo "Run as the regular user, not root; sudo is used where needed." >&2
    exit 1
fi
if ! sudo -n true 2>/dev/null; then
    echo "Passwordless sudo is required (standard on GCE)." >&2
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
APT_GET=(sudo -E apt-get -y
    -o Dpkg::Options::=--force-confdef
    -o Dpkg::Options::=--force-confold)

# --- system packages -------------------------------------------------------
log "apt update + dist-upgrade"
"${APT_GET[@]}" update
"${APT_GET[@]}" dist-upgrade

log "installing system packages"
"${APT_GET[@]}" install \
    zsh tmux ffmpeg \
    git curl ca-certificates rsync htop

# --- data disk -------------------------------------------------------------
if [[ -e "$DATA_DISK_ID" ]]; then
    if ! sudo blkid "$DATA_DISK_ID" > /dev/null 2>&1; then
        log "formatting data disk $DATA_DISK_ID (ext4)"
        sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$DATA_DISK_ID"
    else
        log "data disk already formatted"
    fi
    if ! mountpoint -q "$DATA_MOUNT"; then
        log "mounting $DATA_DISK_ID at $DATA_MOUNT"
        sudo mkdir -p "$DATA_MOUNT"
        if ! grep -qs "$DATA_DISK_ID" /etc/fstab; then
            echo "$DATA_DISK_ID $DATA_MOUNT ext4 discard,defaults,nofail 0 2" | sudo tee -a /etc/fstab > /dev/null
        fi
        sudo mount "$DATA_MOUNT"
        sudo chown "$USER":"$USER" "$DATA_MOUNT"
    else
        log "data disk already mounted at $DATA_MOUNT"
    fi
else
    log "no data disk at $DATA_DISK_ID — skipping (boot disk only)"
fi

# --- oh-my-zsh + default shell ----------------------------------------------
if [[ ! -d "$HOME/.oh-my-zsh" ]]; then
    log "installing oh-my-zsh (unattended)"
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" \
        "" --unattended
else
    log "oh-my-zsh already installed"
fi

if [[ "$(getent passwd "$USER" | cut -d: -f7)" != "$(command -v zsh)" ]]; then
    log "setting default shell to zsh"
    sudo chsh -s "$(command -v zsh)" "$USER"
fi

# --- uv ----------------------------------------------------------------------
if [[ ! -x "$HOME/.local/bin/uv" ]]; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    log "uv already installed"
fi
export PATH="$HOME/.local/bin:$PATH"
if ! grep -qs '.local/bin/env' "$HOME/.zshrc"; then
    printf '\n. "$HOME/.local/bin/env"\n' >> "$HOME/.zshrc"
fi

# --- project -----------------------------------------------------------------
if [[ ! -d "$REPO_DIR" ]]; then
    log "cloning $REPO_URL"
    mkdir -p "$HOME/.ssh"
    grep -qs github.com "$HOME/.ssh/known_hosts" || \
        ssh-keyscan github.com >> "$HOME/.ssh/known_hosts" 2>/dev/null
    git clone "$REPO_URL" "$REPO_DIR"
else
    log "repo already present at $REPO_DIR"
fi

log "uv sync"
(cd "$REPO_DIR" && uv sync)

log "setup complete"
cat <<EOF
Next steps:
  - export HF_TOKEN in ~/.zshrc (collections are login-gated; uploads need write scope)
  - run downloads/conversions under tmux, storing everything in $DATA_MOUNT
EOF
