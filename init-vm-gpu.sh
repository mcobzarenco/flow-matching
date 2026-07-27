#!/usr/bin/env bash
# Initialize a fresh Lambda instance (Ubuntu 24.04) for interactive use and
# training. Fully non-interactive; safe to re-run (idempotent-ish).
#
# Reproduces the manual setup of the first H100 box:
#   dist-upgrade, NVIDIA driver (only when none is working — Lambda "GPU
#   Base" images ship preinstalled drivers, which are kept), zsh + oh-my-zsh
#   (default shell), tmux, ffmpeg (torchcodec links against system libav*),
#   uv, repo clone + uv sync.
#
# Usage:
#   scp init-vm.sh ubuntu@<ip>:
#   ssh ubuntu@<ip> ./init-vm.sh              # reboots only if the driver
#                                             # needs (re)loading
#   ssh ubuntu@<ip> ./init-vm.sh --no-reboot
set -euo pipefail

NVIDIA_DRIVER="nvidia-driver-595-server"
REPO_URL="git@github.com:mcobzarenco/flow-matching.git"
REPO_DIR="$HOME/flow-matching"
REBOOT=1
[[ "${1:-}" == "--no-reboot" ]] && REBOOT=0

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# --- preconditions ---------------------------------------------------------
if [[ "$EUID" -eq 0 ]]; then
    echo "Run as the regular user (ubuntu), not root; sudo is used where needed." >&2
    exit 1
fi
if ! sudo -n true 2>/dev/null; then
    echo "Passwordless sudo is required (standard on Lambda images)." >&2
    exit 1
fi

# Make apt/dpkg/needrestart fully non-interactive: never ask about conffiles,
# never show the "which services should be restarted?" TUI.
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
APT_GET=(sudo -E apt-get -y
    -o Dpkg::Options::=--force-confdef
    -o Dpkg::Options::=--force-confold)

# --- nvidia driver detection ------------------------------------------------
# Lambda GPU Base images ship working preinstalled drivers — keep those:
# layering $NVIDIA_DRIVER on top risks DKMS/version conflicts. Install only
# on bare images where no driver answers. Detect BEFORE apt touches anything.
NEED_DRIVER=1
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
    log "NVIDIA driver already working (version $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)) — keeping it"
    NEED_DRIVER=0
fi

# --- system packages -------------------------------------------------------
log "apt update + dist-upgrade"
"${APT_GET[@]}" update
"${APT_GET[@]}" dist-upgrade

log "installing system packages (shell, tooling, ffmpeg)"
"${APT_GET[@]}" install \
    zsh tmux ffmpeg \
    git curl ca-certificates rsync htop

if [[ "$NEED_DRIVER" -eq 1 ]]; then
    log "no working NVIDIA driver detected — installing $NVIDIA_DRIVER"
    "${APT_GET[@]}" install "$NVIDIA_DRIVER"
fi

# --- oh-my-zsh + default shell --------------------------------------------
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

# --- uv ---------------------------------------------------------------------
if [[ ! -x "$HOME/.local/bin/uv" ]]; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    log "uv already installed"
fi
export PATH="$HOME/.local/bin:$PATH"
# The uv installer appends PATH setup to existing rc files, but .zshrc may
# have been (re)created by oh-my-zsh afterwards on re-runs — guard explicitly.
if ! grep -qs '.local/bin/env' "$HOME/.zshrc"; then
    printf '\n. "$HOME/.local/bin/env"\n' >> "$HOME/.zshrc"
fi

# --- project ----------------------------------------------------------------
if [[ ! -d "$REPO_DIR" ]]; then
    log "cloning $REPO_URL"
    mkdir -p "$HOME/.ssh"
    grep -qs github.com "$HOME/.ssh/known_hosts" || \
        ssh-keyscan github.com >> "$HOME/.ssh/known_hosts" 2>/dev/null
    git clone "$REPO_URL" "$REPO_DIR"
else
    log "repo already present at $REPO_DIR"
fi

log "uv sync (downloads pinned python + all project deps, incl. lerobot extras)"
(cd "$REPO_DIR" && uv sync)

# --- done -------------------------------------------------------------------
log "setup complete"
cat <<'EOF'
Next steps after the reboot: auth (HF + wandb), datasets, smoke tests —
follow docs/init_gpu_machine.md in the repo step by step. Quick check:
  nvidia-smi
EOF

# A reboot is needed when a driver was just installed, or when a
# preinstalled driver stopped answering because dist-upgrade replaced its
# userspace libraries out from under the loaded kernel module
# ("Driver/library version mismatch").
if [[ "$NEED_DRIVER" -eq 0 ]] && nvidia-smi >/dev/null 2>&1; then
    log "preinstalled NVIDIA driver kept — no reboot needed"
elif [[ "$REBOOT" -eq 1 ]]; then
    log "rebooting now to load the NVIDIA driver (pass --no-reboot to skip)"
    sudo reboot
else
    log "reboot skipped — run 'sudo reboot' before using the GPU"
fi
