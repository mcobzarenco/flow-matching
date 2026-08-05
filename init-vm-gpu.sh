#!/usr/bin/env bash
# Initialize a fresh Lambda instance (Ubuntu 24.04) for interactive use and
# training. Fully non-interactive; safe to re-run (idempotent-ish).
#
# NVIDIA drivers are deliberately OUT OF SCOPE: the script neither
# installs, upgrades, holds, nor verifies them — driver handling is
# decided per box, by hand. Note the consequence: dist-upgrade CAN
# replace a preinstalled driver, so check the GPU after running this
# (known failure modes and remedies: docs/init_gpu_machine.md §6).
#
# What it does:
#   dist-upgrade, zsh + oh-my-zsh (default shell), tmux, ffmpeg
#   (torchcodec links against system libav*), uv, repo clone + uv sync.
#
# Usage:
#   scp init-vm-gpu.sh ubuntu@<ip>:
#   ssh ubuntu@<ip> ./init-vm-gpu.sh
set -euo pipefail

REPO_URL="git@github.com:mcobzarenco/flow-matching.git"
REPO_DIR="$HOME/flow-matching"

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

# --- system packages -------------------------------------------------------
log "apt update + dist-upgrade"
"${APT_GET[@]}" update
"${APT_GET[@]}" dist-upgrade

log "installing system packages (shell, tooling, ffmpeg)"
"${APT_GET[@]}" install \
    zsh tmux ffmpeg \
    git curl ca-certificates rsync htop

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
NOTE: this script does not manage NVIDIA drivers (per-box decision) —
verify the GPU now, especially if dist-upgrade touched nvidia packages:
  nvidia-smi                       # driver answers
  uv run python -c "import torch; torch.zeros(1, device='cuda')"
                                   # CUDA context actually initializes
                                   # (nvidia-smi alone does NOT prove this;
                                   # see docs/init_gpu_machine.md §6)
Next steps: auth (HF + wandb), datasets, smoke tests — follow
docs/init_gpu_machine.md in the repo step by step.
EOF
