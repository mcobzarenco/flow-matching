#!/usr/bin/env bash
# Initialize a fresh Lambda instance (Ubuntu 24.04) for interactive use and
# training. Fully non-interactive; safe to re-run (idempotent-ish).
#
# GPU-driver policy (the hard-won part — read before "fixing" it):
#   The script NEVER touches the NVIDIA stack unless --install-driver is
#   passed. On first run it apt-mark HOLDs every installed nvidia*/
#   libnvidia*/fabricmanager package so that dist-upgrade cannot replace
#   the driver out from under the image. Lambda GPU Base images ship a
#   matched driver + fabricmanager pair (580.173.02 at the time of
#   writing); an unintended dist-upgrade to the 595 stack broke CUDA on a
#   2xH100 VM with "Error 802: system not yet initialized" — driver 595
#   put the NVLinked GPUs into a fabric-probe state ("Fabric State: In
#   Progress") gated on a fabric manager that cannot run on that VM (no
#   NVSwitch devices exposed: "NV_WARN_NOTHING_TO_DO"). The matched-580
#   stack does not gate CUDA on the fabric probe. If a box ever ends up
#   mismatched: purge 'nvidia-*' 'libnvidia-*', install
#   nvidia-driver-580-open + nvidia-fabricmanager=580.173.02-1ubuntu1 +
#   nvidia-utils-580, hold them, reboot.
#
# Everything else:
#   dist-upgrade (with the NVIDIA hold in place), zsh + oh-my-zsh (default
#   shell), tmux, ffmpeg (torchcodec links against system libav*), uv,
#   repo clone + uv sync.
#
# Usage:
#   scp init-vm-gpu.sh ubuntu@<ip>:
#   ssh ubuntu@<ip> ./init-vm-gpu.sh                  # never touches drivers,
#                                                     # never reboots
#   ssh ubuntu@<ip> ./init-vm-gpu.sh --install-driver # bare image: install
#                                                     # the pinned 580 stack
#                                                     # and reboot into it
set -euo pipefail

# The blessed stack: matched driver + fabric manager versions (Lambda GPU
# Base preinstall). Bump both together, never one.
NVIDIA_DRIVER_PACKAGES=(
    "nvidia-driver-580-open"
    "nvidia-utils-580"
    "nvidia-fabricmanager=580.173.02-1ubuntu1"
)
REPO_URL="git@github.com:mcobzarenco/flow-matching.git"
REPO_DIR="$HOME/flow-matching"
INSTALL_DRIVER=0
[[ "${1:-}" == "--install-driver" ]] && INSTALL_DRIVER=1

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

# --- NVIDIA stack: freeze BEFORE any apt operation --------------------------
if [[ "$INSTALL_DRIVER" -eq 1 ]]; then
    log "installing the pinned NVIDIA stack: ${NVIDIA_DRIVER_PACKAGES[*]}"
    "${APT_GET[@]}" update
    "${APT_GET[@]}" install "${NVIDIA_DRIVER_PACKAGES[@]}"
fi
INSTALLED_NVIDIA=$(dpkg-query -W -f '${Package}\n' 'nvidia-*' 'libnvidia-*' 2>/dev/null | sort -u)
if [[ -n "$INSTALLED_NVIDIA" ]]; then
    log "holding the installed NVIDIA packages (dist-upgrade must not touch them)"
    # shellcheck disable=SC2086
    sudo apt-mark hold $INSTALLED_NVIDIA >/dev/null
    if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
        log "NVIDIA driver working (version $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)) — held"
    else
        log "NVIDIA packages held, but the driver is not answering — a reboot may be pending"
    fi
else
    log "no NVIDIA packages installed — GPU-less box, or run with --install-driver"
fi

# --- system packages -------------------------------------------------------
log "apt update + dist-upgrade (NVIDIA stack held)"
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
Next steps: auth (HF + wandb), datasets, smoke tests — follow
docs/init_gpu_machine.md in the repo step by step. Quick GPU check:
  nvidia-smi                       # driver answers
  uv run python -c "import torch; torch.zeros(1, device='cuda')"
                                   # CUDA context actually initializes
                                   # (nvidia-smi alone does NOT prove this;
                                   # see the fabric-probe note up top)
EOF

if [[ "$INSTALL_DRIVER" -eq 1 ]]; then
    log "driver was (re)installed — rebooting to load it"
    sudo reboot
fi
