#!/usr/bin/env bash
# pi0.5 policy server (box), stock lerobot from the baselines env.
# Binds localhost only: the server unpickles what it receives, so it must
# never listen on a public interface — the laptop connects through an SSH
# tunnel (see pi05_client.sh). The client tells the server which
# checkpoint to load; paths must be valid ON THE BOX.
#
# Usage: bash pi05_server.sh [gpu]        (default gpu 0)
set -euo pipefail
GPU="${1:-0}"
cd ~/flow-matching/baselines
CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python -m lerobot.async_inference.policy_server \
  --host=localhost \
  --port=8080 \
  --fps=30
