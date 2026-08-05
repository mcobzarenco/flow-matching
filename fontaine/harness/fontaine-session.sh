#!/usr/bin/env bash
# Fontaine harness driver — runs ONE headless Claude Code session.
#
#   fontaine-session.sh {tick|work|bootstrap}
#
# Reference implementation (charter §9): deliberately boring — a
# timer fires ticks, a lock serializes sessions, prompts do the
# thinking. The agent may refine this; keep it a short shell script.
set -euo pipefail

MODE="${1:?usage: fontaine-session.sh {tick|work|bootstrap}}"
REPO="${FONTAINE_REPO:-$HOME/flow-matching}"
# systemd user units start with a minimal PATH; claude + uv live in
# ~/.local/bin.
export PATH="$HOME/.local/bin:$PATH"
DIR="$REPO/fontaine"
STATE="$DIR/harness/state"
LOGS="$DIR/harness/logs"
mkdir -p "$STATE" "$LOGS"

# Secrets + config (Discord token/channel, WANDB_API_KEY, optional
# FONTAINE_MODEL). Created by the owner at ignition; never in git.
ENV_FILE="$HOME/.config/fontaine/env"
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
fi
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072

# One session at a time; a skipped tick is fine (the timer refires).
exec 9>"$STATE/session.lock"
if ! flock -n 9; then
    echo "another session holds the lock; skipping $MODE"
    exit 0
fi

case "$MODE" in
    tick) TIMEOUT=1800 ;;       # 30 min: a tick must never outlive the timer interval by much
    work) TIMEOUT=14400 ;;      # 4 h: bounded per charter §9
    bootstrap) TIMEOUT=28800 ;; # 8 h: dataset mirror + baselines
    *)
        echo "unknown mode: $MODE" >&2
        exit 2
        ;;
esac

PROMPT_FILE="$DIR/prompts/$MODE.md"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$LOGS/${STAMP}_${MODE}.log"

cd "$REPO"
# --dangerously-skip-permissions: the box is single-purpose and the
# real boundaries are credential scope (fontaine-* HF token, branch
# push, one Discord channel) — charter §7 / README "Safety model".
set +e
timeout "$TIMEOUT" claude -p "$(cat "$PROMPT_FILE")" \
    ${FONTAINE_MODEL:+--model "$FONTAINE_MODEL"} \
    --dangerously-skip-permissions \
    >>"$LOG" 2>&1
STATUS=$?
set -e
if [ "$STATUS" -ne 0 ]; then
    echo "session $MODE exited $STATUS (see $LOG)"
fi
exit 0
