#!/usr/bin/env bash
# Fontaine harness driver — runs ONE headless Claude Code session.
#
#   fontaine-session.sh {tick|work|bootstrap}
#
# Reference implementation (charter §9): deliberately boring — a
# timer fires ticks, a lock serializes sessions, prompts do the
# thinking. The agent may refine this; keep it a short shell script.
set -euo pipefail

MODE="${1:?usage: fontaine-session.sh tick|work|bootstrap}"
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
# Extended-thinking budget for every session (env file may override).
# 31999 is the CLI's "ultrathink" tier — the maximum the interactive
# keywords map to.
export MAX_THINKING_TOKENS="${MAX_THINKING_TOKENS:-31999}"
# Allow long in-session sleep-polls (babysitting through a critical
# window WITHOUT ending the session): the bash tool's per-command cap
# defaults to 10 min, which would truncate a `sleep 1800`.
export BASH_MAX_TIMEOUT_MS="${BASH_MAX_TIMEOUT_MS:-3600000}"

# One session at a time; a skipped tick is fine (the timer refires).
exec 9>"$STATE/session.lock"
if ! flock -n 9; then
    echo "another session holds the lock; skipping $MODE"
    exit 0
fi

timeout_for() {
    case "$1" in
        tick) echo 1800 ;;       # 30 min: a hung tick must self-clear fast
        work) echo 14400 ;;      # 4 h: bounded per charter §9
        bootstrap) echo 28800 ;; # 8 h: dataset mirror + baselines
        *)
            echo "unknown mode: $1" >&2
            return 2
            ;;
    esac
}

# Harness-level failure alert, posted WITHOUT the model: if sessions
# themselves cannot run (usage cap, expired auth, broken install),
# the model cannot report its own outage — this can. discord.py is
# stdlib-only on purpose; plain python3, no venv. 1-h cooldown so a
# persistent failure doesn't post once per timer fire.
alert_failure() {
    local mode="$1" status="$2" log="$3" now last
    now="$(date +%s)"
    last="$(cat "$STATE/last_failure_alert" 2>/dev/null || echo 0)"
    if [ $((now - last)) -lt 3600 ]; then
        return 0
    fi
    if python3 "$DIR/harness/discord.py" post \
        "harness alert: $mode session exited $status (log ${log##*/}). If this repeats, sessions may be unable to run at all — usage cap or auth. Check the box." \
        >/dev/null 2>&1; then
        echo "$now" >"$STATE/last_failure_alert"
    fi
}

run_session() {
    local mode="$1" timeout_s stamp log status
    timeout_s="$(timeout_for "$mode")"
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    log="$LOGS/${stamp}_${mode}.log"
    # --dangerously-skip-permissions: the box is single-purpose and the
    # real boundaries are credential scope (fontaine-* HF token, branch
    # push, one Discord channel) — charter §7 / README "Safety model".
    # stream-json (requires --verbose in -p mode) emits one JSONL event
    # per turn/tool-use AS IT HAPPENS — plain text mode buffers
    # everything until session end, so tail -f showed nothing for hours.
    set +e
    timeout "$timeout_s" claude -p "$(cat "$DIR/prompts/$mode.md")" \
        ${FONTAINE_MODEL:+--model "$FONTAINE_MODEL"} \
        --dangerously-skip-permissions \
        --output-format stream-json --verbose \
        >>"$log" 2>&1
    status=$?
    set -e
    if [ "$status" -ne 0 ]; then
        echo "session $mode exited $status (see $log)"
        alert_failure "$mode" "$status" "$log"
    fi
}

timeout_for "$MODE" >/dev/null # validate before running anything
cd "$REPO"
# #21 P3 (owner-signed): versioned pre-commit hook — check.py gates
# code commits; md/state/blog-book commits stay instant. Idempotent.
git config core.hooksPath fontaine/harness/hooks
run_session "$MODE"

# A tick whose findings exceed its 30-min session cap requests a
# chained work session by touching this marker (prompts/tick.md),
# instead of overrunning its own timeout. ONE chain per invocation:
# if more work remains, the next timer fire (≤10 min out) continues
# — bounded lock-holding by construction.
if [ "$MODE" = "tick" ] && [ -f "$STATE/run_work_next" ]; then
    rm -f "$STATE/run_work_next"
    echo "tick requested a chained work session"
    run_session work
fi
exit 0
