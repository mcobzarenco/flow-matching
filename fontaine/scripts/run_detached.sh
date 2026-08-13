#!/usr/bin/env bash
# run_detached.sh — REQUIRED launch wrapper for any job that must
# outlive the driver session (GPU evals/training, long materializes).
#
#   fontaine/scripts/run_detached.sh <unit-name> <cmd> [args...]
#
# Why this exists (3 incidents 2026-08-07, driver-background-task-guard):
# driver sessions run headless under fontaine-tick.service. Jobs
# launched as plain children die when the claude turn completes
# (session-task teardown), and even `setsid nohup` jobs die when the
# service unit stops — systemd kills the unit's whole CGROUP
# (KillMode=control-group default); setsid escapes the terminal
# session, NOT the cgroup. The only reliable escape is a transient
# unit of one's own: systemd-run. This wrapper codifies the working
# 3rd-relaunch recipe (15:58:26Z), including the two gotchas measured
# there: the clean unit env lacks `uv` (exit 127 without the PATH
# setenv) and a bad launch dies within seconds, silently, unless you
# check — so this script verifies the unit is still active after a
# short grace and surfaces the journal tail if not.
#
# Run from the repo root (the unit inherits the caller's cwd via
# --same-dir). Check on the job afterwards with:
#   systemctl --user status <unit-name>
#   journalctl --user -u <unit-name> -f
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: run_detached.sh <unit-name> <cmd> [args...]" >&2
    exit 2
fi
NAME="$1"
shift

if systemctl --user is-active --quiet "$NAME"; then
    echo "ABORT: unit $NAME is already active — pick another name or stop it first" >&2
    exit 1
fi

# Forward MUJOCO_GL when the caller sets it (e.g. `MUJOCO_GL=egl
# run_detached.sh ...`): the transient unit gets the user manager's
# clean env, NOT the caller's — R0-A launch 1 (21:55Z 08-13) died in
# its first sim worker on exactly this (FatalError: no OpenGL platform
# library; the env prefix silently stopped at this script).
systemd-run --user --unit="$NAME" --collect --same-dir \
    --setenv=PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin" \
    --setenv=HOME="$HOME" \
    ${MUJOCO_GL:+--setenv=MUJOCO_GL="$MUJOCO_GL"} \
    "$@"

# A launch that dies inside the grace window is a failed launch (the
# exit-127 class): surface it loudly instead of returning success.
# RUN_DETACHED_GRACE override exists for the driver test only.
sleep "${RUN_DETACHED_GRACE:-5}"
if ! systemctl --user is-active --quiet "$NAME"; then
    echo "LAUNCH FAILURE: unit $NAME is not active ~5s after start; journal tail:" >&2
    journalctl --user -u "$NAME" -n 20 --no-pager >&2 || true
    exit 1
fi
echo "launched $NAME (own transient unit, survives driver teardown)"
echo "  status:  systemctl --user status $NAME"
echo "  follow:  journalctl --user -u $NAME -f"
