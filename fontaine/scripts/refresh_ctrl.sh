#!/usr/bin/env bash
# Refresh the throwaway control checkout (#21 P7, owner-signed).
#
#   refresh_ctrl.sh [SRC_REPO] [DEST]
#   defaults: ~/flow-matching  ~/flow-matching-ctrl
#
# Control evals run from a snapshot so code never syncs under a live
# run — but the snapshot must say what it mirrors (the pre-P7 copy
# recorded nothing). Delete-and-refresh at each use:
#   1. any existing DEST is renamed DEST.prev-<UTC> — moved, never
#      deleted; harvest outputs/ from there, then remove it by hand
#   2. git archive HEAD -> DEST: tracked files only (no .git, no
#      outputs/, no untracked files, no dirty edits — the snapshot
#      IS the commit)
#   3. DEST/CTRL_SOURCE_COMMIT is written: "<sha> <utc> from <src>".
#      Evals launched from DEST cite it in their report notes.
# A dirty SRC tree is allowed (archive reads HEAD, not the tree) but
# the dirty list is printed so "you got HEAD, not your edits" is
# never silent. First `uv run` in a fresh DEST rebuilds its venv.
set -euo pipefail

SRC="${1:-$HOME/flow-matching}"
DEST="${2:-$HOME/flow-matching-ctrl}"

sha="$(git -C "$SRC" rev-parse HEAD)"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"

if [ -e "$DEST" ]; then
    mv "$DEST" "$DEST.prev-$stamp"
    echo "old snapshot -> $DEST.prev-$stamp (harvest outputs/, then rm -rf it)"
fi
mkdir -p "$DEST"
git -C "$SRC" archive "$sha" | tar -x -C "$DEST"
printf '%s %s from %s\n' "$sha" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SRC" \
    >"$DEST/CTRL_SOURCE_COMMIT"

dirty="$(git -C "$SRC" status --porcelain)"
if [ -n "$dirty" ]; then
    echo "note: SRC tree is dirty — the snapshot is HEAD ($sha), not these edits:"
    echo "$dirty"
fi
echo "ctrl refreshed: $sha -> $DEST"
