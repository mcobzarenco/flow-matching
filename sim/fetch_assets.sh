#!/usr/bin/env bash
# Fetch + build the sim's third-party assets (not committed: benchy is
# CC BY-ND 4.0 — no redistribution of derivatives — and the menagerie arm
# is 7 MB of STLs better pinned than vendored).
#   1. menagerie robotstudio_so101 (Apache-2.0), pinned commit
#   2. official 3DBenchy STL -> decimated/UV'd OBJs (convert_benchy.py)
# Our task scene (bijou_pickplace.xml) is tracked in git and survives in
# place; everything else under assets/ (repo root) is fetched.
set -euo pipefail
cd "$(dirname "$0")/.."

MENAGERIE_COMMIT="main"  # pin to a SHA once the prototype graduates

if [[ ! -f assets/robotstudio_so101/so101.xml ]]; then
    tmp="$(mktemp -d)"
    git clone -q --depth 1 --filter=blob:none --sparse \
        https://github.com/google-deepmind/mujoco_menagerie "$tmp"
    git -C "$tmp" sparse-checkout set robotstudio_so101
    git -C "$tmp" checkout -q "$MENAGERIE_COMMIT"
    mkdir -p assets
    rsync -a --exclude bijou_pickplace.xml "$tmp/robotstudio_so101/" assets/robotstudio_so101/
    rm -rf "$tmp"
    echo "fetched menagerie robotstudio_so101"
else
    echo "menagerie arm already present"
fi

if [[ ! -f assets/benchy/benchy_visual.obj ]]; then
    curl -sL -o /tmp/3DBenchy.stl \
        https://raw.githubusercontent.com/CreativeTools/3DBenchy/master/Single-part/3DBenchy.stl
    uv run sim/convert_benchy.py /tmp/3DBenchy.stl
else
    echo "benchy meshes already present"
fi
