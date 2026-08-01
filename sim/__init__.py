"""SO-101 MuJoCo digital twin (prototype). Run scripts as modules from
the repo root, e.g. ``MUJOCO_GL=egl uv run python -m sim.demo_scene``."""

from pathlib import Path

# Generated artifacts (renders, rollout GIFs) go to the repo-wide outputs
# tree, anchored to this file so any cwd works.
OUTPUT_DIR = Path(__file__).parents[1] / "outputs" / "sim"
