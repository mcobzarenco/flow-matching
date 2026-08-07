"""Materialize the AR view of a joint (K-arm) checkpoint — read 4's
trunk-drift instrument (attach-screen pre-reg, 2026-08-07, "Frozen
reads" item 4 + the launch-prep queue item).

A `--joint-ce` checkpoint records TWO decoders: the flow expert
(`decoder` section / `expert.safetensors`) and the phase-1 CE rider
(`joint_ce` section / `joint_ce.safetensors` — a Molmo2ARDecoder, the
phase-1 objective continuing verbatim). The trunk-drift read needs to
greedy-decode K's ADAPTED trunk through that rider on the same panel
plan as the 40k endpoint number — i.e. an `ar_backbone`-view checkpoint
`bijou.eval` can load unchanged. This script synthesizes it:

- `bijou_config.json`: the joint metadata with `decoder` := the
  `joint_ce` section, the `joint_ce` key dropped, and the prompt's
  `residual_exports` cleared (an AR consumer reads the cache, not taps —
  the view's config matches a phase-1 checkpoint's shape; taps would be
  exported and dropped, and the "bit-identical with/without taps"
  oracle says either spelling decodes the same).
- `expert.safetensors` := hardlink of `joint_ce.safetensors` (the rider
  IS the AR decoder; `from_checkpoint` loads it strictly).
- `prompt.safetensors`, `backbone.safetensors` := hardlinks. The
  backbone file is REQUIRED: K trains the trunk, and a view built
  without the adapted snapshot would silently score the warm-start
  trunk — the exact number the drift read compares against.
- `optimizer.pt` is not carried (the view is eval-only).

`train_args` rides verbatim: loaders read only the architecture subset
(CheckpointTrainArgs), and the raw record honestly says what run wrote
the weights. Refuses non-joint checkpoints loudly.

Usage:
    python fontaine/scripts/materialize_joint_ar_view.py \
        --checkpoint outputs/train/fontaine_molmo2_attach_K_10k_ddp4/step_010000
    # writes .../step_010000_ar_view, ready for bijou.eval --checkpoint
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

WEIGHT_LINKS = {
    # view file <- joint-checkpoint source
    "expert.safetensors": "joint_ce.safetensors",
    "prompt.safetensors": "prompt.safetensors",
    "backbone.safetensors": "backbone.safetensors",
}


def ar_view_meta(meta: dict) -> dict:
    """The AR-view ``bijou_config.json`` payload for a joint checkpoint's
    metadata: rider section promoted to ``decoder``, residual taps
    cleared. Pure — raises SystemExit on non-joint input."""
    if meta.get("joint_ce") is None:
        raise SystemExit(
            "not a joint checkpoint: bijou_config.json has no joint_ce "
            "section — the F arm (and every non---joint-ce run) has no AR "
            "view to materialize",
        )
    view = dict(meta)
    view["decoder"] = meta["joint_ce"]
    del view["joint_ce"]
    view["prompt"] = {**meta["prompt"], "residual_exports": []}
    return view


def link_or_copy(source: Path, destination: Path) -> None:
    # bijou.train.link_or_copy's semantics (hardlink, cross-fs copy
    # fallback), local so this CLI never imports torch.
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def materialize(checkpoint: Path, output: Path | None = None) -> Path:
    meta = json.loads((checkpoint / "bijou_config.json").read_text())
    view_meta = ar_view_meta(meta)
    for name, source in WEIGHT_LINKS.items():
        if not (checkpoint / source).exists():
            raise SystemExit(
                f"{checkpoint} has no {source} — "
                + (
                    "a K checkpoint trains the trunk and must carry the "
                    "adapted snapshot; a view without it would silently "
                    "score the warm-start trunk"
                    if source == "backbone.safetensors"
                    else f"required for the AR view's {name}"
                ),
            )
    view_dir = (
        output
        if output is not None
        else checkpoint.parent / (checkpoint.name + "_ar_view")
    )
    view_dir.mkdir(parents=True, exist_ok=True)
    (view_dir / "bijou_config.json").write_text(
        json.dumps(view_meta, indent=2, default=str),
    )
    for name, source in WEIGHT_LINKS.items():
        link_or_copy(checkpoint / source, view_dir / name)
    print(
        f"AR view materialized: {view_dir} (decoder := joint_ce rider, "
        f"step {view_meta.get('step')}) — eval it exactly like a phase-1 "
        "ar_backbone checkpoint",
        flush=True,
    )
    return view_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="view directory (default: <checkpoint>_ar_view alongside)",
    )
    args = parser.parse_args()
    materialize(args.checkpoint, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
