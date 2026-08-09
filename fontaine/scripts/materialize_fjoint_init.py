"""Materialize the F-then-joint composite warm start — Instrument §1 of
the fjoint-rung pre-reg (2026-08-09, `--init-from` audit result: no
train.py surgery needed).

The J arm warm-starts a `--joint-ce` run from the screen's F endpoint:
flow expert = F@10k (the converged capital this rung spends), CE rider =
the phase-1 FAST tables CONTINUING from the 60k AR endpoint (a fresh
rider would restart the action head — exactly what
`--backbone-init-from` would do, discarding F's expert). Neither
checkpoint alone satisfies `--init-from --joint-ce`'s load contract (F
carries no `joint_ce.safetensors`; the phase-1 checkpoint carries no
flow expert), so this script synthesizes the composite — the inverse of
`materialize_joint_ar_view.py`, which splits a joint checkpoint back
into its AR view:

- `expert.safetensors`, `prompt.safetensors`, `backbone.safetensors`
  := hardlinks of the FLOW checkpoint's files (F@10k verbatim; the
  backbone file is REQUIRED — it is the frozen trunk J unfreezes).
- `joint_ce.safetensors` := hardlink of the PHASE-1 checkpoint's
  `expert.safetensors` (its decoder IS a Molmo2ARDecoder; the strict
  rider load consumes it unchanged).
- `bijou_config.json` := the flow metadata with a `joint_ce` section
  := the phase-1 metadata's `decoder` section (the shape
  `save_checkpoint` writes for a real joint run; `train_args` rides
  the flow checkpoint's verbatim — the raw record honestly says what
  run wrote the weights the loaders consume).
- `optimizer.pt` is not carried: the pre-reg's warm start is
  weights-only with a fresh optimizer, which is `--init-from`'s
  contract anyway.

Coherence guard: the rider's tables were trained against the phase-1
trunk, and F's frozen trunk must BE that trunk (the screen's F arm
hardlinked it from the 60k endpoint). The script refuses to compose
checkpoints whose `backbone.safetensors` differ byte-wise — passing the
wrong phase-1 checkpoint (a 40k, say) would otherwise build a composite
whose rider argues with its own trunk.

Usage:
    python fontaine/scripts/materialize_fjoint_init.py \
        --flow-checkpoint outputs/train/fontaine_molmo2_attach_F_10k_ddp4/step_010000 \
        --phase1-checkpoint outputs/train/fontaine_molmo2_ar_60k_resume_ddp4/step_060000
    # writes .../step_010000_fjoint_init, ready for
    #   bijou.train --init-from ... --joint-ce --joint-unfrozen-seam
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

# composite file <- (source checkpoint role, source file)
WEIGHT_LINKS = {
    "expert.safetensors": ("flow", "expert.safetensors"),
    "prompt.safetensors": ("flow", "prompt.safetensors"),
    "backbone.safetensors": ("flow", "backbone.safetensors"),
    "joint_ce.safetensors": ("phase1", "expert.safetensors"),
}


def composite_meta(flow_meta: dict, phase1_meta: dict) -> dict:
    """The composite ``bijou_config.json`` payload: flow metadata plus a
    ``joint_ce`` section from the phase-1 decoder. Pure — raises
    SystemExit on inputs that are not (flow expert, phase-1 AR) pairs."""
    flow_decoder = flow_meta.get("decoder")
    if flow_decoder is None or flow_decoder.get("kind") != "flow":
        raise SystemExit(
            "--flow-checkpoint is not a flow-expert checkpoint "
            f"(decoder kind {None if flow_decoder is None else flow_decoder.get('kind')!r})"
            " — the composite's expert slot is the F arm's converged "
            "flow decoder",
        )
    if flow_meta.get("joint_ce") is not None:
        raise SystemExit(
            "--flow-checkpoint already carries a joint_ce section — it is "
            "a joint checkpoint, and --init-from consumes it directly "
            "(nothing to materialize)",
        )
    phase1_decoder = phase1_meta.get("decoder")
    if phase1_decoder is None or phase1_decoder.get("kind") != "ar_backbone":
        raise SystemExit(
            "--phase1-checkpoint is not an ar_backbone checkpoint "
            f"(decoder kind {None if phase1_decoder is None else phase1_decoder.get('kind')!r})"
            " — the CE rider continues the phase-1 FAST tables, and only "
            "an ar_backbone run's expert.safetensors holds them",
        )
    flow_backbone = flow_meta.get("backbone", {}).get("id")
    phase1_backbone = phase1_meta.get("backbone", {}).get("id")
    if flow_backbone != phase1_backbone:
        raise SystemExit(
            f"backbone mismatch: flow checkpoint records {flow_backbone!r}, "
            f"phase-1 records {phase1_backbone!r} — the rider's tables and "
            "the trunk must come from the same lineage",
        )
    composite = dict(flow_meta)
    composite["joint_ce"] = phase1_decoder
    return composite


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_same_trunk(flow_trunk: Path, phase1_trunk: Path) -> None:
    """Byte-identity of the two trunk snapshots (fast path: same inode —
    the F arm hardlinked its frozen trunk from the 60k endpoint, so on
    the box this is a stat call)."""
    flow_stat, phase1_stat = flow_trunk.stat(), phase1_trunk.stat()
    if (flow_stat.st_dev, flow_stat.st_ino) == (phase1_stat.st_dev, phase1_stat.st_ino):
        return
    if flow_stat.st_size != phase1_stat.st_size or file_digest(
        flow_trunk,
    ) != file_digest(phase1_trunk):
        raise SystemExit(
            f"trunk mismatch: {flow_trunk} and {phase1_trunk} differ — the "
            "phase-1 rider's tables were trained against ITS trunk, and "
            "the flow arm's frozen trunk must be byte-identical to it "
            "(wrong phase-1 checkpoint?)",
        )


def link_or_copy(source: Path, destination: Path) -> None:
    # bijou.train.link_or_copy's semantics (hardlink, cross-fs copy
    # fallback), local so this CLI never imports torch.
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def materialize(
    flow_checkpoint: Path,
    phase1_checkpoint: Path,
    output: Path | None = None,
) -> Path:
    flow_meta = json.loads((flow_checkpoint / "bijou_config.json").read_text())
    phase1_meta = json.loads((phase1_checkpoint / "bijou_config.json").read_text())
    meta = composite_meta(flow_meta, phase1_meta)
    checkpoints = {"flow": flow_checkpoint, "phase1": phase1_checkpoint}
    for name, (role, source) in WEIGHT_LINKS.items():
        if not (checkpoints[role] / source).exists():
            raise SystemExit(
                f"{checkpoints[role]} has no {source} — "
                + (
                    "the F arm inherits its frozen trunk as a hardlinked "
                    "snapshot, and J unfreezes exactly that file; a "
                    "composite without it would warm-start whatever trunk "
                    "the --backbone flag downloads"
                    if (role, source) == ("flow", "backbone.safetensors")
                    else f"required for the composite's {name}"
                ),
            )
    if not (phase1_checkpoint / "backbone.safetensors").exists():
        raise SystemExit(
            f"{phase1_checkpoint} has no backbone.safetensors — the trunk-"
            "coherence check needs the phase-1 trunk snapshot to compare "
            "against",
        )
    ensure_same_trunk(
        flow_checkpoint / "backbone.safetensors",
        phase1_checkpoint / "backbone.safetensors",
    )
    composite_dir = (
        output
        if output is not None
        else flow_checkpoint.parent / (flow_checkpoint.name + "_fjoint_init")
    )
    composite_dir.mkdir(parents=True, exist_ok=True)
    for name, (role, source) in WEIGHT_LINKS.items():
        link_or_copy(checkpoints[role] / source, composite_dir / name)
    (composite_dir / "bijou_config.json").write_text(json.dumps(meta, indent=2))
    return composite_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flow-checkpoint",
        type=Path,
        required=True,
        help="the F arm's endpoint step directory (flow expert on the "
        "frozen trunk; no joint_ce.safetensors)",
    )
    parser.add_argument(
        "--phase1-checkpoint",
        type=Path,
        required=True,
        help="the phase-1 ar_backbone endpoint step directory whose "
        "expert.safetensors holds the FAST tables the CE rider continues "
        "(the 60k AR endpoint — the same checkpoint F's trunk was "
        "hardlinked from)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="composite directory (default: <flow-checkpoint>_fjoint_init "
        "beside the source)",
    )
    args = parser.parse_args()
    composite = materialize(args.flow_checkpoint, args.phase1_checkpoint, args.output)
    print(f"composite warm start materialized at {composite}")
    print(
        "launch with: bijou.train --init-from "
        f"{composite} --joint-ce --joint-unfrozen-seam ... "
        "(fjoint pre-reg flags; fresh optimizer, step 0)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
