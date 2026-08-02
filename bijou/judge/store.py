"""Durable per-dataset verdict store: ``meta/judgments.json``.

The envelope is ``{"judgments": [record, ...]}``; each record is a
``JudgmentRecord`` — provenance plus the verdict — keyed by
(episode_index, model, prompt_hash, num_timesteps, max_image_dim).
Deliberately JSON, not parquet:
verdicts are nested, per-dataset counts are tiny (median ~60 episodes),
and JSON round-trips through the schema-validating dataclasses with no
flattening layer to maintain.

The file lives inside the dataset directory, so hub upload/download
carries it and train-time consumers read it next to the rest of the
metadata — this module needs nothing beyond the stdlib and the (equally
light) schema module, so consumers never pull the judging stacks.
"""

from __future__ import annotations

from pathlib import Path


def discover_datasets(roots: list[Path]) -> list[Path]:
    """Dataset dirs under collection roots (or roots that are datasets).

    Lives here (not in the sweep) so verdict *consumers* — aggregation,
    materialization — can walk a collection without importing the judging
    stacks.
    """
    found: list[Path] = []
    for root in roots:
        root = root.expanduser().resolve()
        if (root / "meta" / "info.json").exists():
            found.append(root)
            continue
        nested = sorted(p.parent.parent for p in root.glob("*/*/meta/info.json"))
        if not nested:
            raise SystemExit(f"no LeRobot datasets under {root}")
        found.extend(nested)
    return found


# The sidecar envelope + I/O live in bijou.annotations (see the note in
# .schema); re-exported for judge-side call sites.
from ..annotations import (
    JUDGMENTS_RELPATH,
    JudgmentRecord,
    load_sidecar,
    write_sidecar,
)

__all__ = [
    "JUDGMENTS_RELPATH",
    "JudgmentRecord",
    "discover_datasets",
    "load_sidecar",
    "write_sidecar",
]
