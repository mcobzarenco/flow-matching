"""Generate outputs/tiny-molmoact2 — the loadable, TRAINABLE tiny
molmoact2-family fixture for the phase-3 CPU loss oracles
(docs/molmoact2-retirement.md: "2-step corridor for ar and joint on
the tiny fixture recorded as new anchors"; the anchors live in
architecture.md's regression-gates section).

Per-checkout artifact like outputs/tiny-gemma4 — regenerating it
re-baselines the molmoact2 oracles, loudly, in the ledger. The builder
itself is :func:`bijou.testing.write_tiny_molmoact2_release` (shared
with the GRPO suite, which fabricates its subject from the same
function under tmp_path).

Run: PYTHONPATH=. uv run python probes/generate_tiny_molmoact2.py
"""

from __future__ import annotations

from pathlib import Path

from bijou.testing import (
    TINY_MOLMOACT2_BLOCK_BASE,
    TINY_MOLMOACT2_VOCAB,
    write_tiny_molmoact2_release,
)

OUTPUT = Path("outputs/tiny-molmoact2")


def main() -> int:
    trunk, checkpoint = write_tiny_molmoact2_release(OUTPUT)
    print(
        f"written {trunk} + {checkpoint} (vocab {TINY_MOLMOACT2_VOCAB}, "
        f"block base {TINY_MOLMOACT2_BLOCK_BASE})",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
