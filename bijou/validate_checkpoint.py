"""Validate and summarize VLA checkpoint directories.

Usage::

    uv run python -m bijou.validate_checkpoint <checkpoint-dir> [...]

Runs the self-containment check (:func:`bijou.checkpoint.validate_checkpoint`)
and prints the metadata summary with per-file link counts (nlink > 1 =
the frozen-part dedup sharing an inode with a parent save or the
imported artifact). A thin module so ``-m`` execution never re-imports
``bijou.checkpoint`` as a second instance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .checkpoint import describe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a VLA checkpoint directory and print its "
        "summary (self-containment check + metadata + per-file link "
        "counts).",
    )
    parser.add_argument("checkpoint", type=Path, nargs="+")
    args = parser.parse_args()
    for directory in args.checkpoint:
        print(describe(directory))


if __name__ == "__main__":
    sys.exit(main())
