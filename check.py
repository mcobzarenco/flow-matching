"""Run all checks: `uv run check.py`; `--fix` applies formatting and lint fixes."""

import subprocess
import sys


def run(*cmd: str) -> int:
    print("$", *cmd)
    return subprocess.run(cmd).returncode


def main() -> int:
    fix = "--fix" in sys.argv[1:]
    steps = [
        ("ruff", "format") + (() if fix else ("--check",)),
        ("ruff", "check") + (("--fix",) if fix else ()),
        ("pyright",),
    ]
    return max(run(*step) for step in steps)


if __name__ == "__main__":
    sys.exit(main())
