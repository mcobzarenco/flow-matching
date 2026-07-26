"""Run all checks: `uv run check.py`; `--fix` applies formatting and lint fixes."""

import subprocess
import sys


def run(*cmd: str) -> int:
    print("$", *cmd)
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    fix = "--fix" in sys.argv[1:]
    # In fix mode, lint fixes run BEFORE the formatter: COM812 inserts
    # trailing commas that the formatter then explodes one-per-line, so a
    # single --fix pass lands on the final (stable) style.
    steps = (
        [
            ("ruff", "check", "--fix"),
            ("ruff", "format"),
            ("pyright",),
        ]
        if fix
        else [
            ("ruff", "format", "--check"),
            ("ruff", "check"),
            ("pyright",),
        ]
    )
    return max(run(*step) for step in steps)


if __name__ == "__main__":
    sys.exit(main())
