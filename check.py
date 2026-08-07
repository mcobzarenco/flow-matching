"""Run all checks: `uv run check.py`; `--fix` applies formatting and lint
fixes; `--gpu` also runs `@pytest.mark.gpu` tests (default excludes them —
tier convention in tests/README.md)."""

import subprocess
import sys


def run(*cmd: str) -> int:
    # flush: under a pipe, print is block-buffered while the subprocess
    # writes straight through — without it the banners drift below the
    # tools' output and a piped tail can hide failures.
    print("$", *cmd, flush=True)
    return subprocess.run(cmd, check=False).returncode


def steps(*, fix: bool, gpu: bool) -> list[tuple[str, ...]]:
    pytest = (
        ("pytest", "-q", "tests") if gpu else ("pytest", "-q", "-m", "not gpu", "tests")
    )
    # In fix mode, lint fixes run BEFORE the formatter: COM812 inserts
    # trailing commas that the formatter then explodes one-per-line, so a
    # single --fix pass lands on the final (stable) style.
    lint: list[tuple[str, ...]] = (
        [("ruff", "check", "--fix"), ("ruff", "format")]
        if fix
        else [("ruff", "format", "--check"), ("ruff", "check")]
    )
    return [*lint, ("pyright",), pytest]


def main() -> int:
    args = sys.argv[1:]
    code = max(run(*step) for step in steps(fix="--fix" in args, gpu="--gpu" in args))
    # An explicit verdict on the LAST line: piped/tailed output must never
    # be able to hide a failure.
    print("CHECKS PASSED" if code == 0 else f"CHECKS FAILED (exit {code})", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
