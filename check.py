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
    # COM812 rides the command line, NOT pyproject: ruff format prints a
    # hardcoded conflict advisory whenever the config selects it (no
    # suppression knob as of 0.16), while --extend-select here is additive
    # and invisible to the formatter — same enforcement, no warning. The
    # editor shows no COM812 squiggles as a result; autofix-only rule, the
    # gate lands it.
    check = ("ruff", "check", "--extend-select", "COM812")
    # In fix mode, lint fixes run BEFORE the formatter: COM812 inserts
    # trailing commas that the formatter then explodes one-per-line, so a
    # single --fix pass lands on the final (stable) style.
    lint: list[tuple[str, ...]] = (
        [(*check, "--fix"), ("ruff", "format")]
        if fix
        else [("ruff", "format", "--check"), check]
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
