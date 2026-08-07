"""Oracles for fontaine/harness/fontaine-session.sh (#21 P5 deadline stamp).

Runs the REAL driver in an isolated HOME + repo with a fake `claude`
binary that dumps its `-p` argument, then asserts the prompt that a
session actually receives ends with the owner-signed deadline stamp
(session start + hard-kill budget). A regression here means sessions
are back to guessing their own wall-clock — the class P5 closed.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

FONTAINE = Path(__file__).resolve().parents[1] / "fontaine"

FAKE_CLAUDE = """#!/usr/bin/env bash
while [ $# -gt 0 ]; do
  if [ "$1" = "-p" ]; then printf '%s' "$2" > "$PROMPT_DUMP"; shift 2; else shift; fi
done
exit 0
"""


def _run_driver(tmp_path: Path, mode: str) -> str:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    (home / ".local" / "bin").mkdir(parents=True)
    (repo / "fontaine").mkdir(parents=True)
    shutil.copytree(FONTAINE / "prompts", repo / "fontaine" / "prompts")
    shutil.copytree(FONTAINE / "harness", repo / "fontaine" / "harness")
    shutil.rmtree(repo / "fontaine" / "harness" / "state", ignore_errors=True)
    shutil.rmtree(repo / "fontaine" / "harness" / "logs", ignore_errors=True)
    # GIT_* scrubbed for the same reason as test_refresh_ctrl._env:
    # under a pre-commit hook in a linked worktree, an inherited
    # ABSOLUTE GIT_DIR makes this init re-initialize the REAL repo.
    subprocess.run(
        ["git", "-C", str(repo), "init", "-q"],
        check=True,
        env={k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
    )
    fake = home / ".local" / "bin" / "claude"
    fake.write_text(FAKE_CLAUDE)
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    dump = tmp_path / "prompt_dump.txt"
    result = subprocess.run(
        ["bash", str(repo / "fontaine" / "harness" / "fontaine-session.sh"), mode],
        env={
            "HOME": str(home),
            "FONTAINE_REPO": str(repo),
            "PROMPT_DUMP": str(dump),
            "PATH": "/usr/bin:/bin",
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,  # returncode asserted below with stderr in the message
    )
    assert result.returncode == 0, result.stderr
    return dump.read_text()


@pytest.mark.parametrize(("mode", "minutes"), [("tick", 30), ("work", 240)])
def test_prompt_ends_with_deadline_stamp(
    tmp_path: Path,
    mode: str,
    minutes: int,
) -> None:
    prompt = _run_driver(tmp_path, mode)
    body = (FONTAINE / "prompts" / f"{mode}.md").read_text()
    assert prompt.startswith(body)
    stamp = re.search(
        r"\nSession start: \d{2}:\d{2}:\d{2}Z; hard kill in (\d+) min\.\n"
        r"Commit and push state comfortably before the deadline\.$",
        prompt,
    )
    assert stamp is not None, (
        f"deadline stamp missing from {mode} prompt tail: {prompt[-200:]!r}"
    )
    assert int(stamp.group(1)) == minutes
