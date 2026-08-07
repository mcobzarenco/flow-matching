"""Oracles for fontaine/scripts/refresh_ctrl.sh (#21 P7 ctrl stamp).

Runs the REAL script against a throwaway git repo and asserts the
snapshot contract: tracked-files-only mirror of HEAD, prior snapshot
moved aside (never deleted), CTRL_SOURCE_COMMIT names the sha the
snapshot actually is. A regression here reopens the silent-drift
class: control evals citing code no one can identify.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1] / "fontaine" / "scripts" / "refresh_ctrl.sh"
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "src"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "code.py").write_text("v1\n")
    (repo / ".gitignore").write_text("outputs/\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c1")
    (repo / "outputs").mkdir()
    (repo / "outputs" / "run.log").write_text("gitignored\n")
    (repo / "untracked.txt").write_text("never committed\n")
    return repo


def _refresh(repo: Path, dest: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), str(repo), str(dest)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_snapshot_is_head_tracked_files_only(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    dest = tmp_path / "ctrl"
    _refresh(repo, dest)
    assert (dest / "code.py").read_text() == "v1\n"
    assert not (dest / ".git").exists()
    assert not (dest / "outputs").exists()
    assert not (dest / "untracked.txt").exists()
    sha, _utc, word, src = (dest / "CTRL_SOURCE_COMMIT").read_text().split()
    assert sha == _git(repo, "rev-parse", "HEAD")
    assert (word, src) == ("from", str(repo))


def test_prior_snapshot_moved_aside_and_stamp_updated(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    dest = tmp_path / "ctrl"
    _refresh(repo, dest)
    sha1 = (dest / "CTRL_SOURCE_COMMIT").read_text().split()[0]
    (dest / "outputs").mkdir()
    (dest / "outputs" / "eval.json").write_text("precious\n")
    (repo / "code.py").write_text("v2\n")
    _git(repo, "commit", "-qam", "c2")
    _refresh(repo, dest)
    sha2 = (dest / "CTRL_SOURCE_COMMIT").read_text().split()[0]
    assert sha2 == _git(repo, "rev-parse", "HEAD") != sha1
    assert (dest / "code.py").read_text() == "v2\n"
    assert not (dest / "outputs").exists()  # fresh snapshot, no carryover
    prev = list(tmp_path.glob("ctrl.prev-*"))
    assert len(prev) == 1  # moved aside, not deleted
    assert (prev[0] / "outputs" / "eval.json").read_text() == "precious\n"


def test_dirty_tree_snapshots_head_and_warns(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / "code.py").write_text("dirty edit\n")
    dest = tmp_path / "ctrl"
    out = _refresh(repo, dest).stdout
    assert (dest / "code.py").read_text() == "v1\n"  # HEAD, not the edit
    assert "dirty" in out and "code.py" in out
