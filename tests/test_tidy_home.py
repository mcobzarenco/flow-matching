"""Oracles for fontaine/scripts/tidy_home.py (#21 P7 home sweep).

The sweep's one hard promise: it NEVER touches anything a live run
depends on — open files (a tee target babysit.toml reads), recent
files, directories — and it never deletes (move + manifest only).
Everything here runs against a throwaway home dir with the REAL
plan/sweep functions.
"""

from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "fontaine" / "scripts" / "tidy_home.py"
spec = importlib.util.spec_from_file_location("tidy_home", SCRIPT)
assert spec is not None and spec.loader is not None
tidy_home = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tidy_home)
MANIFEST, plan, sweep = tidy_home.MANIFEST, tidy_home.plan, tidy_home.sweep


def _old(p: Path, days: float = 5.0) -> Path:
    p.write_text("x")
    past = time.time() - days * 86400
    os.utime(p, (past, past))
    return p


def _actions_by_name(
    actions: list[tuple[str, Path, str]],
) -> dict[str, tuple[str, str]]:
    return {p.name: (a, r) for a, p, r in actions}


def test_plan_classification(tmp_path: Path) -> None:
    _old(tmp_path / "stale.log")
    (tmp_path / "fresh.log").write_text("x")  # mtime now
    (tmp_path / ".dotfile").write_text("x")
    (tmp_path / "somedir").mkdir()
    _old(tmp_path / "keepme.sh")
    held = _old(tmp_path / "held.log")
    with held.open() as fh:  # our own fd shows in the /proc scan
        got = _actions_by_name(plan(tmp_path, 2.0, {"keepme.sh"}))
        assert got["stale.log"][0] == "move"
        assert got["fresh.log"] == ("skip", "younger than 2.0d")
        assert got["held.log"] == ("skip", "open by a process")
        assert got["keepme.sh"] == ("skip", "keeplist")
        assert got["somedir"][0] == "skip"
        assert ".dotfile" not in got  # dotfiles not even listed
        del fh


def test_dry_run_moves_nothing(tmp_path: Path) -> None:
    p = _old(tmp_path / "stale.log")
    n = sweep(tmp_path, apply=False, min_age_days=2.0, keep=set())
    assert n == 1
    assert p.exists()
    assert not (tmp_path / "attic").exists()


def test_apply_moves_with_manifest_and_is_idempotent(tmp_path: Path) -> None:
    p = _old(tmp_path / "stale.log")
    assert sweep(tmp_path, apply=True, min_age_days=2.0, keep=set()) == 1
    assert not p.exists()
    moved = list((tmp_path / "attic").glob("*/stale.log"))
    assert len(moved) == 1 and moved[0].read_text() == "x"
    rows = (tmp_path / "attic" / MANIFEST).read_text().splitlines()
    assert len(rows) == 1
    _utc, src, dst, size, _mtime = rows[0].split("\t")
    assert src == str(p) and dst == str(moved[0]) and size == "1"
    # second sweep: attic/ itself is keeplisted, nothing loose remains
    assert sweep(tmp_path, apply=True, min_age_days=2.0, keep=set()) == 0


def test_name_collision_gets_dup_suffix(tmp_path: Path) -> None:
    sweep_dir = tmp_path  # same name swept twice in one day
    _old(sweep_dir / "a.log")
    sweep(sweep_dir, apply=True, min_age_days=2.0, keep=set())
    _old(sweep_dir / "a.log")
    sweep(sweep_dir, apply=True, min_age_days=2.0, keep=set())
    day_dir = next((tmp_path / "attic").glob("2*"))
    assert sorted(f.name for f in day_dir.iterdir()) == ["a.log", "a.log.dup1"]
