"""Oracles for the driver-background-task-guard fix (3 incidents
2026-08-07: GPU evals silently killed at driver-session end).

Three layers under test:
- fontaine/scripts/driver_guard.py — post-session cgroup straggler scan
  (pure logic against fake /proc + cgroup trees);
- fontaine/harness/systemd/fontaine-tick.service — KillMode=process is
  the regression-guarded safety net (default control-group teardown is
  what SIGKILLed the setsid-detached tsens eval at 15:56:18Z);
- the mechanism itself, reproduced live with transient systemd units:
  a setsid child DIES with a default-KillMode unit (the incident-3
  signature) and SURVIVES with KillMode=process; run_detached.sh puts a
  job in its own unit that outlives the parent unit entirely.

The live tests skip cleanly where no systemd --user manager is
reachable (CI containers).
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "fontaine" / "scripts" / "driver_guard.py"
UNIT_FILE = REPO / "fontaine" / "harness" / "systemd" / "fontaine-tick.service"
RUN_DETACHED = REPO / "fontaine" / "scripts" / "run_detached.sh"

spec = importlib.util.spec_from_file_location("driver_guard", SCRIPT)
assert spec is not None and spec.loader is not None
driver_guard = importlib.util.module_from_spec(spec)
sys.modules["driver_guard"] = driver_guard
spec.loader.exec_module(driver_guard)


# ------------------------------------------------------------ fake trees


class FakeWorld:
    """Builds a fake /proc + cgroup-v2 root under tmp_path."""

    def __init__(self, root: Path) -> None:
        self.proc = root / "proc"
        self.cgroup = root / "cgroup"
        self.unit_dir = self.cgroup / "user.slice" / "fontaine-tick.service"
        self.unit_dir.mkdir(parents=True)
        self.pids: list[int] = []

    def add_proc(self, pid: int, ppid: int, cmdline: str, cgroup_path: str) -> None:
        d = self.proc / str(pid)
        d.mkdir(parents=True)
        (d / "status").write_text(f"Name:\tx\nPPid:\t{ppid}\n")
        (d / "cmdline").write_bytes(cmdline.replace(" ", "\x00").encode())
        (d / "cgroup").write_text(f"0::{cgroup_path}\n")
        self.pids.append(pid)

    def write_cgroup_procs(self) -> None:
        (self.unit_dir / "cgroup.procs").write_text(
            "".join(f"{p}\n" for p in self.pids),
        )


UNIT_PATH = "/user.slice/fontaine-tick.service"


def make_registry(path: Path) -> Path:
    reg = path / "babysit.toml"
    reg.write_text(
        '[[run]]\nname = "r"\nkind = "progress-log"\nhost = "local"\n'
        'pgrep = "stateprobe_q4_draws10_t"\npgrep_min = 2\ngpu_indices = [0]\n'
        'gpu_mem_min_mib = 1\nanchors = []\nboundary = "-"\n'
        'log = "-"\nprogress_re = "x"\nstarted_utc = "2026-08-07T00:00:00Z"\n',
    )
    return reg


class TestScan:
    def test_clean_cgroup(self, tmp_path: Path) -> None:
        w = FakeWorld(tmp_path)
        w.add_proc(100, 1, "bash fontaine-session.sh tick", UNIT_PATH)
        w.add_proc(101, 100, "claude -p", UNIT_PATH)
        w.add_proc(102, 101, "python3 driver_guard.py", UNIT_PATH)
        w.write_cgroup_procs()
        code, lines = driver_guard.scan(100, w.proc, w.cgroup, make_registry(tmp_path))
        assert code == 0
        assert "clean" in "\n".join(lines)

    def test_straggler_detected_and_registered_tagged(self, tmp_path: Path) -> None:
        w = FakeWorld(tmp_path)
        w.add_proc(100, 1, "bash fontaine-session.sh tick", UNIT_PATH)
        # setsid job: reparented to 1, NOT in the driver tree, same cgroup
        # — the exact incident-3 shape (would have died at unit stop).
        w.add_proc(
            200,
            1,
            "python -m bijou.eval --output stateprobe_q4_draws10_t0.5.json",
            UNIT_PATH,
        )
        w.add_proc(201, 1, "sleep 999", UNIT_PATH)
        w.write_cgroup_procs()
        code, lines = driver_guard.scan(100, w.proc, w.cgroup, make_registry(tmp_path))
        assert code == 1
        text = "\n".join(lines)
        assert "2 straggler(s)" in text
        assert "REGISTERED-RUN pid 200" in text
        assert "pid 201" in text and "REGISTERED-RUN pid 201" not in text
        assert "run_detached.sh" in text

    def test_driver_subtree_never_flagged(self, tmp_path: Path) -> None:
        # deep child chain (session -> bash tool -> uv -> python) is ours
        w = FakeWorld(tmp_path)
        w.add_proc(100, 1, "bash fontaine-session.sh work", UNIT_PATH)
        w.add_proc(110, 100, "claude -p", UNIT_PATH)
        w.add_proc(120, 110, "bash -c something", UNIT_PATH)
        w.add_proc(130, 120, "uv run python x.py", UNIT_PATH)
        w.write_cgroup_procs()
        code, _ = driver_guard.scan(100, w.proc, w.cgroup, make_registry(tmp_path))
        assert code == 0

    def test_unresolvable_cgroup_skips(self, tmp_path: Path) -> None:
        w = FakeWorld(tmp_path)
        code, lines = driver_guard.scan(999, w.proc, w.cgroup, make_registry(tmp_path))
        assert code == 2
        assert "skipping" in "\n".join(lines)

    def test_vanished_pid_ignored(self, tmp_path: Path) -> None:
        w = FakeWorld(tmp_path)
        w.add_proc(100, 1, "bash fontaine-session.sh tick", UNIT_PATH)
        w.pids.append(300)  # listed in cgroup.procs, no /proc entry
        w.write_cgroup_procs()
        code, _ = driver_guard.scan(100, w.proc, w.cgroup, make_registry(tmp_path))
        assert code == 0


class TestUnitFile:
    def test_killmode_process_is_set(self) -> None:
        # Regression guard: dropping this line re-arms the incident-3
        # mechanism (cgroup-wide SIGKILL at oneshot completion).
        text = UNIT_FILE.read_text()
        assert "KillMode=process" in text

    def test_installed_unit_matches_repo(self) -> None:
        installed = Path.home() / ".config" / "systemd" / "user" / UNIT_FILE.name
        if not installed.exists():
            pytest.skip("unit not installed on this machine")
        assert installed.read_text() == UNIT_FILE.read_text()


# ------------------------------------------------------------ live systemd


def _user_systemd_available() -> bool:
    if shutil.which("systemd-run") is None:
        return False
    probe = subprocess.run(
        ["systemctl", "--user", "show-environment"],
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


needs_systemd = pytest.mark.skipif(
    not _user_systemd_available(),
    reason="no systemd --user manager reachable",
)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _spawn_setsid_child_in_unit(kill_mode: str | None, pid_file: Path) -> int:
    """Run a transient unit whose main process setsid-spawns `sleep 60`
    and exits; returns the sleep's pid once the unit has finished."""
    name = f"fontaine-test-{uuid.uuid4().hex[:8]}.service"
    cmd = ["systemd-run", "--user", f"--unit={name}", "--collect", "--wait"]
    if kill_mode is not None:
        cmd += ["-p", f"KillMode={kill_mode}"]
    cmd += ["bash", "-c", f"setsid sleep 60 & echo $! > {pid_file}"]
    subprocess.run(cmd, check=True, capture_output=True, timeout=30)
    return int(pid_file.read_text().strip())


@needs_systemd
class TestLiveMechanism:
    def test_default_killmode_kills_setsid_child(self, tmp_path: Path) -> None:
        # Incident-3 reproduction: setsid escapes the session, not the
        # cgroup — the child is SIGKILLed when the oneshot unit finishes.
        pid = _spawn_setsid_child_in_unit(None, tmp_path / "pid")
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and _pid_alive(pid):
            time.sleep(0.1)
        assert not _pid_alive(pid)

    def test_killmode_process_spares_setsid_child(self, tmp_path: Path) -> None:
        # The fontaine-tick.service fix: only the main process is
        # signalled at stop; the straggler survives (and driver_guard
        # makes it loud instead of silent).
        pid = _spawn_setsid_child_in_unit("process", tmp_path / "pid")
        try:
            time.sleep(1.0)
            assert _pid_alive(pid)
        finally:
            if _pid_alive(pid):
                os.kill(pid, 15)

    def test_run_detached_own_unit_survives_parent_unit(
        self,
        tmp_path: Path,
    ) -> None:
        # The compliant path: run_detached.sh from inside a dying unit
        # puts the job in its OWN unit — alive after the parent is gone.
        child_unit = f"fontaine-test-{uuid.uuid4().hex[:8]}"
        parent = f"fontaine-test-{uuid.uuid4().hex[:8]}.service"
        env = {**os.environ, "RUN_DETACHED_GRACE": "1"}
        subprocess.run(
            [
                "systemd-run",
                "--user",
                f"--unit={parent}",
                "--collect",
                "--wait",
                str(RUN_DETACHED),
                child_unit,
                "sleep",
                "60",
            ],
            check=True,
            capture_output=True,
            timeout=30,
            env=env,
        )
        try:
            active = subprocess.run(
                ["systemctl", "--user", "is-active", "--quiet", child_unit],
                check=False,
            )
            assert active.returncode == 0
        finally:
            subprocess.run(
                ["systemctl", "--user", "stop", child_unit],
                check=False,
                capture_output=True,
            )

    def test_run_detached_surfaces_fast_death(self, tmp_path: Path) -> None:
        # The exit-127 class: a launch that dies inside the grace window
        # must return failure, not silent success.
        env = {**os.environ, "RUN_DETACHED_GRACE": "1"}
        name = f"fontaine-test-{uuid.uuid4().hex[:8]}"
        proc = subprocess.run(
            [str(RUN_DETACHED), name, "bash", "-c", "exit 127"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=env,
        )
        assert proc.returncode == 1
        assert "LAUNCH FAILURE" in proc.stderr
