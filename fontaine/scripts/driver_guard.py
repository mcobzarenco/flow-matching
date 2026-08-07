"""Post-session straggler guard (driver-background-task-guard, owner
steering 08-07 13:05Z; 3 incidents 2026-08-07).

Runs at the END of fontaine-session.sh, after the last session of the
invocation: scans the driver's OWN cgroup for processes that are not
part of the driver's process tree. Any such process was launched by a
session as a plain child / setsid job and — before the KillMode=process
fix — would have been silently SIGKILLed moments later when the unit's
cgroup was torn down (that is exactly how incidents 2 and 3 killed live
GPU evals). With KillMode=process the stragglers now SURVIVE unit stop,
but they are still noncompliant launches (a poll-safe job lives in its
own transient unit via run_detached.sh), so the driver posts a loud
alert naming them instead of exiting silently.

Stdlib-only on purpose (same rule as harness/discord.py): the guard
must work even when the repo venv is broken. Exit 0 = cgroup clean,
1 = stragglers found (driver alerts), 2 = cannot scan (no cgroup v2 /
weird environment — never blocks the driver).

Oracles: tests/test_driver_guard.py (fake /proc + cgroup trees, plus a
systemd-run integration test reproducing the incident-3 kill signature
and the KillMode=process survival).
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

TRUNCATE_CMDLINE = 160


def own_cgroup_dir(pid: int, proc: Path, cgroup_root: Path) -> Path | None:
    """The cgroup-v2 directory of `pid` (None if unreadable / not v2)."""
    try:
        text = (proc / str(pid) / "cgroup").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        # cgroup v2: a single "0::<path>" line.
        if line.startswith("0::"):
            rel = line[3:].strip().lstrip("/")
            return cgroup_root / rel
    return None


def cgroup_pids(cgroup_dir: Path) -> list[int]:
    try:
        text = (cgroup_dir / "cgroup.procs").read_text()
    except OSError:
        return []
    return [int(tok) for tok in text.split() if tok.isdigit()]


def ppid_of(pid: int, proc: Path) -> int | None:
    try:
        text = (proc / str(pid) / "status").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("PPid:"):
            return int(line.split()[1])
    return None


def in_driver_tree(pid: int, driver_pid: int, proc: Path) -> bool:
    """Walk the PPid chain; True iff it reaches driver_pid."""
    seen: set[int] = set()
    cur: int | None = pid
    while cur is not None and cur > 1 and cur not in seen:
        if cur == driver_pid:
            return True
        seen.add(cur)
        cur = ppid_of(cur, proc)
    return False


def cmdline_of(pid: int, proc: Path) -> str:
    try:
        raw = (proc / str(pid) / "cmdline").read_bytes()
    except OSError:
        return "(gone)"
    line = raw.replace(b"\x00", b" ").decode(errors="replace").strip()
    if len(line) > TRUNCATE_CMDLINE:
        line = line[: TRUNCATE_CMDLINE - 1] + "…"
    return line or "(empty cmdline)"


def registered_patterns(registry: Path) -> list[str]:
    """pgrep patterns of registered LIVE runs (any host — patterns are
    substrings of cmdlines; a local match against a box-run pattern is
    vanishingly unlikely and still worth flagging)."""
    try:
        with registry.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return []
    return [r["pgrep"] for r in data.get("run", []) if r.get("pgrep")]


def scan(
    driver_pid: int,
    proc: Path,
    cgroup_root: Path,
    registry: Path,
) -> tuple[int, list[str]]:
    """(exit_code, report_lines)."""
    cgroup_dir = own_cgroup_dir(driver_pid, proc, cgroup_root)
    if cgroup_dir is None or not cgroup_dir.is_dir():
        return 2, [f"driver_guard: cannot resolve cgroup of pid {driver_pid}; skipping"]
    patterns = registered_patterns(registry)
    stragglers: list[str] = []
    for pid in cgroup_pids(cgroup_dir):
        if pid == driver_pid or in_driver_tree(pid, driver_pid, proc):
            continue
        # Re-check existence: short-lived pids may exit mid-scan.
        if not (proc / str(pid)).is_dir():
            continue
        cmd = cmdline_of(pid, proc)
        tag = "REGISTERED-RUN " if any(p in cmd for p in patterns) else ""
        stragglers.append(f"  {tag}pid {pid}: {cmd}")
    if not stragglers:
        return 0, [f"driver_guard: cgroup clean ({cgroup_dir.name})"]
    lines = [
        (
            f"driver_guard: {len(stragglers)} straggler(s) in {cgroup_dir.name} — "
            "launched as session children, NOT via run_detached.sh "
            "(they survive unit stop only because of KillMode=process):"
        ),
        *stragglers,
        "driver_guard: relaunch long jobs via fontaine/scripts/run_detached.sh",
    ]
    return 1, lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--driver-pid", type=int, required=True)
    ap.add_argument("--proc", type=Path, default=Path("/proc"))
    ap.add_argument("--cgroup-root", type=Path, default=Path("/sys/fs/cgroup"))
    ap.add_argument(
        "--registry",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "harness" / "babysit.toml",
    )
    args = ap.parse_args()
    code, lines = scan(args.driver_pid, args.proc, args.cgroup_root, args.registry)
    print("\n".join(lines))
    return code


if __name__ == "__main__":
    sys.exit(main())
