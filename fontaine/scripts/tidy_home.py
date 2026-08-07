#!/usr/bin/env python3
"""Sweep loose files from $HOME into a dated attic (#21 P7).

The launch convention scp's launcher scripts to ~ and tees their
console logs there (charter §5 — load-bearing for tmux ergonomics);
nothing is lost, but grep noise grows without bound (59 local / 133
box entries at review time). This mover keeps the convention and
reclaims ~: loose top-level FILES move to ~/attic/<YYYY-MM-DD>/ with
a manifest row each; nothing is ever deleted.

Never touched:
- directories and symlinks (datasets/, checkpoints/, repos, ...)
- dotfiles
- files open by ANY process (/proc fd scan — a live tee target or a
  log babysit.toml is reading stays put)
- files modified within --min-age-days (default 2 — a live run's
  recent artifacts stay put even between fd writes)
- names given via --keep

stdlib-only on purpose: the box copy runs with bare python3 (the box
repo may be mid-run and unpullable — scp this file over and run it).

Run: python3 tidy_home.py [--apply] [--home H] [--min-age-days N]
     [--keep NAME ...]        (default is a dry run)
"""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

MANIFEST = "manifest.tsv"  # utc \t src \t dst \t bytes \t mtime_utc


def open_paths() -> set[str]:
    """Absolute paths of every file open by any visible process."""
    out: set[str] = set()
    proc = Path("/proc")
    for pid in proc.iterdir():
        if not pid.name.isdigit():
            continue
        try:
            for fd in (pid / "fd").iterdir():
                try:
                    out.add(str(fd.readlink()))
                except OSError:
                    continue
        except OSError:  # process gone or not ours
            continue
    return out


def plan(
    home: Path,
    min_age_days: float,
    keep: set[str],
    now: float | None = None,
) -> list[tuple[str, Path, str]]:
    """Return [(action, path, reason)] for every loose entry in home.

    action is "move" or "skip"; pure planning, no filesystem writes.
    """
    now = time.time() if now is None else now
    busy = open_paths()
    actions: list[tuple[str, Path, str]] = []
    for entry in sorted(home.iterdir()):
        if entry.name.startswith("."):
            continue  # dotfiles are config, not sprawl
        if entry.name in ("attic", "logs") or entry.name in keep:
            actions.append(("skip", entry, "keeplist"))
        elif entry.is_symlink() or entry.is_dir():
            actions.append(("skip", entry, "directory/symlink"))
        elif str(entry) in busy:
            actions.append(("skip", entry, "open by a process"))
        elif now - entry.stat().st_mtime < min_age_days * 86400:
            actions.append(("skip", entry, f"younger than {min_age_days}d"))
        else:
            actions.append(("move", entry, ""))
    return actions


def sweep(
    home: Path,
    *,
    apply: bool,
    min_age_days: float,
    keep: set[str],
    now: float | None = None,
) -> int:
    actions = plan(home, min_age_days, keep, now)
    day = datetime.now(UTC).strftime("%Y-%m-%d")
    attic = home / "attic" / day
    moved = 0
    for action, path, reason in actions:
        if action == "skip":
            print(f"skip  {path.name}  ({reason})")
            continue
        dst = attic / path.name
        n = 0
        while dst.exists():  # same name swept twice in one day
            n += 1
            dst = attic / f"{path.name}.dup{n}"
        if apply:
            attic.mkdir(parents=True, exist_ok=True)
            stat = path.stat()
            path.rename(dst)
            with (home / "attic" / MANIFEST).open("a") as f:
                f.write(
                    "\t".join(
                        (
                            datetime.now(UTC).isoformat(timespec="seconds"),
                            str(path),
                            str(dst),
                            str(stat.st_size),
                            datetime.fromtimestamp(
                                stat.st_mtime,
                                UTC,
                            ).isoformat(timespec="seconds"),
                        ),
                    )
                    + "\n",
                )
        print(f"{'move' if apply else 'would'}  {path.name} -> {dst}")
        moved += 1
    print(
        f"{'moved' if apply else 'would move'} {moved} of {len(actions)} "
        f"loose entries{'' if apply else '  (dry run; pass --apply)'}",
    )
    return moved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--apply",
        action="store_true",
        help="actually move (default: dry run)",
    )
    ap.add_argument("--home", type=Path, default=Path.home())
    ap.add_argument("--min-age-days", type=float, default=2.0)
    ap.add_argument(
        "--keep",
        nargs="*",
        default=[],
        help="extra names to leave in place",
    )
    args = ap.parse_args()
    sweep(
        args.home,
        apply=args.apply,
        min_age_days=args.min_age_days,
        keep=set(args.keep),
    )


if __name__ == "__main__":
    main()
