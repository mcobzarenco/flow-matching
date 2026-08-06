#!/usr/bin/env python3
"""Roll aged now.md entries into per-day archive pages (#21).

now.md is the session-boot state file: every boot reads its head, and
only the newest few entries carry live state. Entries below that are
history — they belong on dated archive pages, not in the boot read.

Contract:
- now.md = "# Now" header + entries. Entry 1 starts "*Updated <date>";
  entries 2..n start "*Previous update <date>" (a blank line separates
  entries). Text is preserved verbatim.
- Keep the newest KEEP entries in now.md; move the rest to
  fontaine/blog/src/archive/now-YYYY-MM-DD.md keyed by each entry's
  own date, newest-first within a page. Re-runs append above existing
  archived entries (later batches are always newer), so the page stays
  newest-first without parsing dates beyond the day.
- A pointer line under "# Now" links the archive; SUMMARY.md gains a
  "Now archive" section (idempotent, one line per page).

Run: uv run python fontaine/scripts/archive_now.py [--keep N] [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "blog" / "src"
NOW = SRC / "now.md"
ARCHIVE_DIR = SRC / "archive"
SUMMARY = SRC / "SUMMARY.md"

ENTRY_RE = re.compile(r"^\*(?:Updated|Previous update)\b")
ENTRY_DATE_RE = re.compile(
    r"^\*(?:Updated|Previous update) (\d{4}-\d{2}-\d{2})",
)
POINTER_RE = re.compile(r"^\*Older entries: see the \[now archive\]")

PAGE_HEADER = """# Now archive — {date}

*Aged entries rolled out of [now.md](../now.md) verbatim (newest
first). The head of now.md is the live state; this page is history.*
"""


def split_entries(text: str) -> tuple[str, list[tuple[str, str]], str]:
    """Return (header, [(date, entry_text), ...], tail) in file order.

    The tail is everything from the first standing "## " section after
    the entries (e.g. "## Utilization footer") — live state that stays
    in now.md, never archived.
    """
    lines = text.splitlines(keepends=True)
    starts = [i for i, ln in enumerate(lines) if ENTRY_RE.match(ln)]
    if not starts:
        sys.exit("no entries found — refusing to touch now.md")
    header = "".join(ln for ln in lines[: starts[0]] if not POINTER_RE.match(ln))
    tail_start = next(
        (i for i, ln in enumerate(lines) if i > starts[0] and ln.startswith("## ")),
        len(lines),
    )
    tail = "".join(lines[tail_start:])
    lines = lines[:tail_start]
    starts = [i for i in starts if i < tail_start]
    entries = []
    date = None
    for a, b in zip(starts, [*starts[1:], len(lines)], strict=False):
        m = ENTRY_DATE_RE.match(lines[a])
        # a few early entries predate the dated-marker convention;
        # they inherit the date of the newer entry above them
        if m is not None:
            date = m.group(1)
        if date is None:
            sys.exit(f"first entry has no date (line {a + 1}) — refusing")
        entries.append((date, "".join(lines[a:b]).rstrip("\n") + "\n"))
    return header, entries, tail


def demote(entry: str) -> str:
    """Head entries archive as 'Previous update' like every other."""
    return entry.replace("*Updated ", "*Previous update ", 1)


def append_to_page(date: str, new_entries: list[str], *, dry: bool) -> Path:
    page = ARCHIVE_DIR / f"now-{date}.md"
    block = "\n".join(demote(e) for e in new_entries)
    if page.exists():
        text = page.read_text()
        # newer batch goes directly under the page intro, above old ones
        m = re.search(r"^\*Previous update ", text, re.MULTILINE)
        pos = m.start() if m else len(text)
        out = text[:pos] + block + "\n" + text[pos:]
    else:
        out = PAGE_HEADER.format(date=date) + "\n" + block
    if not dry:
        ARCHIVE_DIR.mkdir(exist_ok=True)
        page.write_text(out)
    return page


def update_summary(pages: list[Path], *, dry: bool) -> None:
    text = SUMMARY.read_text()
    lines = [
        f"  - [{p.stem.removeprefix('now-')}](archive/{p.name})"
        for p in sorted(pages, reverse=True)
    ]
    if "- [Now archive]" not in text:
        text = text.replace(
            "- [Now](now.md)",
            "- [Now](now.md)\n- [Now archive](archive/index.md)",
        )
        if not dry:
            (ARCHIVE_DIR / "index.md").write_text(
                "# Now archive\n\nDated pages of aged now.md entries; "
                "see the sidebar.\n",
            )
    for ln in lines:
        if ln not in text:
            text = text.replace(
                "- [Now archive](archive/index.md)",
                "- [Now archive](archive/index.md)\n" + ln,
            )
    if not dry:
        SUMMARY.write_text(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.keep < 1:
        sys.exit("--keep must be >= 1 (now.md must keep a live head)")

    text = NOW.read_text()
    header, entries, tail = split_entries(text)
    kept, aged = entries[: args.keep], entries[args.keep :]
    if not aged:
        print(f"nothing to archive ({len(entries)} entries <= keep)")
        return 0

    # group aged entries by date, preserving newest-first file order
    by_date: dict[str, list[str]] = {}
    for date, entry in aged:
        by_date.setdefault(date, []).append(entry)

    pages = [
        append_to_page(date, chunk, dry=args.dry_run) for date, chunk in by_date.items()
    ]
    update_summary(pages, dry=args.dry_run)

    pointer = (
        "*Older entries: see the [now archive](archive/index.md) — "
        "one dated page per day, verbatim.*\n\n"
    )
    out = header + pointer + "\n".join(e for _, e in kept)
    if tail:
        out = out.rstrip("\n") + "\n\n" + tail
    if not args.dry_run:
        NOW.write_text(out)

    total = sum(len(v) for v in by_date.values())
    print(
        f"kept {len(kept)}, archived {total} across "
        f"{len(by_date)} page(s): "
        + ", ".join(p.name for p in pages)
        + (" [dry-run]" if args.dry_run else ""),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
