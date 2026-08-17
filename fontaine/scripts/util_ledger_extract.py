"""Extract dated utilization-footer session notes + their GPU-h figures.

One-shot instrument for the utilization-ledger-rebase queue item
(2026-08-17): walks the now-archive pages plus now.md, pulls every
"Session 2026-08-DD HH:MM..." footer note, and lists the GPU-h figures
each note states, so the trailing-7-day rebase sums per-session accruals
instead of hand-summing 550 prose mentions. Audit output, not a ledger:
riding sessions say "accruing" with the run total landing in a later
note, so large runs still get reconciled against babysit.toml prunes.
"""

import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "blog" / "src"
PAGES = [*sorted(SRC.glob("archive/now-2026-08-1[0-7].md")), SRC / "now.md"]

NOTE_RE = re.compile(r"^Session (2026-08-\d\d) (\S+)")
# figures like "+~0.88 GPU-h", "~2.6 GPU-h", "17.8/40 GPU-h", "+~12.7 GPU-h"
GPUH_RE = re.compile(r"([+≈~]*\s*\d+(?:\.\d+)?)\s*(?:/\s*(\d+(?:\.\d+)?))?\s*GPU-h")

for page in PAGES:
    lines = page.read_text().splitlines()
    # collect note blocks: start at "Session 2026-", end at blank line
    i = 0
    while i < len(lines):
        m = NOTE_RE.match(lines[i])
        if not m:
            i += 1
            continue
        block = []
        while i < len(lines) and lines[i].strip():
            block.append(lines[i].strip())
            i += 1
        text = " ".join(block)
        date, span = m.group(1), m.group(2)
        figs = GPUH_RE.findall(text)
        accruing = "accruing" in text
        zero = "0 new GPU-h" in text
        figstr = ", ".join(f"{a.strip()}{'/' + b if b else ''}" for a, b in figs)
        flag = "ACCRUING" if accruing else ("zero" if zero else "")
        head = text[:118]
        print(f"{page.name:22s} {date} {span:22s} [{figstr or '-'}] {flag}")
        print(f"    {head}")
    # ignore non-note GPU-h mentions (Status blocks etc.) by design
print("pages scanned:", len(PAGES), file=sys.stderr)
