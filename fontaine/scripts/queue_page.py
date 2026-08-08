"""Render the Queue page (owner steering 2026-08-08 16:37Z): a
vertical-board view of ``fontaine/queue.json`` at
``fontaine/blog/src/queue.md`` — readable at a glance, generated, never
hand-edited. ``queue.json`` stays canonical; this page is a VIEW.

Board order: LIVE → QUEUED → BLOCKED → DONE (file order reversed inside
each lane, so the most recently touched items sit on top). Each card
shows the item id, class chip, hold flag, a first-clause summary, its
boundary, and the pre-reg link when one exists; the full running record
(queue titles are append-only logs) sits in a fold.

Standard step: run this before every ``mdbook build`` —
``fontaine/scripts/blog_build.sh`` does both (charter session-close
step). ``--check`` exits 1 if the committed page is stale (a cheap
freshness guard for check-time use).

  uv run python fontaine/scripts/queue_page.py          # render
  uv run python fontaine/scripts/queue_page.py --check  # stale => exit 1
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
QUEUE = REPO / "fontaine/queue.json"
PAGE = REPO / "fontaine/blog/src/queue.md"

LANES = [
    ("live", "🔴 Live", "running right now (GPU or owner-window)"),
    ("queued", "🟢 Queued", "ready — waiting on a window or a boundary"),
    ("blocked", "🟡 Blocked", "waiting on a prerequisite, a boundary, or the owner"),
    ("done", "✅ Done", "closed — the full record stays in each fold"),
]
CLASS_CHIP = {
    "cpu": "`cpu`",
    "gpu-local": "`gpu-local`",
    "gpu-box": "`gpu-box`",
    "gpu": "`gpu`",
}


def summary_of(title: str, limit: int = 230) -> str:
    """First clause of the running-log title: cut at the first ' — ',
    ' | ', or sentence end past 60 chars, hard-wrap at ``limit``."""
    cut = len(title)
    for sep in (" | ", " — ", ". "):
        pos = title.find(sep, 60)
        if pos != -1:
            cut = min(cut, pos)
    clause = title[:cut].strip().rstrip(".")
    if len(clause) > limit:
        clause = clause[: limit - 1].rstrip() + "…"
    return clause


def prereg_link(prereg: str | None) -> str | None:
    if not prereg:
        return None
    match = re.search(r"fontaine/blog/src/(.+\.md)$", prereg)
    if match:
        return f"[pre-reg]({match.group(1)})"
    return f"`{prereg}`"


def esc(text: str) -> str:
    """Item text is data, never markup: titles carry literal '<author>'
    style angle brackets that mdbook would otherwise parse as HTML
    (and silently swallow the rest of the paragraph)."""
    return html.escape(text, quote=False)


def card(item: dict) -> str:
    chip = CLASS_CHIP.get(str(item.get("class")), f"`{item.get('class')}`")
    hold = " · **⛔ owner hold**" if item.get("owner_hold") else ""
    lines = [f"**`{item['id']}`** · {chip}{hold}"]
    lines.append(f"\n{esc(summary_of(str(item['title'])))}")
    meta = []
    if item.get("boundary"):
        meta.append(f"**boundary:** {esc(str(item['boundary']))}")
    link = prereg_link(item.get("prereg"))
    if link:
        meta.append(link)
    if meta:
        lines.append("\n" + " · ".join(meta))
    lines.append(
        "\n<details><summary>full record</summary>\n\n"
        f"{esc(str(item['title']))}\n\n</details>",
    )
    return "\n".join(lines)


def render() -> str:
    queue = json.loads(QUEUE.read_text())
    by_status: dict[str, list[dict]] = {key: [] for key, _, _ in LANES}
    for item in queue["items"]:
        by_status.setdefault(str(item.get("status")), []).append(item)

    parts = [
        "# Queue",
        "",
        (
            "*Generated from [`fontaine/queue.json`](https://github.com/"
            "mcobzarenco/flow-matching/blob/fontaine/fontaine/queue.json) — "
            "the canonical queue — by `fontaine/scripts/queue_page.py` "
            "(rides every `blog_build.sh`). Do not hand-edit.*"
        ),
        "",
        f"**Updated:** {queue['updated_utc']}",
        "",
        f"**Depth call:** {queue['depth_reason']}",
        "",
    ]
    open_count = sum(len(by_status.get(k, [])) for k in ("live", "queued", "blocked"))
    counts = " · ".join(
        f"{title.split(' ', 1)[1]} {len(by_status.get(key, []))}"
        for key, title, _ in LANES
    )
    parts += [f"**{open_count} open** ({counts})", ""]

    for key, title, hint in LANES:
        items = by_status.get(key, [])
        parts += [f"## {title} ({len(items)})", "", f"*{hint}*", ""]
        if not items:
            parts += ["*(empty)*", ""]
            continue
        for item in reversed(items):
            parts += [card(item), "", "---", ""]
    # Anything with an unknown status must surface, never vanish.
    known = {key for key, _, _ in LANES}
    strays = [i for i in queue["items"] if str(i.get("status")) not in known]
    if strays:
        parts += ["## ⚠️ Unknown status", ""]
        for item in strays:
            parts += [f"- `{item['id']}`: status `{item.get('status')}`", ""]
    return "\n".join(parts).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the committed page is stale instead of writing",
    )
    args = parser.parse_args()
    content = render()
    if args.check:
        if not PAGE.exists() or PAGE.read_text() != content:
            raise SystemExit("queue.md is STALE — run queue_page.py (no --check)")
        print("queue.md is fresh")
        return
    PAGE.write_text(content)
    print(f"wrote {PAGE.relative_to(REPO)}")


if __name__ == "__main__":
    main()
