"""Queue CLI (#21 P2, owner-signed 2026-08-07): the queue as data.

`fontaine/queue.json` is the CANONICAL queue; `now.md` narrates it
(charter §3 bullet 1). Before P2 the queue lived as prose inside
now.md's head entry — every session re-narrated it, ticks eyeballed
"depth ≥ 2", and nothing machine-checked that a named GPU item
actually had a posted pre-reg. A mis-transcribed queue line in one
mega-paragraph silently became the next session's ground truth.

Schema per item: `id`, `title`, `class` (gpu-local / gpu-box / cpu),
`status` (queued / blocked / live / done), `prereg` (repo-relative
post path — may be null only for cpu items), `owner_hold` (bool),
`boundary` (free-text notes). Top level: `updated_utc`,
`depth_reason` (free text excusing depth < 2), `items` (priority
order: the first `queued` item IS the next pick).

Commands:
  list      all non-done items (id, class, status, hold, title)
  next      the first queued item — the session's default pick
  depth     queued count + open count
  validate  machine gate: schema, unique ids, gpu-* items must name an
            existing pre-reg post, owner_hold forces blocked, and
            queued-depth >= 2 unless depth_reason says why not.
            Exit 1 on any failure — ticks run this instead of
            eyeballing (`queue.py validate || refill`).

Like babysit.py this script only surfaces facts; re-prioritization
stays with the session (charter §6). Oracles: tests/test_queue.py.

Named `queue_cli.py`, not the review's proposed `queue.py`: several
sibling scripts `sys.path.insert` this directory, so a module named
`queue` shadows the stdlib and crashes any child that transitively
imports it (torch spawn workers died exactly this way in check.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
QUEUE = REPO / "fontaine" / "queue.json"

CLASSES = ("gpu-local", "gpu-box", "cpu")
STATUSES = ("queued", "blocked", "live", "done")
REQUIRED_KEYS = ("id", "title", "class", "status", "prereg", "owner_hold")
MIN_DEPTH = 2


def load_queue(path: Path = QUEUE) -> dict[str, Any]:
    with path.open("rb") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        msg = f"{path}: top level must be an object with an 'items' list"
        raise TypeError(msg)
    return data


def open_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [it for it in data["items"] if it.get("status") != "done"]


def queued_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [it for it in data["items"] if it.get("status") == "queued"]


def validate(data: dict[str, Any], repo: Path = REPO) -> list[str]:
    """Return a list of failure strings; empty means the queue is valid."""
    failures: list[str] = []
    seen_ids: set[str] = set()
    for it in data["items"]:
        item_id = it.get("id", "<missing id>")
        failures.extend(
            f"{item_id}: missing key '{key}'" for key in REQUIRED_KEYS if key not in it
        )
        if it.get("class") not in CLASSES:
            failures.append(f"{item_id}: class {it.get('class')!r} not in {CLASSES}")
        if it.get("status") not in STATUSES:
            failures.append(f"{item_id}: status {it.get('status')!r} not in {STATUSES}")
        if item_id in seen_ids:
            failures.append(f"{item_id}: duplicate id")
        seen_ids.add(item_id)
        prereg = it.get("prereg")
        if str(it.get("class", "")).startswith("gpu-"):
            if not prereg:
                failures.append(
                    f"{item_id}: {it.get('class')} item has no pre-reg post "
                    "(prereg may be null only for cpu items)",
                )
            elif not (repo / prereg).is_file():
                failures.append(f"{item_id}: prereg path not found: {prereg}")
        elif prereg and not (repo / prereg).is_file():
            failures.append(f"{item_id}: prereg path not found: {prereg}")
        if it.get("owner_hold") and it.get("status") == "queued":
            failures.append(
                f"{item_id}: owner_hold item cannot be 'queued' — it is not "
                "pickable until the owner releases it (use 'blocked')",
            )
    depth = len(queued_items(data))
    if depth < MIN_DEPTH and not data.get("depth_reason"):
        failures.append(
            f"queue depth {depth} < {MIN_DEPTH} and no depth_reason given "
            "(charter §3: refill or state why not)",
        )
    return failures


def fmt_item(it: dict[str, Any]) -> str:
    hold = " HOLD" if it.get("owner_hold") else ""
    line = f"[{it.get('status'):7s}] {it.get('class'):9s}{hold}  {it.get('id')}: {it.get('title')}"
    if it.get("boundary"):
        line += f"\n           boundary: {it['boundary']}"
    return line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("list", "next", "depth", "validate"))
    parser.add_argument("--queue", type=Path, default=QUEUE)
    args = parser.parse_args(argv)

    data = load_queue(args.queue)

    if args.command == "list":
        for it in open_items(data):
            print(fmt_item(it))
        return 0
    if args.command == "next":
        queued = queued_items(data)
        if not queued:
            print("queue: no queued items (all live/blocked/done)")
            return 1
        print(fmt_item(queued[0]))
        return 0
    if args.command == "depth":
        print(f"queued {len(queued_items(data))}, open {len(open_items(data))}")
        return 0
    # validate
    failures = validate(data)
    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        return 1
    print(
        f"queue OK: depth {len(queued_items(data))} (>= {MIN_DEPTH}), "
        f"{len(open_items(data))} open, updated {data.get('updated_utc')}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
