"""Read a BIJOU_MEM_SNAPSHOT pickle (torch.cuda.memory._dump_snapshot)
and attribute LIVE allocated bytes at dump time by allocation site.

Usage: python fontaine/scripts/mem_snapshot_read.py <snapshot.pickle> [top_n]

Two views:
  1. live bytes by first project frame (…/flow-matching/… in the stack,
     falling back to the innermost frame) — "whose line holds it";
  2. live bytes by innermost torch frame — "what kind of tensor".
The point is replacing component arithmetic (which missed >=10 GiB on
the molmo2 smoke ladder, rungs 4-6) with measured attribution.
"""

from __future__ import annotations

import collections
import pickle
import sys
from pathlib import Path


def frames_of(block: dict) -> list[dict]:
    frames = block.get("frames")
    if frames:
        return frames
    history = block.get("history")
    if history:
        return history[0].get("frames") or []
    return []


def site(frames: list[dict], *, project_first: bool) -> str:
    if not frames:
        return "<no stack recorded>"
    if project_first:
        for f in frames:
            name = f.get("filename", "")
            if "flow-matching" in name and "site-packages" not in name:
                return f"{name.rsplit('/flow-matching/', 1)[-1]}:{f['line']} {f.get('name', '')}"
    f = frames[0]
    return f"{f.get('filename', '?').rsplit('/', 1)[-1]}:{f.get('line', '?')} {f.get('name', '')}"


def main() -> int:
    path = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    with Path(path).open("rb") as fh:
        snap = pickle.load(fh)

    by_project: collections.Counter[str] = collections.Counter()
    by_torch: collections.Counter[str] = collections.Counter()
    total = 0
    reserved = 0
    for seg in snap["segments"]:
        reserved += seg["total_size"]
        for block in seg["blocks"]:
            if block["state"] != "active_allocated":
                continue
            size = block["size"]
            total += size
            frames = frames_of(block)
            by_project[site(frames, project_first=True)] += size
            by_torch[site(frames, project_first=False)] += size

    gib = 2**30
    print(f"live allocated {total / gib:.2f} GiB / reserved {reserved / gib:.2f} GiB\n")
    print(f"== top {top_n} by project allocation site ==")
    for key, size in by_project.most_common(top_n):
        print(f"{size / gib:8.2f} GiB  {key}")
    print(f"\n== top {top_n} by innermost frame ==")
    for key, size in by_torch.most_common(top_n):
        print(f"{size / gib:8.2f} GiB  {key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
