"""Oracles for fontaine/scripts/queue_cli.py (#21 P2 queue-as-data).

The point of P2 is that the queue's invariants are machine-checked
instead of eyeballed: depth >= 2 (or a stated reason), every gpu-*
item names an existing pre-reg post, owner holds cannot silently
become pickable. Each failure mode the review named gets a test, and
the REAL repo queue.json must validate green — a red here means the
canonical queue itself is in a state a tick would have to catch by
reading prose, which is exactly the seam P2 closes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "fontaine" / "scripts" / "queue_cli.py"
spec = importlib.util.spec_from_file_location("queue_cli", SCRIPT)
assert spec is not None and spec.loader is not None
queue_cli = importlib.util.module_from_spec(spec)
sys.modules["queue_cli"] = queue_cli
spec.loader.exec_module(queue_cli)

REPO = Path(__file__).resolve().parents[1]


def _item(**overrides: object) -> dict:
    base: dict[str, object] = {
        "id": "x",
        "title": "t",
        "class": "cpu",
        "status": "queued",
        "prereg": None,
        "owner_hold": False,
        "boundary": None,
    }
    base.update(overrides)
    return base


def _data(items: list[dict], depth_reason: str | None = None) -> dict:
    return {
        "updated_utc": "2026-08-07T01:20:00+00:00",
        "depth_reason": depth_reason,
        "items": items,
    }


class TestValidate:
    def test_real_queue_is_valid(self) -> None:
        data = queue_cli.load_queue(queue_cli.QUEUE)
        assert queue_cli.validate(data) == []

    def test_depth_gate(self) -> None:
        data = _data([_item(id="a")])
        failures = queue_cli.validate(data)
        assert any("depth 1 < 2" in f for f in failures)
        # a stated reason excuses low depth
        assert (
            queue_cli.validate(
                _data(
                    [_item(id="a")],
                    depth_reason="post-endpoint refill next session",
                ),
            )
            == []
        )
        # blocked/live items do not count toward depth
        data = _data([_item(id="a"), _item(id="b", status="blocked")])
        assert any("depth 1 < 2" in f for f in queue_cli.validate(data))

    def test_gpu_item_requires_prereg(self) -> None:
        two_cpu = [_item(id="c1"), _item(id="c2")]
        data = _data(
            [*two_cpu, _item(id="g", **{"class": "gpu-box", "status": "blocked"})],
        )
        failures = queue_cli.validate(data)
        assert any("g: gpu-box item has no pre-reg" in f for f in failures)
        # a prereg path that does not exist on disk also fails
        data = _data(
            [
                *two_cpu,
                _item(
                    id="g",
                    **{
                        "class": "gpu-box",
                        "status": "blocked",
                        "prereg": "no/such/post.md",
                    },
                ),
            ],
        )
        assert any("g: prereg path not found" in f for f in queue_cli.validate(data))
        # a real post path passes
        real = "fontaine/blog/src/posts/2026-08-06-prereg-molmo2-ar-40k.md"
        data = _data(
            [
                *two_cpu,
                _item(
                    id="g",
                    **{"class": "gpu-box", "status": "blocked", "prereg": real},
                ),
            ],
        )
        assert queue_cli.validate(data) == []

    def test_owner_hold_cannot_be_queued(self) -> None:
        data = _data([_item(id="a"), _item(id="b"), _item(id="h", owner_hold=True)])
        failures = queue_cli.validate(data)
        assert any("h: owner_hold" in f for f in failures)
        data = _data(
            [
                _item(id="a"),
                _item(id="b"),
                _item(id="h", owner_hold=True, status="blocked"),
            ],
        )
        assert queue_cli.validate(data) == []

    def test_schema_gates(self) -> None:
        bad = _item(id="s", **{"class": "gpu-cloud", "status": "someday"})
        failures = queue_cli.validate(_data([_item(id="a"), _item(id="b"), bad]))
        assert any("class 'gpu-cloud'" in f for f in failures)
        assert any("status 'someday'" in f for f in failures)
        dupes = _data([_item(id="a"), _item(id="a"), _item(id="b")])
        assert any("a: duplicate id" in f for f in queue_cli.validate(dupes))
        missing = {"id": "m", "title": "t"}
        failures = queue_cli.validate(_data([_item(id="a"), _item(id="b"), missing]))
        assert any("m: missing key 'class'" in f for f in failures)


class TestCommands:
    def test_next_is_first_queued_in_array_order(self) -> None:
        data = _data(
            [
                _item(id="running", status="live"),
                _item(id="held", status="blocked", owner_hold=True),
                _item(id="first-pick"),
                _item(id="second-pick"),
            ],
        )
        assert queue_cli.queued_items(data)[0]["id"] == "first-pick"

    def test_open_excludes_done(self) -> None:
        data = _data([_item(id="a", status="done"), _item(id="b")])
        assert [it["id"] for it in queue_cli.open_items(data)] == ["b"]

    def test_real_queue_has_a_next_pick(self) -> None:
        data = queue_cli.load_queue(queue_cli.QUEUE)
        queued = queue_cli.queued_items(data)
        # an empty pick list is legal only under the same rule validate
        # applies to depth < 2: a stated depth_reason (e.g. an owner
        # pause with supply gated on run boundaries)
        assert queued or data.get("depth_reason"), (
            "canonical queue has no pickable item and no depth_reason"
        )
        # every queued item must be pickable without a GPU decision:
        # gpu-* items enter as blocked/live and are promoted explicitly
        assert all(it["class"] == "cpu" or it["prereg"] for it in queued)
