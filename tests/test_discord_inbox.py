"""Unreplied-inbox oracles (class fix, 2026-08-13 missed-reply
incident): messages `read` surfaces were truncated out of a babysit's
terminal output, and consume-once cursor semantics then buried two
owner questions for ~2 h. The structural fix: every non-bot message
`read` consumes is ALSO appended to state/discord_unreplied.jsonl;
read and babysit print the pending count as a loud FIRST line; only an
explicit `ack <id>` clears an entry — result posts never do.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


discord = sys.modules.get("fontaine_discord") or _load(
    "fontaine_discord",
    _ROOT / "fontaine/harness/discord.py",
)
babysit = _load("fontaine_babysit_inbox", _ROOT / "fontaine/scripts/babysit.py")


def _message(mid: str, *, bot: bool = False, content: str = "hello") -> dict:
    return {
        "id": mid,
        "timestamp": "2026-08-13T19:05:00Z",
        "author": {"username": "owner", "global_name": None, "bot": bot},
        "content": content,
        "attachments": [],
    }


@pytest.fixture
def inbox_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "discord_unreplied.jsonl"
    monkeypatch.setattr(discord, "INBOX_PATH", path)
    return path


def _reader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    messages: list[dict],
) -> None:
    """Wire `read` to a fake channel: cursor pre-set, API returns
    `messages` (newest first, like Discord)."""
    cursor = tmp_path / "discord_cursor"
    cursor.write_text("100")
    monkeypatch.setattr(discord, "CURSOR_PATH", cursor)
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "t")
    monkeypatch.setenv("DISCORD_CHANNEL_ID", "c")
    monkeypatch.setattr(discord, "_request", lambda *a, **k: messages)


def test_read_populates_inbox_and_skips_bot_messages(
    inbox_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _reader(
        monkeypatch,
        tmp_path,
        [
            _message("103", bot=True, content="status post"),
            _message("102", content="which implementation?"),
            _message("101", content="explain this experiment"),
        ],
    )
    discord.read()
    out = capsys.readouterr().out
    entries = [json.loads(ln) for ln in inbox_path.read_text().splitlines()]
    assert [e["id"] for e in entries] == ["101", "102"]  # oldest first, no bot
    assert entries[0]["content"] == "explain this experiment"
    assert entries[0]["author"] == "owner"
    assert "INBOX +2" in out
    assert "result posts do NOT count" in out


def test_read_prints_pending_banner_as_first_line(
    inbox_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inbox_path.write_text(
        json.dumps(
            {"id": "9", "timestamp": "t", "author": "owner", "content": "q"},
        )
        + "\n",
    )
    _reader(monkeypatch, tmp_path, [])
    discord.read()
    first = capsys.readouterr().out.splitlines()[0]
    assert "UNREPLIED INBOX: 1" in first
    assert "9" in first


def test_read_dedupes_by_id(
    inbox_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _reader(monkeypatch, tmp_path, [_message("101")])
    discord.read()
    discord.CURSOR_PATH.write_text("100")  # replay the same message
    discord.read()
    entries = inbox_path.read_text().splitlines()
    assert len(entries) == 1


def test_ack_clears_only_named_ids(
    inbox_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    discord._inbox_write(
        [
            {"id": "1", "timestamp": "t", "author": "owner", "content": "a"},
            {"id": "2", "timestamp": "t", "author": "owner", "content": "b"},
        ],
    )
    discord.ack(["1"])
    out = capsys.readouterr().out
    assert "acked 1; 1 still pending" in out
    entries = [json.loads(ln) for ln in inbox_path.read_text().splitlines()]
    assert [e["id"] for e in entries] == ["2"]


def test_ack_unknown_id_errors_and_clears_nothing(inbox_path: Path) -> None:
    discord._inbox_write(
        [{"id": "1", "timestamp": "t", "author": "owner", "content": "a"}],
    )
    with pytest.raises(SystemExit, match="not in the inbox: 7"):
        discord.ack(["7"])
    assert len(inbox_path.read_text().splitlines()) == 1


def test_inbox_command_reprints_entries_in_full(
    inbox_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    discord._inbox_write(
        [
            {
                "id": "5",
                "timestamp": "2026-08-13T20:03:00Z",
                "author": "owner",
                "content": "which molmoact2 implementation?",
            },
        ],
    )
    discord.inbox()
    out = capsys.readouterr().out
    assert "UNREPLIED INBOX: 1" in out.splitlines()[0]
    assert "which molmoact2 implementation?" in out


def test_babysit_counts_pending_and_banners_when_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "discord_unreplied.jsonl"
    assert babysit.inbox_pending(path) == 0
    path.write_text('{"id": "1"}\n\n{"id": "2"}\n')
    assert babysit.inbox_pending(path) == 2
    babysit.print_inbox_banner(2)
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("!!! UNREPLIED DISCORD INBOX: 2")
    babysit.print_inbox_banner(0)
    assert capsys.readouterr().out == ""
