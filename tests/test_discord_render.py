"""Rendering oracles for the Discord harness reader (#owner 2026-08-08
09:22Z): native reply references and edit markers must survive into the
printed transcript — a reply whose quoted context silently drops is a
steering-loss class (the read loop is the only channel the agent sees).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "fontaine_discord",
    Path(__file__).resolve().parents[1] / "fontaine/harness/discord.py",
)
assert spec is not None and spec.loader is not None
discord = importlib.util.module_from_spec(spec)
sys.modules["fontaine_discord"] = discord
spec.loader.exec_module(discord)


def _message(**overrides: object) -> dict:
    base: dict[str, object] = {
        "id": "1",
        "timestamp": "2026-08-08T10:00:00Z",
        "author": {"username": "owner", "global_name": None, "bot": False},
        "content": "hello",
        "attachments": [],
    }
    base.update(overrides)
    return base


def render(messages: list[dict], capsys: pytest.CaptureFixture[str]) -> str:
    discord._print_messages(messages)
    return capsys.readouterr().out


def test_plain_message_renders_without_reply_or_edit_lines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = render([_message()], capsys)
    assert "owner: hello" in out
    assert "replying to" not in out
    assert "(edited)" not in out


def test_native_reply_renders_referenced_author_and_snippet(
    capsys: pytest.CaptureFixture[str],
) -> None:
    msg = _message(
        content="yes do that",
        message_reference={"message_id": "9"},
        referenced_message={
            "author": {"username": "bot", "global_name": "fontaine", "bot": True},
            "content": "shall I launch the run?",
        },
    )
    out = render([msg], capsys)
    assert "↳ replying to fontaine: shall I launch the run?" in out


def test_reply_snippet_truncates_and_flattens_newlines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    msg = _message(
        message_reference={"message_id": "9"},
        referenced_message={
            "author": {"username": "bot", "global_name": None, "bot": True},
            "content": "line one\nline two " + "x" * 200,
        },
    )
    out = render([msg], capsys)
    line = next(ln for ln in out.splitlines() if "replying to" in ln)
    assert "\n" not in line.removesuffix("\n")
    assert line.endswith("…")
    assert "line one line two" in line


def test_deleted_reference_renders_placeholder(
    capsys: pytest.CaptureFixture[str],
) -> None:
    msg = _message(
        message_reference={"message_id": "9"},
        referenced_message=None,
    )
    out = render([msg], capsys)
    assert "replying to a deleted/unavailable message" in out


def test_edited_message_carries_marker(capsys: pytest.CaptureFixture[str]) -> None:
    msg = _message(edited_timestamp="2026-08-08T10:05:00Z")
    out = render([msg], capsys)
    assert "owner (edited): hello" in out
