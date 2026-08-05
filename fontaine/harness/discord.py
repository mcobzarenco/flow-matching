"""Fontaine's Discord I/O — plain REST over a bot token; the whole "tool".

There is no gateway connection, no MCP server, no library: the
headless sessions shell out to this stdlib script.

Usage (env: DISCORD_BOT_TOKEN, DISCORD_CHANNEL_ID from the harness env
file; optional DISCORD_OWNER_ID for @mentions):

    uv run python fontaine/harness/discord.py read
        Print messages newer than the stored cursor (oldest first),
        then advance the cursor. The first ever read initializes the
        cursor to the channel head WITHOUT printing the backlog —
        stale steering must never replay as new.

    uv run python fontaine/harness/discord.py post "message text"
        Post to the channel (≤2000 chars — substance belongs in the
        blog). To @mention the owner in an escalation, include
        <@$DISCORD_OWNER_ID> in the text.

    uv run python fontaine/harness/discord.py history -n 20
        Print the channel's most recent messages (oldest first)
        WITHOUT touching the cursor — context rebuild for a fresh
        session joining an ongoing conversation (e.g. after a
        tick→work chain mid-chat).

Auth model: a Discord BOT token (developer portal → New Application →
Bot → Reset Token), invited to the private server with View Channel +
Send Messages + Read Message History, and the privileged **Message
Content intent** enabled — without it, guild message content arrives
EMPTY over both gateway and REST. The token is a password: env file
only, never in git, logs, or the blog.

Cursor: ``fontaine/harness/state/discord_cursor`` holds the last-seen
message snowflake id; ids are time-ordered, so ``after=<id>`` is an
exact resume point across sessions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API = "https://discord.com/api/v10"
CURSOR_PATH = Path(__file__).resolve().parent / "state" / "discord_cursor"
MAX_CONTENT = 2000


def _env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        raise SystemExit(
            f"{name} is not set — fill ~/.config/fontaine/env "
            "(see fontaine/README.md) and run via the harness",
        )
    return value


def _request(method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    """One API call; a single bounded retry on rate limit (429)."""
    body = None if payload is None else json.dumps(payload).encode()
    for attempt in (1, 2):
        request = urllib.request.Request(
            f"{API}{path}",
            data=body,
            method=method,
            headers={
                "Authorization": f"Bot {_env('DISCORD_BOT_TOKEN')}",
                "Content-Type": "application/json",
                "User-Agent": "fontaine-harness (flow-matching, v1)",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            if error.code == 429 and attempt == 1:
                time.sleep(2.0)
                continue
            raise SystemExit(
                f"discord {method} {path} failed: HTTP {error.code} — {detail}",
            ) from error
    raise SystemExit(f"discord {method} {path}: rate-limited twice, giving up")


def _print_messages(messages: Any) -> None:
    for message in reversed(messages):  # the API returns newest first
        author = message["author"]
        name = author.get("global_name")
        if name is None or name == "":
            name = author["username"]
        flag = " [BOT]" if author.get("bot", False) else ""
        print(f"{message['timestamp']} {name}{flag}: {message['content']}")
        for attachment in message.get("attachments", []):
            print(f"    attachment: {attachment['url']}")


def read() -> None:
    channel = _env("DISCORD_CHANNEL_ID")
    if not CURSOR_PATH.exists():
        head_batch = _request("GET", f"/channels/{channel}/messages?limit=1")
        head = str(head_batch[0]["id"]) if len(head_batch) > 0 else "0"
        CURSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        CURSOR_PATH.write_text(head)
        print(f"[discord] cursor initialized at {head}; backlog not replayed")
        return
    cursor = CURSOR_PATH.read_text().strip()
    messages = _request(
        "GET",
        f"/channels/{channel}/messages?after={cursor}&limit=100",
    )
    if len(messages) == 0:
        print("[discord] no new messages")
        return
    _print_messages(messages)
    newest = max(int(message["id"]) for message in messages)
    CURSOR_PATH.write_text(str(newest))
    print(f"[discord] {len(messages)} new message(s); cursor -> {newest}")


def history(count: int) -> None:
    if count < 1 or count > 100:
        raise SystemExit(f"history count must be 1..100 (got {count})")
    channel = _env("DISCORD_CHANNEL_ID")
    messages = _request("GET", f"/channels/{channel}/messages?limit={count}")
    if len(messages) == 0:
        print("[discord] channel is empty")
        return
    _print_messages(messages)
    print(f"[discord] last {len(messages)} message(s); cursor untouched")


def post(text: str) -> None:
    if len(text) > MAX_CONTENT:
        raise SystemExit(
            f"message is {len(text)} chars (limit {MAX_CONTENT}) — split it, "
            "or put the substance in the blog and post the link",
        )
    channel = _env("DISCORD_CHANNEL_ID")
    message = _request("POST", f"/channels/{channel}/messages", {"content": text})
    print(f"[discord] posted, id {message['id']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("read", help="print messages newer than the cursor")
    post_parser = subparsers.add_parser("post", help="post to the channel")
    post_parser.add_argument("text", help="message content (<=2000 chars)")
    history_parser = subparsers.add_parser(
        "history",
        help="print the last N messages without moving the cursor",
    )
    history_parser.add_argument("-n", "--count", type=int, default=20)
    args = parser.parse_args()
    if args.command == "read":
        read()
    elif args.command == "post":
        post(args.text)
    else:
        history(args.count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
