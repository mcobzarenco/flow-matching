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
    uv run python fontaine/harness/discord.py post --body-file msg.md
        Post to the channel (≤2000 chars — substance belongs in the
        blog). ``--body-file`` reads the message body from a file
        instead of argv — the safe path for anything with backticks,
        quotes, or ``$``: shell quoting garbled an argv post once
        (2026-08-06 23:38Z) and can never garble a file. To @mention
        the owner in an escalation, include <@$DISCORD_OWNER_ID> in
        the text. ``--attach FILE`` uploads one file with the message
        (≤10 MB, the bot upload cap on an unboosted server);
        recipients get a CDN URL.

    uv run python fontaine/harness/discord.py history -n 20
        Print the channel's most recent messages (oldest first)
        WITHOUT touching the cursor — context rebuild for a fresh
        session joining an ongoing conversation (e.g. after a
        tick→work chain mid-chat).

Both commands render attachments (CDN URL) and emoji reactions
(``reactions: 👍x1``) when present. Caveat (owner asked 2026-08-05):
this is REST polling, not a gateway — a reaction added to an already-
read message only surfaces on a later ``history`` call, so babysits
should ``history`` recent posts when steering-by-reaction matters.

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
import uuid
from pathlib import Path
from typing import Any

API = "https://discord.com/api/v10"
CURSOR_PATH = Path(__file__).resolve().parent / "state" / "discord_cursor"
MAX_CONTENT = 2000
MAX_ATTACHMENT = 10 * 1024 * 1024  # the bot upload cap on an unboosted server


def _env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        raise SystemExit(
            f"{name} is not set — fill ~/.config/fontaine/env "
            "(see fontaine/README.md) and run via the harness",
        )
    return value


def _request(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    body: bytes | None = None,
    content_type: str = "application/json",
) -> Any:
    """One API call; a single bounded retry on rate limit (429).

    ``payload`` is the JSON convenience path; ``body``/``content_type``
    is the raw path (multipart uploads) — mutually exclusive."""
    assert payload is None or body is None
    if payload is not None:
        body = json.dumps(payload).encode()
    for attempt in (1, 2):
        request = urllib.request.Request(
            f"{API}{path}",
            data=body,
            method=method,
            headers={
                "Authorization": f"Bot {_env('DISCORD_BOT_TOKEN')}",
                "Content-Type": content_type,
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


def _author_name(message: Any) -> str:
    author = message["author"]
    name = author.get("global_name")
    if name is None or name == "":
        name = author["username"]
    return str(name)


def _print_messages(messages: Any) -> None:
    for message in reversed(messages):  # the API returns newest first
        name = _author_name(message)
        flag = " [BOT]" if message["author"].get("bot", False) else ""
        edited = " (edited)" if message.get("edited_timestamp") else ""
        print(f"{message['timestamp']} {name}{flag}{edited}: {message['content']}")
        # Native reply context (owner question 2026-08-08 09:22Z): a
        # Discord reply carries the quoted message in
        # `referenced_message` (null when it was deleted; absent for
        # non-replies) — without this line the reply's target drops.
        if "message_reference" in message:
            referenced = message.get("referenced_message")
            if referenced:
                snippet = " ".join(str(referenced.get("content", "")).split())
                if len(snippet) > 120:
                    snippet = snippet[:120] + "…"
                print(f"    ↳ replying to {_author_name(referenced)}: {snippet}")
            else:
                print("    ↳ replying to a deleted/unavailable message")
        for attachment in message.get("attachments", []):
            print(f"    attachment: {attachment['url']}")
        reactions = [
            f"{reaction['emoji'].get('name', '?')}x{reaction['count']}"
            for reaction in message.get("reactions", [])
        ]
        if reactions:
            print(f"    reactions: {' '.join(reactions)}")


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


def post(text: str, attach: Path | None = None) -> None:
    if len(text) > MAX_CONTENT:
        raise SystemExit(
            f"message is {len(text)} chars (limit {MAX_CONTENT}) — split it, "
            "or put the substance in the blog and post the link",
        )
    channel = _env("DISCORD_CHANNEL_ID")
    if attach is None:
        message = _request("POST", f"/channels/{channel}/messages", {"content": text})
    else:
        if not attach.is_file():
            raise SystemExit(f"attachment not found: {attach}")
        data = attach.read_bytes()
        if len(data) > MAX_ATTACHMENT:
            raise SystemExit(
                f"attachment is {len(data) / 1e6:.1f} MB (bot cap "
                f"{MAX_ATTACHMENT / 1e6:.0f} MB on an unboosted server) — "
                "compress it or host it on the blog",
            )
        # Discord's multipart shape: a payload_json field plus files[N]
        # parts, each declared in payload_json's attachments list.
        payload = json.dumps(
            {
                "content": text,
                "attachments": [{"id": 0, "filename": attach.name}],
            },
        )
        boundary = f"----fontaine{uuid.uuid4().hex}"
        body = (
            (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="payload_json"\r\n'
                "Content-Type: application/json\r\n\r\n"
                f"{payload}\r\n"
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="files[0]"; '
                f'filename="{attach.name}"\r\n'
                "Content-Type: application/octet-stream\r\n\r\n"
            ).encode()
            + data
            + f"\r\n--{boundary}--\r\n".encode()
        )
        message = _request(
            "POST",
            f"/channels/{channel}/messages",
            body=body,
            content_type=f"multipart/form-data; boundary={boundary}",
        )
    print(f"[discord] posted, id {message['id']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("read", help="print messages newer than the cursor")
    post_parser = subparsers.add_parser("post", help="post to the channel")
    post_parser.add_argument(
        "text",
        nargs="?",
        default=None,
        help="message content (<=2000 chars); or use --body-file",
    )
    post_parser.add_argument(
        "--body-file",
        type=Path,
        default=None,
        help="read the message body from a file (quoting-proof path)",
    )
    post_parser.add_argument(
        "--attach",
        type=Path,
        default=None,
        help="upload one file with the message (<=10 MB)",
    )
    history_parser = subparsers.add_parser(
        "history",
        help="print the last N messages without moving the cursor",
    )
    history_parser.add_argument("-n", "--count", type=int, default=20)
    args = parser.parse_args()
    if args.command == "read":
        read()
    elif args.command == "post":
        if (args.text is None) == (args.body_file is None):
            raise SystemExit("post needs exactly one of: text argument, --body-file")
        text = args.text
        if args.body_file is not None:
            if not args.body_file.is_file():
                raise SystemExit(f"body file not found: {args.body_file}")
            text = args.body_file.read_text().strip()
        post(text, args.attach)
    else:
        history(args.count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
