"""Generate / verify the frozen golden-ticket candidate bank (#1 screen).

The bank is the pre-reg's M=64 i.i.d. N(0, I) candidates, shape
[64, 50, 6] float32, ticket m drawn from SeedSequence
[TICKET_DOMAIN, 0, m] (bijou.eval.policies.generate_tickets — the
single source; this script only materializes and checks it). Written
once to plans/tickets_goldenticket_m64.npz and committed; np.savez
bytes are deterministic (verified 2026-08-07), so the file sha256 is
the provenance pin every read quotes.

  --verify: regenerate from the seed schedule and compare the committed
  file byte-for-byte (array content AND file sha256) — the launch-time
  bank oracle. Exit nonzero on any mismatch.

Pure CPU, stdlib + numpy + bijou.eval.policies.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bijou.eval.policies import TICKET_DOMAIN, generate_tickets

COUNT, CHUNK, DIM = 64, 50, 6
DEFAULT_OUT = Path("plans/tickets_goldenticket_m64.npz")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    bank = generate_tickets(COUNT, (CHUNK, DIM))
    assert bank.shape == (COUNT, CHUNK, DIM) and bank.dtype == np.float32

    if args.verify:
        if not args.out.exists():
            raise SystemExit(f"VERIFY RED: {args.out} does not exist")
        data = np.load(args.out, allow_pickle=False)
        committed = data["tickets"]
        if not np.array_equal(committed, bank):
            raise SystemExit(
                "VERIFY RED: committed tickets differ from the "
                f"[TICKET_DOMAIN=0x{TICKET_DOMAIN:X}, 0, m] regeneration",
            )
        if int(data["ticket_domain"]) != TICKET_DOMAIN:
            raise SystemExit("VERIFY RED: ticket_domain field mismatch")
        print(
            f"VERIFY GREEN: {args.out} == regeneration "
            f"({COUNT}x{CHUNK}x{DIM} float32), file sha256 "
            f"{file_sha256(args.out)}",
        )
        return

    if args.out.exists():
        raise SystemExit(
            f"{args.out} already exists — the bank is generated ONCE and "
            "committed (its sha256 is pinned in reads and tests); use "
            "--verify, or pick a different --out for a new bank",
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        allow_pickle=False,
        tickets=bank,
        ticket_domain=np.array(TICKET_DOMAIN),
    )
    print(f"wrote {args.out}: tickets [{COUNT}, {CHUNK}, {DIM}] float32")
    print(f"file sha256 {file_sha256(args.out)}")


if __name__ == "__main__":
    main()
