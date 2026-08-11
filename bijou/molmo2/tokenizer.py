"""Native text tokenizer for the Molmo2 checkpoint (TextTokenizer shape).

The shipped AutoProcessor/AutoTokenizer path is trust_remote_code pinned
to transformers 4.x and prompts interactively on load — unusable inside
spawned dataloader workers. The checkpoint's ``tokenizer.json`` through
the ``tokenizers`` backend is the same artifact without the remote-code
surface (the WP3 collator precedent; segment assembly proven equivalent
to whole-string tokenization there).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, override

from tokenizers import Tokenizer

from ..gemma4.loading import resolve_checkpoint_dir

__all__ = ["Molmo2TextTokenizer", "newline_carrier_ids"]


class Molmo2TextTokenizer:
    """The ``bijou.aux_text.TextTokenizer`` protocol over the checkpoint's
    own ``tokenizer.json``. Picklable into dataloader workers (the heavy
    tokenizer is built lazily and dropped on pickle)."""

    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = checkpoint
        self._tokenizer: Any = None

    @override
    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_tokenizer": None}

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            checkpoint_dir = resolve_checkpoint_dir(self.checkpoint)
            tokenizer_file = Path(checkpoint_dir) / "tokenizer.json"
            if not tokenizer_file.exists():
                # An exception, not SystemExit: this fires deep inside
                # library code (often a dataloader worker), not at a CLI
                # boundary.
                raise FileNotFoundError(f"no tokenizer.json in {checkpoint_dir}")
            self._tokenizer = Tokenizer.from_file(str(tokenizer_file))
        return self._tokenizer

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
        ).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)


def newline_carrier_ids(
    tokenizer: Molmo2TextTokenizer,
    *,
    text_vocab_size: int,
    terminator_id: int,
) -> frozenset[int]:
    """Text ids whose decoded bytes CONTAIN a newline without BEING the
    single-token terminator — Qwen's BPE has merged pieces like ``'%\\n'``
    that break the aux boundary contract (encode(value)+encode('\\n') ==
    encode(value+'\\n')); the AR decoder bans these during value decoding
    so termination is always the trained terminator id."""
    decoded = tokenizer.tokenizer.decode_batch(
        [[i] for i in range(text_vocab_size)],
        skip_special_tokens=False,
    )
    return frozenset(
        i for i, piece in enumerate(decoded) if "\n" in piece and i != terminator_id
    )
