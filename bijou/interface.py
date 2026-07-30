"""The observation-encoder ↔ action-decoder seam.

An encoder turns one observation (instruction + camera frames [+ state,
eventually]) into an :class:`EncodedPrefix`: a set of named memory streams
a decoder cross-attends. The streams' static geometry (:class:`
StreamGeometry`) is declared by the encoder at construction time so a
decoder can size its query projections and RoPE behavior without knowing
what kind of trunk produced the memory (see ``docs/plan.md``).

Stream names are defined by each encoder (the Gemma trunk exports its
global layers' K/V as ``"kv{layer}"``); decoder schedules reference those
names, and composition validates the references (unknown name or unused
export = loud error).
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .nn import RopeParameters


def kv_stream_name(layer_idx: int) -> str:
    """The Gemma trunk's stream-naming convention: K/V of backbone layer
    ``layer_idx`` is exported as ``"kv{layer_idx}"``. Shared by the
    producer (prefix encode) and the consumer (the flow decoder's
    int-schedule config) until the encoder abstraction owns it."""
    return f"kv{layer_idx}"


@dataclass(frozen=True, slots=True)
class StreamGeometry:
    """Static per-stream contract, known at construction time.

    ``rope`` set: keys arrive position-encoded and the decoder must RoPE
    its queries at positions ≥ the per-sample real prefix length (the
    Gemma streams' contract). ``rope`` None: positions are baked into the
    memory (e.g. an adapter with learned positions); the decoder applies
    no query RoPE for this stream.
    """

    kv_heads: int
    head_dim: int
    rope: RopeParameters | None


@dataclass(frozen=True, slots=True)
class MemoryStream:
    """One exported stream's K/V.

    Shapes: key/value [B, kv_heads, P, head_dim]."""

    key: Tensor
    value: Tensor


@dataclass(frozen=True, slots=True)
class EncodedPrefix:
    """The value crossing the encoder → decoder seam: named memory streams
    (insertion-ordered as the encoder's exports) plus the (padded) prefix
    width P and, for padded batches, the True-means-real padding mask
    [B, P]. Per-sample real lengths (decoder query position bases) derive
    from the mask; ``length`` is the KV width and the position base only
    for unpadded batches."""

    streams: dict[str, MemoryStream]
    length: int
    padding_mask: Tensor | None = None

    @property
    def batch_size(self) -> int:
        first = next(iter(self.streams.values()))
        return first.key.shape[0]
