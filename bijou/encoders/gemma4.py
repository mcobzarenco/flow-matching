"""The Gemma-trunk observation encoder.

Collation renders ``[instruction][cameras...][instruction]`` per sample
through the Gemma4 processor (chat template, right padding). The camera
slot NAMES in each PromptInputs are deliberately ignored — prompt slots
are positional (community image/image2 keys carry no reliable semantics;
SmolVLA precedent). The processor is built lazily so the strategy can be
pickled into dataloader workers.

Encoding runs the truncated backbone over the multimodal prefix and
exports the K/V of the configured global layers as memory streams — the
expert is, in effect, "more KV-shared layers" grafted onto the trunk's
own read interface.
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, override

import torch
import transformers
from PIL import Image
from torch import Tensor, nn

from ..gemma4.cache import KVCache
from ..gemma4.config import Gemma4Config, LayerType
from ..gemma4.model import Gemma4Model
from ..gemma4.text import DecoderLayer
from ..interface import (
    InputsCollator,
    MemoryStream,
    ObservationMemory,
    PromptInputs,
    StreamGeometry,
    kv_stream_name,
)


@dataclass(frozen=True, slots=True)
class GemmaInputs:
    """The trunk-specific half of a collated batch: one chat-templated
    prompt per sample, ready for ``BijouModel.encode_observation``.

    The prompt layout (which tokens are images, where padding sits, P
    itself) is decided at collate time and carried by ``input_ids``;
    ``has_padding`` is computed CPU-side in the dataloader workers so the
    training loop never needs a device→host sync to decide whether to
    build padding masks.

    Shapes (``images`` = Σ per-sample camera images, not B):
      - input_ids: [B, P]
      - attention_mask: [B, P]  (1 = real token, 0 = right padding)
      - pixel_values: [images, patches, 3·patch_size²]  (RGB in [0, 1])
      - image_position_ids: [images, patches, 2]  ((x, y) spatial ids)
    """

    input_ids: Tensor
    attention_mask: Tensor
    pixel_values: Tensor
    image_position_ids: Tensor
    has_padding: bool

    def tensors(self) -> dict[str, Tensor]:
        return {
            field.name: value
            for field in dataclasses.fields(self)
            if isinstance(value := getattr(self, field.name), Tensor)
        }

    def pin_memory(self) -> GemmaInputs:
        return dataclasses.replace(
            self,
            **{name: t.pin_memory() for name, t in self.tensors().items()},
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> GemmaInputs:
        return dataclasses.replace(
            self,
            **{
                name: t.to(device, non_blocking=non_blocking)
                for name, t in self.tensors().items()
            },
        )


class GemmaInputsCollator:
    """InputsCollator for the Gemma trunk (see the module docstring)."""

    def __init__(self, checkpoint: str, max_soft_tokens: int) -> None:
        self.checkpoint = checkpoint
        self.max_soft_tokens = max_soft_tokens
        self._processor: Any = None

    @override
    def __getstate__(self) -> dict[str, Any]:
        # Never ship a constructed processor across process boundaries; spawn
        # workers rebuild it lazily.
        return {**self.__dict__, "_processor": None}

    def _to_pil(self, image: Tensor) -> Image.Image:
        """One camera frame [3, height, width] (float in [0, 1], CHW as
        LeRobot decodes video) -> PIL RGB image for the processor."""
        array = (image.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
        return Image.fromarray(array)

    def __call__(self, samples: list[PromptInputs]) -> GemmaInputs:
        if self._processor is None:
            # Lazy construction (not import): the collator is pickled into
            # spawned dataloader workers, each of which rebuilds it.
            self._processor = transformers.AutoProcessor.from_pretrained(
                self.checkpoint,
            )
            self._processor.tokenizer.padding_side = "right"

        conversations = []
        for sample in samples:
            content: list[dict[str, Any]] = [
                {"type": "text", "text": sample.instruction},
            ]
            content.extend(
                {"type": "image", "image": self._to_pil(camera.image)}
                for camera in sample.cameras
            )
            content.append({"type": "text", "text": sample.instruction})
            conversations.append([{"role": "user", "content": content}])

        batch = self._processor.apply_chat_template(
            conversations,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            # transformers 5.14: per-call processor kwargs must be nested, and
            # a flat `padding=True` alongside `processor_kwargs` silently
            # drops the latter -- both go inside (verified empirically).
            processor_kwargs={
                "max_soft_tokens": self.max_soft_tokens,
                "padding": True,
            },
        )
        return GemmaInputs(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            image_position_ids=batch["image_position_ids"],
            # Decided here (CPU, in the worker) so the train loop never syncs.
            has_padding=bool((batch["attention_mask"] == 0).any()),
        )


class GemmaEncoder(nn.Module):
    """The Gemma prompt-side strategy: collation, prefix encoding, and the
    trunk's unfreeze surface (see the module docstring). ``exports`` are
    the global layers whose K/V become the memory streams.

    The trunk itself is NOT owned here — BijouModel owns it once and
    passes it into the compute methods; this module carries only
    prompt-side parameters (none today; a projected-state slot is the
    anticipated first). ``config`` is the trunk's (truncated) architecture
    — enough to declare stream geometries without the weights."""

    def __init__(
        self,
        config: Gemma4Config,
        *,
        exports: tuple[int, ...],
        processor_dir: str,
        max_soft_tokens: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.exports = exports
        self.processor_dir = processor_dir
        self.max_soft_tokens = max_soft_tokens

    def stream_geometries(self) -> dict[str, StreamGeometry]:
        """Static geometry per stream name; keys and order match every
        ObservationMemory this encoder produces."""
        text = self.config.text
        geometry = StreamGeometry(
            kv_heads=text.num_global_key_value_heads or text.num_key_value_heads,
            head_dim=text.head_dim_for_type(LayerType.FULL),
            rope=text.rope_parameters[LayerType.FULL],
        )
        return {kv_stream_name(layer): geometry for layer in self.exports}

    def inputs_collator(self) -> InputsCollator[GemmaInputs]:
        """The encoder-specific half of collation (pickleable into
        dataloader workers)."""
        return GemmaInputsCollator(self.processor_dir, self.max_soft_tokens)

    def encode_tensors(
        self,
        trunk: Gemma4Model,
        input_ids: Tensor,
        *,
        pixel_values: Tensor | None = None,
        image_position_ids: Tensor | None = None,
        padding_mask: Tensor | None = None,
        retain_cache: bool = False,
    ) -> ObservationMemory:
        """Run the truncated backbone over the multimodal prefix and export
        the memory streams (grad-transparent — callers choose no_grad).
        Cache the result across flow steps (and, if the observation is
        unchanged, across replans).

        ``retain_cache`` keeps the full prefix KVCache on the returned
        memory (the exported streams are views into it either way);
        the default drops it, freeing the non-exported layers' K/V.

        For right-padded batches (mixed-length instructions), pass the HF
        ``attention_mask`` (True/1 = real token) as ``padding_mask``; it
        masks both the backbone's self-attention and the decoder's
        cross-attention.

        Shapes (P = prompt/prefix tokens = the encoded sequence length;
        ``images`` = Σ per-sample camera images, not B):
          - input_ids: [B, P]
          - pixel_values: [images, patches, 3·patch_size²]
          - image_position_ids: [images, patches, 2]  ((x, y) spatial ids)
          - padding_mask (when present): [B, P]  (True = real token)
          - returns ObservationMemory: streams["kv{layer}"].key/value each
            [B, kv_heads, P, head_dim]; padding_mask [B, P] or None
        """
        inputs_embeds, per_layer_inputs = trunk.embed_multimodal(
            input_ids,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
        )
        cache = KVCache(trunk.config.text)
        trunk.language_model(
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            padding_mask=padding_mask,
            cache=cache,
            # The deepest exported layer's K/V depend only on its input:
            # its attention/MLP and any deeper layers are dead weight here
            # (~1/15 of decoder compute at the default E2B schedule).
            kv_stop_layer=max(self.exports),
        )
        streams: dict[str, MemoryStream] = {}
        for layer_idx in self.exports:
            layer = cache.layers[layer_idx]
            assert layer.keys is not None and layer.values is not None
            streams[kv_stream_name(layer_idx)] = MemoryStream(
                key=layer.keys,
                value=layer.values,
            )
        return ObservationMemory(
            streams=streams,
            length=input_ids.shape[1],
            padding_mask=padding_mask,
            cache=cache if retain_cache else None,
        )

    def encode(
        self,
        trunk: Gemma4Model,
        inputs: GemmaInputs,
        *,
        with_grad: bool,
        retain_cache: bool = False,
    ) -> ObservationMemory:
        """Encode one collated batch (shapes on GemmaInputs); ``with_grad``
        selects the live-trunk training path (grad-transparent, not
        force-enabled) vs the no-grad eval path. ``retain_cache`` as on
        :meth:`encode_tensors`."""
        padding_mask = inputs.attention_mask if inputs.has_padding else None
        with torch.no_grad() if not with_grad else contextlib.nullcontext():
            return self.encode_tensors(
                trunk,
                inputs.input_ids,
                pixel_values=inputs.pixel_values,
                image_position_ids=inputs.image_position_ids,
                padding_mask=padding_mask,
                retain_cache=retain_cache,
            )

    def _trainable_text_parameters(self, trunk: Gemma4Model) -> Iterator[nn.Parameter]:
        """The text-trunk parameters that PARTICIPATE in a prefix encode,
        and only those — the set must be exact because DDP requires every
        grad-enabled parameter to receive gradients each step.

        Participation mirrors ``kv_stop_layer`` (= the deepest exported
        stream): layers below it run fully; the stop layer runs only its
        input layernorm and K/V projections (``TextAttention.project_kv``;
        its v_norm is scale-less — no parameters); deeper layers, the final
        norm and the LM head never run. Token embeddings and the PLE tables
        stay frozen BY DESIGN (few rows touched per batch, dense Adam state
        for a 262k vocab is waste, and frozen embeddings are the cheapest
        forgetting control) — but the PLE *projection* path runs for every
        consumed layer slice, so it trains. The multimodal projector
        (embed_vision) is the vision->text interface and belongs to the
        text group regardless of whether the vision tower itself is
        unfrozen.
        """
        text = trunk.language_model
        stop_layer = max(self.exports)
        for idx, layer in enumerate(text.layers):
            # ModuleList iteration erases the element type (torch types
            # Module.__getattr__ as Tensor | Module): narrow before access.
            assert isinstance(layer, DecoderLayer)
            if idx < stop_layer:
                yield from layer.parameters()
            elif idx == stop_layer:
                yield from layer.input_layernorm.parameters()
                attention = layer.self_attn
                assert attention.k_proj is not None
                assert attention.k_norm is not None
                yield from attention.k_proj.parameters()
                if attention.v_proj is not None:
                    yield from attention.v_proj.parameters()
                yield from attention.k_norm.parameters()
        yield from text.per_layer_model_projection.parameters()
        yield from text.per_layer_projection_norm.parameters()
        assert trunk.embed_vision is not None
        yield from trunk.embed_vision.parameters()

    def param_groups(self, trunk: Gemma4Model) -> dict[str, list[nn.Parameter]]:
        """Named unfreezable trunk subsets — the component-lr flags route
        here. Groups are exact: DDP requires every grad-enabled parameter
        to receive gradients each step.

        ``"text"``: decoder layers up to the stop layer + PLE projections
        + the multimodal projector (see _trainable_text_parameters).
        ``"vision"``: the vision tower (empty when the backbone has none —
        callers decide whether that is an error)."""
        vision = (
            list(trunk.vision_tower.parameters())
            if trunk.vision_tower is not None
            else []
        )
        return {
            "text": list(self._trainable_text_parameters(trunk)),
            "vision": vision,
        }
