"""The Gemma-trunk side of collation: chat-templated multimodal prompts.

Renders ``[instruction][cameras...][instruction]`` per sample through the
Gemma4 processor (chat template, right padding). The camera slot NAMES in
each PromptInputs are deliberately ignored — prompt slots are positional
(community image/image2 keys carry no reliable semantics; SmolVLA
precedent). The processor is built lazily so the strategy can be pickled
into dataloader workers.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, override

import torch
import transformers
from PIL import Image
from torch import Tensor

from ..interface import PromptInputs


@dataclass(frozen=True, slots=True)
class GemmaInputs:
    """The trunk-specific half of a collated batch: one chat-templated
    prompt per sample, ready for ``BijouModel.encode_prefix``.

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
