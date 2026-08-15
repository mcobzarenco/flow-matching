"""The MolmoAct2 prompt-side encoder mode (§8.13).

Their serving prompt, as a first-class bijou encoder strategy:

    <bos> {Image i<|image|> ...} <|im_start|>user\\n
    The task is to {task}. The setup is <setup_start>{setup}<setup_end>.
    The current state of the robot is <state_start><state_N>...<state_end>.
    The expected control mode is <control_start>{control}<control_end>.
    Given these, what action should the robot take to complete the
    task?<|im_end|>\\n<|im_start|>assistant\\n{<action_output> | ε}

- The PREFILL SPLIT POINT is the narration switch:
  narration OFF ends the prefill with ``<action_output>`` — their exact
  serving prompt, byte-for-byte (the parity surface); narration ON ends
  it at the ChatML opener, aux text decodes as suffix and
  ``<action_output>`` appends after it. OFF-ids == ON-ids + one token,
  pinned by test.
- State is DISCRETE: ``PromptInputs.state`` arrives q01/q99-clamp-
  normalized to [-1, 1] (the run's merged-table scheme; the data side
  normalizes with the checkpoint's merged table, NOT per-dataset
  mean/std), and this collator only bins it into ``<state_N>`` tokens.
- Images run their UINT8 single-view path (``to_uint8_rgb`` truncation
  + 378x378 resize — the golden-pinned serving coercion, used for BOTH
  training and inference here, so this mode has zero train/serve skew).
- Camera KINDS are ignored — their format is positional ("Image 1"),
  and the released policy trained with per-episode randomized camera
  order, so our deterministic (kind, name) order is in-distribution.
  ``condition_text`` must be empty: the bracket surfaces are bijou-
  prompt concepts this format has no bytes for (loud, not dropped).
- Special ids resolve from the CHECKPOINT tokenizer at first use (their
  releases re-home the image specials; only the base-vocab ChatML ids
  are pinned and verified). The ``action_mode`` mask flavor computes
  the decoder-conditioning mask over the prompt (EOS/span-strip under
  ``'both'`` — load-bearing for the expert weights) which rides
  ``ObservationMemory.conditioning_mask``.

Assembly is OWNED here (template, split, batching); the leaf transforms
(task normalization, state binning, uint8 image path, token expansion,
sequence budget) are imported from ``bijou.modelling.encoders.molmoact2_processing`` — the
frozen, golden-fixture-pinned reference semantics. The byte gate
(tests/test_molmoact2_encoder.py) compares this collator's ASSEMBLY
against the port's ``pack_action_example``, which stays meaningful
precisely because only the leaves are shared.
"""

from __future__ import annotations

import contextlib
import dataclasses
from dataclasses import dataclass
from typing import Any, override

import torch
from torch import Tensor, nn

from ..gemma4.loading import resolve_checkpoint_dir
from ..interface import (
    InputsCollator,
    ObservationMemory,
    PromptInputs,
)
from ..molmo2.cache import Molmo2KVCache
from ..molmo2.model import (
    Molmo2Model,
    build_multimodal_mask,
    ensure_per_sample_patch_alignment,
)
from ..molmo2.tokenizer import Molmo2TextTokenizer
from .molmoact2_processing import (
    CONTROL_END_TOKEN,
    CONTROL_START_TOKEN,
    IMAGE_TOKEN_STRINGS,
    SETUP_END_TOKEN,
    SETUP_START_TOKEN,
    discrete_state_string,
    encode_action_prompt,
    encoder_attention_mask,
    infer_max_sequence_length,
    normalize_task_text,
    process_image_resize,
    to_uint8_rgb,
)

MOLMOACT2_PROMPT_FORMAT = 1

# Base-vocab ChatML ids — layout-stable across their releases (only the
# >= 151,936 extras re-home); verified against the loaded tokenizer.
BOS_ID = 151_645  # <|im_end|> — their bos convention
PAD_ID = 151_643  # <|endoftext|>
ACTION_OUTPUT_TOKEN = "<action_output>"
GENERATION_OPENER = "<|im_start|>assistant\n"


def robot_prompt(
    *,
    task: str,
    discrete_state: str,
    setup_type: str,
    control_mode: str,
    num_images: int,
    narration: bool,
) -> str:
    """Their ``_build_robot_text`` template with the prefill split point.

    Byte-identical to the port's ``build_robot_prompt`` for
    ``narration=False`` (the gate pins this); ``narration=True`` stops
    at the ChatML opener — the suffix decoder emits aux text there and
    ``<action_output>`` appends after it."""
    setup = f"{SETUP_START_TOKEN}{setup_type}{SETUP_END_TOKEN}"
    control = f"{CONTROL_START_TOKEN}{control_mode}{CONTROL_END_TOKEN}"
    prompt = (
        f"The task is to {task}. The setup is {setup}. The current state "
        f"of the robot is {discrete_state}. The expected control mode is "
        f"{control}. Given these, what action should the robot take to "
        f"complete the task?"
    )
    if num_images == 1:
        image_prefix = "<|image|>"
    else:
        image_prefix = "".join(
            f"Image {index + 1}<|image|>" for index in range(num_images)
        )
    tail = "" if narration else ACTION_OUTPUT_TOKEN
    return (
        f"{image_prefix}<|im_start|>user\n{prompt}<|im_end|>\n{GENERATION_OPENER}{tail}"
    )


@dataclass(frozen=True, slots=True)
class MolmoAct2Inputs:
    """The MolmoAct2-format half of a collated batch. No soft state slot
    — state is discrete in the ids. ``conditioning_mask`` is the
    decoder-conditioning mask over the prompt (the ``action_mode``
    flavor applied), distinct from ``attention_mask`` (real tokens, the
    positions/mask source).

    Shapes (V = views per sample, P = pooled tokens):
      - input_ids: [B, S]  (left-padded)
      - attention_mask: [B, S]  (1 = real token, 0 = left padding)
      - image_type_mask: [B, S]  (bool)
      - conditioning_mask: [B, S]  (bool, True = expert may attend)
      - crops: [B, V, patches, patch_dim]  (float32, -1 padded)
      - pooled_patches_idx: [B, P, pool_group]  (long, -1 padded)
    """

    input_ids: Tensor
    attention_mask: Tensor
    image_type_mask: Tensor
    conditioning_mask: Tensor
    crops: Tensor
    pooled_patches_idx: Tensor
    has_padding: bool

    def tensors(self) -> dict[str, Tensor]:
        return {
            field.name: value
            for field in dataclasses.fields(self)
            if isinstance(value := getattr(self, field.name), Tensor)
        }

    def pin_memory(self) -> MolmoAct2Inputs:
        return dataclasses.replace(
            self,
            **{name: t.pin_memory() for name, t in self.tensors().items()},
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> MolmoAct2Inputs:
        return dataclasses.replace(
            self,
            **{
                name: t.to(device, non_blocking=non_blocking)
                for name, t in self.tensors().items()
            },
        )


class MolmoAct2InputsCollator:
    """InputsCollator for the MolmoAct2 prompt format (module docstring).
    Heavy state (tokenizer, resolved ids) builds lazily and drops on
    pickle — the spawned-dataloader convention."""

    def __init__(
        self,
        checkpoint: str,
        *,
        setup_type: str,
        control_mode: str,
        num_state_tokens: int,
        action_mode: str,
        narration: bool,
    ) -> None:
        if setup_type.strip() == "" or control_mode.strip() == "":
            raise ValueError(
                "setup_type/control_mode render verbatim into the prompt — "
                "empty values build degenerate prompts (load_norm_stats "
                "guards the checkpoint side; construction guards this one)",
            )
        self.checkpoint = checkpoint
        self.setup_type = setup_type
        self.control_mode = control_mode
        self.num_state_tokens = num_state_tokens
        self.action_mode = action_mode
        self.narration = narration
        self._tokenizer: Molmo2TextTokenizer | None = None
        self._image_ids: tuple[int, ...] | None = None
        self._patch_id: int | None = None
        self._eos_id: int | None = None
        self._action_start_id: int | None = None
        self._action_end_id: int | None = None

    @override
    def __getstate__(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "_tokenizer": None,
            "_image_ids": None,
            "_patch_id": None,
            "_eos_id": None,
            "_action_start_id": None,
            "_action_end_id": None,
        }

    def _materialize(self) -> Molmo2TextTokenizer:
        if self._tokenizer is None:
            checkpoint_dir = resolve_checkpoint_dir(self.checkpoint)
            tokenizer = Molmo2TextTokenizer(str(checkpoint_dir))
            backend = tokenizer.tokenizer
            # The base-vocab ChatML pins must hold on ANY MolmoAct2
            # tokenizer; the extras resolve dynamically (releases
            # re-home them).
            for token, expected in (
                ("<|im_end|>", BOS_ID),
                ("<|endoftext|>", PAD_ID),
            ):
                actual = backend.token_to_id(token)
                if actual != expected:
                    raise SystemExit(
                        f"tokenizer maps {token!r} to {actual}, expected "
                        f"{expected} — not a MolmoAct2-family tokenizer",
                    )
            ids = tuple(
                token_id
                for token in IMAGE_TOKEN_STRINGS
                if (token_id := backend.token_to_id(token)) is not None
            )
            if not ids:
                raise SystemExit(
                    f"{checkpoint_dir} tokenizer has none of the MolmoAct2 "
                    "image tokens",
                )
            patch_id = backend.token_to_id("<im_patch>")
            if patch_id is None:
                raise SystemExit(f"{checkpoint_dir} tokenizer has no <im_patch>")
            if backend.token_to_id(ACTION_OUTPUT_TOKEN) is None:
                raise SystemExit(
                    f"{checkpoint_dir} tokenizer has no {ACTION_OUTPUT_TOKEN}",
                )
            # Present on 'both'-mode vocabularies; the mask builder
            # treats None as "no discrete spans to strip".
            self._action_start_id = backend.token_to_id("<action_start>")
            self._action_end_id = backend.token_to_id("<action_end>")
            self._image_ids = ids
            self._patch_id = int(patch_id)
            self._eos_id = BOS_ID  # eos == bos == <|im_end|> (their config)
            self._tokenizer = tokenizer
        return self._tokenizer

    def _sample_ids(self, sample: PromptInputs) -> list[int]:
        """One sample's prompt ids (their pack path: template -> image
        expansion -> BOS insert -> loud sequence budget)."""
        tokenizer = self._materialize()
        if sample.condition_text != "":
            raise ValueError(
                "the MolmoAct2 prompt format has no bytes for the bijou "
                f"condition/[generate|…] block (got "
                f"{sample.condition_text!r}) — conditioning fields cannot "
                "ride this format",
            )
        prompt = robot_prompt(
            task=normalize_task_text(sample.instruction),
            discrete_state=discrete_state_string(
                sample.state,
                num_state_tokens=self.num_state_tokens,
            ),
            setup_type=self.setup_type,
            control_mode=self.control_mode,
            num_images=len(sample.cameras),
            narration=self.narration,
        )
        ids = encode_action_prompt(prompt, tokenizer)
        cap = infer_max_sequence_length(
            num_images=len(sample.cameras),
            state_dim=int(sample.state.shape[0]),
        )
        if len(ids) > cap:
            raise ValueError(
                f"sequence length {len(ids)} exceeds max_sequence_length={cap}",
            )
        return ids

    def __call__(self, samples: list[PromptInputs]) -> MolmoAct2Inputs:
        """Shapes (returned batch, B = len(samples), S = max prompt len):
        - ``input_ids``: [B, S] long, left-padded with PAD_ID
        - ``attention_mask``: [B, S] long, 1 = real token
        - ``image_type_mask``: [B, S] bool
        - ``conditioning_mask``: [B, S] bool (action_mode flavor applied)
        - ``crops``: [B, V_max, 729, 588] fp32, -1-filled pad views
        - ``pooled_patches_idx``: [B, P_max, 4] long, -1-filled
        """
        self._materialize()
        assert self._image_ids is not None and self._patch_id is not None
        sequences: list[list[int]] = []
        crops_per_sample: list[Tensor] = []
        pooled_per_sample: list[Tensor] = []
        for sample in samples:
            if not sample.cameras:
                raise ValueError("MolmoAct2 prompts require at least one camera")
            sequences.append(self._sample_ids(sample))
            images = [
                process_image_resize(to_uint8_rgb(camera.image))
                for camera in sample.cameras
            ]
            crop_base = 0
            pooled: list[Tensor] = []
            for image in images:
                idx = image.pooled_idx
                pooled.append(torch.where(idx >= 0, idx + crop_base, idx))
                crop_base += image.crops.shape[0] * image.crops.shape[1]
            crops_per_sample.append(
                torch.cat([image.crops for image in images], dim=0),
            )
            pooled_per_sample.append(torch.cat(pooled, dim=0))

        batch_size = len(samples)
        width = max(len(ids) for ids in sequences)
        input_ids = torch.full((batch_size, width), PAD_ID, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, width), dtype=torch.long)
        for row, ids in enumerate(sequences):
            input_ids[row, width - len(ids) :] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, width - len(ids) :] = 1
        image_id_tensor = torch.tensor(sorted(self._image_ids), dtype=torch.long)
        image_type_mask = torch.isin(input_ids, image_id_tensor)
        image_type_mask &= attention_mask.bool()

        conditioning_mask = encoder_attention_mask(
            input_ids,
            attention_mask,
            action_mode=self.action_mode,
            eos_token_id=self._eos_id,
            action_start_token_id=self._action_start_id,
            action_end_token_id=self._action_end_id,
        )
        assert conditioning_mask is not None  # attention_mask given

        max_views = max(c.shape[0] for c in crops_per_sample)
        max_pooled = max(p.shape[0] for p in pooled_per_sample)
        patches, patch_dim = crops_per_sample[0].shape[1:]
        group = pooled_per_sample[0].shape[1]
        crops = torch.full(
            (batch_size, max_views, patches, patch_dim),
            -1.0,
            dtype=torch.float32,
        )
        pooled_patches_idx = torch.full(
            (batch_size, max_pooled, group),
            -1,
            dtype=torch.long,
        )
        for row in range(batch_size):
            crops[row, : crops_per_sample[row].shape[0]] = crops_per_sample[row]
            pooled_patches_idx[row, : pooled_per_sample[row].shape[0]] = (
                pooled_per_sample[row]
            )
        ensure_per_sample_patch_alignment(
            input_ids,
            pooled_patches_idx,
            image_patch_id=self._patch_id,
        )
        return MolmoAct2Inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_type_mask=image_type_mask,
            conditioning_mask=conditioning_mask,
            crops=crops,
            pooled_patches_idx=pooled_patches_idx,
            has_padding=bool((attention_mask == 0).any()),
        )


class MolmoAct2Encoder(nn.Module):
    """The MolmoAct2 prompt-side strategy: their-format collation and the
    multimodal prefix encode (a plain module speaking the encoder
    convention — :mod:`bijou.modelling.interface`'s module docstring).
    The whole product is the prefix KV cache (``retain_cache=True`` —
    the molmo_flow decoder conditions on it and the narration suffix
    continues it) plus the ``conditioning_mask``.

    NO prompt-side parameters: state enters as discrete tokens (no soft
    state token, no ``state_proj``), so ``prompt.safetensors`` for this
    mode is an empty section. The trunk unfreeze surface matches the
    molmo2 encoder's (same trunk, same freezing split: embedding
    matrices and the shipped lm_head stay frozen by design)."""

    def __init__(
        self,
        checkpoint: str,
        *,
        setup_type: str,
        control_mode: str,
        num_state_tokens: int,
        action_mode: str,
        narration: bool,
    ) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.setup_type = setup_type
        self.control_mode = control_mode
        self.num_state_tokens = num_state_tokens
        self.action_mode = action_mode
        self.narration = narration
        # Save-side stashes (train sets them from the source checkpoint's
        # sections; loading owns the schema, this module cannot import
        # it): the prompt section dict for the checkpoint round-trip, and
        # the state q01/q99 rows the run normalized prompts with — the
        # written normalization table must carry THE table in use, not
        # the (quantile-less) run aggregate.
        self.prompt_schema: dict[str, Any] | None = None
        self.state_table: tuple[tuple[float, ...], tuple[float, ...]] | None = None
        # The merged ACTION table under --objective ar (retirement phase
        # 3): flow/joint runs read the decoder's own decision-6 tables at
        # save time; an ar run has no flow decoder, so the table the
        # collator tokenized with is stashed here for the written
        # normalization row.
        self.action_table: tuple[tuple[float, ...], tuple[float, ...]] | None = None

    def inputs_collator(self) -> InputsCollator[MolmoAct2Inputs]:
        return MolmoAct2InputsCollator(
            self.checkpoint,
            setup_type=self.setup_type,
            control_mode=self.control_mode,
            num_state_tokens=self.num_state_tokens,
            action_mode=self.action_mode,
            narration=self.narration,
        )

    def encode(
        self,
        backbone: Molmo2Model,
        inputs: MolmoAct2Inputs,
        *,
        with_grad: bool,
        retain_cache: bool = False,
    ) -> ObservationMemory:
        """Run the full multimodal prefix (vision inject + causal-OR-
        image-block mask — NO state splice, state is in the ids), retain
        the prefix cache when asked, and carry the conditioning mask to
        the decoder seam."""
        padding_mask = inputs.attention_mask if inputs.has_padding else None
        with torch.no_grad() if not with_grad else contextlib.nullcontext():
            embeds = backbone.build_input_embeddings(
                inputs.input_ids,
                crops=inputs.crops,
                pooled_patches_idx=inputs.pooled_patches_idx,
            )
            position_ids = (
                Molmo2Model.logical_positions(inputs.attention_mask)
                if padding_mask is not None
                else None
            )
            mask = build_multimodal_mask(
                image_type_mask=inputs.image_type_mask,
                padding_mask=padding_mask,
                dtype=embeds.dtype,
                device=embeds.device,
            )
            cache = (
                Molmo2KVCache(len(backbone.text.transformer.blocks))
                if retain_cache
                else None
            )
            backbone.text.transformer(
                inputs_embeds=embeds,
                position_ids=position_ids,
                attention_mask=mask,
                cache=cache,
            )
        return ObservationMemory(
            streams={},
            length=inputs.input_ids.shape[1],
            padding_mask=padding_mask,
            cache=cache,
            conditioning_mask=inputs.conditioning_mask,
        )

    def param_groups(self, backbone: Molmo2Model) -> dict[str, list[nn.Parameter]]:
        """Same trunk surface as the molmo2 encoder (same trunk, same
        freezing split): ``"text"`` = decoder blocks + ``ln_f``,
        ``"vision"`` = tower + connector; embeddings and the shipped
        lm_head stay frozen by design."""
        text: list[nn.Parameter] = []
        for block in backbone.text.transformer.blocks:
            text.extend(block.parameters())
        text.extend(backbone.text.transformer.ln_f.parameters())
        return {
            "text": text,
            "vision": list(backbone.vision.parameters()),
        }
