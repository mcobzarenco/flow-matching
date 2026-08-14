"""MolmoAct2 first-class end-to-end action prediction (port item 3).

Pre-reg posts/2026-08-10-prereg-molmoact2-firstclass-port.md. This module
ties the three ported pieces into their serving stack's ``predict_action``
semantics on OUR modules: item-2 packing (``bijou.molmoact2.processing``)
-> ``bijou.molmo2`` trunk prompt forward with a retained KV cache ->
item-1 wiring + flow expert (``bijou.molmoact2.wiring``) -> their exact
output tail (dim slice, ``n_obs_steps`` chunk slice, clamp+q01/q99
unnormalize, and the reference's bf16 round-trip). Parity-gated against
their HF ``MolmoAct2ForConditionalGeneration.predict_action`` executing
on the same 240 banked anchor rows
(``fontaine/scripts/molmoact2_e2e_parity.py``, gate G2).

Semantics mirrored off their ``modeling_molmoact2.py`` (facts pinned
2026-08-11):

- Image-token ids are CHECKPOINT-DEPENDENT: the released SO-100/101
  checkpoint ships no depth vocab, so its image specials sit at 154624+;
  rig-ft exports (converted with 128 depth tokens) re-home them to
  155648+. Their processor resolves ``IMAGE_TOKENS`` strings through the
  tokenizer at load; so does this module — never the pinned constants in
  ``processing.py`` (those match the training branch / rig-ft layout).
- Bidirectional attention runs between every pair of image-typed
  positions (their ``token_type_ids`` membership over the resolved id
  set); everything else causal — exactly ``build_multimodal_mask``.
- Horizon facts resolve like their ``predict_action``: generation
  horizon = the tag metadata's ``action_horizon`` (else the config's
  ``max_action_horizon``), executed steps = metadata ``n_action_steps``
  (else the horizon), action width = the tag's action-stats dim;
  ``action_dim_is_pad`` marks dims beyond it up to ``max_action_dim``.
- The output tail keeps the reference's dtype path: the sampled chunk
  (expert dtype, bf16 deployed) is unnormalized in fp32, cast BACK to
  the chunk dtype (their ``torch.as_tensor(..., dtype=x.dtype)``), then
  to fp32 — the bf16 quantization is part of the reference output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from torch import Tensor

from ..decoders.ar_backbone import ActionCaptureStep
from ..gemma4.loading import resolve_checkpoint_dir
from ..molmo2.cache import Molmo2KVCache
from ..molmo2.model import (
    Molmo2Model,
    build_multimodal_mask,
    ensure_per_sample_patch_alignment,
)
from ..molmo2.model import load_model as load_trunk
from ..molmo2.tokenizer import Molmo2TextTokenizer
from .action_expert import (
    ActionExpert,
    ActionExpertConfig,
    load_action_expert_state,
)
from .fast_codec import MolmoAct2FastTokenizer
from .processing import (
    IMAGE_TOKEN_STRINGS,
    QuantileStats,
    load_norm_stats,
    pack_action_example,
    require_single_obs,
    unnormalize_action,
)
from .wiring import (
    encoder_attention_mask,
    extract_kv_states,
    generate_actions,
    validate_inference_config,
)

# IMAGE_TOKEN_STRINGS and require_single_obs moved to the first-class
# encoder side (phase 1 of docs/molmoact2-retirement.md); imported
# through the processing shim above and re-exported here so the port's
# call sites keep working until the package retires (phase 5).
__all__ = ["IMAGE_TOKEN_STRINGS", "MolmoAct2Predictor", "require_single_obs"]


def resolve_image_token_ids(tokenizer: Molmo2TextTokenizer) -> tuple[int, ...]:
    """The checkpoint's image-typed token ids (their processor's
    ``image_token_ids``), skipping strings absent from the vocab."""
    ids = tuple(
        token_id
        for token in IMAGE_TOKEN_STRINGS
        if (token_id := tokenizer.tokenizer.token_to_id(token)) is not None
    )
    if not ids:
        raise ValueError("tokenizer has none of the MolmoAct2 image tokens")
    return ids


def action_expert_from_config(config: dict[str, Any]) -> ActionExpert:
    """Build the (unloaded) expert exactly off a checkpoint's top-level
    ``config.json`` dict; ``llm_kv_dim`` derives from the text config.

    Any dropout-like key in ``action_expert_config`` must be 0.0: this
    builder pins both our dropout fields to 0.0 (correct for the
    released/rig checkpoints), so a checkpoint fine-tuned WITH expert
    dropout would otherwise silently train off-recipe in
    ``bijou.molmoact2.train``. The substring scan is deliberate — their
    config key names are theirs to change."""
    ae_cfg = config["action_expert_config"]
    text_cfg = config["text_config"]
    for key, value in ae_cfg.items():
        if "dropout" in key and float(value) != 0.0:
            raise NotImplementedError(
                f"action_expert_config.{key}={value}: nonzero expert dropout "
                "is not wired (this port builds the expert with all dropout "
                "pinned to 0.0, the released/rig configuration)",
            )
    expert_config = ActionExpertConfig(
        max_horizon=int(config["max_action_horizon"]),
        max_action_dim=int(config["max_action_dim"]),
        hidden_size=int(ae_cfg["hidden_size"]),
        num_layers=int(ae_cfg["num_layers"]),
        num_heads=int(ae_cfg["num_heads"]),
        mlp_ratio=float(ae_cfg["mlp_ratio"]),
        ffn_multiple_of=int(ae_cfg["ffn_multiple_of"]),
        timestep_embed_dim=int(ae_cfg["timestep_embed_dim"]),
        dropout=0.0,
        attn_dropout=0.0,
        context_layer_norm=bool(ae_cfg["context_layer_norm"]),
        qk_norm=bool(ae_cfg["qk_norm"]),
        qk_norm_eps=float(ae_cfg["qk_norm_eps"]),
        rope=bool(ae_cfg["rope"]),
        causal_attn=bool(ae_cfg["causal_attn"]),
    )
    llm_kv_dim = int(text_cfg["head_dim"]) * int(text_cfg["num_key_value_heads"])
    return expert_config.build(llm_kv_dim=llm_kv_dim)


def load_action_expert(
    checkpoint_dir: Path,
    config: dict[str, Any],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> ActionExpert:
    """Load the checkpoint's ``model.action_expert.*`` tensors into our
    expert module (compat tensors injected by ``load_action_expert_state``)."""
    prefix = "model.action_expert."
    state: dict[str, Tensor] = {}
    weight_files = sorted(checkpoint_dir.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"no *.safetensors files in {checkpoint_dir}")
    for weight_file in weight_files:
        with safe_open(weight_file, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118 — safetensors handle, not a dict
                if key.startswith(prefix):
                    state[key] = f.get_tensor(key)
    expert = action_expert_from_config(config)
    load_action_expert_state(expert, state)
    if dtype is not None:
        expert = expert.to(dtype)
    expert = expert.to(device)
    expert.eval()
    expert.requires_grad_(False)
    return expert


def _positive_int(value: Any, *, default: int, what: str) -> int:
    resolved = default if value is None else int(value)
    if resolved < 1:
        raise ValueError(f"{what} must be >= 1, got {resolved}")
    return resolved


@dataclass(frozen=True, slots=True)
class MolmoAct2Predictor:
    """The assembled first-class stack for one checkpoint + norm tag."""

    trunk: Molmo2Model
    expert: ActionExpert
    tokenizer: Molmo2TextTokenizer
    action_stats: QuantileStats
    state_stats: QuantileStats
    metadata: dict[str, Any]
    image_token_ids: tuple[int, ...]
    action_mode: str
    eos_token_id: int | None
    action_start_token_id: int | None
    action_end_token_id: int | None
    max_action_horizon: int
    max_action_dim: int
    n_obs_steps: int
    num_state_tokens: int
    flow_matching_num_steps: int
    mask_action_dim_padding: bool
    # Discrete (AR) pathway resources — owner steering 2026-08-13
    # 10:02Z: the release checkpoint's trained FAST head is the RL
    # target. ``action_token_start_id`` anchors the ``<action_i>``
    # block (release: 151934); ``fast_codec`` is the released OpenFAST
    # artifact. Both optional: the continuous path never touches them.
    action_token_start_id: int | None = None
    fast_codec: MolmoAct2FastTokenizer | None = None

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        norm_tag: str,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
        fast_tokenizer: str | Path | None = None,
    ) -> MolmoAct2Predictor:
        checkpoint_dir = resolve_checkpoint_dir(checkpoint)
        config = json.loads((checkpoint_dir / "config.json").read_text())
        validate_inference_config(config)
        # pack_action_example renders setup/control wrapped in their
        # special tokens unconditionally — the only shipped configuration.
        if not (config.get("add_setup_tokens") and config.get("add_control_tokens")):
            raise NotImplementedError(
                "add_setup_tokens/add_control_tokens=false prompts are not wired",
            )
        tokenizer = Molmo2TextTokenizer(str(checkpoint_dir))
        image_token_ids = resolve_image_token_ids(tokenizer)
        patch_id = tokenizer.tokenizer.token_to_id("<im_patch>")
        if patch_id != int(config["image_patch_id"]):
            raise ValueError(
                f"config image_patch_id {config['image_patch_id']} does not match "
                f"the tokenizer's <im_patch> id {patch_id}",
            )
        action_stats, state_stats, metadata = load_norm_stats(checkpoint_dir, norm_tag)
        n_obs_steps = require_single_obs(config)
        return cls(
            trunk=load_trunk(checkpoint_dir, device=device, dtype=dtype),
            expert=load_action_expert(
                checkpoint_dir,
                config,
                device=device,
                dtype=dtype,
            ),
            tokenizer=tokenizer,
            action_stats=action_stats,
            state_stats=state_stats,
            metadata=metadata,
            image_token_ids=image_token_ids,
            action_mode=str(config.get("action_mode", "continuous")),
            eos_token_id=(
                None
                if config.get("eos_token_id") is None
                else int(config["eos_token_id"])
            ),
            action_start_token_id=(
                None
                if config.get("action_start_token_id") is None
                else int(config["action_start_token_id"])
            ),
            action_end_token_id=(
                None
                if config.get("action_end_token_id") is None
                else int(config["action_end_token_id"])
            ),
            max_action_horizon=_positive_int(
                config.get("max_action_horizon"),
                default=1,
                what="max_action_horizon",
            ),
            max_action_dim=int(config["max_action_dim"]),
            n_obs_steps=n_obs_steps,
            num_state_tokens=int(config["num_state_tokens"]),
            flow_matching_num_steps=int(config["flow_matching_num_steps"]),
            mask_action_dim_padding=bool(config["mask_action_dim_padding"]),
            action_token_start_id=(
                None
                if config.get("action_token_start_id") is None
                else int(config["action_token_start_id"])
            ),
            fast_codec=(
                None
                if fast_tokenizer is None
                else MolmoAct2FastTokenizer.load(fast_tokenizer)
            ),
        )

    @property
    def device(self) -> torch.device:
        return self.expert.action_embed.weight.device

    def batch_inputs(
        self,
        images: list,
        task: str,
        state: Tensor,
        *,
        normalize_language: bool = True,
    ) -> dict[str, Tensor]:
        """Pack one observation (item 2) and collate it to the batch-1
        trunk inputs: per-image pooled indices shift into the sample's
        concatenated view grid exactly like ``Molmo2InputsCollator``.

        Shapes:
        - ``images``: per camera [H, W, 3]-coercible frames
        - ``state``: [D] raw joint units
        - returns: input_ids [1, S], crops [1, V, 729, 588],
          pooled_patches_idx [1, V * 196, 4], image_type_mask [1, S]
        """
        pack = pack_action_example(
            images=images,
            state=state,
            task=task,
            tokenizer=self.tokenizer,
            state_stats=self.state_stats,
            setup_type=str(self.metadata["setup_type"]),
            control_mode=str(self.metadata["control_mode"]),
            normalize_language=normalize_language,
            num_state_tokens=self.num_state_tokens,
        )
        crops = torch.cat([image.crops for image in pack.images], dim=0)
        pooled: list[Tensor] = []
        crop_base = 0
        for image in pack.images:
            idx = image.pooled_idx
            pooled.append(torch.where(idx >= 0, idx + crop_base, idx))
            crop_base += image.crops.shape[0] * image.crops.shape[1]
        input_ids = pack.input_ids[None]
        ensure_per_sample_patch_alignment(
            input_ids,
            torch.cat(pooled, dim=0)[None],
            image_patch_id=self.trunk.image_patch_id,
        )
        input_ids = input_ids.to(self.device)
        image_ids = torch.tensor(
            sorted(self.image_token_ids),
            dtype=torch.long,
            device=self.device,
        )
        return {
            "input_ids": input_ids,
            "crops": crops[None].to(self.device),
            "pooled_patches_idx": torch.cat(pooled, dim=0)[None].to(self.device),
            "image_type_mask": torch.isin(input_ids, image_ids),
        }

    @torch.no_grad()
    def prompt_kv(
        self,
        inputs: dict[str, Tensor],
    ) -> tuple[list[tuple[Tensor, Tensor]], Tensor]:
        """Run the trunk prompt forward (vision inject + causal-OR-image
        mask), retain the cache, and return (per-layer KV states, encoder
        attention mask) for the expert.

        Shapes:
        - ``inputs``: the ``batch_inputs`` dict (see its docstring)
        - returns: per-layer ([1, S, kv_dim], same) pairs and a [1, S]
          bool encoder mask
        """
        embeds = self.trunk.build_input_embeddings(
            inputs["input_ids"],
            crops=inputs["crops"],
            pooled_patches_idx=inputs["pooled_patches_idx"],
        )
        mask = build_multimodal_mask(
            image_type_mask=inputs["image_type_mask"],
            padding_mask=None,
            dtype=embeds.dtype,
            device=embeds.device,
        )
        cache = Molmo2KVCache(len(self.trunk.text.transformer.blocks))
        self.trunk.text.transformer(
            inputs_embeds=embeds,
            attention_mask=mask,
            cache=cache,
        )
        text_config = self.trunk.text.config
        kv_states = extract_kv_states(
            cache,
            num_expert_blocks=len(self.expert.blocks),
            num_attention_heads=text_config.num_attention_heads,
            num_key_value_heads=text_config.num_key_value_heads,
        )
        enc_mask = encoder_attention_mask(
            inputs["input_ids"],
            None,
            action_mode=self.action_mode,
            eos_token_id=self.eos_token_id,
            action_start_token_id=self.action_start_token_id,
            action_end_token_id=self.action_end_token_id,
        )
        assert enc_mask is not None  # input_ids given — the != -1 branch
        return kv_states, enc_mask

    @torch.no_grad()
    def predict_action(
        self,
        *,
        images: list,
        task: str,
        state: Tensor,
        num_steps: int | None = None,
        generator: torch.Generator | None = None,
        normalize_language: bool = True,
    ) -> Tensor:
        """One observation -> the executed action chunk — their
        ``predict_action`` continuous path end-to-end.

        Shapes:
        - ``images``: per camera [H, W, 3]-coercible frames
        - ``state``: [D] raw joint units
        - returns: [1, n_action_steps, action_dim] fp32, CPU, joint units
        """
        inputs = self.batch_inputs(
            images,
            task,
            state,
            normalize_language=normalize_language,
        )
        kv_states, enc_mask = self.prompt_kv(inputs)

        action_dim = int(self.action_stats.q01.numel())
        if action_dim > self.max_action_dim:
            raise ValueError(
                f"tag action dim {action_dim} exceeds max_action_dim "
                f"{self.max_action_dim}",
            )
        action_horizon = _positive_int(
            self.metadata.get("action_horizon"),
            default=self.max_action_horizon,
            what="action_horizon",
        )
        if action_horizon > self.max_action_horizon:
            raise ValueError(
                f"tag action_horizon {action_horizon} exceeds checkpoint "
                f"max_action_horizon {self.max_action_horizon}",
            )
        n_action_steps = _positive_int(
            self.metadata.get("n_action_steps"),
            default=action_horizon,
            what="n_action_steps",
        )
        if n_action_steps > action_horizon:
            raise ValueError(
                f"n_action_steps {n_action_steps} exceeds action_horizon "
                f"{action_horizon}",
            )
        action_dim_is_pad = None
        if action_dim < self.max_action_dim:
            action_dim_is_pad = torch.ones(
                (1, self.max_action_dim),
                dtype=torch.bool,
                device=self.device,
            )
            action_dim_is_pad[:, :action_dim] = False

        chunk = generate_actions(
            self.expert,
            encoder_kv_states=kv_states,
            encoder_attention_mask=enc_mask,
            action_horizon=action_horizon,
            action_dim_is_pad=action_dim_is_pad,
            num_steps=(
                self.flow_matching_num_steps if num_steps is None else int(num_steps)
            ),
            mask_action_dim_padding=self.mask_action_dim_padding,
            generator=generator,
        )

        # Their output tail, in their order: width slice, n_obs_steps
        # chunk slice, clamp + q01/q99 unnormalize in fp32, then the
        # reference's cast back to the sampled dtype before fp32.
        actions = chunk[..., :action_dim]
        start = self.n_obs_steps - 1
        if start + n_action_steps > actions.shape[1]:
            raise ValueError(
                f"chunk rows {start}..{start + n_action_steps} exceed the "
                f"generated horizon {actions.shape[1]}",
            )
        actions = actions[:, start : start + n_action_steps].cpu()
        unnormalized = unnormalize_action(actions, self.action_stats)
        return unnormalized.to(actions.dtype).to(torch.float32)

    @torch.no_grad()
    def predict_action_discrete(
        self,
        *,
        images: list,
        task: str,
        state: Tensor,
        normalize_language: bool = True,
        on_undecodable: str = "raise",
        grammar_masked: bool = False,
        temperature: float | None = None,
        sample_rng: np.random.Generator | None = None,
        action_capture: list[ActionCaptureStep] | None = None,
    ) -> DiscreteActionResult:
        """One observation -> the executed action chunk through the
        checkpoint's TRAINED DISCRETE pathway — their ``predict_action``
        ``inference_action_mode='discrete'`` branch: same prompt as the
        continuous path, then UNCONSTRAINED greedy argmax over the full
        vocabulary until the EOS id (cap ``action_horizon × 16``,
        loud if never hit — the reference raises too), span-extraction
        between ``<action_start>``/``<action_end>`` (tolerant: missing
        markers widen the span to the whole emission; non-action ids
        inside are dropped), OpenFAST decode, their output tail (dim
        slice, ``n_obs_steps`` chunk slice, q01/q99 unnormalize, fp32).

        ``on_undecodable``: ``"raise"`` (default — instrumentation
        wants loud) or ``"zeros"`` (the reference's silent fallback:
        a non-decodable emission becomes a zero NORMALIZED chunk, then
        unnormalizes like any other — parity/eval callers match their
        deployed semantics with this).

        ``grammar_masked`` (the RL decode mode, item (c)): the
        ar_backbone scaffold discipline grafted onto their vocab —
        ``<action_start>`` is FED (a scaffold constant, not a
        decision), every bin step argmaxes over the action block under
        the budget-arithmetic legality mask (piece symbol-length fits
        the remaining T×D budget), and ``<action_end>`` closes the
        stream at budget 0. Every emission decodes by construction —
        no EOS race, no zeros fallback, no cap. Wherever the
        unconstrained argmax was already a legal bin the two modes
        emit identical streams; ``masked_violations`` counts the bin
        steps where the unconstrained full-vocab argmax was NOT a
        legal bin (under greedy exactly the steps where the two modes
        diverge — the record-only divergence instrument; under
        sampling the same mask-binding count, no longer a stream
        diff).

        ``temperature`` + ``sample_rng`` (the RL rollout draw, phase-2
        instrument item 3): each bin step samples the masked softmax
        ``softmax((block/T) | legality mask)`` via Gumbel-max with
        fp32 Exp(1) noise from the caller's keyed CPU generator —
        bit-reproducible under its key, exactly
        :func:`~bijou.decoders.ar_backbone._sample_action_ids`'s
        scheme. Sampling requires ``grammar_masked`` (the RL decode IS
        the masked decode — unconstrained sampling would sample the
        zeros-fallback class) and an explicit generator (no ambient
        RNG).

        ``action_capture`` (requires ``grammar_masked``): appends one
        :class:`ActionCaptureStep` per bin step — pre-mask BLOCK
        logits (fp32, the ``block_vocab`` slice), the applied legality
        mask, the chosen backbone id — so
        ``token_rows_from_capture(capture,
        block_base=action_token_start_id, temperature=...)`` yields
        the TokenRow records the replay collator trains from. Capture
        is observation, never intervention.

        Shapes:
        - ``images``: per camera [H, W, 3]-coercible frames
        - ``state``: [D] raw joint units
        - returns actions [1, n_action_steps, action_dim] fp32, CPU,
          joint units; token_ids = the raw emission (EOS included when
          hit; masked mode: ``[<action_start>, bins, <action_end>]``);
          bins = the extracted codec ids
        """
        if on_undecodable not in ("raise", "zeros"):
            raise ValueError(
                f"on_undecodable must be 'raise' or 'zeros', got {on_undecodable!r}",
            )
        if temperature is not None and temperature <= 0.0:
            raise ValueError(f"temperature {temperature} must be positive")
        if (temperature is None) != (sample_rng is None):
            raise ValueError(
                "sampled decode takes temperature AND sample_rng together "
                "— the draw must be explicitly keyed (no ambient RNG), and "
                "a generator without a temperature has nothing to consume",
            )
        if (temperature is not None or action_capture is not None) and (
            not grammar_masked
        ):
            raise ValueError(
                "temperature/action_capture require grammar_masked — the "
                "RL rollout decode is the masked decode (unconstrained "
                "sampling would sample the zeros-fallback class, and the "
                "capture surface records the masked-softmax distribution)",
            )
        if self.action_mode not in ("discrete", "both"):
            raise ValueError(
                "the discrete pathway requires checkpoint action_mode in "
                f"{{'discrete', 'both'}}, got {self.action_mode!r}",
            )
        codec = self.fast_codec
        if codec is None:
            raise ValueError(
                "no FAST codec attached — load(..., fast_tokenizer=<dir>) "
                "with the released MolmoAct2-FAST-Tokenizer artifact",
            )
        if (
            self.eos_token_id is None
            or self.action_start_token_id is None
            or self.action_end_token_id is None
            or self.action_token_start_id is None
        ):
            raise ValueError(
                "discrete generation requires eos/action_start/action_end/"
                "action_token_start ids in the converted config",
            )
        lm_head = self.trunk.text.lm_head
        assert lm_head is not None  # Molmo2Model requires the full decoder

        inputs = self.batch_inputs(
            images,
            task,
            state,
            normalize_language=normalize_language,
        )
        action_dim = int(self.action_stats.q01.numel())
        action_horizon = _positive_int(
            self.metadata.get("action_horizon"),
            default=self.max_action_horizon,
            what="action_horizon",
        )
        n_action_steps = _positive_int(
            self.metadata.get("n_action_steps"),
            default=action_horizon,
            what="n_action_steps",
        )

        # Prompt prefill: the continuous path's trunk forward (vision
        # inject + causal-OR-image mask), cache retained for the
        # incremental continuation.
        embeds = self.trunk.build_input_embeddings(
            inputs["input_ids"],
            crops=inputs["crops"],
            pooled_patches_idx=inputs["pooled_patches_idx"],
        )
        mask = build_multimodal_mask(
            image_type_mask=inputs["image_type_mask"],
            padding_mask=None,
            dtype=embeds.dtype,
            device=embeds.device,
        )
        transformer = self.trunk.text.transformer
        cache = Molmo2KVCache(len(transformer.blocks))
        hidden = transformer(
            inputs_embeds=embeds,
            attention_mask=mask,
            cache=cache,
        )
        logits = lm_head(hidden[:, -1:])

        prompt_length = int(inputs["input_ids"].shape[1])
        device = embeds.device

        def advance(next_id: Tensor, step: int) -> Tensor:
            positions = torch.full(
                (1, 1),
                prompt_length + step,
                dtype=torch.long,
                device=device,
            )
            fed = transformer(
                inputs_embeds=transformer.wte(next_id[:, None]),
                position_ids=positions,
                cache=cache,
            )
            return lm_head(fed)

        generated: list[Tensor] = []
        masked_violations: int | None = None
        if grammar_masked:
            # The RL decode: scaffold fed, bins budget-masked — every
            # stream consumes exactly T×D symbols and decodes.
            base = self.action_token_start_id
            lengths = torch.from_numpy(codec.symbol_lengths).to(device)
            remaining = action_horizon * action_dim
            masked_violations = 0
            opener = torch.full(
                (1,),
                self.action_start_token_id,
                dtype=torch.long,
                device=device,
            )
            generated.append(opener)
            logits = advance(opener, 0)
            step = 1
            while remaining > 0:
                row = logits[0, -1].float()
                legal = (lengths > 0) & (lengths <= remaining)
                block = row[base : base + codec.block_vocab]
                if temperature is None:
                    choice = int(
                        torch.argmax(block.masked_fill(~legal, float("-inf"))),
                    )
                else:
                    assert sample_rng is not None  # guarded above
                    # Gumbel-max over the masked softmax: argmax of
                    # (block/T | mask) + G with G = -log(Exp(1)) fp32
                    # from the keyed CPU generator — _sample_action_ids'
                    # scheme, one stream per predict.
                    exponential = sample_rng.standard_exponential(
                        codec.block_vocab,
                        dtype=np.float32,
                    )
                    gumbel = -np.log(
                        np.maximum(exponential, np.finfo(np.float32).tiny),
                    )
                    scaled = (block / temperature).masked_fill(
                        ~legal,
                        float("-inf"),
                    )
                    choice = int(
                        torch.argmax(
                            scaled + torch.from_numpy(gumbel).to(scaled.device),
                        ),
                    )
                full_argmax = int(torch.argmax(row))
                if not (
                    base <= full_argmax < base + codec.block_vocab
                    and bool(legal[full_argmax - base])
                ):
                    masked_violations += 1
                if action_capture is not None:
                    action_capture.append(
                        ActionCaptureStep(
                            block_logits=block.detach().float().cpu()[None],
                            allowed=legal.cpu()[None],
                            active=torch.ones(1, dtype=torch.bool),
                            chosen=torch.tensor([base + choice]),
                        ),
                    )
                remaining -= int(lengths[choice])
                chosen = torch.full(
                    (1,),
                    base + choice,
                    dtype=torch.long,
                    device=device,
                )
                generated.append(chosen)
                if remaining > 0:
                    logits = advance(chosen, step)
                    step += 1
            generated.append(
                torch.full(
                    (1,),
                    self.action_end_token_id,
                    dtype=torch.long,
                    device=device,
                ),
            )
        else:
            # Their _continue_discrete_generation_from_output verbatim:
            # greedy argmax over the FULL vocabulary, EOS-terminated,
            # loud when the cap is hit without EOS.
            max_steps = max(1, action_horizon * 16)
            hit_end = False
            for step in range(max_steps):
                next_id = torch.argmax(logits[:, -1, :], dim=-1)
                generated.append(next_id)
                if bool((next_id == self.eos_token_id).all()):
                    hit_end = True
                    break
                logits = advance(next_id, step)
            if not hit_end:
                raise RuntimeError(
                    f"discrete continuation did not emit EOS "
                    f"{self.eos_token_id} within {max_steps} steps — the "
                    "reference raises here too",
                )
        token_ids = torch.stack(generated, dim=1)

        bins = extract_action_bins(
            [int(i) for i in token_ids[0].tolist()],
            action_start_id=self.action_start_token_id,
            action_end_id=self.action_end_token_id,
            action_token_start_id=self.action_token_start_id,
            block_vocab=codec.block_vocab,
        )
        try:
            normalized = torch.from_numpy(
                codec.decode(
                    bins,
                    time_horizon=action_horizon,
                    action_dim=action_dim,
                ),
            ).to(torch.float32)[None]
        except ValueError:
            if on_undecodable == "raise":
                raise
            normalized = torch.zeros(
                (1, action_horizon, action_dim),
                dtype=torch.float32,
            )

        # Their discrete output tail: dim slice (decode already emits
        # the tag width), n_obs_steps chunk slice, q01/q99 unnormalize,
        # fp32 throughout (no bf16 round trip on this path).
        start = self.n_obs_steps - 1
        if start + n_action_steps > normalized.shape[1]:
            raise ValueError(
                f"chunk rows {start}..{start + n_action_steps} exceed the "
                f"generated horizon {normalized.shape[1]}",
            )
        actions = unnormalize_action(
            normalized[:, start : start + n_action_steps].cpu(),
            self.action_stats,
        ).to(torch.float32)
        return DiscreteActionResult(
            actions=actions,
            token_ids=token_ids.cpu(),
            bins=bins,
            masked_violations=masked_violations,
        )


@dataclass(frozen=True, slots=True)
class DiscreteActionResult:
    """``predict_action_discrete``'s full record: the executed chunk
    plus the raw emission the parity harness and the RL instrument
    both need (token_ids includes the terminating EOS when hit).
    ``masked_violations`` is None on the unconstrained (reference)
    mode; under ``grammar_masked`` it counts the bin steps where the
    unconstrained full-vocab argmax differed from the masked choice
    (0 ⟺ the two modes emitted identical streams)."""

    actions: Tensor  # [1, n_action_steps, action_dim] fp32 CPU
    token_ids: Tensor  # [1, K] long CPU
    bins: list[int]
    masked_violations: int | None = None


def extract_action_bins(
    token_ids: list[int],
    *,
    action_start_id: int,
    action_end_id: int,
    action_token_start_id: int,
    block_vocab: int,
) -> list[int]:
    """Their ``_extract_discrete_token_bins`` verbatim, tolerant by
    design: the span opens after the FIRST ``<action_start>`` (or at 0
    when absent), closes at the first ``<action_end>`` after it (or at
    the end when absent), and every id inside that is not an
    ``<action_i>`` row is silently dropped — exactly the reference's
    filter, so parity holds on malformed emissions too."""
    start_index = None
    end_index = None
    for index, token_id in enumerate(token_ids):
        if token_id == action_start_id:
            start_index = index
            break
    if start_index is not None:
        for index in range(start_index + 1, len(token_ids)):
            if token_ids[index] == action_end_id:
                end_index = index
                break
    span_start = 0 if start_index is None else start_index + 1
    span_end = len(token_ids) if end_index is None else end_index
    return [
        token_id - action_token_start_id
        for token_id in token_ids[span_start:span_end]
        if 0 <= token_id - action_token_start_id < block_vocab
    ]
