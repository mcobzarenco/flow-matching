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

import torch
from safetensors import safe_open
from torch import Tensor

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
from .processing import (
    QuantileStats,
    load_norm_stats,
    pack_action_example,
    unnormalize_action,
)
from .wiring import (
    encoder_attention_mask,
    extract_kv_states,
    generate_actions,
    validate_inference_config,
)

#: Their processor's ``IMAGE_TOKENS`` membership strings (the
#: ``token_type_ids`` set), resolved per checkpoint through its
#: tokenizer — see module docstring for why ids cannot be pinned.
IMAGE_TOKEN_STRINGS = (
    "<im_patch>",
    "<im_col>",
    "<im_start>",
    "<low_res_im_start>",
    "<frame_start>",
    "<im_end>",
    "<frame_end>",
    "<im_low>",
)


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


def require_single_obs(config: dict[str, Any]) -> int:
    """Guard a checkpoint config's ``n_obs_steps``: this port packs
    exactly ONE observation per prompt, so only 1 is loadable.

    Refuses a MISSING key too, loudly: their HF config class defaults to
    30 while training used 1 — under their reference a missing key
    silently shifts chunk slicing to start at index 29, and silently
    picking either side of that divergence is worse than stopping."""
    value = config.get("n_obs_steps")
    if value is None or int(value) != 1:
        raise NotImplementedError(
            f"n_obs_steps={value!r}: this port packs exactly one observation "
            "per prompt (all released/rig checkpoints ship 1). A missing key "
            "is refused rather than defaulted — their HF config class "
            "defaults to 30, which shifts the chunk slice to index 29",
        )
    return 1


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

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        norm_tag: str,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
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
