"""MolmoAct2 action-expert fine-tuning in OUR repo (port item 4).

Pre-reg posts/2026-08-10-prereg-molmoact2-firstclass-port.md, gate G4:
reproduce the rung-1 rig fine-tune result class (their trainer, run
``fontaine_so101_rig_ae_r1``, 2026-08-10) with zero patches against
their checkout — the whole path runs on the first-class port
(items 1–3: ``action_expert``/``wiring``/``processing``/``predictor``).

Their training semantics, mirrored off ``olmo/models/molmoact2/
molmoact2.py`` + ``olmo/train`` at the launcher's flag set
(``ft_action_expert=true`` only, ``action_format=continuous``,
``state_format=discrete``; facts pinned 2026-08-11):

- Trainable set = the action expert minus ``state_encoder``/
  ``state_norm`` (their ``freeze_continuous_state_conditioning`` under
  discrete state) and the per-block ``cross_attn.kv_proj`` compat
  tensors (created frozen). Trunk fully frozen, no LM/CE loss (their
  prompt-only examples carry zero loss tokens).
- Loss = flow-matching MSE only: ``t = offset + scale * Beta(alpha,
  beta)`` per example (this checkpoint: 0.001 + 0.999*Beta(1.0, 1.5)),
  ``x_t = (1-t)*noise + t*actions``, target ``actions - noise``, MSE in
  fp32, mean over the valid action dims (padded dims zeroed on both
  actions and noise, excluded from the mean), then mean over
  [batch, horizon].
- Actions/state normalize with ONE shared rig-only q01/q99 table
  (``--norm-stats``: the merged-over-repos stats their launcher
  computed; the same table the predictor uses at read time) and clamp
  to [-1, 1]. Episode-end repeated actions train as real targets (their
  wrapper tracks only max-horizon padding — none at horizon 30/30;
  bijou's own flow loss makes the same call).
- Optimizer AdamW(lr, betas (0.9, 0.95), eps 1e-6, weight_decay 0),
  linear warmup 200 steps from 0 then cosine to ``alpha_f=0.1`` of
  peak at the final step; their increment-before-step convention makes
  the first step run at ``peak/warmup``. Grad clip 1.0 (L2, over the
  trainable set); a non-finite grad norm skips the step. One step =
  ``--global-batch`` examples as ``global/micro`` accumulated
  micro-batches.
- Image augmentation (their ``img_aug=full``, reproduced op-for-op on
  the uint8 PIL path): random 95%-per-side crop -> bilinear resize
  back, rotation U(-5, 5), ColorJitter(0.2, (0.8, 1.2), (0.8, 1.2),
  0.05), 20% GaussianBlur(5, (0.1, 1.0)).

Named deltas vs their run (execution note, 2026-08-11): the frozen
trunk runs DETERMINISTIC — their trainer leaves ``model.train()`` on
with ``llm.residual_dropout=0.1``, so their conditioning KV was
stochastic; ours is our standard frozen-trunk recipe. Dataset mixing is
sqrt-weighted per-frame WITH replacement (theirs: sqrt-weighted source
draw over per-epoch shuffles). Both are regularization-class deltas;
the G4 corridor + anchor reads judge them.

Saves land every ``--save-every`` steps as predictor-consumable step
dirs: ``model.action_expert.*``-prefixed bf16 safetensors +
``config.json`` + the rig ``norm_stats.json`` (trunk/tokenizer stay in
the init checkpoint — the rung read composes them).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, override

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bijou.gemma4.loading import resolve_checkpoint_dir
from bijou.molmo2.cache import Molmo2KVCache
from bijou.molmo2.model import Molmo2Model, build_multimodal_mask
from bijou.molmo2.model import load_model as load_trunk
from bijou.molmo2.tokenizer import Molmo2TextTokenizer
from bijou.molmoact2.action_expert import ActionExpert
from bijou.molmoact2.predictor import (
    load_action_expert,
    resolve_image_token_ids,
)
from bijou.molmoact2.processing import (
    QuantileStats,
    load_norm_stats,
    normalize_state,
    pack_action_example,
)
from bijou.molmoact2.wiring import encoder_attention_mask, extract_kv_states

#: Their AE-side freeze set under discrete state: substrings of
#: parameter names that stay frozen (``freeze_continuous_state_
#: conditioning`` + the construction-frozen kv_proj compat tensors).
FROZEN_NAME_PARTS = ("state_encoder", "state_norm", "cross_attn.kv_proj")

#: The reference run's flow-matching sampling params (checkpoint
#: config.json) — asserted at startup so a differently-configured
#: checkpoint fails loud, not silently off-recipe.
REFERENCE_FLOW_PARAMS = {
    "flow_matching_time_offset": 0.001,
    "flow_matching_time_scale": 0.999,
    "flow_matching_cutoff": 1.0,
    "flow_matching_beta_alpha": 1.0,
    "flow_matching_beta_beta": 1.5,
}


def cosine_warmup_lambda(
    step: int,
    *,
    warmup: int,
    total: int,
    alpha_f: float,
) -> float:
    """Their ``CosWithWarmup`` as an LR multiplier, with the trainer's
    increment-before-step convention folded in: LambdaLR calls this with
    the 0-based completed-step count, so ``step + 1`` is the global step
    the optimizer step runs at (first step = ``1/warmup`` of peak)."""
    current = step + 1
    if current < warmup:
        return current / warmup
    if current >= total:
        return alpha_f
    progress = (current - warmup) / (total - warmup)
    return alpha_f + (1.0 - alpha_f) * 0.5 * (1.0 + math.cos(math.pi * progress))


def sample_flow_times(
    batch_size: int,
    *,
    offset: float,
    scale: float,
    alpha: float,
    beta: float,
    device: torch.device | str,
) -> Tensor:
    """Their ``_sample_beta_timesteps``: ``t = offset + scale * B`` with
    ``B ~ Beta(alpha, beta)`` i.i.d. per example, fp32 [B], drawn from
    the ambient RNG (theirs too — flow times are not example-seeded)."""
    b = (
        torch.distributions.Beta(alpha, beta)
        .sample(torch.Size((batch_size,)))
        .to(device=device, dtype=torch.float32)
    )
    return offset + scale * b


def flow_matching_loss_sums(
    expert: ActionExpert,
    *,
    kv_states: list[tuple[Tensor, Tensor]],
    enc_mask: Tensor,
    actions_norm: Tensor,  # [B, T, max_action_dim] fp32, normalized+clamped
    action_dim_is_pad: Tensor,  # [B, max_action_dim] bool
    times: Tensor,  # [B] fp32
    noise: Tensor,  # [B, T, max_action_dim] fp32
) -> tuple[Tensor, Tensor]:
    """Their ``_compute_flow_matching_loss`` at ``num_flow_timesteps=1``,
    sum form: (sum over [B, T] of the per-position valid-dim means, with
    graph; position count B*T). Mean form = sum / count; the sum form
    keeps micro-batch accumulation exact under unequal batch sizes."""
    valid = ~action_dim_is_pad  # [B, D]
    dim_mask = valid[:, None, :].to(actions_norm.dtype)  # [B, 1, D]
    actions_masked = actions_norm * dim_mask
    noise_masked = noise * dim_mask
    t = times[:, None, None]  # [B, 1, 1]
    xt = (1.0 - t) * noise_masked + t * actions_masked
    target = actions_masked - noise_masked
    pred = expert(
        xt,
        times,
        encoder_kv_states=kv_states,
        encoder_attention_mask=enc_mask,
    )
    err = F.mse_loss(pred.to(torch.float32), target, reduction="none")
    per_dim = err * dim_mask
    counts = valid.sum(dim=-1).clamp(min=1)[:, None]  # [B, 1]
    per_position = per_dim.sum(dim=-1) / counts  # [B, T]
    count = torch.tensor(
        per_position.numel(),
        device=per_position.device,
        dtype=torch.float32,
    )
    return per_position.sum(), count


class RigAugmenter:
    """Their ``_apply_augmentation`` at ``img_aug='full'``, op-for-op on
    the uint8 PIL path (95%-per-side crop -> resize back, rotation,
    ColorJitter, 20% blur). ``rng`` drives the geometric draws exactly
    like their ``np.random`` handle; ColorJitter/RandomApply draw from
    torch's ambient RNG (their torchvision calls do too)."""

    def __init__(self) -> None:
        from torchvision import transforms

        self._jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=(0.8, 1.2),
            saturation=(0.8, 1.2),
            hue=0.05,
        )
        self._blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))

    def __call__(self, image: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        from PIL import Image
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        pil = Image.fromarray(image)
        h, w = pil.height, pil.width
        crop_h = max(int(h * 0.95), 1)
        crop_w = max(int(w * 0.95), 1)
        top = rng.randint(0, max(h - crop_h + 1, 1))
        left = rng.randint(0, max(w - crop_w + 1, 1))
        # torchvision functional accepts PIL images (their code path);
        # its stubs are Tensor-typed.
        pil = TF.crop(pil, top=top, left=left, height=crop_h, width=crop_w)  # type: ignore[arg-type]
        pil = TF.resize(
            pil,
            size=[h, w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        angle = rng.uniform(-5.0, 5.0)
        pil = TF.rotate(pil, angle=angle, interpolation=InterpolationMode.BILINEAR)
        pil = self._jitter(pil)
        if min(pil.size) > 2 and torch.rand(()) < 0.2:
            pil = self._blur(pil)
        return np.array(pil, dtype=np.uint8)  # writable copy: pack wraps it in a tensor


@dataclasses.dataclass
class PackedTrainExample:
    """One example after worker-side packing (item-2 pipeline + the
    normalized action target)."""

    input_ids: Tensor  # [S] long
    crops: Tensor  # [views, patches, patch_dim] fp32
    pooled_idx: Tensor  # [P, group] long (sample-local, pre-shift)
    actions_norm: Tensor  # [T, max_action_dim] fp32
    action_dim_is_pad: Tensor  # [max_action_dim] bool


class MolmoAct2TrainCollator:
    """Dataset item -> packed example -> left-padded batch, the
    ``Molmo2InputsCollator`` conventions on the item-2 pack (heavy
    state rebuilt lazily in spawned dataloader workers)."""

    def __init__(
        self,
        *,
        checkpoint: str,
        state_stats: QuantileStats,
        action_stats: QuantileStats,
        camera_keys: tuple[str, ...],
        setup_type: str,
        control_mode: str,
        max_action_dim: int,
        num_state_tokens: int,
        augment: bool,
    ) -> None:
        self.checkpoint = checkpoint
        self.state_stats = state_stats
        self.action_stats = action_stats
        self.camera_keys = camera_keys
        self.setup_type = setup_type
        self.control_mode = control_mode
        self.max_action_dim = max_action_dim
        self.num_state_tokens = num_state_tokens
        self.augment = augment
        self._tokenizer: Molmo2TextTokenizer | None = None
        self._image_ids: tuple[int, ...] | None = None
        self._augmenter: RigAugmenter | None = None
        self._rng: np.random.RandomState | None = None

    @override
    def __getstate__(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "_tokenizer": None,
            "_image_ids": None,
            "_augmenter": None,
            "_rng": None,
        }

    def _materialize(self) -> tuple[Molmo2TextTokenizer, tuple[int, ...]]:
        if self._tokenizer is None:
            self._tokenizer = Molmo2TextTokenizer(self.checkpoint)
            self._image_ids = resolve_image_token_ids(self._tokenizer)
            if self.augment:
                self._augmenter = RigAugmenter()
            self._rng = np.random.RandomState(torch.initial_seed() % (2**32 - 1))
        assert self._image_ids is not None
        return self._tokenizer, self._image_ids

    def _pack(self, item: dict[str, Any]) -> PackedTrainExample:
        tokenizer, _ = self._materialize()
        images: list[np.ndarray] = []
        for key in self.camera_keys:
            frame = item[key]  # [3, H, W] float [0, 1]
            array = (
                (frame.clamp(0.0, 1.0) * 255.0)
                .round()
                .to(torch.uint8)
                .permute(1, 2, 0)
                .numpy()
            )
            if self._augmenter is not None:
                assert self._rng is not None
                array = self._augmenter(array, self._rng)
            images.append(array)
        pack = pack_action_example(
            images=images,  # type: ignore[arg-type]  # ndarray accepted
            state=item["observation.state"],
            task=str(item["task"]),
            tokenizer=tokenizer,
            state_stats=self.state_stats,
            setup_type=self.setup_type,
            control_mode=self.control_mode,
            num_state_tokens=self.num_state_tokens,
        )
        actions = item["action"].to(torch.float32)  # [T, dim]
        actions_norm = normalize_state(actions, self.action_stats)
        dim = actions_norm.shape[-1]
        if dim > self.max_action_dim:
            raise ValueError(
                f"action dim {dim} exceeds max_action_dim {self.max_action_dim}",
            )
        padded = torch.zeros(
            (actions_norm.shape[0], self.max_action_dim),
            dtype=torch.float32,
        )
        padded[:, :dim] = actions_norm
        dim_is_pad = torch.ones((self.max_action_dim,), dtype=torch.bool)
        dim_is_pad[:dim] = False
        crops = torch.cat([image.crops for image in pack.images], dim=0)
        pooled: list[Tensor] = []
        crop_base = 0
        for image in pack.images:
            idx = image.pooled_idx
            pooled.append(torch.where(idx >= 0, idx + crop_base, idx))
            crop_base += image.crops.shape[0] * image.crops.shape[1]
        return PackedTrainExample(
            input_ids=pack.input_ids,
            crops=crops,
            pooled_idx=torch.cat(pooled, dim=0),
            actions_norm=padded,
            action_dim_is_pad=dim_is_pad,
        )

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Tensor]:
        _, image_ids = self._materialize()
        from bijou.molmoact2.processing import PAD_ID

        examples = [self._pack(item) for item in items]
        batch = len(examples)
        width = max(e.input_ids.shape[0] for e in examples)
        input_ids = torch.full((batch, width), PAD_ID, dtype=torch.long)
        attention_mask = torch.zeros((batch, width), dtype=torch.long)
        for row, example in enumerate(examples):
            length = example.input_ids.shape[0]
            input_ids[row, width - length :] = example.input_ids
            attention_mask[row, width - length :] = 1
        image_id_tensor = torch.tensor(sorted(image_ids), dtype=torch.long)
        image_type_mask = torch.isin(input_ids, image_id_tensor)
        image_type_mask &= attention_mask.bool()

        max_views = max(e.crops.shape[0] for e in examples)
        max_pooled = max(e.pooled_idx.shape[0] for e in examples)
        patches, patch_dim = examples[0].crops.shape[1:]
        group = examples[0].pooled_idx.shape[1]
        crops = torch.full(
            (batch, max_views, patches, patch_dim),
            -1.0,
            dtype=torch.float32,
        )
        pooled_idx = torch.full((batch, max_pooled, group), -1, dtype=torch.long)
        for row, example in enumerate(examples):
            crops[row, : example.crops.shape[0]] = example.crops
            pooled_idx[row, : example.pooled_idx.shape[0]] = example.pooled_idx
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_type_mask": image_type_mask,
            "crops": crops,
            "pooled_patches_idx": pooled_idx,
            "actions_norm": torch.stack([e.actions_norm for e in examples]),
            "action_dim_is_pad": torch.stack([e.action_dim_is_pad for e in examples]),
        }


@torch.no_grad()
def prompt_kv_batch(
    trunk: Molmo2Model,
    batch: dict[str, Tensor],
    *,
    num_expert_blocks: int,
    action_mode: str,
    eos_token_id: int | None,
    action_start_token_id: int | None,
    action_end_token_id: int | None,
) -> tuple[list[tuple[Tensor, Tensor]], Tensor]:
    """The predictor's ``prompt_kv`` generalized to left-padded batches
    (logical positions + padding-aware multimodal mask); the encoder
    mask reproduces their train-time ``_get_encoder_attention_mask``
    (attention_mask minus EOS positions; span strip is a no-op on these
    prompts — no action tokens present)."""
    attention_mask = batch["attention_mask"]
    has_padding = bool((attention_mask == 0).any())
    padding_mask = attention_mask if has_padding else None
    embeds = trunk.build_input_embeddings(
        batch["input_ids"],
        crops=batch["crops"],
        pooled_patches_idx=batch["pooled_patches_idx"],
    )
    position_ids = (
        Molmo2Model.logical_positions(attention_mask) if has_padding else None
    )
    mask = build_multimodal_mask(
        image_type_mask=batch["image_type_mask"],
        padding_mask=padding_mask,
        dtype=embeds.dtype,
        device=embeds.device,
    )
    cache = Molmo2KVCache(len(trunk.text.transformer.blocks))
    trunk.text.transformer(
        inputs_embeds=embeds,
        position_ids=position_ids,
        attention_mask=mask,
        cache=cache,
    )
    text_config = trunk.text.config
    kv_states = extract_kv_states(
        cache,
        num_expert_blocks=num_expert_blocks,
        num_attention_heads=text_config.num_attention_heads,
        num_key_value_heads=text_config.num_key_value_heads,
    )
    enc_mask = encoder_attention_mask(
        batch["input_ids"],
        attention_mask,
        action_mode=action_mode,
        eos_token_id=eos_token_id,
        action_start_token_id=action_start_token_id,
        action_end_token_id=action_end_token_id,
    )
    assert enc_mask is not None  # input_ids given
    return kv_states, enc_mask


def trainable_parameters(expert: ActionExpert) -> list[tuple[str, nn.Parameter]]:
    """Named params of their trainable set (module docstring); loud if
    the freeze left nothing or everything trainable."""
    named = [
        (name, param)
        for name, param in expert.named_parameters()
        if param.requires_grad
    ]
    if not named or len(named) == len(list(expert.parameters())):
        raise SystemExit(
            f"freeze surface broken: {len(named)} trainable of "
            f"{len(list(expert.parameters()))} params",
        )
    return named


def apply_reference_freeze(expert: ActionExpert) -> None:
    """requires_grad routing for their trainable set: everything on,
    then the discrete-state freeze set off."""
    expert.requires_grad_(True)
    for name, param in expert.named_parameters():
        if any(part in name for part in FROZEN_NAME_PARTS):
            param.requires_grad_(False)


def save_step_dir(
    save_dir: Path,
    step: int,
    expert: ActionExpert,
    *,
    config_path: Path,
    norm_stats_path: Path,
) -> Path:
    """One predictor-consumable step dir, written atomically:
    ``model.action_expert.*``-prefixed bf16 tensors (their export dtype)
    + config.json + the rig norm_stats.json."""
    from safetensors.torch import save_file

    step_dir = save_dir / f"step_{step:06d}"
    tmp_dir = save_dir / f".tmp_step_{step:06d}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    state = {
        f"model.action_expert.{name}": tensor.detach().to(torch.bfloat16).cpu()
        for name, tensor in expert.state_dict().items()
    }
    save_file(state, str(tmp_dir / "action_expert.safetensors"))
    shutil.copy2(config_path, tmp_dir / "config.json")
    shutil.copy2(norm_stats_path, tmp_dir / "norm_stats.json")
    if step_dir.exists():
        shutil.rmtree(step_dir)
    tmp_dir.rename(step_dir)
    return step_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MolmoAct2 action-expert fine-tuning in OUR repo (port item 4)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="init checkpoint (HF id or dir)",
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        required=True,
        help="rig norm_stats.json",
    )
    parser.add_argument("--norm-tag", default="so100_so101_molmoact2")
    parser.add_argument("--train-data", type=Path, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--global-batch", type=int, default=64)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--alpha-f", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--img-aug", choices=["full", "none"], default="full")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.global_batch % args.micro_batch != 0:
        raise SystemExit("--global-batch must be a multiple of --micro-batch")
    return args


def main(argv: list[str] | None = None) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint)
    config = json.loads((checkpoint_dir / "config.json").read_text())
    for key, expected in REFERENCE_FLOW_PARAMS.items():
        actual = float(config[key])
        if actual != expected:
            raise SystemExit(
                f"config {key}={actual} != reference {expected} — this "
                "trainer pins the reference run's sampling law",
            )
    action_stats, state_stats, metadata = load_norm_stats(
        args.norm_stats.parent,
        args.norm_tag,
    )
    camera_keys = tuple(str(k) for k in metadata["camera_keys"])
    horizon = int(metadata["action_horizon"])
    max_action_dim = int(config["max_action_dim"])

    fps_expected = 30
    datasets = []
    for data_dir in args.train_data:
        info = json.loads((data_dir / "meta" / "info.json").read_text())
        if int(info["fps"]) != fps_expected:
            raise SystemExit(f"{data_dir} fps {info['fps']} != {fps_expected}")
        repo_id = f"{data_dir.parent.name}/{data_dir.name}"
        datasets.append(
            LeRobotDataset(
                repo_id,
                root=str(data_dir),
                delta_timestamps={
                    "action": [i / fps_expected for i in range(horizon)],
                },
                tolerance_s=0.5 / fps_expected,
            ),
        )
    concat = ConcatDataset(datasets)
    # Their sqrt-weighted source draw, flattened to per-frame weights
    # (uniform within a dataset).
    weights = torch.cat(
        [torch.full((len(ds),), math.sqrt(len(ds)) / len(ds)) for ds in datasets],
    )
    sampler = WeightedRandomSampler(
        weights.double(),  # type: ignore[arg-type]
        num_samples=args.steps * args.global_batch,
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    collator = MolmoAct2TrainCollator(
        checkpoint=str(checkpoint_dir),
        state_stats=state_stats,
        action_stats=action_stats,
        camera_keys=camera_keys,
        setup_type=str(metadata.get("setup_type", "") or ""),
        control_mode=str(metadata.get("control_mode", "") or ""),
        max_action_dim=max_action_dim,
        num_state_tokens=int(config["num_state_tokens"]),
        augment=args.img_aug == "full",
    )
    loader = DataLoader(
        concat,
        batch_size=args.micro_batch,
        sampler=sampler,
        num_workers=args.num_workers,
        prefetch_factor=(args.prefetch_factor if args.num_workers else None),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collator,
        drop_last=True,
    )

    trunk = load_trunk(checkpoint_dir, device=device, dtype=torch.bfloat16)
    trunk.eval()
    trunk.requires_grad_(False)
    expert = load_action_expert(checkpoint_dir, config, device="cpu", dtype=None)
    expert = expert.to(torch.float32).to(device)
    expert.train()
    apply_reference_freeze(expert)
    trainable = trainable_parameters(expert)
    total_params = sum(p.numel() for p in expert.parameters())
    trainable_params = sum(p.numel() for _, p in trainable)
    optimizer = torch.optim.AdamW(
        [p for _, p in trainable],
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-6,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup_lambda(
            step,
            warmup=args.warmup_steps,
            total=args.steps,
            alpha_f=args.alpha_f,
        ),
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.save_dir / "train_log.jsonl"
    micro_per_step = args.global_batch // args.micro_batch
    action_mode = str(config.get("action_mode", "continuous"))
    eos_id = None if config.get("eos_token_id") is None else int(config["eos_token_id"])
    banner = {
        "checkpoint": str(checkpoint_dir),
        "norm_stats": str(args.norm_stats),
        "norm_tag": args.norm_tag,
        "datasets": {
            f"{d.parent.name}/{d.name}": len(ds)
            for d, ds in zip(args.train_data, datasets, strict=True)
        },
        "sampling_pct": [
            round(
                100 * math.sqrt(len(ds)) / sum(math.sqrt(len(x)) for x in datasets),
                1,
            )
            for ds in datasets
        ],
        "camera_keys": camera_keys,
        "horizon": horizon,
        "max_action_dim": max_action_dim,
        "action_mode": action_mode,
        "params_total": total_params,
        "params_trainable": trainable_params,
        "steps": args.steps,
        "global_batch": args.global_batch,
        "micro_per_step": micro_per_step,
        "lr": args.lr,
        "seed": args.seed,
        "img_aug": args.img_aug,
    }
    print(f"molmoact2 AE fine-tune (ours): {json.dumps(banner)}", flush=True)

    data_iter = iter(loader)
    start_time = time.monotonic()
    window_loss: list[float] = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(micro_per_step):
            batch = next(data_iter)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
                kv_states, enc_mask = prompt_kv_batch(
                    trunk,
                    batch,
                    num_expert_blocks=len(expert.blocks),
                    action_mode=action_mode,
                    eos_token_id=eos_id,
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
                )
                actions_norm = batch["actions_norm"]
                times = sample_flow_times(
                    actions_norm.shape[0],
                    offset=REFERENCE_FLOW_PARAMS["flow_matching_time_offset"],
                    scale=REFERENCE_FLOW_PARAMS["flow_matching_time_scale"],
                    alpha=REFERENCE_FLOW_PARAMS["flow_matching_beta_alpha"],
                    beta=REFERENCE_FLOW_PARAMS["flow_matching_beta_beta"],
                    device=device,
                )
                noise = torch.randn_like(actions_norm)
                loss_sum, count = flow_matching_loss_sums(
                    expert,
                    kv_states=kv_states,
                    enc_mask=enc_mask,
                    actions_norm=actions_norm,
                    action_dim_is_pad=batch["action_dim_is_pad"],
                    times=times,
                    noise=noise,
                )
            micro_loss = loss_sum / count
            (micro_loss / micro_per_step).backward()
            step_loss += float(micro_loss.detach()) / micro_per_step
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for _, p in trainable],
            args.grad_clip,
        )
        if not torch.isfinite(grad_norm):
            # Their skip semantics: zero and move on, loudly.
            print(f"[step={step}] non-finite grad norm — step skipped", flush=True)
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            continue
        optimizer.step()
        scheduler.step()
        window_loss.append(step_loss)

        if step % args.log_every == 0 or step == args.steps:
            lr_now = scheduler.get_last_lr()[0]
            mean_loss = sum(window_loss) / max(len(window_loss), 1)
            elapsed = time.monotonic() - start_time
            print(
                f"[step={step}/{args.steps}] "
                f"train/action_flow_loss={mean_loss:.4f} "
                f"lr={lr_now:.3e} elapsed_s={elapsed:.0f}",
                flush=True,
            )
            with log_path.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "step": step,
                            "action_flow_loss": mean_loss,
                            "last_step_loss": step_loss,
                            "lr": lr_now,
                            "grad_norm": float(grad_norm),
                            "elapsed_s": round(elapsed, 1),
                        },
                    )
                    + "\n",
                )
            window_loss = []
        if step % args.save_every == 0:
            save_start = time.monotonic()
            step_dir = save_step_dir(
                args.save_dir,
                step,
                expert,
                config_path=checkpoint_dir / "config.json",
                norm_stats_path=args.norm_stats,
            )
            print(
                f"saved {step_dir} in {time.monotonic() - save_start:.1f}s",
                flush=True,
            )
    print("training complete", flush=True)


if __name__ == "__main__":
    main()
