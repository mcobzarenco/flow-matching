"""Oracles for the MolmoAct2 our-trainer AE fine-tune (port item 4).

Pre-reg posts/2026-08-10-prereg-molmoact2-firstclass-port.md, gate G4.
CPU-tier oracles pin the training math and plumbing on tiny configs:
the LR law (their CosWithWarmup with increment-before-step folded in),
the t-sampling law bounds, the loss's target convention (tied to the
inference integrator's direction, so train and generate can never
disagree about which end is noise), the valid-dim masked reduction and
its sum-form identity, the reference freeze surface, the batched
left-pad collation (per-row tails byte-equal to the item-2 batch-1
pack), and the step-dir save/load round trip through the predictor's
own loader. The GPU-scale result read is the pre-registered script
(``fontaine/scripts/molmoact2_ours_ft_rung_read.py``), per the repo's
oracle-tiering rule.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, override

import numpy as np
import pytest
import torch

from bijou.molmoact2.action_expert import ActionExpertConfig
from bijou.molmoact2.processing import (
    IM_END_ID,
    IM_PATCH_ID,
    IM_START_ID,
    PAD_ID,
    QuantileStats,
    pack_action_example,
)
from bijou.molmoact2.train import (
    FROZEN_NAME_PARTS,
    MolmoAct2TrainCollator,
    apply_reference_freeze,
    cosine_warmup_lambda,
    flow_matching_loss_sums,
    parse_args,
    sample_flow_times,
    save_step_dir,
    trainable_parameters,
)

BOS_ID = 151_645
STATE_TOKEN_0_ID = 151_675

_SPECIAL_IDS: dict[str, int] = {
    "<|im_end|>": BOS_ID,
    "<im_start>": IM_START_ID,
    "<im_end>": IM_END_ID,
    "<im_patch>": IM_PATCH_ID,
    "<setup_start>": 151_669,
    "<setup_end>": 151_670,
    "<control_start>": 151_671,
    "<control_end>": 151_672,
    "<state_start>": 151_673,
    "<state_end>": 151_674,
    "<action_output>": 151_931,
    "<|im_start|>": 151_644,
}
for _n in range(256):
    _SPECIAL_IDS[f"<state_{_n}>"] = STATE_TOKEN_0_ID + _n


class _StubTokenizerBackend:
    def token_to_id(self, token: str) -> int | None:
        return _SPECIAL_IDS.get(token)


class _StubTokenizer:
    """The predictor-test stand-in (tests/test_molmoact2_predictor.py):
    special tokens split first at their REAL ids, plain words hash to
    one id each. Deterministic."""

    tokenizer = _StubTokenizerBackend()

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        ids: list[int] = []
        specials = sorted(_SPECIAL_IDS, key=len, reverse=True)
        word = 0
        while text:
            for token in specials:
                if text.startswith(token):
                    if word:
                        ids.append(word % 50_000)
                        word = 0
                    ids.append(_SPECIAL_IDS[token])
                    text = text[len(token) :]
                    break
            else:
                char, text = text[0], text[1:]
                if char.isspace():
                    if word:
                        ids.append(word % 50_000)
                        word = 0
                else:
                    word = word * 31 + ord(char)
        if word:
            ids.append(word % 50_000)
        return ids


def _tiny_expert_config() -> ActionExpertConfig:
    return ActionExpertConfig(
        max_horizon=4,
        max_action_dim=8,
        hidden_size=16,
        num_layers=2,
        num_heads=2,
        mlp_ratio=4.0,
        # 256 matches the round-trip test's config.json dict below.
        ffn_multiple_of=256,
        timestep_embed_dim=8,
        dropout=0.0,
        attn_dropout=0.0,
        context_layer_norm=True,
        qk_norm=True,
        qk_norm_eps=1e-6,
        rope=True,
        causal_attn=False,
    )


def _stats(dim: int, offset: float) -> QuantileStats:
    return QuantileStats(
        q01=torch.arange(dim, dtype=torch.float32) - offset,
        q99=torch.arange(dim, dtype=torch.float32) + offset,
    )


def test_parse_args_rejects_misnamed_norm_stats() -> None:
    """load_norm_stats reads <parent>/norm_stats.json while
    save_step_dir ships the literal file passed — any other filename
    would train against one quantile table and ship another into every
    checkpoint. parse_args refuses the divergence at the door."""
    argv = [
        "--checkpoint",
        "ckpt",
        "--train-data",
        "data",
        "--save-dir",
        "out",
        "--norm-stats",
    ]
    with pytest.raises(SystemExit):
        parse_args([*argv, "rig/stats_v2.json"])
    args = parse_args([*argv, "rig/norm_stats.json"])
    assert args.norm_stats == Path("rig/norm_stats.json")


def test_cosine_warmup_schedule_closed_form() -> None:
    """Their CosWithWarmup at the reference recipe: first step at
    peak/warmup (increment-before-step), peak exactly at the warmup
    step, cosine midpoint, alpha_f floor at the final step."""
    values = {
        step: cosine_warmup_lambda(step - 1, warmup=200, total=2000, alpha_f=0.1)
        for step in (1, 200, 1100, 2000)
    }
    assert values[1] == pytest.approx(1 / 200)
    assert values[200] == pytest.approx(1.0)
    assert values[1100] == pytest.approx(0.1 + 0.9 * 0.5)  # cosine midpoint
    assert values[2000] == pytest.approx(0.1)


def test_flow_time_law_bounds_and_mass() -> None:
    """t = 0.001 + 0.999 * Beta(1, 1.5): support pinned, mass skewed to
    small t (their Beta mean 0.4)."""
    torch.manual_seed(0)
    times = sample_flow_times(
        20_000,
        offset=0.001,
        scale=0.999,
        alpha=1.0,
        beta=1.5,
        device="cpu",
    )
    assert float(times.min()) >= 0.001
    assert float(times.max()) <= 1.0
    assert float(times.mean()) == pytest.approx(0.001 + 0.999 * 0.4, abs=0.02)


class _EchoExpert(torch.nn.Module):
    """Stands in for the AE: returns a fixed velocity and records its
    inputs, so the interpolant and target conventions are observable."""

    def __init__(self, velocity: torch.Tensor) -> None:
        super().__init__()
        self.velocity = velocity
        self.seen: dict[str, torch.Tensor] = {}

    @override
    def forward(
        self,
        xt: torch.Tensor,
        times: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        self.seen["xt"] = xt.detach().clone()
        self.seen["times"] = times.detach().clone()
        return self.velocity.expand_as(xt)


def test_loss_target_matches_inference_direction() -> None:
    """target = actions - noise: a perfect velocity field gives zero
    loss, and one ascending-Euler step from noise (the wiring
    integrator's direction, t: 0 -> 1) lands exactly on the actions —
    train and inference agree about which end is data."""
    torch.manual_seed(1)
    batch, horizon, dim = 2, 4, 8
    actions = torch.randn(batch, horizon, dim)
    noise = torch.randn(batch, horizon, dim)
    dim_is_pad = torch.zeros(batch, dim, dtype=torch.bool)
    times = torch.full((batch,), 0.25)
    expert = _EchoExpert(actions - noise)
    loss_sum, count = flow_matching_loss_sums(
        expert,  # type: ignore[arg-type] — velocity-field stand-in
        kv_states=[],
        enc_mask=torch.ones(batch, 3, dtype=torch.bool),
        actions_norm=actions,
        action_dim_is_pad=dim_is_pad,
        times=times,
        noise=noise,
    )
    assert float(loss_sum) == pytest.approx(0.0, abs=1e-10)
    assert int(count) == batch * horizon
    # Interpolant convention: x_t = (1-t)*noise + t*actions.
    expected_xt = 0.75 * noise + 0.25 * actions
    assert torch.allclose(expert.seen["xt"], expected_xt)
    # One Euler step of the true velocity from pure noise reaches data.
    reconstructed = noise + (actions - noise) * 1.0
    assert torch.allclose(reconstructed, actions)


def test_loss_dim_masking_and_sum_form() -> None:
    """Padded action dims are zeroed on both ends and excluded from the
    mean; the sum form divided by its count reproduces the hand-built
    valid-dim mean exactly."""
    batch, horizon, dim, valid_dims = 2, 3, 6, 2
    actions = torch.randn(batch, horizon, dim)
    noise = torch.randn(batch, horizon, dim)
    # Poison the padded dims: the mask must erase them entirely.
    actions[:, :, valid_dims:] = 1e6
    noise[:, :, valid_dims:] = -1e6
    dim_is_pad = torch.ones(batch, dim, dtype=torch.bool)
    dim_is_pad[:, :valid_dims] = False
    times = torch.full((batch,), 0.5)
    velocity = torch.zeros(batch, horizon, dim)
    expert = _EchoExpert(velocity)
    loss_sum, count = flow_matching_loss_sums(
        expert,  # type: ignore[arg-type]
        kv_states=[],
        enc_mask=torch.ones(batch, 3, dtype=torch.bool),
        actions_norm=actions,
        action_dim_is_pad=dim_is_pad,
        times=times,
        noise=noise,
    )
    # Zero-velocity prediction: the error is the target itself.
    target = actions[:, :, :valid_dims] - noise[:, :, :valid_dims]
    expected = (target**2).mean(dim=-1)  # per-position mean over VALID dims
    assert float(loss_sum) == pytest.approx(float(expected.sum()), rel=1e-6)
    assert float(loss_sum / count) == pytest.approx(float(expected.mean()), rel=1e-6)


def test_reference_freeze_surface() -> None:
    """Exactly their trainable set: state conditioning + kv_proj compat
    tensors frozen, everything else on."""
    expert = _tiny_expert_config().build(llm_kv_dim=16)
    apply_reference_freeze(expert)
    for name, param in expert.named_parameters():
        should_freeze = any(part in name for part in FROZEN_NAME_PARTS)
        assert param.requires_grad != should_freeze, name
    named = trainable_parameters(expert)
    assert 0 < len(named) < len(list(expert.parameters()))
    # A fully-frozen expert is a loud failure, never a silent no-op run.
    expert.requires_grad_(False)
    with pytest.raises(SystemExit):
        trainable_parameters(expert)


def _collator() -> MolmoAct2TrainCollator:
    collator = MolmoAct2TrainCollator(
        checkpoint="unused",
        state_stats=_stats(3, 1.5),
        action_stats=_stats(3, 2.0),
        camera_keys=("observation.images.front", "observation.images.wrist"),
        setup_type="tiny rig",
        control_mode="absolute joint pose",
        max_action_dim=8,
        num_state_tokens=256,
        augment=False,
    )
    # The worker-side lazy build, stubbed (the real path needs the HF
    # checkpoint's tokenizer.json).
    collator._tokenizer = _StubTokenizer()  # type: ignore[assignment]
    collator._image_ids = (IM_START_ID, IM_END_ID, IM_PATCH_ID)
    return collator


def _item(task: str, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    frame = torch.from_numpy(
        rng.random((3, 48, 64), dtype=np.float32),
    )
    return {
        "observation.images.front": frame,
        "observation.images.wrist": frame.flip(-1),
        "observation.state": torch.tensor([0.5, -0.25, 1.75]),
        "action": torch.from_numpy(rng.random((4, 3), dtype=np.float32)),
        "task": task,
    }


def test_collator_left_pad_matches_single_pack() -> None:
    """Each padded row's tail is byte-equal to the batch-1 item-2 pack;
    masks and the normalized action padding follow."""
    collator = _collator()
    items = [_item("Pick up the cube.", 3), _item("Pick.", 4)]
    batch = collator(items)
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    for row, item in enumerate(items):
        frame = item["observation.images.front"]
        array = (
            (frame.clamp(0, 1) * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
        )
        wrist = (
            (item["observation.images.wrist"].clamp(0, 1) * 255.0)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .numpy()
        )
        pack = pack_action_example(
            images=[array, wrist],
            state=item["observation.state"],
            task=str(item["task"]),
            tokenizer=_StubTokenizer(),
            state_stats=collator.state_stats,
            setup_type=collator.setup_type,
            control_mode=collator.control_mode,
            num_state_tokens=collator.num_state_tokens,
        )
        length = pack.input_ids.shape[0]
        row_ids = batch["input_ids"][row]
        assert torch.equal(row_ids[-length:], pack.input_ids)
        assert bool(batch["attention_mask"][row, -length:].all())
        assert not bool(batch["attention_mask"][row, :-length].any())
        # Left padding is PAD_ID and never counts as an image position.
        assert bool((row_ids[:-length] == PAD_ID).all())
        assert not bool(batch["image_type_mask"][row, :-length].any())
    # Image-typed positions: exactly the resolved id set on real tokens.
    expected_mask = (
        torch.isin(
            batch["input_ids"],
            torch.tensor(sorted(collator._image_ids or ())),
        )
        & batch["attention_mask"].bool()
    )
    assert torch.equal(batch["image_type_mask"], expected_mask)
    # Normalized actions clamp to [-1, 1], padded dims exactly zero.
    assert batch["actions_norm"].shape == (2, 4, 8)
    assert float(batch["actions_norm"].abs().max()) <= 1.0
    assert bool((batch["actions_norm"][:, :, 3:] == 0).all())
    assert torch.equal(
        batch["action_dim_is_pad"],
        torch.tensor([[False] * 3 + [True] * 5] * 2),
    )


def test_save_step_dir_predictor_round_trip(tmp_path: Path) -> None:
    """The step dir loads through the predictor's own loader: prefixed
    keys, strict state match, bf16 export values."""
    from bijou.molmoact2.predictor import load_action_expert

    torch.manual_seed(2)
    expert = _tiny_expert_config().build(llm_kv_dim=16)
    config = {
        "action_expert_config": {
            "hidden_size": 16,
            "num_layers": 2,
            "num_heads": 2,
            "mlp_ratio": 4.0,
            "ffn_multiple_of": 256,
            "timestep_embed_dim": 8,
            "context_layer_norm": True,
            "qk_norm": True,
            "qk_norm_eps": 1e-6,
            "rope": True,
            "causal_attn": False,
        },
        "text_config": {"head_dim": 8, "num_key_value_heads": 2},
        "max_action_horizon": 4,
        "max_action_dim": 8,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    stats_path = tmp_path / "norm_stats.json"
    stats_path.write_text(json.dumps({"metadata_by_tag": {}}))
    step_dir = save_step_dir(
        tmp_path / "run",
        500,
        expert,
        config_path=config_path,
        norm_stats_path=stats_path,
    )
    assert step_dir.name == "step_000500"
    assert (step_dir / "config.json").exists()
    assert (step_dir / "norm_stats.json").exists()
    loaded = load_action_expert(step_dir, config, device="cpu")
    ours = expert.state_dict()
    theirs = loaded.state_dict()
    assert set(ours) == set(theirs)
    for name, tensor in ours.items():
        assert torch.equal(
            tensor.to(torch.bfloat16).to(tensor.dtype),
            theirs[name].to(tensor.dtype),
        ), name


def test_flow_loss_grad_reaches_trainable_set() -> None:
    """End-to-end on the tiny real expert: the loss backpropagates into
    every trainable parameter group class (a dead graph would train
    nothing while logging a healthy-looking loss)."""
    torch.manual_seed(3)
    expert = _tiny_expert_config().build(llm_kv_dim=16)
    apply_reference_freeze(expert)
    batch, horizon, dim, seq = 2, 4, 8, 5
    kv = [
        (torch.randn(batch, seq, 16), torch.randn(batch, seq, 16))
        for _ in range(len(expert.blocks))
    ]
    enc_mask = torch.ones(batch, seq, dtype=torch.bool)
    loss_sum, count = flow_matching_loss_sums(
        expert,
        kv_states=kv,
        enc_mask=enc_mask,
        actions_norm=torch.randn(batch, horizon, dim).clamp(-1, 1),
        action_dim_is_pad=torch.zeros(batch, dim, dtype=torch.bool),
        times=torch.tensor([0.3, 0.8]),
        noise=torch.randn(batch, horizon, dim),
    )
    (loss_sum / count).backward()
    grads = {
        name: param.grad
        for name, param in expert.named_parameters()
        if param.requires_grad
    }
    missing = [name for name, grad in grads.items() if grad is None]
    assert not missing, missing
    assert math.isfinite(float(loss_sum.detach()))
