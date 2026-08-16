"""``--offload-optim`` (offload_optim.py ``CPUOffloadAdamW``): AdamW moments in
host RAM, oracle-pinned to the stock optimizers it replaces.

The claim the flag ships behind is *exact semantics, different
residence*: the offloaded step must BE fp32 AdamW — not an
approximation of it — with only the m/v/step tensors' device changed.
AdamW's update is elementwise (no cross-element reduction anywhere in
Algorithm 2), so the CPU reference kernels are the pinnable ground
truth:

* keystone — a multi-group trajectory (distinct per-group lr/decay,
  the run shape build_optimizer_param_groups emits) stepped N times
  through CPUOffloadAdamW on CUDA params equals torch's own CPU AdamW
  on an identical CPU clone of the problem BITWISE, per step, while a
  live LambdaLR mutates the shared group dicts and an AdamC-style
  weight_decay write lands mid-trajectory (the two in-place group
  mutations the train loop performs);
* the fused-CUDA path it replaces stays within float tolerance (the
  cross-kernel closeness torch itself guarantees, pinned here so a
  fused/CPU divergence would name this flag, not the run);
* a None-grad param is SKIPPED exactly like stock AdamW (no decay
  applied), and steps again when its grad returns;
* zero_grad(set_to_none=True) releases the GPU grads (the live set
  backward writes into), never the pinned mirror buffers;
* state_dict round-trips through the resume path's
  ``torch.load(weights_only=True)`` and the reloaded optimizer
  continues the trajectory bitwise (moments land on CPU — the
  ``rehome_fused_step_tensors`` no-op leg for offload runs).

CUDA-only (the flag targets exactly the one-card live-trunk run);
skipped wholesale off-GPU.
"""

from __future__ import annotations

import io

import pytest
import torch

from bijou.offload_optim import CPUOffloadAdamW
from bijou.train import TrainState, rehome_fused_step_tensors

if not torch.cuda.is_available():
    pytest.skip("--offload-optim targets CUDA runs", allow_module_level=True)


def make_problem(
    seed: int,
    device: str,
) -> tuple[list[torch.nn.Parameter], list[dict]]:
    """Two param groups with distinct lr/weight_decay — the decoder /
    backbone_text shape the construction site feeds AdamW."""
    generator = torch.Generator().manual_seed(seed)
    shapes = [(37, 19), (64,), (11, 5, 3), (128,)]
    params = [
        torch.nn.Parameter(
            torch.randn(*shape, generator=generator, dtype=torch.float32).to(device),
        )
        for shape in shapes
    ]
    groups = [
        {"params": params[:2], "lr": 5e-3, "weight_decay": 0.1},
        {"params": params[2:], "lr": 1e-3, "weight_decay": 0.0},
    ]
    return params, groups


def synthetic_grads(step: int, params: list[torch.nn.Parameter]) -> None:
    """Deterministic per-step grads, seeded off the step index."""
    generator = torch.Generator().manual_seed(1000 + step)
    for param in params:
        grad = torch.randn(
            *param.shape,
            generator=generator,
            dtype=torch.float32,
        )
        param.grad = grad.to(param.device)


STEPS = 7
ADAMC_WRITE_STEP = 3  # mid-trajectory group mutation (apply_adamc path)


def lr_schedule(step: int) -> float:
    """A moving schedule (every step changes lr)."""
    return 1.0 / (1 + step)


def run_offload(seed: int) -> tuple[list[torch.Tensor], CPUOffloadAdamW]:
    params, groups = make_problem(seed, "cuda")
    optimizer = CPUOffloadAdamW(groups, lr=1.0, betas=(0.9, 0.95), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    trajectory: list[torch.Tensor] = []
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        synthetic_grads(step, params)
        if step == ADAMC_WRITE_STEP:
            optimizer.param_groups[0]["weight_decay"] = 0.05
        optimizer.step()
        scheduler.step()
        trajectory.append(
            torch.cat([p.detach().flatten().cpu() for p in params]).clone(),
        )
    return trajectory, optimizer


def run_reference(seed: int, device: str, *, fused: bool) -> list[torch.Tensor]:
    params, groups = make_problem(seed, device)
    optimizer = torch.optim.AdamW(
        groups,
        lr=1.0,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=fused,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    trajectory: list[torch.Tensor] = []
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        synthetic_grads(step, params)
        if step == ADAMC_WRITE_STEP:
            optimizer.param_groups[0]["weight_decay"] = 0.05
        optimizer.step()
        scheduler.step()
        trajectory.append(
            torch.cat([p.detach().flatten().cpu() for p in params]).clone(),
        )
    return trajectory


def test_keystone_bitwise_vs_cpu_reference() -> None:
    """The offloaded trajectory IS the CPU AdamW trajectory — bitwise,
    every step, under live scheduler + adamc-style group writes."""
    offload, _ = run_offload(seed=7)
    reference = run_reference(seed=7, device="cpu", fused=False)
    for step, (ours, theirs) in enumerate(zip(offload, reference, strict=True)):
        assert torch.equal(ours, theirs), (
            f"step {step}: offload diverged from the CPU reference "
            f"(max abs {(ours - theirs).abs().max().item():.3e})"
        )


def test_fused_cuda_closeness() -> None:
    """The fused-CUDA path offload replaces stays within float
    tolerance of the offloaded trajectory (cross-kernel closeness)."""
    offload, _ = run_offload(seed=11)
    fused = run_reference(seed=11, device="cuda", fused=True)
    for step, (ours, theirs) in enumerate(zip(offload, fused, strict=True)):
        assert torch.allclose(ours, theirs, rtol=1e-5, atol=1e-6), (
            f"step {step}: offload vs fused CUDA drifted past float "
            f"tolerance (max abs {(ours - theirs).abs().max().item():.3e})"
        )


def test_none_grad_param_skipped_exactly() -> None:
    """A param with grad None is untouched (stock AdamW's skip — decay
    included), and steps again when its grad returns."""
    params, groups = make_problem(seed=3, device="cuda")
    offload = CPUOffloadAdamW(groups, lr=1.0, betas=(0.9, 0.95), weight_decay=0.0)
    ref_params, ref_groups = make_problem(seed=3, device="cpu")
    reference = torch.optim.AdamW(
        ref_groups,
        lr=1.0,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=False,
    )
    for step in range(4):
        offload.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        synthetic_grads(step, params)
        synthetic_grads(step, ref_params)
        if step in (1, 2):  # param 0 sits out two consecutive steps
            params[0].grad = None
            ref_params[0].grad = None
        before = params[0].detach().cpu().clone()
        offload.step()
        reference.step()
        if step in (1, 2):
            assert torch.equal(params[0].detach().cpu(), before)
        for ours, theirs in zip(params, ref_params, strict=True):
            assert torch.equal(ours.detach().cpu(), theirs.detach())


def test_zero_grad_releases_gpu_grads_only() -> None:
    params, groups = make_problem(seed=5, device="cuda")
    optimizer = CPUOffloadAdamW(groups, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0)
    synthetic_grads(0, params)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in params)
    # Mirror buffers survive (they are managed, overwritten per step).
    assert all(m.grad is not None for _, m in optimizer._pairs)


def test_state_dict_roundtrip_resumes_bitwise() -> None:
    """Save at step 4 through the TrainState payload (the real resume
    format: torch.load weights_only=True), reload into a fresh
    offloaded optimizer, run to STEPS — equals the uninterrupted
    trajectory bitwise. Moments stay CPU: the fused rehome is a no-op."""
    full, _ = run_offload(seed=13)

    params, groups = make_problem(seed=13, device="cuda")
    optimizer = CPUOffloadAdamW(groups, lr=1.0, betas=(0.9, 0.95), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    split = 4
    for step in range(split):
        optimizer.zero_grad(set_to_none=True)
        synthetic_grads(step, params)
        if step == ADAMC_WRITE_STEP:
            optimizer.param_groups[0]["weight_decay"] = 0.05
        optimizer.step()
        scheduler.step()

    payload = TrainState(
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        step=split,
    ).to_payload()
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    restored = TrainState.from_payload(torch.load(buffer, weights_only=True))

    resumed = CPUOffloadAdamW(
        [
            {"params": groups[0]["params"], "lr": 5e-3, "weight_decay": 0.1},
            {"params": groups[1]["params"], "lr": 1e-3, "weight_decay": 0.0},
        ],
        lr=1.0,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    # train.py's resume order: scheduler CONSTRUCTED first (its init
    # step clobbers group lr), then the optimizer load restores the
    # saved groups, then the scheduler state loads.
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed, lr_schedule)
    resumed.load_state_dict(restored.optimizer)
    # The load replaces the inner's group dicts — identity with the
    # wrapper's must be re-established, or post-resume scheduler/adamc
    # writes never reach the CPU kernel.
    assert all(
        ours is theirs
        for ours, theirs in zip(
            resumed.param_groups,
            resumed.inner.param_groups,
            strict=True,
        )
    )
    assert rehome_fused_step_tensors(resumed, fused=False) == 0
    for state in resumed.inner.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu"
    resumed_scheduler.load_state_dict(restored.scheduler)

    for step in range(split, STEPS):
        resumed.zero_grad(set_to_none=True)
        synthetic_grads(step, params)
        resumed.step()
        resumed_scheduler.step()
    final = torch.cat([p.detach().flatten().cpu() for p in params])
    assert torch.equal(final, full[-1])
