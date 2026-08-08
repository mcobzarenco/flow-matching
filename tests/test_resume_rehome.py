"""Regression oracles for ``rehome_fused_step_tensors`` (the molmo2 60k
resume crash, 2026-08-08 10:15Z): a consolidated CPU-tagged
``optimizer.pt`` loaded into a fused-CUDA AdamW leaves the integer
``step`` state on CPU and the first ``optimizer.step()`` aborts. The
GPU test reproduces the crash signature end-to-end and proves the
re-home fixes it; the CPU tests pin the no-op contracts.
"""

from __future__ import annotations

import pytest
import torch

from bijou.train import rehome_fused_step_tensors


def _cpu_payload_roundtrip(optimizer: torch.optim.Optimizer) -> dict:
    """Model the async-save capture: every state tensor CPU-tagged and
    the saved groups stripped of fused/capturable flags (the merged
    consolidated payload's shape)."""
    payload = optimizer.state_dict()
    for state in payload["state"].values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.detach().cpu().clone()
    for group in payload["param_groups"]:
        group.pop("fused", None)
        group.pop("capturable", None)
    return payload


def test_non_fused_is_exact_noop() -> None:
    param = torch.nn.Parameter(torch.zeros(3))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    param.grad = torch.ones_like(param)
    optimizer.step()
    before = {k: v.clone() for k, v in optimizer.state[param].items()}
    assert rehome_fused_step_tensors(optimizer, fused=False) == 0
    after = optimizer.state[param]
    assert set(before) == set(after)
    for key, value in before.items():
        assert torch.equal(value, after[key])
    assert after["step"].device.type == "cpu"


def test_matched_devices_move_nothing() -> None:
    param = torch.nn.Parameter(torch.zeros(3))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    param.grad = torch.ones_like(param)
    optimizer.step()
    # fused=True but every tensor already on the param's (CPU) device.
    assert rehome_fused_step_tensors(optimizer, fused=True) == 0


@pytest.mark.gpu
def test_zero1_fused_cuda_resume_crashes_without_rehome_and_steps_with_it() -> None:
    """The measured incident shape: a ZeRO-1 wrapper resumed from a
    CPU-tagged consolidated payload leaves the local shard's ``step``
    tensors on CPU (plain AdamW's ``load_state_dict`` re-casts them
    from the constructed group's fused flag; the ZRO shard-load path
    does not), and fused AdamW aborts at the first step."""
    import os

    from torch.distributed.optim import ZeroRedundancyOptimizer

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    torch.distributed.init_process_group("gloo", rank=0, world_size=1)
    try:
        device = torch.device("cuda:0")

        def build() -> tuple[torch.nn.Parameter, ZeroRedundancyOptimizer]:
            p = torch.nn.Parameter(torch.zeros(4, device=device))
            opt = ZeroRedundancyOptimizer(
                [p],
                optimizer_class=torch.optim.AdamW,
                lr=1e-3,
                fused=True,
            )
            return p, opt

        param, optimizer = build()
        param.grad = torch.ones_like(param)
        optimizer.step()
        optimizer.consolidate_state_dict(to=0)
        payload = _cpu_payload_roundtrip(optimizer)

        param2, resumed = build()
        # ZRO lazily builds shard buckets at first step; prime it the
        # way the training loop does before any resume matters.
        param2.grad = torch.ones_like(param2)
        resumed.load_state_dict(payload)
        inner = resumed.optim
        step_state = inner.state[param2]["step"]
        if step_state.device.type == "cpu":
            # The crash signature this fix exists for.
            with pytest.raises(RuntimeError, match="state_steps is on cpu"):
                resumed.step()
            moved = rehome_fused_step_tensors(resumed, fused=True)
            assert moved == 1
            assert inner.state[param2]["step"].device.type == "cuda"
        else:
            pytest.fail(
                "payload load re-homed step by itself — the incident "
                "mechanism changed; re-diagnose before trusting the fix",
            )
        resumed.step()  # must not raise
    finally:
        torch.distributed.destroy_process_group()
