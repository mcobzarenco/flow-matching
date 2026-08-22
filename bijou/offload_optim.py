"""``bijou.train --offload-optim``'s optimizer, in its own module so the
train-loop file carries only the flag wiring (the class is the bulk of
the diff and the likely rebase-conflict surface as train.py evolves on
main)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, override

import torch
from torch import Tensor, nn


class CPUOffloadAdamW(torch.optim.Optimizer):
    """AdamW whose moments live in host RAM (``--offload-optim``): the
    fp32 m/v/step state for every trainable parameter — the single
    largest resident block of a live-trunk run (measured 33.7 GiB for
    the 4.2B-param molmoact2_joint set, vs the H100's 80 GiB) — never
    touches the GPU. Update semantics are EXACT fp32 AdamW: the step
    runs torch's own CPU kernels on pinned fp32 mirrors of the params;
    AdamW is elementwise, so the trajectory is the reference CPU path's
    bitwise (oracle: tests/test_offload_optim.py pins both the CPU
    identity and GPU-fused closeness).

    Wiring: the INNER CPU optimizer's param-group dicts are registered
    as this wrapper's own (same objects), so the LR scheduler and the
    AdamC pre-step decay writes reach the kernel that steps without any
    forwarding layer. Per step: clipped GPU grads copy into pinned
    mirror grads (D2H), the inner optimizer steps on host, updated
    mirrors copy back into the live GPU params (H2D, stream-ordered —
    the next forward waits on the copy by ordering, and the next step's
    device synchronize fences the host side). A param whose GPU grad is
    None steps exactly like stock AdamW: skipped entirely (its mirror
    grad is set to None for that step).

    Cost: ~2×(trainable bytes) PCIe traffic + a CPU foreach-AdamW pass
    per optimizer step; host holds mirrors + pinned grad buffers +
    moments (~4× trainable bytes total). Single-process only — the
    construction site refuses it under torchrun."""

    def __init__(
        self,
        param_groups: list[dict[str, Any]],
        *,
        lr: float,
        betas: tuple[float, float],
        weight_decay: float,
    ) -> None:
        self._pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
        # Mirrors snapshotted here go stale if checkpoint weights load
        # into the live params AFTER construction (the construction
        # site's real order under --init-from/--resume:
        # load_family_weights runs after the optimizer is built) — the
        # first write-back would silently revert those params to their
        # built init (REALIZED 2026-08-22: the squint flow-head reset).
        # The invariant is mirror == live param at step ENTRY, so the
        # first step() re-captures before it reads.
        self._mirrors_synced = False
        # Pinned grad buffers survive None-grad steps (the mirror's
        # .grad is swapped to None to reproduce AdamW's skip exactly,
        # then restored from here when the grad returns).
        self._grad_buffers: dict[nn.Parameter, Tensor] = {}
        mirror_groups: list[dict[str, Any]] = []
        for group in param_groups:
            mirror_group = dict(group)
            mirrors: list[nn.Parameter] = []
            for param in group["params"]:
                mirror = nn.Parameter(
                    param.detach().to("cpu", torch.float32, copy=True).pin_memory(),
                    requires_grad=False,
                )
                buffer = torch.zeros_like(mirror).pin_memory()
                mirror.grad = buffer
                self._grad_buffers[mirror] = buffer
                self._pairs.append((param, mirror))
                mirrors.append(mirror)
            mirror_group["params"] = mirrors
            mirror_groups.append(mirror_group)
        self.inner = torch.optim.AdamW(
            mirror_groups,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            fused=False,
        )
        # Register the inner's group dicts (same objects) as our own:
        # scheduler lr writes and AdamC weight_decay writes land
        # directly in the groups the CPU kernel reads.
        super().__init__(self.inner.param_groups, dict(self.inner.defaults))
        assert all(
            ours is theirs
            for ours, theirs in zip(
                self.param_groups,
                self.inner.param_groups,
                strict=True,
            )
        )  # add_param_group must keep dict identity (torch API contract)

    @override
    def zero_grad(self, set_to_none: bool = True) -> None:
        # The LIVE grads are the GPU params' (backward writes there;
        # clipping reads there). Mirror grads are managed buffers,
        # fully overwritten each step — never zeroed.
        for param, _ in self._pairs:
            if param.grad is None:
                continue
            if set_to_none:
                param.grad = None
            else:
                param.grad.detach_()
                param.grad.zero_()

    @override
    @torch.no_grad()
    def step(  # type: ignore[override]
        self,
        closure: Callable[[], float] | None = None,
    ) -> float | None:
        assert closure is None  # the train loop never passes one
        if not self._mirrors_synced:
            # First step: re-capture the mirrors from the live params
            # (D2H into pinned memory, fenced by the synchronize below
            # with the grad copies). No-op for a fresh run — nothing
            # mutates params between construction and step 1 except
            # post-construction checkpoint loads, which is exactly the
            # staleness being erased.
            for param, mirror in self._pairs:
                mirror.data.copy_(param.data, non_blocking=True)
            self._mirrors_synced = True
        for param, mirror in self._pairs:
            if param.grad is None:
                mirror.grad = None  # stock AdamW skips the param
                continue
            if mirror.grad is None:
                mirror.grad = self._grad_buffers[mirror]
            mirror.grad.copy_(param.grad, non_blocking=True)
        # Fence the D2H copies before the CPU kernel reads them — and
        # (next step) the previous H2D copies before the host mutates
        # their pinned sources again.
        torch.cuda.synchronize()
        self.inner.step()
        for param, mirror in self._pairs:
            if mirror.grad is None:
                continue  # skipped above: the GPU param is current
            param.data.copy_(mirror.data, non_blocking=True)
        return None

    @override
    def state_dict(self) -> dict[str, Any]:
        return self.inner.state_dict()

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.inner.load_state_dict(state_dict)
        # torch's load REPLACES the inner's group dicts (update_group
        # builds new ones from the saved payload) — re-share them, or
        # post-resume scheduler/adamc writes would land in orphaned
        # dicts and the stepping kernel would never see an LR change.
        self.param_groups = self.inner.param_groups
