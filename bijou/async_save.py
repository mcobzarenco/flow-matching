"""Async checkpoint saves: the optimizer-state capture/gather/merge side.

The sync save path stalls stepping for the whole serialize+write — and
under ``--zero1`` for ``ZeroRedundancyOptimizer.consolidate_state_dict``
first, which pickles each rank's ENTIRE shard, round-trips it through a
device ByteTensor, and broadcasts rank-by-rank over the training NCCL
group while every other rank idle-spins in the collective (measured on
the molmo2 AR 4xDDP run: ~15.5 min/save, ~14 min of it consolidate,
every ~92 min of stepping — ~14% of wall time).

The async design splits the save at the device boundary:

* **Capture (main thread, seconds)** — every rank deep-copies its LOCAL
  optimizer shard to CPU (``capture_optimizer_state``); rank 0 also
  copies the merge skeleton (param groups + any stale wrapper state) and
  the global partition index map. No collective, no disk.
* **Gather+merge+write (background thread)** — the CPU shards travel to
  rank 0 with ``gather_object`` over a DEDICATED gloo (CPU) group — the
  background thread must never touch the training NCCL communicator or
  the GPU — and rank 0 merges them into the standard optimizer format
  with exactly ``ZeroRedundancyOptimizer.state_dict()``'s indexing
  (``merge_zero1_shards``), then runs the caller's write closure.

Byte-identity with the sync path is the contract (oracle:
``tests/test_async_save.py``): same merged dict, same ``torch.save``
bytes. One representational exception, load-path invisible: a non-zero1
GPU run's sync ``optimizer.pt`` stored CUDA-device tensors; the async
capture stores their CPU copies (``load_state_dict`` re-homes state to
each param's device on resume — and the zero1 sync path already stored
CPU tensors via consolidate).

Concurrency contract: one save in flight per rank. ``submit`` joins the
previous save first — per-rank thread joins give the shared gloo group a
consistent collective order across ranks. Background failures are
stored and re-raised loudly at the next ``submit``/``join`` (a dropped
checkpoint must never be silent).
"""

from __future__ import annotations

import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer


def copy_to_cpu(value: Any, _memo: dict[int, Any] | None = None) -> Any:
    """Recursive CPU deep copy of a state-dict-shaped object: tensors are
    copied off-device (always copied, even CPU->CPU — the snapshot must
    not alias live optimizer state the next step mutates); containers are
    rebuilt; leaves pass through. Identity-memoized like ``deepcopy``:
    an object shared inside the source (e.g. one ``betas`` tuple
    referenced by every param group) stays ONE object in the copy —
    pickle memoizes shared references, so byte-identity with the sync
    path depends on preserving the sharing structure, not just the
    values."""
    memo = _memo if _memo is not None else {}
    if id(value) in memo:
        return memo[id(value)]
    copied: Any
    if isinstance(value, torch.Tensor):
        copied = value.detach().to("cpu", copy=True)
    elif isinstance(value, dict):
        copied = {}
        memo[id(value)] = copied
        for key, item in value.items():
            copied[key] = copy_to_cpu(item, memo)
        return copied
    elif isinstance(value, list):
        copied = []
        memo[id(value)] = copied
        for item in value:
            copied.append(copy_to_cpu(item, memo))
        return copied
    elif isinstance(value, tuple):
        copied = tuple(copy_to_cpu(item, memo) for item in value)
    else:
        return value
    memo[id(value)] = copied
    return copied


@dataclass(frozen=True, slots=True)
class OptimizerCapture:
    """The CPU snapshot ``submit`` consumes. Non-zero1: ``local_shard``
    is the full standard-format state dict (rank 0 only). Zero1:
    ``local_shard`` is this rank's ``optimizer.optim.state_dict()`` and
    rank 0 additionally carries the merge inputs."""

    local_shard: dict[str, Any]
    # Rank 0, zero1 only: the wrapper-level dict the merge starts from —
    # param_groups with GLOBAL indices plus any (stale) wrapper state,
    # exactly what super().state_dict() returns inside
    # ZeroRedundancyOptimizer.state_dict().
    skeleton: dict[str, Any] | None = None
    # Rank 0, zero1 only: partition_indices[rank][group] is the list of
    # GLOBAL param indices for that rank's local param group, in local
    # order — the local->global translation state_dict() derives from
    # _partition_parameters() and _param_to_index.
    partition_indices: tuple[tuple[tuple[int, ...], ...], ...] | None = None


def capture_optimizer_state(
    optimizer: torch.optim.Optimizer,
    *,
    zero1: bool,
    is_main: bool,
) -> OptimizerCapture:
    """Main-thread snapshot at the save boundary (device->CPU copies,
    seconds). Must run BEFORE stepping resumes — the copies are the save
    step's values."""
    if not zero1:
        return OptimizerCapture(local_shard=copy_to_cpu(optimizer.state_dict()))
    assert isinstance(optimizer, ZeroRedundancyOptimizer)
    # The local (per-rank) inner optimizer's shard — consolidate's
    # payload, without the collective.
    local_shard = copy_to_cpu(optimizer.optim.state_dict())
    if not is_main:
        return OptimizerCapture(local_shard=local_shard)
    # Merge inputs, captured synchronously so a scheduler.step() between
    # boundary and background merge cannot leak a later lr into
    # param_groups. _partition_parameters() and _param_to_index are
    # construction-time caches — private torch surface, pinned by the
    # byte-identity oracle against torch upgrades.
    zero: Any = optimizer
    skeleton = copy_to_cpu(torch.optim.Optimizer.state_dict(optimizer))
    partition_indices = tuple(
        tuple(
            tuple(zero._param_to_index[param] for param in group["params"])
            for group in rank_groups
        )
        for rank_groups in zero._partition_parameters()
    )
    return OptimizerCapture(
        local_shard=local_shard,
        skeleton=skeleton,
        partition_indices=partition_indices,
    )


def merge_zero1_shards(
    skeleton: dict[str, Any],
    shards: list[dict[str, Any]],
    partition_indices: tuple[tuple[tuple[int, ...], ...], ...],
) -> dict[str, Any]:
    """``ZeroRedundancyOptimizer.state_dict()``'s merge, replicated over
    captured data instead of the live optimizer: every param's state
    entry comes from the one shard that owns it, keyed back to its
    global index; entries sort; param_groups come from the skeleton."""
    state = dict(skeleton["state"])
    for shard, rank_indices in zip(shards, partition_indices, strict=True):
        for local_group, global_indices in zip(
            shard["param_groups"],
            rank_indices,
            strict=True,
        ):
            for local_index, global_index in zip(
                local_group["params"],
                global_indices,
                strict=True,
            ):
                if local_index in shard["state"]:
                    state[global_index] = shard["state"][local_index]
    return {
        "state": dict(sorted(state.items())),
        "param_groups": skeleton["param_groups"],
    }


class AsyncCheckpointSaver:
    """One background save in flight per rank; see the module docstring.

    ``group`` is the dedicated gloo group (zero1 + distributed only —
    ``None`` otherwise); ``write`` on rank 0 receives the final
    standard-format optimizer state dict and does everything
    model/disk-side (already-captured CPU tensors only)."""

    def __init__(
        self,
        *,
        group: object | None,
        is_main: bool,
        world_size: int,
        zero1: bool,
    ) -> None:
        if zero1 and world_size > 1 and group is None:
            raise ValueError(
                "zero1 async saves need a dedicated (gloo) process group "
                "for the background shard gather",
            )
        self._group = group
        self._is_main = is_main
        self._world_size = world_size
        self._zero1 = zero1
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def submit(
        self,
        *,
        capture: OptimizerCapture,
        write: Callable[[dict[str, Any]], Any] | None,
    ) -> None:
        """Called at the save boundary on every participating rank (all
        ranks under zero1 — the gather is collective; rank 0 only
        otherwise). Joins the previous save first."""
        self.join()
        if self._is_main and write is None:
            raise ValueError("rank 0 submit needs the write closure")

        def run() -> None:
            try:
                if self._zero1 and self._world_size > 1:
                    shards: list[dict[str, Any]] | None = (
                        [{} for _ in range(self._world_size)] if self._is_main else None
                    )
                    dist.gather_object(
                        capture.local_shard,
                        shards,
                        dst=0,
                        group=self._group,  # type: ignore[arg-type]
                    )
                    if self._is_main:
                        assert shards is not None
                        # gather_object round-trips even rank 0's own
                        # contribution through pickle, which de-interns
                        # its dict keys; the sync path copies rank 0's
                        # shard directly. Byte-identity of the final
                        # torch.save depends on that object structure
                        # (pickle memoizes shared strings), so keep the
                        # local capture for our own slot.
                        shards[0] = capture.local_shard
                        assert capture.skeleton is not None
                        assert capture.partition_indices is not None
                        merged = merge_zero1_shards(
                            capture.skeleton,
                            shards,
                            capture.partition_indices,
                        )
                        assert write is not None
                        write(merged)
                elif self._is_main:
                    assert write is not None
                    write(capture.local_shard)
            except BaseException as error:  # noqa: BLE001 — re-raised at join
                traceback.print_exc()
                self._error = error

        self._thread = threading.Thread(
            target=run,
            name="bijou-async-save",
            daemon=True,
        )
        self._thread.start()

    def join(self) -> None:
        """Wait for the in-flight save; re-raise its failure loudly."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            error, self._error = self._error, None
            raise RuntimeError("async checkpoint save failed") from error
