"""Explicit chunked-gradient allreduce (``--chunk-grad-allreduce``):
the memory fix for the Molmo2 AR 4xDDP rung's step-1 wall. The OOM
snapshot (2026-08-06, rungs 3-7) showed DDP's reducer bucket buffers —
a full fp32 gradient copy — are allocated AT CONSTRUCTION, so merely
bypassing the sync (the first version of this flag) removed nothing.
The flag now skips the DDP wrapper entirely: rank-0 state broadcast at
init (``broadcast_module_states`` — the one thing DDP's constructor
provided besides the reducer), plain autograd gradient accumulation
across chunks, one explicit in-place allreduce per step (sum / world —
DDP's averaging semantics, differing only in fp reduction order).

Verified on a real 2-process gloo group over a float64 model with
rank-distinct data and sum-form losses normalized by the global count
(the train loop's chunked semantics):

* gradient/update oracle — N optimizer steps under the flag (bare
  module, broadcast init from DELIBERATELY rank-divergent weights) ==
  N steps under DDP's final-chunk sync == the single-process
  reference whose loss is the global-batch objective, params equal to
  1e-12 on both ranks;
* the CLI guards — the flag without torchrun, or without
  --backward-chunks > 1, dies loudly before any data/model build.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
from torch.multiprocessing.spawn import spawn

from bijou.train import allreduce_gradients, broadcast_module_states

WORLD = 2
STEPS = 3
CHUNKS = 3
PER_RANK = 6  # samples per rank per step; CHUNKS equal chunks of 2
IN_DIM, HIDDEN, OUT_DIM = 5, 8, 3
LR = 0.1
# Global sum-form normalizer, constant across steps (the train loop
# computes it from the FULL step's counts before any forward).
NORMALIZER = WORLD * PER_RANK * OUT_DIM


def build_model(seed: int = 0) -> torch.nn.Sequential:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(IN_DIM, HIDDEN, dtype=torch.float64),
        torch.nn.Tanh(),
        torch.nn.Linear(HIDDEN, OUT_DIM, dtype=torch.float64),
    )


def rank_batch(step: int, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic, rank-DISTINCT data — the allreduce must actually
    average something."""
    gen = torch.Generator().manual_seed(1000 + 10 * step + rank)
    x = torch.randn(PER_RANK, IN_DIM, dtype=torch.float64, generator=gen)
    y = torch.randn(PER_RANK, OUT_DIM, dtype=torch.float64, generator=gen)
    return x, y


def run_reference() -> list[torch.Tensor]:
    """Single-process ground truth: the global-batch objective both DDP
    paths must realize — mean over ranks of per-rank sum-form losses
    over the global normalizer."""
    model = build_model()
    params = list(model.parameters())
    optimizer = torch.optim.SGD(params, lr=LR)
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        for rank in range(WORLD):
            x, y = rank_batch(step, rank)
            (((model(x) - y) ** 2).sum() / (NORMALIZER * WORLD)).backward()
        optimizer.step()
    return [p.detach().clone() for p in params]


def ddp_worker(rank: int, tmp: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{tmp}/rendezvous",
        rank=rank,
        world_size=WORLD,
    )
    results: dict[str, list[torch.Tensor]] = {}
    for mode in ("ddp_sync", "allreduce"):
        if mode == "ddp_sync":
            model = build_model()
            step_module: torch.nn.Module = torch.nn.parallel.DistributedDataParallel(
                model,
                # The train loop's DDP-path shape (static_graph off,
                # buffers unbroadcast, grads as bucket views).
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
        else:
            # The flag's path: NO DDP wrapper. Deliberately divergent
            # per-rank init — broadcast_module_states must be the thing
            # that synchronizes the replicas, exactly like DDP's
            # construction-time broadcast would have.
            model = build_model(seed=rank)
            broadcast_module_states(model)
            step_module = model
        params = list(model.parameters())
        optimizer = torch.optim.SGD(params, lr=LR)
        for step in range(STEPS):
            x, y = rank_batch(step, rank)
            optimizer.zero_grad(set_to_none=True)
            per_chunk = PER_RANK // CHUNKS
            for c in range(CHUNKS):
                xs = x[c * per_chunk : (c + 1) * per_chunk]
                ys = y[c * per_chunk : (c + 1) * per_chunk]
                last = c == CHUNKS - 1
                sync_ctx = (
                    nullcontext()
                    if (mode == "allreduce" or last)
                    else step_module.no_sync()  # type: ignore[union-attr]
                )
                with sync_ctx:
                    (((step_module(xs) - ys) ** 2).sum() / NORMALIZER).backward()
            if mode == "allreduce":
                allreduce_gradients(params)
            optimizer.step()
        results[mode] = [p.detach().clone() for p in params]
    (Path(tmp) / f"rank{rank}.pkl").write_bytes(pickle.dumps(results))
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def test_chunk_grad_allreduce_matches_ddp_sync_and_reference() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        spawn(ddp_worker, args=(tmp,), nprocs=WORLD, join=True)
        reference = run_reference()
        for rank in range(WORLD):
            results = pickle.loads((Path(tmp) / f"rank{rank}.pkl").read_bytes())
            for mode in ("ddp_sync", "allreduce"):
                for got, want in zip(results[mode], reference, strict=True):
                    assert torch.allclose(got, want, rtol=0.0, atol=1e-12), (
                        f"rank {rank} mode {mode} diverged from the "
                        "global-batch reference"
                    )


def test_chunk_grad_allreduce_requires_torchrun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bijou.train import main

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "bijou.train",
            "--family",
            "gemma_flow",
            "--train-data",
            "/nonexistent",
            "--backward-chunks",
            "2",
            "--batch-size",
            "12",
            "--chunk-grad-allreduce",
        ],
    )
    with pytest.raises(SystemExit, match="world size > 1"):
        main()


def test_chunk_grad_allreduce_requires_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard fires before torch.distributed initializes, so a fake
    WORLD_SIZE is safe — no process group is ever created."""
    from bijou.train import main

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(
        "sys.argv",
        [
            "bijou.train",
            "--family",
            "gemma_flow",
            "--train-data",
            "/nonexistent",
            "--chunk-grad-allreduce",
        ],
    )
    with pytest.raises(SystemExit, match="backward-chunks"):
        main()
