"""ZeRO-1 optimizer sharding (``--zero1``): the memory fix for the
Molmo2 AR 4xDDP rung, whose per-rank static budget (~76-77 GiB
measured) left no chunk size that fits — Adam moments shard across
ranks instead (2026-08-06 smoke-ladder verdict).

The contract the flag ships on is EXACTNESS: ZeroRedundancyOptimizer
partitions each parameter's Adam state onto exactly one rank and
broadcasts updated shards after each step, so with identical per-rank
gradients (what DDP's allreduce guarantees) every replica must produce
the SAME parameters plain AdamW would — bitwise, not approximately.
Verified here on a real 2-process gloo group (CPU, file-store
rendezvous) over the run's actual param-group shape (decoder group +
decayed/undecayed backbone groups at a different lr) with a LambdaLR
stepping every iteration (the lr-sync path ZRO handles inside step()):

* update oracle — N steps of ZRO(AdamW) across 2 ranks == N steps of
  single-process AdamW, params bit-equal on both ranks;
* checkpoint round-trip — consolidate_state_dict -> state_dict on rank
  0 is the STANDARD optimizer format: loading it into a fresh plain
  AdamW (the un-sharded resume direction) and into a fresh ZRO (the
  sharded resume direction) both continue bit-identically to the
  never-checkpointed reference;
* the CLI guard — --zero1 without torchrun dies loudly before any
  data/model build.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path

import pytest
import torch
from torch.multiprocessing.spawn import spawn

WORLD = 2
STEPS = 4
RESUME_STEPS = 3
SEED = 0
DIMS = (13, 7)  # odd sizes: exercise an uneven greedy partition


def build_params() -> tuple[list[torch.nn.Parameter], list[dict]]:
    """The run's param-group shape: decoder @ lr1 (decayed), backbone
    decayed @ lr2, backbone undecayed @ lr2 / wd 0."""
    torch.manual_seed(SEED)
    params = [
        torch.nn.Parameter(torch.randn(d_out, d_in, dtype=torch.float64))
        for d_out in DIMS
        for d_in in DIMS
    ]
    groups = [
        {"params": params[:2], "lr": 1e-2},
        {"params": params[2:3], "lr": 3e-3},
        {"params": params[3:], "lr": 3e-3, "weight_decay": 0.0},
    ]
    return params, groups


def fake_grads(params: list[torch.nn.Parameter], step: int) -> None:
    """Deterministic per-step gradients, identical on every rank —
    exactly what DDP's allreduce hands the optimizer."""
    gen = torch.Generator().manual_seed(1000 + step)
    for p in params:
        p.grad = torch.randn(p.shape, dtype=p.dtype, generator=gen)


def lr_lambda(step: int) -> float:
    return 1.0 / (1 + step)


def run_reference(steps: int, adamw_state: dict | None = None) -> list[torch.Tensor]:
    """Single-process plain AdamW: the ground truth ZRO must match."""
    params, groups = build_params()
    optimizer = torch.optim.AdamW(groups, lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    start = 0
    if adamw_state is not None:
        # Fast-forward the reference to the checkpoint boundary first so
        # the continuation is a true resume comparison.
        for step in range(STEPS):
            fake_grads(params, step)
            optimizer.step()
            scheduler.step()
        optimizer.load_state_dict(adamw_state)
        start = STEPS
    for step in range(start, start + steps):
        fake_grads(params, step)
        optimizer.step()
        scheduler.step()
    return [p.detach().clone() for p in params]


def zero1_worker(rank: int, tmp: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{tmp}/rendezvous",
        rank=rank,
        world_size=WORLD,
    )
    from torch.distributed.optim import ZeroRedundancyOptimizer

    params, groups = build_params()
    optimizer = ZeroRedundancyOptimizer(
        groups,
        optimizer_class=torch.optim.AdamW,
        lr=1e-2,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    for step in range(STEPS):
        fake_grads(params, step)
        optimizer.step()
        scheduler.step()

    # The save path bijou.train runs at every boundary: collective
    # consolidation, then rank 0 serializes the standard-format dict.
    optimizer.consolidate_state_dict(to=0)
    if rank == 0:
        (Path(tmp) / "params.pkl").write_bytes(
            pickle.dumps([p.detach().clone() for p in params]),
        )
        (Path(tmp) / "optimizer.pkl").write_bytes(
            pickle.dumps(optimizer.state_dict()),
        )
    torch.distributed.barrier()

    # Sharded-resume direction: a FRESH ZRO loads the consolidated dict
    # (bijou.train --resume --zero1) and continues.
    resumed_params, resumed_groups = build_params()
    resumed = ZeroRedundancyOptimizer(
        resumed_groups,
        optimizer_class=torch.optim.AdamW,
        lr=1e-2,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed, lr_lambda)
    resumed.load_state_dict(
        pickle.loads((Path(tmp) / "optimizer.pkl").read_bytes()),
    )
    saved = pickle.loads((Path(tmp) / "params.pkl").read_bytes())
    with torch.no_grad():
        for p, s in zip(resumed_params, saved, strict=True):
            p.copy_(s)
    for _ in range(STEPS):
        resumed_scheduler.step()  # scheduler state rides optimizer.pt in train
    for step in range(STEPS, STEPS + RESUME_STEPS):
        fake_grads(resumed_params, step)
        resumed.step()
        resumed_scheduler.step()
    if rank == 0:
        (Path(tmp) / f"rank{rank}_resumed.pkl").write_bytes(
            pickle.dumps([p.detach().clone() for p in resumed_params]),
        )
    (Path(tmp) / f"rank{rank}_final.pkl").write_bytes(
        pickle.dumps([p.detach().clone() for p in params]),
    )
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def test_zero1_matches_plain_adamw_and_roundtrips() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        spawn(
            zero1_worker,
            args=(tmp,),
            nprocs=WORLD,
            join=True,
        )
        reference = run_reference(STEPS)

        # Update oracle: both ranks landed exactly the plain-AdamW params.
        for rank in range(WORLD):
            ranked = pickle.loads((Path(tmp) / f"rank{rank}_final.pkl").read_bytes())
            for got, want in zip(ranked, reference, strict=True):
                assert torch.equal(got, want), f"rank {rank} diverged from AdamW"

        # Un-sharded resume direction: the consolidated dict IS a plain
        # AdamW state dict — load it into one and continue.
        optimizer_state = pickle.loads((Path(tmp) / "optimizer.pkl").read_bytes())
        continued = run_reference(RESUME_STEPS, adamw_state=optimizer_state)
        never_checkpointed = run_reference(STEPS + RESUME_STEPS)
        for got, want in zip(continued, never_checkpointed, strict=True):
            assert torch.equal(got, want)

        # Sharded resume direction: fresh ZRO + consolidated dict.
        resumed = pickle.loads((Path(tmp) / "rank0_resumed.pkl").read_bytes())
        for got, want in zip(resumed, never_checkpointed, strict=True):
            assert torch.equal(got, want)


def test_zero1_requires_torchrun(monkeypatch: pytest.MonkeyPatch) -> None:
    """--zero1 in a single-process run dies loudly before any build
    (the guard sits right after world_size is read, ahead of dataset
    selection and the model load)."""
    from bijou.train import main

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        ["bijou.train", "--train-data", "/nonexistent", "--zero1"],
    )
    with pytest.raises(SystemExit, match="world size > 1"):
        main()
