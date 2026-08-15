"""Async checkpoint saves (``bijou.async_save`` + train.py's atomic
writer): the oracles the owner-priority item ships behind.

The async path replaces (a) ``--zero1``'s
``consolidate_state_dict`` — whole-shard pickle round-tripped through a
device ByteTensor and broadcast rank-by-rank over the TRAINING
communicator, ~14 of the ~15.5 min/save measured on the molmo2 AR 4xDDP
run — with a device->CPU shard snapshot plus a background
``gather_object`` over a dedicated gloo group, and (b) the in-loop
serialize+write with a background writer publishing via one atomic
directory rename. The contract is byte-identity: same merged optimizer
dict, same ``torch.save``/safetensors bytes as the sync path.

* keystone — on a real 2-process gloo group over the run's actual
  param-group shape, the async capture->gather->merge ``optimizer.pt``
  payload is BYTE-identical to consolidate->state_dict at two
  consecutive boundaries (the second exercises the join-pending path,
  and the background gather runs concurrently with main-thread
  collectives, the production overlap);
* the full checkpoint directory written through the async machinery is
  byte-identical file-by-file to ``save_checkpoint``'s, and its
  ``optimizer.pt`` round-trips through the resume path's
  ``torch.load(weights_only=True)``;
* a crash mid-write leaves NO ``step_*`` directory (only ``.tmp``
  debris) and every earlier checkpoint intact — atomic rename;
* background failures re-raise loudly at join (a dropped checkpoint is
  never silent).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
from test_checkpoint_backbone import DIM, make_args, pristine_dir, tiny_model
from test_zero1 import WORLD, build_params, fake_grads, lr_lambda
from torch.multiprocessing.spawn import spawn

from bijou.async_save import (
    AsyncCheckpointSaver,
    capture_optimizer_state,
    copy_to_cpu,
)
from bijou.train import (
    Normalizer,
    Normalizers,
    TrainState,
    build_vla_metadata,
    capture_checkpoint_tensors,
    save_checkpoint,
    write_checkpoint,
)

BOUNDARIES = (4, 3)  # steps before the 1st and between the two saves


def zero1_async_worker(rank: int, tmp: str) -> None:
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
    # Exactly train.py's construction: the gather rides a dedicated
    # group, never the training one.
    side_group = torch.distributed.new_group(backend="gloo")
    saver = AsyncCheckpointSaver(
        group=side_group,
        is_main=rank == 0,
        world_size=WORLD,
        zero1=True,
    )
    step_count = 0
    for boundary, steps in enumerate(BOUNDARIES):
        for _ in range(steps):
            fake_grads(params, step_count)
            optimizer.step()
            scheduler.step()
            step_count += 1

        capture = capture_optimizer_state(optimizer, zero1=True, is_main=rank == 0)
        write = None
        if rank == 0:
            scheduler_state = copy_to_cpu(scheduler.state_dict())

            def write_async(
                merged: dict[str, Any],
                *,
                _boundary: int = boundary,
                _scheduler: dict[str, Any] = scheduler_state,
                _step: int = step_count,
            ) -> None:
                # Same basename as the sync file: torch.save embeds the
                # archive name (derived from the filename) in the zip,
                # so byte-comparison needs identical names.
                target = Path(tmp) / f"async_{_boundary}" / "optimizer.pt"
                target.parent.mkdir(exist_ok=True)
                torch.save(
                    TrainState(
                        optimizer=merged,
                        scheduler=_scheduler,
                        step=_step,
                    ).to_payload(),
                    target,
                )

            write = write_async

        # No join before the sync reference below: the background gather
        # (side group) deliberately overlaps the main thread's
        # consolidate collective (default group) — the production
        # concurrency. The NEXT submit joins the pending save first.
        saver.submit(capture=capture, write=write)

        optimizer.consolidate_state_dict(to=0)
        if rank == 0:
            target = Path(tmp) / f"sync_{boundary}" / "optimizer.pt"
            target.parent.mkdir(exist_ok=True)
            torch.save(
                TrainState(
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    step=step_count,
                ).to_payload(),
                target,
            )
    saver.join()
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def test_zero1_async_gather_bytes_match_consolidate() -> None:
    """Keystone: capture->gather->merge == consolidate->state_dict, to
    the byte, at consecutive boundaries on a live 2-rank group."""
    with tempfile.TemporaryDirectory() as tmp:
        spawn(zero1_async_worker, args=(tmp,), nprocs=WORLD, join=True)
        for boundary in range(len(BOUNDARIES)):
            sync_bytes = (Path(tmp) / f"sync_{boundary}" / "optimizer.pt").read_bytes()
            async_bytes = (
                Path(tmp) / f"async_{boundary}" / "optimizer.pt"
            ).read_bytes()
            assert len(sync_bytes) > 1000  # a real payload, not a stub
            assert async_bytes == sync_bytes, f"boundary {boundary} diverged"
        # The two boundaries are genuinely different saves.
        assert (Path(tmp) / "sync_0" / "optimizer.pt").read_bytes() != (
            Path(tmp) / "sync_1" / "optimizer.pt"
        ).read_bytes()
        # The async payload is exactly what --resume reads.
        payload = torch.load(Path(tmp) / "async_1" / "optimizer.pt", weights_only=True)
        state = TrainState.from_payload(payload)
        assert state.step == sum(BOUNDARIES)


def populated_optimizer(
    model: Any,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
    optimizer = torch.optim.AdamW(model.flow_decoder.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    generator = torch.Generator().manual_seed(7)
    for _ in range(2):
        for parameter in model.flow_decoder.parameters():
            parameter.grad = 0.01 * torch.randn(
                parameter.shape,
                dtype=parameter.dtype,
                generator=generator,
            )
        optimizer.step()
        scheduler.step()
    return optimizer, scheduler


def test_async_checkpoint_directory_byte_identical(tmp_path: Path) -> None:
    """The full directory through the async machinery == the sync
    ``save_checkpoint``, file by file (the pristine backbone/ mirror
    hard-links the same inodes on both paths); the async
    ``optimizer.pt`` round-trips through the resume loader."""
    model = tiny_model()
    args = make_args(tmp_path)
    optimizer, scheduler = populated_optimizer(model)
    normalizers = Normalizers(
        action=Normalizer(mean=torch.zeros(DIM), std=torch.ones(DIM)),
        state=Normalizer(mean=torch.zeros(DIM), std=torch.ones(DIM)),
    )
    sync_dir = save_checkpoint(
        model,
        model.backbone,
        args=args,
        normalizers=normalizers,
        per_dataset_stats={},
        optimizer=optimizer,
        scheduler=scheduler,
        step=5,
        adapted_backbone_source=None,
        pristine_trunk_dir=pristine_dir(tmp_path),
    )
    sync_dir = sync_dir.rename(tmp_path / "sync")

    capture = capture_optimizer_state(optimizer, zero1=False, is_main=True)
    tensors = capture_checkpoint_tensors(
        model,
        model.backbone,
        args=args,
        adapted_backbone_source=None,
        pristine_trunk_dir=pristine_dir(tmp_path),
    )
    metadata = build_vla_metadata(
        model,
        args=args,
        normalizers=normalizers,
        per_dataset_stats={},
        step=5,
        adapted_backbone_source=None,
    )
    scheduler_state = copy_to_cpu(scheduler.state_dict())
    saver = AsyncCheckpointSaver(group=None, is_main=True, world_size=1, zero1=False)
    saver.submit(
        capture=capture,
        write=lambda optimizer_state: write_checkpoint(
            tmp_path / "step_000005",
            metadata=metadata,
            tensors=tensors,
            train_state_payload=TrainState(
                optimizer=optimizer_state,
                scheduler=scheduler_state,
                step=5,
            ).to_payload(),
        ),
    )
    saver.join()
    async_dir = (tmp_path / "step_000005").rename(tmp_path / "async")

    sync_files = sorted(path.name for path in sync_dir.iterdir())
    async_files = sorted(path.name for path in async_dir.iterdir())
    assert sync_files == async_files
    assert "optimizer.pt" in sync_files
    assert "flow_decoder.safetensors" in sync_files
    assert "backbone" in sync_files  # the pristine mirror directory
    for name in sync_files:
        if (sync_dir / name).is_dir():
            assert sorted(p.name for p in (async_dir / name).iterdir()) == sorted(
                p.name for p in (sync_dir / name).iterdir()
            )
            continue
        assert (async_dir / name).read_bytes() == (sync_dir / name).read_bytes(), (
            f"{name} diverged between sync and async writers"
        )

    payload = torch.load(async_dir / "optimizer.pt", weights_only=True)
    state = TrainState.from_payload(payload)
    fresh = torch.optim.AdamW(model.flow_decoder.parameters(), lr=1e-4)
    fresh.load_state_dict(state.optimizer)  # the --resume direction
    assert state.step == 5


def test_write_checkpoint_atomic_on_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash mid-write publishes nothing: no ``step_*`` directory for
    the failed save (only ``.tmp`` debris), earlier checkpoints
    untouched."""
    model = tiny_model()
    args = make_args(tmp_path)
    metadata = build_vla_metadata(
        model,
        args=args,
        normalizers=Normalizers(
            action=Normalizer(mean=torch.zeros(DIM), std=torch.ones(DIM)),
            state=Normalizer(mean=torch.zeros(DIM), std=torch.ones(DIM)),
        ),
        per_dataset_stats={},
        step=5,
        adapted_backbone_source=None,
    )
    tensors = capture_checkpoint_tensors(
        model,
        model.backbone,
        args=args,
        adapted_backbone_source=None,
        pristine_trunk_dir=pristine_dir(tmp_path),
    )
    payload = {"optimizer": {}, "scheduler": {}, "step": 5}
    first = write_checkpoint(
        tmp_path / "step_000005",
        metadata=metadata,
        tensors=tensors,
        train_state_payload=payload,
    )
    first_bytes = (first / "flow_decoder.safetensors").read_bytes()

    calls = {"count": 0}
    import bijou.checkpoint as checkpoint_module

    real_save_file = checkpoint_module.save_file

    def failing_save_file(state: dict[str, Any], path: str) -> None:
        calls["count"] += 1
        if calls["count"] == 2:  # die after the first component file
            raise OSError("disk gone")
        real_save_file(state, path)

    monkeypatch.setattr(checkpoint_module, "save_file", failing_save_file)
    with pytest.raises(OSError, match="disk gone"):
        write_checkpoint(
            tmp_path / "step_000010",
            metadata=metadata,
            tensors=tensors,
            train_state_payload={"optimizer": {}, "scheduler": {}, "step": 10},
        )
    assert not (tmp_path / "step_000010").exists()  # nothing published
    assert (tmp_path / "step_000010.tmp").exists()  # debris, as documented
    assert (first / "flow_decoder.safetensors").read_bytes() == first_bytes
    monkeypatch.undo()
    # The next attempt clobbers the debris and lands.
    retried = write_checkpoint(
        tmp_path / "step_000010",
        metadata=metadata,
        tensors=tensors,
        train_state_payload={"optimizer": {}, "scheduler": {}, "step": 10},
    )
    assert (retried / "flow_decoder.safetensors").exists()
    assert not (tmp_path / "step_000010.tmp").exists()


def test_saver_surfaces_background_errors() -> None:
    """A failed background save re-raises at the next join/submit — and
    the saver stays usable afterwards."""
    saver = AsyncCheckpointSaver(group=None, is_main=True, world_size=1, zero1=False)
    capture = capture_optimizer_state(
        torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=1e-3),
        zero1=False,
        is_main=True,
    )

    def explode(_: dict[str, Any]) -> None:
        raise ValueError("writer died")

    saver.submit(capture=capture, write=explode)
    with pytest.raises(RuntimeError, match="async checkpoint save failed"):
        saver.join()

    landed: list[dict[str, Any]] = []
    saver.submit(capture=capture, write=landed.append)
    saver.join()
    assert len(landed) == 1 and "param_groups" in landed[0]
