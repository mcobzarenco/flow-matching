"""Oracles for --prune-superseded-optim (2026-08-19 root-disk-full
incident: six ~31 GiB offload-optim optimizer.pt saves filled the disk
mid-run). The helper deletes optimizer.pt from all but the newest two
published step_* checkpoints; weights and metadata are never touched."""

from pathlib import Path

from bijou.train.args import TrainArgs, _build_parser
from bijou.train.saving import prune_superseded_optimizers


def _make_step(save_dir: Path, step: int, *, optimizer: bool = True) -> Path:
    directory = save_dir / f"step_{step:06d}"
    directory.mkdir(parents=True)
    (directory / "model.safetensors").write_bytes(b"weights")
    if optimizer:
        (directory / "optimizer.pt").write_bytes(b"moments")
    return directory


def test_prunes_all_but_newest_two(tmp_path: Path) -> None:
    dirs = [_make_step(tmp_path, s) for s in (500, 1000, 1500, 2000)]
    pruned = prune_superseded_optimizers(tmp_path)
    assert pruned == [dirs[0] / "optimizer.pt", dirs[1] / "optimizer.pt"]
    assert not (dirs[0] / "optimizer.pt").exists()
    assert not (dirs[1] / "optimizer.pt").exists()
    assert (dirs[2] / "optimizer.pt").exists()
    assert (dirs[3] / "optimizer.pt").exists()
    # Weights untouched everywhere — pruned directories stay loadable.
    assert all((d / "model.safetensors").read_bytes() == b"weights" for d in dirs)


def test_two_or_fewer_saves_are_a_no_op(tmp_path: Path) -> None:
    assert prune_superseded_optimizers(tmp_path) == []
    _make_step(tmp_path, 500)
    assert prune_superseded_optimizers(tmp_path) == []
    _make_step(tmp_path, 1000)
    assert prune_superseded_optimizers(tmp_path) == []
    assert (tmp_path / "step_000500" / "optimizer.pt").exists()


def test_ignores_staging_debris_and_foreign_names(tmp_path: Path) -> None:
    _make_step(tmp_path, 500)
    _make_step(tmp_path, 1000)
    _make_step(tmp_path, 1500)
    # .tmp staging debris, a non-step directory, and a step-named FILE
    # must never match — only published step_* directories count.
    debris = tmp_path / "step_002000.tmp"
    debris.mkdir()
    (debris / "optimizer.pt").write_bytes(b"mid-write")
    (tmp_path / "step_extra").mkdir()
    (tmp_path / "step_9").write_bytes(b"a file, not a checkpoint")
    pruned = prune_superseded_optimizers(tmp_path)
    assert pruned == [tmp_path / "step_000500" / "optimizer.pt"]
    assert (debris / "optimizer.pt").read_bytes() == b"mid-write"


def test_already_pruned_directories_are_skipped(tmp_path: Path) -> None:
    _make_step(tmp_path, 500, optimizer=False)
    _make_step(tmp_path, 1000)
    _make_step(tmp_path, 1500)
    _make_step(tmp_path, 2000)
    pruned = prune_superseded_optimizers(tmp_path)
    assert pruned == [tmp_path / "step_001000" / "optimizer.pt"]


def test_flag_parses_and_defaults_off() -> None:
    parser = _build_parser()
    base = ["--family", "gemma_flow", "--train-data", "corpus"]
    raw = parser.parse_args(base)
    assert (
        TrainArgs.from_namespace(raw, parser, checkpoint=None).prune_superseded_optim
        is False
    )
    raw = parser.parse_args([*base, "--prune-superseded-optim"])
    assert (
        TrainArgs.from_namespace(raw, parser, checkpoint=None).prune_superseded_optim
        is True
    )
