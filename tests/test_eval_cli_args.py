"""bijou.eval argument interactions — pure parse-time, no model or data.

--dump-draws only makes sense on a run that actually ensembles flow
draws: at draws=1 the prediction IS the single draw (already covered by
--dump-predictions), and without a checkpoint there is no bijou policy
to tap. Both misuses must die at the parser, before any data loads.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from bijou.eval.cli import parse_args


def _parse(monkeypatch: pytest.MonkeyPatch, *extra: str) -> argparse.Namespace:
    monkeypatch.setattr(
        "sys.argv",
        ["bijou.eval", "--data", "corpus", *extra],
    )
    return parse_args()


def test_dump_draws_alone_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--dump-draws", "draws.npz")


def test_dump_draws_at_single_draw_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--dump-draws", "draws.npz", "--checkpoint", "ckpt")


def test_dump_draws_without_checkpoint_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--dump-draws", "draws.npz", "--sample-draws", "10")


def test_dump_draws_on_an_ensembled_checkpoint_parses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parse(
        monkeypatch,
        "--dump-draws",
        "draws.npz",
        "--checkpoint",
        "ckpt",
        "--sample-draws",
        "10",
    )
    assert args.dump_draws == Path("draws.npz")
    assert args.sample_draws == 10
