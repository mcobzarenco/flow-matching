"""Oracle for the pdnorm panel anchor-ladder chart (PRE-GO prep).

Rung values and labels are the pre-reg's frozen wear-audit ladder
(text renders inside the PNG, so rows are asserted structurally, house
convention); the endpoint slot stays a placeholder until --endpoint
stamps it on GO.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest

from fontaine.scripts.pdnorm_panel_ladder_chart import (
    ENDPOINT_LABEL,
    LADDER,
    PENDING_NOTE,
    build_rungs,
    main,
)


def test_ladder_rungs_are_the_frozen_prereg_values() -> None:
    assert [(r[0], r[1]) for r in LADDER] == [
        ("disc-1000 raw (worn demos global table)", 58.14),
        ("disc-1000 re-worn (honest per-repo rows)", 27.40),
        # Measured 08:22Z 08-18 (record-only) — no longer a FILL slot.
        ("released pre-SFT (own table, measured 08-18)", 25.89),
        ("repo-midpoint null (constant)", 25.15),
        ("worn-box clamp floor", 14.40),
        ("state-copy", 8.37),
    ]


def test_endpoint_slot_pending_then_stamped() -> None:
    pending = build_rungs(None)
    assert pending[0][0] == ENDPOINT_LABEL
    assert pending[0][1] is None
    assert pending[0][3] == PENDING_NOTE
    assert pending[1:] == build_rungs(23.5)[1:]  # ladder itself frozen
    stamped = build_rungs(23.5)
    assert stamped[0][1] == 23.5
    assert "FILL" not in stamped[0][3]


@pytest.mark.parametrize("endpoint", [None, 23.5])
def test_main_writes_png_and_matching_b64(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    endpoint: float | None,
) -> None:
    png = tmp_path / "ladder.png"
    b64 = tmp_path / "ladder.b64"
    argv = [
        "pdnorm_panel_ladder_chart.py",
        "--out-png",
        str(png),
        "--out-b64",
        str(b64),
    ]
    if endpoint is not None:
        argv += ["--endpoint", str(endpoint)]
    monkeypatch.setattr(sys, "argv", argv)
    assert main() == 0
    data = png.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert base64.b64decode(b64.read_text()) == data
