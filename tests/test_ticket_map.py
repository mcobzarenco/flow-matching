"""Per-dataset ticket routing oracles (#1 noise-ladder rung 2, pre-reg
2026-08-08): the --noise-ticket-map instrument's frozen contracts, CPU
only.

Load-bearing guarantees: (1) the committed stage-01 routing map loads
and its canonical-form sha256 reproduces the pre-registered
`15d9293553ac1a88…` EXACTLY — the stage-2 run's provenance is checked
against this constant; (2) ROUTING — _flow_noise under a map feeds each
item bit-exactly its dataset's bank row, never ticket 0; (3) every
refusal branch aborts loud (map without bank, draws != 1, unmapped
dataset, non-integer / out-of-range routes) — a silent fallback would
blend routed and unrouted rows inside one npz. The policy-NAME
(_ticketmap) and report/npz provenance contracts are GPU-side and ride
the preflight adjudicator (noise_ladder_preflight.sh), which
byte-compares a routed decode against a plain single-ticket decode —
the pre-reg's stage-2 oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bijou.eval.cli import parse_args
from bijou.eval.policies import BijouPolicy, load_ticket_map, load_tickets

REPO = Path(__file__).resolve().parents[1]
BANK = REPO / "plans/tickets_goldenticket_m64.npz"
ANALYSIS = REPO / "reports/analysis__noise_ladder_stage01.json"
# Pre-registered map sha (posts/2026-08-08-prereg-noise-ladder-perdataset.md
# stage 1): the stage-2 run must carry exactly this in its provenance.
MAP_SHA = "15d9293553ac1a8878e0b7b0c385f03127a518d96e408bc1f496f5d8c4ec2173"
TOP10 = [33, 2, 0, 51, 10, 59, 38, 28, 15, 36]
SHAPE = (50, 6)


def _parse(monkeypatch: pytest.MonkeyPatch, *extra: str) -> argparse.Namespace:
    monkeypatch.setattr(
        "sys.argv",
        ["bijou.eval", "--data", "corpus", *extra],
    )
    return parse_args()


def stub_policy(
    tickets: torch.Tensor,
    ticket_map: dict[str, int] | None,
) -> SimpleNamespace:
    """The _flow_noise seam's inputs, without a checkpoint load."""
    return SimpleNamespace(
        tickets=tickets,
        ticket_map=ticket_map,
        noise_key="stable",
        seed=0,
    )


def items_for(repos: list[str]) -> list[dict[str, object]]:
    return [
        {"repo_id": repo, "episode_index": 3, "frame_index": 40 + i}
        for i, repo in enumerate(repos)
    ]


def routed_noise(
    policy: SimpleNamespace,
    repos: list[str],
) -> torch.Tensor:
    items = items_for(repos)
    return BijouPolicy._flow_noise(
        policy,  # type: ignore[arg-type]
        items,
        list(range(len(items))),
        1,
        SHAPE,
    )


# --- the committed map -------------------------------------------------


def test_committed_map_reproduces_preregistered_sha() -> None:
    mapping, sha = load_ticket_map(ANALYSIS, 64)
    assert sha == MAP_SHA
    assert len(mapping) == 792
    # Frozen routing rule: image within top-10 + {33} (33 is in the top-10).
    assert set(mapping.values()) <= set(TOP10)


def test_bare_map_loads_with_same_canonical_sha(tmp_path: Path) -> None:
    mapping = {"b/two": 2, "a/one": 33}
    path = tmp_path / "map.json"
    path.write_text(json.dumps(mapping))
    loaded, sha = load_ticket_map(path, 64)
    assert loaded == mapping
    expected = hashlib.sha256(
        json.dumps(mapping, sort_keys=True).encode(),
    ).hexdigest()
    assert sha == expected


# --- routing correctness ----------------------------------------------


def test_routed_noise_is_the_datasets_bank_row() -> None:
    bank, _sha = load_tickets(BANK)
    mapping = {"a/one": 2, "b/two": 51, "c/three": 33}
    noise = routed_noise(
        stub_policy(bank, mapping),
        ["b/two", "a/one", "b/two", "c/three"],
    )
    assert noise.shape == (4, *SHAPE)
    for row, ticket in zip(noise, [51, 2, 51, 33], strict=True):
        assert torch.equal(row, bank[ticket])


def test_unrouted_stub_keeps_the_plain_ticket_path() -> None:
    bank, _sha = load_tickets(BANK)
    noise = routed_noise(stub_policy(bank, None), ["a/one", "b/two"])
    # Plain ticket mode at draws=1: every frame integrates from
    # tickets[0] (the golden-ticket contract, unchanged by the routing
    # seam).
    assert torch.equal(noise, bank[:1].repeat_interleave(2, dim=0))


# --- refusal branches (all abort, never silent) ------------------------


def test_unmapped_dataset_aborts() -> None:
    bank, _sha = load_tickets(BANK)
    with pytest.raises(SystemExit, match="covers no route"):
        routed_noise(stub_policy(bank, {"a/one": 2}), ["a/one", "z/unmapped"])


def test_routed_multi_draw_aborts() -> None:
    bank, _sha = load_tickets(BANK)
    policy = stub_policy(bank, {"a/one": 2})
    with pytest.raises(SystemExit, match="single-draw"):
        BijouPolicy._flow_noise(
            policy,  # type: ignore[arg-type]
            items_for(["a/one"]),
            [0],
            2,
            SHAPE,
        )


def test_map_without_structure_aborts(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    path.write_text(json.dumps(["not", "a", "map"]))
    with pytest.raises(SystemExit, match="no routing map"):
        load_ticket_map(path, 64)


def test_non_integer_route_aborts(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"a/one": "33"}))
    with pytest.raises(SystemExit, match="non-integer"):
        load_ticket_map(path, 64)
    path.write_text(json.dumps({"a/one": True}))
    with pytest.raises(SystemExit, match="non-integer"):
        load_ticket_map(path, 64)


def test_out_of_bank_route_aborts(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"a/one": 64}))
    with pytest.raises(SystemExit, match="outside the bank"):
        load_ticket_map(path, 64)


def test_policy_refuses_map_without_bank() -> None:
    # The guard fires before the checkpoint loads, so a bogus path
    # never reaches from_checkpoint.
    with pytest.raises(SystemExit, match="requires --noise-tickets"):
        BijouPolicy(
            Path("nonexistent-checkpoint"),
            device=torch.device("cpu"),
            seed=0,
            ticket_map=ANALYSIS,
        )


def test_policy_refuses_routed_multi_draw() -> None:
    with pytest.raises(SystemExit, match="no routed semantics"):
        BijouPolicy(
            Path("nonexistent-checkpoint"),
            device=torch.device("cpu"),
            seed=0,
            sample_draws=2,
            tickets=BANK,
            ticket_map=ANALYSIS,
        )


# --- parser interactions ----------------------------------------------


def test_parser_refuses_map_without_tickets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit):
        _parse(
            monkeypatch,
            "--checkpoint",
            "ckpt",
            "--noise-ticket-map",
            "map.json",
        )


def test_parser_refuses_map_at_multi_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit):
        _parse(
            monkeypatch,
            "--checkpoint",
            "ckpt",
            "--noise-tickets",
            "bank.npz",
            "--noise-ticket-map",
            "map.json",
            "--sample-draws",
            "10",
        )


def test_parser_accepts_routed_single_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parse(
        monkeypatch,
        "--checkpoint",
        "ckpt",
        "--noise-tickets",
        "bank.npz",
        "--noise-ticket-map",
        "map.json",
    )
    assert args.noise_ticket_map == Path("map.json")
    assert args.sample_draws == 1
