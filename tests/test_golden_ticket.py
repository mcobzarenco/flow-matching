"""Golden-ticket instrument oracles (#1 screen, pre-reg 2026-08-07):
the ticket noise mode's frozen contracts, on the tiny FlowDecoder
fixture and the committed bank.

The load-bearing guarantees: (1) CONTRACT — ticket mode at draws=1
feeds exactly tickets[0] to the solver, and the batched call
reproduces bit-exact the direct sample_actions(noise=ticket) call on
the same frames; (2) TICKET PROPERTY — within one run, different
frames receive byte-identical noise for the same draw index, asserted
on the tensor the policy actually produces (not by construction);
(3) DETERMINISM — bank generation, npz round-trip, and the noise path
are all byte-stable across runs, and the COMMITTED bank file matches
its pinned sha256 (regenerating it can never silently move the
screen's candidates); (4) the keyed-noise path is byte-identical to
the pre-refactor inline assembly — landing tickets must not redraw
any existing flow number. The pooling-reuse oracle (banked 6.5997 +
the 10 per-draw probe MAEs) lives in fontaine/scripts/ticket_scores.py
--oracle — it needs the banked local npz artifacts.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from test_batched_draws import randomized_decoder
from test_flow_decoder import ACTION_DIM, BATCH, CHUNK, fabricate

from bijou.eval.policies import (
    BijouPolicy,
    generate_tickets,
    load_tickets,
    noise_for_item,
)
from bijou.model import SamplingMethod

# The committed bank (generated once by make_golden_tickets.py). Both
# pins are frozen: FILE_SHA is the provenance every read quotes;
# CONTENT_SHA pins the candidate values themselves.
BANK = Path(__file__).resolve().parents[1] / "plans/tickets_goldenticket_m64.npz"
BANK_FILE_SHA = "9bb13bc47a92f7cc764e81022a9a7b05dbb9ec391eb9ba8ab14d675c955cc7c0"
BANK_CONTENT_SHA = "a07c062a6f0aa1638e022166cd3fa351ee94fe6c91e0ca701ad5bc023f4015da"

DRAWS = 3
SHAPE = (CHUNK, ACTION_DIM)


def fixture_bank() -> torch.Tensor:
    return torch.from_numpy(generate_tickets(4, SHAPE))


def stub_policy(tickets: torch.Tensor | None) -> SimpleNamespace:
    """The _flow_noise seam's inputs, without a checkpoint load."""
    return SimpleNamespace(
        tickets=tickets,
        ticket_map=None,
        noise_key="stable",
        seed=0,
    )


def items_and_indices() -> tuple[list[dict[str, object]], list[int]]:
    """Two frames with DIFFERENT identity triples and concat indices —
    under any keyed mode their noise must differ; under ticket mode it
    must not."""
    return (
        [
            {"repo_id": "a/x", "episode_index": 0, "frame_index": 3},
            {"repo_id": "b/y", "episode_index": 7, "frame_index": 41},
        ],
        [5, 991],
    )


def seam(
    policy: SimpleNamespace,
    items: list[dict[str, object]],
    indices: list[int],
    draws: int,
    shape: tuple[int, int],
) -> torch.Tensor:
    # The stub carries exactly the attributes _flow_noise reads; the
    # cast keeps pyright honest about the deliberate self-substitution.
    return BijouPolicy._flow_noise(
        cast("BijouPolicy", policy),
        items,
        indices,
        draws,
        shape,
    )


def flow_noise(policy: SimpleNamespace, draws: int) -> torch.Tensor:
    items, indices = items_and_indices()
    return seam(policy, items, indices, draws, SHAPE)


def test_committed_bank_matches_both_pins() -> None:
    assert hashlib.sha256(BANK.read_bytes()).hexdigest() == BANK_FILE_SHA
    bank, sha = load_tickets(BANK)
    assert sha == BANK_FILE_SHA
    assert bank.shape == (64, 50, 6) and bank.dtype == torch.float32
    content = hashlib.sha256(bank.numpy().tobytes()).hexdigest()
    assert content == BANK_CONTENT_SHA
    # The seed schedule regenerates the committed values exactly.
    assert np.array_equal(generate_tickets(64, (50, 6)), bank.numpy())


def test_ticket_property_asserted_in_process() -> None:
    """Oracle 2: the tensor the policy path produces gives every frame
    byte-identical noise per draw index — and it IS the bank row."""
    bank = fixture_bank()
    noise = flow_noise(stub_policy(bank), DRAWS)
    assert noise.shape == (DRAWS * BATCH, CHUNK, ACTION_DIM)
    for draw in range(DRAWS):
        for item in range(BATCH):
            row = noise[draw * BATCH + item]
            assert torch.equal(row, bank[draw])
    # And the mode genuinely switches: keyed noise on the same frames
    # differs per frame and per draw.
    keyed = flow_noise(stub_policy(None), DRAWS)
    assert not torch.equal(keyed[0], keyed[1])
    assert not torch.equal(keyed[0], noise[0])


def test_draws1_contract_bit_exact() -> None:
    """Oracle 1: ticket mode at draws=1 reproduces bit-exact the direct
    sample_actions(noise=ticket) call on the same frames."""
    decoder = randomized_decoder()
    memory, state, _, _ = fabricate()
    bank = fixture_bank()
    noise = flow_noise(stub_policy(bank), 1)
    assert torch.equal(noise, bank[0].expand(BATCH, CHUNK, ACTION_DIM))
    via_mode = decoder.sample_actions(
        memory,
        state,
        noise=noise,
        num_steps=4,
        method=SamplingMethod.HEUN,
    )
    direct = decoder.sample_actions(
        memory,
        state,
        noise=bank[0].expand(BATCH, CHUNK, ACTION_DIM).clone(),
        num_steps=4,
        method=SamplingMethod.HEUN,
    )
    assert torch.equal(via_mode, direct)


def test_two_run_determinism() -> None:
    """Oracle 3: bank generation and the noise path are byte-stable."""
    assert np.array_equal(generate_tickets(4, SHAPE), generate_tickets(4, SHAPE))
    bank = fixture_bank()
    assert torch.equal(
        flow_noise(stub_policy(bank), DRAWS),
        flow_noise(stub_policy(bank), DRAWS),
    )


def test_keyed_path_unchanged_by_refactor() -> None:
    """Landing tickets must not redraw any existing flow number: the
    seam's keyed output is byte-identical to the pre-refactor inline
    assembly, for both keyings, at draws 1 and 3."""
    items, indices = items_and_indices()
    for key in ("stable", "index"):
        for draws in (1, DRAWS):
            via_seam = seam(
                SimpleNamespace(tickets=None, noise_key=key, seed=11),
                items,
                indices,
                draws,
                SHAPE,
            )
            inline = torch.cat(
                [
                    torch.stack(
                        [
                            noise_for_item(key, 11, item, index, draw, SHAPE)
                            for item, index in zip(items, indices, strict=True)
                        ],
                    )
                    for draw in range(draws)
                ],
            )
            assert torch.equal(via_seam, inline)


def test_loud_refusals(tmp_path: Path) -> None:
    bank = fixture_bank()
    with pytest.raises(SystemExit, match="tickets in"):
        flow_noise(stub_policy(bank), 5)  # draws > bank count
    with pytest.raises(SystemExit, match="shaped"):
        seam(stub_policy(bank), *items_and_indices(), 1, (CHUNK + 1, ACTION_DIM))
    empty = tmp_path / "empty.npz"
    np.savez(empty, other=np.zeros(3, dtype=np.float32))
    with pytest.raises(SystemExit, match="no 'tickets'"):
        load_tickets(empty)
    f64 = tmp_path / "f64.npz"
    np.savez(f64, tickets=np.zeros((2, 50, 6)))
    with pytest.raises(SystemExit, match="float32"):
        load_tickets(f64)


def test_npz_roundtrip_and_sha(tmp_path: Path) -> None:
    bank = generate_tickets(4, SHAPE)
    path = tmp_path / "bank.npz"
    np.savez(path, tickets=bank)
    loaded_a, sha_a = load_tickets(path)
    loaded_b, sha_b = load_tickets(path)
    assert sha_a == sha_b == hashlib.sha256(path.read_bytes()).hexdigest()
    assert torch.equal(loaded_a, loaded_b)
    assert np.array_equal(loaded_a.numpy(), bank)
