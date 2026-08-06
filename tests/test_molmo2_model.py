"""Molmo2 full-model assembly (WP4 slice) — CPU oracles on the tiny
fixture:

- multimodal mask semantics vs an independent brute-force rendering of the
  reference's ``token_type_ids_mask_function`` composition (causal OR
  both-image, then key padding excluded),
- vision injection: additive at ``image_patch_id`` positions only, count
  asserted, non-patch embeddings untouched,
- compose forward: shapes, finiteness, and left-pad invariance end-to-end
  (the bidirectional image block must not admit pad keys),
- cache-free greedy decode: deterministic continuation, stop-id handling,
  and agreement with the model's own full-sequence argmax.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bijou.molmo2.config import Molmo2Config
from bijou.molmo2.model import Molmo2Model, build_multimodal_mask, load_model
from bijou.molmo2.testing import tiny_config_json, write_tiny_text_checkpoint


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_tiny_text_checkpoint(
        tmp_path_factory.mktemp("molmo2-model") / "tiny-molmo2",
    )


@pytest.fixture(scope="module")
def model(tiny_checkpoint: Path) -> Molmo2Model:
    return load_model(str(tiny_checkpoint), dtype=torch.float32)


def tiny_ids() -> tuple[torch.Tensor, torch.Tensor]:
    """A hand-built two-row batch in the tiny id space (vocab 512,
    image_patch_id 514): row 0 unpadded with a 4-patch image block plus
    markers, row 1 shorter, left-padded. Returns (input_ids,
    attention_mask)."""
    config = Molmo2Config.from_dict(tiny_config_json())
    patch = config.image_patch_id  # 514
    marker = patch + 1  # another extension id: an im_col/im_end stand-in
    row0 = [1, marker, patch, patch, patch, patch, marker, 2, 3, 4]
    row1 = [5, patch, patch, marker, 6, 7]
    width = len(row0)
    pad = 0
    input_ids = torch.tensor([row0, [pad] * (width - len(row1)) + row1])
    attention_mask = torch.tensor(
        [[1] * width, [0] * (width - len(row1)) + [1] * len(row1)],
    )
    return input_ids, attention_mask


def image_type_mask_of(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    config = Molmo2Config.from_dict(tiny_config_json())
    patch = config.image_patch_id
    return ((input_ids == patch) | (input_ids == patch + 1)) & attention_mask.bool()


def tiny_vision_inputs(
    num_patch_tokens: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crops + pooling indices producing exactly ``num_patch_tokens[b]``
    valid pooled tokens for each row (single view, singleton pooling
    groups, -1 rows pad the shorter sample)."""
    config = Molmo2Config.from_dict(tiny_config_json())
    assert config.vit is not None
    torch.manual_seed(2)
    batch = len(num_patch_tokens)
    crops = torch.randn(batch, 1, config.vit.image_num_pos, config.vit.patch_dim)
    max_tokens = max(num_patch_tokens)
    idx = torch.full((batch, max_tokens, 1), -1, dtype=torch.long)
    for row, count in enumerate(num_patch_tokens):
        assert count <= config.vit.image_num_pos
        idx[row, :count, 0] = torch.arange(count)
    return crops, idx


def test_multimodal_mask_matches_reference_semantics() -> None:
    input_ids, attention_mask = tiny_ids()
    image = image_type_mask_of(input_ids, attention_mask)
    spec = build_multimodal_mask(
        image_type_mask=image,
        padding_mask=attention_mask,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert not spec.is_causal
    assert spec.tensor is not None
    batch, seq = input_ids.shape
    allowed = spec.tensor[:, 0] == 0.0
    for b in range(batch):
        for q in range(seq):
            for kv in range(seq):
                # The reference composition: causal OR'd with the token-
                # type function (both image => allowed), AND key is real.
                causal = kv <= q
                image_block = bool(image[b, q]) and bool(image[b, kv])
                expected = (causal or image_block) and bool(attention_mask[b, kv])
                assert allowed[b, q, kv] == expected, (b, q, kv)


def test_injection_additive_at_patch_positions_only(model: Molmo2Model) -> None:
    input_ids, _ = tiny_ids()
    patch_counts = [int((row == model.image_patch_id).sum()) for row in input_ids]
    crops, pooled_idx = tiny_vision_inputs(patch_counts)

    embeds = model.build_input_embeddings(
        input_ids,
        crops=crops,
        pooled_patches_idx=pooled_idx,
    )
    base = model.text.transformer.wte(input_ids)
    features = model.vision(crops, pooled_idx)

    is_patch = input_ids == model.image_patch_id
    # Non-patch positions (markers, text, padding) are untouched.
    torch.testing.assert_close(embeds[~is_patch], base[~is_patch])
    # Patch positions got exactly base + feature, in scatter order.
    torch.testing.assert_close(embeds[is_patch], base[is_patch] + features)


def test_injection_count_mismatch_dies_loud(model: Molmo2Model) -> None:
    input_ids, _ = tiny_ids()
    crops, pooled_idx = tiny_vision_inputs([2, 2])  # 4 + 2 patches expected
    with pytest.raises(ValueError, match="pooled feature rows"):
        model.build_input_embeddings(
            input_ids,
            crops=crops,
            pooled_patches_idx=pooled_idx,
        )


def test_compose_forward_and_left_pad_invariance(model: Molmo2Model) -> None:
    input_ids, attention_mask = tiny_ids()
    image = image_type_mask_of(input_ids, attention_mask)
    patch_counts = [int((row == model.image_patch_id).sum()) for row in input_ids]
    crops, pooled_idx = tiny_vision_inputs(patch_counts)

    logits = model(
        input_ids,
        crops=crops,
        pooled_patches_idx=pooled_idx,
        image_type_mask=image,
        attention_mask=attention_mask,
    )
    vocab = model.text.config.vocab_size
    assert logits.shape == (*input_ids.shape, vocab)
    assert torch.isfinite(logits).all()

    # Row 1 alone, unpadded, must reproduce its padded logits exactly:
    # padding may leak through neither the causal part nor the
    # bidirectional image block.
    real = attention_mask[1].bool()
    solo_ids = input_ids[1:, real]
    solo_logits = model(
        solo_ids,
        crops=crops[1:],
        pooled_patches_idx=pooled_idx[1:],
        image_type_mask=image[1:, real],
        attention_mask=None,
    )
    torch.testing.assert_close(
        logits[1, real],
        solo_logits[0],
        rtol=1e-4,
        atol=1e-4,
    )


def test_greedy_generate_matches_argmax_and_stops(model: Molmo2Model) -> None:
    input_ids, attention_mask = tiny_ids()
    image = image_type_mask_of(input_ids, attention_mask)
    patch_counts = [int((row == model.image_patch_id).sum()) for row in input_ids]
    crops, pooled_idx = tiny_vision_inputs(patch_counts)

    emitted = model.greedy_generate(
        input_ids,
        crops=crops,
        pooled_patches_idx=pooled_idx,
        image_type_mask=image,
        attention_mask=attention_mask,
        max_new_tokens=3,
    )
    assert [len(row) for row in emitted] == [3, 3]

    # First emitted token == the model's own last-position argmax.
    logits = model(
        input_ids,
        crops=crops,
        pooled_patches_idx=pooled_idx,
        image_type_mask=image,
        attention_mask=attention_mask,
    )
    first = logits[:, -1].argmax(-1)
    assert [row[0] for row in emitted] == [int(t) for t in first]

    # A stop id ends the row at its first emission.
    stopped = model.greedy_generate(
        input_ids,
        crops=crops,
        pooled_patches_idx=pooled_idx,
        image_type_mask=image,
        attention_mask=attention_mask,
        max_new_tokens=3,
        stop_ids=frozenset({int(first[0])}),
    )
    assert stopped[0] == [int(first[0])]
