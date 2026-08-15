"""Molmo2 vision tower + connector (WP2) — CPU oracles on the tiny fixture:

- the backbone instantiates only ``max(vit_layers) + 1`` blocks (the
  build-time truncation the released checkpoint ships);
- tap extraction concatenates the named block outputs in ``vit_layers``
  order (raw block outputs — the tower has no final norm);
- the 2x2 attention pooling honors the validity mask: a partial group
  (-1-padded members) pools identically to the same group expressed
  without padding, and all--1 rows (padding tokens) are dropped;
- the gated projector convention (``w2(act(w1(x)) * w3(x))``);
- ``float32_attention`` preserves the caller's dtype;
- the loader strict-loads the vision keys from the real layout and the
  text loader still skips them.

HF parity on real weights/processor inputs is ``verify_parity --vision``
(port plan §4), run CPU-side like the text gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from bijou.modelling.molmo2.config import Molmo2Config
from bijou.modelling.molmo2.loading import load_text_model, load_vision_backbone
from bijou.modelling.molmo2.testing import tiny_config_json, write_tiny_text_checkpoint
from bijou.modelling.molmo2.vision import Molmo2VisionBackbone


def tiny_backbone() -> Molmo2VisionBackbone:
    torch.manual_seed(0)
    config = Molmo2Config.from_dict(tiny_config_json())
    assert config.vit is not None and config.adapter is not None
    return Molmo2VisionBackbone(config.vit, config.adapter)


def tiny_images(batch: int = 1, views: int = 2) -> torch.Tensor:
    torch.manual_seed(1)
    backbone = tiny_backbone()
    vit = backbone.vit_config
    return torch.randn(batch, views, vit.image_num_pos, vit.patch_dim)


def test_tower_truncated_to_deepest_tap() -> None:
    backbone = tiny_backbone()
    # vit_layers (-2, -4) of a 4-layer tower -> absolute (2, 0) -> 3 blocks.
    assert backbone.vit_layers == (2, 0)
    assert len(backbone.image_vit.transformer.resblocks) == 3


def test_tap_concat_order_and_raw_block_outputs() -> None:
    backbone = tiny_backbone()
    images = tiny_images()
    batch, views, num_patches, _ = images.shape
    hidden_states = backbone.image_vit(images.view(batch * views, num_patches, -1))
    features = backbone.encode_image(images)
    expected = torch.cat([hidden_states[2], hidden_states[0]], dim=-1)
    assert torch.equal(features, expected.view(batch, views, num_patches, -1))


def test_pooling_mask_and_padding_token_semantics() -> None:
    backbone = tiny_backbone()
    images = tiny_images()

    # Three output tokens: a full pair, a partial group (one -1 member),
    # and an all--1 padding row that must vanish from the output.
    idx = torch.tensor([[[0, 1], [2, -1], [-1, -1]]])
    out = backbone(images, idx)
    assert out.shape == (2, backbone.adapter_config.text_hidden_size)

    # The partial group must pool exactly like the same group without the
    # padded slot — the validity mask, not the clipped index (0), decides
    # what the query averages and what the attention sees.
    out_single = backbone(images, torch.tensor([[[2]]]))
    torch.testing.assert_close(out[1], out_single[0])


def test_projector_gating_convention() -> None:
    backbone = tiny_backbone()
    projector = backbone.image_projector
    x = torch.randn(3, backbone.adapter_config.hidden_size)
    expected = projector.w2(F.silu(projector.w1(x)) * projector.w3(x))
    torch.testing.assert_close(projector(x), expected)


def test_float32_attention_preserves_caller_dtype() -> None:
    backbone = tiny_backbone().to(torch.bfloat16)
    images = tiny_images().to(torch.bfloat16)
    out = backbone(images, torch.tensor([[[0, 1]]]))
    assert out.dtype == torch.bfloat16


def test_vision_loader_strict_and_text_loader_skips(tmp_path: Path) -> None:
    checkpoint = write_tiny_text_checkpoint(tmp_path / "tiny-molmo2")
    vision = load_vision_backbone(checkpoint)
    assert len(vision.image_vit.transformer.resblocks) == 3

    images = tiny_images()
    out = vision(images, torch.tensor([[[0, 1], [2, 3]]]))
    assert out.shape == (2, vision.adapter_config.text_hidden_size)

    # Text loader unaffected by the vision keys sharing the file.
    text = load_text_model(checkpoint, truncate_layers=2)
    assert len(text.transformer.blocks) == 2


def test_dynamic_grid_refused() -> None:
    backbone = tiny_backbone()
    images = tiny_images()
    with pytest.raises(NotImplementedError, match="image_num_pos"):
        backbone(images[:, :, :4], torch.tensor([[[0, 1]]]))
