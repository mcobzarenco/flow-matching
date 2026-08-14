"""Oracles for MolmoAct2 action-side processing (port item 2).

Two tiers:

- Fixture parity: ``fontaine/scripts/molmoact2_processing_goldens.py``
  (run in the molmoact2 venv) drove THEIR real lerobot pipeline —
  masked q01/q99 normalizer -> clamp -> ``MolmoAct2PackInputsProcessorStep``
  over the shipped HF AutoProcessor — on deterministic inputs and froze
  the outputs; the tests here reproduce every tensor from
  ``bijou.molmoact2.processing`` byte-exact (measured max|Δ| = 0.0 on
  all cases at banking time, including the uint8 resize path across
  torchvision 0.25/0.26).
- Pure oracles: pinned token ids vs the checkpoint tokenizer, the
  discrete-state binning, task-text normalization, the prompt template,
  and the sequence-budget guard — no fixtures needed.

The goldens script is imported as a module (its module level is
stdlib+numpy+torch only) so inputs are regenerated from the SAME code
that banked them.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from bijou.encoders.molmoact2_processing import (
    ACTION_OUTPUT_ID,
    BOS_ID,
    IM_END_ID,
    IM_PATCH_ID,
    IM_START_ID,
    RESIZE_GRID,
    STATE_TOKEN_0_ID,
    QuantileStats,
    build_robot_prompt,
    discrete_state_string,
    encode_action_prompt,
    image_token_ids_resize,
    infer_max_sequence_length,
    load_norm_stats,
    normalize_q01q99,
    normalize_state,
    normalize_task_text,
    pack_action_example,
    process_image_resize,
    to_uint8_rgb,
    unnormalize_action,
    unnormalize_q01q99,
)
from bijou.molmo2.tokenizer import Molmo2TextTokenizer

_GOLDENS_PATH = (
    Path(__file__).resolve().parents[1]
    / "fontaine/scripts/molmoact2_processing_goldens.py"
)
_spec = importlib.util.spec_from_file_location(
    "molmoact2_processing_goldens",
    _GOLDENS_PATH,
)
assert _spec is not None and _spec.loader is not None
goldens = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(goldens)

needs_checkpoint = pytest.mark.skipif(
    not goldens.DEFAULT_HF_DIR.exists(),
    reason=f"MolmoAct2 checkpoint export not present at {goldens.DEFAULT_HF_DIR}",
)


Case = tuple[str, list[tuple[int, int, str]], list[float], str]
StatsAndMeta = tuple[QuantileStats, QuantileStats, dict[str, Any]]


@pytest.fixture(scope="module")
def stats_and_meta() -> StatsAndMeta:
    return load_norm_stats(goldens.DEFAULT_HF_DIR, goldens.NORM_TAG)


@pytest.fixture(scope="module")
def tokenizer() -> Molmo2TextTokenizer:
    return Molmo2TextTokenizer(str(goldens.DEFAULT_HF_DIR))


@needs_checkpoint
@pytest.mark.parametrize("case", goldens.CASES, ids=[c[0] for c in goldens.CASES])
def test_pack_matches_their_pipeline(
    case: Case,
    stats_and_meta: StatsAndMeta,
    tokenizer: Molmo2TextTokenizer,
) -> None:
    """Full input-side parity vs the banked reference: ids, mask shape,
    pixel values (strided samples + per-crop stats), pooling indices,
    grids, and the normalized prompt state — all exact."""
    name, image_specs, state_row, task = case
    ref = np.load(goldens.FIXTURE_DIR / f"{name}.npz")
    _, state_stats, meta = stats_and_meta

    packed = pack_action_example(
        images=goldens.case_images(image_specs),
        state=torch.tensor(state_row, dtype=torch.float32),
        task=task,
        tokenizer=tokenizer,
        state_stats=state_stats,
        setup_type=meta["setup_type"],
        control_mode=meta["control_mode"],
    )

    np.testing.assert_array_equal(packed.input_ids.numpy(), ref["input_ids"][0])
    assert ref["attention_mask"].shape == ref["input_ids"].shape
    assert (ref["attention_mask"] == 1).all()

    np.testing.assert_array_equal(
        packed.normalized_state.numpy(),
        ref["state_norm"][0],
    )

    pixels = torch.cat([crops.crops for crops in packed.images]).numpy()
    assert pixels.shape == tuple(ref["pixel_shape"])
    strides = goldens.PIXEL_STRIDES
    np.testing.assert_array_equal(
        pixels[:: strides[0], :: strides[1], :: strides[2]],
        ref["pixel_samples"],
    )
    np.testing.assert_array_equal(pixels.mean(axis=(1, 2)), ref["pixel_mean"])
    np.testing.assert_array_equal(pixels.std(axis=(1, 2)), ref["pixel_std"])

    np.testing.assert_array_equal(
        np.concatenate([crops.pooled_idx.numpy() for crops in packed.images]),
        ref["image_token_pooling"],
    )
    np.testing.assert_array_equal(
        ref["image_grids"],
        np.tile(np.array(RESIZE_GRID), (len(image_specs), 1)),
    )
    np.testing.assert_array_equal(ref["image_num_crops"], np.ones(len(image_specs)))
    assert all(crops.grid == RESIZE_GRID for crops in packed.images)


@needs_checkpoint
@pytest.mark.parametrize(
    "case",
    goldens.ACTION_CASES,
    ids=[c[0] for c in goldens.ACTION_CASES],
)
def test_unnormalize_action_matches_their_pipeline(
    case: tuple[str, float],
    stats_and_meta: StatsAndMeta,
) -> None:
    """Output-side parity: clamp -> q01/q99 unnormalize, exact."""
    name, _scale = case
    ref = np.load(goldens.FIXTURE_DIR / f"action_{name}.npz")
    action_stats, _, _ = stats_and_meta
    out = unnormalize_action(torch.from_numpy(ref["action_norm"].copy()), action_stats)
    np.testing.assert_array_equal(out.numpy(), ref["action_out"])


@needs_checkpoint
def test_pinned_token_ids_match_checkpoint_tokenizer(
    tokenizer: Molmo2TextTokenizer,
) -> None:
    """Every pinned id resolves from the tokenizer as ONE token with the
    pinned value (the ids are NOT molmo2's — see module docstring)."""
    pinned = {
        "<setup_start>": 151_669,
        "<control_start>": 151_671,
        "<state_start>": 151_673,
        "<state_0>": STATE_TOKEN_0_ID,
        "<state_255>": STATE_TOKEN_0_ID + 255,
        "<action_output>": ACTION_OUTPUT_ID,
        "<im_start>": IM_START_ID,
        "<im_end>": IM_END_ID,
        "<im_patch>": IM_PATCH_ID,
        "<|im_end|>": BOS_ID,
    }
    for token, token_id in pinned.items():
        assert tokenizer.encode(token, add_special_tokens=False) == [token_id], token


@needs_checkpoint
def test_encode_inserts_bos_and_expands_images(tokenizer: Molmo2TextTokenizer) -> None:
    prompt = build_robot_prompt(
        task="wave",
        discrete_state="",
        setup_type="",
        control_mode="",
        num_images=1,
    )
    ids = encode_action_prompt(prompt, tokenizer)
    assert ids[0] == BOS_ID
    assert ids[1:199] == image_token_ids_resize()
    assert ids.count(IM_PATCH_ID) == 196
    assert ids[-1] == ACTION_OUTPUT_ID


def test_discrete_state_binning() -> None:
    """-1 -> bin 0, 0 -> bin 128 (rint half-to-even on 127.5), +1 -> 255;
    nan -> 0.0 -> 128, +-inf -> edge bins; out-of-range clamps."""
    text = discrete_state_string(
        np.array([-1.0, 0.0, 1.0, np.nan, np.inf, -np.inf, 7.0]),
    )
    expected = (
        "<state_start><state_0><state_128><state_255>"
        "<state_128><state_255><state_0><state_255><state_end>"
    )
    assert text == expected


def test_discrete_state_rejects_bad_bins() -> None:
    with pytest.raises(ValueError, match="num_state_tokens"):
        discrete_state_string(np.zeros(3), num_state_tokens=0)


def test_normalize_task_text_reference_cases() -> None:
    cases = {
        "Task: Pick up the cube.": "pick up the cube",
        "'grab the bottle'": "grab the bottle",
        "Pick the block. Place it left!": "pick the block; place it left",
        "The task is to wipe the table": "wipe the table",
        "PLACE THE FORK": "place the fork",
        "“fold the towel”": "fold the towel",
        "goal: stack cups; carefully": "stack cups; carefully",
        "instruction- sort  the   fruit": "sort the fruit",
        "": "",
        "  ": "",
    }
    for raw, expected in cases.items():
        assert normalize_task_text(raw) == expected, raw


def test_robot_prompt_template_exact() -> None:
    prompt = build_robot_prompt(
        task="wave",
        discrete_state="<state_start><state_1><state_end>",
        setup_type="lab arm",
        control_mode="absolute joint pose",
        num_images=2,
    )
    assert prompt == (
        "Image 1<|image|>Image 2<|image|><|im_start|>user\n"
        "The task is to wave. The setup is <setup_start>lab arm<setup_end>. "
        "The current state of the robot is <state_start><state_1><state_end>. "
        "The expected control mode is "
        "<control_start>absolute joint pose<control_end>. "
        "Given these, what action should the robot take to complete the task?"
        "<|im_end|>\n<|im_start|>assistant\n<action_output>"
    )
    # No state clause when the discrete string is empty; single image has
    # no "Image N" prefix; already-wrapped descriptors are not re-wrapped.
    bare = build_robot_prompt(
        task="wave",
        discrete_state="",
        setup_type="<setup_start>x<setup_end>",
        control_mode="",
        num_images=1,
    )
    assert "current state" not in bare
    assert bare.startswith("<|image|><|im_start|>user\n")
    assert "<setup_start><setup_start>" not in bare


def test_infer_max_sequence_length_reference_values() -> None:
    # Their formula: 2 images * 196 + 80 + 32 + state 6 + 32 = 542 -> 576.
    assert infer_max_sequence_length(num_images=2, state_dim=6) == 576
    assert infer_max_sequence_length(num_images=1, state_dim=6) == 384
    # Discrete branch (fidelity only): + 4 + 30 * max(6, ceil(6*0.95)).
    assert (
        infer_max_sequence_length(
            num_images=2,
            state_dim=6,
            include_discrete_action=True,
            action_dim=6,
            action_horizon=30,
        )
        == 768
    )


def test_load_norm_stats_rejects_empty_prompt_metadata(tmp_path: Path) -> None:
    """setup_type/control_mode render verbatim into the prompt — a
    missing or empty value must be loud, not a silent 'The setup is .'
    off-distribution prompt."""
    rows = {"q01": [0.0], "q99": [1.0]}
    base = {
        "action_stats": rows,
        "state_stats": rows,
        "setup_type": "tiny rig",
        "control_mode": "absolute joint pose",
    }
    for key, value in (
        ("setup_type", ""),
        ("control_mode", "  "),
        ("setup_type", None),
    ):
        broken = {**base, key: value}
        if value is None:
            del broken[key]
        (tmp_path / "norm_stats.json").write_text(
            json.dumps({"metadata_by_tag": {"tag": broken}}),
        )
        with pytest.raises(ValueError, match=key):
            load_norm_stats(tmp_path, "tag")
    (tmp_path / "norm_stats.json").write_text(
        json.dumps({"metadata_by_tag": {"tag": base}}),
    )
    action_stats, state_stats, meta = load_norm_stats(tmp_path, "tag")
    assert meta["setup_type"] == "tiny rig"
    assert action_stats.q01.shape == state_stats.q01.shape == (1,)


def test_quantile_normalize_roundtrip_and_eps() -> None:
    stats = QuantileStats(
        q01=torch.tensor([-2.0, 5.0, 3.0]),
        q99=torch.tensor([2.0, 15.0, 3.0]),  # last dim zero-width -> eps
    )
    raw = torch.tensor([[0.0, 10.0, 3.0], [-2.0, 5.0, 4.0], [2.0, 15.0, 2.0]])
    norm = normalize_q01q99(raw, stats)
    np.testing.assert_allclose(norm[:, 0].numpy(), [0.0, -1.0, 1.0], atol=1e-7)
    np.testing.assert_allclose(norm[:, 1].numpy(), [0.0, -1.0, 1.0], atol=1e-7)
    assert norm[1, 2] > 1e7  # zero-width range explodes through eps, as theirs does
    back = unnormalize_q01q99(
        norm[:, :2],
        QuantileStats(q01=stats.q01[:2], q99=stats.q99[:2]),
    )
    np.testing.assert_allclose(back.numpy(), raw[:, :2].numpy(), atol=1e-6)
    # normalize_state clamps; unnormalize_action clamps FIRST.
    assert normalize_state(torch.tensor([100.0, 100.0, 3.0]), stats).max() == 1.0
    clamped = unnormalize_action(
        torch.tensor([5.0, -5.0]),
        QuantileStats(
            q01=stats.q01[:2],
            q99=stats.q99[:2],
        ),
    )
    np.testing.assert_allclose(
        clamped.numpy(),
        [stats.q99[0].item(), stats.q01[1].item()],
    )


def test_to_uint8_rgb_branches() -> None:
    chw = torch.full((3, 4, 5), 0.5)
    out = to_uint8_rgb(chw)
    assert out.shape == (4, 5, 3) and out.dtype == np.uint8 and out[0, 0, 0] == 127
    # Floats above 1 are clipped, not rescaled.
    assert to_uint8_rgb(np.full((4, 5, 3), 300.0))[0, 0, 0] == 255
    gray = to_uint8_rgb(np.zeros((4, 5), dtype=np.uint8))
    assert gray.shape == (4, 5, 3)
    rgba = to_uint8_rgb(np.zeros((4, 5, 4), dtype=np.uint8))
    assert rgba.shape == (4, 5, 3)
    with pytest.raises(ValueError, match="unsupported image shape"):
        to_uint8_rgb(np.zeros((2, 3, 4, 5, 6)))


def test_process_image_resize_contract() -> None:
    crops = process_image_resize(goldens.synthetic_frame(96, 128, seed=0))
    assert crops.grid == RESIZE_GRID
    assert tuple(crops.crops.shape) == (1, 729, 588)
    assert tuple(crops.pooled_idx.shape) == (196, 4)
    # Every source patch pooled exactly once; 27 is odd so one padded
    # row/col of -1 members exists.
    members = crops.pooled_idx.numpy().reshape(-1)
    assert sorted(members[members >= 0]) == list(range(729))
    assert crops.crops.min() >= -1.0 and crops.crops.max() <= 1.0
    with pytest.raises(ValueError, match="uint8"):
        process_image_resize(np.zeros((8, 8, 3), dtype=np.float32))


@needs_checkpoint
def test_pack_sequence_budget_guard(
    stats_and_meta: StatsAndMeta,
    tokenizer: Molmo2TextTokenizer,
) -> None:
    _, state_stats, meta = stats_and_meta
    with pytest.raises(ValueError, match="exceeds max_sequence_length"):
        pack_action_example(
            images=[goldens.synthetic_frame(32, 32, seed=0)],
            state=torch.zeros(6),
            task="wave",
            tokenizer=tokenizer,
            state_stats=state_stats,
            setup_type=meta["setup_type"],
            control_mode=meta["control_mode"],
            max_sequence_length=16,
        )
