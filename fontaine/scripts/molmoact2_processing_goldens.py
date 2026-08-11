"""Bank MolmoAct2 action-side processing goldens from THEIR stack.

Runs in the molmoact2 venv (`~/molmoact2/.venv/bin/python`), NOT ours —
it drives the real lerobot policy pipeline
(``make_molmoact2_pre_post_processors``: masked q01/q99 normalizer ->
clamp -> ``MolmoAct2PackInputsProcessorStep`` over the shipped HF
AutoProcessor) on deterministic inputs and freezes every output;
``tests/test_molmoact2_processing.py`` then reproduces them from
``bijou.molmoact2.processing`` in our env (the
``bank_processor_goldens.py`` precedent: reference semantics cross the
env boundary as fixtures, never as shared code). The test imports THIS
module (stdlib+numpy+torch at module level, lerobot only inside
``main``) to regenerate the identical inputs — images are synthesized,
not banked; pixel outputs are stored as strided samples + per-crop
stats like the molmo2 bank.

    ~/molmoact2/.venv/bin/python fontaine/scripts/molmoact2_processing_goldens.py \
        [--hf-dir ~/checkpoints/molmoact2-so101-rig-r1-step2000-hf]

Writes ``tests/fixtures/molmoact2_processing/<case>.npz`` (input side)
and ``action_<case>.npz`` (output side).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

FIXTURE_DIR = (
    Path(__file__).resolve().parents[2] / "tests/fixtures/molmoact2_processing"
)
DEFAULT_HF_DIR = Path("~/checkpoints/molmoact2-so101-rig-r1-step2000-hf").expanduser()
NORM_TAG = "so100_so101_molmoact2"
STATE_DIM = 6

# Stride over the [crops, patches, patch_dim] pixel stack — dense enough
# that any resize/normalize drift shows, small enough to commit.
PIXEL_STRIDES = (1, 13, 11)

# (name, [image specs], state row, task). Image spec = (H, W, kind) with
# kind: 'chw_float' = torch float32 CHW in [0,1] (the LeRobot decode
# convention their eval feeds), 'hwc_uint8' = torch uint8 HWC,
# 'chw_float255' = float CHW in [0,255] (exercises the nanmax>1 branch
# of their uint8 coercion: clip WITHOUT the *255 rescale).
CASES = [
    (
        "base_2cam",
        [(480, 640, "chw_float"), (480, 640, "chw_float")],
        [10.0, -40.0, 60.0, 30.0, 50.0, 12.0],
        "pick up the red block and place it in the box",
    ),
    (
        "one_cam_prefix",
        [(640, 480, "chw_float")],
        [0.0, -90.0, 20.0, 80.0, -100.0, 2.0],
        "Task: Pick up the cube.",
    ),
    (
        "mixed_sizes_quoted",
        [(480, 640, "chw_float"), (240, 320, "chw_float")],
        [-40.0, 50.0, 90.0, -90.0, 160.0, 40.0],
        "'grab the bottle'",
    ),
    (
        "uint8_multisentence",
        [(480, 640, "hwc_uint8"), (480, 640, "hwc_uint8")],
        [4.0, -33.0, 55.0, 30.0, 53.0, 10.0],
        "Pick the block. Place it left!",
    ),
    (
        "small_image_prefix2",
        [(96, 96, "chw_float")],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "The task is to wipe the table",
    ),
    (
        "native_size_upper",
        [(378, 378, "chw_float"), (378, 378, "chw_float")],
        [-64.0, 64.0, 98.0, 102.0, 174.0, 43.0],
        "PLACE THE FORK",
    ),
    (
        "state_clamp_far",
        [(480, 640, "chw_float")],
        [500.0, -500.0, 1e6, -1e6, 0.0, 100.0],
        "“fold the towel”",
    ),
    (
        "state_nonfinite",
        [(480, 640, "chw_float")],
        [float("nan"), float("inf"), float("-inf"), 0.0, 1.0, -1.0],
        "goal: stack cups; carefully",
    ),
    (
        "float255_spacing",
        [(480, 640, "chw_float255"), (240, 320, "chw_float255")],
        [12.0, -70.0, 33.0, 21.0, 90.0, 20.0],
        "instruction- sort  the   fruit",
    ),
]

# Output-side cases: normalized action chunks [1, T, D] through clamp ->
# q01/q99 unnormalize. Values are deterministic (no RNG — the parity
# test regenerates them with this same function).
ACTION_CASES = [
    ("act_inrange", 0.7),
    ("act_outofrange", 2.0),
    ("act_edges", 1.0),
]


def synthetic_frame(height: int, width: int, *, seed: int) -> np.ndarray:
    """Deterministic HWC uint8 RGB frame — integer color gradients with a
    filled circle and rectangle (numpy-only so both envs regenerate it
    bit-identically; smooth enough that fixtures compress)."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = (np.arange(width, dtype=np.uint32) * 255 // width)[None, :]
    frame[..., 1] = (np.arange(height, dtype=np.uint32) * 255 // height)[:, None]
    frame[..., 2] = (128 + 37 * seed) % 256
    yy, xx = np.mgrid[0:height, 0:width]
    cy, cx = height * (5 + seed) // 16, width * (6 + seed) // 16
    radius = min(height, width) * 3 // 16
    frame[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2] = (220, 40, 40)
    y0, x0 = height // 8, width * (2 + seed) // 32
    frame[y0 : y0 + height // 4, x0 : x0 + width // 8] = (40, 200, 80)
    return frame


def make_image(height: int, width: int, kind: str, *, seed: int) -> torch.Tensor:
    frame = synthetic_frame(height, width, seed=seed)
    if kind == "chw_float":
        return (
            torch.from_numpy(frame.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .contiguous()
        )
    if kind == "chw_float255":
        return torch.from_numpy(frame.astype(np.float32)).permute(2, 0, 1).contiguous()
    if kind == "hwc_uint8":
        return torch.from_numpy(frame)
    raise ValueError(kind)


def case_images(image_specs: list[tuple[int, int, str]]) -> list[torch.Tensor]:
    return [
        make_image(h, w, kind, seed=idx) for idx, (h, w, kind) in enumerate(image_specs)
    ]


def action_chunk(name: str, scale: float) -> np.ndarray:
    """Deterministic [1, 30, D] normalized action chunk in [-scale, scale]."""
    t = np.arange(30, dtype=np.float32)[:, None]
    d = np.arange(STATE_DIM, dtype=np.float32)[None, :]
    chunk = np.sin(t * 0.37 + d * 1.3 + float(len(name))) * scale
    chunk = chunk.astype(np.float32)[None]
    if name == "act_edges":
        chunk[0, 0] = 1.0
        chunk[0, 1] = -1.0
    return chunk


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", default=str(DEFAULT_HF_DIR))
    args = parser.parse_args()

    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
    from lerobot.policies.molmoact2.processor_molmoact2 import (
        make_molmoact2_pre_post_processors,
    )
    from lerobot.processor.converters import create_transition
    from lerobot.types import TransitionKey
    from lerobot.utils.constants import ACTION, OBS_STATE

    def build(num_images: int) -> tuple[Any, Any]:
        config = MolmoAct2Config(
            checkpoint_path=args.hf_dir,
            norm_tag=NORM_TAG,
            action_mode="continuous",
            normalize_gripper=True,
            device="cpu",
            image_keys=[f"observation.images.cam{i}" for i in range(num_images)],
            input_features={
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            },
            output_features={
                ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(STATE_DIM,)),
            },
        )
        return make_molmoact2_pre_post_processors(config)

    pipelines = {n: build(n) for n in {len(specs) for _, specs, _, _ in CASES}}
    postproc = next(iter(pipelines.values()))[1]

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, image_specs, state_row, task in CASES:
        steps = list(pipelines[len(image_specs)][0].steps)
        clamp_index = next(
            i
            for i, s in enumerate(steps)
            if type(s).__name__ == "MolmoAct2ClampNormalizedProcessorStep"
        )
        images = case_images(image_specs)
        state = torch.tensor(state_row, dtype=torch.float32)

        observation = {OBS_STATE: state}
        for idx, image in enumerate(images):
            observation[f"observation.images.cam{idx}"] = image
        transition = create_transition(
            observation=observation,
            complementary_data={"task": task},
        )
        state_norm = None
        for i, step in enumerate(steps):
            transition = step(transition)
            if i == clamp_index:
                state_norm = (
                    transition[TransitionKey.OBSERVATION][OBS_STATE]
                    .detach()
                    .cpu()
                    .numpy()
                    .copy()
                )
        packed = transition[TransitionKey.COMPLEMENTARY_DATA]
        pixels = packed["pixel_values"].detach().cpu().numpy()

        np.savez_compressed(
            FIXTURE_DIR / f"{name}.npz",
            state_raw=state.numpy(),
            state_norm=state_norm,
            input_ids=np.asarray(
                packed["input_ids"].detach().cpu().numpy(),
                dtype=np.int64,
            ),
            attention_mask=np.asarray(
                packed["attention_mask"].detach().cpu().numpy(),
                dtype=np.int64,
            ),
            image_token_pooling=np.asarray(
                packed["image_token_pooling"].detach().cpu().numpy(),
                dtype=np.int64,
            ),
            image_grids=np.asarray(
                packed["image_grids"].detach().cpu().numpy(),
                dtype=np.int64,
            ),
            image_num_crops=np.asarray(
                packed["image_num_crops"].detach().cpu().numpy(),
                dtype=np.int64,
            ),
            pixel_samples=pixels[
                :: PIXEL_STRIDES[0],
                :: PIXEL_STRIDES[1],
                :: PIXEL_STRIDES[2],
            ].astype(np.float32),
            pixel_shape=np.asarray(pixels.shape, dtype=np.int64),
            pixel_mean=pixels.mean(axis=(1, 2)).astype(np.float64),
            pixel_std=pixels.std(axis=(1, 2)).astype(np.float64),
        )
        print(f"{name}: ids {packed['input_ids'].shape} crops {pixels.shape}")

    for name, scale in ACTION_CASES:
        chunk = action_chunk(name, scale)
        out = postproc(torch.from_numpy(chunk.copy()))
        np.savez_compressed(
            FIXTURE_DIR / f"action_{name}.npz",
            action_norm=chunk,
            action_out=out.detach().cpu().numpy(),
        )
        print(f"action_{name}: {chunk.shape} -> {tuple(out.shape)}")

    print(
        f"WROTE {len(CASES)} input + {len(ACTION_CASES)} action goldens to {FIXTURE_DIR}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
