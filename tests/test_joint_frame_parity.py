"""The discrete-serving joint-frame seam + loop-vs-anchor prompt parity
(the R2 serving-parity fix's CPU oracle).

The R2 wave-0 kill (2026-08-19): ``sim.grpo_loop`` served every
checkpoint through the port-era v30→v21 shim, correct only for
unremapped v2.1-table releases — on a v3.0-frame bijou table it clamps
lift/elbow state bins at the table edge and inverts the chunk map (the
arm drives out of range; the boat is never touched). The fix routes
both sim discrete drivers through ``resolve_joint_frame`` — fingerprint
the checkpoint's state table (docs/so101-joint-conventions.md §4),
refuse mismatches loudly. This suite pins:

1. the fingerprint classifier on the REAL table shapes in play
   (v2.1 released, conversion-remapped released, bijou-trained v2);
2. resolve semantics — auto routing, ask-don't-guess on unclassifiable
   tables, loud refusal of explicit/classified mismatches;
3. the shim literals against ``JointFrameTransform`` (one source of
   truth for the map, exercised in both directions);
4. PROMPT PARITY: the loop stack's ``prompt_inputs`` packing against
   the BijouPolicy/Collator packing on the SAME observation — collated
   ``MolmoAct2Inputs`` tensors bit-equal, so under the identity frame
   the two serving paths feed the trunk identical bytes (they already
   share ``predict_ar`` and the family's one quantile table out the
   other side — the frame map was the ONLY unshared seam).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from bijou.checkpoint import read_metadata
from bijou.rollout import SO_MOTORS, observation_to_item
from bijou.rollout_safety import JointFrameTransform
from bijou.testing import write_tiny_molmoact2_release
from sim.grpo_loop import parse_args as loop_parse_args
from sim.rollout_sim import TASK
from sim.rollout_sim_parallel import (
    MOLMOACT2_OFFICIAL_OFFSETS_DEG,
    MOLMOACT2_OFFICIAL_SIGNS,
    classify_state_frame,
    resolve_joint_frame,
)

# Real table shapes (docs/so101-joint-conventions.md §4 + live
# checkpoint metadata, rounded): the classifier must place all three.
V21_STATE_Q01 = (-42.1, 45.2, 35.4, 4.9, -65.6, -0.3)
V21_STATE_Q99 = (48.6, 186.1, 173.6, 93.4, 43.5, 44.8)
# molmoact2-so101-released after the conversion-time remap — the
# flipped shoulder_lift lands as a DESCENDING pair.
REMAPPED_STATE_Q01 = (-41.9, 46.3, -51.6, 5.7, -63.4, 0.9)
REMAPPED_STATE_Q99 = (48.3, -95.3, 83.1, 91.8, 42.9, 44.1)
# fontaine_grasp_sft_joint_corrected/step_002000_v2 (bijou-trained).
V2_STATE_Q01 = (-36.9, -110.0, 14.4, 23.5, -157.2, 0.0)
V2_STATE_Q99 = (4.8, 12.2, 88.6, 81.0, 157.2, 42.7)


class _Stats:
    def __init__(self, q01: tuple[float, ...], q99: tuple[float, ...]) -> None:
        self.state_q01 = q01
        self.state_q99 = q99


def test_classify_state_frame_fingerprints() -> None:
    assert classify_state_frame(V21_STATE_Q01, V21_STATE_Q99) == "v21"
    assert classify_state_frame(REMAPPED_STATE_Q01, REMAPPED_STATE_Q99) == "v30"
    assert classify_state_frame(V2_STATE_Q01, V2_STATE_Q99) == "v30"
    # Quantile-less / short / mid-band tables: ask, don't guess.
    assert classify_state_frame(None, None) is None
    assert classify_state_frame((0.0, 1.0), (1.0, 2.0)) is None
    assert classify_state_frame((0.0,) * 6, (10.0,) * 6) is None
    # v2.1-band lift WITHOUT the elbow corroboration stays unclassified.
    assert classify_state_frame((0.0, 45.0, 0.0), (10.0, 186.0, 10.0)) is None


def test_resolve_joint_frame_auto_routes_by_fingerprint() -> None:
    resolved, shim = resolve_joint_frame("auto", _Stats(V2_STATE_Q01, V2_STATE_Q99))
    assert resolved == "rig"
    assert torch.equal(shim.scale, torch.ones(6))
    assert torch.equal(shim.offset, torch.zeros(6))
    resolved, shim = resolve_joint_frame(
        "auto",
        _Stats(REMAPPED_STATE_Q01, REMAPPED_STATE_Q99),
    )
    assert resolved == "rig"
    resolved, shim = resolve_joint_frame("auto", _Stats(V21_STATE_Q01, V21_STATE_Q99))
    assert resolved == "v30-to-v21"
    assert shim.scale.tolist() == list(MOLMOACT2_OFFICIAL_SIGNS)
    assert shim.offset.tolist() == list(MOLMOACT2_OFFICIAL_OFFSETS_DEG)


def test_resolve_joint_frame_refuses_unclassifiable_auto() -> None:
    with pytest.raises(SystemExit, match=r"ask, don't guess"):
        resolve_joint_frame("auto", _Stats((0.0,) * 6, (10.0,) * 6))


def test_resolve_joint_frame_refuses_classified_mismatch() -> None:
    # The R2 kill class: the shim on a v3.0-frame table.
    with pytest.raises(SystemExit, match="contradicts"):
        resolve_joint_frame("v30-to-v21", _Stats(V2_STATE_Q01, V2_STATE_Q99))
    # The double-remap class from the other side.
    with pytest.raises(SystemExit, match="contradicts"):
        resolve_joint_frame("rig", _Stats(V21_STATE_Q01, V21_STATE_Q99))
    with pytest.raises(SystemExit, match="one of"):
        resolve_joint_frame("v21-to-v30", _Stats(V2_STATE_Q01, V2_STATE_Q99))


def test_resolve_joint_frame_trusts_explicit_on_unclassifiable() -> None:
    resolved, shim = resolve_joint_frame("rig", _Stats((0.0,) * 6, (10.0,) * 6))
    assert resolved == "rig"
    assert torch.equal(shim.scale, torch.ones(6))
    resolved, shim = resolve_joint_frame(
        "v30-to-v21",
        _Stats((0.0,) * 6, (10.0,) * 6),
    )
    assert resolved == "v30-to-v21"
    assert shim.offset.tolist() == list(MOLMOACT2_OFFICIAL_OFFSETS_DEG)


def test_shim_literals_and_map_match_joint_frame_transform() -> None:
    """One source of truth: the sim-local literals equal
    JointFrameTransform.lerobot_v30_to_v21()'s, and the resolved
    AffineMap computes the SAME map in both directions (state in via
    state_to_model, chunks out via chunk_to_arm)."""
    frame = JointFrameTransform.lerobot_v30_to_v21()
    assert frame.signs == MOLMOACT2_OFFICIAL_SIGNS
    assert frame.offsets == MOLMOACT2_OFFICIAL_OFFSETS_DEG
    _, shim = resolve_joint_frame("auto", _Stats(V21_STATE_Q01, V21_STATE_Q99))
    state = torch.tensor([1.0, -22.5, 47.0, -3.0, 15.0, 30.0])
    assert shim.apply(state).tolist() == pytest.approx(
        frame.state_to_model(state.tolist()),
    )
    chunk = torch.tensor([[10.0, 160.0, 120.0, 40.0, -5.0, 20.0]])
    assert torch.allclose(shim.invert(chunk), frame.chunk_to_arm(chunk))
    # Round trip is exact (signs are ±1).
    assert torch.allclose(shim.invert(shim.apply(state)), state)


def test_loop_parse_args_joint_frame() -> None:
    base = ["--checkpoint", "ckpt", "--total-steps", "1"]
    assert loop_parse_args(base).joint_frame == "auto"
    assert loop_parse_args([*base, "--joint-frame", "rig"]).joint_frame == "rig"
    with pytest.raises(SystemExit):
        loop_parse_args([*base, "--joint-frame", "official"])


# --- prompt parity: the loop stack vs the BijouPolicy collator --------


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("joint-frame-parity") / "tiny"
    write_tiny_molmoact2_release(root)
    return root / "checkpoint_vla"


def _observation(state: torch.Tensor) -> dict[str, Any]:
    generator = np.random.default_rng(11)
    frames = {
        "top": generator.integers(0, 256, size=(48, 64, 3), dtype=np.uint8),
        "wrist": generator.integers(0, 256, size=(48, 64, 3), dtype=np.uint8),
    }
    return {
        **{f"{m}.pos": float(state[i]) for i, m in enumerate(SO_MOTORS)},
        **frames,
    }


def test_prompt_parity_stack_vs_bijou_collator(tiny_checkpoint: Path) -> None:
    """The two serving paths pack IDENTICAL trunk bytes: the loop
    stack's prompt_inputs (raw frames + identity-frame state) against
    the BijouPolicy collator on the equivalent sim item — every
    collated MolmoAct2Inputs tensor bit-equal. Beyond this seam the
    paths share predict_ar and the family's one quantile table, so
    prompt parity + the frame map IS serving parity."""
    from bijou.eval.policies import BijouPolicy
    from bijou.grpo_replay import MolmoAct2DiscreteStack

    stack = MolmoAct2DiscreteStack.load(
        tiny_checkpoint,
        device="cpu",
        dtype=torch.float32,
        fast_tokenizer=str(
            Path(__file__).parent / "fixtures" / "molmoact2_fast_tokenizer",
        ),
    )
    policy = BijouPolicy(
        tiny_checkpoint,
        device=torch.device("cpu"),
        seed=0,
    )
    metadata = read_metadata(tiny_checkpoint)
    state = torch.tensor([0.5, -0.25, 1.75, 10.0, -4.0, 2.0])
    observation = _observation(state)

    item = observation_to_item(
        observation,
        TASK,
        stats=metadata.stats,
        chunk_size=policy.info.chunk_size,
        camera_kinds={"top": "top", "wrist": "wrist"},
    )
    item["repo_id"] = "sim/eval100"
    item["episode_index"] = 0
    item["frame_index"] = 0
    batch = policy.collator([item])

    inputs, normalized = stack.prompt_inputs(
        [observation["top"], observation["wrist"]],
        TASK,
        state,
    )

    stack_tensors = inputs.tensors()
    anchor_tensors = batch.encoder_inputs.tensors()
    assert stack_tensors.keys() == anchor_tensors.keys()
    for name, tensor in stack_tensors.items():
        assert torch.equal(tensor, anchor_tensors[name]), (
            f"collated {name} diverges between the loop stack and the "
            "BijouPolicy collator — the two serving paths feed the trunk "
            "different bytes"
        )
    # The state the stack normalized is the merged-table clamp map —
    # the exact scheme the collator binned (equality of input_ids above
    # pins the bins; this pins the pre-binned vector).
    table_q01 = torch.tensor(metadata.stats.state_q01)
    table_q99 = torch.tensor(metadata.stats.state_q99)
    expected = (2.0 * (state - table_q01) / (table_q99 - table_q01) - 1.0).clamp(
        -1.0,
        1.0,
    )
    assert torch.allclose(normalized, expected, atol=1e-6)
