"""Phase-2 acceptance gate: first-class MolmoAct2ARDecoder vs the
reference discrete decodes (docs/molmoact2-retirement.md, phase-2
gates; fixture: tests/fixtures/molmoact2_discrete/decode_anchors.npz,
generated from fontaine's predict_action_discrete at his 3cacc531-era
worktree on this box's H100, bf16).

PRE-REGISTERED EXPECTATIONS (a failure is a re-baseline decision,
never a tolerance bump):
- masked-mode token_ids, bins and executed actions BYTE-EQUAL on all
  6 rows (discrete surfaces are byte-portable; the executed fp32
  chunk pins the whole pipeline: prompt assembly → prefill → masked
  decode → codec decode → clamp+unnormalize op order);
- per-bin-step chosen logprobs within 1e-5 of the fixture (fp32
  softmax drift class; recomputed from OUR capture with the
  generator's exact ops);
- MEASURED (not gated): the teacher-forced re-forward of the decoded
  bins vs the capture logprobs — a CROSS-SURFACE comparison (one wide
  suffix forward vs L single-token forwards), which under a bf16
  trunk sits at the batch-shape reduction-order floor (measured
  2026-08-14: ≤5.6e-2 worst-step across the 6 rows; fixture-logprob
  agreement at 2.4e-7 simultaneously — the decode path itself is
  exact). The registered 1e-5 replay bound is a SAME-surface bound
  (teacher-forced vs teacher-forced, phase 4's frozen-wave gate) and
  does not apply here; ``--trunk-dtype float32`` is the mechanism
  diagnostic (deltas collapse ⇒ shape numerics, not semantics).
If token IDS mismatch: suspect the sdpa pin (the reference ran the
full dispatcher) and the trunk mount dtype FIRST.

Run on the box (GPU, ~2 min):
  PYTHONPATH=. uv run python probes/probe_molmoact2_ar_parity.py \\
    --checkpoint converted/molmoact2_so100_101_release \\
    --data ~/datasets/mcobzarenco/so101_pick_place_clean \\
           ~/datasets/mcobzarenco/so101_pick_place_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bijou.data import EpisodeSplit, select_datasets
from bijou.fast.molmoact2 import QuantileStats, normalize_state
from bijou.loading import (
    MOLMOACT2_FAST_TOKENIZER_REF,
    MolmoAct2PromptConfig,
    MolmoFlowDecoderConfig,
    build_molmoact2_ar_decoder,
    checkpoint_sections,
    molmo_flow_state_table,
    molmoact2_ar_config_from_flow_section,
    read_checkpoint_info,
    resolve_checkpoint_dir,
)
from bijou.model import BijouModel
from bijou.modelling.encoders.molmoact2 import MolmoAct2Encoder
from bijou.modelling.interface import (
    ActionCaptureStep,
    CameraFrame,
    CollatedBatch,
    NormStats,
    PromptInputs,
)
from bijou.modelling.molmo2.loading import load_config as load_molmo2_config
from bijou.modelling.molmo2.model import load_model as load_molmo2_model

FIXTURE = Path("tests/fixtures/molmoact2_discrete/decode_anchors.npz")
NUM_ROWS = 6


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--fast-tokenizer", default=MOLMOACT2_FAST_TOKENIZER_REF)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument(
        "--trunk-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="bfloat16 = the fixture generator's mount (byte gates apply); "
        "float32 = the replay-delta mechanism diagnostic ONLY (the "
        "fixture is bf16-specific, byte gates are skipped): if the "
        "teacher-forced-vs-incremental logprob delta collapses in fp32, "
        "the bf16 batch-shape reduction-order class (sharding.py's "
        "caveat) is confirmed as the mechanism",
    )
    args = parser.parse_args()
    byte_gates = args.trunk_dtype == "bfloat16"

    fixture = np.load(args.fixture, allow_pickle=False)
    checkpoint = Path(args.checkpoint)
    info = read_checkpoint_info(checkpoint)
    sections = checkpoint_sections(
        json.loads((checkpoint / "bijou_config.json").read_text()),
    )
    prompt = sections.prompt
    flow = sections.decoder
    assert isinstance(prompt, MolmoAct2PromptConfig), type(prompt)
    assert isinstance(flow, MolmoFlowDecoderConfig), type(flow)

    # --- the first-class stack (the release AR read) ---
    trunk_dir = resolve_checkpoint_dir(info.backbone)
    config = molmoact2_ar_config_from_flow_section(
        flow,
        prompt,
        info.backbone,
        fast_tokenizer=args.fast_tokenizer,
    )
    decoder = build_molmoact2_ar_decoder(
        config,
        prompt,
        load_molmo2_config(trunk_dir).text,
        info.backbone,
    )
    backbone = load_molmo2_model(
        trunk_dir,
        device="cuda",
        dtype=torch.bfloat16 if args.trunk_dtype == "bfloat16" else torch.float32,
    )
    encoder = MolmoAct2Encoder(
        info.backbone,
        setup_type=prompt.setup_type,
        control_mode=prompt.control_mode,
        num_state_tokens=prompt.num_state_tokens,
        action_mode=prompt.action_mode,
        narration=prompt.narration,
    )
    model = BijouModel(backbone=backbone, encoder=encoder, decoder=decoder)
    model.eval()
    print(
        f"stack: block_base={config.block_base} vocab_total="
        f"{config.vocab_total} chunk={config.chunk_size}x{config.action_dim}",
    )

    state_q01, state_q99 = molmo_flow_state_table(info.normalization)
    state_stats = QuantileStats(q01=state_q01, q99=state_q99)
    assert info.normalization.action_q01 is not None
    assert info.normalization.action_q99 is not None
    action_q01 = torch.tensor(info.normalization.action_q01, dtype=torch.float32)
    action_q99 = torch.tensor(info.normalization.action_q99, dtype=torch.float32)

    # --- the fixture's exact row selection ---
    selection = select_datasets(
        tuple(Path(d).expanduser() for d in args.data),
        (),
        config.chunk_size,
        episode_split=EpisodeSplit.HOLDOUT,
        holdout_fraction=0.1,
        split_seed=0,
    )
    dataset = selection.concat()
    rows: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for index in range(len(dataset)):
        item = dataset[index]
        key = (str(item["repo_id"]), int(item["episode_index"]))
        if int(item["frame_index"]) != 0 or key in seen:
            continue
        seen.add(key)
        rows.append(item)
        if len(rows) == NUM_ROWS:
            break
    assert len(rows) == NUM_ROWS, f"only {len(rows)} first-frames in the holdout"
    fixture_rows = [
        (str(repo), int(episode))
        for repo, episode in zip(
            fixture["repos"].tolist(),
            fixture["episodes"].tolist(),
            strict=True,
        )
    ]
    ours_rows = [(str(r["repo_id"]), int(r["episode_index"])) for r in rows]
    assert fixture_rows == ours_rows, (
        f"row identity drift: fixture {fixture_rows} vs selected {ours_rows}"
    )

    collator = encoder.inputs_collator()
    base = config.block_base
    failures: list[str] = []
    max_logprob_delta = 0.0
    max_replay_delta = 0.0

    for row_idx, item in enumerate(rows):
        cameras = tuple(
            CameraFrame(
                name=key.removeprefix("observation.images."),
                kind="unknown",
                image=item[key],
            )
            for key in sorted(k for k in item if k.startswith("observation.images."))
        )
        normalized_state = normalize_state(item["observation.state"], state_stats)
        inputs = collator(
            [
                PromptInputs(
                    instruction=str(item["task"]),
                    cameras=cameras,
                    condition_text="",
                    state=normalized_state,
                ),
            ],
        ).to("cuda")
        stats = NormStats(
            mean=torch.zeros(1, config.action_dim),
            std=torch.ones(1, config.action_dim),
            q01=action_q01[None],
            q99=action_q99[None],
        )
        batch = CollatedBatch(
            encoder_inputs=inputs,
            state=normalized_state[None].to("cuda"),
            actions=torch.zeros(1, config.chunk_size, config.action_dim),
            action_is_pad=torch.zeros(1, config.chunk_size, dtype=torch.bool),
            action_stats=stats,
            state_stats=stats,
            action_tokens=None,
            suffix_tokens=None,
            suffix_is_aux=None,
        )
        memory = model.encode(inputs, with_grad=False)
        capture: list[ActionCaptureStep] = []
        prediction = model.ar_predict_greedy(memory, batch, action_capture=capture)

        bins = [int(step.chosen[0]) - base for step in capture if bool(step.active[0])]
        token_ids = np.array([[base - 2, *(base + b for b in bins), base - 1]])
        logprobs: list[float] = []
        for step in capture:
            rebased = step.chosen - base
            logits = step.block_logits.float().masked_fill(
                ~step.allowed,
                float("-inf"),
            )
            logprobs.append(
                logits.log_softmax(-1).gather(-1, rebased[..., None]).item(),
            )

        key = f"release_masked_{row_idx}"
        checks: dict[str, bool] = {}
        delta = 0.0
        if byte_gates:
            checks["token_ids"] = bool(
                np.array_equal(fixture[f"{key}_token_ids"], token_ids),
            )
            checks["bins"] = bool(
                np.array_equal(fixture[f"{key}_bins"], np.array(bins, dtype=np.int64)),
            )
            checks["actions"] = bool(
                np.array_equal(
                    fixture[f"{key}_actions"],
                    prediction.actions.cpu().float().numpy(),
                ),
            )
            delta = float(
                np.abs(fixture[f"{key}_logprobs"] - np.array(logprobs)).max(),
            )
            max_logprob_delta = max(max_logprob_delta, delta)
            checks["logprobs<=1e-5"] = delta <= 1e-5

        # Ratio contract: teacher-forced re-forward of the decoded bins
        # against a FRESH memory reproduces the capture logprobs
        # (unchanged policy ⇒ ratio 1). Same masked-softmax read.
        replay_memory = model.encode(inputs, with_grad=False)
        forced = model.ar_teacher_forced_block_logits(replay_memory, [bins])[0]
        assert forced is not None
        lengths = decoder.symbol_lengths.cpu()
        remaining = config.chunk_size * config.action_dim
        replay_logprobs: list[float] = []
        for position, bin_id in enumerate(bins):
            legal = (lengths > 0) & (lengths <= remaining)
            row_logits = forced[position].cpu().masked_fill(~legal, float("-inf"))
            replay_logprobs.append(
                float(row_logits.log_softmax(-1)[bin_id]),
            )
            remaining -= int(lengths[bin_id])
        replay_delta = float(
            np.abs(np.array(replay_logprobs) - np.array(logprobs)).max(),
        )
        max_replay_delta = max(max_replay_delta, replay_delta)
        # MEASURED, not asserted: teacher-forced (one wide suffix
        # forward) vs incremental (L single-token forwards) is the
        # batch-shape reduction-order class — bf16 floor measured
        # 2026-08-14 at ≤5.6e-2 worst-step on the release trunk; fp32
        # collapses it (the --trunk-dtype diagnostic). Not a gate;
        # phase 4's frozen-wave replay registers its own bound.

        verdict = "PASS" if all(checks.values()) else "FAIL"
        print(
            f"row {row_idx} ({ours_rows[row_idx][0]} ep "
            f"{ours_rows[row_idx][1]}): {verdict} "
            f"{checks} logprob_delta={delta:.2e} replay_delta={replay_delta:.2e}",
            flush=True,
        )
        if verdict == "FAIL":
            failures.append(f"row {row_idx}: {checks}")

    print(
        f"max logprob delta {max_logprob_delta:.3e}; "
        f"max replay delta {max_replay_delta:.3e} "
        f"(teacher-forced vs incremental, {args.trunk_dtype} trunk — "
        "measured, not gated)",
    )
    if not byte_gates:
        print("DIAGNOSTIC MODE (fp32 trunk): no byte gates — read the deltas")
        return 0
    if failures:
        print("ACCEPTANCE: FAIL — a re-baseline decision, never a tolerance bump")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("ACCEPTANCE: PASS — masked ids/bins/actions byte-equal, logprobs in-bound")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
