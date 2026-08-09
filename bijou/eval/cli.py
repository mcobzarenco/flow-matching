"""Open-loop evaluation CLI (invoked via ``python -m bijou.eval``).

Samples K frames (without replacement, seeded; default = every frame of
the selection) from a dataset selection and scores every requested policy
on the SAME frames: the trivial state-copy baseline always, a bijou
checkpoint when ``--checkpoint`` is given, and a SmolVLA policy when
``--smolvla`` is given. Reports pad-masked chunk metrics in raw action
units plus paired per-frame comparisons against the baseline.

Ground truth is the recorded action chunk at each frame — this is offline /
open-loop evaluation (no robot, no simulator). For held-out scoring, point
``--data`` at datasets the checkpoint was not trained on.

Usage::

    uv run python -m bijou.eval \
        --data ~/datasets/mcobzarenco/so101_pick_place_clean \
        --checkpoint outputs/train/<run>/step_040000 \
        --smolvla lerobot/smolvla_base \
        --num-samples 256 --device cuda --output-json eval.json

Multi-GPU: launch under torchrun and the sampled frames are sharded
round-robin across ranks (rank 0 aggregates, prints and writes exactly
the single-process outputs — see ``sharding.py`` for the determinism
contract)::

    uv run torchrun --standalone --nproc-per-node=4 -m bijou.eval ...
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..aux_text import (
    EVENT_NONE,
    AuxField,
    aux_label_text,
    label_values,
    parse_visibility,
)
from ..data import EpisodeSplit, select_datasets
from ..model import SamplingMethod
from .metrics import (
    DatasetSlice,
    FrameScore,
    PairedComparison,
    PolicySummary,
    compare_paired,
    format_table,
    score_frame,
    slice_by_dataset,
    summarize,
)
from .policies import (
    NOISE_KEYS,
    BijouPolicy,
    ChunkPolicy,
    MaskedContrastSubgoalPolicy,
    NarratedBijouPolicy,
    NormalizedStateCopyPolicy,
    SelectedSubgoalPolicy,
    SelfSubgoalPass1Policy,
    SelfSubgoalPolicy,
    StateCopyPolicy,
)
from .subgoal_scoring import (
    eligible_indices,
    likelihood_pick,
    mean_chosen_logprob,
    medoid_pick,
    self_certainty,
)
from .subgoal_swap import SubgoalSwapMap, build_swap_map

# Q2 buckets, the train-side convention (train.OUTCOME_BUCKETS): frames
# sliced by their episode's TRUE outcome label; unlabeled = no
# requestable outcome (UNCLEAR completion or unjudged dataset).
OUTCOME_BUCKETS = ("success", "partial", "failure", "unlabeled")
from .plan import (
    SamplePlan,
    build_plan,
    episode_tables,
    resolve_plan,
    validate_plan,
)
from .report import THEMES, ReportSample, ReportTable, render_report
from .sharding import ShardResults, merge_shards
from .smolvla import SmolVLAEvalPolicy

# #6 rung (c) npz key namespace — the mcselect_results.py contract
# (mcselect:kl / mcselect:cand_pred / mcselect:pred_masked).
MCSELECT_PREFIX = "mcselect:"


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Schema of the --output-json payload: what was evaluated (data,
    split, sampling, policies) plus the per-policy summaries and paired
    comparisons. The JSON contract other tooling parses — extend, don't
    reshape."""

    data: list[str]
    episodes: str
    holdout_episodes: float
    split_seed: int
    fps: list[float] | None
    camera_counts: list[int] | None
    num_samples: int
    seed: int
    checkpoint: str | None
    smolvla: str | None
    # Scoring semantics — everything else that shapes the numbers.
    # Exact reproduction additionally needs batch_size and world_size:
    # batch composition perturbs bf16 decodes (see sharding.py), and a
    # condition_override run is a counterfactual, not a deployment read.
    exclude: list[str]
    aux_prompt_hash: str | None
    sample_steps: int
    sample_method: str
    sample_draws: int
    # AR sampled-draws instrument: the action-block sampling
    # temperature (None = greedy — every AR eval before 2026-08-06).
    # With sample_draws > 1 the bijou row is the mean-of-samples read,
    # the AR mirror of flow noise-draw ensembling.
    ar_temperature: float | None
    # Flow target-time conditioning s: "t" = standard s=t forwards;
    # "zero" = the SnapFlow one-step shortcut field (φ_s checkpoints
    # only) — with euler/1 this is the 1-NFE read.
    target_time: str
    noise_key: str
    # Golden-ticket mode (#1 screen): the tickets file and its sha256
    # (None = keyed noise). A ticket read must never pass as a keyed
    # read — consumers check these before pairing against keyed npzs.
    noise_tickets: str | None
    tickets_sha256: str | None
    # Noise-ladder rung 2 (#1): per-dataset ticket routing map and the
    # sha of its canonical form (None = unrouted). A routed (_ticketmap)
    # read must never pool as a single-ticket (_ticket) read — the
    # stage-2 scorer checks this sha against the pre-registered map.
    noise_ticket_map: str | None
    ticket_map_sha256: str | None
    # State-reliance probe: bijou policy fed the dataset state mean
    # (zero-information soft state token) — a diagnostic, never a
    # deployment read; the policy name carries _state-masked too.
    mask_state: bool
    # #6 rung (a)/(b) subgoal-conditioning probes: None = every eval
    # before 2026-08-07; "oracle" = per-frame TRUE-label [subgoal|…]
    # conditioning; "self" = the two-pass self-subgoal arms (policy
    # names carry _oraclesubgoal / _narrsubgoal / _selfsubgoal);
    # "draws" = rung (b) candidate selection — pass 1 additionally
    # decodes sampled subgoal candidates, pass 2 runs BOTH selection
    # arms (_bonsubgoal / _ceilsubgoal) on the same rows.
    subgoal_mode: str | None
    # #6 subgoal-swap content read (pre-reg 2026-08-09): oracle-path
    # derangement seed (None = every eval before 2026-08-09; the policy
    # name carries _swapsubgoal) and the identity-check flag (donor =
    # self on every episode — the launcher's byte-reproduction oracle,
    # name _swapidentity, NEVER a read).
    subgoal_swap_seed: int | None
    subgoal_swap_identity: bool
    # Rung (b) candidate set: sampled draws beyond the greedy candidate
    # and their sampling temperature (None outside draws mode; 0 draws
    # = the bit-exactness preflight limit).
    subgoal_draws: int | None
    subgoal_temperature: float | None
    # Rung (b') clean-list rule (None = unfiltered rung (b)): "clean"
    # means every scorer operated on the eligible (non-truncated)
    # candidate list, greedy fallback on all-truncated rows — the
    # report records the filter alongside the scorer id (pre-reg).
    subgoal_candidate_filter: str | None
    # Oracle-(i)/(ii) live check: pass 2 ran with every hint forced
    # EMPTY (must reproduce the baseline decode; the policy names carry
    # _emptyhint) — never an arm read.
    selfsubgoal_force_empty: bool
    # #6 rung (c) mcselect mode (None/absent outside it): the reference
    # tempering τ, the injected candidates file, and its sha256 — the
    # read script refuses a run whose sha does not match the banked
    # rung-(b') width (every comparator would be void).
    mcselect_tau: float | None
    mcselect_candidates_file: str | None
    candidates_sha256: str | None
    generate: list[str] | None
    condition_override: list[str]
    batch_size: int
    world_size: int
    summaries: list[PolicySummary]
    paired: list[PairedComparison]
    motor_names: list[str]
    # Q2: per-policy chunk MAE sliced by the episode's TRUE outcome
    # label ({} when the eval carries no labels).
    outcome_slices: dict[str, dict[str, float]]
    # Q3: mean |Δ prediction| on labeled non-success frames when outcome
    # is counterfactually forced to "success" (None = not measured:
    # no condition-trained checkpoint, or a manual --condition-override).
    condition_sensitivity: float | None
    condition_sensitivity_frames: int
    # Narrated-pass aux metrics vs the (weak) judge labels over ALL
    # labeled sampled frames (None = no narrated pass). Event accuracy
    # is presence detection ("none" vs any event); visible accuracy is
    # exact set-match of the parsed positional slots.
    holding_accuracy: float | None
    holding_frames: int
    progress_mae: float | None
    progress_frames: int
    event_accuracy: float | None
    event_frames: int
    visible_accuracy: float | None
    visible_frames: int
    # Sample-plan provenance (None/0 when sampling was uniform): the
    # headline summaries cover core_frames; labeled_frames rode along
    # for the aux metrics only.
    sample_plan: str | None
    plan_seed: int | None
    core_frames: int
    labeled_frames: int
    # Per-dataset breakdown, ordered by frame count descending (see
    # metrics.slice_by_dataset for the small-n caveat).
    per_dataset: dict[str, DatasetSlice]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "data": self.data,
            "episodes": self.episodes,
            "holdout_episodes": self.holdout_episodes,
            "split_seed": self.split_seed,
            "fps": self.fps,
            "camera_counts": self.camera_counts,
            "num_samples": self.num_samples,
            "seed": self.seed,
            "checkpoint": self.checkpoint,
            "smolvla": self.smolvla,
            "exclude": self.exclude,
            "aux_prompt_hash": self.aux_prompt_hash,
            "sample_steps": self.sample_steps,
            "sample_method": self.sample_method,
            "sample_draws": self.sample_draws,
            "ar_temperature": self.ar_temperature,
            "target_time": self.target_time,
            "noise_key": self.noise_key,
            "noise_tickets": self.noise_tickets,
            "tickets_sha256": self.tickets_sha256,
            "noise_ticket_map": self.noise_ticket_map,
            "ticket_map_sha256": self.ticket_map_sha256,
            "mask_state": self.mask_state,
            "subgoal_mode": self.subgoal_mode,
            "subgoal_swap_seed": self.subgoal_swap_seed,
            "subgoal_swap_identity": self.subgoal_swap_identity,
            "subgoal_draws": self.subgoal_draws,
            "subgoal_temperature": self.subgoal_temperature,
            "subgoal_candidate_filter": self.subgoal_candidate_filter,
            "selfsubgoal_force_empty": self.selfsubgoal_force_empty,
            "mcselect_tau": self.mcselect_tau,
            "mcselect_candidates_file": self.mcselect_candidates_file,
            "candidates_sha256": self.candidates_sha256,
            "generate": self.generate,
            "condition_override": self.condition_override,
            "batch_size": self.batch_size,
            "world_size": self.world_size,
            "summaries": [s.to_dict() for s in self.summaries],
            "paired": [c.to_dict() for c in self.paired],
            "motor_names": self.motor_names,
            "outcome_slices": self.outcome_slices,
            "condition_sensitivity": self.condition_sensitivity,
            "condition_sensitivity_frames": self.condition_sensitivity_frames,
            "holding_accuracy": self.holding_accuracy,
            "holding_frames": self.holding_frames,
            "progress_mae": self.progress_mae,
            "progress_frames": self.progress_frames,
            "event_accuracy": self.event_accuracy,
            "event_frames": self.event_frames,
            "visible_accuracy": self.visible_accuracy,
            "visible_frames": self.visible_frames,
            "sample_plan": self.sample_plan,
            "plan_seed": self.plan_seed,
            "core_frames": self.core_frames,
            "labeled_frames": self.labeled_frames,
            "per_dataset": {
                repo_id: dataset_slice.to_dict()
                for repo_id, dataset_slice in self.per_dataset.items()
            },
        }


def identity_collate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep raw items as a list (module-level: spawn workers pickle this)."""
    return items


def eval_worker_init(_worker_id: int) -> None:
    """Single-threaded + file_system tensor sharing, IN the worker: tensors
    are serialized worker-side, so the sharing strategy must be set there —
    the main-process call alone still produced fd-based storages."""
    torch.set_num_threads(1)
    torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="dataset dirs and/or collection roots (same semantics as "
        "bijou.train --train-data)",
    )
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument(
        "--episodes",
        choices=[s.value for s in EpisodeSplit],
        default=EpisodeSplit.ALL.value,
        help="which side of the per-dataset episode holdout to evaluate; "
        "'holdout' scores exactly the episodes a training run with the "
        "same --holdout-episodes and --split-seed never saw",
    )
    parser.add_argument(
        "--holdout-episodes",
        type=float,
        default=0.0,
        help="holdout fraction — must match the training run's "
        "--holdout-episodes for the split to line up",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="split seed — must match the training run's --split-seed",
    )
    parser.add_argument(
        "--fps",
        type=float,
        nargs="+",
        default=None,
        help="keep only datasets recorded at one of these frame rates; "
        "default keeps every fps. Must match the training run's --fps for "
        "scores to be comparable (any filter changes the concatenated "
        "frame indexing and thus the sampled eval frames)",
    )
    parser.add_argument(
        "--camera-counts",
        type=int,
        nargs="+",
        default=None,
        help="keep only datasets with one of these camera counts; must "
        "match the training run's --camera-counts (same comparability "
        "caveat as --fps)",
    )
    parser.add_argument(
        "--aux-prompt-hash",
        default=None,
        help="pin: datasets whose judge-annotation stamp carries any other "
        "prompt hash render as unjudged, loudly — pass the training run's "
        "pin so eval and training agree on the prompt distribution "
        "(without it a pinned run's eval renders full tags for datasets "
        "the run trained as unjudged)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="bijou checkpoint directory; omit to evaluate the baseline only",
    )
    parser.add_argument(
        "--smolvla",
        default=None,
        help="optional SmolVLA policy id/path (e.g. lerobot/smolvla_base)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="frames sampled without replacement across the selection; "
        "omit to score every frame of the selection (mind the size: "
        "large holdouts are hours even multi-GPU — the printed "
        "'eval data' line has the frame count). Mutually exclusive "
        "with --sample-plan",
    )
    parser.add_argument(
        "--sample-plan",
        type=Path,
        default=None,
        help="stratified panel artifact (JSON): per-episode core frames "
        "drive the headline metrics, oversampled judge-labeled frames "
        "feed the aux metrics. Loads the file when it exists (a frozen "
        "panel — cross-checkpoint comparisons become paired); builds "
        "and writes it deterministically when it does not",
    )
    parser.add_argument(
        "--plan-seed",
        type=int,
        default=0,
        help="seed for building a NEW sample plan (ignored when loading; "
        "deliberately distinct from --seed, which keeps governing "
        "policy noise)",
    )
    parser.add_argument(
        "--frames-per-episode",
        type=int,
        default=4,
        help="core-panel frames drawn uniformly per episode when "
        "building a new sample plan (ignored when loading)",
    )
    parser.add_argument(
        "--labeled-per-episode",
        type=int,
        default=2,
        help="additional judge-labeled frames per episode when building "
        "a new sample plan (ignored when loading); scored but excluded "
        "from headline aggregation — they feed the aux metrics",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--noise-key",
        choices=list(NOISE_KEYS),
        default="stable",
        help="flow-noise derivation: 'stable' (frame-identity triple — "
        "corpus-composition-invariant; the quoted keying for all new "
        "flow numbers since the 2026-08-06 anchor re-bank) or 'index' "
        "(legacy, corpus-relative concat index — comparable only at "
        "frozen corpus composition; pass explicitly to reproduce "
        "historical index-keyed reports). The keyings are DIFFERENT "
        "draws, so numbers are not comparable across them",
    )
    parser.add_argument(
        "--noise-tickets",
        type=Path,
        default=None,
        help="golden-ticket noise mode (#1 screen): path to a committed "
        "tickets npz ([count, chunk, dim] float32) replacing per-frame "
        "flow-noise keying entirely — draw d at EVERY frame integrates "
        "from tickets[d], so one --sample-draws M run scores every "
        "candidate on every frame. Flow checkpoints only; --sample-draws "
        "must not exceed the bank. The policy name gains _ticket and the "
        "report/dump provenance carries the file's sha256 — a ticket "
        "read must never pass as a keyed-noise (--noise-key) read",
    )
    parser.add_argument(
        "--noise-ticket-map",
        type=Path,
        default=None,
        help="noise-ladder rung 2 (#1): per-dataset ticket routing — a "
        "json {repo_id: bank index} map (bare, or the committed stage-01 "
        "analysis json's stage1.routing_map). Every frame integrates "
        "from its DATASET's routed ticket of the --noise-tickets bank; "
        "deterministic single decode (--sample-draws 1). The policy "
        "name gains _ticketmap (a routed read must never pool as a "
        "single-ticket _ticket read) and the report/dump provenance "
        "carries the map's canonical-form sha256; a dataset absent from "
        "the map is a hard abort, never a silent fallback",
    )
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="bijou prefix-encode batch",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--sample-steps",
        type=int,
        default=10,
        help="flow ODE solver steps",
    )
    parser.add_argument(
        "--sample-method",
        choices=[m.value for m in SamplingMethod],
        default=SamplingMethod.HEUN.value,
    )
    parser.add_argument(
        "--target-time",
        choices=["t", "zero"],
        default="t",
        help="flow target-time conditioning s (SnapFlow φ_s checkpoints "
        "only): 't' = standard s=t forwards (the default; also the only "
        "valid value for unextended checkpoints); 'zero' = one-step "
        "shortcut mode — combine with --sample-method euler "
        "--sample-steps 1 for the 1-NFE read. Never inferred from step "
        "count: 1-NFE claims require this flag explicitly",
    )
    parser.add_argument(
        "--sample-draws",
        type=int,
        default=1,
        help="average this many stochastic decodes per frame: flow "
        "noise draws (prefix encoded once; draw 0 reproduces the "
        "single-draw numbers exactly), or — with --ar-temperature — "
        "temperature-sampled AR chunk decodes sharing one prefill. "
        ">1 is UNCONSTRAINED-class inference — the policy name gains "
        "a _drawsN suffix so ensembled numbers can never pass as "
        "deployment reads",
    )
    parser.add_argument(
        "--ar-temperature",
        type=float,
        default=None,
        help="ar_backbone sampled-draws instrument (the flow "
        "ensembling's mirror): temperature-sample the ACTION block "
        "(Gumbel-max over the grammar-masked softmax; aux value lines "
        "stay greedy) — combine with --sample-draws N for the "
        "mean-of-samples read. The policy name gains a _tT suffix. "
        "Draw RNGs are always frame-identity keyed (a new instrument "
        "has no legacy index path; --noise-key governs flow noise "
        "only). The narrated pass is skipped under sampling — its "
        "greedy voice would pair a different inference class",
    )
    parser.add_argument(
        "--generate",
        nargs="*",
        choices=[f.value for f in AuxField],
        default=None,
        help="ar_backbone request set: fields to elicit before the "
        "actions (template order; 'actions' is implicit and terminal). "
        "Omit for the deployment fast path [generate|actions] — "
        "comparable to aux-less arms. Requires an aux-trained "
        "checkpoint; other decoder kinds reject it",
    )
    parser.add_argument(
        "--mask-state",
        action="store_true",
        help="state-reliance probe: feed the bijou policy its dataset's "
        "state MEAN instead of each frame's true state (the normalized "
        "soft state token collates to exactly zero — no information, "
        "in-distribution magnitude). The policy name gains a "
        "_state-masked suffix; baselines keep the intact state "
        "(state-copy stays the reference). Diagnostic only — never a "
        "deployment read",
    )
    parser.add_argument(
        "--subgoal-mode",
        choices=["oracle", "self", "draws", "mcselect"],
        default=None,
        help="#6 rung (a)/(b) subgoal-conditioning probes "
        "(condition-trained ar_backbone checkpoints): 'oracle' renders "
        "each frame's TRUE segment label through the trained "
        "[subgoal|…] slot (label-less frames decode identically to "
        "baseline; policy name gains _oraclesubgoal); 'self' runs the "
        "two-pass loop — pass 1 greedy-decodes the model's own subgoal "
        "on the planner-less prompt ([generate|subgoal actions]; its "
        "actions are the _narrsubgoal arm, free), pass 2 feeds that "
        "text back through the prompt slot and decodes on the "
        "deployment fast path (_selfsubgoal); 'draws' is rung (b) "
        "candidate selection — pass 1 additionally decodes "
        "--subgoal-draws sampled candidates off the shared prefill, "
        "pass 2 runs BOTH selection arms on the same rows: the frozen "
        "self-certainty pick (_bonsubgoal, deployment-honest) and the "
        "record-only oracle-similarity ceiling (_ceilsubgoal); "
        "'mcselect' is rung (c) masked-contrast scoring — NO in-run "
        "sampling: candidates are injected from the banked rung-(b') "
        "file (--subgoal-candidates-file), each eligible candidate gets "
        "a conditioned greedy decode + a teacher-forced planner-less "
        "reference forward, and the per-candidate KL(p_cond || "
        "p_masked^(1/tau)) is dumped for the offline argmax "
        "(mcselect_results.py owns the pick). Probe "
        "reads, never deployment reads; the banked planner-less "
        "baseline is NOT re-run",
    )
    parser.add_argument(
        "--subgoal-candidates-file",
        type=Path,
        default=None,
        help="requires --subgoal-mode mcselect: the banked rung-(b') "
        "candidates JSON (rows keyed by global frame index, 9 "
        "candidates each) — the run scores EXACTLY this width; the "
        "file's sha256 lands in the report for the read script's "
        "comparator guard",
    )
    parser.add_argument(
        "--mcselect-tau",
        type=float,
        default=None,
        help="requires --subgoal-mode mcselect: the masked reference's "
        "tempering τ (the pre-reg pins 4.0, MG-Select's setting, "
        "adopted verbatim — the read script refuses any other value); "
        "explicit, no silent default on a pre-registered knob",
    )
    parser.add_argument(
        "--subgoal-swap-seed",
        type=int,
        default=None,
        help="requires --subgoal-mode oracle: #6 subgoal-swap content "
        "read (pre-reg 2026-08-09) — replace each labeled frame's TRUE "
        "segment label with a DIFFERENT episode's fraction-matched label "
        "(seeded within-dataset derangement; single-labeled-episode "
        "datasets render an EMPTY slot, recorded; label-less frames stay "
        "baseline-identical). Policy name gains _swapsubgoal; the "
        "pre-reg seed is 0",
    )
    parser.add_argument(
        "--subgoal-swap-identity",
        action="store_true",
        help="requires --subgoal-swap-seed: force every episode to be "
        "its own donor (ALL swap plumbing live) — the pre-launch oracle "
        "(ii) run, which must reproduce the banked oracle arm "
        "byte-exactly. Policy name gains _swapidentity; never a read",
    )
    parser.add_argument(
        "--dump-subgoal-swaps",
        type=Path,
        default=None,
        help="requires --subgoal-swap-seed: write the per-frame swap "
        "provenance as JSON (identity triple, TRUE label, donor episode, "
        "rendered text) — the mechanical oracle-(iv) check recomputes "
        "every row from the judgment sidecars and compares",
    )
    parser.add_argument(
        "--subgoal-draws",
        type=int,
        default=None,
        help="requires --subgoal-mode draws: sampled subgoal candidates "
        "beyond the greedy candidate (pre-reg width 8; 0 = the greedy-"
        "only bit-exactness preflight limit)",
    )
    parser.add_argument(
        "--subgoal-temperature",
        type=float,
        default=None,
        help="requires --subgoal-mode draws with --subgoal-draws > 0: "
        "candidate sampling temperature (pre-reg 1.0); the greedy "
        "candidate and both conditioned decodes stay greedy",
    )
    parser.add_argument(
        "--subgoal-candidate-filter",
        choices=["clean"],
        default=None,
        help="requires --subgoal-mode draws: rung (b') clean-list rule "
        "(pre-reg 2026-08-08-…-cleanlist) — every scorer (both selection "
        "arms and the record-only alternates in the candidates dump) "
        "operates on the eligible list: candidates with truncated == "
        "false, empty list -> greedy fallback (recorded). Pass-1 decode "
        "and dump bytes are unchanged; policy names gain "
        "_boncleansubgoal/_ceilcleansubgoal",
    )
    parser.add_argument(
        "--dump-subgoal-candidates",
        type=Path,
        default=None,
        help="requires --subgoal-mode draws: write the per-frame "
        "candidate table as JSON — identity triple, TRUE label, all "
        "candidate texts + per-step distribution stats (self-certainty "
        "exactly recomputable offline), scorer values and the picks "
        "both arms actually conditioned on — the stage-1 table and the "
        "read script consume this",
    )
    parser.add_argument(
        "--dump-subgoals",
        type=Path,
        default=None,
        help="requires --subgoal-mode self: write pass 1's per-frame "
        "generations as JSON (frame identity triple, instruction, TRUE "
        "segment label, generated text) — the stage-1 validity table "
        "and the results post's qualitative block read this",
    )
    parser.add_argument(
        "--selfsubgoal-force-empty",
        action="store_true",
        help="requires --subgoal-mode self: force pass 2's hint EMPTY on "
        "every frame (the no-hint limit) — the pre-launch oracle (i) "
        "run, which must reproduce the baseline decode bit-exact. The "
        "policy name gains _emptyhint; never a self-arm read",
    )
    parser.add_argument(
        "--condition-override",
        nargs="*",
        default=[],
        metavar="FIELD=VALUE",
        help="counterfactual conditioning (e.g. outcome=success): force "
        "the field(s) instead of each item's true hindsight label — the "
        "Q3 sensitivity diagnostic. Default: TRUE-label conditioning "
        "(score against truth, condition on truth). Condition-trained "
        "checkpoints only",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="write summaries as JSON",
    )
    parser.add_argument(
        "--dump-predictions",
        type=Path,
        default=None,
        help="write every policy's predicted chunks (plus truth/valid/"
        "frame identity) as a compressed .npz — for offline analyses that "
        "must share this eval's exact prediction conventions (e.g. the "
        "DCT-truncation sweep) instead of re-implementing them",
    )
    parser.add_argument(
        "--dump-draws",
        type=Path,
        default=None,
        help="requires --sample-draws > 1: write the bijou "
        "policy's PRE-AVERAGE per-draw chunks [frames, draws, chunk, dim] "
        "(plus truth/valid/frame identity) as a compressed .npz — the "
        "per-draw data that ensembling otherwise averages away (draw "
        "dispersion, best-of-N bounds); the prediction path is untouched",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="write a self-contained HTML report (tables + per-datapoint "
        "prediction charts) to this path",
    )
    parser.add_argument(
        "--report-samples",
        type=int,
        default=12,
        help="datapoints charted in the report (evenly spread over the "
        "sampled frames; bounds report size on large evals)",
    )
    parser.add_argument(
        "--report-theme",
        choices=sorted(THEMES),
        default="dark",
        help="report color scheme",
    )
    args = parser.parse_args()
    if args.sample_plan is not None and args.num_samples is not None:
        parser.error(
            "--sample-plan and --num-samples are mutually exclusive: the "
            "plan IS the sample",
        )
    if args.mask_state and args.checkpoint is None:
        parser.error(
            "--mask-state rewrites the bijou policy's state input — it "
            "requires --checkpoint",
        )
    if args.mask_state and args.smolvla is not None:
        parser.error(
            "--mask-state applies only to the bijou policy; a panel "
            "mixing a masked bijou with an intact smolvla would compare "
            "different inputs — run them separately",
        )
    if args.dump_draws is not None and (
        args.checkpoint is None or args.sample_draws <= 1
    ):
        parser.error(
            "--dump-draws needs a bijou checkpoint ensembling draws "
            "(--checkpoint plus --sample-draws > 1): at draws=1 there is "
            "no per-draw stack to dump — that run's prediction IS the "
            "single draw, --dump-predictions already covers it",
        )
    if args.ar_temperature is not None and args.checkpoint is None:
        parser.error(
            "--ar-temperature samples the bijou policy's AR action "
            "decode — it requires --checkpoint",
        )
    if args.noise_tickets is not None:
        if args.checkpoint is None:
            parser.error(
                "--noise-tickets substitutes the bijou flow policy's "
                "initial noise — it requires --checkpoint",
            )
        if args.smolvla is not None:
            parser.error(
                "--noise-tickets applies only to the bijou policy; a "
                "panel mixing a ticket bijou with a keyed-noise smolvla "
                "would compare different noise modes — run them "
                "separately",
            )
        if args.ar_temperature is not None:
            parser.error(
                "--noise-tickets substitutes flow initial noise; "
                "--ar-temperature samples the AR decode — the modes are "
                "disjoint",
            )
    if args.noise_ticket_map is not None:
        if args.noise_tickets is None:
            parser.error(
                "--noise-ticket-map routes datasets INTO a ticket bank — "
                "it requires --noise-tickets",
            )
        if args.sample_draws != 1:
            parser.error(
                "--noise-ticket-map is a deterministic single decode "
                "(each frame integrates from its dataset's routed "
                "ticket) — --sample-draws must be 1",
            )
    if args.subgoal_mode is not None:
        if args.checkpoint is None:
            parser.error(
                "--subgoal-mode conditions/generates through a trained "
                "checkpoint — it requires --checkpoint",
            )
        if args.generate is not None:
            parser.error(
                "--subgoal-mode owns the request set (pass 1 requests "
                "exactly [generate|subgoal actions], pass 2 exactly "
                "[generate|actions]) — drop --generate",
            )
        if args.ar_temperature is not None or args.sample_draws > 1:
            parser.error(
                "--subgoal-mode is a greedy deployment-fast-path probe; "
                "mixing it with sampled/ensembled decodes would pair "
                "different inference classes — run them separately",
            )
        if args.smolvla is not None:
            parser.error(
                "--subgoal-mode applies only to the bijou policy — run "
                "smolvla separately (the --mask-state precedent)",
            )
        if args.mask_state:
            parser.error(
                "--subgoal-mode with --mask-state mixes two probes in "
                "one read — not pre-registered; run them separately",
            )
        if any(pair.partition("=")[0] == "subgoal" for pair in args.condition_override):
            parser.error(
                "--subgoal-mode and --condition-override subgoal=… are "
                "two sources for the same prompt slot — pick one",
            )
    if args.subgoal_swap_seed is not None and args.subgoal_mode != "oracle":
        parser.error(
            "--subgoal-swap-seed rides the oracle conditioning path (the "
            "swap arm IS the oracle arm with deranged labels) — it "
            "requires --subgoal-mode oracle",
        )
    if args.subgoal_swap_identity and args.subgoal_swap_seed is None:
        parser.error(
            "--subgoal-swap-identity forces the swap map to identity — "
            "it requires --subgoal-swap-seed",
        )
    if args.dump_subgoal_swaps is not None and args.subgoal_swap_seed is None:
        parser.error(
            "--dump-subgoal-swaps writes the swap arm's per-frame "
            "provenance — it requires --subgoal-swap-seed",
        )
    if args.dump_subgoals is not None and args.subgoal_mode not in ("self", "draws"):
        parser.error(
            "--dump-subgoals writes pass 1's generations — it requires "
            "--subgoal-mode self or draws",
        )
    if args.selfsubgoal_force_empty and args.subgoal_mode not in ("self", "draws"):
        parser.error(
            "--selfsubgoal-force-empty forces pass 2's hint empty — it "
            "requires --subgoal-mode self or draws",
        )
    if args.subgoal_mode == "draws":
        if args.subgoal_draws is None:
            parser.error(
                "--subgoal-mode draws needs an explicit --subgoal-draws "
                "(the pre-reg width is 8; 0 is the preflight limit) — "
                "no silent default on a pre-registered knob",
            )
        if args.subgoal_draws < 0:
            parser.error(f"--subgoal-draws must be >= 0, got {args.subgoal_draws}")
        if args.subgoal_draws > 0 and args.subgoal_temperature is None:
            parser.error(
                "--subgoal-draws > 0 samples candidates — set "
                "--subgoal-temperature explicitly (the pre-reg value is 1.0)",
            )
        if args.subgoal_draws == 0 and args.subgoal_temperature is not None:
            parser.error(
                "--subgoal-temperature with --subgoal-draws 0 is a no-op "
                "— the greedy-only limit samples nothing; drop it",
            )
    else:
        for flag, value in (
            ("--subgoal-draws", args.subgoal_draws),
            ("--subgoal-temperature", args.subgoal_temperature),
            ("--subgoal-candidate-filter", args.subgoal_candidate_filter),
            ("--dump-subgoal-candidates", args.dump_subgoal_candidates),
        ):
            if value is not None:
                parser.error(f"{flag} requires --subgoal-mode draws")
    if args.subgoal_mode == "mcselect":
        if args.subgoal_candidates_file is None:
            parser.error(
                "--subgoal-mode mcselect injects banked candidates — "
                "set --subgoal-candidates-file (the rung-(b') dump)",
            )
        if args.mcselect_tau is None:
            parser.error(
                "--subgoal-mode mcselect needs an explicit --mcselect-tau "
                "(the pre-reg value is 4.0) — no silent default on a "
                "pre-registered knob",
            )
        if args.dump_predictions is None:
            parser.error(
                "--subgoal-mode mcselect without --dump-predictions "
                "would measure and throw the mcselect:* arrays away — "
                "the read script consumes the npz",
            )
        if args.selfsubgoal_force_empty:
            parser.error("--selfsubgoal-force-empty requires --subgoal-mode self")
    else:
        for flag, value in (
            ("--subgoal-candidates-file", args.subgoal_candidates_file),
            ("--mcselect-tau", args.mcselect_tau),
        ):
            if value is not None:
                parser.error(f"{flag} requires --subgoal-mode mcselect")
    return args


def main() -> int:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    # Eval workers ship RAW items (10+ tensor storages each, vs training's
    # one collated batch); the default fd-based sharing pins one shm fd per
    # storage and blows the 1024-fd ulimit -> 'received 0 items of ancdata'.
    # The file_system strategy shares via named files instead.
    torch.multiprocessing.set_sharing_strategy("file_system")

    # Multi-GPU sharding (torchrun): gloo, not nccl — eval's only
    # collective is one object gather at the end, and gloo gathers
    # pickled CPU objects directly (nccl would stage the report tensors
    # through the GPU). Inference still runs on cuda:LOCAL_RANK per rank.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = 0
    device = torch.device(args.device)
    if distributed:
        torch.distributed.init_process_group("gloo")
        rank = torch.distributed.get_rank()
        if device.type == "cuda":
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
            torch.cuda.set_device(device)
    is_main = rank == 0

    if args.subgoal_swap_seed is not None and distributed:
        # The pre-registered swap arm is a single-process local run; the
        # per-frame provenance records live on the policy object and are
        # not shard-gathered — refuse rather than dump a partial map.
        raise SystemExit(
            "--subgoal-swap-seed is a single-process instrument (swap "
            f"records are not shard-gathered) — got WORLD_SIZE={world_size}",
        )

    episode_split = EpisodeSplit(args.episodes)
    if episode_split is not EpisodeSplit.ALL and args.holdout_episodes <= 0:
        raise SystemExit(f"--episodes {args.episodes} requires --holdout-episodes > 0")
    selection = select_datasets(
        tuple(args.data),
        tuple(args.exclude),
        args.chunk_size,
        episode_split=episode_split,
        holdout_fraction=args.holdout_episodes,
        split_seed=args.split_seed,
        allowed_fps=tuple(args.fps) if args.fps else None,
        allowed_camera_counts=(
            tuple(args.camera_counts) if args.camera_counts else None
        ),
        required_prompt_hash=args.aux_prompt_hash,
        # Condition-trained checkpoints render each item's TRUE labels;
        # loading them costs seconds and is harmless for older models.
        load_episode_annotations=args.checkpoint is not None,
    )
    dataset = selection.concat()
    if is_main:
        print(
            f"eval data: {len(selection.datasets)} datasets, "
            f"{selection.total_episodes} episodes, {len(dataset)} frames, "
            f"action/state dim {selection.action_dim}/{selection.state_dim}",
            flush=True,
        )
        if episode_split is not EpisodeSplit.ALL:
            print(
                f"episode split: {episode_split.value} "
                f"(fraction {args.holdout_episodes}, split seed {args.split_seed}; "
                f"{selection.held_out_episodes} held-out episodes across "
                f"{selection.held_out_datasets} datasets)",
                flush=True,
            )
        if selection.dropped:
            print(
                f"dropped {len(selection.dropped)} incompatible datasets:",
                flush=True,
            )
            for reason in selection.dropped:
                print(f"  - {reason}", flush=True)

    subgoal_swap: SubgoalSwapMap | None = None
    if args.subgoal_swap_seed is not None:
        # Pure-CPU metadata pre-pass over the SELECTED datasets (the
        # derangement domain is pinned to the eval's own selection);
        # deterministic in the seed, so a re-run reproduces the map.
        subgoal_swap = build_swap_map(
            {
                wrapped.dataset.repo_id: Path(wrapped.dataset.root)
                for wrapped in selection.datasets
            },
            seed=args.subgoal_swap_seed,
            identity=args.subgoal_swap_identity,
        )
        print(
            f"subgoal swap map (seed {args.subgoal_swap_seed}"
            + (", IDENTITY check mode" if args.subgoal_swap_identity else "")
            + f"): {len(subgoal_swap.episodes)} datasets with labeled "
            f"episodes, "
            f"{sum(len(e) for e in subgoal_swap.episodes.values())} labeled "
            f"episodes, "
            f"{sum(len(d) for d in subgoal_swap.donors.values())} donor-"
            f"mapped; {len(subgoal_swap.skipped)} dataset(s) skipped "
            "(single labeled episode -> labeled frames render an EMPTY "
            "slot, recorded)",
            flush=True,
        )

    plan: SamplePlan | None = None
    if args.sample_plan is not None:
        # One column scan serves both build and resolve (~8 min on
        # curated-v0 — measured when it ran twice).
        tables = episode_tables(selection)
        if args.sample_plan.exists():
            plan = SamplePlan.load(args.sample_plan)
            validate_plan(
                plan,
                episodes=args.episodes,
                holdout_episodes=args.holdout_episodes,
                split_seed=args.split_seed,
                fps=list(args.fps) if args.fps else None,
                camera_counts=(
                    list(args.camera_counts) if args.camera_counts else None
                ),
            )
            if is_main:
                print(
                    f"sample plan LOADED from {args.sample_plan} "
                    f"(plan seed {plan.plan_seed}, "
                    f"{plan.frames_per_episode}/episode core + "
                    f"≤{plan.labeled_per_episode}/episode labeled, "
                    f"created {plan.created_at})",
                    flush=True,
                )
        else:
            # Deterministic: every rank builds the identical plan; only
            # rank 0 writes it.
            plan = build_plan(
                tables,
                plan_seed=args.plan_seed,
                frames_per_episode=args.frames_per_episode,
                labeled_per_episode=args.labeled_per_episode,
                episodes=args.episodes,
                holdout_episodes=args.holdout_episodes,
                split_seed=args.split_seed,
                fps=list(args.fps) if args.fps else None,
                camera_counts=(
                    list(args.camera_counts) if args.camera_counts else None
                ),
            )
            if is_main:
                plan.save(args.sample_plan)
                print(
                    f"sample plan BUILT and written to {args.sample_plan} "
                    f"(plan seed {args.plan_seed})",
                    flush=True,
                )
        indices, core_indices = resolve_plan(plan, tables)
        num_samples = len(indices)
        if is_main:
            print(
                f"panel: {len(core_indices)} core frames (headline) + "
                f"{num_samples - len(core_indices)} labeled-oversample "
                "frames (aux metrics only)",
                flush=True,
            )
    else:
        num_samples = (
            len(dataset)
            if args.num_samples is None
            else min(args.num_samples, len(dataset))
        )
        indices = sorted(
            random.Random(args.seed).sample(range(len(dataset)), num_samples),
        )
        core_indices = set(indices)
    # Round-robin over the SORTED indices: every rank sees a near-even
    # spread across datasets, and shard membership is a pure function of
    # (seed, world_size).
    shard_indices = indices[rank::world_size]
    if is_main:
        print(
            f"sampling {num_samples} frames (seed {args.seed})"
            + (
                f", sharded over {world_size} ranks (~{len(shard_indices)}/rank)"
                if distributed
                else ""
            ),
            flush=True,
        )

    policies: list[ChunkPolicy] = [
        StateCopyPolicy(args.chunk_size),
        NormalizedStateCopyPolicy(args.chunk_size),
    ]
    # "zero" = SnapFlow one-step shortcut conditioning; BijouPolicy
    # validates the checkpoint carries φ_s.
    target_time = 0.0 if args.target_time == "zero" else None
    if target_time is not None and args.checkpoint is None:
        raise SystemExit(
            "--target-time zero conditions the flow expert — it requires --checkpoint",
        )
    bijou_policy: BijouPolicy | None = None
    narrated_policy: NarratedBijouPolicy | None = None
    pass1_policy: SelfSubgoalPass1Policy | None = None
    self_policy: SelfSubgoalPolicy | None = None
    bon_policy: SelectedSubgoalPolicy | None = None
    ceil_policy: SelectedSubgoalPolicy | None = None
    mcselect_policy: MaskedContrastSubgoalPolicy | None = None
    candidates_sha256: str | None = None
    if args.checkpoint is not None:
        overrides: dict[str, str] = {}
        for pair in args.condition_override:
            field, _, value = pair.partition("=")
            if not value:
                raise SystemExit(
                    f"--condition-override expects FIELD=VALUE, got {pair!r}",
                )
            overrides[field] = value
        bijou_policy = BijouPolicy(
            args.checkpoint,
            device=device,
            seed=args.seed,
            sample_steps=args.sample_steps,
            method=SamplingMethod(args.sample_method),
            sample_draws=args.sample_draws,
            ar_temperature=args.ar_temperature,
            target_time=target_time,
            noise_key=args.noise_key,
            tickets=args.noise_tickets,
            ticket_map=args.noise_ticket_map,
            mask_state=args.mask_state,
            generate=tuple(AuxField(f) for f in (args.generate or ())),
            condition_override=overrides,
            # Subgoal conditioning renders only when explicitly forced
            # (it is an operator input, not a hindsight label — the
            # deployment default is planner-less).
            include_subgoal_condition="subgoal" in overrides,
            subgoal_mode=args.subgoal_mode,
            subgoal_swap=subgoal_swap,
        )
        if bijou_policy.info.chunk_size != args.chunk_size:
            raise SystemExit(
                f"checkpoint chunk size {bijou_policy.info.chunk_size} != "
                f"--chunk-size {args.chunk_size}",
            )
        if args.subgoal_mode == "self":
            # The two-pass arms REPLACE the plain bijou row: the
            # planner-less baseline is banked, never re-run (pre-reg).
            # Pass 1 must sit before pass 2 in the list — the runner
            # scores policies in order per batch, and pass 2 reads pass
            # 1's generations for exactly those frames.
            pass1_policy = SelfSubgoalPass1Policy(bijou_policy)
            self_policy = SelfSubgoalPolicy(
                bijou_policy,
                pass1_policy,
                force_empty=args.selfsubgoal_force_empty,
            )
            policies.extend([pass1_policy, self_policy])
            if is_main:
                print(
                    f"subgoal mode SELF: {pass1_policy.name} (pass 1, "
                    f"narrated-subgoal arm) feeds {self_policy.name} "
                    "(pass 2, prompt-slot conditioning); plain bijou row "
                    "skipped (baseline is banked)",
                    flush=True,
                )
        elif args.subgoal_mode == "draws":
            # Rung (b): the same pass-1-first contract; BOTH selection
            # arms ride one run so they share pass 1's candidate decode
            # and stay row/composition-matched to each other.
            pass1_policy = SelfSubgoalPass1Policy(
                bijou_policy,
                draws=args.subgoal_draws,
                temperature=args.subgoal_temperature or 1.0,
            )
            bon_policy = SelectedSubgoalPolicy(
                bijou_policy,
                pass1_policy,
                mode="bon",
                force_empty=args.selfsubgoal_force_empty,
                candidate_filter=args.subgoal_candidate_filter,
            )
            ceil_policy = SelectedSubgoalPolicy(
                bijou_policy,
                pass1_policy,
                mode="ceil",
                force_empty=args.selfsubgoal_force_empty,
                candidate_filter=args.subgoal_candidate_filter,
            )
            policies.extend([pass1_policy, bon_policy, ceil_policy])
            if is_main:
                print(
                    f"subgoal mode DRAWS: {pass1_policy.name} decodes "
                    f"1 greedy + {args.subgoal_draws} sampled candidates "
                    f"(T={args.subgoal_temperature}) and feeds "
                    f"{bon_policy.name} (self-certainty pick) + "
                    f"{ceil_policy.name} (record-only oracle-similarity "
                    "ceiling); plain bijou row skipped (baseline is "
                    "banked)"
                    + (
                        "; CLEAN-LIST filter active (rung (b'): scorers "
                        "see non-truncated candidates only, greedy "
                        "fallback recorded)"
                        if args.subgoal_candidate_filter == "clean"
                        else ""
                    ),
                    flush=True,
                )
        elif args.subgoal_mode == "mcselect":
            # Rung (c): candidates are INJECTED from the banked rung-(b')
            # file — no in-run sampling; the plain bijou row is skipped
            # (the masked reference rides the policy itself and the
            # baseline is banked — the npz must never re-carry it).
            assert args.subgoal_candidates_file is not None  # parse_args
            candidates_bytes = args.subgoal_candidates_file.read_bytes()
            candidates_sha256 = hashlib.sha256(candidates_bytes).hexdigest()
            candidates_doc = json.loads(candidates_bytes)
            doc_checkpoint = Path(str(candidates_doc.get("checkpoint", "")))
            if doc_checkpoint.parts[-2:] != args.checkpoint.parts[-2:]:
                raise SystemExit(
                    f"candidates file decoded from {doc_checkpoint} but "
                    f"this run scores {args.checkpoint} — cross-checkpoint "
                    "re-ranking voids every banked comparator, stop",
                )
            mcselect_policy = MaskedContrastSubgoalPolicy(
                bijou_policy,
                candidates_by_index={
                    int(row["index"]): row["candidates"]
                    for row in candidates_doc["rows"]
                },
                tau=args.mcselect_tau,
            )
            policies.append(mcselect_policy)
            if is_main:
                print(
                    f"subgoal mode MCSELECT: {mcselect_policy.name} scores "
                    f"{mcselect_policy.n_candidates} banked candidates/row "
                    f"(file sha {candidates_sha256[:12]}…, tau="
                    f"{args.mcselect_tau:g}) — conditioned greedy decode + "
                    "teacher-forced masked reference per eligible "
                    "candidate; the ARGMAX lives in the read script; "
                    "plain bijou row skipped (baseline is banked)",
                    flush=True,
                )
        else:
            policies.append(bijou_policy)
            if args.subgoal_mode == "oracle" and is_main:
                if subgoal_swap is None:
                    print(
                        f"subgoal mode ORACLE: {bijou_policy.name} renders "
                        "each frame's TRUE segment label through the "
                        "[subgoal|…] slot (label-less frames = baseline "
                        "context)",
                        flush=True,
                    )
                else:
                    print(
                        f"subgoal mode ORACLE+SWAP: {bijou_policy.name} "
                        "renders each labeled frame's DONOR "
                        "fraction-matched label through the [subgoal|…] "
                        "slot (label-less frames = baseline context; "
                        "unswappable labeled frames = EMPTY slot, "
                        "recorded)"
                        + (
                            " — IDENTITY mode: donor = self, must "
                            "byte-reproduce the banked oracle arm"
                            if subgoal_swap.identity
                            else ""
                        ),
                        flush=True,
                    )
        if (
            bijou_policy.aux_fields
            and args.generate is None
            and args.ar_temperature is None
            and args.subgoal_mode is None
        ):
            # The narrated pass rides automatically on aux-trained
            # checkpoints (shared model, ~2x bijou inference): its
            # paired chunk MAE is the full-sample does-narration-help
            # answer, and its generations feed the aux metrics + report
            # blocks. An explicit --generate means the MAIN policy
            # already narrates — no second pass then. Skipped under
            # --ar-temperature too: the narrated pass decodes greedily,
            # and pairing it against a sampled base row would compare
            # different inference classes.
            narrated_policy = NarratedBijouPolicy(bijou_policy)
            policies.append(narrated_policy)
            if is_main:
                print(
                    f"narrated pass ON: {narrated_policy.name} requests "
                    f"{[f.value for f in narrated_policy.fields]}",
                    flush=True,
                )
    if args.smolvla is not None:
        policies.append(
            SmolVLAEvalPolicy(
                args.smolvla,
                device=device,
                seed=args.seed,
                lerobot_stats=selection.lerobot_stats,
                noise_key=args.noise_key,
            ),
        )

    # Fetch each sampled frame once (parallel decode; spawn context — the
    # main process may hold CUDA state and torchcodec is fork-unsafe) and
    # stream it through every policy so comparisons stay paired.
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, shard_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=identity_collate,
        worker_init_fn=eval_worker_init if args.num_workers > 0 else None,
        multiprocessing_context="spawn" if args.num_workers > 0 else None,
    )
    # Datapoints whose predictions are retained for the report: evenly
    # spread over the sampled frames so charts span datasets, bounded so
    # large evals stay renderable.
    report_stride = max(len(indices) // max(args.report_samples, 1), 1)
    report_indices = set(indices[::report_stride][: args.report_samples])
    report_samples: dict[int, ReportSample] = {}

    scores: dict[str, list[FrameScore]] = {p.name: [] for p in policies}
    # Identity/truth columns are shared by both dump flavors; the heavy
    # per-policy and per-draw payloads fill only when their flag asks.
    dumping = args.dump_predictions is not None or args.dump_draws is not None
    dump_predictions: dict[str, list[torch.Tensor]] = {p.name: [] for p in policies}
    dump_draws: list[torch.Tensor] = []
    dump_truth: list[torch.Tensor] = []
    dump_valid: list[torch.Tensor] = []
    dump_repo: list[str] = []
    dump_index: list[int] = []
    dump_episode: list[int] = []
    dump_frame: list[int] = []
    # Frame-aligned label records for Q2 slices and the aux metrics,
    # keyed by global frame index (None = unlabeled at that frame).
    outcomes: dict[int, str | None] = {}
    holding_labels: dict[int, float] = {}
    progress_labels: dict[int, float] = {}
    event_labels: dict[int, str] = {}
    visible_labels: dict[int, str] = {}
    # Q3: measured iff outcome-conditioning is trained AND the run isn't
    # already a manual counterfactual — nor a subgoal-mode probe (its
    # rows are conditioned reads; flipping outcome on top would measure
    # a compound counterfactual at double GPU cost).
    q3 = (
        bijou_policy is not None
        and "outcome" in bijou_policy.info.condition_fields
        and not bijou_policy.condition_override
        and args.subgoal_mode is None
    )
    sensitivity_deltas: list[float] = []
    done = 0
    for batch_number, items in enumerate(loader):
        batch_indices = shard_indices[done : done + len(items)]
        for item, index in zip(items, batch_indices, strict=True):
            outcomes[index] = item.get("condition_outcome")
            for key, store in (
                ("annotation.holding", holding_labels),
                ("annotation.progress", progress_labels),
            ):
                value = item.get(key)
                if value is not None and bool(torch.isfinite(value)):
                    store[index] = float(value)
            # Event/visible labels via the shared presence rules (event
            # carries the explicit "none" on judge-sampled frames).
            texts = label_values(item, (AuxField.EVENT, AuxField.VISIBLE))
            if (event := texts.get(AuxField.EVENT)) is not None:
                event_labels[index] = event
            if (visible := texts.get(AuxField.VISIBLE)) is not None:
                visible_labels[index] = visible
        batch_predictions: dict[str, list[torch.Tensor]] = {}
        for policy in policies:
            start = time.perf_counter()
            predictions = policy.predict(items, batch_indices)
            elapsed = (time.perf_counter() - start) / len(items)
            batch_predictions[policy.name] = predictions
            if (
                args.dump_draws is not None
                and bijou_policy is not None
                and policy is bijou_policy
            ):
                # Collected before the narrated/Q3 passes can call
                # predict again and overwrite the policy's last batch.
                assert bijou_policy.last_draws is not None  # draws > 1
                for item, row in zip(
                    items,
                    bijou_policy.last_draws,
                    strict=True,
                ):
                    dump_draws.append(row[:, : item["action"].shape[0]].float())
            for item, index, predicted in zip(
                items,
                batch_indices,
                predictions,
                strict=True,
            ):
                truth = item["action"]
                predicted = predicted[: truth.shape[0]].float()
                scores[policy.name].append(
                    score_frame(
                        index=index,
                        repo_id=str(item["repo_id"]),
                        predicted=predicted,
                        truth=truth.float(),
                        valid=~item["action_is_pad"],
                        inference_seconds=elapsed,
                    ),
                )
                if args.dump_predictions is not None:
                    dump_predictions[policy.name].append(predicted)
                if dumping and policy is policies[0]:
                    dump_truth.append(truth.float())
                    dump_valid.append(~item["action_is_pad"])
                    dump_repo.append(str(item["repo_id"]))
                    dump_index.append(index)
                    dump_episode.append(int(item["episode_index"]))
                    dump_frame.append(int(item["frame_index"]))
                if args.report is not None and index in report_indices:
                    sample = report_samples.get(index) or ReportSample(
                        index=index,
                        episode=int(item["episode_index"]),
                        frame_in_episode=int(item["frame_index"]),
                        repo_id=str(item["repo_id"]),
                        task=str(item["task"]),
                        state=item["observation.state"].float(),
                        cameras={
                            k.removeprefix("observation.images."): v
                            for k, v in item.items()
                            if k.startswith("observation.images.")
                        },
                        truth=truth.float(),
                        valid=~item["action_is_pad"],
                        predictions={},
                        aux_generated=None,
                        aux_label=(
                            aux_label_text(item, bijou_policy.aux_fields) or None
                            if bijou_policy is not None
                            else None
                        ),
                    )
                    sample.predictions[policy.name] = predicted
                    report_samples[index] = sample
        if q3:
            assert bijou_policy is not None  # q3 construction
            flipped = [
                (position, item)
                for position, item in enumerate(items)
                if item.get("condition_outcome") not in (None, "success")
            ]
            if flipped:
                forced_items = [
                    {**item, "condition_outcome": "success"} for _, item in flipped
                ]
                forced_indices = [batch_indices[position] for position, _ in flipped]
                forced = bijou_policy.predict(forced_items, forced_indices)
                for (position, item), prediction in zip(flipped, forced, strict=True):
                    base = batch_predictions[bijou_policy.name][position]
                    valid = ~item["action_is_pad"]
                    length = item["action"].shape[0]
                    delta = (
                        (prediction[:length].float() - base[:length].float())
                        .abs()[valid]
                        .mean()
                    )
                    sensitivity_deltas.append(float(delta))
        done += len(items)
        if is_main and batch_number % 5 == 0:
            print(
                f"  scored {done}/{len(shard_indices)} frames"
                + (" (rank 0 shard)" if distributed else ""),
                flush=True,
            )

    # One downstream path for any world size: gather every rank's shard,
    # merge on rank 0 (index-sorted, so results are world-size-invariant
    # for AR decodes — sharding.py has the flow caveat), others exit.
    local = ShardResults(
        scores=scores,
        outcomes=outcomes,
        holding_labels=holding_labels,
        progress_labels=progress_labels,
        event_labels=event_labels,
        visible_labels=visible_labels,
        sensitivity_deltas=sensitivity_deltas,
        report_samples=report_samples,
        generations=(
            narrated_policy.generations if narrated_policy is not None else {}
        ),
        subgoal_records=(pass1_policy.records if pass1_policy is not None else {}),
        subgoal_candidates=(
            pass1_policy.candidates if pass1_policy is not None else {}
        ),
        subgoal_picks={
            policy.mode: policy.picks
            for policy in (bon_policy, ceil_policy)
            if policy is not None
        },
        mcselect_kl=(mcselect_policy.kl if mcselect_policy is not None else {}),
        mcselect_cand_pred=(
            mcselect_policy.cand_pred if mcselect_policy is not None else {}
        ),
        mcselect_pred_masked=(
            mcselect_policy.pred_masked if mcselect_policy is not None else {}
        ),
        dump_predictions=dump_predictions,
        dump_truth=dump_truth,
        dump_valid=dump_valid,
        dump_repo=dump_repo,
        dump_index=dump_index,
        dump_episode=dump_episode,
        dump_frame=dump_frame,
        dump_draws=dump_draws,
    )
    if distributed:
        gathered: list[ShardResults | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, local)
        torch.distributed.destroy_process_group()
        if not is_main:
            return 0
        results = merge_shards([shard for shard in gathered if shard is not None])
    else:
        results = merge_shards([local])

    def dump_identity() -> dict[str, np.ndarray]:
        """Truth + frame-identity columns shared by both dump flavors."""
        return {
            "truth": torch.stack(results.dump_truth).numpy(),
            "valid": torch.stack(results.dump_valid).numpy(),
            "index": np.array(results.dump_index),
            "repo_id": np.array(results.dump_repo),
            # Dataset-local identity: 'index' is a CONCAT index, valid
            # only under this eval's exact selection — these two keep
            # rows addressable when the corpus composition changes.
            "episode_index": np.array(results.dump_episode),
            "frame_index": np.array(results.dump_frame),
            # Core-panel membership (all-True without a sample plan).
            "core": np.array([i in core_indices for i in results.dump_index]),
        }

    if args.dump_predictions is not None:
        payload: dict[str, np.ndarray] = dump_identity()
        for name, chunks in results.dump_predictions.items():
            payload[f"pred:{name}"] = torch.stack(chunks).numpy()
        if mcselect_policy is not None:
            # The rung-(c) contract keys (mcselect_results.py): KL
            # [N, C] float64 NaN-at-ineligible, per-candidate decoded
            # chunks [N, C, S, D], the planner-less reference [N, S, D]
            # — all row-aligned with the identity columns.
            payload[f"{MCSELECT_PREFIX}kl"] = np.array(
                [results.mcselect_kl[i] for i in results.dump_index],
                dtype=np.float64,
            )
            payload[f"{MCSELECT_PREFIX}cand_pred"] = torch.stack(
                [results.mcselect_cand_pred[i] for i in results.dump_index],
            ).numpy()
            payload[f"{MCSELECT_PREFIX}pred_masked"] = torch.stack(
                [results.mcselect_pred_masked[i] for i in results.dump_index],
            ).numpy()
        if args.noise_tickets is not None:
            # Ticket-mode provenance (pre-reg 2026-08-08 rung 2, item 5):
            # the single-decode npz must carry the bank sha — and, when
            # routed, the map sha — so a scorer can refuse to pool a
            # ticket/routed npz against keyed or single-ticket anchors.
            assert bijou_policy is not None  # parse_args enforced
            payload["noise_tickets"] = np.array(str(args.noise_tickets))
            payload["tickets_sha256"] = np.array(bijou_policy.tickets_sha256 or "")
            payload["noise_ticket_map"] = np.array(
                str(args.noise_ticket_map or ""),
            )
            payload["ticket_map_sha256"] = np.array(
                bijou_policy.ticket_map_sha256 or "",
            )
        np.savez_compressed(args.dump_predictions, allow_pickle=False, **payload)
        print(f"dumped predictions to {args.dump_predictions}", flush=True)
    if args.dump_draws is not None:
        assert bijou_policy is not None  # parse_args enforced
        np.savez_compressed(
            args.dump_draws,
            allow_pickle=False,
            **dump_identity(),
            # [frames, draws, chunk, dim] pre-average stacks; mean over
            # axis 1 reproduces pred:<policy> up to float32 rounding.
            draws=torch.stack(results.dump_draws).numpy(),
            # Scoring semantics, so the npz stays interpretable standalone
            # (the report JSON records the full set — #18.1 precedent).
            policy=np.array(bijou_policy.name),
            sample_steps=np.array(args.sample_steps),
            sample_method=np.array(args.sample_method),
            sample_draws=np.array(args.sample_draws),
            target_time=np.array(args.target_time),
            noise_key=np.array(args.noise_key),
            # Ticket-mode provenance (empty strings = keyed noise): the
            # scorer refuses to pool a ticket npz against keyed anchors
            # without these matching its expectations.
            noise_tickets=np.array(str(args.noise_tickets or "")),
            tickets_sha256=np.array(bijou_policy.tickets_sha256 or ""),
            mask_state=np.array(args.mask_state),
            seed=np.array(args.seed),
        )
        print(f"dumped per-draw chunks to {args.dump_draws}", flush=True)
    if args.dump_subgoals is not None:
        # Sorted by global frame index — world-size-invariant like every
        # other output; rows carry the dataset-local identity triple so
        # they stay addressable when the corpus composition changes.
        rows = [
            dataclasses.asdict(record)
            for _, record in sorted(results.subgoal_records.items())
        ]
        args.dump_subgoals.parent.mkdir(parents=True, exist_ok=True)
        args.dump_subgoals.write_text(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "subgoal_mode": args.subgoal_mode,
                    "selfsubgoal_force_empty": args.selfsubgoal_force_empty,
                    "seed": args.seed,
                    "rows": rows,
                },
                indent=2,
            ),
        )
        print(
            f"dumped {len(rows)} per-frame subgoals to {args.dump_subgoals}",
            flush=True,
        )
    if args.dump_subgoal_swaps is not None:
        # Single-process by the WORLD_SIZE guard, so the policy's own
        # record map IS the full map; sorted by the identity triple.
        assert bijou_policy is not None  # parse_args enforced the mode
        assert subgoal_swap is not None
        swap_rows = [
            dataclasses.asdict(record)
            for _, record in sorted(bijou_policy.swap_records.items())
        ]
        args.dump_subgoal_swaps.parent.mkdir(parents=True, exist_ok=True)
        args.dump_subgoal_swaps.write_text(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "subgoal_swap_seed": args.subgoal_swap_seed,
                    "subgoal_swap_identity": args.subgoal_swap_identity,
                    "seed": args.seed,
                    "skipped_datasets": list(subgoal_swap.skipped),
                    "swapped": sum(
                        1 for row in swap_rows if row["donor_episode_index"] is not None
                    ),
                    "empty_rendered": sum(
                        1 for row in swap_rows if row["donor_episode_index"] is None
                    ),
                    "rows": swap_rows,
                },
                indent=2,
            ),
        )
        print(
            f"dumped {len(swap_rows)} per-frame subgoal swaps to "
            f"{args.dump_subgoal_swaps}",
            flush=True,
        )
    if args.dump_subgoal_candidates is not None:
        # Sorted by global frame index (world-size-invariant). Scorer
        # values ride precomputed for readability, but the per-step
        # stats make every one exactly recomputable offline — and the
        # dumped picks are the arms' LIVE picks (cross-checked against
        # an offline recompute by the read script, oracle-grade).
        candidate_rows = []
        for index, record in sorted(results.subgoal_records.items()):
            row_candidates = results.subgoal_candidates.get(index)
            if row_candidates is None:
                raise SystemExit(
                    f"frame {index} has a subgoal record but no candidate "
                    "list — pass 1 candidate capture broke, stop",
                )
            # Every scorer — the live picks and the record-only
            # alternates — operates on the same eligible list (the full
            # list when unfiltered; the rung-(b') clean list otherwise).
            eligible = (
                eligible_indices([c.truncated for c in row_candidates])
                if args.subgoal_candidate_filter == "clean"
                else list(range(len(row_candidates)))
            )
            candidate_row: dict[str, Any] = {
                "index": index,
                "repo_id": record.repo_id,
                "episode_index": record.episode_index,
                "frame_index": record.frame_index,
                "instruction": record.instruction,
                "true_subgoal": record.true_subgoal,
                "greedy_subgoal": record.generated_subgoal,
                "candidates": [
                    dataclasses.asdict(candidate) for candidate in row_candidates
                ],
                "self_certainty": [
                    self_certainty(c.mean_logprob, c.allowed_vocab)
                    for c in row_candidates
                ],
                "mean_chosen_logprob": [
                    mean_chosen_logprob(c.chosen_logprob) for c in row_candidates
                ],
                "picks": {
                    "bon": results.subgoal_picks.get("bon", {}).get(index),
                    "ceil": results.subgoal_picks.get("ceil", {}).get(index),
                    # Record-only alternates, computed offline here —
                    # never conditioned on (pre-reg: no post-hoc
                    # promotion).
                    "likelihood": eligible[
                        likelihood_pick(
                            [row_candidates[i].chosen_logprob for i in eligible],
                        )
                    ],
                    "medoid": eligible[
                        medoid_pick([row_candidates[i].text for i in eligible])
                    ],
                },
            }
            if args.subgoal_candidate_filter is not None:
                # Membership in the scorer's operating list (a fallback
                # row keeps [candidate 0] though it is truncated).
                members = set(eligible)
                candidate_row["eligible"] = [
                    i in members for i in range(len(row_candidates))
                ]
                candidate_row["fallback"] = all(c.truncated for c in row_candidates)
            candidate_rows.append(candidate_row)
        args.dump_subgoal_candidates.parent.mkdir(parents=True, exist_ok=True)
        args.dump_subgoal_candidates.write_text(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "subgoal_mode": args.subgoal_mode,
                    "subgoal_draws": args.subgoal_draws,
                    "subgoal_temperature": args.subgoal_temperature,
                    "subgoal_candidate_filter": args.subgoal_candidate_filter,
                    "selfsubgoal_force_empty": args.selfsubgoal_force_empty,
                    "seed": args.seed,
                    "rows": candidate_rows,
                },
                indent=2,
            ),
        )
        print(
            f"dumped {len(candidate_rows)} per-frame candidate tables to "
            f"{args.dump_subgoal_candidates}",
            flush=True,
        )

    # Headline aggregation (summaries, per-dataset, Q2, paired) runs over
    # the CORE panel only — under a sample plan the labeled-oversample
    # frames would bias every frame-mean toward judged frames. Aux
    # metrics and Q3 read ALL scored frames: that oversampling is their
    # reason to exist. Without a plan, core == everything.
    core_scores = {
        name: [score for score in frame_scores if score.index in core_indices]
        for name, frame_scores in results.scores.items()
    }
    summaries = [
        summarize(name, frame_scores) for name, frame_scores in core_scores.items()
    ]
    motor_names = selection.action_names or [
        f"motor_{i}" for i in range(selection.action_dim)
    ]
    per_dataset = slice_by_dataset(core_scores)

    # Q2: chunk MAE sliced by TRUE outcome (a bucketing of the one
    # true-label-conditioned pass — the same semantics as the in-run
    # eval/chunk_mae_* series; not a counterfactual). Buckets align with
    # the merged (index-sorted) score lists via the outcome-by-index map.
    reference_scores = next(iter(core_scores.values()))
    buckets = [
        outcome if outcome in OUTCOME_BUCKETS else "unlabeled"
        for outcome in (results.outcomes[score.index] for score in reference_scores)
    ]
    outcome_slices: dict[str, dict[str, float]] = {}
    if len(set(buckets)) > 1:
        for name, frame_scores in core_scores.items():
            outcome_slices[name] = {}
            for bucket in OUTCOME_BUCKETS:
                subset = [
                    s for s, b in zip(frame_scores, buckets, strict=True) if b == bucket
                ]
                if subset:
                    outcome_slices[name][bucket] = summarize(name, subset).chunk_mae

    # Aux metrics: narrated generations vs the (weak) judge labels over
    # every labeled sampled frame — the proper-n version of the in-run
    # 12-row probes. Weak supervision (~80% inter-judge holding
    # agreement, ±15% progress MAE): ceilings sit near the label noise.
    holding_accuracy: float | None = None
    holding_frames = 0
    progress_mae: float | None = None
    progress_frames = 0
    event_accuracy: float | None = None
    event_frames = 0
    visible_accuracy: float | None = None
    visible_frames = 0
    if narrated_policy is not None:
        holding_hits = [
            int(generation.holding == bool(int(label)))
            for index, label in results.holding_labels.items()
            if (generation := results.generations.get(index)) is not None
            and generation.holding is not None
        ]
        if holding_hits:
            holding_frames = len(holding_hits)
            holding_accuracy = sum(holding_hits) / holding_frames
        progress_errors = [
            abs(generation.progress - label)
            for index, label in results.progress_labels.items()
            if (generation := results.generations.get(index)) is not None
            and generation.progress is not None
        ]
        if progress_errors:
            progress_frames = len(progress_errors)
            progress_mae = sum(progress_errors) / progress_frames
        # Event: presence detection (generated "none" vs any event) —
        # free-text match would punish paraphrase, presence is the
        # decision that matters. Negatives exist only on judge-sampled
        # frames (explicit-"none" labels).
        event_hits = [
            int((generation.event.strip() == EVENT_NONE) == (label == EVENT_NONE))
            for index, label in results.event_labels.items()
            if (generation := results.generations.get(index)) is not None
            and generation.event is not None
        ]
        if event_hits:
            event_frames = len(event_hits)
            event_accuracy = sum(event_hits) / event_frames
        # Visibility: set-equality of the parsed positional slots (both
        # object and gripper must match exactly; order-insensitive).
        visible_hits = [
            int(generated_slots == label_slots)
            for index, label in results.visible_labels.items()
            if (generation := results.generations.get(index)) is not None
            and generation.visible is not None
            and (generated_slots := parse_visibility(generation.visible)) is not None
            and (label_slots := parse_visibility(label)) is not None
        ]
        if visible_hits:
            visible_frames = len(visible_hits)
            visible_accuracy = sum(visible_hits) / visible_frames

    print("\n== chunk metrics (raw action units, pad-masked) ==", flush=True)
    print(
        format_table(
            [
                [
                    s.name,
                    f"{s.chunk_mae:.3f}",
                    f"{s.mae_p50:.3f}",
                    f"{s.mae_p90:.3f}",
                    f"{s.first_mae:.3f}",
                    f"{s.chunk_mse:.1f}",
                    f"{s.seconds_per_frame * 1000:.0f}",
                ]
                for s in summaries
            ],
            ["policy", "chunk_mae", "p50", "p90", "first_mae", "chunk_mse", "ms/frame"],
        ),
        flush=True,
    )

    print("\n== per-motor chunk MAE ==", flush=True)
    print(
        format_table(
            [[s.name, *(f"{v:.2f}" for v in s.per_motor_mae)] for s in summaries],
            ["policy", *motor_names],
        ),
        flush=True,
    )

    if outcome_slices:
        print("\n== Q2: chunk MAE by TRUE outcome label ==", flush=True)
        counts = {b: buckets.count(b) for b in OUTCOME_BUCKETS if b in buckets}
        print(
            format_table(
                [
                    [
                        name,
                        *(
                            f"{slices[b]:.3f}" if b in slices else "-"
                            for b in OUTCOME_BUCKETS
                        ),
                    ]
                    for name, slices in outcome_slices.items()
                ],
                [
                    "policy",
                    *(f"{b} (n={counts.get(b, 0)})" for b in OUTCOME_BUCKETS),
                ],
            ),
            flush=True,
        )
    if q3 and results.sensitivity_deltas:
        deltas = results.sensitivity_deltas
        print(
            f"\n== Q3: condition sensitivity == mean |Δ| "
            f"{sum(deltas) / len(deltas):.4f} over "
            f"{len(deltas)} labeled non-success frames "
            "(outcome forced to 'success' vs true-label conditioning)",
            flush=True,
        )
    if narrated_policy is not None and holding_frames + progress_frames > 0:
        nan = float("nan")
        print(
            f"\n== aux vs weak labels ({narrated_policy.name}) == "
            f"holding acc "
            f"{holding_accuracy if holding_accuracy is not None else nan:.3f} "
            f"(n={holding_frames}), progress MAE "
            f"{progress_mae if progress_mae is not None else nan:.3f} "
            f"(n={progress_frames}), event acc "
            f"{event_accuracy if event_accuracy is not None else nan:.3f} "
            f"(n={event_frames}), visible acc "
            f"{visible_accuracy if visible_accuracy is not None else nan:.3f} "
            f"(n={visible_frames})",
            flush=True,
        )

    baseline = policies[0].name
    comparisons = [
        compare_paired(
            s.name,
            core_scores[s.name],
            baseline,
            core_scores[baseline],
        )
        for s in summaries[1:]
    ]
    if narrated_policy is not None and bijou_policy is not None:
        # The full-sample does-narration-help pairing, first class.
        comparisons.append(
            compare_paired(
                narrated_policy.name,
                core_scores[narrated_policy.name],
                bijou_policy.name,
                core_scores[bijou_policy.name],
            ),
        )
    if self_policy is not None and pass1_policy is not None:
        # The channel read, paired in-eval: (nearly) the same text
        # entering through the prompt slot (pass 2) vs the suffix voice
        # (pass 1) — where the text enters, separated from whether text
        # helps. The Δ-vs-banked-baseline reads stay offline (frozen).
        comparisons.append(
            compare_paired(
                self_policy.name,
                core_scores[self_policy.name],
                pass1_policy.name,
                core_scores[pass1_policy.name],
            ),
        )
    if comparisons:
        print("\n== paired comparisons (negative delta = better) ==", flush=True)
        print(
            format_table(
                [
                    [
                        c.policy,
                        c.reference,
                        f"{c.mean_delta:+.3f}",
                        f"{c.delta_p50:+.3f}",
                        f"{100 * c.win_rate:.0f}%",
                    ]
                    for c in comparisons
                ],
                ["policy", "vs", "mean_delta_mae", "delta_p50", "win_rate"],
            ),
            flush=True,
        )

    if args.output_json is not None:
        report = EvalReport(
            data=[str(p) for p in args.data],
            episodes=args.episodes,
            holdout_episodes=args.holdout_episodes,
            split_seed=args.split_seed,
            fps=list(args.fps) if args.fps else None,
            camera_counts=list(args.camera_counts) if args.camera_counts else None,
            num_samples=num_samples,
            seed=args.seed,
            checkpoint=str(args.checkpoint) if args.checkpoint else None,
            smolvla=args.smolvla,
            exclude=list(args.exclude),
            aux_prompt_hash=args.aux_prompt_hash,
            sample_steps=args.sample_steps,
            sample_method=args.sample_method,
            sample_draws=args.sample_draws,
            ar_temperature=args.ar_temperature,
            target_time=args.target_time,
            noise_key=args.noise_key,
            noise_tickets=(
                str(args.noise_tickets) if args.noise_tickets is not None else None
            ),
            tickets_sha256=(
                bijou_policy.tickets_sha256 if bijou_policy is not None else None
            ),
            noise_ticket_map=(
                str(args.noise_ticket_map)
                if args.noise_ticket_map is not None
                else None
            ),
            ticket_map_sha256=(
                bijou_policy.ticket_map_sha256 if bijou_policy is not None else None
            ),
            mask_state=args.mask_state,
            subgoal_mode=args.subgoal_mode,
            subgoal_swap_seed=args.subgoal_swap_seed,
            subgoal_swap_identity=args.subgoal_swap_identity,
            subgoal_draws=args.subgoal_draws,
            subgoal_temperature=args.subgoal_temperature,
            subgoal_candidate_filter=args.subgoal_candidate_filter,
            selfsubgoal_force_empty=args.selfsubgoal_force_empty,
            mcselect_tau=args.mcselect_tau,
            mcselect_candidates_file=(
                str(args.subgoal_candidates_file)
                if args.subgoal_candidates_file is not None
                else None
            ),
            candidates_sha256=candidates_sha256,
            generate=list(args.generate) if args.generate is not None else None,
            condition_override=list(args.condition_override),
            batch_size=args.batch_size,
            world_size=world_size,
            summaries=summaries,
            paired=comparisons,
            motor_names=motor_names,
            outcome_slices=outcome_slices,
            condition_sensitivity=(
                sum(results.sensitivity_deltas) / len(results.sensitivity_deltas)
                if results.sensitivity_deltas
                else None
            ),
            condition_sensitivity_frames=len(results.sensitivity_deltas),
            holding_accuracy=holding_accuracy,
            holding_frames=holding_frames,
            progress_mae=progress_mae,
            progress_frames=progress_frames,
            event_accuracy=event_accuracy,
            event_frames=event_frames,
            visible_accuracy=visible_accuracy,
            visible_frames=visible_frames,
            sample_plan=str(args.sample_plan) if args.sample_plan else None,
            plan_seed=plan.plan_seed if plan is not None else None,
            core_frames=len(core_indices),
            labeled_frames=num_samples - len(core_indices),
            per_dataset=per_dataset,
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report.to_json_dict(), indent=2))
        print(f"\nwrote {args.output_json}", flush=True)

    if args.report is not None:
        # Local wall-clock time, offset-aware (naive now() is ambiguous
        # when comparing reports generated on different machines).
        generated = datetime.datetime.now(datetime.UTC).astimezone()
        config_lines = [
            f"generated: {generated.isoformat(timespec='seconds')}",
            f"data: {', '.join(str(p) for p in args.data)}",
            (
                f"selection: {len(selection.datasets)} datasets, "
                f"{selection.total_episodes} episodes, {len(dataset)} frames "
                f"({len(selection.dropped)} dropped)"
            ),
            f"samples: {num_samples} frames, seed {args.seed}",
            f"episodes: {args.episodes}"
            + (
                f" (holdout fraction {args.holdout_episodes}, "
                f"split seed {args.split_seed})"
                if episode_split is not EpisodeSplit.ALL
                else ""
            ),
            f"checkpoint: {args.checkpoint or '-'}",
            f"smolvla: {args.smolvla or '-'}",
            f"sampler: {args.sample_method}-{args.sample_steps}",
            "target time: "
            + (
                "zero (SnapFlow one-step shortcut field)"
                if args.target_time == "zero"
                else "t (standard)"
            ),
            f"noise key: {args.noise_key}"
            + (
                f" OVERRIDDEN by tickets {args.noise_tickets} (sha256 "
                f"{bijou_policy.tickets_sha256})"
                if bijou_policy is not None and bijou_policy.tickets_sha256
                else ""
            )
            + (
                f" ROUTED per-dataset by {args.noise_ticket_map} (map "
                f"sha256 {bijou_policy.ticket_map_sha256})"
                if bijou_policy is not None and bijou_policy.ticket_map_sha256
                else ""
            ),
            "state: "
            + (
                "MASKED to dataset mean (state-reliance probe)"
                if args.mask_state
                else "intact"
            ),
            f"fps filter: {args.fps or 'all'}",
            f"camera-count filter: {args.camera_counts or 'all'}",
            f"generate: {args.generate if args.generate is not None else '(fast path)'}",
            "subgoal mode: "
            + (
                args.subgoal_mode
                + (
                    f" (1 greedy + {args.subgoal_draws} sampled "
                    f"T={args.subgoal_temperature})"
                    if args.subgoal_mode == "draws"
                    else ""
                )
                + (" (FORCED-EMPTY hint)" if args.selfsubgoal_force_empty else "")
                if args.subgoal_mode is not None
                else "none (planner-less)"
            ),
            (
                f"sample plan: {args.sample_plan} (plan seed {plan.plan_seed}, "
                f"{len(core_indices)} core + "
                f"{num_samples - len(core_indices)} labeled frames; headline "
                "metrics = core panel only)"
                if plan is not None
                else "sample plan: none (uniform frame sampling)"
            ),
        ]
        extra_tables: list[ReportTable] = []
        if outcome_slices:
            slice_counts = {
                bucket: buckets.count(bucket)
                for bucket in OUTCOME_BUCKETS
                if bucket in buckets
            }
            extra_tables.append(
                ReportTable(
                    title="Q2: chunk MAE by TRUE outcome label",
                    header=[
                        "policy",
                        *(
                            f"{bucket} (n={slice_counts.get(bucket, 0)})"
                            for bucket in OUTCOME_BUCKETS
                        ),
                    ],
                    rows=[
                        [
                            name,
                            *(
                                f"{slices[bucket]:.3f}" if bucket in slices else "-"
                                for bucket in OUTCOME_BUCKETS
                            ),
                        ]
                        for name, slices in outcome_slices.items()
                    ],
                ),
            )
        diagnostics: list[list[str]] = []
        if results.sensitivity_deltas:
            diagnostics.append(
                [
                    "Q3 condition sensitivity (mean |Δ|, outcome→success)",
                    f"{sum(results.sensitivity_deltas) / len(results.sensitivity_deltas):.4f}",
                    str(len(results.sensitivity_deltas)),
                ],
            )
        if holding_accuracy is not None:
            diagnostics.append(
                [
                    "holding accuracy vs weak labels (narrated pass)",
                    f"{holding_accuracy:.3f}",
                    str(holding_frames),
                ],
            )
        if progress_mae is not None:
            diagnostics.append(
                [
                    "progress MAE vs weak labels (narrated pass)",
                    f"{progress_mae:.3f}",
                    str(progress_frames),
                ],
            )
        if event_accuracy is not None:
            diagnostics.append(
                [
                    "event presence accuracy vs weak labels (narrated pass)",
                    f"{event_accuracy:.3f}",
                    str(event_frames),
                ],
            )
        if visible_accuracy is not None:
            diagnostics.append(
                [
                    "visibility slot-set accuracy vs weak labels (narrated pass)",
                    f"{visible_accuracy:.3f}",
                    str(visible_frames),
                ],
            )
        if diagnostics:
            extra_tables.append(
                ReportTable(
                    title="Conditioning & aux diagnostics",
                    header=["metric", "value", "n"],
                    rows=diagnostics,
                ),
            )
        policy_names = [s.name for s in summaries]
        # Worst-first by the evaluated policy's MAE (the actionable tail
        # on top); the baseline sorts it when no checkpoint was given.
        # Subgoal modes skip the plain bijou row (the baseline is
        # banked), so the sorter must be a policy that actually RAN —
        # sorting by the absent bare name was a KeyError that silently
        # cost the rung-(b') q4 run its HTML (caught by the mcselect
        # smoke 2026-08-09).
        if self_policy is not None:
            sort_policy = self_policy.name
        elif bon_policy is not None:
            sort_policy = bon_policy.name
        elif mcselect_policy is not None:
            sort_policy = mcselect_policy.name
        elif bijou_policy is not None and bijou_policy.name in policy_names:
            sort_policy = bijou_policy.name
        else:
            sort_policy = policy_names[0]
        collapsible_tables = [
            ReportTable(
                title=(
                    f"Per-dataset chunk MAE ({len(per_dataset)} datasets; "
                    f"sorted by {sort_policy} MAE, worst first; rows with "
                    "few frames are noise-dominated)"
                ),
                header=["dataset", "frames", *policy_names],
                rows=[
                    [
                        repo_id,
                        str(dataset_slice.frames),
                        *(
                            f"{dataset_slice.chunk_mae[name]:.3f}"
                            for name in policy_names
                        ),
                    ]
                    for repo_id, dataset_slice in sorted(
                        per_dataset.items(),
                        key=lambda pair: -pair[1].chunk_mae[sort_policy],
                    )
                ],
            ),
        ]
        samples = [results.report_samples[i] for i in sorted(results.report_samples)]
        if narrated_policy is not None:
            samples = [
                dataclasses.replace(
                    sample,
                    aux_generated=(
                        generation.text or "(empty)"
                        if (generation := results.generations.get(sample.index))
                        is not None
                        else None
                    ),
                )
                for sample in samples
            ]
        render_report(
            args.report,
            config_lines,
            summaries,
            comparisons,
            motor_names,
            samples,
            total_scored=num_samples,
            theme=THEMES[args.report_theme],
            extra_tables=extra_tables,
            collapsible_tables=collapsible_tables,
        )
        print(f"wrote {args.report}", flush=True)
    return 0
