# The VLA architecture — families over dispatch

Status: agreed design, pre-implementation. This is the plan document
for replacing `BijouModel` with the `VLA` trait lattice and model
families; it will be subsumed into `docs/architecture.md` section by
section as phases land, and then reads as the historical record with
per-phase verdicts (the `docs/molmoact2-retirement.md` pattern).

Companion conventions: `docs/code-styleguide.md` (notably the
"Classes and inheritance" section — this design is its first large
application) and `docs/working-together.md` (gates, oracles,
re-baselining discipline).

**Backwards compatibility is explicitly out of scope.** Old checkpoints
we care about are converted by a script (§10); everything else — state
dict keys, checkpoint layouts, CLI flags, wandb series names — is free
to change. Parity with the pre-refactor world is proven by oracles
(§11), not by byte-stable artifacts.

## 0. For Fontaine — how to adopt these changes

Your lines consume four surfaces this plan rewrites. Nothing lands
mid-run: each phase is a normal `main` commit, and the two phases that
affect you (5 and 8) get an in-channel boundary post before they merge.

1. **Checkpoint format changes (phase 3/8).** Every bijou checkpoint
   directory gains `metadata.json` + per-component safetensors and
   loses the old section layout. `bijou.convert_legacy` converts old
   directories in place-adjacent copies; your artifacts on the
   inventory (er-60k, the stage-C grasp-SFT endpoints, rig-r1
   converts) are converted at phase 8 with parity receipts posted.
   Until you rebase past phase 5, your checkout keeps reading the old
   format — old code reads old checkpoints, new code reads new; do not
   mix within one checkout.
2. **`bijou.train` CLI reshape (phase 5).** Fresh runs name a family
   (`--family molmoact2_joint`); `--expert-*` flags get decoder-scoped
   names (§8 ledger). Your verbatim-class launchers need a one-time
   re-pin after you rebase past phase 5; the diff-receipt discipline
   you already use is exactly right for it. The stats seam you pinned
   (molmo_flow inheriting the SOURCE checkpoint's baked q01/q99)
   becomes a first-class conversion operation — `--replace-stats` at
   convert time (§7), so the 3-step corrected-table prep collapses to
   one flag.
3. **`BijouPolicy --checkpoint` keeps its CLI surface** through the
   port (your wrist-screen checkpoint-surface verdict stays true); it
   loads new-format checkpoints after phase 5.
4. **GRPO wave/row formats are untouched.** Row NPZ and loop `.pt`
   stay frozen — they are run artifacts, not checkpoints. The replay
   stack re-points at phase 5; the box gate re-runs mask bit-equality
   and the registered 1e-4 cross-decomposition logprob bound on your
   banked R1-A/R1-B waves before the phase closes. Post-migration GRPO
   runs start from converted checkpoints (decision 11 already says
   fresh starts).

Rebase guidance: rebase at your next run boundary after phase 5 lands;
phases 0–4 are additive or pure motion and safe to absorb any time.

## 1. Background — why BijouModel goes

`bijou/model.py`'s `BijouModel` is a generic composition root: one
trunk (`Gemma4Model | Molmo2Model` behind a type parameter), one
`ObservationEncoder`, one decoder out of a five-way union, plus three
mutable run-property slots (`distill`, `joint_ce` + weight,
`insulate_expert`). Every operation is a `match` over the decoder
union (~10 methods × 5 arms), the trunk generic is un-told by
`_gemma_backbone()`/`_molmo2_backbone()` narrowing helpers, `encode`
asks `isinstance(self.decoder, …)` to decide cache retention, and the
predict knobs (`target_time`, `generate`, `noise`) are each legal for
one decoder kind and runtime-guarded for the rest.

The encoder×decoder matrix promised composability that never existed:
there are ~6 real models, not 15 combinations. The `match` statements
are exhaustive (pyright-gated) but no model's logic lives anywhere —
it is sliced across `model.py`, `BijouTrainStep`'s autocast seams, and
`train.py`'s adamc/eval-probe branches.

The replacement: a small abstract **`VLA`** trait that train/eval/
rollout program against, capability sub-traits that make "this model
has a discrete action head" a type-level fact, and one concrete
**family class per real model**. Families own their assembly,
precision policy, and loss composition; encoders and decoders remain
shared building blocks; illegal states stop being representable.

## 2. Decision register

| # | Decision | Rationale (short) |
|---|----------|-------------------|
| D1 | Trait lattice `VLA` / `ARVLA` / `FlowVLA` / `NarratingVLA`, all stateless `nn.Module + abc.ABC` | capabilities as types; consumers state head requirements in signatures; styleguide's trait rule |
| D2 | One family class per (trunk, trained-surface set): six families (§5) | class = what the model IS; loading refuses capability claims the checkpoint can't back |
| D3 | Objective is a **constructor payload** (frozen dataclass union per family), never a class split, never a mutable slot, never a forward argument | run constant; param_groups/DDP/encode topology derive from it; rank divergence unrepresentable; kills `distill`/`joint_ce`/`insulate_expert` slots |
| D4 | Objective (graph facts: terms, weights, gradient gates) vs optimizer (LRs, decay, schedule) are separate layers; `param_groups()` is the interface: structural offer, reconciled against LR flags in train.py, contradictions error at startup | model never sees LRs; the one policy→structure feedback edge (frozen trunk kills `state_proj`'s grad path) stays in train.py, documented |
| D5 | DDP-correct losses are two-phase: `loss_counts(batch)` → all-reduce → `forward(batch, counts=…)`; `Loss(sum, count)` is the component currency; `LossReport.objective` is the per-rank graph scalar whose DDP mean is the global objective | a single (sum, count) pair cannot represent a multi-term objective with uneven per-rank counts; this is today's `loss_component_sums`/`count_normalizers` protocol, made the contract |
| D6 | `VLA.predict(batch)` takes no knobs and runs the **recorded serving operating point** (checkpoint metadata); knobbed inference lives on capability traits with statically-legal signatures | kills the knob-legality raise matrix; cross-family paired evals compare like with like |
| D7 | Per-trait prediction structs: `ARPrediction(actions)`, `FlowPrediction(actions, noise)`, `NarratedPrediction(actions, generations)`; `BijouPrediction` retires | the None-union dies; a flow prediction with generations is unconstructible |
| D8 | Narration is its own trait, orthogonal to the action head; `predict_narrated` requires non-empty `generate` and a batch collated with the same request | format-6 AR narrates never; a future narrated-flow family narrates without an AR head |
| D9 | New checkpoint format (§7): `metadata.json` + per-component files; `backbone.safetensors` **always present, hard-linked** to the HF blob when pristine; explicit `backbone_trained` flag; stats tables and serving point are metadata; stats replacement is a first-class conversion op | self-containment without disk cost; presence-as-signal (the frozen⇒pristine trap) dies; fontaine's baked-q01/q99 seam becomes one flag |
| D10 | Naming ledger (§8): **decoder** = action/text-emitting component; **head** = shallow projection only (`lm_head`, `fast_head`); **backbone/trunk** and **encoder** as before; **expert** retired from identifiers | four words, four concepts; "Molmo2ARHead reading the trunk's lm_head" was the overload D10 prevents |
| D11 | Package split: `modelling/` = building blocks (trunks, encoders, decoders, interface, nn, aux_text), `models/` = families + objectives; `vla.py` top-level | gemma4 beside `models/` was a layer violation in the directory tree |
| D12 | Codec/tokenizer split follows the styleguide ledger: tokenizers (artifact + math) stay `fast/`; codecs (AR conventions: specials, block anchoring, symbol lengths) move to `modelling/decoders/codecs.py` | codecs are coupled to head id-space conventions, not to the artifact |
| D13 | Instrument currencies (`ARSampling`, `ActionCaptureStep`, `ValueCandidate`) live in `modelling/interface.py`; `vla.py` imports downward | nothing in `modelling` may import `vla` |
| D14 | Parity with the pre-refactor world is proven by the five loss oracles + decode fixtures + instrument parity (§11), re-anchored loudly when a CLI shape changes but never when a number would move | bitwise numbers are format-independent truths; bytes are not |

## 3. The trait lattice — `bijou/vla.py`

Full target code. Docstrings are the contract; families implement
every method (the traits own no state and no algorithm).

```python
"""The VLA trait lattice — the model contract train/eval/rollout
program against.

A :class:`VLA` is a complete vision-language-action model: it owns its
trunk, prompt side, and action/text decoders, and how those are
assembled is entirely the implementation's business. The base trait is
what EVERY consumer needs (collate, train, predict, route parameters,
persist); the sub-traits — :class:`ARVLA`, :class:`FlowVLA`,
:class:`NarratingVLA` — are capabilities a family declares by
inheritance, so "this model has a discrete action head" is a
type-level fact: consumers state requirements in signatures
(``def replay[I](model: ARVLA[I])``) and the wrong family is a type
error, not a runtime raise.

All traits are STATELESS (no fields, no ``__init__``) — pure interface
plus contracts. Families live in ``bijou/models/``, one class per
(trunk, trained-surface set); shared machinery is composed modules and
free functions, never intermediate base classes.

Pairing contract: batches passed to :meth:`VLA.forward` and the
predict methods come from THIS model's :meth:`VLA.collator` — the type
parameter ``I`` names that coupling, and loading is the single
boundary where it erases to ``VLA[Any]`` (generic consumer functions
restore precision).

Import DAG: ``models/*`` → ``vla`` → ``modelling/*`` → ``fast``.
Nothing in ``modelling`` imports this module.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Self, override

import torch
from torch import Tensor, nn

from .modelling.aux_text import AuxField, AuxGeneration
from .modelling.interface import (
    ActionCaptureStep,
    ARSampling,
    BatchInputs,
    CollatedBatch,
    InputsCollator,
    SamplingMethod,
    ValueCandidate,
)

# ---------------------------------------------------------------------
# Identity


class VLAFamily(Enum):
    """The closed set of model families — loading's registry keys and
    the value recorded/printed for provenance (rollout banner, eval
    reports, checkpoint metadata). One member per (trunk,
    trained-surface set); the surface set is also visible in the family
    class's trait bases, and the two must agree."""

    GEMMA_FLOW = "gemma_flow"
    GEMMA_AR = "gemma_ar"
    MOLMO2_AR = "molmo2_ar"
    MOLMOACT2_FLOW = "molmoact2_flow"
    MOLMOACT2_AR = "molmoact2_ar"
    MOLMOACT2_JOINT = "molmoact2_joint"


@dataclass(frozen=True, slots=True)
class VLASpec:
    """The model's identity card — what rollout prints and eval
    records. Derived by each family from its configs (a VIEW, never
    independent state)."""

    family: VLAFamily
    chunk_size: int
    action_dim: int


# ---------------------------------------------------------------------
# Training currency


@dataclass(frozen=True, slots=True)
class Loss:
    """A loss in transportable form: the unnormalized sum over
    contributing elements plus the count that normalizes it — the
    DDP-correct currency: ranks all-reduce ``sum`` and ``count``
    separately and divide AFTER, so uneven per-rank counts (aux text,
    holdout rows) cannot skew the mean. The scalar loss is always the
    quotient — there is deliberately no ``.value`` shortcut.

    Shapes:
      - ``sum``: [] (0-d) — graph-connected on the training path
      - ``count``: [] (0-d) — detached; elements contributing to sum
    """

    sum: Tensor
    count: Tensor


@dataclass(frozen=True, slots=True)
class LossReport:
    """One batch's training objective, decomposed for logging.

    ``objective`` is the per-rank graph scalar built with the
    ALL-REDUCED counts the loop passed to :meth:`VLA.forward` — its
    DDP gradient average equals the gradient of the global objective
    even under uneven per-rank counts. ``components`` are the chart
    series, keyed by the family's own component names ("action",
    "aux", …) — a dict by design: the key set is family-dynamic but
    run-constant, and equals :meth:`VLA.loss_counts`' key set (the
    loop enforces this). Component sums are graph-connected (they are
    the objective's addends); logging detaches."""

    objective: Tensor
    components: dict[str, Loss]


# ---------------------------------------------------------------------
# Prediction currency (one struct per trait method — a flow prediction
# with generations, or an AR one with a noise draw, cannot be built)


@dataclass(frozen=True, slots=True)
class ARPrediction:
    """A discrete-decoder decode's product.

    Shapes:
      - ``actions``: [B, chunk, action_dim] — RAW action units
        (mirrors ``CollatedBatch.actions``, the ground truth it is
        scored against)
    """

    actions: Tensor


@dataclass(frozen=True, slots=True)
class FlowPrediction:
    """A flow integration's product. ``noise`` is ALWAYS the initial
    draw the solver actually integrated (supplied or drawn) — paired
    re-decodes must reuse it, or sampling variance floors any
    conditioning-sensitivity signal.

    Shapes:
      - ``actions``: [B, chunk, action_dim] — RAW action units
      - ``noise``: [B, chunk, action_dim] — normalized units
    """

    actions: Tensor
    noise: Tensor


@dataclass(frozen=True, slots=True)
class NarratedPrediction:
    """A narrated pass's product: actions plus one
    :class:`AuxGeneration` per batch row (raw text is the report
    ground truth; a lenient-parse failure is a None field inside the
    generation, never a missing row).

    Shapes:
      - ``actions``: [B, chunk, action_dim] — RAW action units
    """

    actions: Tensor
    generations: list[AuxGeneration]


# ---------------------------------------------------------------------
# The traits


class VLA[I: BatchInputs](nn.Module, abc.ABC):
    """A complete vision-language-action model (module docstring).

    ``nn.Module`` because implementations must BE modules (DDP wrap,
    optimizer, device moves, ``state_dict``); abstract because the
    trait owns no state and no algorithm — families implement every
    method.

    Contracts every implementation honors:

    - batches come from this model's :meth:`collator` (the ``I``
      pairing); behavior on foreign batches is undefined by type
      design, not defensively checked;
    - :meth:`forward` owns its OWN precision policy (autocast regions,
      fp32 seams, loss-term ordering) — callers apply no ambient
      autocast;
    - the predict surfaces run under ``torch.no_grad``
      (implementations decorate) and return RAW action units via the
      batch's per-dataset quantile stats;
    - models are ALWAYS constructed with an objective — eval/rollout
      construction passes the checkpoint's recorded one, so a loaded
      model can always compute its own training loss.
    """

    @property
    @abc.abstractmethod
    def spec(self) -> VLASpec:
        """Identity card, derived from the family's configs."""

    @abc.abstractmethod
    def collator(self) -> InputsCollator[I]:
        """The sole producer of this model's batches. Pickleable — it
        crosses into spawned dataloader workers."""

    @abc.abstractmethod
    def loss_counts(self, batch: CollatedBatch[I]) -> dict[str, Tensor]:
        """Per-component element counts for this batch (detached,
        0-d). The loop all-reduces these BEFORE :meth:`forward` —
        global normalizers are what make the DDP gradient average
        exact under uneven per-rank counts. Key set is run-constant
        and equals the report's ``components`` keys."""

    @override
    @abc.abstractmethod
    def forward(
        self,
        batch: CollatedBatch[I],
        *,
        counts: dict[str, Tensor],
    ) -> LossReport:
        """One batch's training objective (the DDP entry point).
        ``counts`` are the all-reduced returns of :meth:`loss_counts`
        (passed through un-reduced on a single process). Chunked
        backward is supported by construction: calling forward on
        micro-batch slices with the SAME global counts yields
        objective addends whose sum equals the full-batch objective."""

    @abc.abstractmethod
    def predict(self, batch: CollatedBatch[I]) -> Tensor:
        """Actions at the model's RECORDED serving operating point
        (checkpoint metadata — solver and step count for flow, greedy
        decode for AR), taking no knobs so every family can answer and
        cross-family paired evals compare like with like. Knobbed
        inference lives on the capability traits.

        Shapes:
          - returns: [B, chunk, action_dim] — RAW action units
        """

    @abc.abstractmethod
    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        """Named parameter groups — the LR-routing vocabulary
        ("decoder", "backbone_text", "backbone_vision") — as the
        STRUCTURAL offer under this model's construction: a parameter
        appears iff the objective's graph can deliver it a gradient
        (insulation empties a flow-only run's backbone groups;
        construction-frozen tensors never appear). Which offered
        groups actually train, and at what LR, is optimizer policy —
        train.py cross-checks the flags against this offer and errors
        on contradictions. Groups are disjoint; DDP's exactness
        contract rides on their union minus policy freezes."""

    @abc.abstractmethod
    def output_head_parameters(self) -> list[nn.Parameter]:
        """The trainable OUTPUT-projection parameters — the subset
        that keeps standard AdamW decay under ``--optimizer adamc``
        while every hidden matrix gets the corrected decay. May be
        empty (adamc then degenerates to uniform corrected decay); a
        family with a frozen output head answers with the empty list,
        explicitly."""

    @abc.abstractmethod
    def checkpoint_components(self) -> dict[str, nn.Module]:
        """Component name → module subtree, the family's declaration
        of its checkpoint sections (§7). The loading toolkit maps each
        entry to ``<name>.safetensors``; the backbone is NOT listed
        here (the toolkit handles it via the hard-link rule)."""

    @classmethod
    @abc.abstractmethod
    def from_checkpoint(
        cls,
        checkpoint: "Path",  # noqa: F821 — sketch; real code imports Path
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Self:
        """Reconstruct the trained model from a checkpoint directory
        with no reference to any other directory (self-containment is
        the format's invariant, §7)."""


class ARVLA[I: BatchInputs](VLA[I], abc.ABC):
    """The discrete-action-decoder capability: this model emits its
    action chunk as tokens, so the block can be teacher-forced,
    sampled, and captured — the GRPO and chunk-NLL instrument surface.
    Nothing about text: narration is :class:`NarratingVLA`."""

    @abc.abstractmethod
    def predict_ar(
        self,
        batch: CollatedBatch[I],
        *,
        sampling: ARSampling | None = None,
        capture: list[ActionCaptureStep] | None = None,
    ) -> ARPrediction:
        """The action-block decode, never any text: prompt encode, the
        family's opener, BOA forced (its identity is scaffold, not a
        decision), then the grammar-masked block decode — each step
        masks ids whose symbol expansion exceeds the remaining budget,
        PAD legal only at budget zero.

        ``sampling=None`` decodes greedily (deterministic per frame —
        the deployment and paired-eval path); an :class:`ARSampling`
        switches the ACTION block to per-row temperature sampling.
        ``capture``, when given, receives one
        :class:`ActionCaptureStep` per decode step, taken from the
        very logits the decode chose from — no re-forward, no numeric
        drift vs the executed decode."""

    @abc.abstractmethod
    def teacher_forced_block_logits(
        self,
        batch: CollatedBatch[I],
        action_ids: Tensor,
    ) -> Tensor:
        """Block logits for GIVEN action ids under teacher forcing —
        one prefill plus one suffix forward, no decode loop (the
        chunk-NLL metric and the GRPO replay-logprob surface).
        Deterministic per frame; batch composition moves bf16
        reduction order, so cross-run comparisons pin batch shape.

        Shapes:
          - ``action_ids``: [B, S] long — block-relative ids
          - returns: [B, S, vocab_total] float32 — position j scores
            ``action_ids[:, j]``
        """


class FlowVLA[I: BatchInputs](VLA[I], abc.ABC):
    """The flow-matching capability: actions come from integrating a
    learned velocity field from Gaussian noise."""

    @abc.abstractmethod
    def predict_flow(
        self,
        batch: CollatedBatch[I],
        *,
        num_steps: int,
        method: SamplingMethod,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> FlowPrediction:
        """Integrate the velocity field with an EXPLICIT operating
        point — the knobs are required here; defaults live only in
        :meth:`VLA.predict`. ``noise`` supplies the initial draw
        (paired re-decodes reuse a prior prediction's); otherwise it
        is drawn from ``generator`` on CPU (device-independent draws,
        eval's seeding convention).

        Shapes:
          - ``noise``: [B, chunk, action_dim] — normalized units
        """


class NarratingVLA[I: BatchInputs](VLA[I], abc.ABC):
    """The text-surface capability: the model was trained to emit aux
    value lines (subgoal, holding, progress, event, visible) and can
    generate them at inference — beside whichever action decoder the
    family has. Orthogonal to :class:`ARVLA`: a format-5 AR family
    narrates inside its one AR pass; a narrated-flow family decodes a
    text suffix beside flow-sampled actions."""

    @abc.abstractmethod
    def predict_narrated(
        self,
        batch: CollatedBatch[I],
        *,
        generate: tuple[AuxField, ...],
    ) -> NarratedPrediction:
        """The narrated pass: requested fields decoded as text, then
        actions. ``generate`` must be non-empty (action-only inference
        is :meth:`VLA.predict` / :meth:`ARVLA.predict_ar`), a subset
        of the checkpoint's TRAINED fields in template order (AuxField
        declaration order), and equal to the request the batch was
        collated with — a mismatch sits off the conditioning and is a
        loud error.

        Per field, in order: greedy text decode under the field's
        value budget until the ``\\n`` terminator — budget exhaustion
        forces the terminator and counts a loud fallback;
        newline-carrier ids are banned mid-value; HOLDING's first
        token is constrained to its candidate set."""

    @abc.abstractmethod
    def predict_with_value_candidates(
        self,
        batch: CollatedBatch[I],
        *,
        field: AuxField,
        generate: tuple[AuxField, ...],
        draws: int,
        sampling_for_draw: Callable[[int], ARSampling],
    ) -> tuple[NarratedPrediction, list[list[ValueCandidate]]]:
        """The subgoal-draws instrument: ONE prefill, then (a) the
        full narrated pass, op-identical to :meth:`predict_narrated`
        so the draws=0 limit stays bit-exact against it, and (b)
        ``draws + 1`` text-only decodes of ``field`` against the
        restored prefix cache — candidate 0 greedy (its per-step stats
        make the greedy candidate scorable), candidates 1..draws
        temperature-sampled under ``sampling_for_draw(draw)``.
        Free-text fields only (typed check); the greedy candidate's
        text must equal the full pass's parsed value — a mismatch is a
        broken instrument and exits loudly. Returns (full-pass
        prediction, per-row candidates)."""
```

Consumer patterns the lattice is designed for:

```python
def train_loop[I: BatchInputs](model: VLA[I], ...) -> None: ...
def nll_section[I: BatchInputs](model: ARVLA[I], ...) -> ...: ...
def heun_sweep[I: BatchInputs](model: FlowVLA[I], ...) -> ...: ...

# Narrowing happens ONCE, at the load boundary, and failure on an
# explicitly requested instrument is a SystemExit with the spec in the
# message — never a silent skip (the eval/policies.py L637 scar):
model = loading.load(args.checkpoint, device=..., dtype=...)
if args.generate and not isinstance(model, NarratingVLA):
    raise SystemExit(
        f"--generate requested but {model.spec.family.value} has no "
        "narration surface",
    )
```

## 4. Objectives — constructor payloads

The objective is a typed value: per family, a closed union of frozen
dataclasses (Rust enum-with-payload). Each variant carries exactly the
knobs that parameterize its term composition, so a knob without its
term is unrepresentable. Shared payloads live in
`models/objectives.py`; family-unique ones beside their family.

```python
# models/objectives.py
@dataclass(frozen=True, slots=True)
class FlowObjective:
    """Plain flow matching over the action chunk."""

    # unit variant — no knobs


@dataclass(frozen=True, slots=True)
class SnapflowObjective:
    """Self-distillation mix over a φ_s-extended flow decoder:
    α·mean(fm) + (1−α)·shortcut_weight·mean(shortcut). FM runs s=t
    (φ_s trained, not bypassed); the shortcut term regresses the
    one-step field at pure noise onto the model's own multi-step
    integration (stop-grad teacher). Admissible on any flow family
    whose decoder config has ``target_time_embed`` — the ctor
    validates and names the remedy (extend at --init-from; the φ_s
    MLP is zero-initialized, so extension is function-preserving)."""

    alpha: float  # FM share of the mix, in (0, 1)
    shortcut_weight: float  # the shortcut term's multiplier, > 0
```

```python
# models/molmoact2_joint.py
@dataclass(frozen=True, slots=True)
class JointObjective:
    """L = mean(flow) + ce_weight·mean(CE), optionally with the flow
    gradients stopped at the KV seam so the trunk learns only from CE
    (knowledge insulation)."""

    ce_weight: float
    insulate_flow: bool

    def __post_init__(self) -> None:
        if not self.ce_weight > 0:
            raise ValueError(
                f"ce_weight must be > 0, got {self.ce_weight} — a "
                "zero-weight CE term is the flow objective; construct "
                "MolmoAct2FlowVLA instead",
            )
```

Rules (D3, D4):

- **Why the constructor**: the objective determines construction-time
  invariants — which parameters can ever receive gradients (the ctor
  materializes objective → `requires_grad`), hence `param_groups()`'s
  offer, hence DDP's bucket set; whether the prefix cache is retained;
  which component keys exist (stable wandb series). A `forward`-time
  objective could disagree with all of those, and between ranks.
- **Objective vs optimizer**: LRs, weight decay, adamc, schedules are
  optimizer policy and never appear in payloads. train.py reconciles
  policy against the structural offer in both directions: an LR flag
  for an empty group errors; an offered group without an LR freezes
  loudly (today's opt-in convention).
- Objectives are recorded in `train_args` for provenance and
  **serialized into checkpoint metadata** (the loaded model
  reconstructs as-it-was-trained); they are never model state.
- `--objective`/weight flags parse into payloads at the TrainArgs
  boundary; under `--resume` the recorded objective is locked (flag
  refused), under `--init-from` a new objective may be declared
  (stage-2 flows), same ARCH-flag discipline as today.
- SnapFlow's α/λ_s stop being module constants (`SNAPFLOW_ALPHA`/
  `SNAPFLOW_LAMBDA`) and become recorded run parameters.

## 5. The families — `bijou/models/`

| Family class | Traits | Objective union | Building blocks |
|---|---|---|---|
| `GemmaFlowVLA` | `FlowVLA[GemmaInputs]` | `FlowObjective \| SnapflowObjective` | `Gemma4Model` + `GemmaEncoder` + `FlowDecoder` |
| `GemmaARVLA` | `ARVLA + NarratingVLA [GemmaInputs]` | `ARObjective(aux_loss_weight)` | Gemma + encoder + `GemmaARDecoder` |
| `Molmo2ARVLA` | `ARVLA + NarratingVLA [Molmo2Inputs]` | `ARObjective` | `Molmo2Model` + `Molmo2Encoder` + `Molmo2ARDecoder` |
| `MolmoAct2FlowVLA` | `FlowVLA[MolmoAct2Inputs]` | `FlowObjective` (+`SnapflowObjective` iff φ_s lands) | trunk + `MolmoAct2Encoder` + `MolmoFlowDecoder` |
| `MolmoAct2ARVLA` | `ARVLA[MolmoAct2Inputs]` | `ARObjective` (no aux — format 6) | trunk + encoder + `MolmoAct2ARDecoder` |
| `MolmoAct2JointVLA` | `ARVLA + FlowVLA [MolmoAct2Inputs]` | `JointObjective` | trunk + encoder + both decoders |

Classes = trained-surface sets (checkpoint facts loading enforces:
constructing a `NarratingVLA` family over a checkpoint with no trained
aux is refused). Objective = what receives gradients this run. The
three MolmoAct2 variants share assembly through free functions in
their modules — never a `MolmoAct2Base` class (styleguide).
Family-specific instruments that are not capabilities stay concrete
methods on their family: `GemmaFlowVLA.predict_flow_sde`
(`sde_noise_level`, Euler-only) and its `target_time` φ_s read;
`GemmaFlowVLA` under `SnapflowObjective` is where `distill` went.

### Worked example — `MolmoAct2JointVLA`

Sketch-fidelity (exact tensor plumbing lands in phase 4; the KV-before-
CE ordering, autocast ownership, and normalization formula are the
contract):

```python
class MolmoAct2JointVLA(
    ARVLA[MolmoAct2Inputs],
    FlowVLA[MolmoAct2Inputs],
):
    """MolmoAct2 trunk with both action decoders (see §4 for the
    objective payload). forward owns the joint precision policy: the
    CE branch runs inside bf16 autocast (a phase-1 step, verbatim);
    the flow branch runs fp32 outside it; the prompt-only KV is
    extracted for the flow decoder BEFORE the CE suffix extends the
    cache — ordering is a trained contract, not an implementation
    accident."""

    def __init__(
        self,
        backbone: Molmo2Model,
        encoder: MolmoAct2Encoder,
        flow_decoder: MolmoFlowDecoder,
        ar_decoder: MolmoAct2ARDecoder,
        *,
        objective: JointObjective,
        serving: FlowServing,  # recorded operating point (§7)
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.flow_decoder = flow_decoder
        self.ar_decoder = ar_decoder
        self.objective = objective
        self.serving = serving
        # objective → requires_grad materializes HERE (D3/D4): under
        # insulation the trunk still trains via CE, so backbone groups
        # stay offered; a hypothetical flow-only payload on a family
        # with a live trunk would empty them instead.

    @override
    def loss_counts(self, batch: CollatedBatch[MolmoAct2Inputs]) -> dict[str, Tensor]:
        return {
            "action": flow_element_count(batch),  # B·chunk·action_dim
            "aux": ce_target_count(self.ar_decoder, batch),
        }

    @override
    def forward(
        self,
        batch: CollatedBatch[MolmoAct2Inputs],
        *,
        counts: dict[str, Tensor],
    ) -> LossReport:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            memory = self.encoder.encode(
                self.backbone,
                batch.encoder_inputs,
                retain_cache=True,
            )
            # Prompt-only KV for the flow decoder, extracted BEFORE
            # the CE rider appends suffix K/V to the cache. Insulation
            # is a detach at exactly this seam.
            kv = extract_prompt_kv(memory)
            if self.objective.insulate_flow:
                kv = kv.detach()
            ce = ar_ce_loss_sums(self.ar_decoder, self.backbone, memory, batch)
        flow = flow_loss_sums(self.flow_decoder, kv, batch)  # fp32, outside
        world = dist.get_world_size() if dist.is_initialized() else 1
        # Per-rank scalar whose DDP MEAN is the global objective:
        # sum_r · W / global_count per term (D5).
        objective = (
            flow.sum * world / counts["action"]
            + self.objective.ce_weight * ce.sum * world / counts["aux"]
        )
        return LossReport(
            objective=objective,
            components={"action": flow, "aux": ce},
        )

    @override
    def predict(self, batch: CollatedBatch[MolmoAct2Inputs]) -> Tensor:
        # The deployment path is the flow decoder at the RECORDED
        # operating point (their serving semantics).
        return self.predict_flow(
            batch,
            num_steps=self.serving.num_steps,
            method=self.serving.method,
        ).actions

    # predict_flow / predict_ar / teacher_forced_block_logits delegate
    # to the composed decoders; param_groups returns the structural
    # offer {"decoder": flow_decoder trainables (+ prompt-side),
    # "backbone_text": …, "backbone_vision": …} — the AR decoder owns
    # zero parameters (trunk-native rows), so it contributes none.

    @override
    def checkpoint_components(self) -> dict[str, nn.Module]:
        return {"prompt": self.encoder, "flow_decoder": self.flow_decoder}
        # ar_decoder is parameterless → no section (explicitly, here).
```

### Adding family #7 — the recipe this design optimizes

1. `models/<family>.py`: the class (declare traits by inheritance),
   its objective union (or reuse shared), `from_checkpoint` + a fresh-
   construction builder, `checkpoint_components`.
2. One `VLAFamily` member + one registry entry in `loading.py`.
3. Reuse or add building blocks: an existing encoder/decoder if the
   trunk/decoder exists; else a new leaf module in `modelling/` with
   its loss kernel beside it.
4. A tiny-trunk fixture builder + recorded loss anchors (the
   `outputs/tiny-*` pattern), plus the family's collator/decode tests.
5. Nothing else: train/eval/rollout/GRPO light up through the traits;
   narrated eval sections appear iff the class is `NarratingVLA`;
   wandb components come from the objective's key set.

## 6. Package layout and DAG

```
bijou/
  vla.py                    # §3
  models/                   # §4–5: objectives.py + one file per family
  modelling/
    interface.py            # CollatedBatch, InputsCollator, SamplingMethod,
                            #   ARSampling, ActionCaptureStep, ValueCandidate,
                            #   per-trunk memory types (phase 7)
    nn.py, aux_text.py
    gemma4/  molmo2/        # trunks, moved wholesale
    encoders/               # gemma4.py, molmo2.py, molmoact2.py (+processing)
    decoders/               # flow.py, molmo_flow.py, ar_suffix.py,
                            #   ar_gemma.py, ar_molmo2.py, ar_molmoact2.py,
                            #   blocks.py, codecs.py
  fast/                     # tokenizer ARTIFACT layer only (fit CLI + math)
  data.py, annotations.py, judge/        # untouched
  loading.py                # registry + checkpoint toolkit (§7)
  train.py, eval/, rollout*.py, grpo_replay.py, train_grpo.py,
  convert_molmoact2.py, convert_legacy.py, async_save.py, testing.py
```

DAG (importing downward only):
`apps → loading → models → vla → modelling/{encoders,decoders} →
modelling/{gemma4,molmo2,interface,aux_text,nn} → fast`. Nothing in
`modelling` imports `vla`; no module above `models` names "all
decoders" or "all encoders".

## 7. Checkpoint format

```
checkpoint/
  metadata.json
  backbone.safetensors      # ALWAYS present; hard-linked when pristine
  prompt.safetensors        # per checkpoint_components(): encoder params
  flow_decoder.safetensors  # …and/or ar_decoder.safetensors
  optimizer.pt              # optional (kept for run-seeding checkpoints)
```

`metadata.json` (illustrative):

```json
{
  "schema_version": 1,
  "family": "gemma_flow",
  "spec": {"chunk_size": 50, "action_dim": 6},
  "backbone": {"id": "google/gemma-4-e2b", "trained": false},
  "objective": {"kind": "flow"},
  "serving": {"kind": "flow", "num_steps": 5, "method": "heun"},
  "components": {"prompt": {...config...}, "flow_decoder": {...}},
  "artifacts": {"fast_tokenizer": "mcobzarenco/fast_tokenizer_v2"},
  "stats": {"per_dataset": {...q01/q99 tables...}},
  "train_args": {...provenance, verbatim...}
}
```

Rules:

- **Self-contained, always** (working-together invariant, now without
  the disk cost): the backbone file is always present. When the trunk
  is pristine, the toolkit `os.link`s the resolved HF-cache blob
  (`os.link(realpath(snapshot_file), dst)`); `EXDEV`/cross-device
  falls back to a full copy with a printed note. A frozen-trunk run's
  every `step_NNNNNN` links the same inode.
- **Presence is not a signal.** `backbone.trained` is the explicit
  fact (the frozen⇒pristine inference trap dies).
- **The serving operating point is recorded**, written explicitly at
  save/convert time — no silent family defaults (D6).
- **Stats are metadata**, and stats replacement is a first-class
  operation: `bijou.convert_legacy --replace-stats <table>` (and the
  same op on `convert_molmoact2`) writes a corrected table with a
  provenance note — the molmo_flow baked-q01/q99 seam becomes one
  flag.
- **Objective payload serialized** as a tagged dict; loading
  reconstructs it for construction (eval-loaded models can compute
  their training loss).
- Transfer caveat (documented in working-together when phase 3 lands):
  local `rsync` needs `-H` to preserve hard links; without it the copy
  is correct but fully materialized. HF hub dedupes by content hash.

Loading becomes a registry plus a toolkit:

```python
FAMILIES: dict[VLAFamily, type[VLA[Any]]] = {
    VLAFamily.GEMMA_FLOW: GemmaFlowVLA,
    ...
}

def load(checkpoint: Path, *, device: DeviceLike, dtype: torch.dtype) -> VLA[Any]:
    meta = CheckpointMetadata.read(checkpoint / "metadata.json")
    return FAMILIES[meta.family].from_checkpoint(
        checkpoint, device=device, dtype=dtype,
    )

def save(model: VLA[Any], directory: Path, *, metadata: CheckpointMetadata) -> None:
    """The toolkit: atomic writes, component files from
    checkpoint_components(), the backbone hard-link rule, metadata
    validation, self-containment check before the directory is
    renamed into place."""
```

The train loop core (two-phase, D5), including the LR reconciliation:

```python
groups = model.param_groups()
for flag, group in (
    ("--decoder-lr", "decoder"),
    ("--backbone-text-lr", "backbone_text"),
    ("--backbone-vision-lr", "backbone_vision"),
):
    if lr(flag) > 0 and len(groups[group]) == 0:
        parser.error(
            f"{flag} given, but {group!r} receives no gradients "
            f"under {model.spec.family.value}'s objective"
        )

counts = model.loss_counts(batch)
if dist.is_initialized():
    for key in sorted(counts):
        dist.all_reduce(counts[key])
report = model(batch, counts=counts)  # DDP entry point
(report.objective / grad_accum).backward()
log({k: (v.sum.detach(), v.count) for k, v in report.components.items()})
```

(Chunked backward: forward per micro-slice with the same global
`counts`; objective addends sum to the full-batch objective — the
existing `test_chunked_backward`/`test_chunk_grad_allreduce` gates
adapt to assert exactly this.)

## 8. Naming ledger (D10) and the rename set

- **backbone/trunk**: the pretrained network (unchanged rule).
- **encoder**: the prompt side (unchanged).
- **decoder**: an action- or text-emitting component mounted on a
  trunk. Inside `modelling/gemma4` and `modelling/molmo2`, "decoder
  block/layer" keeps its standard transformer meaning — the scope
  boundary is the package.
- **head**: a shallow output projection only (`lm_head`, `fast_head`,
  `output_head_parameters`). Never a decoder-sized component.
- **expert**: retired from identifiers; allowed in prose.

| Old | New |
|---|---|
| `ARBackboneDecoder` | `GemmaARDecoder` |
| `ARBackboneConfig` | `ARDecoderConfig` |
| `ExpertConfig` | `FlowDecoderConfig` |
| `decoders/ar_backbone.py` | `modelling/decoders/ar_suffix.py` (scaffold) + `ar_gemma.py` (concrete) |
| `--expert-dtype` / `--expert-init` / `--insulate-expert` | `--flow-decoder-dtype` / `--flow-decoder-init` / `--insulate-flow` |
| `--decoder` (section declarator) | subsumed by `--family` + objective |
| decoder-kind metadata strings (`ar_backbone`, …) | `VLAFamily` values |
| `BijouModel`, `BijouPrediction`, `ObservationEncoder` (ABC), universal `ObservationMemory` | deleted (replaced per §3/§9 phase 7) |

Kept as-is (correct all along): `FlowDecoder`, `MolmoFlowDecoder`,
`Molmo2ARDecoder`, `MolmoAct2ARDecoder`, `ARSuffixDecoder`,
`--decoder-lr`, the `"decoder"` param group, `prompt.safetensors`.
wandb group/series names follow the ledger; naming breaks in charts
are acceptable (no compat).

## 9. Per-file disposition

Every file under `bijou/` (84 at time of writing), plus the repo-level
follow-ons. "move" = `git mv` + import rewrite, no logic change.

### Top level

| File | Disposition |
|---|---|
| `__init__.py` | keep; exports updated |
| `annotations.py` | unchanged (judge artifact contract) |
| `async_save.py` | unchanged; called by the checkpoint toolkit |
| `aux_text.py` | move → `modelling/aux_text.py` |
| `convert_molmoact2.py` | keep as CLI; emits the NEW format (family `molmoact2_flow`); gains `--replace-stats` |
| `data.py` | unchanged |
| `grpo_replay.py` | re-pointed: the stack becomes a thin adapter over `MolmoAct2ARVLA` (or direct trait use); replay math (`grammar_masks_from_bins`, `replay_logprobs`, sums/loss) unchanged; row-NPZ + loop-`.pt` formats frozen |
| `interface.py` | move → `modelling/interface.py`; loses `ObservationEncoder` ABC, universal `ObservationMemory`, `BijouPrediction` (phases 6–7); keeps `CollatedBatch`/`BatchInputs`/`InputsCollator`/`SamplingMethod`; gains `ARSampling`/`ActionCaptureStep`/`ValueCandidate` (phase 1) |
| `loading.py` | rewritten: `VLAFamily` registry + checkpoint toolkit; the old reader logic is frozen into `convert_legacy.py`, not kept live |
| `model.py` | DELETED (phase 6) — `BijouModel` dissolves into families |
| `nn.py` | move → `modelling/nn.py` |
| `rollout.py` | ported to `predict`/`predict_flow`/`predict_narrated` + capability narrowing; flag renames |
| `rollout_async.py` | ported (the policy thread calls trait methods); loop logic untouched |
| `rollout_safety.py` | unchanged |
| `testing.py` | fixture builders emit the new format; one tiny builder per family |
| `train.py` | slimmed: `TrainArgs` gains family+objective parsing; two-phase counts loop; LR-vs-offer reconciliation; `BijouTrainStep` autocast seams and adamc/eval-probe decoder matches move into families |
| `train_grpo.py` | re-pointed to `ARVLA`; wave formats frozen |

### `decoders/` → `modelling/decoders/`

| File | Disposition |
|---|---|
| `__init__.py` | move; re-export surface updated |
| `ar_backbone.py` | SPLIT: scaffold + losses → `ar_suffix.py`; Gemma concrete → `ar_gemma.py` (`GemmaARDecoder`) |
| `ar_molmo2.py` | move; dead `text_vocab_size` deleted |
| `ar_molmoact2.py` | move |
| `blocks.py` | move, unchanged |
| `flow.py` | move; snapflow constants → payload parameters (with `SnapflowObjective` CLI, phase 7); returns raw products in phase 7 |
| `molmo_flow.py` | move |
| — new: `codecs.py` | `ActionCodec` Protocol + `FastActionCodec` + `MolmoAct2ActionCodec`, moved from `fast/` (D12) |

### `encoders/` → `modelling/encoders/`

All four files move; the ABC parent drops in phase 7 (concrete
modules); `stream_geometries` retreats into `gemma4.py`;
`molmoact2_processing.py` unchanged.

### `eval/`

| File | Disposition |
|---|---|
| `__init__.py`, `__main__.py` | trivial updates |
| `cli.py` | ported: family-aware banners from `VLASpec`, capability narrowing with the loud-SystemExit rule |
| `leakage.py`, `metrics.py`, `plan.py`, `report.py`, `sharding.py`, `smolvla.py`, `subgoal_scoring.py` | unchanged (import paths only) |
| `molmo_norm.py` | absorbed: stats replacement is a conversion op (§7); file deleted |
| `policies.py` | `BijouPolicy` slims to trait consumption + narrowing; keeps its `--checkpoint` CLI surface (fontaine promise, §0) |
| `subgoal_swap.py` | ported to `FlowPrediction.noise` reuse + `NarratingVLA` draws |

### `fast/`

`__init__.py`, `__main__.py`, `cli.py`, `tokenizer.py` stay (artifact
layer). `codec.py` and `molmoact2.py` split: codec classes →
`modelling/decoders/codecs.py`; tokenizer halves stay (D12).

### `gemma4/` and `molmo2/`

All 23 files move wholesale to `modelling/gemma4/` and
`modelling/molmo2/`, internals unchanged (bench, cache, config,
generation/processor, loading, masks/goldens, model, testing, text,
tokenizer, verify_parity, vision, `__init__`s). Their parity gates and
goldens are untouched.

### `judge/`

All 13 files unchanged — the judge → data/annotations DAG never
touches the model stack.

### New files

`vla.py`; `models/{__init__,objectives,gemma_flow,gemma_ar,molmo2_ar,`
`molmoact2_flow,molmoact2_ar,molmoact2_joint}.py`; `convert_legacy.py`;
`modelling/{__init__,decoders/codecs}.py`.

### Repo-level follow-ons

- `sim/`: `rollout_sim.py` + `rollout_sim_parallel.py` re-pointed
  (consume policies); `collect_demos.py`, `scripted_expert.py`,
  `wrist_transform.py`, `so101_sim.py` unchanged (policy-free).
- `tests/`: `test_state_dict_keys.py`'s frozen-attribute gate is
  REPLACED by a checkpoint-schema test; `test_loading_schema.py`
  rewritten for the new metadata; the `ARCH_FLAGS`↔`CheckpointTrainArgs`
  sync test becomes family-schema tests; NEW `test_vla_parity.py`
  (phase 4, retired at phase 6 when `BijouModel` dies — the anchors
  remain as the standing gate); chunked-backward tests re-target the
  counts invariant; everything else re-points imports.
- `probes/`: `generate_tiny_molmoact2.py` emits the new format after
  phase 5; the parity probes re-point; `probe_grpo_replay_parity.py`
  re-runs as the phase-5 box gate.

## 10. Migration — phase by phase

Every phase is one or a few `main` commits, lands green, and names its
gate. A phase whose gate would move an oracle NUMBER has failed —
numbers never re-baseline in this plan; only CLI SHAPES re-record
(loudly), at phase 5.

**Phase 0 — anchor freeze (no code change).** Re-run all five loss
oracles at current HEAD (post-fontaine-rebase) and confirm the
recorded numbers reproduce; regenerate/verify both tiny fixtures.
Gate: five anchors bitwise at HEAD. This is the pre-registration the
whole migration is measured against.
VERDICT: PASS — all five anchors reproduced bitwise on the laptop
(gemma flow 2.7903/1.9152, gemma ar 27.8306/27.767, molmoact2 flow
1.3906/1.3305, ar 12.2254/12.3317, joint 13.616/13.6621 with the
cross-oracle exact); `check.py` 828 green at the freeze commit.

**Phase 1 — pure motion.** Create `modelling/`; move `gemma4/`,
`molmo2/`, `encoders/`, `decoders/`, `nn.py`, `aux_text.py`,
`interface.py` into it; the §8 renames; the `ar_backbone.py` file
split; the codec split out of `fast/`; instrument currencies into
`modelling/interface.py`. File moves, class/flag renames, and import
rewrites ONLY — no function bodies change (the deferred
`suffix_positions`/`continue_molmo2_suffix` extractions wait for
phase 7). Gate: `check.py` green; all five oracles bitwise via the
OLD CLI; decode fixtures green. The commit message claims
"motion-only" and the oracles prove it.
VERDICT: PASS — all five anchors bitwise post-move (same numbers as
phase 0), `check.py` 828 green. Two amendments discovered at
execution: (1) `codecs.py` lives at `modelling/codecs.py`, not
`modelling/decoders/codecs.py` — `interface.py` imports `ActionCodec`
(collators carry codecs), so a decoders/ home would invert the DAG;
(2) `loading.py` already owned the name `FlowDecoderConfig` for its
checkpoint-schema record — schema records now carry the `Section`
suffix (`FlowDecoderSection`; `MolmoFlowDecoderConfig` follows at its
next touch), a transitional naming the new format retires in phases
5–6. `AnyFloatArray` moved to `fast/tokenizer.py` (both layers need
it; the artifact layer is the lower one).

**Phase 2 — trait scaffolding (additive).** `vla.py`,
`models/objectives.py`, `VLAFamily`; unit tests for the currencies and
payload validation. Nothing consumes them yet. Gate: `check.py`.
VERDICT: PASS — `check.py` 838 green (10 new tests:
`tests/test_vla_types.py` — payload validation, frozen currencies,
trait abstractness/statelessness, a stub family proving narrowing +
forward/backward through the two-phase counts protocol).

**Phase 3 — checkpoint schema + toolkit + converter.**
`CheckpointMetadata`, the save/load toolkit (atomic writes, hard-link
rule with `EXDEV` fallback, self-containment check),
`convert_legacy.py` embedding a frozen copy of the old reader,
`--replace-stats`. Gate: convert both tiny fixtures → metadata
validates → conversion is idempotent (re-convert = byte-identical
directory); hard-link semantics tested (link when same-fs, loud copy
fallback); `check.py`.
VERDICT: PASS — `check.py` 851 green (13 new in
`tests/test_vla_checkpoint.py`); real-artifact smoke: the phase-0
gemma-flow and molmoact2-joint tiny oracle checkpoints both convert
and validate (pristine trunk → hard-linked `backbone/` mirror,
nlink=2; trained trunk → `backbone.safetensors` linked; joint family
inferred from the recorded objective). Amendments at execution: the
toolkit lives in **`bijou/checkpoint.py`** (its own module; loading.py
keeps only the family registry when phase 5 lands — owner call);
pristine trunks are SHARDED SNAPSHOT DIRECTORIES, so the pristine form
is a hard-linked `backbone/` mirror of the whole snapshot (loadable as
a local model dir, tokenizer/processor files included — true
self-containment), not a single file; `components` records
`{"config", "weights"}` per entry so parameterless decoders declare
themselves explicitly instead of by file absence; component configs
are carried VERBATIM as the legacy tagged section dicts (families
parse them with the same section machinery — conversion cannot drift
architecture); `convert_legacy` imports the LIVE legacy reader until
phase 6 freezes a copy in; formats 1/2 are refused (nothing on the
inventory is older than format 3).

**Phase 4 — families + registry + the parity suite (dual world).**
The six family classes composing the phase-1 building blocks verbatim
(loss paths delegate to the existing kernels in today's op order);
`from_checkpoint` reads the NEW format; `loading` registry.
`tests/test_vla_parity.py`: for each anchor — old path (`BijouModel`
from the old-format fixture) vs new path (family from the CONVERTED
fixture) — 2-step losses bitwise equal, `predict`/decode outputs
bitwise equal, decode-anchor fixtures pass through the family,
`param_groups` name→param-set identical, instrument parity
(`teacher_forced_block_logits`, value candidates, capture steps
bit-equal). Gate: the parity suite, plus the old-path oracles still
green, plus `check.py`.

**Phase 5 — apps port + CLI reshape.** `train.py` (family/objective
parsing, two-phase loop, LR reconciliation), `eval/`, `rollout`(+
async), `grpo_replay`/`train_grpo`, `sim/` re-points, flag renames,
`--family`. Gate: the five oracles re-anchored through the NEW CLI on
CONVERTED fixtures — same numbers bitwise, new command shapes
recorded in architecture.md's regression-gates section; eval tiny
smoke (a 4-frame report end-to-end); box gate on fontaine's machine:
GRPO masks bit-equal + logprobs within the registered 1e-4
cross-decomposition bound on the banked R1-A/R1-B waves;
`probe_grpo_replay_parity` PASS; fontaine boundary post (§0).

**Phase 6 — delete the old world.** `model.py`, the live old-format
read path in `loading`, `BijouPrediction`, the phase-4 parity bridges
and `test_vla_parity.py` (the anchors remain as the standing gate).
Gate: `check.py`; grep gates — zero references to `BijouModel`,
`BijouPrediction`, `ObservationEncoder`, decoder-kind strings; the
five oracles green (new CLI).

**Phase 7 — seam dissolutions**, each sub-step its own commit, each
oracle-gated: (a) encoder ABC removal; (b) per-trunk memory types —
the `isinstance(cache, Molmo2KVCache)` guards become parameter types;
(c) decoders return raw products, families wrap into prediction
structs; (d) the `suffix_positions` + `continue_molmo2_suffix`
extractions (three copies → one); (e) snapflow constants →
`SnapflowObjective` fields + CLI. Gate per sub-step: `check.py` + the
five oracles bitwise + decode fixtures.

**Phase 8 — conversion campaign + adoption.** Convert the real
checkpoint inventory, verify each (recorded eval/loss numbers where
they exist, else load + smoke predict + spec check), re-upload to HF
hub (uploads before deletions; optimizer kept only for run-seeding
checkpoints), fontaine boundary for his artifacts + launcher re-pin.
Inventory (finalized with owner + fontaine at execution): gemma AR
mainline 100k; flow 80k + the restarted 80k; the rig fine-tunes
(`bijou_arb_rcond_ft_rig_4k_ddp4`/10k, `bijou_ft_rig_ar_armL`);
converted molmoact2 (release, rig-r1 step2000, release_rigtable);
`gate_d_lite` step2000; er-60k; the stage-C grasp-SFT endpoints; both
tiny fixtures. Gate: per-checkpoint parity receipts posted; the
deletion inventory convention for anything pruned.

### Gate matrix

| Phase | check.py | oracles (old CLI) | parity suite | oracles (new CLI) | decode fixtures | box gates | grep gates |
|---|---|---|---|---|---|---|---|
| 0 | ✓ | ✓ (freeze) | — | — | ✓ | — | — |
| 1 | ✓ | ✓ bitwise | — | — | ✓ | — | — |
| 2 | ✓ | unchanged | — | — | — | — | — |
| 3 | ✓ | unchanged | converter gates | — | — | — | — |
| 4 | ✓ | ✓ (old path) | ✓ bitwise | — | ✓ via families | — | — |
| 5 | ✓ | retired | ✓ | ✓ bitwise, shapes re-recorded | ✓ | GRPO waves | — |
| 6 | ✓ | — | retired | ✓ | ✓ | — | ✓ |
| 7 | ✓ | — | — | ✓ per sub-step | ✓ | — | — |
| 8 | ✓ | — | — | ✓ | — | per-checkpoint receipts | — |

## 11. Oracles and parity — the exact truth set

The five loss anchors (2-step tiny-fixture reproductions through
`bijou.train`; CPU-deterministic; exact CLI shapes live in
architecture.md's regression-gates section and are re-recorded there
at phase 5):

| Oracle | step-1 / step-2 loss |
|---|---|
| gemma flow | 2.7903 / 1.9152 |
| gemma ar | 27.8306 / 27.767 |
| molmoact2 flow | 1.3906 / 1.3305 |
| molmoact2 ar | 12.2254 / 12.3317 |
| molmoact2 joint (KI, λ=1) | 13.6160 / 13.6621 — `loss_action` ≡ the flow anchors and `loss_aux` ≡ the ar anchors, bitwise |

Fixtures: `outputs/tiny-gemma4`
(`python -m bijou.modelling.gemma4.testing --output outputs/tiny-gemma4`; the
module path gains a `modelling.` segment at phase 1) and
`outputs/tiny-molmoact2`
(`PYTHONPATH=. uv run python probes/generate_tiny_molmoact2.py`).
Regenerating a fixture re-baselines its oracles — loudly, in the
ledger; a regeneration inside this migration would invalidate the
parity chain and is forbidden until phase 8 closes.

Secondary truth set:

- `tests/fixtures/molmoact2_discrete/decode_anchors.npz` — masked
  ids/bins/actions byte-equal, logprobs at the recorded tolerance.
- molmo_flow port fixtures — `allclose` at the registered ~1e-6 abs
  bound (the cross-box ULP class; bitwise only same-box).
- The mcselect candidate-0 ≡ full-pass tripwire and the value-budget
  fallback counters (loud instruments, not tolerances).
- GRPO replay: masks bit-equal on banked waves; logprobs ≤ the
  registered 1e-4 cross-decomposition bound (box gate, phase 5).

Tolerance discipline (standing lessons, restated because this
migration will be tempted to violate them): "bitwise" claims hold
only same-box, same-dtype, same batch composition; 1e-5-class bounds
apply only between IDENTICAL forward decompositions; a monolithic vs
prefill+continuation split drifts ~5e-5 in fp32 and that is a
DIFFERENT decomposition, not a bug — but phases 1–6 never change the
decomposition, so their claims stay bitwise.

## 12. Out of scope, deliberately

- Narration on Molmo trunks (§8.13 step 6): the `NarratingVLA` slot
  exists; nothing here builds it.
- A Gemma joint family: only if an experiment demands one (trigger
  recorded in architecture.md §2.2a).
- molmo_flow φ_s extension / distillation: designed for (§4), not
  built.
- Multi-turn training, new objectives, any training-semantics change:
  this plan is a refactor — the models it produces are numerically
  the models we already have, and every gate exists to prove exactly
  that.