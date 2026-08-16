"""Step and probe machinery of the training loop.

The step side: :class:`DevicePrefetcher` (one-batch-lookahead
host->device transfer), :class:`ChunkedBatch`/:class:`ChunkingCollator`
(collate-time splitting for ``--backward-chunks`` gradient
accumulation), :func:`summed_loss_counts` (the full step's
per-component counts, summed over the chunk micro-batches BEFORE any
forward, so every chunk normalizes by the SAME totals -- the
chunked-backward exactness contract), and the no-DDP-wrapper
collectives (:func:`broadcast_module_states`,
:func:`allreduce_gradients`).

The probe side: seeded, rank-sharded MAE probe sets
(:class:`ProbeSet`, :func:`build_probe_set`) and :func:`validate` --
sampled-chunk MAE in raw action units through the capability traits,
plus the wandb rich tables and the conditioning-sensitivity tripwires.
"""

from __future__ import annotations

import dataclasses
import itertools
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import torch
import wandb
from torch import Tensor

from ..annotations import ConditionField
from ..modelling.aux_text import AuxField, AuxGeneration, aux_label_text
from ..modelling.interface import (
    BatchInputs,
    CollatedBatch,
    Collator,
    SamplingMethod,
)
from ..vla import ARVLA, VLA, FlowVLA, NarratingVLA

# Rows in the wandb probe tables (each costs camera images + a
# matplotlib figure per eval — TWICE for aux runs, the fast-path table
# and the all-fields table): a spot check, deliberately small — 32 rows
# of figures were a measured ~34s/eval rank-0 straggler (2026-08-03).
EVAL_TABLE_ROWS = 12


class DevicePrefetcher[I: BatchInputs]:
    """One-batch-lookahead host->device transfer on a side CUDA stream.

    Centralizes every H2D copy of the training pipeline: the loop receives
    batches already device-resident, and the copy of batch N+1 overlaps the
    compute of batch N (dataloader workers cannot produce CUDA tensors —
    each would need its own CUDA context — so this is the closest torch gets
    to "the loader hands you device batches"). Degrades to plain synchronous
    transfers on non-CUDA devices.
    """

    def __init__(
        self,
        loader: Iterable[CollatedBatch[I] | ChunkedBatch[I]],
        device: torch.device,
    ) -> None:
        self.loader = loader
        self.device = device

    def __iter__(self) -> Iterator[CollatedBatch[I] | ChunkedBatch[I]]:
        if self.device.type != "cuda":
            for batch in self.loader:
                yield batch.to(self.device)
            return

        stream = torch.cuda.Stream(self.device)
        compute_stream = torch.cuda.current_stream(self.device)
        batches = iter(self.loader)

        def preload() -> CollatedBatch[I] | ChunkedBatch[I] | None:
            cpu_batch = next(batches, None)
            if cpu_batch is None:
                return None
            with torch.cuda.stream(stream):
                return cpu_batch.to(self.device, non_blocking=True)

        batch = preload()
        while batch is not None:
            compute_stream.wait_stream(stream)
            # The tensors were allocated on the side stream; tell the caching
            # allocator they are consumed on the compute stream.
            for tensor in batch.all_tensors():
                tensor.record_stream(compute_stream)
            yield batch  # consumer enqueues the step's compute, then returns
            batch = preload()  # blocks on workers while the GPU crunches


@dataclass(frozen=True, slots=True)
class ChunkedBatch[I: BatchInputs]:
    """One optimizer step's samples, collated as ``--backward-chunks``
    equal sub-batches for chunked backward. Splitting happens at COLLATE
    time (each chunk pads to its own max length — position ids are
    padding-mask cumsums, so padding width is inert to the math), which
    also shrinks the per-forward activation footprint below a sliced
    full-width batch. Duck-types CollatedBatch's transfer surface for
    the DataLoader pin hook and DevicePrefetcher."""

    chunks: tuple[CollatedBatch[I], ...]

    def all_tensors(self) -> list[Tensor]:
        return [t for chunk in self.chunks for t in chunk.all_tensors()]

    def pin_memory(self) -> ChunkedBatch[I]:
        return ChunkedBatch(tuple(c.pin_memory() for c in self.chunks))

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> ChunkedBatch[I]:
        return ChunkedBatch(
            tuple(c.to(device, non_blocking=non_blocking) for c in self.chunks),
        )


@dataclass
class ChunkingCollator[I: BatchInputs]:
    """Train-loader collate for chunked backward: split the step's item
    list into ``chunks`` contiguous, size-balanced sub-lists (equal by
    construction — --backward-chunks divides --batch-size and the train
    loaders drop last; a short straggler batch would still split
    near-evenly and stay exact, since normalization is by global
    counts) and collate each separately. Sample composition per step is
    identical to the unchunked loader's; the probe collator is never
    wrapped."""

    collator: Collator[I]
    chunks: int

    def __call__(self, samples: list[Any]) -> ChunkedBatch[I]:
        bounds = [(len(samples) * i) // self.chunks for i in range(self.chunks + 1)]
        return ChunkedBatch(
            tuple(
                self.collator(samples[start:stop])
                for start, stop in itertools.pairwise(bounds)
                if stop > start
            ),
        )


def broadcast_module_states(module: torch.nn.Module) -> None:
    """The one-time rank-0 state broadcast DDP's constructor performs,
    without the reducer DDP would also build — its bucket buffers are a
    full fp32 gradient copy allocated AT CONSTRUCTION, not at first
    sync (the measured 13.6 GiB block from the molmo2 smoke-ladder
    snapshot). Params and buffers both; state_dict values
    are live views, so in-place broadcast lands in the module."""
    for tensor in module.state_dict().values():
        if isinstance(tensor, torch.Tensor):
            torch.distributed.broadcast(tensor, src=0)


def allreduce_gradients(parameters: list[torch.nn.Parameter]) -> None:
    """The ``--chunk-grad-allreduce`` gradient sync: one explicit
    in-place allreduce of every accumulated ``param.grad`` (sum, then
    divide by world — DDP's averaging semantics, differing only in fp
    reduction order). Runs after the full chunk loop, so no reducer
    bucket buffers ever coexist with the accumulated gradients. Every
    parameter handed in must carry a gradient (the same trainable-
    partition contract DDP's static bucketing relies on); a missing one
    dies loudly rather than letting replicas desynchronize."""
    grads: list[torch.Tensor] = []
    for p in parameters:
        if p.grad is None:
            raise RuntimeError(
                "chunk-grad-allreduce: a trainable parameter has no "
                f"gradient after the chunk loop (shape {tuple(p.shape)}) "
                "— the every-parameter-gets-gradients contract is broken",
            )
        grads.append(p.grad)
    handles = [torch.distributed.all_reduce(g, async_op=True) for g in grads]
    for handle in handles:
        assert handle is not None  # async_op=True always returns a Work
        handle.wait()
    world = float(torch.distributed.get_world_size())
    for g in grads:
        g.div_(world)


def summed_loss_counts(
    model: VLA[Any],
    chunks: Sequence[CollatedBatch[Any]],
) -> dict[str, Tensor]:
    """The full step's per-component counts, summed over the chunk
    micro-batches BEFORE any forward (data-only): each chunk's forward
    then divides by the SAME global normalizers, so the objective
    addends sum to the full-batch objective — the chunked-backward
    exactness contract (``VLA.forward``'s docstring)."""
    per_chunk = [model.loss_counts(chunk) for chunk in chunks]
    keys = list(per_chunk[0])
    for counts in per_chunk[1:]:
        if list(counts) != keys:
            raise SystemExit(
                f"loss_counts key set moved across chunks ({keys} vs "
                f"{list(counts)}) — component keys are run-constant by "
                "contract",
            )
    return {
        key: torch.stack([counts[key] for counts in per_chunk]).sum() for key in keys
    }


def _chunk_plot(
    predicted: Tensor,
    truth: Tensor,
    valid: Tensor,
    state: Tensor,
    action_names: list[str],
) -> Any:
    """Per-joint curves over the action chunk: ground truth, the model's
    prediction, and the trivial state-copy baseline (hold current joint
    positions — the minimum bar a learned policy must clear). Returns a
    matplotlib figure (caller logs and closes it).

    Shapes: predicted/truth [chunk, action_dim]; valid [chunk] (bool);
    state [state_dim] — one sample, CPU-resident."""
    dims = predicted.shape[-1]
    ncols = 3
    nrows = (dims + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 2.5 * nrows),
        squeeze=False,
    )
    steps = range(predicted.shape[0])
    n_valid = int(valid.sum())
    for dim in range(dims):
        ax = axes[dim // ncols][dim % ncols]
        ax.plot(
            steps[:n_valid],
            truth[:n_valid, dim].tolist(),
            label="truth",
            color="black",
            linewidth=1.8,
        )
        ax.plot(
            steps[:n_valid],
            [float(state[dim])] * n_valid,
            label="state-copy",
            color="tab:blue",
            linestyle="--",
            linewidth=1.2,
        )
        ax.plot(
            steps[:n_valid],
            predicted[:n_valid, dim].tolist(),
            label="predicted",
            color="tab:orange",
            linestyle="--",
            linewidth=1.2,
        )
        name = action_names[dim] if dim < len(action_names) else f"dim {dim}"
        ax.set_title(name, fontsize=9)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    return fig


@dataclass(frozen=True, slots=True)
class RichRow:
    """One probe sample's prediction, kept (CPU-side) for the wandb table.
    ``noise`` is the flow draw the prediction integrated (None for AR) —
    the Q3 tripwire re-decodes these rows and must reuse it."""

    sampled: Tensor
    truth: Tensor
    valid: Tensor
    state: Tensor
    noise: Tensor | None = None


@dataclass(frozen=True, slots=True)
class ProbeSet[I: BatchInputs]:
    """This rank's shard of a seeded MAE probe, CPU-resident between evals.

    ``total`` is the global probe size across ranks. ``rich_items`` are raw
    items kept for wandb rich logging at ``rich_positions`` (positions in
    this shard's streaming order, strided across the shard so the table
    spans the concatenated datasets instead of the earliest ones).
    """

    total: int
    batches: list[CollatedBatch[I]]
    rich_items: list[dict[str, Any]]
    rich_positions: tuple[int, ...]
    # Per shard item (streaming order): the episode's hindsight outcome
    # label (None = unlabeled) — the Q2 slicing key: per-outcome MAE
    # buckets, with the success slice as the open-loop deployment proxy
    # and the unlabeled slice as the continuity anchor.
    outcomes: tuple[str | None, ...]
    # Per-dataset slicing (--eval-dataset-breakdown, owner work order
    # 2026-08-16): each shard item's repo id, and the GLOBAL bucket
    # order — derived from the concat dataset, identical on every rank
    # so the fixed-size slice tensor stays collective-aligned.
    repo_ids: tuple[str, ...] = ()
    repo_buckets: tuple[str, ...] = ()


OUTCOME_BUCKETS = ("success", "partial", "failure", "unlabeled")


def build_probe_set[I: BatchInputs](
    dataset: torch.utils.data.ConcatDataset[dict[str, Any]],
    collator: Collator[I],
    num_samples: int,
    seed: int,
    rank: int,
    world_size: int,
    batch_size: int,
    *,
    keep_rich: bool,
) -> ProbeSet[I]:
    """Draw, fetch and collate one probe set's shard for this rank.

    The frame draw is exactly bijou.eval's: seeded sampling without
    replacement over the same selection scores the same frames. The sorted
    draw is striped round-robin across ranks, so every shard spreads evenly
    over the concatenated datasets.
    """
    num = min(num_samples, len(dataset))
    indices = sorted(random.Random(seed).sample(range(len(dataset)), num))
    shard = indices[rank::world_size]
    items = [dataset[i] for i in shard]
    batches = [
        collator(items[i : i + batch_size]) for i in range(0, len(items), batch_size)
    ]
    rich_positions: tuple[int, ...] = ()
    rich_items: list[dict[str, Any]] = []
    if keep_rich and items:
        stride = max(len(items) // EVAL_TABLE_ROWS, 1)
        rich_positions = tuple(range(0, len(items), stride))[:EVAL_TABLE_ROWS]
        rich_items = [items[p] for p in rich_positions]
    # Bucket order must be rank-identical: derived from the concat
    # dataset's members (sorted, deduped across --dataset-repeat
    # replicas), never from this rank's item stream. Members without a
    # repo id (synthetic test datasets) contribute no bucket.
    member_ids = {
        str(repo_id)
        for member in getattr(dataset, "datasets", ())
        for repo_id in (getattr(getattr(member, "dataset", None), "repo_id", None),)
        if repo_id is not None
    }
    return ProbeSet(
        total=num,
        batches=batches,
        rich_items=rich_items,
        rich_positions=rich_positions,
        outcomes=tuple(item.get("condition_outcome") for item in items),
        repo_ids=tuple(str(item.get("repo_id", "")) for item in items),
        repo_buckets=tuple(sorted(member_ids)),
    )


def holding_accuracy(
    generations: list[AuxGeneration],
    items: list[dict[str, Any]],
) -> float | None:
    """Generated-vs-label holding accuracy over the labeled table rows —
    read straight off the table decode (the request-conditioned format
    elicits every requested field, so a separate likelihood probe is no
    longer needed: the constrained value in the MAIN decode is the
    measurement, in exactly the training context). None when no row is
    labeled."""
    correct = 0
    labeled = 0
    for generation, item in zip(generations, items, strict=True):
        value = item.get("annotation.holding")
        if value is None or not bool(torch.isfinite(value)):
            continue
        if generation.holding is None:
            continue
        correct += int(generation.holding == bool(int(value)))
        labeled += 1
    if labeled == 0:
        return None
    return correct / labeled


def probe_prediction(
    model: VLA[Any],
    batch: CollatedBatch[Any],
    *,
    generator: torch.Generator,
    flow_probe_method: SamplingMethod | None,
    noise: Tensor | None = None,
) -> tuple[Tensor, Tensor | None]:
    """The probe's (actions, noise | None) through the capability
    traits: flow families integrate 10 steps at the family's recorded
    serving METHOD (eval is a measurement — integration error well
    below model error; 0.018 vs 0.05 mean deviation at the 5-step
    deployment default), AR families decode greedily. The joint family
    is both — the flow read is its deployment path and wins."""
    if isinstance(model, FlowVLA):
        assert flow_probe_method is not None  # supplied for every flow family
        prediction = model.predict_flow(
            batch,
            num_steps=10,
            method=flow_probe_method,
            noise=noise,
            generator=generator,
        )
        return prediction.actions, prediction.noise
    assert isinstance(model, ARVLA)  # every family carries one of the two
    return model.predict_ar(batch).actions, None


@torch.no_grad()
def validate(
    model: VLA[Any],
    probe: ProbeSet[Any],
    device: torch.device,
    seed: int,
    *,
    distributed: bool = False,
    wandb_run: Any = None,
    collator: Collator[Any] | None = None,
    action_names: list[str] | None = None,
    step: int = 0,
    table_key: str = "eval/samples",
    aux_fields: tuple[AuxField, ...] = (),
    flow_probe_method: SamplingMethod | None = None,
    dataset_breakdown: bool = False,
) -> float:
    """Sampled-chunk MAE in raw action units over this rank's shard of the
    probe set; with ``distributed`` the sums all-reduce to the global value
    (collective — every rank must call this at the same step). Batches
    arrive CPU-resident and visit the device one at a time, and the
    observation memory is re-encoded per eval, so probe size costs host
    RAM, not GPU memory. The valid-element-weighted aggregation is exactly
    bijou.eval's chunk_mae. Normalization is per dataset (each sample's
    own stats, matching training). With a wandb run and a probe carrying
    rich items, also logs a table under ``table_key``: camera images,
    task, state, per-joint predicted-vs-truth plots. ``aux_fields`` are
    the run's TRAINED aux fields — non-empty only for a
    :class:`~bijou.vla.NarratingVLA` (the narrated side-channel table)."""
    totals = torch.zeros(2, device=device)  # [abs-error sum, valid elements]
    rich_rows: list[RichRow] = []
    slice_totals = torch.zeros(len(OUTCOME_BUCKETS), 2, device=device)
    # --eval-dataset-breakdown: [abs-error sum, valid elements, samples]
    # per repo bucket — fixed [D, 3] tensor, collective-aligned (the
    # bucket order is rank-identical by construction).
    dataset_breakdown = dataset_breakdown and len(probe.repo_buckets) > 0
    dataset_totals = torch.zeros(max(len(probe.repo_buckets), 1), 3, device=device)
    wanted = iter(probe.rich_positions)
    next_rich = next(wanted, None)
    base = 0
    generator = torch.Generator(device=device).manual_seed(seed)
    for cpu_batch in probe.batches:
        batch = cpu_batch.to(device)
        # The probe scores the deployment surface through the traits:
        # AR-suffix probes run the ACT fast path here (comparable across
        # aux-on / aux-off arms); the rich table below is the FREE-mode
        # surface for narrating checkpoints.
        sampled, sampled_noise = probe_prediction(
            model,
            batch,
            generator=generator,
            flow_probe_method=flow_probe_method,
        )
        # The molmo_flow decoder returns CPU fp32 actions (its
        # reference-parity unnormalize runs on host, §8.13) — score on
        # the probe's device; no-op for decoders that stay on device.
        # Measured crash: joint family, first in-train eval, all ranks
        # (cuda vs cpu at the subtraction) 2026-08-16 18:08Z.
        sampled = sampled.to(device)
        truth = batch.actions.float()
        valid = ~batch.action_is_pad
        error = (sampled - truth).abs()
        totals[0] += error[valid].sum()
        totals[1] += valid.sum() * error.shape[-1]
        # Q2 slices: per-frame error bucketed by the episode's hindsight
        # outcome (probes condition on TRUE labels, so the success slice
        # is the open-loop deployment proxy and the unlabeled slice the
        # continuity anchor). Fixed [4, 2] tensor — collective-aligned.
        frame_error = (error * valid[..., None]).sum(dim=(1, 2))
        frame_count = valid.sum(dim=1) * error.shape[-1]
        for i in range(sampled.shape[0]):
            outcome = probe.outcomes[base + i]
            bucket = (
                OUTCOME_BUCKETS.index(outcome)
                if outcome in OUTCOME_BUCKETS
                else OUTCOME_BUCKETS.index("unlabeled")
            )
            slice_totals[bucket, 0] += frame_error[i]
            slice_totals[bucket, 1] += frame_count[i]
            if dataset_breakdown:
                repo = probe.repo_ids[base + i]
                if repo in probe.repo_buckets:
                    d = probe.repo_buckets.index(repo)
                    dataset_totals[d, 0] += frame_error[i]
                    dataset_totals[d, 1] += frame_count[i]
                    dataset_totals[d, 2] += 1
        # Batches stream in shard order: pick off the rich positions
        # (matching probe.rich_items one-to-one) as they pass.
        while next_rich is not None and next_rich < base + sampled.shape[0]:
            i = next_rich - base
            rich_rows.append(
                RichRow(
                    sampled=sampled[i].cpu(),
                    truth=truth[i].cpu(),
                    valid=valid[i].cpu(),
                    state=batch.state[i].cpu(),
                    noise=(
                        sampled_noise[i].cpu() if sampled_noise is not None else None
                    ),
                ),
            )
            next_rich = next(wanted, None)
        base += sampled.shape[0]
    if distributed:
        torch.distributed.all_reduce(totals)
        torch.distributed.all_reduce(slice_totals)
        if dataset_breakdown:
            torch.distributed.all_reduce(dataset_totals)
    mae = float(totals[0] / totals[1].clamp(min=1))
    metric_prefix = table_key.split("/")[0]
    if wandb_run is not None:
        # Labeled buckets only (all-unlabeled probes log nothing extra);
        # skip the redundant all-in-one-bucket case.
        labeled = {
            bucket: float(slice_totals[i, 0] / slice_totals[i, 1])
            for i, bucket in enumerate(OUTCOME_BUCKETS)
            if float(slice_totals[i, 1]) > 0
        }
        if len(labeled) > 1:
            wandb_run.log(
                {
                    f"{metric_prefix}/chunk_mae_{bucket}": value
                    for bucket, value in labeled.items()
                },
                step=step,
            )
        if dataset_breakdown:
            # Per-dataset MAE scalars (a wandb line per repo) + one
            # counts table: how many of the probe's samples came from
            # each dataset this eval (global, after the all-reduce).
            per_dataset = {
                f"{metric_prefix}/chunk_mae_dataset/{repo}": float(
                    dataset_totals[d, 0] / dataset_totals[d, 1].clamp(min=1),
                )
                for d, repo in enumerate(probe.repo_buckets)
                if float(dataset_totals[d, 2]) > 0
            }
            counts = wandb.Table(columns=["dataset", "eval_samples", "chunk_mae"])
            for d, repo in enumerate(probe.repo_buckets):
                counts.add_data(
                    repo,
                    int(dataset_totals[d, 2]),
                    float(dataset_totals[d, 0] / dataset_totals[d, 1].clamp(min=1)),
                )
            wandb_run.log(
                {**per_dataset, f"{metric_prefix}/dataset_counts": counts},
                step=step,
            )

    if wandb_run is not None and probe.rich_items and collator is not None:
        # Two surfaces over the same rich rows (rank-0-only, no
        # collectives, bounded to EVAL_TABLE_ROWS):
        #   {table_key} — chunk columns straight off the scalar pass
        #     (fast path: prompt says [generate|actions], the suffix
        #     carries NO aux text — actions condition on the user
        #     message only; chunk_mae matches the logged scalar's
        #     measurement condition). The aux_generated/aux_label
        #     columns are a SIDE-CHANNEL from the all-fields decode of
        #     the same items — what the model says for this observation
        #     next to the fast-path chunk, deliberately mixed
        #     conditions, labeled here so nobody rediscovers it as a
        #     bug (owner-requested pairing, 2026-08-03).
        generations: list[AuxGeneration] | None = None
        rich_actions: Tensor | None = None
        if isinstance(model, NarratingVLA) and len(aux_fields) > 0:
            table_collator = dataclasses.replace(
                collator,
                generate_override=aux_fields,
            )
            rich_batch = table_collator(probe.rich_items).to(device)
            narrated = model.predict_narrated(rich_batch, generate=aux_fields)
            generations = narrated.generations
            rich_actions = narrated.actions.cpu()

        # Cameras vary per sample across mixed datasets: generic positional
        # columns, padded with None where a sample has fewer cameras.
        per_item_cameras = [collator.cameras_of(item) for item in probe.rich_items]
        n_slots = max(len(cams) for cams in per_item_cameras)

        def row_images(item: dict[str, Any], cams: list[str]) -> list[Any]:
            images: list[Any] = [
                wandb.Image(
                    (item[camera].clamp(0, 1) * 255)
                    .to(torch.uint8)
                    .permute(1, 2, 0)
                    .numpy(),
                    caption=camera.removeprefix("observation.images."),
                )
                for camera in cams
            ]
            return images + [None] * (n_slots - len(cams))

        def state_str(item: dict[str, Any]) -> str:
            return ", ".join(f"{x:.1f}" for x in item["observation.state"].tolist())

        table = wandb.Table(
            columns=[
                "sample",
                *(f"camera_{i}" for i in range(n_slots)),
                "task",
                "state",
                "chunk_mae",
                "pred_vs_truth",
                *(["aux_generated", "aux_label"] if generations is not None else []),
            ],
        )
        for i, (item, row) in enumerate(zip(probe.rich_items, rich_rows, strict=True)):
            figure = _chunk_plot(
                row.sampled,
                row.truth,
                row.valid,
                row.state,
                action_names or [],
            )
            aux_columns: tuple[str, ...] = ()
            if generations is not None:
                aux_columns = (
                    generations[i].text,
                    aux_label_text(item, aux_fields),
                )
            table.add_data(
                i,
                *row_images(item, per_item_cameras[i]),
                str(item["task"]),
                state_str(item),
                float((row.sampled - row.truth).abs()[row.valid].mean()),
                wandb.Image(figure),
                *aux_columns,
            )
            plt.close(figure)
        wandb_run.log({table_key: table}, step=step)

        if generations is not None and rich_actions is not None:
            # Paired signal, free (the decode already ran for the aux
            # columns): masked MAE of the chunks that followed the
            # model's SELF-GENERATED field lines, same 12 rows as the
            # fast-path table — does narration help or hurt the
            # actions? Small-n, directional only; never compared to
            # the full-probe scalar. (The dedicated all-fields table
            # was dropped 2026-08-03 — visually a subset of the main
            # table once it regained the aux columns.)
            all_fields_mae = [
                float((rich_actions[i] - row.truth).abs()[row.valid].mean())
                for i, row in enumerate(rich_rows)
            ]
            wandb_run.log(
                {
                    f"{table_key}_all_fields_mae": sum(all_fields_mae)
                    / len(all_fields_mae),
                },
                step=step,
            )

            if AuxField.HOLDING in aux_fields:
                accuracy = holding_accuracy(generations, probe.rich_items)
                if accuracy is not None:
                    wandb_run.log(
                        {f"{table_key}_holding_acc": accuracy},
                        step=step,
                    )

        # Q3 — conditioning sensitivity, THE tripwire for silent
        # conditioning collapse: on labeled non-success rich rows, decode
        # once more with outcome overridden to "success" and log the mean
        # |Δ| against the true-conditioned scalar-pass predictions.
        # Pre-registered: > 0 and growing; ≈ 0 means the model ignores
        # the label and the failed-demo mass trained as-if-good. The
        # override decode reuses each row's scalar-pass noise: with a
        # fresh draw, a flow decoder's |Δ| has a floor at the sampling
        # variance even when the model is conditioning-blind — the exact
        # state this alarm exists to catch (deep-dive finding 3).
        if ConditionField.OUTCOME in collator.condition_fields:
            flipped = [
                (i, item)
                for i, item in enumerate(probe.rich_items)
                if item.get("condition_outcome") not in (None, "success")
            ]
            if flipped:
                override_items = [
                    {**item, "condition_outcome": "success"} for _, item in flipped
                ]
                override_batch = collator(override_items).to(device)
                flipped_noise = [
                    row
                    for row in (rich_rows[i].noise for i, _ in flipped)
                    if row is not None
                ]
                override_actions, _ = probe_prediction(
                    model,
                    override_batch,
                    generator=generator,
                    flow_probe_method=flow_probe_method,
                    noise=(
                        torch.stack(flipped_noise).to(device)
                        if len(flipped_noise) == len(flipped)
                        else None
                    ),
                )
                deltas = [
                    float(
                        (override_actions[j].cpu() - rich_rows[i].sampled)
                        .abs()[rich_rows[i].valid]
                        .mean(),
                    )
                    for j, (i, _) in enumerate(flipped)
                ]
                wandb_run.log(
                    {
                        f"{metric_prefix}/condition_sensitivity": sum(deltas)
                        / len(deltas),
                    },
                    step=step,
                )
    return mae
