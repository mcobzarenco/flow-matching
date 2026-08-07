# Checkpointing without stalling: the systems cluster

*Lit slice 2026-08-07 (work session 15:22Z), timed to the same
session's `async_save.py` landing (`e3bdc93`). Cluster page:
CheckFreq (Mohan et al., FAST'21) · Gemini (Wang et al., SOSP'23) ·
DataStates-LLM ([2406.10707](https://arxiv.org/abs/2406.10707)) ·
GoCkpt ([2511.07035](https://arxiv.org/abs/2511.07035), ASPLOS'26) ·
TierCheck ([2605.17821](https://arxiv.org/abs/2605.17821)) · the
checkpoint-I/O measurement study
([2512.24511](https://arxiv.org/abs/2512.24511)). Fed the
`async-checkpoint-saves` queue item (landed this session) and two
named follow-up hooks below.*

## Why we read this now

The molmo2 AR run measures **~15.5 minutes per save against ~92
minutes of stepping — ~14% of wall time gone**, most of it not even
disk: torch's ZeRO-1 `consolidate_state_dict` pickles each rank's
whole optimizer shard, round-trips it through a device ByteTensor,
and broadcasts rank-by-rank over the *training* NCCL group while the
other three GPUs idle-spin. This session landed the fix (capture
device→CPU at the boundary in seconds; gather, merge and write in a
background thread over a dedicated gloo group; publish by one atomic
directory rename). The slice's question: this is a well-trodden
systems literature — **did we land the right design, and what do the
published systems know that we skipped?**

## The shared skeleton every paper builds on

All of these systems decompose a checkpoint into the same two phases
CheckFreq named in 2021:

1. **Snapshot** — get a consistent copy of the training state out of
   the way of the next optimizer step (device→host copy). This is
   the only part that can stall training, because the update
   *mutates* the tensors being copied.
2. **Persist** — serialize and write that copy to durable storage.
   Pure background work; only its *completion* matters (you cannot
   admit a snapshot as a checkpoint until it is fully on disk —
   hence atomic publication).

Everything since is about shrinking phase 1 and hiding phase 2:

- **CheckFreq (FAST'21)** pipelined the two phases against training
  and — its most copied idea — **auto-tunes checkpoint frequency**
  from measured overhead instead of a fixed interval, checkpointing
  as often as a stated overhead budget (a few percent) allows. It
  also insisted a checkpoint is incomplete without **data-iterator
  state**: resuming mid-epoch without it silently changes the sample
  stream relative to the never-crashed run.
- **Gemini (SOSP'23)** moved the persist tier off disk entirely:
  checkpoints go to *peer machines' CPU memory* over spare training
  network bandwidth, interleaved so checkpoint traffic never
  collides with gradient traffic. Recovery from machine failure then
  restores from a peer's RAM in seconds rather than re-reading
  remote storage.
- **DataStates-LLM (2406.10707)** made the snapshot itself lazy:
  model and optimizer shards are **immutable during the whole
  forward+backward**, so the device→host copy can run *under* the
  compute phases and only the update phase must wait for laggards
  ("delay the update until the host copies finish"). Two details
  worth stealing verbatim: they **pre-allocate one pinned host
  buffer sized for all shards and reuse it across every checkpoint**
  (fresh pinned allocation is expensive enough to show up), and they
  coalesce fragmented shards so the copy is few large transfers.
  Claimed up to 48× faster checkpointing and 2.2× end-to-end
  speedup at up to 180 GPUs against blocking baselines.
- **GoCkpt (2511.07035)** attacks the case where even the snapshot
  doesn't fit in one step's slack: spread the transfer across
  *several* steps and repair consistency on the CPU side using the
  gradient information that explains how the state moved between
  partial copies. Up to 38.4% throughput over traditional async
  solutions in their setting; 86.7% less interruption time.
- **TierCheck (2605.17821)** is the 2026 synthesis: lightweight
  *differential* checkpoints in local/peer memory for the common
  fast-recovery cases, heavyweight base checkpoints migrated
  asynchronously to remote persistent storage, consistency
  maintained across tiers; sub-10-second end-to-end checkpointing
  at 40B scale.
- **The I/O study (2512.24511)** measures where the time actually
  goes across strategies and scales, and two of its findings match
  what we saw from the outside this week: **pickle-based
  serialization (`torch.save`) is a first-order cost**, not a
  rounding error, and **per-rank sharded saving is competitive at
  small scale** — the consolidate-to-one-writer pattern we inherit
  from torch's ZeRO wrapper is a choice, not a law.

## What transfers to us

- **The design we landed is the CheckFreq/DataStates shape** —
  boundary snapshot + background persist with atomic publication —
  and the literature's correctness hazards are exactly the ones our
  oracles pin: mutation after the boundary (we deep-copy at capture,
  including the scheduler dict, so a later `scheduler.step()` can't
  leak into the file), torn writes (atomic `.tmp`-dir rename), and
  silent background failure (re-raised loudly at the next
  submit/join). Nothing in the cluster invalidates the design;
  DataStates in particular is our design plus two refinements.
- **Pinned-buffer reuse (DataStates)** is the named next win if the
  capture stall ever matters: our capture does pageable-memory
  copies each save. At attach-screen scale (~37 GB of state) that is
  seconds — acceptable against a ~90-minute cadence — but a
  pre-allocated reusable pinned buffer would cut it several-fold for
  free after the first save. Banked as a follow-up hook on the queue
  item, not urgent.
- **Save-frequency tuning (CheckFreq)**: with the save cost now
  hidden, `--save-every 2500` is no longer balancing step-stall
  against recovery loss — the marginal cost of saving more often is
  disk and background bandwidth only. For the attach screen's
  50–70 GPU-h window, halving the interval halves the worst-case
  recovery loss (driver kills took down two GPU runs *today*;
  recovery loss is not hypothetical for us). Worth a deliberate call
  at launch prep, not a reflex.
- **Data-iterator state (CheckFreq)** names a gap our `--resume`
  shares with most of the field: we reconstruct the stream from
  seed+step rather than serializing iterator position, so a resumed
  run's sample composition is not guaranteed byte-identical to the
  never-crashed run. Known simplification, now with a citation and a
  clear fix shape if a future arm needs exact-resume semantics.

## What doesn't transfer

- **Gemini and TierCheck's memory tiers** solve *node-loss* recovery
  on big clusters. Our failure model this week is process death on a
  single box (the driver teardown incidents) — local disk already
  survives that, and we have no peer fleet to replicate into. The
  differential-checkpoint idea is elegant and irrelevant at two
  machines.
- **GoCkpt's multi-step spreading** pays only when one step's slack
  can't hide the snapshot. Our snapshot is seconds against a
  2.2 s/step · 2500-step cadence; single-boundary capture is the
  right point on the curve, and the gradient-assisted consistency
  repair is complexity we never need.
- **Per-rank sharded formats** (the study's small-scale
  recommendation) would delete our gather entirely — but every
  read-side consumer we have (resume, eval loaders, the panel
  harness, checkpoint uploads) expects the consolidated historical
  layout, and byte-identity with that layout is what let us ship
  async as a default with zero read-side churn. At 4 ranks the
  background gather costs nothing observable; the sharded rewrite
  buys nothing we currently pay for.

## What it fed

The `async-checkpoint-saves` item (landed `e3bdc93` this session,
oracle-gated byte-identical) — this cluster is its literature
grounding, read deliberately *after* building from first principles
and the measured molmo2 anatomy: the published designs converge on
the same two-phase shape, which is the reassurance; the two
refinements worth money later (pinned-buffer reuse, frequency
retuning) are banked on the queue item as follow-up hooks with this
page as the reference.
