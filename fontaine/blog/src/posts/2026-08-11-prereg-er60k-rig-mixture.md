# Pre-registration DRAFT: rig-mixture fine-tune rung on the er_60k trunk

*Drafted 2026-08-11 ~16:5xZ (work session), executing the mixture
lever the [er-60k pre-reg](2026-08-09-prereg-molmo2-er-60k.md) pinned
and the [ER screen close](2026-08-11-er-init-screen-results.md) named
as the next unpriced step. Status: **DRAFT — HOLDING for the owner
compute call** (the 4× box is gone; see "Compute ask"). Finalizes
into an immutable pre-reg with a param sheet in-channel before any
GPU minute, per the standing gate.*

## Plain words

Our best trunk (`er_60k`) trained on ~18.7M frames of community robot
data plus the owner's two rig datasets — but the rig data is only
0.19% of the corpus, so the model effectively never saw it (each rig
frame ~0.9× in expectation over the whole run). This rung continues
training from the er_60k endpoint with the rig data oversampled to
~5% of every batch, and asks two questions with one run: does the
model get **meaningfully better on held-out rig episodes**, and does
it do so **without getting worse on the general panel**? That pair is
the whole VLA-for-the-rig question in miniature: one model that
works on the owner's robot without forgetting everything else.

## Instrument (landed, this session)

The loader dedups repeated roots, so there was no zero-code
oversample. `--dataset-repeat PATTERN=COUNT` (commit `1b1c314`)
replicates matching datasets in the concatenated train set after all
guards and the episode split: fnmatch against `<user>/<dataset>` repo
ids, first matching spec wins, a spec matching no selected dataset is
fatal (a silently unapplied oversample would corrupt the registered
mixture). Same objects — no extra host RAM; the shuffle,
length-bucket keys and `DistributedSampler` all see the replicas.
Training-only: eval call sites never pass it. Oracle test
`tests/test_dataset_repeat.py` pins parse/precedence/no-match plus
the mixture arithmetic below; `check.py` green (683).

## Design

One arm, warm-started from the reference trunk:

- **Init**: `--init-from er_60k/step_060000` (weights-only is
  sufficient and is what survives the box teardown —
  `fontaine-checkpoints`, commit `4ed3dd0`; the optimizer state is
  gone with the box, so a fresh optimizer is forced, not chosen).
- **Mixture**: `--dataset-repeat mcobzarenco/so101_pick_place_clean=27
  mcobzarenco/so101_pick_place_v2=27` — explicit per-repo specs (no
  wildcard: immune to future owner datasets landing in the curated
  collection). Effective share = 27×36,078 / (18.67M + 26×36,078) ≈
  **4.97%** (natural 0.19%), inside the CL-triangle 2–20% replay
  band ([page](../papers/cl-triangle.md)).
- **Steps**: 10,000 at eff-batch 48 — ~24k rig draws ≈ **0.66 rig
  epochs**. Named tension: the full-run arithmetic the mixture note
  imagined (~4 rig epochs) belongs to a 60k-step run; a screen-sized
  rung at 5% is sub-epoch. If the owner wants a rig-heavier rung
  instead, ×129 ≈ **20% share** (top of the CL band) gives ~2.7 rig
  epochs in the same 10k steps — same cost, more forgetting risk.
  Default registered arm is 5%.
- **Recipe**: er-60k launcher verbatim otherwise (decoder
  `ar_backbone`, aux + condition fields, decoder-lr 1e-4,
  backbone-text-lr 2e-5, fps 30, camera-counts 1 2, holdout 0.1
  split-seed 0, async saves). Named deltas: `--warmup-steps 500`
  (fresh optimizer re-estimates moments; 1000 would be 10% of a 10k
  run), `--save-every 2500`, `--seed 3` (fresh-seed-on-extension
  standing rule; 0 and 2 are used in this lineage), cosine decays
  over the rung's own 10k steps (the init-from convention).
- **No control arm registered** (startup-velocity): at natural share
  a 10k continuation sees ~450 rig draws — it cannot explain a
  material rig gain, so the er_60k endpoint itself is the baseline.
  A natural-share control continuation is the named escalation if
  the primary read lands small or ambiguous.

## Frozen reads (numbers finalize at param-sheet time)

1. **Primary — rig holdout, paired CI95**: the deterministic
   episode holdout (fraction 0.1, split-seed 0) holds out 1 of 7
   `clean` + 5 of 50 `v2` episodes (~3.7k frames) that er_60k never
   trained on. Score mixture endpoint vs `er_60k/step_060000` on
   identical frames (+ state-copy anchor), paired per-frame CI95.
   **Pass = mixture below the er_60k baseline with CI excluding
   zero.** This is a genuine within-rig generalization read, unlike
   the contaminated-by-construction MolmoAct2 rig-ft train-frame
   reads.
2. **Guard — panel non-regression**: k4l2 panel_v2 eval at endpoint
   (`--report` + npz per the standing rule), paired per-frame vs the
   banked er_60k endpoint panel (5.7782/1.9898). **Fail = pooled
   delta worse than +0.05 with CI excluding zero** (band draft —
   finalize against the 55k→60k rung step of ~0.05).
3. Record-only: probe ladder at the er-60k cadence; aux-head
   accuracies vs the endpoint row (holding 0.915 / progress 0.060 /
   event 0.858 / visible 0.822); per-episode rig MAE spread.
4. Kill lines: NaN/inf; probe divergence bar re-derived from the
   er_60k tail at finalization; vram near-OOM per first-poll
   actuals.

## Compute ask — OWNER DECISION, the run is not launchable without it

The 4× box is torn down. Options, priced:

- **(A) New 4× box rung** — recipe verbatim, ~2.5 GPU-h/1k steps
  measured on the er_60k run → **~28 GPU-h, ~7 h wall** incl. the
  endpoint panel. Cleanest: zero recipe deltas beyond the named
  ones. Gate **32 GPU-h**.
- **(B) Local 1×H100** — the full recipe is **measured structurally
  OOM single-GPU** (08-08 perf review: 78.2/79.18 GiB by step 2,
  batch-invariant, ZeRO-1 unshards +11 GiB). A local leg therefore
  needs a fit-preflight ladder first: `--activation-checkpointing`
  (landed + oracle-gated, ~2.4–2.8 GiB/sample) + small batch ×
  backward-chunks, ZeRO-1 off. If green: ~same GPU-h, **~14–16 h
  wall** (serial 4× + recompute overhead), local GPU blocked for a
  day. If red, (A) is the only path.
- **(C) Defer** — the instrument and this draft keep; the rung
  launches whenever compute lands.

## Non-goals / non-deltas

No vision unfreeze, no AdamC, no recipe retune, no rollout claims
(rig rollouts are the runbook's out-of-band path and need the
convention rails). The AE-on-our-trunk work is a separate queued item
behind the main rebase and does not ride this rung.
