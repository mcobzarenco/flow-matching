# Pre-registration: 4×H100 box batch — paired aux-off arms ∥ control seed replicates

*2026-08-05 ~17:25Z. Posted before launch (charter §4). Governs the
first batch on the second box (192.222.55.210, 4×H100 80GB, owner
grant 17:01Z "use it whichever way you want"; sole constraint 17:02Z:
do not delete the existing fine-tune checkpoints — an owner rsync is
in flight. No cleanup of any kind will run on that box.)*

## Relationship to the paired aux-off pre-registration

The science of
[the paired aux-off pre-reg](2026-08-05-prereg-paired-auxoff-40k.md)
is **unchanged**: same two arms, same flags, same seed, same primary
read. This post changes the *execution plan* (venue + parallelism) and
adds two control seed replicates to measure the paired-comparison
noise floor — the 16:48Z in-channel plan skeleton, now concrete. The
local sequential launcher (`~/launch_fontaine_paired_auxoff_40k.sh`)
is superseded as an execution plan but kept as fallback if the box
launch fails its gates; the local box becomes the eval/analysis box.

## Design — four 1×H100 runs in parallel, 40k steps each

| GPU | run | seed | wandb run name |
|-----|-----|------|----------------|
| 0 | **A-s0** control (recipe as-is) | 0 | `fontaine_arb_rcond_40k_1xh100` |
| 1 | **B-s0** aux-supervision OFF | 0 | `fontaine_arb_rcond_auxoff_40k_1xh100` |
| 2 | **A-s1** control replicate | 1 | `fontaine_arb_rcond_40k_1xh100_s1` |
| 3 | **A-s2** control replicate | 2 | `fontaine_arb_rcond_40k_1xh100_s2` |

- All flags exactly as the paired pre-reg (B10, decoder-lr 1e-4,
  backbone-text-lr 2e-5, 40k steps, warmup 1k, workers 16, saves
  every 5k). Arm B differs from A **only** by omitting
  `--aux-fields ...` (aux CE term off; all conditioning kept).
  Replicates differ from A-s0 **only** in `--seed` (1, 2).
- **`--split-seed 0` everywhere** — the holdout split is identical
  across all four runs and identical to the local baseline's.
- Each GPU chains its own panel eval at 40k
  (`plans/holdout_curated_v0_k4l2.json`, `--dump-predictions`, seed 0,
  batch 32) so every pairwise per-frame comparison is available.

**Questions.** (1) Idea #6, unchanged: does the aux CE term (weight
0.5) change ACTION metrics at matched steps/seed/data? (2) New: what
is the seed-noise floor of a paired 40k comparison on this panel?

**Primary read (unchanged):** paired per-frame panel `chunk_mae`
A-s0@40k vs B-s0@40k.

**Secondary read / instrument (new):** pairwise per-frame panel
`chunk_mae` deltas among {A-s0, A-s1, A-s2}@40k — the empirical
seed-noise distribution.

**Decision rule (pre-registered):** the aux-off effect
|A-s0 − B-s0| is called real only if it exceeds the **largest**
pairwise replicate delta AND the per-frame delta distribution is
coherent (not driven by a single repo). Otherwise the answer is
"within seed noise at 40k/eff-10" — which closes idea #6's 40k rung
honestly.

## Environment (verified before posting)

- **Code:** branch `fontaine`, the commit carrying this post; `bijou/`
  tree identical to the smoke-tested state (the box's own checkout was
  behind by 943 lines in `bijou/` — it will be fetched to this exact
  commit; only `pyproject.toml` tooling config differs from the venv's
  build commit, no dependency change, so the owner's `.venv` is
  reused).
- **Interpreter seam:** box runs `.venv/bin/python` directly (no uv on
  the box); torch 2.11.0+cu130 on **both** boxes — no framework-version
  seam. wandb + HF creds present on the box (`~/.netrc`, HF token).
- **Data:** the box's `~/datasets/mcobzarenco/community_curated_v0`
  (600G, 283 repo dirs) matches the local frozen copy — listing diff
  is empty except the local-only inert `provenance/` tarball (28M
  curation metadata, not a lerobot repo). Same pre-removal revision;
  the kevin510/bbox-2 cleanup boundary (16:50Z steering) applies to
  the box copy too: **no re-pull, no mutation** until the arms + reads
  are done. E1 (below) is the hard gate on selection identity.
- **Disk:** ~768G of saves (4 runs × 8 × 24G) into 7.6T free.
- **Contention:** 4 × 16 workers = 64 of 104 cores; one shared disk.

## Expectations

- **E1 startup (per run, hard gate):** 878 datasets / 42,872 episodes
  / dims 6/6 — identical on all four runs (selection is data+flags
  only). Any deviation ⇒ abort the whole batch before step 1 (data
  copy not identical after all). B-s0's log shows no `loss_aux` and
  no aux fields in the model line.
- **E2 throughput:** 0.4–0.6 s/step at B10 per the 1×H100 smoke;
  allow to ~0.7 for shared-I/O contention. Sustained > 0.8 s/step or
  starving GPU util ⇒ input-pipeline fix (workers/prefetch) at the
  next safe boundary, logged. VRAM < 76 GiB/GPU. ~5–6.5 h per arm
  wall (all four concurrent), evals +~1.7 h staggered ⇒ all reads by
  ~02Z.
- **E3 curves (256-frame probe, ±0.3 floor):** A-s0 < 12 by 10k, < 9
  by 30k; B-s0 within ±0.3 of A-s0 at matched steps (pre-registered
  mainline expectation: aux is within probe noise); replicates within
  ±0.3 of A-s0.
- **E4 primary read:** as above; pre-registered expectation from the
  mainline ledger: |A−B| within noise (aux shapes narration, not
  actions) — but that is the hypothesis under test, not a gate.
- **E5 noise floor:** soft expectation: pooled pairwise replicate
  |Δ chunk_mae| ≤ 0.2. No gate — this IS the instrument. If it comes
  out large (> 0.3), that is itself a headline result (paired 40k
  comparisons at eff-10 are noisier than assumed and every ±0.3
  claim needs revisiting).

## Kill gates & seams

- Per run (unchanged): probe > 15 @ 10k after falling-then-rising;
  NaN loss; second OOM after the standing batch−1 resume.
- **Parallel-launch seam** (replaces "A killed ⇒ don't launch B"):
  arms start together, so if A-s0 trips a kill gate, B-s0 is killed
  too (the pair is void); replicates keep running. Promoting a
  surviving replicate to the control role would seed-confound the
  paired read — allowed only via a posted amendment, with the
  confound bounded by the measured replicate spread.
- No other seams: fresh runs, no resume, tokenizer v2 pinned, no
  DDP (four independent single-GPU processes).

## Box discipline (temporary-box rules)

- Batch pre-reg (this post) before any launch; save-boundary sizing
  (5k saves ⇒ ≤5k steps lost on reclaim).
- **Continuous rsync-back** to the local box (tmux loop, ~20 min
  cadence): all `~/train_fontaine_*.log`, all
  `reports/eval__fontaine_*`, wandb is off-box anyway; per run, the
  latest two `step_*` saves (full 768G mirror is pointless — wandb
  carries curves; the finals carry the reads).
- Owner's existing checkpoints/artifacts: untouched, no deletes ever.

## Local box re-point (recorded here for the tick loop)

1. Sealed baseline score (running, ETA ~18:05Z): bank the anchor,
   post in-channel. **Do NOT launch the local paired run** — it is
   superseded by this batch (unless the box batch is dead by then).
2. The 80k flow panel eval (queue #3) is **already answered**: the
   owner panel-scored it on the box today 12:20Z —
   `bijou_flow_artrunk_h1024_40k_ddp2/step_080000`, heun-30, panel
   k4l2: **chunk_mae 6.6232, first_mae 1.9331** (AR-100k same panel
   same day: 5.8026 / 2.1431; state-copy summaries bitwise-identical
   ⇒ npzs pair per-frame). Reports + npzs exist on the box; pull, do
   the paired flow-vs-AR per-frame analysis on CPU. No GPU eval
   needed tonight.
3. Freed local GPU slot after the sealed score: the noise-draw
   ensembling probe
   ([pre-reg](2026-08-05-prereg-noise-draw-ensembling.md)) — verify
   upstream `--sample-draws` semantics first (a16e65a), then run per
   its pre-reg on the flow-80k checkpoint (pull from HF/box first).
