# Pre-reg (DRAFT): molmo2 vision-unfreeze rung — #17

*Drafted 2026-08-07 (owner question 17:04Z: "what evidence on
unfreezing our SigLIP encoder in molmo2, helpful or harmful?" →
[vision-encoder-freeze](../papers/vision-encoder-freeze.md) lit
slice same day). **Amendment 1, 2026-08-07 18:xxZ (owner steering
18:02Z): the from-scratch 10k screen is replaced by a warm-start
two-arm continuation from the 40k endpoint** — frozen-continue vs
thawed-continue; rationale in §2, the superseded design recorded in
§8. **Amendment 2, 2026-08-07 18:4xZ (owner steering 18:31Z +
18:39Z): 5k steps per arm (was 3k), gate 32 GPU-h (was 24); the
fresh-AdamW `--init-from` route is now owner-confirmed** — a
resume-with-injected-vision-group patch was offered in-channel and
declined ("you're right re: fresh adam optimisers", 18:39Z); seed
and rewarmup steering from 18:31Z was already satisfied by
`--seed 1` and `--warmup-steps 200`. **STATUS: DRAFT — this is not
yet a posted pre-registration.**
Execution is blocked on: (a) the finalization amendment below, (b)
an owner go, (c) a box window after the attach-screen chain
(~08-09+). Nothing launches off this page as it stands.*

## 1. Question and prior

The live molmo2 AR 40k trunk run trains with the SigLIP tower frozen
(no `--backbone-vision-lr`); its pre-reg names a vision-unfreeze
rung as a follow-on. Does unfreezing the tower improve the panel
number in our regime?

Written prior (frozen here, before any number is read): **it should
help.** Our run is squarely the embodiment-adaptation regime — 18.6M
frames, panel = held-out episodes of the same distribution — and
that regime's published evidence is consistent: OpenVLA's
frozen-encoder underperformance (2406.09246), VLM4VLA's
frozen-encoder collapse (PaliGemma/SigLIP, Calvin 3.51 → 0.50,
[vla-initialization](../papers/vla-initialization.md)), the
assembly-domain swing (frozen SigLIP 0.14 → trainable 0.74,
2607.10172). The known harm cases (MAPS 2511.19878, dual-encoder
2509.11417) live on the OOD-retention axis, which the panel barely
measures — so the *panel* read should favor unfreezing, and the
declared blind spot in §5 is where the harm would hide if it exists
for us.

Recipe prior from the same slice: **full-FT the tower at low LR;
never LoRA-on-SigLIP** (2607.10172's 0.43 uncanny valley).

**Honest caveat on the warm-start shape (stated in the 18:2xZ
in-channel reply, frozen here)**: a late low-LR thaw can understate
what unfreeze-from-*scratch* buys — the lit's ablations co-adapt
vision from step 0, and 5k tail steps may not recover that
co-adaptation. The bet is asymmetric: a positive upgrades the
deployment artifact immediately at ~half the from-scratch screen's
cost; a null leaves the from-scratch question open but cheap to
revisit.

## 2. Design — warm-start two-arm continuation (owner steering 18:02Z)

Both arms continue **from the 40k endpoint checkpoint**
(`outputs/train/fontaine_molmo2_ar_40k_ddp4/step_040000`, box) via
`--init-from` — weights-only warm start, fresh optimizer/scheduler.
Base recipe = the 40k launcher
(`fontaine/scripts/box/launch_box_fontaine_molmo2_ar_40k_ddp4.sh`)
byte-identical — same data gate, collator, freezing split, aux/
condition/dropout flags, B12/rank 4×DDP global 48, ZeRO-1 + 6×2
chunked backward + `--chunk-grad-allreduce` — with the continuation
deltas pinned below, identical across arms except exactly one flag:

- **Frozen-continue** (`fontaine_molmo2_ar_vu5k_frozen_ddp4`) — the
  control. Tower stays frozen. The essential arm: extra steps alone
  move the number, so nothing is readable against the 40k endpoint
  without it.
- **Thawed-continue** (`fontaine_molmo2_ar_vu5k_thawed_ddp4`) — plus
  **`--backbone-vision-lr 2e-6`** (frozen at the draft's value; an
  LR sweep is a different pre-reg). Drafted as 0.1× the original
  text peak (2e-5); in this continuation it equals the text *tail*
  LR — stated, not re-picked.

Continuation deltas (both arms, identical):

- `--init-from .../step_040000` (NOT `--resume`, on two grounds,
  both verified in `bijou/train.py` at HEAD: **mechanical** — the
  thawed arm builds two extra optimizer param groups
  (`backbone_vision` decayed/no-decay: five groups vs the 40k
  checkpoint's three), and `optimizer.load_state_dict` against the
  40k run's `optimizer.pt` raises on the group-count mismatch;
  **methodological**
  — `--init-from` gives both arms the same fresh-AdamW treatment, so
  the warm-restart transient is common-mode in the paired read. The
  tower has no optimizer state at 40k in any case; fresh moments +
  a short ramp is the standard late-unfreeze mechanic).
- `--steps 5000 --warmup-steps 200`: 5k continuation steps (amendment 2; was 3k — owner 18:31Z, more room for the late-thaw co-adaptation §1 worries about), 200-step
  linear ramp (the "short tower warmup"; the ramp is global — all
  groups share it, symmetric across arms). Cosine floors at 10% of
  peak; endpoint `step_005000` always saves (`bijou.train`
  save-boundary rule), cadence 2500 kept verbatim.
- **LRs = the 40k tail values**: `--decoder-lr 1e-5
  --backbone-text-lr 2e-6` (the 40k cosine floors at 10% of peaks
  1e-4/2e-5, so these are the LRs the run ends at — "text side
  continues at tail LR"). The fresh 5k cosine anneals from there;
  schedule shape is common-mode in the paired read.
- `--seed 1` (fresh vs the 40k run's seed 0; same seed both arms →
  identical batches and τ/ε streams, so the arms differ in the one
  flag and nothing else).
- Probe cadence unchanged (`--eval-every 500`, 256 samples) → probe
  curves overlay at matched continuation steps.
- Chained endpoint panel eval per arm, the 40k launcher's eval
  command verbatim (plan `plans/holdout_curated_v0_k4l2.json`,
  `--report-samples 32`, dumps + json), stems
  `eval__fontaine_molmo2_ar_vu5k_{frozen,thawed}_ddp4__step_005000__panel_curated_v0_k4l2.*`.

**Order: frozen-continue first** (box arms are sequential on the 4
GPUs). Its probe curve and endpoint bank become the thawed arm's
mechanical kill-line references (§4).

Warm-start facts, stated for honesty (symmetric, common-mode in the
paired Δ; visible in the record-only arm-vs-endpoint reads): fp32
masters restart snapped to the checkpoint's bf16 grid
(`backbone_snapshot` serializes bf16); text/decoder Adam moments
restart fresh; the data stream restarts at epoch 0 under seed 1
(frames the 40k run has seen, new order — unavoidable for any
continuation). Molmo2 snapshots are full-model (tower included), so
both arms inherit the identical backbone.

## 3. Memory (the binding constraint for the thawed arm) and the ladder

Unchanged from the original draft — the contrast moved, the memory
fight didn't. The baseline recipe peaks **67.07 GiB vs the 71.0
gate**. Unfreezing the ~428M-param so400m tower adds, per rank: fp32
grads ≈ 1.7 GiB (`--chunk-grad-allreduce` holds full-size grads),
ZeRO-1-sharded Adam moments ≈ 0.9 GiB, plus tower activation graphs
at microbatch 2 (order ~1 GiB). Projected peak ≈ 70–72 GiB —
**straddling the gate**, so the smoke is load-bearing. Ladder, in
order, semantics exact at every rung (thawed arm only; the frozen
arm is the banked recipe and takes R0 by construction):

- **R0**: recipe as-is + the flag (B12, `BACKWARD_CHUNKS=6`).
- **R1**: `BACKWARD_CHUNKS=12` (microbatch 1; gradient exactly
  equivalent, activation footprint halves).
- **R2**: R1 + `--activation-checkpointing` (decoder blocks only —
  it does not cover the tower; it buys headroom by shrinking the
  decoder's share).
- **All red** → stop. A matched *downshift* is NOT on this ladder:
  batch semantics changes poison the contrast. Named ways forward:
  tower-side activation checkpointing (new code, own oracle-gated
  item) or an owner call.

Gate to launch: a 150-step smoke of the winning rung with
`vram_alloc_peak_gib ≤ 71.0`, peak + rate quoted in the finalization
amendment. The smoke also confirms the trainable-param banner counts
the tower (~4.3e8 vision params; `bijou.train` hard-aborts if the
backbone has no tower, so a silent no-op unfreeze cannot happen).

## 4. Kill lines (kills wait for save boundaries)

- E1 data banner byte-identical to the baseline's (878 datasets /
  38,571 episodes / 18,636,749 frames / dims 6/6), both arms — any
  deviation aborts before step 1.
- NaN/inf loss → kill.
- **Frozen-arm sanity line**: frozen-continue probe > (banked 40k
  endpoint probe) + 2.0, sustained ×3 consecutive evals → the
  *control itself* is broken (a tail-LR continuation should not
  regress) → stop the screen, no thawed launch, report loud. The
  endpoint probe value is quoted in the finalization amendment.
- **Vision-damage line (thawed arm)**: thawed probe > (frozen-arm
  probe at the same continuation step) + 2.0, sustained ×3
  consecutive evals, any time after step 1000 → kill.
- vram > 71.0 GiB sustained, or OOM → dead rung, ladder or stop (no
  mid-run batch surgery).
- Cost gate: **≤ 32 GPU-h** for the screen (amendment 2; est. ~26
  GPU-h train — frozen ≈ 12.2 at the measured 2.2 s/step, thawed ≈
  13.9 at a projected ~2.4–2.6 s/step with the tower backward — plus
  two chained panel evals + smoke margin). Overrun projected at a
  babysit check → kill at the next save boundary, partial reported
  as partial.

## 5. Frozen reads (before launch, per charter)

1. **Primary**: paired per-frame Δ of chunk-pooled panel MAE,
   **thawed@5000 − frozen@5000**, on the k4l2 plan, CI95 (the
   `draws10_t1_results.py` pairing convention). Expectation: Δ < 0.
   Bands: CI95 excluding 0 **and** |Δ| > 0.07 (the banked seed-trio
   spread — the empirical null scale for a pooled panel delta) →
   real effect; anything inside either bound → tie. Sign positive
   with CI excluding 0 and |Δ| > 0.07 → **harm, a real result**
   (report loud, feeds the MAPS-leash follow-on).
2. **Record-only**: each arm@5000 vs the banked 40k endpoint panel —
   the "extra steps alone" channel the frozen control exists to
   subtract, plus the warm-start transient; never a headline.
3. Probe-curve overlay at matched continuation cadence (record-only;
   adaptation speed, not a decision input).
4. Critical-frame re-pool of the primary Δ via
   `critical_frame_repooling.py` (the #16 instrument; robustness
   check).
5. State-copy separation from the arms' dumps (record-only).
6. **Declared blind spot, quoted in any adopt decision**: the panel
   cannot see the MAPS/2509.11417 OOD-retention tax. An
   unfrozen-vision checkpoint that wins here may still pay under
   visual perturbation — invisible until a #16-style rig/OOD
   benchmark exists.

## 6. Decision rule

- Screen **helps** (Δ < 0, real per §5.1) → the thawed@5000
  checkpoint is the new deployment-artifact candidate (that is the
  point of the warm-start shape: the winner is kept, not re-derived);
  tower-unfrozen becomes the default for future trunk recipes,
  §5.6 caveat quoted. The from-scratch co-adaptation question (§1
  caveat) stays open as a named, deprioritized escalation.
- **Tie** → rung dies; frozen tower stays the default at tail-thaw
  scale. Recorded honestly as "a 5k late thaw doesn't move the
  panel", NOT "unfreezing doesn't help" — the §1 caveat bounds the
  claim.
- **Harm** → loud report; the MAPS-style L2-to-init leash becomes
  the named follow-on hedge (own pre-reg, new code).

## 7. What this draft does NOT license

No launch before the finalization amendment + owner go. No post-hoc
LR re-pick (2e-6 frozen; a sweep is its own pre-reg). No
LoRA-on-SigLIP, ever. No MAPS leash or dual-encoder anchor in this
rung. No batch-semantics change under any memory pressure. No
reading the thawed arm without the frozen control landing first.

## 8. Superseded design (recorded, not licensed)

The original draft's primary was a from-scratch 10k screen: the 40k
recipe + the flag trained fresh to 10k, read against the banked
baseline `step_010000` checkpoint (~27 GPU-h, ~40 GPU-h gate).
Replaced by owner steering 2026-08-07 18:02Z ("startup mindset,
shortest time to high quality rollouts"): the warm-start two-arm
form was ~15 GPU-h train at amendment 1's 3k/arm (amendment 2 took
it to 5k/arm, ~26), upgrades the actual deployment artifact on
a win, and replays none of the easy curriculum. What the swap gives
up is stated in §1's caveat and §6's tie wording.

## Finalization amendment checklist (converts DRAFT → posted)

1. Byte-audit the base launcher + eval stems at HEAD
   (audit-queue-items-against-git rule; flag existence via `--help`;
   `--init-from` + `--seed 1` + tail-LR flags verified against
   `bijou.train`'s actual CLI surface).
2. Run the §3 smoke (thawed recipe, 150 steps from the endpoint
   checkpoint); quote peak vram, rate, winning ladder rung — and
   confirm the first async save's "captured in Xs" line (#18.9
   first-real-run validation note carries over).
3. Quote the banked 40k endpoint probe value for §4's frozen-arm
   sanity line.
4. Land the two arm launchers (siblings of the 40k script,
   `vu5k_frozen`/`vu5k_thawed` naming, `run_detached.sh` wrapper) +
   prepared `babysit.toml` entries, frozen-first ordering explicit.
5. Owner go + window confirmation (post-attach-screen chain).
