# Pre-reg (DRAFT): molmo2 vision-unfreeze rung — #17

*Drafted 2026-08-07 (owner question 17:04Z: "what evidence on
unfreezing our SigLIP encoder in molmo2, helpful or harmful?" →
[vision-encoder-freeze](../papers/vision-encoder-freeze.md) lit
slice same day). **STATUS: DRAFT — this is not yet a posted
pre-registration.** Execution is blocked on: (a) the finalization
amendment below, (b) an owner go, (c) a box window after the
attach-screen chain (~08-09+). Nothing launches off this page as it
stands.*

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
declared blind spot in §6 is where the harm would hide if it exists
for us.

Recipe prior from the same slice: **full-FT the tower at low LR;
never LoRA-on-SigLIP** (2607.10172's 0.43 uncanny valley).

## 2. Design — one variable

Arm = the 40k launcher
(`fontaine/scripts/box/launch_box_fontaine_molmo2_ar_40k_ddp4.sh`)
byte-identical — same data gate, collator, freezing split
(`wte`/`lm_head` frozen, decoder + FAST rows trainable), aux/
condition/dropout flags, `--decoder-lr 1e-4 --backbone-text-lr 2e-5`,
seed 0, B12/rank 4×DDP global 48, ZeRO-1 + 6×2 chunked backward +
`--chunk-grad-allreduce` — plus exactly one flag:

- **`--backbone-vision-lr 2e-6`** (0.1× the text LR — the standard
  published tower/text ratio for full-FT vision under a leashed
  language side; "low LR" per the banked recipe prior). Frozen at
  this value: an LR sweep is a different pre-reg, not a post-hoc
  branch of this one.

**Primary rung = a 10k screen, not a 40k re-run.** Matched-steps
contrast: train the arm to 10,000 steps
(`RUN_NAME=fontaine_molmo2_ar_vu10k_ddp4`), then run the endpoint
panel eval — the 40k launcher's chained eval command verbatim (same
plan `plans/holdout_curated_v0_k4l2.json`, `--report-samples 32`,
dumps + json) — on **both** `step_010000` checkpoints: the arm's and
the banked baseline's (`fontaine_molmo2_ar_40k_ddp4/step_010000`,
saved at the 2500 cadence). Baseline stems
`eval__fontaine_molmo2_ar_40k_ddp4__step_010000__panel_curated_v0_k4l2.*`,
arm stems the `vu10k` twin. Same probe cadence as the baseline
(`--eval-every 500`, 256 samples, seed 0 → identical probe batches),
so the probe curves overlay at matched steps.

The full-40k arm is the *escalation* rung (§7), not this launch —
~110 GPU-h does not get spent before the ~27 GPU-h screen says the
direction is real.

## 3. Memory (the binding constraint) and the pre-registered ladder

The baseline sits at **67.07 GiB peak vs the 71.0 gate**. Unfreezing
the ~428M-param so400m tower adds, per rank: fp32 grads ≈ 1.7 GiB
(params already load fp32; `--chunk-grad-allreduce` holds full-size
grads), ZeRO-1-sharded Adam moments ≈ 0.9 GiB, plus tower activation
graphs at microbatch 2 (order ~1 GiB; the tower forward currently
runs grad-free and keeps nothing). Projected peak ≈ 70–72 GiB —
**straddling the gate**, so the smoke is load-bearing, not a
formality. Ladder, in order, semantics exact at every rung:

- **R0**: recipe as-is + the flag (B12, `BACKWARD_CHUNKS=6`).
- **R1**: `BACKWARD_CHUNKS=12` (microbatch 1; gradient exactly
  equivalent, activation footprint halves).
- **R2**: R1 + `--activation-checkpointing` (molmo2 **decoder blocks
  only** — it does not cover the tower; it buys headroom by
  shrinking the decoder's share).
- **All red** → stop. A matched *downshift* is NOT on this ladder:
  the baseline is banked at global 48, so changing batch semantics
  poisons the contrast this rung exists to make. The named ways
  forward are tower-side activation checkpointing (new code, its own
  oracle-gated item) or an owner call.

Gate to launch: an F1-style 150-step smoke of the winning rung with
`vram_alloc_peak_gib ≤ 71.0`, peak + rate quoted in the finalization
amendment. The smoke also confirms the trainable-param banner counts
the tower (~4.3e8 vision params; `bijou.train` hard-aborts if the
backbone had no tower, so a silent no-op unfreeze cannot happen).

## 4. Kill lines (kills wait for save boundaries, cadence 2500)

- E1 data banner must be byte-identical to the baseline's (878
  datasets / 38,571 episodes / 18,636,749 frames / dims 6/6) — any
  deviation aborts before step 1.
- NaN/inf loss → kill.
- **Vision-damage line**: probe > (banked baseline probe at the same
  step) + 2.0, sustained ×3 consecutive evals, any time after step
  2000 → kill. The baseline curve values at 2500/5000/7500/10000 are
  quoted in the finalization amendment so the gate is mechanical.
- vram > 71.0 GiB sustained, or the run OOMs → dead rung, ladder or
  stop (no mid-run batch surgery).
- Cost gate: **≤ 40 GPU-h** for the screen (est. ~27 GPU-h train at
  ~2.3–2.5 s/step + the two panel evals). Overrun projected at a
  babysit check → kill at the next save boundary, partial result
  reported as partial.

## 5. Frozen reads (before launch, per charter)

1. **Primary**: paired per-frame Δ of chunk-pooled panel MAE,
   arm@10k − baseline@10k, on the k4l2 plan, CI95 (the
   `draws10_t1_results.py` pairing convention). Expectation: Δ < 0.
   Bands: CI95 excluding 0 **and** |Δ| > 0.07 (the banked seed-trio
   spread 7.7966/7.8052/7.7355 — the empirical null scale for a
   pooled panel delta) → real effect; anything inside either bound →
   tie. Sign positive with CI excluding 0 and |Δ| > 0.07 → **harm, a
   real result** (the gradient-quality channel showing up in-regime;
   report loud, feeds the MAPS-leash follow-on).
2. Probe-curve overlay at matched cadence (record-only; adaptation
   speed, not a decision input).
3. Critical-frame re-pool of the primary Δ via
   `critical_frame_repooling.py` (the #16 instrument; robustness
   check — does the verdict hold where the CI-MSE critique says
   pooled MAE lies?).
4. State-copy separation from the arm's dump (record-only).
5. **Declared blind spot, quoted in any adopt decision**: the panel
   cannot see the MAPS/2509.11417 OOD-retention tax. An unfrozen-
   vision checkpoint that wins here may still pay under visual
   perturbation — that cost is invisible until a #16-style rig/OOD
   benchmark exists.

## 6. Decision rule

- Screen **helps** (Δ < 0, real per §5.1) → escalation rung opens:
  full 40k matched arm, its own finalization amendment + owner go
  (~110 GPU-h class; also the natural donor trunk for a later attach
  rung — out of scope here).
- **Tie** → rung dies; frozen tower stays the default; the banked
  conclusion is that our embodiment-adaptation gradients don't need
  the tower to move at 10k scale (worth one ideas-page line, no
  follow-on).
- **Harm** → loud report; the MAPS-style L2-to-init leash becomes
  the named follow-on hedge (own pre-reg, new code).

## 7. What this draft does NOT license

No launch before the finalization amendment + owner go. No post-hoc
LR re-pick (2e-6 is frozen; a sweep is its own pre-reg). No
LoRA-on-SigLIP, ever, per the banked 0.43 evidence. No MAPS leash or
dual-encoder anchor in this rung (the former is the harm-branch
follow-on; the latter alters the interface the attach screen depends
on). No batch-semantics change under any memory pressure.

## Finalization amendment checklist (converts DRAFT → posted)

1. Byte-audit the base launcher + eval stems at HEAD (the
   audit-queue-items-against-git rule; save cadence, plan sha, flag
   existence via `--help`).
2. Run the §3 smoke; quote peak vram, rate, winning ladder rung.
3. Quote the baseline probe values at 2500/5000/7500/10000 for §4.
4. Land the arm launcher (sibling of the 40k script, `vu10k` naming,
   `run_detached.sh` wrapper) + prepared `babysit.toml` entry.
5. Owner go + window confirmation (post-attach-screen chain).
