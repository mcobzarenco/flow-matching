# Pre-registration: state-dropout 0.8 — the anti-shortcut arm (ideas #9)

*2026-08-06 ~08:1xZ. Posted before launch (charter §4). The
state-reliance probe's branch rule fired
([results](2026-08-06-state-probe-results.md): D = +0.702
[0.498, 0.916] — aux-off leans harder on the state shortcut, and all
four checkpoints degrade +15–16 chunk MAE under state masking), so the
literature's counter-lever gets its paired arm:
**[Adapt Your Body](https://arxiv.org/abs/2506.23944)-style state
dropout at p = 0.8**, the causal-confusion line's standard mitigation
([2509.18644](https://arxiv.org/abs/2509.18644) goes to state-free
entirely and reports better spatial generalization). Explore-class
arm (charter §3 budget): the modal outcome is "within band"; the tail
is a vision-reliant policy, which is what the §0 north star (few-shot
rig transfer) actually needs — a policy that extrapolates
proprioception transfers worse than one that looks.*

## Instrument (landed with this post, before launch)

`--state-dropout p` in `bijou.train`: with probability p per sample,
the collator rewrites the item through `mask_state_item` — the SAME
primitive as the eval probe's `--mask-state` (state → dataset mean ⇒
normalized soft state token EXACTLY zero, raw decoder-side state at
the mean, actions/targets untouched), so the regularizer trains
exactly the condition the probe measures and the two can never drift.
p=0 is bitwise-inert and consumes no RNG (all three CPU loss oracles
reproduce exactly: flow 2.7903/1.9152, ar_fast 4.9232/4.8631,
ar_backbone 27.8262/27.7701; `check.py` 240 green; SnapFlow stage-0
recipe verify re-run green with the new field at its inert default).
In-run probes and panel evals score INTACT state (dropout-0 clones —
the deployment condition); the masked readout is the offline probe.

## Design — one arm, one variable

**Arm C: `fontaine_arb_rcond_statedrop80_40k_1xh100`** — the A-s0
recipe verbatim (seed 0, 40k steps, B10, ar_backbone rcond, box GPU 0,
launcher
`fontaine/scripts/box/launch_box_gpu0_fontaine_arb_rcond_statedrop80_40k_1xh100.sh`)
plus exactly `--state-dropout 0.8` (the literature's value; p is the
one free parameter and 0.8 is the aggressive end — if it fails while
the mechanism reads clean, a lower-p screen is the pre-declared
branch, a NEW pre-reg). Paired against **A-s0's banked panel npz**
(chunk 7.7966 / first 3.9422 @40k) — zero re-evals on the intact side.
Chained in-launcher after training: full panel eval, then the
masked-subset reliance eval (frozen q4 plan, sha256 asserted at launch
before any GPU work).

**Known seam, stated now:** the per-sample dropout draw comes from the
same per-worker generator as the other train-time regularizers, so
arm C's camera-kind/instruction/condition dropout decisions shift
relative to A-s0's (equivalent to re-seeding the regularizer streams
only; data order, model init, and τ/ε draws are untouched — separate
generators). This sub-seed effect is bounded above by the full
seed-noise measurement: max pairwise replicate delta 0.0697, σ_seed
0.038. The 0.15 band floor already absorbs it.

## Frozen reads

1. **Primary — paired per-frame panel chunk_mae, C − A-s0** (17,204
   core frames, seeded bootstrap CI, the box-batch instrument's
   paired-read path re-oracled on the new pair before the real read).
   Band = max(3σ_seed, 0.15) = **0.15**:
   - C − A-s0 < −0.15 ⇒ state dropout HELPS actions in-distribution →
     adopt as a recipe default in future arms (its own follow-up
     pre-reg for the next lineage run).
   - |C − A-s0| ≤ 0.15 ⇒ headline-neutral → decision moves to reads
     2–3: a free hardening lever is still adopted if the mechanism
     reads clean (vision reliance is north-star-relevant even when the
     in-distribution panel can't see it).
   - C − A-s0 > +0.15 ⇒ p=0.8 costs actions → adopt NOTHING; branch:
     if read 2 shows the mechanism worked (reliance collapsed), a
     p=0.3 screen is the one sanctioned follow-up; if read 2 also
     fails, #9's dropout leg is falsified at this scale.
2. **Reliance readout — masked-subset eval of C @40k** (frozen q4
   plan, 4,301 rows; intact side pooled from C's own panel npz —
   strict row-subset, the probe instrument's established pattern).
   Execution oracles inherited from the probe pre-reg: state-copy/
   -norm rows byte-match C's own full-panel npz on the subset rows;
   report JSON records `mask_state: true`; policy name carries
   `_state-masked`. Reads, with banked comparators:
   - **Sanity gate:** Δ_first(C) = masked − intact first_mae < 5.0
     (vs A-s0's +19.950): masking is in-distribution for C by
     construction — this gate only proves the regularizer trained the
     intended condition; it is NOT evidence for the vision-shift
     hypothesis.
   - **The capability number:** C's masked first_mae — vision-only
     first-frame prediction. Banked comparators on the identical rows:
     intact state-copy 2.4316 (the proprioceptive-extrapolation
     floor), A-s0 masked 23.8154, B masked 24.0783. Masked-C beating
     state-copy (< 2.43) would be a qualitative first (no existing
     arm is within 20 of it); pre-declared reporting thresholds:
     < 6.0 = strong vision capability signal, ≥ 15 = dropout failed
     to build one.
3. **Grounding read — C's intact panel first_mae vs A-s0's 3.9422**
   (the grounding-sensitive column; state-copy 2.6202 floor).
   Interpretive trap pre-declared: B (aux-off) IMPROVED first_mae
   (3.5009) by leaning on state — read 3 is only meaningful jointly
   with read 2 (better first_mae + collapsed reliance = vision did it;
   better first_mae + intact reliance = shortcut did it).

## Expectations (banked before launch)

- **E1 startup (hard gate):** 878 datasets / 42,872 episodes / dims
  6/6, identical to the batch; banner prints
  `state dropout: p=0.8`; `bijou_config.json` records
  `state_dropout: 0.8`. Any deviation ⇒ abort before step 1.
- **E2 throughput:** 0.4–0.6 s/step at B10 (idle box, no contention);
  sustained > 0.8 ⇒ input fix at a save boundary. VRAM < 76 GiB
  (state masking changes no tensor shapes; memory must match A-s0's
  profile).
- **E3 in-run probe (intact state, ±0.3 floor):** expect a trajectory
  ≤ ~1 above A-s0's (6.955@39k) — 80% of samples lose an informative
  input, some convergence cost is honest to expect. Soft: < 13 @10k,
  < 10.5 @30k. **Kill:** > 13 @10k after falling-then-rising; NaN;
  second OOM after the standing B−1 resume. **Formal final gate:
  probe < 10 @40k** — above that, p=0.8 is too aggressive at this
  scale and reads 1–3 still run (a negative is a deliverable), but
  no adoption path opens from this arm.
- **E4 panel (the honest prior):** chunk_mae C in **7.65–8.30** —
  centered slightly worse than A-s0 (information removed at train
  time; the literature's wins are OOD/spatial-generalization wins,
  which this in-distribution panel largely cannot see). first_mae
  anywhere in 3.4–4.4 given the read-3 trap. The pre-registered
  question is WHERE in the band it lands and what read 2 says, not
  whether C beats the baseline.
- **E5 reliance:** Δ_first(C) < 5.0 (sanity); masked-C first_mae is
  reported against the three banked comparators above, whatever it is.

## Cost & environment

~5.0 h train (40k × ~0.45 s/step, 1×H100, idle box GPU 0) + ~2.5–3.3 h
panel eval + ~25 min masked eval ≈ **8.5 GPU-h**, on a box otherwise
idle awaiting owner steer (the posted E4B follow-on recommendation was
"box to grounding arms" — this IS a grounding arm, from the same
mechanism chain). Disk: one run × 8 saves × 24G = 192G into 6.8T free.
Same corpus, same selection, no derived data ⇒ no new leakage surface
(E1 asserts selection identity with the batch). Box code fetched to
the commit carrying this post before launch (box idle — no live-run
sync); wandb `fontaine`, run name = save dir name, one lineage.

*Falsification honesty: this arm cannot distinguish "state dropout
builds vision grounding" from "state dropout merely tolerates masking"
if read 1 is neutral AND read 2's capability number stays near the
20s. That combination — trained-in mask tolerance without transfer to
intact-state grounding — is the pre-declared "mechanism inert"
outcome, and it kills the dropout leg as cleanly as a band miss.*
