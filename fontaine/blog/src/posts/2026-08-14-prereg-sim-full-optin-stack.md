# Pre-reg — full opt-in stack read (pricing the combined promotion)

*2026-08-14, drafted 11:0xZ, posted in-channel before the read. Queue
item `sim-full-optin-stack-read`. The owner has three appearance
promotions pending, each measured SEPARATELY:
[clutter real-crop patches](2026-08-13-prereg-sim-arm-split.md)
0.713→0.556 (fg-fix read, pre-reg'd in-channel 05:23Z 08-13),
[arm photometrics](2026-08-14-prereg-sim-arm-photometric-links.md)
0.713→0.698, and
[mount material](2026-08-14-prereg-sim-mount-material-split.md)
(alone n.s.; rides the material stack to 0.702). If the flags flip
together, the interactions are unmeasured — this read measures the
full stack in one paired harness.*

**Plain words.** We have three separate fixes that each make the
simulated top-camera view look more like the real robot's footage:
pasting photos of the real desk clutter over the rendered stand-ins
(the big win), fixing the arm's surface colors and shine, and fixing
the white camera bracket's color. Each was measured on its own. But
the plan is to turn them all on together, and improvements don't
always add up — one fix can hide or undo another. This read renders
100 paired scenes with everything off vs everything on and asks the
policy's own frozen vision encoder whether the combination is at
least as good as the best single fix, and whether the pieces still
add.

## Question

Does the full opt-in stack (clutter patches + `arm_photometrics='v1'`
+ `mount_material='v1'`) read at least as real as the best single fix
— i.e. do the material fixes still contribute on top of the clutter
patches, or do the promotions interact?

## Design

`fontaine/scripts/sim_full_optin_stack_read.py`: TWO production v3
instances (numpy post backend) over the same 20 seeds × 5 appearance
draws — default materials vs the two-flag material stack — both
hooked at `_composite` with the noise RNG state restored per arm (the
fg-fix harness, verbatim). Arms embedded:

- `v3` — baseline production output (bit-exact by construction);
- `patched` — baseline materials, no_clutter mask + the mined real
  crops pasted at the drawn poses: the **in-run replication of the
  banked best single** (0.5561);
- `stack_full` — THE ARM: graded materials + no_clutter mask + the
  pasted crops. The exact frame the combined promotion would make
  the production default.

Encoder probe: er_60k trunk, knn5 vs held-out real-B, the established
protocol. In-run oracles: per-slot qpos bit-equality across
instances (the grades consume no RNG); clutter draws + episode
affine bit-equal across instances; changed-px fraction between the
two production frames ≤ 30% (an RNG-stream divergence flips ~100% of
pixels through the sensor noise; the material grades touch only
arm/servo/mount pixels).

## Registered anchors / aborts

**ABORT (no claims) unless BOTH**: in-run `v3` knn5 AUROC in
**0.713 ± 0.005** (the established 20×5 gate) AND in-run `patched`
AUROC in **0.5561 ± 0.010** (the banked fg-fix best single must
replicate — it anchors the "beats best single" comparison).

## Decision rule (frozen before the read)

- **PRIMARY PASS** (as queued): paired Δknn5 CI95 (10k resamples,
  rng 0) of `stack_full` vs `v3` entirely **below 0** AND
  `stack_full` AUROC ≤ **0.5511** (banked best single 0.5561 − ε,
  **ε = 0.005** registered) → the combined promotion is priced: the
  stack beats every single fix, materials still pay on top of
  clutter.
- **CI below 0 but stack > 0.5511** → the stack helps vs v3 but the
  materials' contribution is absorbed/interacted away next to the
  clutter patches — the promotion case reduces to clutter-only
  first; flagged on the asks.
- **CI straddles or above 0** → interaction pathology (the stack
  reads LESS real than default despite three individually-good
  parts) — inspect the dumped frames before any claim; promotions
  flagged.
- **Record-only riders**: additivity — measured stack AUROC vs the
  additive prediction `v3_inrun − 0.1566 − 0.0103` (banked clutter Δ
  + banked material-stack Δ; ≈ 0.546 at a centered anchor), the
  deviation is the interaction term; paired `stack_full` vs
  `patched` (the materials' marginal contribution ON TOP of clutter
  — the banked material-stack rider CI [−2.45, −0.57]e-07 is the
  no-interaction reference); clean anchor.

## Scope notes

- No promotion has landed (all three flags are opt-in at HEAD; the
  asks are unanswered) — baseline is the production v3 default, per
  the queue item's re-scope clause.
- This is a TOP-camera read. The wrist side of the material stack is
  already measured (neutral at reset poses, wrist read 06:0xZ); the
  clutter paste is top-only by construction.

## Cost

CPU renders (100 slots × 2 instances + the paste arms) + ~0.02 GPU-h
embeds (300 sim + 400 real frames) on the er_60k trunk. R1-B owns
gpu0 at ~34 GiB / 80 — the embed job fits in the headroom and does
not perturb training (established: the texture-read embeds ran
alongside R1-A/B without incident).

---

## RESULTS (10:58Z 08-14, executed same session — MIDDLE BRANCH: stack beats v3, misses the best-single bar by 0.001)

All gates green, exit 0: in-run `v3` **0.7127** dead-center
(band 0.708–0.718); in-run `patched` **0.5561** — bit-matching the
banked fg-fix read (deterministic renders, the two harnesses agree
exactly, same cross-check shape as the wrist read's top rider); qpos
+ clutter draws + episode affine bit-equal across both instances ×
100 slots; changed-px tripwire quiet (max 12.3% of frame — the
material grades' arm/servo/mount footprint, nowhere near the ~100%
RNG-divergence signature).

- **PRIMARY — NOT PASSED, by the ε margin.** Paired Δknn5 of
  `stack_full` vs `v3`: **−2.075e-06, CI95 [−2.254, −1.891]e-06**,
  99/100 slots closer — massively below zero; the stack is far
  better than the current default. But `stack_full` AUROC landed at
  **0.5521** vs the registered bar **0.5511** (best single 0.5561 −
  ε, ε = 0.005): the stack beats the best single fix by only
  −0.0040, under the registered ε. Per the frozen middle branch:
  **the combined promotion is NOT priced as strictly better than
  clutter-only — the promotion case reduces to clutter-first, with
  the materials' add-on unresolved.**
- **The materials' marginal on top of clutter (record-only rider):**
  paired `stack_full` vs `patched` **−5.50e-08, CI95 [−1.44e-07,
  +3.37e-08]**, 56/100 — straddles zero. Against the banked
  no-interaction reference (materials vs v3 alone: mean −1.49e-07,
  CI [−2.45, −0.57]e-07): the marginal's mean is ~⅓ of the solo
  effect and its CI now includes 0. The material fixes' small,
  real solo effect is **attenuated ~3× and statistically absorbed**
  once the clutter patches are in.
- **Additivity (record-only):** additive prediction 0.5458, measured
  0.5521 → **interaction term +0.0063 AUROC (sub-additive)** —
  roughly two-thirds of the materials' banked contribution fails to
  survive composition with the clutter patches.
- **Mechanism sketch (not registered, offered for the asks):** the
  clutter patches remove the strongest fake cue; what remains is
  dominated by the rendered arm *geometry/relief* signature (the
  texture programme's surviving hypothesis), against which the
  ±few-count photometric grades are second-order. Consistent with
  the arm-split finding that patched surface 0.556 ≫ real-fg 0.328
  — the remaining gap lives in the foreground render itself, not in
  its albedo.

**Disposition.** Item closed. For the three pending promotion asks
(all still unanswered): the measured facts are (1) clutter patches
carry essentially the whole combined gain — promote first or alone;
(2) the material flags cost nothing when stacked (no regression;
CI straddles zero, point estimate still negative) but their measured
solo gain does not survive composition at n=100 — flipping them
together with clutter is *safe* but should not be sold as additive;
(3) a bigger-n read could resolve the residual marginal, priced
separately if the owner wants the material flags' stacked value
pinned before promoting. No auto-queued follow-up — the decision is
promotion-shaped and sits with the owner.

Artifacts: [analysis](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_full_optin_stack_read.json)
· [chart](https://mcobzarenco-fontaine-reports.static.hf.space/chart__full_optin_stack_read.png)
· [frame strip](https://mcobzarenco-fontaine-reports.static.hf.space/strip__full_optin_stack_read.png)
