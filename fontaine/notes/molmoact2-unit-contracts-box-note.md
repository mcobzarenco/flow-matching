**Note for Fontaine — molmoact2 unit contracts and what they mean for sim
evaluation** (main @ `63155d4`; full detail in `bijou/eval/molmo_norm.py`'s
docstring and the 2026-08-12 eval
`reports/eval__molmoact2_so100_101_release__step_000000__holdout256_euler10.*`)

**Finding.** A molmo_flow/molmoact2 checkpoint's ONE global q01/q99 table is a
*unit contract*, and the two artifacts we converted carry near-disjoint
contracts. Release (`allenai/MolmoAct2-SO100_101`): lift box
**[+45.2, +186.1]**, elbow [+35.4, +173.6] — the degree-zero conventions
dominating older community data. Your rig-r1-step2000: lift
**[−103.7, +48.6]** — controller-native v3 units (the boxes overlap on a 3.4°
sliver). Both carry `norm_tag "so100_so101_molmoact2"`: **tag equality ≠ table
equality — key nothing off the tag name, read the values.** Mechanically,
truth outside the box is unreachable (predictions clamp into the box in
normalized space) and state outside the box is invisible (bins saturate at
the edge). On our curated-v0 256-frame panel this floored 52% of frames and
produced the release's 21.4 chunk MAE vs state-copy 7.3; per-frame charts
show near-perfect 30-step tracking on in-convention frames — the model is
fine where it can see and reach.

**For your sim, per checkpoint class:**

1. **ftrig checkpoints (rig-recomputed table)** — structurally sound
   end-to-end, assuming the sim actuates in the same controller-native units
   the rig recorded: sim state normalizes in-distribution, decoded actions
   come out in controller units. Two tripwires before trusting results:
   (a) check your sim task's joint trajectories stay inside the checkpoint
   box (print it from `bijou_config.json` / norm_stats; outside = blind +
   unreachable, a *structural* failure no training fixes — scene design
   should respect the box); (b) your own oob-plan rule generalizes: a
   first-action-vs-current-state check catches any unit/sign mismatch
   instantly (our release read had first_mae 18.0 vs state-copy 2.5 — that
   signature *is* the unit bug detector). Note only step2000 exists in our
   converted format; step 500/1000/1500 are upstream HF exports only.

2. **Released checkpoint, raw in a v3 sim — will not work, and the number
   would be meaningless.** v3 state (lift ≈ −30 at rest) sits below its box
   floor (+45) → state tokens saturate → the model is blind, then emits
   actions in its own convention → the controller receives ~90–180°
   equivalent offsets. You'd measure the unit mismatch, not the policy. This
   also retro-explains your rig zero-shot anchor (28.95 ≈ 3.2× state-copy):
   floor-dominated, same mechanism — and the "odd actions" early in your
   fine-tune (released weights suddenly decoding through the new rig table =
   convention-scrambled until relearned).

3. **Release with an on-the-fly unit shim — legitimate and cheap if you want
   a release-in-sim read.** The sim knows the exact per-joint affine both
   ways (degrees-with-offset-zeros ↔ calibrated-range→[−100,100]); wrap
   state-in (v3 → model units before its table normalization) and action-out
   (model output → v3 before the controller). Label it off-contract (never
   pool with contract reads; our eval suffixes `_convmap` for exactly this),
   and treat it as a lower bound — the release trained on a *mixture* of
   conventions through one table, so its outputs are mixture-blurred even
   under a perfect shim. Also check the *mapped* reachable set
   `A⁻¹(release box)` covers the sim task's workspace — the clamp travels
   with the model. (Precision: the real axis is *calibration convention*,
   not dataset format version — v2.1/v3 is a correlated proxy; the community
   corpus mixes conventions and one tick-scale outlier regardless of format
   version.)

**Cross-checkable anchors** if you want them: our stats-side estimator
(`fit_convention_map`) snapped the curated panel at lift +180°×46 frames,
wrist_roll ±90° wrap ×78, elbow +90°×28, exactly one untranslatable
tick-scale dataset (`willnorris/bbox-2`) — if your sim calibration implies a
materially different lift/elbow map for old-convention data, one of us has a
sign/offset wrong and I'd like to know which. Paired `--molmo-norm` arms on
the release are staged (`~/launch_eval_molmoact2_release_normarms.sh`,
pre-registered header) for whenever you release the GPU.
`