# Pre-registration DRAFT: the Squint-twin qualification screen — first use of the SO-101 digital twin, relative reads only

*Draft cut 2026-08-22 00:2xZ (work session riding the gripfix
endpoint battery), queue item `squint-twin-screen-prereg` — the
[eval-design v0](2026-08-21-vla-eval-design-v0.md)'s named slot 2.
**NO launch from this item.** DRAFT status: the reads, gates, and
expectation grid below are frozen now and do not move; the
finalization amendment (before any launch, in-channel per the
standing no-GO-ask rule) fills exactly the named slots — the
preflight-2 receipts (§7), the adaptation-recipe command block, the
frozen instruction strings, and the pair-2 slot that waits on
tonight's gripfix frozen-grid verdict. Substrate:
[Squint](../papers/squint.md) (MIT, local checkout `~/squint`),
preflighted GO 2026-08-14
([note](2026-08-14-squint-twin-preflight.md)).*

**Plain words.** We have a second simulator available: a digital twin
of exactly our robot arm, published by another lab, with automatic
success grading. Before ever letting it influence a decision, we have
to qualify it — the way you'd qualify a new measuring instrument
against one you already trust. The plan has three gates. First,
mechanical: prove our policies' outputs actually drive the twin
correctly (units, joint order, camera plumbing), by replaying a real
robot episode inside it. Second, a positive control: teach one of our
models the twin's alien-looking world with a small dose of in-twin
demonstrations — if even an adapted model can't score, the twin can't
referee anything and we close the screen as a broken instrument.
Third, the actual measurement: take two of our checkpoints whose
quality difference we know precisely from our own simulator (28/100
vs 8/100 at grasping), adapt both identically, and ask whether the
twin sees the same ordering. If it does, we've bought an independent
second opinion for future model comparisons — plus an unlimited
supply of ground-truth-labeled robot videos for calibrating failure
detectors. If it doesn't, we record that and the twin stays a toy.
Absolute scores in the twin are never claims — only differences
between our own models, measured under identical conditions.

## The tier decision, resolved (the item's first deliverable)

The preflight note's rule (2026-08-14): *"Squint is the successor
tier if the wrist-transfer screen hits F-instrument or the success
floor holds."* Both branches have since fired:

- The [wrist-transfer screen closed
  **F-instrument**](2026-08-15-wrist-screen-results.md) (08-15): its
  control couldn't certify its own instrument at n=25, and the
  success-rate form of the transfer question was left standing.
- Its successor-requirement #1 — *"a competence floor first: re-screen
  on a policy whose success rate can actually move"* — is now met:
  the grasp-SFT lineage has a 28/100 checkpoint
  ([onerig](2026-08-19-prereg-demos-plus-one-rig.md)) and a certified
  20-point paired spread below it.

**Tier decision: GO for qualification.** Not GO for judging: per
[eval-design v0](2026-08-21-vla-eval-design-v0.md), the twin's
registered role until this screen passes is *none* — it screens
nothing, gates nothing. This pre-registration IS the gate before
first verdict-adjacent use.

## The question and the registered role

One question: **can the Squint twin, with the domain gap held
constant across arms, reproduce a capability ordering that sim100
certified?** Pass → the twin registers in eval-design v1 as a
**relative screening instrument** (A/B deltas between our
checkpoints, never absolutes, never a verdict-gater). Fail → it
registers as a labeled-rollout generator only (idea #6), and the
menu's next tier stays sim100-only.

## Gate 0 — mechanical adapter (CPU, no policy)

The adapter (a ~100-line gym loop per the preflight) carries four
frozen conversions:

- **Units + joint order:** policy outputs LeRobot servo degrees; the
  twin's `pd_joint_pos` consumes absolute radians
  (`normalize_action=False`, bounds = joint limits). Arm joints:
  `deg2rad`. Gripper: the twin's own deploy path encodes a
  gripper-specific affine (sim convention −10°..120° ↔ servo open
  constant; `deploy_utils/manipulator.py:49-52,135`) — we invert it
  with OUR calibration's open value (demos command exactly {0,
  41.69}); the two affine constants are a preflight-2 receipt.
- **Rate:** our chunks are 30 Hz (sim100 `CONTROL_HZ = 30`); the twin
  is hard-pinned at `control_freq=10` (`envs/base_random_env.py:148`).
  Frozen: **subsample every 3rd action**, execute 10 env-steps per
  replan — 1 s wall-equivalent per replan, the same cadence as
  sim100's execute-horizon 30. The published twin dynamics are not
  touched (no `SimConfig` overrides).
- **Cameras:** the dual-camera subclass the preflight priced (wrist
  mount + third mount both exist in the base classes), 224×224 via
  `sensor_configs`, `apply_overlay=False` (raw renders — the
  `rgb+segmentation` trap is moot), `domain_randomization=False`
  (deterministic paired seeds, verified in preflight). Wrist frame →
  `[wrist camera|…]`; the third-person frame's kind tag is judged by
  `resolve_camera_kinds` on the saved preflight frame — a
  preflight-2 receipt, frozen at finalization, identical across arms.
- **Decode + stats:** sim100 protocol constants verbatim — euler-10,
  bf16 decoder, each checkpoint's own training stats
  (`grasp_demos_v2/merged` worn-row rule carried).

**Gate 0 read (pass/fail):** (a) determinism entry gate — one seed
rolled twice, per-step obs bit-equal (wrist-screen precedent); (b)
**rig-episode replay** — a banked `grasp_demos_v2` episode's action
trace driven through the adapter into the twin: tracking p50 error
< 0.05 rad across arm joints, no joint-limit violations, gripper
open/close events reproduced. Fail → fix or close; nothing downstream
launches on a failed adapter.

## Gate 1 — the sim-adaptation sanity arm (the positive control)

Separates "renderer is alien" from "policy is bad". The wrist
screen's successor rule applies: **the control runs at treatment n**
(n=100), not a cut-rate pilot — a control that can't fire is
decoration.

- **Demo source:** the repo ships no policy checkpoints — we train
  Squint's own state-based SAC expert per task (`train_squint.py`,
  2–9 min/task on a 3090-class GPU; priced ≤0.5 GPU-h for the 2-task
  set), then roll it out and keep ~100 success episodes/task,
  re-rendered through OUR adapter config (dual-camera 224, raw, DR
  off) — the twin gives ground-truth success labels for free.
- **Conversion:** twin episodes → LeRobot format (actions/state back
  in servo-degree convention through the inverse adapter — the same
  constants Gate 0 verified). Conversion oracle: one episode
  round-tripped frame- and action-bit-exact.
- **Adaptation:** short SFT via `bijou.train` (the only training
  path), starting from each arm's checkpoint, **one frozen recipe,
  identical across arms** — same steps, LR, seed, data; no tuning,
  no second attempt. Budget hard cap ≤2.5 GPU-h/arm; exact step
  count in the finalization command block.
- **Gate 1 read (pass/fail):** the adapted *stronger* arm scores
  **≥20/100** on its best task after the band pilot (below). Fail →
  **F-instrument**: the twin cannot be made legible to our stack at
  this budget; screen closes, no relative read is attempted, the
  labeled-rollout deliverable (#6) survives.

## Gate 2 — the qualification read (the screen proper)

- **Pair 1 (frozen now, verdict-independent):**
  `grasp_sft_v2_joint_1gpu_pdnorm_onerig` @3000 (**28/100**) vs
  `…_democlean` @3000 (**8/100**) — the maximum-contrast banked pair,
  20 points on paired seeds, and (pending tonight's gripfix verdict)
  a gripper-amplitude mechanism that Lift/Place tasks exercise
  directly. **Pair-2 slot:** filled at finalization from the gripfix
  frozen-grid verdict (≥20 → gripfix joins as the hygiene-rule arm);
  pair 2 is optional and never blocks pair 1.
- **Tasks:** `SO101LiftCube-v1` primary, `SO101PlaceCube-v1`
  secondary — piloted by the Gate-1 adapted arm into the **20–80%
  band** (the banked bench rule); out-of-band → substitution ladder
  Reach (easier) / Stack (harder), frozen order, logged.
- **Cells:** n=100 paired seeds (seeds 0–99), DR off, both arms
  adapted by the identical Gate-1 recipe. The **unadapted** pair
  runs the same cells as a free record-only rider (expected at/near
  floor; if it isn't, that's worth knowing and costs nothing extra).
- **Primary read:** paired per-seed **McNemar** on success, adapted
  onerig vs adapted democlean, primary task — the standing sim100
  machinery (`sim100_paired_read.py` class).
- **Co-primary (rig-grammar rehearsal):** per-episode
  time-to-predicate curves from the twin's honest `info` predicates
  (`reached_object` → `is_item_grasped` → `item_lifted`),
  Kaplan–Meier with timeouts right-censored + KS with
  episode-clustered bootstrap — the eval-design tier-3 analysis run
  for the first time on cheap rollouts. Full CDF panel published;
  no scalar summaries without it.

**Frozen expectation grid:**

| outcome (primary task, adapted pair) | verdict |
|---|---|
| Δ(onerig−democlean) > 0, McNemar CI excl. 0 (or KS agrees, pre-reg direction) | **ordering preserved** — twin registers in eval-design v1 as a relative screening instrument |
| CI straddles 0 | **underpowered/insensitive at n=100** — twin stays non-instrument for this capability class; record, no claim either way |
| Δ < 0, CI excl. 0 | **substrate divergence** — loudest possible flag; both reads recorded, NO ruling on which substrate is right (only the rig can arbitrate); twin stays non-instrument |

## Budget and gates

SAC experts ~0.5 + adaptation ≤2.5×2 + rollout inference (~800
episodes × ~5 replans × ~0.6 s + pilots) ~1 + margin ≈ **cell gate
≤7 GPU-h**, abort-and-close at the gate. Sim overhead itself is
noise (1.35 s/ep at the CPU floor). Every GPU leg launches
`systemd-run` detached at a free window, policy-server checked, per
the standing rules; the twin never preempts a sim100 verdict leg.

## Preflight-2 receipts (CPU, before finalization)

1. Dual-camera subclass renders both views in one env (saved frames).
2. `resolve_camera_kinds` judgment on the third-person frame → frozen
   kind tag.
3. Gripper affine constants extracted (ours: open 41.69; twin sim
   convention −10..120) + Gate-0 replay green.
4. LeRobot conversion oracle green on one round-tripped episode.
5. `train_squint.py` smoke (imports + a few CPU env steps; no GPU).

## Claims contract (frozen)

1. **Relative deltas only**, domain gap held constant across arms.
2. **Absolute twin numbers are never claims** — the paper's own
   in-domain BC baseline at 41.9% sim success is the permanent
   warning label; ours sit under a far larger gap.
3. **No sim-to-real claims** — the twin is visually far-OOD from
   both the rig and sim100; nothing here measures transfer.
4. **One role per instrument:** this screen can promote the twin to
   *relative screening instrument*, nothing higher; twin reads never
   gate a sim100 verdict, and a passed screen is not license to skip
   sim100 — it adds a second opinion, it replaces nothing.
5. **Silence proves nothing:** a null Gate-2 read demotes the twin,
   it does not exonerate either checkpoint.

## What this feeds

Eval-design v1 slot 2 (this pre-reg), idea #16 (the substrate menu:
sim100 / twin / rollout-free / rig), idea #6 (the labeled-rollout
corpus: every twin episode banks frames + per-step predicates +
ground-truth success — detector calibration data at unlimited
volume).

---

## Appendix (2026-08-22 00:4xZ, same session): preflight-2 receipts — EXECUTED

*The §"Preflight-2 receipts" list, run CPU-only in the twin's venv
while the gripfix battery rode gpu0 (CUDA masked + lavapipe, GPU
untouched). Probe: `fontaine/scripts/squint_preflight2.py`; facts:
`outputs/squint_preflight2/facts.json` (mirrored to
[fontaine-reports](https://mcobzarenco-fontaine-reports.static.hf.space/squint_preflight2/facts.json)).
These are receipts, not results — the finalization amendment freezes
the two re-priced lines below before any launch.*

- **R1 dual-camera subclass: GREEN.** One env serves both views at
  (1, 224, 224, 3) — the wrist camera inherited (mount + per-step
  pose update), a static `third_camera` added on the base env's
  existing `camera_mount` with `ThirdCameraEnv`'s published
  pose/FOV constants. ~25 lines, as priced.
  [Wrist](https://mcobzarenco-fontaine-reports.static.hf.space/squint_preflight2/dual_base_camera_224.png) ·
  [third](https://mcobzarenco-fontaine-reports.static.hf.space/squint_preflight2/dual_third_camera_224.png).
  **Kind-tag observation for the frozen slot:** the third view is an
  elevated three-quarter view (eye 0.25 m up, ~40° down), not an
  overhead — the honest vocabulary choice is `front` (or `side`),
  not `top`; our checkpoints trained on top+wrist only, so the tag
  is OOD either way. Frozen at finalization.
- **R2 determinism entry gate: GREEN.** Same seed + same 10-step
  action sequence twice → sensor bytes and qpos bit-equal (DR off).
- **R3 Gate-0 rig-episode replay: tracking GREEN, limit line needs
  the re-price.** `grasp_demos_v2/merged` ep 0 (449 frames @30 Hz →
  150 twin steps @10 Hz) through the frozen mapping: **tracking p50
  0.0025 rad** (gate <0.05 — 20× under), p95 0.148 (fast-transient
  lag, consistent with 10 Hz steps over a 30 Hz path), gripper
  transitions **3/3 commanded→achieved**. But the drafted "zero
  limit violations" line is unmeetable as written: **41 clips**, all
  ≤0.047 rad — forensics: 36× `wrist_flex` grazing its limit by
  ≤0.0032 rad, 3× `shoulder_lift` at ≤0.0471 rad (the twin's
  −1.7453 rad limit sits ~2.7° inside our demo's deepest reach
  −1.792), 2× `elbow_flex` ≤0.003. A banked substrate fact, not an
  adapter bug. **Finalization re-price (named slot):* Gate-0 limit
  line becomes "no arm clip > 0.05 rad"* — preserving the intent
  (catch unit/order/sign errors, which produce radian-scale clips)
  without failing on the twin's genuinely tighter joint limits.
- **R4 `train_squint.py` smoke: GREEN after dep install** (`wandb`,
  `tensordict`, `torchrl`, `tqdm`, `torchvision` added to the
  isolated venv — the 08-14 preflight installed eval deps only).
  **Recorded venv drift:** the `torchrl` install upgraded the venv's
  torch 2.6.0-cpu → 2.13.0+cu130. All receipts re-taken under the
  final stack; the replay statistics are **bit-identical** across
  the torch change (CPU PhysX physics unmoved). The exec session
  inherits the venv as-is; `CUDA_VISIBLE_DEVICES` masking is what
  keeps CPU legs off the GPU now that the build can see it.

Open before FINAL: the LeRobot conversion oracle (needs the
conversion tooling the exec item builds), the adaptation-recipe
command block, instruction strings, the third-camera kind tag, the
Gate-0 limit-line re-price above, and the pair-2 slot (gripfix
verdict, tonight).

---

## FINALIZATION AMENDMENT (2026-08-22 02:5xZ) — all named slots frozen; launch follows this post

*Exec session `squint-twin-screen-exec` part (b), announced
in-channel per the standing no-GO-ask rule. Every open slot above is
resolved here; nothing else in the pre-registration moves. The
pipeline below was smoke-tested end-to-end (junk 30k-step expert →
rollout → re-render → LeRobot → oracle) before this amendment;
smoke artifacts deleted, receipts quoted where they price a frozen
line.*

**Slot 1 — Gate-0 limit line (re-priced as named):** "no arm clip
> 0.05 rad". R3 stands green under this line (41 clips, all ≤0.047
rad, twin-limit geometry not adapter error).

**Slot 2 — third-camera kind tag: `front`.** Dataset key
`observation.images.front`, mirroring `grasp_demos_v2`'s own key
pair — the twin dataset wears exactly `{wrist, front}` like the rig
demos (twin `base_camera` → `wrist`, `third_camera` → `front`).

**Slot 3 — instruction strings (frozen):**
- LiftCube: `Pick up the red cube.`
- PlaceCube: `Pick up the red cube and place it in the bin.`

**Slot 4 — conversion spec + oracle re-price.** Twin episodes (10 Hz)
→ LeRobot v3 at **fps 30 by repeat-3 upsampling**, so the deploy
adapter's subsample-every-3rd *exactly inverts* the conversion.
Actions = the rollout controller's absolute radian targets
(target-delta control is `PDJointPosController` with
use_target/use_delta — `_target_qpos` IS the pd_joint_pos-equivalent
command), inverse-adapted to servo degrees; state likewise from qpos.
Oracle re-price: the drafted "action-bit-exact" becomes **round-trip
max |Δ| < 1e-5 rad** (deg↔rad float paths aren't bit-stable; smoke
measured 6.6e-8), frames bit-exact at the pre-encode boundary, and
decoded-video PSNR recorded as a fact, not a gate — our own demos
ride the same mp4 path (smoke: ~36–40 dB). Dataset:
`~/datasets/fontaine/squint_twin_demos_v1`, ≤100 episodes/task.

**Slot 5 — expert + collection config (frozen).** Experts:
`train_squint.py` paper defaults with `--no-env-domain-randomization`
(DR off everywhere in this screen), seed 1, no wandb, ckpt = last
100k-step eval save, hard cap 45 min/task. Collection: rollout in the
expert's native env config; re-render through the dual-camera 224 raw
DR-off `pd_joint_pos` env, same seed + `set_state_dict`; the **final
success label is the re-render's own end-state success** (episodes
losing success in replay are dropped; keep-rate + replay-divergence
receipts banked — smoke divergence p50 0.0038 rad). Per-step honest
predicates banked per episode (the idea-#6 corpus starts here).

**Slot 6 — adaptation recipe command block (frozen, identical both
arms).** Priced from the pdnorm lineage's measured 16.2 s/step at
eff-96: **500 steps ≈ 2.25 GPU-h/arm** ≤ the 2.5 cap.

```
uv run python -m bijou.train \
  --train-data ~/datasets/fontaine/squint_twin_demos_v1 \
  --init-from ~/checkpoints/finetune/grasp_sft_v2_joint_1gpu_pdnorm_{onerig|democlean}/step_003000 \
  --objective joint --joint-ce-weight 1.0 --insulate-flow \
  --recompute-stats --per-dataset-flow-norm \
  --flow-decoder-init inherit --image-augment 0.8 \
  --decoder-lr 5e-5 --backbone-text-lr 1e-5 \
  --steps 500 --batch-size 96 --backward-chunks 8 \
  --activation-checkpointing --offload-optim --prune-superseded-optim \
  --holdout-episodes 0.1 --eval-every 100 --eval-samples 256 \
  --save-every 250
```

Adapted arms wear their own recomputed stats (the twin rows); the
unadapted record-only rider wears the sim100 worn-row rule verbatim.

**Slot 7 — pair-2: EMPTY.** The gripfix frozen-grid verdict landed
**5/100 (≤10 band)**; the frozen rule ("≥20 → gripfix joins") does
not fire. The screen runs pair 1 only.

**Launch plan:** leg A now (experts → collection → conversion →
oracle, one detached unit, ~0.8 GPU-h); leg B (adaptation, both arms
chained, ~4.5 GPU-h) after the oracle receipt is verified; Gate-2
rollout harness at the next window. Cell gate ≤7 GPU-h unchanged.

---

## RESULTS (2026-08-22 13:57Z; appended 2026-08-22 23:0xZ work session): Gate 1 FAILED — screen closed F_INSTRUMENT

*Full record: [results page](2026-08-22-squint-screen-results.md).
Verdict artifact `outputs/squint_screen/eval/gate1.log`.*

- **Gate 0: PASS** (preflight-2 receipts + leg A oracle — replay
  tracking p50 0.0025 rad, no arm clip > 0.05 rad, round-trip
  6.5e-8 rad, frames bit-exact pre-encode).
- **Leg A COMPLETE** 05:43Z (~1.9 GPU-h): experts success 1.00 both
  tasks; `squint_twin_demos_v1` banked, 100+100 episodes with
  per-step ground-truth predicates.
- **Leg B COMPLETE** 13:33Z (r4 full retrain ~4.4 GPU-h after the
  ENOSPC + [resume-bug](2026-08-22-offload-mirror-bug.md) incident,
  ~2.7 GPU-h re-spend, crossing recorded in-channel): frozen Slot-6
  recipe both arms, probe twins onerig 2.47@500 / democlean
  2.5187@500.
- **Gate 1: FAIL.** Adapted onerig @500: band pilots 0/20 both tasks
  (BELOW_BAND), cells completed to n=100 anyway — **0/100 lift,
  0/100 place** vs the ≥20/100 best-task bar → `FAIL_F_INSTRUMENT`,
  the pre-registered valid end. Milestone forensics: reached_object
  20/100, grasps 3–4/100, item_lifted 7–10/100, success 0 — partial
  competence, not a transport flatline. **No relative read attempted;
  Gate-2 spend skipped; substitution ladder (Reach) logged, never
  auto-run** — a future pre-registered session's call.
- **Registered role per the frozen fail branch:** the twin is a
  labeled-rollout generator only (idea #6 — the corpus survives and
  banked its first 400 predicate-labeled episodes). It screens
  nothing, gates nothing; the next eval tier stays sim100-only.
  onerig 28/100 vs democlean 8/100 stands exactly where sim100
  certified it.
