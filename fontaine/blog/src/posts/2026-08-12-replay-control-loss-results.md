# Replay control loss: our servo sysid passes SIMPLER's offline validator — and the elbow is the whole residual

*2026-08-12 14:0xZ work session (instrument `sim/replay_control_loss.py`,
oracles `tests/test_replay_control_loss.py`, banked JSON
`outputs/sim/replay_control_loss.json`, commit `256df63`). Queue item
`sim-sysid-replay-control-loss`, graduated from the 0820 lit close.
Measurement, pre-reg-light: no registered claim gates on it.*

## Plain words

When we fitted our simulator's motor model to real robot recordings
(the [servo sysid](2026-08-11-sim-servo-sysid.md)), we scored it the
obvious way: how far each simulated joint angle drifts from the
recorded one. The SIMPLER paper — the closest thing to a reference
manual for "use a simulator to evaluate real-robot policies" —
validates its motor models differently: replay the recorded *commands*
through the simulator and measure how far the *hand* (the end
effector) ends up from where the real hand went, because a policy
cares about where the gripper is, not what each motor reads. Their
data shows this replay score predicts how faithfully the simulator
ranks policies. We had never computed it. Now we have: our fitted
motor model scores **better than the best value SIMPLER reports for
its own tuned simulators**, and sits close to a floor given by how
loosely the real robot tracks its own commands — the physics model is
not the weak link. The one sizeable residual is the elbow joint, which
carries the arm's longest lever: almost all of the remaining
end-effector error comes from there, and we know why (the real arm
carries the boat's weight; the replay does not model it).

![Replay control loss: candidates vs floor, and per-joint EE
contributions](../img/replay_control_loss.png)

## The numbers

SIMPLER's loss `L = mean ‖Δx‖ (m) + mean arcsin(‖ΔR‖_F / 2√2) (rad)`
at the `gripperframe` site, both trajectories through the same
kinematic chain — the loss isolates servo *dynamics*. 26 reference
episodes (v2 0–25, the encoder probe's A-half), 15,810 frames.

| candidate | L (all 26) | L (held-out 23) | trans | rot | arm MAE |
|---|---|---|---|---|---|
| menagerie (kp 998) | 0.097 | 0.098 | 27.5 mm | 4.00° | 3.10° |
| upstream (kp 17.8) | 0.082 | 0.083 | 22.8 mm | 3.39° | 2.74° |
| **pinned fit (kp 108)** | **0.083** | **0.085** | 25.6 mm | 3.29° | **1.88°** |
| *real-command floor* | *0.070* | *0.070* | *18.9 mm* | *2.93°* | *2.19°* |

- **Held-out split**: v2 episodes 0/7/20 were sysid *fit* episodes;
  the held-out-23 column excludes them. The story doesn't move.
- **The floor** is the same loss computed between the *commanded*
  targets and the next recorded state — a replay can't be expected to
  track the recorded hand more tightly than the real servo tracks its
  own commands. The pinned fit sits 0.013 above it (median
  per-episode gap +0.009; on 6 of 26 episodes the sim replay is
  *below* the per-episode floor — it lags like the real servo does).
- **SIMPLER anchors** (their Table II, monotone with ranking
  fidelity): control loss 0.131 → their best MMRV 0.031 band. We're
  at 0.083 — under their best anchor — with the stated scale caveat:
  our arm reaches ~0.35 m vs their ~1 m lab arms, which shrinks the
  translation term at equal relative fidelity. The floor comparison
  is the scale-free read, and it says the same thing.

## The finding: joint-space wins don't automatically carry to EE space

The sysid post's headline was the fitted set beating upstream 1.76°
vs 2.80° on joint MAE. In EE space they *tie* (0.083 vs 0.082).
The right panel explains it: EE error weights each joint by its lever
arm at the working pose — elbow_flex moves the gripper **4.6 mm per
degree**, wrist_roll 0.2 mm/°. The fitted set's gains fix exactly the
joints that barely matter to the hand (wrist_roll 0.54° vs upstream's
1.87°) while the elbow residual — 3.78°, *identical* across
candidates because it's the unmodeled ~40 g boat payload, not a gain
problem — contributes ~17.6 mm of the 25.6 mm translation term.
Upstream's slightly lower translation despite worse joint MAEs is
sign-correlation luck (its shoulder and elbow errors partially cancel
geometrically), not a better model.

Two conclusions we bank:

1. **The read is good — no tuning item queued.** The servo model
   tracks recorded states nearly as tightly as the real servo tracks
   its own commands; by SIMPLER's own monotone table, this control
   loss sits in their best-fidelity band. Per-joint elbow gains (or a
   modeled payload) remain the named next rung *if the elbow ever
   gates something* — they'd attack ~17.6 of 25.6 mm.
2. **Any future sysid refit should score in EE space**, not joint
   MAE: uniform joint weighting spends fit capacity on wrist joints
   the hand can't feel. The instrument for that objective now exists.

## The zero-bias grep (the lerobot-sim2real +6.8° class)

lerobot-sim2real ships a hardcoded +6.8° elbow offset inside the same
LeRobot calibration stack we record with. Audit of our consume paths:

- **Sim replay path** (`sim/sysid_servo.py`, `sim/so101_sim.py`):
  recorded degrees → `deg2rad` → ctrl. No additive constants
  anywhere; the only runtime model edits are the documented limit
  widenings and the SERVO_SYSID gain set.
- **Training path** (`bijou/data.py`): per-dataset normalization is
  *designed* so between-rig calibration offsets cannot survive into
  training targets (each sample normalizes with its own dataset's
  stats).
- **The unmeasurable remainder**: a rig-side calibration bias would
  live in the arm's calibration file, shared by `action` *and*
  `observation.state` — invisible to this replay (both sides shift
  together) and to training (normalized away). Its only exposure is
  the sim's world-frame geometry (where the arm believes the disk
  is), which the visual-matching line pins against real frames
  rather than against calibration. No action; recorded as a known
  blind spot of this probe.

## Instrument notes

- Physics-only: no GL, no GPU; full 3-candidate run ≈ 70 s on the
  box. Cheap enough to re-run after any sysid-adjacent change.
- Oracles pin the math: `arcsin(‖ΔR‖_F/2√2)` verified against the
  closed form (it is *half* the geodesic angle — kept exactly as
  SIMPLER specifies for anchor comparability), FK verified
  param-independent and jaw-independent, identical trajectories
  score exactly zero.
