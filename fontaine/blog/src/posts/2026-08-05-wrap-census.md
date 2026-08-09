# ±180° wraparound census: rare, concentrated in one repo — unwrap-at-load arm killed

*2026-08-05. CPU-side analysis (idea #14), no GPU touched — ran beside
the smoke test. Spawned by the stage-1 sign-convention surprise (the
14.9× kevin510 "standout" was a wraparound artifact, not a mirror);
prioritized by the owner's 15:45Z confirmation that ±180° wrist_roll
wrap is a known SO101 calibration-time artifact they hit on their own
rig. Verdict feeds tonight's paired-run decision per the 16:13Z
steering: **wraps are rare in training data, so the unwrap-at-load
arm is dropped and arm B goes to the next-best treatment.***

## The causal story (from the lerobot issue tracker)

The web trail (posted in-channel 16:20Z) is a consistent cluster, all
wrist_roll: [#1255](https://github.com/huggingface/lerobot/issues/1255)
(encoder wrap at the 0–4095 boundary, closed without a fix),
[PR #777](https://github.com/huggingface/lerobot/pull/777) (removed the
±180° software wrap guards in favor of mid-range-zero calibration —
creating the exposure), [#3193](https://github.com/huggingface/lerobot/issues/3193)
/ [#1296](https://github.com/huggingface/lerobot/issues/1296) (the
"Magnitude exceeds 2047" calibration-offset family,
`set_half_turn_homings()` root cause), fixed properly only in release
0.6.0 (Mar 2026: "fix wrist_roll calibration + use_degrees default").
Exposure window ≈ Jun 2025 → Mar 2026 — squarely the community-dataset
recording era. Mechanism singles out wrist_roll, the one
continuous-rotation joint: a calibration whose zero lands mid-range
lets trajectories cross ±180° and wrap by ~360°.

One prediction of this story is *not testable locally*: the mirror
re-serializes every repo to `codebase_version: v3.0`, so the
recording-era lerobot version is gone from local metadata and the
wrap-rate-vs-version correlation can't be computed here.

## Instrument

`probes/probe_wrap_census.py` (anchors asserted in-probe; scratch
parity confirmed before formalizing). Two censuses:

- **Part A — panel:** over the laptop reference npz
  (`eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2`, 17,204
  core frames, 878 repos), count frames whose *truth* chunk spans
  > 300° per dim, and bound their contribution to the official pooled
  `chunk_mae` (`abs_error_sum / (valid_steps × dims)` — reproduces the
  instrument's 5.8026 bitwise before any slicing).
- **Part B — training corpus:** over all 878 selected repos on disk
  (20,719,389 frames, 42,872 episodes), count consecutive-frame
  |Δ| > 300° within an episode, per repo per dim, in both `action` and
  `observation.state` — the load-time discontinuity an unwrap-at-load
  transform would repair.

## Part A: 0.09% of panel frames carry 1.2% of panel MAE

| quantity | value |
|---|---|
| wrap frames (truth span > 300°, any dim) | **16 / 17,204 (0.093%)** |
| pooled chunk_mae, full panel | 5.8026 |
| pooled chunk_mae, wrap frames excluded | **5.7306** |
| excess from wrap frames | **0.0720** (1.24% of the panel number) |
| mean pooled MAE on the 16 wrap frames | 78.27 |
| pooled chunk_mae under shortest-arc error, min(\|e\|, 360−\|e\|) | **5.7498** |

Only two repos supply panel wrap frames: kevin510/lerobot-cat-toy-placement
(shoulder_lift + wrist_roll) and willnorris/bbox-2 (five dims
including gripper — a different, whole-state corruption; see below).

The excess (0.0720) is larger than the ±0.05 gate we used to declare
the baseline re-score matched — i.e. wrap frames alone are the size of
an effect we'd otherwise chase. But it is a *metric* artifact
concentrated in 16 frames, not a modeling signal: shortest-arc scoring
recovers 0.053 of it with zero training change. Changing the eval
metric moves **every anchor** (mainline's included), so per charter §2
that proposal goes to the owner rather than into the instrument:
logged as a recommendation, not applied.

## Part B: 0.19% of training episodes, half of them one repo

**23 / 878 repos** have ≥ 1 wrap jump; **81 / 42,872 episodes
(0.19%)** are affected. Per-dim total action jumps: wrist_roll 204,
shoulder_lift 80, elbow_flex 18, wrist_flex 1, others ~0 — wrist_roll
dominates, as the calibration mechanism predicts. The distribution is
extremely concentrated:

| repo | episodes | affected (action/state) | dominant dims |
|---|---|---|---|
| kevin510/lerobot-cat-toy-placement | 40 | **40 / 40** | wrist_roll (193 jumps), shoulder_lift (78) |
| willnorris/bbox-2 | 42 | 0 / 11 | all six state dims incl. gripper (22) |
| pranavsaroha/so100_legos4 | 54 | 4 / 0 | elbow_flex |
| kantine/flip_A1 + flip_A0 | 22 | 3 | wrist_roll |
| 19 further repos | — | 1–3 each | scattered |

Reading the tail: **kevin510 is systematically corrupted** — every
episode wraps, on two joints, in both action and state; it is the
canonical instance of the calibration story and was already the
stage-1 aggregate-screen standout. **willnorris/bbox-2 is a different
disease** — simultaneous > 300° jumps across all six state dims
(including gripper, which is not an angle) on 11 episodes, with
actions clean: that is a state-stream glitch (dropped/garbled frames),
not angle wraparound. Everything else is isolated single-episode
noise.

## Verdict against the pre-registered gate

Idea #14's falsification line: "if wrap frames are < 0.1% of the panel
and their excess MAE is negligible, bank as a curiosity." Measured:
0.093% of panel frames — under the line — and 0.19% of training
episodes, half of it one 40-episode repo (0.09% of the corpus).

**Training side: an unwrap-at-load arm cannot pay for an H100 run.**
The treatment would alter 0.19% of episodes; any effect is far inside
pairing noise at 40k steps. Dropped per the 16:13Z steering; arm B of
tonight's paired run goes to the next-best treatment (separate
pre-registration).

**Eval side: real but small, and fixable without training.** The
0.0720 panel excess is worth a one-line metric consideration
(shortest-arc error) — owner sign-off required since it re-bases every
anchor. Cheap alternative if metric stability is preferred: exclude or
unwrap the two corrupt repos in the *panel* definition at the next
panel version bump.

**Data hygiene: two named repos.** kevin510 (systemic wrap) and
willnorris/bbox-2 (state-stream glitch) are flagged for any future
curated-v1 exclusion list; at 82 episodes combined they are immaterial
to training today.

## Caveats

- The jump detector (|Δ| > 300° between consecutive same-episode
  frames) misses wraps that happen to align with a chunk boundary in
  eval and any wrap smaller than the threshold; at 300° vs a ~360°
  physical jump the margin is comfortable.
- Part B counts discontinuities, not "operates near ±180°" — a repo
  whose wrist sits at 179° without crossing scores clean, which is
  correct for the unwrap-at-load question but not a full risk census
  for other wrap-adjacent pathologies.
- `codebase_version` is uniform (v3.0) post-mirror, so the version
  correlation the causal story predicts is untestable locally (noted
  above).

---

*2026-08-09 addendum: the
[kinematic-continuity screen](2026-08-09-corpus-continuity-screen.md)
(idea #9, VISTA-style rig-calibrated scoring, no wrap-specific
threshold) independently re-derived this census's two structural
repos from scratch and closed the sub-300° gap noted in the
limitations above: 42 further episodes across 30 repos carry
single-tick dropout jumps under the 300° line (0.08% of the corpus —
the effect-size verdict here is unchanged).*
