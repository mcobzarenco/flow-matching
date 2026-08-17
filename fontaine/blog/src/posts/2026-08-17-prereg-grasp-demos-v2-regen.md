# Pre-registration: grasp-demos-v2 regen

*2026-08-17 06:16Z, posted in-channel (msg 1538793633703268372) before
launch; launched 06:16:38Z as unit `demo_gen_v2` on the A100 box.*

**Plain words**: we are re-generating the 5,000-episode scripted-expert
demo dataset with three improvements queued since v1 shipped — a
smoother, more accurate expert; the robot's camera bracket drawn the
way the real one looks; and the wrist camera moved to the pose we just
fitted against real rig footage. Everything else — spawn protocol,
tints, seeds, scale — is deliberately identical to v1, so any change
in the keep-rate or downstream training is attributable to those three
knobs.

## Command (dry-run verified)

```
sim.collect_demos_sharded \
  --out ~/datasets/fontaine/grasp_demos_v2 \
  --repo-id fontaine/grasp_demos_v2 \
  --shards 96 --target-kept 5000 \
  --seed-start 10000 --seeds-per-shard 2000 \
  --spawn-version v2.1 --tint-band mix70 \
  --bracket-appearance real --wrist-pose refit
```

96 shards round-robined over the box's 8×A100; SAME seed universe as
v1 (10000+, stride 2000) so spawn draws match and deltas attribute
cleanly (seed policy: same seeds for comparability — this is a regen,
not a resume).

## Changes vs v1 (all else identical)

- **expert v1.3** (`9ba7d30`): place-center bar 3→1.5 cm, retreat
  glide 5°/tick, tail budget 450 — bench kept 52.5%, parked 98.6%
  (n=120).
- **`bracket_appearance='real'`** (`4a9bf5c`): leader bracket hidden,
  follower ring filled; render-only, physics oracle-pinned.
- **`wrist_pose='refit'`** (`4b14b1f`): fitted wrist-cam pose (pitch
  −23°, yaw +14°, roll −9.5°, camera-frame offset +3.3/+1.3/−3.0 cm);
  held-out G2+G3 PASS, G1 −44.5% vs the −50% bar (disclosed) — riding
  per the 05:46Z ship-and-ride recommendation unless vetoed.

**Receipt**: launch HEAD `7078cf0` (banked as `expert_head` in
provenance; the driver manifest now carries `bracket_appearance` and
`wrist_pose` — pass-through plumbing landed this session, check.py 980
green).

## Anchors and gates

- **Kept-rate anchor 45.9%** — v1's realized rate (5,000/10,883
  attempted, same seeds). Expert v1.3's bench says ≥ that.
  **Halt-and-diagnose bar: sustained aggregate < 40% once ≥ 500
  attempted** — that is a render/protocol regression, not luck.
- **Complete**: 5,000/5,000 kept, 0 failed shards; provenance carries
  the three new knobs + expert head.
- **GPU-hours gate**: 40 (v1 ran 2h07m wall ≈ 16.9 GPU-h).
- **Boundary**: merge (`sim.merge_demo_shards`, bit-identical-oracle
  path) → upload public `fontaine-grasp-demos-v2` → dataset card +
  results post with the kept-rate verdict. ETA ~2.5–3.5 h from v1's
  wall clock.
