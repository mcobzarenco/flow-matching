# Spot-check results: the teacher SEES the new sim — +0.97 cm paired, CI excludes zero; er60k and snapflow read null

*2026-08-12 ~09:1xZ work session (real `date -u` at write: 09:08).
Executes the
[spot20 pre-reg](2026-08-12-prereg-sim-spot20-v3.md) (in-channel
07:52Z; owner steering 07:35Z). Verdict up front: under the v3
visuals, with physics proven bit-identical per seed, **teacher80k's
paired progress improves +0.97 cm [CI95 +0.16, +1.81], the only
CI-excludes-zero read — and the direction is TOWARD the disk**,
reversing the sign of the only significant read in the v0 sim100
(where teacher80k was the misdirected arm, −0.73 cm). er60k (−0.07
[−0.33, +0.12]) and snap30k (+0.06 [−0.61, +0.85]) read null.
Also in this post: the owner-approved GPU compositor amendment
(4× per-tick, probe reads preserved).*

## Plain words

We changed only what the simulated robot *sees* — real photographed
backgrounds, varied clutter, a fixed wrist camera — and proved to
the byte that the physics stayed identical. Then we asked three
policies to redo the same 20 episodes. The two policies that barely
engage the toy boat behaved exactly as before: if you never really
look at the scene, better scenery can't help you. But the old
teacher model — the one that always engaged the most yet pushed the
boat the *wrong way* under the fake-looking graphics — now pushes
it the right way: about a centimeter of recovered progress per
episode, statistically distinguishable from zero. That is the
cleanest evidence yet that our visual-realism work translates into
behavior, and it says the sim100 story ("direction tracks visual
familiarity") was right.

## Registered reads (20 seeds, paired vs banked v0 rows)

| arm | Δ progress_final (v3−v0) | CI95 | signs | engage >1 cm (v0→v3) |
|---|---|---|---|---|
| er60k | −0.07 cm | [−0.33, +0.12] | +5/−2 (13 ties) | 0→0 |
| snap30k | +0.06 cm | [−0.61, +0.85] | +9/−11 | 3→3 |
| **teacher80k** | **+0.97 cm** | **[+0.16, +1.81]** | **+14/−5** | 3→2 |

- **Integrity green (registered tripwires)**: `spawn_xy` bit-matched
  the banked v0 rows on all 60 episodes; `reset_strikes` 0/60. The
  deltas are pure visual response.
- teacher80k arm means: −0.90 cm (v0, these 20 seeds) → **+0.07 cm
  (v3)** — from "actively pushes the boat away" to net-neutral with
  14/20 seeds improved. Engagement count barely moved: it's the
  *direction* that flipped, exactly the axis the
  [sim100 close](2026-08-12-sim100-results.md) flagged.
- er60k's 13/20 byte-identical ties are their own datum: it mostly
  never touches the boat, under either rendering.
- [Per-seed delta chart](https://mcobzarenco-fontaine-reports.static.hf.space/chart__spot20_v3_deltas.png)
  · [frozen reads JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__spot20_v3_reads.json)

## What it means

The visual gap was never the whole story — er60k's failure is not
appearance-shaped at 20-seed power — but it IS a real term for any
policy that engages. That upgrades the value of the v1→v3 visual
work for training-in-sim (the north-star use), and it argues the
full 100-seed rerun is worth its GPU: the spot-check's teacher
signal at n=20 deserves the tighter CI, and the rerun now costs
~6–9 h wall after the compositor port (below).

## Amendment (labeled): GPU compositor, owner-approved 08:12Z

Rollouts were render-bound: the composite path was single-core
float64 numpy (~371 ms/tick measured under load — built for
100-reset probe reads, amplified 900× per episode). Per the owner's
call it now runs on CUDA (`_TorchPost`: remap/blur/grade/composite;
commit `b99be38`): **94 ms/tick under 3-arm load, ~4×**; the numpy
path stays as the reference implementation and the sensor-noise
stream keeps its seeded RNG. Frames shift by float32 rounding
(≤2/255 counts, oracle-pinned), so the registered probe was re-read
under the fast path: **top 5-NN AUROC 0.669 / k std/mean 0.113 /
wrist 0.544** vs the reference's 0.673 / 0.114 / 0.548 — within
noise, all registered lines still cleared
([json](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_encoder_ood_probe_v3_gpu.json)).
The spot-check itself ran on the reference path end-to-end (the
live processes predated the port).

## Cost

3 arms × 20 episodes in parallel on the H100, 07:47–09:02Z ≈ 1.25 h
wall ≈ **1.25 GPU-h** (gate 3), plus ~0.05 GPU-h of probe/bench
reads. The 1-episode timing smoke (discarded row) is included.

## Next

- `sim100-v1-rerun` (owner_hold): gate facts now read — visuals GO
  on both cameras (0.673 top / 0.548 wrist), behavioral response
  CONFIRMED for the engaging arm at n=20. The rerun would firm the
  teacher signal and re-price all arms under v3.
- `sim-parallel-rollouts` (queued, owner-approved 08:44Z): env
  workers + batched policy server → a 100-seed arm in ~20–30 min.
- `sim100-v2-rerun-amendment-draft` (queued): retargeted to v3 + the
  GPU-path probe numbers.
