# Pre-registration: sim content diversity v3 — per-episode plate bank + clutter-state draws

*Registered 2026-08-12 ~07:0xZ (work session). Successor of
[visual matching v2](2026-08-12-sim-visual-inpainting-results.md),
which met its AUROC bar (top 0.773 ≤ 0.790) but left the
homogeneity read untouched: **sim k std/mean ~4% vs real ~45%** —
every read since the
[encoder OOD probe](2026-08-11-prereg-sim-policy-eval-100seeds.md)
names this the remaining axis. Lighting jitter (v1) moved it ~3%;
a fixed real background (v2) moved it not at all. This item attacks
it with per-reset CONTENT variation. Instrument and A/B semantics
inherited unchanged from the
[v1](2026-08-12-prereg-sim-visual-matching.md) /
[v2](2026-08-12-prereg-sim-visual-inpainting.md) pre-regs.*

## Plain words

Real camera frames differ from each other: the daylight changes
between recordings, the computer mouse and the mug wander around
the table, sometimes things enter or leave the scene. Our simulator
frames are near-clones of each other — same light, same clutter in
the same spot, every time. A policy's vision encoder sees a tight
little cluster where reality is a wide cloud, and that sameness is
itself a give-away (and a robustness risk: train on the clone, fail
on the cloud). This pass makes each simulated episode draw a
different *real* background — one per real recording, carrying that
recording's actual lighting — and scatters the desk clutter the way
the real operator actually scattered it between recordings. The
test: the spread of the simulator's frames under the encoder should
grow toward the real spread, without the frames getting any easier
to tell from real.

## Baseline (measured, frozen)

Reset-render probe on the shipped v2 default (08-12): top 5-NN
AUROC **0.773** (100 seeds; 0.774 at 20×5), top sim k std/mean
**0.038** (100 seeds; 0.042 at 20×5) vs real held-out **0.447**.
Per-draw mean-k spread 0.5%. Wrist (v1 path inside v2): 5-NN 0.548
after the [periphery re-tune](2026-08-12-sim-wrist-periphery-results.md).

## Method (v3 scope)

1. **Per-episode plate bank** (`make_clean_plates.py --bank`): one
   top-cam plate per A-half episode (26). The mining pass masks ALL
   transient/novel content by construction — no boat ghosts can
   bake in (the naive per-episode median, inspected before this
   registration, bakes in the boat parked on the disk, both arm
   rest poses and the operator's hand):
   - fit a per-episode per-channel gain to the global plate on
     agreeing pixels (the episode's exposure/white-balance state);
   - a frame's pixel is an **inlier** iff it sits within a
     registered-by-inspection threshold of the gain-corrected
     global plate; plate pixel = median over inlier samples;
   - pixels with too few inliers (anything parked: boat, arms,
     hands, the real disk, moved clutter) fall back to the
     gain-corrected global plate, feathered at the boundary.
   Plates carry the episode's real lighting field and photometric
   state; they contain NO object-level novelties. Bank manifest
   pins episodes, thresholds, coverage stats, commit.
2. **Clutter-state draws**: the four contype-0 stand-ins (mouse,
   mug, laptop, PCB) get per-reset poses drawn from the **measured
   real between-episode spread**: the same mining pass extracts,
   per episode, the change-blob centroid near each object's
   canonical position (image → pinhole ray → table plane → world
   xy, through the sim's own camera model); draw ranges = empirical
   min/max boxes (+ yaw jitter for the non-circular objects). If an
   object is detectably absent in some A episodes, presence is
   drawn at its empirical frequency (absent = parked outside both
   frusta). Numeric ranges land in the results post + manifest —
   the method is what registers here. Clutter stand-ins are
   contype/conaffinity 0: physics untouchable by construction.
3. **`render_style="v3"`** = the v2 composite with (a) the top
   plate drawn per reset from the bank (uniform over the 26) and
   (b) clutter poses drawn per reset — both from the appearance RNG
   **after every existing draw**, so all v2 appearance draws are
   stream-identical and the **wrist path stays bit-identical to
   v2** (wrist keeps the v1 render path; its plate mush and its
   0.548 read are not touched).
4. **Not in scope**: disk-position draws — the disk is task
   geometry (success is measured against it); drawing it from the
   real between-episode distribution changes task semantics and
   needs its own pre-reg (queue note, carried forward). Real
   clutter kept in-plate (vs rendered stand-ins) is a named
   follow-up lever, not this pass.

## Reads (instrument pinned, order registered)

Foreground, idle H100; **gate ≤ 0.3 GPU-h for the whole item**
(~0.02 GPU-h per read). Plate-bank/clutter mining and inspection
are CPU and encoder-free; at most **3 encoder-probe iterations**
on the composite before the candidate freezes.

1. **v3 read (registered primary)** — reset-render probe, seeds
   0..99, `--render-style v3`: top 5-NN AUROC, top k std/mean,
   k-ratios, centroid secondaries.
2. **Homogeneity read (registered, co-primary spread figure)** —
   seeds 0..19 × 5 appearance draws: k std/mean where every reset
   draws plate + clutter independently of the spawn.
3. **Wrist guard (registered)**: wrist frames bit-identical to v2
   for the same (seed, appearance_seed) — asserted on renders
   before any read is credited.
4. Physics oracles before any read is credited: reset qpos
   bit-identical across v0/v1/v2/v3 and across appearance seeds;
   spawn stream bit-matches banked sim100.

## Success bar (registered)

- **v3 lands** if, on the primary 100-seed read: **top sim k
  std/mean ≥ 0.15** (≥ ~4× the v2 figure, one third of the real
  0.447) **AND top 5-NN AUROC ≤ 0.790** (no regression past the
  registered v2 line — diversity must not come at the price of
  realism).
- **Overfit tripwire** (inherited): AUROC < 0.5 reported as a
  warning, not a win.
- Record-only: whether AUROC *improves* (more spread should
  overlap the real cloud more), per-plate k breakdown, per-draw
  spread at 20×5.
- Miss → reported as the result with the measured spread; the
  shipped default flips v2 → v3 only if the bar is met.

## Why this bar

k std/mean is the pre-registered homogeneity instrument every read
since the OOD probe has quoted (4% vs 45%). 0.15 is not "matched to
real" — real spread includes mid-episode content (arm sweeps,
operator hands, the boat mid-carry) that reset renders can never
show; it is the "the lever visibly moved" line, same semantics as
v1's ≥0.10-absolute AUROC drop. The AUROC guard keeps the failure
mode honest: spread is trivially inflatable with garbage draws that
would also make sim MORE separable — both numbers must hold at
once.
