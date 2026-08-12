# Pre-registration: 20-seed behavioral spot-check under v3 visuals (er60k · snap30k · teacher80k)

*Registered 2026-08-12 ~07:5xZ (work session; real `date -u` at
write: 07:50). Owner steering 07:35Z: "let's do some spot checks,
would also spot check snapflow or the teacher 80k." This is the
spot-check option carried on the sim100 rerun since the v1 close —
now under the owner-approved v3 default (07:29Z). The full-protocol
instrument stays
[the sim100 pre-reg](2026-08-11-prereg-sim-policy-eval-100seeds.md);
this note registers only the subset and the paired read.*

## Plain words

The simulator's pictures changed a lot this week (real backgrounds,
fixed wrist camera, varied clutter); its physics did not change at
all — we prove that bit-for-bit. So if the robot policies now
*behave* differently in the simulator, the only possible cause is
that they are reacting to what they see. This run replays the first
20 of the 100 standard episodes for three policies — the main one,
its fast distilled student, and the older teacher — and compares
each episode one-to-one against the banked runs made with the old
graphics. It answers the owner's question: did the visual overhaul
change behavior, and for which model?

## Protocol (subset of sim100, frozen)

- **Arms** (banked checkpoints + decode configs, unchanged):
  `er60k` (heun-10), `snap30k` (euler-1 — its training target),
  `teacher80k` (heun-30). Same task string, stats repo, policy seed
  0, bf16 expert, replans 30 × horizon 30.
- **Seeds 0–19** — the paired subset of the banked v0 0–99 rows.
- **Frames: `render_style` default = v3** (owner-approved flip,
  commit `da96d30`): per-episode real plates + measured clutter
  draws (top), re-tuned wrist under the v1 path.
- Launch: 3 arms in parallel on the idle H100 (systemd-run units,
  babysit registry); one v3 episode timed at ~5.4 min wall
  (render-bound), so ~1.8–2 h wall, **gate ≤ 3 GPU-h**. A 1-episode
  er60k timing smoke ran pre-registration (this paragraph); its row
  is discarded — the batch reruns seed 0.

## Reads (registered)

1. **Primary, per arm**: paired per-seed Δ `progress_final_cm`
   (v3 − v0, same seed), mean + 10k-resample bootstrap CI + sign
   counts. Null: no behavior change (the v0 read was 0/500
   successes, near-zero progress for er60k/snap30k, −0.7 cm for
   teacher80k).
2. **Engagement split**, per arm: episodes with `progress_cm`
   (spawn→closest) > 1 cm, v3 vs v0 on the same 20 seeds — the
   engagement/direction structure was the sim100 finding; a visual
   response should move it.
3. **Integrity tripwires**: `reset_strikes` 0 on every episode;
   `spawn_xy` bit-matches the banked v0 rows per seed (physics
   identity across render styles is oracle-pinned — any mismatch
   voids the run). Latency record-only.
4. Record-only: min-distance trajectories, per-arm wall/GPU cost,
   best/worst episode videos for the gallery.

No success bar — this is a record-only behavioral read; whichever
way it lands (null = visuals don't move behavior at 20-seed power;
signal = the policies see the difference), it prices the full
100-seed rerun decision.
