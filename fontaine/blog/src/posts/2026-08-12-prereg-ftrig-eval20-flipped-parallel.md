# Pre-reg + results: ftrig MolmoAct2 rerun on flipped-mount physics — 18/20 bit-identical, the bracket is innocent

*Registered 2026-08-12 15:3xZ (work session; real `date -u` at write:
15:35). Owner prio 15:27:11Z: "Prio: can we re-run the 20 episodes on
the flipped camera physics? Use many parallel workers so it goes
fast." Design confirmed in-channel 15:28Z. Rough/exploratory pass —
this note pins the settings and the asterisk before launch; no
registered claim gates on it.*

## Plain words

This morning's 20-episode look at the owner's MolmoAct2 checkpoint ran
with the simulator's wrist-camera bracket mounted upside down — a bug
the owner spotted in the videos, now fixed (the bracket points at the
ceiling, like the real arm). The owner wants the same 20 episodes
re-run on the fixed physics, fast, using parallel simulation workers.
Because our parallel mode is known to produce slightly different
numbers than the trusted sequential mode (a floating-point batching
effect), we run BOTH the old physics and the fixed physics through the
same parallel mode and compare within it, episode by episode — that
way the one thing that changes between the two runs is the bracket
fix, and the comparison is fair even though the absolute numbers carry
an asterisk.

## Design (frozen before launch)

- **Two arms, both `sim.rollout_sim_parallel` at `--workers 8`**, same
  checkpoint (`~/marius-convert-gate/converted/molmoact2_rig_r1_step2000`),
  same settings as the banked sequential run (euler-10, horizon 30,
  replans 15, bf16 expert, v3 frames, seeds 0–19, videos on):
  - **post-flip**: flipped-mount physics (`d5cf9fd`, the registered
    geometry) — the number the owner asked for.
  - **pre-flip**: `--no-mount-flip` (new flag, this session): the
    mirrored-Menagerie bracket, physics-verified to reproduce the
    pre-flip settled bracket height exactly (camera_box2 40.2 mm at
    home — the probe-measured pre-flip value; flipped: 156.6 mm).
- **The sanctioned read is the paired per-seed delta WITHIN the
  parallel path** (pre-reg
  [parallel rollouts](2026-08-12-prereg-sim-parallel-rollouts.md)
  frozen rule, applied): the parallel oracle FAILED 14:37Z (batched
  bf16 decode diverges), so parallel rows are never
  registered-comparable to the banked sequential rows. Cross-arm
  comparisons here are parallel-vs-parallel only; both arms inherit
  the identical scheduler, seed partition, and stable-key noise.
- **Reads** (exploratory, no gate): paired `progress_final_cm` delta
  (flip effect), knock-away count change (pre-flip sequential had
  4/20 ≥1 cm; hypothesis: bracket-table collisions contributed),
  approach count, videos side-by-side for the worst movers.
  Incidental datum: parallel-vs-sequential drift on the pre-flip arm
  (same physics as the banked run, different decode path).
- **Instrument changes this session** (committed before launch):
  `flip_camera_mount` constructor toggle on `SO101Sim` (default True =
  registered geometry; CPU probe verifies the mirror restores all 3
  mount geoms and the 40.2 mm settled height), and the parallel driver
  gains the sequential driver's merged-stats fallback (converted
  checkpoints carry no per-dataset table) + `--no-mount-flip` +
  `mount_flip` recorded in the rows JSON. Harness oracle
  (`tests/test_sim_parallel_rollouts.py`) 5/5 green after the change.
- **Gate**: ≤0.5 GPU-h total (est. ~2× 5–8 min at workers=8).
  Outputs under `outputs/sim/ftrig_eval20_flip_parallel/{postflip,preflip}/`.
- **What would change our mind about the flip**: nothing here — the
  flip is registered on physics evidence (replay control loss
  0.0831→0.0751, below-table sweep 31.9%→1.4%). This rerun asks
  whether the *policy's observed behavior* (knock-aways, grinding)
  moves with it; a null is a real answer (the bracket wasn't the
  binding constraint on THIS policy's failures).

## Results (same session, 15:42–15:59Z)

**The pre-registered null is the answer, almost exactly: 18/20 seeds
are BIT-IDENTICAL across the two physics — this policy's rollouts
almost never touch the bracket.** The 2 seeds where the bracket did
engage both IMPROVED post-flip. The knock-aways are jaw-side, not
bracket-side.

| read | post-flip (par) | pre-flip (par) | pre-flip (seq, banked) |
|---|---|---|---|
| success | 0/20 | 0/20 | 0/20 |
| mean progress_final | −1.14 cm | −1.21 cm | −0.84 cm |
| median | 0.00 | 0.00 | 0.00 |
| knock-aways ≥1 cm | 6 | 6 | 4 |
| approaches ≥0.5 cm | 1 | 0 | 2 |

- **Paired flip effect (the sanctioned read)**: mean +0.072 cm,
  18/20 exactly tied, 2/20 improved, 0 worsened. The movers:
  [seed 15](https://mcobzarenco-fontaine-reports.static.hf.space/ftrig_eval20_flip_parallel/postflip/rollout_seed015.mp4)
  +0.29 → +1.10 cm
  ([pre-flip video](https://mcobzarenco-fontaine-reports.static.hf.space/ftrig_eval20_flip_parallel/preflip/rollout_seed015.mp4))
  and
  [seed 5](https://mcobzarenco-fontaine-reports.static.hf.space/ftrig_eval20_flip_parallel/postflip/rollout_seed005.mp4)
  −5.51 → −4.90 cm — both low-reaching episodes where the old
  bracket ground the table.
- **Interpretation**: the flip matters where the probes said it
  does — real-pose replay fidelity (control loss −62%, the servo
  read) — but this policy fails before it reaches the poses the
  bracket used to block. Its 6 knock-aways persist unchanged under
  both geometries: the jaws shove the boat, the bracket is
  innocent. The MolmoAct2 diagnosis from the sequential run stands
  unchanged on the fixed physics.
- **Physics-difference sanity check**: bit-identity on 18 seeds is
  the *expected* signature — the mount geoms only enter the dynamics
  through contact (their mass never moved, a compile-time residual
  noted at the flip), so trajectories that never collide with the
  bracket are unchanged to the bit.
- **Incidental determinism datum**: a launcher-flag slip ran the
  post-flip config twice first — the two runs came out bit-identical
  on all 20 rows, so lockstep-parallel at workers=8 is exactly
  reproducible run-to-run (a useful property for the GRPO probe).
- **Parallel-vs-sequential drift, quantified at outcome level** (both
  pre-flip physics, same 20 seeds): mean −0.37 cm, 11/20 seeds moved
  >0.1 cm, max 6.0 cm (seed 15 flipped sign, seed 17 recovered 4 cm).
  The 14:37Z oracle FAIL was not cosmetic — cross-path comparisons
  stay barred, the asterisk on this page's absolute numbers is real.
- **Cost**: 3 × 5.4 min arms ≈ **0.27 / 0.5 GPU-h** (incl. the
  accidental replicate). All rows + 40 videos under
  [`/ftrig_eval20_flip_parallel/`](https://mcobzarenco-fontaine-reports.static.hf.space/ftrig_eval20_flip_parallel/postflip/rows.json).
