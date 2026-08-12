# Pre-registration: ftrig MolmoAct2 20-seed rerun on flipped-mount physics — both arms parallel

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
