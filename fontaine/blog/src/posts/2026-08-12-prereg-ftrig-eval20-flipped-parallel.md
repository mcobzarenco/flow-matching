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

## Results (same session, 15:42–15:59Z) — SUPERSEDED, see the correction below

*Kept verbatim for the record: this readout measured only the
collision-box half of the flip. The owner spotted (16:07Z) that the
videos showed the bracket unmoved — a MuJoCo `sameframe` compile
optimization was making the kinematics ignore the runtime pose edit
on the visual mesh. The corrected run is in the next section.*

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

## Correction (16:07–16:4xZ): the render never flipped — and fixing it revealed the real effect

The owner, comparing the two videos (16:07Z): the bracket still
pointed at the table in *both*. Root cause, probe-confirmed: MuJoCo's
compiler stamps geoms whose frame coincides with an already-computed
frame with a `sameframe` fast path, and `mj_kinematics` then **never
reads `geom_pos`/`geom_quat` again** for them. The bracket's visual
mesh carried flag 2 (frame ≡ the mount body's inertial frame), so the
load-time flip edit was written into the model and silently ignored
every render. `camera_box1` carried flag 3 (rotation-only skip —
harmless: a 180° flip maps a box onto itself); `camera_box2` carried
flag 0 and moved correctly. Net: **physics flipped, appearance
didn't.** The one-line fix: clear `geom_sameframe` on the edited
geoms. Verified: the mesh's settled world position moves from
(74, 10, 48) mm — jaw side, table-ward — to (137, −22, 149) mm,
wrapped around the camera at (150, 0, 150), ceiling side, matching
the hand-computed prediction. Stills:
[fixed](https://mcobzarenco-fontaine-reports.static.hf.space/ftrig_eval20_flip_parallel/zoom_fixed_front.png)
·
[broken](https://mcobzarenco-fontaine-reports.static.hf.space/ftrig_eval20_flip_parallel/zoom_broken_front.png).

The post-flip arm was then re-run (same 20 seeds, same driver). The
planned bit-identity oracle **failed — correctly**: 13/20 seeds
changed. The policy is vision-driven and the bracket is *visible in
the top camera*; un-sticking the mesh changed the policy's input.
(The real frames do show the bracket ceiling-side — the fixed render
closes an appearance gap, it doesn't add one.) So the section above
was a physics-only read, and the TRUE flip effect is:

| read | TRUE post-flip (v2) | pre-flip | morning's "post-flip" |
|---|---|---|---|
| success | 0/20 | 0/20 | 0/20 |
| mean progress_final | **−0.46 cm** | −1.21 cm | −1.14 cm |
| knock-aways ≥1 cm | **2** | 6 | 6 |
| paired Δ vs pre-flip | **+0.75 cm** [−0.33, +2.26] | — | +0.07 |

- The two catastrophes dissolve:
  [seed 4](https://mcobzarenco-fontaine-reports.static.hf.space/ftrig_eval20_flip_parallel/postflip_v2/rollout_seed004.mp4)
  −12.3 → −0.05 cm, seed 5 −5.5 → +0.1. Two seeds worsen (s11 −2.4,
  s14 −2.2); 9 tie exactly. The CI crosses zero — n=20 rough read,
  as registered.
- Character shift: less shoving, more freezing — several pre-flip
  movers now end at exactly 0.00. Plausibly the visible bracket
  makes frames *more* like the training distribution and the policy
  defers to its (frozen-ish) prior; the encoder-OOD probe follow-up
  named in the sequential pre-reg would adjudicate.
- Physics-side claims are untouched by the correction (the mesh
  never collides): replay control loss −62%, sweep 31.9%→1.4%, and
  the box-only paired read above stand as what they measured.
- Lesson registered for every future runtime geom edit: **clear
  `geom_sameframe` after editing `geom_pos`/`geom_quat`** — the
  compiler's fast path silently swallows the edit otherwise. Audited
  the existing runtime edits: cameras (`_repose_wrist_cam`) have no
  such flag; material/light edits are unaffected.
- Corrected totals: 4 arms ≈ **0.36 / 0.5 GPU-h**. Corrected rows +
  videos under
  [`/ftrig_eval20_flip_parallel/postflip_v2/`](https://mcobzarenco-fontaine-reports.static.hf.space/ftrig_eval20_flip_parallel/postflip_v2/rows.json).

## Owner extension (16:37Z): the step-500 checkpoint, same seeds

The owner asked for the rig fine-tune's step-500 checkpoint through
the same read. Converted fresh (`bijou.convert_molmoact2`, same
recipe/norm tag as step-2000, kept at
`outputs/converted/molmoact2_rig_r1_step500`), then the same 20 seeds
on the fixed post-flip sim, same parallel driver:

| read | step-500 | step-2000 (corrected) |
|---|---|---|
| success | 0/20 | 0/20 |
| mean progress_final | **+0.02 cm** | −0.46 cm |
| moved (>0.05 cm) | **9** | 6 |
| knock-aways ≥1 cm | **1** | 2 |
| best seed | s0 **+1.59 cm** | s19 +0.13 |

Paired per-seed (500 − 2000): **+0.48 cm, CI95 [−0.06, +1.13]** — 9
better / 3 worse / 8 tied. step-500 dissolves step-2000's two worst
episodes (s11 −3.6 → −0.25, s14 −5.0 → +0.03) and posts the day's
best approach
([seed 0](https://mcobzarenco-fontaine-reports.static.hf.space/ftrig_eval20_flip_parallel/step500/rollout_seed000.mp4)
+1.59 cm); its one knock-away is
[seed 1](https://mcobzarenco-fontaine-reports.static.hf.space/ftrig_eval20_flip_parallel/step500/rollout_seed001.mp4)
(−1.8). Not CI-clean at n=20, but directionally: the extra 1500
fine-tune steps are not buying sim-side competence — the earlier
checkpoint engages more and shoves less, consistent with the
fine-tune narrowing toward rig appearance (sim frames sit further
from step-2000's distribution). Both checkpoints freeze on the same
~8–10 seeds. Day total across all 5 arms: **~0.45 / 0.5 GPU-h**.
