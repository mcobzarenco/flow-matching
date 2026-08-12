# Pre-registration: ftrig MolmoAct2 (rig-r1 step2000) — 20-seed sim eval, rough numbers + videos

*Registered 2026-08-12 13:5xZ (work session; real `date -u` at
write: 13:53). Owner-called 13:16Z/13:36Z: "I would like to also
queue evaluating the ftrig molmoact2 latest checkpoint on the sim
(top prio, when GPU is back) … let's do 20 episodes first, keen to
get some rough numbers and videos on that checkpoint." Record-only
pass — this note pins the settings and framing before launch; no
registered claim gates on it.*

## Plain words

The owner fine-tuned a different robot brain — MolmoAct2, a large
open vision-language-action model — on our own robot's recordings,
and converted it into the format our code loads. Before deciding
anything about it, we want the cheapest possible look: drop it into
our simulator for 20 episodes, watch the videos, and read the same
progress numbers we track for our home-grown models. This is
explicitly a first look, not a verdict: our simulator has only been
sanity-checked against our own model family, and a twin that ranks
one family faithfully can still misread a new one (the AutoEval
caution from the sim-as-eval literature). Whatever number comes out
tells us where to look next, not how good the checkpoint is.

## Checkpoint

- `~/marius-convert-gate/converted/molmoact2_rig_r1_step2000`
  (in-house format 3; decoder `molmo_flow` 36×768, horizon 30;
  backbone ref `~/checkpoints/molmoact2-so101-rig-r1-step2000-hf`;
  `read_checkpoint_info` verified on this branch at queue time).
  "Latest" of the rig-r1 series per the owner's 13:16Z message
  (step500/1000/1500/2000 exist; 2000 assumed, flagged in-channel).

## Settings (pinned)

- **Seeds 0–19** of the sim100 list — same spawn stream as every
  banked run; deterministic per-seed rows.
- **v3 frames** (current default render), videos ON for all 20
  episodes (the owner asked to see them).
- Policy served through `BijouPolicy` exactly as `bijou.eval`
  wires molmo_flow checkpoints (Euler, the checkpoint's recorded
  step count); replans/execute-horizon at `rollout_sim` defaults
  unless the checkpoint's horizon forces otherwise — any deviation
  recorded in the results post.
- **Execution path**: the parallel rollout driver if
  `sim_parallel_oracle.py` is GREEN (this eval is its first
  consumer, owner 13:36Z); sequential `rollout_sim` fallback if the
  oracle fails (paired-only rule from the parallel pre-reg — no
  mixing either way at n=20).

## Reads (record-only)

- Per-seed `progress_final_cm`, min-distance, success flag, guard
  trips (strikes / upright / knock-offs) — the sim100 conventions.
- Side-by-side context rows: er60k v3 (−0.07 cm spot20, 0/20
  success) and teacher80k v3 (+0.97 cm spot20) — context, not a
  comparison claim: different training data, different stack.
- Videos linked from the results post; a qualitative note on
  failure modes (the first molmoact2 rollout through our sim —
  integration edges are themselves findings).

## Framing caveats (stated before the numbers exist)

1. **AutoEval caution**: our sim's eval-fidelity evidence is
   family-local (our trunk lineage). First read on a foreign stack
   is exploratory by construction.
2. **n=20** bounds everything: a 0/20 or a 3/20 is a coarse signal,
   not a rate estimate.
3. This checkpoint trained on rig data whose cameras the v3 render
   approximates for OUR encoder; molmoact2's encoder may sit
   elsewhere on the sim-real gap. If the numbers are near-floor,
   the encoder-OOD probe rerun on molmoact2 features is the named
   follow-up before any "the checkpoint is weak" reading.

Gate: ≤1 GPU-h (20 episodes, parallel path expected ~15–20 min;
sequential worst case ~30 min + load time).
