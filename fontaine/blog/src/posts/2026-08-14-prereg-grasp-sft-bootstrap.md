# Pre-reg (DRAFT) — grasp-rich SFT bootstrap: competence before RL pressure

*2026-08-14, ~22:2xZ. Drafted at the owner's go (22:07Z: "do as much
in parallel as you reasonably can") following the 21:53Z question —
"what should we do next to train a policy which solves over 90% of
seeds successfully?" Status: **DRAFT** — finalization (frozen params,
objection window) before any GPU stage runs. Stage A is CPU-executable
immediately.*

**Plain words.** Our robot policies touch the toy boat but almost
never finish the job — the best row completes the task on about 1 of
100 tries. Reinforcement learning can only amplify behaviors a policy
already produces sometimes; at a 1% success rate there is nearly
nothing to amplify, which is exactly what our last RL run measured
(it learned to shove the boat, not grasp it). The unlock is to first
*teach* the policy to grasp, using something we have for free in
simulation: the simulator knows exactly where every object is, so a
scripted robot — no learning, just move-above/close/lift/place
against known coordinates — can generate hundreds of *successful*
demonstrations cheaply. We then fine-tune the policy on those
successes and only afterwards apply RL to push the success rate up.
This document freezes that plan into measurable stages with abort
rules, so each step tells us something even when it fails.

## §1 Hypothesis and the causal chain

**H:** the policy's ~0–1% sim success floor is a *competence* gap
(no grasp behavior in-distribution), not an *incentive* gap — so
dense successful demonstrations move success rate where GRPO
re-pricing measurably did not (R1-B frozen verdict: deciding behavior
worsened at pinch competence ~4/64; the registered next shape is
grasp-rich SFT before RL pressure).

Chain to the owner's 90% target: scripted expert (privileged state)
→ successful demos → SFT (≥20% base) → fresh-pre-reg GRPO per
Decision 11 (groups finally carry success variance) → success-recycle
flywheel for the tail. This pre-reg covers the chain UP TO the ≥20%
gate; GRPO is its own registration.

## §2 Stages, gates, and pricing (measured pace 0.0094 GPU-h/episode)

| stage | what | cost (est) | gate at boundary |
|---|---|---|---|
| A | scripted expert vs privileged sim state (waypoint policy over `sim.data` object pose: approach → descend → close → lift → traverse → place), validated on **non-eval seeds** | CPU + ~0.2 GPU-h validation (20 episodes) | scripted success **≥ 70%** on 20 held demo seeds; below → the sim itself can't host the behavior (§4 F-physics) |
| B | demo collection: scripted rollouts on demo seeds, **successes kept**, obs (top/wrist/state) + executed actions written in the training format | ~2–4 GPU-h for 300–600 kept successes | ≥ 300 kept successes inside the budget |
| C | SFT: molmoact2 **`--objective ar`** fine-tune on the demo set (the GRPO-ready pathway, new-stack objective matrix; rig-ft recipe class, measured ~2.7 GPU-h) + optional ftrig4k-recipe flow arm as comparison | ~3–5 GPU-h | train-loss sanity + in-train greedy decode emits legal streams |
| D | sim100 eval of the SFT endpoint(s): the frozen 100 seeds, standard gates | ~1–1.5 GPU-h | **primary read, frozen:** successes **≥ 20/100** → GRPO GO (fresh pre-reg); 5–19 → iterate B/C once (more demos) before GRPO; **< 5** → §4 F-transfer |

Worst-case ≈ **11 GPU-h; proposed gate ≤ 13** (both iteration arms
included). Stage boundaries are hard stops with in-channel posts.

## §3 Contamination + comparability guards (frozen at draft)

- **Eval seeds 0–99 NEVER appear in demos**: demo seeds drawn from
  1000+ (same spawn distribution class, disjoint stream). The frozen
  sim100 seed set stays a pure holdout for every read in the chain.
- Demos are **sim-rendered under the production visual config**
  (v3 + fitted lens + numpy post) — the policy trains on the pixels
  it will be evaluated on; any visual-config change between B and D
  is a registered amendment.
- The honesty instrument (er_60k knn5) never trains — unchanged.
- Success detection = the sim100 harness's own `success_tick`
  machinery (no new success definition).
- Seed policy: fresh demo seeds are variance-motivated (the standing
  policy's stated-reason branch); the eval seeds stay the frozen
  0–99 for comparability with every banked row.

## §4 Falsifiers and consequences

- **F-physics** — the scripted expert with privileged state can't
  reach ≥ 70%: our sim (contact/servo model) can't host the grasp;
  demos would teach nothing. Escalate to the Squint twin tier
  (preflight GO banked 08-14) where competence is buildable from
  state, and the sim keeps its screening role only.
- **F-transfer** — script succeeds, SFT'd policy stays < 5/100: the
  gap is observation-side (visual/pose OOD), not competence — the
  wrist-transfer screen's read (running in parallel) becomes the
  binding diagnosis, and renderer-class/visual work outranks more
  demos.
- **F-live** — ≥ 20/100: competence-first confirmed; GRPO registers
  fresh on the new stack (Decision 11) with success-variance groups,
  targeting the 90% north star via SFT+GRPO alternation.

## §5 Interplay with the wrist-transfer screen (running in parallel)

The screen prices whether wrist-appearance honesty moves behavior;
this bootstrap prices whether competence moves success. They share
the sim100 substrate but not GPU windows (screen first — it is
already registered FINAL). If the screen lands **F-null/F-flat**
(wrist channel dead/tolerated), stage-C's optional flow arm drops
and the AR arm carries alone; if **F-live**, the demo render config
inherits whatever wrist-fidelity decision the owner takes, as a
registered amendment before stage B.

## §6 Status

**DRAFT.** Finalization checklist before any GPU stage: (1) freeze
demo-seed range, kept-success target, SFT recipe params (LR/steps
from the rig-ft runbook class) and the stage-D arm list; (2) owner
objection window in-channel; (3) HEAD re-pin. **Stage A (the scripted
expert itself) is CPU-executable now** and lands with its own oracles
(waypoint reachability on 3 spawn draws, jaw-close force sanity, no
eval-seed usage asserted in code).
