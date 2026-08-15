# Pre-reg (FINAL) — grasp-rich SFT bootstrap: competence before RL pressure

*2026-08-14, ~22:2xZ. Drafted at the owner's go (22:07Z: "do as much
in parallel as you reasonably can") following the 21:53Z question —
"what should we do next to train a policy which solves over 90% of
seeds successfully?" Status: **FINAL as of 2026-08-15 ~01:4xZ** — §6
freezes the checklist (demo seeds, targets, recipe params, arm list,
convention seam) and opens the objection window. Stage A landed
CPU-side while the draft stood (14/16 on the engineering smoke).*

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

## §6 Status — FINAL (frozen 2026-08-15 ~01:4xZ)

The draft checklist, resolved item by item:

**(1) Frozen parameters.**

- **Stage-A gate read: 20 held seeds 1020–1039**, ≥ 70% (≥ 14/20)
  scripted success, rendered (videos banked for the record, ~0.2
  GPU-h class). Honesty note: seeds **1000–1015 were the engineering
  smoke set the expert's fixes were tuned on** (14/16 there) — the
  gate runs on a disjoint stream precisely so tuning-to-the-smoke
  can't pass it.
- **Stage-B collection**: demo seeds ascend from 1000 (the smoke
  seeds are legitimate demos), production visual config, successes
  kept. Target **400 kept successes; gate ≥ 300 within ≤ 4 GPU-h**.
- **Stage-C primary (AR)**: molmoact2 `train_lerobot.py`,
  base `allenai/MolmoAct2-SO100_101`, the rig-ft recipe class
  verbatim — `ft_action_expert=true` only (`ft_vlm=false`,
  `ft_embedding=none`, `lora=false`), action-expert LR **5e-5**,
  global batch **64** (device 8), save every 500 —
  with **max_duration 3000 steps** (the runbook's 2000 was sized for
  the smaller rig set; 300–600 kept demos ≈ 45–90k frames ≈ 2–4
  epochs at batch 64; every 500-step checkpoint is retained as
  fallback). Endpoint = final step.
- **Stage-C optional flow arm**: ftrig4k recipe verbatim (4k steps,
  decoder LR 1e-5, dataset swapped to the demo set). The wrist
  screen closed **F-instrument** (01:3xZ 08-15) — not F-null/F-flat —
  so the §5 drop clause does **not** fire: the flow arm stays in,
  conditional only on the ≤ 13 GPU-h gate after the primary
  chain lands.
- **Stage-D arm list**: the SFT-AR endpoint on the frozen sim100
  (100 eval seeds, standard gates + reset-strike checks); the flow
  endpoint too if trained. Context anchors (not gates): banked
  ftrig4k +0.08 cm / 47 moved / ~1 success class, and the fresh
  stage-1 W0 in-run row (+0.054 / 44 / 2 successes, post-fitted-lens
  + v3-wrist substrate).

**(4) Convention seam (owner question 23:19Z 08-14), frozen.** Stage
C trains against a **recomputed per-dataset q01/q99 table** (the
rig-ft recipe default), so demo rows are written in the
**controller-native rig frame, identity — no shim**; the rows JSON
carries `state_units: "rig (identity — recomputed dataset table)"` as
provenance. The official v3.0→v2.1 shim
(`MOLMOACT2_OFFICIAL_SIGNS/OFFSETS`) is the released-global-table
contract and is NOT applied anywhere in stages B–D: stage-D rollouts
consume the endpoint through the same recomputed-table frame it
trained in. Any move to the release's global table is a registered
amendment that flips demo-writing to shim-frame (the GRPO
training-row contract).

**(2) Objection window.** Open at the in-channel finalization post
(~01:4xZ 08-15). GPU stages launch at the **next work-session
boundary** absent objection; an explicit owner go collapses the
window; any objection re-opens finalization. Stage A's gate read is
the first GPU leg.

**(3) HEAD re-pin.** The finalization commit id rides the in-channel
post; the gate read launches from that HEAD or later on the
`fontaine` branch (expert code: `b564337` → `d1b2552` → `2435a6d`,
14/16 smoke).

**Stage-A status at finalization**: landed CPU-side with 5 oracles
(IK reachability over the spawn band, perpendicular alignment < 6°,
scratch-data purity, eval-seed refusal at `DEMO_SEED_BASE` 1000) plus
the settle-before-release and deck-strike jam-recovery mechanisms;
14/16 on the tuning smoke. The gate read (seeds 1020–1039) is what
counts.

## §7 Amendment A1 — fresh held gate set after the 08-15 gate FAIL
*(registered 2026-08-15 ~02:5xZ, in-channel post before any new read)*

The §6 gate read ran 02:0x–02:1xZ 08-15 (HEAD `e371e2b`, rendered,
videos + `reports/analysis__grasp_sft_stageA_gate.json` banked):
**FAIL, 11/20** — the held stream caught tuning-smoke overfit (14/16
on 1000–1015). Integrity receipts: rendered ≡ unrendered bit-identical
on 3 re-run seeds; failure classes measured, not guessed (4× lower
radial stall, 3× mid-carry grip loss, jam-on-both-branches tail).
Record-only honesty note: 11 clean grasp-lift-place runs contradict
F-physics' mechanistic reading ("the sim can't host the grasp") — the
gap was expert *coverage*, so the boundary decision went to the owner
(boundary post 02:14Z) with a registered amendment as the stated
default absent steering.

**A1, frozen:**

- **Seeds 1020–1039 are reclassified as tuning data** (burned by the
  read + diagnosis). The robustness pass tuned on them is `77776fd`:
  lower-phase place droop, re-grasp recovery, jam-flip budget 3,
  retry-scoped dwell + droop reset — measured 11/20 → **16/20** on
  the burned set, 14/16 → 15/16 smoke, no regressions.
- **Fresh held gate set: seeds 1040–1059**, protocol otherwise
  verbatim §6 (rendered, videos banked, production `SO101Sim()`
  substrate, ≥ 14/20, HEAD `77776fd` or later).
- **One amendment only**: if the fresh read fails, §4 F-physics fires
  as frozen (Squint twin tier) with **no further expert tuning** — the
  amendment loop is capped here, before the second read, so it cannot
  become a re-roll ladder.
- Window: the fresh read runs at this session's close (≥ 30 min from
  the in-channel A1 post) absent objection; stages B–D are unchanged
  from §6 and launch at the next session boundary absent objection if
  the read passes. Demo collection (stage B) draws seeds ascending
  from 1000 **excluding nothing** — tuning seeds are legitimate demos
  (§6), and the gate stream stays disjoint from eval seeds 0–99 as
  always.

## §8 Stage-B collection record (record-only, 2026-08-15)

*(Appended mid-collection ~04:3xZ; no frozen parameter changes — this
section records measured facts for the boundary decision and the
results page.)*

Collection launched 03:29:18Z (owner 👍 on the finalization = explicit
go). Babysit surfaced a pace drop at ~04:06 (52 kept / 76 attempted,
window rate ~1.1 kept/min); diagnosed in-session rather than letting
the wall clock burn:

- **Determinism receipt**: the 6 freshly re-run miss seeds
  (1061–1074 band) reproduce the collector's rows *exactly* — same
  tick counts, same final distances — in a clean process. No collector
  state leak, no new failure class (jam / pinch-miss taxonomy
  throughout).
- **No spawn drift**: spawn x/y/yaw distributions are statistically
  identical across the smoke (1000–1015), held (1040–1059) and
  forward (1060–1099) bands.
- **True expert rate: 62.5%** (125/200, seeds 1078–1277, unrendered
  CPU re-runs, ~3 min wall, zero GPU contention). The gate reads'
  75–80% was n=20 optimism — 15/20 has a CI of roughly [53%, 89%] —
  and the 4/16 stretch at 1060–1075 was a bad run inside a ~62%
  process, not a regime change. Lesson for future collections: when
  the expert is deterministic and CPU-replayable, price the GPU
  collection off a cheap large-n unrendered measurement, not the
  n=20 gate read.
- **Wall projection**: ~295 kept at the 4 h self-stop (07:29Z) —
  borderline against the ≥ 300 gate. Per the frozen terms the run
  rides untouched; if it lands just under, the anchor's priced path
  is a **recorded top-up** (the collector is resume-capable; ~10 kept
  ≈ ~10 min GPU) before stage C. Mid-ride status posted in-channel
  04:13Z with the objection window open until the wall.
