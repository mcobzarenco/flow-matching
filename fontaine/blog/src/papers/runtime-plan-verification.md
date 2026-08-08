# Runtime plan verification: gate, refresh, recover

*Lit slice 2026-08-08 ~01:1xZ, read while the #6 self-subgoal arms
were decoding — the escalation side of tonight's readout. Three
papers that all ask: once a policy (or its planner) has committed to
a plan, who checks it at runtime, and what happens when the check
fails? Sources: SV-VLA
([2604.02965](https://arxiv.org/abs/2604.02965), under review
04-2026), Do What You Say
([2510.16281](https://arxiv.org/abs/2510.16281)), VINE
([2512.03913](https://arxiv.org/abs/2512.03913)).*

Our #6 probe conditions an action decode on a self-generated subgoal
*once, per frame, offline*. Every escalation beyond it — refresh
policies, gated conditioning, candidate selection — is a runtime-loop
design. These three papers are the current published shapes of that
loop.

## SV-VLA — verify cheap, replan heavy (2604.02965)

**Contribution.** Chunked open-loop execution is fast but blind;
per-step closed-loop is robust but slow. SV-VLA runs a heavyweight
VLA as a low-frequency macro-planner (chunk K=64) and a *17×-cheaper*
verifier (frozen ViT-Tiny + the planner's context feature, 0.081 s
vs 1.373 s per call) at high frequency. The verifier predicts a
"reference action" from the current observation plus the original
plan intent, compares it to the pre-planned action by normalized L1,
and triggers a full replan only when the discrepancy crosses τ=0.2.

**Experiments.** LIBERO (Goal/Spatial/Object): base K=8 96.0% at
1.0× speed; base K=64 79.5% at 3.15×; SV-VLA 90.9% at 2.17× —
+11.4 pts over open-loop at two-thirds of its speedup, biggest gains
exactly where drift hurts (Object +18.1, Spatial +15.0). The
ablation table is the real payload: drop the planning-context feature
→ 73.7%; drop the current observation → 63.7%; **keep verification
but drop replanning → 15.5%**. A gate with no recovery path is worse
than useless. Threshold sensitivity is real (τ=0.1 → 83.1%, τ=0.4 →
77.4%) and the authors name adaptive triggering as open.

**Transfers / doesn't.** The design lesson transfers whole: any #6
refresh policy needs (cheap drift score → gated re-decode →
*recovery*, not just detection), and the monitor should carry the
plan intent, not just the current frame. The numbers don't: LIBERO
closed-loop success under injected dynamics is a different regime
from our offline panel, and their verifier is *trained* (L1
regression on K−1 steps) — a data+training cost our zero-training
rung deliberately avoids. Also directly relevant to **#22** (async
staleness): this is a drift monitor for chunked execution,
competing on cost with PAINT/TT-RTC-class answers.

## Do What You Say — the faithfulness gap (2510.16281)

**Contribution.** Names and measures *embodied CoT faithfulness*:
a reasoning VLA can emit a correct textual plan and then execute
actions that don't follow it. Their runtime fix: sample several
candidate action sequences from the same model, predict each one's
outcome (simulation), and let a pre-trained VLM pick the sequence
whose outcome best matches the model's own stated plan. No
retraining; up to +15 pts on behavior-composition tasks on a
reasoning-annotated LIBERO-100 with OOD perturbations.

**Transfers / doesn't.** The concept is the mirror image of tonight's
read. Our Δ_oracle/Δ_self split prices *plan generation* noise
(stage 1's phase-offset rows); their faithfulness gap is *execution*
infidelity — text right, actions wrong. A conditioned decode can fail
either way, and the two need different fixes (better planner vs
selection-by-alignment). If tonight lands "oracle helps, self
doesn't," their sample-and-align is a named escalation — but an
expensive one (N action draws + outcome prediction + a VLM judge per
decision), and their outcome-prediction step leans on simulation we
don't have. The cheap fragment that does transfer: alignment
*scoring* between subgoal text and decoded actions could reuse our
existing draws machinery (#19) with the subgoal as the scoring
target rather than a reward model.

## VINE — subgoal search with failure-aware values (2512.03913)

**Contribution.** A hierarchical system where System 2 does batched
MCTS over *candidate subgoals* (nodes = 2D scene graphs, edges =
verifiable subgoal transitions), scored by a value function trained
on successes AND failures — "probability of reaching the goal before
the failure set" — and System 1 executes the chosen subgoal with a
flow-matching policy. Test-time compute scales in the expansion
width: K=1→5 lifts unseen plug-insertion 28.9%→44.4% (peak K=4,
latency ~linear 23.9 s→32.5 s). Failure conditioning alone is +46%
relative (28.9→42.2). Uncertainty-triggered replanning adds +6.5 pts
on drawer packing *without retraining*.

**Experiments.** MuJoCo plug insertion / drawer packing + Simpler +
real sponge/towel packing; beats π0 (42.2 vs 26.7 unseen insertion;
65 vs 55, 55 vs 30 real-world) and VLM-as-planner variants
(GPT-4o+failures 68.3 vs their 75.2 on drawers).

**Transfers / doesn't.** Two fragments transfer. (1) *Draws over
subgoals*: their expansion-width scaling says candidate-subgoal
diversity is worth test-time compute — for us, pass 1 of the self
arm could sample N subgoal texts (our batched-draws instrument, #1)
and score them, a rung strictly cheaper than their MCTS since our
"tree" is depth-1. (2) Replanning-without-retraining echoes SV-VLA
from an independent group. What doesn't: the load-bearing piece —
the failure-aware value function — needs failure-labeled
demonstrations we don't have (our corpus is success-only, like
their System 1, which they name as *their* limitation); and the
scene-graph abstraction presumes an object-relation extractor.

## What this fed

- **#6 escalation map**, ahead of tonight's readout: the ladder
  above rung (a) now has published shapes with numbers — refresh =
  SV-VLA's gate (cheap monitor carrying plan intent + mandatory
  recovery; threshold sensitivity is the named open problem);
  selection = VINE's subgoal-draws (test-time width scaling, peak at
  small K) or Do-What-You-Say's alignment pick (needs outcome
  prediction). All three are rollout-granularity — none change the
  frozen reads or the E5 falsifier; they price what an escalation
  pre-reg would cost if Δ_self earns one.
- **#22 async staleness**: SV-VLA is a direct competitor entry for
  the chunked-execution drift-monitor slot (PAINT → A2C2 → TT-RTC
  ladder) — training-required but 17× cheap at runtime.
- **#19/#1 bridge**: "sample N subgoals, score, condition on best"
  is implementable with the existing batched-draws instrument —
  banked as a hook, gated (like everything here) on tonight's
  Δ_oracle being alive.
