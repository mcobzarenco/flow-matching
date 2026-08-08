# Progress from logits: zero-shot phase signals for the planner

*Lit slice 2026-08-08 ~04:1xZ (standing allocation), targeted at the
#6 rung-(b) escalation routing before its execution reads out. Rung
(a) located the self-subgoal failure in single-frame **phase
estimation** — the model plans a valid step of the task, just often
the wrong one (~10/60 stage-1 rows phase-offset). This slice asked
three questions: does anything published beat self-certainty
label-free at inference (lane a); what exists for phase/progress
estimation beyond a single frame (lane b); and is MG-Select's
masked-contrast prerequisite actually met for us (lane c — answered
in the [self-certainty page's](self-certainty.md) corrected
escalation note). Sources: "TOPReward: Token Probabilities as Hidden
Zero-Shot Rewards for Robotics"
([2602.19313](https://arxiv.org/abs/2602.19313)) and "ProgVLA:
Progress-Aware Robot Manipulation Skill Learning"
([2605.28231](https://arxiv.org/abs/2605.28231)). Cross-references:
[test-time selection](test-time-selection.md),
[runtime plan verification](runtime-plan-verification.md).*

## TOPReward — ask the logits, not the text

**The contribution.** Zero-shot task-progress and success rewards
read directly from a video VLM's token logits. Given a trajectory
*prefix* and the task instruction, pose one binary completion query —
"Does this trajectory complete the task? True or False" — and take
`log p("True")` as the reward for that prefix; min-max normalize
within the episode to get a [0,1] progress curve. No training, no
fine-tuning, no reward model: the signal is one logit of a frozen
VLM. The prior zero-shot state of the art (GVL) asked the model to
*generate* calibrated progress numbers as text, and that is exactly
where it dies on open models — LLMs generate badly calibrated
numerals. Reading the logit instead of sampling the number is the
whole trick.

**The experiments.** On ManiRewardBench (130 tasks, 653 episodes,
Franka/SO-100/YAM arms) TOPReward on Qwen3-VL-8B reaches 0.947 mean
VOC vs GVL's 0.332; on Open X-Embodiment 0.857 vs 0.194 — GVL is
near chance on open-source VLMs while the logit probe is not. Models
tested include **Molmo2-8B — our trunk family**. Downstream:
success-detection ROC-AUC 0.654 (GVL 0.519 ≈ chance), and
advantage-weighted behavior cloning with its progress curves lifted
real SO-100 task success ~43%. One ablation worth remembering:
running the query through a **chat template halves the VOC** — the
probe works when it aligns with the raw next-token pretraining
objective, which rhymes with our raw prompt-slot rendering.

**What transfers.** The load-bearing fact for us is *what the input
is*: a video **prefix**, not a frame. Progress is recoverable
zero-shot — on our own trunk family — when the model sees history;
our planner pass decodes its subgoal from a single frame, and
single-frame phase estimation is precisely the rung-(a) failure
mechanism. That makes "condition the planner on history" (or:
re-rank candidate subgoals with a TOPReward-style logit probe over
recent frames) a *nameable, evidence-backed* escalation if rung (b)
reads no-scorer — at the cost of feeding multiple frames through the
planner pass. Caveats: per-episode min-max normalization means no
absolute cross-episode scale (per-frame candidate ranking, our use,
is unaffected); the paper never selects among language plans, so it
is **not** a scorer competitor to self-certainty — lane (a) stands:
nothing published beats SC label-free on open-ended text selection.
(RoVer and EVE, surfaced in the same sweep, are *trained*
verifier/reward modules — named escalations under our
no-labels-at-inference constraint, not competitors.)

## ProgVLA — progress as a training-time reweighter, not an inference signal

**The contribution.** A 0.1B flow-matching policy that co-trains
progress heads (an expectile state-value head, a near-completion
success classifier, and a state-action critic, all on the policy's
own context tokens) against a purely temporal remaining-horizon
target — then uses their detached predictions only as
**multiplicative weights on the imitation loss**, up-weighting
high-advantage, near-completion samples. The heads are never
deployed; inference is the plain policy.

**The experiments.** LIBERO 91.1% average (88.6% long-horizon),
beating SmolVLA-2.25B by 2.4 and OpenVLA-7B by 14.6 points at ~20×
fewer parameters; Meta-World +7 to +21 on the harder tiers. But the
ablation column that matters: removing the progress objectives costs
only **−2.3 points** (concentrated on long-horizon), vs −16.0 for
their context resampler and −13.5 for unfreezing vision — the
progress machinery is the *smallest* lever in their own accounting.

**What transfers, and what doesn't.** We already co-train aux
progress/subgoal heads (that is idea #6's whole premise, and aux-off
cost us +0.462 — a far bigger effect than their −2.3, though ours
measures representation shaping, not loss reweighting).
Progress-weighted imitation is a training-side idea, off the current
inference-rung path; parked. Crucially ProgVLA never ablates
single-frame vs history-conditioned progress — it does not unblock
lane (b) at inference. The history direction is instead carried by
TOPReward above (and by DIM-WAM's multi-scale historical memory in
world-action models — same "history fixes phase" direction, much
heavier; one-liner only).

## What this fed

- **#6 escalation routing (the point of the slice)**: if rung (b)
  reads **no-scorer** (ceiling ≫ bon), two escalations are now
  mapped with prerequisites checked: (1) **masked-contrast
  selection** — prerequisite *met*, see the corrected note on the
  [self-certainty page](self-certainty.md): the subgoal-masked
  reference is our planner-less forward, trained at 50% dropout,
  N+1 teacher-forced action passes; (2) **history-conditioned
  planning** — TOPReward shows phase is zero-shot recoverable from a
  video prefix on our trunk family; the planner-side fix attacks the
  measured ~10/60 phase-offset mechanism directly. If rung (b) reads
  **no-diversity**, none of this fires and the family closes.
- **Lane (a) verdict banked**: no published label-free
  inference-time scorer beats self-certainty on open-ended text —
  the rung-(b) frozen scorer cell survives this slice unchanged.
- **Training-side (parked)**: progress-weighted imitation (ProgVLA)
  and TOPReward-style advantage-weighted BC are data-flywheel ideas
  for a future training rung, not this family.
