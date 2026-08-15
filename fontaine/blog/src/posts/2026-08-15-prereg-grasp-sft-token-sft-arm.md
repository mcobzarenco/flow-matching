# Pre-reg (DRAFT, owner-gated) — token-SFT arm: measured competence for the discrete head, via bijou.train

*2026-08-15, ~14:4xZ. This is the arm that [GRPO R2 Amendment
A1](2026-08-15-prereg-grpo-r2-post-sft.md) §7 decision 2 requires: the
owner's registered direction (10:14Z exchange) is that if token-GRPO
is the tool, a token-SFT run must precede it, with its own
pre-registration and its own sim eval. Status: **DRAFT — doubly
owner-gated** (the token-GRPO route itself is an owner choice, and the
GPU is owner-reserved since 13:35Z). Nothing here launches.*

**Plain words.** Our robot model has two "mouths" it can speak actions
through: a *flow* head (the one we fine-tuned on 313 scripted
demonstrations — it went from 9 to 28 successes out of 100) and a
*token* head that spells actions as discrete symbols (the one
reinforcement learning would actually push on, per the owner's stack).
The fine-tuning never touched the token head — it still speaks with
the factory weights, which on our task almost never succeed. Running
reinforcement learning on a head that can't yet do the task is the
exact mistake our earlier experiments priced: the training signal is
noise when there are no successes to compare against failures. This
document freezes the fix: teach the token head the same 313
demonstrations first, measure it on the same 100 held-out scenarios,
and only if it passes the same competence bar (≥ 20/100) does the
reinforcement-learning plan reactivate on top of it. One extra
wrinkle: the symbol encoding squashes actions into a fixed range
before spelling them out, using the same "typical range" table that we
found corrupted this morning — so this run must also start from the
corrected-table base we already built.

## §1 Hypothesis, and why this arm exists

**H:** SFT on the grasp demo set moves the **discrete head** into the
competence band (≥ 20/100 unseen) the same way it moved the flow head
(9 → 28), giving token-GRPO the competent base its §1 premise
requires.

The seam this closes (A1, owner-caught): stage-C SFT trained only the
action expert; the AR/FAST token head — the surface
`bijou/grpo_replay.py` trains — still carries released weights. Every
banked read of that head is floor-level: R1-B held-out 2/20, phase-2
waves 4/3/3 of 64. Token-GRPO from today's checkpoints would push on
an incompetent policy, which is precisely the banked-negative R1
shape.

**The table seam applies to this head too (verified in code this
session):** `bijou/fast/codec.py` normalizes action chunks with the
baked q01/q99 table *before* DCT+BPE encoding, and unnormalizes on
decode — the token targets and the served actions both ride the
quantile table. Training or serving this head under the corrupt table
would bake the same wrist_roll clamp distortion (`[35.5, 94.4]` vs
true `[−157.2, 157.2]`) into the token stream. This arm therefore
inits from the **corrected-table conversion** already landed
(`~/checkpoints/converted/molmoact2_base_corrected_stats_v0`, retrain
pre-reg [§2](2026-08-15-prereg-grasp-sft-retrain-corrected-table.md)).

## §2 Frozen run params (proposal; finalization may adjust LR only)

```
uv run python -m bijou.train \
  --train-data ~/datasets/fontaine/grasp_sft_demos_v0 \
  --init-from  ~/checkpoints/converted/molmoact2_base_corrected_stats_v0 \
  --objective  ar              # CE over trunk-native token rows \
  --backbone-text-lr 1e-5 \
  --steps 2000 --batch-size 64 \
  --save-every 500 \
  --save-dir ~/checkpoints/finetune/fontaine_grasp_sft_token_ar
```

- `--objective ar` trains the **trunk's text layers only** (the AR
  head owns no parameters; no expert is built, `--expert-init` is
  refused). The released FAST codec rides the checkpoint — no
  `--fast-tokenizer` flag on this path.
- **LR 1e-5** is the repo's measured AR-objective smoke value
  (docs/architecture.md objective-matrix oracles); 2e-5 (the
  historical trunk-recipe value) is the registered alternative,
  judged at finalization against the first-poll loss trace. Note this
  is a *backbone* LR — a 5e-5 decoder-class value is NOT proposed for
  trunk weights.
- Steps/batch match the stage-C endpoint actually probed (2000 ×
  gb64, ~2.4 epochs of 54k frames) — deliberate, so the flow-head
  result (28/100 at this exact data budget) is the cross-head
  comparison row.
- Fresh seed n/a — new run, no resume lineage (house policy).

## §3 Eval protocol + anchors (frozen)

Verbatim the step2000 probe protocol — **unseen 0–99**, then train
band 1000–1099, 30 s episodes — except serving: the AR head's
grammar-masked **greedy** decode (the objective matrix's `ar` serving
convention), not euler-10.

| row | unseen /100 | status |
|---|---|---|
| this arm @2000 | ? | **primary** |
| stage-C flow head @2000 (corrupt table) | 28 | cross-head context |
| released base, flow head | 9 | context |
| released base, token head | not banked | optional anchor leg, see below |

**Primary read**: unseen count vs the R2 activation bar — **≥ 20/100 →
R2's competent-base premise holds for the head it trains**;
finalization there proceeds per its §6 with this endpoint as the base.
5–19 → the head lags its flow sibling materially — owner decision
(iterate vs re-scope R2 to the flow head vs park). < 5 → token-SFT
did not transfer; token-GRPO stays parked and the discrepancy vs the
flow head's 28 is itself the finding.

**Optional anchor leg** (finalization decision, +~1.3 GPU-h): base
token-head sim100 — makes the SFT delta a measured pair instead of
inferring the floor from R1-B's 2/20 (different protocol, 20 seeds).
Default: run it; it is the row every downstream claim divides by.

## §4 Route context — three ways to spend the next SFT GPU-hours

For the owner's route decision, the three live options and what each
buys (this pre-reg is option B; A is the [retrain
draft](2026-08-15-prereg-grasp-sft-retrain-corrected-table.md)):

- **A — flow retrain, corrected table** (~5.5 GPU-h): prices the
  table-fix cleanly on the head we've measured. No token-GRPO
  progress.
- **B — this arm** (~5–6.5 GPU-h): unlocks token-GRPO per A1. No
  table-fix pricing on the flow head.
- **C — one `--objective joint` run** (L_flow + λ·CE, λ=1.0
  default): both heads in one budget — BUT it confounds the retrain
  draft's table-fix read (objective changes alongside the table) and
  the joint recipe class is unmeasured on our data. If the owner
  wants both heads warm and accepts the confound, C replaces A+B;
  the pre-regs would merge under a registered amendment.

Sequencing note: A and B are independent runs from the same corrected
base and can go in either order; neither blocks the other's read.

## §5 Budget

Train ~3–4.5 GPU-h (backbone-text training is heavier than the
AE-only 5.2 s/step; pace pinned at first poll, standing util check) +
probe eval ~2.5 + optional base anchor ~1.3 = **~7–8 expected, gate
≤ 9**. Detached unit + babysit entry at launch per standing rules.

## §6 Checkpoint-format note (owner main `4fd6875`, 13:56Z today)

The phase-3 VLA checkpoint format landed on main (`bijou/checkpoint.py`
+ `convert_legacy`) after our tooling was built. Two implications
recorded now so finalization doesn't trip on them: (1) this run's
endpoint (format-3 today) should get a `convert_legacy` pass +
`validate_checkpoint` before it becomes R2's pinned base — the
receipt R2 §6.2 wants is cleanest in the new format, where the stats
table and its provenance (`stats_note`) are first-class metadata; (2)
`convert_legacy --replace-stats` is the format-level spelling of our
`--norm-stats-from` seam — if the owner's convention lands before
this arm launches, the corrected-table declaration should ride that
flag rather than our side-channel provenance JSON. Our launchers and
babysit readers parse `bijou_config.json` (format 3) and will need a
small follow-up when `bijou.train` starts writing the new format —
flagged, not blocking.

## §7 Gating

**Nothing launches from this document.** Order of gates: (1) owner
route choice (A/B/C above — B activates this page); (2) GPU freed
(owner-reserved since 13:35Z); (3) finalization (LR decision, anchor
leg decision, HEAD re-pin, objection window) — then launch at the
next free boundary. If the owner instead re-scopes R2 to the flow
head, this page parks with the token-GRPO lane.
