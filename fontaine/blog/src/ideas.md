# Ideas

The backlog, one page per idea (sidebar, or the index below). Every
idea page carries: hypothesis, expected effect, cost, cheapest
falsification, and the dated record of everything that has happened
to it since. Seeded 2026-08-05 from charter §8 (which distills the
mainline ledger, `docs/architecture.md` §7–8). Status tags: `queued`
/ `screening` / `running` / `confirmed` / `falsified` / `parked`.

This page is the **index**: what is hot right now vs what is on ice.
It is updated whenever an idea moves (the per-idea page is the
record; the line here is the hook). *Index last updated 2026-08-07.*

## Hot — actively pursued

- **`ar-draws` [#19 AR sampled-draws eval](ideas/19-ar-sampled-draws.md)** —
  `screening`. The AR side of the draws fairness programme.
  draws10_t1 read out 2026-08-07: all three pre-registered
  expectations met (Δ_AR −0.145, ~9× smaller than the flow gain —
  the mean-collapse shape). Next: T-sensitivity q4 rung (launcher
  ready), then the molmo2 arm at its endpoint.
- **`seam-screen` [#4 Stage-2 attachment seam](ideas/04-stage2-attachment.md)** —
  `screening`. F (frozen) vs K (KI-joint) screen pre-registered;
  instruments, launchers, smoke ladder and frozen-read script all
  landed oracle-gated. Launches at the molmo2 40k endpoint (~08-08).
- **`new-trunks` [#17 New trunks / architectures](ideas/17-new-trunks.md)** —
  standing owner mandate; the Molmo2-4B AR 40k trunk run is LIVE on
  the box (endpoint ~08-08), K1 gate crossed green.
- **`aux-subgoals` [#6 Aux attribution](ideas/06-aux-attribution.md)** —
  `confirmed` (aux HELPS actions, +0.462 cost when off), and its
  rung-(a) self-subgoal probe is pre-registered, waiting on the next
  quiet local-GPU window.
- **`noise-draws` [#1 Noise-draw ensembling](ideas/01-noise-draw-ensembling.md)** —
  flow mean-of-10 banked (5.365, beats the AR anchor on both
  columns); fairness + energy-score reads in; batched draws merged
  2026-08-07 — mean-of-N at single-draw latency (teacher 9.1×,
  student 2.5× single-stream). Next candidate rung: the
  golden-ticket noise screen (needs its own pre-reg).

## Standing

- **`rig-benchmark` [#16 Few-shot rig-transfer benchmark](ideas/16-rig-transfer-benchmark.md)**
  — **the north star**; execution parked by owner (better rig data
  later), instruments banked. Short-term proxy: comm-holdout MAE +
  attribution. New 2026-08-07: the proxy itself got a lit slice
  ([offline-validation](papers/offline-validation.md) — raw MSE
  measured at ρ −0.61 vs rollout success, sign flips exist);
  critical-frame re-pooling rung banked (CPU, existing npz dumps).
- **`lit-arms` [#15 Literature-sourced arms](ideas/15-literature-arms.md)** —
  the arXiv radar; every borrowed idea cites its source, every
  "novel" idea gets a search first. Feeds the Papers section.
- **`infra-hardening` [#18 Instrument & infra hardening](ideas/18-infra-hardening.md)**
  — the bijou deep-dive fix queue + everything oracle-shaped;
  several items done, rest queued by leverage. New 2026-08-07: item
  9 async checkpoint saves LANDED (owner HIGH; byte-identical
  oracle, ~14% wall-time payoff at the attach screen) + its
  [checkpointing-systems lit page](papers/checkpointing-systems.md).

## On ice — queued or parked, each with its named trigger

- **`throughput-compile` [#2 Throughput: bucketing + compile](ideas/02-throughput-bucketing-compile.md)**
  — 2a landed; GPU A/B conditional on a widened-selection corpus
  (padding ceiling too small under the current recipe).
- **`longer-training` [#3 Longer training](ideas/03-longer-training.md)** — needs the
  own-baseline reference arm first.
- **`tokenizer-v3` [#5 FAST tokenizer v3](ideas/05-fast-tokenizer-v3.md)** — CPU
  refit on curated-v0 quantiles; token metrics reset; entropy/
  utilization gate before any learned-VQ arm.
- **`stream-schedule` [#7 Stream-schedule re-test](ideas/07-stream-schedule.md)** —
  enters at the short-run screen rung.
- **`vocab-head` [#8 Shortlist/output-vocab head](ideas/08-shortlist-vocab-head.md)**
  — VRAM lever for ar_backbone; design concretized, unbuilt.
- **`data-levers` [#9 Data levers](ideas/09-data-levers.md)** — state-dropout arm
  C answered "adopt nothing"; p=0.3 branch survives on our own
  branch rule only; calibrated-noise/GAP are the literature levers.
- **`base-vs-it` [#10 E2B base-vs-IT swap](ideas/10-e2b-base-vs-it.md)** —
  backbone-swap arm, pre-registered prediction ±0.2.
- **`visual-grounding` [#11 Visual grounding arms](ideas/11-visual-grounding.md)** —
  the open front; arch batch #1 pre-registered, arm A (img280) HELD
  for a fresh owner go.
- **`one-step` [#12 Solver/Heun-gap + 1-NFE distill](ideas/12-solver-heun-gap.md)**
  — SnapFlow 1-NFE student banked (holds the panel, single draw
  beats AR); rig fine-tune diagnosed, next rung opens with rig data
  (#16).
- **`sign-convention` [#13 Sign-convention repair](ideas/13-sign-convention.md)** —
  stage 2 hit the escalation branch (3/4 reference populations not
  sign-consistent); parked pending a decision on the reference set.
- **`async-staleness` [#22 Async staleness bridging](ideas/22-async-staleness.md)** —
  RTC-class rollout question; parked, waits on #16 (closed-loop by
  construction).

## Answered — banked results

- **`wrap-census` [#14 ±180° wraparound census](ideas/14-wraparound-census.md)** —
  measured: 1.24% of pooled panel MAE; under the gate, banked.
- **`activation-ckpt` [#20 Activation checkpointing](ideas/20-activation-checkpointing.md)**
  — landed + oracle-gated; the GPU ladder lives on as #4's K smoke
  item.
- **`loop-review` [#21 Agentic-loop deep review](ideas/21-agentic-loop-review.md)**
  — CLOSED: P1–P7 all landed, owner-signed.
