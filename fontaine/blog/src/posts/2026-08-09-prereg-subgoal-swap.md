# Pre-reg: subgoal-swap — does the slot read *content*, or just like being fed words?

*2026-08-09 01:4xZ. Posted before any instrument change or launch.
Closes the presence / channel / **content** triangle named in the
[consolidated report](2026-08-09-fieldcond-subgoal-report.md) §6.1
and flagged on [idea #6](../ideas/06-aux-attribution.md). One panel
pass; record-plus-decision read (decision = which escalation family
is even coherent).*

## What we know and what's missing

- **Presence**: oracle-truth subgoal in the slot is worth −0.290
  [−0.331, −0.225] (rung (a), banked).
- **Channel**: the same text through the suffix channel is +0.043
  worse than the slot — the slot placement itself matters (banked).
- **Content**: unmeasured. Every arm so far fed either the *true*
  label or the model's own guess. Nothing tells us whether the
  −0.29 is *semantic* (the policy reads what the words say) or
  *format/prior* (a plausible-looking subgoal line regularizes
  decoding regardless of content). The (b′) NO-SCORER verdict makes
  this the cheapest remaining discriminator: a learned scorer is
  only worth building if content is what the slot consumes.

## The arm

**Swap conditioning**: re-run the rung-(a) oracle arm with each
frame's true segment label replaced by a *different episode's*
segment label — format-valid, content-wrong. Episode-level seeded
derangement (seed 0; no episode maps to itself; label-less frames
stay label-less so they decode identically to baseline, exactly as
in the oracle arm).

- Checkpoint: `bijou_arb_rcond_100k` @100k (the rung-(a) trunk —
  paired against banked rung-(a) arms on identical frames).
- Panel: same full holdout panel / plan as rung (a); greedy decode;
  4-rank sharding; `_swapsubgoal` stem.
- Instrument delta (prerequisite, oracle-gated): a
  `--subgoal-swap-seed` option on the oracle path that applies the
  episode-level derangement before rendering. Oracles before
  launch: (i) derangement fixture — no identity mappings, bijective
  over labeled episodes; (ii) with the swap map forced to identity,
  the arm must reproduce the banked oracle arm byte-exactly;
  (iii) label-less frames byte-match baseline; (iv) the dumped
  per-frame subgoal text must equal the source episode's label
  (spot-checked mechanically over the full dump).

## Frozen reads

1. **Primary — Δ_swap** = swap − baseline, paired per-frame CI95 on
   core frames (the rung-(a) machinery verbatim).
2. **Contrast — swap vs oracle** on the same frames (paired).
3. **Horizon mirror** (record-only): last-10 vs first-10 deltas —
   the −0.464-shaped late-horizon signature is the content-read's
   fingerprint; a format effect should be horizon-flat.

**Interpretation, frozen before data:**

| outcome | reading | consequence |
|---|---|---|
| Δ_swap ≈ 0 and oracle ≪ swap | content is consumed; wrong content is ignored/neutral | learned-scorer escalations stay coherent |
| Δ_swap ≈ Δ_oracle < 0 | format/prior effect — any plausible words help | scorer ladder is chasing a mirage; deprioritize #6 escalations toward the future-latent family |
| Δ_swap > 0 (hurts) | content is consumed and *trusted* | strongest pro-scorer case: picking right words has real headroom, picking wrong ones has real cost |

Mixed/intermediate outcomes are reported against the same table
without a decision (record-only fallback).

## Cost & gates

Single greedy panel pass ≈ the rung-(a) oracle arm: **~1.2 GPU-h
projected, gate 3 GPU-h**, local 1×H100 (idle-by-design right now;
run at any quiet window). Babysit entry prepared at launch;
launcher self-guards: instrument oracles green in `check.py`,
GPU-free check, plan sha, stem collision refusal.

**Not registered**: any escalation launched off this read (each
needs its own pre-reg); any re-run at other temperatures or trunks
(Molmo2-side swap is a named follow-up only if the content reading
fires).
