# E4B screen: NO-LAUNCH — the memory ladder exhausted; E4B does not fit this recipe on 80 GB

*2026-08-06, ~05:4xZ. The pre-registered terminal branch of the
[E4B screen pre-reg](2026-08-05-prereg-e4b-screen.md) fired: the
B12 memory smoke ran all four ladder rungs (B12 direct → 2×6 → 3×4
→ 4×3 chunked backward) on a box H100-80GB and **every rung OOM'd
before completing a single optimizer step**. Per the pre-reg — "if
even 4×3 doesn't fit: do not launch; post the finding and take the
follow-on decision to the owner" — E4B does not launch under the
matched mainline recipe. This post is the finding; the measured
detail lives in the pre-reg's finalization Amendment 2.*

## The measurement

`smoke_e4b_b12.sh`: 1×H100, the exact mainline recipe (verbatim
E2B flags + `--backbone google/gemma-4-e4b-it`), 60 steps at loader
batch 12, 2-s VRAM sampler, expandable segments on, box code
`9ddcfe3` (chunked backward + oracles landed per Amendment 1).

| Rung | Config | Outcome | Sampler peak (MiB / 81,559) | Died at |
|---|---|---|---|---|
| 1 | B12 direct | OOM | 81,035 | before first logged step |
| 2 | 2×6 chunks | OOM | 81,059 | forward SDPA, first train_step |
| 3 | 3×4 chunks | OOM | 81,035 | backward, first train_step |
| 4 | 4×3 chunks | OOM | 81,049 | backward, first train_step |

All four rungs printed identical E1 selection/model lines (878
datasets, 42,872 episodes, dims 6/6; 42 layers, text 3,975.3M live
params) and the correct per-rung chunked-backward banner — the
ladder was exercised as registered.

## Why this is decisive, not a near-miss

No rung reached `optimizer.step()`, so **Adam's fp32
exp_avg/exp_avg_sq for the 3,975.3M live text params — ~31.8 GiB —
were never allocated**. The OOMs at ~78 GiB torch-allocated happened
during the *first* forward/backward. Steady-state training therefore
needs roughly **≥110 GiB/rank**: fp32 masters (~15.9) + fp32 grads
(~15.9) + bf16 weights (~8) + frozen embedding/PLE tables + ~32 GiB
Adam + activations that alone overflow the remainder even at
3-sample chunks. Batch chunking reduces only the activation term; it
cannot close a fixed-cost gap of ~30 GiB. Consistency check: the E2B
arms (text ~1.8B live) ran 71–75 GiB at the same B12 recipe — scaling
the live-trunk memory terms ~2.2× lands exactly where the smoke died.

## What this is as a datum

- **The attribution question stays open, not answered.** This is a
  *feasibility* negative (E4B live-trunk + fp32-master AdamW recipe
  > 80 GB/rank), not evidence about whether trunk scale helps — the
  probe/panel gates never ran. The external prior
  ([VLM-to-VLA redundancy](https://arxiv.org/pdf/2606.31382): bigger
  backbones don't consistently help after adaptation) is unrefuted
  and untested by us.
- **The screen's zero-port-cost premise is dead as-is.** E4B was
  ranked rung 1 *because* it was in-family and config-driven; a
  memory-recipe change spends real implementation+oracle budget, at
  which point rank 2 (Molmo2-4B) competes on closer-to-even terms.

## Follow-on options (owner decision — none is pre-registered)

Any E4B re-entry is a **new pre-reg with a changed recipe**, not an
amendment — the matched-recipe premise is what died.

1. **Optimizer-state sharding (ZeRO-1 / `ZeroRedundancyOptimizer`)
   across the 4 ranks** — Adam m/v ~31.8 → ~8 GiB/rank, gradient
   math unchanged, recipe otherwise verbatim. Cheapest faithful
   re-entry; ~24 GiB saved likely fits 2×6 or 3×4, but that is a
   *prediction, not a measurement* — a new smoke ladder would gate
   it. Cost: impl + oracles + resume-path audit (~one work session)
   before any launch.
2. **Activation checkpointing on the trunk** — attacks the wrong
   term alone (fixed cost still ~80 GiB with Adam); only viable
   *combined* with option 1. More moving parts, more oracle surface.
3. **Recipe change at the precision layer** (bf16 optimizer states /
   8-bit Adam / no fp32 masters) — breaks matched-recipe comparison
   worst of all three; the E2B baseline would arguably need a
   matching re-run to keep the comparison honest.
4. **Drop E4B, redirect the box** — take the feasibility negative as
   the datum, move the trunk question to **Molmo2-4B (survey rank
   2)** on its own pre-reg timeline, and give the box to the **#11
   grounding arms** (the kill/tie branch's designated use) or the
   E4B slot's GPU-hours to the queue. The lit prior (state-shortcut
   mechanism, probe running tonight) currently points at grounding,
   not scale, as the binding limit.

My recommendation, stated for the record: **option 4 now, option 1
as a queued candidate** — the box has pre-registered work waiting
(#11 grounding arms follow the state-probe read landing ~06:1x–06:4xZ),
and option 1's session of infra work should compete for its slot
against Molmo2-4B rather than pre-empt the queue by default.

## Cost of the finding

~50 min of 1×GPU smoke time (four rungs, 04:31–05:21Z), zero launch
hours burned, the 26–31 h run never started — the pre-registered
ladder did exactly what it was for.
