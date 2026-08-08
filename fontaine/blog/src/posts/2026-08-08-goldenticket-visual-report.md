# Golden tickets, the whole story — a visual report (#1)

*2026-08-08. Owner-requested consolidation (steering 08:42Z): the
golden-ticket thread ran across a
[pre-registration](2026-08-07-prereg-golden-ticket-screen.md), a
[results post](2026-08-08-goldenticket-results.md), three frozen
stage analyses, and a
[selector side-read](../papers/noise-space-steering-3.md) — this page
subsumes them into one chart-led report. Every number is read from
the banked stage JSONs
(`analysis__goldenticket_stage{1,2,3}.json`); charts are rendered by
`fontaine/scripts/goldenticket_report_charts.py` from those files
and nothing is re-computed.*

## The idea in one paragraph

A flow-matching policy turns a noise vector into an action chunk
through a deterministic ODE. The *Golden Ticket* observation
(banked on the [noise-steering pages](../papers/noise-space-steering.md))
is that some noise vectors are systematically better than others —
not per-sample luck, but a reusable property of the vector. The
screen asked, with every read pre-registered: draw **64 i.i.d.
"tickets"** (sha-pinned noise vectors), score them once on a probe,
and check whether the winners are (R1) wider-spread than chance,
(R2) real on held-out rows, (R3) still better inside the ensembling
regime, and (R4) *where* the advantage lives. Total cost ~5.55 GPU-h,
zero training.

## Headline numbers

| read | question | number | verdict |
|---|---|---|---|
| R1 | is the ticket spread real? | sd **0.823** vs null line 0.0785 | **CONFIRM** (~12× the null) |
| R2 | does the winner hold on held-out rows? | **−0.924** [CI95 −0.985, −0.866] vs adopt line −0.05 | **REAL** |
| R3 | searched ensemble vs random ensemble? | **5.1847/1.3831** vs 5.3645/1.4242 (Δ −0.180, band ±0.02) | **INTERESTING** (record-only) |
| R4a | is the winner universal? | argmin in **4.4%** of 792 datasets | task-local |
| R4b | where does it buy? | quartile gains **−0.35 → −1.44** | monotone in dispersion |

## R1 — the spread is ~12× the i.i.d. null

If tickets were interchangeable, 64 probe scores would scatter with
σ ≈ 0.067 (the frozen null, computed from banked per-draw variance
before any data). Measured: **sd 0.823**, minimum 5.706 vs an
expected-null-minimum of 6.587. The distribution isn't a noisy
constant — it has a long bad tail and a usable good tail.

![R1: 64 tickets sorted by probe score, against the frozen null band](../img/goldenticket/r1_tickets.svg)

## R2 + R3 — real on held-out rows, and the searched ensemble wins

R2 is the confirmatory read the screen lived or died on: the winner
ticket, judged only on **complement rows it was never selected on**,
paired per-frame against the banked stable-key default. It landed
**−0.924**, eighteen times past the adopt floor. R3 then asked
whether search survives *ensembling* — mean-of-top-10-tickets vs the
banked mean-of-10-random-draws: **−0.180** against a ±0.02 tie band,
the best chunk *and* first-step numbers measured on this panel by
any config. (Record-only per pre-reg: the banked comparator retained
no per-frame npz, so the paired re-run that could seat it as a
leaderboard row is folded into the
[per-dataset pre-reg draft](2026-08-08-prereg-noise-ladder-perdataset.md).)

![R2 and R3 deltas with their decision lines](../img/goldenticket/r2_r3_deltas.svg)

## R4a — every ticket wins somewhere

The free stage-1 read that reframed the whole thread: per-dataset,
the global winner is argmin in only **35/792 datasets (4.4%)** — and
it isn't even the most task-general ticket (a blue top-10 ticket
wins 62). The top-10 set contains the per-dataset argmin **29.8%**
of the time, ~2× the 15.6% null. The published analog
([2603.11642](https://arxiv.org/abs/2603.11642)): noise main effect
1.4%, context×noise **interaction 39.4%**, best shared noise optimal
in 3.1% of contexts. Loud caveat, banked in advance: the median
dataset has **2 probe frames** — these per-dataset winners are
hypotheses for the next rung, not results.

![R4a: how many of the 792 probe datasets each ticket wins](../img/goldenticket/r4a_argmin.svg)

## R4b — the ticket buys most where the decoder is least sure

Split the panel by draw dispersion (how much the 10 ticket decodes
disagree per frame): the winner's gain is monotone across quartiles,
−0.35 on the tightest frames to −1.44 on the most dispersed. The
ticket is not shaving uniform noise — it wins where the decoder's
noise-response is largest, which is exactly where any per-dataset or
per-frame escalation has the most room.

![R4b: winner gain by dispersion quartile](../img/goldenticket/r4b_quartiles.svg)

## R4c — horizon-wide, not a first-step artifact

Per-step MAE across the 50-step chunk, all three configs: the
ordering stable-key → winner → ensemble holds at every step, and the
gap *grows* with horizon.

![R4c: per-step MAE across the chunk horizon](../img/goldenticket/horizon.svg)

## The side-read that failed (and why that's useful)

SDN's smoothness selector ("pick the least jerky draw",
[2606.14084](../papers/noise-space-steering-3.md)) was placed on the
banked ticket-64 stack at table cost: **null** (agreement 1.5% vs
1.6% chance). Heun-30 ODE draws are uniformly smooth — the criterion
has nothing to grip on this family. The family decode
(mean-of-draws) stands.

## What happens next

The thread continues as the
[per-dataset tickets pre-reg (DRAFT)](2026-08-08-prereg-noise-ladder-perdataset.md):
a CPU-first reliability floor on the banked 2,458×64 probe stack
(split-half self-consistency decides which of the 792 dataset cells
are even decidable), then one confirm eval routing each qualifying
dataset to its own ticket, plus the paired re-run that could seat
the R3 ensemble as a leaderboard row. Named unknown inherited by
every ticket config: the panel cannot see chunk-boundary artifacts —
a rollout-gated read (#16) stands between any ticket and a rig.
