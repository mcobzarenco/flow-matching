# Pre-registration DRAFT: per-dataset golden tickets (#1, noise-ladder rung 2)

*2026-08-08 ~09:2xZ. **DRAFT — not yet the immutable pre-reg.** The
design below is frozen in structure; the finalization checklist at the
bottom converts DRAFT → posted before any instrument or launch work
(the [vu5k precedent](2026-08-07-prereg-molmo2-vision-unfreeze.md)).
Entry condition met 2026-08-08: the
[golden-ticket screen](2026-08-08-goldenticket-results.md) closed
R1 CONFIRM → R2 REAL → R3 INTERESTING. Idea
[#1](../ideas/idea-01.md). Priors from
[2603.11642](../papers/noise-space-steering-3.md) and our own R4a/R4b
reads, quoted below with their numbers. Zero training.*

## Question

The screen proved **one shared** searched noise vector is real:
ticket 33 beat the stable-key default by **−0.924** [CI95 −0.985,
−0.866] on held-out complement rows. But both the published
decomposition and our own free reads say the shared ticket is the
*small* half of the channel:

- [2603.11642](https://arxiv.org/abs/2603.11642) (192 contexts × 16
  noises): boundary-gap variance splits **59.1% context main effect /
  1.4% noise main effect / 39.4% context×noise interaction**; the
  globally best noise is optimal in only **3.1%** of contexts.
- Our R4a shadow (792 probe datasets × 64 tickets, banked stage-1
  matrix): ticket 33 is the per-dataset argmin in only **4.4%** of
  datasets; every ticket wins somewhere; the per-dataset argmin lands
  in the global top-10 **29.8%** of the time (~2× the 15.6% null).
- R4b: the winner's gain is monotone in draw dispersion (−0.35 →
  −1.44 by quartile) — the channel is largest exactly where
  per-dataset structure has the most room.

Rung 2 asks: **does routing each dataset to its own ticket beat the
shared winner on held-out rows — at one extra panel eval of cost?**
The honest obstacle, banked in advance: the **median probe dataset
has 2 frames**, so most per-dataset argmins are selection noise, not
signal. This pre-reg's job is to spend CPU on banked data first to
find out *which* cells are decidable, and only then spend GPU
confirming the survivors.

## Data already in hand (no GPU for stages 0–1)

The stage-1 ticket eval retained per-frame, per-ticket predictions:
`…drawsprobe_s7_ticket_draws64_heun30_draws.npz` — draws
`(2458, 64, 50, 6)` + truth/valid/repo_id/core, sha-pinned tickets
(`e537f4cd…` top-10 subset of the M=64 bank `a07c062a…`). Every
stage-0/1 quantity below is a pure function of this file plus the
banked complement npzs (`…panel_curated_v0_k4l2_ticket33_heun30.npz`,
`…stablekey_heun30.npz`).

## Stage 0 — reliability floor from split-half self-consistency (CPU)

For every dataset with ≥ 4 probe frames: split its frames into two
halves by frame-index parity (deterministic, no seed knob); pick the
argmin ticket on half A; measure its regret on half B against half
B's own argmin. Pool the regret curve **by cell size n** and compare
each n-bin against a permutation null (ticket labels shuffled within
dataset, 1,000 permutations, seed 0). The **floor F** is the smallest
n whose median split-half regret beats the null's 5th percentile —
i.e. the smallest cell size at which the argmin carries any
out-of-half information at all.

**Frozen decision rules:**

- If NO n-bin beats its null (selection noise dominates at every
  available cell size), **the rung closes at CPU cost** — a real
  result: per-dataset search is undecidable on this probe's cell
  sizes; the escalation (bigger probe) needs its own pre-reg.
- Otherwise: **qualifying set** = datasets with ≥ F probe frames AND
  ≥ 20 held-out complement core rows (the confirm needs rows to
  judge on). The qualifying set, its panel-row weight, and F are all
  published in the stage-0 table before stage 2 launches.

## Stage 1 — per-dataset ticket assignment (CPU, frozen selector)

For each qualifying dataset: ticket = argmin of pooled probe MAE over
its frames, **restricted to the global top-10 ticket set** (the R4a
containment read says the argmin lands there 2× null; restricting to
10 pre-vetted tickets cuts the 64-way selection-noise surface by
6.4× and reuses tickets that already passed R1). Ties break toward
the global winner 33. Non-qualifying datasets route to ticket 33
(the shared winner is the fallback, so the deployed map degrades to
rung 1, never below it). The full map (dataset → ticket) is
committed before stage 2.

## Stage 2 — one confirm eval (GPU), paired frozen reads

One full-panel eval, deterministic single decode, with the
per-dataset ticket map (instrument: the `--noise-tickets` machinery
gains a per-dataset routing mode; oracle: rows of a dataset mapped to
ticket t must decode byte-identical to a plain ticket-t run of those
rows at matched composition — the [rung-(b) preflight
pattern](2026-08-08-prereg-subgoal-draws.md)). Cost ≈ the ticket33
run (~0.9 GPU-h). Reads, all paired per-frame with seeded bootstrap
CI95 (seed 0, 10,000 resamples), **clustered by dataset** (frames
within a dataset share the routing decision — an unclustered CI
would overstate precision):

1. **Primary: Δ_route = map vs ticket 33**, qualifying datasets'
   core rows only — the marginal value of per-dataset routing over
   the shared winner. Pass = CI95 entirely below 0.
2. **Δ_route vs stable-key** on the same rows (record-only context;
   rung-1 already banked the shared-vs-stable number).
3. **Per-dataset win table**: fraction of qualifying datasets where
   the routed ticket beats ticket 33 on held-out rows, vs the 50%
   null (sign test).
4. Horizon + dispersion-quartile mirrors of read 1 (R4b form).
5. Execution oracles (abort): state-copy rows byte-match; identity
   columns; routing-map sha in the report; non-qualifying rows
   byte-match the banked ticket33 run at matched composition.

**Falsifier:** read-1 CI95 not entirely below 0 ⇒ per-dataset
routing at this probe's cell sizes buys nothing over one shared
ticket — the interaction slice is not harvestable at panel scale
with this selector, and the rung records that against the 39.4%
prior. Read 3 then adjudicates *why* (broad small losses = selection
noise; a few large losses = floor too low).

## Folded-in arm: seating the R3 number (GPU, record-only → row)

R3 measured mean-of-top-10-tickets at **5.1847 / 1.3831** — the best
chunk and first numbers on this panel — but the banked random-noise
mean-of-10 row (5.3645 / 1.4242) retained no per-frame npz, so R3
was record-only. This arm re-runs the **random-noise draws-10**
config with `--dump-predictions` retained (~3.0 GPU-h, the stage-3
cost), enabling the paired per-frame read the board row seating
requires: mean-of-top-10 vs mean-of-random-10, paired CI95. CI
entirely below 0 ⇒ the top-10-ticket ensemble takes the mean-of-10
board row; otherwise the row stands and R3 stays a record. This arm
is independent of stages 0–2 and runs in the same GPU window.

## Numbered expectations (banked before data)

1. Stage 0 finds a floor F ≤ 16 with a non-empty qualifying set
   covering ≥ 25% of panel core rows — confidence medium.
2. Δ_route (read 1) lands below 0 but small (the qualifying set is
   the easy-cell minority; 2603.11642's 93.8%-of-gap number is a
   16-noise, per-context ceiling we do not expect at 10-ticket,
   per-dataset resolution) — confidence medium-low.
3. Per-dataset win rate (read 3) beats 50% — confidence medium.
4. The R3 seating arm confirms (CI below 0) and the board row moves
   to the top-10 ensemble — confidence medium-high (Δ was 9× the
   band, but unpaired).
5. Gains concentrate in the upper dispersion quartiles (R4b form) —
   confidence medium.

## Cost & scheduling

Stages 0–1 are CPU on banked data (any GPU-busy window). Stage 2 ≈
0.9 GPU-h; seating arm ≈ 3.0 GPU-h; **ceiling ≤ 6 GPU-h total**,
local GPU, quiet window after the #6 rung-(b) chain resolves; every
launch via `run_detached.sh`; babysit entries at launch. Boundary
artifact (chunk hand-offs) remains a named unknown of every ticket
config — panel-blind, rollout-gated, inherited from the screen.

## Finalization checklist (DRAFT → posted)

1. Run stage 0 + stage 1 (CPU, banked data); publish F, the
   qualifying set + row weights, and the routing map in this post.
2. Pin the instrument oracle list (routing mode + byte-match checks)
   once the instrument design is audited against `bijou.eval` at
   HEAD.
3. Re-date the post, drop the DRAFT banner, commit as immutable;
   execution gets its own queue entry and babysit entries.
