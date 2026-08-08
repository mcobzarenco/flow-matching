# Pre-registration: per-dataset golden tickets (#1, noise-ladder rung 2)

*2026-08-08 ~13:2xZ — **finalized; immutable from this commit.**
Drafted ~09:2xZ the same day (structure unchanged); finalized after
the CPU stages ran on banked data: stage-0/1 results are published
below (they were the draft's finalization checklist item 1), the
instrument oracle list is pinned after an audit of `bijou.eval` at
HEAD (item 2), and one wording clarification is flagged inline where
it occurs. Entry condition met 2026-08-08: the
[golden-ticket screen](2026-08-08-goldenticket-results.md) closed
R1 CONFIRM → R2 REAL → R3 INTERESTING. Idea
[#1](../ideas/idea-01.md). Priors from
[2603.11642](../papers/noise-space-steering-3.md) and our own R4a/R4b
reads, quoted below with their numbers. Zero training. Remaining GPU
cost at launch: **≤ 4 GPU-h** (stage 2 ≈ 0.9 + seating arm ≈ 3.0; the
draft's ≤ 6 ceiling included CPU-stage contingency that is now
spent).*

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

## Stage 0–1 results (executed 2026-08-08 ~13:1xZ — CPU, banked data)

Instrument: `fontaine/scripts/noise_ladder_stage01.py` (oracles a–d
GREEN first: pooling + complement reproduction against the banked
stage-1/2 jsons; planted signal/null/at-line split-half worlds;
provenance refusal; routing tie-breaks). Full table:
`reports/analysis__noise_ladder_stage01.json`.

**Stage 0: floor F = 6 — the rung stays open, thinly.** 147 of the
153 ≥4-frame datasets split (6 have a single-parity frame set and are
listed in the json). The bin table is honest and noisy: n=4 (30
datasets) fails (median regret 1.855 vs null 5th-pctl 1.766), n=5
(20) fails, **n=6 (39 datasets) passes — 1.5675 vs 1.5965** (~2%
under the line), **n=7 (10) passes clearly — 0.846 vs 1.084**, and
the sparse bins n=8–22 (1–11 datasets each) all fail; three
single-dataset bins at n=23/37/86 pass on their own cells. The floor
rule binds at the smallest passing n: **F = 6**. Caveats recorded at
the moment of judgment: the pass at the floor is marginal, the
pattern over n is not monotone (small-bin power is the likely reason,
but that is an interpretation, not a measurement), and the frozen
qualification rule (≥ F frames) admits datasets from failing bins.
Stage 2's held-out reads exist precisely to adjudicate whether this
thin floor carries transferable signal — the falsifier stands
unchanged.

**Stage 1: the routing map is committed.** Qualifying set (≥ 6 probe
frames AND ≥ 20 complement rows): **97 datasets**, covering **7,028
panel core rows (40.8%)** and **6,014 complement rows** (expectation
1's ≥ 25% met). **88 of 97 route away from ticket 33**; the routed
tickets span all ten of the top-10 set. The full dataset → ticket map
(all 792 datasets; non-qualifying → 33) is in the analysis json;
**map sha256 `15d9293553ac1a88…`** — the stage-2 run must carry
exactly this sha in its provenance.

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
   **held-out complement core rows only** (clarified at finalization:
   the draft said "core rows"; probe rows selected the tickets and
   never judge — the R2 pattern, and the reason the ≥ 20
   complement-row qualification floor exists) — the marginal value of
   per-dataset routing over the shared winner. Pass = CI95 entirely
   below 0.
2. **Δ_route vs stable-key** on the same rows (record-only context;
   rung-1 already banked the shared-vs-stable number).
3. **Per-dataset win table**: fraction of qualifying datasets where
   the routed ticket beats ticket 33 on held-out rows, vs the 50%
   null (sign test).
4. Horizon + dispersion-quartile mirrors of read 1 (R4b form).
5. Execution oracles — **pinned at finalization** after the audit of
   `bijou.eval` at HEAD (the substitution point is
   `BijouPolicy._flow_noise`, which already has per-item identity in
   hand; ticket provenance already rides both the npz dump and the
   report json). All abort, never silent:
   - **Provenance:** report + npz carry the m64 bank sha
     (`9bb13bc4…`) AND a new `ticket_map_sha256` equal to the
     committed map sha `15d9293553ac1a88…`; the policy name gains
     `_ticketmap` (distinct from plain `_ticket` — a routed read must
     never pool as a single-ticket read); `sample_draws == 1`.
   - **Routing byte-match (preflight, before the panel run):** on a
     small plan of rows from ≥ 2 datasets mapped to a non-33 ticket
     t, the routed decode must be byte-identical to a plain
     `--noise-tickets` ticket-t decode of the same plan at matched
     composition (the rung-(b) preflight pattern).
   - **Non-qualifying rows** (mapped to 33) of the full panel run
     byte-match the banked ticket33 npz at matched composition (same
     plan file, same batch size — same row order).
   - **Identity columns** byte-match the banked panel npzs;
     state-copy rows byte-match.
   - **Map coverage:** every panel dataset appears in the map; the
     map's image ⊆ top-10 ∪ {33}.

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

## Finalization record (DRAFT ~09:2xZ → posted ~13:2xZ, same day)

1. ✅ Stage 0 + stage 1 ran on banked data (oracles first); F, the
   qualifying set + row weights, and the routing map sha are
   published above; the full map is committed in
   `reports/analysis__noise_ladder_stage01.json`.
2. ✅ Instrument oracle list pinned (stage-2 item 5) after auditing
   `bijou.eval` at HEAD. The routing mode itself is instrument work
   that happens at execution time, gated on its preflight oracle.
3. ✅ This commit: re-dated, DRAFT banner dropped, immutable from
   here; execution gets its own queue entry and babysit entries at
   launch. One wording clarification at finalization is flagged
   inline (read 1: complement rows); expectations 1–5 were banked in
   the draft **before** stage 0 ran and are unchanged — expectation 1
   is already CONFIRMED (F = 6 ≤ 16; 40.8% ≥ 25%).
