# Pre-registration: golden-ticket noise screen (#1, teacher-first)

*2026-08-07 ~18:0xZ. Immutable once posted. Idea
[#1](../ideas/01-noise-draw-ensembling.md), from the Golden Ticket
deep read
([2603.15757](https://arxiv.org/abs/2603.15757) at v3 — the
[sampling-beyond-selection page](../papers/sampling-beyond-selection.md)).
Zero training; eval-side only. The instrument (a "ticket" noise-key
mode) does NOT exist yet: it lands oracle-gated before launch, and if
implementation forces any semantic deviation from this post, an
amendment posts before launch (the
[#19 amendment](2026-08-06-prereg-ar-sampled-draws.md) precedent).*

## Question

For our frozen flow teacher, does a *single searched initial-noise
vector* — substituted at every frame — beat fresh Gaussian noise on
the panel? Golden Ticket showed tickets beat Gaussian sampling on
46/51 task–policy pairs, searched by episodic rollout return. Their
search needs environment rollouts; **our panel is the offline
criterion they lack** — the whole search reduces to one batched
draws-style eval where the "draws" are candidate tickets.

The honest prior is *against* a large effect: our panel spans many
tasks, and the paper's own structure says **tickets are task-local**
— per-task tickets always gain (+13 LIBERO-Spatial), the best single
*shared* ticket per suite regressed in all three suites (−2.6 to
−12.0). A panel-wide ticket is the shared regime. That is precisely
why this is a cheap screen and not an execution arm: stage 1 asks
only whether global ticket structure exists at all, and the
per-dataset geometry (which comes free from the dump) reads the
task-locality question on our own data.

**Teacher-first** (the deep read's design note): the SnapFlow student
compiled away most of its draw spread (draw-averaging gain −0.236 vs
the teacher's −1.258), so its searchable ticket space may have
collapsed. This screen targets the teacher's noise space only; any
student rung is an escalation amendment gated on stage-2 passing.

## Frozen design

**Checkpoint / decode:** `bijou_flow_artrunk_h1024_40k_ddp2` @
`step_080000`, Heun-30 — the leaderboard's flow-teacher config.
Banked anchors: single draw (stable-key) **6.5997 / 1.9355**,
mean-of-10 **5.3645 / 1.4242**.

**Tickets:** M = **64** candidates, i.i.d. N(0, I), shape
[50, 6] (chunk × action_dim), generated once at instrument-land time
from a domain-separated seed sequence (`[TICKET_DOMAIN, 0, m]` for
m = 0..63), saved to a single tickets npz whose sha256 is quoted in
every read. Random search only — CEM or any adaptive refinement is
out of scope for the screen (escalation material).

**Stage 1 — search (probe subset):** one eval on the
`plans/holdout_curated_v0_k4l2_drawsprobe_s7.json` probe (2,458
frames, f_eff 2,341.7) with `--sample-draws 64` under the new ticket
noise key: draw m at *every* frame uses ticket m — the defining
ticket property. `--dump-draws` retains per-ticket per-frame
predictions; the per-ticket score is pooled core-frame chunk MAE
through the same pooling as the fairness reads. Winner = argmin
(tie-break: lower first_mae); top-10 = the 10 lowest.

**Stage 2 — confirmatory (full panel, complement rows):** the winner
ticket, full 25,800-frame panel, `--sample-draws 1` with ticket
noise. Primary read pools **complement core rows only** (panel core
frames minus the probe plan's frame-identity triples; ≈14,147 f_eff)
— the probe rows selected the winner, so they are excluded from the
read that judges it. Paired per-frame Δ vs the banked stable-key
single-draw npz
(`eval__..._panel_curated_v0_k4l2_stablekey_heun30.npz`) re-pooled
on the identical rows.

**Stage 3 — the "both" cell (only if stage 2 passes):** mean of the
top-10 tickets, full panel, `--sample-draws 10` ticket noise — does
searched noise beat random noise *inside the ensembling regime*?
Compared to the banked mean-of-10 row (5.3645) at pooled level (its
per-frame npz was not retained; both cells' pooled draw-noise scales
are ≤ σ_draw/√10 ≈ 0.008, so a pooled comparison with a ±0.02 tie
band is honest).

## Null scales (all banked, none new)

- Per-draw pooled spread at probe size: **σ_probe = 0.0669** (the 10
  stable-key draws of `analysis__sigma_draw_direct.json`; range
  6.5766–6.7977). Under the null (no global ticket effect; noise
  effects frame-idiosyncratic and exchangeable) 64 ticket scores
  spread with exactly this σ.
- Expected minimum of 64 null tickets: mean − 2.345·σ_probe =
  **mean − 0.157**, sd of that minimum ≈ 0.45·σ_probe ≈ 0.030
  (Monte-Carlo, 2·10⁵ trials, seeded).
- Panel-scale σ_draw = **0.02367**; at complement size σ ≈ 0.0256,
  so the standing **0.05 adopt floor ≈ 2σ** on the stage-2 read.

## Frozen reads and decision lines

- **R1 (stage 1, headroom):** sample sd of the 64 ticket scores, and
  the minimum. Ticket structure is declared *worth confirming* iff
  **sd > 0.0785** (the upper 95% χ²₆₃ edge of σ_probe = 0.0669) OR
  **min < mean − 0.22** (expected null min −0.157, minus 2 sd of the
  min). Otherwise **KILL before stage 2**: results post records the
  distribution and the screen closes at ~1.5 GPU-h spent.
- **R2 (stage 2, confirmatory):** paired per-frame Δ (winner −
  stable-key) on complement core rows, bootstrap CI95. Ticket is
  **REAL** iff Δ ≤ **−0.05** and CI95 excludes 0. The full-panel
  pooled number is quoted alongside for board continuity; a
  leaderboard row (draws/keying stated: `ticket`, sha-pinned) only
  if REAL.
- **R3 (stage 3):** pooled Δ (mean-of-top-10-tickets − banked
  mean-of-10 5.3645). Interesting iff ≤ **−0.02** (beyond the tie
  band); either way record-only in this screen — mean-of-10's row
  is not displaced without a paired follow-up.
- **R4 (record-only, free from the dumps):** (a) per-dataset
  per-ticket score matrix — do datasets disagree on the argmin
  ticket (the task-locality read; the paper predicts they do)?
  (b) dispersion-quartile geometry of the winner's per-frame gain
  (the `selection_ceiling_results.py` quartile machinery);
  (c) per-step horizon profile of the gain.

No other numbers are read. R1's kill line is the point of the
screen's staging: under the null the probe winner's edge is
frame-idiosyncratic luck and would evaporate on complement rows —
we do not pay for stage 2 to learn what R1 already said.

## Instrument (to land, oracle-gated, before launch)

`bijou.eval` gains a **ticket noise mode** (semantics frozen here;
flag spelling is implementation's): a tickets-npz path replaces
`noise_for_item`'s per-frame keying with `noise = tickets[draw]`,
independent of the frame — reusing the batched draws-major tiling,
`--dump-draws`, and the scoring path unchanged. Policy/report
provenance carries the mode and the tickets-file sha256 (a ticket
read must never pass as a stable-key read).

**Oracles (abort-on-red before launch):**

1. Contract: ticket mode at draws=1 reproduces bit-exact the direct
   `sample_actions(noise=ticket)` call on the same frame.
2. Ticket property: within one run, two different frames receive
   byte-identical noise for the same draw index (asserted
   in-process, not by construction).
3. Determinism: same tickets file → two runs byte-identical dumps.
4. Pooling reuse: the search-stage scorer, run over the banked
   stable-key full-panel npz, reproduces **6.5997** exactly; run
   over the banked drawsprobe npz per-draw, reproduces the 10
   banked per-draw pooled MAEs of `analysis__sigma_draw_direct.json`
   exactly.

## Cost gate and execution window

Stage 1 ≈ 1.5 GPU-h (draws-64 batched on 2,458 frames; microbench
marginal ≈ 33 ms/frame/draw), stage 2 ≈ 0.9, stage 3 ≈ 2.9. **Gate:
6.0 GPU-h total**, local 1×H100 only, launched via
`run_detached.sh`, GPU-free guard. Window: a quiet local-GPU window
strictly after the tsens rungs complete and behind the already-queued
selfsubgoal probe (#6) — this screen does not preempt anything
pre-registered before it.

## Caveats carried from the paper and from us

- A fixed ticket makes the policy **deterministic** — and can fail
  hard at unsearched positions; the panel cannot see either rollout
  property (the #16 offline-vs-rollout gap applies in full).
- The shared-ticket prior: the paper's own shared-per-suite cells
  regressed. A null result here is informative, not a failure — it
  would close #1's ticket rung and leave mean-of-10 as the flow
  family's decode.
- Ticket effects, if REAL, bind to the **teacher** (30 NFE,
  unconstrained class until distilled); nothing here licenses a
  student claim — that is the escalation amendment's job, gated on
  the student's own draw-response pre-check.
