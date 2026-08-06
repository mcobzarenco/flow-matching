# Pre-registration: architecture batch #1 — bigger images & full-residual conditioning (DDP3, panel-v2)

*2026-08-06 ~12:2xZ. Status: pre-registered, POSTED BEFORE LAUNCH.
Owner steering 2026-08-06 11:44Z: "I would like to have a run on more
than 1 GPU aimed at trying some new more fundamental architecture
changes, like a new trunk or if not, paying attention to full
residuals rather than few exported layers, or bigger images (i.e.
more tokens / image)." This batch is examples 2 and 3, paired, on the
best lineage; the trunk swap (Molmo2-4B, survey rank 2) is its own
follow-on pre-reg either way. Posted for an owner look before launch
per the 11:46Z exchange; proceeding unless steered.*

## Question

The grounding evidence says vision is the binding limit of the
current architecture: best flow `first_mae` 1.9453 vs state-copy
2.5851 on panel-v2 (ahead, but the whole model is worth ~0.64 of
first-step error over a policy that ignores images); the acuity probe
located position information as *sharpest at the vision-tower output*
(8.4 px linear readout) and degraded through the LM layers; the
state-reliance probe measured the proprio-shortcut mechanism directly
(D = +0.702, 14× threshold). Two architecture levers attack this
without touching the trunk weights:

- **(A) Bigger images — more visual tokens per image.** The Gemma 4
  processor natively supports soft-token budgets {70, 140, 280, 560,
  1120}/image (measured on the real processor today: patch count
  scales exactly linearly, 1260 → 10080 patches for 140 → 1120). We
  train at 140. Raising the budget gives the LM's visual workspace
  4× the slots (560) over the same content — each soft token covers a
  finer cell. Honest physics: sources are 640×480, so the processor
  upscales (~2× linear at 560) — no new *pixels*, but a finer patch
  grid and more token capacity where the acuity probe says
  information currently dies.
- **(B) Full-residual conditioning.** The flow expert today reads the
  trunk through exactly three exported K/V streams (global-attention
  prefix layers {4, 9, 14}, schedule 4-4-7). The AR path that the
  trunk was shaped by consumes *all* layers. Arm B gives the expert
  the full residual stream: one new conditioning stream per prefix
  layer (hidden states after layers 0..14), each with a learned
  RMSNorm + K/V projection into the expert's existing cross-attention
  geometry, expert layer *i* reading trunk layer *i* (1:1 ascending).
  ~23.6M new trained params (15 × 1536 × 512 × 2; trunk stays
  frozen). Mainline's stream question (#4) carried a pre-registered
  null-result caveat — this is that headroom, tested.

Family: **stage-2** (flow expert h1024, adaRMS, bidirectional, on the
FROZEN `bijou_arb_rcond_100k_ddp4/step_100000` trunk) — the best
lineage (teacher `bijou_flow_artrunk` @80k: 6.7151/1.9453 on
panel-v2), cheap to train (no trunk grads), and conditioning-side is
exactly where both levers live.

## Arms (one variable each; sequential DDP3 on box GPUs 1–3)

| arm | run name | delta vs arm 0 |
|---|---|---|
| 0 | `fontaine_flow_arch0_base_40k_ddp3` | — (own-baseline, teacher recipe verbatim) |
| A | `fontaine_flow_archA_img560_40k_ddp3` | `--max-soft-tokens 560` (fallback rung 280, gate F2) |
| B | `fontaine_flow_archB_fullresid_40k_ddp3` | residual-stream conditioning res0..res14 replaces kv4/9/14 |

Common recipe (teacher-verbatim except where stated): 3 corpora
(community_curated_v0 + both so101 repos), fps 30, cameras {1,2},
holdout 0.1/seed 0, `--decoder flow`, `--backbone-init-from
outputs/train/bijou_arb_rcond_100k_ddp4/step_100000`, h1024/8h/4096/8xh,
`--stream-counts 4 4 7` (arms 0+A), `--self-attention-mode
bidirectional`, `--time-conditioning adarms`, camera-kind-dropout 0.1,
instruction-augment 0.5, condition fields subgoal/outcome/smoothness
(dropout 0.1/0.5), chunk 50, **`--batch-size 32` per rank × 3 ranks =
eff-96 (matches teacher's ddp2×48)**, 40k steps, decoder-lr 1e-4,
warmup 500, wd 1e-5, grad-clip 10.0, `--seed 0` (all arms matched;
teacher used seed 1 — irrelevant to paired reads, stated for the
record), eval-every 500 (256-frame probe, eval-seed 0), save-every
2500, wandb project `fontaine` (named verify item — the d9dd385
teacher-verbatim trap). Launch: `CUDA_VISIBLE_DEVICES=1,2,3 uv run
torchrun --standalone --nproc-per-node=3 -m bijou.train …` under tmux
via a `~`-scp'd launcher, console tee'd, MALLOC_* set. Order: 0 → A →
B (0 and A run on today's box code `bcbf101`; B's new code syncs at
arm C's 40k boundary — never under a live run).

**Deltas from teacher, stated once:** 40k steps vs 80k, DDP3-eff-96 vs
DDP2-eff-96 (same samples/step, different topology), seed 0 vs 1, our
box. Arm-0-vs-teacher is therefore directional context only; **every
frozen read below is paired arm-vs-arm-0 on identical frames** and
inherits none of these deltas.

## Endpoint evals (per arm, chained in-launcher on the rank-0 GPU)

Panel-v2 (`plans/holdout_curated_v0_k4l2_panel_v2.json` — first
pre-reg under the v2 adoption, owner 11:44Z), Heun-30, **draws=1,
`--noise-key stable`, seed 0** (deployment class), batch 32,
`--dump-predictions` npz + JSON + HTML report. State-copy control rows
ride along as always. Instrument: `fontaine/scripts/arch_batch_results.py`,
built and oracled BEFORE any arm's data exists (standing box-batch
pattern, 5th application): degenerate self-comparison → exactly 0/CI
[0,0]; synthetic ±known-delta recovery; misaligned-index hard abort;
anchor reproduction of the v2 teacher/state-copy rows from banked
npzs.

## Frozen reads & decision rules

Per arm X ∈ {A, B}, paired per-frame vs arm 0 on the v2 core rows,
CI95 by frame bootstrap:

1. **Primary (headline column):** Δchunk = chunk_mae(X) −
   chunk_mae(arm 0). **Adopt-lever iff Δchunk ≤ −0.15 AND CI95
   excludes 0.** (Floor borrowed from the E4B convention:
   max(3σ_seed, 0.15) with σ_seed = 0.038 measured on the AR family
   at 40k — the borrow across families is a stated approximation;
   the flow-side draw noise is controlled by stable keying + matched
   seed, σ_draw 0.0237 ≪ the floor.)
2. **Grounding read (the column these arms aim at):** Δfirst ≤ −0.10
   AND CI95 excludes 0 ⇒ the lever moved grounding specifically
   (context: v2 state-copy first = 2.5851 is the ignore-images
   floor).
3. **Falsified for its mechanism** iff CI95 contains 0 or Δchunk >
   +0.15 (actively worse ⇒ record and kill the lever at this scale).
4. **Verdict assembly:** any adopt-lever ⇒ follow-on pre-reg
   (combine winning levers and/or 80k extension of the winner;
   winner also becomes the preferred SnapFlow teacher config, #12).
   Both arms null/falsified ⇒ conditioning-side levers are dead at
   this scale ⇒ **the trunk swap (Molmo2-4B) is promoted to the next
   multi-GPU pre-reg** — that outcome is decision-relevant, not a
   failure. Arm B adopt ⇒ offer upstream (mainline #4 stream
   question).

Expectations (banked before data): arm 0 endpoint chunk ∈ [6.7, 7.9],
first ∈ [1.90, 2.35] (teacher @80k is 6.7151/1.9453; 40k is half
its training — outside the band = surprise-log entry, paired reads
unaffected). Arm A modal outcome per the acuity-probe prior (the LM's
*use* of tokens, not token count, looked binding): |Δchunk| < 0.15 —
this is an explore arm; the tail worth buying is Δfirst −0.1 to −0.3.
Arm B is genuinely open (mainline never tested it); a null here
closes #4's caveat with data.

## Gates

- **F1 (memory smoke, before arm 0 launches):** 200-step smoke of ALL
  THREE configs on GPUs 1–3 at B32/rank. Any OOM ⇒ the whole batch
  drops to the largest B ∈ {24, 16} that fits all three with ≥5 GiB
  headroom — one eff-batch for the whole batch, never per-arm
  (batch semantics never change mid-comparison). Expected fit: B64
  flow was ~40 GiB on 1×H100; arm A's 4× prefix and arm B's 15
  streams are the unknowns the smoke exists for.
- **F2 (rate, arm A):** if the smoke rate projects arm A's 40k > 30 h
  wall, drop 560 → 280 and re-smoke (pre-registered rung, still 2×
  tokens); if 280 also projects > 30 h, arm A reduces to a 10k screen
  at 280 and the result is labeled screen-rung. Estimates (measured
  at smoke, these are priors): arms 0/B ~0.7–0.9 s/step ⇒ ~8–10 h
  each; arm A at 560 ~2–3× arm 0 (prefix encode was 79% of step time
  at 140 tokens) ⇒ ~20–28 h.
- **K1 (in-run kill, arms A/B):** 256-frame probe > (arm 0's probe at
  the matched step) + 3.0 at any eval ≥ step 5k ⇒ kill at the next
  save boundary (catastrophic-only; the probe's ±0.3 floor makes
  tighter in-run kills noise-trading).
- **K2 (liveness):** standard babysit — first-poll util+rate rule,
  30-min polls, kills wait for save boundaries, OOM ⇒ B−1 ladder
  is FORBIDDEN here (would break matching) — an OOM after F1 passes
  is a bug to diagnose, not a knob to turn.
- **Pre-launch (arm B only):** implementation + oracles landed +
  `check.py` green + box code synced at an arm-C-free boundary.
  Oracles owed: (i) res-stream K/V shapes/positions match the
  existing stream geometry contract (loader asserts); (ii) trunk
  params bitwise-frozen through a train step (zero grad, zero drift);
  (iii) grads flow to all 15 projections; (iv) config round-trips
  (checkpoint records the res schedule; eval loads it with no flags);
  (v) arms 0/A code path bitwise-unaffected by the new code (the
  three CPU loss oracles + stage-0 re-verify). Arm B does NOT launch
  until all five pass.

## Seams & caveats (stated now, shipped with any claim)

- Sources are 480p: arm A's gain, if any, is token-capacity +
  finer patch grid, not new pixels. A camera upgrade is the rig-side
  dual, out of scope here.
- At 560 tokens/camera the prompt (~1,130 tokens) exceeds the trunk's
  512-token sliding window on 4-of-5 layers — in-family for Gemma 4
  (windows are its training regime), but the global layers carry the
  long-range load; noted as an interpretation seam for arm A, not a
  defect.
- Arm B's res streams include sliding-window layers' hidden states —
  the information is real (residual stream, not K/V), no window
  truncation applies to hidden-state reads.
- 40k ≠ 80k: an arm that nulls at 40k could win at 80k; the branch
  rule buys the 80k extension only for a winner (velocity rule).
- Panel-v2 + stable keying is the new-bank convention (adopted
  11:44Z); the teacher anchor 6.7151/1.9453 quoted here is
  index-keyed v2 (derived from the banked npz) — context only, no
  frozen read consumes it.
- Both arms bill to the ≥20% exploration budget (explore class).

## Cost

~35–50 GPU-h on GPUs 1–3 total (F2-dependent), ~1.5–2 days wall
sequential, +~2–3 h panel eval per arm on one GPU. Disk ~60–90 GB
across the three runs at save-every 2500 (prune to endpoint+latest
after reads, uploads before deletions). GPU 0 (arm C) untouched;
SnapFlow local untouched.
