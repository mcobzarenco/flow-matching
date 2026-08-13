# Design memo + pre-reg DRAFT: token-GRPO phase 2 on the AR head (t=1.0)

*2026-08-13 06:0xZ. The queue's phase-2 design item, executed per the
[signal probe](2026-08-12-prereg-grpo-signal-probe.md)'s frozen
decision rule (both families cleared 08-13 00:0xZ–01:1xZ: **phase 2 =
token-GRPO on the AR head first, at t=1.0**; Flow-GRPO SDE second).
Everything here is CPU-side design: **nothing is registered, nothing
launches** — the launch pends the owner phase-2 go (open ask since
08-12) plus a finalized pre-reg. The [08-12 design
memo](2026-08-12-grpo-sim-design-memo.md) §5 sketched this run before
the probe existed; this memo replaces that sketch with measured
numbers, and corrects its cost estimate ~5× upward.*

## Plain words

The probe answered the go/no-go question: when our AR policy replays
the same starting position 8 times with mild sampling noise (t=1.0),
the attempts spread out enough to rank (median spread 0.771 cm, 3×
the bar) without getting meaningfully worse than the deterministic
policy. So group-relative RL has something to push on. This memo
turns that into a concrete first training run: how attempts are
scored, which weights move, how big a batch of attempts each update
uses, what it costs on the measured simulator pace, and the exact
tripwires that stop it if it starts learning the wrong thing (the
probe warned that most of the spread comes from *violent* attempts —
so the first thing RL will likely learn is "don't knock the boat
off," and the run is instrumented to see exactly that). It also
prices the honest bad news: at the measured rollout pace the run
costs ~3× the old sketch, so it is laddered — a 2-step smoke, a
15-step read, and a conditional extension — with a decision boundary
before each escalation.

## 1. What the probe fixed (inputs to this design)

From [amendment 1](2026-08-12-prereg-grpo-signal-probe.md) (all cells
complete, 15/15 groups, 3.57 GPU-h):

- **t=1.0, not SimpleVLA-RL's 1.6.** Cell 1 (t=1.0): median group
  std **0.771 cm** (3.1× bar), competence cost −0.351 cm CI
  [−1.117, +0.207] (includes 0), knock-aways 10/120. Cell 2 (t=1.6):
  3.2× the spread but cost CI [−1.556, −0.634] entirely negative and
  4× the knock-aways — at our floor the extra spread is violence,
  not information.
- **Non-degeneracy 13/15** groups at t=1.0 — the dynamic-sampling
  filter would keep most groups, but ~13% are dead weight per step.
- **The spread is knock-away-tailed.** The within-group variance is
  dominated by violent draws, not gentle trajectory diversity. First
  learnable signal ≈ "don't swat the boat." The reward and reads
  below are designed so that is measurable, not hidden.
- **Measured rollout pace** (the number that reprices everything):
  **1.13 GPU-h per 120 episodes** at workers=8 (the parallel driver
  runs one seed's 8 draws as a worker-wave; a 30 s episode-wave takes
  ~4.5 min wall) → **~0.0094 GPU-h/episode**. The 08-12 sketch
  assumed ~12–15 min per 128 episodes; reality is ~5× slower. All
  budgets below use the measured number.

## 2. Algorithm (SimpleVLA-RL mapping, frozen candidate)

Per RL step:

1. Draw **S=8 fresh spawn seeds** from a dedicated stream (disjoint
   from sim100's 0–99, the probe's 0–14, and the held-out eval set),
   **G=8** sampled rollouts each at `--ar-temperature 1.0` → 64
   episodes, v3 frames, sim100 episode conventions
   (`--episode-seconds 30` = 30 replans × execute-horizon 30).
2. Score each episode with the composite reward (§3); z-score within
   each group (ddof=0 — the probe's primary statistic); **drop
   zero-variance groups** (std < 0.05 cm, the probe's non-degeneracy
   line — expect ~1/8 dropped).
3. One gradient pass (μ=1, strictly on-policy — no replay epochs in
   the first ladder): advantage-weighted **token-level clipped CE**
   over the action block only (value lines excluded), broadcast
   advantage to all action tokens, **clip-higher [0.8, 1.28]**
   (DAPO), ratio = π_new/π_old from per-token logprobs under the
   grammar-masked softmax at t=1.0. **KL penalty off**, but KL to
   the frozen er60k anchor is *measured* every step (one reference
   forward on the training batch) — it is a tripwire input and the
   hacking early-warning, per Flow-GRPO's lesson.
4. lr **5e-6** flat (SimpleVLA-RL's value; our SFT decoder-lr 1e-4
   is a pre-training rate, not an RL rate), AdamW as in train.py,
   no warmup, constant schedule for the ladder.

**Old-policy logprobs come almost free**: the decode path already
captures pre-mask block logits + the applied grammar mask + chosen
ids per step (`ActionCaptureStep`, built for mcselect) — the rollout
driver just needs to keep `log_softmax(masked logits / T)[chosen]`
per token. No new math.

## 3. Reward (frozen candidate constants)

Per episode, in centimeters so every term is commensurate with the
probe's spread numbers:

```
r = progress_final_cm                      # dense base (initial − final boat→disk)
  + 10.0 · [success_tick is not None]      # the actual goal
  −  2.0 · [final_upright < 0.9]           # tipped-boat penalty
  −  5.0 and episode flagged, if reset_strikes > 0   # hard fault
```

Design notes: knock-aways need no extra penalty — they are already
negative progress (the dense base *is* the anti-swat gradient);
success at +10 deliberately dominates a group when it happens (median
group std is 0.771 — a success should win its group outright, this is
the SimpleVLA binary signal grafted on top of the dense floor);
the strike penalty should never fire (probe measured 0/360) — if it
fires at all it is also a tripwire input. Rewards are z-scored within
group, so only within-group *differences* matter; constants are about
ordering, not scale.

Frames are **production-default visuals at finalization time** — v3
today; v4 (contact shadows / clutter patches) only if amendment 5
lands first. The rendering default is pinned in the finalized pre-reg
and never changes mid-run.

## 4. Trainable surface (the one real fork — owner input wanted)

- **Option A — patch-only (~11M)**: the FAST-block embedding rows +
  tied head columns. Cheapest, safest, but it cannot change the
  trunk's computation — only re-map action-token embeddings/logits.
  Plausibly too weak to move behavior; no published precedent this
  narrow.
- **Option B — patch + text stack at 5e-6, vision frozen
  (RECOMMENDED)**: the er60k SFT shape (decoder + text-lr) at the RL
  rate. SimpleVLA-RL full-fine-tunes its 7B at 5e-6 — the only
  published token-GRPO-on-VLA recipe votes B. Memory is measured:
  the rig-mixture option-B preflight ran the full er60k recipe
  single-H100 at 69.2 GiB peak with act-ckpt — the RL gradient pass
  is the same batch shape.
- πRL's frozen-trunk precedent does not map cleanly here: their
  "expert" is a 300M module; our ar_backbone's "expert" IS the trunk
  (the ~11M patch is just vocabulary furniture).

Registered fallback: if B trips the instability wires (§7), A is the
named retreat arm, not a new design.

## 5. Ladder + budget (measured-pace arithmetic)

Per-step cost: 64 rollout episodes ≈ 0.60 GPU-h (8 waves × ~4.5 min)
+ gradient pass ≈ 0.13 GPU-h (64 eps × 30 replans = 1,920 suffix
sequences ≈ 40 eff-48 steps at the measured ~12 s/step single-H100
pace) + amortized eval ≈ **~0.75 GPU-h/step**.

| rung | steps | reads at boundary | cost | cum. |
|---|---|---|---|---|
| R0 smoke | 2 | plumbing oracles green end-to-end; pace read (incl. a workers-12/16 throughput probe — if 16 holds, every later number halves); KL/entropy telemetry sane | ≤ 2 GPU-h | 2 |
| R1 | 15 | reward slope + guard rates + held-out greedy eval (§6) | ~12 GPU-h | ~14 |
| R2 (conditional) | +25 | full frozen reads | ~+19 GPU-h | **~33; gate 35** |

R1→R2 boundary rule (frozen candidate): extend iff (a) no tripwire
fired, and (b) the held-out greedy composite reward is not *worse*
than step-0 (paired CI not entirely below 0). R2 is the read, R1 is
the safety check — 15 steps × 64 eps ≈ 1k episodes of experience is
not expected to move a 2B policy far; stopping at R1 for a *positive*
result would be premature, stopping for a *negative* safety read
would not. Beyond R2 is a new pre-reg, not an extension.

Rollouts and gradient steps alternate on one GPU (πRL-style
synchronous loop) — no second GPU, no async staleness. Venue: local
H100, inside the standing GPU-day sequencing, only on owner go.

## 6. Frozen reads (draft bars — finalized numbers at pre-reg)

Held-out eval set: **seeds 200–219, greedy, 20 episodes**, run at
step 0 (before any update) and every 5 steps (~0.19 GPU-h each,
~0.04/step amortized).

1. **Primary: paired Δ composite reward (held-out greedy, endpoint
   vs step-0)**, seeded 10k bootstrap CI95 paired by seed.
   IMPROVED if CI entirely above 0; the honest expectation for R2's
   ~2.6k episodes is "measurably not-worse with guard-rate movement"
   — the bar for *calling phase 2 promising* is CI-above-0 on either
   the primary or read 2.
2. **Knock-away rate under sampling** (the probe's working
   hypothesis made testable): training-rollout knock-away fraction,
   step-series; bar = endpoint 5-step window below the probe's
   t=1.0 baseline 10/120 with a binomial CI excluding it.
3. **Success count** (record + headline if > 0): held-out greedy
   successes (probe baseline: 0 greedy; 1/120 sampled).
4. **Record-only**: per-step median group std (does the spread the
   probe measured survive training?), non-degenerate-group fraction,
   KL-to-anchor curve, action-token entropy, tip rate, per-seed
   progress traces.

## 7. Tripwires (stop + re-scope in-channel; never silently continue)

- NaN/inf loss, or any reset strike in training rollouts (probe
  baseline 0/360).
- **Entropy/spread collapse**: median group std < 0.05 cm for 3
  consecutive steps (advantage signal gone — the Flow-GRPO
  diversity-collapse signature).
- **Violence explosion**: knock-away rate > 2× the 10/120 baseline
  for 3 consecutive steps (reward hacking toward swatting).
- **Competence crash**: held-out greedy eval worse than −1.0 cm vs
  step-0, paired CI entirely below (the probe's own floor line).
- KL-to-anchor runaway: step-series slope goes vertical relative to
  the R0/R1 trend (recorded, judged at eval boundaries — exact
  numeric line set at finalization from R0's measured scale).

## 8. Instrument delta (bounded, CPU-buildable before any go)

1. **Rollout logprob + frame capture**: `--emit-training-rows` on
   the parallel driver — per replan: the two observation frames
   (jpeg), sampled backbone ids, per-token chosen logprobs (from the
   existing `ActionCaptureStep` surface), state vector, RNG key.
   ~0.3 GB/step, pruned after the gradient pass. Oracle: at greedy,
   emitted logprobs match a teacher-forced re-forward bit-for-bit
   (same masked softmax); draw-0 rows reproduce banked sequential
   rows.
2. **GRPO step** (`bijou/train_grpo.py` or a train.py mode):
   advantage-weighted clipped token-CE. Oracles: ratio≡1 (fresh
   policy) reduces to advantage-weighted CE exactly; zero advantage
   → zero grad on every parameter; grammar mask at train time equals
   the rollout mask per token.
3. **Replay collator**: training rows → `CollatedBatch` (prompt
   re-encode + suffix teacher-forcing of the *sampled* ids). Oracle:
   re-encoded prompt logits reproduce the rollout decode's captured
   block logits within bf16 tolerance on a fixture episode.
4. **Loop harness**: rollout wave → score → z-score/filter → grad
   step → periodic held-out eval → babysit-readable heartbeat + rows
   stream; registry entry at launch.

Estimated: 2–3 CPU sessions, oracle-gated, no behavior change to any
existing path (new flags default-off).

## 9. What this asks of the owner

1. **The phase-2 go itself** (open since 08-12) — now with an exact
   shape and an honest price: **~33 GPU-h, gate 35**, laddered with
   two decision boundaries (vs the 08-12 sketch's 10–15; the probe's
   measured pace repriced it).
2. **The §4 fork**: option B (patch + text stack at 5e-6, vision
   frozen) recommended; A is the registered fallback. Veto welcome.
3. Whether the **instrument build (§8) may start before the go** —
   it is CPU-only, oracle-gated, and makes the go instantly
   actionable; but it is ~2–3 sessions of build the owner may prefer
   spent on the sim-visuals lane.
4. Nothing launches on this memo. On go: this draft finalizes
   (constants frozen, HEAD + checkpoint pinned, objection window)
   as its own immutable pre-reg post, per protocol.

Sources: [signal probe pre-reg + results](2026-08-12-prereg-grpo-signal-probe.md) ·
[08-12 design memo](2026-08-12-grpo-sim-design-memo.md) ·
[GRPO for our two heads (deep-read)](../papers/grpo-for-vla-heads.md) ·
[sim100 results](2026-08-12-sim100-results.md) ·
[er-60k continuation pre-reg](2026-08-08-prereg-molmo2-ar-60k-continuation.md)
