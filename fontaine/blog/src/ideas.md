# Ideas

The backlog. Every entry: hypothesis, expected effect, cost, cheapest
falsification. Seeded 2026-08-05 from charter §8 (which distills the
mainline ledger, `docs/architecture.md` §7–8); ordering ≈ expected
information × cheapness. Status tags: `queued` / `screening` /
`running` / `confirmed` / `falsified` / `parked`.

## 1. Inference-time noise-draw ensembling — `queued`, natural first

- **Hypothesis:** mean-of-N noise draws through the flow expert cuts
  panel MAE substantially in the unconstrained class (mainline
  measured 5.30°→2.88° on motion frames for a ft'd model,
  mean-of-10); the stage-2 flow-on-AR-trunk lineage (6.623 panel)
  should benefit similarly.
- **Expected effect:** large on the flow lineage's panel number;
  unconstrained-class only (charter §2) until distilled.
- **Cost:** ~20 lines eval-side + one eval burst per N in {1,5,10}.
  Zero training.
- **Falsification:** paired panel eval, same checkpoint, draws
  stated. Check unimodality of draws first (averaging multi-modal
  draws is wrong): per-frame draw spread on a few hundred panel
  frames. If mean-of-10 does not beat single-draw beyond the panel's
  pairing noise, kill.
- **Open sub-question:** an AR-family analogue (temperature/nucleus
  chunk ensembles, chunk-level medians) — separate screen.

## 2. Throughput: bucketed batching + torch.compile on the frozen prefix — `queued`, natural first

- **Hypothesis:** length-bucketed batching + `torch.compile` of the
  prefix encode (79% of step time) buys ≥20% step-time on 1×H100 —
  compounding interest on every later run.
- **Cost:** a day of implementation + measured A/B on a short run.
  No GPU-idle risk (screens run at run boundaries).
- **Falsification:** measured s/step and samples/s on identical
  configs, before/after, on THIS box. If <10% combined, bank the
  numbers and deprioritize.

## 3. Longer training on the best recipe — `queued`

- **Hypothesis:** rcond-100k was still improving at 100k
  (75k→100k bought 0.05–0.3); an extension banks a cheap win.
- **Cost:** a multi-day 1×H100 run (own-baseline rule: needs the
  eff-10/11 reference arm first, charter §4). Resume traps: fresh
  `--seed`, `--steps` = new TOTAL, cosine re-heat semantics.
- **Falsification:** panel MAE at matched eval cadence vs the
  own-baseline arm's curve; kill if the extension's curve is flat
  over its first 10–15k steps.

## 4. Stage-2 follow-ups (flow expert on AR trunk) — `queued`

Inherited questions from mainline §8.11 (banked: 6.57 in-run / 6.62
panel @80k, 2.2× smaller expert): more/deeper export streams (AR
adaptation lives in all 35 layers; the expert reads {4,9,14} —
untested headroom), expert width h512/h1536 on the better features, a
second-generation AR trunk re-measured through the stage-2 lens.
Cost: one screen-rung run per arm. Falsification: paired screens at
matched steps.

## 5. FAST tokenizer v3 — `queued`

- **Hypothesis:** refitting on curated-v0's exact quantiles removes
  the ~1.94%-of-chunks clip rate; small but real MAE effect on
  clipped chunks.
- **Cost:** CPU-only fit (~32 min measured for v2); token metrics
  RESET (never cross tokenizer versions) — coordinate with run seams.
- **Falsification:** paired arms (same seed/data/arch, only the
  artifact differs — the v1-vs-v2 precedent); recon error + clip rate
  in the fit report before any training touches it.

## 6. Aux attribution arms — `queued`

The still-owed paired aux-on vs aux-off fine-tunes from a common base
(does aux supervision shape the representation, separate from "does
narrating help" — the 100k run answered only the latter).
Pre-registered mainline expectation: within probe noise (±0.3).
Cost: two short matched fine-tunes.

## 7. Stream-schedule re-test — `queued`

0-0-16 vs 4-4-8 vs shallow-heavy (8-4-4) at scale: the acuity probe
(shallow stream carries sharpest position) and streams0016's rig hint
pull opposite directions — measure. Config-diff cheap per arm; enters
at the short-run screen rung.

## 8. Shortlist/output-vocab head for ar_backbone — `queued`

The 262k-vocab CE softmax is the VRAM headroom eater; a shortlist
head raises feasible batch on 1×H100 (mainline queued it as the
structural fix after the B12 OOM). Cost: real code + an equivalence
check (loss oracle moves → loud re-baseline). Payoff multiplies every
future ar_backbone run on this box.

## 9. Data levers — `queued`

`--trim-leading-idle` (~6.7% of frames), state-noise augmentation,
judge-score-weighted sampling (never yet run). Each is a cheap paired
arm at the screen rung. Any derived corpus ships with the leakage
check (charter §2) before training touches it.

## 10. E2B base-vs-IT swap — `queued`

Pre-registered mainline prediction ±0.2 MAE; backbone-swap arm, tests
whether instruct tuning matters at our instruction distribution.
Verify the -pt checkpoint ships the vision tower first.

## 11. Visual grounding arms — `queued`, the open front

Re-anchor probe: error is frame-dependent level mis-estimation;
acuity probe: the text stack's use of visual tokens is the
bottleneck. Arms: trunk shaping, schedules, vision-side aux tasks —
chartered on the community panel; `first_mae` is the
grounding-sensitive column (2.143 vs copy 2.620 — headroom).
High-variance; counts toward the ≥20% exploration budget.

## 12. Solver/Heun-gap work — `queued`, re-opened by a surprise

The h1536 adaRMS Heun-gap collapse did NOT transfer to h1024-on-AR-trunk
(measured −0.28 at 10→30, first_mae −0.46): sampler quality is back on
the table for the best flow lineage. Arms: step-count sweeps, solver
variants, consistency/distillation toward 1–2-step deployment decodes
(the distillation leg pairs with idea 1).

## 13. Literature-sourced arms — standing

The arXiv radar (VLA/robot learning, flow matching, action
tokenization, data curation) feeds this list; every borrowed idea
cites its source in the pre-registration; every "novel" idea gets a
search first. Local canon: π0, π0.5, SmolVLA, FAST
(arXiv:2501.09747).
