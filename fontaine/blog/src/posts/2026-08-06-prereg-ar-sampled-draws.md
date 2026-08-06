# Pre-registration: AR sampled-draws eval (mean-of-samples)

*2026-08-06 ~23:1xZ. Immutable once posted. Ideas [#19](../ideas.md)
— the owner's fairness ask (2026-08-06 19:15Z): when we quote flow
mean-of-N draws, the AR family gets temperature-sampled draws-N +
mean-of-samples too — both sides get the same instrument or neither.
Instrument landed and gated this session (commit `78c9f56`);
execution queued for the next quiet GPU window.*

## Question

Does the flow family's mean-of-N ensembling advantage survive when
the AR family is given the same read? The banked asymmetry: flow
gains ~1.26 chunk MAE from 10-draw averaging (teacher draws1 6.6232
→ draws10 5.365 on the k4l2 panel), while every AR row to date is a
single greedy decode — the AR deployment anchor 5.8026 has never had
an ensembled counterpart. If greedy decode already sits near the
posterior mean (the mean-collapse shape the SnapFlow arc measured on
the 1-NFE student), AR sampling + averaging should gain little; if
AR draw diversity survives like the flow teacher's, the 5.8026
anchor understates the family.

## Instrument (landed this session, `check.py` 351 green)

`bijou.eval --ar-temperature T --sample-draws N`: the action block
of the backbone-suffix AR decode is temperature-sampled N times per
frame and the decoded chunks are averaged in raw units
(`collapse_draws`, the flow path's own mean). Mechanics, each
oracle-pinned (`tests/test_ar_sampling.py`, 9 tests):

- **Sampling = Gumbel-max over the grammar-masked softmax**
  (`ARSampling`, `ar_backbone.py`): argmax(logits/T + G) with
  Gumbel G drawn fp32 CPU-side — exactly softmax(logits/T)
  restricted to legal ids; the grammar mask is unchanged, illegal
  ids sit at −inf and can never win. Aux value lines stay GREEDY —
  only the action block is a distribution read.
- **Keying**: one RNG stream per (frame identity triple, draw) —
  `stable_sample_rng`, blake2b → SeedSequence like `stable_noise`
  but domain-separated (an AR draw never replays a flow draw's
  bitstream). Invariant to corpus composition, batch composition,
  shard order and device. No legacy index path exists (new
  instrument); `--noise-key` governs flow noise only.
- **Draws share one prefill**: the prefix cache is snapshotted by
  reference and restored between draws
  (`ARSuffixDecoder.cache_snapshot`, sound under the append-only
  cache contract; restored-cache decode ≡ fresh-encode decode,
  bit-exact oracle). This makes the fairness caveat literal: AR
  draws are cheaper per draw than flow draws (shared prefill, ~30–60
  sequential suffix steps each) while flow draws re-integrate the
  full solver per draw.
- **The T→0 limit recovers the greedy decode exactly** (oracle) —
  greedy rows and sampled rows share one decision point; `sampling=None`
  IS the historical path, so all banked AR numbers stand.
- **Naming/provenance**: the policy row carries `_drawsN_tT` (an
  ensembled, sampled read must never pass as a deployment read —
  charter §2); report JSON records `ar_temperature`; the narrated
  pass is skipped under sampling (different inference class). Guards:
  `--ar-temperature` on flow / without `--checkpoint` / ≤ 0 dies
  loudly; `--sample-draws > 1` without a stochastic decode still dies.

## Design

**Temperature: T = 1.0, pinned, untuned — primary.** The open design
point in #19 (fit T on a probe set) is resolved by the fairness rule
that motivates the instrument: the flow side's draws are i.i.d.
samples from the model's own untuned noise distribution, so the AR
mirror samples the model's own untuned softmax. Tuning T on any data
would hand AR a fitted knob flow was never given. One pre-registered
sensitivity rung — T ∈ {0.5, 0.7, 1.3} at draws 10 on the frozen q4
subset (4,301 rows, the state-probe artifact) — is RECORD-ONLY:
quoted as a dT diagnostic, never a headline, and never a license to
re-pick T post hoc.

**Arms (k4l2 panel, v1 keying conventions, seed 0):**

| arm | checkpoint | rows |
|---|---|---|
| A-s0 (gemma4 AR aux-on 40k) | `fontaine_arb_rcond_40k_1xh100/step_040000` | greedy (banked, 5.8026) + `_draws10_t1` |
| molmo2 AR 40k | `fontaine_molmo2_ar_40k_ddp4/step_040000` (endpoint, lands ~2026-08-08) | greedy + `_draws10_t1`, same command stems |

Anchors (no new GPU): flow teacher draws1 6.6232 / draws10 5.365,
teacher draws10-heun30 5.3645/1.4242, SnapFlow student mean-of-10
5.3675/1.5927 — all banked k4l2 reads from the draws-fairness arc.

**Cost gate (pre-registered fallback, no improvisation at launch):**
AR decode is sequential, so draws10 ≈ up to 10× a greedy panel eval
(~2 h at ~200 f/min) per arm. Measure the rate over the first ~200
frames; if a full-panel draws10 run projects > 24 GPU-h, BOTH arms
drop to the frozen q4 subset (4,301 rows) and every comparison row
(greedy, flow anchors) is re-pooled onto those rows from banked npzs
— the state-probe subset precedent; the switch is recorded, not
silent. Venue: local GPU for A-s0 (idle-by-design; this pre-reg is
its required paper), box at the molmo2 endpoint boundary for the
molmo2 arm. Never co-located with a pre-registered training run's
eval chain.

## Frozen reads

Per arm, on identical rows (paired per-row, seeded bootstrap 95% CI,
seed 0, 10,000 resamples — the draws-fairness assembly conventions):

1. **Primary: Δ_AR = chunk_mae(_draws10_t1) − chunk_mae(greedy).**
   The AR ensembling gain, quoted with CI.
2. **Fairness comparison: Δ_AR vs the flow teacher's −1.258** (its
   draws1 → draws10 gain on the same panel) — does the AR family
   ensemble like flow, or is greedy already the mean?
3. **Family read: does A-s0 `_draws10_t1` reach the flow draws10
   band (5.365)?** Both families then hold mean-of-10 reads under
   their own stochasticity — the first symmetric-instrument
   flow-vs-AR comparison.
4. first_mae mirrors of 1–3; the T-sensitivity rung (record-only).
5. Execution oracles (abort on failure): state-copy / state-copy-norm
   rows byte-match the banked panel values (row pairing + baselines
   untouched by sampling); report JSON carries `ar_temperature: 1.0`;
   policy name carries `_draws10_t1`.

## Numbered expectations (banked before data)

1. Δ_AR < 0 (averaging 10 sampled decodes beats one of them — and
   beats greedy at least slightly) — confidence medium.
2. |Δ_AR| < 1.258: the AR gain is SMALLER than the flow teacher's —
   greedy decode already sits near the predictive mean, so sampling
   mostly adds noise the average removes (the mean-collapse shape) —
   confidence medium-high. This is the informative read either way.
3. A-s0 `_draws10_t1` does NOT overtake flow draws10 5.365 —
   confidence medium.
4. **Falsified if** Δ_AR > +0.1 (sampling + averaging actively hurts
   the AR family at T=1.0): the mean-of-samples premise fails for
   this family; the result is recorded and the instrument retires to
   diagnostic use — no temperature fishing beyond the pre-registered
   sensitivity rung.

## Cost & scheduling

A-s0 arm: ≤ 24 GPU-h full-panel (else the q4 fallback, ~4 GPU-h);
sensitivity rung 3 × q4 draws10 ≈ 12 GPU-h worst case, run ONLY if
the primary lands inside the gate. molmo2 arm: same stems at its
endpoint boundary (~2026-08-08), decided by the same gate. Queue
position: after the molmo2 40k babysit obligations; local launch in
a work session with first-poll util+rate checks per standing rule.
