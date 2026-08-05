# Pre-registration: paired 40k arms — control vs aux-supervision-off

*2026-08-05. Immutable once posted; the launcher header points here.
Supersedes the [own-baseline pre-registration](2026-08-05-prereg-own-baseline.md)
per the owner's 16:13Z steering ("make something else ready" + GPU
never idles): the standalone 100k baseline is replaced by a paired
40k design where **arm A doubles as the own-topology control**. The
originally slated treatment (unwrap-at-load) was killed by the
[wrap census](2026-08-05-wrap-census.md) — wraps touch 0.19% of
training episodes, far inside pairing noise — so arm B goes to the
next-best zero-new-code treatment: **aux attribution** (idea #6, a
still-owed mainline question).*

## Question

Does aux text supervision (the CE loss over judge-annotation fields,
weight 0.5) change *action* performance at matched steps, seed, and
data — or is narration a free rider on the action objective? Mainline
pre-registered "within probe noise (±0.3)" but never ran the paired
arms; the 100k run answered only "does conditioning help at
inference". Secondary payoff: arm A is the charter §4 own-baseline
(topology control) at 40k for every future training delta on this box.

## Arms (sequential on the 1×H100, chained in one launcher)

- **Arm A (control):** `fontaine_arb_rcond_40k_1xh100` — the
  mainline-best recipe verbatim from the superseded pre-reg (B10,
  workers 16, prefetch 4, LRs 1e-4/2e-5 unrescaled, seed 0), `--steps
  40000`, `--warmup-steps 1000`, save-every 5000, eval-every 500.
- **Arm B (treatment):** `fontaine_arb_rcond_auxoff_40k_1xh100` —
  identical except the aux loss is OFF: `--aux-fields`,
  `--aux-dropout`, `--field-dropout` omitted ("train actions only, the
  historical objective" path). Conditioning flags
  (`--condition-fields subgoal outcome smoothness`,
  `--condition-dropout 0.1`, `--subgoal-dropout 0.5`,
  `--instruction-augment 0.5`, `--camera-kind-dropout 0.1`) stay —
  the arms isolate the aux CE term alone.
- After both arms: frozen-panel evals of A@40k and B@40k with
  `--dump-predictions`, so the paired per-frame analysis runs on CPU
  without re-scoring.

40k (not 100k) per the owner-amended cap: early ablations pair at
≤ 40k; extend later only if a decision needs it.

## Numbered expectations

1. **Startup (both arms):** selection 878 datasets / 42,872 episodes;
   ~11M decoder params + text trunk at 2e-5. Arm B's model line shows
   no aux fields and its log lines carry no `loss_aux`.
2. **Throughput:** 0.4–0.6 s/step at B10 (smoke measured 0.39–0.45);
   VRAM < 76 GiB (smoke peak 67.4); ~5–6.5 h per arm, both arms +
   evals inside ~14 h.
3. **Curves** (256-frame in-run probe, ±0.3 floor): arm A below 12 by
   10k and below 9 by 30k (inherited from the superseded pre-reg).
   Arm B's *action* probe within ±0.3 of arm A at matched steps — the
   mainline pre-registered expectation, now actually tested. Arm B
   total loss is lower by construction (one term fewer); only the
   action component is comparable.
4. **Primary read:** panel `chunk_mae` A@40k vs B@40k, paired
   per-frame (same 17,204 core frames). Decision rule: |Δ| within
   pairing noise (bootstrap CI from the paired per-frame deltas) →
   aux supervision is action-neutral at this scale — banked, idea #6
   closed at 40k/eff-10. B better than A beyond noise → the aux term
   taxes actions (weight 0.5 too high — flag to mainline, follow-up
   arm at lower weight). A better than B beyond noise → aux
   supervision helps actions (representation shaping is real —
   strengthens the aux program).
5. **Kill gates (per arm):** probe > 15 at 10k with a
   falling-then-rising shape; NaN loss; second OOM after the standing
   B−1 resume. Slowness alone is data. Arm A being killed does not
   launch arm B (the pair is the experiment; B alone answers nothing).

## Known seams and confounds

- Same seed (0) in both arms: data order, augmentations, and dropout
  masks are as matched as the harness allows; residual nondeterminism
  (cuDNN autotune, atomics) is the pairing-noise floor the bootstrap
  CI measures.
- Arm B changes the *sequence content* (no aux value lines rendered
  before BOA), so its per-step token count differs — throughput and
  loss scale shift is expected and not a finding.
- eff-10 vs mainline eff-40: any cross-topology comparison stays
  directional; the paired A-vs-B contrast is the claim-grade result.
- 40k is 40% of the recipe's step budget; a null here bounds the
  effect at this horizon only.

## Cost

~11–13 h GPU for both arms + ~2×35 min panel evals (25,800 frames at
the measured ~320 f/min), ~384 GB checkpoints (8 saves/arm × 24 GB;
2.2 T free), zero API spend. Launches tonight the moment the
sealed-panel baseline score frees the GPU, gated on the smoke's E1–E4
(passed 16:32Z except final-step formality — see log).
