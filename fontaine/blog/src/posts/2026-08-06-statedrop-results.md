# 2026-08-06 — State-dropout 0.8 results: mechanism WORKED, actions PAID — adopt nothing (#9)

*Frozen reads of the [pre-registration](2026-08-06-prereg-state-dropout-40k.md)
via `fontaine/scripts/statedrop_results.py` (pairing + state-copy
byte-match oracles green; banked
`reports/analysis__statedrop_40k_k4l2.json`). Run:
`fontaine_arb_rcond_statedrop80_40k_1xh100` — the mainline AR recipe +
`--state-dropout 0.8` (mean-masking at collation via the shared
`mask_state_item` primitive), 40k steps, completed 2026-08-06 ~16:02Z;
panel + masked-reliance evals completed 19:01Z.*

## Verdict: **COSTS — adopt nothing.** The sanctioned follow-up is a p=0.3 screen.

- **Read 1 (primary, paired per-frame Δchunk vs A-s0 core frames):**
  **+2.64** MAE [CI95 2.55–2.74], median +1.27, C wins only **23.9%**
  of 17,204 frames. Far beyond the pre-registered ±0.15 band; no
  single-repo exclusion moves the mean below +2.58. Pooled: C
  10.5024/8.5606 vs A-s0 7.7966/3.9422.
- **Read 2 (reliance, the mechanism check):** state-masked first_mae
  **8.11** vs the intact read 8.56 — masking state barely moves the
  model (it even helps slightly), while the baseline model collapses
  to **24.08** under the same mask. Capability class: **partial**;
  the anti-shortcut mechanism did exactly what the causal-confusion
  literature promises — state reliance is gone.
- **Read 3 (grounding transfer):** no first_mae gain to attribute —
  C's 8.56 sits +4.62 above A-s0's 3.94. Breaking the state shortcut
  did NOT rebuild the lost accuracy out of vision at this dropout
  rate; it just removed the crutch.
- **E4 prior band:** chunk 10.50 vs the expected [7.65, 8.3] —
  outside, on the costly side. The pre-registered final probe gate
  (`< 10 @40k`) had already failed at 10.90.

## Reading

p=0.8 is too aggressive at this rung: the model was denied state on
80% of samples and could not (at this data/step budget) replace the
information visually — the cost lands on both chunk and first-action
error. The mechanism/capability split is the transferable result:
**dropout kills the shortcut without teaching the replacement.** The
pre-registered branch keeps exactly one follow-up alive — a p=0.3
screen (mostly-intact state, shortcut still perturbed) — queued, not
launched; the box belongs to the Molmo2 AR run tonight.

Eval reports: hosted under [Reports](../reports.md) once synced
(`eval__fontaine_arb_rcond_statedrop80_40k_1xh100__step_040000__*`).
