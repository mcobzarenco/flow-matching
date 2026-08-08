# Pre-reg note: accuracy-by-field (narrated pass) evals

*2026-08-08 ~11:2xZ. Record-only diagnostic (the microbench
precedent). Queue item `fieldgen-accuracy-eval` (owner steering
2026-08-08 10:08Z: "We should also run the eval script with field
generation enabled, reports have a table with accuracy by field which
would also be good to see"). Posted BEFORE the one GPU run it
registers.*

## A correction first — the AR-100k half is already banked

My 10:49Z in-channel answer said the banked panel runs didn't enable
field generation. **That was wrong for AR-100k.** The narrated pass
(`bijou@100000+fields` — the model decodes every trained aux field,
then its actions) rides *automatically* on aux-trained checkpoints,
and the banked AR-100k greedy panel has carried the accuracy-by-field
block all along
([report](https://mcobzarenco-fontaine-blog.static.hf.space/reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2.html)):

| field | metric | value | frames |
|---|---|---|---|
| holding | accuracy | **0.807** | 8,987 |
| progress | MAE | **0.062** | 8,987 |
| event | presence accuracy | **0.878** | 8,987 |
| visible | slot-set accuracy | **0.319** | 8,260 |

Caveats that travel with the table: labels are the weak judge's
(~80% inter-judge holding agreement, ±15% progress MAE — ceilings sit
near the label noise); event scores presence ("none" vs any), visible
scores exact set-equality of parsed slots (the 0.319 is the strictest
metric here, not comparable across columns). Free companion read:
narration *costs* on AR-100k — `+fields` chunk MAE 5.8565 vs base
5.8026 (+0.054, the known does-narration-help sign at 100k).

**Consequence: no AR-100k GPU run.** The queued ~1–2 GPU-h local run
is cancelled as redundant — the table above is the deliverable.

## Why molmo2 has no table — a found-and-fixed bug

The molmo2 40k panel shows all-None accuracies next to 8,596 labeled
frames. Root cause (fixed in `2f4d575`, regression-tested):
`BijouPolicy` gated the narrated pass on the *Gemma concrete*
(`isinstance(decoder, ARBackboneDecoder)`); `Molmo2ARDecoder` is a
sibling concrete of the shared `ARSuffixDecoder` scaffold, so
aux-trained molmo2 checkpoints silently reported no trained fields
and the pass never rode. Prompt bytes are unchanged by the fix on
every banked read (molmo2 checkpoints record `generate_bracket=True`;
the request renders identically) — banked numbers all stand; the only
behavior change is that the narrated arm now *exists* on molmo2.

## The one run this note registers

**`fontaine_molmo2_ar_60k_ddp4` @60k fields panel** (box 4×DDP), via
`fontaine/scripts/box/eval_box_molmo2_60k_fields_panel.sh` +
`run_detached.sh`, strictly AFTER:

1. the 60k endpoint saves and its **chained panel eval lands**
   (tonight's chain runs the box checkout as launched — charter:
   never sync box code under a live run — so the chained eval stays
   narrated-arm-free and byte-comparable to the 40k panel, which is
   what the paired 60k-vs-40k read wants anyway), and
2. the box control checkout is refreshed via `refresh_ctrl.sh` to a
   commit carrying `2f4d575` (the launcher greps the fixed gate in
   `bijou/eval/policies.py` and refuses to run pre-fix code).

Exact command = the chained eval's, byte-identical flags/plan/seed,
output stems `..._panel_curated_v0_k4l2_fields`. No new CLI flags:
post-fix the narrated pass rides automatically.

**Frozen reads (all record-only):**

1. **The accuracy-by-field table** for molmo2@60k — same four metrics
   as above, same weak-label caveats. This is the owner deliverable.
2. **Does-narration-help on molmo2**: `+fields` vs `bijou@60000`
   paired chunk MAE (AR-100k anchor: +0.054 cost). Record-only — no
   decision hangs on it.
3. **Validity oracle, checked mechanically by the launcher**: the
   fields run's base `bijou@60000` chunk MAE must equal the chained
   eval's to full JSON precision (same instrument, same 4-rank
   sharding, greedy decode — the index-sorted merge is
   world-size-invariant). Disagreement = instrument finding, abort
   the read.

**Cost**: base + narrated ≈ 2× the 40k chained panel (~1.7 GPU-h) ⇒
**~3.5 GPU-h, gate 6 GPU-h** — inside the box window after the 60k
chain, before the attach screen opens (the attach decision consumes
the 60k-vs-40k read, not this diagnostic).

**Not registered here**: any 40k-side narrated run (superseded — the
60k trunk is the continuation the board tracks), any headline claim
from field accuracies (weak labels), any re-run of the AR-100k table.
