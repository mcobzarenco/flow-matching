# Molmo2 AR 40k endpoint: BEATS — the trunk bet pays

*2026-08-08 05:0xZ. Reads frozen in the
[pre-registration](2026-08-06-prereg-molmo2-ar-40k.md) §5, executed by
`fontaine/scripts/molmo2_endpoint_results.py` (oracle-green before the
data landed; output
`reports/analysis__molmo2_endpoint_k4l2.json`). Run:
`fontaine_molmo2_ar_40k_ddp4`, 40,000 steps on 4×H100 (~29 h + saves),
Molmo2-4B trunk, aux + subgoal co-training, endpoint probe
6.2075@40000 (run low 5.91@26500). Panel:
`panel_curated_v0_k4l2`, 25,800 frames / 17,204 core, chained greedy
eval at the endpoint.*

## Read 1 (primary): BEATS, by 3.4× the bar

| policy | pooled chunk MAE | first MAE |
|---|---|---|
| **molmo2 AR 40k, greedy** | **6.0079** | **2.1871** |
| E2B AR anchor (A-s0 own-topology 40k control) | 7.7966 | 3.9422 |

Classification bar: BEATS < 7.30 (anchor − 0.5, ~the family's seed
spread). Measured: **6.0079** — not a margin call. The paired
per-frame read (identity-aligned rows, seeded frame bootstrap):
**Δ = −1.717 [CI95 −1.797, −1.635]** over all 17,204 core frames.
First-step MAE nearly halves (2.19 vs 3.94).

**Frozen decision executes: Molmo2 becomes the phase-2 flow-trunk
candidate** — the AR-adapted prefix is what the attachment screen
(#4, launching next on this box) will hold frozen, killing the −2.7
topology confound the pre-reg named.

Context the board gives for free: at 40k steps the Molmo2 trunk sits
**0.21 behind AR-100k's greedy 5.8026 at 2.5× fewer steps**, on a
trunk that reads our rig scenes zero-shot. The #19 draws10_t1 arm
(running now, same chain) prices its sampled-ensemble headroom.

## Read 2 (instrument integrity): green, with one recorded slip

The state-copy and state-copy-norm columns **byte-match** the banked
same-plan panel columns — same rows, same fallbacks, no instrument
drift. Pooled state-copy quotes 11.7847/2.6202 under the panel
convention. Recorded, not silently corrected: the pre-reg's
parenthetical quoted "11.7639/2.5851", which reproduces under *no*
pooling of this plan (core, all-rows and norm variants all checked) —
a drafting slip in the pre-reg text; the operative byte-match oracle
is unaffected.

## Read 3 (context, narrative only)

The e4b screen milestone family (7.54@10k probe) and arm C statedrop
(10.50) both sit well above this endpoint. Nothing here changes those
banked readings.

## What was consumed, what it cost, what broke

~29 h train on 4×H100 (2.17–2.55 s/step; ~4 h of wall clock was the
save windows) + ~0.5 h chained greedy panel eval. vram_alloc_peak
67.13 ≤ 71 GiB throughout; K1 kill line crossed green at 10k
(7.1652 vs 12.0944) and never looked back.

One incident at the boundary: the chained greedy eval **died 4
minutes in** on its first launch — `float != BFloat16` at the first
suffix-attention matmul. The suffix decoder's `torch.where` silently
promoted mixed-dtype embeds (bf16 mounted trunk, fp32 trainable FAST
patch); training probes run under autocast, so this standalone-eval
path had never executed against a real bf16 checkpoint, and the tests
loaded the tiny fixture in fp32. One-line cast fix (the idiom
`_logits` already used), a red-then-green regression test
(`test_bf16_mounted_trunk_decodes_with_fp32_patch`), and the #19
launcher's pre-built greedy-if-missing clause recovered the chain
(`5a43b15`). The dead-chain survival path being designed in advance
is why this cost ~10 minutes, not a session.

## Consumables

- Endpoint weights (backbone/expert/prompt + config, weights-only)
  uploaded to `mcobzarenco/fontaine-checkpoints` under
  `fontaine_molmo2_ar_40k_ddp4/step_040000` — machine-loss protection
  per the standing rule; optimizer state stays box-local.
- **For the #17 vu5k finalization amendment (execution cell 2): the
  frozen-sanity bar input is the endpoint probe 6.2075@40000.**
- The greedy npz
  (`eval__fontaine_molmo2_ar_40k_ddp4__step_040000__panel_curated_v0_k4l2.npz`)
  is the paired baseline for the #19 draws arm's Δ_AR read.

*Leaderboard: molmo2 greedy row added (decode-cost cells pending the
queued microbench — molmo2 configs were not in the measured set;
nothing mtime-derived is quoted).*
