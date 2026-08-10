# MolmoAct2 on our panel: results, and exactly how the comparison was built

*2026-08-10 · the results + methods companion to the
[pre-registration](2026-08-10-prereg-molmoact2-oob-panel.md) and the
[implementation deep-read](2026-08-10-molmoact2-oob-eval-plan.md).
The interactive artifact this post documents:
**[the 3-policy side-by-side report](https://mcobzarenco-fontaine-reports.static.hf.space/eval__molmoact2_oob_3policy__panel_curated_v0_k4l2.html)**.*

**In plain words:** we took AllenAI's released robot-control model
(MolmoAct2, the SO-100/101 community fine-tune — built on the *same*
Molmo2-ER backbone our current training run starts from) and scored
it on the exact 25,800-frame test panel we use for our own models,
without changing anything about how their model runs. The result:
on the ~31% of our panel that comes from datasets *their model
trained on*, it performs like a sane policy — slightly better than
the copy-the-current-state floor, though still well behind our best
model. On the ~69% of the panel from robot setups it has never seen,
it falls apart: twice the error of simply repeating the robot's
current position. The released checkpoint memorizes its training
rigs rather than learning a transferable SO-100 controller — which
matters for the "just use an open VLA" question on any *new* rig,
including ours.

## Headline numbers

Matched 1.0 s window (chunk steps 0–29, their native horizon), core
frames, `willnorris/bbox-2` excluded (owner amendment — wraparound
units):

| policy (trunk) | pooled | clean-632 | contam-245 |
|---|---|---|---|
| flow teacher 80k top-10-tickets (Gemma-4-E2B + flow) | **3.90** | 3.97 | 3.75 |
| ar 60k continuation (Molmo2-4B, AR) | 4.46 | 4.58 | 4.22 |
| ar 40k endpoint (Molmo2-4B, AR) | 4.56 | 4.68 | 4.31 |
| flow teacher 80k stable-key (Gemma-4-E2B + flow) | 5.06 | 5.17 | 4.84 |
| flow teacher 80k heun-30 original (Gemma-4-E2B + flow) | 5.09 | 5.20 | 4.86 |
| er 60k @15000 (Molmo2-ER, AR, mid-training) | 5.89 | 6.04 | 5.57 |
| state-copy (no model) | 8.32 | 8.58 | 7.75 |
| **MolmoAct2 SO100_101 (Molmo2-ER + their flow expert)** | **13.87** | **16.97** | **7.00** |

Every paired per-frame read (MolmoAct2 − arm, seeded bootstrap CI95,
n = 17,188 / 11,856 / 5,332) classifies MOLMOACT2-WORSE; the closest
it gets is +1.51 [+1.31, +1.71] vs our quarter-trained er run on the
contaminated split. Against state-copy it wins only on contaminated
frames (7.00 vs 7.75).

## How it was computed (the method, end to end)

1. **Their model, untouched.** The public
   `allenai/MolmoAct2-SO100_101` checkpoint (bf16), driven by its
   own `predict_action`: their processor (378×378 squash-resize),
   their prompt template with the state discretized into 256-level
   `<state_N>` tokens, their q01/q99 normalization from the
   checkpoint's `norm_stats.json`, their 10-step Euler flow expert.
   We adapted nothing in the model path; two load-time workarounds
   (a tokenizer config quirk and an fp32→bf16 input cast) come from
   AllenAI's own example server.
2. **Exactly our panel rows.** The predictor iterates the identical
   25,800 frames of every banked eval (plan-pinned, holdout split
   seed 0), and per frame hard-aborts unless the dataset row matches
   the banked npz identity (repo/episode/frame) *and* its raw action
   chunk reproduces the banked ground truth to 1e-6 — so "same
   frames" is verified, not assumed.
3. **Deterministic and resumable.** The flow expert's noise
   generator is seeded per frame with the frame's global index, so
   any subset (the 500-frame smoke) is byte-reproducible inside the
   full sweep. Predictions checkpoint every 500 frames.
4. **Matched horizon.** Their model predicts 30 steps = 1.0 s at
   30 fps; ours predict 50 steps = 1.67 s. Later steps are strictly
   harder (see the MAE-by-timestep charts in the report), so all
   primary numbers re-pool *both* sides over steps 0–29 only — our
   banked full-50 predictions are simply sliced; no re-evaluation of
   our models. Full-50 numbers stay in the report as secondary, for
   our arms only.
5. **Contamination measured, not estimated.** 245 of our 878 panel
   repos appear in their fine-tune mixture list (from their own
   `data_constants.py`, re-derived live at read time with a
   hard-abort pin: 245 repos / 7,996 frames / 5,332 core frames).
   Every read lands pooled / clean / contaminated. "Clean" still
   carries an asterisk: their *pre-training* uses the same repo
   lists at lower weight. And the asymmetry runs the other way for
   us — our models trained on this panel's repo distribution
   (holdout episodes, same repos), so the contaminated split is the
   closest thing to a fair fight in the table.
6. **Oracles gate every number.** Identity columns and state-copy
   rows byte-match across all seven npzs; the MolmoAct2 rows must be
   all-finite inside the window and all-NaN after it; every banked
   arm's full-50 re-pool must reproduce its own report json to
   5e-3; the reads instrument ships a planted-delta `--oracle`
   selftest with every abort branch exercised.
7. **Smoke before sweep.** 500 strided frames with pre-registered
   tripwires (per-dim range sanity vs truth, MAE < 3× state-copy)
   caught nothing — and the smoke's contamination split (parity with
   state-copy on trained-on repos) is what established the
   harness was correct when the pooled number looked shockingly bad.

## What it means

The one-sentence take: **scale + a 1,220-repo community fine-tune
does not buy rig transfer** — visual workspace calibration on an
unseen SO-100 setup is the binding constraint, and no amount of
"more community repos of other rigs" in their mixture solved it.
Their expert predicts kinematically sane trajectories in the wrong
part of the workspace. For the programme this prices the
"off-the-shelf VLA on our rig" path: without fine-tuning on
rig-local demos it starts *below state-copy*. It also sets up an
unusually clean follow-up: our er_60k run shares MolmoAct2's exact
backbone, so tomorrow's endpoint panel compares our from-scratch
decoder (in-domain data) against their flow expert (300k-episode
mixture) on the same trunk, same frames, same window.

Artifacts: [report](https://mcobzarenco-fontaine-reports.static.hf.space/eval__molmoact2_oob_3policy__panel_curated_v0_k4l2.html) ·
[frozen reads json](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__molmoact2_oob_panel_k4l2.json) ·
[contaminated repo list](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__molmoact2_contamination_repos.json) ·
[sweep metadata](https://mcobzarenco-fontaine-reports.static.hf.space/eval__molmoact2_so100_release__panel_curated_v0_k4l2_oob.meta.json).
Instruments: `fontaine/scripts/molmoact2_panel_{predict,reads,report}.py`.
(The proposed full-panel mean-of-10-draws add was cancelled by the
owner at 15:01Z — 0 GPU-h spent.)

*Naming note (owner question 15:10Z): the 80k model here is the
**flow teacher** `bijou_flow_artrunk_h1024_40k_ddp2@80k` — a
Heun-30 multi-step flow expert on the frozen Gemma-4-E2B AR trunk.
Earlier messages called it "snapflow 80k"; strictly, *SnapFlow* is
the 1-NFE student distilled FROM this teacher (a different, faster,
slightly worse checkpoint). All report labels now say flow teacher.*
