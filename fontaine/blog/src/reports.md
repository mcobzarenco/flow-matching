# Reports

Every panel eval dumps a self-contained HTML report (headline tables,
per-repo breakdowns, worst-frame galleries) plus a JSON that the
frozen results instruments consume. The HTML reports and the frozen
analysis JSONs are hosted on the dedicated
[fontaine-reports Space](https://mcobzarenco-fontaine-reports.static.hf.space/)
(moved off this Space 2026-08-10, owner request — the ~10 MB
self-contained report files were the bulk of this Space's storage);
this page indexes them. Posts link the specific reports behind their
numbers.

## Owner-side reports

- [AR-pretrained trunks for flow decoders (interim, 2026-08-05)](https://mcobzarenco-fontaine-reports.static.hf.space/stage2_ar_trunk_report.html)
  — the paired two-arm stage-2 phase behind the −2.7 MAE
  AR-adaptation number: same expert/init/seed/data order, trunk
  stock vs AR-pretrained; Δ−2.69 (−20%) at step 2,500, ~8× the probe
  noise floor. Shared by the owner 2026-08-06; the direct motivation
  for the Molmo2 AR-first amendment.

Naming: `eval__<run>__<checkpoint>__<panel+sampler>` — `heun30` =
Heun 30-step, `1nfe_euler1` = single Euler step (1 expert eval),
`drawsN` = mean-of-N ensembling, `stable`/`stablekey` = stable noise
keying ([#18.2](posts/2026-08-05-noise-reseed-prereg.md)), unmarked =
legacy index keying. `panel_curated_v0_k4l2` is the v1 25,800-frame
panel; `panel_v2` is the [dedup-hardened
revision](posts/2026-08-06-panel-v2-amendment.md).

## SnapFlow 1-NFE distillation ([results](posts/2026-08-06-snapflow-results.md))

- [student 1-NFE, single draw (primary)](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_flow_snapdistill_h1024_30k_1xh100__step_030000__panel_curated_v0_k4l2_1nfe_euler1.html)
- [student 1-NFE, mean-of-5](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_flow_snapdistill_h1024_30k_1xh100__step_030000__panel_curated_v0_k4l2_1nfe_euler1_draws5.html)
- [student 1-NFE, mean-of-10 (deployment headline)](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_flow_snapdistill_h1024_30k_1xh100__step_030000__panel_curated_v0_k4l2_1nfe_euler1_draws10.html)
- [frozen analysis JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__snapflow_distill_30k_k4l2.json)
  (`snapflow_results.py`, pre-registered reads)

## Flow teacher `bijou_flow_artrunk_h1024_40k_ddp2` @80k

- [Heun-30, single draw (v1 anchor)](https://mcobzarenco-fontaine-reports.static.hf.space/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_curated_v0_k4l2_draws1_heun30.html)
- [Heun-30, mean-of-5](https://mcobzarenco-fontaine-reports.static.hf.space/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_curated_v0_k4l2_draws5_heun30.html)
- [Heun-30, mean-of-10](https://mcobzarenco-fontaine-reports.static.hf.space/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_curated_v0_k4l2_draws10_heun30.html)
- [Heun-10, single draw](https://mcobzarenco-fontaine-reports.static.hf.space/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_curated_v0_k4l2_draws1_heun10.html)
- [Heun-10, mean-of-10](https://mcobzarenco-fontaine-reports.static.hf.space/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_curated_v0_k4l2_draws10_heun10.html)
- [Heun-30, stable keying (re-banked anchor)](https://mcobzarenco-fontaine-reports.static.hf.space/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_curated_v0_k4l2_stablekey_heun30.html)
  ([results](posts/2026-08-06-stablekey-rebank-results.md))
- [legacy k4l2 panel, Heun-30](https://mcobzarenco-fontaine-reports.static.hf.space/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_k4l2_heun30.html)
- [state-masked Q4 probe](https://mcobzarenco-fontaine-reports.static.hf.space/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__stateprobe_q4_state-masked.html)
  ([results](posts/2026-08-06-state-probe-results.md))
- Draws-fairness frozen reads:
  [analysis](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__draws_fairness_k4l2.json) ·
  [validate](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__draws_fairness_k4l2_validate.json)
  ([results](posts/2026-08-06-draws-fairness-results.md))
- σ_draw:
  [finalization](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sigma_draw_finalization.json) ·
  [direct measurement](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sigma_draw_direct.json)
  ([amendment](posts/2026-08-06-sigma-draw-finalization.md))

## Flow teacher @40k (arch-batch control)

- [panel-v2, Heun-30, stable keying](https://mcobzarenco-fontaine-reports.static.hf.space/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_040000__panel_v2_ctrl_heun30_draws1_stable.html)
  — the [arch batch #1](posts/2026-08-06-prereg-arch-batch-1.md)
  control ·
  [ctrl-only analysis](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__arch_batch_1_ctrl_only.json)

## AR mainline `bijou_arb_rcond_100k_ddp4` @100k

- [curated_v0 panel (anchor)](https://mcobzarenco-fontaine-reports.static.hf.space/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_curated_v0_k4l2.html)
- [sealed split](https://mcobzarenco-fontaine-reports.static.hf.space/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_curated_v0_k4l2_sealed.html)
  ([sealed plan](posts/2026-08-05-sealed-plan-v2.md))
- [legacy k4l2 panel](https://mcobzarenco-fontaine-reports.static.hf.space/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2.html)
  — carries the **accuracy-by-field block** (narrated `+fields` arm:
  holding 0.807 · progress MAE 0.062 · event 0.878 · visible 0.319
  over ~9k judge-labeled frames; curated_v0 panel above has its own:
  0.814/0.063/0.879/0.316) —
  [pre-reg note](posts/2026-08-08-prereg-accuracy-by-field.md)
- [state-masked Q4 probe](https://mcobzarenco-fontaine-reports.static.hf.space/eval__bijou_arb_rcond_100k_ddp4__step_100000__stateprobe_q4_state-masked.html) ·
  [state-probe analysis](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__state_probe_q4.json)
  ([results](posts/2026-08-06-state-probe-results.md))

## Box batch 40k AR arms ([results](posts/2026-08-06-box-batch-results.md))

- [A-s0 `fontaine_arb_rcond_40k_1xh100` panel](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_arb_rcond_40k_1xh100__step_040000__panel_curated_v0_k4l2.html) ·
  [state-masked probe](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_arb_rcond_40k_1xh100__step_040000__stateprobe_q4_state-masked.html)
- [A-s1 seed replicate panel](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_arb_rcond_40k_1xh100_s1__step_040000__panel_curated_v0_k4l2.html)
- [A-s2 seed replicate panel](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_arb_rcond_40k_1xh100_s2__step_040000__panel_curated_v0_k4l2.html)
- [aux-off arm state-masked probe](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_arb_rcond_auxoff_40k_1xh100__step_040000__stateprobe_q4_state-masked.html)
- [batch analysis JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__box_batch_40k_k4l2.json)

## State-dropout arm C `fontaine_arb_rcond_statedrop80_40k_1xh100` @40k ([results](posts/2026-08-06-statedrop-results.md))

- [endpoint panel (curated v0 k4l2)](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_arb_rcond_statedrop80_40k_1xh100__step_040000__panel_curated_v0_k4l2.html)
- [masked-state reliance eval (stateprobe q4)](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_arb_rcond_statedrop80_40k_1xh100__step_040000__stateprobe_q4_state-masked.html)
- [frozen analysis JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__statedrop_40k_k4l2.json)

## Molmo2 AR trunk `fontaine_molmo2_ar_40k_ddp4` @40k ([results](posts/2026-08-08-molmo2-endpoint-results.md))

- [endpoint panel, greedy (curated v0 k4l2)](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_molmo2_ar_40k_ddp4__step_040000__panel_curated_v0_k4l2.html)
  — the #17 BEATS read (6.0079/2.1871 vs A-s0, paired −1.717)
- [frozen endpoint analysis JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__molmo2_endpoint_k4l2.json)
  (`molmo2_endpoint_results.py`, pre-registered reads)
- [draws10_t1 frozen analysis JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__draws10_t1_molmo2_40k_k4l2.json)
  — leaderboard row 9, Δ_AR −0.154 mean-collapse read
- [decode-cost microbench JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__leaderboard_decode_microbench_molmo2.json)
  — rows 8+9 cost cells (box-measured)
- Accuracy-by-field: **missing from this panel by bug** (the narrated
  pass silently skipped molmo2 checkpoints; found + fixed `2f4d575`
  2026-08-08). The table landed at the 60k endpoint instead — see the
  [fields panel results](posts/2026-08-09-molmo2-fields-panel-results.md)
  in the @60k section below

## Molmo2 AR trunk `fontaine_molmo2_ar_60k_ddp4` @60k ([results](posts/2026-08-09-molmo2-60k-results.md) · [fields panel](posts/2026-08-09-molmo2-fields-panel-results.md))

- [endpoint panel json, greedy (curated v0 k4l2)](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_molmo2_ar_60k_ddp4__step_060000__panel_curated_v0_k4l2.json)
  — 5.8602/2.0719, the IMPROVED endpoint (−0.139 paired vs 40k) that
  repointed the attach screen to `step_060000`
- [accuracy-by-field table json](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_molmo2_ar_60k_ddp4__step_060000__panel_curated_v0_k4l2_fields.json)
  — the registered fields panel read at this endpoint (visible slots
  0.32 → 0.82 vs the Gemma trunk)
- [frozen 60k-vs-40k analysis JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__molmo2_60k_vs_40k_k4l2.json)
  (`molmo2_60k_results.py`, pre-registered reads)
- [browsable HTML panel, greedy](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_molmo2_ar_60k_ddp4__step_060000__panel_curated_v0_k4l2.html)
  — per-frame retained predictions (32 samples), rendered at eval
  time 23:49Z 08-08. (Correction: the eval DID run with `--report` —
  the launcher always had it; the HTML sat unsynced on the box. The
  earlier "needs a ~1 GPU-h re-run" note was wrong; no re-run needed.)
- [browsable HTML panel, fields run](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_molmo2_ar_60k_ddp4__step_060000__panel_curated_v0_k4l2_fields.html)
  — same panel with the narrated `+fields` arm riding (the
  accuracy-by-field source run)
- Checkpoint weights on the hub:
  [`fontaine-checkpoints/fontaine_molmo2_ar_60k_ddp4/step_060000`](https://huggingface.co/mcobzarenco/fontaine-checkpoints/tree/main/fontaine_molmo2_ar_60k_ddp4/step_060000)

## Molmo2-ER trunk `fontaine_molmo2_er_60k_ddp4` @15k (mid-training, owner-requested)

- [browsable HTML panel, greedy (curated v0 k4l2)](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_molmo2_er_60k_ddp4__step_015000__panel_curated_v0_k4l2.html)
  — owner request 08:29Z 08-10: the @15000 checkpoint hub-copied,
  downloaded locally, and panel-evaled mid-run; per-frame retained
  predictions (32 samples), rendered 11:51Z 08-10
- [panel json](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_molmo2_er_60k_ddp4__step_015000__panel_curated_v0_k4l2.json)
  — pooled 7.5283/3.5590 (quarter-training snapshot; the run's own
  decision point stays the @60k endpoint panel)
- [frozen record-only reads JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__er15k_panel_vs_banked_k4l2.json)
  (`er15k_panel_reads.py`) — paired vs banked 40k endpoint
  +1.52 CI95 [+1.39, +1.54], vs 60k continuation +1.67
  [+1.52, +1.68], both ABOVE-BASELINE as expected at 15k/60k steps
- Checkpoint weights on the hub:
  [`fontaine-checkpoints/fontaine_molmo2_er_60k_ddp4/step_015000`](https://huggingface.co/mcobzarenco/fontaine-checkpoints/tree/main/fontaine_molmo2_er_60k_ddp4/step_015000)

## MolmoAct2-SO100_101 out-of-band panel ([pre-reg](posts/2026-08-10-prereg-molmoact2-oob-panel.md) · [plan/deep-read](posts/2026-08-10-molmoact2-oob-eval-plan.md))

- [3-policy side-by-side HTML report](https://mcobzarenco-fontaine-reports.static.hf.space/eval__molmoact2_oob_3policy__panel_curated_v0_k4l2.html)
  — owner spec 11:59Z 08-10: flow teacher 80k (top-10-tickets + stable-key + heun-30 original)
  vs the released MolmoAct2 SO-100/101 fine-tune vs state-copy, same
  25,800 frames, matched 30-step/1.0 s window, 32-frame gallery with
  4 policies overlaid per joint; rendered 14:25Z 08-10
- [frozen matched-window reads JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__molmoact2_oob_panel_k4l2.json)
  (`molmoact2_panel_reads.py`) — matched-window chunk MAE core
  frames (`willnorris/bbox-2` excluded, owner amendment 13:14Z):
  flow-teacher top-10-tickets **3.90** / state-copy 8.32 / MolmoAct2
  **13.87** pooled, **16.97** clean-633 vs **7.00** contaminated-245
  (beats state-copy only on repos in its own fine-tune mixture,
  −0.75; trails the flow teacher by +3.29 [+3.11, +3.48] even there)
- [contamination repo list](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__molmoact2_contamination_repos.json)
  — 245/878 panel repos in their `SO100_SO101_MOLMOACT2` mixture
  (7,996 frames, 5,332 core), derived live from their repo file
- [sweep metadata](https://mcobzarenco-fontaine-reports.static.hf.space/eval__molmoact2_so100_release__panel_curated_v0_k4l2_oob.meta.json)
  — their `predict_action` end-to-end, bf16, 10-step Euler, seed =
  concat index; 25,800 frames at 352 f/min, ~1.3 GPU-h total

## er_60k ENDPOINT @60000 — THE ER decision read ([pre-reg](posts/2026-08-09-prereg-molmo2-er-60k.md)): ER init WINS both legs

- [endpoint eval report](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_molmo2_er_60k_ddp4__step_060000__panel_curated_v0_k4l2.html)
  — chained in-unit panel (rc 13:28Z 08-11, ~153/155 GPU-h run
  total): fast path **5.7782/1.9898** core — the best banked trunk
  number to date; narrated arm 5.83 (+0.055 pairing, 45% win); aux
  holding 0.915 / progress MAE 0.060 / event 0.858 / visible 0.822
- [paired decision reads JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__er60k_endpoint_vs_banked_k4l2.json)
  (`er15k_panel_reads.py`, key `bijou@60000`) — vs 40k endpoint
  (6.0079) pooled **−0.2297** [CI95 −0.281, −0.154] BELOW-BASELINE;
  vs 60k-cont (5.8602) pooled **−0.0821** [CI95 −0.126, −0.025]
  **BELOW-BASELINE, CI excludes zero** = the pre-registered decision
  read: the ER-init trunk beats both banked baselines at matched
  panel class; state-copy integrity byte-match ×3. Rung trajectory
  15k +1.52 → 35k +0.28 → 55k −0.18 → **60k −0.23** vs the 40k
  endpoint. Rig-data effect read at endpoint: NOT split-compatible —
  the panel contains no owner-rig repos (checked against the npz
  `repo_id` identity), recorded as skipped per the pre-reg's
  if-clause
- Weights: [`step_060000`](https://huggingface.co/mcobzarenco/fontaine-checkpoints/tree/main/fontaine_molmo2_er_60k_ddp4/step_060000)
  (weights-only, hub-uploaded 12:44Z in 42.0 s, commit 4ed3dd0)

## er_60k @60000 events one-off (owner request 12:44Z 08-11, record-only)

What events does the model actually see? Generated event strings vs
the weak judge labels on the 8,987 judge-labeled panel frames, via
the new `--dump-generations` instrument (commit 7f43c54 — main-arm
generations retained under explicit `--generate`).

- [standalone report](https://mcobzarenco-fontaine-reports.static.hf.space/report__er60k_events_oneoff.html)
  — 13-class model×gt confusion (incl. none/none), per-class P/R,
  and 136 image cards across hit / class-swap / miss / false-alarm
  galleries + probe examples (repo-diverse selection)
- Headline: both-none 7,238 · hits 333 · swaps 129 · **misses 683** ·
  false alarms 604. On the 1,145 gt-event frames the model speaks on
  40%, but class-agrees 72% when it does; exact-string match 3.6%
  (same event, different words)
- [constrained-probe JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__er60k_events_probe.json)
  — on the misses, a 1-step `none`-ban re-decode (frame's own
  generated prefix replayed; unbanned replay reproduced `none`
  bit-exact 679/683): **forced guess lands the gt class 63%** →
  the dominant miss mode is *saw-it-under-threshold*, not blindness
  (idle 86% / release-place 80% / occlusion 72% / blur 62%; camera
  quirks 10% and episode markers 0% are the genuinely-not-encoded
  tail)
- [confusion JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__er60k_events_confusion.json)
  · [dump-pass eval json](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_molmo2_er_60k_ddp4__step_060000__panel_curated_v0_k4l2_events.json)
  · [per-frame generations dump](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_molmo2_er_60k_ddp4__step_060000__panel_curated_v0_k4l2_events_generations.json)
  (25,800 rows, ~1.55/4 GPU-h). Instrument oracle: presence acc
  0.8568 vs banked 0.8582 — Δ 13 frames, inside the documented
  cross-world-size bf16 batch-composition band (banked ran 4-way on
  the box)

## er_60k @55000 owner-requested panel, standard both-arms ([pre-reg](posts/2026-08-09-prereg-molmo2-er-60k.md), record-only)

- [standard eval report](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_molmo2_er_60k_ddp4__step_055000__panel_curated_v0_k4l2.html)
  — the @55000 read (rc=0 12:00Z 08-11, ~2.2/8 GPU-h): fast path
  **5.8269/2.0172** core + narrated arm 5.869 (+0.039 pairing, 46%
  win — same ~0.04–0.05 narration-cost class as 15k/35k); aux vs
  weak labels at full n≈8,987: holding acc 0.915→**0.920**, progress
  MAE 0.065→**0.060**, event acc 0.875→0.858, visible acc
  0.823→0.822
- [class-matched paired reads JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__er55k_panel_vs_banked_k4l2.json)
  (`er15k_panel_reads.py`, key `bijou@55000`) — vs 40k endpoint
  (6.0079) pooled **−0.1810** [CI95 −0.232, −0.105]
  **BELOW-BASELINE** (first such read for the ER trunk; @35000 was
  +0.281 above), vs 60k-cont (5.8602) −0.0334 [−0.078, +0.024]
  CI-SPANS-0 = parity at 92% training; state-copy integrity
  byte-match ×3; record-only — the @60000 endpoint panel decides
- Weights: [`step_055000`](https://huggingface.co/mcobzarenco/fontaine-checkpoints/tree/main/fontaine_molmo2_er_60k_ddp4/step_055000)
  (weights-only, hub-uploaded 09:4xZ in 42.9 s)

## er_60k @35000 owner-requested panel, standard both-arms ([pre-reg](posts/2026-08-09-prereg-molmo2-er-60k.md), record-only — SUPERSEDES the aux-arm read below)

- [standard eval report](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_molmo2_er_60k_ddp4__step_035000__panel_curated_v0_k4l2.html)
  — the complete 35k read (rc=0 00:41Z 08-11, ~2.2/8 GPU-h): fast
  path **6.2892/2.3746** core + narrated arm 6.342 (+0.047 pairing,
  44% win — narration costs the same ~0.05-class as at 15k); aux vs
  weak labels at full n≈8,987, ALL FOUR improved from 15k: holding
  acc 0.899→**0.915**, progress MAE 0.075→**0.065**, event acc
  0.862→**0.875**, visible acc 0.704→**0.823**
- [class-matched paired reads JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__er35k_panel_vs_banked_k4l2.json)
  (`er15k_panel_reads.py`, key `bijou@35000`) — vs 40k endpoint
  (6.0079) pooled **+0.2813** [CI95 +0.199, +0.337], vs 60k-cont
  (5.8602) +0.4290 [+0.353, +0.467]; ABOVE-BASELINE at 58% training,
  the 15k gap (+1.52) ~82% closed; state-copy integrity byte-match ×3

## er_60k @35000 aux-narrated arm (superseded by the standard read above)

- [aux-narrated eval report](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_molmo2_er_60k_ddp4__step_035000__panel_curated_v0_k4l2_aux.html)
  — owner request 20:47Z 08-10: `--generate subgoal holding progress
  event visible` (actions follow the model's own generated aux
  lines); core **6.3425/2.3770** at 58% training (er15k
  narrated-class was 7.601), win-rate 77% vs state-copy, Q3
  condition sensitivity 1.62
- [paired reads JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__er35k_aux_panel_vs_banked_k4l2.json)
  — vs 40k endpoint +0.335 [+0.247, +0.387], vs 60k-cont +0.482
  [+0.399, +0.517]; **cross-class caveat** (narrated arm vs
  fast-path baselines) — the standard both-arms eval relaunched
  same-session supersedes these with class-matched reads when it
  lands (~01:0xZ 08-11)
- Weights: [`step_035000`](https://huggingface.co/mcobzarenco/fontaine-checkpoints/tree/main/fontaine_molmo2_er_60k_ddp4/step_035000)
  (weights-only, hub-uploaded 20:5xZ in 42.4 s)

## MolmoAct2 SO-101 rig fine-tune ([pre-reg](posts/2026-08-10-prereg-molmoact2-rig-finetune.md) · [runbook](posts/2026-08-10-molmoact2-rig-finetune-runbook.md) · [results](posts/2026-08-10-molmoact2-rig-ft-results.md))

- [anchor-rung HTML report](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_so101_rig_ae_r1__anchor_rungs.html)
  — `rig_ft_r1` (AE-only, 2000 steps, ~2.7/12 GPU-h): rung curve
  zero-shot 28.95 → **3.23@2000** vs state-copy 9.08 on the 240 rig
  anchor frames; per-timestep curves, motion-corr small multiples,
  8-frame strided trajectory gallery. Pre-reg PASS at every gate;
  reads are train-frame sanity (contaminated by construction — the
  real eval is on-rig rollouts, runbook §3–4)
- Frozen reads:
  [zero-shot/preflight](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__molmoact2_rig_preflight.json) ·
  [step 500](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__molmoact2_rig_ft_step500.json) ·
  [step 1000](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__molmoact2_rig_ft_step1000.json) ·
  [step 1500](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__molmoact2_rig_ft_step1500.json) ·
  [step 2000](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__molmoact2_rig_ft_step2000.json)
  (`molmoact2_rig_preflight.py --model <rung>`, identical 240 rows)
- Weights on the hub:
  [`molmoact2_so101_rig_r1_step2000`](https://huggingface.co/mcobzarenco/fontaine-checkpoints/tree/main/molmoact2_so101_rig_r1_step2000)
  — AE + resized-embedding delta vs the released checkpoint (trunk
  deduplicated, 704/707 tensors sha-verified byte-identical);
  serve-ready dir stays local at
  `~/checkpoints/molmoact2-so101-rig-r1-step2000-hf`

## Golden-ticket noise screen ([close-out](posts/2026-08-08-goldenticket-results.md) · [visual report](posts/2026-08-08-goldenticket-visual-report.md))

- Frozen stage analyses:
  [stage 1 (R1 CONFIRM)](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__goldenticket_stage1.json) ·
  [stage 2 (R2 REAL)](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__goldenticket_stage2.json) ·
  [stage 3 (R3 INTERESTING + R4a/R4b)](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__goldenticket_stage3.json)
- [jerk-pick selector read](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__jerkpick_selector.json)
  — SDN smoothness-prior test on banked draw stacks (flow null / AR small)

## Frozen-trunk flow experts @10k, panel_v2 ([attach memo](posts/2026-08-09-molmo2-stage2-attachment-decision.md) · [tiny results](posts/2026-08-10-tiny-expert-results.md))

Both experts sit on the hard-frozen 60k trunk (sha `e6ed783b`),
scored on the panel_v2 k4l2 plan (15,056 core frames pooled) —
numbers compare within this section, not with the v1 scoreboard.

- [F arm `fontaine_molmo2_flow_frozen_10k_ddp4` @10k, Heun-30 single-draw stable](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_molmo2_flow_frozen_10k_ddp4__step_010000__panel_v2_heun30_draws1_stable.html)
  — 9.4157/2.9581, the attach-screen F endpoint (previously box-only;
  pushed with the tiny readout)
- [tiny arm `fontaine_molmo2_flow_tiny_h256_10k_1xh100` @10k, same decode](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_molmo2_flow_tiny_h256_10k_1xh100__step_010000__panel_v2_heun30_draws1_stable.html)
  — the T1 capacity rung endpoint
- [frozen Δ_capacity analysis JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__tiny10k_delta_capacity.json)
  (`attach_seam_results.py` read-1 machinery at explicit paths, per
  the pre-reg)
- Checkpoints on the hub:
  [`F/step_010000`](https://huggingface.co/mcobzarenco/fontaine-checkpoints/tree/main/fontaine_molmo2_flow_frozen_10k_ddp4/step_010000) ·
  [`tiny/step_010000`](https://huggingface.co/mcobzarenco/fontaine-checkpoints/tree/main/fontaine_molmo2_flow_tiny_h256_10k_1xh100/step_010000)
  (both weights-only, backbone deduplicated to the 60k trunk)

## 100-seed sim policy eval ([pre-reg](posts/2026-08-11-prereg-sim-policy-eval-100seeds.md) · [results](posts/2026-08-12-sim100-results.md))

Five arms × seeds 0–99 in the v0 SO-101 sim (sysid'd servos), paired
design; primary metric = boat→disk progress (cm). 0/500 successes;
the engagement/direction split is the finding.

- [HTML report + video gallery](https://mcobzarenco-fontaine-reports.static.hf.space/report__sim100_seed_eval.html)
  — per-arm tables, paired CIs, four charts, best/median/worst clips
  per arm (+ the er60k reach-but-miss money shots)
- [frozen analysis JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim100_seed_eval.json)
  (`sim100_reads.py`: gates, summaries, paired bootstrap reads,
  ordering read auto-skipped — rung arms killed by the phase-2
  amendment)

## Contact-shadow pass v4 — the composite's missing shadow, fitted and gated ([lit page](papers/composite-shadows.md), 08-13)

The v3 composite's pasted arm casts no shadow on the real plate —
the one physics law every real frame obeys that no composite frame
did. Leg (a) measured the real arm's own shadow from 200 frames × 25
bank episodes (frame ÷ episode-plate darkening vs the sim-replayed
silhouette slid along candidate light directions): **real and
directional** — contrast +0.091 CI95 [0.081, 0.100] vs ring control,
zenith 30° / azimuth 112.5° (85% bootstrap stability), strength
0.392, softness σ 24 px. `render_style="v4"` = v3 + the fitted
shadow multiply-darkening the top plate (shared projector
`sim/shadow.py`, 12 oracles; wrist bit-identical to v3). Paired
encoder gate (seeds 0..99, fresh both arms — the banked v3 anchor
0.673 predates the bracket flip; fresh v3 reads 0.721): top 5-NN
AUROC 0.721 → **0.715**, and the paired per-seed read is decisive —
Δknn5 −1.04e-07 CI95 [−1.53e-07, −5.6e-08], 66/100 seeds closer,
**~10% of the remaining top-cam knn5 excess closed**. Wrist 100/100
tied. GO recorded; default stays v3 pending the sim100 amendment-5
owner call. ~0.04 GPU-h. For scale: v1 scene −0.049, v2 inpainting
−0.103, v3 content −0.100, v4 shadows −0.006 — the tail is thinning.

- [light-fit JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__contact_shadow_fit.json)
  · [gate v3 arm](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_encoder_ood_probe_v4gate_v3arm.json)
  · [gate v4 arm](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_encoder_ood_probe_v4gate_v4arm.json)
- [direction/softness chart](https://mcobzarenco-fontaine-reports.static.hf.space/chart__contact_shadow_fit.png)
  · [v4 sample frame](https://mcobzarenco-fontaine-reports.static.hf.space/chart__contact_shadow_v4_sample.png)

## Fitted wrist lens — cubemap render path + gate: the fit's center term double-counts the pose, the curve-only refit passes ([lit page](papers/fisheye-lens-fitting.md), 08-13)

The deployed wrist warp assumed an ideal equidistant lens centered
at the image midpoint; the plumb-line fit on the 150 pinned real
frames (leg (a), 08-13 01:4xZ) measured the real module off-center
(22 px left / 14 px down, ~5σ) with stronger peripheral compression
(−12.8 px at the corner, CI-excludes-0). Leg (b) landed the render
path that can draw ANY lens: the wrist source is a pinhole cubemap
around the camera axis (output→face map precomputed, so runtime is
one bilinear gather; only referenced faces render; face focal
matched to the deployed source so A/Bs read geometry, not
sharpness; the camera-riding headlight is re-pointed at the base
axis per face — without that, face boundaries carry a shading
seam, caught by the rotated-cubemap oracle at mean|Δ| 6.77). Gate
read (pre-reg 03:27Z, 20 seeds × 5 draws, er60k trunk, control
0.560): **full fit 0.667 FAIL — and a labeled post-hoc center-only
arm reads 0.672, reproducing the whole regression**. The 08-12
wrist pose re-tune was fit to real frames under the deployed lens,
so it already absorbed the principal-point offset (~2.6°
yaw-equivalent); bolting the fitted center on top applies it
twice. The **curve-only refit (k2 +0.101, k4 −0.036) passes: 0.523
≤ the 0.548 gate**, paired Δknn5 −7.6e-07 CI95 [−8.5e-07,
−6.8e-07], **96/100 frames closer** — ~7× the contact-shadow GO
effect, and cost-neutral (single face covers the frame: 73 vs 70
ms/tick). `lens_model="fitted"` now pins the curve-only params;
default stays equidistant pending the sim100 amendment-6 owner
call. Top cam bit-identical across all arms (0.713 — now the
frontier number). Full-fit center use is parked behind a joint
pose+lens refit (`sim-joint-pose-lens-refit`, owner-held).
~0.04 GPU-h total (4 probe arms).

- [gate chart](https://mcobzarenco-fontaine-reports.static.hf.space/chart__lens_gate.png)
  · sample frames: [equidistant](https://mcobzarenco-fontaine-reports.static.hf.space/wrist_lens_seed0_equidistant.png)
  · [full fit](https://mcobzarenco-fontaine-reports.static.hf.space/wrist_lens_seed0_fitted_full.png)
  · [curve-only](https://mcobzarenco-fontaine-reports.static.hf.space/wrist_lens_seed0_fitted_curveonly.png)
- gate JSONs: [equidistant](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_encoder_ood_probe_lensgate_equidistant_arm.json)
  · [full fit](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_encoder_ood_probe_lensgate_fitted_arm.json)
  · [center-only](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_encoder_ood_probe_lensgate_centeronly_arm.json)
  · [curve-only](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_encoder_ood_probe_lensgate_curveonly_arm.json)
- leg (a): [fit JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__wrist_lens_fit.json)
  · [fit chart](https://mcobzarenco-fontaine-reports.static.hf.space/chart__wrist_lens_fit.png)
  — plumb-line θ→r fit, decompositions, bootstrap

## Appearance screen CONSOLIDATED ([report](posts/2026-08-14-appearance-screen-report.md), 08-14)

The whole top-cam appearance programme in one chart-led post,
written for the three pending promotion decisions: nine
pre-registered reads, ~0.2 GPU-h total — clutter patches carry the
removable share (0.713 → 0.556), materials are free riders, texture
refuted twice, wrist-neutral, stack 0.5521 sub-additive; the
remaining gap to the 0.328 pipeline floor is arm geometry/relief
(renderer-class, priced separately). The individual entries below
remain the frozen records.

- [ladder + paired-reads chart](https://mcobzarenco-fontaine-reports.static.hf.space/chart__appearance_screen_ladder.png)

## Top-cam gap decomposition — the whole 0.713 residue lives in the rendered foreground pixels (08-13)

With the wrist at 0.523 under the curve-only fitted lens, the
frontier is the top cam's 5-NN AUROC 0.713 — unmoved by every lens
arm. The decomposition probe asked WHERE it lives before spending on
any lever: five ablation arms on the pinned 20×5 reset-render
harness (numpy backend, fresh in-run v3 baseline reproduces the
banked anchor exactly at 0.713). The verdict is unambiguous.
**Swapping the rendered foreground for real dynamic pixels — real
arm/boat/hand pixels mined from bank-episode frames, re-lit
source→target episode via the bank affines, pasted on a *different*
drawn plate by the exact production arithmetic — collapses the read
to 0.328**, at the clean-repo anchor (0.283) and below the 0.5 null:
the compositing pipeline (plates, mask feathering, grade, noise) can
reach real-level; only the pixels it pastes are wrong. The
arithmetic residue read agrees from the other side: fg→plate (0.869)
vs plate-only (0.865) differ by +0.004 AUROC (paired Δknn5 +2.3e-07
CI95 [1.4e-07, 3.3e-07] — real but ~5% of the armless shift and far
under the +0.05 registered bar), so edge-weighting + foreground-PSF
carry ~nothing. Two labeled surprises: (1) the fg→plate/plate-only
arms read FARTHER than v3 (0/100 frames closer, +4.5e-06) — an
armless frame is itself OOD (every real frame has the arm
somewhere), so registered rule 1's first clause misfired for a
measured, explainable reason; the real-fg arm carries the
localization instead. (2) The shadow-band crop read is near-ceiling
(v3 0.989, v4 0.988, crop clean anchor 0.304): within the table
region the encoder separates sim from real almost perfectly, and the
fitted shadow does NOT close it locally (paired crop Δ CI includes
0) — but the registered box grew to cover most of the lower frame
(89:480, 81:640), i.e. it includes the rendered arm itself, so it
localizes the signal to "the region containing the pasted render",
consistent with the real-fg verdict rather than a separate shadow
story. v4's full-frame paired read replicated the shadow gate on the
20×5 protocol (−8.3e-08 CI [−1.34e-07, −3.1e-08], 66/100 closer).
**Decision (registered rule): the next leg is foreground appearance**
— and the sample frames name the prime suspect: the untextured
gray clutter stand-ins (cylinder mug, white disk) sit next to
photoreal plates; queued as `sim-foreground-appearance-pass` with a
content-split leg (clutter vs arm vs benchy, keeping the rest
rendered to dodge the armless confound) before any material work.
~0.02 GPU-h embeds; renders CPU.

- [analysis JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_top_gap_decomposition.json)
  · [chart](https://mcobzarenco-fontaine-reports.static.hf.space/chart__top_gap_decomposition.png)
- [arm strip](https://mcobzarenco-fontaine-reports.static.hf.space/top_gap_arms_seed0_strip.png)
  — v3 render / real-fg composite / plate-only, same slot

## Foreground content split — the clutter stand-ins (~5% of pixels) carry the removable share (08-13)

Leg (a) of the appearance pass asked WHICH rendered class carries
the 0.713: arm bodies (96 geoms, ~7.1% of pixels), benchy (341,
~0.1%), the clutter stand-ins mouse/mug/laptop/pcb (~5.1%), or the
disk (~0.5%, split out of "clutter" as the always-rendered named
suspect). One production v3 instance was hooked at `_composite`, so
every slot yields all 10 arms — v3, plate-only, no_(class),
only_(class) — through the exact production arithmetic with a
segmentation-restricted mask: same physics, same drawn plate, same
sensor noise (RNG-state restore), making the paired Δ exactly the
class's visible-pixel effect (in-run oracle: hooked v3 bit-exact ==
the production observation, all 100 slots; fresh v3 read 0.7127,
inside the registered abort band). **Removing the clutter stand-ins
alone collapses the read 0.713 → 0.576** (paired Δknn5 −1.73e-06
CI95 [−1.92e-06, −1.54e-06], 99/100 frames closer) — the unique
class past the registered ±0.05 material bar: no_disk −0.006 and
no_benchy −0.002 are CI-excl-0 but immaterial, and no_arm reads
+0.113 WORSE, the armless-content confound the decomposition
labeled (every real frame has the arm). The keep-only duals all
pull toward real when added to the bare plate (only_arm 0.654,
only_clutter 0.824, only_benchy/only_disk 0.832 vs plate-only
0.866), so no class is rendered badly enough to overwhelm its own
content benefit — the ranking rests on the removal direction, which
is also the honest one for clutter (real episodes genuinely vary
clutter presence; the bank plates are mined clutter-free).
**Registered primary rule fires: leg (b) target = clutter
appearance** (real-crop textures or plate-sourced patches for the
gray untextured stand-ins). Ceiling note, registered before leg
(b): no_clutter's 0.576 still sits far above the real-fg anchor
0.328, and the arm carries most of that remainder (only_arm 0.654
vs the real-content direction ~0.33) — clutter alone cannot close
the gap, it is just the best ROI per rendered pixel. Renders CPU
(~5 min), embeds 12 groups ~0.02 GPU-h.

- [analysis JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_fg_content_split.json)
  · [chart](https://mcobzarenco-fontaine-reports.static.hf.space/chart__sim_fg_content_split.png)
- [arm strip](https://mcobzarenco-fontaine-reports.static.hf.space/fg_split_arms_seed0_strip.png)
  — v3 / no_clutter / only_clutter, same slot

## Foreground appearance fix — real-crop clutter patches beat the removal ceiling, gate PASS (08-13)

Legs (b)+(c) of the appearance pass (pre-reg in-channel 05:23Z)
executed the registered follow-up: replace the untextured gray
stand-ins with **real-pixel crops pasted into the plate**.
`make_clutter_crops.py` mined per-object RGBA crops from the bank
episodes' naive per-pixel medians (source episode = largest measured
blob; alpha = the feathered static-novelty mask vs the
gain/bias-corrected global plate — the same statistic the bank pass
localized the objects with; recomputed areas bit-match the manifest),
normalized to global-plate lighting. `clutter_patch.py` pastes them
at the drawn poses by inverse warp through the verified analytic
fisheye model (target pixel → object-height plane → rigid
drawn→mined transform → source pixel, bilinear), so translation, yaw
jitter and the fisheye's local scale all ride the camera model; the
active episode's affine grades the patch exactly like the rendered
foreground; the `fixed_canonical` pcb pastes at its real measured
location (identity). Zero extra appearance-RNG draws — slots pair
1:1 with production v3. The leg (a) harness then read three arms off
one hooked instance: **patched 0.556 vs v3 0.713 (ΔAUROC −0.157,
paired Δknn5 −2.02e-06 CI95 [−2.21e-06, −1.83e-06], 100/100 slots
closer) — the registered −0.05 gate passes at 3× the bar**, and
patched lands 0.020 BELOW the no_clutter removal ceiling 0.576
(75/100 closer, CI-excl-0): real-looking clutter beats clutter-free
plates, as the real reference (clutter present in 15–77% of
episodes) predicts. Integrity: in-run v3 0.7127 inside the abort
band, no_clutter 0.5764 reproduces leg (a) within the registered
±0.01, hooked-v3 bit-exact all 100 slots, clean anchor 0.283
unchanged. Promotion of the patch paste into production v3/v4 is an
owner call (asked in-channel 05:40Z); the remaining ceiling to
real-fg 0.328 is the arm's ~7% of pixels — a separate future item.
Renders CPU (~4 min), embeds 5 groups ~0.02 GPU-h.

- [analysis JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_fg_appearance_fix.json)
  · [chart](https://mcobzarenco-fontaine-reports.static.hf.space/chart__sim_fg_appearance_fix.png)
- [v3 vs patched strip](https://mcobzarenco-fontaine-reports.static.hf.space/fg_fix_v3_vs_patched_strip.png)
  — same slots, gray stand-ins vs pasted real crops
- [crops strip](https://mcobzarenco-fontaine-reports.static.hf.space/fg_fix_crops_strip.png)
  — mined RGBA crops: on checker / source naive median / identity paste

## Wrist-view read of the arm material fixes — wrist-neutral: the two-flag stack moves ~230 raw px and the CI straddles zero ([pre-reg](posts/2026-08-14-prereg-sim-wrist-view-material-read.md), 08-14)

The wrist-side fact the two pending promotion asks (photometrics +
mount) assumed rather than measured. Both flags are model-level
material writes, so the wrist camera — inches from the recolored
surfaces, its frame a RAW render (no composite) — sees them directly.
Two paired production instances, 20 seeds × 5 draws, settled resets,
er_60k knn5 probe, both cameras; gates all green (in-run TOP 0.713
dead-center; WRIST 0.561 in the registered [0.50, 0.60] reset band;
qpos bit-equal ×100; changed-px tripwire quiet at 0.56% max).
**PRIMARY: paired wrist Δknn5 −1.39e-08, CI95 [−4.53, +1.73]e-08
straddles zero (46/100) — wrist-neutral**; AUROC 0.561 → 0.560. The
mechanism is visibility: at the home pose the wrist camera sees ~230
raw px of graded surface (servo 208 / PLA 21 / mount 1), so there is
nearly nothing for the encoder to read — no regression (the texture
failure mode did not fire), no gain. The top rider **replicated the
mount read's combo delta bit-for-bit** (−1.4937e-07, CI [−2.451,
−0.570]e-07, 0.713 → 0.702) — production `reset()` observations and
the `_composite` hook path produce identical frames: the hook was
bit-exact. Registered limitation stands: the 0.828 ROLLOUT-pose wrist
gap (gripper filling the frame mid-manipulation) is a different,
still-open fact — needs banked trajectories or fresh rollouts, priced
separately. Renders CPU (~9 min), embeds 8 groups ~0.02 GPU-h.

- [analysis JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_wrist_material_read.json)
  · [chart](https://mcobzarenco-fontaine-reports.static.hf.space/chart__wrist_material_read.png)
- [frame strip](https://mcobzarenco-fontaine-reports.static.hf.space/strip__wrist_material_read.png)
  — v3 / stack / amplified-Δ (near-black) / real wrist

## Arm micro-texture — a clean negative: statistically-matched grain reads MORE fake, both registered CIs above zero ([pre-reg](posts/2026-08-14-prereg-sim-arm-texture-followup.md), 08-14)

The registered residual branch of the photometric close, executed and
decisively refuted — the cheap kind of negative result. The graded arm
is locally FLAT vs real (PLA print-layer local contrast 8.36 vs 4.66;
servo glint tail p97 205.6 vs 125.2), so a composite-stage micro-texture
(opt-in `arm_texture="v1"`, deterministic static fields from a private
pinned RNG, zero shared-stream draws, applied under seg masks before
the production remap/blur/noise; 6 test oracles + init checks) was
fitted THROUGH the composite to the mined real statistics: PLA local
contrast landed **8.24 vs real 8.36**, servo 10.46 vs 9.22, glint tail
~20% closed, photometric guard loss improved on both populations. The
registered 20×5 read, all gates green (in-run v3_photo 0.698
dead-center, anchors exact): **PRIMARY v3_tex vs v3_photo +9.33e-07
CI95 [+8.27, +10.42]e-07 entirely ABOVE zero, 3/100 closer, AUROC
0.698 → 0.751; MECHANISM only_links_tex +1.30e-06 CI95 [+1.22,
+1.38]e-06, 0/100 closer, 0.652 → 0.740** — the texture undoes most of
the grade's gain. Reading: the pooled per-pixel statistics moved toward
real while the encoder moved away — **the probe sees spatial structure,
not marginal statistics**; screen-fixed band-limited grain reads as
blotchy mottling (the zoom strip shows it), not as anisotropic,
surface-tracking, shading-coupled print ridges. Composite-stage
stats-matching is the wrong instrument class for texture; the branch
dies in one session at ~0.02 GPU-h. Disposition per the frozen rule:
no promotion ask; `sim-arm-surface-texture-mjspec` (true UV-mapped
surface texture via the recompile path, physics-preservation oracles
as its bar) queued as the escalation, not auto-run; the photometric
grade (0.698/0.652) remains the arm-appearance frontier.

- [analysis JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_arm_texture_read.json)
  · [chart](https://mcobzarenco-fontaine-reports.static.hf.space/chart__arm_texture_read.png)
  · [fit record](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__arm_texture_fit.json)
- [frame strip](https://mcobzarenco-fontaine-reports.static.hf.space/strip__arm_texture_read.png)
  — v3_photo / v3_tex, three slots
- [arm zoom 2×](https://mcobzarenco-fontaine-reports.static.hf.space/zoom__arm_texture_read.png)
  — the mottling the encoder flagged, side by side with the smooth grade

## Arm SURFACE texture (mjSpec) — the SECOND refutation: true surface-tracking bands still read MORE fake ([pre-reg + results](posts/2026-08-14-prereg-sim-arm-surface-texture-mjspec.md), 08-14)

The micro-texture refutation's registered escalation, executed and
refuted in one session. `arm_texture="v2"` bakes a quasi-periodic
layer-line texture INTO the 18 PLA link materials via an mjSpec
recompile — bands live in OBJECT space and track the surface, the
exact property the first refutation demanded. Physics hard bar 11/11
oracles green (every model field bit-equal, qpos bit-equal incl. a
60-tick excursion); zero-clip tanh generator with grade-preserving
mean compensation; registered reflection rider (the texture
legitimately shows in the tabletop's 0.02-reflectance mirror of the
arm — and is then fully absorbed by the PSF blur: composited max |Δ|
0). Fit honesty: period 32 frozen at the plausibility bound (lc
response monotonic — fine bands die in the blur chain), amplitude
CAPPED at the 0.42 no-clip headroom → realized PLA local contrast
**6.43 of real 8.36** (grade-only 4.66): the albedo-modulation channel
closes ~41% of the quadrature gap and cannot close the rest. The
registered 20×5 read, all gates green (in-run v3_photo 0.698
dead-center): **PRIMARY v3_surf vs v3_photo +3.07e-07 CI95 [+2.42,
+3.71]e-07 entirely ABOVE zero, 14/100 closer, AUROC 0.698 → 0.718;
MECHANISM only_links_surf +1.98e-07 CI95 [+1.36, +2.59]e-07, 27/100,
0.652 → 0.671** — about a third of the micro-texture's harm, but
confidently fake-ward. Coherence was NOT the missing ingredient.
Diagnostics: the cube shrink-wrap renders sunburst fans on several
faces (not clean layers), and the bands are pure albedo modulation
while real print layers are RELIEF — shading/specular structure the
classic renderer cannot express without a normal-map path. The
arm-texture direction is COLD at this abstraction level; the graded
arm (0.698/0.652) stays the production frontier; no further texture
rung auto-queued.

- [analysis JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_arm_surface_texture_read.json)
  · [chart](https://mcobzarenco-fontaine-reports.static.hf.space/chart__arm_surface_texture_read.png)
  · [fit record](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__arm_surface_texture_fit.json)
- [frame strip](https://mcobzarenco-fontaine-reports.static.hf.space/strip__arm_surface_texture_read.png)
  — v3_photo / v3_surf / amplified-Δ / link zooms (the sunburst fans)

## Camera-mount material split — mechanism lands (93/100), whole-frame null: the part is fixed but too small to move the frame read ([pre-reg](posts/2026-08-14-prereg-sim-mount-material-split.md), 08-14)

The arm-split's per-pixel worst offender, measured and fixed — with a
split verdict the pre-reg's decision rule adjudicates cleanly. The
mount (the wrist camera's white 3D-printed bracket) shared a material
with a black gripper piece; the fix first made the material
mount-exclusive via a **byte-identical detach** (the gripper geom
drops to `matid=-1` with the color copied — the shipped material
carries exactly mjv's material-less defaults; oracle-pinned), then
mined the real bracket at recorded poses. The white part can't
darkness-snap, so its mask **rode the dark gripper/wrist per-body
locks** plus a brightness guard: 81/156 frames, 91k px — the real
mount region reads **neutral light gray [123, 120, 125], luma p50 121**
vs the recolor-black composite's 55. Fit through the production
composite chose **the same specular ceiling as both link populations**
(spec 1.0, shin 0.1; albedo 0.455/0.430/0.431), loss 177188 → 9028,
composited medians dead-on real. The registered 20×5 read (in-run v3
0.713 dead-center; bridges reproduce the arm-split anchors exactly):
**MECHANISM PASSES decisively — only_mount_v1 −1.03e-06 CI95 [−1.16,
−0.90]e-06, AUROC 0.821 → 0.793, 93/100 closer**, and against the bare
plate the graded mount reads −2.67e-06 with **100/100 closer** — with
the right color, mount *presence* now beats absence (the no_mount
amputation confound, reversed). But **PRIMARY FAILS — v3_mount vs v3
CI95 [−0.07, +1.42]e-07 includes zero**, 45/100, AUROC 0.713 → 0.713:
at ~0.66% of pixels the fixed part is below the whole-frame read's
detection floor. Per the frozen rule: no promotion ask for the mount
flag alone. Record-only rider: the **two-flag stack** (mount +
photometrics, what the pending promotion asks would flip together)
reads **0.713 → 0.702, CI95 [−2.45, −0.57]e-07 entirely below zero**
(61/100) — the photometrics carries it; the mount flag rides at zero
measured frame-level cost if the owner flips both. Amendment 1 logged
pre-read: the locality oracle's bit-equality was amended to a bound —
the tabletop's 0.02 reflectance mirrors any arm color change (≤24 px,
≤5 counts measured across all 200 oracle slots vs the 3000 px /
6 count bound). Renders CPU (3 sequential instances), embeds 8 groups
~0.02 GPU-h on the R1-A-freed GPU.

- [analysis JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_mount_material_read.json)
  · [chart](https://mcobzarenco-fontaine-reports.static.hf.space/chart__mount_material_read.png)
  · [mine](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__mount_material_mine.json)
  · [fit](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__mount_material_fit.json)
- [frame strip](https://mcobzarenco-fontaine-reports.static.hf.space/strip__mount_material_read.png)
  — v3 / v3_mount / v3_full_fix and only_mount / only_mount_v1, same slot
- [mining overlay](https://mcobzarenco-fontaine-reports.static.hf.space/overlay__mount_material_mine_ep000_f0635.png)
  — the mount mask (blue) riding the gripper-cluster lock (red) on a real frame

## Arm link photometrics — a measured material grade lands, both registered CIs below zero ([pre-reg](posts/2026-08-14-prereg-sim-arm-photometric-links.md), 08-14)

The execution of the arm-split verdict. Instead of guessing a better
arm color, the real arm's pixels were MEASURED: the sim posed at the
recorded joints of 142 real v2 frames, its silhouette projected
through the production fisheye onto them (per-body FFT darkness-snap
±60 px absorbs the tens-of-px registration offset; ring + absolute
darkness guards, wrist excluded for its dark distractors), pooling
436k printed-PLA and 77k servo-casing pixels. The real black
hardware is brighter than the flat recolor (median luma 66 vs 54),
cool-cast [60, 66, 83], and 16–18% glints — sim rendered 5%/0%.
**The missing term was shine, not paint.** Albedo solved per channel
through the production composite, specular × shininess by grid: both
populations chose the specular ceiling (spec 1.0, shin 0.1); fit
loss ↓8.5× (PLA) / 2.3× (servo). Landed as opt-in
`arm_photometrics="v1"` (default byte-identical, zero RNG draws,
5 oracles). The registered 20×5 read, all gates green (in-run v3
0.713 dead-center): **PRIMARY v3_photo −2.22e-07 CI95 [−3.08e-07,
−1.38e-07] entirely below 0, AUROC 0.713 → 0.698 (72/100 closer);
MECHANISM only_links_photo −7.37e-07 CI95 [−8.35, −6.42]e-07, 0.705
→ 0.652 (96/100)** — the graded links alone now match the no_mount
amputation best (0.654) without removing anything. Residuals for the
registered texture follow-up: print-layer local contrast (real 8.4
vs graded 4.7) and the servo glint tail (p97 206 vs 125). Rider
finding: the camera mount is WHITE in reality, black in sim, and its
material is shared with the gripper wrist-roll piece — a mount fix
needs a material split first. Production-default promotion pends the
owner go. Renders CPU (~9 min, two sequential instances), embeds 7
groups ~0.02 GPU-h alongside R1-A.

- [analysis JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_arm_photometric_read.json)
  · [chart](https://mcobzarenco-fontaine-reports.static.hf.space/chart__arm_photometric_read.png)
  · [mine](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__arm_photometric_mine.json)
  · [fit](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__arm_photometric_fit.json)
- [before/after strip](https://mcobzarenco-fontaine-reports.static.hf.space/strip__arm_photometric_read.png)
  — v3 / v3_photo and only_links / only_links_photo, same slot
- [mining overlay](https://mcobzarenco-fontaine-reports.static.hf.space/overlay__arm_photometric_mine_ep000_f0303.png)
  — snapped per-body masks on a real frame (green PLA, red servo)

## Arm sub-part split — the links carry 88% of the arm's signature; mounts are the per-pixel worst ([pre-reg](posts/2026-08-13-prereg-sim-arm-split.md), 08-13)

The arm-class follow-on to the content split: the rendered arm
(~7.1% of pixels) is the biggest remaining rendered class after the
clutter patches (patched 0.556 ≫ real-fg 0.328), so WHICH arm
sub-part carries it? Same hooked harness — one production v3
instance, `_composite` re-run per segmentation subset with RNG-state
restore, 20 seeds × 5 draws — over two exact partitions of the 96
arm-class geoms: gripper+jaw (46 geoms, 0.3% px) / links base→wrist
(44, 6.1%) / camera mounts (6, 0.7%), and follower (48, 3.6%) /
leader (48, 3.5%). All gates green: in-run v3 0.713 (band ±0.005),
bridges plate_only 0.866 / only_arm 0.654 / no_arm 0.825 all inside
their registered ±0.02. **The registered rule names LINKS**: 88% of
the whole arm's keep-only paired delta (only_links −4.63e-06 of
−5.26e-06, CI-excl-0; only_links alone reads 0.705 vs plate-only
0.866 — nearly the full v3 0.713). Gripper 26% and mount 31% sit
below both the 60% and 35% thresholds. The instance axis is
sub-additive — only_follower −4.05e-06 and only_leader −4.14e-06
each carry ~77–79% alone — so the encoder saturates on either
instance and **a fix must treat both arms**. Record-only but
striking: `no_mount` is the ONLY removal that moves v3 TOWARD real
(0.713 → 0.654, 97/100 frames closer, CI-excl-0) despite the
absence-OOD confound that makes no_arm read +0.113 WORSE — the six
camera-mount geoms are per-pixel the most sim-distinctive thing in
the frame, a cheap rider for the photometric rung. Follow-on queued:
`sim-arm-photometric-links` (links material fix, both instances,
mount-retexture rider). Renders CPU (~7 min), embeds 16 groups
~0.03 GPU-h.

- [analysis JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_arm_split.json)
  · [chart](https://mcobzarenco-fontaine-reports.static.hf.space/chart__sim_arm_split.png)
- [frame strip](https://mcobzarenco-fontaine-reports.static.hf.space/strip__sim_arm_split.png)
  — v3 / only_links / only_gripper / only_mount / plate-only, same slot

## 20-seed behavioral spot-check under v3 ([pre-reg](posts/2026-08-12-prereg-sim-spot20-v3.md), [results](posts/2026-08-12-sim-spot20-v3-results.md), 08-12)

Same 20 seeds, physics bit-identical (spawn rows byte-matched
in-run), only the rendering changed v0 -> v3. **teacher80k improves
+0.97 cm paired [CI95 +0.16, +1.81] — the only CI-excludes-zero
read, direction flipped toward the disk**; er60k (-0.07) and
snap30k (+0.06) null. Visual familiarity moves the arm that
engages. Amendment: GPU compositor (owner-approved) — 371 -> 94
ms/tick, probe reads preserved (0.669/0.113/0.544). ~1.3 GPU-h
(gate 3).

- [reads JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__spot20_v3_reads.json)
  · [per-seed delta chart](https://mcobzarenco-fontaine-reports.static.hf.space/chart__spot20_v3_deltas.png)
  · [GPU-path probe re-read](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_encoder_ood_probe_v3_gpu.json)

## Sim content diversity v3 — plate bank + clutter draws ([pre-reg](posts/2026-08-12-prereg-sim-content-diversity.md), [results](posts/2026-08-12-sim-content-diversity-results.md), 08-12)

Per-reset content variation for the v2 composite: a bank of 26
per-episode clean plates (each carrying its real episode's lighting;
ghost-free by inlier-median mining) + clutter presence/pose draws
from the measured real between-episode spread. Registered bar (top
k std/mean ≥ 0.15 AND 5-NN AUROC ≤ 0.790) **MISSED on the spread
leg** (0.038 → 0.114) while the AUROC leg fell 0.773 → **0.673**
(k-ratio 1.02× — top composites inside the real spread, best
top-cam read yet). Wrist bit-identical to v2 (0.548). Default stays
v2 per the registered flip rule; flip put to the owner. Record-only:
the real disk wanders 8–29 cm × ±19 cm between episodes. ~0.08
GPU-h (gate 0.3).

- [v3 primary](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_encoder_ood_probe_v3_content.json)
  · [homogeneity 20×5](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_encoder_ood_probe_v3_homog.json)
- [k-distance strips](https://mcobzarenco-fontaine-reports.static.hf.space/chart__sim_content_diversity_kdist.png)
  · [REAL | v2 | v3 gallery](https://mcobzarenco-fontaine-reports.static.hf.space/chart__sim_content_diversity_top_gallery.png)

## Sim wrist-cam periphery re-tune ([pre-reg](posts/2026-08-12-prereg-sim-wrist-periphery.md), [results](posts/2026-08-12-sim-wrist-periphery-results.md), 08-12)

One runtime pose change in `_repose_wrist_cam`: camera moved from
the wrist top behind the gripper to over the jaw base (≈10 cm
forward, 55°→65° down) — under the 72° fisheye source the old pose
filled the bottom ~40% of frame with gripper-body mass the real
camera never sees. Registered bar (wrist 5-NN AUROC ≤ 0.786)
**smashed on the first candidate**: 0.900 → **0.548**, k-ratio
0.97× — sim wrist frames sit inside the real embedding spread.
Guard green (top 0.773 bit-identical); 20×5 sensitivity 0.550.
Per-episode wrist-plate axis retired. ~0.04 GPU-h (gate 0.2).

- [primary 100-seed](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_wrist_periphery_fix.json)
  · [sensitivity 20×5](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_wrist_periphery_sensitivity.json)
- [REAL | old | new gallery](https://mcobzarenco-fontaine-reports.static.hf.space/chart__sim_wrist_periphery_before_after.png)

## Sim visual matching v2 — real-frame inpainting ([pre-reg](posts/2026-08-12-prereg-sim-visual-inpainting.md), [results](posts/2026-08-12-sim-visual-inpainting-results.md), 08-12)

Real clean plates (per-pixel median over the 26 reference-half
episodes; A/B pixel-disjointness verified in video-frame indices)
composited under segmentation-masked rendered dynamic content (arms,
benchy, disk + on-table clutter whose real twins move between
episodes). Registered bar (top-cam 5-NN AUROC ≤ 0.790) **MET**:
0.890 (v0) → 0.876 (v1) → **0.773**; overfit tripwire clear. Wrist
composite regressed (0.951 vs 0.900 — the plate is cross-episode
mush) so the shipped `render_style="v2"` (new default) keeps the v1
wrist path. Homogeneity unchanged (~4% vs 45% k std/mean) — content
variation stays the diversity lever. ~0.06 GPU-h (gate 0.3).

- [v2 primary](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_encoder_ood_probe_v2_inpaint.json)
  · [homogeneity 20×5](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_encoder_ood_probe_v2_homog.json)
  · [shipped config](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_encoder_ood_probe_v2_shipped.json)
- REAL | v1 | v2 galleries:
  [top](https://mcobzarenco-fontaine-reports.static.hf.space/chart__sim_visual_inpaint_top_before_after.png)
  · [wrist](https://mcobzarenco-fontaine-reports.static.hf.space/chart__sim_visual_inpaint_wrist_before_after.png)

## Sim visual matching v1 — appearance pass + probe re-reads ([pre-reg](posts/2026-08-12-prereg-sim-visual-matching.md), [results](posts/2026-08-12-sim-visual-matching-results.md), 08-12)

Scene rebuild (real table texture, clutter layout, wrist-cam re-pose,
fisheye remap, color grade, sensor emulation, per-reset appearance
jitter) shipped as `render_style="v1"`; physics oracle-pinned
bit-identical. Registered bar (top-cam 5-NN AUROC 0.890 → ≤0.790 on
the reset-render probe) **missed**: best 0.874, final 0.876. Wrist
responded to the camera re-pose (0.835 → 0.786 scene-only) then
regressed under fisheye+grade (0.900). Sim stays ~10× too homogeneous
at the encoder; lighting jitter moves per-seed distance only ~3%.
Named next lever: real-frame inpainting (SIMPLER-RT style).

- [v0-render baseline](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_visual_match_v0render.json)
  · [scene](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_visual_match_v1scene.json)
  · [fisheye](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_visual_match_v1fisheye.json)
  · [grade](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_visual_match_v1grade.json)
  · [sensor](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_visual_match_v1sensor.json)
  · [sensitivity 20×5](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_visual_match_v1sensitivity.json)
- before/after composites:
  [top](https://mcobzarenco-fontaine-reports.static.hf.space/chart__sim_visual_match_top_before_after.png)
  · [wrist](https://mcobzarenco-fontaine-reports.static.hf.space/chart__sim_visual_match_wrist_before_after.png)

## Encoder OOD probe — sim-vs-real at the policy's eyes (rides the [sim100 pre-reg](posts/2026-08-11-prereg-sim-policy-eval-100seeds.md), owner ask 01:11Z 08-12)

Sim frames (banked er60k-arm rollouts) vs real rig frames through the
frozen er_60k vision trunk, per camera. Measured gap: top-cam 5-NN
AUROC 0.885 / gap ratio 1.54× (wrist 0.828 / 1.33×); the clean-repo
control lands inside the real spread (AUROC 0.26) so the shift is
sim-specific. Sim is at the edge of the real manifold, not off it —
the baseline the visual-matching lever must move.

- [frozen analysis JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_encoder_ood_probe.json)
  (`sim_encoder_ood_probe.py`: pinned frame selection, centroid-cosine
  primary + 5-NN secondary, AUROC/gap-ratio reads, per-frame distances)
- [distance strip chart](https://mcobzarenco-fontaine-reports.static.hf.space/chart__sim_encoder_ood_probe.png)
  — per camera × metric, three groups; sim's tight blob at the real
  distribution's right tail is the whole story in one look

## Grasp-SFT route C `fontaine_grasp_sft_joint_corrected` @2000 ([amendment](posts/2026-08-16-amendment-grasp-sft-route-c-joint.md) · [chain page](posts/2026-08-15-grasp-sft-chain-results.md))

- [flow-head unseen-100 eval report](https://mcobzarenco-fontaine-reports.static.hf.space/eval__grasp_sft_joint_step2000__flow_unseen100.html)
  — owner request 08:25Z 08-16: **44/100 successes** on unseen seeds
  0–99 (euler-10) vs base 9 / corrupt-table stage-C 28 — A §5 verdict
  **TABLE_FIX_POSITIVE** (44 > 28+3, overlap band moot); anchor bar,
  per-seed spawn→final strip, 4-clip gallery, full table; rendered
  08:5xZ 08-16 from the banked leg json
- Remaining probe legs (flow-train memorization read, token-unseen vs
  R2 bar ≥20, token-base anchor) land ~12:3xZ 08-16; consolidated
  verdicts JSON `analysis__grasp_sft_joint_probes.json` + report to
  follow
- [standard 256-sample eval report](https://mcobzarenco-fontaine-reports.static.hf.space/eval__grasp_sft_joint_step2000_train256.html)
  ([json](https://mcobzarenco-fontaine-reports.static.hf.space/eval__grasp_sft_joint_step2000_train256.json))
  — owner request 09:06Z 08-16, stage-C train256 protocol reproduced
  (state-copy anchors bitwise 9.3562/9.8678): joint chunk MAE
  **3.24** vs corrupt-table stage-C **12.56** (which sat WORSE than
  state-copy 9.36 — the wrist_roll clamp); ~3.9× tighter fit on the
  same 256 demo frames with the corrected box
- Weights: [`molmoact2_grasp_sft_joint_corrected_step2000`](https://huggingface.co/mcobzarenco/fontaine-checkpoints/tree/main/molmoact2_grasp_sft_joint_corrected_step2000)
  (weights-only, corrected table baked)

## Grasp-SFT v1 `grasp_sft_v1_joint_8xa100` @3000 ([results](posts/2026-08-16-grasp-sft-v1-results.md) · [flow isolation](posts/2026-08-17-sft-v1-flow-isolation.md) · [drift saga](posts/2026-08-17-sft-drift-saga.md))

- [3-leg sim100 chain panel](https://mcobzarenco-fontaine-reports.static.hf.space/eval__grasp_sft_v1__sim100_chain.html)
  — the chain that dated the collapse and separated the heads
  (14:17:56Z 08-17, ~6.2/12 GPU-h): step500 flow **4/100** / step500
  token **16/100** / endpoint token under the serving fix `b779ba4`
  **14/100** vs probe flow 44 and endpoint flow 5 — token ~flat
  across training while flow never leaves the floor ⇒ the mis-fit
  normalization table poisons the flow targets, not the shared
  trunk; anchors bar, head-asymmetry slopegraph, per-seed strips,
  combined table, 9-clip gallery
- [frozen chain summary JSON](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sft_v1_chain.json)
  (`sft_v1_chain_report.py` — headline numbers reproduce from the
  banked leg JSONs)
- [endpoint flow-head unseen-100 report](https://mcobzarenco-fontaine-reports.static.hf.space/eval__grasp_sft_v1_step3000__flow_unseen100.html)
  — **5/100** vs probe 44 (per-seed data log-reconstructed after the
  box wipe; see the results page's integrity note)
- Weights: [`grasp_sft_v1_joint_step3000`](https://huggingface.co/mcobzarenco/fontaine-checkpoints/tree/main/grasp_sft_v1_joint_step3000)
  (weights-only, byte-verified post-upload)

## Grasp-SFT v2 drift discriminator `grasp_sft_v2_demosonly_1gpu_disc` @1000 ([pre-reg](posts/2026-08-17-prereg-sft-drift-discriminator.md) · [verdict](posts/2026-08-18-sft-drift-discriminator-verdict.md))

The first non-drifting v2-corpus checkpoint (verdict HEALTHY 00:42Z
08-18 ⇒ distributed path convicted), panel-reported per the standing
HTML-reports rule.

- [browsable HTML eval report](https://mcobzarenco-fontaine-reports.static.hf.space/eval__grasp_sft_v2_demosonly_1gpu_disc__step_001000__demos_holdout256_euler10.html)
  ([json](https://mcobzarenco-fontaine-reports.static.hf.space/eval__grasp_sft_v2_demosonly_1gpu_disc__step_001000__demos_holdout256_euler10.json))
  — current-stack eval on the probe-matched pins (demos holdout 0.1 /
  split-seed 0 / 256 samples seed 0 / chunk 30 / euler-10 / batch 12),
  32 charted frames: chunk MAE **5.763** vs state-copy 7.671 (paired
  −1.95); reproduces the old-stack parity read 5.7626 to 3 decimals —
  the in-train probe's 5.8989 is the known ×1.024 probe-vs-eval
  instrument shift from the [verdict post](posts/2026-08-18-sft-drift-discriminator-verdict.md).
  wrist_roll 12.31 stays the worst motor (3.5× state-copy's 3.99),
  the residue the `--per-dataset-flow-norm` rerun targets
- Weights: [`grasp_sft_v2_demosonly_1gpu_disc`](https://huggingface.co/mcobzarenco/fontaine-checkpoints/tree/main/grasp_sft_v2_demosonly_1gpu_disc)
  (steps 500 + 1000, weights-only, banked 00:5xZ 08-18)

## Cross-family analyses

- [flow-vs-AR paired per-step read](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__flow_vs_ar_paired_k4l2.json)
  ([post](posts/2026-08-05-flow-vs-ar-paired.md))

New reports land here as their evals finish; if a number in a post
has no link yet, its report predates this page — ask and it gets
pushed.
