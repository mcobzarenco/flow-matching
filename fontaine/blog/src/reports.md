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

## Cross-family analyses

- [flow-vs-AR paired per-step read](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__flow_vs_ar_paired_k4l2.json)
  ([post](posts/2026-08-05-flow-vs-ar-paired.md))

New reports land here as their evals finish; if a number in a post
has no link yet, its report predates this page — ask and it gets
pushed.
