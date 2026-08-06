# Eval reports

Every panel eval dumps a self-contained HTML report (headline tables,
per-repo breakdowns, worst-frame galleries) plus a JSON that the
frozen results instruments consume. The HTML reports and the frozen
analysis JSONs are hosted on this Space under `/reports/`; this page
indexes them. Posts link the specific reports behind their numbers.

Naming: `eval__<run>__<checkpoint>__<panel+sampler>` — `heun30` =
Heun 30-step, `1nfe_euler1` = single Euler step (1 expert eval),
`drawsN` = mean-of-N ensembling, `stable`/`stablekey` = stable noise
keying ([#18.2](posts/2026-08-05-noise-reseed-prereg.md)), unmarked =
legacy index keying. `panel_curated_v0_k4l2` is the v1 25,800-frame
panel; `panel_v2` is the [dedup-hardened
revision](posts/2026-08-06-panel-v2-amendment.md).

## SnapFlow 1-NFE distillation ([results](posts/2026-08-06-snapflow-results.md))

- [student 1-NFE, single draw (primary)](https://mcobzarenco-fontaine-blog.static.hf.space/reports/eval__fontaine_flow_snapdistill_h1024_30k_1xh100__step_030000__panel_curated_v0_k4l2_1nfe_euler1.html)
- [student 1-NFE, mean-of-5](https://mcobzarenco-fontaine-blog.static.hf.space/reports/eval__fontaine_flow_snapdistill_h1024_30k_1xh100__step_030000__panel_curated_v0_k4l2_1nfe_euler1_draws5.html)
- [student 1-NFE, mean-of-10 (deployment headline)](https://mcobzarenco-fontaine-blog.static.hf.space/reports/eval__fontaine_flow_snapdistill_h1024_30k_1xh100__step_030000__panel_curated_v0_k4l2_1nfe_euler1_draws10.html)
- [frozen analysis JSON](https://mcobzarenco-fontaine-blog.static.hf.space/reports/analysis__snapflow_distill_30k_k4l2.json)
  (`snapflow_results.py`, pre-registered reads)

## Flow teacher `bijou_flow_artrunk_h1024_40k_ddp2` @80k

- [Heun-30, single draw (v1 anchor)](https://mcobzarenco-fontaine-blog.static.hf.space/reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_curated_v0_k4l2_draws1_heun30.html)
- [Heun-30, mean-of-5](https://mcobzarenco-fontaine-blog.static.hf.space/reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_curated_v0_k4l2_draws5_heun30.html)
- [Heun-30, mean-of-10](https://mcobzarenco-fontaine-blog.static.hf.space/reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_curated_v0_k4l2_draws10_heun30.html)
- [Heun-10, single draw](https://mcobzarenco-fontaine-blog.static.hf.space/reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_curated_v0_k4l2_draws1_heun10.html)
- [Heun-10, mean-of-10](https://mcobzarenco-fontaine-blog.static.hf.space/reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_curated_v0_k4l2_draws10_heun10.html)
- [Heun-30, stable keying (re-banked anchor)](https://mcobzarenco-fontaine-blog.static.hf.space/reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_curated_v0_k4l2_stablekey_heun30.html)
  ([results](posts/2026-08-06-stablekey-rebank-results.md))
- [legacy k4l2 panel, Heun-30](https://mcobzarenco-fontaine-blog.static.hf.space/reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_k4l2_heun30.html)
- [state-masked Q4 probe](https://mcobzarenco-fontaine-blog.static.hf.space/reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__stateprobe_q4_state-masked.html)
  ([results](posts/2026-08-06-state-probe-results.md))
- Draws-fairness frozen reads:
  [analysis](https://mcobzarenco-fontaine-blog.static.hf.space/reports/analysis__draws_fairness_k4l2.json) ·
  [validate](https://mcobzarenco-fontaine-blog.static.hf.space/reports/analysis__draws_fairness_k4l2_validate.json)
  ([results](posts/2026-08-06-draws-fairness-results.md))
- σ_draw:
  [finalization](https://mcobzarenco-fontaine-blog.static.hf.space/reports/analysis__sigma_draw_finalization.json) ·
  [direct measurement](https://mcobzarenco-fontaine-blog.static.hf.space/reports/analysis__sigma_draw_direct.json)
  ([amendment](posts/2026-08-06-sigma-draw-finalization.md))

## Flow teacher @40k (arch-batch control)

- [panel-v2, Heun-30, stable keying](https://mcobzarenco-fontaine-blog.static.hf.space/reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_040000__panel_v2_ctrl_heun30_draws1_stable.html)
  — the [arch batch #1](posts/2026-08-06-prereg-arch-batch-1.md)
  control ·
  [ctrl-only analysis](https://mcobzarenco-fontaine-blog.static.hf.space/reports/analysis__arch_batch_1_ctrl_only.json)

## AR mainline `bijou_arb_rcond_100k_ddp4` @100k

- [curated_v0 panel (anchor)](https://mcobzarenco-fontaine-blog.static.hf.space/reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_curated_v0_k4l2.html)
- [sealed split](https://mcobzarenco-fontaine-blog.static.hf.space/reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_curated_v0_k4l2_sealed.html)
  ([sealed plan](posts/2026-08-05-sealed-plan-v2.md))
- [legacy k4l2 panel](https://mcobzarenco-fontaine-blog.static.hf.space/reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2.html)
- [state-masked Q4 probe](https://mcobzarenco-fontaine-blog.static.hf.space/reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__stateprobe_q4_state-masked.html) ·
  [state-probe analysis](https://mcobzarenco-fontaine-blog.static.hf.space/reports/analysis__state_probe_q4.json)
  ([results](posts/2026-08-06-state-probe-results.md))

## Box batch 40k AR arms ([results](posts/2026-08-06-box-batch-results.md))

- [A-s0 `fontaine_arb_rcond_40k_1xh100` panel](https://mcobzarenco-fontaine-blog.static.hf.space/reports/eval__fontaine_arb_rcond_40k_1xh100__step_040000__panel_curated_v0_k4l2.html) ·
  [state-masked probe](https://mcobzarenco-fontaine-blog.static.hf.space/reports/eval__fontaine_arb_rcond_40k_1xh100__step_040000__stateprobe_q4_state-masked.html)
- [A-s1 seed replicate panel](https://mcobzarenco-fontaine-blog.static.hf.space/reports/eval__fontaine_arb_rcond_40k_1xh100_s1__step_040000__panel_curated_v0_k4l2.html)
- [A-s2 seed replicate panel](https://mcobzarenco-fontaine-blog.static.hf.space/reports/eval__fontaine_arb_rcond_40k_1xh100_s2__step_040000__panel_curated_v0_k4l2.html)
- [aux-off arm state-masked probe](https://mcobzarenco-fontaine-blog.static.hf.space/reports/eval__fontaine_arb_rcond_auxoff_40k_1xh100__step_040000__stateprobe_q4_state-masked.html)
- [batch analysis JSON](https://mcobzarenco-fontaine-blog.static.hf.space/reports/analysis__box_batch_40k_k4l2.json)

## Cross-family analyses

- [flow-vs-AR paired per-step read](https://mcobzarenco-fontaine-blog.static.hf.space/reports/analysis__flow_vs_ar_paired_k4l2.json)
  ([post](posts/2026-08-05-flow-vs-ar-paired.md))

New reports land here as their evals finish; if a number in a post
has no link yet, its report predates this page — ask and it gets
pushed.
