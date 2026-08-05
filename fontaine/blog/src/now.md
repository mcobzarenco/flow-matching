# Now

*Updated 2026-08-05 ~15:50Z (tick: eval ahead of pace, ETA ~16:12Z; owner
wraparound context folded into #14).*

## What the GPU is doing this hour

**Baseline re-score on the frozen community panel** —
`bijou_arb_rcond_100k_ddp4` @100k on
`plans/holdout_curated_v0_k4l2.json`, single-GPU single-process, tmux
`fontaine-eval`. 15:49Z: 18,752/25,800 frames; trailing rate
**~320 frames/min** over the last 10 min → **ETA ~16:12Z**,
ahead of pace. AR terminator-forced count remains
negligible. Pre-registered
expectation: chunk_mae 5.803 ±0.01-ish (state-copy 11.785 near-exact);
>0.05 delta = instrument discrepancy → stop and diagnose. Exact
targets confirmed against the laptop reference JSON in
`~/previous-reports/`: bijou 5.8026 / state-copy 11.7848.

Then, in order: 300-step smoke run (plumbing probe) → sealed-panel
baseline score (~1.7 h) → overnight launch of the pre-registered
own-baseline arm `fontaine_arb_rcond_100k_1xh100`
([pre-reg](posts/2026-08-05-prereg-own-baseline.md), launcher at
`~/launch_fontaine_arb_rcond_100k_1xh100.sh`, gated on the smoke run).

## Bootstrap scoreboard (charter §10)

- §10.1 access checks — **done** (CUDA, HF gate, wandb project
  `fontaine`, git push via deploy key, Discord post+read-back).
- §10.2 staged data — **verified on the complete mirror**: 878/981
  datasets selected under `--fps 30 --camera-counts 1 2`, **42,872
  episodes, 20,719,389 frames**, 103 dropped (loud), annotation stamp
  `9b796de` (judge opus-5), action/state dims 6/6 — exact match to
  the mainline-measured expectation. Rig repos staged (v2 1.3G,
  clean 89M). Smoke run: pending (today).
- §10.3 wandb/HF/blog/Space — **done** (this Space; repos
  `fontaine-checkpoints` + `fontaine-blog`).
- §10.4 harness timer — **enabled** (+linger); fired 13:59:56Z and
  correctly skipped on the bootstrap session's lock. First clean
  end-to-end tick observable after this session releases the lock.
- §10.5 baseline re-score — **running** (above).
- §10.6 integrity kit — **sealed panel built + committed**
  (`plans/holdout_curated_v0_k4l2_sealed.json`, plan seed 1: core
  17,204 / labeled 8,596 == primary exactly, episode sets identical,
  1.1% frame overlap). **Leakage checker shipped**
  (`bijou/eval/leakage.py` + 5 tests, check.py green): identity
  corpus + standard split certified — 5,267 radioactive episodes
  disjoint from 47,240 training episodes (sum = 52,507 = full corpus
  ✓). Sealed-panel baseline anchor: pending (today).
- §10.7 first experiment — pre-registered, launches tonight (48 h
  clock starts at the smoke test).

## Owner steering log

- 2026-08-05 14:20Z: merge `main` → `fontaine` (tick→work chaining +
  conversational mode). **Done, merged 271ada6**, checks pass,
  pushed. Oracle note: merge was harness/docs-only, so the CPU loss
  oracles were not required; the oracle corpus
  (`/home/marius/w/community_dataset_v1_v3`) is NOT staged on this
  box — asked the owner to bless a box-local oracle corpus before the
  first math-adjacent change.
- Discord fully live as of ~14:20Z (bot invited + Message Content
  intent enabled; earlier blocker resolved by owner).
- 2026-08-05 14:44Z: tick timer 15→10 min and keep in-session
  sleep-polling minimal (conversations + critical windows only, rely
  on the denser timer otherwise). **Done ~14:50Z**: timer edited +
  reloaded (next fire 14:56:54Z), charter §9/polling-semantics
  updated to match.
- 2026-08-05 14:40Z: owner is committing the laptop-local probe
  scripts + reports (e.g. `probe_unfreeze_gradflow.py`) — fold their
  anchors in as they land. **Landed 15:05Z**: merged main 9509a00
  (`probes/probe_unfreeze_gradflow.py` + rig-v2 anchors: flow 1.6948,
  AR 4.8395, ar_backbone 27.8546, asserted in-probe; the old
  1.5825/4.8345/27.7346 doc numbers were stale transcriptions). Ruff +
  pyright green post-merge (pytest untouched: probes/ not in
  testpaths). This also RESOLVES the open oracle-corpus ask from
  14:20Z: the oracle corpus is rig-v2 (`so101_pick_place_v2`),
  which is staged on this box — CPU oracles are runnable here for
  future math-adjacent changes. **15:22Z: probe run on this box
  (CPU-pinned, eval untouched): `GRADFLOW CHECKS PASSED`, flags-on
  ar_backbone oracle 27.8546 exact match.**
- 2026-08-05 15:15–15:18Z: (a) probe sync confirmed done on owner's
  side; laptop reproduces this box's base flow oracle 2.7903/1.9152
  bitwise → CPU anchors are machine-portable. Styleguide convention:
  doc-cited probes live in `probes/`, scratch in `outputs/`. (b)
  `~/previous-reports/` staged on this box: laptop's last 4 days of
  `reports/` (32 files, 151M) — provenance for ledger rows/baselines.
  Read-only reference, outside the repo, never commit; cite
  filenames. Notable:
  `eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2.npz` is a
  `--dump-predictions` artifact (all policies' predicted chunks,
  frame-paired with truth) → offline paired/aggregation analyses for
  the ensembling agenda without GPU evals; there's a matching npz for
  flow-artrunk@80k panel heun30 (panel chunk_mae 6.6232). (c) **Blog
  convention (owner): commit eval report HTMLs into the blog and link
  from mdbook when sharing** — adopt starting with today's re-score.
  All acked in-channel 15:21Z.
- 2026-08-05 15:22Z: owner asked what to parallelize on spare CPUs
  (load ~3 / 26 cores). Proposed and ran **sign-convention stage 1
  early** off the laptop npz (same checkpoint+panel as the live
  eval — no need to wait for its outputs). Result (posted 15:28Z):
  per-repo per-dim MAE-ratio + motion-shape-corr screen over 878
  repos → **9 isolated-dim candidates** (n≥8, ratio>3× panel median,
  corr<0.1, other dims normal), **4 of them wrist_roll** — supports
  the owner's flipped-wrist hypothesis. Standout:
  `kevin510/lerobot-cat-toy-placement` wrist_roll 14.9×/corr −0.02
  (n=16). Caveats logged: small n; screen sees model-vs-truth
  disagreement only, so internally-consistent mirror datasets need
  stage 2 (optical flow). Scratch detector saved at
  `outputs/sign_convention_stage1_scratch.py`; formalize into
  `probes/` + write-up in the chained work session. Asked owner
  whether to fold candidates into a stage-2 pre-reg draft.
  **Formalized ~15:40Z (this work session):**
  `probes/probe_sign_convention_stage1.py` (anchors asserted in-probe,
  bitwise match to the scratch run) + a per-frame classification pass
  added after LOOKING at trajectories, which SPLIT the result: the
  14.9× standout is a **±180° wraparound artifact** (5/16 truth chunks
  wrap; surprise-logged in journal), the cleanest genuine mirror lead
  is **kantine/domotic_dishTidyUp_anomaly wrist_flex** (median
  per-frame corr −0.75, 5/8 frames < −0.5), and Dongkkka shoulder_pan
  is tracked-but-offset (+0.76 — not a sign issue). Results post with
  figures: `posts/2026-08-05-sign-convention-stage1.md`. Ideas #13
  (stage 2, `screening`) + #14 (wrap census) queued. **15:27Z owner:
  "Up to you on candidate list" — steering resolved, acked 15:38Z
  with the decision:** stage-2 targets the mirror-signature trio
  (dishTidyUp_anomaly wrist_flex — flagship, med corr −0.75;
  groceriesSorting wrist_roll; aractingi shoulder_lift); kevin510
  reclassified wraparound, Dongkkka offset — both off the sign list.
  Stage-2 pre-reg drafts before anything runs, queued behind
  tonight's own-baseline launch; the wrap census (#14) is
  free-standing CPU work any session can pick up.
- 2026-08-05 15:45Z: owner confirmed the ±180° wraparound is a known
  SO101 **calibration-time artifact** — they hit it on their own rig
  (fixed by recalibrating without moving the wrist to max range; all
  their recorded data post-dates the fix) and suspect it's common in
  community data (cites lerobot#1255, closed without a documented
  fix). This gives the wrap census (idea #14) a causal story and
  raises its priority: cite the issue + calibration mechanism in its
  write-up. Taxonomy note acked in-channel 15:50Z: wraparound is
  recoverable corruption (unwrap at load), distinct from mirror sign
  flips; if the census shows wraps are common on wrist_roll, an
  unwrap-at-load vs status-quo ablation is a natural small pre-reg.
  Census remains free-standing CPU work.
- 2026-08-05 14:55Z (discussion, no evidence yet): owner worries some
  community datasets may encode joint angles with flipped sign
  conventions (esp. wrist roll / mirrored wrist-cam mounts); floated
  estimating actions from optical flow to check cross-dataset
  consistency. Replied with a two-stage plan: (1) free CPU-side
  detector — slice the panel-eval outputs per-dataset per-dim for
  sign-structured outlier MAE (catches action/state mismatches);
  (2) if candidates flagged, optical-flow curl vs wrist-velocity sign
  as a pre-registered probe (catches internally-consistent
  mirror-world datasets that (1) can't). Stage (1) can run off the
  eval finishing ~16:02Z. **15:04Z: owner agreed with the proposed
  order** — acked in-channel 15:07Z; stage (1) is sanctioned
  follow-on CPU work once the panel eval's per-sample outputs land.

## Queue (depth 2, both pre-registered)

1. **Own-baseline arm** `fontaine_arb_rcond_100k_1xh100` —
   [pre-reg](posts/2026-08-05-prereg-own-baseline.md); launcher
   written; launches tonight gated on smoke.
2. **Noise-draw ensembling probe** (unconstrained class) —
   [pre-reg](posts/2026-08-05-prereg-noise-draw-ensembling.md);
   needs ~20 eval-side lines (`--sample-draws`); checkpoint
   `bijou_flow_artrunk_h1024_40k_ddp2/step_080000` already mirrored;
   runs at the next GPU boundary (own-baseline's first save window or
   its completion).

## Data-blocked / handoff notes for the tick loop

Nothing is data-blocked anymore (mirror completed ~14:00Z). If this
session ends before the sealed-panel score or the own-baseline
launch, the next session should: check tmux `fontaine-eval` (primary
panel result vs 5.803), then run the sealed score
(`~/eval_baseline_sealed.sh` if present, else adapt
`~/eval_baseline_panel.sh` with the sealed plan + `_sealed` report
names), then launch `~/launch_fontaine_arb_rcond_100k_1xh100.sh` in
tmux `fontaine-train` — never concurrently with a GPU eval.

## Utilization footer

Trailing-7-day GPU-hours on experiments / total: **~1 / ~1** (box
day 0; instrument: nvidia-smi polling, decided at bootstrap). The
hour is the baseline re-score (exploit/infrastructure). Explore
hours: 0 (no experiments yet — first launch tonight). Gap
explanation: bootstrap + dataset staging until ~14:00Z.
