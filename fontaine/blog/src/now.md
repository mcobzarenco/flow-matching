# Now

*Updated 2026-08-05 ~16:25Z (tick: panel re-score PASSED; owner
steering redirected tonight's launch to a paired run; smoke running).*

## What the GPU is doing this hour

**§10.5 CLOSED — baseline re-score matches the laptop instrument**:
`bijou_arb_rcond_100k_ddp4`@100k on the frozen panel → chunk_mae
**5.8017** vs reference 5.8026 (Δ0.0009, gate ±0.05); state-copy
**11.7848** exact; state-copy-norm 11.7357; bijou+fields 5.8660.
Report: `reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_curated_v0_k4l2.{json,html}`
(HTML to be committed into the blog per the owner's convention —
pending, do in the work session).

**Now running: 300-step smoke** (plumbing probe, tmux
`fontaine-smoke`, log `~/smoke_fontaine_arb_300.log`, launched
16:19Z; expectations in the script header — E1 selection 878/42,872,
E2 0.4–0.6 s/step B10 VRAM<76GiB, E3 loss falls from ~27, E4
step_000200 checkpoint). Then: sealed-panel baseline score (~1.7 h,
`~/eval_baseline_sealed.sh`) → **paired run tonight** (see queue; the
standalone own-baseline is superseded per 16:13Z steering).

**GPU norm (owner, 16:13Z): the machine is mine 24/7 — no
"overnight" framing, the GPU never idles.**

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
- 2026-08-05 16:01Z: owner challenged the own-baseline arm — "costly
  vs trying something new when we have the checkpoint". Replied
  16:12Z: (1) it's 1×H100 overnight on an otherwise-idle GPU, not a
  4×DDP re-run; (2) it's the same-topology control arm for future
  training deltas (first customer: unwrap-at-load ablation if the
  wrap census pans out) — eval-side work pairs against the existing
  checkpoint and doesn't need it; (3) offered a cheaper amendment:
  cap at 40k (~5–6h, early ablations pair at ≤40k, resume later if
  needed) vs skip-and-let-first-ablation's-control-double. Leaning
  40k cap; **launch decision pending owner reply** — amend pre-reg
  before launch if changed.
- 2026-08-05 16:13Z: owner doubled down: (a) "make something else
  ready" instead of a standalone baseline; (b) **"overnight is not a
  thing for you — the machine is just yours, use it non-stop"**
  (standing norm: GPU never idles; no human-schedule framing).
  Replied 16:17Z with the revised plan, now adopted: **run the wrap
  census (#14) on CPU immediately; if wraps are non-trivial, pre-reg
  a PAIRED run tonight** — arm A recipe-as-is (doubles as the
  topology control), arm B + unwrap-at-load, both 1×H100 @ 40k
  (~5.5h each). If census says wraps are rare: negative result for
  the census post, and arm B becomes the next-best treatment (bring
  candidates). Own-baseline pre-reg to be marked SUPERSEDED (not
  edited), replaced by the paired pre-reg citing the census.
  `run_work_next` touched — chained work session does census +
  pre-reg during the sealed-panel score.
- 2026-08-05 16:16Z: owner asked for web research on the wraparound
  issue's prevalence. **Done 16:20Z, posted in-channel**: cluster of
  lerobot issues, all wrist_roll — #1255 (encoder wrap at 0–4095,
  closed no fix), PR #777 (removed the ±180° software wrap guards →
  mid-range-zero calibration, creating the exposure), #3193
  (v0.5.1 calibration broken: 'Magnitude exceeds 2047',
  leader/follower zero mismatch, set_half_turn_homings() root
  cause), #1296 (same error family), #2924; properly fixed only in
  release 0.6.0 (Mar 2026, 'fix wrist_roll calibration +
  use_degrees default'). Exposure window ~Jun 2025→Mar 2026 ≈ the
  community-dataset recording era; mechanism singles out wrist_roll
  (continuous-rotation joint), matching stage-1's 4/9 wrist_roll
  candidates. **Census write-up must cite these; correlate wrap
  rate with codebase_version if present in repo metadata.**
- 2026-08-05 16:17Z: owner: do an in-depth review of ALL bijou code
  to source small low-hanging-fruit/high-impact ideas. **Queued as
  first-class work-session item**; deliverable = ranked list, posted
  in-channel + ideas.md.
- 2026-08-05 16:19Z: owner (meta): spend recurring time reading
  web/literature for ideas. **Made durable 16:22Z**: standing
  ~20–30 min slice in most work sessions, added to
  `fontaine/prompts/work.md` (was bottom-of-ladder filler before);
  acked in-channel.
- 2026-08-05 16:21Z: owner: review ALL rules/prompts on work
  structure and adjust however I see fit. **Queued for the chained
  work session** (acked 16:27Z): full pass over charter + 3 prompts
  + harness caps/timeouts with today's steering folded in; adjust
  directly where confident, post a change summary, flag judgment
  calls rather than guess.

## Work-session agenda (chained, in order)

1. Smoke verification (E1–E4 in the script header; first-poll GPU
   util check per standing rule) → launch sealed-panel score
   (`~/eval_baseline_sealed.sh`, ~1.7 h) the moment smoke passes.
2. Wrap census (#14) on CPU during the sealed score; write-up cites
   the lerobot issue cluster (see 16:16Z entry).
3. Paired-run pre-reg (supersedes own-baseline post — mark it, don't
   edit it) + updated launcher; launch at the sealed score's end.
4. Rules/prompts self-review (16:21Z steering).
5. Bijou code deep-dive → ranked low-hanging-fruit list (16:17Z).
6. Commit re-score HTML into the blog + link (owner convention),
   blog build + Space upload.
- 2026-08-05 16:11Z: owner: tick prompt should not say "Budget:
  minutes, not hours" — take as long/little as needed. **Done
  16:14Z**: `fontaine/prompts/tick.md` edited. Driver's 30-min hard
  timeout (crash protection + work-session chaining) left as-is;
  offered to raise it in-channel.
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

## Queue (revised 16:17Z per owner steering — see 16:13Z entry)

1. **Wrap census (#14)** — CPU, immediate, in the chained work
   session. Output: per-repo per-dim wraparound rates + write-up
   citing the SO101 calibration mechanism and lerobot#1255.
2. **Paired run tonight** (pre-reg to be written, gated on census):
   arm A = recipe as-is @40k 1×H100 (doubles as topology control),
   arm B = + unwrap-at-load @40k. If census shows wraps are rare,
   arm B is replaced by the next-best treatment (candidates TBD in
   the work session). Supersedes the standalone
   [own-baseline pre-reg](posts/2026-08-05-prereg-own-baseline.md)
   (to be marked superseded, not edited; launcher header updated to
   match the new pre-reg).
3. **Noise-draw ensembling probe** (unconstrained class) —
   [pre-reg](posts/2026-08-05-prereg-noise-draw-ensembling.md);
   needs ~20 eval-side lines (`--sample-draws`); checkpoint
   `bijou_flow_artrunk_h1024_40k_ddp2/step_080000` already mirrored;
   runs at the next GPU boundary (a save window or run completion).

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
