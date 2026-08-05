# Now

*Updated 2026-08-05 ~17:50Z (tick: box batch healthy at step
1500–2000 ×4 @ 0.38–0.39 s/step, B-s0 action loss tracks A-s0 within
noise; sealed eval 16.9k/25.8k @ ~315 f/min ⇒ ETA ~18:16Z; owner
17:42Z conversational — 26B-A4B training cost discussed, replied
(MoE saves FLOPs not memory; QLoRA the realistic path; E4B/12B the
practical full-FT trunks).*

## ⚡ The second box (192.222.55.210) — batch RUNNING

Pre-reg: [box batch](posts/2026-08-05-prereg-box-batch-4xh100.md)
(commit cc0b922, posted before launch). Four 1×H100 40k runs launched
17:12Z in per-GPU tmux sessions (`launch_box_gpu{0..3}_*`):

| GPU | run | seed | tmux / log |
|-----|-----|------|------------|
| 0 | A-s0 control | 0 | `~/train_fontaine_arb_rcond_40k_1xh100.log` |
| 1 | B-s0 aux-off | 0 | `~/train_fontaine_arb_rcond_auxoff_40k_1xh100.log` |
| 2 | A-s1 control | 1 | `..._s1.log` |
| 3 | A-s2 control | 2 | `..._s2.log` |

- **E1 hard gate PASSED on all four** (17:15Z): 878 datasets /
  38,571 train + 4,301 holdout = 42,872 episodes / dims 6/6 / 103
  dropped — identical, and B-s0's log carries **no aux line** while
  A's shows fields + weight 0.5. Box data copy verified against local
  (listing diff = inert `provenance/` tarball only).
- **E2 first-poll PASSED (17:18Z, util rule):** all four stepping at
  0.43–0.54 s/step (band 0.4–0.7 — no contention penalty so far),
  VRAM ~64–67 GiB, util 53–94% sampling jitter, loss falling from
  ~21 on all arms; B-s0's step lines carry no `loss_aux`, replicates
  do. wandb runs: `vr8b8hpy` (A-s0), `skdz5ppa` (B-s0), `790g1ccm`
  (s1), `d0xmdcnz` (s2), project `fontaine`.
- Each GPU chains its panel eval (k4l2, `--dump-predictions`) after
  40k. ~5–6.5 h train + ~1.7 h eval ⇒ all reads by ~02Z.
- **Babysit every ~30 min of session time**: liveness + s/step
  (0.4–0.7 healthy, >0.8 sustained = starvation → fix at boundary)
  + probe curve vs anchors (<12 @10k, <9 @30k; B within ±0.3 of A).
  Kill gates in launcher headers; A-s0 killed ⇒ kill B-s0 (pair
  void), replicates continue.
- **rsync-back live**: local tmux `fontaine-rsync`
  (`~/boxsync_loop.sh`, 20-min cadence): logs + eval reports + latest
  two saves per run → `~/boxsync/`.
- **Owner constraint (17:02Z): do NOT delete the box's existing
  fine-tune checkpoints** (owner rsync in flight). No cleanup of any
  kind runs on that box.
- Code on box: branch `fontaine` @ cc0b922 (pushed over direct SSH;
  box `.venv` reused — torch 2.11.0+cu130 both boxes, no seam).

## What the LOCAL GPU is doing this hour

**Sealed-panel baseline score running** (tmux `fontaine-eval`,
launched 16:33Z, ETA ~18:05Z; healthy at 17:03Z: 6,624/25,800,
~305 f/min, util 73%). Expectations in the script header: chunk_mae
within ~0.15 of 5.8017; state-copy ≈ 11.785.

**At sealed-score end — the local paired launch is SUPERSEDED (do
NOT run `~/launch_fontaine_paired_auxoff_40k.sh`; the box batch is
running it).** Instead: (1) bank + post the sealed anchor; (2)
**launch the noise-draw chain**: `tmux new-session -d -s
fontaine-eval-draws 'bash ~/eval_flow80k_draws_panel.sh'` — the
[pre-reg](posts/2026-08-05-prereg-noise-draw-ensembling.md) is fully
verified ready: `--sample-draws` is a16e65a (my 14:31Z commit —
test-gated: draw 0 byte-identical, mean in raw degrees, `_drawsN`
naming), checkpoint rsync'd from the box to
`outputs/train/bijou_flow_artrunk_h1024_40k_ddp2/step_080000`, and
the launcher header carries the E1 instrument gate (N=1 heun-30 must
reproduce 6.6232 ±0.03 or the chain stops). 5 runs ≈ 9 h overnight.
The unimodality probe (per-draw dumps) runs before the results post,
next work session — ordering noted in the launcher header.

## Banked this session (no GPU needed): 80k flow panel number

Queue #3 dissolved — the owner had **already panel-scored flow-80k
on the box today 12:20Z** (heun-30, panel k4l2, with
`--dump-predictions`), alongside a same-day AR-100k panel rerun with
dumps:

- **flow-80k @ heun-30: chunk_mae 6.6232, first_mae 1.9331**
- **AR-100k: chunk_mae 5.8026 (anchor, bitwise), first_mae 2.1431**
- state-copy summaries bitwise-identical across the two reports ⇒
  the npzs pair per-frame. Flow still trails AR by 0.82 pooled but
  **beats it on first_mae** (1.93 vs 2.14, the grounding-sensitive
  column).

All eight files pulled to local `reports/` (17:14Z). Queued CPU
analysis: paired per-frame flow-vs-AR deltas (where does flow win?)
— feeds a results post + the solver/ensembling ideas (#1, #12).

## This work session (17:03Z→) — what happened

1. Read the owner's 17:02Z constraint (keep box fine-tune ckpts) —
   honored: zero deletes on the box.
2. Verified box: 4×H100 idle, creds present (netrc/HF), torch parity,
   dataset copy parity (283 dirs, 600G; local-only `provenance/`
   tarball inert), owner's checkout behind → pushed `fontaine` over
   SSH, checked out cc0b922, imports OK.
3. Wrote + posted the [batch pre-reg](posts/2026-08-05-prereg-box-batch-4xh100.md)
   (execution supersedes the local sequential plan; science of the
   [paired pre-reg](posts/2026-08-05-prereg-paired-auxoff-40k.md)
   unchanged; new E5 = seed-noise floor with pre-registered decision
   rule). Banner added to the paired pre-reg. `check.py` green.
4. Generated 4 per-GPU launchers (diff-verified: replicates differ
   only in GPU/seed/name; B differs only by dropped aux flags),
   launched 17:12Z, E1 gate passed on all four.
5. Discovered the owner's existing flow-80k + AR-100k panel reports
   on the box → banked the numbers above, pulled the npzs.
6. rsync-back loop started (`fontaine-rsync` tmux).

## Bootstrap scoreboard (charter §10)

- §10.1–§10.6 — done (sealed anchor: running, ETA ~18:05Z).
- §10.7 first experiment — **RUNNING** (paired aux-off + replicates
  on the box; 48 h clock started at the smoke test — beaten).

## Owner steering log (active items)

- 17:50–18:01Z (conversational, replied in-channel): **(a) trunk
  survey mandate** — deep review of in-scope open-weights models:
  budget **<7B, ideally ~3B**, video-trained preferred; method per
  owner 18:01Z: read the **arXiv paper** (if any) + HF config per
  candidate, not just model cards. Multi-turn = later-stage research
  area (noted, not started). → queued in the owner-steered reviews
  block (item 5c). **(b) Ministral 3 3B** flagged by owner —
  first-read posted (3.4B LM + 0.4B vision enc, 256k ctx, Apache
  2.0, Dec 2025 = post-cutoff; images only, no video/audio on the
  card; arch details undisclosed → config read needed). Candidate on
  size/license; misses the video-trained preference. **(c) owner
  asked after the rules/prompts + bijou reviews** — answered
  honestly (not done; eaten by box launch + Gemma 4 docs);
  **committed in-channel to a chained work session**
  (`run_work_next` touched 17:58Z) with order: rules/prompts pass →
  bijou deep-dive → trunk survey → literature slice.
- 17:31Z: **research the Gemma 4 lineage** (owner: PLE only on
  E2B/E4B, 12B unified-multimodal no-audio, "MoE I think?"; read
  the HF blog) → **DONE this tick**: blog read, `docs/gemma4.md`
  family section rewritten with all 5 variants (E2B/E4B/12B
  Unified/26B-A4B/31B, params, ctx, modalities). Blog corrections
  posted in-channel: PLE is in E2B/E4B *and* 12B; 12B *does* take
  audio (raw waveforms linearly projected, encoder-free); only
  26B-A4B is MoE (8/128 experts, 4B active). Summary posted 17:41Z.
- 17:26Z: **Gemma 4 is post-cutoff — never reason from Gemma-3
  priors** (I wrote "Gemma-3-class" in ideas #17). → **DONE this
  tick**: `docs/gemma4.md` written (code-derived from
  `bijou/gemma4/`), wake-up memory `gemma4-post-cutoff` installed
  (loaded every session via MEMORY.md), ideas #17 line fixed to
  "larger Gemma-4 variants (E4B/12B)". Also 17:26Z: 👍 on the
  "run only what changes the next decision" rule — no action.
- 17:20–17:23Z: **three big steers, all acted on this session**:
  (1) "You push" the README → **DONE**, dataset-repo commit
  `a9f652f` (known-issues section + pre-removal revision hash
  `250f6ed2c45c…` recorded in it). (2) Remove the census repos from
  the sealed plan → **DONE**:
  `plans/holdout_curated_v0_k4l2_sealed_v2.json` (core −52 frames /
  13 eps, labeled −26; [amendment posted](posts/2026-08-05-sealed-plan-v2.md);
  v1 deprecated; v2 anchor re-pools from the v1 report's per-dataset
  means when the running eval lands — note: sealed run has NO npz
  dump; the recompute (`fontaine/scripts/sealed_v2_anchor.py`,
  sanity-checked against the primary report) is **approximate, not
  exact as earlier claimed** — the pooled summary weights by valid
  chunk elements, not frames, so re-pooling per-dataset means
  reproduces it only to ~5e-3 (bijou) / ~4e-2 (state-copy); method
  error ~15× smaller than the −0.07 v1→v2 shift, negligible vs the
  0.15 band, quoted with the anchor).
  (3) **North star declared: a VLA for the owner's rig — prove
  few-shot transfer (new SO101 arm, tens of episodes)** → saved to
  memory + ideas.md #16 (benchmark pre-reg to write after the box
  batch lands); backlog reweighted toward rig transfer.
- 17:08Z: **(a) update the dataset README** — draft posted in-channel
  17:2xZ; owner 17:18Z: "README section text is good 🎉" → resolved
  by 17:20Z "you push" above.
  **(a2) 17:16Z Discord formatting** — owner: posts render as text
  blobs; adopted Discord-markdown house style (headers/bullets/
  backticks, ≤2000 chars, long-form on the blog) + saved to memory. **(b) sealed plan
  "overly strict"** — steering adopted: outcomes measurable +
  pre-registered, but the sealed plan is *versioned*; a wrong measure
  is fixed by a posted amendment (sealed_v2 + reason + fresh anchors,
  v1 deprecated loudly), never silent edits. Codify in the rules pass
  (queued next session). Concrete case queued: post-removal sealed_v2
  redraw with census-predicted baseline pre-registered first.
- 17:02Z: **box fine-tune checkpoints must survive** (owner rsync in
  flight) — honored; no deletes ever on that box.
- 16:50Z dataset cleanup (kevin510/bbox-2 upstream removal):
  sequencing proposed in-channel, unconfirmed. **Boundary extended to
  the box copy**: no re-pull/mutation of `community_curated_v0` on
  EITHER box until the batch arms + reads are done. Record the
  pre-removal HF revision hash before any upstream push lands.
- 16:52Z 80k checkpoint: **resolved** — owner's own panel eval found
  on the box (numbers above); remaining work is CPU analysis, no GPU
  eval needed.
- 16:17Z bijou code deep-dive + 16:21Z rules/prompts review: queued,
  next work session (first-class items).
- 16:19Z literature slice (~20–30 min most sessions): **not spent
  this session either** (consumed by the box launch chain) — two
  sessions running; the next work session should start with it
  unless a run needs surgery.

## Queue (depth 5)

1. **Babysit the box batch** (every ~30 min session time; see box
   section). At arm completion: check panel evals ran, then the
   **results post**: primary read A-s0 vs B-s0 + E5 noise floor
   (decision rule in the pre-reg) — closes idea #6's 40k rung.
2. **Sealed anchor** at ~18:05Z: bank, post in-channel (tick loop
   instructions below).
3. **Noise-draw ensembling probe** on local GPU after the sealed
   score: verify `--sample-draws` (a16e65a) → pull flow-80k ckpt →
   run per pre-reg. Draw-spread unimodality check first (in pre-reg).
4. **Paired flow-vs-AR per-frame analysis** (CPU, npzs in local
   `reports/`) — results post; any session.
5. **Owner-steered reviews** (chained work session, in order): (a)
   rules/prompts full pass, (b) bijou deep-dive → ranked list, (c)
   **trunk survey** (open-weights, <7B ideally ~3B, video-trained
   preferred; arXiv paper + HF config per candidate; Ministral 3 3B
   + Gemma 4 E2B/E4B seed the list; ranked doc on the blog).
6. Stage-2 sign-convention pre-reg draft (mirror trio) — backlog.

## Data-blocked / handoff notes for the tick loop

If a tick lands after `fontaine-eval` finishes: (1) read
`~/eval_baseline_sealed.log` tail + report JSON; chunk_mae within
~0.15 of 5.8017, state-copy ≈ 11.785 (bigger gap ⇒ the two panel
draws disagree → diagnose, charter §2); (2) post the sealed anchor
in-channel (Discord-markdown style, short); (2b) **recompute the
sealed_v2 anchor** from the report JSON (drop the 3 removed repos'
per-dataset means × counts from the pool — exact for frame means;
plan v2 + the amendment post give repos and counts) and post both
v1 + v2 anchors; (3) **do NOT launch the local paired run** —
superseded by the box batch; (4) launch the
noise-draw chain: `tmux new-session -d -s fontaine-eval-draws 'bash
~/eval_flow80k_draws_panel.sh'` (fully verified + checkpoint local;
E1 gate stops the chain itself if the instrument disagrees), then
first-poll util check per the standing rule.

Box babysit one-liner (tick or work):
`ssh ubuntu@192.222.55.210 'tail -2 ~/train_fontaine_*.log; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'`

Known safe-to-ignore: `wandb/` untracked at repo root (smoke
scratch); owner tmux sessions on the box (`5`, `rigjudge`,
`watchdog`) — theirs, do not touch.

## Utilization footer

Trailing-7-day GPU-hours on experiments / total: local **~1.6 / ~1.8**
(eval bursts + smoke; sealed eval running), box **4 GPU-streams live
since 17:12Z** (~22–26 GPU-h queued today: 2 exploit-attribution
arms + 2 instrument replicates). Explore/exploit: aux-off arm B +
noise-floor replicates ≈ instrument/attribution (exploit-side);
explore hours proper start with the noise-draw probe + flow-vs-AR
analysis tonight. Literature slice: 0 h two sessions running
(flagged above).
