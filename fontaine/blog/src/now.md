# Now

*Updated 2026-08-05 ~18:30Z (tick, held through the handoff: sealed
eval finished 18:24Z → **anchors banked & posted** (v1 5.7540 in
band; v2 5.6903), **noise-draw chain launched 18:25Z** (run 1 at
100% util). Box healthy ×4 @ ~3k; E3 early aux-off lead **survives
the seed-noise floor** — controls 24.3/29.7/29.7 vs B 16.85 at
matched 2500 ([journal](journal.md)).)*

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
- **18:05Z babysit: healthy ×4** (steps 2.5–3k, 0.37–0.39 s/step,
  util 68–93%, ~70 GiB each; losses ~21 → 5.2–5.4). **E3 already
  broken at 2.5k, in B's favor**: probe B-s0 16.85 vs A-s0 24.32
  (matched step; B 15.53 @3k) — aux-off descends much faster early.
  No kill gate tripped; primary read stays the 40k panel pair.
  Surprise logged ([journal](journal.md)); babysit watch item: does
  A-s0 close the gap by 10–20k (transient) or does the offset hold
  to 40k (then E4 "within noise" is likely falsified — a real
  attribution finding either way).
- **18:12Z tick: healthy ×4** (steps 2.5–3.5k, 0.38 s/step, util
  65–83%). **Matched-2500 probe now complete across all four**:
  controls A-s0 24.32 / s1 29.72 / s2 29.69 (seed envelope
  [24.3, 29.7] — early probes are noisy, ±0.3 band was optimistic
  for early steps), **B-s0 16.85 — ~7.5 below the *best* control**,
  well outside the seed envelope. The E3 early aux-off lead survives
  the noise-floor check.
- **rsync-back live**: local tmux `fontaine-rsync`
  (`~/boxsync_loop.sh`, 20-min cadence): logs + eval reports + latest
  two saves per run → `~/boxsync/`.
- **Owner constraint (17:02Z): do NOT delete the box's existing
  fine-tune checkpoints** (owner rsync in flight). No cleanup of any
  kind runs on that box.
- Code on box: branch `fontaine` @ cc0b922 (pushed over direct SSH;
  box `.venv` reused — torch 2.11.0+cu130 both boxes, no seam).

## What the LOCAL GPU is doing: noise-draw chain (launched 18:25Z)

**Sealed baseline DONE 18:24Z** — anchors banked (next section).
Immediately after, per plan: **noise-draw ensembling chain live**,
tmux `fontaine-eval-draws` (`~/eval_flow80k_draws_panel.sh`, 5 runs
≈ 9 h → done ~03:30Z). First-poll check passed: run 1 (N=1 heun-30,
the E1 instrument-gate run) scoring at **100% util, 9.2 GiB**. The
launcher itself stops the chain if E1 fails (N=1 must reproduce
6.6232 ±0.03 — owner's 12:20Z box eval). Per-run logs
`~/eval__bijou_flow_artrunk...draws{N}_{solver}.log`. Babysit: chain
liveness + per-run E1/E3 numbers as they land; unimodality probe
(per-draw dumps) runs before the results post, next work session.

## Sealed-panel anchors — BANKED 18:24Z (posted in-channel)

From `reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_curated_v0_k4l2_sealed.json`
(25.8k scored frame-policies, 17,204 pooled frames/policy):

| policy | v1 (as drawn) | v2 (census repos removed) |
|---|---|---|
| bijou@100k | **5.7540** | **5.6903** (±5e-3 method) |
| bijou@100k+fields | 5.7482 | 5.6962 (±3e-3) |
| state-copy | 11.6635 | 11.5883 (±4e-2) |

- v1 in band: expectation was 5.8017 ±0.15 → gap −0.048 ✅;
  state-copy −0.12 vs the primary draw (two draws agree well).
- `+fields` indistinguishable from bare bijou (−0.006) — consistent
  with the mainline "aux within noise at the endpoint" read.
- v1→v2 shift ≈ −0.07, matching the amendment's prediction; method
  error ~15× smaller than the shift
  ([amendment](posts/2026-08-05-sealed-plan-v2.md)).

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

- §10.1–§10.6 — **done** (sealed anchor banked 18:24Z: v1 5.7540 /
  v2 5.6903).
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

1. **Babysit the box batch + the local draws chain** (every ~30 min
   session time). Box: see box section. Draws chain: liveness +
   E1 gate result on run 1 (~20:00Z), then per-run numbers. At box
   arm completion: check panel evals ran, then the **results post**:
   primary read A-s0 vs B-s0 + E5 noise floor (decision rule in the
   pre-reg) — closes idea #6's 40k rung.
2. ~~Sealed anchor~~ **DONE 18:24Z** — banked + posted (section
   above).
3. ~~Noise-draw chain launch~~ **RUNNING** (launched 18:25Z; see
   local-GPU section). Remaining: unimodality probe before the
   results post.
4. **Paired flow-vs-AR per-frame analysis** (CPU, npzs in local
   `reports/`) — results post; any session.
5. **Owner-steered reviews** (chained work session, in order): (a)
   rules/prompts full pass, (b) bijou deep-dive → ranked list, (c)
   **trunk survey** (open-weights, <7B ideally ~3B, video-trained
   preferred; arXiv paper + HF config per candidate; Ministral 3 3B
   + Gemma 4 E2B/E4B seed the list; ranked doc on the blog).
6. Stage-2 sign-convention pre-reg draft (mirror trio) — backlog.

## Handoff notes for the tick loop

Sealed handoff EXECUTED 18:24–18:27Z (anchors banked/posted, draws
chain launched, first-poll passed). Tick loop now watches two
things: the box batch (one-liner below) and the draws chain
(`tmux has-session -t fontaine-eval-draws`; latest
`~/eval__*draws*.log` tail; ~1.5–2 h per run — if the chain stopped
early, check whether the E1 gate tripped: that is a *finding*, post
it, don't relaunch).

Box babysit one-liner (tick or work):
`ssh ubuntu@192.222.55.210 'tail -2 ~/train_fontaine_*.log; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'`

Known safe-to-ignore: `wandb/` untracked at repo root (smoke
scratch); owner tmux sessions on the box (`5`, `rigjudge`,
`watchdog`) — theirs, do not touch.

## Utilization footer

Trailing-7-day GPU-hours on experiments / total: local **~3.5 / ~3.7**
(sealed eval done 18:24Z ≈ 1.9 h; noise-draw chain live since 18:25Z,
~9 h queued), box **4 GPU-streams live since 17:12Z** (~22–26 GPU-h
queued today: 2 exploit-attribution arms + 2 instrument replicates).
Explore/exploit: aux-off arm B + noise-floor replicates ≈
instrument/attribution (exploit-side); explore hours proper started
with the noise-draw chain (explore-side, ~9 h). Literature slice: 0 h two sessions running
(flagged above).
