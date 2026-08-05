# Now

*Updated 2026-08-05 ~17:10Z (tick: sealed eval healthy — 6,144/25,800
@ ~305 f/min, ETA ~18:05Z; owner steering on dataset cleanup + 80k
ckpt answered; **4×H100 second box granted** — work session chained
to write the batch pre-reg).*

## ⚡ NEW: second box (owner 17:01Z)

`ssh ubuntu@192.222.55.210` — 4×H100 80GB (idle), 104 cores, 885G
RAM, 7.6T free. Has `~/flow-matching`, `~/datasets`, previous
checkpoints + owner eval artifacts (looks like the owner's training
box). Owner 17:02Z: "feel free to use it whichever way you want."
Duration unstated ⇒ **temporary-box discipline: batch pre-reg before
any launch, save-boundary sizing, continuous rsync-back of
checkpoints/logs to the local box.** Plan skeleton = the 16:48Z
4×H100 post: (1) paired aux-off arms in parallel there (~5.5h wall),
(2) 2–3 seed replicates of the 40k control for the noise floor,
(3) local box becomes the eval box → 80k flow panel eval can run
locally TONIGHT after the sealed score. Chained work session:
verify repo/venv/data state on the box, write the batch pre-reg,
launch, and re-point the local queue (the local paired launch at
sealed-eval end is superseded IF the box pair launches first —
otherwise keep the local launch; GPU never idles either way).*

## What the GPU is doing this hour

**Sealed-panel baseline score running** (tmux `fontaine-eval`,
launched 16:33Z, `~/eval_baseline_sealed.sh`, ~1.7 h → ETA ~18:20Z,
log `~/eval_baseline_sealed.log`). First-poll check 16:45Z: 254 f/min
(healthy band), util 66–100%, VRAM 12 GiB — workers 8 is not
starving this run. Expectations in the script header:
chunk_mae within ~0.15 of the primary read (5.8017), state-copy near
11.785.

**The moment it finishes → launch the paired run** (see queue #1):
`tmux new-session -d -s fontaine-train 'bash
~/launch_fontaine_paired_auxoff_40k.sh'` — never concurrently with a
GPU eval. Chain: arm A (control 40k) → arm B (aux-off 40k) → both
panel evals with `--dump-predictions`. ~13 h total; per-arm kill gates
in the launcher header; arm A killed ⇒ do NOT launch arm B.

**GPU norm (owner, 16:13Z): the machine is mine 24/7 — the GPU never
idles.**

## This work session (16:22Z→) — what happened

1. **Smoke relaunch + PASS.** First smoke (16:19Z) died at
   `wandb.init`: no API key in fresh tmux shells (key lived only in
   the harness env). **Durable fix: `wandb login` → `~/.netrc`
   (0600).** Relaunched 16:23Z, all 300 steps: E1 selection 878
   datasets / 42,872 episodes / 103 dropped / dims 6/6 ✓; E2
   0.39–0.45 s/step at B10, VRAM peak 67.4 GiB < 76 ✓; E3 loss
   19.7 (step 20) → 6.33 (step 300), falling throughout ✓; E4
   step_000200 + step_000300 checkpoints with loadable
   `bijou_config.json` ✓. wandb run `fontaine_smoke_arb_300_1xh100`
   synced.
2. **Wrap census (#14) measured and closed** —
   [write-up](posts/2026-08-05-wrap-census.md), instrument
   `probes/probe_wrap_census.py` (anchors asserted in-probe, ANCHORS
   PASSED; pooled-MAE part reproduces the official 5.8026 bitwise).
   Panel: 16/17,204 wrap frames (0.093%) carry 0.0720 of panel
   chunk_mae (wrap frames average 78.27; shortest-arc re-score
   5.7498). Corpus (all 20.7M frames scanned): 81/42,872 episodes
   (0.19%) in 23 repos; **kevin510 systemically corrupted (40/40
   eps, wrist_roll+shoulder_lift, action+state)**;
   willnorris/bbox-2 is a distinct state-stream glitch (all six
   state dims incl. gripper, actions clean). wrist_roll dominates
   (204 action jumps) — matches the SO101 calibration story
   (lerobot#1255, PR#777, #3193; fixed in 0.6.0). codebase_version
   is uniformly v3.0 post-mirror ⇒ version correlation untestable
   locally. **Verdict: training-side wraps rare → unwrap-at-load arm
   killed per 16:13Z steering.** Open recommendation for owner:
   shortest-arc eval metric (recovers 0.053 with zero training; moves
   every anchor, needs sign-off).
3. **Paired-run pre-reg posted**
   ([pre-reg](posts/2026-08-05-prereg-paired-auxoff-40k.md)): arm B =
   **aux-supervision-off** (idea #6, the still-owed mainline
   attribution question; zero new code — `--aux-fields` omitted, all
   conditioning kept). Runner-up candidates considered:
   `--stream-counts` re-test (semantics need study first),
   `--trim-leading-idle` (docs-only — no such flag exists yet, needs
   implementation + oracles), FAST v3 (token-metric reset seam).
   Own-baseline pre-reg marked SUPERSEDED (banner, body untouched).
   Launcher: `~/launch_fontaine_paired_auxoff_40k.sh` (header = the
   pre-reg; eval flags verified against `bijou/eval/cli.py`).
4. **Re-score HTML committed into the blog** per owner convention:
   [assets/reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_curated_v0_k4l2.html](assets/reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_curated_v0_k4l2.html).

Incidental instrument note: `bijou.eval` already ships
`--sample-draws` — the noise-draw ensembling pre-reg's "~20 eval-side
lines" may already exist upstream; verify before writing code
(queue #3).

## Bootstrap scoreboard (charter §10)

- §10.1–§10.4 — done (see git history; timer live, Discord live).
- §10.5 baseline re-score — **CLOSED** (5.8017 vs 5.8026, gate ±0.05;
  state-copy 11.7848 exact).
- §10.6 integrity kit — sealed panel built + leakage checker shipped;
  **sealed-panel baseline anchor: running now** (above).
- §10.7 first experiment — pre-registered (paired aux-off), launches
  at the sealed score's end. 48 h clock started at the smoke test.

## Owner steering log (active items)

- 16:50Z: **remove corrupted datasets locally + push removal to the
  upstream HF dataset, note reason in README**; owner will change soon
  but defer the new eval-baseline recompute. **Answered in-channel
  ~17:00Z**: agree; kevin510 (both repos) clear-cut, bbox-2 too
  (separate README line, different failure mode), leave the scattered
  0.19% wrap eps in the other 23 repos. Flagged: removal re-draws the
  0.1/split-seed-0 holdout ⇒ new baseline epoch, not a small delta;
  both kevin510 repos + bbox-2 are pinned in the sealed plan, so
  today's sealed anchor only replays against the pre-removal HF
  revision — record the revision hash before pushing. Proposed
  sequencing: owner pushes upstream anytime; local copy stays frozen
  (no re-pull) until after the paired arms + 80k eval; then cleanup +
  baseline recompute at the following eval slot, with a
  census-derived *predicted* post-removal baseline pre-registered
  first. Awaiting owner reaction; not yet applied locally.
- 16:52Z: **80k flow-matching checkpoint exists** (40k + 40k more) —
  confirmed on HF: `bijou_flow_artrunk_h1024_40k_ddp2/step_080000`.
  Plan proposed in-channel: panel-score it after the paired arms
  (~07–09Z tomorrow) on the pre-removal revision so it's comparable
  to the 100k AR 5.8017; offered tonight-slot (between sealed score
  and paired launch, paired slips to ~20:00Z) if owner wants the
  number sooner. **Default = pre-registered order unless owner says
  otherwise.** → New queue item below.
- 16:43Z: hypothetical — "if I gave you a 4×H100 box via ssh for
  18h, what would you prioritise?" **Answered in-channel 16:48Z**:
  breadth + noise floor over one big run — (1) seed-replicate the
  40k control ×2–3 to measure the paired-comparison noise floor,
  (2) paired aux-off arms in parallel, (3) 1–2 more pre-registered
  40k arms (stream-counts, FAST v3), (4) local box becomes the eval
  box; temporary-box discipline (batch pre-reg first, continuous
  rsync-back, save-boundary sizing). If access materializes, that
  post is the plan skeleton — draft the batch pre-reg before
  touching the box.
- 16:13Z: paired run supersedes standalone own-baseline; GPU 24/7
  norm. **Done this session** (pre-reg posted, launcher ready,
  own-baseline marked superseded).
- 16:16Z: wraparound web research + census must cite the lerobot
  cluster. **Done** (census post cites #1255, PR#777, #3193, #1296,
  0.6.0; version correlation noted untestable post-mirror).
- 16:17Z: in-depth review of ALL bijou code → ranked
  low-hanging-fruit list. **Queued, next work session** (first-class
  item).
- 16:19Z: standing ~20–30 min literature slice most sessions.
  **Adopted** in `fontaine/prompts/work.md`; not spent this session
  (bounded session consumed by the launch chain — noted, not
  skipped silently).
- 16:21Z: review ALL rules/prompts on work structure, adjust as I see
  fit. **Queued, next work session** (with the bijou review; both
  outrank new analysis per the ladder).
- Earlier resolved items: see git history of this file (15:0x–16:1xZ
  entries) — probe sync, sign-convention stage 1 + decision on the
  mirror trio, wraparound calibration context.

## Queue (depth 3)

1. **Paired aux-off run** — launch at sealed-score end (~18:15Z):
   `tmux new-session -d -s fontaine-train 'bash
   ~/launch_fontaine_paired_auxoff_40k.sh'`. Then per-arm liveness +
   curve checks every ~30 min of session time (E1–E3 in the launcher
   header). Results: paired per-frame analysis (CPU, off the dumped
   npzs) + results post; closes idea #6 at 40k/eff-10.
2. **Owner-steered reviews** (next work session, in order): (a)
   rules/prompts full pass (16:21Z), (b) bijou code deep-dive →
   ranked list posted in-channel + ideas.md (16:17Z).
3. **80k flow panel eval** (owner 16:52Z) — after the paired arms
   finish, BEFORE any local dataset cleanup: pull
   `bijou_flow_artrunk_h1024_40k_ddp2/step_080000`, panel-score on
   the current (pre-removal) data + both plans (curated_v0_k4l2 +
   sealed) for comparability with 5.8017. ~1.7 h/plan.
4. **Noise-draw ensembling probe** (unconstrained class) —
   [pre-reg](posts/2026-08-05-prereg-noise-draw-ensembling.md);
   FIRST verify upstream `--sample-draws` already does the job; runs
   at the next GPU boundary (post-paired-run, or during a save
   window).
5. **Stage-2 sign-convention pre-reg draft** (mirror trio:
   dishTidyUp_anomaly wrist_flex flagship) — optical-flow probe,
   CPU-heavy, pre-reg before running.

**Dataset-cleanup boundary (owner 16:50Z, sequencing proposed
in-channel, unconfirmed): do NOT re-pull / mutate the local
`community_curated_v0` copy until the paired arms AND the 80k eval
are done** — the sealed plan pins kevin510 + bbox-2 episodes, and the
paired arms are pre-registered on the current revision. Before any
upstream push lands locally: record the pre-removal HF revision hash
in the ledger.

## Data-blocked / handoff notes for the tick loop

If a tick lands after `fontaine-eval` finishes: (1) read
`~/eval_baseline_sealed.log` tail + report JSON, check chunk_mae
within ~0.15 of 5.8017 and state-copy ≈ 11.785 (larger gap = the two
panel draws disagree → diagnose before trusting either, charter §2);
(2) post the sealed anchor in-channel; (3) launch the paired run (tmux
command above — GPU must not idle); (4) first-poll util check per the
standing rule (smoke showed 100% util / 0.4 s/step healthy; eval
showed starvation fixed at workers 20).

Known safe-to-ignore: `wandb/` untracked dir at repo root (wandb
scratch from the smoke's working dir; gitignored outputs/ holds the
real run dirs).

## Utilization footer

Trailing-7-day GPU-hours on experiments / total: **~1.4 / ~1.6** (box
day 0; instrument: nvidia-smi polling). This session: smoke 0.1 h
(exploit/infra), sealed eval started (infra anchor). Explore hours: 0
so far — first experiment (paired arms) launches ~18:15Z and runs
~11–13 h, of which arm B (~5.5 h + eval) counts explore. Gap
explanation: bootstrap + staging until ~14:00Z, eval bursts since.
