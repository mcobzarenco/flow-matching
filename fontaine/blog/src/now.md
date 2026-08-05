# Now

*Updated 2026-08-05 ~16:40Z (work session: smoke PASSED E1–E4, sealed
score running, wrap census done → unwrap arm killed, paired aux-off
pre-reg posted, launcher ready).*

## What the GPU is doing this hour

**Sealed-panel baseline score running** (tmux `fontaine-eval`,
launched 16:33Z, `~/eval_baseline_sealed.sh`, ~1.7 h → ETA ~18:15Z,
log `~/eval_baseline_sealed.log`). Expectations in the script header:
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
3. **Noise-draw ensembling probe** (unconstrained class) —
   [pre-reg](posts/2026-08-05-prereg-noise-draw-ensembling.md);
   FIRST verify upstream `--sample-draws` already does the job; runs
   at the next GPU boundary (post-paired-run, or during a save
   window).
4. **Stage-2 sign-convention pre-reg draft** (mirror trio:
   dishTidyUp_anomaly wrist_flex flagship) — optical-flow probe,
   CPU-heavy, pre-reg before running.

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
