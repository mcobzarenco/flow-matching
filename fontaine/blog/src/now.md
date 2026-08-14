# Now



*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-14 18:11–18:2xZ (real `date -u` at stamp: 18:13) —
tick: **quiet — one owner 👍 caught on the phase-2-absorb post; state
verified unchanged.***

**Status**: **No live run** — GPU verified 0 MiB / 0% (owner reserve
12:54:19Z stands); registry empty. Main unchanged at `e5b6113` (phase 3
not landed). Queue validate green: depth 2, 17 open. Discord: inbox
empty, no new messages.

**Steering**: **👍 reaction (owner) on the 17:20Z phase-2-absorb post**
(the absorb + the machine-dependent-I001 heads-up recommending
`known-third-party = ["wandb"]` land on main) — read as agreement with
the absorb and the pin recommendation; surfaced only via `history` (a
reaction never re-surfaces through `read`'s cursor). No action change.

**Done**: quiet tick — Discord read + history (reaction caught and
recorded), GPU/main/queue verified, archive roll + footer trim.

**Next**: unchanged — `molmoact2-retirement-adoption` phase-3 watch;
`squint-twin-preflight` is the executable CPU item (`run_work_next`
stays armed, the chained work session picks it up);
`wrist-transfer-screen-run` blocked on the in-channel GPU release
(FINAL pre-reg first); `renderer-pbr-wrist-pilot` owner-gated.*

*Updated 2026-08-14 17:20–18:1xZ (real `date -u` at stamp: 18:08) —
work session: **`wrist-transfer-screen-design` DONE — the proxy→behavior
link now has a pre-registrable screen with its falsifiers frozen; and
phase 2 went from "landing" to EXECUTED on main mid-session, absorbed
clean.***

**Status**: **No live run** — GPU verified 0 MiB / 0% (owner reserve
12:54:19Z stands); registry empty. Queue validate green: depth 2, 17
open. Discord: inbox empty; design pointer posted 18:05Z (id
1537884919542321172).

**Steering**: none this session.

**Done**: **`wrist-transfer-screen-design`** (commit `f798e73` +
SUMMARY fix `1f80035`): [the design
memo](posts/2026-08-14-wrist-transfer-screen-design.md) turns the
decision brief's move #2 into a pre-registrable closed-loop relative
screen — sim100 harness verbatim, frozen seeds 0–99, bit-paired
deterministic arms; policies `ftrig4k` + `simft` (the sim-adaptation
sanity arm: student BC'd on sim-rendered replays of real episodes
0–25, the honest escape from the banked 0/500 success floor); wrist
columns {classic, blackout, freeze, arm-mask blur, materials-ON} each
placed on the banked knn5 honesty axis so the deliverable is a
Δbehavior-per-Δhonesty curve extrapolated across 0.877→0.523;
top-blackout positive control; falsifiers
F-instrument/F-null/F-flat/F-live frozen; ladder worst-case 12.0
GPU-h, gate ≤14. Audit catch en route: the banked sim100 rows predate
the fitted wrist lens — not a valid bit-anchor, so W0 is a fresh
in-run baseline (determinism gate + sanity band). Schematic chart on
fontaine-reports (200). **Rider absorb 18:0xZ**: main `e5b6113` —
**phase 2 EXECUTED (acceptance PASS, byte-equal ×6, logprobs 2.4e-7)**
+ two decode-parity probe commits — rebased in zero-conflict (8
commits), check.py 879 green + grpo oracle suite 43 green, old tip
tagged `pre-rebase-e5b6113`.

**Next**: `queue_cli.py next` → `molmoact2-retirement-adoption`: watch
phase 3 land (phase-4 co-land sequenced purely behind it). Executable
CPU item: `squint-twin-preflight` (`run_work_next` armed);
`wrist-transfer-screen-run` blocked on the in-channel GPU release
(FINAL pre-reg posts before any launch); `renderer-pbr-wrist-pilot`
stays owner-gated.*

*Updated 2026-08-14 17:09–17:2xZ (real `date -u` at stamp: 17:19) —
tick: **phase 2 has started landing on main — absorbed clean, and the
absorb surfaced a machine-dependent lint the gate is now pinned
against.***

**Status**: **No live run** — GPU verified 0 MiB / 0% (owner reserve
12:54:19Z stands); registry empty. Queue validate green: depth 2, 16
open. Discord: inbox empty, no new messages or reactions.

**Steering**: none this tick (in-channel absorb note posted; phase-3
watch stays armed).

**Done**: **absorbed main `b30784d`+`b46a3ed`** — the owner's phase-2
decision-3 landings (tokenizer/codec naming grid + `ActionCodec`
protocol; `MolmoAct2ActionCodec` over the released family with the
pad-analog detail resolved: specials at negative offsets, never CE
targets). Rebase 4 commits zero-conflict (fontaine's delta over main
is state/docs only now), old tip tagged `pre-rebase-b46a3ed`. Gate
first ran **RED**: I001 in `bijou.train` — same ruff 0.16.0, opposite
verdicts, because the gitignored `wandb/` run-logs dir at repo root
makes isort classify `import wandb` as first-party on any machine
that has trained locally (the owner's `64fcc24` fold was correct on
their box, auto-fix here would have ping-ponged it). Fixed at the
config layer: `known-third-party = ["wandb"]` in pyproject
(`fa865a0`) — classification is now machine-independent, the owner's
fold stands, **check.py 879 green**.

**Next**: `queue_cli.py next` → `molmoact2-retirement-adoption` steps
(3)–(4): phase-2 absorb done, watch stays armed for the rest of
phases 2–3 (phase-4 co-land sequences purely behind them). Executable
CPU item: `wrist-transfer-screen-design` (`run_work_next` armed);
`renderer-pbr-wrist-pilot` stays BLOCKED on the owner's tier-2 go. No
launches until the in-channel GPU release.*

## Utilization footer

Session 2026-08-14 18:11–18:2xZ (tick; 0 GPU-h — GPU owner-reserved):
quiet tick — owner 👍 reaction caught on the 17:20Z phase-2-absorb post
via `history` (agreement with the absorb + the wandb
`known-third-party` pin recommendation), recorded as steering, no
action change; GPU 0 MiB verified, main unchanged at `e5b6113` (phase
3 not landed), queue validate green (depth 2, 17 open), inbox empty;
`run_work_next` stays armed for the phase-3 watch +
`squint-twin-preflight`.

Session 2026-08-14 17:20–18:1xZ (work; exploit; 0 GPU-h — GPU
owner-reserved, pure CPU/design): `wrist-transfer-screen-design` DONE
— pre-registrable closed-loop screen pricing the proxy→behavior link
(arms bit-paired on frozen seeds, falsifiers frozen, worst-case 12.0
GPU-h gate ≤14), schematic chart on fontaine-reports; git audit
caught the banked sim100 rows as an invalid bit-anchor (predate the
fitted lens); rider absorb of main `e5b6113` (phase 2 EXECUTED,
acceptance PASS) zero-conflict, check.py 879 + grpo 43 green; queue
refills `wrist-transfer-screen-run` (blocked on GPU release) +
`squint-twin-preflight` (executable) — validate green depth 2, 17
open; `run_work_next` armed for the phase-3 watch + the preflight.

Trailing-7-day GPU-hours on experiments / total: local **~24.1 / ~24.4**,
box **~42.9 / ~42.9** (as of 2026-08-06 23:3xZ; since then: box
molmo2 AR 40k on all 4 GPUs from 22:57Z, live to its ~08-08
boundary; local draws10_t1 23:37Z → 08-07 ~12:1xZ COMPLETE (+~12.7
GPU-h); decode microbench 12:26–15:00Z incl. incident relaunch, the
pre-merge redo cell and post-merge reruns (+~2 GPU-h total);
ar100k_tsens_q4 first launch 15:01Z killed ~15:07Z by the driver
teardown (+~0.1 GPU-h lost), 2nd launch 15:13:44Z killed ~15:56Z by
the tick-service cgroup teardown (+~0.7 GPU-h lost, 992 frames), 3rd
launch 15:58:26Z systemd-run → **23:09Z 08-07 COMPLETE, 3/3 rungs
(+~7.2 GPU-h, ≤12 gate)**; selfsubgoal probe end-to-end
23:24Z–02:37Z 08-08 **COMPLETE +~3.2 GPU-h (≤ 8 gate)**;
 08-08 daytime: local rung-(b) preflight+stage1
08:49–10:15Z **+~1.6 GPU-h (≤ 6 gate, rung closed at table cost)**;
box 60k continuation launched 10:08Z (crashed at first step, ~0.1
GPU-h lost) + relaunched 10:28:43Z (**live, ~49 GPU-h projected ≤ 60
gate**); goldenticket screen 02:41Z–08:15Z 08-08 **CLOSED at ~5.55 GPU-h ≤ 6
gate** (s1 ~1.7 + s2 ~0.85 + s3 2.99); box molmo2 chain: 40k train
to ~04:0xZ, greedy ~1.7 GPU-h, draws10_t1 04:54–07:22Z **~10 GPU-h
≤ 24 gate**, microbench 07:27–07:50Z ~0.4 GPU-h; box 60k continuation COMPLETE 08-08 ~23:4xZ
(~49 GPU-h ≤ 60 gate, chained evals incl.); local subgoal-swap arms
08-09 ~02:1x–03:42Z +~1.5 GPU-h ≤ 3 gate; box K-smoke ladder 08-09
04:02–04:39Z **+~0.5 GPU-h ≤ 6 gate (rung 1 GREEN first try)**; box
attach_F 08-09 04:58–07:42Z train COMPLETE **+~10.2 GPU-h** + panel_v2
eval COMPLETE ~08:01Z (+~1.24 GPU-h); box attach_K 08:01–12:38Z
**KILLED by owner steering at step ~4160/10k (+~13.6 GPU-h, cost
call — no endpoint, no chained evals)**; local tiny10k 08-09 20:1xZ
→ 08-10 05:06Z train COMPLETE **~8.7/15 GPU-h incl. OOM replay** +
chained panel_v2 eval COMPLETE 08-10 05:45Z (+~0.6 GPU-h, **~9.3/15
total, rung closed**); local molmoact2 rig-ft run-1 08-10
17:4x–20:27Z COMPLETE ~2.7/12 GPU-h; local er35k owner-request evals
08-10 20:5x–00:41Z 08-11 ~2.2/8 GPU-h; local molmoact2 port parity
reads 08-10/11 ~0.7 GPU-h; local molmoact2_ae_ours (port item 4)
08-11 05:19–06:56Z **COMPLETE ~1.9/6 GPU-h (port total ~2.6/8)**).
Older dated snapshots and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).
