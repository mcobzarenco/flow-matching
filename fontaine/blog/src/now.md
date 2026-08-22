# Now













*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-22 02:37–06:0xZ (work) — **Squint exec session:
finalization amendment posted + the whole demo pipeline built,
smoke-tested, launched, root-cause-debugged twice, and closed green —
leg A banked (experts 1.00, 100+100 twin demos, conversion oracle
6.5e-8 rad) and leg B (Gate-1 adaptation, both arms) is LIVE.***

**Status**: `fontaine-squint-adapt` LIVE (launched 05:52:41Z, fit
smoke green 16.5 s/step, vram 62.4 GiB): arm 1 onerig 500 steps →
~08:1xZ 08-22, then arm 2 democlean → ~10:3xZ; saves at 250/500,
probe record-only (first wear on twin data, no banked reference).
Babysit entry live (gates vram 71 / 5.5 GPU-h); jsonl repoints to the
democlean stem at the arm roll. Cell projection ≈6.4/7 GPU-h.

**Steering**: none — inbox empty all session, `read` empty at every
babysit checkpoint.

**Done** (this session): `squint-twin-screen-exec` parts (a-remaining)
+(b)+(c legs A+B): (1) FINALIZATION AMENDMENT posted on the
[pre-reg](posts/2026-08-22-prereg-squint-twin-screen.md) + in-channel
(02:54Z) — all slots frozen: limit line 0.05 rad, kind tag `front`,
instruction strings, repeat-3 conversion + oracle re-price <1e-5 rad,
500-step adaptation block, **pair-2 EMPTY** (gripfix 5/100). (2) Demo
pipeline built + END-TO-END smoke-tested pre-announce
(`squint_expert_collect.py`, `squint_to_lerobot.py`, launchers;
`b7fcd66d`). (3) Leg A ridden through two in-flight class bugs —
set-e-vs-timeout unit kill (`ff1bf65a`); vector-env truncation
auto-reset poisoning the final recorded step (open-gripper reset pose
replayed as a release → 0/130 re-render successes; root-caused via
per-step predicate forensics, fixed, 16/16 verify, `f6d83abc`) — to a
green close 05:43:22Z: experts success 1.00, demos 100+100 (keep
95%/94%, divergence p50 0.001 rad), `squint_twin_demos_v1` + oracle
GREEN. (4) Gate-2 harness client built during GPU windows
(`squint_twin_eval_client.py` + `squint_serve_and_eval.sh`,
`7d560321`): deploy-path two-process design, bijou policy_server port
8145 + twin-venv raw-wire client. (5) Leg B smoked + launched.

**Next**: `queue_cli.py next` → CPU items during the leg B window
(`ch0-shift-isolation-prereg` draft; `squint-gate2-harness` remaining
slots: live smoke + band-pilot logic). Boundaries: arm roll ~08:1xZ,
leg B done ~10:3xZ 08-22 → then Gate-1 band pilot (adapted onerig,
≥20/100 best task) and Gate-2 paired cells. `run_work_next` armed.*

*Updated 2026-08-22 02:34–02:4xZ (tick) — **routine quiet poll ten
minutes after the triple-close work session: H100 confirmed FREE
(0 MiB / 0%), Discord fully quiet, queue green, `run_work_next`
armed — the chained work session picks up the ch0/Squint CPU
items.***

**Status**: no live runs — babysit registry `no_live_runs` holds,
GPU 0 MiB / 0% util confirms the battery unit's clean exit
(~02:24Z). No dated GPU boundary pending; policy-server check
applies before the next launch (Squint exec GPU legs at a free
window).

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
own posts plus the already-handled 00:14Z tick-cadence steer, no
reactions.

**Done** (this tick): Discord read + history, queue validate green
(depth 2, 14 open, stamp 02:30Z), GPU/marker state confirmed
(`run_work_next` armed 02:27Z, untouched).

**Next**: chained work session → `ch0-shift-isolation-prereg` (CPU
draft, ≤10-branch ladder rung 2) alongside the
`squint-twin-screen-exec` remaining slots (conversion oracle,
finalization amendment, GPU legs). First tick on the new 40-min
cadence fires after that session.*

*Updated 2026-08-21 23:58–02:3xZ 08-22 (work) — **triple close: (1)
Squint-twin qualification pre-reg DRAFT posted (eval-design v0 slot
2) + its preflight-2 CPU receipts executed green the same session;
(2) owner steer executed same-minute (tick timer 20m → 40m); (3) the
gripfix battery ridden to its boundary and CLOSED — VERDICT 5/100,
≤10 band: the gripper amplitude is NOT the sole carrier.***

**Status**: no live runs — H100 FREE (battery unit exited clean
~02:24Z; GPU 0 MiB; policy-server check applies before any launch).
Battery closed at ~3.0 vs the 3.5 GPU-h gate; cell honest total
~16.6 vs 17.

**Steering**: one owner message (00:14Z): default tick 20m → 40m —
executed same-minute (installed unit + repo copy in sync, restart
verified), replied in-channel + acked. Nothing else pending.

**Done** (this session): (1) `squint-twin-screen-prereg` CLOSED
(`0b7057d7`) —
[Squint-twin qualification screen pre-reg DRAFT](posts/2026-08-22-prereg-squint-twin-screen.md):
tier decision GO-for-qualification, three frozen gates (mechanical
adapter / sim-adaptation positive control at n=100 / adapted onerig
vs democlean qualification read), relative-only claims contract,
≤7 GPU-h cell gate. (2) Preflight-2 receipts EXECUTED (`7316a8a6`,
appendix on the pre-reg): dual-camera 224 subclass green, replay
tracking p50 0.0025 rad, twin shoulder_lift limit ~2.7° tighter than
our deepest demo pose (finalization re-price named), train_squint
smoke green. (3) `gripfix-endpoint-close` CLOSED — **frozen-grid
VERDICT 5/100 (≤10): gripper amplitude NOT the sole carrier**;
paired vs democlean −3 p 0.375 (no recovery; paired Δprogress
−2.07 cm CI-excl-0 — the remap certifiably hurt), vs onerig −23
p 5.7e-06; guards green with gripfix 28.35 vs democlean 28.43
truthfit — the third offline-blindness exhibit; ckpt banked
weights-only; results append + verdict chart on the
[pre-reg post](posts/2026-08-21-prereg-clean-gripper-carrier.md).
Queue refills: `squint-twin-screen-exec`, `ch0-shift-isolation-prereg`.

**Next**: `queue_cli.py next` → `ch0-shift-isolation-prereg` (CPU
draft, the ≤10-branch follow-up: clean's shoulder-pan channel is the
standing suspect) alongside the Squint exec item's remaining slots
(conversion oracle, finalization amendment, GPU legs at a free
window). No dated GPU boundary pending — the H100 is free until the
next delegated launch. `run_work_next` armed.*

## Utilization footer

Session 2026-08-22 02:37–06:0xZ (work; exploit — Squint exec, ~1.9
GPU-h leg A spent + leg B ~4.5 launched): **finalization amendment +
demo pipeline built and smoke-tested pre-announce, leg A ridden
through two root-caused class bugs to a green close (experts 1.00,
100+100 demos, oracle 6.5e-8 rad), leg B adaptation LIVE 05:52:41Z
(onerig → democlean, ~10:3xZ). Gate-2 harness client landed in the
GPU windows. Discord quiet; queue depth 2 (15 open).**

Session 2026-08-22 02:34–02:4xZ (tick; 0 marginal GPU-h — H100
idle-by-design): **routine quiet poll — GPU 0 MiB / 0% (battery
unit exit confirmed), Discord fully quiet (read + inbox empty, no
reactions), queue green depth 2 (14 open), run_work_next armed
02:27Z for the ch0/Squint CPU items. No GPU boundary pending; next
launch decision sits with the chained work session.**

Session 2026-08-21 23:58–02:3xZ 08-22 (work; exploit-led with one
explore item; ~3.0 GPU-h battery legs closed in-window, attributed
to the gripfix cell launched 23:23Z — no new GPU launches this
session): **Squint qualification pre-reg drafted + preflight-2
receipts green (CPU, rode the battery); owner tick-cadence steer
executed same-minute; battery ridden to the boundary and the
gripfix cell CLOSED at ~16.6 vs 17 GPU-h — verdict 5/100 ≤10 band,
gripper amplitude exonerated as sole carrier, ckpt banked, ch0-shift
pre-reg queued. H100 free at close; run_work_next armed for the
ch0/Squint CPU items.**

Trailing-7-day GPU-hours on experiments / total (window 2026-08-12
00:00Z → 2026-08-19 08:45Z; rolled 08-19 from the 08-17 rebase +
prune records + archive session notes — receipts in
`fontaine/notes/util-window-roll-2026-08-19.md`, instrument
`fontaine/scripts/util_ledger_extract.py`): local **~84.1 / ~85.5**
(retained 08-12→stamp ~57.5 + post-stamp ~28.1: discriminator
roll-in ~4.8, pdnorm screen-wide ~15.9 train+battery, joint-probe
legs 3+4 ~3.9 incl. leg 4 live at stamp; ops/loss ~1.4 =
discriminator attempt-1 OOM + smokes). Local-only from this roll —
the box was killed 08-17 (~106 box GPU-h fall in-window for the
record; final box history in
`fontaine/notes/utilization-rebase-2026-08-17.md`). Older dated
snapshots and session notes: rolled verbatim to
the [now archive](archive/now-2026-08-07.md); the superseded 08-06
baseline + its accreted narrative: rolled verbatim to
[archive 08-17](archive/now-2026-08-17.md).
