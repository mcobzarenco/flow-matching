# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-07 18:37–19:0xZ (real `date -u`) — tick (babysit)
turned conversational: owner live in-channel — **#17 amendment 2
landed** (5k/arm, fresh-Adam route owner-confirmed 18:39Z) +
**golden-ticket in-depth explainer posted** (owner's 18:33Z
question); recovered the killed 18:24 session's uncommitted
param-group correction.*

**Status** (babysit 18:38Z):
- box molmo2 AR 40k — 27140/40k, loss 2.9399 (falling −0.016 over
  the window), 2.167 s/step, vram 67.07 ≤ 71. Probe 6.81@27000
  (5.91@26500 stands as the low). Gate margin 4.93. ~7.7 h to 40k.
- local **ar100k_tsens_q4 rung t0.7** — healthy: 352/4301 at the
  18:38:39 flush (160-frame chunks 32→192→352, ~20 f/min incl.
  model load; util 24–25% steady). Babysit **exit-3 "gate
  crossing" (projection 59.6 h) judged FALSE POSITIVE** — the
  cumulative baseline still anchors at the 15:58Z t0.5 launch while
  the per-rung frame counter reset at the 18:21Z roll; artifact
  anchor added to `babysit.toml`. Real cumulative ≈ 2.7 GPU-h ≤ 12.
  t0.7 ends ~21Z, t1.3 ~23:3xZ → dT read ~00Z.

**Steering** (owner live 18:31–18:39Z, conversational mode): (1)
18:31Z seed/rewarmup/5k/LR message → answered 18:35Z by the prior
session; (2) 18:33Z "tell me more in depth about optimising the
initial noise vector" → in-depth explainer posted 18:40Z (ODE-map
claim, why the panel makes the search ~free, banked-null machinery,
shared-ticket prior against, per-dataset escalation path); (3)
18:39Z **"you're right re: fresh adam optimisers"** → the offered
resume-with-injected-vision-group patch is DROPPED, fresh-AdamW
`--init-from` confirmed → amendment 2.

**Done**: tick — **#17 amendment 2** (5k steps/arm, `vu5k` naming
incl. eval stems, gate 24→32 GPU-h with recomputed arm costs
12.2/13.9, provenance header, §8 cost note); recovered + re-verified
the 18:24 session's uncommitted 5-vs-3 group-count correction
(`bijou/train.py:3385-3410`: decoder 1 group, +2 decay/no-decay per
unfrozen backbone group) and stated the correction in-channel; blog
built + Space pushed (post curl-verified 200, amendment content
live); check.py 460 green; queue validate green (depth 2, 14 open);
`run_work_next` armed. Three Discord posts (explainer, lock-in,
amendment confirmation).

**Next**: #17 → finalization amendment (byte-audit + memory-ladder
smoke + endpoint-probe quote + `vu5k` launchers) + owner go, window
post-attach-screen. GPU-busy windows →
`idea1-golden-ticket-instrument` (CPU). tsens dT read opens ~00Z;
molmo2 endpoint ~08-08 morning → #19 box obligations → K smoke
ladder → attach-screen window. **Every GPU launch goes through
`run_detached.sh`.**

*Updated 2026-08-07 18:00–18:2xZ (real `date -u`) — tick (babysit):
**owner steering 18:02Z on #17** (warm-start the unfreeze from the
40k checkpoint, two arms frozen/thawed — replied in-channel,
agreed, amendment falls to the chained session) + **tsens rung roll
t0.5 → t0.7 caught live 18:21Z**, babysit stem repointed.*

**Status** (babysit 18:00Z + 18:21Z):
- box molmo2 AR 40k — 26700/40k, loss 2.9714 (falling −0.038 over
  the window), 2.181 s/step, vram 67.07 ≤ 71. Probe **5.91@26500 —
  new low** (prior best 5.97@22500). Gate margin 4.93. ~8.1 h to
  40k → endpoint ~08-08 morning.
- local **ar100k_tsens_q4** — **rung t0.5 COMPLETE 4301/4301
  ~18:21Z** (json + html + npz written); **t0.7 launched 18:21Z**
  (`--ar-temperature 0.7` confirmed on the live process), babysit
  `log` stem repointed t0.5 → t0.7 in `babysit.toml`. The 18:00
  zero-window was the flush-quantization artifact again (log
  flushes in 160-frame chunks; mtime 17:56 at 3552). Cumulative
  gate projection 2.5 ≤ 12. t0.7 ends ~21Z, t1.3 ~23:3xZ → dT read
  ~00Z.

**Steering** (owner 18:02Z, replied 18:2xZ): on #17 — start from
the 40k checkpoint, two arms frozen/thawed instead of the
from-scratch 10k screen; "startup mindset, shortest time to high
quality rollouts". **Agreed in the reply**: frozen-continue is the
control (extra steps alone move the number), read = thawed vs
frozen paired per-frame Δ; ~15 GPU-h (2 × ~3k steps) vs ~27, and
it upgrades the deployment artifact directly. Caveat stated: late
low-LR thaw can understate unfreeze-from-scratch (lit co-adapts
vision from step 0) — asymmetric bet, acceptable. **#17 draft
amendment = next chained-session item** (arms, steps, tower Adam
warmup, kill lines; execution window unchanged post-attach-screen,
still owner-held).

**Done**: tick — babysit 18:00Z exit 0 both green; held the session
through the rung boundary (charter §6), verified the roll on the
live process list, repointed the stem; owner reply posted
in-channel; `queue_cli.py validate` green (depth 2, 14 open);
`run_work_next` armed (was already, 17:59). No blog build (Discord
reply + now.md only).

**Next**: chained work session → **#17 draft amendment to the
warm-start two-arm design** (owner steering, jumps the queue) +
rejoin the thread via `history`; then
`idea1-golden-ticket-instrument` (CPU) in GPU-busy windows.
**idea19-tsens-dt-read-execution** opens at rungs completion ~00Z.
molmo2 endpoint ~08-08 morning → #19 box obligations → K smoke
ladder → attach-screen window. **Every GPU launch goes through
`run_detached.sh`.**

*Updated 2026-08-07 17:47–18:0xZ (real `date -u`) — work session
(bounded, one item): **#1 golden-ticket noise screen pre-reg
POSTED** ([pre-reg](posts/2026-08-07-prereg-golden-ticket-screen.md),
`e162eb1`) — not a draft; every design constant pinned from banked
data before posting.*

**Status** (babysit 17:58Z):
- box molmo2 AR 40k — 26080/40k, loss 2.9898 (falling −0.034 over
  the window), 2.171 s/step, vram 67.07 ≤ 71. Probe 6.67@26000
  (in-band, no ≥7.5 pair). Gate margin 4.93. ~8.4 h to 40k →
  endpoint ~08-08 morning.
- local **ar100k_tsens_q4 rung t0.5** — 3552/4301, window 43.9
  f/min, cumulative 29.6 f/min, projection 2.4 ≤ 12 gate, ~0.4 h
  left. Rung roll t0.5 → t0.7 **~18:2xZ** (babysit `log` stem
  repoint at the first tick after); all rungs ~00Z → dT read.

**Steering**: none (boot poll + babysit-forced poll 17:58Z: no new
messages; `history -n 5`: our own posts only).

**Done**: this session — **#1 golden-ticket screen pre-registered**
(`e162eb1`): teacher-first (flow_artrunk@80k Heun-30; student =
escalation amendment only), M=64 sha-pinned tickets scored as the
draws of ONE batched draws-64 eval on drawsprobe_s7 (~1.5 GPU-h);
null frozen from banked `sigma_draw_direct` (σ_probe 0.0669, null
min₆₄ = mean − 0.157, MC-verified); R1 kill line BEFORE stage 2
(sd > 0.0785 or min < mean − 0.22); R2 = winner on COMPLEMENT core
rows paired vs the banked stable-key npz, adopt floor −0.05 = 2σ;
R3 mean-of-top-10-tickets vs banked 5.3645 (tie band 0.02); R4 free
per-dataset task-locality read (the paper's shared-ticket
regression is the stated prior against). Instrument = a ticket
noise-key mode at the `noise_for_item` seam, 4 oracles frozen in
the post. check.py 460 green; posts/index.md drift fixed (4 missing
entries added). Queue: draft item done, instrument item (CPU,
queued) + execution item (gpu-local, blocked) added; validate green
depth 2. Blog built + Space pushed (post curl-verified 200);
Discord close post.

**Next**: `queue_cli.py next` → **idea19-tsens-dt-read-execution**
(opens at rungs completion ~00Z tonight); GPU-busy windows →
**idea1-golden-ticket-instrument** (CPU: ticket mode + tickets npz
+ 4 oracles). Dated boundaries: tsens rung roll ~18:2xZ (babysit
stem repoint t0.5 → t0.7 at first tick after) → all rungs ~00Z →
dT read; molmo2 endpoint ~08-08 morning → #19 box obligations → K
smoke ladder → attach-screen window. **Every GPU launch goes
through `run_detached.sh`.**

## Utilization footer

Trailing-7-day GPU-hours on experiments / total: local **~24.1 / ~24.4**,
box **~42.9 / ~42.9** (as of 2026-08-06 23:3xZ; since then: box
molmo2 AR 40k on all 4 GPUs from 22:57Z, live to its ~08-08
boundary; local draws10_t1 23:37Z → 08-07 ~12:1xZ COMPLETE (+~12.7
GPU-h); decode microbench 12:26–15:00Z incl. incident relaunch, the
pre-merge redo cell and post-merge reruns (+~2 GPU-h total);
ar100k_tsens_q4 first launch 15:01Z killed ~15:07Z by the driver
teardown (+~0.1 GPU-h lost), 2nd launch 15:13:44Z killed ~15:56Z by
the tick-service cgroup teardown (+~0.7 GPU-h lost, 992 frames),
accruing from the 15:58:26Z systemd-run 3rd launch, ≤12 GPU-h gate). Older dated
snapshots and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).

Session 18:37–19:0xZ: conversational tick, 0 GPU-h new (tsens +
molmo2 accruing under their own gates) — owner live in-channel:
#17 amendment 2 landed (5k/arm, gate 32, fresh-Adam
owner-confirmed) + golden-ticket in-depth explainer; recovered the
killed 18:24 session's uncommitted 5-vs-3 group-count correction;
tsens t0.7 exit-3 crossing judged false positive (cross-rung
projection artifact, anchor added). Blog pushed, check 460 green.
Note: the 18:24–18:4x work session (amendment 1 + seed/rewarmup
reply) hit the hard cap before committing its last edit — its
Discord posts are the record; the edit landed here.

Session 17:47–18:0xZ: all-CPU bounded work session, 0 GPU-h new
(tsens + molmo2 accruing under their own gates) — queue-refill/
pre-reg: #1 golden-ticket screen pre-registered (design + nulls
frozen entirely from banked data; staged kill line before any
full-panel spend); queue 1 done + instrument/execution items added,
depth 2.
