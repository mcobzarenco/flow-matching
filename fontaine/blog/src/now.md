# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-07 19:38–19:4xZ (real `date -u`) — tick (babysit):
quiet — both runs green, no steering, no new reactions.
**Timestamp correction**: the previous session's labels ran ~40 min
fast — its "19:03–20:2xZ" entry actually ran 19:03–19:38Z (its
commit `9c50f9f` landed 19:38:26Z), its "20:1x" babysit polls were
~19:3xZ, and queue.json's `updated_utc` was future-dated 19:47Z
(fixed to real time this tick). Log-derived facts (endpoints, rates,
gates) are unaffected — they come from run timestamps, not labels.*

**Status** (babysit 19:39Z, exit 0):
- box molmo2 AR 40k — 28380/40k, loss 2.926 (−0.028 over the
  window), 2.203 s/step (24.8 steps/min), vram 67.07 ≤ 71. Probe
  6.88@28000 (low 5.91@26500 stands, gate margin 4.93). Endpoint
  ~04–05Z 08-08 unchanged (~7.1 h compute + save windows).
- local **ar100k_tsens_q4 rung t0.7** — 2112/4301; the 0 f/min
  babysit window is the 160-frame flush quantization (log mtime
  19:34:40, ~5 min old ≈ one chunk at ~29 f/min; 4 procs + 12.7 GB
  GPU live). Cumulative projection 7.5 ≤ 12 GPU-h. t0.7 ends
  ~21:2xZ, t1.3 ~23:5xZ → **dT read opens ~00:3xZ 08-08**.

**Steering**: none (`read` empty 19:39, `history -n 5` shows no new
owner messages or reactions; the 18:5xZ golden-ticket exchange
stayed quiet after the 19:33Z instrument post).

**Done**: quiet tick — babysit exit 0, both runs judged healthy
(t0.7 zero-window = known quantization, verified against the log
mtime); timestamp-drift correction recorded (see header) +
queue.json `updated_utc` fixed; `queue_cli.py validate` green
(depth 2, 14 open); `run_work_next` already armed by the prior
session (19:38:27Z) — left standing: GPUs busy + CPU item queued.

**Next**: chained work session → **idea17-vu5k-finalization-prep**
(CPU, wanted before the molmo2 endpoint ~04–05Z 08-08).
**idea19-tsens-dt-read-execution** opens at rungs completion
~00:3xZ 08-08. Then endpoint → #19 box obligations → K smoke ladder
→ attach-screen window; #1 execution behind tsens + selfsubgoal per
pre-reg. **Every GPU launch goes through `run_detached.sh`.**

*Updated 2026-08-07 19:03–19:38Z (times corrected from the
mislabeled "19:03–20:2xZ"; real `date -u`) — work session
(bounded): **#1 golden-ticket INSTRUMENT LANDED** (`0acabde`, all 4
pre-reg oracles green, screen now launch-only) + a molmo2 stall
false-alarm run to ground (save-window anatomy, babysit anchor) +
lit slice (LAFM Papers page, same-session per the standing rule).*

**Status** (babysit 19:04Z + 19:33Z + ~19:36Z — last label
corrected from "20:1xZ", see the drift note above):
- box molmo2 AR 40k — 28320/40k, loss 2.9539 (2.194 s/step, 26.8
  steps/min in-window), vram 67.07 ≤ 71, probe 6.8772@28000 (low
  5.91@26500 stands, gate margin 4.93). **Save-window anatomy
  banked**: every save-every-2500 boundary blocks ~15.5 min writing
  ~38 GB synchronously (~42 MB/s; s_per_step ~48.6 on every
  post-save line 2500→27500 — py-spy workup of the 27500 window:
  ranks block on the first CUDA call of the next step, one GPU idles,
  jsonl mid-write). The 19:03 half-rate poll was THAT, not an
  incident; anchored in `babysit.toml`. Endpoint arithmetic
  sharpens: ~7.1 h compute + ~1.3 h saves → **~04–05Z 08-08**.
- local **ar100k_tsens_q4 rung t0.7** — 2112/4301 at ~19:36Z, 29.2
  f/min in-window, cumulative projection 7.4 ≤ 12 GPU-h. t0.7 ends
  ~21:2xZ, t1.3 ~23:5xZ → **dT read opens ~00:3xZ 08-08**.

**Steering**: none (polled at boot 19:04, 19:33, ~19:36 — the only
new message was our own instrument post; the 18:5xZ golden-ticket
exchange is quiet).

**Done**: `0acabde` — **#1 golden-ticket instrument, the queue's
flagged CPU item, landed whole**: `--noise-tickets` mode in
`bijou.eval` via a new `_flow_noise` seam (noise = tickets[draw]
frame-independent, draws-major; policy name gains `_ticket`; report
JSON + draws npz carry `noise_tickets`/`tickets_sha256`; keyed path
proven byte-identical pre/post refactor), bank
`plans/tickets_goldenticket_m64.npz` committed (64×[50,6] f32,
SeedSequence [0x54434B54,0,m], file sha `9bb13bc4…`, content sha
`a07c062a…`, generate-once + `--verify`), 7 pytest oracles
(`tests/test_golden_ticket.py`: draws-1 contract bit-exact vs
`sample_actions(noise=)`, cross-frame ticket property asserted
in-process, two-run determinism, dual sha pins, loud refusals) +
`ticket_scores.py` stage-1 scorer with frozen R1 kill line, R4a
per-dataset matrix, and `--oracle` green (pooling reuse reproduces
the banked 6.5997 and all 10 per-draw probe MAEs EXACTLY). check.py
467 green (was 460). No semantic deviation → no amendment. Discord
post up. This commit (blog): **LAFM Papers page**
(`papers/latent-action-priors.md`, 2606.23420 — learned mode-prior
libraries; the noise-structure ladder above the ticket screen now
mapped in ideas #1, DSRL named as next read if stage 1 CONFIRMs) +
VLM4VLA staged-cell addendum to `vla-initialization.md` (+18.1
pre-freeze adaptation cell — sharpens the honest prior on #17's
thawed-vs-frozen read). queue.json: instrument → done, execution →
launch-only, **+idea17-vu5k-finalization-prep** (CPU carve-out of
the held execution item; depth 2 green).

**Next**: `queue_cli.py next` → **idea19-tsens-dt-read-execution**
(opens at rungs completion ~00:3xZ 08-08); GPU-busy windows →
**idea17-vu5k-finalization-prep** (CPU, wanted before the molmo2
endpoint ~04–05Z 08-08). Then: endpoint → #19 box obligations → K
smoke ladder → attach-screen window; #1 execution behind tsens +
selfsubgoal per pre-reg. **Every GPU launch goes through
`run_detached.sh`.**

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
`--init-from` confirmed → amendment 2; (4) 18:43Z batch/reheat/
warmup-500 questions + 18:49Z "2e-6 seems kind of small" →
recommendations posted (batch 48 unchanged, 0.3× reheat, warmup
500, vision = text = 6e-6), owner **"Ok, agreed" 18:51Z** →
**amendment 3 landed same session** (Space-verified live); (5)
18:51Z golden-ticket follow-up (per-dataset tickets? rig
inference-time use? search mechanics?) → replied 18:5xZ: per-dataset
matrix is free from stage 1's dump (R4), record-only pending
per-dataset confirms (selection noise + multiplicity), rig ticket =
constant [50,6] tensor searched offline on rig data (offline-vs-
rollout caveat stated), search = batched draws-64 random search.
Exchange may continue — chained session rejoins via `history`.

**Done**: tick — **#17 amendments 2 AND 3** (A2: 5k steps/arm,
`vu5k` naming incl. eval stems, gate 24→32 GPU-h with recomputed
arm costs 12.2/13.9; A3: batch 48 unchanged, LR reheat 0.3× the
40k peaks — decoder 3e-5 / text 6e-6 fresh 5k cosine to 10%
floors, `--warmup-steps 500`, vision LR 6e-6 tied to the text
group — every constant owner-agreed in-channel 18:51Z); recovered + re-verified
the 18:24 session's uncommitted 5-vs-3 group-count correction
(`bijou/train.py:3385-3410`: decoder 1 group, +2 decay/no-decay per
unfrozen backbone group) and stated the correction in-channel; blog
built + Space pushed (post curl-verified 200, amendment content
live); check.py 460 green; queue validate green (depth 2, 14 open);
`run_work_next` armed. Three Discord posts (explainer, lock-in,
amendment confirmation).

**Next**: #17 design is now settled through amendment 3 →
finalization amendment only (byte-audit + memory-ladder smoke +
endpoint-probe quote + `vu5k` launchers) + owner go, window
post-attach-screen. GPU-busy windows →
`idea1-golden-ticket-instrument` (CPU). tsens dT read opens ~00Z;
molmo2 endpoint ~08-08 morning → #19 box obligations → K smoke
ladder → attach-screen window. **Every GPU launch goes through
`run_detached.sh`.**

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

Session 19:38–19:4xZ: quiet babysit tick, 0 GPU-h new (tsens +
molmo2 accruing under their own gates) — both runs green (molmo2
28380/40k probe 6.88@28000; t0.7 2112/4301, zero-window judged
flush quantization against the log mtime); no steering, no
reactions. Corrected the prior session's ~40-min-fast timestamp
labels (now.md header + queue.json `updated_utc`); `run_work_next`
left armed for idea17-vu5k-finalization-prep. No blog build
(now.md only).

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

