# Now

















*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-18 13:44–15:3xZ (real `date -u` at write: 15:28) —
work session (chained, bounded): **sim-clutter-patch-promotion
EXECUTED + its registered re-gate PASSED same session — real-crop
clutter patches are now the production v3/v4 clutter appearance.
Session then rode to the step-1000 drift-guard read in-turn: PASS.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm` LIVE — babysit exit 0 at
15:26 (drift read): step 1000/3000, 5 procs, util 84%, VRAM 62.21/71
gate stable through save@1000; **drift guard PASS: Δ eval(1000−500) =
−2.13 vs ≤ +0.30** (8.2419@500 → 6.1069@1000; curve
12.91/8.24/6.65/6.11 vs disc 12.51/7.57/6.59/5.90 — healthy shape,
slightly above throughout). Host RAM 46 GiB available (vs 48 at
13:17) — no second step-drop after save@1000, watch routine. ~8.4 h
to endpoint, ETA ~23:5xZ. Queue green depth 3 (16 open).

**Steering**: none — read + inbox empty at boot (13:45) and the 14:19
poll; no owner reply yet to the triage posts.

**Done** (commits `9fe3ead` promotion + `0c0a8fb` re-gate; post
1539282325907570728): the 13:3x GO executed end to end.
`sim/clutter_patch.py` (moved from fontaine/scripts, camera model
inlined + oracle-pinned); `SO101Sim clutter_appearance` knob, default
**patched** — crops pasted onto the drawn plate at `_draw_content`,
stand-ins parked off-frustum (dropped from top render/mask/v4
shadow), wrist path untouched, zero extra RNG draws (slot-pairing
survives). Oracles +5 in `tests/test_sim_appearance.py` (wrist
bit-exact patched-vs-standins v3+v4; top bit-exact outside
clutter-affected px; physics/stream identity; camera-model freeze);
check.py 1048 green + 3 gpu render legs. **Re-gate PASS**
(`sim_clutter_promotion_regate.py`, ~0.02 GPU-h beside the live run):
production patched AUROC 0.554 vs gate 0.556 (dev −0.002, bar
±0.010), standins anchor 0.713 dead-center — patched substrate
CLEARED for behavioral evals. Sequencing: pdnorm prereg **Amendment
1** pins tonight's sim100 legs to `--clutter-appearance standins`
(11/100 baseline + demos are stand-ins-era); pin also in the endpoint
queue item + archived gate instruments; er_60k probe ckpt
re-converted to current schema (`er_60k_step_060000_vla_v2`).
**Drift-guard read** ridden in-turn (foreground hold to eval@1000,
posted 1539294445630005277): PASS, no knob moves (READ not kill).

**Next**: `queue_cli.py next` CPU pointer →
**expert-approach-quasistatic-redesign** (chained session).
**pdnorm-endpoint-close** at step 3000 (~23:5xZ, sim100 pinned
`--clutter-appearance standins` per Amendment 1);
**grasp-sft-bootstrap** token legs 3/4 after it. Routine babysits own
the interim (host-RAM watch routine). `run_work_next` ARMED at
close.*

*Updated 2026-08-18 13:41–13:4xZ (real `date -u` at write: 13:44) —
tick: **quiet babysit ~2 min after the digest session closed — no
delta; pdnorm healthy.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm` LIVE — babysit exit 0
at 13:41: 5 procs, step 590/3000, loss 0.6241, instantaneous rate
4.0 steps/min (= 15.0 s/step, on plan; the CLI's 18.1 s/step window
figure spans the eval+save@500 pause), VRAM 62.21/71, probe
8.24@500 unchanged. Host RAM available 48 GiB — same plateau as the
13:17 investigation, no further drop; re-check stays armed for the
~15:2xZ drift-read tick. Queue green depth 4 (17 open).

**Steering**: none — read + inbox empty, history shows no new
reactions or owner reply to the 13:33 triage posts.

**Done**: babysit CLI (facts above), Discord poll, queue validate,
`free -g` re-glance. No post (nothing new).

**Next**: unchanged — ~15:2xZ tick owns the step-1000 drift-guard
read (bar eval@1000 ≤ 8.5419, PROVISIONAL) + the RAM re-check;
endpoint battery ~23:4x–00:0xZ. `run_work_next` ARMED (marker
present, 13:40) — chained work session owns
**sim-clutter-patch-promotion**.*

*Updated 2026-08-18 13:24–13:5xZ (real `date -u` at write: 13:39) —
work session (chained, bounded): **owner-pending-decisions-digest
EXECUTED — all 20 blocked/owner-hold items re-triaged under the
10:25Z delegation: 4 closed, 3 unblocked, 6 owner-holds converted to
Fontaine-decides deferrals, 5 genuinely owner-owned digested
in-channel.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm` LIVE — babysit exit 0
at 13:38: liveness 5 procs, step 580/3000, loss 0.6329, 16.1 s/step
over the eval+save@500 window (~10.8 h to endpoint, ETA
~23:4x–00:0xZ), VRAM 62.21/71 gate, probe 12.91@250 → 8.24@500 vs
disc 12.51/7.57. Queue green depth 4 (17 open).

**Steering**: none — read + inbox empty at boot (13:25) and at the
13:38 babysit poll; no owner reply yet to the triage posts.

**Done** (commit `7cdb533`; posts 1539265964892360786 decisions +
1539265974660890717 owner digest): the digest item executed end to
end — every blocked/owner_hold item re-triaged under the no-go-asks
delegation. Closed 4: pdnorm-run item (superseded-by-execution),
token-SFT arm B (superseded by the owner's route-C pick),
photometrics default flip (**decided NO-GO** — wrist manipulation
regression CI-excl-0 + top gain absorbed next to clutter), sim100 v3
rerun (**superseded-by-outcomes** — joint-2k's 44/100 answers the
≥1/100 goal; 2–4 GPU-h saved). Unblocked 3: grasp-sft-bootstrap
remainder (joint-ckpt **token probe legs 3/4**, GPU-free-gated
behind tonight's endpoint; the 08-16 GPU-pause hold was stale),
clutter-patch promotion (**decided GO** — the payload of the visual
stack), quasistatic approach redesign (**decided GO** — executes the
owner's own 19:42Z 08-16 smoothness ask); also decided GO on the
v1.1 realcal disk exemption inside its still-blocked item. Converted
6 owner-holds to my deferrals with stated reopen conditions
(rig-mixture C-defer, img280, F-then-joint rung, renderer PBR pilot,
lens refit relabeled technical, v1.1 regen). Owner-owned digest
posted: molmo_flow lane ×2, released-stats fix (owner-taken), box
provisioning, wandb key rotation. check.py 1045 green.

**Next**: `queue_cli.py next` → **sim-clutter-patch-promotion**
(CPU, ~1 session; registered probe re-gate before behavioral moves),
then **expert-approach-quasistatic-redesign** (CPU). Ticks own the
~15:2xZ step-1000 drift read (bar eval@1000 ≤ 8.5419, PROVISIONAL) +
the host-RAM re-check. **pdnorm-endpoint-close** at step 3000
(~23:4x–00:0xZ); **grasp-sft-bootstrap** token legs 3/4 after it.
`run_work_next` ARMED — GPU busy, CPU queue non-empty.*

## Utilization footer

Session 2026-08-18 13:44–15:3xZ (work, bounded, exploit-side; ~0.02
GPU-h in-session (re-gate embeds) — pdnorm train continues, ~4.4 h
elapsed at close): **clutter-patch promotion EXECUTED + production
re-gate PASS same session (patched 0.554 vs gate 0.556, standins
anchor 0.713; commits `9fe3ead`+`0c0a8fb`); tonight's sim100 pinned
standins via prereg Amendment 1; step-1000 drift read ridden in-turn
— PASS (Δ −2.13 vs ≤ +0.30 bar)** — `run_work_next` ARMED:
quasistatic redesign next; endpoint battery ~23:5xZ.

Session 2026-08-18 13:41–13:4xZ (tick; 0 GPU-h new — pdnorm train
continues, ~2.6 h elapsed): **quiet babysit 2 min after the digest
session closed — exit 0, step 590/3000, 15.0 s/step instantaneous,
probe 8.24@500, RAM plateau 48 GiB unchanged; Discord quiet, no
post** — `run_work_next` stays ARMED: clutter promotion next; drift
read ~15:2xZ; endpoint ~23:4x–00:0xZ.

Session 2026-08-18 13:24–13:5xZ (work, bounded, exploit-side CPU; 0
GPU-h new — pdnorm train continues, ~2.6 h elapsed at poll):
**delegation digest executed — 20 items re-triaged: 4 closed (incl.
killing the sim100 v3 rerun, 2–4 GPU-h saved), 3 unblocked (token
probe legs 3/4 gated behind tonight's endpoint, clutter promotion
GO, quasistatic redesign GO), 6 owner-holds → Fontaine-decides
deferrals, owner-owned digest posted (4 entries)** — `run_work_next`
ARMED: clutter promotion next; ticks own the 15:2xZ drift read;
endpoint battery ~23:4x–00:0xZ.

Trailing-7-day GPU-hours on experiments / total (window 2026-08-10
00:00Z → 2026-08-17 19:45Z; rebased 08-17 from per-run prune records
+ archive session notes — receipts in
`fontaine/notes/utilization-rebase-2026-08-17.md`, instrument
`fontaine/scripts/util_ledger_extract.py`): local **~80.0 / ~80.2**
(incl. the discriminator at ~1.0 in-window; run COMPLETE 08-18
00:42Z at ~5.8 total — post-window ledger row landed in the 00:49
work-session note above, ~4.8 rolls into the next window), box **~250 /
~254 FINAL** (box killed by owner 08-17 ~15:xxZ; er_60k pro-rated
~147 in-window of its ~153; sim100 eval ~5 is the one estimated
figure). Older dated snapshots and session notes: rolled verbatim to
the [now archive](archive/now-2026-08-07.md); the superseded 08-06
baseline + its accreted narrative: rolled verbatim to
[archive 08-17](archive/now-2026-08-17.md).
