# Utilization ledger rebase — 2026-08-17 (receipts)

Queue item `utilization-ledger-rebase`, work session 19:20–2x:xxZ
08-17. The now.md footer's trailing-7-day baseline was dated
2026-08-06 23:3xZ (11 days stale) with a per-run "since then"
narrative accreted onto it. This note is the per-run reconciliation
behind the fresh figure; the instrument that dumped the session-note
figures is `fontaine/scripts/util_ledger_extract.py` (one-shot, over
the now-archive pages + now.md).

**Window**: 2026-08-10 00:00Z → 2026-08-17 19:45Z (the queue item's
"08-10 onward"; the only window-crossing runs are pro-rated at the
08-10 edge using their own babysit accrual readings, and the live
discriminator is counted at the stamp).

**Sources**: babysit registry prune records
(`fontaine/harness/babysit.toml`, per-run totals — authoritative for
detached runs, since tick notes log "0 new" while units accrue) +
archive session notes for foreground/probe spends that never had a
registry entry. Owner's own usage of either machine (box 08-12
reservation, local owner-reserved windows 08-14/15) is not ours and
is not counted.

## Local H100 (~80.2 GPU-h total in-window)

| run / spend | GPU-h | note |
|---|---|---|
| tiny10k train+eval (in-window part) | 5.4 | total 9.3; 3.9 accrued pre-08-10 (babysit read "4.0/15" at 00:13Z) |
| er15k panel eval | 2.5 | |
| molmoact2_sweep | 1.3 | |
| rig_ft preflight+smoke / rig_ft_r1 | 0.1 / 2.7 | |
| er35k aux+panel evals | 2.2 | registry combined figure |
| molmoact2 port parity reads | 0.7 | |
| molmoact2_ae_ours | 1.9 | |
| er55k_panel | 2.2 | local (box still training; 09:42Z tick confirms) |
| er60k_events_dump | 1.55 | |
| sim100b all phases | 5.5 | |
| sim foreground probes 08-12 | 0.2 | 0.02+0.12+0.06 |
| spot20 3-arm | 1.25 | |
| sim_parallel_oracle | 0.25 | FAIL leg, gate decided |
| ftrig_eval20 + flip_parallel | 0.51 | 0.15 + 0.36 |
| release_officialmap | 0.25 | |
| seed6_30s + release_100ep | 0.88 | |
| grpo_signal_probe | 3.57 | run total, 08-12 21:39→08-13 01:08 |
| molmoact2_ar100 A+B (+smoke) | 2.25 | |
| grpo_phase2 R0 (4 launches) | 3.8 | registry cum vs 5.5 gate |
| sim gate probes 08-13 | 0.25 | six ~0.02–0.1 embeds |
| grpo_phase2 r0a + r1a + r1b | 8.07 | 2.12 + ~3.0 + ~2.95 (ladder cum ~8.1 ✓) |
| 08-14 embeds/misc + stage-0 parity | 0.45 | |
| wrist_screen_stage1 | 3.1 | |
| grasp stage-A reads | 0.6 | |
| grasp stage-B collect | 4.0 | |
| grasp stage-C AR train (owner kill @2040) | 2.7 | |
| step2000 two-arm probe | 3.4 | |
| train256 eval extra 08-15 | 0.5 | the non-overlapping part of the "~1.8 in-session" note |
| grasp_sft_joint_corrected train | 5.7 | |
| route-C probes + eval256 + smoke (08-16) | 3.6 | 3.3 in-session + 0.3 leg-3 partial (owner pause) |
| norm_audit_legs | 1.1 | |
| run-1b sim20 (flow-regression) | 0.5 | |
| sft_v1_eval_chain (3 legs) | 6.2 | |
| discriminator (LIVE at stamp) | ~1.0 | launched 18:44Z, single GPU, accruing |
| **local total** | **~80.2** | experiments ~80.0 (ops-only: rig preflight 0.1, smoke fractions ~0.1) |

## Box (~254 GPU-h total in-window; FINAL — box killed by owner 08-17 ~15:xxZ)

| run | GPU-h | note |
|---|---|---|
| er_60k (4-GPU, in-window part) | 147.4 | total ~153 incl. evals; 5.6 accrued pre-08-10 (babysit "5.6/155" at 00:13Z) |
| demo-gen v1c (08-16) | 16.9 | incl. ~0.5 smokes |
| grasp_sft_v1_joint launch-1 | 3.7 | incl. ~2.5 eval-crash loss |
| grasp_sft_v1_joint run-1b (+smoke) | 14.7 | killed 21:07Z 08-16 |
| grasp_sft_v1_joint run-2 | 31.0 | registry wall×8 |
| sft-v1-sim100 eval | ~5 | ESTIMATED — never gate-logged (sharded 4×25 ×2 legs, ~1.2 h wall); the one soft figure here |
| demo_gen_v2 | 17.8 | |
| grasp_sft_v2_joint_8xa100 (owner kill) | 2.6 | |
| grasp_sft_v2_demosonly_8xa100 (owner kill) | 4.0 | |
| grasp_sft_rigonly_8xa100 | 10.5 | |
| **box total** | **~253.6** | experiments ~250.3 (ops/loss: eval-crash 2.5, smokes ~0.8) |

Precision class: local ±2, box ±5 (the sim100 estimate + er_60k
pro-rate). Owner kills are counted as experiments (measured claims
came out of every one). Next rebase: rerun the extract script, take
prune records for anything detached, and pro-rate at the new window
edge from babysit accrual readings exactly as above.
