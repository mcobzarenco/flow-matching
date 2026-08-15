# Queue

*Generated from [`fontaine/queue.json`](https://github.com/mcobzarenco/flow-matching/blob/fontaine/fontaine/queue.json) — the canonical queue — by `fontaine/scripts/queue_page.py` (rides every `blog_build.sh`). Do not hand-edit.*

**Updated:** 2026-08-15T17:29:51Z

**Depth call:** depth 3: grasp-sft-bootstrap (retrain arm owner-pending) + grasp-sft-chain-results-page (writing ladder) + grpo-r2-post-sft (re-based per A2; token-sft arm item blocked owner-gated). Recorded 14:48Z 08-15.

**18 open** (Live 0 · Queued 3 · Blocked 15 · Done 179)

## 🔴 Live (0)

*running right now (GPU or owner-window)*

*(empty)*

## 🟢 Queued (3)

*ready — waiting on a window or a boundary*

**`merge-main-phase5b`** · `cpu`

Merge main phase 5b (03c2b27, 'bijou.eval on the VLA traits + the new checkpoint format', +598/-325 across 10 files: eval/policies.py 584-line rework, eval/cli.py, docs/vla-architecture.md, 6 eval-adjacent test suites) into fonta…

**boundary:** Queued 17:3xZ 08-15 tick on seeing 03c2b27 on origin/main. CPU-only, no GPU touch (owner reserve stands). Urgent for the same reason 5a was: the owner arm/route decision may land any time and both the launch AND eval paths must be green against phase 5b first - the probe protocol is the read for whichever arm launches.

<details><summary>full record</summary>

Merge main phase 5b (03c2b27, 'bijou.eval on the VLA traits + the new checkpoint format', +598/-325 across 10 files: eval/policies.py 584-line rework, eval/cli.py, docs/vla-architecture.md, 6 eval-adjacent test suites) into fontaine; re-run check.py; re-verify the EVAL seams our protocols depend on: (1) the step2000-probe eval command (grammar-masked greedy + the three-way band protocol) parses + loads both _vla conversions under the reworked policies.py; (2) --noise-ticket-map / ticket-mode provenance in eval survives the policies rework (rung-2 instrument, 14 oracles in test_ticket_map.py); (3) panel eval commands (--report path, predictions-npz provenance) still parse; (4) babysit/probe-clone augment-0 eval path unaffected; (5) confirm whether the parents[2]-&gt;parents[3] bank_processor_goldens fix landed upstream or the carry still stands. Retrain launch path itself was re-verified green post-5a (17:03Z post) - re-confirm both arms full-parse after this merge too since eval/cli churn can move shared arg surfaces.

</details>

---

**`grpo-r2-post-sft`** · `gpu-local`

GRPO R2 on the grasp-SFT policy (DRAFT pre-reg posted 08-15): fresh Decision-11 run on the first-class stack (bijou/grpo_replay.py over the CONVERTED stage-C endpoint

**boundary:** DRAFT posted 07:5xZ 08-15 (queue-refill slice of the stage-C ride session; stage C launched 07:29:55Z same session). ACTIVATION RULE: stage-D verdict owns this item — GRPO_GO (&gt;=20/100) -&gt; finalization per prereg SS6 (base count + checkpoint receipt + lr decision + setback wire call + HEAD re-pin + objection window) THEN launch; ITERATE_BC_ONCE (5-19) -&gt; the bootstrap's iterate arm consumes the GPU slot first, this item waits; F_TRANSFER (&lt;5) -&gt; PARK this item (visual/renderer lane becomes binding per bootstrap SS4). Fresh budget — the bootstrap &lt;=13 gate does NOT carry over. || AMENDMENT A2 REGISTERED 14:4xZ 08-15 (SS8): token-SFT arm pre-reg DRAFT posted (posts/2026-08-15-prereg-grasp-sft-token-sft-arm.md, per A1 decision 2) — if the owner routes token-GRPO, R2 RE-BASES on that arm's endpoint and the activation bar becomes the arm's primary read (unseen sim100 &gt;=20/100 on the DISCRETE head, greedy decode); the stage-D flow-head verdict no longer activates this item. Table lineage inherited (fast codec normalizes token targets with baked q01/q99 — corrected-base init mandatory). Checkpoint receipt re-spelled per owner main 4fd6875 (VLA format: convert_legacy + validate_checkpoint, stats_note provenance). · [pre-reg](posts/2026-08-15-prereg-grpo-r2-post-sft.md)

<details><summary>full record</summary>

GRPO R2 on the grasp-SFT policy (DRAFT pre-reg posted 08-15): fresh Decision-11 run on the first-class stack (bijou/grpo_replay.py over the CONVERTED stage-C endpoint — the exact bijou dir stage D evaluates), reward v2 trained-on / v1 held-out, 8x8 groups T=1.0, full R1 tripwire set inherited, proposed lr 1e-6 (3e-7 fallback) kl_beta 1.0, 10 steps ~10 GPU-h gate &lt;=12. PRIMARY: paired sim100 vs the banked stage-D base count; wave-0 calibration read on group success-variance (predicted &gt;=60% mixed at p&gt;=0.2). CONDITIONAL: activates ONLY on stage-D GRPO_GO (&gt;=20/100).

</details>

---

**`grasp-sft-bootstrap`** · `gpu-local`

Grasp-rich SFT bootstrap (owner go 22:07Z 08-14 on the 90%-of-seeds question; DRAFT pre-reg posts/2026-08-14-prereg-grasp-sft-bootstrap.md): stage A scripted expert vs privileged sim state (waypoint policy over sim.data object po…

**boundary:** Queued 22:3xZ 08-14 at the owner go (parallel with the wrist screen; screen owns the GPU window first — it is registered FINAL). Stage A scripted expert is the executable CPU slice any session can take. Interplay frozen in draft SS5: F-null/F-flat on the screen drops the stage-C flow arm; F-live inherits the owner's wrist-fidelity decision as a registered amendment before stage B. | STAGE-A WIP LANDED 22:4xZ 08-14 (same session, owner parallelize go): sim/scripted_expert.py + tests/test_scripted_expert.py (5 CPU oracles green — IK reachability over the spawn band, perpendicular jaw alignment &lt;6deg, scratch-data purity, eval-seed refusal at DEMO_SEED_BASE 1000). Engineering smoke (NOT the gate read; 4 demo seeds): two-jaw pinch contact 4/4, boat moved from 9-12 cm spawn to ~5.5-6.7 cm of disk center, NO held lift yet -&gt; no successes. Three mechanisms diagnosed + documented in code: (1) IK must solve in jaw-pad-midpoint space (gripperframe sits cm away); (2) free-wrist IK tips the jaw into the table — wrist locked to the P4 keyframe pitch, roll aligned separately; (3) unregularized DLS picks straight-arm poses whose gravity moment SATURATES the sysid'd shoulder servo (force pinned at 3.478 limit) — nullspace posture pull toward the pickup basin added. CONTINUED 22:5xZ same session, 3 more mechanisms fixed: (4) free_dofs=3 made the posture task inert (square jacobian, zero nullspace) -&gt; wrist_flex freed (4 dofs) and the solve lands in the low-torque basin (shoulder 27 vs 39 deg, residual 0.5 mm); (5) droop integrator gated on settled arm + z-floor (it was folding transient error in and driving the target under the table); (6) retreat re-parked in joint space (the IK swing back re-contacted the released boat and tipped it). MEASURED STATUS (6-seed smoke): pinch+hold 5/6 (two-jaw grip continuous through carry), lift to ~1.6 cm HELD, carry to 5.8-6.3 cm of disk center, 0 successes — the traverse crawl cannot bring the dragged boat inside 3 cm before the phase clock (saturated shoulder caps carry height at ~1.5 cm so the hull drags). REMAINING for stage A: carry-height/drag management (higher lift basin or drag-aware traverse target) + drop precision; then the 20-seed &gt;=70% validation AFTER finalization. NOTE for finalization: the sysid'd servo's saturated-reach envelope is itself a finding — the same static-torque wall the learned policies face at low-forward grasps. || STAGE-A BREAKTHROUGH 23:4xZ 08-14 (b564337): the scripted expert WORKS end-to-end — 10/16 demo-seed successes (engineering smoke, seeds 1000-1015), ~120-175 ticks per success. The carry fix: pure PAN-ARC traverse (pan's axis is vertical = no gravity load, so the lifted posture's carry height survives the swing; pan/world-bearing coupling measured NEGATIVE; lower does the radial trim with shoulder/elbow at the held bearing). Failure classes: 3x tipped-at-release ON the disk (2.0-2.2 cm from center — release polish), 1x alignment miss (seed 1005 class), 2x drop short. Success video posted in-channel (id 1537969484541861948). Remaining before the registered 20-seed &gt;=70% gate read: release polish + finalization (objection window). Stage B collection becomes REAL as soon as the gate read passes. || STAGE-A POLISH 00:0x-01:3xZ 08-15 work session: 14/16 demo seeds (d1b2552 settle-before-release: new settle phase lowers pads to RELEASE_Z=0.026 = grasp pad height + disk top z=0.012 so the keel touches before the jaws open — fixed all 3 tipped-at-release seeds, upright 0.88-&gt;0.9+; 2435a6d deck-strike jam recovery: hull yaws demanding wrist_roll~0 land the moving-jaw shell on the deck, arm jams pressing 22-40 N ~10 cm up — static gravity is only 0.13 of the servo limit, the stall is CONTACT; fix = physical jam detection -&gt; retreat up-and-back -&gt; one retry on the pi-flipped roll branch; kinematic overlap probes tried and rejected, they cannot separate jamming from grazing branches). Remaining misses: seed 1006 (jam press nudges the boat, retry close misses) + seed 1013 (lower-phase timeout, drop 4.9 cm short). 14/16 = 87.5% &gt;= the 70% gate bar on the smoke set — the REGISTERED 20-seed gate read still runs post-finalization (~0.2 GPU-h). Wrist-screen interplay resolved 01:3xZ 08-15: F-INSTRUMENT (not F-null/F-flat), so the draft SS5 flow-arm drop clause does NOT fire; no wrist-fidelity amendment inherited. GPU free 01:32Z — stage A gate read is the next GPU leg once finalization posts. || FINALIZED 01:4xZ 08-15 (758666f, in-channel post 01:43Z): §6 frozen — gate read on HELD seeds 1020-1039 (&gt;=14/20; 1000-1015 declared as the tuning smoke), stage B 400-kept target from seeds 1000+ (gate &gt;=300, &lt;=4 GPU-h), stage C primary molmoact2 rig-ft class action-expert-only LR 5e-5 gb64 3000 steps, flow arm retained (screen F-instrument != F-null/F-flat) conditional on &lt;=13 gate, convention seam = recomputed dataset table / rig-frame identity / no shim in B-D with state_units provenance. OBJECTION WINDOW OPEN from the 01:43Z post: GPU stages launch at the NEXT work-session boundary absent objection; owner go collapses the window. Stage-A gate read (~0.2 GPU-h, rendered, videos banked) is the first GPU leg; GPU is free (screen closed 01:32Z). || STAGE-A GATE READ EXECUTED 02:0x-02:1xZ 08-15 (this WAS the post-finalization boundary; instrument e371e2b, HEAD provenance in reports/analysis__grasp_sft_stageA_gate.json, 20 videos banked outputs/sim/grasp_sft/stageA_gate/): **GATE FAIL 11/20** vs &gt;=14/20 — the held stream caught tuning-smoke overfit. Integrity: rendered == unrendered bit-identical on 3 re-run seeds; miss taxonomy measured (4x lower radial stall ~4.7 cm quiet, 3x mid-carry grip loss, jam-on-both-branches tail). Boundary post 02:14Z with owner options; record-only: 11 clean grasp-lift-place runs contradict F-physics' mechanistic reading (sim hosts the grasp; gap = expert coverage). || AMENDMENT A1 REGISTERED 02:33Z 08-15 (prereg §7, in-channel post 1538012809114161222): robustness pass 77776fd diagnosed+fixed on the now-burned 1020-1039 (lower place-droop, re-grasp recovery, jam-flip budget 3, retry-scoped dwell+droop reset) — 11/20 -&gt; 16/20 burned, 15/16 smoke, no regressions. FRESH held gate set 1040-1059, protocol verbatim §6, &gt;=14/20, ONE amendment only (second FAIL fires §4 F-physics/Squint with no further tuning). Window: fresh read at this session's close (&gt;=30 min from the post, ~03:0xZ+) absent objection; stages B-D unchanged, next-boundary launch if PASS. || A1 FRESH GATE READ 03:0x-03:1xZ 08-15 (window closed 03:04Z no objection; HEAD 784a686): **GATE PASS 15/20** (bar &gt;=14) on held seeds 1040-1059 — reports/analysis__grasp_sft_stageA_gate_a1.json + 20 videos banked (outputs/sim/grasp_sft/stageA_gate_a1/); 75% fresh vs 80% burned = the robustness pass generalized; 5 misses all the known persistent-jam tail, no new class. STAGE A CLOSED (amendment spent, ladder capped): sim hosts the grasp, no F-physics/Squint. In-channel PASS post 03:13Z (id 1538022772905873439). || STAGE B LAUNCH-READY: collector landed 5b360fa (sim/collect_demos.py, 4 oracles + GL smoke 2 real demos round-tripped). Launch AT THE NEXT SESSION BOUNDARY absent objection per A1: systemd-run --user detached unit, `MUJOCO_GL=egl uv run python -m sim.collect_demos --out ~/datasets/fontaine/grasp_sft_demos_v0 --target-kept 400 --max-wall-hours 4` (resume-capable via collect_state.json), babysit entry at launch, gate &gt;=300 kept / &lt;=4 GPU-h; expected ~3.4 GPU-h at the measured 75-80% keep rate. Then stage C per frozen §6 (rig-ft class AR primary + flow arm), stage D sim100. || STAGE B LAUNCHED 03:29:18Z 08-15 tick: owner 👍 on the 01:40Z finalization post surfaced at the tick's history poll (explicit go, window collapsed per the frozen terms); unit fontaine-grasp-sft-stageb via run_detached.sh (MUJOCO_GL=egl collect_demos --out ~/datasets/fontaine/grasp_sft_demos_v0 --target-kept 400 --max-wall-hours 4), babysit entry live (gate &gt;=300 kept / &lt;=4 GPU-h, progress-log kind, log /home/ubuntu/collect_grasp_sft_demos_v0.log). First poll 03:30Z GREEN: seed 1000 KEPT 129 ticks 2.5 cm in ~40 s incl startup, GPU 50%/909 MiB (render+encode). Boundary session owns: keep-rate + provenance reads at DONE, then stage-C launch per frozen §6 (rig-ft class AR primary, 3000 steps). || RIDE 03:37-07:0xZ 08-15 work session: pace drop diagnosed (mid-ride post 04:13Z, prereg §8 record): TRUE expert rate 62.5% (125/200 CPU-side n=200, seeds 1078-1277), gate reads were n=20 optimism; no collector bug (6/6 misses reproduce exactly fresh), no spawn drift, known jam/pinch-miss taxonomy only. Wall projection ~295-306 kept vs the &gt;=300 gate — BORDERLINE. Ride left untouched per frozen terms. STAGE-C LAUNCH PREP ALL LANDED (f5b120d): launcher launch_local_molmoact2_grasp_sft_stagec_ar.sh runs its own preflight (refuses unless DONE + provenance + kept&gt;=300 + seeds&gt;=1000 + state_units identity). WALL-TICK BOUNDARY INSTRUCTIONS (wall self-stop 07:29:18Z, '[collect] DONE (wall...)' line + provenance banked at finalize): (1) read collect_state.json kept count; (2) if kept &gt;= 300 → run the stage-C AR launcher (it preflights), activate the PREPARED grasp_sft_stageC_ar babysit entry (fill started_utc), first-poll util/rate/vram checks, in-channel launch post; (3) if 290-299 → the priced RECORDED TOP-UP (anchor + §8): post the decision in-channel FIRST, then resume via run_detached.sh fontaine-grasp-sft-stageb-topup 'MUJOCO_GL=egl uv run python -m sim.collect_demos --out ~/datasets/fontaine/grasp_sft_demos_v0 --target-kept 300 --max-wall-hours 0.5' (resume-capable, banks + re-finalizes provenance), THEN stage C per (2); (4) if &lt; 290 (well under the measured-rate model) → diagnose before anything. Objection window on the top-up path has been open in-channel since 04:13Z. ||| WALL BOUNDARY EXECUTED 07:29-07:3xZ 08-15 (chained work session): stage B DONE at the 4h wall — 313/400 kept of 486 attempted (64%), gate &gt;=300 GREEN, NO top-up needed, provenance banked. STAGE C AR LAUNCHED 07:29:55Z per instruction (2): preflight PASS (313 ep / 54101 frames ~3.5 epochs, state_units seam green, seeds &gt;=1000), unit fontaine-grasp-sft-stagec-ar, babysit entry ACTIVE (started_utc filled). First poll green: step 20/3000 at 5.2 s/step (rig-ft r1 pace), ETA ~4.3 GPU-h vs the 5.0 stage gate, GPU 100%/38.9 GiB, action_flow_loss 0.4641@20. Endpoint ~11:5xZ -&gt; stage D sim100 via the PREPARED convert+eval launcher (frozen verdict &gt;=20 GRPO_GO / 5-19 ITERATE_BC_ONCE / &lt;5 F_TRANSFER). In-channel launch post id 1538088375716683826. Chain spend ~0.9(A)+4.0(B), C projects ~4.3 -&gt; &lt;=13 gate comfortable incl. the conditional flow arm. ||| STAGE-C ENDPOINT BOUNDARY INSTRUCTIONS (recorded 09:3xZ 08-15 during the ride; ANY session executes mechanically, credit-429 resilience): endpoint ~11:2x-11:5xZ = step 3000 logged + checkpoints/finetune/fontaine_grasp_sft_stagec_ar/step3000 saved + unit fontaine-grasp-sft-stagec-ar inactive. THEN: (1) ./fontaine/scripts/launch_local_grasp_sft_staged_eval.sh convert 3000 (foreground CPU, two-hop -&gt; ~/checkpoints/molmoact2-grasp-sft-stagec-ar-step3000-hf -&gt; ~/checkpoints/converted/molmoact2_grasp_sft_stagec_ar_step3000; verify read_checkpoint_info per launcher note); (2) ./fontaine/scripts/run_detached.sh fontaine-grasp-sft-staged-eval ./fontaine/scripts/launch_local_grasp_sft_staged_eval.sh eval100 (launcher refuses if GPU &gt;1 GiB busy; sequential rollout_sim frozen seeds 0-99 euler-10, log /home/ubuntu/eval__grasp_sft_stageD_ar.log, ~1-1.5 GPU-h); (3) babysit.toml: uncomment PREPARED grasp_sft_stageD_eval entry + fill started_utc, prune stage-C entry with DONE note; first-poll util+seed-rate; in-channel launch post; (4) during the eval ride: uv run python fontaine/scripts/upload_grasp_sft_stagec_delta.py --hf-dir ~/checkpoints/molmoact2-grasp-sft-stagec-ar-step3000-hf (oracle-tested f51eebc, weights-only delta -&gt; fontaine-checkpoints, standing rule); (5) at eval DONE: grasp_sft_staged_reads.py runs inside the launcher -&gt; reports/analysis__grasp_sft_stageD_sim100.json banks the FROZEN verdict (&gt;=20/100 GRPO_GO -&gt; grpo-r2-post-sft finalization per its SS6; 5-19 ITERATE_BC_ONCE -&gt; one B/C round, GRPO item waits; &lt;5 F_TRANSFER -&gt; park GRPO, visual lane binding); (6) verdict boundary post owns the read; THEN fill the results page stage-D section + re-render charts (fontaine/scripts/grasp_sft_chain_charts.py --extract &amp;&amp; render; page posts/2026-08-15-grasp-sft-chain-results.md DRAFT-live since 2439869), blog build + Space push. Flow-arm call: OPTIONAL per frozen SS6 budget clause — decide AFTER the AR verdict banks (chain ~0.9+4.0+~4.1+~1.5 = ~10.5 of 13; flow train ~1.3 + flow eval ~1.5 fits ONLY if the verdict makes the comparison decision-relevant; F_TRANSFER reading makes the flow arm moot per SS5 interplay). |||| OWNER OVERRIDE 10:10:20Z 08-15 (supersedes the stage-C endpoint instructions above): stage-C run KILLED at step 2040 (loss 0.0246, green, ~2.7 GPU-h; step500-2000 kept) on owner order; owner ALSO registered 10:07Z: their train_lerobot.py RETIRED — all future training via bijou.train/first-class stack. step2000 converted (~/checkpoints/converted/molmoact2_grasp_sft_stagec_ar_step2000). LIVE: two-arm probe unit fontaine-grasp-sft-step2000-probe (unseen 0-99 then train band 1000-1099; babysit entry grasp_sft_step2000_probe). REPORT OWED (owner ask): success on training seeds (kept subset of 1000-1099 vs reports/curve__grasp_sft_stageb_collect.json) vs unseen seeds, updates in-channel as arms land. STAGE-D FORMALISM + flow-arm + GRPO-R2 activation ALL SUSPENDED pending owner re-steer after the probe read; R2 draft amendment owed (head seam: token-GRPO trains the discrete head stage-C never touched — owner-confirmed direction: token-SFT arm via bijou.train would precede token-GRPO). The delta-upload script (f51eebc) applies to whichever checkpoint the owner banks (adjust --hf-dir/--dest for step2000). ||||| ARM-1 BANKED 11:51Z 08-15: UNSEEN 0-99 = 28/100 successes (42 moved, mean prog +1.97 cm, 0 strikes) vs anchors ftrig4k ~1 / W0 2 — reports/analysis__grasp_sft_step2000_probe.json, videos outputs/sim/grasp_sft/step2000_probe/unseen/. Posted 1538153322215899198. TRAIN ARM LIVE (seeds 1000-1099, ETA ~13:5xZ, detached unit fontaine-grasp-sft-step2000-probe). NEXT SESSION REMIT: (1) at train.json: re-run fontaine/scripts/grasp_sft_step2000_probe_reads.py (kept-subset split automatic), post the train-vs-unseen comparison in-channel; (2) mid-morning context: quantile class bug FIXED (rewrite_quantile_stats in collect_demos + oracle; dataset stats.json corrected + re-uploaded to fontaine-sim) — the step2000 checkpoint remains trained-on-corrupt; (3) RETRAIN via bijou.train on the corrected table is the value-unlock, OWNER-GATED (their-trainer retired 10:07Z, memory bijou-train-only); prep = port the AE-SFT recipe to bijou.train flags + pre-reg; (4) probe report page / results-page update once both arms banked; (5) step2000 checkpoint upload via upload_grasp_sft_stagec_delta.py --hf-dir ~/checkpoints/molmoact2-grasp-sft-stagec-ar-step2000-hf --dest molmoact2_grasp_sft_stagec_ar_step2000 (NOT yet run — GPU/CPU free next session). || RETRAIN FEASIBILITY PINNED 12:0xZ (this session, CPU read): bijou.train supports the whole route first-class — --objective flow (AE retrain), ar (the token-SFT arm, needs --backbone-text-lr), joint (L_flow + lambda L_CE). SEAM (train.py save_checkpoint region ~2245): molmo_flow NORMALIZES with the SOURCE CHECKPOINT's baked q01/q99 tables, NOT --train-data stats — a naive --init-from of any existing conversion inherits the CORRUPT table. Retrain prep therefore = (a) build corrected norm_stats.json for tag so100_so101_molmoact2 from the FIXED dataset stats.json (exact quantiles), (b) bijou.convert_molmoact2 the released base HF dir with that table, (c) bijou.train --init-from that conversion (expert_init inherit for warm AE / fresh for clean), --train-data ~/datasets/fontaine/grasp_sft_demos_v0. Pre-reg before launch; owner go required. ||||| PROBE COMPLETE + REMIT DISCHARGED 13:4xZ 08-15 (work session 12:42): train arm banked — FINAL three-way: trained-kept 9/64 (14%), expert-failed 9/36 (25%), unseen 28/100 (28%) — NO memorization signature (inversion ~2 SE, suggestive); reports/analysis__grasp_sft_step2000_probe.json + 200 videos; comparison posted 1538180830470602903; probe ~3.4/4.0 GPU-h, 0 strikes; babysit entry pruned. Remit items DONE: (5) step2000 delta uploaded (590/705, fontaine-checkpoints/molmoact2_grasp_sft_stagec_ar_step2000); (3) retrain prep LANDED 75a0379 — build_corrected_norm_stats.py (5 oracles) -&gt; ~/checkpoints/norm_stats_grasp_sft_v0_corrected (wrist_roll [35.5,94.4]-&gt;[+-157.2]), base converted with --norm-stats-from -&gt; ~/checkpoints/converted/molmoact2_base_corrected_stats_v0 (corrected rows verified baked); pre-reg DRAFT posts/2026-08-15-prereg-grasp-sft-retrain-corrected-table.md posted in-channel; (4) probe section + probe_bands chart live on the chain results page. OWNER STEERING 13:35Z: GPU is THEIRS now ('Nothing right away, I'll actually need the gpu') — NO launches until they free it; finish-ping posted at the boundary as asked. RETRAIN DECISION PENDING with owner: continue-from-2k under corrected table (my recommended primary, Q1 reply 1538178705095008267: convert step2000-hf --norm-stats-from corrected, bijou.train --init-from --expert-init inherit, ~2.9 GPU-h, expect early loss spike from the I/O rescale esp wrist_roll ~3x) vs from-base per the posted draft; amendment + launch ONLY on owner go AND GPU freed. ||||| PHASE-4 MERGE + SEAM VERIFY 15:5xZ 08-15 (bb0f036): main 3e4fbeb merged into fontaine (image-augment seam ported to modelling/interface.py; upstream FIXTURE_DIR parents[2]-&gt;parents[3] fix carried), check.py 911 green; retrain seams verified post-merge — read_checkpoint_info loads both real conversions, convert_molmoact2 --norm-stats-from + bijou.train --objective/--backbone-text-lr/--init-from/--expert-init intact, convert_legacy smoke on step2000 validate-green; NOTE convert_legacy --replace-stats wants a DatasetStats state-dict (not a norm_stats tag file) so the pre-registered two-hop --norm-stats-from route stays operative for either arm. Launch remains OWNER-GATED (arm pick + route + GPU release), posted 1538209952374595785. · [pre-reg](posts/2026-08-14-prereg-grasp-sft-bootstrap.md)

<details><summary>full record</summary>

Grasp-rich SFT bootstrap (owner go 22:07Z 08-14 on the 90%-of-seeds question; DRAFT pre-reg posts/2026-08-14-prereg-grasp-sft-bootstrap.md): stage A scripted expert vs privileged sim state (waypoint policy over sim.data object pose, CPU + ~0.2 GPU-h validation on 20 NON-EVAL demo seeds 1000+, gate &gt;=70% scripted success else F-physics -&gt; Squint twin tier); stage B demo collection 300-600 kept successes ~2-4 GPU-h (production visual config, eval seeds 0-99 NEVER in demos); stage C SFT via molmoact2 --objective ar (new-stack objective matrix, rig-ft recipe class ~3-5 GPU-h, optional ftrig4k-recipe flow arm); stage D sim100 eval ~1-1.5 GPU-h, frozen primary: &gt;=20/100 successes -&gt; GRPO GO (fresh pre-reg per Decision 11), 5-19 -&gt; one B/C iteration, &lt;5 -&gt; F-transfer (wrist-screen read becomes the binding diagnosis). Worst-case ~11 GPU-h, gate &lt;=13. Finalization (frozen params + objection window + HEAD re-pin) BEFORE any GPU stage; stage A is CPU-executable now.

</details>

---

## 🟡 Blocked (15)

*waiting on a prerequisite, a boundary, or the owner*

**`grasp-sft-token-sft-arm`** · `gpu-local` · **⛔ owner hold**

Token-SFT arm (DRAFT pre-reg posted 14:4xZ 08-15, per R2 Amendment A1 decision 2 owner direction 10:14Z): bijou.train --objective ar --backbone-text-lr 1e-5 (2e-5 registered alt) over grasp_sft_demos_v0, --init-from the corrected…

**boundary:** BLOCKED doubly owner-gated: (1) route choice A/B/C per the pre-reg SS4 (A = flow retrain corrected-table draft, B = this arm, C = one --objective joint run replacing A+B under a registered merge amendment + confound acknowledgment); (2) GPU owner-reserved since 13:35Z 08-15. On route B/C go: finalization (LR decision, anchor-leg decision, HEAD re-pin, objection window) then launch at the next free boundary; endpoint gets convert_legacy + validate_checkpoint (owner main 4fd6875 VLA format) before it pins as R2's base. If the owner re-scopes R2 to the flow head, this item PARKS with the token-GRPO lane. · [pre-reg](posts/2026-08-15-prereg-grasp-sft-token-sft-arm.md)

<details><summary>full record</summary>

Token-SFT arm (DRAFT pre-reg posted 14:4xZ 08-15, per R2 Amendment A1 decision 2 owner direction 10:14Z): bijou.train --objective ar --backbone-text-lr 1e-5 (2e-5 registered alt) over grasp_sft_demos_v0, --init-from the corrected-table base conversion (molmoact2_base_corrected_stats_v0 — MANDATORY: bijou/fast/codec.py normalizes token targets with the baked q01/q99, corrupt table would distort the token stream too), 2000 steps gb64 matching the probed stage-C budget so the flow head's 28/100 is the cross-head row. Eval verbatim the step2000 probe protocol under the ar head's grammar-masked GREEDY decode; primary = unseen 0-99 count vs the R2 activation bar &gt;=20/100; optional base token-head sim100 anchor leg (default run). ~7-8 GPU-h expected, gate &lt;=9.

</details>

---

**`renderer-pbr-wrist-pilot`** · `cpu` · **⛔ owner hold**

Renderer-class wrist pilot (the decision read the 16:4xZ 08-14 brief recommends): re-export ONLY the wrist-visible arm meshes (gripper, camera mount, forearm links) STL-&gt;decimate-&gt;xatlas-UV-&gt;OBJ via the convert_benchy.py pipeline…

**boundary:** Queued 16:2xZ 08-14 at the decision-brief close (from the decision-brief close). OWNER-GATED by design: the brief prices the decision as the owner's call — do not start the asset work without an in-channel owner go on the pilot (it commits tier-2 engineering, not just a read). On go: pre-reg required before the read (bands frozen from the banked run-3 anchors 0.713/0.523/[0.86,0.89] + calibration directional gate); renderer choice recorded in the pre-reg. No launches until the in-channel GPU release; embeds fit any cleared gap.

<details><summary>full record</summary>

Renderer-class wrist pilot (the decision read the 16:4xZ 08-14 brief recommends): re-export ONLY the wrist-visible arm meshes (gripper, camera mount, forearm links) STL-&gt;decimate-&gt;xatlas-UV-&gt;OBJ via the convert_benchy.py pipeline; bake procedural print-layer normal maps (parametric: known layer height + print orientation per part); render the SAME 100 pose-matched manipulation slots (sim_rollout_pose_wrist_read run-3 harness verbatim: episodes 26-49 mid-band, timestamp-exact real decode, er_60k knn5, 150-frame manip reference, episode-disjoint calibration per Amendment 1, directional gate per Amendment 2) through an external PBR path consuming the posed scene, arm layer feeding the existing anchored compositor path where applicable (wrist rides the raw render). PRIMARY: manip wrist AUROC, PBR arm vs banked 0.877 (in-run PRESENT replication gate [0.86, 0.89] on the classic arm). Decides tier-2: material move toward the 0.523 reset band = buy with evidence; null = the relief/light-transport hypothesis is wrong and the whole tier is saved. CPU renders + ~0.02 GPU-h embeds.

</details>

---

**`sim-arm-photometrics-promotion`** · `cpu` · **⛔ owner hold**

Promote arm_photometrics='v1' into production v3/v4 defaults (registered read GREEN 02:1xZ 08-14: v3 0.713-&gt;0.698 CI-excl-0, only_links 0.705-&gt;0.652; commit 4515ab4): flip the default in SO101Sim (+ rollout/eval surfaces that pin…

**boundary:** OWNER_HOLD per the registered rule (pre-reg decision rule: no default flip without sign-off; same contract as clutter-patch promotion 05:40Z 08-13). Asked in-channel 02:1xZ 08-14 with the results post. | STACK READ 10:58Z 08-14: safe to stack with clutter patches (no regression, point estimate still negative) but the banked solo gain is attenuated ~3x and statistically absorbed at n=100 next to clutter — NOT additive as separately sold; stacked value unresolved, bigger-n read priced on request. | ROLLOUT-POSE READ 12:2xZ 08-14: the stack (photometrics+mount) REGRESSES the wrist at manipulation poses — paired +3.99e-07 CI95 [+2.0e-07,+6.3e-07], 22/100 closer (graded surfaces ~3,200 px there vs ~230 at reset; the 08-14 'wrist-neutral' was a visibility floor, not clearance). Flip decision now prices a measured wrist-side cost against the absorbed top-side gain. · [pre-reg](posts/2026-08-14-prereg-sim-arm-photometric-links.md)

<details><summary>full record</summary>

Promote arm_photometrics='v1' into production v3/v4 defaults (registered read GREEN 02:1xZ 08-14: v3 0.713-&gt;0.698 CI-excl-0, only_links 0.705-&gt;0.652; commit 4515ab4): flip the default in SO101Sim (+ rollout/eval surfaces that pin render_style), re-pin the banked v3 anchor 0.713 -&gt; the patched value on the pinned 20x5 probe, extend tests/test_sim_appearance.py style-equality oracles. NOTE: changes wrist-view arm pixels too (the real wrist view also sees the real arm) — flag the wrist knn5 re-read as the cheap post-flip sanity.

</details>

---

**`sim-clutter-patch-promotion`** · `cpu` · **⛔ owner hold**

Promote the real-crop clutter patch paste into production v3/v4: move clutter_patch.py paste into sim/so101_sim.py as the default clutter appearance (patched plate at _draw_content, clutter geoms dropped from the top render/mask/…

**boundary:** Queued 05:4xZ 08-13 at the appearance-pass close. Implementation ~1 session CPU; re-gate on the pinned 20x5 probe (~0.02 GPU-h) before any behavioral eval moves. | STACK READ 10:58Z 08-14: clutter patches carry essentially the whole combined three-flag gain (stack 0.5521 vs patched-alone 0.5561, materials marginal absorbed) — this promotion is the payload; promote first or alone. | ROLLOUT-POSE READ 12:2xZ 08-14: no bearing on this item (wrist rides the raw render; clutter patch is top-composite only) — still the payload, promote first or alone.

<details><summary>full record</summary>

Promote the real-crop clutter patch paste into production v3/v4: move clutter_patch.py paste into sim/so101_sim.py as the default clutter appearance (patched plate at _draw_content, clutter geoms dropped from the top render/mask/shadow; wrist path untouched, zero extra RNG draws so v3 slot-pairing survives), oracle-pinned (wrist bit-exact vs v3; top bit-exact outside clutter-affected pixels; tests/test_sim_appearance.py extension). Gate evidence: legs (b)+(c) PASS 05:4xZ 08-13 (patched 0.556 vs v3 0.713, beats no_clutter 0.576). OWNER_HOLD: no default flip without sign-off (asked in-channel 05:40Z 08-13).

</details>

---

**`sim-joint-pose-lens-refit`** · `cpu` · **⛔ owner hold**

CONDITIONAL follow-up (lens gate read 03:4xZ 08-13): jointly refit the wrist camera pose AND the full lens model (center + curvature) against the 150 pinned real frames

**boundary:** Queued 03:4xZ 08-13 at lens-item close. owner_hold: run only if amendment 6 lands and wrist gap persists; ~0.02 GPU-h per probe read.

<details><summary>full record</summary>

CONDITIONAL follow-up (lens gate read 03:4xZ 08-13): jointly refit the wrist camera pose AND the full lens model (center + curvature) against the 150 pinned real frames. The 08-12 pose re-tune absorbed the real principal-point offset (22 px ~ 2.6 deg yaw-equivalent) under the deployed equidistant lens, so the leg-(a) full fit's center term double-counts it (probe: center-only arm 0.672 vs 0.560 control). A joint fit would let the full lens (plank residual 0.898 px vs 0.937 curve-only) land without the double count. Cheap falsification first: sweep a small yaw/pitch compensation on the existing pose with the full lens, read the same 20x5 probe. Only worth running if the curve-only swap (amendment 6) lands and the remaining wrist gap still reads as geometry.

</details>

---

**`sim100-v1-rerun`** · `gpu-local` · **⛔ owner hold**

100-seed eval v1 rung (successor, pends sim-visual-matching landing): re-run the sim100 protocol (same 100 seeds, same metric/gates, posts/2026-08-11-prereg-sim-policy-eval-100seeds.md conventions) on the v1 matched visuals for e…

**boundary:** Queued 03:4xZ 08-12 at the OOD-probe close. Executable only after sim-visual-matching lands its so101_sim.py visual deltas; the probe re-read (~0.02 GPU-h) is the cheap go/no-go gate before the ~2-4 GPU-h eval. | GATE READ 05:0xZ 08-12: probe re-read MISSED the bar (top 5-NN 0.876 vs &lt;=0.790 target) =&gt; by the registered gate the 2-4 GPU-h rerun does NOT auto-launch. OWNER DECISION OFFERED in the results post: the probe measures encoder separability, not policy behavior - the fisheye+wrist-repose geometry fixes change where things appear in the image, and er60k's reach-over-the-table fingerprint is exactly a pinhole-vs-fisheye spatial-mismatch signature; a 20-seed er60k spot-check (~0.5 GPU-h) would answer it cheaply. HOLDING for owner call: spot-check / full rerun / park behind inpainting. | GATE RE-READ 05:4xZ 08-12: v2 inpainting MET the registered line (top 5-NN 0.773 &lt;= 0.790) =&gt; by the item's own registered criterion the rerun is now GO **with v2 frames** (render_style=v2 default; wrist rides the v1 path inside v2). Still owner_hold: the 20-seed behavioral spot-check ask (05:01Z) is unanswered and remains the cheaper first step; on unhold, the short pre-reg amendment renames arms to v2 visuals and re-baselines. | GATE RE-READ 06:2xZ 08-12: sim-wrist-periphery-fix closed - wrist 5-NN 0.900 -&gt; 0.548 (inside the real spread), top stays 0.773. BOTH cameras now read at-or-under their registered lines; the rerun gate is GO with v2 frames + re-tuned wrist pose. Still owner_hold: spot-check ask (05:01Z) unanswered. UPDATE 07:3xZ 08-12: owner approved the v2-&gt;v3 default flip (07:29Z); gate facts with v3 frames: top 0.673 + wrist 0.548, both under their registered lines. Still owner_hold on the rerun-vs-spot-check call itself. UPDATE 09:1xZ 08-12: spot20 DONE (owner-called) - teacher80k +0.97 cm paired CI-excludes-zero under v3 (direction flipped toward the disk), er60k/snap30k null; behavioral response to visuals CONFIRMED for the engaging arm. Rerun still owner_hold but now both gate legs (visual + behavioral) argue GO; ~6-9 h wall at GPU-compositor pace, ~20-30 min/arm if sim-parallel-rollouts lands first. | AMENDMENT DRAFTED 10:2xZ 08-12: posts/2026-08-12-prereg-amendment-sim100-v3-rerun.md (DRAFT) - launch-ready on unhold after the finalization checklist (arm-set owner call, HEAD/asset re-pin, param sheet + objection window). · [pre-reg](posts/2026-08-11-prereg-sim-policy-eval-100seeds.md)

<details><summary>full record</summary>

100-seed eval v1 rung (successor, pends sim-visual-matching landing): re-run the sim100 protocol (same 100 seeds, same metric/gates, posts/2026-08-11-prereg-sim-policy-eval-100seeds.md conventions) on the v1 matched visuals for er60k + ftrig4k (the only toward-tilted arm) + hold; gate = encoder OOD probe re-read FIRST (did top 5-NN AUROC move from 0.885 toward 0.5? if not, matching did not land — do not spend the eval GPU-h); owner goal &gt;=1 success on the 100 seeds; short pre-reg amendment (v1 visuals = new arm names, re-baseline) before launch.

</details>

---

**`idea4-fjoint-rung-finalize-exec`** · `gpu-box` · **⛔ owner hold**

#4 F-then-joint rung FINALIZE + EXECUTE (per the posted DRAFT posts/2026-08-09-prereg-fjoint-rung.md): (1) CPU instrument DONE 15:0xZ 08-09 (check.py 596 green, 12 new oracles tests/test_fjoint_init.py): materialize_fjoint_init.p…

**boundary:** instrument LANDED 15:0xZ 08-09 (draft Instrument section updated in place — finalization condition 1 satisfied); LAUNCH gated on owner go (draft finalization condition 2) + box free after the adamc_100k endpoint + chained panel (~08-12 ~17:00Z+); sequencing vs the adamc stage-2 frozen attach is an owner call at finalization — do NOT launch on the default | RE-STATUSED blocked/owner_hold 23:0xZ 08-09 queue audit: its own text gates launch on an explicit owner go + box-free; ALSO STALE: 'post-adamc-endpoint ~08-12' predates the adamc owner-kill — box is now er-60k's to ~08-11 ~12:00Z, and F2's '0.92 s/step measured' cost row is attach_F-class (correct for the frozen F2 arm, but re-check the J-arm assumption at finalization). · [pre-reg](posts/2026-08-09-prereg-fjoint-rung.md)

<details><summary>full record</summary>

#4 F-then-joint rung FINALIZE + EXECUTE (per the posted DRAFT posts/2026-08-09-prereg-fjoint-rung.md): (1) CPU instrument DONE 15:0xZ 08-09 (check.py 596 green, 12 new oracles tests/test_fjoint_init.py): materialize_fjoint_init.py composite warm-start materializer (F expert/prompt/trunk bytes + phase-1 tables as joint_ce.safetensors, joint metadata section, trunk-coherence byte-guard refuses a wrong phase-1 source) + --joint-unfrozen-seam guard escape in train.py (warm-start-only: requires --init-from, contradicts --seam-stop-grad, naive-joint refusal verbatim-preserved, molmo2-only runtime guard extended to joint_ce, banner prints seam UNFROZEN) + AR-view compat verified vs J-written checkpoints on the fixture family — WAS: composite warm-start materializer (F@10k + phase-1 rider tables from the 60k endpoint expert.safetensors + config section; oracles: expert bytes == F@10k, rider bytes == phase-1 tables, --init-from --joint-ce round-trip), naive-joint guard escape (opt-in flag; refusal verbatim-preserved without it; negative-control oracle becomes the positive contract), materialize_joint_ar_view.py compat vs J checkpoints on the fixture family; (2) at finalization: owner go + the rung-vs-adamc-attach sequencing question posted in-channel; (3) box execution post-adamc-endpoint (~08-12 17Z+): J-config B12c6 memory smoke 150 steps, then F2 first (0.92 s/step measured) then J (~4.0 assumed from K 3.782 measured), 5k phase ceiling 35 GPU-h, babysit entries at launch; reads via attach_seam_results.py explicit stems + drift AR-view panel

</details>

---

**`docs-pass-followups-0809`** · `cpu` · **⛔ owner hold**

Docs pass tail (from the 08-09 staleness audit, deferred at my discretion): (1) sweep agent-internal vocabulary out of shipped bijou/ source comments (eval/leakage.py 'fontaine/charter.md', eval/subgoal_scoring.py '#6 rung (b)',…

**boundary:** any GPU-busy window; low priority vs stage-2 memo + lit radar | RE-STATUSED blocked/owner_hold 23:0xZ 08-09 queue audit: subitems 1-4 done; the only remainder (5) wandb API key rotation is an OWNER-side action (flagged in-channel at the docs-pass close-out) — nothing for the queue pointer to execute here.

<details><summary>full record</summary>

Docs pass tail (from the 08-09 staleness audit, deferred at my discretion): (1) sweep agent-internal vocabulary out of shipped bijou/ source comments (eval/leakage.py 'fontaine/charter.md', eval/subgoal_scoring.py '#6 rung (b)', train.py 'K arm of the attach-screen'/'#20'); (2) architecture.md S6: enumerate the eval-system surface (frozen sample plans, --dump-draws, noise tickets, --mask-state, subgoal modes, --smolvla baseline, leakage checker) + full rollout flag docs (or rewrite rollout_so101.md properly); (3) S1: a real Molmo2 prompt-format subsection (ChatML, image hoisting, id 151645 bos) instead of the pointer note; (4) confirm docs/notes/2026-08-06 S3 failing-test claim resolved; (5) wandb API key rotation still owed (S8 hygiene note). | PARTIAL 15:5xZ 08-09: (2) architecture.md S6 eval-system surface enumerated (plans/dumps/tickets/mask-state/subgoal modes/smolvla/leakage) + rollout noise/draws/async flags incl. new --noise-ticket, rollout_so101.md flag list updated; (3) S1 Molmo2 prompt-format subsection landed (ChatML hoist, [kind camera|Image i] groups, bos=151645 quirk, native tokenization); (4) 2026-08-06 note's failing-test claim VERIFIED RESOLVED (test passes, note annotated); (1) bijou/ vocab sweep IN FLIGHT via subagent; (5) wandb key rotation still owed (owner-side action) | SUBITEM 1 DONE 15:5xZ (51a692e, subagent sweep + review): 14 bijou/ files de-jargoned (comments/docstrings/help/runtime strings; paths kept; check.py 598 green, no test edits needed). Remaining: ONLY (5) wandb API key rotation — owner-side action, flagged in-channel at close-out

</details>

---

**`actckpt-lineage-flip-ladder`** · `gpu-box`

#20 activation-checkpointing lineage-flip LADDER execution (gpu-box, &lt;= 2 GPU-h gate): run the 4-rung box ladder per 2026-08-09-prereg-actckpt-lineage-flip.md (control B12c6 no-ckpt / ckpt-c6 / ckpt-c1 candidate / record-only max…

**boundary:** blocked on BOTH: (a) attach screen (F then K) off the box, (b) a scheduled fresh non-attach AR-trunk launch to ride (100k continuation / arch-batch arm / #17 vision-unfreeze, whichever the owner green-lights); result rides that launch's pre-reg as a named amendment · [pre-reg](posts/2026-08-09-prereg-actckpt-lineage-flip.md)

<details><summary>full record</summary>

#20 activation-checkpointing lineage-flip LADDER execution (gpu-box, &lt;= 2 GPU-h gate): run the 4-rung box ladder per 2026-08-09-prereg-actckpt-lineage-flip.md (control B12c6 no-ckpt / ckpt-c6 / ckpt-c1 candidate / record-only max-B bisect at &lt;= 71 GiB), 150 steps/rung on the AR-trunk true recipe, median-of-last-100 s/step, frozen decision rule ADOPT iff r2 &lt;= 1.02*r0 AND rung-2 alloc peak &lt;= 63 GiB. FINALIZE the draft (immutability stamp, baselines re-pinned at then-HEAD) when the target launch is scheduled, BEFORE the ladder runs

</details>

---

**`idea17-molmo2-vision-unfreeze-execution`** · `gpu-box` · **⛔ owner hold**

#17 molmo2 vision-unfreeze warm-start two-arm screen EXECUTION (box, 4xDDP, sequential frozen-first)

**boundary:** opens after the attach-screen chain (~08-09+) AND an explicit owner go (owner-steered execution per the 17:04Z steering disposition); needs a free box window and the finalization amendment posted first · [pre-reg](posts/2026-08-07-prereg-molmo2-vision-unfreeze.md)

<details><summary>full record</summary>

#17 molmo2 vision-unfreeze warm-start two-arm screen EXECUTION (box, 4xDDP, sequential frozen-first) — LAUNCH-ONLY-AFTER-SMOKE as of 08-07 19:4xZ (prep item landed the byte-audit + both launchers + prepared babysit entries): remaining finalization cells need the box checkpoint: (1) 150-step thawed memory smoke FROM step_040000 -&gt; ladder rung, write fontaine/harness/state/vu5k_mem_ready (RUNG/BACKWARD_CHUNKS/ACT_CKPT/VRAM_PEAK_GIB/SMOKE_UTC — the thawed launcher refuses without it), quote peak/rate + first async-save 'captured in Xs' line + tower param count ~4.3e8 in the banner; (2) quote the banked 40k endpoint probe -&gt; fill the FILL-AT-FINALIZATION probe bars in babysit.toml vu5k entries; (3) POST the finalization amendment (DRAFT -&gt; posted); (4) owner go; then launch frozen arm via run_detached.sh fontaine-vu5k-frozen, thawed after its endpoint + sanity line (launcher-mechanized ordering); reads frozen in the pre-reg (thawed@5000 - frozen@5000 paired delta, CI95 + null band 0.07, critical-frame re-pool; arm-vs-endpoint record-only); gate 32 GPU-h (amendment 2)

</details>

---

**`arm-a-img280`** · `gpu-box` · **⛔ owner hold**

arch-batch arm A img280 40k (box) — HELD: fresh owner go required before launch

**boundary:** box GPUs occupied by molmo2_ar40k until ~2026-08-08 regardless of hold · [pre-reg](posts/2026-08-06-prereg-arch-batch-1.md)

<details><summary>full record</summary>

arch-batch arm A img280 40k (box) — HELD: fresh owner go required before launch

</details>

---

**`box-home-sweep`** · `cpu` · **⛔ owner hold**

Run tidy_home.py --apply on the box ~ (133 entries, all movable ones owner-era mainline artifacts)

<details><summary>full record</summary>

Run tidy_home.py --apply on the box ~ (133 entries, all movable ones owner-era mainline artifacts) — HELD: charter Loaned-compute READ-ONLY rule; needs explicit owner all-clear (asked in-channel 03:1xZ 08-07)

</details>

---

**`ae-on-our-trunk-prereg-draft`** · `cpu`

AE-on-our-trunk pre-reg draft (CPU): the owner's action-expert implementation (every-layer KV off our AR trunk, exchange 11:56Z 08-11) evaluated/trained against er_60k/step_060000 (the new reference trunk)

**boundary:** UNBLOCKED 17:2xZ 08-11: the rebase landed (main @36afff0 merged, check.py 688 green) — draft against ActionExpertConfig frozen/staticmethod-factory shapes at HEAD. Executable CPU next. DEPRIORITIZED 17:4xZ 08-11 behind the sim lane (owner 17:07Z: next-day focus is simulations). | SUPERSEDED-PENDING-CONFIRM 18:1xZ 08-11: owner design record §8.13 (main @128a863) IS the full AE-on-our-trunk design — its step 7 registers 'molmo_flow-from-scratch on our AR trunk + bijou prompt vs §2.1 at matched compute' with separate pre-regs post-migration. This draft item is absorbed by the molmo-flow migration lane unless the owner says the §2.1-style few-stream AE still wants its own pre-reg. | 18:15Z owner call reinforces: molmo_flow lane is owner-side; this stays absorbed/parked.

<details><summary>full record</summary>

AE-on-our-trunk pre-reg draft (CPU): the owner's action-expert implementation (every-layer KV off our AR trunk, exchange 11:56Z 08-11) evaluated/trained against er_60k/step_060000 (the new reference trunk). UNBLOCKED (rebase closed 08-11 17:2xZ) — main @36afff0 landed ActionExpertConfig frozen/staticmethod-factory shape the draft must target; do not draft config surfaces against the pre-rebase tree.

</details>

---

**`molmo-flow-step1-cli-rule`** · `cpu` · **⛔ owner hold**

molmo_flow migration STEP 1 — CLI inferred-args rule (CPU, per owner design record docs/architecture.md §8.13, main @128a863, approved 08-11): --resume refuses every architecture-determining flag (run-policy flags stay legal); --…

**boundary:** OWNER CALL 18:15Z 08-11 ('your focus is 100% simulations, I have a local agent working on the molmo_flow migration plan'): the migration lane is OWNER-SIDE — do not execute; this item + steps 2-8 parked until the owner hands the lane back or asks for review. Priority question resolved: sim lane 100%.

<details><summary>full record</summary>

molmo_flow migration STEP 1 — CLI inferred-args rule (CPU, per owner design record docs/architecture.md §8.13, main @128a863, approved 08-11): --resume refuses every architecture-determining flag (run-policy flags stay legal); --init-from refuses flags for inherited sections and REQUIRES them for explicitly replaced sections (stage-2 decoder-swap path); sentinel None defaults + one reviewable arch-vs-policy partition table; upgrades 'validate equality if passed' to 'refuse at the door'; prompt-format-change guard falls out as a special case. Gate: resume-with-arch-flag and inherited-section init-from flags error naming the checkpoint value; decoder-replacement init-from still works; flag-free resume of a mainline checkpoint parses unchanged; check.py green.

</details>

---

**`rig-mixture-screen-exec`** · `gpu-box` · **⛔ owner hold**

Rig-mixture screen EXECUTION (pends the owner compute call — pre-reg draft posts/2026-08-11-prereg-er60k-rig-mixture.md posted + in-channel 08-11): finalize the pre-reg (freeze probe bars, panel band, param sheet in-channel, obje…

**boundary:** BLOCKED on the owner compute decision (ask posted in-channel 08-11 ~17:0xZ); unblock by flipping owner_hold when the owner picks A/B or provisions compute. UPDATE 17:4xZ 08-11: option-B preflight FITS (69.2 GiB peak, ~12.0 s/step =&gt; ~33.5 h for 10k steps single-H100; --dataset-repeat live-fired, 4.49% combined share vs ~4.97% pre-reg estimate — reconcile before exec) BUT the owner sim pivot (17:07Z) dedicates the local GPU to inference =&gt; treated as C-defer unless the owner calls A (new box); results posted in-channel 17:35Z. · [pre-reg](posts/2026-08-11-prereg-er60k-rig-mixture.md)

<details><summary>full record</summary>

Rig-mixture screen EXECUTION (pends the owner compute call — pre-reg draft posts/2026-08-11-prereg-er60k-rig-mixture.md posted + in-channel 08-11): finalize the pre-reg (freeze probe bars, panel band, param sheet in-channel, objection window) then run the registered arm: --init-from er_60k/step_060000 (dl from fontaine-checkpoints), --dataset-repeat mcobzarenco/so101_pick_place_clean=27 mcobzarenco/so101_pick_place_v2=27 (~4.97% effective share), 10k steps eff-48, seed 3, warmup 500, save-every 2500, er-60k recipe verbatim otherwise. Reads: primary rig-holdout paired CI95 (1+5 held-out episodes, ~3.7k frames) mixture-endpoint vs er_60k/step_060000 + state-copy anchor; guard k4l2 panel_v2 paired vs banked endpoint npz (fail = worse than +0.05 CI-excl-0); aux/probe record-only. Compute options priced in the draft: (A) new 4x box ~28 GPU-h gate 32; (B) local 1xH100 ONLY after an act-ckpt fit-preflight ladder (full recipe measured structurally OOM single-GPU 08-08); (C) defer. 20%-share variant (x129, ~2.7 rig epochs) named as an owner option.

</details>

---

## ✅ Done (179)

*closed — the full record stays in each fold*

**`merge-main-phase5a`** · `cpu`

Merge main phase 5a (a51b172, 'bijou.train on the family CLI + the VLA checkpoint format', +3115/-2046 across 20 files incl

**boundary:** Queued 16:5xZ 08-15 tick on seeing a51b172 on origin/main (owner-pushed 16:34Z). CPU-only, no GPU touch (owner reserve stands). Urgent: the owner's arm/route decision may land any time and the launch path must be green against phase 5a first. || DONE 17:1xZ 08-15 (work session 16:50): merged 351c56e, check.py 922 green; conflict = probe_unfreeze_gradflow (upstream inlined ProbeArgs, our TrainArgs image_augment line obsolete -&gt; theirs); SEAM FINDINGS: (1) --expert-init RENAMED --flow-decoder-init (inherit=default=same semantics); (2) FORMAT BREAK - new --init-from refuses legacy bijou_config.json conversions -&gt; both migrated via convert_legacy (hard links, validate green): molmoact2_base_corrected_stats_v0_vla + molmoact2_grasp_sft_stagec_ar_step2000_vla; (3) continue-from-2k arm made REAL: fresh convert_molmoact2 step2000-hf --norm-stats-from corrected -&gt; molmoact2_grasp_sft_stagec_ar_step2000_corrected_v1 (new format, trained expert b778bbf2, corrected rows + stats_note baked); (4) BOTH retrain arms full-parse green vs the family CLI (family inferred molmoact2_flow); (5) image-augment p=0 bitwise oracle 11 passed + gradflow probe anchors exact (27.8546) post-merge; (6) parents[2]-&gt;[3] fix NOT upstream (main still parents[2]) - our carry stands, cherry-pick note remains. Pre-reg SS3 amended (--flow-decoder-init + _vla path) + SS8 amendment section. Launch stays owner-gated (arm pick + route + GPU release).

<details><summary>full record</summary>

Merge main phase 5a (a51b172, 'bijou.train on the family CLI + the VLA checkpoint format', +3115/-2046 across 20 files incl. train args, checkpoint_backbone, convert_molmoact2, test_train_vla) into fontaine; re-run check.py; re-verify the retrain-prep seams exactly as the 15:37Z 08-15 post did (read_checkpoint_info on both real conversions, convert_molmoact2 --norm-stats-from, bijou.train --objective/--backbone-text-lr/--init-from/--expert-init, --image-augment p=0 bitwise oracle, convert_legacy+validate_checkpoint smoke); confirm the pre-registered retrain launch commands still parse against the new family CLI (391-line test_train_args churn suggests arg surface moved). Also confirm whether the parents[2]-&gt;parents[3] bank_processor_goldens fix from bb0f036 landed upstream or still needs the cherry-pick note.

</details>

---

**`wrist-screen-results-post`** · `cpu`

Wrist-transfer screen results page (blog, chart-led per the owner standing preference): the F-instrument story end-to-end

**boundary:** Queued 01:3xZ 08-15 at the F-instrument close (depth refill: the screen item closed same session). Not urgent; any writing-ladder session takes it. The Discord boundary post (01:34Z) is the canonical short record until this lands. | DONE 02:4xZ 08-15 (fb1e672): posts/2026-08-15-wrist-screen-results.md — plain-words open, receipts, T1 gate table, power analysis (control ±0.28 at n=25 vs wrist effects ~2x smaller; successor rule: control at treatment n), W3 record-only engagement finding, successor needs. Charts delta_strips + engagement_split (house dark scheme) via fontaine/scripts/wrist_screen_close_charts.py (recompute-and-abort guard vs the banked analysis JSON).

<details><summary>full record</summary>

Wrist-transfer screen results page (blog, chart-led per the owner standing preference): the F-instrument story end-to-end — design, stage-0/1 receipts, the T1 control failure with the power analysis (n=25 control cannot resolve its own +0.16 point estimate while W3 shows +18/100 CI [+0.06,+0.29] at n=100), the record-only W3 engagement finding, and what a successor screen needs (competence floor first — the grasp-SFT line — and a control priced at n=100). Charts: paired per-seed delta strips per arm, gate table, the engagement-flip split. Links reports/analysis__wrist_screen_stage1.json; plain-words open per the papers rule if it lands as a Papers-adjacent page (it is a results post — Status/plain-words block up top either way).

</details>

---

**`grasp-sft-stage-b-collector`** · `cpu`

Grasp-SFT stage-B demo collector (CPU instrument, prereg §2/§6): scripted-expert rollouts on demo seeds ascending from 1000, rendered under the production visual config, SUCCESSES KEPT, obs (top/wrist/state) + executed action tar…

**boundary:** Queued 02:4xZ 08-15 with amendment A1 (depth refill at the results-post close). Collection launches only after the A1 fresh held gate read passes &gt;=14/20 and per its window terms; the writer instrument itself is the executable CPU slice any session can take. | DONE 03:2xZ 08-15 (5b360fa): instrument landed with 4 CPU oracles (eval-seed refusal, success-only + round-trip bit-equal, resume-appends, foreign-dir refusal) + GL smoke (2 real demos collected, reloaded, schema+values verified). The GPU collection leg rides the grasp-sft-bootstrap item per A1 window terms. · [pre-reg](posts/2026-08-14-prereg-grasp-sft-bootstrap.md)

<details><summary>full record</summary>

Grasp-SFT stage-B demo collector (CPU instrument, prereg §2/§6): scripted-expert rollouts on demo seeds ascending from 1000, rendered under the production visual config, SUCCESSES KEPT, obs (top/wrist/state) + executed action targets written in the molmoact2 train_lerobot.py training format with state_units provenance 'rig (identity — recomputed dataset table)' (§6 item 4, no shim anywhere in B-D). Eval seeds 0-99 refused at the writer (the run_expert_episode guard rides through). Target 400 kept / gate &gt;=300 within &lt;=4 GPU-h; checkpointable (resume from the last banked seed) so the GPU leg can ride any window. Oracles: format round-trip vs a real training-set row, seed-refusal, success-only filter, action-vs-replay consistency on one episode. CPU-buildable NOW; the GPU collection leg is gated on the A1 fresh gate read (seeds 1040-1059) passing.

</details>

---

**`grasp-sft-chain-results-page`** · `cpu`

Grasp-SFT chain results page (blog, chart-led per the owner standing preference; writing-ladder item): the competence-before-RL story end-to-end once stage D banks its verdict

**boundary:** Queued 04:2xZ 08-15 at the stage-d-eval-prep close (depth refill). Blocked in practice until stage D banks analysis__grasp_sft_stageD_sim100.json; any writing-ladder session takes it after that. The Discord boundary posts are the canonical short record meanwhile. | DONE 15:4x-15:5xZ 08-15 (work session; draft 2439869, probe section 13:4xZ, finalized this commit): page live + in-channel post 1538212636477624320 — the stage-D verdict section became the SUSPENSION record (10:10Z re-steer; formal exam never ran) plus the A2 re-base note (a flow-head sim100 no longer triggers R2; the token arm's discrete-head count does), and a new 'Where this goes next' section carries the three pending owner decisions (retrain arm / route A/B/C / GPU release) with pre-reg links. All 5 dark-mode charts curl-verified live; charts regenerate from banked JSONs via grasp_sft_chain_charts.py; blog built + Space pushed. Chain ledger on the page: ~11 GPU-h vs the &lt;=13 gate. · [pre-reg](posts/2026-08-14-prereg-grasp-sft-bootstrap.md)

<details><summary>full record</summary>

Grasp-SFT chain results page (blog, chart-led per the owner standing preference; writing-ladder item): the competence-before-RL story end-to-end once stage D banks its verdict — stage-A scripted-expert arc (breakthrough mechanisms, gate FAIL -&gt; A1 -&gt; PASS, the n=20 CI lesson vs the measured 62.5% true rate at n=200), stage-B collection facts (keep rate, seed integrity, provenance), stage-C SFT curves vs the rig-ft r1 reference, stage-D sim100 verdict vs the banked context anchors (ftrig4k / W0), and what the verdict means for the GRPO registration (Decision 11). Plain-words open per the papers rule; dark-mode charts (per-band keep-rate strip, SFT loss curve, per-seed progress strip vs anchors); links the banked analysis JSONs + demo videos.

</details>

---

**`grasp-sft-stage-d-eval-prep`** · `cpu`

Grasp-SFT stage-D eval prep (CPU, prereg §6 frozen arm list): the sim100 eval leg for the stage-C endpoint(s)

**boundary:** Queued 03:5xZ 08-15 at the stage-c-launch-prep close (depth refill). Executable CPU once stage C is training (conversion/launcher code can be built against the rig-ft r1 precedent before the endpoint exists); the GPU leg rides the grasp-sft-bootstrap ladder. | DONE 04:2xZ 08-15 same session (stage-B ride window, no-idle rule): (1) launch_local_grasp_sft_staged_eval.sh — convert mode (two-hop: their convert_molmoact2_to_hf on the olmo step dir -&gt; HF serve dir carrying the demo-set recomputed norm_stats, then bijou.convert_molmoact2 --norm-tag so100_so101_molmoact2 -&gt; converted dir; rig-r1 runbook + eval20 precedent) + eval100 mode (sequential rollout_sim ONLY — 08-12 parallel oracle FAIL froze sequential; frozen seeds 0-99, --episode-seconds 30 per the eval20 replans-vs-chunk lesson, euler-10 AR / euler-1 flow, videos + out-json, GPU-free guard). (2) grasp_sft_staged_reads.py — the frozen §2 decision surface as code: success == recorded success_tick (sim100 convention, verified against the banked ftrig4k arm JSON schema), refuses any seed set != 0-99, reset-strikes gate, verdict bands ORACLE-TESTED at the edges (20-&gt;GRPO_GO, 19/5-&gt;ITERATE_BC_ONCE, 4-&gt;F_TRANSFER, strikes flag, non-frozen-set refusal all green); context anchors ride record-only. (3) PREPARED babysit entry grasp_sft_stageD_eval (2.0 GPU-h gate). NO launch — stage D rides the ladder after stage C. · [pre-reg](posts/2026-08-14-prereg-grasp-sft-bootstrap.md)

<details><summary>full record</summary>

Grasp-SFT stage-D eval prep (CPU, prereg §6 frozen arm list): the sim100 eval leg for the stage-C endpoint(s) — (1) checkpoint conversion: convert_molmoact2_to_hf on the stage-C AR final step (rig-ft r1 precedent: serve dir ~/checkpoints/...-hf) + the flow endpoint if trained; (2) eval launcher on the FROZEN 100 eval seeds 0-99 (standard sim100 harness gates + reset-strike checks, --report HTML per the owner standing rule), consuming the endpoint through the same recomputed-table rig-identity frame it trained in (§6: NO shim anywhere in B-D); (3) prepared babysit entry (~1-1.5 GPU-h class); (4) the frozen primary read wired as the decision surface: &gt;=20/100 -&gt; GRPO GO (fresh pre-reg per Decision 11), 5-19 -&gt; one B/C iteration, &lt;5 -&gt; F-transfer (wrist-screen F-instrument read becomes the binding diagnosis). Context anchors (not gates): banked ftrig4k +0.08 cm / 47 moved / ~1 success; stage-1 W0 in-run row +0.054 / 44 / 2 successes. NO launch from this item — stage D launches per the frozen ladder after stage C lands.

</details>

---

**`grasp-sft-stage-c-launch-prep`** · `cpu`

Grasp-SFT stage-C launch prep (CPU, prereg §6 frozen params): the train_lerobot.py launcher script for the AR primary

**boundary:** Queued 03:2xZ 08-15 at the collector close (depth refill). Executable CPU now (launcher + audit code); its preconditions (the demo set) land with stage B. | DONE 03:5xZ 08-15 work session: (1) mixture so101_grasp_sft landed in ~/molmoact2 (7fb6552 on fontaine-so101-rig): demo repo fontaine/grasp_sft_demos_v0, tag so100_so101_molmoact2 so per-tag q01/q99 recompute over the demo repo only (§6 item 4), import-verified. (2) AR launcher launch_local_molmoact2_grasp_sft_stagec_ar.sh — rig-ft r1 verbatim-class, mechanical diff receipt in the header AND re-verified by diffing the torchrun blocks: ONLY mixture/wandb-name/max_duration 2000-&gt;3000/save_folder/unit/log differ. NOTE: no --objective flag exists in molmoact2 train_lerobot.py (the item title's '--objective ar' was draft-era language; MolmoAct2 is natively AR — frozen §6 text, which governs, never mentions the flag). (3) flow-arm launcher launch_local_grasp_sft_stagec_flow_4k.sh — ftrig4k train block verbatim, diff receipt = ONLY --train-data swapped + run name; conditional on the &lt;=13 GPU-h budget after the primary lands. (4) shared preflight grasp_sft_stagec_preflight.py refuses launch unless: collector finished + provenance banked, kept &gt;=300, ALL kept seeds &gt;=1000, state_units == frozen identity string, info.json consistent; prints epoch math (3000xgb64 and 4000xb24 vs total frames). Oracle-tested: PASS path + 3 refusal paths (kept&lt;300, eval-seed 99, shim state_units) + live-collector refusal all green. (5) PREPARED babysit entries grasp_sft_stageC_ar + grasp_sft_stageC_flow (commented, FILL-AT-LAUNCH started_utc, gates 5.0/1.5 GPU-h). · [pre-reg](posts/2026-08-14-prereg-grasp-sft-bootstrap.md)

<details><summary>full record</summary>

Grasp-SFT stage-C launch prep (CPU, prereg §6 frozen params): the train_lerobot.py launcher script for the AR primary — base allenai/MolmoAct2-SO100_101, rig-ft recipe class verbatim (ft_action_expert=true only, ft_vlm=false, ft_embedding=none, lora=false), action-expert LR 5e-5, global batch 64 (device 8), save every 500, max_duration 3000 steps, --objective ar, dataset = the stage-B demo set with RECOMPUTED per-dataset q01/q99 (identity rig frame, §6 item 4 — verify no shim flag anywhere in the arg list) — plus the optional ftrig4k-recipe flow arm variant (4k steps, decoder LR 1e-5, dataset swapped). Deliverables: launcher(s) with a demo-set precondition audit (&gt;=300 kept per gate, provenance/state_units check, epoch math logged at gb64), prepared babysit.toml entries, and the arg lists diffed against the banked rig-ft run-1 command line as the verbatim-class receipt. NO launch from this item — stage C launches per the frozen ladder after stage B's gate (&gt;=300 kept) reads green.

</details>

---

**`image-augment-sim2real`** · `cpu`

image-augment-sim2real: train-time photometric augmentation in bijou.train (owner ask 13:09Z 08-15, Q2 reply 1538178752582787093): --image-augment flag applying brightness/contrast/saturation/hue jitter + gamma, Gaussian sensor n…

**boundary:** Queued 13:4xZ 08-15 from the owner's sim2real question. Value prices in only at rig transfer (sim100 may dip slightly with aug on — expected, say so in the pre-reg). Composes with whichever retrain arm the owner picks (flag on the same run or a follow-up arm). Render-time domain randomization (lighting/textures/camera pose at collection) is the recorded heavier alternative — needs demo re-collection, machinery exists from the arm-photometrics/texture screens. || DONE 14:2xZ 08-15 (work session, commit 09129af + blog commit): --image-augment landed in bijou.train — bijou/image_augment.py (v0 spec frozen: crop/translate 0.90-1.0, brightness +-0.15, contrast/sat 0.7-1.3, hue +-0.05, gamma log-U(0.8,1.25), noise p=.5 sigma .002-.02, blur p=.25, JPEG p=.25 q40-85), Collator.image_augment per-frame gate at the CameraFrame seam, probe clone augment-0, eval-side default 0.0. 11 oracles (p=0 identity + zero-RNG bitwise pin, determinism, non-mutation, probe-clone convention); check.py green 865. Pre-reg page live (posts/2026-08-15-prereg-image-augment-sim2real.md, curl-200) with a clean-vs-7-draws grid on a real stage-B frame; in-channel post 1538191003574607885. Recommended first use --image-augment 0.8 on the owner-picked retrain arm (direct = confounded vs the 28/100 floor, follow-up arm = clean A/B ~2.9 GPU-h more) — owner's call, recorded in the page SS4.

<details><summary>full record</summary>

image-augment-sim2real: train-time photometric augmentation in bijou.train (owner ask 13:09Z 08-15, Q2 reply 1538178752582787093): --image-augment flag applying brightness/contrast/saturation/hue jitter + gamma, Gaussian sensor noise, slight defocus blur, JPEG artifacts, small random crop/translate to camera frames at TRAIN time only (pi0/OpenVLA-class sim2real recipe); aug-off path oracle-pinned bitwise to today's pipeline; eval NEVER augmented. CPU-implementable (feature + oracles) any session; pre-reg the aug recipe params before any training arm uses it.

</details>

---

**`wrist-transfer-screen-prereg-final`** · `cpu`

FINAL pre-registration for the wrist-transfer screen (the design memo 2026-08-14-wrist-transfer-screen-design.md frozen into a launchable pre-reg): freeze arm list {ftrig4k, simft} x {W0..W4} + T1, seeds 0-99, the knn5 honesty ax…

**boundary:** Queued 18:2xZ 08-14 at the squint-preflight close (depth refill; charter S4). Executable any CPU window. Simft training-data build steps stay inside the run item; this item is ONLY the frozen pre-reg document + post. | DONE 18:5xZ 08-14 (work session): FINAL pre-reg posted (posts/2026-08-14-prereg-wrist-transfer-screen.md) — design memo sections 5-7 frozen VERBATIM (programmatically diffed byte-identical), arm grid {ftrig4k, simft} x {W0..W4} + T1 frozen with seeds 0-99 (T1 0-24), knn5 honesty anchors 0.877-&gt;0.523 frozen, ladder + &lt;=14 GPU-h gate frozen, amendment policy stated (in-channel before the affected stage, never retroactive). Design-memo caption erratum fixed in place (said '&lt;=12 gate'; section-9 text's &lt;=14 was always the registered figure) with a dated erratum note. wrist-transfer-screen-run converted to GPU-release-only.

<details><summary>full record</summary>

FINAL pre-registration for the wrist-transfer screen (the design memo 2026-08-14-wrist-transfer-screen-design.md frozen into a launchable pre-reg): freeze arm list {ftrig4k, simft} x {W0..W4} + T1, seeds 0-99, the knn5 honesty axis anchors (0.877-&gt;0.523 span), the W0 determinism gate + sanity band, falsifiers F-instrument/F-null/F-flat/F-live verbatim, ladder + 14 GPU-h gate, and the abort/success readouts; post to the blog + in-channel pointer. CPU-only writing task; its completion converts wrist-transfer-screen-run from double-blocked (prereg + GPU) to GPU-release-only, so the run launches the moment the owner frees the GPU.

</details>

---

**`squint-twin-preflight`** · `cpu`

Squint SO-101 twin preflight (lit 0819, papers/squint.md; the wrist-transfer design memo's successor tier for the success-rate form of the question): CPU-side only - install the MIT repo in an isolated venv, verify the 8 SO101*-v…

**boundary:** Queued 17:3xZ 08-14 at the wrist-transfer design close (depth refill; executable any window, pure CPU - no GPU needed for the preflight). Not a commitment to the twin tier: the note prices it, the wrist-transfer screen's outcome decides it. | DONE 18:2xZ 08-14 (work session): CPU-only preflight executed GO — 8 envs register+step headless (physx_cpu + lavapipe, GPU 0 MiB throughout); pd_joint_pos verified raw absolute-joint radians end-to-end (hold drift 0.0 rad, random-walk p50 track 0.014 rad, 50-step truncation, success/info predicates plumbed); 224x224 via sensor_configs kwarg, wrist+greenscreen+third frames saved to outputs/squint_preflight/ + fontaine-reports; step cost 1.9 ms state / 27 ms wrist-rgb224 / 128 ms third-rgb224 at the CPU floor. Two API traps documented (overlay needs rgb+segmentation obs mode or silently no-ops; CAMERA_TYPE is a per-process module constant, in-process alias flip impossible). Feasibility note on the blog (2026-08-14-squint-twin-preflight.md); probe script fontaine/scripts/squint_preflight.py. Tier decision stays with the wrist-transfer screen outcome.

<details><summary>full record</summary>

Squint SO-101 twin preflight (lit 0819, papers/squint.md; the wrist-transfer design memo's successor tier for the success-rate form of the question): CPU-side only - install the MIT repo in an isolated venv, verify the 8 SO101*-v1 ManiSkill3 envs register and step headless; render one wrist + one third-person frame at policy-relevant resolution (224+) with apply_overlay=False and save to outputs/ for a look; verify pd_joint_pos normalize_action=False consumes our absolute-joint LeRobot convention end-to-end with a scripted hold + a random-walk episode (success/info plumbing observed); note per-step wall time at 1 env CPU. Deliverable: a short feasibility note on the blog (what works, what needs a subclass, measured step cost) feeding the tier decision if the wrist-transfer screen hits F-instrument or the success floor holds.

</details>

---

**`wrist-transfer-screen-run`** · `gpu-local`

Execute the wrist-transfer screen per the 08-14 design memo (posts/2026-08-14-wrist-transfer-screen-design.md, frozen sections 5-7 become the pre-reg verbatim): stage 0 wrist-transform hook (--wrist-transform {none,blackout,freez…

**boundary:** Queued 17:3xZ 08-14 at the design close. BLOCKED on the in-channel GPU release (owner reserve 12:54:19Z 08-14 stands). prereg field points at the design memo (the registered skeleton); launch still requires a posted FINAL pre-reg freezing its sections 5-7 verbatim; stage boundaries are hard stops. Stage 0 is CPU-preparable during the reserve if a session wants it early - the transform hook + oracles land without touching the GPU; honesty placement (er_60k knn5) is the only stage-0 GPU-adjacent step (~0.1 GPU-h class, still gated on the release). | PREREG-FINAL POSTED 18:5xZ 08-14 (posts/2026-08-14-prereg-wrist-transfer-screen.md): the item is now GPU-RELEASE-ONLY — the in-channel release is the single remaining blocker; stage 0 launches under the FINAL pre-reg with no further paperwork. Stage-0 CPU-preparable slice split out as wrist-transfer-stage0-cpu-prep (hook + transform oracles land under the reserve; the none bit-replay oracle + honesty placement stay GPU-gated inside this item). | GPU RELEASED in-channel 21:14Z 08-14 ('Your GPU is all yours') — the single registered blocker is CLEARED. Launch sequenced behind main-review-molmoact2-final deliverable (c) only: the retirement re-pointed checkpoint loading (bijou checkpoints, not HF-layout dirs + norm tags), so the frozen ftrig4k/simft launch surfaces must be verified or amended in-channel BEFORE stage 0 per the pre-reg's own amendment policy. Stage-0 CPU prep landed (64c93e6): critical path is none bit-replay + honesty placement + stage 1. | DELIVERABLE-(c) VERDICT 21:4xZ 08-14 (main-review-molmoact2-final): NO AMENDMENT NEEDED — P1 ftrig4k (outputs/train/fontaine_flow_snapdistill_ftrig_4k_1xh100) and the stage-2 simft fine-tune are flow-pathway bijou checkpoints served via BijouPolicy --checkpoint, untouched by the phase 3-5 re-point (which moved only grpo_loop --checkpoint and rollout_sim_parallel --molmoact2-discrete); from_checkpoint changes are additive for pre-existing checkpoints. UNBLOCKED: stage 0 (none bit-replay oracle + honesty placement) launches under the FINAL pre-reg with no further paperwork — the next work session takes it FIRST. | STAGE 0 EXECUTED 22:2xZ 08-14 (c5be36f, oracles ALL GREEN): honesty placement PASS on the serving substrate (equidistant wrist — wrist_arm_mask's registered path; W0 in-run 0.8769 reproduces the banked 0.877 manip anchor, W1 blackout 1.0, W3 arm_blur 0.8867 with paired W3-W0 CI95 [9.6e-08, 4.7e-07] excl-0 — small vs the W1 bracket, registered; mask coverage mean 1.6%); none bit-replay PASS (ftrig4k W0 seed 0 x2, episode row bit-equal, run-twice determinism form per the registered config drift); --top-transform blackout landed for T1 (chain_transforms seam, oracles green). STAGE 1 LAUNCHED 22:24:42Z unit wrist-screen-stage1 (launch_wrist_screen_stage1.sh): det gate x2 -&gt; hold(25) -&gt; W0/W1/W3(100 each) + T1(25), ~3-3.5 GPU-h, rc ETA ~01:0x-01:4xZ 08-15, babysit entry live (gate 5 GPU-h). Stage-1 boundary session owns the reads (sanity band, hold floor, T1 CI, spawn_xy pairing, first W1/W3 deltas) + in-channel post BEFORE stage-2 spend. || CLOSED 01:3xZ 08-15 work session at the STAGE-1 BOUNDARY, verdict F-INSTRUMENT (frozen section 4): unit rc 01:32:02Z (~3.1 GPU-h; screen total ~3.3 of &lt;=14) — det gate PASS (launcher, 10 seeds bit-equal x2), sanity band PASS (+0.054 cm / 44 moved / strikes 0), hold floor PASS (+0.0000, strikes 0), spawn_xy pairing PASS, T1 top-blackout control FAIL (dEngagement +0.16 [-0.12,+0.44], d|progress| -0.28 [-1.29,+0.62], both CI95 straddle 0 at n=25; hook consumption receipted — 24/25 T1 rows bit-differ from W0 under deterministic draw-0). Screen ABORTS per the frozen falsifier: stages 2/3 NEVER LAUNCH, no transfer-link claim in either direction; the tier-2 renderer pilot keeps only its proxy-unit case; the fidelity-&gt;behavior question escalates to the competence tier (= the grasp-sft-bootstrap line). RECORD-ONLY: W3 arm-blur flips engagement +18/100 CI95 [+0.06,+0.29] excl-0 (62 vs 44 moved) — a wrist-appearance corruption with a detectable n=100 behavior effect while the n=25 control cannot resolve its own +0.16 point estimate; successor-design lesson: the control was underpowered ~2x vs the effect sizes the wrist arms show. W1 blackout ~0 on every channel (dProgress -0.12 [-0.80,+0.51], flips +3 [-12,+17]). Reads banked reports/analysis__wrist_screen_stage1.json (wrist_stage1_reads.py, 1a857ea); boundary post + owner status reply in-channel 01:34Z. GPU FREE 01:32Z. · [pre-reg](posts/2026-08-14-prereg-wrist-transfer-screen.md)

<details><summary>full record</summary>

Execute the wrist-transfer screen per the 08-14 design memo (posts/2026-08-14-wrist-transfer-screen-design.md, frozen sections 5-7 become the pre-reg verbatim): stage 0 wrist-transform hook (--wrist-transform {none,blackout,freeze,arm_blur} on obs.wrist in both rollout drivers) + oracles (golden frames, bit-replay of none, qpos invariance, W3 mask spot-check) + honesty placement of W1/W3 on the banked 100 manip pose slots; stage 1 ftrig4k x {W0,W1,W3} 100 seeds + T1 top-blackout 25 (~3.3 GPU-h); stage 2 simft fine-tune (sim-rendered replays of real episodes 0-25 + recorded actions, ftrig4k recipe) + P2 x {W0,W1,W3} (~4.8); stage 3 conditional W2/W4 ladder (~3.8). Gates/aborts per memo section 6; worst-case 12.0 GPU-h, gate &lt;=14, hard-stop boundaries with in-channel posts.

</details>

---

**`wrist-transfer-stage0-cpu-prep`** · `cpu`

Stage-0 CPU-preparable slice of the wrist-transfer screen (per the FINAL pre-reg section 1 implementation contract + the run item's standing note that the hook + oracles land without touching the GPU): land the --wrist-transform…

**boundary:** Queued 18:5xZ 08-14 at the prereg-final close (depth refill; charter S4). Executable any CPU window under the reserve. Completing it shortens the post-release critical path to: none bit-replay + honesty placement + stage 1. | DONE 21:2xZ 08-14 (tick, orphan recovery): the 18:59Z work session landed the full item then died on the usage cap before lint+commit (harness alert 19:17Z); this tick audited the diff, fixed lint+pyright, tests 11/11 + check.py 901 green, committed (64c93e6 post-rebase tip) — hook in both drivers ('none' routes around the hook), wrist_arm_mask W3 path, oracles, spotcheck script. · [pre-reg](posts/2026-08-14-prereg-wrist-transfer-screen.md)

<details><summary>full record</summary>

Stage-0 CPU-preparable slice of the wrist-transfer screen (per the FINAL pre-reg section 1 implementation contract + the run item's standing note that the hook + oracles land without touching the GPU): land the --wrist-transform {none,blackout,freeze,arm_blur} hook in both rollout drivers (applied to obs.wrist after observe(), before policy packing — the SimObservation seam in rollout_sim_parallel.py), the W3 per-tick wrist segmentation mask path (arm+gripper geom ids, Gaussian blur inside the mask only), and the CPU-side oracles: golden-frame test per transform + W3 mask visual spot-check on 3 banked pose slots + a transform-purity check (transforms touch pixels, never state). check.py green. The two GPU-adjacent stage-0 steps (none bit-replay of a banked seed; W1/W3 honesty placement on the 100 pose slots, ~0.1 GPU-h class) stay inside wrist-transfer-screen-run, gated on the release.

</details>

---

**`main-review-molmoact2-final`** · `cpu`

Review the molmoact2-retirement final code on main 26ac1e6 (owner ask 21:14Z 08-14: 'reviewing the new code from main after you rebase and let me know your thoughts'): phases 3-5

**boundary:** Queued 21:2xZ 08-14 at the owner ask (21:14Z message + the handoff attachment). Executable immediately — chained work session takes it FIRST; wrist-transfer-screen-run launch is sequenced behind deliverable (c). || CLOSED 21:4xZ 08-14 work session, all 4 deliverables: (a) review post posts/2026-08-14-molmoact2-retirement-review.md + in-channel summary — verdict ADOPT, re-baseline JUDGMENT AGREE (mechanism self-verified: port replay is monolithic cat(prompt,suffix) forward, first-class is prefill+cached continuation — genuine cross-decomposition, 4.4-5.7e-5 in the phase-2 fp32 diagnostic's decade, ratio impact 0.01% vs clip band; forcing decomposition-match to keep 1e-5 would gate surviving code on the deleted port's kernel schedule); 4 nits ranked (train.py ~4420 dead unreachable+false print after the backbone-init-from rider raise; codec.hole_count per-DataLoader-worker undercount; molmoact2_discrete/generate.py missing the run-at-tag header note; from_numpy non-writable warning); (b) probe_grpo_replay_parity RERUN local ~21:5xZ PASS — masks bit-equal ALL 1903+1904 rows, spreads v1 med 5.68e-1/p90 1.29/max 3.92, v2 med 5.52e-1/p90 1.58/max 8.84 (report-only per registration); (c) VERDICT NO AMENDMENT — ftrig4k/simft ride BijouPolicy --checkpoint (flow pathway, untouched); re-point moved only grpo_loop --checkpoint + rollout --molmoact2-discrete; from_checkpoint additive (objective defaults flow, rider mounts only on joint_ce metadata) — wrist-transfer-screen-run LAUNCH-READY as registered; (d) Decision 11 + masked-only + full-width-Gumbel absorbed as the dated post-retirement note on the R1-B record (+ probe receipts); posts index drift fixed (squint + prereg-final entries restored).

<details><summary>full record</summary>

Review the molmoact2-retirement final code on main 26ac1e6 (owner ask 21:14Z 08-14: 'reviewing the new code from main after you rebase and let me know your thoughts'): phases 3-5 — train.py objective matrix (--objective {flow,ar,joint} + --joint-ce-weight + --expert-init + quantization-hole policy, c18d033/ba57b29), bijou/grpo_replay.py re-point (f560528) + replay-parity gate (f219a2d/f77a8c7/6bb6439 re-baseline receipts vs my signed 1e-5 shape — the decomposition-class argument needs my judgment), phase-5 deletion (26ac1e6). Deliverables: (a) in-channel thoughts post; (b) rerun probe_grpo_replay_parity.py on my banked waves locally if cheap (CPU/GPU now free); (c) VERDICT on the wrist-transfer-screen pre-reg: do ftrig4k/simft arm checkpoint-loading surfaces sit on re-pointed code (loop consumes bijou checkpoints now, not HF-layout dirs + norm tags) — if the frozen launch commands change, in-channel amendment BEFORE stage 0 per the pre-reg policy; (d) Decision 11 + masked-only decode + full-width-Gumbel notes absorbed into ledger/docs where my line cites the old behavior. [owner-requested review; no GPU gate — probe reruns ride the free-GPU window]

</details>

---

**`wrist-transfer-screen-design`** · `cpu`

Wrist-transfer screen design doc (the decision brief's move #2, design only

**boundary:** Queued 16:2xZ 08-14 at the decision-brief close (depth refill; executable any window, pure CPU/writing). The design must state its own falsifiers before execution is queued; no launches until the in-channel GPU release. | DONE 17:3xZ 08-14 (work session): design memo posted (posts/2026-08-14-wrist-transfer-screen-design.md) with arms (P1 ftrig4k + P2 simft sim-adaptation sanity arm; wrist columns W0 classic / W1 blackout / W2 freeze / W3 arm-mask blur / W4 measured-materials ON; T1 top-blackout positive control), honesty placement of every arm on the banked knn5 axis, paired frozen-seed 0-99 draw-0 stats, gates (W0 determinism + sanity band vs banked sim100 - git audit found the banked rows predate the fitted lens, NOT a bit-anchor; hold floor; T1 must move; placement sanity), falsifiers F-instrument/F-null/F-flat/F-live, staged ladder worst-case 12.0 GPU-h gate &lt;=14. Schematic chart on fontaine-reports (200). Execution queued as wrist-transfer-screen-run (blocked on GPU release); squint-twin-preflight queued as the successor-tier CPU prep.

<details><summary>full record</summary>

Wrist-transfer screen design doc (the decision brief's move #2, design only — no run): design the closed-loop relative screen that prices whether the banked 0.877 wrist dishonesty moves SUCCESS RATE at all (the unpriced link between the encoder-honesty proxy and the north star). Squint-style per ideas.md 0819: deterministic-seed sim rollouts as RELATIVE screens (domain gap held constant across arms), arms = wrist-conditioned policy on {classic arm render, wrist feed ablated/degraded controls}, plus the required sim-adaptation sanity arm; instrument, seeds policy (fresh-seed rule), n and CI plan, gates + abort bands, GPU-h budget. Deliverable: a pre-registrable design doc on the blog — execution is a separate, owner-visible pre-reg when a GPU window opens.

</details>

---

**`renderer-class-decision-brief`** · `cpu`

Renderer-class decision brief (analysis/writing, no run): consolidate the now-complete arm-appearance price into ONE owner-facing decision post

**boundary:** Queued 15:3xZ 08-14 at the content-split close (depth refill). Executable any window (pure CPU/writing). Not a pre-reg — no run, no claims beyond banked numbers; the owner call it supports is theirs to make. | DONE 16:2xZ 08-14 (work session): brief posted (posts/2026-08-14-renderer-class-decision-brief.md, chart-led, banked numbers only; lead chart chart__renderer_class_decision.png on fontaine-reports, curl 200). Tiers priced: tier-0 albedo spent (refuted x2), tier-1 in-classic cannot express relief (no normal-map input), tier-2 = STL-&gt;UV re-export (convert_benchy.py precedent) + procedural layer-line normal maps + external PBR path feeding the anchored compositor (validation tail is the real cost). Recommendation: pilot before buying — wrist-visible meshes only at the 100 banked manip slots (decides the tier for ~0.02 GPU-h), or the closed-loop transfer read; both owner-gated.

<details><summary>full record</summary>

Renderer-class decision brief (analysis/writing, no run): consolidate the now-complete arm-appearance price into ONE owner-facing decision post — top residual 0.552-&gt;0.328 after clutter (banked), wrist 0.877 at manipulation poses attributed to the RENDERED ARM itself (content term nil per sim-manip-wrist-content-split 15:2xZ 08-14: paired delta +3.28e-07 straddling zero, blind-slot control ~0, ABSENT AUROC 0.888), material-stack regression at manip poses (+4e-07 CI excl. 0). Brief covers: what a normal-map/PBR + real-gripper-geometry upgrade would plausibly buy per channel, rough implementation cost tiers (texture-only vs mjSpec recompile vs asset re-export), what stays unpriced without it, and the recommendation. Chart-led per the standing preference; lands on the blog + in-channel pointer.

</details>

---

**`sim-manip-wrist-content-split`** · `cpu`

Manipulation-pose wrist content split: price how much of the banked 0.877 manip-pose wrist AUROC (sim-rollout-pose-wrist-read, 12:2xZ 08-14) is scene-content mismatch (no boat in the sim jaw, benchy at spawn, no real clutter) vs…

**boundary:** Queued 12:3xZ 08-14 at the rollout-pose read close (depth refill). Executable any GPU-busy window; pre-reg in-channel BEFORE the read; the registered caveat section of 2026-08-14-prereg-sim-rollout-pose-wrist.md is the contract this item discharges. | DONE 15:2xZ 08-14 (work session): pre-reg posted 15:13Z (owner 👍 = ack + gap-go, interpretation stated in-channel 15:21Z with veto window), CPU renders + ~30 s embed gap (~0.005 GPU-h, card at 0 MiB, reserve otherwise untouched). All gates green (reset 0.713/0.523, calibration 0.268 low-note, PRESENT manip AUROC 0.877 = banked digit, render oracles bit-exact x100). VERDICT content_nil: paired dknn5 ABSENT-PRESENT +3.28e-07 CI95 [-2.26e-07, +8.39e-07] straddles zero, content share -3.8% of the pose effect; ABSENT AUROC 0.888 (still fake-side); blind-slot control +6.4e-09; benchy-px&lt;-&gt;|d| corr 0.011. The banked 0.877 caveat is DISCHARGED strengthening it: the rendered arm carries the whole manipulation-pose wrist gap — renderer-class decision keeps its full wrist price. Results on the pre-reg page + chart__wrist_content_split.png.

<details><summary>full record</summary>

Manipulation-pose wrist content split: price how much of the banked 0.877 manip-pose wrist AUROC (sim-rollout-pose-wrist-read, 12:2xZ 08-14) is scene-content mismatch (no boat in the sim jaw, benchy at spawn, no real clutter) vs the rendered arm itself. Same harness: re-render the 100 pose-matched manip slots with (a) benchy REMOVED (clean table) and (b) benchy at spawn (the banked arm), paired per slot; optionally a real-frame arm-crop rider if separable. er_60k knn5 vs the same manip reference; paired deltas + AUROC per arm. If the content term is small, the rendered arm carries the gap -&gt; the renderer-class decision (normal-map/PBR + gripper geometry) gets its wrist-side price; if large, the 0.877 overstates the camera's dishonesty and the honest number is lower. Pre-reg required (bands frozen from the banked run: anchors 0.713/0.523, calibration 0.268 directional gate per amendment 2). CPU renders + ~0.02 GPU-h embeds.

</details>

---

**`molmoact2-retirement-adoption`** · `cpu`

Adopt the molmoact2 retirement plan (docs/molmoact2-retirement.md, main 02a58e0; owner ask 12:46Z 08-14, thoughts + sign-offs posted in-channel 12:50Z): (1) rebase fontaine onto main &gt;= db0a141 (T1 ar_fast retirement + T2 residua…

**boundary:** Queued 12:5xZ 08-14 at the owner ask. Step (1) executable at the current run boundary AFTER grpo-r1b-boundary-reads (banked rows consumed via current import paths first); steps 3-4 sequenced with the owner's phase landings — never under a live run. | Moved ahead of sim-manip-wrist-content-split 13:1xZ 08-14 (the 12:5x signed order: rebase after the boundary reads; main is already &gt;= db0a141 at 51704c0, so the rebase step is executable now). | STEP (1) DONE 13:3xZ 08-14 (work session): fontaine rebased onto main 51704c0 (137 commits replayed, one conflict — model.py ar_predict_sampled docstring, action_capture doc kept + retired ar_fast mention dropped, exactly plan §0's predicted surface); check.py 858 green + grpo oracle suite 43 green post-rebase; pushed --force-with-lease (old tip tagged pre-rebase-51704c0). Steps 2-4 remain: track phases 1-3 as they land on main; phase-4 co-land blocked on owner ladder adjudication + phase landings. | PHASES 0a+1 LANDED on main c57ce05 (observed tick 13:4xZ 08-14: vendored parity fixtures + leaf promotion; 16 files, +5604/-743 incl. tests/test_fast_molmoact2.py). Step (2) adoption executable: rebase fontaine 51704c0-base onto c57ce05, check.py + grpo oracle suite green post-rebase — chained work session takes it ahead of the wrist-content-split pre-reg. Phases 2-3 not yet landed; phase-4 co-land still blocked on ladder adjudication. | STEP (2) DONE 13:5xZ 08-14 (work session): fontaine rebased onto main 0312ab7 (c57ce05 phases 0a+1 + the convert_molmoact2 --norm-stats-from commit); 140 commits replayed, ZERO conflicts (phase-1 predictor shim merged clean next to the discrete-pathway imports; vendored fast-tokenizer fixtures blob-identical to the carried ones, dropped as already-applied); grpo oracle suite 43 green; check.py 863 green + 2 FAILED both INHERITED from main — tests/test_molmo_flow.py byte-parity pair fails on clean origin/main 0312ab7 on this machine (fixture not byte-portable: forward max |d| 4.17e-7, &lt;=40 ULP, 84/96 elements — kernel-order class), finding posted in-channel 13:52Z (1537821299538264114) for the owner's call (allclose-with-tol vs per-machine regen); pushed --force-with-lease, old tip tagged pre-rebase-0312ab7. Steps 3-4 remain: track phases 2-3 as they land; phase-4 co-land blocked on ladder adjudication + phase landings. | PHASE 0(b) LANDED on main 7d89f53-&gt;77246a9 (observed 14:1x-14:5xZ 08-14: discrete-AR-head decode fixture + generator lint/pyright excludes). Adoption DEFERRED deliberately: owner 14:11Z says their local agent will push a byte-parity-fixture fix — one combined rebase (0b + the fix) closes the red pre-commit gate green in a single replay instead of two skip-checks closes. Watch held to 14:5xZ, fix not yet landed; next session/tick adopts on landing. | COMBINED REBASE DONE 15:3xZ 08-14 (work session): fontaine onto main 3131f82 (7423ec3 fixture bounds = my measurement registered, + joint-frame remap + gate-d-lite PASS doc), 143 commits zero-conflict, check.py 874 GREEN (parity pair passes) + grpo suite 43 green — pre-commit gate green again, old tip tagged pre-rebase-3131f82. | LADDER ADJUDICATED STOP 15:31Z 08-14: owner delegated ('waits on your ladder adjudication'), I adjudicated STOP per the 13:1xZ recommendation, owner ratified 15:31Z/15:36Z (recorded in the doc at 5a2a395). Phase-4 co-land now sequenced PURELY behind the owner's phases 2-3 landings (frozen-wave parity gate incl. the v2-reward wave unchanged). | ABSORB 15:5xZ 08-14: main 0a3bed8 (phases-2-3 handoff + anchor probe) + 5a2a395 (adjudication record) rebased in, 145 commits zero-conflict, check.py 874 green post-absorb. Remaining: watch phases 2-3 land, then phase-4 co-land + phase-5 sign-off. | ABSORB 18:0xZ 08-14 (work session): main e5b6113 (PHASE 2 EXECUTED - acceptance PASS byte-equal x6, logprobs 2.4e-7 + the two decode-parity probe commits) rebased in, 8 commits zero-conflict, check.py 879 green + grpo oracle suite 43 green post-absorb, pushed --force-with-lease (old tip tagged pre-rebase-e5b6113). Remaining: watch phase 3 land, then phase-4 co-land + phase-5 sign-off. | ALL PHASES COMPLETE + ADOPTED 21:2xZ 08-14 (tick): owner delegated finishing the plan — phases 3-5 landed on main 26ac1e6 (objective matrix, grpo_replay re-point + frozen-wave replay-parity gate executed on my banked R1-A/R1-B waves with receipts [masks bit-equal, logprobs &lt;=5.7e-5 vs re-baselined 1e-4, per-token deltas &lt;=6.5e-8], bijou/molmoact2/ DELETED); fontaine rebased onto 26ac1e6 ZERO conflicts (16 commits, 836 non-GPU green, old tip tagged pre-rebase-26ac1e6). Phase-4 co-land + phase-5 sign-off overtaken by the owner-side execution; my review/sign-off moved to main-review-molmoact2-final (owner ask 21:14Z).

<details><summary>full record</summary>

Adopt the molmoact2 retirement plan (docs/molmoact2-retirement.md, main 02a58e0; owner ask 12:46Z 08-14, thoughts + sign-offs posted in-channel 12:50Z): (1) rebase fontaine onto main &gt;= db0a141 (T1 ar_fast retirement + T2 residual-conditioning removal; conflict surface per plan §0: flow.py sample_actions_sde, model.py action_capture kwarg, eval/policies.py TokenRow/stable_sde_step_noise vs tile_memory, train.py 3 lines — expect trivial); check.py green + grpo oracle suite green post-rebase (test_grpo_step/test_token_rows/test_molmoact2_replay/test_grpo_loop). (2) Track phases 1-3 as they land on main, adopt at convenience. (3) PHASE 4 CO-LAND (my instrument, boundary I signed: after grpo-r1b-boundary-reads land + owner ladder adjudication): thin replay builder + loop/driver re-point to BijouPolicy+MolmoAct2ARDecoder; gate = frozen-wave replay parity on banked R1-B waves (rewards equal, logprobs in registered 1e-5+JPEG bounds) INCLUDING one v2-reward wave (grip-trace keys preserved — the gate addition asked in-channel). (4) Phase-5 sign-off after 4 is green.

</details>

---

**`sim-rollout-pose-wrist-read`** · `cpu`

Rollout-pose wrist gap read: the one unmeasured leg the consolidated report flags

**boundary:** Queued 11:2xZ 08-14 at the consolidated-report close (depth refill). Executable any GPU-busy window; pre-reg in-channel BEFORE the read; if the owner answers the promotion asks first, fold the flipped defaults into the arm set. | CLOSED 12:2xZ 08-14: executed as sim_rollout_pose_wrist_read.py with the registered premise correction (no banked sim qpos traces existed — poses taken from the REAL held-out episodes' recorded observation.state, exact pose-matched). Two registered ABORTS banked first (interleaved-calibration temporal leakage 0.129; symmetric band vs the protocol's real-real drift floor 0.268 — amendments 1+2 on the pre-reg page). Final: manip-pose wrist AUROC 0.877 = GAP REAL (understated in this calibration direction); material stack REGRESSES the wrist at manip poses (+3.99e-07 CI [+2.0,+6.3]e-07); pose effect +8.7e-06 (1/100). Riders replicated banked digits (reset top -1.49e-07, reset wrist neutral, anchors 0.713/0.523 x3 runs).

<details><summary>full record</summary>

Rollout-pose wrist gap read: the one unmeasured leg the consolidated report flags — the banked 0.828 wrist anchor is ROLLOUT-frame (gripper filling the frame mid-manipulation); every wrist read so far is reset-pose (0.548-0.561 band). Render wrist frames at banked rollout trajectories' recorded qpos (settled mid-episode poses, production v3 + fitted curve-only lens + re-tuned pose), pair against real mid-manipulation wrist frames from the held-out episodes, er_60k knn5 probe. Answers whether the wrist camera is honest where it matters for policy (manipulation frames), prices the material flags' wrist-side effect at poses where the arm FILLS the frame (~230 px at reset vs most-of-frame mid-grasp). Pre-reg required before the read (anchors: reset band 0.548-0.561, rollout 0.828 banked rollout-frame; bars frozen at pre-reg). CPU renders + ~0.02 GPU-h embeds.

</details>

---

**`sim-appearance-consolidated-report`** · `cpu`

Appearance programme consolidated report (chart-led, closed-screen rule): the sim top-cam appearance screen is measured end-to-end

**boundary:** Queued 11:1xZ 08-14 at the full-optin-stack close (depth refill; owner standing preference: chart-led consolidated reports for closed screens). Fully CPU, no pre-reg needed (no new claims). If an owner promotion call lands first, fold the decision into the report rather than re-scoping. | CLOSED 11:2xZ 08-14: report posted (posts/2026-08-14-appearance-screen-report.md, chart-led, banked numbers only), lead chart chart__appearance_screen_ladder.png on fontaine-reports (curl 200), reports.md consolidated entry added, in-channel post 11:28:26Z. Promotion guidance on record: clutter = payload, materials = free riders, remainder = renderer-class (geometry/relief).

<details><summary>full record</summary>

Appearance programme consolidated report (chart-led, closed-screen rule): the sim top-cam appearance screen is measured end-to-end — v3 0.713 anchor; clutter real-crop patches 0.556 (carries the gap's removable share); arm_photometrics 0.698; mount rides the material stack to 0.702; texture REFUTED twice (albedo channel exhausted, relief/light-transport hypothesis banked); wrist-side neutral at reset poses; full opt-in stack 0.5521 (materials absorbed next to clutter, interaction +0.0063 sub-additive). One blog-post report telling the whole story with the ladder chart + frame strips, written for the owner's three pending promotion decisions — what to flip, in what order, what each flag is worth alone vs stacked, and what remains (real-fg 0.328 floor: geometry/light-transport, renderer-upgrade priced separately). No new measurements — banked numbers only.

</details>

---

**`grpo-r1b-boundary-reads`** · `cpu`

R1-B boundary reads at rc (ETA ~19:3xZ 08-14, unit grpo-phase2-r1b): execute the pre-reg's registered reads

**boundary:** Queued 09:5xZ 08-14 at the R1-B launch. Blocked until rc/tripwire; any session at rc executes; babysit rides it meanwhile (~30-min checkpoints, poll forced last). | UNBLOCKED 12:5xZ 08-14 tick: R1-B SELF-STOPPED on the knockaway wire 12:40:50Z (fresh steps 0.328/0.3125/0.4531 all &gt; 0.167, exit 3 rc 3) — the S6-style stop reads apply. Banked endpoint step_0006.pt (step-7 update exited pre-save); step-7 telemetry REVERSED step 6 (earned 1.66-&gt;0.58 cm, reward_mean -0.26-&gt;-1.21). Pre-reg §4 registered contingency = the finding: wire re-fired under v2 =&gt; shoving not reward-driven at this surface. Reads owed: paired delta at step_0006, behavior-prediction judgment, ladder verdict for owner adjudication; step_0006 weights-only upload; results section + chart + in-channel. Cost ~2.95 GPU-h, ladder cum ~8.1 of 22. EXECUTE FIRST in the chained work session. | CLOSED 13:1xZ 08-14 work session: all reads executed on the banked jsonl (CPU only, GPU untouched under the owner reserve). Calibration PASS (8/8 groups kept every wave, median std 3.27/3.02/2.14 -&gt; no lambda amendment); PRIMARY flat (+0.0246 CI95 [-0.0716,+0.1455] at step_0006 vs the 1.868 pairing, probe digit-identical steps 5/6); behavior prediction FALSIFIED on the deciding channel (ungrasped_disp decayed 4.98-&gt;4.60-&gt;4.20 cm but knockaway rose to run-max 0.4531, earned collapsed to 0.58) -&gt; registered finding: shoving is a competence artifact, not reward-driven. Recommended ladder verdict STOP posted for owner adjudication (post 1537810884318199889). step_0006_weights.pt (2.9 GiB) + train.jsonl + meta.json on fontaine-checkpoints grpo_phase2_r1b/; chart__grpo_r1b_boundary.png on fontaine-reports; results section on the pre-reg page. · [pre-reg](posts/2026-08-14-prereg-token-grpo-phase2-r1b.md)

<details><summary>full record</summary>

R1-B boundary reads at rc (ETA ~19:3xZ 08-14, unit grpo-phase2-r1b): execute the pre-reg's registered reads — PRIMARY held-out paired delta CI (v1 metric, banked 1.868 baseline) -&gt; accumulate = R2 pricing discussion, flat+wires-quiet = ladder STOPS as a banked negative; calibration read (first-wave earned/shoved decomposition, &gt;=6/8-drop bar -&gt; lambda re-price amendment); behavior predictions (knockaway_frac decay vs R1-A 0.41-&gt;0.36-&gt;0.31, setback_frac, earned/shoved trends). Results section on the pre-reg page + in-channel post + chart; endpoint checkpoint upload if boundary-worthy; babysit entry prune. If the run tripwired instead, the S6-style stop reads apply (which wire, wave facts, banked step).

</details>

---

**`grpo-r1b-repriced-launch`** · `gpu-local`

R1-B re-priced ladder (option 1, owner-approved 09:16Z 08-14, sequenced behind grpo-reward-patch-prereg): resume from banked step_0004 (fontaine-checkpoints/grpo_phase2_r1a, weights-only) under the PATCHED reward with lr 3e-7 + k…

**boundary:** LAUNCHED 09:43:20Z (unit grpo-phase2-r1b), rc ETA ~19:3xZ 08-14; babysit registry entry live. Boundary reads at rc per the pre-reg (accumulate or the ladder stops). | SELF-STOPPED 12:40:50Z 08-14 at fresh-step 3-of-3 (jsonl step 7): knockaway wire 0.328/0.3125/0.4531 vs 0.167 x3 — registered exit 3, unit rc 3, GPU released. step_0006.pt banked on disk. ~2.95 GPU-h this run (09:43-12:40Z). Boundary reads item unblocked. · [pre-reg](posts/2026-08-14-prereg-token-grpo-phase2-r1b.md)

<details><summary>full record</summary>

R1-B re-priced ladder (option 1, owner-approved 09:16Z 08-14, sequenced behind grpo-reward-patch-prereg): resume from banked step_0004 (fontaine-checkpoints/grpo_phase2_r1a, weights-only) under the PATCHED reward with lr 3e-7 + kl_beta 1.0, wire unchanged (knock tripwire stays as belt even with the in-reward fix), ~0.96 GPU-h/step; 10 steps ~9.6 GPU-h, ladder cum ~5.1 of the 22 gate -&gt; fits with headroom. Own pre-reg (final constants + boundary reads) before launch.

</details>

---

**`sim-full-optin-stack-read`** · `cpu`

Full opt-in stack read (prices the combined promotion): the owner has three pending appearance promotions measured SEPARATELY (clutter real-crop patches 0.713-&gt;0.556; arm_photometrics 0.713-&gt;0.698; mount rides the stack at 0.713-…

**boundary:** CLOSED 10:58Z 08-14 rc=0, all gates green (in-run v3 0.7127 band-center; in-run patched 0.5561 bit-matching the banked fg-fix read; cross-instance qpos/draws/affine bit-equal x100; changed-px 12.3% material footprint). MIDDLE BRANCH of the frozen rule: paired stack vs v3 -2.075e-06 CI [-2.254,-1.891]e-06 (99/100) but stack AUROC 0.5521 &gt; bar 0.5511 (beats best single by only -0.0040 &lt; eps 0.005). Materials' marginal on top of clutter -5.50e-08 CI [-1.44e-07,+3.37e-08] straddles 0 (~1/3 of banked solo effect) — attenuated ~3x, statistically absorbed. Additivity: predicted 0.5458 measured 0.5521, interaction +0.0063 sub-additive. Disposition: clutter carries the combined gain (promote first/alone); material flags safe to stack but not additive as separately sold; bigger-n marginal read priced only on owner request. Results in the pre-reg post + in-channel 11:00Z. · [pre-reg](posts/2026-08-14-prereg-sim-full-optin-stack.md)

<details><summary>full record</summary>

Full opt-in stack read (prices the combined promotion): the owner has three pending appearance promotions measured SEPARATELY (clutter real-crop patches 0.713-&gt;0.556; arm_photometrics 0.713-&gt;0.698; mount rides the stack at 0.713-&gt;0.702). If they flip together, interactions are unmeasured. One paired 20x5 top read: v3 default vs the full opt-in stack (clutter patches + arm_photometrics + mount_material), er_60k knn5, in-run v3 gate 0.713+/-0.005. PRIMARY: paired dknn5 CI95 &lt; 0 AND full stack &lt;= best single (0.556) - epsilon registered in the pre-reg; record additivity vs the sum of parts. Pre-reg with explicit bar before any read.

</details>

---

**`grpo-reward-patch-prereg`** · `cpu`

GRPO reward patch (option 2, owner-approved 09:16Z 08-14): the R1-A tripwire caught REWARDED shoving leakage - composite_reward's dense progress term pays cm-for-cm for boat displacement without a grasp, and knock-away is only th…

**boundary:** CLOSED 09:4xZ 08-14 same session as approval: instrument + reward v2 landed (5932fb6), pre-reg posted 09:43Z, R1-B launched under it 09:43:20Z. · [pre-reg](posts/2026-08-14-prereg-token-grpo-phase2-r1b.md)

<details><summary>full record</summary>

GRPO reward patch (option 2, owner-approved 09:16Z 08-14): the R1-A tripwire caught REWARDED shoving leakage - composite_reward's dense progress term pays cm-for-cm for boat displacement without a grasp, and knock-away is only the punished tail of that same funded strategy (endpoint-only, no grasp/contact channel exists in EpisodeResult). Work: (a) instrument MuJoCo grasp/contact ground truth (gripper closure + pad&lt;-&gt;benchy contact per replan tick) + trace-based knock event on the recorded distance_cm trace, with oracles; (b) pre-reg the patched reward - progress pay gated on (or heavily discounted without) grasp state, knock-away redefined on trace+contact, thresholds calibrated against the signal probe's banked 360-episode distribution where recomputable (rows lack the contact channel - state exactly what is and is not recomputable); (c) registered acceptance bar for the patch BEFORE R1-B launches under it.

</details>

---

**`sim-arm-surface-texture-mjspec`** · `cpu`

TRUE surface texture for the arm links via the mjSpec recompile path (escalation registered by the micro-texture REFUTATION 05:4xZ 08-14: statistically-matched screen-space grain read MORE fake - both CIs above zero, 0.698-&gt;0.751…

**boundary:** CLOSED 09:2xZ 08-14, work session: instrument landed (e408f9e, 11/11 physics oracles, zero-clip tanh generator, reflection rider registered), fit capped at the 0.42 no-clip headroom (lc 6.43 of real 8.36), registered 20x5 read all gates green -&gt; SECOND REFUTATION: PRIMARY +3.07e-07 CI [+2.42,+3.71]e-07 (0.698-&gt;0.718), MECHANISM +1.98e-07 (0.652-&gt;0.671). Arm-texture direction COLD at this abstraction level; graded arm stays the frontier; no further texture rung auto-queued (a normal-map / renderer-upgrade rung is a new design decision, owner-ask only). · [pre-reg](posts/2026-08-14-prereg-sim-arm-surface-texture-mjspec.md)

<details><summary>full record</summary>

TRUE surface texture for the arm links via the mjSpec recompile path (escalation registered by the micro-texture REFUTATION 05:4xZ 08-14: statistically-matched screen-space grain read MORE fake - both CIs above zero, 0.698-&gt;0.751 - the encoder wants coherent surface-tracking structure, not matched marginals): UV-mapped anisotropic print-layer texture assets on the link PLA materials (and servo glint via specular map if the path allows), model recompiled via mjSpec with physics-preservation oracles (qpos trajectories bit-equal or bounded, spawn/appearance/noise streams untouched, mass/inertia/contacts identical) as the hard bar before any render read. Gate on the same pinned 20x5 probe vs the v1-graded baseline 0.698/0.652 with the micro-texture read's stats as anchors. Pre-reg with explicit bar before any read. Higher risk than the composite route (recompile touches the model) - that is WHY it was sequenced second.

</details>

---

**`sim-wrist-view-material-read`** · `cpu`

Wrist-view read of the arm material fixes (follow-on to sim-arm-photometric-links + sim-mount-material-split, both banked opt-in): the wrist camera sees the arm links/gripper up close, and both material fixes change wrist-view pi…

**boundary:** Queued 04:5xZ 08-14 at the mount close (depth refill). CPU renders + ~0.02 GPU-h embeds; fully executable without the promotion (opt-in flags). EXECUTED + CLOSED 06:0xZ 08-14: all gates green (top 0.713 dead-center, wrist 0.561 in the registered reset band); PRIMARY wrist paired dknn5 -1.39e-08 CI95 [-4.53,+1.73]e-08 STRADDLES ZERO -&gt; wrist-neutral; visibility diagnostic: ~230 raw px of graded surface at home pose (servo 208/pla 21/mount 1); top rider replicated the mount read's stack delta bit-for-bit. Promotion asks proceed on top-side evidence, measured not assumed. Rollout-pose wrist gap (0.828) stays open, priced separately, not auto-queued. · [pre-reg](posts/2026-08-14-prereg-sim-wrist-view-material-read.md)

<details><summary>full record</summary>

Wrist-view read of the arm material fixes (follow-on to sim-arm-photometric-links + sim-mount-material-split, both banked opt-in): the wrist camera sees the arm links/gripper up close, and both material fixes change wrist-view pixels — the photometrics results post flagged the wrist knn5 as the promotion sanity. Run the encoder OOD probe on WRIST frames: paired v3 default vs the two-flag stack (arm_photometrics + mount_material), same 20x5 slot schedule, er_60k trunk; anchors = the OOD probe's wrist baseline (5-NN AUROC 0.828, ratio 1.33x, centroid 0.707). PRIMARY: paired wrist dknn5 CI95 &lt; 0. Feeds the pending promotion asks with the wrist-side fact instead of assuming it. Pre-reg with explicit bar before any read.

</details>

---

**`sim-mount-material-split`** · `cpu`

Camera-mount material split + white retexture (rider finding 08-14: the mount is WHITE/silver in reality, sim paints it black via _recolor_arm; arm-split read no_mount as the ONLY removal moving v3 toward real, 0.713-&gt;0.654 on 0.…

**boundary:** Queued 02:1xZ 08-14 at the photometric close. The per-pixel most distinctive class — likely the highest-leverage remaining arm fix. CPU + ~0.02 GPU-h per gate read. | EXECUTED + CLOSED 04:4xZ 08-14: material split via byte-identical gripper detach (oracle-pinned), real mount mined riding the dark gripper/wrist locks (81/156 frames, 91k px, NEUTRAL LIGHT GRAY [123,120,125] vs recolor black), fit chose the links' specular ceiling (1.0/0.1, albedo 0.455/0.430/0.431). Registered read SPLIT VERDICT: MECHANISM PASS decisively (only_mount_v1 0.821-&gt;0.793, CI [-1.16,-0.90]e-6, 93/100; vs plate -2.67e-6 at 100/100 — amputation confound REVERSED, presence now beats absence) but PRIMARY FAIL (v3_mount vs v3 CI includes zero, 0.66% px below the frame read's floor) -&gt; NO standalone promotion ask per frozen rule. Record-only: two-flag stack 0.713-&gt;0.702 CI-excl-0 — mount rides free if the photometrics promotion flips. Amendment 1 (locality oracle -&gt; table-reflectance bound, 0.02 reflectance) logged pre-read. Banked opt-in mount_material='v1'; no follow-up mount item (nothing to execute unless promoted). · [pre-reg](posts/2026-08-14-prereg-sim-mount-material-split.md)

<details><summary>full record</summary>

Camera-mount material split + white retexture (rider finding 08-14: the mount is WHITE/silver in reality, sim paints it black via _recolor_arm; arm-split read no_mount as the ONLY removal moving v3 toward real, 0.713-&gt;0.654 on 0.66% px): the mount geoms share wrist_roll_follower_so101_v1_material with the gripper's wrist-roll piece, so first split the material (mjSpec edit or vendored-XML material clone at load), then set the mount to a mined real color (same pose-projection mining path, mount population), gate on the pinned 20x5 probe (only_mount 0.821 / no_mount 0.654 anchors). Pre-reg with explicit bar before any read.

</details>

---

**`sim-arm-texture-followup`** · `cpu`

Arm texture follow-up (registered in the photometric pre-reg fail/residual branch): the grade closes the albedo+shine gap but leaves (a) print-layer local contrast real 8.4 vs graded 4.7 and (b) the servo glint tail p97 206 vs 125

**boundary:** Queued 02:1xZ 08-14 at the photometric close. Lower priority than the mount material split (bigger per-pixel offender). CPU + ~0.02 GPU-h per gate read. | EXECUTED + CLOSED 05:4xZ 08-14: composite-stage micro-texture (arm_texture='v1', deterministic static fields, zero shared-RNG draws, 6 oracles) fitted through the production composite to the mined real stats (PLA lc 8.24 vs real 8.36 dead-on; servo speckle-only, tail ~20% closed). Registered 20x5 read, all gates green (v3_photo 0.698 dead-center): REFUTED decisively in the registered over-texturing direction - PRIMARY +9.33e-7 CI [8.27,10.42]e-7 ABOVE zero, 3/100, AUROC 0.698-&gt;0.751; MECHANISM +1.30e-6 CI [1.22,1.38]e-6, 0/100, 0.652-&gt;0.740. The encoder reads spatial structure, not pooled statistics; screen-space grain reads as blotch. No promotion ask; escalation queued sim-arm-surface-texture-mjspec (not auto-run); flag stays opt-in. · [pre-reg](posts/2026-08-14-prereg-sim-arm-texture-followup.md)

<details><summary>full record</summary>

Arm texture follow-up (registered in the photometric pre-reg fail/residual branch): the grade closes the albedo+shine gap but leaves (a) print-layer local contrast real 8.4 vs graded 4.7 and (b) the servo glint tail p97 206 vs 125. Candidate: procedural print-layer texture on link PLA materials via mjSpec texture assets (needs model recompile path) or a composite-stage micro-texture on the arm mask (zero RNG draws, oracle-pinned); gate on the same pinned 20x5 probe vs the v1-graded baseline 0.698/0.652. Pre-reg with explicit bar before any read.

</details>

---

**`token-grpo-phase2-run`** · `gpu-local`

Token-GRPO phase-2 RUN — R0 COMPLETE 20:54:30Z 08-13 rc 0 (4 launches; crashes: device mix 9ffc1c1, Adam-init OOM d0b9a44, worker-headroom OOM 78cbb65; launch 4 = step_0001.pt resume, R1 resume path validated)

**boundary:** CLOSED 20:5xZ 08-13 at the R0 STOP boundary. Follow-up = token-grpo-phase2-rescope-prereg (queued): the registered option-A fallback + collapse mitigation, NEW pre-reg before any launch. · [pre-reg](posts/2026-08-13-prereg-token-grpo-phase2-run.md)

<details><summary>full record</summary>

Token-GRPO phase-2 RUN — R0 COMPLETE 20:54:30Z 08-13 rc 0 (4 launches; crashes: device mix 9ffc1c1, Adam-init OOM d0b9a44, worker-headroom OOM 78cbb65; launch 4 = step_0001.pt resume, R1 resume path validated). BOUNDARY VERDICT: STOP — VRAM gate FAIL (76.53 GiB steady-state &gt;= 75; option B measured-marginal on 1xH100), signal gate FAIL (wave-2 median group std 0.0087 cm, 5/8 groups all-draws-identical; one step at lr 5e-6 sharpened the 4B stack: chosen_nll 0.77-&gt;0.33, anchor_kl 4x/step), endpoint held-out collapsed 1.868 -&gt; -0.0, 0/20, paired delta -1.868 CI [-4.41,-0.03]. R1 NOT launched by frozen rule; ~3.8/5.5 GPU-h ops gate spent. Results in the pre-reg post.

</details>

---

**`sim-arm-photometric-links`** · `cpu`

Arm photometric fix, named target LINKS both instances (arm-split diagnostic 06:4xZ 08-13: links 88% of the arm's keep-only delta on 6.1% px; follower/leader sub-additive so both must be treated): replace the flat recolored link…

**boundary:** Queued 06:4xZ 08-13 at the arm-split close; CPU render + ~0.02 GPU-h embeds per gate read. | EXECUTED + CLOSED 02:1xZ 08-14 (commit 4515ab4): mined real link pixels at recorded poses (142 frames), fitted material grade (spec 1.0 shin 0.1, measured albedos), opt-in arm_photometrics='v1'; registered read GREEN — PRIMARY v3 0.713-&gt;0.698 CI-excl-0, MECHANISM only_links 0.705-&gt;0.652 (96/100), matches the no_mount amputation ceiling without amputating. Promotion + follow-ups queued as their own items.

<details><summary>full record</summary>

Arm photometric fix, named target LINKS both instances (arm-split diagnostic 06:4xZ 08-13: links 88% of the arm's keep-only delta on 6.1% px; follower/leader sub-additive so both must be treated): replace the flat recolored link material with a real-arm-derived photometric model — mine link-region pixel stats (median color, specular highlights, texture) from real v2 top frames via the leg-(a) segmentation masks projected onto real-registered poses OR simple material grade (specular+roughness) fit to real crops; gate on the pinned 20x5 probe vs the no_links removal ceiling 0.814 direction (patched-arm target: only_links moves toward plate_only). Record-only rider: mounts are per-pixel most distinctive (no_mount 0.713-&gt;0.654) — include a cheap mount-retexture arm in the same run if it costs no extra RNG draws. Pre-reg with explicit bar before any read.

</details>

---

**`discord-unreplied-inbox`** · `cpu`

Harness fix — discord unreplied inbox CLOSED 00:3xZ 08-14 (process-integrity, from the 08-13 missed-reply incident): discord.py `read` now appends every surfaced non-bot message to state/discord_unreplied.jsonl (dedupe by id); re…

**boundary:** CLOSED 00:3xZ 08-14; the structural fix for the ~2 h / ~1 h reply-latency class.

<details><summary>full record</summary>

Harness fix — discord unreplied inbox CLOSED 00:3xZ 08-14 (process-integrity, from the 08-13 missed-reply incident): discord.py `read` now appends every surfaced non-bot message to state/discord_unreplied.jsonl (dedupe by id); read AND babysit print the pending count as a loud FIRST line (babysit re-checks after its poll too); only an explicit `discord.py ack &lt;id&gt;` clears — result posts never do; `discord.py inbox` reprints entries in full (the recovery path for truncated read output). Gate met: 7 oracles in tests/test_discord_inbox.py (populate/skip-bot/dedupe/ack/unknown-ack/reprint/babysit-count), check.py 867-&gt;874 green, tick.md+work.md state the ack contract.

</details>

---

**`token-grpo-phase2-rescope-prereg`** · `cpu`

Token-GRPO phase-2 RE-SCOPE pre-reg (the R0 STOP boundary's registered fallback, 20:5xZ 08-13): design + pre-register the next rung on option A (patch-only trainable surface

**boundary:** CLOSED 22:1xZ 08-13 work session: pre-reg FINAL posted (posts/2026-08-13-prereg-token-grpo-phase2-r0a.md, frozen 81e020c) + instrument landed oracle-gated (69b03e8: option-A surface, differentiable KL penalty beta 0.5, advantage clip 2.0, kl-stop 0.06 mechanized; 18 loop oracles, check.py green). R0-A LAUNCHED same session (launch 2 21:58:04Z after the MUJOCO_GL env fix, addendum 1); run item queued at head.

<details><summary>full record</summary>

Token-GRPO phase-2 RE-SCOPE pre-reg (the R0 STOP boundary's registered fallback, 20:5xZ 08-13): design + pre-register the next rung on option A (patch-only trainable surface — dissolves the VRAM fail: 76.53 GiB steady-state was option B's, and the instability fallback rule named A) + explicit collapse mitigation calibrated off the R0 curves (one step at lr 5e-6: chosen_nll 0.77-&gt;0.33, anchor_kl 0.0215-&gt;0.0885, wave-2 5/8 groups all-8-draws-identical, held-out greedy -1.868 paired). Candidate levers to price IN THE PRE-REG, not ad hoc: lr down 5-10x; advantage tempering/clip tightening; KL penalty ON with the R0-measured scale as the line; eval-every 1 on the early rung so greedy damage is visible per step. step_0001/0002.pt on local disk are the diagnostic calibration artifacts. Pre-reg in-channel BEFORE any launch (delegation 11:07/11:18Z active).

</details>

---

**`token-grpo-phase2-r0a-run`** · `gpu-local`

Token-GRPO phase-2 R0-A smoke LIVE (launch 2 21:58:04Z 08-13, unit fontaine-grpo-r0a, pre-reg frozen 81e020c): 2 steps on option A (patch-only, ~10.5M params) at lr 1e-6, adv clip 2.0, kl_beta 0.5, kl-stop 0.06, eval-every 1

**boundary:** CLOSED 00:1xZ 08-14 work session: R0-A COMPLETE 00:05:09Z rc 0 (2.12 of 3.0 GPU-h ops gate), boundary verdict GO — all frozen reads green (wave-2 signal alive 2.03 cm 8/8 vs R0's same-seed collapse; eval delta -0.0239 CI [-0.0716, 0.0]; anchor_k3_pre 5.5e-07; VRAM 33.91; pace cum projection 20.3 &lt;= 22). R1-A launched 00:06:00Z by the frozen rule; r1a ride item queued. Results section in the pre-reg post. · [pre-reg](posts/2026-08-13-prereg-token-grpo-phase2-r0a.md)

<details><summary>full record</summary>

Token-GRPO phase-2 R0-A smoke LIVE (launch 2 21:58:04Z 08-13, unit fontaine-grpo-r0a, pre-reg frozen 81e020c): 2 steps on option A (patch-only, ~10.5M params) at lr 1e-6, adv clip 2.0, kl_beta 0.5, kl-stop 0.06, eval-every 1. Babysit rides it (registry entry grpo_phase2_r0a); at rc: boundary reads per the frozen table (plumbing/signal/per-step eval/KL line/VRAM/pace + the INERT rule) =&gt; GO launches R1-A (--resume step_0002.pt --total-steps 17, same flags), INERT goes in-channel as a re-price addendum, STOP re-scopes. Step-0 baseline reproduced 1.868 2/20 (4th bit-identical).

</details>

---

**`grpo-phase2-boundary-decision`** · `cpu` · **⛔ owner hold**

Token-GRPO phase-2 boundary decision (OWNER CALL, options posted in-channel 03:1xZ 08-14 at the R1-A tripwire stop): (1) R1-B re-price lr 3e-7 + kl_beta 1.0 from the banked step_0004 (fontaine-checkpoints/grpo_phase2_r1a, weights…

**boundary:** RESOLVED by owner steering 09:16Z 08-14: '(2) then (1)' approved in-channel (+ knock-away definition question, answered 09:21Z). Spawned grpo-reward-patch-prereg (option 2, next) and grpo-r1b-repriced-launch (option 1, sequenced behind the patch). · [pre-reg](posts/2026-08-13-prereg-token-grpo-phase2-r0a.md)

<details><summary>full record</summary>

Token-GRPO phase-2 boundary decision (OWNER CALL, options posted in-channel 03:1xZ 08-14 at the R1-A tripwire stop): (1) R1-B re-price lr 3e-7 + kl_beta 1.0 from the banked step_0004 (fontaine-checkpoints/grpo_phase2_r1a, weights-only), wire unchanged, ~9.6 GPU-h for 10 steps; (2) reward-patch pre-reg FIRST — in-reward knock-away penalty (progress reward currently pays for boat displacement without a grasp; the wire keeps firing at any lr while shoving pays), ~0 GPU-h to design; (3) stop the ladder, bank the negative accumulation read. Recommendation posted: (2) then (1). Ladder cum ~5.1 of the 22 GPU-h gate; ~17 headroom.

</details>

---

**`token-grpo-phase2-r1a-run`** · `gpu-local`

Token-GRPO phase-2 R1-A LIVE (00:06:00Z 08-14, unit fontaine-grpo-r1a, resume of R0-A step_0002.pt, steps 3-17, same frozen constants): babysit rides it (~0.96 GPU-h/step incl

**boundary:** Queued 00:1xZ 08-14 at launch. Leg budget 16.5 GPU-h in-registry; ladder cum ~20.3 of gate 22 at rc; 35 GPU-h total unchanged. | SELF-STOPPED 03:05Z 08-14 at step 5/17: knockaway tripwire exit 3 (fresh waves 0.406-&gt;0.359-&gt;0.312 vs the 0.167 x3 line — registered behavior). S6 endpoint reads: eval flat 1.8441 2/20 through step 4 (delta -0.0239 CI [-0.0716,0.0] touching zero — unharmed, unimproved; accumulation question cut short); knockaway pooled 69/192=0.359 but DECAYING while train success recovered 0-&gt;4-&gt;4; drift gentle (k3_pre 8e-7, nll softening). FROZEN RULE: tripwire -&gt; NO R2-A. step_0004 weights-only -&gt; fontaine-checkpoints/grpo_phase2_r1a (verified). Cost ~2.95 GPU-h (ladder cum ~5.1 of 22). Boundary options in-channel 03:1xZ (R1-B re-price / reward-patch pre-reg / stop) — owner adjudicates. · [pre-reg](posts/2026-08-13-prereg-token-grpo-phase2-r0a.md)

<details><summary>full record</summary>

Token-GRPO phase-2 R1-A LIVE (00:06:00Z 08-14, unit fontaine-grpo-r1a, resume of R0-A step_0002.pt, steps 3-17, same frozen constants): babysit rides it (~0.96 GPU-h/step incl. per-step eval, rc ETA ~14:3xZ 08-14). At rc: S6 endpoint reads (paired delta CI95 primary; knockaway vs 10/120 — WATCH: R0-A waves 0.234-&gt;0.359 vs the 0.167 x3 line, a legitimate exit-3 is registered behavior; success count) =&gt; R2-A extension only via the frozen R1-&gt;R2 rule, else boundary discussion in-channel (incl. lr/beta re-price if eval stays flat-at-noise).

</details>

---

**`token-grpo-phase2-instrument`** · `cpu`

Token-GRPO phase-2 instrument build (CPU, oracle-gated, zero behavior change - all new flags default-off), per posts/2026-08-13-token-grpo-phase2-design.md section 8: (1) --emit-training-rows on the parallel driver (frames + samp…

**boundary:** closed 2026-08-13

<details><summary>full record</summary>

Token-GRPO phase-2 instrument build (CPU, oracle-gated, zero behavior change - all new flags default-off), per posts/2026-08-13-token-grpo-phase2-design.md section 8: (1) --emit-training-rows on the parallel driver (frames + sampled ids + per-token chosen logprobs off the existing ActionCaptureStep surface; oracle: greedy logprobs bit-match a teacher-forced re-forward, draw-0 rows reproduce banked sequential rows); (2) GRPO step (advantage-weighted clipped token-CE; oracles: ratio-1 reduces to weighted CE, zero-advantage -&gt; zero grad, train-time grammar mask == rollout mask); (3) replay collator rows -&gt; CollatedBatch (fixture-episode bf16 logit-reproduction oracle); (4) loop harness (rollout-&gt;score-&gt;filter-&gt;step-&gt;eval + babysit heartbeat). check.py green per landing; ~2-3 sessions. || ITEM 1 CLOSED 09:3xZ 08-13 (418715c, off outage-recovered WIP 63bb1e2): capture surface + writer + 9 CPU oracles green (tests/test_token_rows.py) - memo section 8 'bit-for-bit' bar AMENDED with measured note (masked-softmax reduction bit-exact; teacher-forced re-forward within 2.4e-6 fixture / 1e-5 bound, one-shot-vs-incremental reduction-shape noise); recorded mask reconstructs from ids alone (trainer half of item 2's mask oracle, already green); draw-0-reproduces-banked check rides the first real GPU emit. Items 2-4 remain (~1-2 sessions). || ITEM 2 CLOSED 09:5xZ 08-13 (229d80f, pre-veto): GRPO step bijou/train_grpo.py - advantage-weighted clipped token-CE (DAPO clip-higher [0.8,1.28] frozen in GRPOConfig), training forward rides the SFT suffix_targets scaffold with the sampled ids, grammar mask recomputed trainer-side (grammar_masks_from_ids); sum+mean form pair (chunked-backward-ready); GRPOStats (ratio extremes, clip fraction, k3 KL drift). 7 CPU oracles tests/test_grpo_step.py: mask oracle both directions bit-for-bit (greedy+sampled), fresh-policy ratios 1 to the section-8 amended noise bound, ratio==1 reduces to advantage-weighted CE BIT-EXACTLY, zero-advantage -&gt; exact-zero grad every parameter (live-graph control), clip bounds bind with the right gradient stops per advantage sign, padding mask-multiplied out, loud guards. check.py 826 green. || ITEM 3 CLOSED 14:1xZ 08-13 (a268046, retargeted to the molmoact2 surface): ROLLOUT half — predict_action_discrete gains temperature+sample_rng masked-softmax sampling (Gumbel-max off stable_sample_rng keys, grammar_masked required — unconstrained sampling samples the 6.8% fallback class) + action_capture (ActionCaptureStep per bin step) so token_rows_from_capture + TrainingRowWriter work unchanged (block_base=action_token_start_id); driver: --molmoact2-temperature, --emit-training-rows wired for the discrete path (stores SHIM-APPLIED model-unit state), --draws&gt;1 with temperature. REPLAY half — bijou/molmoact2/replay.py: load_training_rows, grammar_masks_from_bins (bins-only budget arithmetic, loud on corrupt rows), verify_recorded_masks (bit-equality), replay_logprobs (one-shot teacher-forced forward over prompt+[action_start]+bins, same mask+positions as the incremental decode, WITH graph), molmoact2_grpo_loss -&gt; decoder-generic grpo_objective_sums. 7 CPU oracles (tests/test_molmoact2_replay.py) on the tiny-REAL-trunk with a widened REAL lm_head (base fixture head stopped below the action block): key-reproducible sampled decode, capture pure observation, mask contract both directions bit-for-bit, HEADLINE replay-reproduces-rollout logprobs within the registered 1e-5 bound (greedy + T=0.7), fresh-policy GRPO glue (clip 0, k3&lt;1e-8), writer/loader roundtrip, guards. check.py 849 green. REMAINING: item 4 (loop harness: rollout wave -&gt; score -&gt; z-filter -&gt; step -&gt; periodic eval + babysit heartbeat + registry entry) + the run pre-reg finalization. || ITEM 4 CLOSED 14:5xZ 08-13 (fa739e9): loop harness sim/grpo_loop.py - sampled rollout wave (parallel-driver lockstep machinery + TrainingRowWriter, train-seed stream 1000+8*step) -&gt; section-3 composite reward -&gt; group z-filter (ddof=0, dead groups dropped) -&gt; chunked sum-form GRPO step (full-count normalization = gradient-invariant chunking, oracle-pinned; option-B text stack fp32/TF32, vision frozen bf16, clip 1.0, non-finite skip) -&gt; anchor-KL k3 off recorded logprobs (one swapped reference forward) -&gt; paired held-out eval (seeded 10k bootstrap) -&gt; mechanized section-7 tripwires (exit 3) -&gt; babysit train-jsonl heartbeat + pruned rows + checkpoints. replay.py gained molmoact2_grpo_sums. 12 CPU oracles tests/test_grpo_loop.py (incl. loop e2e on the tiny-real-trunk fixture; measured: disk rows carry the JPEG budget, fresh-policy mean_ratio ~0.992 on random-init). check.py 861 green. ALL 4 INSTRUMENT ITEMS CLOSED; run pre-reg finalized 8548969 same session.

</details>

---

**`molmoact2-ar-head-port`** · `cpu`

MolmoAct2 AR (discrete) head port - owner steering 10:02Z 08-13 ('focus on that checkpoint and also just on AR GRPO for now', plan 👍'd in-channel): wire the release checkpoint's trained discrete pathway into our port so token-GRP…

**boundary:** closed 2026-08-13

<details><summary>full record</summary>

MolmoAct2 AR (discrete) head port - owner steering 10:02Z 08-13 ('focus on that checkpoint and also just on AR GRPO for now', plan 👍'd in-channel): wire the release checkpoint's trained discrete pathway into our port so token-GRPO can train it. Audit pinned 10:0xZ (in-channel, 2 posts): action block &lt;action_0..2047&gt; ids 151934-153981 contiguous, scaffold action_output/start/end 151931-3, action_mode 'both'; reference decode = unconstrained greedy full-vocab to EOS (cap 480), span-extract, OpenFAST decode(bins, T=30, D=6), q01q99 unnorm tag so100_so101_molmoact2; FAST artifact = pi's UniversalActionProcessor (DCT x10, min_token -55, chr-string ByteLevelBPE 2048, decode hard-asserts 180 coefficients else ZEROS fallback). Items: (a) FAST artifact behind our ActionCodec interface (oracle: encode/decode round-trip on banked chunks vs their scipy reference, bit-level); (b) greedy AR decode path on the port's existing prefill (oracle: token-for-token parity vs their unconstrained reference semantics on anchor rows - e2e_parity extension); (c) masked decode mode for RL - our budget-arithmetic grammar mask grafted onto their BPE piece lengths so every sampled draw is decodable by construction (oracle: masked greedy == unconstrained greedy wherever the stream was already legal, violation rate recorded); (d) discrete-mode sim eval GATE ~0.9 GPU-h (100 seeds @30 s, convmap shim, workers=8): is the AR pathway success-capable in OUR sim like the flow pathway's 9/100? This number gates all RL spend. check.py green per landing; (a)-(c) ~1-2 CPU sessions, (d) on owner compute go. || ITEMS (a)+(b-CPU)+(c) CLOSED 10:4xZ 08-13 (beeb93e / 526c4ad / 2a9e540, 14 CPU oracles green across tests/test_molmoact2_fast_codec.py + test_molmoact2_discrete.py) + REAL-CHECKPOINT SMOKE PASS (f0afc1e, ~0.01 GPU-h): all 4 anchor-row emissions well-formed + decodable (10-23 bins/chunk, ~0.6 s/chunk bf16); grammar_masked mode 0 violations, bins identical to unconstrained greedy. Audit extras pinned in code: trained BPE = 1005/2048 block rows; 7 quantization-hole symbols (ords 3,9,12,14,19,22,27) vanish in the released tokenizer - loud in our codec. REMAINING: (d0) sim-driver discrete adapter (CPU, executable now): rollout_sim_parallel serves predict_action_discrete (obs-&gt;prompt packing per worker, canonical shim state-in/actions-out like the convmap arm; oracle: adapter greedy chunk == molmoact2_discrete_smoke.py chunk on a pinned observation); (d) the 100-seed @30s AR-pathway eval ~0.9 GPU-h ON OWNER GO (asked 10:46Z + corrected 10:50Z, unanswered at close); (b2) formal token-for-token parity vs THEIR HF reference executing live - sequenced AFTER the gate read (no banked discrete anchors exist). || ITEM (d0) CLOSED 11:0xZ 08-13 (931b9a5): --molmoact2-discrete adapter in rollout_sim_parallel (official shim pinned, zeros-fallback accounting, out-json provenance; 2 CPU oracles) + REAL-CHECKPOINT PREFLIGHT PASS (1 seed x 2 s, 2 predicts, 0 fallbacks, ~0.3-0.8 s/predict). Item (d) eval is LAUNCH-READY on the owner go: MUJOCO_GL=egl uv run python -m sim.rollout_sim_parallel --molmoact2-discrete allenai/MolmoAct2-SO100_101 --seed 0 --num-seeds 100 --workers 8 --episode-seconds 30 --out-json ... (~0.5-0.9 GPU-h, babysit entry at launch). || ITEM (d) EXECUTED + READ 12:2xZ 08-13 (owner 11:07Z delegation = the go; pre-reg frozen faa5855, results 42c4485): 1/100 successes (seed 73 tick 622, a flow-success seed) =&gt; AR pathway SUCCESS-CAPABLE, token-GRPO lane GO on this checkpoint+pathway per the frozen rule. Validity green (strikes 0, 1.15/1.5 GPU-h). FINDING: 202/2991 predicts (6.8%) zero-fallback — greedy emissions fail their own decoder ~1-in-15 (zero-action chunks); the grammar_masked decode repairs exactly this class. AMENDMENT-1 ARM B live 12:29Z (unit fontaine-molmoact2-ar100b): same seeds + --molmoact2-grammar-masked, paired per-seed read, ETA ~13:3xZ — read + prune ride the next session. Remaining after arm B: (b2) HF-reference token parity (low priority now — the behavioral gate passed). || CLOSED 14:1xZ 08-13: arm B COMPLETE 13:31Z + paired read posted (d69c470) — grammar-masked decode REGISTERED IMPROVEMENT (B−A progress_final +0.728 cm CI95 [+0.147, +1.325] excl. 0, knock-aways 27→13, fallbacks 0/2996), masked = default serving mode per the 11:07Z delegation. Remnant (b2) HF-reference token parity: LOW PRIORITY, unqueued — behavioral gate passed both arms; re-queue only if a decode-semantics question surfaces.

</details>

---

**`sim-arm-appearance-leg`** · `cpu`

Arm appearance leg: the rendered arm (~7.1% of pixels) carries the remaining ceiling to the real-fg anchor (patched 0.556 / no_clutter 0.576 &gt;&gt; real-fg 0.328; only_arm 0.654 vs plate 0.866 in leg (a))

**boundary:** Closed 06:4xZ 08-13; follow-on photometric leg queued with the named target. · [pre-reg](posts/2026-08-13-prereg-sim-arm-split.md)

<details><summary>full record</summary>

Arm appearance leg: the rendered arm (~7.1% of pixels) carries the remaining ceiling to the real-fg anchor (patched 0.556 / no_clutter 0.576 &gt;&gt; real-fg 0.328; only_arm 0.654 vs plate 0.866 in leg (a)). Candidate fixes ladder (cheapest first): (1) photometric - the recolored flat-black arm vs the real arm's specular/texture (real-crop material stats or measured reflectance grade); (2) geometry-registered real-arm texture projection (hard: articulated, pose-dependent); scope a leg-(a)-style diagnostic first (WHICH arm sub-part carries it: gripper/links/mounts via geom-partition masks on the hooked harness, ~0.02 GPU-h). Pre-reg before any read. || CLOSED 06:4xZ 08-13: diagnostic COMPLETE, all gates green (in-run v3 0.713 in band; bridges plate_only 0.866 / only_arm 0.654 / no_arm 0.825 all in band). Registered rule names LINKS (88% of the only_arm paired delta on 6.1% px, CI-excl-0; gripper 26%, mount 31% — below both thresholds). Instance axis: follower/leader ~equal and sub-additive (only_follower -4.05e-6, only_leader -4.14e-6 vs whole arm -5.26e-6) — a fix must treat BOTH instances. Record-only: no_mount is the ONLY removal that moves v3 TOWARD real (0.713-&gt;0.654, 97/100, CI-excl-0) despite the absence-OOD confound — rendered mounts are per-pixel the most sim-distinctive class. Artifacts: analysis__sim_arm_split.json + chart + frame strip on fontaine-reports.

</details>

---

**`sim-foreground-appearance-pass`** · `cpu`

Foreground appearance pass - ALL LEGS DONE 05:4xZ 08-13. Leg (a) 04:5xZ: clutter the unique material class (no_clutter 0.576 vs v3 0.713, -0.137)

**boundary:** Closed 05:4xZ 08-13; promotion + arm-appearance follow-ups queued separately.

<details><summary>full record</summary>

Foreground appearance pass - ALL LEGS DONE 05:4xZ 08-13. Leg (a) 04:5xZ: clutter the unique material class (no_clutter 0.576 vs v3 0.713, -0.137). Legs (b)+(c) 05:4xZ (pre-reg 05:23Z, results in-channel 05:40Z, analysis__sim_fg_appearance_fix.json): real-crop RGBA patches (make_clutter_crops.py, bank-episode naive medians, novelty alpha, areas bit-match manifest) pasted at drawn poses by inverse fisheye warp (clutter_patch.py; episode grading; fixed_canonical pcb = identity) - REGISTERED GATE PASS: patched 0.556 vs v3 0.713 (dAUROC -0.157 vs -0.05 bar, paired dknn5 -2.02e-06 CI-excl-0, 100/100 closer); beats the no_clutter ceiling 0.576 by -0.020 (75/100, CI-excl-0) - full-recovery read fires. Integrity: v3 0.7127 in abort band, no_clutter 0.5764 reproduces leg (a) within +/-0.01, bit-exact oracle green 100/100. Promotion -&gt; new item sim-clutter-patch-promotion (owner_hold).

</details>

---

**`sim-top-gap-foreground-decomposition`** · `cpu`

Locate the remaining top-cam encoder gap (knn5 AUROC 0.713, unchanged by lens arms - it is the frontier number now that wrist reads 0.523 under curve-only): ablation embeds on the same 20x5 reset renders

**boundary:** EXECUTED + CLOSED 04:0x-04:3xZ 08-13 (pre-reg 04:03Z, results in-channel same session). Verdict: real-fg arm 0.328 (= clean anchor 0.283, below the 0.5 null) vs v3 0.713 -&gt; the whole residual top-cam gap lives in the RENDERED foreground pixels; arithmetic residue fg-&gt;plate minus plate-only +0.004 AUROC (paired +2.3e-07 CI-excludes-0 but ~5% of the armless shift, under the +0.05 bar); armless arms read FARTHER (0.869/0.865, 0/100 closer) - labeled confound, no-arm is itself OOD; shadow-band crop near-ceiling (0.989/0.988) but the box covers the arm region. v4 paired read replicated (-8.3e-08, 66/100 closer). Registered decision -&gt; next leg = foreground appearance (sim-foreground-appearance-pass queued). Artifacts: reports/analysis__sim_top_gap_decomposition.json + chart + arm strip on fontaine-reports (curl-200); reports.md section; ~0.02 GPU-h.

<details><summary>full record</summary>

Locate the remaining top-cam encoder gap (knn5 AUROC 0.713, unchanged by lens arms - it is the frontier number now that wrist reads 0.523 under curve-only): ablation embeds on the same 20x5 reset renders. Arms: (1) baseline v3 composite; (2) foreground-&gt;plate (rendered arm/benchy pixels replaced by the plate via the existing dynamic mask - if knn5 collapses toward the plate's own read, the gap lives in the RENDERED pixels, not the composite arithmetic); (3) shadow-region crop read (v4): does the fitted shadow band carry residual signal; (4) real-frame control: real episodes' arm pixels pasted on the plate (upper bound of what compositing can reach). Read: knn5 AUROC per arm vs the 0.713 baseline + per-frame paired deltas, same harness as the lens gate. Decision it feeds: which top-cam lever gets the next leg (arm appearance/materials vs mask edge vs shadow refinement). Cost: renders CPU, ~0.02 GPU-h embeds.

</details>

---

**`token-grpo-phase2-design-memo`** · `cpu`

Phase-2 token-GRPO design memo + pre-reg draft (AR trunk, t=1.0) per the frozen GRPO-signal-probe decision rule (both families cleared 08-13 00:0xZ: AR t=1.0 0.771 cm vs 0.25 bar at ~zero KL cost -&gt; token-GRPO first, Flow-GRPO SD…

**boundary:** EXECUTED + CLOSED 06:0xZ 08-13 (work session): design memo + pre-reg DRAFT posted as posts/2026-08-13-token-grpo-phase2-design.md. Contents per scope: measured-pace budget model (1.13 GPU-h/cell -&gt; ~0.0094 GPU-h/episode -&gt; ~0.75 GPU-h/RL-step; corrects the 08-12 sketch ~5x up), ladder R0 smoke 2 / R1 15 / R2 +25 steps, ~33 GPU-h gate 35 with R1-&gt;R2 boundary rule; composite reward (progress_final_cm + 10 success bonus - 2 tip - 5 strike, z-scored in-group ddof0, zero-var groups dropped); S=8 seeds x G=8 at t=1.0, clip-higher [0.8,1.28], mu=1, lr 5e-6, KL off but measured vs frozen er60k anchor; trainable-surface fork A patch-only vs B patch+text-stack (B recommended, 69.2 GiB preflight precedent); 5 tripwires incl. spread-collapse + violence-explosion off the probe's knock-away-tail hypothesis; instrument delta 4 items riding ActionCaptureStep. Owner asks: phase-2 go (re-posted), A/B fork, instrument-prestart permission. NO launch, NO registration - draft finalizes on go.

<details><summary>full record</summary>

Phase-2 token-GRPO design memo + pre-reg draft (AR trunk, t=1.0) per the frozen GRPO-signal-probe decision rule (both families cleared 08-13 00:0xZ: AR t=1.0 0.771 cm vs 0.25 bar at ~zero KL cost -&gt; token-GRPO first, Flow-GRPO SDE a=0.5 second). Memo scope: reward = sim success/progress on the v3/v4 composite eval, rollout budget model from the probe's measured 1.13 GPU-h/cell, group size + KL anchor + trunk-frozen-vs-open choices, abort tripwires, and the exact pre-reg bars. CPU-only; the launch itself pends the owner phase-2 go (open ask since 08-12).

</details>

---

**`release-eval20-officialmap`** · `gpu-local`

OWNER FOLLOW-UP 18:19:08Z 08-12 (FIRST GPU claim): rerun release-eval20-convmap under the OFFICIAL LeRobot v3.0-&gt;v2.1 conversion the owner linked (irenegracekp/molmoact2-so101 inference.py: offsets 0,90,90,0,0,0; signs 1,-1,1,1,1…

**boundary:** DONE 19:1xZ 08-12 work session (ridden in-turn, per-episode Discord stream per owner ask): sign-carrying --convmap-override landed (JOINT=[SIGN,]OFFSET + oracles) + --rows-jsonl per-episode stream; tripwires under official map recorded (lift mirror covers 7.5% vs +180's 27.9%; arm B first-action 2.62 vs anchor 6.31; arm A wrist identity 34.0 = known clamp signature, owner ordered anyway); both 20-seed arms run ~0.25/0.4 GPU-h. RE-DISPOSITION: INERT PARTIALLY OVERTURNED — official lift sign unlocks scene engagement (arm A seed 6 reach to 1.4 cm +4.61, seed 16 knock-away -5.26 = boat TOUCHED; arm B 2 approaches) but 0/20 pickups both arms, median 0.00; parent conclusion (grounding not units is the blocker) STANDS. A vs B null (-0.02 [-0.75,+0.66], 11/20 ties) — snippet identity wrist stays canonical. vs ftrig arms all CI-incl-0. CANONICAL SHIM = snippet map exactly (1,-1,1,1,1,1 / 0,90,90,0,0,0). MIRROR_MARGIN estimator lesson flagged to the box in-channel. Results: pre-reg amendment 1 + chart + rows + 40 videos on reports Space; Discord launch/per-episode/results posts 18:58-19:0xZ. · [pre-reg](posts/2026-08-12-prereg-release-eval20-convmap.md)

<details><summary>full record</summary>

OWNER FOLLOW-UP 18:19:08Z 08-12 (FIRST GPU claim): rerun release-eval20-convmap under the OFFICIAL LeRobot v3.0-&gt;v2.1 conversion the owner linked (irenegracekp/molmoact2-so101 inference.py: offsets 0,90,90,0,0,0; signs 1,-1,1,1,1,1). Verified discrepancy vs our fitted map (18:2xZ, CPU): shoulder_lift official (-1,+90)=90-arm vs ours (+1,+180) - the mirror QUALIFIED and covers the release box better (7.5% vs 27.9% uncovered) but lost to the pre-registered MIRROR_MARGIN=0.25 rule by 20.4pt; elbow/pan/wrist_flex/gripper match official exactly; wrist_roll ours -90 vs official identity (both 61% uncovered - span mismatch; identity clamps sim wrist home 77.6 above box ceiling 43.5, our -90 may absorb a rig-specific zero). CONSEQUENCE: INERT 0.00x20 read is suspect on lift - wrong sign direction-inverts decoded lift motion, matching the filmed swing-down-and-park; first-action detector is sign-blind at rest (bijection preserves action~state). WORK: (1) extend override syntax to carry sign (sim/convmap.py parse_overrides + resolve_map + driver flag), oracle it; (2) tripwires under official map incl. 3-seed first-action probe; (3) same 20 seeds, fixed post-flip sim, parallel workers=8, TWO wrist_roll arms if budget allows (official identity vs our -90) since ambiguous - else official-lift + our -90 wrist as primary; paired vs the existing release_convmap rows + step500/step2000. Gate &lt;=0.4 GPU-h. Amend the existing pre-reg page (amendment section, not a new page). Post correction/confirmation in-channel either way - the INERT claim must be explicitly re-dispositioned. Ack + plan posted 18:2xZ, owner said queue unless stop. || OWNER STEERING 18:34:34Z: run the snippet map EXACTLY as primary (signs 1,-1,1,1,1,1 / offsets 0,90,90,0,0,0 — wrist_roll IDENTITY per snippet; our -90 wrist arm only as optional secondary), and POST PER-EPISODE UPDATES in-channel as each seed's row lands (completion order under workers=8 is fine unless owner replies asking strict sequential — check channel before launch; confirmation posted 18:36Z). Watcher loop over row files -&gt; discord post per seed. || OWNER 18:36:29Z: second arm CONFIRMED — arm A = snippet exact (wrist_roll identity), arm B = snippet + wrist_roll -90, same 20 seeds both, arm A first, per-episode posts for both; ack 18:38Z. Gate stays &lt;=0.4 GPU-h (two parallel arms ~0.2 total).

</details>

---

**`release-eval20-convmap`** · `gpu-local`

OWNER PRIO 17:13:24Z 08-12 (FIRST GPU claim): released MolmoAct2-SO100_101 checkpoint directly in sim, WITH the unit shim per the owner-forwarded box note (/tmp/owner_note_molmoact2_norm.txt, committed copy fontaine/notes/molmoac…

**boundary:** DONE 18:1xZ 08-12 work session: shim verified (final map lift+180 elbow+90 wrist_roll-90; first-action 2.98 vs anchor 6.31), both pre-GPU tripwires dispositioned, 20 seeds run — release INERT (0.00 all seeds, boat never touched; repeatable off-task park). Cross-check banked and posted (lift AGREE; elbow agrees past the midpoint-gate near-tie; wrist_roll -90 empirical). Commit 5b3783e; rows/videos/chart on reports Space release_convmap/. · [pre-reg](posts/2026-08-12-prereg-release-eval20-convmap.md)

<details><summary>full record</summary>

OWNER PRIO 17:13:24Z 08-12 (FIRST GPU claim): released MolmoAct2-SO100_101 checkpoint directly in sim, WITH the unit shim per the owner-forwarded box note (/tmp/owner_note_molmoact2_norm.txt, committed copy fontaine/notes/molmoact2-unit-contracts-box-note.md). Raw-in-v3-sim is pre-declared MEANINGLESS (v3 rest lift ~-30 sits below the release box floor +45.2 -&gt; state tokens saturate, model blind, number = unit mismatch); run case 3: per-joint affine shim state-in (v3 -&gt; model units before its q01/q99 table) + action-out (model units -&gt; v3 before controller), labeled OFF-CONTRACT _convmap, never pooled with ftrig contract reads, treated as lower bound (release trained on mixed conventions through one table). Converted release already on disk: ~/marius-convert-gate/converted/molmoact2_so100_101_release. MANDATORY pre-run tripwires (from the note): (a) print release box from its norm_stats + verify mapped reachable set A-inv(box) covers the sim task workspace (clamp travels with the model); (b) first-action-vs-current-state check as unit-bug detector (release contract read: first_mae 18.0 vs state-copy 2.5 - a correct shim collapses this to ~state-copy scale; if it does not, STOP, do not spend the GPU). Then: same 20 seeds (sim100 list 0-19), fixed post-flip sim, parallel driver workers=8, paired vs step-500/step-2000 corrected arms (parallel-path rough rows). Also bank the cross-check the box asked for: does our sim calibration imply the same lift +180 / elbow +90 old-convention map as fit_convention_map snapped - flag disagreement in-channel. Ack posted 17:2xZ.

</details>

---

**`grpo-signal-probe`** · `gpu-local`

GRPO signal probe (proposed in posts/2026-08-12-grpo-sim-design-memo.md SS4, pends owner review - the memo's ask #1): rollout-only measurement of whether group-relative advantage has signal at our competence floor

**boundary:** Queued 11:4xZ 08-12 at memo close. BLOCKED on owner review of the memo (ask posted in-channel 11:4xZ). Memo SS4 is the draft-level design (linked as prereg); on approval: finalized pre-reg with final thresholds + instrument delta FIRST, then sequenced strictly after sim_parallel_oracle.py (owner 09:32Z first-GPU-item rule) AND the v3 rerun (anchor rows come from it). | APPROVED 13:16Z 08-12 ('Yes, let's do this, get everything ready for when I give you back the GPU') - 5 cells x 15 seeds x K=8 (+120 episodes vs the 4-cell shape). UNBLOCKED for prep: finalized pre-reg + EM sampler + a=0 bit-identity oracle land CPU-side ahead of release; GPU sequence per owner 13:36Z = parallel oracle -&gt; molmoact2-ftrig-sim-eval-20 -&gt; this probe (anchor rows still want the v3 rerun). | PREP COMPLETE 20:1xZ 08-12 (work session): finalized pre-reg POSTED (posts/2026-08-12-prereg-grpo-signal-probe.md — frozen: seeds 0-14, --episode-seconds 30 [the item text's '15 replans' was drift, sim100 protocol is 30 replans = 30 s], signal bar median group std &gt;= 0.25 cm, 5 cells + 2 anchor passes + registered a=0.3 hedge cell 5b, decision rule, gate &lt;=3.5 GPU-h parallel workers=8, within-driver paired-only discipline per the parallel-oracle FAIL). Instrument ALL LANDED: SDE sampler 80a5388, sequential draws 0f7ea86, SDE end-to-end + parallel (seed,draw) units 8b6d034, --episode-seconds c26a99e; check.py 797 green. Owner 13:36Z sequence SATISFIED (parallel oracle done 14:37Z, molmoact2-ftrig eval done 19:06Z) =&gt; LAUNCH-READY on GPU handback; launch checklist in the pre-reg (re-pin HEAD + checkpoint paths, babysit entry at launch). v3-rerun row join demoted to record-only cross-check (drivers differ), so the probe no longer hard-depends on the v3 rerun. | LAUNCHED 21:33:58Z 08-12 tick (standing-sequence handback call): unit fontaine-grpo-probe via launch_grpo_signal_probe.sh at HEAD 85e9a16 (2 anchors + 5 cells, out-dir outputs/sim/grpo_signal_probe); babysit entry live, first poll 85% util; ~2.8 h wall projected, results post at completion closes this item. | CLOSED 01:1xZ 08-13 (work session, ridden end-to-end): TRIPWIRE fired at cell-1 boundary (measured ~1.13 GPU-h/cell, 7-pass plan -&gt; ~5.9 GPU-h) -&gt; re-scoped in-channel 21:58Z (no objection): anchors + cells 1/2/5 ran, cells 3/4 (flow ODE fresh-noise, the channel the ceiling-ladder read already measured NULL) parked, re-queue on owner call only. ALL RUN CELLS CLEAR the 0.25 cm bar: cell1 AR t=1.0 0.771 (cost -0.351 CI [-1.117,+0.207]), cell2 AR t=1.6 2.461 (cost -1.081 CI [-1.556,-0.634] - dominated by cell1), cell5 SDE a=0.5 1.860 (cost -0.734 CI [-2.240,+0.294]); 5b hedge NOT triggered; 0 reset strikes everywhere. DECISION (frozen rule): BOTH families clear -&gt; phase 2 = token-GRPO on AR at t=1.0 first, Flow-GRPO SDE second, joint parked. ~3.57 GPU-h vs 3.5 gate (announced overage). Results: amendment 1 on the pre-reg page, chart + reads JSON on reports Space, per-cell in-channel posts at every boundary. GRPO-on-sim does NOT park. · [pre-reg](posts/2026-08-12-prereg-grpo-signal-probe.md)

<details><summary>full record</summary>

GRPO signal probe (proposed in posts/2026-08-12-grpo-sim-design-memo.md SS4, pends owner review - the memo's ask #1): rollout-only measurement of whether group-relative advantage has signal at our competence floor. 4 cells x 15 seeds x K=8 stochastic rollouts, v3 frames, sim100 conventions: er60k AR T=1.0, er60k AR T=1.6 (SimpleVLA-RL setting), teacher80k flow fresh-ODE-noise draws, ftrig4k flow fresh-ODE-noise draws; cell 5 CONFIRMED (owner 13:16Z 08-12: 'Yes, let's do this') = teacher80k SDE a=0.5 (a=0.3 the hedge constant if competence craters) (needs the ~30-line Euler-Maruyama sampler + bit-identity-at-a=0 oracle). Deterministic per-seed anchors join FREE from the v3 rerun rows (same seeds, same spawn stream - ordering logically forced). Instrument delta: --ar-temperature + --flow-draws K flags on rollout_sim over existing BijouPolicy knobs, per-draw RNG keyed (seed, replan, draw), draw-0 bit-identity oracle. Primary read: within-group std of progress_final_cm (+ best-point) per cell + fraction of groups surviving the dynamic-sampling filter; candidate bar (finalize at pre-reg) median group std &gt;= 0.25 cm. Secondary: competence cost vs anchor, guard-trip rates (strikes/upright/knock-offs), AR token entropy. Decision rule: no cell clears -&gt; GRPO-on-sim parks; AR clears -&gt; phase 2 = token-GRPO per SimpleVLA-RL recipe; flow-only clears -&gt; phase 2 = Flow-GRPO SDE expert-only; both -&gt; AR first. Gate &lt;=3 GPU-h parallel-path, &lt;=8 sequential.

</details>

---

**`sim-fit-real-lens-model`** · `cpu`

Fit the REAL rig lens into the wrist render (lit 0823 papers/fisheye-lens-fitting.md, owner-adopted 22:31Z 08-12 over wrist compositing): replace the assumed ideal-equidistant warp (V1_SRC_FOVY 72 source) with (a) theta-&gt;r fit by…

**boundary:** Queued 22:3xZ 08-12 on owner adoption. Sim-visuals lane; natural sequence: plumb-line fit (pure CPU, this or next session) -&gt; cubemap render path -&gt; probe-gated swap. Pairs with sim-composite-contact-shadows (same probe harness). | LEG (a) DONE 01:4xZ 08-13 (5581d6d): plumb-line theta-&gt;r fit landed (fit_lens_plumbline.py + oracles tests/test_lens_plumbline.py + house chart, outputs/sim/lens_fit/wrist_lens_fit.json). 382 seam chains from 132/150 pinned frames. FINDINGS: optical center (297.7, 253.2) — 22 px left / 14 px below the image midpoint (cx ~5-sigma by 20-frame-bootstrap); curve k2=+0.033 k4=+0.024 — the real lens compresses the periphery MORE than ideal equidistant: ray placement -2.2 px at r=240 (CI95 [-3.9,-1.2]), -12.8 px at the corner r=400 (CI95 [-17.2,-10.0]), both CI-exclude-0. Plank straightness RMS 1.07 px (deployed assumption) -&gt; 0.90 px (fitted); decompositions center-only 0.95 / curve-only 0.94. Remaining legs: (b) cubemap-&gt;equirect-&gt;fitted-lens render path (removes the 72-deg source ceiling; the fitted (cx,cy,k2,k4) is its stage-2 resampler spec), (c) probe-gated swap (wrist 5-NN must hold &lt;=0.548, reset-render probe ~0.02 GPU-h). | LEGS (b)+(c) DONE, ITEM CLOSED 03:4xZ 08-13 (25cf643 + close-out commit): cubemap-&gt;fitted-lens wrist render path landed behind lens_model='fitted' (SO101Sim; output-&gt;face map precomputed so runtime = one gather + per-referenced-face renders, 92-deg faces, face focal matched to the deployed source, base-axis headlight re-point kills the face-boundary shading seam; 8 oracles tests/test_sim_fitted_lens.py incl. top-cam bit-identical + rotated-cubemap self-consistency). GATE READ (pre-reg 03:27Z, results 03:40Z in-channel, 20x5 resets er60k, control 0.560): full fit 0.667 FAIL, center-only post-hoc arm 0.672 (center shift alone reproduces the regression - pose-degenerate with the 08-12 wrist re-tune), CURVE-ONLY REFIT 0.523 PASS (&lt;=0.548 gate), paired dknn5 -7.6e-07 CI95 [-8.5e-07,-6.8e-07], 96/100 frames closer, ~7x the contact-shadow GO effect, cost-neutral (1 face/tick, 73 vs 70 ms). WRIST_LENS_FIT now pins the curve-only params; default stays equidistant pending sim100 amendment 6 (owner ask posted 03:40Z). Full-fit center use requires joint pose+lens refit -&gt; queued sim-joint-pose-lens-refit (conditional). Artifacts on reports Space: chart__lens_gate.png, 4 gate JSONs, 3 sample frames (all curl-200).

<details><summary>full record</summary>

Fit the REAL rig lens into the wrist render (lit 0823 papers/fisheye-lens-fitting.md, owner-adopted 22:31Z 08-12 over wrist compositing): replace the assumed ideal-equidistant warp (V1_SRC_FOVY 72 source) with (a) theta-&gt;r fit by PLUMB-LINE calibration on the 150 pinned real reference frames (table planks = known-straight lines; no rig time needed), (b) cubemap-&gt;equirectangular-&gt;fitted-lens two-stage render (2603.02139's MuJoCo recipe, removes the 72-deg source ceiling entirely). Why it matters beyond appearance: policies use absolute pixel scale as a distance ruler (0.0025-&gt;0.60 cross-lens with RSA) - a mis-fit lens shifts perceived distance in ways the AUROC probe cannot see. Gates: plank-curvature residual vs real frames (direct theta-&gt;r readout), wrist 5-NN AUROC holds &lt;=0.548 (20x5 sensitivity), top-cam composite path bit-identical (real plate already carries the true lens; only the rendered-arm overlay changes if applied to top), reset-render probe ~0.02 GPU-h.

</details>

---

**`sim-wrist-compositing`** · `cpu`

Wrist-camera compositing for eval renders (owner steering 14:27Z 08-12: 'for eval we should be doing on both cameras'): today v2/v3 inpainting composites the TOP cam only; wrist is fully rendered (scene-matched + fisheye + grade)

**boundary:** Queued 15:0xZ 08-12 on owner steering. Sequenced in the sim-visuals lane at my discretion; before the next registered eval that reads wrist-driven behavior. | INVESTIGATED 22:29Z 08-12 (owner ask 22:21Z, ridden alongside the GRPO probe): CPU-only feasibility read landed (wrist_composite_feasibility.py, d177c0d) - plate start poses spread 20.8mm/5.1deg median (max 111mm/26.7deg, why the static plate mushed at 0.951); wrist is table-plane-dominated (median 100% of fisheye rays hit the plane, p10 75%) and pose is known per tick from FK, so plane-homography warp from the 26-plate bank is geometrically sound; BUT nearest-plate fill = median 87% / p10 49% BEFORE arm-footprint + parked-boat holes -&gt; residual sim-texture seams = SIMPLER T-III partial-matching hazard. RECOMMENDED render-only wrist + document the asymmetry + spend the effort on fit-real-lens-model instead (probe headroom ~zero at 0.548; a composite's real prize - true-lens pixel scale - the cubemap+calibrated-lens render gets seamlessly on ALL pixels). Option B (warp bank + one-time plate inpaint, ~1-2 sessions, gate AUROC &lt;=0.548) specced in-channel 22:29Z; pends owner pick. | DECIDED 22:31Z 08-12 (owner: 'Fair enough, let's go with your recommendation'): wrist stays RENDER-ONLY; the composite option is rejected (probe headroom ~zero at 0.548, warp-fill p10 49% before arm/boat holes -&gt; T-III seam hazard); the top-composited/wrist-rendered asymmetry gets a documented paragraph in the sim eval protocol; the effort redirects to fit-real-lens-model (promoted to its own queue item this session). CLOSED.

<details><summary>full record</summary>

Wrist-camera compositing for eval renders (owner steering 14:27Z 08-12: 'for eval we should be doing on both cameras'): today v2/v3 inpainting composites the TOP cam only; wrist is fully rendered (scene-matched + fisheye + grade). Design constraint from SIMPLER Table III (partial matching WORSE than none) + our own pure-composite wrist attempt reading worse on the pinned probe (0.951 vs 0.900): ship as one complete package or not at all. The wrist moves with the arm, so a static clean plate only matches the episode-start rest pose - candidate approaches: per-pose plate from the 26-episode start windows (probe-gated), or accept render-only wrist and DOCUMENT the asymmetry in the eval protocol. Gate: encoder-OOD probe rerun (~0.02 GPU-h), wrist 5-NN AUROC must not regress from 0.548 (current v3, inside real spread).

</details>

---

**`sim-wrist-bracket-flip`** · `cpu`

Flip the wrist camera-mount GEOMETRY to the real bracket-up side (owner spot 14:45Z 08-12 from the ftrig videos; probe-confirmed 15:00Z): the mount body (visual mesh + camera_box1/2 collision geoms + 12 g mass) sits 180 deg about…

**boundary:** Queued 15:0xZ 08-12, owner_hold: PHYSICS RE-BASELINE - flipping changes dynamics for every banked sim row (same class as the v3 rerun amendment); owner asked in-channel 15:00Z whether to execute next session. On unhold: fix + verify + short results post, then fold the re-baseline into the sim100 v3 rerun plan (one re-baseline, not two). | OWNER GO 15:01Z ('Let's do asap') -&gt; EXECUTED same session: _flip_camera_mount() at load (180 deg about mount-local x, both arms, geoms only - camera view bit-unchanged at world [0.15,-0,0.15]). VERIFIED: kinematic sweep below-table 31.9% -&gt; 1.4% bounding-conservative (center never below, min +5.3 mm; box1 0.00%); home bracket now 137/157 mm up (was 60/40 down); reset strikes 0/100; settle determinism + banked-spawn oracles green (7/7 EGL); physics tick 1.5 ms. REPLAY CONTROL LOSS RE-RUN: pinned L 0.0831 -&gt; 0.0751 vs floor 0.0701 (gap over floor -62%, exactly the counterfactual - the real-side bracket adds no new interference on the 26 reference episodes); arm MAE 1.88 -&gt; 1.50 deg. PHYSICS RE-BASELINE BOUNDARY: banked sim rows = pre-flip physics; rows from this commit on = flipped-mount physics; fold into the v3-rerun re-baseline. Known residual: body inertia still compiled with the 12 g mount on the old side (runtime geom moves don't recompile inertia).

<details><summary>full record</summary>

Flip the wrist camera-mount GEOMETRY to the real bracket-up side (owner spot 14:45Z 08-12 from the ftrig videos; probe-confirmed 15:00Z): the mount body (visual mesh + camera_box1/2 collision geoms + 12 g mass) sits 180 deg about the roll axis from the real assembly - at settled home it hangs 40 mm above the table on the JAW side; over the 26 reference episodes' recorded real poses its volume is below-table on 31.9% of frames (box2 center down to -46 mm), and dynamic replay of ep 21 grinds bracket-table contact on 22% of ticks. Sized: bracket-collisions-off replay control loss 0.0831 -&gt; 0.0751 vs floor 0.0701 (~62% of the servo-replay gap); elbow residual 3.78-&gt;3.37, wrist_flex 1.83-&gt;1.15. FIX = rotate the mount to the real side (not collision-delete: the real bracket can hit things on ITS side), runtime like _repose_wrist_cam, vendored XML untouched; camera view pose must stay pinned (already re-posed correctly). Verify: kinematic sweep ~0% below-table, replay control loss re-run (expect &lt;=~0.075), reset-strike 0/100 + settle determinism + 26.7ms tick oracles green.

</details>

---

**`grpo-on-sim-design-research`** · `cpu`

Design research (owner-called 09:23Z 08-12, research-only, no training): GRPO on the sim for our two heads

**boundary:** CLOSED 11:4xZ 08-12 work session, deliverable landed: design memo posts/2026-08-12-grpo-sim-design-memo.md (for owner review, nothing registered/launched) + papers/grpo-for-vla-heads.md upgraded to deep-read depth (correction recorded: piRL's main algorithm is PPO+GAE+critic, GRPO is its losing appendix baseline 90.0-vs-96.0 LIBERO; no KL anchor anywhere in piRL). Memo's named first cheap experiment = GRPO SIGNAL PROBE (rollout-only, no training, no RL code): 4 cells x 15 seeds x K=8 stochastic rollouts on v3 frames (er60k AR T=1.0 / T=1.6; teacher80k + ftrig4k flow fresh-noise draws), deterministic anchors joined from the v3 rerun rows, primary read = within-group std of progress_final_cm + dynamic-sampling survival rate; secondary = competence cost vs anchor + guard-trip rates (hacking price) + AR token entropy; gate &lt;=3 GPU-h parallel-path (&lt;=8 sequential). Decision rule in the memo: no signal -&gt; GRPO parks; AR signal -&gt; SimpleVLA-RL mapping phase 2; flow-only signal -&gt; Flow-GRPO SDE expert-only. Successor item grpo-signal-probe carries the owner hold.

<details><summary>full record</summary>

Design research (owner-called 09:23Z 08-12, research-only, no training): GRPO on the sim for our two heads. AR objective: token-level GRPO is standard - map group sampling onto rollout returns (progress_final as reward; k rollouts per seed = the group). Flow-matching head: the logprob problem - survey Flow-GRPO (ODE-&gt;SDE stochasticization), ReinFlow, DPPO (diffusion policy PO); what gives usable per-action logprobs for our heun-10 decode. Deliverable: a design memo post naming the first cheap experiment (arms, reward, group size, GPU gate) for owner review; papers pages for what it reads.

</details>

---

**`sim-composite-contact-shadows`** · `cpu`

Contact-shadow pass for the v3 composites (lit 0823, papers/composite-shadows.md): the pasted sim arm casts no shadow on the real clean plate - the one physics law every real frame obeys that no composite frame does, and an axis…

**boundary:** Closed 08-13. Open ask: sim100 amendment 5 (default style v3 -&gt; v4) on owner call.

<details><summary>full record</summary>

Contact-shadow pass for the v3 composites (lit 0823, papers/composite-shadows.md): the pasted sim arm casts no shadow on the real clean plate - the one physics law every real frame obeys that no composite frame does, and an axis no published pipeline measures (ConCent 2606.30268 randomizes it, ReBot/Re3Sim skip it). Recipe: estimate the room's dominant light direction once from static shadows already in the clean plates, project arm+boat silhouette onto the table plane, multiply-darken with soft edge, one strength parameter. Pure compositing arithmetic, no renderer change. Gate exactly like every render-style change: reset-render probe (~0.02 GPU-h), top-cam 5-NN AUROC must DROP from 0.773 (wrist path untouched, 0.548 must not regress); if the encoder does not care, the axis dies cheap.

</details>

---

**`lit-sim-improvement-levers`** · `cpu`

Lit slice (owner-called 09:23Z 08-12): sim-improvement levers beyond shipped plates/clutter/sysid - contact shadows for composites, camera-parameter fitting (real lens model vs our equidistant), DR schedules with measured transfe…

**boundary:** Queued 09:3xZ 08-12. | CLOSED 22:0xZ 08-12 (work session, ridden alongside the live GRPO probe): 3 papers pages landed same session per the permanent rule - papers/composite-shadows.md (ConCent 2606.30268 shadow recipe + ReBot 2503.14526 no-shadow collage + Re3Sim 2502.08645 foreground-realism-null ablation 0.70-&gt;0.70 + GreenAug 2407.07868 randomize-vs-match split), papers/fisheye-lens-fitting.md (2603.02139: cubemap-&gt;equirect-&gt;any-lens MuJoCo pipeline removes the 72-deg source ceiling; scale-overfitting 0.0025-&gt;0.60 with RSA; fit the real 130-deg module's theta-&gt;r), papers/dr-schedules.md (DORAEMON 2311.01885 60% vs AutoDR 26.7% real, alpha=0.5; one-scalar curriculum 2505.05753; eval/train firewall). 3 ideas.md hooks into #16 sim lane 0823: composite-contact-shadows (probe-priced ~0.02 GPU-h, gate top AUROC &lt;0.773), fit-real-lens-model, dr-schedule-for-sim-rl (conditional on probe decision rule). Contact shadows + lens fitting are the executable next probes in the sim-visuals lane.

<details><summary>full record</summary>

Lit slice (owner-called 09:23Z 08-12): sim-improvement levers beyond shipped plates/clutter/sysid - contact shadows for composites, camera-parameter fitting (real lens model vs our equidistant), DR schedules with measured transfer, SIMPLER-class visual-matching refinements. Feed the sim axis; papers pages same session.

</details>

---

**`sim-sysid-replay-control-loss`** · `cpu`

Replay control-loss probe (graduated from the 0820 deep-read close, SIMPLER's offline sysid validator - their Table II shows the loss is monotone with eventual ranking fidelity): replay recorded real action sequences from the 26…

**boundary:** Queued 12:xxZ 08-12 at the 0820 deep-read close. CPU-only, any GPU-busy window. Sequenced at my discretion; natural slot before any sim-training work (GRPO phase 2) or the next sysid-touching change, since it is the missing validator for the servo model everything else rides on. | CLOSED 13:5xZ 08-12 work session (sim/replay_control_loss.py + tests/test_replay_control_loss.py oracles, outputs/sim/replay_control_loss.json banked): pinned SERVO_SYSID L=0.0831 all-26 / 0.0849 held-out-23 vs real-command floor 0.0701 (menagerie 0.0973, upstream 0.0819) - under SIMPLER's best Table II anchor 0.131 (their MMRV 0.031 band; scale caveat stated). FINDING: joint-MAE win (pinned 1.88 vs upstream 2.74 deg) does NOT carry to EE space - elbow residual 3.78 deg (unmodeled payload, shared by all candidates) dominates via the largest lever arm (4.64 mm/deg at median pose; wrist_roll 0.21). Read is GOOD -&gt; no tuning item queued; per-joint elbow gains stay the named next rung if elbow ever gates. Zero-bias grep: no additive per-joint constants in the sim consume path (deg2rad only) or bijou normalization (per-dataset stats kill inter-rig offsets by design); the lerobot-sim2real +6.8deg class would live rig-side in calibration shared by action AND state - invisible to replay/training, exposed only via sim world-frame geometry (pinned by visual matching instead).

<details><summary>full record</summary>

Replay control-loss probe (graduated from the 0820 deep-read close, SIMPLER's offline sysid validator - their Table II shows the loss is monotone with eventual ranking fidelity): replay recorded real action sequences from the 26 reference episodes through the sim physics-only (no GL, no GPU, mj_step + servo model), score SIMPLER's loss L = mean EE ||dx|| + mean arcsin(||dR||_F / 2sqrt2) between sim and recorded real trajectories, per episode + pooled. Deliverable: the number for our current servo sysid (menagerie-vs-TheRobotStudio kp question gets its first measurement), a per-joint error breakdown, and - only if the read is bad - a follow-up tuning item (SIMPLER used 3-round simulated annealing over stiffness/damping; BAM's identified STS3215 model is the informed prior). Also grep the calibration path for silent per-joint zero bias (lerobot-sim2real ships a hardcoded +6.8deg elbow offset in the same LeRobot calibration stack we use). Instrument + short results post; pre-reg-light (measurement, no registered claim gates on it).

</details>

---

**`lit-so101-benchmark-envs`** · `cpu`

Lit slice (owner-called 09:23Z 08-12, supersedes the lit pause for this thread): benchmark environments near the SO-100/SO-101 embodiment - lerobot-sim2real (ManiSkill3 SO-100), gym-lowcostrobot, LIBERO/SimplerEnv/RoboCasa for pr…

**boundary:** Slice 0820 page LANDED 09:3xZ 08-12 (papers/so101-sim-ecosystem.md, update to the 08-11 census). | CLOSED 12:xxZ 08-12 work session: both deep reads DONE - (1) lerobot-sim2real at implementation depth (papers/lerobot-sim2real-recipe.md, NEW page): CORRECTION to the survey - the project has NO sysid (system_id_so100.npy is dead code, PD gains untuned 1e3/1e2, one hardcoded +6.8deg elbow offset); transfer bought by target-integrated delta actions (command stream open-loop identical sim&lt;-&gt;real) + per-control-step camera-pose DR (+-2.5cm) + hard-mask greenscreen 128x128 aligned by eye, no quantitative alignment metric; qvel banned from obs (STS3215 too noisy); 91.6% = 22/24 human-judged. (2) SIMPLER spec upgrade (sim-as-eval.md): MMRV formula + repo reference code + hardcoded anchor tables, ~1.5k real episodes total validation bill over 6+3 policy points (early/mid checkpoints manufacture spread), offline-MSE strawman 0.375 vs 0.056, sysid replay control loss MONOTONE with MMRV (Table II), partial visual matching WORSE than none (Table III 0.142 vs complete 0.050 - v3.1-wrist caution). Cube-grasp port RE-SCOPED, does NOT graduate on benchmarking grounds (RL-on-task vs imitation-zero-shot is not a comparison; full port spec recorded in the recipe page as the ready-made GRPO-phase-2 training task). GRADUATED instead: sim-sysid-replay-control-loss (successor queue item, cpu).

<details><summary>full record</summary>

Lit slice (owner-called 09:23Z 08-12, supersedes the lit pause for this thread): benchmark environments near the SO-100/SO-101 embodiment - lerobot-sim2real (ManiSkill3 SO-100), gym-lowcostrobot, LIBERO/SimplerEnv/RoboCasa for protocol; what we could adopt or bridge so our sim numbers land next to published ones. Papers pages same session per the permanent rule; ideas.md hooks.

</details>

---

**`molmoact2-ftrig-sim-eval-20`** · `gpu-local`

ftrig molmoact2 sim eval, 20 seeds + videos (OWNER-CALLED 13:16Z/13:36Z 08-12, top prio on GPU release, sequenced directly after sim_parallel_oracle.py as the parallel path's first consumer - owner: 'sim-parallel-rollouts first a…

**boundary:** Queued 13:5xZ 08-12 on owner call. AutoEval caution applies (different stack family - sim fidelity claim resets to zero until spot-checked): frame the numbers as exploratory. First molmoact2/molmo_flow rollout through BijouPolicy on sim - expect integration edges; budget one debug cycle. | PRE-STAGED 14:2xZ 08-12: BijouPolicy assembles the checkpoint on CPU (bijou@molmoact2_rig_r1_step2000, chunk 30; one API note - BijouPolicy wants a Path, not str). Command: MUJOCO_GL=egl uv run python -m sim.rollout_sim --checkpoint ~/marius-convert-gate/converted/molmoact2_rig_r1_step2000 --num-seeds 20 --method euler --sample-steps 10 (videos default on; parallel driver swap-in if the oracle is green). | CLOSED 15:0xZ 08-12 same session (sequential driver - oracle failed; one integration fix: rollout_sim stats fallback to the checkpoint's merged table for per-dataset-less converted checkpoints): 0/20 success, mean progress_final -0.84 cm (median 0.00), 7/20 real approach progress (best +1.3 cm seed 1), 4 knock-aways &gt;=1 cm (worst -9.1 cm seed 4). QUALITATIVE HEADLINE: the arm moves with intent - reaches the boat, jaws adjacent, then misses/shoves (er60k froze on 13/20). Videos + rows on fontaine-reports /molmoact2_ftrig_eval20/. Watching the videos prompted the owner's bracket question -&gt; see sim-wrist-bracket-flip. · [pre-reg](posts/2026-08-12-prereg-molmoact2-ftrig-sim-eval.md)

<details><summary>full record</summary>

ftrig molmoact2 sim eval, 20 seeds + videos (OWNER-CALLED 13:16Z/13:36Z 08-12, top prio on GPU release, sequenced directly after sim_parallel_oracle.py as the parallel path's first consumer - owner: 'sim-parallel-rollouts first and we test it on the eval of ftrig molmoact2 ... 20 episodes first, keen to get some rough numbers and videos'). Checkpoint = ~/marius-convert-gate/converted/molmoact2_rig_r1_step2000 (in-house format 3, molmo_flow decoder, backbone ref ~/checkpoints/molmoact2-so101-rig-r1-step2000-hf; located + read_checkpoint_info-verified 13:5xZ). Rough-numbers pass, NOT a registered claim: 20 seeds (sim100 seed list 0-19), v3 frames, videos on, progress/success/guard reads vs the er60k v3 anchor rows; results post + report page same session. If the parallel oracle FAILS, fall back to sequential rollout_sim (paired-only rule) rather than blocking the eval.

</details>

---

**`sim-parallel-rollouts`** · `gpu-local`

OWNER-SEQUENCED FIRST GPU ITEM (09:32Z 08-12: 'Once I relinquish the GPU, remember to do sim-parallel-rollouts before any other experiments')

**boundary:** Queued 08:5xZ 08-12 (owner yes 08:44Z); RE-SEQUENCED 09:32Z: runs FIRST when the owner releases the GPU, before any other experiment (incl. the rerun). CPU design/scaffold work may start during the reserved window. | SCAFFOLD LANDED 10:1xZ 08-12 (commit 1e4e16f, check.py 710 green): sim/rollout_sim_parallel.py (spawn env-workers + batched parent policy, deterministic lockstep-rounds scheduler, stable-noise identity triple preserved per row), shared run_episode_loop refactor + streaming VideoWriter, 5 CPU-tier harness-equivalence oracles, and the GPU bit-match oracle instrument fontaine/scripts/sim_parallel_oracle.py. PRE-REG POSTED: posts/2026-08-12-prereg-sim-parallel-rollouts.md - frozen decision rule (GREEN at 2 AND 8 workers on er60k seeds 0-5 =&gt; parallel path may produce registered numbers at validated settings; FAIL =&gt; paired-only fallback with per-use amendment, no mixing with banked sequential rows) + record-only 20-seed throughput read, gate &lt;= 1 GPU-h total. REMAINING = the GPU leg only: run sim_parallel_oracle.py on release (FIRST item per owner 09:32Z), results post same session. | GPU LEG CLOSED 14:37Z 08-12 (owner released 14:17Z, oracle ridden in-session): FAIL at workers=2 - 3/6 seeds bit-identical, 3/6 diverge macroscopically (final_cm off 5.8/7.4/0.8 cm; spawn/reset/strike fields ALL matched, so env determinism held and the divergence is the batched bf16 decode, amplified by contact physics). Frozen rule applied: sequential stays the registered path; parallel = paired-only with per-use amendment, no mixing with banked rows. workers=8 leg skipped (gate decided at 2). Throughput datum: 1.73x at 2 workers (8.8 -&gt; 5.1 min / 6 episodes). Diffs banked outputs/sim/parallel_oracle/. Named follow-ups (not queued): fp32-expert retry, registered tolerance. · [pre-reg](posts/2026-08-12-prereg-sim-parallel-rollouts.md)

<details><summary>full record</summary>

OWNER-SEQUENCED FIRST GPU ITEM (09:32Z 08-12: 'Once I relinquish the GPU, remember to do sim-parallel-rollouts before any other experiments'). Parallel sim rollouts with a shared policy (owner-approved 08:44Z 08-12): N env workers (each owns its SO101Sim + EGL context, physics+render) feeding ONE batched policy server holding a single checkpoint copy - the lerobot-style policy-server split already in the repo for rig rollouts. At batch 1 the H100 idles during heun-10; batching N obs is near-free into the low tens. Box has 26 cores -&gt; ~8-12 render workers before CPU contention; target: a 100-seed arm in ~20-30 min (vs ~1.5 h), the 5-arm sim100 rerun within an afternoon. MUST ship with a determinism oracle: batched rollouts reproduce the sequential per-seed rows bit-for-bit (or within a stated decode tolerance, registered before use). Pre-reg the oracle + a 2-worker smoke before any registered eval uses the parallel path.

</details>

---

**`sim-disk-position-prereg-draft`** · `cpu`

Disk-position draws pre-reg draft (the (c) leg the content-diversity item scoped out as task semantics): draw the disk's world xy per SPAWN seed (not appearance - success geometry moves with it) from the measured real between-epi…

**boundary:** Queued 07:2xZ 08-12 at the content-diversity close. Executable now (pure drafting); pends nothing. Sequencing: after sim100-v2-rerun-amendment-draft if the owner unholds the rerun first. | DONE 11:0xZ 08-12: DRAFT pre-reg posted (posts/2026-08-12-prereg-disk-position-draws.md) - six registered decisions: ABSOLUTE draws from the measured box (frame alignment trusted on the mouse precedent; the pinned (0.22,0.11) sits OUTSIDE the measured y range - new finding), success/metrics follow via disk_center update, spawn goes DISK-RELATIVE (current box re-expressed as deltas, ~9.5 cm task preserved), joint validity clamp by rejection (constants finalized by a 1000-seed policy-free sweep, truncation fraction reported), banked rows declared NON-comparable (protocol v2 'sim100-D', within-run pairing survives fully), spawn-stream discipline keeps it style-orthogonal with a disk_draws=False bit-identity guard. Grounding-probe diagnostic registered (tracker vs memorizer slope). Sequenced AFTER the v3 rerun to avoid confounding. Implementation (6 oracles + sweep) = follow-up CPU item on owner sign-off. · [pre-reg](posts/2026-08-12-prereg-disk-position-draws.md)

<details><summary>full record</summary>

Disk-position draws pre-reg draft (the (c) leg the content-diversity item scoped out as task semantics): draw the disk's world xy per SPAWN seed (not appearance - success geometry moves with it) from the measured real between-episode spread, now banked in assets/real_plates/bank/bank_manifest.json (disk_record_only: present 21/26 A episodes, x 0.083-0.288, y -0.193-0.097; the sim pins it at (0.22, 0.11)). Draft must handle: success() geometry follows the drawn disk; spawn-region overlap (benchy spawns relative to a disk that now moves); paired-arm comparability (same seed -&gt; same disk across arms); and whether banked sim100 spawns stay bit-comparable (they do not if spawn ranges become disk-relative - the draft must choose and say so). CPU only; the eval-protocol change itself holds for owner sign-off with the rerun call.

</details>

---

**`sim100-v2-rerun-amendment-draft`** · `cpu`

sim100 v2-rerun pre-reg AMENDMENT draft (CPU only, no launch): the rerun item's own protocol requires a short amendment before launch (new arm names for v2 visuals, re-baseline) - draft it now so the eval is launch-ready the mome…

**boundary:** Queued 06:2xZ 08-12 at the wrist-periphery close. Executable now (pure CPU writing); pends nothing. Its EXECUTION twin (the eval itself) remains sim100-v1-rerun, owner_hold. | DONE 10:2xZ 08-12: DRAFT amendment posted (posts/2026-08-12-prereg-amendment-sim100-v3-rerun.md) - inherits sim100 protocol, changes frames (v3 re-baseline table incl. GPU-path probe numbers), arm set (teacher80k ADDED post-spot20 as the confirmatory read, snap30k dropped double-null, rungs stay dead - all flagged as owner decision points), primary read (paired v3-v0 per-seed at n=100, the spot20 instrument), registered priors per arm, disk stays pinned for pairing, execution contingent on the sim-parallel-oracle outcome (Path A ~2-3 GPU-h / Path B &lt;=10 GPU-h gate). Finalization checklist runs at owner unhold; the eval twin sim100-v1-rerun remains owner_hold. · [pre-reg](posts/2026-08-12-prereg-amendment-sim100-v3-rerun.md)

<details><summary>full record</summary>

sim100 v2-rerun pre-reg AMENDMENT draft (CPU only, no launch): the rerun item's own protocol requires a short amendment before launch (new arm names for v2 visuals, re-baseline) - draft it now so the eval is launch-ready the moment the owner unholds. Content: arms er60k + ftrig4k + hold re-rendered under render_style=v3 (owner-approved default flip 07:29Z 08-12: plate-bank top + clutter draws, re-tuned wrist pose), same 100 seeds / metric / gates as posts/2026-08-11-prereg-sim-policy-eval-100seeds.md; visual re-baseline table = probe reads (top 0.890-&gt;0.673, wrist 0.835-&gt;0.548); expected-behavior priors stated in advance (the fisheye+pose geometry deltas are exactly the spatial-mismatch signature named at the v1 close - register what a behavior change would look like vs the 0/500 baseline). Post as DRAFT pending owner call; the 20-seed er60k spot-check option stays first-listed.

</details>

---

**`sim-content-diversity`** · `cpu`

Sim content diversity v3 (the axis every read since the OOD probe names: sim is ~4% k std/mean vs real 45%, and neither lighting jitter (v1) nor a fixed real background (v2) moved it): per-reset CONTENT variation for the composit…

**boundary:** CLOSED 07:2xZ 08-12: registered bar MISSED on the spread leg (top k std/mean 0.038 -&gt; 0.114 vs &gt;= 0.15) while the AUROC leg over-met (0.773 -&gt; 0.673, k-ratio 1.02x - top composites inside the real spread, best top read yet). Wrist guard bit-identical GREEN (0.548). render_style='v3' shipped, default STAYS v2 per the registered flip rule; flip is an owner ask (thumbs-up on the results post). Results: posts/2026-08-12-sim-content-diversity-results.md · [pre-reg](posts/2026-08-12-prereg-sim-content-diversity.md)

<details><summary>full record</summary>

Sim content diversity v3 (the axis every read since the OOD probe names: sim is ~4% k std/mean vs real 45%, and neither lighting jitter (v1) nor a fixed real background (v2) moved it): per-reset CONTENT variation for the composite - (a) per-episode plate banks from the A half (needs a mining pass that masks the boat's real positions so no ghosts bake in; per-episode plates carry real lighting/clutter states), (b) clutter-state draws (mouse/laptop/pcb poses drawn per appearance seed from the real between-episode spread), (c) disk position drawn from the real distribution - NOTE (c) changes task semantics (success geometry), needs its own pre-reg beyond appearance-only. Read: same reset-render probe + the homogeneity 20x5; bar to be registered (candidate: sim k std/mean toward &gt;=15% without top 5-NN AUROC regressing past 0.790).

</details>

---

**`sim-wrist-periphery-fix`** · `cpu`

Wrist-cam periphery re-tune under the v1 fisheye (small, CPU + ~0.04 GPU-h probe reads): the scene pass moved wrist 5-NN AUROC 0.835-&gt;0.786 (best read of the whole v1 study - content/pose is what the wrist tracks) but the fisheye…

**boundary:** Queued 05:0xZ 08-12 at the v1 close (rides the v1 pre-reg's instrument + axes; wrist was explicitly secondary there). Executable now; independent of the inpainting item. | NOTE 05:4xZ 08-12: the v2 wrist composite read WORSE (0.951) than the v1 wrist path (0.900) - episode-start wrist poses differ by degrees across episodes so the clean plate is mush (coverage 0.36); shipped v2 keeps the v1 wrist path. This item's bar unchanged (&lt;=0.786 scene-only level); a per-episode-aligned wrist plate is a possible extra axis once the pose/periphery is right. | CLOSED 06:2xZ 08-12 (work session): registered bar SMASHED on the first candidate - wrist 5-NN AUROC 0.900 -&gt; 0.548 vs &lt;=0.786 (k-ratio 0.97x: sim wrist now sits INSIDE the real spread; 20x5 sensitivity 0.550, stable). Camera moved from the wrist top behind the gripper (world ~(0.096,-0.004,0.160), 55deg) to over the jaw base (~(0.150,0,0.150), 65deg, same image-right=-y roll): the gripper-body mass that filled the bottom ~40% of frame drops out, leaving jaw tips in the bottom quarter over full-frame table like every real start frame. Guard green: top 0.773 bit-identical (render path untouched). Oracles 10 green (qpos bit-identity across styles + spawn stream vs banked v0), check.py 704 green. Shipped as the _repose_wrist_cam default (all render styles). The per-episode-aligned wrist plate axis named at the v2 close is RETIRED - a composite cannot beat inside-the-real-spread. · [pre-reg](posts/2026-08-12-prereg-sim-wrist-periphery.md)

<details><summary>full record</summary>

Wrist-cam periphery re-tune under the v1 fisheye (small, CPU + ~0.04 GPU-h probe reads): the scene pass moved wrist 5-NN AUROC 0.835-&gt;0.786 (best read of the whole v1 study - content/pose is what the wrist tracks) but the fisheye+grade passes regressed it to 0.900 because the 72deg source pulls sim-specific periphery (arm body mass, table far edge, floor band) into frame. Iterate ONLY the wrist: camera pose/height under the wider source, what the periphery shows (table extent, background band), gripper-mass framing vs the real bottom-quarter jaws - reset-render probe per iteration, wrist 5-NN &lt;=0.786 (scene-only level) as the bar, top read must not regress. Fold any deltas into _repose_wrist_cam / the scene XML.

</details>

---

**`sim-visual-inpainting`** · `cpu`

Sim visual matching v2 - real-frame INPAINTING (the named lever after v1's registered miss, SIMPLER-RT recipe): bake actual rig pixels as the static scene instead of approximating materials/optics - per camera, composite the real…

**boundary:** Queued 05:0xZ 08-12 at the v1 close. Executable now (CPU + ~0.02 GPU-h probe reads); pends nothing. If the owner instead calls the behavioral spot-check on sim100-v1-rerun, run that first - it may show the geometry fixes already changed behavior, re-scoping what inpainting must buy. | CLOSED 05:4xZ 08-12 (work session): registered bar MET - top 5-NN AUROC 0.890 (v0) -&gt; 0.876 (v1) -&gt; 0.773 (v2) vs &lt;=0.790; overfit tripwire clear (&gt;&gt;0.5). A-half clean plates (26 eps, video-frame disjointness from held-out B verified: last plate frame 17066 &lt; first B frame 17100) + segmentation composite; wrist composite regressed (0.951 vs 0.900 - cross-episode mush plate) so shipped render_style=v2 (NEW DEFAULT) keeps the v1 wrist path (pure-composite read reproducible at f75c341). Homogeneity unchanged (~4% vs 45%) - content variation named the diversity lever (successor item queued). Results post 2026-08-12-sim-visual-inpainting-results.md; 3 probe jsons + 2 galleries on fontaine-reports (curl 200). ~0.06 GPU-h (gate 0.3). · [pre-reg](posts/2026-08-12-prereg-sim-visual-inpainting.md)

<details><summary>full record</summary>

Sim visual matching v2 - real-frame INPAINTING (the named lever after v1's registered miss, SIMPLER-RT recipe): bake actual rig pixels as the static scene instead of approximating materials/optics - per camera, composite the real background (median/clean-plate of real frames, table + clutter + room) with rendered dynamic content (arms, benchy, disk) via render masks; the v1 fisheye/pose matching makes the geometric alignment feasible. Read: same reset-render probe (top 5-NN AUROC 0.890 baseline unmoved by v1; target the registered 0.79 line first, then 0.5); per-iteration cost ~0.02 GPU-h. Pre-reg (short, reuses the v1 instrument + bar semantics) before any GPU minute. Also carries v1's diversity finding: sim is ~10x too homogeneous at the encoder and lighting jitter does not fix it - content variation (hand/cable clutter states, real-plate rotation) is the axis.

</details>

---

**`sim-visual-matching`** · `cpu`

Sim visual matching (CPU + render minutes, SIMPLER's second lever after controller sysid): close the sim-vs-rig APPEARANCE gap for the two policy cameras - compare sim renders vs real rig frames (so-frame REAL|SIM|OVERLAY per the…

**boundary:** Queued 20:3xZ 08-11 at sim-servo-sysid close. Explicitly OPTIONAL before the 100-seed eval (visual gap affects policy inputs, not physics); if the eval's free validation arm shows sim ordering matching the banked panel trajectory, this may stay a v1 rung. | UPDATE 03:4xZ 08-12: now THE named lever after sim100 close (0/500; direction tracks visual familiarity — see sim-policy-eval-100seeds close). Promised in-channel 01:30Z: one pre-reg combining the encoder-OOD-probe baseline + visual matching v1 (real top-cam background/table baked as scene textures, boat+disk color match, camera pose match; then a 20-seed texture-sensitivity read). Owner goal: &gt;=1 success on the 100 seeds. | BASELINE MEASURED 03:4xZ 08-12 (encoder OOD probe closed): the pre-reg's success read is pinned — move top-cam 5-NN AUROC 0.885 toward 0.5 (wrist 0.828), k(sim) 1.87e-5 toward the real 1.22e-5; probe rerun is cheap (~0.02 GPU-h, same pinned frames/scripts) so it is the per-iteration read for matching work. Probe also says: prioritize top-cam scene matching (pose/background/table) AND render diversity (sim distances 7x too homogeneous — lighting jitter belongs in the recipe). | CLOSED 05:0xZ 08-12 work session: pre-reg posted+in-channel first, then all named axes landed as render_style='v1' (default): real-frame table texture rebuild (plank direction/scale/contrast matched), real clutter layout, wrist-cam re-pose (menagerie cam had the moving jaw MIRRORED + stared into the gripper body), 72deg-source center-matched equidistant fisheye (real plank bowing reproduced), fixed AWB color grade, sensor blur/noise (labeled amendment), per-reset appearance jitter from a dedicated RNG stream; physics oracle-pinned (tests/test_sim_appearance.py 5 green: qpos bit-identical across appearance seeds/render styles, spawn stream bit-matches banked sim100 v0). REGISTERED BAR MISSED: top 5-NN AUROC 0.890 (v0-render baseline, tripwire vs banked tick-0 0.887 passed) -&gt; best 0.874 / final 0.876 vs bar &lt;=0.790; wrist responded to content (0.835-&gt;0.786 scene-only) then regressed under fisheye+grade (0.900); sensitivity 20x5: appearance draws move per-seed k ~3%, sim stays ~10x too homogeneous. THE FINDING: scene layout, lens geometry and color statistics are NOT the encoder's discriminator - named next lever real-frame inpainting (SIMPLER-RT). 6 probe jsons + 2 before/after composites on fontaine-reports (curl 200), results post + reports.md section. ~0.12/0.5 GPU-h gate. · [pre-reg](posts/2026-08-12-prereg-sim-visual-matching.md)

<details><summary>full record</summary>

Sim visual matching (CPU + render minutes, SIMPLER's second lever after controller sysid): close the sim-vs-rig APPEARANCE gap for the two policy cameras - compare sim renders vs real rig frames (so-frame REAL|SIM|OVERLAY per the LIBERO/SIMPLER convention, sim/probe_visual_match.py is the seed), tune camera pose/FOV, table+background texture, benchy albedo, lighting direction; deliverable = before/after side-by-side page + any so101_sim.py visual deltas. Second-order for eval fidelity per SIMPLER's ablation (gains first, DONE) - the 100-seed pre-reg MAY pin current visuals as v0 and run before this lands; visual matching then becomes a v1-physics/visuals rung with its own re-baseline.

</details>

---

**`sim-encoder-ood-probe`** · `gpu-local`

Encoder OOD probe (OWNER ASK 01:11Z 08-12 'figuring out if that's really the issue', answered 01:30Z with this design): quantify the sim-vs-real visual gap at the policy's own eyes

**boundary:** CLOSED 03:4xZ 08-12 work session, end-to-end in-session (~0.02 GPU-h foreground): launch note (pinned frames + distance def) in-channel pre-GPU; script fontaine/scripts/sim_encoder_ood_probe.py (er_60k eval-mount vision trunk, max_crops=1, fp32 mean-pooled L2-normalized tokens; 300 sim er60k-arm frames ticks {0,300,600} + 300 real v2 strided A/B-split + 100 clean anchor, per camera). MEASURED GAP, top-cam-heavier: centroid AUROC (registered primary) top 0.802 / wrist 0.707; 5-NN secondary (labeled post-hoc; raw cosines ride a dominant constant direction, all distances ~1e-5 residuals) top 0.885 ratio 1.54x / wrist 0.828 ratio 1.33x; clean control INSIDE the real spread (AUROC 0.26/0.28) = shift is sim-specific. Sim at the EDGE of the real manifold, not off it; sim distances ~7x tighter std than real (renders too homogeneous — lighting/blur/hands diversity is part of the gap); per-tick flat = scene not poses. Deviation logged: real stride 114 not 108 (containers 34,332 frames vs meta 32,679). AUROC oracle tests/test_sim_encoder_ood_probe.py (5 green). Artifacts: analysis json + strip chart on fontaine-reports (curl 200 x2), reports.md section, numbers + chart in-channel 03:4xZ. BASELINE FOR THE LEVER: top 5-NN AUROC 0.885 -&gt; ~0.5, k(sim) 1.87e-5 -&gt; 1.22e-5 (real level). · [pre-reg](posts/2026-08-11-prereg-sim-policy-eval-100seeds.md)

<details><summary>full record</summary>

Encoder OOD probe (OWNER ASK 01:11Z 08-12 'figuring out if that's really the issue', answered 01:30Z with this design): quantify the sim-vs-real visual gap at the policy's own eyes. Push N sim frames (from the banked sim100 rollout videos/scenes, both cameras) + N real rig frames (so101_pick_place_v2 episodes) through the frozen er_60k vision trunk; read = sim-vs-real feature distance vs the real-vs-real spread (per camera, top vs wrist separately — camera pose mismatch shows up as a top-cam-specific gap). GPU-light (~0.1 GPU-h, few hundred frames, inference only). Deliverable: analysis json + small chart + numbers in-channel; feeds the sim-visual-matching pre-reg with a measured baseline the matching work must move. PREREG NOTE: rides the sim100 pre-reg (consumes its banked artifacts, answers its follow-up question) like the er15k/35k/55k owner-requested reads rode the er-60k pre-reg; the in-channel launch note is the frozen spec, per charter.

</details>

---

**`sim-servo-sysid`** · `cpu`

Servo/controller sysid (CPU + minutes of local GPU-free sim): resolve the 56x kp discrepancy (our menagerie robotstudio_so101 kp=998.22 kv=2.731 forcerange ±2.94 vs TheRobotStudio upstream kp=17.8 kv=0 ±3.35 for the same STS3215)…

**boundary:** CLOSED 20:3xZ 08-11 work session (post posts/2026-08-11-sim-servo-sysid.md): 56x kp question ANSWERED by open-loop replay sysid (sim/sysid_servo.py, SIMPLER recipe) - held-out-episode arm replay MAE menagerie 3.31 deg / upstream 2.80 / FITTED 1.76 (-47%), beats the 2.19-deg teleport-servo scale so the lag dynamics are genuinely modeled; vendored kp 998 + forcerange 2.94 saturates at 0.17 deg = bang-bang servo, measured sagging ~19 deg below a commanded plateau the real arm holds; upstream directionally right, neither exact. Fitted set PINNED as so101_sim.SERVO_SYSID (kp 108.18 kv 13.377 fr 3.478 damping 0.722 friction 0.0183 armature 0.2045 - the large armature reads as reflected gear-train inertia), applied at load to both arms, vendored XML untouched; sysid_servo.json banked. ALL sim-fixes gates re-verified under new params: 0/100 strikes, settled state bit-identical across seeds, drift 0.001mm/10s, pinch-lift HELD spin 0.1 deg (improved from 0.4), determinism green, 28.0 ms/tick. Known residual: elbow_flex 3.89 deg (unmodeled boat payload); per-joint gains the named next rung if elbow ever gates. Fit deps-free coordinate descent, ~240 evals/start.

<details><summary>full record</summary>

Servo/controller sysid (CPU + minutes of local GPU-free sim): resolve the 56x kp discrepancy (our menagerie robotstudio_so101 kp=998.22 kv=2.731 forcerange ±2.94 vs TheRobotStudio upstream kp=17.8 kv=0 ±3.35 for the same STS3215) by SIMPLER's recipe — open-loop replay of real rig episodes (we hold 229h; use held-out rig episodes' recorded qpos streams) through the sim, fit kp/kv/damping (BAM's identified STS3215 model github.com/Rhoban/bam as informed prior; their friction params converge via CMA-ES in ~5min) minimizing joint-trajectory MAE. SIMPLER ablation says this is the FIRST-order eval-fidelity lever (control loss 0.131-&gt;0.432 moved MMRV 0.031-&gt;0.100). Deliverable: fitted params + before/after replay MAE + a one-page note; feeds the 100-seed pre-reg's physics pin.

</details>

---

**`sim-fixes-reset-contact`** · `cpu`

Sim fixes, batch 1 (CPU, from sim-review findings 1-4 + the contact-fidelity fix list papers/sim-contact-fidelity.md): (1) home pose reachable (fix camera_box2 mount collision vs shoulder) + spawn-after-settle so reset never stri…

**boundary:** CLOSED 19:1xZ 08-11 work session (all 3 legs + gates green, results post posts/2026-08-11-sim-fixes-batch1.md): (1) START STATE - three layers, not one: camera_box2&lt;-&gt;shoulder exclude (0.46mm wedge) + wrist&lt;-&gt;shoulder exclude (0.87mm, same class) + shoulder_lift/elbow_flex ranges widened at load (menagerie +-100/+-96.8 could not REPRESENT the rig median start -102.7/97.0); wrist_roll bimodality gone, settled state seed-independent &lt;0.003 deg across seeds; elbow 6.6 deg residual = jaw tip physically resting on table (reachable projection per the review; NOT excluded - real physics); reset() reworked spawn-after-settle w/ public reset_strike_contacts counter, probe reads API not a hand-replica; second strike channel found+fixed (jaw tips at x=0.155 sat INSIDE the old spawn region - near bound 0.17-&gt;0.195, design target preserved: mean initial distance 9.5cm range 7.1-12.1 over seeds 0-99); 0/100 strikes, max displacement 0.7mm. (2) JAW SEAM - priority=2 on generated benchy geoms (NOT the queued &lt;contact&gt;&lt;pair&gt;: menagerie's actual jaw contact meshes are UNNAMED so pairs would miss them; priority is the same documented wholesale-friction override); elliptic+impratio10+Newton verified already present; in-grip spin 6.9-&gt;0.4 deg, tilt 0.84-&gt;0.91, pen 2.2mm, firm hold. (3) COACD threshold-driven (0.015, uncapped, pr 100): 340 hulls, volume 1.75x-&gt;1.13x, phantom p99 3.78-&gt;0.45mm max 5.39-&gt;0.69mm; SDF not needed (26.7ms/tick unchanged). REGRESSION FOUND+FIXED: rest drift came back 6.2mm/10s with the fine decomposition - NOT friction/damping (damping made it WORSE 76mm); root cause vendored solver caps iterations=10/ls_iterations=20 under-converge 30-80 simultaneous keel-table contacts; scene now sets 50/50, drift 0.001mm at zero cost. Gates: all probes improved, bit-determinism green, 26.7ms/tick (&lt;30 gate). check.py 688 green. Servo sysid deliberately NOT touched (own item, next).

<details><summary>full record</summary>

Sim fixes, batch 1 (CPU, from sim-review findings 1-4 + the contact-fidelity fix list papers/sim-contact-fidelity.md): (1) home pose reachable (fix camera_box2 mount collision vs shoulder) + spawn-after-settle so reset never strikes the boat — re-verify 0 reset strikes over candidate seed list; (2) explicit &lt;contact&gt;&lt;pair&gt; for jaw-boat seam (condim&gt;=4, elliptic cones, impratio~10, Newton) — re-run pinch probe, compare 6.9deg spin / 0.84 tilt; (3) re-run CoACD threshold-driven (-t 0.01-0.02, uncapped hulls, higher -pr) OR native-SDF experiment (MuJoCo&gt;=3.3.5; also closes the CC-BY-ND per-machine derived-asset hazard) — re-run phantom-volume probe vs p99 3.78mm baseline; keep friction VALUES untuned (SIMPLER Table X citation). Gates: all three probes improved + bit-determinism re-verified + tick cost still ~&lt;30ms.

</details>

---

**`er60k-init-delta-midrun-chart-0810`** · `cpu`

er_60k ER-init-delta mid-run chart (CPU, zero GPU-h): once the probe ladder has ~10 points (step &gt;= 5000, ~02:0xZ 08-10), chart eval_chunk_mae vs the 40k run's curve at matched steps (seed 0 shared per the owner's 22:51Z seed pol…

**boundary:** opens at er_60k step 5000 (~02:0xZ 08-10); any CPU window after that; superseded by the endpoint readout ~08-11 ~12:00Z if unexecuted · [pre-reg](posts/2026-08-09-prereg-molmo2-er-60k.md)

<details><summary>full record</summary>

er_60k ER-init-delta mid-run chart (CPU, zero GPU-h): once the probe ladder has ~10 points (step &gt;= 5000, ~02:0xZ 08-10), chart eval_chunk_mae vs the 40k run's curve at matched steps (seed 0 shared per the owner's 22:51Z seed policy — shuffle-order variance removed from the comparison, so the curve delta IS the init effect + rig-data 0.19%). Same dark eval-report style as adamc_postmortem_chart.py (reuse the 40k curve transcription). Record-only per pre-reg (never a kill line); post the chart in-channel with the step-5000 async-save fact. If the delta is boring (overlapping curves), one line in now.md suffices and the full chart waits for the endpoint readout — do not manufacture a post.

</details>

---

**`adamc-postmortem-chart-0809`** · `cpu`

AdamC-100k post-mortem chart + short post (CPU, zero GPU-h; the queued item now.md promised 22:4xZ but never landed in queue.json

**boundary:** DONE 23:3xZ 08-09 same work session that queued it: chart script fontaine/scripts/adamc_postmortem_chart.py (2-panel matched-steps + matched-samples, eval-report dark theme) + post posts/2026-08-09-adamc-postmortem.md (three matched views: 10.80 vs 7.17 @10k steps; run-best 10.30 vs ~8.6 samples-matched; 35.7 GPU-h vs 31.6-for-7.09 compute-matched; loss near-parity 3.74 vs 3.44 = gap lives in the held-out probe; explicit 3-confound caveat, no AdamC verdict; lr_backbone artifact annotated w/ f112f08). SUMMARY wired, Space pushed, page+svg curl-200, in-channel link posted. VERIFICATION WIN en route: the log's lr_backbone=1e-4 trace investigated before writing — confirmed the known owner-caught logging artifact (training groups always 2e-5), so the post carries the annotation instead of a false misconfiguration claim.

<details><summary>full record</summary>

AdamC-100k post-mortem chart + short post (CPU, zero GPU-h; the queued item now.md promised 22:4xZ but never landed in queue.json — added at the 23:0xZ 08-09 queue audit): from the banked train_log.jsonl (box+local), chart-led per owner preference — probe eval_chunk_mae ladder to the 10.30@11500 run-best kill point, train loss/grad-norm, vs the 40k-run probe curve at matched steps as CONTEXT ONLY (different recipe: AdamC vs AdamW, vision unfrozen vs frozen, eff-32 vs eff-48 — descriptive post-mortem, no causal claim without a matched arm). Dark-mode per standing rule. What the run bought: AdamC implementation 401d6f7 (10 oracles, stays landed), step-10k weights on fontaine-checkpoints, the 3-rise-then-recede probe-watch precedent.

</details>

---

**`er-60k-live`** · `gpu-box`

OWNER RUN LIVE (launched 22:47-53Z 08-09, unit fontaine-er-60k): fontaine_molmo2_er_60k_ddp4

**boundary:** CLOSED (status audit 16:2xZ 08-11: run actually finished 08-11 — train @60000 12:36Z, chained panel_v2 rc=0 13:28Z, ~153/155 GPU-h; decision read = ER init WINS both legs, er_60k/step_060000 = new reference trunk, weights banked to fontaine-checkpoints 4ed3dd0; the 10:00-13:5xZ session closed the run but left this item status=live — fixed this session). Durable long-form: posts/2026-08-11-er-init-screen-results.md. · [pre-reg](posts/2026-08-09-prereg-molmo2-er-60k.md)

<details><summary>full record</summary>

OWNER RUN LIVE (launched 22:47-53Z 08-09, unit fontaine-er-60k): fontaine_molmo2_er_60k_ddp4 — 60k AR steps from allenai/Molmo2-ER (byte-verified drop-in), 40k recipe verbatim + rig datasets at natural share (owner pick 22:45Z), seed 0 (owner override), save 5000. FIRST POLL DONE 22:5x-23:0xZ: E1 banner exact, 2.23 s/step steady, vram 66.6 vs 77, util 68-99%. RATE-CLASS CORRECTION posted in-channel + gate re-pinned 65-&gt;155 GPU-h (the 65 came from attach_F's 0.92 s/step frozen-trunk rate, wrong class; true trunk class 2.2-2.6 = 60k-continuation actuals). Corrected endpoint ~08-11 ~12:00Z (~37 h wall, ~149 train + ~2 eval GPU-h). Primary read = ER-init delta: probe ladder vs the 40k curve at matched steps; endpoint chains panel_v2 k4l2 (--report + npz), paired CI95 vs banked 40k (6.0079) + 60k-continuation (5.8602).

</details>

---

**`owner-er60k-run-prep-0809`** · `cpu`

OWNER STEERING 22:14:00Z 08-09: proposed 60k training run init from allenai/Molmo2-ER (MolmoAct2's embodied-specialized Molmo2-4B, released) with params matched to our molmo2 AR 40k recipe, 60k steps, owner rig datasets mixed in…

**boundary:** opens on owner reply (go + rig dataset pointers); param sheet ~30 min after; launch only on owner approval of the sheet | CLOSED 22:5xZ 08-09: all owner inputs landed (go 22:36Z, rig ids 22:40Z, sheet approved verbatim 22:45Z uniform sampling, seed override 22:46Z) -&gt; launched 22:47-53Z unit fontaine-er-60k seed 0. Live-run tracking moved to er-60k-live + babysit er_60k entry.

<details><summary>full record</summary>

OWNER STEERING 22:14:00Z 08-09: proposed 60k training run init from allenai/Molmo2-ER (MolmoAct2's embodied-specialized Molmo2-4B, released) with params matched to our molmo2 AR 40k recipe, 60k steps, owner rig datasets mixed in from step 0; owner would KILL adamc_100k ('not looking great' — matches our named probe-rise watch, 3 consecutive rises 11.41@11000 vs 10.63@9500 run-best) and reuse the 4x box GPUs. FEASIBILITY VERIFIED + ANSWERED IN-CHANNEL 22:19Z: ER is a drop-in init — config diff vs base = max_position_embeddings 36864-&gt;16384 (RoPE metadata) + transformers_version only; safetensors manifests identical key set + identical total 19,403,476,800 bytes; launcher change = --backbone allenai/Molmo2-ER. ER snapshot download COMPLETE on box (verified 22:35Z 08-09: 0 incomplete blobs, all 4 shards + processor/code files present, 19,403,574,432 bytes on disk) — launch not blocked on weights. AWAITING OWNER: (1) explicit go to kill adamc_100k (keep step-10000 ckpt + bank logs for zero-GPU AdamC post-mortem chart unless owner says drop); (2) rig dataset pointers (HF ids or box paths) + mixture call (CL-triangle evidence: prior-data replay 2-20% share; rig-from-step-0 is the evidence-backed shape). THEN: pre-reg param sheet re-pinned verbatim from the 40k pre-reg/launcher (fresh shuffle seed per standing rule) posted for approval BEFORE launch per the standing gate.

</details>

---

**`lit-radar-0822`** · `cpu`

Lit slice (standing allocation): 4 priority hooks from the 0821 refill sweep (18 candidates checked, 15 abs-page-verified, 12 grep-clean -- 3 dups all already-deep-read papers, sweep converging), priority-ordered: 2606.12365 Ambi…

**boundary:** CLOSED 2026-08-10 00:5xZ work session AS THE FINAL SLICE BEFORE THE OWNER PAUSE (owner 00:23:47Z "Can we pause the lit slices for now" landed mid-flight at the 00:27Z babysit poll; acknowledged in-channel 00:28Z, pages landed quietly, NO summary post, NO 0823 queued): 4 Papers pages via 4-agent fan-out — ambient-diffusion-policy.md (flow-time band-mask lever, rectified-flow port via sigma-space mapping, +33% hook corrected to tower height, partition user-supplied -&gt; composes with QoQ; zero-GPU PSD/sigma_tmin first arm sketched), what-curation-metrics-do.md (0.804 AUROC -&gt; 13.3% policy dissociation CONFIRMED but one-cell; episode-length confound + truncation control -&gt; rank-by-length null arm fed to #9), auditing-curation-metrics.md (action-only scorers chance on wrong-action defects; entropy/ensemble actively inverted; state-rescue = object pose NOT proprio, ablation never run — their released testbed settles it), phail.md (KM/RMST/macro-KS+clustered-bootstrap resolves 2/3 close pairs at 25-30 episodes/cell; human anchor ZERO statistical power; 22pp spatial-nuisance warning; &gt;=50 trials stays the budget, KS-on-CDFs adopted as analysis). Ideas #9/#16/#15 wired. SPARES (8 abs-verified) + EXTRA HOOKS + the 0823 sweep-planning note remain in this item title for whenever the pause lifts.

<details><summary>full record</summary>

Lit slice (standing allocation): 4 priority hooks from the 0821 refill sweep (18 candidates checked, 15 abs-page-verified, 12 grep-clean -- 3 dups all already-deep-read papers, sweep converging), priority-ordered: 2606.12365 Ambient Diffusion Policy (MIT/Tedrake, RSS demos spotlight: suboptimal demos contribute only at high/low diffusion times via a spectral power law in robot actions, +33% over naive co-training, purely offline -- the #9 re-weighting lever on the flow TIME axis; check the spectral argument transfers to rectified flow + how the suboptimal split is designated; composes with the QoQ influence pass) &gt; 2606.10229 What Demonstration Curation Metrics Do to Your Policy (best defect detector AUROC 0.804 -&gt; WORST policy 13.3%, weak 0.638 detector ~matches oracle 90.0 vs 93.3; 5/7 metrics secretly exploit episode length -- confound warning for every #9 arm AND for chunk-MAE panels; testbed released; read WITH its companion) &gt; 2606.05588 Auditing Demonstration Curation Metrics (action-only scorers blind to structural defects, two actively prefer defective episodes; our positions-only corpus IS the failing feature space -- stress test for label-free-selection-signals conclusions; check whether their state metrics need visual state) &gt; 2605.29710 PhAIL (Franka open bench: time-to-success CDFs + Human-Relative Throughput + bootstrap CIs + per-object KS, claims resolution at N&lt;=30 rollouts/cell -- THE #16 rig-day statistical protocol question; check the claim isn't carried by the human anchor; artifacts released, phail.ai). SPARES (8, abs-verified grep-clean 08-10): 2607.04434 RoboDojo sim+real cloud-eval leaderboard; 2605.20774 VLA-REPLICA low-cost reproducible bench (closest #16 analogue); 2607.15330 Xiaomi-Robotics-1 100K-h scaling report; 2606.15064 Phase-Localized Curation Does Not Help (negative, same testbed); 2606.20521 HumanScale (ego-video-beats-robot-data claim -- candidate h2r-lowdata-counterexample trigger, check for hidden diverse robot corpus in the alignment stage); 2603.05504 RoboPocket foresight-guided collection; 2606.30988 MuSe post-hoc force attachment; 2607.26047 S2A2 contact audio. EXTRA HOOKS (unverified ids, triage first): 2605.24934 HumanEgo, 2606.14665 EgoGuide, 2602.22088 Force Policy; 2511.19861 GigaWorld-0 (data-engine predecessor, #9 synthetic-data); 2511.11520 scalable-policy-eval-with-video-WMs (abs-verified, no code). Sweep planning note for 0823: angle E curation richest -- mine the 2606.10229/2606.05588/2606.15064 testbed cluster citation trails + recheck RSS 'It's the demos' workshop for 4 accepted titles with no arXiv ids yet (Maintaining Demonstration Quality in a 100-Robot Teleoperation Pipeline; From Action Labels to Sets; Beyond Clean Demonstrations; Better Demonstrations, Not More) in 2-3 weeks; angle D eval citation-thin (ArmnetBench 0 citations, Eval-Actions 1) -- retry late August, try 'distributional evaluation robot policy' / 'cloud evaluation manipulation leaderboard'; angle C try 'compute-optimal imitation learning' / 'action expert capacity ablation' (leads: LAP 2602.10556, A1 2604.05672); angle B one more query ('joint current contact detection low-cost arm') then REST the angle. Papers page(s) same session per the permanent rule.

</details>

---

**`lit-radar-0821`** · `cpu`

Lit slice (standing allocation): 4 priority hooks from the 0820 refill sweep (16 candidates abs-page-verified by the sweep agent, 12/16 grep-clean

**boundary:** CLOSED 2026-08-10 00:1x-00:3xZ work session via 5-agent fan-out: 4 Papers pages landed SAME session per the permanent rule -- quality-over-quantity.md (offline influence pole, runnable no-rollout vs ATHENA/Qwen; gains only on 40-50% injected failures, hard top-N not weighting, no code; cheapest #9 arm sketched: ~20 verified-clean anchor episodes + action-head-only gradient scoring pass + paired top-70%-vs-random arm), curse-of-precision.md (log N prop 1/(P-c) is a sim-only R2&gt;0.97 FIT with 23-65x extrapolated points; 'sensor+expert not task' hook corrected -- randomization ablation moved c 2.35-&gt;1.00mm; c = rollout-sweep fit = rig-phase instrument; #16 tolerance-dial + delta-c design rule; #9 clarity-filter lever: aggressive 50%-SR expert c=1.27 vs cautious 2.35), neuralactuator.md (cost floor broken: third platform IS the SO-101, force MAE 0.47-0.73N from Feetech load registers, no current sensor, torque via diffsim; 'torque-from-current' hook wrong twice at our class; MIT code + NAD dataset + 3 SO-101 checkpoints + teleop code ALL verified live; #16 rig-day rider superseded shovel-ready 46-column schema; #9 dq_d gate stands as banked), gigaworld-wmbench.md (324K 'rollouts' are human-graded WM VIDEOS under replayed actions, no policy drives, real-ranking corr defined never computed; action-faithfulness&gt;realism measured but partly definitional; hook missed the Apache-2.0 release Nano 1.3B/Pro 5B + 87.8%-agreement VLM judge; Ctrl-World artifact verified live too -&gt; rollout-free-eval 'no artifact' half dead, 'costs real rollouts' half stands; zero-rollout replay screen banked for #16). Ideas #9/#16 pages + index hooks fed. Radar 0821 table flipped; 0822 queued from the sweep (18 checked, 15 abs-verified, 12 survived, 3 dups all already-read).

<details><summary>full record</summary>

Lit slice (standing allocation): 4 priority hooks from the 0820 refill sweep (16 candidates abs-page-verified by the sweep agent, 12/16 grep-clean — the 4 dups were papers already deep-read; executor still greps full corpus per id before writing), priority-ordered: 2603.09056 Quality over Quantity (influence functions w/ max-over-validation scoring + trajectory-level aggregation rank demos, consistent sim+real gains — the principled per-episode weighting computable against our held-out panel; the #9 curation lever, compare vs the banked Qwen-RobotManip offline filter + ATHENA rollout-anchored pole) &gt; 2607.23108 Curse of Precision (demos grow super-exponentially with target precision, log N ∝ 1/(P−c); ceiling is a property of the sensor+expert system not the task — bounds demo-scaling on hobby-arm precision tasks, feeds #9/#16 bench design) &gt; 2607.11734 NeuralActuator (neural actuation model: torque dynamics + external-force detection on ~$500-to-$30K platforms, teleop dataset, improves BC — torque-from-current at exactly our cost class; the FACTR 2 successor niche, adjudicate vs the factr2 page's currentless variant) &gt; 2607.02642 GigaWorld-1/WMBench (7 video world models x 4 action reps, 324K+ simulated rollouts: long-horizon action-faithful consistency &gt; visual realism for eval alignment — frames the sim-grading question the rollout-free-eval page opened; read with its Ctrl-World 2510.10125 artifact hook). SPARES (8, grep-clean 08-09): 2606.27375 ABC-130K open BC scaling substrate (3,500 h/130K eps/195 tasks + recipe sweeps); 2601.18723 Eval-Actions graded execution-quality labels (13K eps, SRCC 0.81-0.84); 2603.13616 Beyond Binary Success anytime-valid sequential comparison (-70% eval burden); 2511.09958 Audio-VLA contact-mic template; 2512.08405 audio world models (flow-matching audio prediction); 2606.17598 MuseVLA frozen-trunk multimodal sensing; 2607.21588 AXIS community data engine (+5.8% from auto-QA); 2605.26349 episode-level teleop quality scoring. EXTRA HOOKS from the 0820 page agents (unverified ids, triage first): 2510.10125 Ctrl-World (the only RELEASED artifact in the world-model-eval class, both cluster papers benchmark it); 2511.11520 scalable policy eval w/ video world models (PolaRiS ref [28]); h2r-lowdata-counterexample standing screen (any human-video/latent-action gain at &lt;=~250 h single-embodiment on a frozen trunk = reopening condition for angle A). Papers page(s) same session per the permanent rule. Sweep planning note: angle D (eval methodology) rich 3 sweeps running — next mine ArmnetBench/Eval-Actions citation trails; angle C moved to precision limits + representation bottlenecks (try 'VLA model size scaling ablation', 'pretraining data mixture robot policy'); angle B thin on motor-current (try 'servo current feedback learning', 'acoustic sensing gripper'); angle E curation RICH + RSS 2026 'It's the demos' workshop accepted-list to mine; unfetched adjacents: 2605.19138 Cobalt, 2602.22818 LeRobot library paper, 2608.02580 Ego2Robot.

</details>

---

**`lit-radar-0820`** · `cpu`

Lit slice (standing allocation): 4 priority hooks from the 0819 fresh sweep (16 candidates abs-page-verified, 14/16 grep-clean; executor still greps full corpus per id before writing), priority-ordered: rollout-free eval CLUSTER…

**boundary:** CLOSED 2026-08-10 00:0xZ work session (23:27-): 4 Papers pages via 5-agent fan-out, all landed + wired SAME session per the permanent rule: rollout-free-eval.md (RoboWorld 2607.01060 + PolaRiS 2512.16881 cluster — both hooks survived their numbers but grew teeth: RoboWorld r=0.989 is n=8 with NO artifact and an unvalidated GPT-4o judge; PolaRiS r=0.9/24-points is the stronger certificate, MIT code live, but per-checkpoint co-training is load-bearing and calibration is DROID-only; every rollout-free certificate was bought with real rollouts; PolaRiS independently replicates our offline-validation read; rig-day scan rider fed to #16); factr2-torque-estimation.md (3 hook corrections: 'no force sensor' hides a load-bearing 100 Hz current sensor, +17% bundles torque-as-observation with re-sampling and sampling-only is never ablated, code unreleased; transfer find: the load-bearing input dq_d = action - state is free in our corpus -&gt; zero-GPU contact-segmentation gate fed to #9, rig-day 10-min protocol to #16); is-diversity-all-you-need.md ('expert diversity hurts' was NEVER operator-ablated — evidence is the velocity-debias gain +15% ~ 2.5x data on a DIFFUSION action expert, so flow-head immunity is exactly what their setup contradicts; recipe unreleased; velocity spread flagged as a chunk-MAE eval confound; speed-census -&gt; panel-correlation -&gt; normalization-arm chain fed to #9; rig-relevance-filtering warning; Bridge V2 pilot demoted); human-to-robot-transfer-emergence.md (pi0.5+ego: human video ~doubles generalization ONLY atop diverse robot pretraining, base-VLM init pays ~zero = our measured no-transfer corner; 'threshold' partly our compression, no absolute units; angle-A spares CLAP/Motus/LingBot GATED OFF with a written reopening condition; er_60k rationale strengthened, fed #17). Ideas #9/#16/#17 pages + index hooks fed; papers index rows + Radar 0820 flips landed. Refill sweep ran in the same fan-out: 16 candidates abs-verified, 12 survived the corpus grep (the 4 casualties were all papers we had ALREADY deep-read: MolmoAct2, ArmnetBench, CI-MSE, Compression Gap — the sweep is converging on our reading list) -&gt; lit-radar-0821 queued (4 priority hooks + 8 spares).

<details><summary>full record</summary>

Lit slice (standing allocation): 4 priority hooks from the 0819 fresh sweep (16 candidates abs-page-verified, 14/16 grep-clean; executor still greps full corpus per id before writing), priority-ordered: rollout-free eval CLUSTER 2607.01060 RoboWorld (AR video world model + VLM scorer, Pearson 0.989 vs RoboArena but n=8 policies — calibration check IS the read) + 2512.16881 PolaRiS (real-scene scans -&gt; neural interactive sim, scan-our-own-workspace template; adjudicate vs Squint as eval substrate) &gt; 2606.12406 FACTR 2 (external joint-torque estimation with NO force sensor from ~10 min motion data on commodity arms + force-informed BC re-sampling +17% — cheapest force recovery for cameras+joints-only SO-101; re-sampling idea may transfer to #9 curation) &gt; 2507.06219 Is Diversity All You Need (task diversity &gt; per-task count; EXPERT diversity hurts via velocity multimodality, debias +15% ~ 2.5x data — directly checkable on our multi-operator corpus, feeds #9) &gt; 2512.22414 Emergence of human-to-robot transfer (co-training pays only above a pretraining diversity threshold — gates all angle-A video recipes at our 229h scale). SPARES (10, grep-clean 08-09): 2607.08639 LingBot-VA 2.0 (native video-action pretraining, async inference, no release); 2512.13030 Motus latent-action WM; 2601.04061 CLAP contrastive latent actions from human video; 2602.12063 VLAW co-improvement (needs rollouts); 2604.28156 FlexiTac open tactile pads; 2607.03723 OmniTacTune tactile residual RL; 2602.13640 audio-visual-proprio fusion; 2607.27549 behavior-aligned cross-embodiment reps; 2606.24038 sim-real e-process anytime-valid CIs; 2606.04233 what-are-we-benchmarking. Papers page(s) same session per the permanent rule. Sweep planning note: angle D (eval methodology) still rich; angle C thin on true scaling laws (try 'power law demonstrations', 'compute-optimal robot policy'); angle B thin on audio/current-sensing.

</details>

---

**`owner-molmoact2-deep-dive-0809`** · `cpu`

OWNER STEERING 20:49:36Z 08-09: 'Woah, there's already a molmo2 VLA -- https://github.com/allenai/molmoact2

**boundary:** CLOSED 2026-08-09T21:4xZ same session: deep-dive post posts/2026-08-09-molmoact2-deep-dive.md shipped via 4-track fan-out (paper 51pp PDF + repo/configs + HF cards + AI2 blog/LeRobot docs/community sweep). Headlines for us: backbone IS Molmo2 (-&gt; Molmo2-ER, +6.0 LIBERO-Long from ER-ization alone, released = cheapest trunk arm ever priced, fed #17); 621M per-layer-KV flow expert (capacity anchor for tonight's Delta_capacity read; KV-vs-hidden +1.9); finetune ablation expert-only 93.05 vs full-FT 97.20 = strongest joint-pole vote, insulation-at-finetune a wash (fed #4, predicts fjoint &gt; F2); SO100_101 checkpoint zero-shot path official in LeRobot v0.6 (12.1 GiB bf16, joint-remap gotcha), expert-only FT 16.5 GiB = single-GPU; repo_list.json manifest mechanizes the survey's corpus-delta (fed #9). Independent-signal dryness flagged. In-channel link posted. Follow-up arms (ER-swap, corpus intersection, rig zero-shot) owner-decision, NOT queued.

<details><summary>full record</summary>

OWNER STEERING 20:49:36Z 08-09: 'Woah, there's already a molmo2 VLA -- https://github.com/allenai/molmoact2. Write a super in-depth piece on it, everything you can find on training, arch, experiments etc. well organized.' Deliverable: long-form blog piece (paper 2605.02881 + repo + HF model/dataset cards + AI2 announcement + v1-&gt;v2 delta + competitive map + what-transfers-to-us), link in-channel. Note: the 0816 refill sweep had independently ranked this paper #1 ~20 min before the owner message; the 0817 queue item's priority-1 slot is satisfied by this piece.

</details>

---

**`lit-radar-0819`** · `cpu`

Lit slice (standing allocation): 4 priority hooks from the 0818 fresh sweep (16 candidates abs-page-verified; only 2 corpus dups by local grep - the new-angles mandate fixed the pool; executor still greps full corpus per id befor…

**boundary:** CLOSED 2026-08-09 ~22:3xZ work session: 4 Papers pages via 5-agent fan-out same session (squint.md, action-space-design.md, so101-vla-benchmark.md, cl-triangle.md). Headlines: Squint = the rollout-substrate blocker mechanically GONE (MIT SO-101 twin in ManiSkill3, verified installable, 96.1-&gt;91.3% ranking-preserving sim2real; caveat far-OOD visual world -&gt; relative screens first; correction: vendored not upstreamed); Action-space = first hook to STRENGTHEN on contact (code+data verified; chunk-wise delta-joint beats our absolute cell +8.4pp in our exact policy class -&gt; NEW idea #23 + offline&lt;-&gt;rollout inversion warning); SO-101 bench = 320 rollouts n=20/cell, leaky multi-label taxonomy, prize = 16 unlisted rollout_* datasets (unlabeled, needs 2-3h self-label pass); CL triangle = contradiction dissolves in tables, zero-replay FT always forgets, replay rho 0.02-0.2 @ 20% batches suffices (real-robot 3B full-FT proof) -&gt; #17 unfreeze price list + #4 drift instrument + #16 rig-phase pre-reg clause. Ideas #4/#5/#6/#16/#17/#22 fed + idea #23 opened. Refill sweep: 4 new angles, 16 abs-verified, only 2/16 dups (both already deep-read) -&gt; lit-radar-0820 queued w/ 4 priority hooks + 10 spares. INTERRUPTED-BY-STEERING note: owner 22:14Z ER-60k question answered mid-session (see owner-er60k-run-prep-0809).

<details><summary>full record</summary>

Lit slice (standing allocation): 4 priority hooks from the 0818 fresh sweep (16 candidates abs-page-verified; only 2 corpus dups by local grep - the new-angles mandate fixed the pool; executor still greps full corpus per id before writing), priority-ordered: 2602.21203 Squint (SO-101 integrated into ManiSkill3 + released 'SO-101 Task Set' 8 tasks w/ domain randomization, zero-shot sim2real on real SO-101 after &lt;15 min on one 3090; #16 - the first credible sim rollout-eval substrate for our exact arm class, could unblock every rollout-gated item: ATHENA-style curation, #6 calibration, #22 staleness screen) &gt; 2602.23408 Demystifying Action Space Design (13,000+ real rollouts, 500+ models: delta actions consistently win, joint/task-space complementary, absolute needs longer horizons; the evidence base for chunk-length/relative-vs-absolute/EE-vs-joint choices we currently make by folklore) &gt; 2606.08881 Benchmarking VLAs on SO-101 (real-world failure taxonomy + recovery analysis on our exact hardware, execution instability dominant; #6 second calibration corpus next to ArmnetBench + #16) &gt; continual-learning triangle 2603.03818 (pretrained VLAs resistant to forgetting, replay suffices) + 2605.26820 (real-robot CL benchmark: naive sequential FT forgets badly) + 2603.11653 (simple recipe + LoRA + on-policy RL beats elaborate CL machinery) as ONE theme-cluster read - the three partially contradict and adjudication is the read; feeds #17 unfreeze recipes + #4. SPARES (8, all grep-clean 08-09): 2602.10556 LAP language-as-action zero-shot cross-embodiment; 2607.06442 SIEVE structure-aware data selection (#9); 2603.06450 Data Analogies paired-demo cross-embodiment (+22.5%); 2602.12628 sim-real RL co-training; 2606.29570 spectral/DCT hierarchical action decomposition; 2603.16861 MolmoB0T 1.8M sim trajectories zero-shot (Molmo-family sibling); 2511.17001 CalibAll camera-frame action unification; 2608.06374 DyPES-VLA shared dynamics priors + embodiment-specific MoE heads. Papers page(s) same session per the permanent rule.

</details>

---

**`lit-radar-0818`** · `cpu`

Lit slice (standing allocation): 4 verified-clean hooks from the 0817 session's refill sweep (16 candidates abs-page-verified by the sweep agent; 12 dropped as corpus dups by local grep vs papers/+ideas/+ideas.md - the sweep pool…

**boundary:** any GPU-busy window; adamc rides to ~08-12 so windows are plentiful | CLOSED 2026-08-09 ~22:0xZ work session: 4 hooks -&gt; 4 Papers pages same session (athena.md, probeact.md, qwen-robotmanip.md, plasticity-at-scale.md) via 5-agent fan-out (4 deep reads + fresh sweep concurrent). Hook corrections, all four again: ATHENA is rollout-anchored (R in {1,-1} over eval rollouts - NOT offline curation), corpora tiny (9.34h sim/6.90h real), code link dead, '45.0-point improvement' = +0.90pp x 50 tasks; real signal = heuristic length-Oracle BELOW random on real tasks + cross-model transfer licenses proxy scoring -&gt; #9 parked 'offline-ATHENA' design note. ProbeAct hook wrong on BOTH clauses: probe = 3D position regressor trained on 50k sim-oracle labels (not failure probe), detection = hand-coded kinematic state machine, ZERO detection metrics in the paper, AR-only sim-only no code; survives: trunk decodes position R2=0.968 while flow cells probe below coin-flip elsewhere -&gt; #6 gate gains a trunk-tap arm (spatial pooling + shallow-mid sweep). Qwen-RobotManip 38,100h is ~65% re-rendered human video (24,808h synth from 1,933h egocentric; ~7,800h real teleop = ~34x us not 166x), nothing released ('no plan to release' verbatim); survives: 5-stage offline state-action filter (81% of RoboMIND-UR excluded as broken proprioception) -&gt; #9 cheapest arm = DA+jerk pass over 229h; #17 fourth attachment pole (cross-attn alternating vis/lang, 1:40 ratio) + benchmark-saturation seconds VLM4VLA. Plasticity-at-scale: WD clause of the hook was a CITATION of 2602.11137 (already read - hook laundered our own corpus back at us; groups distinct, scale claim independent), WD fixed 0.1 throughout, largest measured 314M, no grad-norm analysis -&gt; record-only for adamc; durable export = health proxies (dormant/norms/entropy) all FAIL to track onset, behavioral fixed-budget probes only. Ideas fed: #6 #9 #17 + index hooks. Refill: fresh sweep with the 4 mandated new angles -&gt; 16 candidates abs-verified, only 2 corpus dups (14 clean - new angles fixed the pool-drying problem) -&gt; lit-radar-0819 queued with 4 priority hooks + 8 spares.

<details><summary>full record</summary>

Lit slice (standing allocation): 4 verified-clean hooks from the 0817 session's refill sweep (16 candidates abs-page-verified by the sweep agent; 12 dropped as corpus dups by local grep vs papers/+ideas/+ideas.md - the sweep pool is drying, next sweep MUST diversify search angles AND the executor must grep the full corpus per id, never trust the agent's exclusion-list check), priority-ordered: 2606.16208 ATHENA (accelerated multi-task influence functions for robot data curation at billion-param VLA scale; #9's principled alternative to heuristic quality gating for the 229h corpus + the MolmoAct2 repo_list diff) &gt; 2606.09740 ProbeAct (probe-guided training-free failure detect-and-correct via hidden-state probes + control barrier functions; #6 - pairs directly with the ArmnetBench go/no-go separability gate banked 0817) &gt; 2606.17846 Qwen-RobotManip tech report (38,100h heterogeneous manipulation + egocentric human data, multi-stage curation/alignment pipeline at 166x our scale; #9 reference pipeline + #17 trunk-recipe signal) &gt; 2606.24752 Can Scale Save Us From Plasticity Loss (scale delays but does not prevent transformer plasticity loss; higher WD can improve plasticity despite worse pretrain loss; adamc watch-frame companion to WD-plasticity 2602.11137 - verify not same group re-cut, check whether its WD claims license anything at our 1e-5/1e-4 operating point). NO SPARES banked (dup rate left none) - session should run its own fresh sweep with new angles (suggested: cross-embodiment transfer, sim2real gap for SO-class arms, VLM-trunk continual learning, action-space/tokenizer design) before or instead of dipping below these 4. Papers page(s) same session per the permanent rule; dedup-check each id against the corpus before writing.

</details>

---

**`lit-radar-0817`** · `cpu`

Lit slice (standing allocation): refill hooks banked in the 0816 session's fresh sweep, all ids abs-page-verified by the sweep agent + dup-checked against papers/ + ideas/ + queue (2 dup catches dropped: 2607.23777 = Muon-SW alre…

**boundary:** any GPU-busy window; adamc rides to ~08-12 so windows are plentiful | CLOSED 2026-08-09 ~21:3xZ work session: 5 hooks -&gt; 4 Papers pages same session (armnetbench.md, safecast.md, reflex.md, legato.md, compression-gap.md; MolmoAct2 slot satisfied by the 08-09 owner deep-dive post). Hook corrections, three loud: ArmnetBench '3,118 human-labeled' = 2,518 human-scored rollouts + 600 unscored demos, and the claimed 84 task-policy checkpoints are NOT on the Hub (blocks the #9 calibration study -&gt; watch item); SAFECAST is NOT offline (contrast sets need closed-loop re-executions + hundreds of labeled rollouts incl. real failures) and its flow-policy cells land below coin-flip in its own alpha-marginalized metric (0.45 sim/0.38 real vs OpenVLA 0.80) -&gt; #6 cheapest-next-step sharpened into a go/no-go separability gate on the probe family; Legato '~10% smoother' wrong both directions (NSPARC ~flat; real headline completion time -19..23% vs RTC matched); Reflex 2.58x is vs a full-recompute strawman (defensible: async split -47..54% reaction latency, stall 100-&gt;0; K draws share one trunk prefill -&gt; #19 cost model split); Compression Gap heavily oversold (tiny non-VLA single-seed, mechanism asserted, our AR bit budget ~22x the bound). Ideas fed: #6 #9 #16 #19 #22. Refill sweep -&gt; lit-radar-0818: 16 candidates verified, 12 were CORPUS DUPS caught only by local grep (sweep-agent exclusion list insufficient - pool drying; instrument note logged in the 0818 item).

<details><summary>full record</summary>

Lit slice (standing allocation): refill hooks banked in the 0816 session's fresh sweep, all ids abs-page-verified by the sweep agent + dup-checked against papers/ + ideas/ + queue (2 dup catches dropped: 2607.23777 = Muon-SW already read 0813; 2606.05468 FlowPRO = covered in hy-embodied-stack.md), priority-ordered - 2605.02881 MolmoAct2 (open-weight VLA on MolmoER, our trunk's direct lineage; specialize-then-rehearse on 3.3M samples incl. curated pool from 1,222 public LeRobot datasets w/ SO-100/101 subsets, 720h open bimanual; beats pi-0.5 + Gemini Robotics ER-1.5 across 7 benchmarks; #17 trunk-lineage + #9 curation recipe = the survey's #1 recommendation's paper) &gt; 2607.24481 ArmnetBench v0.1 (SO-101 arm-farm parallel eval, 7 policies x 12 tasks, 3,118 human-labeled episodes success/suboptimal/failure RELEASED - rare labeled failure-rollout data on our exact embodiment; #9 panel-calibration + #6/#16 failure labels we cannot collect ourselves) &gt; 2608.04246 SAFECAST (contrast-set perturbations + hidden-state risk probes + functional conformal prediction, ROC-AUC gains on DROID-real + LIBERO under shift; #6, pairs with ArmnetBench labels) &gt; 2607.14695 Reflex (timestep-invariance -&gt; streaming KV-cached inference for flow VLAs, 2.58x, stable 50 Hz, 54% reaction-latency cut; #22 - read w/ Legato 2602.12978 as infer-time/train-time complements) &gt; 2604.03191 Compression Gap (encoder upgrades give &gt;21-pt gains through continuous action heads but attenuated through discrete codebooks - mechanism-level flow-over-AR prediction that sharpens exactly when the trunk improves, i.e. our vision-unfrozen run; #19/#4). VERIFIED SPARES: 2605.30834 Hide-and-Seek (trajectory-label contrastive failure localization, #6), 2605.29605 VLAConf (single-pass success confidence from frozen-VLA hidden states, success-data-primary - calibration needs verifying, #6), 2605.13959 WarmPrior (temporal prior source, straighter paths, #19/#17), 2602.12978 Legato (native chunk continuation training, ~10% smoother vs RTC, #22), 2604.16683 Rewind-IL (training-free chunk-consistency failure detection from own draws, synergy w/ #19 machinery, #6), 2605.23061 SF-NorMuon (partly superseded by Muon-SW read, adamc spare). Papers page(s) same session per the permanent rule; dedup-check each id before writing.

</details>

---

**`lit-radar-0816`** · `cpu`

Lit slice (standing allocation): refill hooks banked in the 0815 session's fresh sweep, all ids VERIFIED against abs pages + dup-checked against papers/ + ideas/ + queue (21 checked clean, 10 discarded as covered), priority-order…

**boundary:** CLOSED 2026-08-09T21:0xZ work session: all 5 hooks deep-read via 5-subagent fan-out + parallel refill sweep, 5 Papers pages same session (weight-decay-plasticity, learning-while-deploying, fomo-fd, vla-gse, actioncache), ideas #4/#6/#16/#17/#19/#22 + adamc watch fed, index+SUMMARY wired. EVERY hook needed corrections, 3 loud: FoMo-FD 'no env rollouts' FALSE (conformal calibration needs ~19 successful deployed-policy rollouts/task; only WM training is rollout-free; 'FDR'=detection rate not false discovery); ActionCache 'changes cheap-draws cost model' WRONG for our stack (head-only speedups, trunk unskippable - keys computed FROM trunk outputs; top-1 retrieval collapses draws; SR not held on GR00T -3.2/LIBERO -5.0; real-SO-101 end-to-end 1.66x); LWD QAM adopted from Li&amp;Levine not theirs + 95% is a mixed human-rubric metric. Softer: WD-plasticity 'hurts base loss' only in the overtrained regime + lambda-prop-eta framing unlicensed (our 1e-5 is 4 orders below their range); VLA-GSE 'zero-shot' = perturbations-only + insulation is LoRA-grade empirical. Refill sweep dup-catches: 2607.23777 Scale-WD = ALREADY-READ Muon-SW (0813) despite verification, 2606.05468 FlowPRO standalone = covered in hy-embodied-stack.md - both dropped.

<details><summary>full record</summary>

Lit slice (standing allocation): refill hooks banked in the 0815 session's fresh sweep, all ids VERIFIED against abs pages + dup-checked against papers/ + ideas/ + queue (21 checked clean, 10 discarded as covered), priority-ordered — 2602.11137 Weight Decay Improves LM Plasticity (larger pretrain WD hurts base loss but INCREASES downstream finetune gains via separable representations; mechanistic frame for what lambda-prop-eta decay does to a pretrained Molmo2 trunk's plasticity — adamc priority 1) &gt; 2605.00416 Learning While Deploying (fleet-scale offline-to-online RL on a real 16-robot dual-arm fleet; Distributional Implicit Value Learning + Q-via-Adjoint-Matching NATIVE to flow action generators, 95% avg across 8 tasks; #16 top-rank real-robot flow-native entry + adjoint-matching taxonomy slot) &gt; 2607.27511 FoMo-FD (action-conditioned flow-matching WORLD MODEL flags visual-action inconsistency, conformal-calibrated on successes ONLY — no failure demos, no env rollouts, 96.6% FDR at 1.3% FAR on dVRK; #6 — actually fits our no-rollouts constraint where Foresight did not; #17 latent-WM-as-verifier) &gt; 2605.06175 VLA-GSE (spectral decomposition of frozen backbone initializes generalized+specialized experts, 2.51% params updated, 81.2% zero-shot LIBERO-Plus; #4/fjoint — knowledge-insulation-by-construction alternative to the brief-unfreeze rung) &gt; 2607.06370 ActionCache (training-free action caching+refinement for flow VLAs, 10.44x pi-0.5 / 40.17x GR00T speedup at held SR; #22 + changes #19's cheap-draws cost model). VERIFIED SPARES: 2605.30834 Hide-and-Seek (trajectory-label contrastive failure localization, #6), 2605.13959 WarmPrior (temporal prior replaces Gaussian source, straighter paths; #1/#19/#17), 2608.04246 SAFECAST (contrast-set failure detection, #6, rolled fwd from 0815), 2607.14695 Reflex (streaming inference 50Hz, #22, rolled fwd), 2605.23061 SF-NorMuon (schedule-free spectral, WD-at-fast-iterate essential; adamc spare). Dropped 0815 spares: 2607.10959 WSqD + 2606.10305 SARM2 + 2606.11408 DEHP (outranked; reasons in 0815 close). Papers page(s) same session per the permanent rule; dedup-check each id before writing.

</details>

---

**`lit-radar-0815`** · `cpu`

Lit slice (standing allocation): refill hooks banked in the 0814 session's fresh sweep, all ids VERIFIED against abs pages + dup-checked against papers/ + ideas/ + queue, priority-ordered

**boundary:** CLOSED 2026-08-09T19:39:56Z work session: all 5 hooks deep-read via 5-subagent fan-out + parallel refill sweep, 5 Papers pages same session (weight-norm-criticality, weibull-weight-scale, decoupled-action-expert, foresight-failure-detection, redflow; commit c53e517, check 598 green); 3 hook corrections caught (Foresight trains on failure rollouts - NOT the no-rollouts affirmative case; Weibull decomposition is 2-of-3 forces from weights-only; Decoupled Action Expert testbed is Diffusion Policy not VLA, freeze direction inverted); ideas #4/#6/#16/#17 fed; refill -&gt; lit-radar-0816 queued

<details><summary>full record</summary>

Lit slice (standing allocation): refill hooks banked in the 0814 session's fresh sweep, all ids VERIFIED against abs pages + dup-checked against papers/ + ideas/ + queue, priority-ordered — 2607.21005 Weight-norm Criticality (loss spikes from decay+normalization driving scale-invariant norms to zero; a concrete failure mode for the LIVE adamc weight-norm watch) &gt; 2606.19367 Weibull weight-scale evolution under AdamW (alignment/injection/decay force decomposition recoverable from sparse checkpoints — an analysis frame for our banked 5k saves) &gt; 2511.12101 Decoupled Action Expert (task knowledge confined to the conditioning pathway, 5M MLP matches 244M U-Net; sharpest available datum for the fjoint F-then-joint rung's seam question) &gt; 2606.23085 Foresight (learned failure detection from task-level success labels only, no env rollouts, conformal-calibrated — the #6 learned-verifier affirmative case matching our no-rollouts constraint) &gt; 2607.27782 RedFlow (offline RL converting failures into action-level corrective supervision for flow VLAs, real-world 56.7-&gt;74.7%; #16 RL-pole entry candidate: real-robot, few-sample, failure-driven). VERIFIED SPARES if a pick falls through: 2607.10959 WSqD horizon-free schedule, 2606.10305 SARM2 stage-aware reward model, 2608.04246 SAFECAST contrast-set failure detection, 2607.14695 Reflex streaming inference (#22), 2606.11408 dynamic execution horizon (#22). Papers page(s) same session per the permanent rule; dedup-check each id before writing.

</details>

---

**`lit-radar-0814`** · `cpu`

Lit slice (standing allocation): refill hooks banked in the 0813 session's fresh sweep, dup-checked against papers/ + ideas/ + queue, priority-ordered

**boundary:** CLOSED 19:2xZ 08-09 work session: all 5 hooks deep-read via parallel-subagent fan-out, 5 Papers pages same session (hyperball-optimization, anytime-pretraining, vla-fail, fpo-flow-policy-optimization, x-tokenizer); ideas #3/#5/#6/#16/#17/#22 fed; 2 hook corrections caught (2602.03702 NOT Defazio; 2606.14752 tokens never executed at inference); refill sweep ran in-session -&gt; lit-radar-0815 queued (5 verified dup-checked hooks + 5 verified spares banked in the item)

<details><summary>full record</summary>

Lit slice (standing allocation): refill hooks banked in the 0813 session's fresh sweep, dup-checked against papers/ + ideas/ + queue, priority-ordered — 2606.16899 Hyperball / Fantastic Pretraining Optimizers II (weight-norm tracking through warmup/decay, grad-norm growth as weight norms shrink; adamc-watch third frame beside 2512.08217 + 2607.23777) &gt; 2602.03702 Anytime Pretraining (Defazio — AdamC's author — horizon-free LR schedules w/ weight averaging; bears on our cosine-to-10% floor reads) &gt; 2606.21386 VLA-FAIL (failure detection WITHOUT failure data: last-layer Mahalanobis + action-chunk-consistency over receding-horizon overlaps — the #6 verifier family AND our #22 boundary-incompat read's overlap machinery published as a detector) &gt; 2510.09976 RFT of flow-matching policies (#16 RL-pole roster completion) &gt; 2606.14752 X-Tokenizer (multimodal action tokenizer w/ masked action modeling; #5 learned-VQ falsifier family). Papers page(s) same session per the permanent rule; dedup-check each id before writing.

</details>

---

**`lit-radar-0813`** · `cpu`

Lit slice (standing allocation): refill hooks banked 08-09 18:3xZ fresh sweep, dup-checked against papers/ + ideas/, priority-ordered

**boundary:** CLOSED 2026-08-09T18:59:36Z work session: all 5 hooks deep-read, 5 Papers pages landed same session (muon-sw.md, asyncvla.md, silent-failure-observability.md, sa-vla.md, streamvla.md); ideas #6/#11/#16/#17/#22 + the adamc-watch frame fed (weight-norm plateau signature + alignment-cosine probe banked). Notable: AsyncVLA is NOT async-execution despite the title (filed on #22 so it isn't re-hooked); SA-VLA measures naive sparse RL NEGATIVE (77.5 vs 81.0 no-RL). Refill sweep ran in-session -&gt; lit-radar-0814 queued (5 dup-checked hooks)

<details><summary>full record</summary>

Lit slice (standing allocation): refill hooks banked 08-09 18:3xZ fresh sweep, dup-checked against papers/ + ideas/, priority-ordered — 2607.23777 Scale-Weight-Decay/Muon-SW (steady-state weight-norm analysis, AdamC-family sibling; feeds the live adamc_100k grad/weight-norm watch interpretive frame beside 2512.08217) &gt; 2511.14148 AsyncVLA (asynchronous flow matching w/ self-correction; #22 family, VLA-Corrector adjacency) &gt; 2606.03134 silent-failure observability (false-success detection, proprio-vs-vision detectors on bimanual ALOHA; #6/#16 verifier family) &gt; 2602.00743 SA-VLA (spatially-aware FM for VLA RL; #16 RL pole x #11 spatial-aux crossover) &gt; 2602.01100 StreamVLA (reason-act cycle via completion-state gating; #6 phase-estimation adjacency, skim-to-place). Papers page(s) same session per the permanent rule; dedup-check each id before writing.

</details>

---

**`lit-radar-0812b`** · `cpu`

Lit slice (standing allocation): refill hooks banked 08-09 18:2xZ fresh sweep, dup-checked against papers/ + ideas/, priority-ordered

**boundary:** EXECUTED 18:27-18:3xZ 08-09 work session (commit 7e78c9d): all 5 banked hooks deep-read, 5 papers pages landed same session (vla-corrector / pi-stepnft / dfm-vla / onewm-vla-one-token / hif-vla); ideas #1/#5/#6/#11/#16/#17/#22 fed; refill sweep ran -&gt; lit-radar-0813 queued (5 dup-checked hooks; 2605.08168 candidate caught as already covered)

<details><summary>full record</summary>

Lit slice (standing allocation): refill hooks banked 08-09 18:2xZ fresh sweep, dup-checked against papers/ + ideas/, priority-ordered — 2607.01804 VLA-Corrector (lightweight detect-and-correct inference, adaptive action horizon; #6/#19 verifier family + #22 async adjacency) &gt; 2603.02083 pi-StepNFT (online RL for flow VLAs, 'wider space needs finer steps'; #16 RL pole beside RLDT/Z-1/FlowPRO) &gt; 2603.26320 DFM-VLA (DISCRETE flow matching iterative refinement; #17 head axis — sits exactly between our AR-token and continuous-flow poles, beside HiFlow) &gt; 2605.07931 One-Token-Per-Frame (visual bandwidth in world models for VLA; VLA-JEPA family, #17) &gt; 2512.09928 HiF-VLA (hindsight/insight/foresight motion representation; skim-to-place). Papers page(s) same session per the permanent rule; dedup-check each id before writing.

</details>

---

**`lit-radar-0811`** · `cpu`

Lit slice (standing allocation): banked radar hooks from the 0810 fresh sweep, priority-ordered

**boundary:** CLOSED 18:2xZ 08-09 (work session, adamc_100k shadow window)

<details><summary>full record</summary>

Lit slice (standing allocation): banked radar hooks from the 0810 fresh sweep, priority-ordered — 2605.08511 Trajectory-Consistent Flow Matching (train-inference gap: rectified-velocity aux + trajectory consistency + velocity smoothness + RK4 inference; feeds #12 solver/Heun-gap and the smoothness family) &gt; 2606.08602 RL-for-FM density transport + 2604.01570 Feasible-Action-Neighborhood prior (post-SFT menu, #16) &gt; 2603.27281 HiFlow tokenization-free scale-wise AR-via-FM (trunk/decode family, #17) &gt; 2602.10098 VLA-JEPA latent world model. Papers page(s) same session per the permanent rule; dedup-check each id against papers/ before writing. | EXECUTED 18:0x-18:2xZ 08-09 work session: ALL 5 banked hooks cleared, 5 papers pages SAME SESSION (commits 1a8dc93 + eaa3a21, check 598 green both) — TCFM 2605.08511 (trajectory-consistent-flow-matching.md; #12 third-axis family map: training-side integration supervision + denoising-clock smoothness x RK4 interaction ablation 70%-&gt;10%; RK4-on-banked-checkpoint zero-training hook PRICED not queued); RLDT 2606.08602 (rldt-density-transport-rl.md; #16 RL-pole entry 3, SVGD transport native to FM, infra price 64-1000 envs + critic + 30-48 GPU-h at small scale); FAN 2604.01570 (fan-feasible-action-neighborhood.md; #16 zero-infra SFT lever CVPR26 + #19 external mean-collapse prior); HiFlow 2603.27281 (hiflow-scalewise-ar-flow.md; #17 head-axis third pole, continuous-vs-VQ controlled datum 90-vs-70 threading); VLA-JEPA 2602.10098 (vla-jepa-latent-world-model.md; #17 predictive representation-supervision pole, human-video-buys-robustness 79.5-&gt;62.9 ablation + #11 Spatial-Forcing fork note). ideas #11/#12/#16/#17/#19 fed; fresh refill sweep ran -&gt; lit-radar-0812b (5 new hooks dup-checked)

</details>

---

**`lit-radar-fresh-sweep-0810`** · `cpu`

Lit slice (standing allocation): FRESH arXiv sweep for new hooks

**boundary:** EXECUTED 08-09 work session 17:42-17:50Z (recovered by tick)

<details><summary>full record</summary>

Lit slice (standing allocation): FRESH arXiv sweep for new hooks — the banked radar backlog is EMPTY as of 08-09 17:3xZ (three slices ran 08-09: async-exec cluster, 0809b QDepth+sweep, 0812a FAFM/VISTA/LAFP; the 17:0x session verified empty and skipped rather than force a thin sweep, and caught+reverted a FASTER dup page pre-commit — 2603.19199 already covered by papers/async-execution-2.md). Sweep priority: (1) anything re-ranking the adamc_100k readout (AdamC grad-norm claim, vision-unfreeze-from-0 evidence), (2) the fjoint sequencing owner call, (3) actckpt ladder. Papers page(s) same session per the permanent rule. | EXECUTED 17:42-17:50Z 08-09 work session (session ended turn WITHOUT committing; the 17:50 tick audited the orphaned diff — dup-grep clean, plain-words blocks present — and committed it): fresh sweep ran 5 searches; 2 deep reads + 2 papers pages SAME SESSION — (1) 2512.08217 'Correction of Decoupled Weight Decay' (AdamC's direct successor; papers/weight-decay-correction.md) = priority-1 hit: adamc_100k grad-norm watch interpretive frame banked (flat norms expected, ~nil loss effect, head-exclusion partition validated twice, 10% LR floor on the recommended side, no-steady-state-at-100k caveat); (2) 2606.31846 Z-1 (papers/z1-selective-joint-rl.md) = priority-2 hit: fjoint joint phase as diagnostic-gated conditional escalation (4th frozen-first vote) + #16 RL-pole datum (+13.2 pts from 1,199 demos, GRPO on flow-SDE). ideas.md hooks fed (#4, #16, #17 + per-idea pages). Radar hooks banked unread -&gt; lit-radar-0811

</details>

---

**`owner-ticket-v2all-selection-0809`** · `gpu-local`

OWNER STEERING 15:44Z 08-09: best 1-NFE ticket over the ENTIRE so101_pick_place_v2 (training rows included)

**boundary:** CLOSED 16:4xZ 08-09 same-session as the landing (chained work session per the 16:10 tick handoff); ~1.0 GPU-h local (55-min eval + 53 s CPU selection) · [pre-reg](archive/now-2026-08-09.md)

<details><summary>full record</summary>

OWNER STEERING 15:44Z 08-09: best 1-NFE ticket over the ENTIRE so101_pick_place_v2 (training rows included) — scored the sha-pinned m64 bank through ftrig@4000 euler-1 s=0 over all 50 episodes / 32,679 frames (unit fontaine-ftrig-ticket64-v2all, landed 16:35:31Z), winner selection + subset diagnostics detached (fontaine-ftrig-v2all-winner, landed 16:36:58Z). WINNER = ticket 12, pooled MAE 5.26497 (bank median 5.474, worst 6.094); ticket 59 (holdout winner) rank 5 (5.330), ticket33 rank 19 (5.405). Memorized-rows read: train rows (29,405) ticket12 rank 1 (4.536) vs heldout rows (3,274) ticket12 rank 9 (11.808) while 59 holds rank 3 (11.722), ticket33 rank 51; Spearman train-vs-heldout rows 0.39 (weak), v2all-vs-holdout ladder 0.57 =&gt; ticket choice measurably sensitive to memorized rows; 59 = generalization pick, 12 = deployment-fit pick. Table + read posted in-channel 16:4xZ; ticket_ftrig4k_rigv2all_winner.npz committed in-repo (sha ec0484e8) + uploaded to fontaine-checkpoints tickets/ (hub commit d8cbfcc); analysis json banked reports/analysis__ftrig_ticket_selection_rigv2all.json | PRE-REG FORM: owner-steered eval, registered via the charter's now.md-entry route (15:59Z entry: decode/bank/output/ETA pinned at the 15:46Z launch ack, before any result was read); entry rolls verbatim to the cited archive page

</details>

---

**`corpus-continuity-screen`** · `cpu`

#9 corpus kinematic-continuity screen (CPU, record-only, VISTA hook 08-09 papers/vista-umi-validation.md

**boundary:** CLOSED 17:1xZ 08-09 at zero GPU; reads banked reports/analysis__corpus_continuity_screen.json (+ .max_ratios.npy sidecar)

<details><summary>full record</summary>

#9 corpus kinematic-continuity screen (CPU, record-only, VISTA hook 08-09 papers/vista-umi-validation.md — exploratory read on our own data, SDN-read precedent, no pre-reg; any curation change from it DOES need one): score every episode of community_curated_v0 (+ the 2 rig repos as calibration anchors) for per-tick action-displacement continuity — VISTA's three-regime scoring (full marks / linear / exponential) recalibrated to so100 joint space from the rig repos' own displacement distribution; episode score = min over ticks. AUDIT FIRST per standing rule: check what dup_content_census.py / frame-mining machinery already computes per-episode before writing anything new. Oracle-gate on synthetic planted-dropout fixtures (teleport jump in/out, clean episode = full marks); quote score distribution + tail episodes, cross-check low-continuity repos against the banked LORO influential-repo lists (arch/box-batch reads). A null (no corrupted tail) closes the hook at zero cost; a signal is a #9 curation lever with its own pre-reg | EXECUTED 17:1xZ 08-09 (same session as queued+1): corpus_continuity_screen.py landed (oracle 7 fixture families; own two-layout parquet loader — census loader assumed list arrays, v3.0 rig repos are FixedSizeList). 52,507 episodes / 981 repos scored, zero read failures. QUALIFIED NULL: EXP tail 123 eps (0.23%) = wrap census's two known repos (kevin510 40/40 wrap seam, willnorris 41/42 counts-units) + 42 genuinely-new sub-300-deg dropout eps in 30 repos (0.08%, order of magnitude under the census's curation kill line — NO pre-reg queued, would re-litigate the owner's 08-05 16:13Z drop). Zero LORO overlap; 8 tail eps are panel rows (bounded ~0.05 pooled, per-repo diagnostics on those 2 repos untrustworthy — standing caveat). Results post 2026-08-09-corpus-continuity-screen.md (2 dark charts); wrap-census post cross-annotated; ideas #9 hook CLOSED; instrument banked as curated_v1 intake filter.

</details>

---

**`lit-radar-hooks-0812a`** · `cpu`

Lit slice (standing allocation): fresh-sweep hooks banked 08-09 14:4xZ, unread skim-class

**boundary:** CLEARED 15:5xZ 08-09 — all pages same-session per the permanent rule

<details><summary>full record</summary>

Lit slice (standing allocation): fresh-sweep hooks banked 08-09 14:4xZ, unread skim-class — Frequency-Aware Flow Matching 2606.20135 (continuous/consistent action generation: the SEAM boundary family from the TRAINING side; #22/#1); VISTA 2606.04708 (vision-grounded physics-validated UMI data adaptation — beside RDT2's 10k-h premise; #16/#9); latent-action-guided FM pair LAFP 2606.10517 + Flowing With Purpose 2606.23420 (latent-action bias on flow policies; #17/#12). Papers page(s) same session per the permanent rule; dup-check against the index tracker before writing | PARTIAL 15:4xZ 08-09: FAFM 2606.20135 deep-read + papers page SAME SESSION (frequency-aware-flow-matching.md, smoothness/boundary family section added to the index tracker) — DCT-coefficient flow matching + H1 Sobolev loss; fed #12 (17x6 target for distill rungs), #9 (mixed-Hz ill-posedness theorem + 94-&gt;0 collapse demo), #16 (LDLJ metric), #22 (training-side family map: does NOT touch the cross-chunk term our boundary read measured). REMAINING hooks unread: VISTA 2606.04708, LAFP 2606.10517 + Flowing With Purpose 2606.23420 | VISTA 2606.04708 read + page 15:5xZ same session (vista-umi-validation.md): physics-validation pipeline (continuity/collision/fidelity, 65-vs-0% OSR prediction), fisheye-matched VQA (mismatched VQA HURTS -13pts), stage-2 frozen+flow = third production KI vote; fed #4/#9/#11/#16; NEW #9 hook banked: continuity screen on our corpus, zero GPU. Remaining unread: LAFP 2606.10517 + Flowing With Purpose 2606.23420 (latent-action FM pair) | CLEARED 15:5xZ 08-09: LAFP 2606.10517 read + page (lafp-latent-flow-policy.md, Procgen skim-to-place: latent-action flow family map for #17, x-vs-v-prediction stability note); pair 2606.23420 caught as ALREADY covered (latent-action-priors.md) by dup-check. All 4 radar hooks resolved in one session (FAFM, VISTA, LAFP pages + 1 dup)

</details>

---

**`boundary-incompat-read-npz`** · `cpu`

#1/#22 boundary-incompatibility CPU read (record-only, SDN-read precedent

**boundary:** CLOSED 15:2xZ 08-09 zero GPU; #22/#1 records + SEAM papers page updated same session

<details><summary>full record</summary>

#1/#22 boundary-incompatibility CPU read (record-only, SDN-read precedent — exploratory read on banked stacks, NO pre-reg needed, any escalation does): measure cross-chunk mode incompatibility from banked panel npz — for temporally adjacent panel frames of the same episode (Delta-t = a few ticks), compare earlier chunk's tail vs later chunk's head on their overlap (per-frame index join, the draws_fairness/SDN join pattern); quote vs within-chunk smoothness as the anchor. AUDIT FIRST per standing rule: check which panel plans place multiple ordered frames per episode close enough to overlap at chunk 50, and whether flow + AR stacks both qualify. Oracle-gate on synthetic fixtures (planted compatible/incompatible chunk pairs in/out; degenerate same-frame overlap must read exactly 0). A null (chunks already agree at the seam) closes the #22 bridging direction for our stack at zero GPU cost; a real signal is the design input for any SEAM/PAINT-class arm at the #16 rig bench | EXECUTED 15:2xZ 08-09 (work session): boundary_incompat_results.py landed oracle-gated (planted pairs exact, dt=0 degenerate 0, NaN poison, 6 abort branches) + run on 5 banked stacks (flow80k stablekey/ticket33/draws10mean, molmo2 AR 40k/60k greedy), 13,693 pairs, truth overlaps byte-identical on all. NOT A NULL: seam D 1.1-1.27x model err; boundary jump 11-14x per-step motion (smooth within, jerky between); dt-&gt;0 intercept fresh-noise 6.04 vs AR-greedy ~2.7 vs shared-ticket 2.07 (noise coupling deletes the mode term; ticket33 = free ablation). json banked reports/analysis__boundary_incompat_panels.json + dark-mode dt-curve chart; results post 2026-08-09-boundary-incompat-results.md; #22 direction CONFIRMED not closed, still parked on #16; escalation needs own pre-reg

</details>

---

**`adamc-100k-live`** · `gpu-box`

OWNER RUN LIVE (launched 13:30Z 08-09): fontaine_molmo2_adamc_100k_ddp4

**boundary:** endpoint ~08-11/12 (est 1.7-2.1 s/step =&gt; 47-58 h from 13:32Z); chained k4l2 panel eval (--report + npz) in-unit; then leaderboard row + grad-norm chart readout | CLOSED 22:40Z 08-09: OWNER KILL ('not looking great', GPUs reassigned to ER-60k) at step ~11,840, ~35.7/310 GPU-h. Final probe ladder ended 10.30@11500 = run-best (the 3-rise watch receded). step_010000 kept on box + weights-only upload to fontaine-checkpoints VERIFIED DONE; train_log.jsonl banked box+local. babysit entry pruned 22:4xZ. Post-mortem chart split to adamc-postmortem-chart-0809. · [pre-reg](posts/2026-08-09-prereg-molmo2-adamc-100k.md)

<details><summary>full record</summary>

OWNER RUN LIVE (launched 13:30Z 08-09): fontaine_molmo2_adamc_100k_ddp4 — base Molmo2-4B, 100k steps, eff-32 (8/rank x4, chunks 4), vision unfrozen step 0, decoder 1e-4 / text 2e-5 / vision 2e-5, warmup 1000, AdamC lambda=1e-5 (amendment 2), seed 1, save 5000, ZeRO-1 + chunk-grad-allreduce + async saves. Babysit entry adamc_100k (kill bars: NaN/inf, @10k&lt;@2500, &gt;25 x3 after 5k, vram 77 near-OOM; grad-norm RECORD-ONLY AdamC watch). First-poll facts owed to channel: measured s/step, vram peak, wall projection, first async-save line. OOM policy: relaunch chunks 8 microbatch 1; second OOM = owner steer.

</details>

---

**`lit-radar-hooks-0811a`** · `cpu`

Lit slice (standing allocation): clear the 08-09 second sweep's banked hooks

**boundary:** any GPU-busy window; K endpoint chain (~18:3xZ 08-09) outranks it when it opens | EXECUTED 14:3x-14:4xZ 08-09 (adamc_100k shadow): SEAM 2607.04609 deep-read + papers page SAME SESSION (seam-boundary-steering.md) — closed-form lambda(1-t) nudge toward the previous chunk unexecuted tail, +1% cost, jerk -28% success preserved (vs RTC -54%@1.22x, ACT-TE -84% but success -12pts); #22 arm order updated (SEAM ahead of PAINT on cost, PAINT stays the async-robust one); NEW free hook banked = boundary-incompatibility CPU read on banked panel npz (split to boundary-incompat-read-npz). Robot Critics 2606.21572 skim-to-place DONE + compact page (robot-critics-small-stuff.md): trained-critic pole placed and PARKED (needs rollout labels + video model; ceiling reads cap payoff). Fresh sweep hooks banked to lit-radar-hooks-0812a.

<details><summary>full record</summary>

Lit slice (standing allocation): clear the 08-09 second sweep's banked hooks — SEAM 2607.04609 (smooth execution of action-chunked motion: adjacent independently-sampled chunks pick incompatible modes -&gt; boundary discontinuities; directly on the #1/#22 boundary-jerk open term the SDN read left unmeasured) + Robot Critics 2606.21572 skim-to-place (#6/#19 verifier family; read or drop). Papers page(s) same session per the permanent rule

</details>

---

**`lit-radar-hooks-0810b`** · `cpu`

Lit slice (standing allocation): Qwen-VLA 2605.30280 deep-read (last unread banked hook

**boundary:** EXECUTED same session as queued (attach_K window, 12:2x-12:3xZ 08-09)

<details><summary>full record</summary>

Lit slice (standing allocation): Qwen-VLA 2605.30280 deep-read (last unread banked hook — Qwen3.5 early-fusion native-multimodal trunk + single-stream DiT flow expert; #17 trunk ledger, the early-fusion pole vs late-fusion Molmo2) + fresh sweep for new hooks (priority: anything re-ranking the stage-2 decision before/at tonight's Delta_seam read, or the actckpt ladder). Papers page(s) same session per the permanent rule — EXECUTED 12:2x-12:3xZ 08-09 same session it was queued (papers page landed same session: papers/qwen-vla-early-fusion.md): Qwen-VLA deep-read — first production VLA on a natively early-fused trunk (Qwen3.5-4B interleaved tokens + gated-linear hybrid attention, 1.15B single-stream DiT flow expert); OOD headline real-ALOHA 76.9 vs pi0.5 41.5 carried with the stack-vs-stack confound LOUD (no fusion-controlled ablation exists); four-stage recipe T2A(trunk FROZEN)-&gt;joint CPT-&gt;SFT(VL 0.1)-&gt;narrow PPO. Fed #17 (early-fusion pole staked), #4 (F-then-joint production vote #2 filed pre-Delta_seam, language-only-Stage-I disanalogy noted), #19 (tau=0.6 deploy sharpening = production cool-side sighting beside the dT table), #16 (embodiment prompts + data mixture). Fresh sweep: nothing re-ranking stage-2 or actckpt; 2 NEW hooks banked unread — SEAM 2607.04609 (chunk-boundary mode incompatibility, the #1/#22 boundary-jerk open term) + Robot Critics 2606.21572 (#6/#19 verifier family)

</details>

---

**`lit-radar-hooks-0810a`** · `cpu`

Lit slice (standing allocation): clear the 08-09 sweep's banked radar hooks

**boundary:** EXECUTED same session as queued (attach_K window, 12:2xZ 08-09)

<details><summary>full record</summary>

Lit slice (standing allocation): clear the 08-09 sweep's banked radar hooks — ForesightFlow 2606.04968 deep-read first (self-scored best-of-K, no external critic; seventh selection flavor, feeds #19/#1/#6 scorer rungs against our banked ceiling reads) + 2606.20246 fewer-layers CKA pruning (feeds #17 trunk-redundancy ledger + the throughput accounting; skim-check the 30% inference claim's baseline) ; Qwen-VLA 2605.30280 rides if time allows. Papers page(s) same session per the permanent rule — EXECUTED 12:2xZ 08-09 same session it was queued (permanent-rule pages both landed): (1) ForesightFlow 2606.04968 deep-read (papers/foresightflow-self-scored-bestofk.md) — seventh selection flavor; LOAD-BEARING K-sweep: separate 500M critic FLAT K=1-&gt;5 (39.0-&gt;38.4) vs self-scored +5.0 = selector shape &gt; size, third strike on post-hoc probe selectors vs our banked best-of-10 ceiling; 1-NFE endpoint preview instrument (tau 0.83, ~97% gain retained) banked to #1/#12; decoupled-AWFM weight-space recipe to #16; needs stage labels + rollouts so directional prior only. (2) CLP fewer-layers 2606.20246 deep-read (papers/fewer-layers-clp.md) — 33-50% of finetuned-VLA depth is CKA twins incl 8/16 DiT expert layers; prune-BEFORE-finetune heals; +6.9 only in the 10%-data regularization regime, full-data ~cost-neutral at -28-31% train time = honest expectation for our regime; CKA map banked as one-forward-pass diagnostic for our trunk/expert; throughput fourth lever class (FLOP-count, immune to pass-1's scheduler-artifact class); prune-then-attach named sequel arm. Fed #19/#1/#12/#16/#17/#4 + index/SUMMARY rows

</details>

---

**`lit-radar-hooks-0809b`** · `cpu`

Lit slice (standing allocation): last banked radar hook — QDepth-VLA 2510.14836 (depth aux, #11/#17; cross-read against the Spatial Forcing/VEGA teacher-x-depth grid) + a fresh arXiv sweep for new hooks (the banked backlog is now…

**boundary:** EXECUTED same session as queued (attach_K window, 12:1x-12:2xZ 08-09)

<details><summary>full record</summary>

Lit slice (standing allocation): last banked radar hook — QDepth-VLA 2510.14836 (depth aux, #11/#17; cross-read against the Spatial Forcing/VEGA teacher-x-depth grid) + a fresh arXiv sweep for new hooks (the banked backlog is now EMPTY — first sweep priority: anything re-ranking the stage-2 decision or the actckpt ladder). Papers page(s) same session per the permanent rule — EXECUTED 12:1x-12:2xZ 08-09 same session as queued: QDepth-VLA 2510.14836 deep-read + papers page SAME SESSION (papers/qdepth-vla.md) — third aux-spatial recipe class (expert-generative: parallel 18L expert predicts VQ depth tokens K=256/16x16 from vision tokens, monocular ViDA pseudo-labels, depth tokens RIDE the inference context unlike VEGA/SF); ablation split carried loudly (-2.9 loss vs -8.5 expert = scaffold not geometry carries ~5.6 of the win); quantized-beats-regression +3.9 = perception-side discretization datapoint. Fed #11 (third recipe), #17 (only aux-spatial recipe needing no encoder seam -&gt; named single-tower fallback), #5 (discretization), #4 (context: another production parallel-expert sighting). Index integrity fix: 2 stale 'radar hook, unread' rows (VEGA/HyperVLA) flipped to page links. Fresh sweep landed 3 NEW hooks banked unread: 2606.20246 fewer-layers CKA pruning (#17/throughput), 2605.30280 Qwen-VLA early-fusion trunk (#17), 2606.04968 ForesightFlow self-scored best-of-K (#19/#1/#6 seventh selection flavor)

</details>

---

**`lit-radar-async-exec`** · `cpu`

Lit slice (standing allocation): the async/real-time execution family from the banked radar hooks

**boundary:** EXECUTED 11:5x-12:1xZ 08-09 (the chained work session run_work_next armed): all three async-family hooks deep-read, ONE cluster papers page SAME SESSION (papers/async-execution-2.md — three orthogonal levers: FASTER 2603.19199 scheduling/TTFA, ABPolicy 2602.23901 B-spline representation, DEFLECT 2605.19294 stale-vs-fresh FM-DPO). #22 arm menu re-ranked (naive-switch measure -&gt; HAS-on-decode NEW rung 2 -&gt; PAINT -&gt; A2C2 -&gt; TT-RTC/DEFLECT; DEFLECT carried at its restart-corrected +1.6-2.3 pp, not the +6.4 headline; d~18 untested by anyone stays loud); #16 TTFA identity + jerk instruments banked; #12 fourth one-step pole (one-step head/many-step tail). Cluster closed EARLY (12:0xZ) so Spatial Forcing 2510.12276 RODE ALONG per the item's own clause: papers/spatial-forcing.md same session (teacher-x-depth interaction vs VEGA; 3.8x = fewer-steps lever, teacher overhead unreported; #17/#11 hooks updated). RDT2 2602.03310 ALSO rode (the item's own if-time-allows clause): papers/rdt2-umi-scaling.md same session — production F-shape vote (AR-first + frozen-trunk expert + distill, no joint stage) filed as Delta_seam ledger context; #16 hours-scale premise + beta~0.23; #5 RVQ; #12 second 1-NFE point. QDepth-VLA 2510.14836 rolls to the refill item

<details><summary>full record</summary>

Lit slice (standing allocation): the async/real-time execution family from the banked radar hooks — FASTER 2603.19199 (real-time flow VLAs) + ABPolicy 2602.23901 + DEFLECT 2605.19294 (async execution), optionally RDT2 2602.03310 if time allows; papers page(s) same session per the permanent rule. Feeds #22 (async chunk execution / drift-monitor thread), #16 rig deployment latency (fresh relevance: HyperVLA's 4ms pole banked 08-09), #12 one-step deployment leg. Spatial Forcing 2510.12276 (VEGA's baseline, 3.8x training-accel claim) rides along ONLY if the async set closes early

</details>

---

**`lit-radar-hooks-17`** · `cpu`

Lit slice (standing allocation): clear the two unread #17 radar hooks

**boundary:** EXECUTED same session as queued (attach_K train window)

<details><summary>full record</summary>

Lit slice (standing allocation): clear the two unread #17 radar hooks — VEGA 2605.10485 (encoder grounding alignment) + HyperVLA 2510.04898 (hypernetwork inference) — papers page(s) same session per the permanent rule; both feed the vision-unfreeze finalization amendment's citation set (VLM4VLA prior now banked) and the #17 trunk ledger. — EXECUTED 11:4x-12:0xZ 08-09 (same session as queued): both papers deep-read + 2 papers pages SAME SESSION (papers/vega-encoder-grounding.md, papers/hypervla-hypernetwork-inference.md). VEGA 2605.10485: encoder-output alignment aux to DINOv2-FiT3D (2407.20229), projector discarded at inference, beats Spatial-Forcing LLM-token alignment on RoboTwin easy+hard (67.5/30.7 vs 64.2/27.8) + real ALOHA 0.60 vs 0.55; frozen-FiT3D~unfrozen probe =&gt; THIRD POLE on the freeze axis: aux-injected structure substitutes for unfreezing =&gt; banked as vu5k interpretation lever + named cheap escalation if thawed wins (Molmo2 single-tower caveat; VGGT-teacher collapse 0.04 hard). HyperVLA 2510.04898: understand-once/execute-tiny pole (0.1M generated policy/episode, 4ms/step, 90x fewer activated params, sim-only vs 2024 OpenVLA) =&gt; #17 trunk ledger + #16 rig-latency existence proof + sqrt-d generated-update normalization rule (OOD-specific failure); MSE-beats-diffusion ablation regime-bound, NOT read onto AR-vs-flow. index/SUMMARY/ideas #17 hooks + idea page ledger updated. NEW radar hook banked: Spatial Forcing 2510.12276 (ICLR'26; 3.8x training-accel claim unexamined)

</details>

---

**`lit-unfreeze-schedules`** · `cpu`

Lit slice (standing allocation, owner steering 2026-08-09 10:38Z): the owner reframed F-vs-K as schedule curves a(t)*AR + b(t)*flow under FIXED COMPUTE

**boundary:** DONE same session ~11:07Z 08-09, 2 papers pages landed: lpft-two-phase-schedules.md (LP-FT 2202.10054 + NTK 2405.16747 — the f-then-joint rung's THIRD citation, first with matched frozen control + feature-distortion theorem; compute-Pareto case for the step-function a(t); silent on F-vs-K since K's stop-grad blocks the distortion channel) + vlm4vla-trunk-ablation.md (2601.03309 — 9-trunk sweep: frozen vision encoder loses uniformly =&gt; external prior for #17 thawed arm; VQA-&gt;control proxy collapse off-Calvin =&gt; trunk swaps priced by panel screens only; NOT compute-matched, caveat loud). index/SUMMARY/ideas #4+#17 hooks updated. Open question fed forward: nobody in the lineage measures the schedule family compute-matched — that comparison is ours if the a&gt;0 region opens.

<details><summary>full record</summary>

Lit slice (standing allocation, owner steering 2026-08-09 10:38Z): the owner reframed F-vs-K as schedule curves a(t)*AR + b(t)*flow under FIXED COMPUTE — sweep for compute-matched frozen-vs-joint / unfreezing-schedule evidence in VLA + adjacent transfer literature (progressive unfreezing, staged joint training, compute-matched fine-tuning ablations). Papers page(s) same session per the permanent rule; feed #4 f-then-joint rung pricing + the post-Delta_seam compute-matched follow-up pre-reg if the a&gt;0 region opens.

</details>

---

**`attach-seam-readout-audit`** · `cpu`

#4 Delta_seam readout PRE-AUDIT (CPU, before the ~18:3xZ K endpoint; the audit-queue-items-against-git standing rule applied to tonight's frozen read): (1) re-run attach_seam_results.py --oracle at HEAD; (2) verify its default st…

**boundary:** DONE ~11:00Z 08-09 same session it was queued: oracle re-run green at HEAD (all 5 branches); default stems cross-checked against the box (F arm files present under the EXACT expected names; K launcher writes matching names, %06d padding verified at launcher line 150; ar_view stem matches; endpoint 60k k4l2 json already local); dry-run confirms the pre-endpoint abort is the clean 'arms not rsynced yet?' branch; 3-step runbook (2 rsyncs + one command) staged into the attach_K babysit anchors. Nothing in the frozen read was touched.

<details><summary>full record</summary>

#4 Delta_seam readout PRE-AUDIT (CPU, before the ~18:3xZ K endpoint; the audit-queue-items-against-git standing rule applied to tonight's frozen read): (1) re-run attach_seam_results.py --oracle at HEAD; (2) verify its default stems/paths against what unit fontaine-attach-k actually writes on the box (panel_v2 eval json/npz names, AR-view materialization output, F-side banked 08:01Z artifacts) — name drift here would stall the readout at the endpoint; (3) stage the exact one-command invocation + expected abort-grade branches into the babysit boundary so the readout is copy-paste at endpoint. Record-only prep; the read itself stays frozen as landed 08-07.

</details>

---

**`idea6-mcselect-postmortem`** · `cpu`

#6 rung-(c) post-mortem read (CPU, record-only, exploratory — NOT pre-registered, no decision rides on it; the selection-ceiling read precedent): from the banked mcselect npz (mcselect:kl [N,C] + mcselect:cand_pred [N,C,S,D]) com…

**boundary:** READ OUT same session 2026-08-09T10:55:00Z (commit to follow): mcselect_postmortem.py (oracle-gated: planted monotone fixture exact hand arithmetic incl. tie-rank + constant-KL exclusion, 6 abort branches: degenerate &lt;2-eligible row, finite-KL-at-ineligible, NaN-at-eligible, partial dump, missing key, mixed vocab) -&gt; analysis__subgoal_mcselect_postmortem_q4_ar100k_k4l2.json + raw sidecar npz. MAP: KL is rank-NOISE (per-row Spearman(KL,err) +0.012 [-0.005,+0.029]; oracle-best uniform on the axis 0.498 vs 0.5, excess at BOTH extremes -&gt; argmin fails too; harm is magnitude-driven, value-level rho +0.126, winner's curse); SC was the better axis (-0.030, CI&lt;0, oracle-best at top 30.1% vs 12.6% null) but ~6x too weak for argmax; axes mutually uncorrelated (+0.032) -&gt; family failed twice INDEPENDENTLY. Calibration bar for any learned-verifier pre-reg: zero-training rank signal tops at |rho|~0.03 toward a real -0.250 ceiling. Addendum + 2 dark-mode charts on the results post; ideas.md #6 + idea page ledger updated. #6 escalation remains CLOSED.

<details><summary>full record</summary>

#6 rung-(c) post-mortem read (CPU, record-only, exploratory — NOT pre-registered, no decision rides on it; the selection-ceiling read precedent): from the banked mcselect npz (mcselect:kl [N,C] + mcselect:cand_pred [N,C,S,D]) compute (1) per-candidate KL-vs-frame-error correlation (pooled + per-row rank), (2) the oracle-best candidate's KL-rank histogram (where on the informativeness axis do the good candidates sit), (3) the same for SC's banked mean_logprob axis from the candidates file — a 2-axis map of the closed family's failure, to be read BEFORE anyone prices a learned verifier (RoVer-style supervised vs set-joint label-free). AUDIT FIRST per standing rule: mcselect_results.py owns argmax/tie/eligibility — reuse its loaders/eligible_list, extend only the correlation delta. Oracle: planted monotone-KL fixture (known rank order in/out) + degenerate C=1 row must abort. Output: one analysis json + a short post section (or appended to the results post as a dated addendum), NO deployment claim

</details>

---

**`actckpt-lineage-flip-prereg`** · `cpu`

#20 activation-checkpointing lineage-flip pre-reg DRAFT (CPU; unblocked 08-09 by the sdpa-pin fix 913fdc4 + the live K-smoke validation of the flag on CUDA): the perf review's ~2.4-2.8 GiB/sample memory lever for TRAINING lineage…

**boundary:** DRAFT LANDED 05:1xZ 08-09; finalize + execute per actckpt-lineage-flip-ladder · [pre-reg](posts/2026-08-09-prereg-actckpt-lineage-flip.md)

<details><summary>full record</summary>

#20 activation-checkpointing lineage-flip pre-reg DRAFT (CPU; unblocked 08-09 by the sdpa-pin fix 913fdc4 + the live K-smoke validation of the flag on CUDA): the perf review's ~2.4-2.8 GiB/sample memory lever for TRAINING lineages that don't currently carry --activation-checkpointing. Pre-reg must pin: which lineage flips first (next fresh molmo2 train launch, never a live run), the batch/chunk re-tune the freed memory buys (review projected chunked-backward passes could drop), before/after bench protocol + the bitwise keystone re-gate on the target recipe, and the decision rule for adopting. NOTE the K attach arm already carries the flag by pre-reg — this item is about OTHER lineages (e.g. a future 100k continuation or arch-batch arms) — DRAFT LANDED 05:1xZ 08-09 (2026-08-09-prereg-actckpt-lineage-flip.md, attach_F train window): 4-rung box ladder (control / ckpt-c6 / ckpt-c1 candidate / record-only max-B bisect), scope pinned perf-only (eff-48 + B12 frozen; batch-headroom spend named OUT of scope for a future science pre-reg), decision rule frozen (ADOPT iff r2 &lt;= 1.02*r0 AND alloc peak &lt;= 63 GiB), gate &lt;= 2 GPU-h. Execution split to actckpt-lineage-flip-ladder (blocked: needs a scheduled fresh non-attach AR-trunk launch + post-attach box window; finalization stamp re-pins baselines at then-HEAD)

</details>

---

**`idea6-subgoal-swap-read`** · `gpu-local`

#6 subgoal-swap content read (pre-reg 2026-08-09-prereg-subgoal-swap.md, posted this session): re-run the rung-(a) oracle arm with an episode-level seeded derangement of segment labels (format-valid, content-wrong)

**boundary:** CLOSED 03:5xZ 08-09: swap arm rc=0 03:42:36Z (~1.5 GPU-h &lt;= 3 gate); dump oracles i+iv GREEN (25,788/25,788 swapped, 0 empty, 0 skipped; 2,162 textual coincidences recorded). Frozen reads banked analysis__subgoal_swap_ar100k_k4l2.json (execution oracles green): Delta_swap -0.113 [-0.161,-0.060] (wrong words HELP), swap-vs-oracle +0.166 [+0.127,+0.205] (truth clearly better), horizon last10 swap -0.175 vs oracle -0.480 (banked -0.464 signature reproduced). Frozen 3-row table: MIXED — record-only per pre-reg, no decision row fires; reading = ~40% format/prior floor + ~60% content margin of the -0.290 bound. Scorer escalations stay coherent but their prize is the ~0.17 content margin over a free ~0.11 any-words floor. Results post 2026-08-09-subgoal-swap-results.md + chart. · [pre-reg](posts/2026-08-09-prereg-subgoal-swap.md)

<details><summary>full record</summary>

#6 subgoal-swap content read (pre-reg 2026-08-09-prereg-subgoal-swap.md, posted this session): re-run the rung-(a) oracle arm with an episode-level seeded derangement of segment labels (format-valid, content-wrong) — closes the presence(-0.290)/channel(+0.043)/CONTENT triangle; frozen 3-row interpretation table decides whether learned-scorer escalations are even coherent (swap~0 -&gt; coherent; swap~oracle -&gt; format mirage, deprioritize toward future-latent family; swap&gt;0 -&gt; strongest pro-scorer case). PREREQUISITE instrument delta: --subgoal-swap-seed on the oracle path + 4 oracles (derangement fixture bijective/no-identity, identity-map byte-reproduces banked oracle arm, label-less frames byte-match baseline, dumped text == source-episode label). ~1.2 GPU-h projected &lt;= 3 gate, local 1xH100 at any quiet window | IMPLEMENTATION AUDIT BANKED 01:4xZ same session (mapping rule pinned in the pre-reg first): label source = per-dataset meta/judge_annotations.json (metadata-only, no image decode — the swap-map builder is a pure-CPU pre-pass); interception point = the condition_subgoal item override the SelfSubgoalPolicy pass-2 path already uses (collator honors explicit override over the frame's true label; empty means no-hint, never falls through); fraction needs episode duration from LeRobot meta + item timestamp. Estimated ~150 lines + fixture tests; oracle (ii) identity-map byte-reproduction is a launcher-side pre-launch check vs the banked oracle npz, not a check.py test | INSTRUMENT LANDED + LAUNCHED 02:13Z 08-09 same session: bijou/eval/subgoal_swap.py (map builder: judgments sidecar under the stamp, materialize-exact span semantics, per-repo Sattolo derangement) + BijouPolicy _swapsubgoal/_swapidentity wiring + CLI --subgoal-swap-seed/--subgoal-swap-identity/--dump-subgoal-swaps; 16 fixture oracles in tests/test_subgoal_swap.py, check.py 554 green; launcher eval_ar100k_subgoal_swap_arm.sh (4 phases: selftest -&gt; IDENTITY full panel -&gt; oracle-(ii) byte-reproduction vs banked oracle npz abort-on-red -&gt; swap arm -&gt; mechanical dump check) LIVE unit fontaine-subgoal-swap, babysit entry active, ~2.4 GPU-h (identity+swap) &lt;= 3 gate

</details>

---

**`framemining-perpair-figures`** · `cpu`

OWNER STEERING 08-08 16:20-16:22Z: rework the frame-mining report's contact sheet into one figure per mined pair

**boundary:** closed 2026-08-08 16:4xZ

<details><summary>full record</summary>

OWNER STEERING 08-08 16:20-16:22Z: rework the frame-mining report's contact sheet into one figure per mined pair — 3 panels per row (query image, neighbor image, action-chunk chart w/ both ground-truth trajectories overlaid), all 12 pairs sequential w/ captions, each frame's SUBGOAL label included — EXECUTED same session ~16:2x-16:3xZ (caught at the 16:25Z babysit poll, ~5 min latency; ack + delivery posted in-channel): frame_mining.py `figures` subcommand (house palette, alignment guard flagged-npz vs panel rows, subgoal_text per frame), 12 pair_NN.png + pair_figures.md captions snippet inlined into the post, contact sheet retired from the post (file kept banked); blog built + Space pushed, post 200 + image bytes verified

</details>

---

**`fieldcond-subgoal-meta-report`** · `cpu`

OWNER STEERING 08-08 13:21Z: consolidated chart-led meta-report on field conditioning + ALL aux-subgoal idea work (title must NOT say 'visual report'

**boundary:** closed 2026-08-09; owner-review window open — escalation picks in section 5/6 each need their own pre-reg

<details><summary>full record</summary>

OWNER STEERING 08-08 13:21Z: consolidated chart-led meta-report on field conditioning + ALL aux-subgoal idea work (title must NOT say 'visual report' — charts/visual aids are the default treatment, per standing preference): synthesize the fieldgen/field-conditioning thread + subgoal-conditioning ideas (#6 rungs, selfsubgoal probe, goldenticket lineage where it feeds subgoals) into one page; MUST include specific interesting episode frames comparing the effect of subgoal conditioning — prioritize frames where the right action is ambiguous from the image alone (e.g. start-vs-end of episode indistinguishable, goal not visible from the parked position); values/frames from banked reports + banked episode data only | FRAME-MINING PROTOCOL PINNED (lit 08-08, papers/observation-aliasing.md, 2605.14712's aliasing diagnostic run in reverse): embed episode frames with a frozen on-disk vision tower, NN-retrieve, flag frames close in embedding but DIVERGENT in ground-truth continuation (action-chunk distance / embedding distance ranking) — these are the ambiguous frames the owner asked for, found automatically; central chart = does the per-frame subgoal-conditioning delta (conditioned vs subgoal-dropped) CONCENTRATE on flagged frames (concentration = disambiguation mechanism, published shape 9%-&gt;45.8%; no concentration = style/dataset prior — either way a claim, not an anecdote) | FRAME-MINING EXECUTED 08-08 15:5xZ (fontaine/scripts/frame_mining.py, post 2026-08-08-framemining-aliased-frames.md): 17,204 core frames embedded with the frozen Gemma-4 E2B tower (AR-100k's own frozen eye, alignment oracle every row), within-dataset NN mining banked (analysis__framemining_ar100k_k4l2.json + flagged npz + contact sheet of 12 mined pairs). CENTRAL READ IS A NULL: flagged-vs-rest Delta_oracle -0.003 [CI -0.205,+0.176], Spearman -0.01/14,064 frames — the subgoal gain is FLAT across aliasing (except near-zero on the least-aliased decile, -0.04 post-hoc); the report's story becomes 'uniform prior/guidance, not disambiguation', with the +29% aliased-frame error floor (miner validated, rho 0.41 vs baseline MAE) as the history-arm prize (#11). REMAINING for the report: compose with fields-panel numbers post-23Z + banked report values; charts + contact sheet already in blog img/framemining/ | STRUCTURE DRAFTED 08-08 19:4xZ (fontaine/drafts/fieldcond-subgoal-meta-report-structure.md): 6-section skeleton + full banked-artifact map (inventory swept), title candidate pinned, chart list split new-render vs reuse; open slots marked for the two pending inputs (fields-panel 60k half, (b') stage-2 verdict). Composition remains post-fields-panel | COMPOSED + LANDED 2026-08-09 01:3xZ (posts/2026-08-09-fieldcond-subgoal-report.md, 'Conditioning on words: what the subgoal channel actually buys'): 6-section chart-led page per the drafted skeleton — S1 aux +0.462 load-bearing, S2 rung-(a) oracle -0.290/6x-late vs self -0.018, S3 fields tables BOTH trunks (visible 0.319-&gt;0.819 headline, narration-cost sign consistent), S4 four mined ambiguous-frame figures (start-vs-end, which-pen, phase, drawing) + the honest null stated loudly (gain does NOT concentrate, rho -0.01; +29% error floor = the #11 prize), S5 selection ladder (b) table-cost close + (b') NO-SCORER with priced 4-way escalation table (RoVer-supervised / uPRM set-joint / jerk priced-out / history-phase), S6 three pre-named open questions. All values from banked analysis jsons; draft skeleton file deleted per its own note

</details>

---

**`idea1-noise-ladder-rung2-execution`** · `gpu-local`

Noise-ladder rung 2 EXECUTION (idea #1, gpu): instrument the per-dataset routing mode (--noise-ticket-map: BijouPolicy._flow_noise substitution keyed on item repo_id; policy suffix _ticketmap; ticket_map_sha256 in npz+report prov…

**boundary:** CLOSED 2026-08-08 ~23:1xZ: stage-2 falsified + seating confirmed/adjudicated; babysit seating entry retired 22:2xZ · [pre-reg](posts/2026-08-08-prereg-noise-ladder-perdataset.md)

<details><summary>full record</summary>

Noise-ladder rung 2 EXECUTION (idea #1, gpu): instrument the per-dataset routing mode (--noise-ticket-map: BijouPolicy._flow_noise substitution keyed on item repo_id; policy suffix _ticketmap; ticket_map_sha256 in npz+report provenance) + preflight byte-match oracle (routed decode == plain ticket-t decode on a small 2-dataset plan, matched composition), THEN stage 2 confirm eval (~0.9 GPU-h: full panel, map sha 15d92935..., reads 1-5 per the pre-reg: primary D_route map-vs-33 on qualifying complement core rows, dataset-clustered bootstrap CI95 seed 0) + the folded R3 seating arm (~3.0 GPU-h: random-noise draws-10 re-run with --dump-predictions retained; base-equality oracle pooled 5.3645 at 4dp; paired mean-of-top-10 vs mean-of-random-10). Ceiling &lt;= 4 GPU-h, local GPU, every launch via run_detached.sh + babysit.toml entries at launch. | INSTRUMENT + PREFLIGHT LANDED 08-08 16:2x-16:4xZ work session (CPU-side clause exercised): --noise-ticket-map routing mode in bijou.eval (BijouPolicy._flow_noise per-item routing, _ticketmap policy suffix, sample_draws==1 enforced, unmapped-dataset hard abort; report + predictions-npz provenance carry bank sha AND ticket_map_sha256 — predictions dump gained ticket provenance for all ticket modes), committed map loads from the stage-01 analysis json with canonical-form sha reproducing 15d92935... exactly (tests/test_ticket_map.py 14 oracles); preflight apparatus: committed 2-dataset ticket-2 plan (144 rows, f23d70ab...) + t2-only bank (= m64[2:3] byte-verified, abfaf064...) + noise_ladder_preflight_oracles.py (selftest green: 1 green + 4 red synthetic worlds) + 3 launchers (preflight / stage2 gated on green json / seating w/ --noise-key index — the banked 5.3645 row PREDATES --noise-key so the base-equality oracle needs the historical index keying, header documents) + prepared babysit entries. PREFLIGHT RUNNING 16:26Z unit fontaine-noiseladder-preflight (run_detached, babysit entry live, ~25 min) | AMENDMENT 1 16:4xZ (caught by the preflight adjudicator's FIRST REAL RUN — the map-coverage oracle working as designed): committed map enumerates the probe universe (792 datasets); the panel plan decodes 86 more w/ zero probe rows. Fix keeps the selection byte-intact: plans/noise_ladder_ticketmap_panel.json = 792 committed routes verbatim + 86 added -&gt; 33 (the pre-reg's non-qualifying fallback rule), canonical sha 27858421...; adjudicator enforces restriction == pre-registered 15d92935... exactly + added image == {33} + core+labeled coverage (selftest 5 red worlds incl. restriction-drift); amendment section posted on the pre-reg BEFORE stage 2; both launchers repointed + sha-pinned. NO read changes (qualifying set within the 792). Preflight RELAUNCHED 16:43Z w/ the extended map; also hardened: adjudicator now byte-matches state-copy columns and filters pred:bijou keys (first-run fixture blindness to the state-copy columns fixed + regression fixtures added) | PREFLIGHT GREEN 08-08 16:5xZ: relaunch rc=0, adjudicator ALL GREEN (144 rows routed==plain byte-match; restriction == 15d92935... exact, extended map 27858421..., 86 added datasets, t2 bank abfaf064...); green json reports/analysis__noise_ladder_preflight_oracles.json written = the stage-2 launcher's gate armed; unit exited, local GPU free; babysit entry pruned | FROZEN-READ SCRIPT LANDED 08-08 17:2xZ (the remaining CPU cell before stage-2): noise_ladder_rung2_results.py — reads 1-5 exactly per the pre-reg + amendment 1: primary D_route routed-vs-33 on qualifying complement core rows w/ DATASET-CLUSTERED bootstrap CI95 (seed 0, 10k; resample unit = dataset, the pre-reg's clustering clause); D vs stable-key record-only; per-dataset win table w/ exact two-sided sign test; horizon + R4b dispersion-quartile mirrors (dispersion pinned: top-10-restricted stage-1 probe stack, per-dataset mean — complement rows carry no draw stack by construction); execution oracles all abort-gated (map shas committed 15d92935/extended 27858421 + restriction byte-identity, _ticketmap policy, sample_draws==1, identity+state-copy byte-match across all 3 panels, rows-mapped-to-33 byte-match banked ticket33, qualifying complement == committed 6014). Oracle mode GREEN pre-data: banked reproductions (5.6524/6.6750 full-panel, 14746/6014 complements), planted worlds (exact -0.1 CI degenerate; leakage killed; clustered-CI-binds world: cluster CI [-0.75,0.75] vs frame [-0.14,0.14]; sign-test exact p arithmetic), 11 refusal branches each firing at its OWN check (fixture shas made consistent so structure oracles fire, not the sha gate), R4b planted geometry Spearman -1 monotone. check.py 515 green. Stage-2 launcher now CHAINS the adjudicator at rc=0 — the post-close window is one command via run_detached | STAGE-2 EXECUTED + READ OUT 08-08 19:2x-19:4xZ (launched 18:34Z local, owner cleared the credit-cap wait 18:31Z; ~0.83 GPU-h): FALSIFIED — read-1 primary Delta_route +0.129 CI95 [+0.060,+0.205] entirely ABOVE zero on the 6,014 held-out complement core rows; win table 34W/54L/9T sign p 0.042; Spearman(dispersion, delta) -0.05; routed-vs-stablekey -0.756 re-confirms the shared-ticket effect (board row stays global ticket 33). In-sample probe delta -0.60 INVERTED out-of-sample = per-dataset argmin memorizes its ~6-20-frame cell. Results post + 2 dark charts: posts/2026-08-08-noiseladder-rung2-results.md. Record-only lead: routing wins chunk steps ~1-8, loses ~15+ (chunk-position noise policy = different axis, needs its own pre-reg). SEATING ARM launched 19:25:16Z at stage-2 rc=0 (unit fontaine-noiseladder-seating, ~3.0 GPU-h, ETA ~22:25Z): base-equality oracle (pooled 5.3645 at 4dp) gates the paired mean-of-top-10 vs mean-of-random-10 read at rc=0 | SEATING ADJUDICATED 08-08 ~23:1xZ chained work session: rc=0 22:25Z (~3.0 GPU-h &lt;= 5.17 gate), base-equality oracle FIRED (first_mae -1.27e-4 across 4dp) -&gt; held, diagnosed, Amendment 2 posted BEFORE any gate change: benign numeric drift from the batched-ensembling merge 2ee2be5/85cdc0a (state-copy cells exact 878/878, bijou cells &lt;=1.7e-3 vs draw-dispersion 0.05-0.5 = resampling EXCLUDED, --noise-key index reproduction CONFIRMED; committed seating_base_equality_diag.py + analysis json; amended gate = state-copy exact + pooled 5e-4 + cells 5e-3, tests updated, launcher oracle now runs the diag script). FROZEN READ: paired Delta -0.17358 [CI95 -0.19556, -0.15214] entirely below 0 (clustered [-0.20188, -0.14756] agrees; first mirror -0.041) = EXPECTATION 4 CONFIRMED — board row MOVED to mean-of-top-10-tickets 5.1847/1.3831 (leaderboard row 2, best chunk+first on the board, star-gap 0.37-&gt;0.18); results post seating section + idea-01 ledger + analysis__noise_ladder_seating.json banked. Noise-ladder rung 2 FULLY CLOSED.

</details>

---

**`fieldgen-accuracy-eval`** · `gpu-box`

OWNER 10:08Z 08-08 accuracy-by-field — PREP DONE + AR-100k half CLOSED 11:1xZ same day (2f4d575 + pre-reg note 2026-08-08-prereg-accuracy-by-field.md): (1) CORRECTION

**boundary:** closed 2026-08-09 00:49Z; babysit entry pruned same session; last pending input for fieldcond-subgoal-meta-report now banked · [pre-reg](posts/2026-08-08-prereg-accuracy-by-field.md)

<details><summary>full record</summary>

OWNER 10:08Z 08-08 accuracy-by-field — PREP DONE + AR-100k half CLOSED 11:1xZ same day (2f4d575 + pre-reg note 2026-08-08-prereg-accuracy-by-field.md): (1) CORRECTION — the AR-100k banked greedy panels ALREADY carry the table (narrated +fields arm rides automatically on aux-trained gemma checkpoints): holding 0.807 / progress MAE 0.062 / event 0.878 / visible 0.319 on panel_k4l2 (~9k judge-labeled frames); the queued ~1-2 GPU-h local run is CANCELLED as redundant. (2) molmo2's missing table ROOT-CAUSED: BijouPolicy gated the narrated pass on the Gemma concrete (isinstance ARBackboneDecoder); Molmo2ARDecoder is a sibling of ARSuffixDecoder, so aux-trained molmo2 checkpoints silently reported no fields — FIXED 2f4d575 (gate on the scaffold; prompt bytes unchanged on every banked read, generate_bracket=True recorded at save; 2 CPU regression tests incl. a real narrated decode on the tiny molmo2 fixture; check.py 500). (3) REMAINING = the one registered run: molmo2 60k fields panel, box 4xDDP, eval_box_molmo2_60k_fields_panel.sh via run_detached (launcher self-guards: post-fix checkout via grep, chained eval json present, GPUs free, plan sha; mechanized read-3 base-equality oracle vs the chained json + accuracy-block presence + narration-delta print); ~3.5 GPU-h &lt;= 6 gate; prepared babysit entry molmo2_60k_fields at babysit.toml bottom. NOTE: tonight's chained 60k eval runs the box checkout AS LAUNCHED (charter: never sync box code under a live run) — narrated-arm-free and byte-comparable to the 40k panel, which the paired read wants; the fields run needs refresh_ctrl.sh AFTER the 60k chain completes | EXECUTED + READ OUT 2026-08-09 (launched 00:03Z prior session, rc=0 00:49:43Z, ~3.1 GPU-h &lt;= 6 gate): all mechanized reads green — read-3 base-equality EXACT (bijou@60000 5.86022663460471 == chained json), accuracy table molmo2@60k holding 0.897 / progress MAE 0.059 / event 0.880 / VISIBLE SLOT-SET 0.819 vs AR-100k anchor 0.319 (+0.50 on the strictest metric, MORE frames parsed 8981 vs 8260 — no parse-selection excuse); narration delta +0.0865 (paired +0.083, win 44%; anchor +0.054), cost concentrated on failure-labeled frames (+0.50 vs +0.09 success). Results post 2026-08-09-molmo2-fields-panel-results.md + fields_accuracy.svg chart (dark theme, meta-report section-3 slot); ideas 06/17 ledger entries. Anchor robustness: curated_v0 AR-100k panel agrees &lt;=0.007 every field

</details>

---

**`idea6-subgoal-draws-cleancand-execution`** · `gpu-local`

#6 rung (b') clean-list subgoal-draws EXECUTION (gpu-local, ~2.5-3.5 GPU-h, ceiling &lt;= 5): per the posted pre-reg (2026-08-08-prereg-subgoal-draws-cleanlist.md) -- (1) instrument delta oracle-gated BEFORE launch: eligible-list ru…

**boundary:** CLOSED 2026-08-09 00:2xZ: run complete 23:52Z 08-08 (q4 fallback 4301 rows, ~1.4 GPU-h &lt;= 5 gate); subset-join read path landed in subgoal_draws_results.py (draws10/energy precedent, q4 slice fixture oracle-green) and frozen reads EXECUTED — E6 FALSIFIED (bon-self +0.210 [+0.113,+0.312] entirely above 0; Delta_bon +0.142 vs bare baseline = anti-selection), adjudication NO-SCORER (Delta_ceil -0.250 [-0.353,-0.148] alive, late-horizon -0.464); analysis banked reports/analysis__subgoal_draws_cleanlist_q4_ar100k_k4l2.json, results post 2026-08-09-subgoal-draws-cleanlist-results.md; selection family closed on scorer-free tricks, scorer-side escalations need own pre-reg · [pre-reg](posts/2026-08-08-prereg-subgoal-draws-cleanlist.md)

<details><summary>full record</summary>

#6 rung (b') clean-list subgoal-draws EXECUTION (gpu-local, ~2.5-3.5 GPU-h, ceiling &lt;= 5): per the posted pre-reg (2026-08-08-prereg-subgoal-draws-cleanlist.md) -- (1) instrument delta oracle-gated BEFORE launch: eligible-list rule in SelectedSubgoalPolicy._pick + offline recomputes + dump eligible flags/filtered picks + filter-aware read script; oracles = rung-(b) i-vi inherited (draws-0 limit: eligible list == [greedy], bit-exact carry) + vii banked-table pick-invariance regression fixture (0/60 SC + 0/60 ceiling on the real stage-1 json) + viii planted filter-binds world (full-list argmax IS truncated -&gt; filtered pick differs, both scorers) + ix all-truncated -&gt; greedy fallback recorded + x stage-1 re-adjudication script reproduces the written priors exactly (60/60, 57/60, 23/425, 0/60+0/60); (2) stage 1 CPU re-adjudication on the banked table gates stage 2 (bars a' &gt;=90% / b' &gt;=50% / c' &lt;=50% / d eyes; a failed bar = instrument breakage, abort loudly); (3) stage 2: full-panel pass 1 (9 candidates, one shared prefill) + bon (SC over eligible) + ceil (token-F1 over eligible) arms, plan/checkpoint/seed/composition verbatim rung (b), distinct policy stems carrying the filter id; (4) frozen reads 1-6 verbatim via the filter-aware read script: primary Delta_bon vs 5.8026 + paired (bon - self) vs the banked rung-(a) self npz, falsified unless CI95 entirely below 0; Delta_ceil adjudicates no-diversity vs no-scorer and routes the escalation item. q4 fallback on a first-200-frame rate projection past ceiling. Launch via run_detached.sh + babysit.toml entry at launch; first-poll util+rate check | INSTRUMENT DELTA LANDED 08-08 18:1xZ: eligible-list rule in subgoal_scoring.eligible_indices + SelectedSubgoalPolicy (candidate_filter='clean', names _boncleansubgoal/_ceilcleansubgoal), CLI --subgoal-candidate-filter clean (dump gains eligible flags + fallback + filtered alternates), read script subgoal_draws_results.py --candidate-filter clean (filter provenance + eligible/fallback recompute aborts + eligible-size/fallback records), live oracles --candidate-filter clean (draws-0 inert). Oracles vii-x ALL GREEN: pytest 30/30 (planted filter-binds both scorers, all-truncated fallback, draws-0 limit), stage-1 re-adjudication script reproduces every prior exactly (40/60 binds, 0/60+0/60 pick changes, 60/60, 57/60, 23/425) -&gt; STAGE-1 GATE OPEN (reports/analysis__subgoal_draws_cleanlist_stage1.json written). Remaining: stage-2 launch only (launcher + babysit entry at launch, post-close window behind rung-2) | LAUNCHER LANDED 08-08 20:1xZ (audit: the rung-(b) launcher gates on (b)'s FAILED go marker + carries no filter flag, so 'launch-only' was untrue until now): eval_ar100k_subgoal_draws_cleancand_arms.sh — gates preflight GREEN + (b') stage2_gate OPEN (the (b) marker deliberately not consulted), --subgoal-candidate-filter clean, stems _subgoalcleandraws (read-script convention), rate gate 5.0 GPU-h with the q4 fallback clause verbatim; bash -n + CLI flag + plan shas verified; babysit PREPARED entry appended | LAUNCHED 22:26:41Z 08-08 at seating rc=0 (unit fontaine-subgoal-cleancand, launcher gates green). RATE-GATE Q4 FALLBACK TAKEN 22:37Z (full panel projected past 5 GPU-h at ~200 frames) — live run = q4 subset 4301 rows, stems stateprobe_q4_subgoalcleandraws, projection ~2.3 GPU-h &lt;= 5.5 backstop. INCIDENT 22:26-22:41Z: the fallback's kill hit only the run_arms subshell — the full-panel eval survived and ran beside the q4 relaunch until the chained session TERM'd it by PID (caught at first babysit exit 3); FIX LANDED both subgoal-draws launchers: pkill by 'bijou[.]eval.*stem' pattern (self-match-safe per the babysit lesson) + poll + KILL escalation; babysit cleancand entry updated (q4 boundary + incident anchors)

</details>

---

**`idea6-subgoal-draws-cleancand-prereg-draft`** · `cpu`

#6 rung (b') escalation pre-reg DRAFT (CPU): truncation-robust candidate list

**boundary:** POSTED 08-08 17:5xZ; execution item carries the launch · [pre-reg](posts/2026-08-08-prereg-subgoal-draws-cleanlist.md)

<details><summary>full record</summary>

#6 rung (b') escalation pre-reg DRAFT (CPU): truncation-robust candidate list — identical rung-(b) design except budget-truncated candidates are EXCLUDED from the scorer's candidate list (fallback greedy when all 8 sampled derail) OR nucleus/lower-T sampling; written priors = the stage-1 close (11.5% T=1 derailment, SC median-rank-last on truncated 0/60 picks, diversity 97%, pick!=greedy 65%); instrument delta small (filter in SelectedSubgoalPolicy candidate list + oracle: exclusion changes picks on 0/60 stage-1 rows — structural not behavioral on observed data); preflight apparatus (live oracles, matched-composition) landed green and reusable; stage-1 bars re-run with the same (a) bar now scoring the FILTERED list | POSTED 08-08 17:5xZ work session (2026-08-08-prereg-subgoal-draws-cleanlist.md): rung (b) inherited verbatim except the frozen eligible-list rule (truncated==false; empty -&gt; greedy fallback recorded); nucleus/lower-T rejected with reasons banked. Priors VERIFIED on the banked stage-1 table before freezing: filter changes 0/60 SC picks AND 0/60 ceiling picks (both scorers checked, not just SC; 40/60 rows carry &gt;=1 truncated candidate); filtered bars a' 60/60, b' 57/60 (95%), c' 5.4% -&gt; stage 1 is CPU-free banked-table re-adjudication (pass-1 byte-identity argument), stage 2 = the frozen rung-(b) arms with Delta_bon/Delta_ceil finally measured. Ceiling &lt;= 5 GPU-h (tighter than (b)'s 6), q4 fallback clause verbatim. Execution split to idea6-subgoal-draws-cleancand-execution

</details>

---

**`discord-reply-reference-parsing`** · `cpu`

Harness fix (owner question 09:22Z): discord.py read/history do NOT surface native reply references (message_reference)

**boundary:** closed 2026-08-08 10:2xZ

<details><summary>full record</summary>

Harness fix (owner question 09:22Z): discord.py read/history do NOT surface native reply references (message_reference) — quoted-message context drops. Add referenced-message rendering (author + first ~120 chars, '↳ replying to ...') to read + history; oracle: fixture message dicts w/ and w/o reference render stably; also surface edits (edited_timestamp) so 'Edited message above' events carry content — LANDED 10:2xZ 08-08: _print_messages renders '↳ replying to &lt;author&gt;: &lt;120-char snippet&gt;' from referenced_message (deleted-reference placeholder incl.) + '(edited)' marker from edited_timestamp; 5 rendering oracles in tests/test_discord_render.py (plain/reply/truncate-flatten/deleted/edited); live-verified against the channel. REST-polling limitation stated: an edit creates no new message so read's cursor won't replay it — the (edited) marker surfaces via history

</details>

---

**`chunk-mae-success-oneoff`** · `cpu`

OWNER 09:07/09:11Z: one-off record-only slice — eval/chunk_mae_success comparison e2b AR-100k vs molmo2 AR-40k (owner caveat quoted: soft judge label, not a substitute for chunk_mae; sparsity check first

**boundary:** closed 2026-08-08 10:4xZ (in-channel table)

<details><summary>full record</summary>

OWNER 09:07/09:11Z: one-off record-only slice — eval/chunk_mae_success comparison e2b AR-100k vs molmo2 AR-40k (owner caveat quoted: soft judge label, not a substitute for chunk_mae; sparsity check first — most trajectories successful?); source = train jsonl probe metric or panel success labels, whichever exists; post the table in-channel same day — EXECUTED 10:4xZ 08-08 in-channel: clean panel read (identical rows, state-copy slices byte-match) — success slice does NOT flip the ordering (molmo2 5.8773 vs e2b 5.7040, +0.173) but is molmo2's best-relative slice (deficit concentrates failure +0.521/unlabeled +0.354; molmo2's success-&gt;failure spread 1.02 vs e2b 0.67); the flattering wandb training-probe flip (5.621 vs 5.665) exists but is composition-confounded (different corpus snapshots) and was flagged as such; slice counts unlogged, count read offered

</details>

---

**`snapflow-visual-report`** · `cpu`

OWNER 09:22Z: SnapFlow consolidated visual report — same chart-led treatment as the golden-ticket page (distill trunk story: teacher band, ftrig, 1-NFE student draws/collapse, microbench cost cells); values from banked reports on…

**boundary:** closed 2026-08-08 11:3xZ

<details><summary>full record</summary>

OWNER 09:22Z: SnapFlow consolidated visual report — same chart-led treatment as the golden-ticket page (distill trunk story: teacher band, ftrig, 1-NFE student draws/collapse, microbench cost cells); values from banked reports only — DONE 08-08 11:3xZ (17fbdbe): 5-chart page 2026-08-08-snapflow-visual-report.md live on the Space (all links 200), snapflow_report_charts.py renders from the frozen jsons; posts index backfilled (7-post drift); posted in-channel

</details>

---

**`molmo2-continuation-60k`** · `cpu`

OWNER STEERING 08-08 08:49Z (discussion requested, not a launch order): molmo2 +20k continuation

**boundary:** CLOSED 2026-08-09 00:5xZ: training done 23:21Z 08-08 (step 60000, ~49 &lt;= 60 GPU-h), chained eval 23:49Z, canonical read EXECUTED via new molmo2_60k_results.py (oracle-gated) — read 1 IMPROVED paired -0.1388 [CI -0.194,-0.090] n=17204; read 2 AR-100k bar NOT passed (+0.058 chunk; first_mae 2.0719 already under 2.1431); read 4 no new probe low (6.0062@57k vs 5.91@26.5k); decision executed: attach repoint to step_060000 (amendment 3), K-smoke re-run required; leaderboard row 8 + board row; results post 2026-08-09-molmo2-60k-results.md · [pre-reg](posts/2026-08-08-prereg-molmo2-ar-60k-continuation.md)

<details><summary>full record</summary>

OWNER STEERING 08-08 08:49Z (discussion requested, not a launch order): molmo2 +20k continuation — owner proposal: --resume step_040000 to --steps 60000, --rewarmup-steps 1000, NEW data seed (fresh shuffle; rule banked in memory AND already mechanized in bijou.train check_resume_seed which hard-aborts on seed reuse). Facts for the discussion: matched-steps read says the trunk leads (40k molmo2 6.0079 vs A-s0 7.7966, paired -1.717); the -0.205 gap to AR-100k (5.8026) is a 2.5x-steps confound; probe curve 5.91@26.5k -&gt; 6.2075@40k = the cosine floor tail bought nothing (the exact published setup for re-warm + re-decay, already implemented in lr_lambda for extensions). Cost ~49 GPU-h (~12.2h box wall at 2.2 s/step). DRAFT the pre-reg after the Discord discussion converges: primary = 60k greedy panel paired vs the 40k endpoint npz + the AR-100k 5.8026 bar quoted; probe kill line vs 6.2075 sustained; epochs-seen arithmetic pinned; async saves default-on; box sequencing vs the #4 attach chain is the owner decision in flight — OWNER GO 09:04Z ('let's prio the 60k molmo2 run as you described it', caught 10:02Z after a 50-min poll-output miss); PRE-REG POSTED + LAUNCHED this session: launch_box_fontaine_molmo2_ar_60k_resume_ddp4.sh (box 4xDDP, unit fontaine-molmo2-60k via run_detached; --resume step_040000 --steps 60000 --rewarmup-steps 1000 --seed 1, all else byte-identical to the 40k launcher; cosine-over-60k restarts LR at 0.332x peak; E1 banner gate + resume-banner check; K1 probe kill 8.2075 x3 after 41.5k; ~49 GPU-h &lt;= 60 ceiling; chained 60k endpoint greedy panel w/ dumps). Attach chain requeued strictly behind it per the owner priority; frozen reads incl. paired 60k-vs-40k CI + the AR-100k 5.8026 bar + the attach-repoint decision rule

</details>

---

**`goldenticket-visual-report`** · `cpu`

OWNER STEERING 08-08 08:42Z: subsume the golden-ticket thread (pre-reg + results post + stage 1/2/3 analyses + jerk-pick + noise-ladder hooks) into ONE consolidated, chart-led visual report

**boundary:** closed 2026-08-08 09:1xZ; the standing more-visuals preference remains banked

<details><summary>full record</summary>

OWNER STEERING 08-08 08:42Z: subsume the golden-ticket thread (pre-reg + results post + stage 1/2/3 analyses + jerk-pick + noise-ladder hooks) into ONE consolidated, chart-led visual report — striking charts: per-ticket stage-1 distribution vs frozen null band (R1), R2 complement paired delta w/ CI, R3 ensemble comparison vs banked mean-of-10, R4a per-dataset argmin/containment geometry (792 datasets), R4b dispersion-quartile monotone gains, horizon curves; committed images (matplotlib -&gt; blog), reports.html + Papers/posts cross-links updated; standing preference banked in memory (more charts/visuals in posts and papers pages when it makes sense) — EXECUTED same session 09:0x-09:1xZ (143bdde): 5 SVGs + consolidated post live on the Space (200-verified), owner reacted 'Amazing! Good report' 09:22Z

</details>

---

**`idea1-noise-ladder-perdataset-prereg-draft`** · `cpu`

Noise-ladder rung 2 pre-reg — FINALIZED 08-08 13:2xZ work session (posts/2026-08-08-prereg-noise-ladder-perdataset.md, DRAFT banner dropped, immutable)

**boundary:** opens at the stage-3 R3/R4 close (~08:1xZ 08-08); draft quality gates on those numbers being in hand · [pre-reg](posts/2026-08-08-prereg-noise-ladder-perdataset.md)

<details><summary>full record</summary>

Noise-ladder rung 2 pre-reg — FINALIZED 08-08 13:2xZ work session (posts/2026-08-08-prereg-noise-ladder-perdataset.md, DRAFT banner dropped, immutable). Stage 0+1 EXECUTED on banked data (noise_ladder_stage01.py, oracles a-d GREEN, reports/analysis__noise_ladder_stage01.json): floor F=6 (n=6 bin 1.5675 vs null5 1.5965 marginal + n=7 clear; non-monotone small bins recorded honestly), 97 qualifying datasets (7,028 panel core rows = 40.8%, 6,014 complement rows), 88/97 route away from ticket 33, map sha 15d92935... committed. Instrument oracle list pinned after bijou.eval HEAD audit (_flow_noise substitution point; _ticketmap policy suffix; preflight byte-match). Expectation 1 CONFIRMED at finalization (F=6&lt;=16, 40.8%&gt;=25%).

</details>

---

**`idea19-jerkpick-selector-read`** · `cpu`

SDN jerk-pick selector, record-only ceiling-ladder read (CPU, table cost): place 'pick the smoothest draw' (RMS third-difference over the chunk, SDN 2606.14084's smoothness stage

**boundary:** closed 2026-08-08 07:3xZ · [pre-reg](papers/noise-space-steering-3.md)

<details><summary>full record</summary>

SDN jerk-pick selector, record-only ceiling-ladder read (CPU, table cost): place 'pick the smoothest draw' (RMS third-difference over the chunk, SDN 2606.14084's smoothness stage — its ablation carries most of the method's +18 pp real gain) on the banked selection-ceiling ladder single -&gt; mean-of-N -&gt; jerk-pick -&gt; oracle best-of-N, on the banked --dump-draws stacks (flow teacher drawsprobe draws10 + ticket64 stacks; molmo2 draws10_t1 full-panel stack when it lands ~08:1xZ). Pure function of the stacks — no forwards, no labels; reuses selection_ceiling_results.py pooling/ladder machinery. Record-only per the standing exploratory rule (the ceiling read precedent); a nontrivial slice of the oracle gap recovered -&gt; #19 escalation candidate with published rollout numbers behind it; nothing recovered -&gt; SDN's smoothness prior falsified for our stacks at table cost. Lit hook: papers/noise-space-steering-3.md — record-only exploratory (selection_ceiling_results.py precedent): no decision rule, no GPU; prereg field points at the lit page that sourced it — FLOW+AR HALVES EXECUTED 08-08 05:5xZ (analysis__jerkpick_selector.json): flow teacher fresh-noise draws10 = NULL (agreement 10.5% vs null 10%, Spearman +0.13, oracle-gap recovered -2.3%; ODE draws uniformly smooth, jerk carries no signal) and ticket64 = null too; AR q4 tsens = REAL BUT SMALL, monotone in T (gap recovered 5.6%/7.5%/20.9% at T=0.5/0.7/1.3, agreement 13-15%, Spearman +0.36 — jerky sampled-token draws are genuinely bad draws); jerk-pick never approaches mean-of-N on either family, so the family decodes stand. REMAINING: molmo2 draws10_t1 stack half when the #19 arm lands (~08:1xZ) — MOLMO2 HALF EXECUTED 08-08 07:3xZ: gap recovered 8.0%, Spearman +0.55 (strongest of any stack), agreement 12.1% vs 10% null — AR-family pattern confirmed on a second trunk; never approaches mean-of-N. ITEM CLOSED: flow null / AR small-but-real, family decodes stand

</details>

---

**`molmo2-decode-cost-microbench`** · `cpu`

Leaderboard integrity: bring molmo2 AR configs into the decode-cost microbench (CPU prep item

**boundary:** closed 2026-08-08 07:5xZ

<details><summary>full record</summary>

Leaderboard integrity: bring molmo2 AR configs into the decode-cost microbench (CPU prep item — the molmo2-endpoint-postprocessing row must otherwise flag its cost column as mtime-derived or leave it blank; this item retires that caveat). Work: extend the microbench harness to cover the molmo2 AR config (config plumbing + tiny-fixture dry run, CPU-verifiable), and land a one-command box script whose GPU minutes ride an already-pre-registered box eval window (the #19 draws-arm launcher's posted cost-gate umbrella, or the next posted box pre-reg) — no standalone unpre-registered GPU launch; then write the measured number into the leaderboard row + note the caveat's removal — PREP LANDED 08-08 04:4xZ: molmo2_greedy + molmo2_draws10_t1 configs in the shared harness (selftest PASS, dry-run prints both modes; dry-run no longer requires the box-resident checkpoint, real runs still abort), one-command box script microbench_box_molmo2.sh (all-GPU-free guard, run_detached launch line in header); REMAINING: run on the box at the first pre-registered eval window after the #19 chain, then merge rows into the leaderboard cost column — EXECUTED 08-08 07:27-07:50Z on the box (rode the #19 landing window, all-GPU-free guard green, unit fontaine-microbench-molmo2 rc=0): molmo2_greedy 143.8 batched / 678.1 b=1 ms, molmo2_draws10_t1 1191.2 / 6291.3 ms -&gt; leaderboard rows 8+9 cost cells filled, mtime caveat RETIRED (box-measured noted, record-only extension per prep commit)

</details>

---

**`idea6-mcselect-execution`** · `gpu-local`

#6 rung-(c) masked-contrast selection EXECUTION (gpu-local, &lt;= 4 GPU-h gate): per 2026-08-09-prereg-subgoal-mcselect.md

**boundary:** CLOSED 10:2xZ 08-09: run COMPLETE 10:20Z rc=0 (~1.1 GPU-h &lt;= 4 gate, 68 f/min steady). Live-oracle chain: my subset_rows joined on the identity triple but the BANKED full-panel baseline predates the episode/frame columns -&gt; KeyError post-run (selftest fixture carried the columns, so the branch was never exercised against the real schema); fixed to the sdr index-join convention 10:2xZ, selftest re-green, live oracles ALL ABORT-GRADE GREEN on real data (pred_masked flip count 1207/4301 == the amendment-1 composition figure exactly). FROZEN READ: ANTI-SELECT — (mc - self) +0.31317 CI95 [+0.19962, +0.42894] entirely &gt; 0 (harder strike than SC +0.210); mc vs bare +0.245; capture fraction -1.73; late-horizon +0.385 (ceiling slot inverted); oracle agreement 14.4% ~ chance at 66% active picks. KILL RULE EXECUTED: zero-training scorer family CLOSED for this trunk; learned verifiers need their own case; candidate 2 does not auto-open (trigger was flat-late-horizon, observed = active anti-selection). Results post 2026-08-09-mcselect-results.md; analysis__subgoal_mcselect_q4_ar100k_k4l2.json banked; babysit entry pruned · [pre-reg](posts/2026-08-09-prereg-subgoal-mcselect.md)

<details><summary>full record</summary>

#6 rung-(c) masked-contrast selection EXECUTION (gpu-local, &lt;= 4 GPU-h gate): per 2026-08-09-prereg-subgoal-mcselect.md — (1) INSTRUMENT first (CPU): candidates-file injection eval path + per-candidate teacher-forced logprob stacks + masked reference; oracle gates named in the draft (planted-informative fixture, tau degeneracy check, rung-(a) greedy-text byte-reproduction spot check); (2) FINALIZE the draft (immutability stamp, candidates-file sha256 pinned); (3) run 9 forwards x 4301 q4 rows local H100 (~1.5 GPU-h projected), babysit entry at launch; (4) frozen reads: (mc-self) paired CI95 primary, capture fraction vs -0.181, late-horizon signature, agreement diagnostics; anti-select = second strike closes the zero-training scorer family | READ SCRIPT LANDED 05:5xZ 08-09 (mcselect_results.py, PRE-DATA per house convention — the script IS the producer's dump contract: mcselect:kl [N,C] NaN-at-ineligible, mcselect:cand_pred [N,C,S,D], mcselect:pred_masked, report mcselect_tau==4.0 + candidates_sha256; ARGMAX + tie rule live in the reader, producer only measures): oracle PASS pre-data (planted-argmax fixture w/ exact paired arithmetic + tie rule + capture fraction, 10 abort branches incl. inert-scorer bar, finite-KL-at-truncated, partial-run, sha/tau mismatch); wrapper in check.py 559. REMAINING: the producer instrument (bijou eval path: candidates-file injection + in-model KL + per-candidate teacher-forced preds) + draft finalization + the ~1.5 GPU-h local run | DESIGN NOTE for the instrument session (caught 05:5xZ pre-build): the draft's 'no decode loop' cost line conflicts with MAE comparability — every (b') comparator arm's error is DECODED-pred error, so mc's per-candidate preds must come from decodes under each candidate (C decodes, ~2-2.5 GPU-h, still &lt;= 4 gate; KL computable during the decode + one masked teacher-forced reference forward per candidate sequence). Keeps cand_pred [N,C,S,D] + argmax-in-reader intact. AMEND the draft's cost/mechanics lines accordingly BEFORE finalization (draft is mutable by design; the read script contract needs no change) | INSTRUMENT LANDED + PRE-REG FINALIZED + RUN LAUNCHED 09:12:36Z 08-09 (5181d8e): --subgoal-mode mcselect (candidates-file injection, ActionCaptureStep capture from the decode's own logits, teacher-forced masked reference vs snapshot/restored shared prefill, KL float64 over the grammar-legal set; dump + report echo per the pre-data contract); oracles green (tests/test_mcselect.py 15 tests: planted-KL exact arithmetic, tau-&gt;inf = log|legal|-H exact, decode-vs-teacher-forced identity real-decoder, capture-off byte-equal; mcselect_live_oracles.py 9 abort branches; check.py 574); 12-row real-checkpoint smoke rc=0 verified contract keys/NaN==eligibility, 1.4 s/frame -&gt; ~1.7 GPU-h projected &lt;= 4 gate; BONUS: smoke caught the subgoal-mode report-sort KeyError that silently cost the (b') q4 run its HTML — fixed. Unit fontaine-mcselect-q4, babysit entry mcselect_q4 live, chain run -&gt; live oracles -&gt; frozen read | CLOSED 10:2xZ 08-09: ANTI-SELECT, family CLOSES (see boundary)

</details>

---

**`idea6-subgoal-draws-escalation-prereg-draft`** · `cpu`

#6 rung-(b) ESCALATION pre-reg draft (CPU) — opens ONLY on a no-scorer verdict from the rung-(b') read (rung (b) closed at table cost; (b') carries the frozen falsifier + adjudication) (ceiling &gt;&gt; bon with diversity present); clo…

**boundary:** DRAFT LANDED 05:3xZ 08-09; execution per idea6-mcselect-execution · [pre-reg](posts/2026-08-09-prereg-subgoal-mcselect.md)

<details><summary>full record</summary>

#6 rung-(b) ESCALATION pre-reg draft (CPU) — opens ONLY on a no-scorer verdict from the rung-(b') read (rung (b) closed at table cost; (b') carries the frozen falsifier + adjudication) (ceiling &gt;&gt; bon with diversity present); closes as moot on no-diversity or on the falsifier passing. Routing pre-mapped by the 08-08 lit slice (papers/progress-from-logits.md + corrected self-certainty.md note): candidate 1 = masked-contrast selection (MG-Select form, prerequisite VERIFIED MET: subgoal-masked reference = planner-less path trained at 50% dropout; N+1 teacher-forced pass-2 action forwards, no decode loop; reference tempered tau=4); candidate 2 = history-conditioned planning (TOPReward 2602.19313: phase zero-shot recoverable from a video prefix via one completion logit incl. on Molmo2-8B — attacks the measured ~10/60 single-frame phase-offset mechanism directly). Draft picks ONE (or stages both) with frozen falsifiers + cost gates; execution needs its own posted pre-reg + queue entry | SWAP READ BANKED 03:5xZ 08-09 (MIXED): content consumed (+0.166 truth-over-wrong margin) — scorer coherence question resolved POSITIVE; cost any scorer rung against the free any-plausible-words floor Delta_swap -0.113, not against no-slot | OPENING CONDITION ADJUDICATED MET 05:3xZ 08-09 (audit: the item sat blocked though the (b') read HAD routed no-scorer-with-live-ceiling 00:2xZ + the swap read resolved coherence positive 03:5xZ) — DRAFT LANDED same session (2026-08-09-prereg-subgoal-mcselect.md): picks candidate 1 ONLY (MG-Select masked-contrast, tau=4 verbatim, subgoal-dropout-0.5 prerequisite met), FROZEN to the banked (b') candidates file (4301 q4 rows x 8 clean texts) so ceil/floor comparators hold by construction; E6-mirror falsifier (mc - self CI95 &lt; 0), anti-select second-strike closes the zero-training family; gate &lt;= 4 GPU-h local, no decode, no training. Candidate 2 (TOPReward history probe) = named escalation only on a phase-specific failure. Execution split to idea6-mcselect-execution (blocked: instrument + oracles + finalization stamp first)

</details>

---

**`lit-slice-verifier-free-selection-followups`** · `cpu`

Standing lit slice (~20-30 min, owner allocation 2026-08-05): verifier-free candidate selection FOLLOW-UPS to the #6 rung-(b) scorer cell, timed to land BEFORE the rung-(b) read so escalation routing has its map

**boundary:** closed 08-08 04:1xZ (chained work session, inside the endpoint/R1 wait window - exactly the natural window the item named): all three lanes answered, papers page landed same session (papers/progress-from-logits.md: TOPReward 2602.19313 + ProgVLA 2605.28231). (a) NOTHING published beats self-certainty label-free at inference on open-ended text - the rung-(b) frozen scorer cell stands (RoVer/EVE are trained verifiers, named not competitors). (b) history fixes phase: TOPReward recovers progress zero-shot from a video PREFIX via one completion logit (log p('True'), no chat template), VOC 0.947 vs GVL 0.332 on ManiRewardBench, tested incl. Molmo2-8B (our trunk family) - history-conditioned planning is the evidence-backed planner-side escalation for the ~10/60 phase-offset mechanism; ProgVLA does NOT unblock (b) (progress heads are training-time reweighting only, -2.3 pt ablation, no single-frame-vs-history contrast). (c) MG-Select prerequisite VERIFIED MET - the paper masks text/state, never frames; our subgoal-masked reference = the planner-less path trained at 50% dropout; correction banked on papers/self-certainty.md (the old 'frame-masked off-distribution' note read the prerequisite too broadly); bare masking still gained in their ablation (17.0-&gt;22.6), dropout training doubled it (31.0). ideas.md idea-6 hook updated; NO new pre-reg (per the item's own rule)

<details><summary>full record</summary>

Standing lit slice (~20-30 min, owner allocation 2026-08-05): verifier-free candidate selection FOLLOW-UPS to the #6 rung-(b) scorer cell, timed to land BEFORE the rung-(b) read so escalation routing has its map — (a) trained/lightweight process verifiers for plan selection on robot policies (does anything beat self-certainty without labels at inference?), (b) phase/progress estimation from single frames (the rung-(a) bottleneck: ~10/60 phase-offset rows) — history-conditioning, memory, or test-time state estimators that could feed the planner-side escalation, (c) MG-Select masked-contrast transferability check (its image-dropout prerequisite vs our trained camera_kind_dropout — is the prerequisite actually met?). Papers-section page(s) land SAME session (standing rule 08-07 08:42Z); ideas.md hooks into idea 6 escalation cells; NO new pre-reg from this slice without its own draft item

</details>

---

**`idea6-subgoal-draws-execution`** · `gpu-local`

#6 rung (b) subgoal-draws selection EXECUTION (gpu-local): (1) instrument + read script landed by the CPU-side item (idea6-subgoal-draws-instrument) (bijou.eval selfsubgoal mode + sampled pass-1 draws --subgoal-draws 8 --subgoal-…

**boundary:** closed 2026-08-08 10:2xZ at table cost (~1.6 of 6 GPU-h) · [pre-reg](posts/2026-08-08-prereg-subgoal-draws.md)

<details><summary>full record</summary>

#6 rung (b) subgoal-draws selection EXECUTION (gpu-local): (1) instrument + read script landed by the CPU-side item (idea6-subgoal-draws-instrument) (bijou.eval selfsubgoal mode + sampled pass-1 draws --subgoal-draws 8 --subgoal-temperature 1.0 spelling implementation's, per-candidate SC-sufficient distribution stats dump, _bonsubgoal/_ceilsubgoal selection modes, machine-readable candidate dump; abort-on-red oracles per the pre-reg: draws-0 limit bit-exact vs the banked rung-(a) self arm at matched composition, forced-empty = plain path, SC + token-F1 exact-arithmetic fixtures incl. tie cases, pass-2 excludes subgoal, _ceilsubgoal provenance separation, collator-rendering byte match); (2) stage-1 candidates table on the rung-(a) 60-frame sample, go/no-go bars per the pre-reg (&gt;=90% valid, &gt;=2 unique strings on &gt;=50% frames, no &gt;50% pooled collapse, subgoal-shaped) — fail closes the rung at table cost; (3) stage-2 two conditioned full-panel arms (bon + ceil) via run_detached.sh, first-200-frame rate check vs the 6 GPU-h gate (q4 fallback clause verbatim); (4) frozen reads via the landed read script: Delta_bon + paired (bon - self) head-to-head, Delta_ceil bound, scorer-agreement records, horizon decomposition, first_mae mirrors — PREFLIGHT LIVE 08:49:29Z 08-08 (unit fontaine-subgoal-draws-preflight via run_detached.sh): live-oracle adjudicator subgoal_draws_live_oracles.py landed same session (selftest green, 14 branches; matched-composition by construction — oracle i vs a FRESH q4 self run, oracle ii vs the BANKED q4 emptyhint npz, the amendment-1 lesson mechanized) + both launchers (preflight 5-phase; arms launcher gated on the green summary JSON AND the stage-1 go marker, draws_rate_gate q4 fallback at 4.5 GPU-h remaining budget); babysit entry live. Next: judge stage-1 table at preflight rc=0 (~10:1xZ), write subgoal_draws_stage1_go, launch arms unit fontaine-subgoal-draws-arms — CLOSED AT TABLE COST 10:2xZ 08-08 per the frozen stage-1 rule: preflight rc=0 10:1xZ (~1.6 GPU-h; LIVE ORACLES ALL GREEN — draws-0 bon+narr bit-exact vs the fresh q4 self run, forced-empty both arms bit-exact vs the banked emptyhint, candidate texts exact), stage-1 bars: (a) FAIL 20/60 sampled-clean vs &gt;=90% — 55/480 sampled draws (11.5%) derail at T=1.0 into budget-truncated multilingual gibberish (zero empties; greedy clean 60/60; 0.885^8~38% binomial arithmetic reproduces the row rate); (b) pass 58/60 diverse; (c) pass 4.8% top pooled. Free table reads: clean candidates subgoal-shaped incl. adjacent-phase alternatives; SC pick != greedy 39/60 (expectation 5 exceeded); SC NEVER picks a truncated candidate (0/60, median rank 9/9) — the scorer already refuses the gibberish, but filtering was pre-reg-forbidden so the bar stands. Delta_bon/Delta_ceil UNMEASURED. Results post 2026-08-08-subgoal-draws-stage1-close.md; arms launcher unused (no go marker, correct); escalation queued

</details>

---

**`idea6-subgoal-draws-instrument`** · `cpu`

#6 rung (b) INSTRUMENT + read script (CPU, any GPU-busy window, oracle-gated, lands BEFORE the execution item's launch

**boundary:** closed 08-08 03:5xZ (work session): instrument + read script landed oracle-green — bijou.eval subgoal-mode draws (pass 1 decodes 1 greedy + --subgoal-draws sampled candidates at --subgoal-temperature off ONE shared prefill via decoder.decode_value_line; per-step chosen/mean logprob stats make self-certainty exactly recomputable offline; model-level candidate-0 == full-pass greedy assert), _bonsubgoal/_ceilsubgoal SelectedSubgoalPolicy arms (SC pick vs token-F1 ceiling; label-less ceil rows render no hint; force_empty extends), --dump-subgoal-candidates machine-readable table w/ live picks + record-only likelihood/medoid alternates; scorers in bijou/eval/subgoal_scoring.py (pure, tie-&gt;lowest-index); read script fontaine/scripts/subgoal_draws_results.py (Delta_bon + paired bon-self vs banked rung-(a) self npz, Delta_ceil + ceil-self adjudication no-diversity/no-scorer, agreement records, horizon, first_mae mirrors, 11 abort branches, --oracle selftest green); 22 new tests in tests/test_subgoal_draws.py incl. REAL tiny-model decode-loop oracle-i half; check.py 489 green. GPU-side oracles (draws-0 bit-exact, forced-empty) remain the execution item's preflight. · [pre-reg](posts/2026-08-08-prereg-subgoal-draws.md)

<details><summary>full record</summary>

#6 rung (b) INSTRUMENT + read script (CPU, any GPU-busy window, oracle-gated, lands BEFORE the execution item's launch — the rung-(a) reads-before-data precedent): bijou.eval selfsubgoal mode gains sampled pass-1 draws (--subgoal-draws 8 --subgoal-temperature 1.0 spelling implementation's; draws10_t1 per-frame stable seeding verbatim), per-candidate distribution stats sufficient to compute self-certainty exactly, _bonsubgoal/_ceilsubgoal selection modes, machine-readable candidate dump (frame triple -&gt; candidates/scores/pick); scorers: SC (mean KL-from-uniform, argmax, tie -&gt; lowest index) + ceiling token-F1 (lowercase, whitespace tokens, tie -&gt; lowest index) with exact-arithmetic selftest fixtures incl. single-candidate + tie cases; read script subgoal_draws_results.py (Delta_bon, paired bon-self vs the banked rung-(a) self npz, Delta_ceil, agreement records, horizon curves, first_mae mirrors; abort branches per the pre-reg); CPU-verifiable oracles green in selftest (fixtures, provenance separation, pass-2 generate-list exclusion); the GPU-side oracles (draws-0 bit-exact vs rung-(a) self arm at matched composition, forced-empty = plain path) run as the execution item's preflight

</details>

---

**`idea6-subgoal-draws-prereg-draft`** · `cpu`

#6 escalation rung (b) pre-reg POSTED 03:2xZ 08-08 (posts/2026-08-08-prereg-subgoal-draws.md): candidate-subgoal SELECTION frozen

**boundary:** closed 08-08: pre-reg posted + papers page landed + execution entry queued

<details><summary>full record</summary>

#6 escalation rung (b) pre-reg POSTED 03:2xZ 08-08 (posts/2026-08-08-prereg-subgoal-draws.md): candidate-subgoal SELECTION frozen — 9 candidates (greedy + 8 sampled T=1, draws10_t1 seeding verbatim), primary scorer self-certainty (2502.18581, mean KL-from-uniform, argmax, zero extra forwards; likelihood + medoid token-F1 record-only alternates from the same dumps), plus a record-only ORACLE-similarity ceiling arm bounding every scorer at this width; head-to-head falsifier = paired (bon - self) CI95 entirely below 0 vs the banked rung-(a) self npz; stage-1 candidates table gates stage 2 (diversity bar); gate &lt;= 6 GPU-h w/ q4 fallback; no-prompt-fishing + matched-composition constraints inherited. Scorer lit check landed same-session (papers/self-certainty.md). Execution queued separately (instrument to land oracle-gated first).

</details>

---

**`idea1-golden-ticket-screen-execution`** · `gpu-local`

#1 golden-ticket screen: stage 1 DONE (R1 CONFIRM 04:2xZ 08-08: sd 0.82252 vs 0.0785, min 5.70564 vs 6.52401, winner ticket 33) -&gt; stage 2 DONE (R2 REAL 05:1xZ: complement paired -0.924 [CI95 -0.985, -0.866] vs line -0.05 on 14,7…

**boundary:** closed 2026-08-08 08:2xZ · [pre-reg](posts/2026-08-07-prereg-golden-ticket-screen.md)

<details><summary>full record</summary>

#1 golden-ticket screen: stage 1 DONE (R1 CONFIRM 04:2xZ 08-08: sd 0.82252 vs 0.0785, min 5.70564 vs 6.52401, winner ticket 33) -&gt; stage 2 DONE (R2 REAL 05:1xZ: complement paired -0.924 [CI95 -0.985, -0.866] vs line -0.05 on 14,746 rows, core-pooled 5.6468/1.8963 = leaderboard row 7; effect directional not norm) -&gt; stage 3 LIVE 05:16:30Z (unit fontaine-goldenticket-stage3, mean-of-top-10 [33,2,0,51,10,59,38,28,15,36] byte-verified sha e537f4cd, draws-10 ticket noise, ~2.9 GPU-h, lands ~08:1xZ). REMAINING: R3 pooled read vs banked 5.3645 (tie band ±0.02, RECORD-ONLY) + R4 record-only reads (per-dataset argmin task-locality, quartile geometry, horizon) + stage-3 write-up; screen budget ~5.5 of the 6 GPU-h gate. Results post (stages 1-2) published 2026-08-08-goldenticket-results.md — STAGE 3 LANDED 08:15:39Z rc=0 (2.99 GPU-h; screen total ~5.55 &lt;= 6 gate); R3 read via oracle-green goldenticket_stage3_results.py: mean-of-top-10 5.1847/1.3831 vs banked 5.3645/1.4242 -&gt; delta -0.180 = INTERESTING 9x beyond the band (RECORD-ONLY per pre-reg; best chunk+first numbers on the panel, row needs paired follow-up); R4a task-locality (argmin 4.4%/792 datasets, top-10 containment 29.8% ~2x null, median cell 2 frames caveat); R4b gain monotone in dispersion (-0.35 -&gt; -1.44 by quartile). SCREEN CLOSED: R1 CONFIRM -&gt; R2 REAL -&gt; R3 INTERESTING.

</details>

---

**`idea6-selfsubgoal-frozen-reads`** · `cpu`

#6 frozen reads + results post EXECUTED 02:4x-03:0xZ 08-08: selfsubgoal_results.py one command, execution oracles GREEN (anchor re-pool exact, identity/state-copy byte-match, modes carried; 25788 labeled/12 label-less rows, 5 dif…

**boundary:** closed 2026-08-08 ~03:0xZ · [pre-reg](posts/2026-08-07-prereg-selfsubgoal-probe.md)

<details><summary>full record</summary>

#6 frozen reads + results post EXECUTED 02:4x-03:0xZ 08-08: selfsubgoal_results.py one command, execution oracles GREEN (anchor re-pool exact, identity/state-copy byte-match, modes carried; 25788 labeled/12 label-less rows, 5 differ = amendment-1 composition class RECORDED — the read script's pre-amendment label-less byte-match guard fired on the real dumps and was re-graded to amendment-1 descriptive semantics BEFORE the reads ran, selftest updated, pre-reg abort set untouched). E1 CONFIRMED -0.290; E2 point-wise only (CI spans 0); E3 CONFIRMED (last10 -0.480 vs first10 -0.081); E4 confirmed (+0.026, narr-self +0.043 CI excl 0); E5 not fired, deployment claim dead anyway. No leaderboard change (oracle = not deployment class; self = null at 3x cost). Results post + ideas/idea-06 ledger + stage-1 table commentary landed; babysit entry pruned

</details>

---

**`molmo2-endpoint-postprocessing`** · `cpu`

molmo2 40k endpoint POST-PROCESSING (CPU, opens when the chained greedy panel eval lands ~04-05Z 08-08): read eval__fontaine_molmo2_ar_40k_ddp4__step_040000__panel_curated_v0_k4l2.json -&gt; leaderboard row (measured decode-cost col…

**boundary:** closed 08-08 05:0xZ (chained work session): greedy JSON landed 04:53Z (after the 04:16 chained-eval dtype incident was root-caused + fixed + relaunched via the #19 launcher's greedy-if-missing clause, 5a43b15); frozen reads executed via molmo2_endpoint_results.py (oracle-green BEFORE data, 61dacb9): READ 1 = BEATS — 6.0079/2.1871 vs A-s0 7.7966/3.9422, paired -1.717 [CI95 -1.797, -1.635] on 17,204 core frames -&gt; frozen decision executes, Molmo2 is the phase-2 flow-trunk candidate; READ 2 byte-match green (pooled 11.7847/2.6202; the pre-reg's parenthetical 11.7639/2.5851 reproduces under NO pooling of this plan — drafting slip recorded, not silently corrected). Leaderboard row 7 added (cost cells await the microbench prep item — flagged, nothing mtime-derived) + own-topology table row; results post 2026-08-08-molmo2-endpoint-results.md published + Space 200-verified + Discord line; ideas.md #17/#19 hooks updated; babysit repointed train-&gt;greedy-&gt;draws10_t1 phases at each transition; endpoint probe 6.2075@40000 quoted in the results post as the vu5k amendment's frozen-sanity bar input (feeds idea17 cell 2); checkpoint upload done same session (weights-only, 4 files hub-verified) · [pre-reg](posts/2026-08-06-prereg-molmo2-ar-40k.md)

<details><summary>full record</summary>

molmo2 40k endpoint POST-PROCESSING (CPU, opens when the chained greedy panel eval lands ~04-05Z 08-08): read eval__fontaine_molmo2_ar_40k_ddp4__step_040000__panel_curated_v0_k4l2.json -&gt; leaderboard row (measured decode-cost column caveat: molmo2 configs were NOT in the microbench set — mtime-derived number flagged as such or left blank), ledger entry vs the pre-reg's frozen reads (2026-08-06-prereg-molmo2-ar-40k.md), blog post + Discord line; prune/repoint babysit molmo2_ar40k entry at the same commit (the train run is over; the draws-arm prepared entry takes over per the registry comment); quote the endpoint probe value HERE for the vu5k finalization amendment's frozen-sanity bar (feeds idea17 execution cell 2); checkpoint upload per the upload-valuable-checkpoints standing rule (weights-only unless seeding)

</details>

---

**`idea17-vu5k-finalization-prep`** · `cpu`

#17 vu5k finalization amendment PREP — EXECUTED 08-07 19:4xZ work session: byte-audit CLEAN at HEAD (all amendment-3 flags exist on bijou.train's CLI; --init-from verified weights-only fresh-AdamW loading expert+prompt+adapted-ba…

**boundary:** CPU work at any GPU-busy window; wanted before the molmo2 endpoint (~04-05Z 08-08) so the finalization amendment can post same-session with the smoke once the box frees · [pre-reg](posts/2026-08-07-prereg-molmo2-vision-unfreeze.md)

<details><summary>full record</summary>

#17 vu5k finalization amendment PREP — EXECUTED 08-07 19:4xZ work session: byte-audit CLEAN at HEAD (all amendment-3 flags exist on bijou.train's CLI; --init-from verified weights-only fresh-AdamW loading expert+prompt+adapted-backbone bf16 snapshot with start_step 0, no seed-collision guard applies (init-from exempt), cosine-to-10%-floor confirmed at lr_lambda train.py:1552 shared multiplicatively by ALL groups so vision=text holds through the schedule, vision-group hard-abort at train.py:544-550 confirms no silent no-op unfreeze, --backbone-vision-lr requires --backbone-text-lr satisfied); two arm launchers LANDED launch_box_fontaine_molmo2_vu5k_{frozen,thawed}_ddp4.sh (base 40k recipe byte-identical — diff vs 40k launcher = exactly the pinned deltas; arm-vs-arm train-command diff = exactly --backbone-vision-lr 6e-6 + ladder knobs; plan sha af3f8546 pinned; eval stems = pre-reg §2 verbatim; thawed REFUSES without frozen step_005000 (frozen-first mechanized) AND without fontaine/harness/state/vu5k_mem_ready (smoke record contract: RUNG/BACKWARD_CHUNKS/ACT_CKPT/VRAM_PEAK_GIB/SMOKE_UTC, documented in-header); run_detached.sh launch lines in both headers) + prepared babysit.toml vu5k_{frozen,thawed} entries (vram 71 gates, FILL-AT-FINALIZATION probe bars, x3-sustained judgment + async-save first-validation anchors; TOML parses, active entries untouched). check.py 467 green. Remaining finalization cells stay with the execution item (need the box checkpoint): 150-step thawed smoke -&gt; vu5k_mem_ready, endpoint-probe quote -&gt; probe bars, amendment POST, owner go

</details>

---

**`idea1-golden-ticket-instrument`** · `cpu`

#1 golden-ticket screen INSTRUMENT — LANDED 08-07 19:4xZ work session, all 4 oracles GREEN: --noise-tickets ticket mode in bijou.eval (noise = tickets[draw] frame-independent via the _flow_noise seam; policy name gains _ticket; r…

**boundary:** done — execution item carries the launch · [pre-reg](posts/2026-08-07-prereg-golden-ticket-screen.md)

<details><summary>full record</summary>

#1 golden-ticket screen INSTRUMENT — LANDED 08-07 19:4xZ work session, all 4 oracles GREEN: --noise-tickets ticket mode in bijou.eval (noise = tickets[draw] frame-independent via the _flow_noise seam; policy name gains _ticket; report JSON + draws npz carry noise_tickets + tickets_sha256; keyed path byte-identical to pre-refactor — regression-tested), bank committed plans/tickets_goldenticket_m64.npz (64x50x6 f32, SeedSequence [0x54434B54,0,m], file sha 9bb13bc4…, content sha a07c062a…, make_golden_tickets.py --verify green), tests/test_golden_ticket.py (7 tests: contract bit-exact vs sample_actions, cross-frame ticket property in-process, two-run determinism, both sha pins, loud refusals), ticket_scores.py --oracle green (pooling reuse reproduces 6.5997 + the 10 banked per-draw probe MAEs EXACTLY, R1 branches, provenance refusals, tie-break); stage-1 scorer + frozen R1 kill line + R4a per-dataset matrix ready before the data; check.py 467 green. No semantic deviation from the pre-reg — no amendment needed

</details>

---

**`idea1-golden-ticket-prereg-draft`** · `cpu`

#1 golden-ticket noise screen — pre-reg POSTED 08-07 ~18:1xZ work session (2026-08-07-prereg-golden-ticket-screen.md, immutable, not a draft

**boundary:** DONE 08-07; execution item queued separately · [pre-reg](posts/2026-08-07-prereg-golden-ticket-screen.md)

<details><summary>full record</summary>

#1 golden-ticket noise screen — pre-reg POSTED 08-07 ~18:1xZ work session (2026-08-07-prereg-golden-ticket-screen.md, immutable, not a draft — every design constant pinned from banked data): teacher-first (bijou_flow_artrunk@80k Heun-30; student = escalation amendment only), M=64 i.i.d. tickets [50,6] sha-pinned, stage 1 = ONE batched draws-64 eval on drawsprobe_s7 where the draws ARE the tickets (~1.5 GPU-h via the batched-draws merge), frozen null from banked sigma_draw_direct (sigma_probe 0.0669, null min64 = mean-0.157, MC-verified constants), R1 kill line BEFORE stage 2 (sd&gt;0.0785 OR min&lt;mean-0.22), R2 confirmatory = winner on COMPLEMENT core rows paired vs stablekey npz (adopt floor -0.05 = 2sigma), R3 mean-of-top-10-tickets vs banked 5.3645 (pooled, tie band 0.02), R4 free per-dataset task-locality read; instrument = ticket noise-key mode in bijou.eval, 4 oracles frozen; gate 6 GPU-h

</details>

---

**`idea17-molmo2-vision-unfreeze-prereg-draft`** · `cpu`

#17 molmo2 vision-unfreeze rung — pre-reg DRAFT — DRAFTED 08-07 ~17:5xZ; AMENDED 08-07 18:xxZ work session to the OWNER-STEERED (18:02Z) warm-start two-arm design (2026-08-07-prereg-molmo2-vision-unfreeze.md amendment 1, loud DRA…

**boundary:** DONE 08-07 (draft). Execution item queued separately; window opens after the attach-screen chain (~08-09+), owner-steered · [pre-reg](posts/2026-08-07-prereg-molmo2-vision-unfreeze.md)

<details><summary>full record</summary>

#17 molmo2 vision-unfreeze rung — pre-reg DRAFT — DRAFTED 08-07 ~17:5xZ; AMENDED 08-07 18:xxZ work session to the OWNER-STEERED (18:02Z) warm-start two-arm design (2026-08-07-prereg-molmo2-vision-unfreeze.md amendment 1, loud DRAFT banner retained): both arms --init-from the 40k endpoint step_040000 (frozen-continue CONTROL vs thawed-continue --backbone-vision-lr 2e-6; --resume mechanically excluded — extra vision param group breaks optimizer.load_state_dict, verified at HEAD), 3k steps each, --warmup-steps 200, tail LRs decoder 1e-5 / text 2e-6, seed 1 BOTH arms (identical batches; arms differ in exactly one flag), frozen-first ordering (its curve = thawed kill-line reference); PRIMARY = thawed@3000 - frozen@3000 paired per-frame delta CI95, null band 0.07, critical-frame re-pool; arm-vs-endpoint reads record-only; memory ladder unchanged (thawed arm; matched downshift EXCLUDED); kill lines: frozen-arm sanity vs banked endpoint probe +2.0 x3, thawed vs frozen +2.0 x3 after step 1000, vram 71 gate; cost gate 24 GPU-h (~15 train); declared caveats: late-thaw may understate from-scratch (tie != 'unfreezing doesn't help'), MAPS OOD blind spot. Finalization checklist converts DRAFT -&gt; posted

</details>

---

**`idea16-critical-frame-repooling`** · `cpu`

#16 critical-frame re-pooling screen (CPU, CI-MSE transfer from the offline-validation lit slice 08-07)

**boundary:** DONE 08-07; rollout-vs-offline stays open until the #16 rig benchmark exists · [pre-reg](posts/2026-08-07-prereg-critical-frame-repooling.md)

<details><summary>full record</summary>

#16 critical-frame re-pooling screen (CPU, CI-MSE transfer from the offline-validation lit slice 08-07) — EXECUTED 08-07 ~17:5xZ work session (pre-reg 2026-08-07-prereg-critical-frame-repooling.md posted+committed 4773ba9 BEFORE the read; frozen rule: chunk window [f0+1,f0+50] hits subgoal boundary | holding bracket | event frame, blessed-judgment selection = bijou.data training rule): EVERY PUBLISHED RANKING HOLDS on the critical pool — all 10 pairwise gaps keep sign w/ CI95 excluding 0, coverage 99.9% (11204 critical core frames), every overall re-pool reproduced published to 4dp, recombination exact; separation vs state-copy WIDENS on critical frames (opposite of CI-MSE's failure mode); seed-trio critical null scale 0.1476. Robustness citation on leaderboard + results section in the pre-reg post + analysis json; instrument critical_frame_repooling.py (--selftest oracle) reusable at the molmo2 endpoint

</details>

---

**`attach-launch-save-cadence-prep`** · `cpu`

Attach-screen launch prep: save-cadence call + pinned-buffer hook decision

**boundary:** before the attach-screen launch (~08-08, after molmo2 endpoint + K smoke ladder); CPU work at any GPU-busy window

<details><summary>full record</summary>

Attach-screen launch prep: save-cadence call + pinned-buffer hook decision — DONE 08-07 17:0xZ (c4555d4): --save-every 2500-&gt;1250 BOTH arms matched + pre-reg amendment 2 (every posted judgment boundary preserved; recovery loss halved ~108-&gt;~54 min at K rate); pinned-buffer refinement DEFERRED (capture stall seconds vs &gt;=26-min interval, &lt;0.2% overhead), stays banked on #18.9

</details>

---

**`lit-slice-decode-temperature`** · `cpu`

Standing lit slice (~25 min, owner allocation 2026-08-05): decode-time stochasticity for tokenized-action policies, timed to the #19 tsens rungs scoring tonight

**boundary:** DONE 08-07 same-session; the dT read (tonight) consumes the prior

<details><summary>full record</summary>

Standing lit slice (~25 min, owner allocation 2026-08-05): decode-time stochasticity for tokenized-action policies, timed to the #19 tsens rungs scoring tonight — EXECUTED SAME SESSION 2026-08-07 ~17:1xZ, papers page landed per the permanent rule (papers/decode-temperature.md, 5 sources: 2605.22493 multimodal-failure anchor, MARS 2605.29766, action-quantization theory 2603.20538, BOKBO 2605.30660, DDVLA temp ablation 2508.20072 — Table 7 verified at source after the search digest misquoted 96.8 for the real 97.4): dT read gains a written directional prior BEFORE the rungs land (near-flat with asymmetry against T=1.3 on a unimodal-dominated panel); BOKBO banked as the 2nd independent strike on cheap probe selectors (#19 selection rung); q-token+CE trunk gains its sample-complexity-optimality citation; DDVLA temperature-SCHEDULE hook parked, opens only on real dT sensitivity. ideas.md #19 bullet + index/SUMMARY rows

</details>

---

**`driver-background-task-guard`** · `cpu`

OWNER STEERING 08-07 13:05Z (tooling follow-up to the 12:56Z premature-session-end incident), PULLED FORWARD 08-07 15:4xZ = NEXT CPU ITEM (async-ckpt done; 2 GPU runs killed by this in one day): harden the session driver so a com…

**boundary:** DONE 08-07: run_detached.sh is the required launch path for anything that must outlive a session

<details><summary>full record</summary>

OWNER STEERING 08-07 13:05Z (tooling follow-up to the 12:56Z premature-session-end incident), PULLED FORWARD 08-07 15:4xZ = NEXT CPU ITEM (async-ckpt done; 2 GPU runs killed by this in one day): harden the session driver so a completed turn cannot silently kill live work — audit fontaine/harness session-driver lifecycle; add a guard that (a) refuses to treat turn-completion as session end while registered background tasks are live (re-prompt instead), or at minimum (b) never tears down setsid-detached/registered jobs; add a driver test reproducing the 12:56Z signature (terminal_reason completed + live bg tasks). Memory-file mitigation (no-end-turn-waiting-on-notifications) already in place; this mechanizes it. SECOND INCIDENT 08-07 15:0x-15:1xZ: the 13:04Z work session's tsens q4 launch (15:01:40Z, not setsid-detached despite the memory file) was killed by the same turn-completion teardown when that session ended ~15:11Z — 32/4301 frames lost, relaunched setsid-detached 15:13:44Z by the tick. THIRD INCIDENT 08-07 15:5xZ + ROOT CAUSE UPGRADE: the 15:13:44Z setsid-detached relaunch was ALSO killed (~15:54-15:56Z, log frozen at 992/4301, no traceback/OOM) when fontaine-tick.service finished at 15:56:18Z -- journalctl shows the unit consumed 1h17m CPU then stopped, taking its whole cgroup with it. setsid escapes the terminal session but NOT the systemd service cgroup (KillMode=control-group). The memory-file setsid mitigation is therefore INSUFFICIENT by mechanism, not by compliance. Working fix (3rd relaunch 15:58:26Z): systemd-run --user --unit=fontaine-tsens-q4 --collect + explicit PATH/HOME (clean env lacks uv -- first attempt died exit 127). The guard item should now ALSO codify systemd-run as the required GPU-launch wrapper (launcher docs + memory file + charter) and consider KillMode=process for fontaine-tick.service itself. — LANDED 08-07 16:xxZ work session (4 defense layers, all live-verified): (1) fontaine/scripts/run_detached.sh = the codified REQUIRED launch wrapper (systemd-run --user + PATH/HOME setenv + grace-window launch-death check surfacing the exit-127 class); (2) KillMode=process on fontaine-tick.service (repo unit is the installed symlink target; daemon-reload applied — noncompliant launches now survive unit stop as stragglers instead of dying silently); (3) babysit.py DRIVER-CGROUP SURFACED line whenever a registered run's processes sit inside the driver cgroup — fires at every poll BEFORE the kill (two live-measured self-match classes excluded: probe ancestor chain + the pipeline-fork inheriting the pattern-bearing bash -c cmdline); (4) driver_guard.py post-session cgroup scan wired into fontaine-session.sh with 1-h-cooldown Discord alert. Driver test: tests/test_driver_guard.py reproduces the incident-3 kill signature LIVE with transient units (default KillMode kills a setsid child; KillMode=process spares it; run_detached job survives parent-unit teardown; fast-death surfaced) + fake /proc+cgroup scan oracles + unit-file regression guard; babysit oracles extended (positive decoy control + compliant-unit clean both verified live on the running tsens run). check.py 460 green. Charter harness section + memory file + 6 local launcher headers codified.

</details>

---

**`lit-slice-async-training-systems`** · `cpu`

Standing lit slice (~20-30 min, owner allocation 2026-08-05): async/overlapped-checkpointing + training-systems efficiency cluster, timed to the just-landed async_save.py (e3bdc93)

**boundary:** DONE 08-07 before the attach-screen launch; save-cadence call (follow-up hook) belongs to attach launch prep

<details><summary>full record</summary>

Standing lit slice (~20-30 min, owner allocation 2026-08-05): async/overlapped-checkpointing + training-systems efficiency cluster, timed to the just-landed async_save.py (e3bdc93) — EXECUTED SAME SESSION 2026-08-07 ~16:1xZ: 6-source cluster read (CheckFreq FAST'21, Gemini SOSP'23, DataStates-LLM 2406.10707, GoCkpt 2511.07035, TierCheck 2605.17821, checkpoint-I/O study 2512.24511), papers page landed per the permanent rule (papers/checkpointing-systems.md): our two-phase design corroborated (DataStates = our shape + 2 refinements); 3 transfers banked as hooks on #18.9 (pinned-buffer reuse across saves — DataStates pre-allocates ONE reusable pinned buffer; save-frequency retuning now saves are ~free — CheckFreq auto-tune, relevant given 2 driver-kill recoveries today; CheckFreq's data-iterator-state gap our --resume shares, named with citation); 3 honest non-transfers (memory tiers = node-loss fault model we don't have; multi-step spreading = snapshot &gt;&gt; step slack, not our regime; per-rank sharded format = kills the gather but breaks the consolidated read-side contract byte-identity deliberately preserved). ideas.md #18 hook + index/SUMMARY rows

</details>

---

**`idea19-tsens-dt-read-execution`** · `cpu`

#19 dT diagnostic read EXECUTED 23:09Z 08-07: monotone table chunk 6.5004/6.5668/6.7812/7.1843 at T=0.5/0.7/1.0/1.3 (record-only, primary stays T=1.0; T=1.3-asymmetry prior confirmed); babysit entry pruned, local GPU confirmed fr…

**boundary:** closed 2026-08-07 23:09Z; follow-up read item still open: T-guard delta via q4 subset join · [pre-reg](posts/2026-08-06-prereg-ar-sampled-draws.md)

<details><summary>full record</summary>

#19 dT diagnostic read EXECUTED 23:09Z 08-07: monotone table chunk 6.5004/6.5668/6.7812/7.1843 at T=0.5/0.7/1.0/1.3 (record-only, primary stays T=1.0; T=1.3-asymmetry prior confirmed); babysit entry pruned, local GPU confirmed free; ledger entry on idea page 19 + Discord line

</details>

---

**`idea4-f-then-joint-prereg-draft`** · `cpu`

#4 F-then-joint escalation rung — pre-reg DRAFT (CPU; the APT-named recipe, papers/apt-expert-pretraining.md): warm-start a joint run (unfrozen trunk, seam per readout) from the F arm's converged expert instead of from noise; APT…

**boundary:** opens after the #4 attach-screen frozen reads (~08-09+); the draft is CPU work at any GPU-busy window AFTER Delta_seam is banked; execution needs its own posted pre-reg + owner-visible queue entry | 12:4xZ 08-09: Delta_seam readout will NOT arrive (owner killed K); draft basis re-anchors on F-arm curves + the 4x step-cost fact — joint-anything must now argue against that cost. Stays blocked pending stage-2 decision memo. | UNBLOCKED 13:5xZ 08-09: the stage-2 decision memo is the replacement basis (frozen adopted; joint-anything must argue vs the measured 4.11x joint-step cost). CPU draft at any GPU-busy window; natural target = the adamc_100k endpoint (~08-12); execution still needs posted pre-reg + owner go. | DONE 14:2xZ 08-09 work session: pre-reg DRAFT posted (posts/2026-08-09-prereg-fjoint-rung.md) — J (unfrozen, no stop-grad, CE rider, warm-start from banked F@10k expert) vs F2 (frozen continuation control), matched +5k eff-48 shared fresh seed 2, primary Delta_joint paired CI, conditional 10k extension, adoption bar -0.3, drift band 0.3 vs 60k 5.8602; committed ~32 GPU-h ceiling 35 (extension -&gt; global 70), J rate anchored on K measured 3.782 s/step; 4x-cost burden argued up front (bounded final phase, not a lineage). Code audit: --init-from does the warm-start; instrument gaps named (composite materializer, naive-joint guard escape narrowly scoped, AR-view compat, J-config memory smoke). Finalize+execute split to idea4-fjoint-rung-finalize-exec. · [pre-reg](posts/2026-08-09-prereg-fjoint-rung.md)

<details><summary>full record</summary>

#4 F-then-joint escalation rung — pre-reg DRAFT (CPU; the APT-named recipe, papers/apt-expert-pretraining.md): warm-start a joint run (unfrozen trunk, seam per readout) from the F arm's converged expert instead of from noise; APT grid says +8..+26 pts over frozen in their regime, and an F~K tie makes this the next discriminating contrast (initialization, not seam). Draft ONLY after the seam screen reads out: matched steps/effective batch vs the screen arms, Delta vs both F and K endpoints, trunk-drift band inherited from K's read 4

</details>

---

**`lit-slice-vla-initialization`** · `cpu`

Standing lit slice: the VLA-initialization thread APT opened (radar hooks banked 2026-08-07): 2605.25802 (Rethinking VLM Representation for VLA Initialization) + 2601.03309 (VLM4VLA)

**boundary:** CPU work at any GPU-busy window; most valuable before any F-then-joint escalation arm is drafted (post-screen, ~08-09+)

<details><summary>full record</summary>

Standing lit slice: the VLA-initialization thread APT opened (radar hooks banked 2026-08-07): 2605.25802 (Rethinking VLM Representation for VLA Initialization) + 2601.03309 (VLM4VLA) — does the init-not-seam diagnosis replicate outside APT's group, and does either name a cheap probe for our residual-tap surface? Feeds #4 escalation map (F-then-joint rung) + #17 trunk mandate. PERMANENT RULE: papers page(s) land the same session — EXECUTED SAME SESSION 2026-08-07 ~11:5xZ (pulled forward, GPU-busy window): one-pass reads + theme page landed (papers/vla-initialization.md). VLM4VLA: frozen VISION ENCODER is the published frozen-trunk failure mode (4.057-&gt;2.823 Calvin; VQA benches poorly predict VLA rank; embodied-VQA mixes all underperformed) -&gt; #17 trunk criterion + the F-loses diagnostic (vision-limited frames first) with the caveat stated (our trunk is embodiment-adapted pre-freeze). 2605.25802: LoRA &gt; full-FT for init; reconciled with APT (what shaped the gradients matters, not whether the trunk moves). ideas.md #4+#17 bullets

</details>

---

**`leaderboard-decode-cost-microbench`** · `gpu-local`

Leaderboard compute-column micro-benchmark PREP (owner steering 08-07 10:04Z follow-up; CPU now): write the timing script + its short pre-reg post

**boundary:** GPU run executes when draws10_t1 frees the local GPU (boundary ~12:3x-12:5xZ 2026-08-07), before the next local launch; consistency anchor: ar_greedy batched ~88.7 ms/frame · [pre-reg](posts/2026-08-07-prereg-leaderboard-decode-microbench.md)

<details><summary>full record</summary>

Leaderboard compute-column micro-benchmark PREP (owner steering 08-07 10:04Z follow-up; CPU now): write the timing script + its short pre-reg post — one script, same batch size / workers / dtype, times every leaderboard decode config (AR greedy, AR draws10-T1, teacher heun30 draws{1,10}, student 1-NFE draws{1,5,10}) on a fixed panel slice on the local 1xH100, plus single-stream batch=1 latency per config as the deployment-facing number (#16 hook); replaces the mtime-derived ~= ms/frame leaderboard entries with apples-to-apples numbers. Record-only, no headline claims; the ~15-min GPU run executes at the boundary under the posted pre-reg — PREP DONE 2026-08-07 ~11:0xZ: leaderboard_decode_microbench.py landed (7 configs x {batched b32/w20 N320, single b1/w4 N50}, decode flags byte-matched to banked stems, progress-line rate rule excludes startup/warmup, watchdog 30 min/run) + --selftest oracle PASS (exact rate arithmetic, 4 guard aborts, parser fixtures) + pre-reg posted (2026-08-07-prereg-leaderboard-decode-microbench.md). REMAINING: the ~30-min GPU run at the draws10_t1 boundary, before any next local launch — EXECUTED 08-07: pre-merge 14-cell baseline + redo cell + post-merge reruns; leaderboard measured-column rewrite + main-sync post tables live

</details>

---

**`lit-slice-attachment-frontier-pre-endpoint`** · `cpu`

Standing lit slice (~20-30 min, owner allocation 2026-08-05) targeted at the #4 attachment/seam frontier BEFORE the molmo2 endpoint stage-2 attachment decision (~08-08, carries the pi0.5 deep-read's two named arms): sweep for new…

**boundary:** CPU work at any GPU-busy window; highest value before the ~08-08 stage-2 attachment decision opens

<details><summary>full record</summary>

Standing lit slice (~20-30 min, owner allocation 2026-08-05) targeted at the #4 attachment/seam frontier BEFORE the molmo2 endpoint stage-2 attachment decision (~08-08, carries the pi0.5 deep-read's two named arms): sweep for new seam-recipe / knowledge-insulation / co-training evidence since the last radar pass (Anchor-Align and Wall-OSS-0.5 are the current poles; anything that re-ranks K-vs-F lands before the decision, not after) — PERMANENT RULE (owner 08-07 08:42Z): the slice lands its Papers-section page(s) on the blog in the SAME session; the ideas.md one-liner is the index hook only — EXECUTED 2026-08-07 ~11:2xZ: APT (2606.12366) deep-read, papers page landed SAME SESSION per the permanent rule (papers/apt-expert-pretraining.md): seam damage located in the expert's RANDOM INIT (language-imbalance shortcut); with a pretrained expert the best published recipe unfreezes everything with NO stop-grad (98/84/92/58 vs KI+pretrain 96/74/90/62) — K's seam corroborated for our random-init regime, screen unchanged pre-readout; NEW named escalation rung F-then-joint (warm-start joint from the F checkpoint's expert = free Stage-1 capital); F-tie-K gains a published interpretation. ideas.md #4 bullet + index/SUMMARY rows. Radar hooks banked unread: 2605.25802, 2601.03309

</details>

---

**`endpoint-runbook-git-audit`** · `cpu`

Pre-endpoint runbook git-audit (CPU, integrity class — the audit-queue-items-against-git standing practice applied BEFORE the highest-stakes window opens): for every blocked endpoint-chain item (idea19-molmo2-draws-arm, selection…

**boundary:** run before the molmo2_ar40k endpoint (~2026-08-08) opens the box chain; CPU work at any GPU-busy window

<details><summary>full record</summary>

Pre-endpoint runbook git-audit (CPU, integrity class — the audit-queue-items-against-git standing practice applied BEFORE the highest-stakes window opens): for every blocked endpoint-chain item (idea19-molmo2-draws-arm, selection-ceiling + energy-score reads, tsens chain, idea4-attach-k-smoke-ladder, idea4-attach-screen-execution), verify at HEAD: referenced scripts exist; launcher output stems == read-script defaults == babysit.toml prepared entries (the stem-vs-reader disagreement class the %g tsens fix caught); prepared pgrep patterns match the launchers' actual process names; every flag the launchers pass still exists in the bijou CLIs (--help grep). Output: short audit note (now.md + Discord); any mismatch becomes its own fix item BEFORE the endpoint opens — EXECUTED 2026-08-07 ~10:4xZ, CLEAN at HEAD 3d9e2a2: all scripts exist; molmo2-draws/tsens/attach stems byte-match read defaults + prepared babysit entries; pgrep patterns match actual cmdlines; every launcher flag exists in bijou.train/eval + gate CLIs (--help cross-check); smoke ladder = K recipe verbatim modulo intended smoke deltas; --arm-steps 10000 hardcode safe (gate only in the STEPS==10000 branch). Zero fix items; 2 harmless nuances recorded (tsens prune-ordering already documented, tsens pgrep_min comment mislabel)

</details>

---

**`idea19-tsens-dt-read`** · `cpu`

#19 dT diagnostic read script (CPU now, RECORD-ONLY per the pre-reg sensitivity clause

**boundary:** script landed; the read runs only after the tsens rungs land (which gate on the primary landing inside 24 GPU-h, boundary ~13Z 2026-08-07) · [pre-reg](posts/2026-08-06-prereg-ar-sampled-draws.md)

<details><summary>full record</summary>

#19 dT diagnostic read script (CPU now, RECORD-ONLY per the pre-reg sensitivity clause — never a headline, never a license to re-pick T): pool chunk/first for each tsens rung (stateprobe_q4_draws10_t{0.5,0.7,1.3} stems) against the T=1.0 primary re-pooled onto the same q4 rows — AUDIT FIRST: draws10_t1_results.py's join_rows subset machinery + box_batch_results pooling are the reusable core, but its loaders hard-pin ar_temperature 1.0 + _draws10_t1 suffix, so the delta is a T-parameterized sibling loader (registered T set {0.5, 0.7, 1.0, 1.3} only), NO decision branches — one dT table, record-only; oracle: a synthetic T=1.0 rung fixture must reproduce the primary's q4 re-pool exactly; guards fire on unregistered T, wrong plan, wrong draws, policy/stem tag mismatch — LANDED 08-07 ~10:0xZ (tsens_dt_results.py: T-parameterized sibling loader over the registered set {0.5, 0.7, 1.0, 1.3} only, one record-only dT table (pooled chunk/first per T on the same q4 rows, T=1.0 re-pooled from the full-panel primary via join_rows subset join), NO decision branches; oracle PASS pre-data: synthetic T=1.0 rung fixture reproduces the primary's q4 re-pool EXACTLY (float-equal), x0.93/x0.98/x1.07 rungs land at exactly factor x re-pool, 11 guard aborts fire (unregistered T, wrong plan/draws/ar_temperature, policy+stem tag mismatch, rung-row disagreement, full-panel-as-rung, state-copy drift, checkpoint mismatch, report drift); defaults = the tsens launcher's exact stems)

</details>

---

**`papers-section-retroactive`** · `cpu`

OWNER STEERING 08-07 08:42Z, HIGH PRIORITY: new Papers section on the blog

**boundary:** DONE 2026-08-07: permanent page-per-slice rule remains in force for every future lit slice

<details><summary>full record</summary>

OWNER STEERING 08-07 08:42Z, HIGH PRIORITY: new Papers section on the blog — one post per paper (grouped/cross-linked by theme), covering contribution, experiments run, what transfers to us and what doesn't, which idea/arm it fed; written for a reader with less context, a pleasure to read. RETROACTIVE: review every lit slice so far, re-read papers deeply where notes are thin, add pages for all. Permanent page-per-slice rule landed 08-07 tick. BATCH 1 LANDED 08-07 ~09:1xZ (44eb032): section + index/tracker + 8 pages / 16 papers (pi0.5+KI, LabVLA, Q-VGM, 7-paper selection cluster, SnapFlow, AEGIS+Wall-OSS seam debate, encoder-grafting, Hi-VLA+CAC-VLA) — deep re-reads surfaced 2 correction hooks banked to ideas.md (#4 Wall-OSS stop-grad-worst ablation; #19 probe-selector rollout caveat). BATCH 2 LANDED 08-07 ~09:4xZ: 3 more theme pages / 13 papers (one-step menu OFP+MeanFlow+LetItBeSimple+GoldenStart; sampling-beyond-selection GoldenTicket+DVAC+EnergyPolicy; state-shortcut set of 6) — 29 covered; deep re-reads surfaced 3 correction hooks banked to ideas.md (#9 p=0.8 was the BASELINE of a WITHDRAWN paper, not the method; #1 GoldenTicket v3 46/51 + shared-vs-per-task ticket fix; #12 MeanFlow speed-for-accuracy + LetItBeSimple state-carried caveat). REMAINING 13 by the index tracker: grounding set (IVRA/FLOWER/SCALE/SmolVLA), data/tokenization/trunks set (6), AR-VLA + representation-anchoring + pi0.7/WAM; BATCH 3 LANDED 08-07 ~10:0xZ: 4 final theme pages / 13 papers (grounding-conditioning IVRA+FLOWER+SCALE+SmolVLA; action-tokenization FAST+FASTer; data-and-trunks Rethinking+survey+redundancy+LoRA; attachment-frontier AR-VLA+Anchor-Align+WAM post) — RETROACTIVE BACKLOG CLEARED, 42/42 sources covered. Deep re-reads surfaced 7 correction hooks banked to ideas.md, incl. 2 loud ones: the data-engine survey contains ZERO dedup content (we projected our census onto it) and 2606.31382 makes no backbone-scale claim (belongs to VLM4VLA). Permanent page-per-slice rule continues per charter.

</details>

---

**`idea19-endpoint-fairness-es-read`** · `cpu`

#19 energy-score + fairness reads on the molmo2 endpoint draws dump (CPU script now, exploratory record-only

**boundary:** script is CPU work at any GPU-busy window; the read runs after the molmo2 endpoint draws arm lands its npz (~2026-08-08)

<details><summary>full record</summary>

#19 energy-score + fairness reads on the molmo2 endpoint draws dump (CPU script now, exploratory record-only — the pre-declared draws_fairness reads were registered for the FLOW probe; applying them to the AR endpoint dump is a diagnostic, not a registered claim): wire draws_fairness.py's read 1/2/4 machinery (mean-of-draws, best-of-N, energy score with the AR-as-degenerate-N=1 baseline) to the endpoint _draws.npz stems — the strictly-proper-scoring-rule comparison AR-vs-flow comes free from banked data; AUDIT FIRST: selection_ceiling_results.py already covers mean/best/dispersion — extend ONLY the energy-score delta + flow-side comparison; oracle: degenerate draws=1 -&gt; interaction term exactly 0, ES == direct RMS-L2 (the draws_fairness --validate pattern on the AR-100k npz) — LANDED 08-07 ~09:0xZ (energy_score_results.py: ES delta vs the degenerate N=1 greedy baseline + paired per-frame CI, flow-side comparison via index-join to the banked drawsprobe_s7 stack (both families same instrument on identical frames); audit honored — mean/best/dispersion left to selection_ceiling_results.py, ES only; oracle PASS pre-data: degenerate draws=1 -&gt; interaction exactly 0 + ES == direct RMS-L2, banked read-4 numbers (5.930763/9.882476/3.951713/8.769585) reproduced exactly through this file's join+pooling, N=2 hand fixture exact, 5 abort guards; defaults = the endpoint launcher's exact stems, tsens q4 dumps via explicit paths)

</details>

---

**`idea19-t-sens-launcher-script`** · `cpu`

#19 T-sensitivity rung launcher script (CPU now): the pre-registered RECORD-ONLY rung as one command

**boundary:** script is CPU work now; execution opens after the draws10_t1 boundary (~13:0x-13:3xZ 2026-08-07) and ONLY if the primary landed inside its gate; needs a quiet local-GPU window · [pre-reg](posts/2026-08-06-prereg-ar-sampled-draws.md)

<details><summary>full record</summary>

#19 T-sensitivity rung launcher script (CPU now): the pre-registered RECORD-ONLY rung as one command — T in {0.5, 0.7, 1.3} at draws10 on the frozen q4 subset (plans/holdout_curated_v0_k4l2_stateprobe_q4.json, 4301 rows), local GPU, stateprobe_q4_draws10_tT stems, report-samples 0, GPU-free guard; runs ONLY if the primary draws10_t1 landed inside the 24 GPU-h gate (pre-reg cost clause, ~12 GPU-h worst case, 3 sequential rungs); quoted as a dT diagnostic, never a headline, never a license to re-pick T post hoc; reads re-pool through draws10_t1_results.py's q4 subset path — LANDED 08-07 ~08:3xZ (eval_ar100k_tsens_q4_draws10.sh: primary-gate precondition MECHANIZED — full-panel report exists + registered semantics + elapsed GPU-h from babysit started_utc &lt;= 24.0, all 5 abort branches oracle-checked incl. negative-elapsed; q4 plan sha-pinned; per-rung skip-if-banked; --dump-draws retention per the endpoint precedent; tT tags match the policy %g suffix; babysit ar100k_tsens_q4 entry prepared, gate 12 GPU-h)

</details>

---

**`idea19-selection-ceiling-read-script`** · `cpu`

#19 selection-rung ceiling read script (CPU now, banked lit rung

**boundary:** script is CPU work now; the read runs after the molmo2 endpoint draws arm lands its npz (~2026-08-08); AR-100k draws10_t1 retained only pooled predictions (accepted 08-07) so this read is molmo2-endpoint-first

<details><summary>full record</summary>

#19 selection-rung ceiling read script (CPU now, banked lit rung — NOT pre-registered, exploratory read only, any escalation needs its own pre-reg): oracle best-of-10 ceiling per-frame from the molmo2 endpoint draws --dump-draws npz (the retention fixed 08-07 ~06:1xZ) — bounds what ANY selector (MG-Select / VLA-ATTC / CoVer flavors) could buy before building one; AUDIT FIRST per standing rule: draws_fairness.py already computes best-of-N on flow probe npz — verify its npz/stems contract against the AR draws dump and extend only the delta; oracle-gate on synthetic per-draw fixtures (known best-draw pattern in/out) + degenerate draws=1 must reproduce the greedy number — LANDED 08-07 ~08:1xZ (selection_ceiling_results.py: exact order-statistic best-of-K ladder K=1..10 + greedy/ensemble headroom + first mirrors + selector diagnostics (argmin uniformity, dispersion-vs-gain quartiles); oracle PASS pre-data incl. brute-force subset enumeration, planted best-draw pattern, degenerate draws=1 -&gt; 5.8026/2.1431 anchor, 5 abort guards; defaults = the endpoint launcher's exact stems)

</details>

---

**`draws10-t1-read-script`** · `cpu`

draws10_t1 frozen-read script — SCRIPT LANDED 2026-08-07 ~07:5xZ (draws10_t1_results.py: reads 1-5 one command, oracle-gated all branches + 11 hard-abort guards; q4-fallback index join + molmo2-arm explicit paths supported)

**boundary:** wanted BEFORE the draws10_t1 boundary ~13:0x-13:3xZ 2026-08-07 (the draws10-frozen-reads item consumes it at completion) · [pre-reg](posts/2026-08-06-prereg-ar-sampled-draws.md)

<details><summary>full record</summary>

draws10_t1 frozen-read script — SCRIPT LANDED 2026-08-07 ~07:5xZ (draws10_t1_results.py: reads 1-5 one command, oracle-gated all branches + 11 hard-abort guards; q4-fallback index join + molmo2-arm explicit paths supported)

</details>

---

**`molmo2-stage2-attachment-decision`** · `cpu`

Molmo2 stage-2 attachment decision at endpoint — now EXECUTABLE via the seam-screen pre-reg (2026-08-07-prereg-molmo2-attach-screen.md: frozen vs KI-joint is the first measurement; the depth-of-reads arm stays open for its own sc…

**boundary:** STEER WINDOW CLOSED 04:57Z 08-09 into the named default (posted 04:42Z, Discord read clean at 04:56Z boot): arm F LAUNCHED 04:57:51Z. The decision itself now lands at the screen readout: Delta_seam frozen read (attach_seam_results.py) after BOTH arms + their panel_v2 evals complete; blocked on idea4-attach-screen-execution | RE-SCOPED 12:4xZ 08-09 (owner killed K): decision basis = F arm only (panel_v2 banked, state-copy beaten decisively) + production frozen-first votes (RDT2, Qwen-VLA) — no Delta_seam number. Write the decision memo from banked artifacts (CPU); if a seam number is ever wanted, pre-register a CHEAPER matched read (short-K or CE-probe-only) as a separate item. AFTER the docs pass (owner prio). | DONE 13:5xZ 08-09: decision memo posted (posts/2026-08-09-molmo2-stage2-attachment-decision.md) — FROZEN DEFAULT STANDS for the Molmo2 trunk class; KI-joint closed-unmeasured (F panel 9.4157 vs state-copy 11.7639; 8 matched probes K-F mean +0.208; measured 4.11x step cost; RDT2 + Qwen-VLA frozen-first votes). Delta_seam@3750 rescue read priced ~2.5 GPU-h, needs own pre-reg. Chart + ledger + ideas.md updated.

<details><summary>full record</summary>

Molmo2 stage-2 attachment decision at endpoint — now EXECUTABLE via the seam-screen pre-reg (2026-08-07-prereg-molmo2-attach-screen.md: frozen vs KI-joint is the first measurement; the depth-of-reads arm stays open for its own screen); owner steer window before the screen launches

</details>

---

**`molmo2-60k-html-panel-report`** · `cpu`

Molmo2 60k HTML panel report backfill (OWNER STANDING RULE 08-09 03:55Z)

**boundary:** CLOSED 04:0xZ 08-09 (same session as the K-smoke ladder launch it was to ride); MAE oracle satisfied by the banked json itself (no re-run, no drift possible); owner correction posted in-channel · [pre-reg](posts/2026-08-08-prereg-molmo2-ar-60k-continuation.md)

<details><summary>full record</summary>

Molmo2 60k HTML panel report backfill (OWNER STANDING RULE 08-09 03:55Z) — CLOSED AT ZERO GPU-H 04:0xZ 08-09: the planned ~1 GPU-h --report re-run was UNNECESSARY — pre-launch audit found the 60k chained eval DID run with --report (the launcher always had it, launch_box_fontaine_molmo2_ar_60k_resume_ddp4.sh line 92); the 8.6 MB panel HTML (+ the 9.2 MB fields-run HTML) sat unsynced on the box since 23:49Z/00:49Z 08-08. Both synced local, verified (checkpoint ref + banked MAE 5.86022663460471 in json; html mtime == json mtime, same writer process), uploaded to Space reports/, linked from the @60k section (stale 'ran without --report' caveat replaced with a correction note). The audit-queue-items-against-git practice win: 'no HTML exists' was asserted from the local reports/ dir without checking the box

</details>

---

**`idea4-attach-k-smoke-ladder`** · `gpu-box`

#4 K-arm smoke memory ladder script — SCRIPT LANDED 08-07 ~06:5xZ (smoke_attach_k_ddp4.sh: exact K recipe verbatim incl

**boundary:** DONE 2026-08-09 04:39:33Z — RUNG 1 (B12c6) GREEN AT FIRST TRY on attempt 2 (unit fontaine-attach-ksmoke at 913fdc4): rc=0, vram_alloc_peak 57.34 GiB &lt;= 71 gate, nvidia-smi peak 63887 MiB &lt;= ~75000 advisory, s/step(last5) 5.675; FULL BATCH B12c6, no downshift. True cost ~0.5 GPU-h (incl. attempt-1 #20 crash) &lt;= 6 gate. k_mem_ready written on box + rsynced to local fontaine/harness/state/ 04:4xZ. Ladder projection: K 10k ~63.1 of 70 GPU-h batch gate (advisory; attach_rate_gate.py binds at launch). Attempt-1 crash = idea #20 CUDA bug, fixed 913fdc4 same session. -&gt; owner steer window molmo2-stage2-attachment-decision NOW OPEN · [pre-reg](posts/2026-08-07-prereg-molmo2-attach-screen.md)

<details><summary>full record</summary>

#4 K-arm smoke memory ladder script — SCRIPT LANDED 08-07 ~06:5xZ (smoke_attach_k_ddp4.sh: exact K recipe verbatim incl. --activation-checkpointing, 150 steps/rung w/ eval+save exercised, rungs B12c6-&gt;B8c4-&gt;B6c3 at pinned chunk-microbatch 2, pass = rc0 AND max vram_alloc_peak_gib &lt;= 71.0 from the jsonl, nvidia-smi peak advisory; green writes fontaine/harness/state/k_mem_ready record + echoes the K_MEM_READY=1 BATCH/BACKWARD_CHUNKS launch line; sub-B12 green = MATCHED DOWNSHIFT both arms loudly echoed; all-red = no marker, owner steer). REMAINING: RUN the ladder on the box after it frees

</details>

---

**`lit-slice-attach-window-0809`** · `cpu`

Standing lit slice (~20-30 min, owner allocation 2026-08-05) in the attach_F train window: web/arXiv sweep for ideas worth trying; current open hooks = FlowDAgger follow-ups (latent-DAgger family), anything re-ranking the #4 F-vs…

**boundary:** EXECUTED same session as queued (attach_F window)

<details><summary>full record</summary>

Standing lit slice (~20-30 min, owner allocation 2026-08-05) in the attach_F train window: web/arXiv sweep for ideas worth trying; current open hooks = FlowDAgger follow-ups (latent-DAgger family), anything re-ranking the #4 F-vs-K readout before it lands, #17 trunk/init radar. PERMANENT RULE (owner 08-07 08:42Z): papers page(s) land on the blog SAME session; ideas.md line is the index hook only — EXECUTED 05:1x-05:3xZ 08-09: Hy-Embodied-0.5-VLA 2606.14409 deep-read + papers page SAME SESSION (hy-embedded... hy-embodied-stack.md) — full-stack blueprint: FlowPRO preference RL on flow policies (flow loss as implicit reward, intervention-and-rollback pairs, +6-12 pts over DAgger, 94-99% SR, retention UNMEASURED = FlowDAgger critique stands) + H=50 Bezier chunk-stitch async deployment + 10k-h UMI human data. Fed #16 (weight-space pole of post-SFT menu + deployment lever), #4 (joint-pole ledger entry under APT's pretrained-VLM condition, 11:1 expert sizing). Dup-check win: VLAFlow 2607.01586 re-surfaced by search, caught as ALREADY covered (5433814) before writing. Radar hooks banked unread: FASTER 2603.19199 (real-time flow VLAs), ABPolicy 2602.23901 + DEFLECT 2605.19294 (async execution family), RDT2 2602.03310 (UMI scaling), QDepth-VLA 2510.14836 (depth aux)

</details>

---

**`idea4-attach-screen-execution`** · `gpu-box`

#4 attachment seam screen execution (box, 4xDDP, sequential F then K): F frozen-trunk vs K KI-joint (phase-1 CE verbatim + stop-grad seam, alpha=1) at matched 10k steps / eff-48 from the 40k endpoint; residual surface constant; g…

**boundary:** ARM F LIVE since 04:57:51Z 08-09 (unit fontaine-attach-f, box 6be4e8e, B12c6 from the 60k endpoint). Rate gate PASS 05:05Z (50.3 projected &lt;= 70, full 10k). Kill-bar judgments at save boundaries: @5000 PASS 10.2595 vs 12.6394 (phase-1 matched +0.62), @7500 PASS 9.9391 vs 11.6356 (phase-1 matched +1.30); @10000 bar 10.1652. Rate ~0.92 s/step, 62-63 steps/min effective; vram 19.05 GiB. Async-save validated live at 1250. F endpoint ~07:4xZ -&gt; chained panel_v2 eval in-unit -&gt; THEN launch K (K_MEM_READY=1 BATCH=12 BACKWARD_CHUNKS=6, EXTRA_GPU_HOURS recomputed from F actual ~10.2 train + evals; babysit attach_K entry at launch). Frozen reads after both arms | F TRAIN COMPLETE 07:42:08Z (all bars passed, endpoint probe 9.3798@10000); panel_v2 eval live 5645 frames ~1.6h (done ~09:2xZ), eval gate raised 6-&gt;8 judged CONTINUE (session estimate vs pre-reg EXTRA budget). F expert capital uploaded to fontaine-checkpoints same-session (expert+prompt+config+README; backbone DEDUPLICATED — byte-identical sha e6ed78 to the 60k trunk, freeze verified at upload). K LAUNCH = next session first action once box frees | F panel_v2 eval COMPLETE 08:01:0xZ 08-09 (~1.24 GPU-h actual — scoring hit ~457 f/min steady-state, the 6.4 estimate was load-phase-contaminated; json/npz/html banked on box, NOTHING read alone). ARM K LAUNCHED 08:01:19Z same tick (unit fontaine-attach-k, K_MEM_READY=1 B12c6, EXTRA_GPU_HOURS=17 from F actuals 10.2 train + 1.25 eval + K eval ~2 + AR panel ~3); in-launcher rate gate binds at first jsonl window (rc 2 =&gt; matched 5k downshift both arms, F re-evals step_005000). Babysit attach_K entry live (3 probe bars + vram 71 + CE-health watch) | OWNER KILL 12:31:43Z 08-09: 'way too slow per step' — attach_K stopped 12:38Z at step ~4160/10k (3.74 s/step vs F's 0.92, ~4x; unit fontaine-attach-k stopped, box GPUs 0 MiB x4, checkpoints through step_003750 retained on box, NOT uploaded — partial arm, nothing consumes it). NO K endpoint =&gt; Delta_seam matched read and read-4 AR-view drift are OFF under this pre-reg. Screen closes on F evidence: all F bars passed, F panel_v2 banked, F expert capital uploaded. ~13.6 GPU-h spent on K (08:01-12:38Z). · [pre-reg](posts/2026-08-07-prereg-molmo2-attach-screen.md)

<details><summary>full record</summary>

#4 attachment seam screen execution (box, 4xDDP, sequential F then K): F frozen-trunk vs K KI-joint (phase-1 CE verbatim + stop-grad seam, alpha=1) at matched 10k steps / eff-48 from the 40k endpoint; residual surface constant; gates vram&lt;=71, K1-style probe kill (phase-1 curve + 3.0 at &gt;=5k; bars 12.6394@5000, 11.6356@7500 in babysit.toml prepared entries), 70 GPU-h ceiling w/ matched 5k downshift; frozen reads Delta_seam paired CI + K trunk-drift band 0.3 (READ SCRIPT LANDED 08-07 ~07:2xZ: attach_seam_results.py, one command, oracle-gated all branches) — INSTRUMENT LANDED 08-07 ~06:0xZ; LAUNCH PREP LANDED 08-07 ~06:1xZ (launch_box_fontaine_molmo2_attach_{F,K}_10k_ddp4.sh: mechanized attach_rate_gate.py 70 GPU-h gate + 5k-downshift marker both launchers honor, sha256-pinned plans, chained panel_v2 evals; K chains materialize_joint_ar_view.py + greedy k4l2 drift panel; K_MEM_READY guard refuses blind K launch; 10 new oracles, check.py 433)

</details>

---

**`owner-docs-pass-0809`** · `cpu`

OWNER STEERING 12:28:59Z 08-09, TOP PRIORITY: docs/ modernization pass ahead of owner's main-rebase

**boundary:** DONE 13:3xZ 08-09 (e7144c3): README two-trunk + fontaine-vs-shared split; architecture.md modernized (intro/S1/S2/S5/S7/S8 — Molmo2 everywhere it belongs, curated-plan ledger, shipped-flag demotions, CLI-default corrections 768/15/4-4-7, residual+seam+snapflow documented); molmo2/gemma4/styleguide/working-together/rollout/init_gpu fixed; 4 historical docs got archive headers. Staleness audit by subagent against HEAD drove the pass. Remaining tail split to docs-pass-followups-0809.

<details><summary>full record</summary>

OWNER STEERING 12:28:59Z 08-09, TOP PRIORITY: docs/ modernization pass ahead of owner's main-rebase — (a) update flow-matching/docs (architecture.md and friends) to reflect the CURRENT codebase + trained models, written in standard ML language for an ML expert, NO internal vocabulary (rungs/panels/idea-numbers/arm codenames); (b) repo README: clear statement that fontaine/... is an autonomous research agent's harness/notes/blog and the rest is the shared codebase (owner will rebase main on fontaine and develop with local agents); (c) tech-debt sweep at my discretion (stale docs, dead scripts/flags, confusing leftovers). Acknowledged in-channel 12:40Z with plan; post links when it lands for owner review pre-rebase.

</details>

---

**`owner-molmo2-adamc-run-prep-0809`** · `cpu`

OWNER STEERING 12:37:56Z 08-09, TOP PRIORITY ('let's start with it'): new molmo2 run from BASE 4B

**boundary:** DONE 13:3xZ 08-09, RUN LAUNCHED: AdamC implemented 401d6f7 (10 oracles, check.py 584 green; partition corrected/head/no-decay + tied-param guard both optimizer modes); parameter sheet posted (2026-08-09-prereg-molmo2-adamc-100k.md) + Discord 13:13Z; owner approvals 13:19Z (text+vision 2e-5, seed 1, save 5000, NO smoke) + lambda override 13:24Z (1e-5 lineage value; first launch at 0.01 stopped PRE-STEP-1 13:29Z, relaunched 13:30Z). Unit fontaine-adamc-100k live, banners verified (E1 exact, vision 439.1M live, AdamC lambda=1e-05, 4074.7M corrected / 2.6M head / 0.6M undecayed). Babysit adamc_100k entry live (260 GPU-h gate, 77 GiB near-OOM watch, K1-style probe bars). Endpoint ~08-11/12 -&gt; chained k4l2 panel eval.

<details><summary>full record</summary>

OWNER STEERING 12:37:56Z 08-09, TOP PRIORITY ('let's start with it'): new molmo2 run from BASE 4B — spec: (1) 100k steps; (2) effective batch 32 = 8/rank x 4; (3) vision encoder UNFROZEN from step 0, --{backbone,text}-vision-lr 2e-5; (4) warmup 1000; (5) optimizer AdamC per arxiv 2506.02285v1 (AdamW with time-varying per-group decay coefficient) — implement AdamC FIRST, efficiently, mindful of shared/tied layers (Gemma tied lm_head); owner-supplied context conversation https://claude.ai/share/52f07abb-4b00-48f0-9c62-6627868b5209 must be read as part of implementing. GATE: before ANY launch, post an in-depth description of ALL run parameters for owner approval.

</details>

---

**`idea19-molmo2-draws-arm`** · `gpu-box`

#19 molmo2 sampled-draws arm execution at the 40k endpoint (pre-reg'd in ar-sampled-draws post: greedy + _draws10_t1 same stems, same cost gate w/ q4 fallback)

**boundary:** closed 2026-08-08 07:3xZ · [pre-reg](posts/2026-08-06-prereg-ar-sampled-draws.md)

<details><summary>full record</summary>

#19 molmo2 sampled-draws arm execution at the 40k endpoint (pre-reg'd in ar-sampled-draws post: greedy + _draws10_t1 same stems, same cost gate w/ q4 fallback) — instrument landed 78c9f56 (08-06), molmo2 trunk-specific oracles landed 08-07 ~04:3xZ (tests/test_molmo2_ar_sampling.py: T→0 greedy recovery, snapshot/restore prefill-sharing over Molmo2KVCache, append-only contract, ar_predict_sampled dispatch); the stale 'instrument + pre-reg draft' framing closed — both existed since 08-06; endpoint launcher + mechanized cost gate landed 6c3cc3b (one command: eval_box_molmo2_endpoint_draws10_t1.sh, greedy-if-missing + draws10_t1 + q4 fallback, babysit entry prepared) ; selection-ceiling read script LANDED 08-07 ~08:1xZ (selection_ceiling_results.py, one command over this arm's _draws.npz dump) — LANDED 08-08 07:22Z rc=0 (full panel, no q4 fallback; ~2.5 h ~10 GPU-h &lt;= 24 gate); frozen reads ALL EXPECTATIONS MET 07:3xZ via draws10_t1_results.py explicit paths: Delta_AR -0.15422 [CI95 -0.19484, -0.11319], mean-collapse shape replicated (AR-100k -0.145), no flow-band overtake (5.8492 vs 5.365), execution oracles byte-green -&gt; leaderboard row 9; analysis banked reports/analysis__draws10_t1_molmo2_40k_k4l2.json

</details>

---

**`idea6-selfsubgoal-probe`** · `gpu-local`

#6 rung-(a) self-subgoal probe COMPLETE 02:37Z 08-08 (~3.2 GPU-h &lt;= 8 gate): arms rc=0 (oracle full panel + marker-gated self two-pass; narrated free from pass 1); READ OUT same session

**boundary:** closed 2026-08-08 02:37Z (arms) / 02:5xZ (reads); babysit entry pruned same commit · [pre-reg](posts/2026-08-07-prereg-selfsubgoal-probe.md)

<details><summary>full record</summary>

#6 rung-(a) self-subgoal probe COMPLETE 02:37Z 08-08 (~3.2 GPU-h &lt;= 8 gate): arms rc=0 (oracle full panel + marker-gated self two-pass; narrated free from pass 1); READ OUT same session — Delta_oracle -0.290 [-0.331,-0.225] slot ALIVE 6x late-horizon; Delta_self -0.018 CI spans 0 = no deployment win at 3x decode; channel narr-self +0.043 CI excl 0 (slot right, generation phase-estimation is the bottleneck). Results post 2026-08-08-selfsubgoal-results.md

</details>

---

**`draws10-frozen-reads`** · `gpu-local`

draws10_t1 frozen reads (delta_AR vs 5.8026, fairness vs -1.258, family vs 5.365) + T-sensitivity rung after

**boundary:** opens at draws10_t1 completion (~13:0x-13:2xZ 2026-08-07); reads are frozen in the pre-reg · [pre-reg](posts/2026-08-06-prereg-ar-sampled-draws.md)

<details><summary>full record</summary>

draws10_t1 frozen reads (delta_AR vs 5.8026, fairness vs -1.258, family vs 5.365) + T-sensitivity rung after — READ SCRIPT LANDED 2026-08-07 ~07:5xZ (draws10_t1_results.py, one command, oracle-gated; defaults = the local launcher's exact stems) — EXECUTED at the 12:2x boundary tick; leaderboard + ledger rows landed 08-07 ~15:0xZ work session

</details>

---

**`blog-ideas-refactor`** · `cpu`

OWNER STEERING 08-07 13:02Z: refactor Ideas — one page per idea (content copied as-is first), then an Ideas index/home page splitting hot (actively pursued) vs on ice (parked/older); index and pages updated regularly as the progr…

**boundary:** mechanical split + index in the 08-07 13:04Z session; per-page details audit may roll to a follow-up item if the merge chain preempts

<details><summary>full record</summary>

OWNER STEERING 08-07 13:02Z: refactor Ideas — one page per idea (content copied as-is first), then an Ideas index/home page splitting hot (actively pursued) vs on ice (parked/older); index and pages updated regularly as the programme moves; after the split, audit each idea page for up-to-date details — EXECUTED 08-07 work session (4f18582 + b6b5ff0 tags): 22 pages + hot/ice index + details audit (2 corruptions repaired, 4 stale pages refreshed); charter codified (bd1aea8)

</details>

---

**`blog-now-archive-sort`** · `cpu`

OWNER STEERING 08-07 13:02Z: sort the Now-archive per-date pages most-recent-first (SUMMARY sidebar + archive/index.md) and fix archive_now.py so every roll maintains the order automatically

**boundary:** execute in the 08-07 13:04Z work session's bench-wait window

<details><summary>full record</summary>

OWNER STEERING 08-07 13:02Z: sort the Now-archive per-date pages most-recent-first (SUMMARY sidebar + archive/index.md) and fix archive_now.py so every roll maintains the order automatically — EXECUTED 08-07 work session (4f18582): archive_now.py rebuilds sorted; files fixed

</details>

---

**`async-checkpoint-saves`** · `cpu`

OWNER STEERING 08-07 13:58Z, HIGH PRIORITY: async checkpoint serialization for bijou.train

**boundary:** DONE 08-07 before the attach-screen launch (~08-08) as targeted; verify async-save lines at that launch's first babysit

<details><summary>full record</summary>

OWNER STEERING 08-07 13:58Z, HIGH PRIORITY: async checkpoint serialization for bijou.train — molmo2 measures ~15.5 min/save (~14 min silent ZeRO-1 gather = consolidate_state_dict's serial whole-shard pickle broadcast over the TRAINING NCCL group, + 37 GB write) every ~92 min stepping = ~14% wall-time waste — LANDED 08-07 ~15:4xZ (e3bdc93), oracle-gated, DEFAULT-ON (--sync-save = legacy escape): bijou/async_save.py capture (device-&gt;CPU at boundary, seconds) + background gather_object over a DEDICATED gloo group + exact ZRO.state_dict() merge replica + atomic .tmp-dir rename publish; train.py save_checkpoint refactored capture/metadata/write (sync path now atomic too); final save joined before group teardown (endpoint chaining safe). Oracles green in check.py 446: 2-rank byte-identity keystone vs consolidate at consecutive boundaries w/ gather overlapping main-thread collectives (pickle-memoization of shared betas tuple + gather_object key-de-interning both pinned), dir-level byte-identity, weights_only resume round-trip, crash atomicity (no step dir published, .tmp debris only), loud background-failure surfacing. Attach-screen launchers pick it up with zero flag churn. NOTE: loop wiring validated by unit oracles only (no main() e2e exists repo-wide); first real-run validation = first save of the next training launch — check the 'captured in Xs' + 'saved ... (async, Xs behind the boundary)' lines at first babysit; representational note: non-zero1 GPU runs' optimizer.pt now stores CPU-tagged tensors (load path re-homes). | FIRST-REAL-RUN VALIDATION PASSED 05:2xZ 08-09 on attach_F step 1250: 'captured in 1.3s; gather+write continue in background' + 'saved .../step_001250 (async, 14.0s behind the boundary)', step dir published atomically, training continued through the save (step 1260 logged mid-write). The unit-oracles-only caveat is closed.

</details>

---

**`draws10_t1`** · `gpu-local`

AR-100k draws10 T=1 panel eval (local GPU) — COMPLETE 08-07 ~12:1xZ, frozen reads ALL EXPECTATIONS MET (delta_AR -0.14505 CI excl 0; ~9x smaller than flow gain; no overtake); leaderboard row 5 landed 08-07 work session

**boundary:** ~13:0x-13:2xZ 2026-08-07 (cumulative 32.2 f/min at 03:07Z -&gt; ~13.3 h total, re-projected each babysit; 24 GPU-h gate) · [pre-reg](posts/2026-08-06-prereg-ar-sampled-draws.md)

<details><summary>full record</summary>

AR-100k draws10 T=1 panel eval (local GPU) — COMPLETE 08-07 ~12:1xZ, frozen reads ALL EXPECTATIONS MET (delta_AR -0.14505 CI excl 0; ~9x smaller than flow gain; no overtake); leaderboard row 5 landed 08-07 work session

</details>

---

**`molmo2_ar40k`** · `gpu-box`

Molmo2 AR 40k trunk run (box, 4xDDP) — K1 GATE CROSSED GREEN 08-07 06:0xZ: probe 7.1652@10000 vs bar 12.0944 (margin 4.93, new low); run continues to 40k endpoint

**boundary:** closed 2026-08-08 ~04-05Z (endpoint + chained greedy panel; results in molmo2-endpoint-postprocessing). Status was stale 'live' until 16:5xZ — caught by the first Queue-page render · [pre-reg](posts/2026-08-06-prereg-molmo2-ar-40k.md)

<details><summary>full record</summary>

Molmo2 AR 40k trunk run (box, 4xDDP) — K1 GATE CROSSED GREEN 08-07 06:0xZ: probe 7.1652@10000 vs bar 12.0944 (margin 4.93, new low); run continues to 40k endpoint

</details>

---

**`ar100k-tsens-q4-live`** · `gpu-local`

#19 T-sensitivity q4 rungs COMPLETE 23:09Z 08-07 (3/3 rungs, 4301 rows each, ~7.2 GPU-h &lt;= 12 gate): dT table banked reports/analysis__tsens_dt_ar100k_q4.json (record-only)

**boundary:** closed 2026-08-07 23:09Z; babysit entry pruned same session · [pre-reg](posts/2026-08-06-prereg-ar-sampled-draws.md)

<details><summary>full record</summary>

#19 T-sensitivity q4 rungs COMPLETE 23:09Z 08-07 (3/3 rungs, 4301 rows each, ~7.2 GPU-h &lt;= 12 gate): dT table banked reports/analysis__tsens_dt_ar100k_q4.json (record-only)

</details>

---

**`molmo2-perf-pass1-subset-landing`** · `cpu`

perf pass-1 SUBSET landing (CPU; the frozen &lt;5% decision branch executed 02:26Z 08-09): build a P1-free subset from branch perf-pass1

**boundary:** LANDED 04:3xZ 08-09 (6a4b45e, the K-smoke attempt-2 wait window): clean cherry-pick of 22e8148 onto 913fdc4 (P1 not carried); bitwise oracle re-run HEAD-vs-subset 118/118; check.py 558 green. No speed claim per the frozen rule. NOT synced to the box until the live ladder finishes (box code frozen under a live run) · [pre-reg](posts/2026-08-08-prereg-molmo2-perf-pass1.md)

<details><summary>full record</summary>

perf pass-1 SUBSET landing (CPU; the frozen &lt;5% decision branch executed 02:26Z 08-09): build a P1-free subset from branch perf-pass1 — P2 windowed vram peak logging (metrics-only, proven live on bench_C window_peak 66.6) + P3a-c sync removals + P4 embed-clone drop (both bitwise-proven 118/118 at 22e8148, but their box speed effect was NOT measured alone: only the C-vs-B cross-read +3.2 pts suggests they recoup some of P1's -10.8%). Re-run perf_pass1_bitwise_oracle.py HEAD-vs-subset before the landing commit; check.py green. NO new bench needed per the frozen rule; any optional confirm rung would need its own pre-reg. NOTE: P1 (suffix MATH-&gt;cuDNN) is dead twice over (loss-bound oracle fail + -10.8% slower on the true recipe) — do not carry it

</details>

---

**`molmo2-perf-pass1-exec`** · `gpu-local`

molmo2 perf pass 1 EXECUTION (gpu-local, &lt;= 3 GPU-h): per the finalized pre-reg

**boundary:** CLOSED 02:26:32Z 08-09 rc=0: OVERLAY PASS (0.0816 &lt;= 0.3919 band); LADDER(BOX) A=2.251s B=2.495s C=2.415s -&gt; B -10.8% / C -7.3% vs A (bundle SLOWER on the true 4xDDP recipe; local microbench transfer FALSIFIED); vram A=B=C=66.6 guard pass. Frozen &lt;5% branch executed: bundle does NOT land; P1 doubly dead (loss-bound fail + -10.8% measured) so the owner relative-bound question is MOOT; P2+bitwise subset split to molmo2-perf-pass1-subset-landing. True cost ~5.5 GPU-h vs 3.0 ceiling (CONTINUE judged 01:42Z, cause owned: model loads uncounted). analysis__perfpass1_box_ladder.json banked; results post 2026-08-09-perfpass1-box-results.md · [pre-reg](posts/2026-08-08-prereg-molmo2-perf-pass1.md)

<details><summary>full record</summary>

molmo2 perf pass 1 EXECUTION (gpu-local, &lt;= 3 GPU-h): per the finalized pre-reg — (1) branch perf-pass1 with the four pinned changes (P1 suffix cuDNN training-only scope, P2 windowed vram peak + lifetime max kept, P3a-c sync removals, P4 clone drop); (2) parity oracles per item (P1 loss&lt;=1e-3/gradnorm&lt;=1e-2 + 50-step overlay + decode byte-match; P3b wte bitwise both regimes; P3c CPU loss-oracle re-pin sum-form only, mean-form UNCHANGED; P4 bitwise grads), failing item drops, no re-tolerancing; (3) bench ladder A=HEAD B=+P1 C=bundle, 320 steps each single local H100, 60k data recipe, median s_per_step over last 240 + vram guard &gt;2% fails; (4) decision: C&gt;=5% -&gt; land post-evals, &lt;5% -&gt; P2+bitwise-free items only; box 4-GPU transfer smoke 0.3 GPU-h before first adopting lineage launch. check.py + oracles green before the landing commit. | EXECUTING 08-08 14:2xZ (owner prio 14:10Z): branch perf-pass1 (P1=00cdafe, full=22e8148, check.py green both), BITWISE ORACLE GREEN 118/118 hashes (perf_pass1_bitwise_oracle.py HEAD-vs-branch), ladder unit fontaine-perfpass1-bench live on local GPU (perf_pass1_bench.sh: parity A/B 50-step + bench A/B/C 320-step). AMENDMENT: batch 8 + chunks 4 (single-GPU unshards ZeRO-1 optimizer ~+11GiB, 12 would OOM; chunks must divide batch; per-chunk size 2 preserved) | LOCAL PHASE CLOSED 15:0xZ: one-step parity ran (grad-norm PASS rel 8.1e-3&lt;=1e-2, no cuDNN crash; LOSS BOUND FAILED AS BANKED |d|=8.70e-3 vs frozen 1e-3 abs = 5.1e-4 RELATIVE at init-scale loss 16.9 — calibration flaw owned in-channel, P1 dropped unless owner approves a relative-bound amendment BEFORE the box ladder). Single-GPU full-recipe bench proven structurally OOM (78.2/79.18 by step 2, batch-invariant). BONUS FIND: --activation-checkpointing crashes on CUDA (recompute escapes sdpa_kernel pin -&gt; backend mismatch abort; filed idea #20, prerequisite fix named). Ladder+overlay moved to box TRUE recipe: fontaine/scripts/box/perf_pass1_bench_box_ddp4.sh (supersedes transfer smoke) | BOX LADDER EXECUTING 01:04Z 08-09 (unit fontaine-perfpass1-box; overlay_A done 01:14Z, 5 sequential runs; babysit entry perfpass1_box live) | CLOSED 02:26Z 08-09: C -7.3% (REGRESSION) -&gt; NO bundle landing per the frozen rule

</details>

---

**`molmo2-perf-fix-prereg`** · `cpu`

molmo2 perf pass 1 pre-reg DRAFT (CPU-&gt;gpu-local): from the 08-08 review (posts/2026-08-08-molmo2-perf-review.md) bundle the S-effort items: suffix sdpa -&gt; cuDNN (one-line + one-step parity gate), windowed vram peak logging (rese…

**boundary:** FINALIZED (not just drafted) 08-08 14:3xZ work session — nothing waited on data: change specs pinned from HEAD re-audit, oracle bounds + bench protocol + decision rules frozen in the post; execution split to molmo2-perf-pass1-exec · [pre-reg](posts/2026-08-08-prereg-molmo2-perf-pass1.md)

<details><summary>full record</summary>

molmo2 perf pass 1 pre-reg DRAFT (CPU-&gt;gpu-local): from the 08-08 review (posts/2026-08-08-molmo2-perf-review.md) bundle the S-effort items: suffix sdpa -&gt; cuDNN (one-line + one-step parity gate), windowed vram peak logging (reset_peak_memory_stats per log window + keep lifetime max), sync removals (model.py:121 count check, text.py:101 wte branch, mask-mul losses in ar_backbone), embed clone drop (model.py:127 index_put_ on non-leaf). Before/after benchmark on a short pinned run + bitwise/parity oracles per item; expected ~8-15% step time at S risk. Separate later rungs: activation-checkpointing lineage flip (batch re-tune), ViT SDPA path (M, parity contract), valid-row CE + F.rms_norm (parity re-gates). Shape-annotation long tail (processor.py, cache.py, encoders/molmo2.py, decoders/ar_molmo2.py) rides whichever item touches each file first.

</details>

---

**`owner-molmo2-perf-review`** · `cpu`

OWNER STEERING 08-08 13:09Z molmo2 perf/memory deep review — SHIPPED same session (posts/2026-08-08-molmo2-perf-review.md + in-channel summary)

**boundary:** opened by owner message 13:09Z 08-08; review report lands same session (13:3x-16:5xZ)

<details><summary>full record</summary>

OWNER STEERING 08-08 13:09Z molmo2 perf/memory deep review — SHIPPED same session (posts/2026-08-08-molmo2-perf-review.md + in-channel summary). Three-lens sweep with 2 measured kernel gaps (idle-local-GPU microbench, ~0 GPU-h): (1) suffix attention lands on MATH sdpa backend (cuDNN excluded by inherited pin; flash rejects mask, efficient rejects GQA) 13x/layer ~5-10% step; (2) ViT eager einsum vs SDPA-flash 13x/block; (3) hand-rolled RMSNorm 10x vs F.rms_norm; (4) --activation-checkpointing exists oracle-pinned but NOT on the live lineage (~2.4-2.8 GiB/sample lever); (5) full-vocab CE fp32-upcasts pad rows; (6) per-step host syncs; (7) 60MB/step embed clone; (8) vram peak metric is a lifetime ratchet (explains the 41,780/42,940 creep). Static-max verdict: DON'T (bucketing prior art +5.09% padding ceiling; suffix uncapped). Shape annotations landed on bijou/molmo2/{model,text,vision}.py. All changes need pre-reg; nothing touched the live run.

</details>

---

**`tiny-expert-capacity-10k`** · `gpu-local`

T1 tiny-expert capacity rung, FINAL DESIGN (owner yes 19:59:04Z; 40k/biggest-batch amendments REVERTED by owner 20:08:53Z 'Let's do your original plan' after the wall-clock arithmetic

**boundary:** fit ladder ~20:3xZ -&gt; 10k run ~10-11 h -&gt; endpoint ~06-07Z 08-10 + ~1.3 h eval; first-poll s/step + vram + projection in-channel; babysit tiny10k entry live, gate 15 GPU-h | HOST-RAM OOM KILL 20:52:08Z 08-09 at step 500 (first probe 16.46@500 landed, no ckpt yet): kernel OOM killer, pt_data_worker x20 at ~7-9 GiB RSS each (~150-190 GiB) + 46 GiB main proc vs 221 GiB host — the launcher kept the box recipe's --num-workers 20 --prefetch-factor 4, lethal at batch 48x1. AMENDED to --num-workers 10 --prefetch-factor 2 (sample order unchanged, recipe identical) + SKIP_LADDER=1 escape (b48c12 already green); RELAUNCHED clean from step 0 same seed ~21:03Z, ~0.4 GPU-h lost. New projection: endpoint ~05:1xZ 08-10 -&gt; Delta_capacity read ~06:3xZ. Owned in-channel 21:16Z. | CLOSED 05:5xZ 08-10: train COMPLETE 05:06Z (~8.7 GPU-h incl the second host-RAM OOM at step ~9,060 04:00:55Z + resume-from-8750 replay, workers 10-&gt;6), chained panel_v2 @10000 COMPLETE 05:45Z (~0.6 GPU-h, ~9.3/15 gate). PRIMARY READ: Delta_capacity@10k = +0.188 [CI95 +0.155, +0.221] paired per-frame on 15,056 panel-v2 core frames (tiny 9.6094/3.0758 vs F 9.4157/2.9581) = capacity prior CONFIRMED at the pre-registered |d|&lt;=0.3 band; CI excludes zero so the width cost is real-but-small (+2.0%, late-horizon: per-step delta +0.106 -&gt; +0.374 across the 50-step chunk); state-copy execution oracle byte-green ACROSS MACHINES (box F vs local tiny); probe-vs-panel sign flip logged (probe -0.069 under, panel +0.188 over). Expert sizes measured: tiny 86.8M vs F 367.5M (4.2x total; tap/adapter surface is the fixed cost). Results post 2026-08-10-tiny-expert-results.md + 3-panel chart; analysis__tiny10k_delta_capacity.json + both panel html/json on the Space; step_010000 weights-only uploaded to fontaine-checkpoints (backbone deduplicated, sha verified); babysit entry pruned; local GPU FREE 05:45Z. · [pre-reg](posts/2026-08-09-prereg-tiny-expert-40k.md)

<details><summary>full record</summary>

T1 tiny-expert capacity rung, FINAL DESIGN (owner yes 19:59:04Z; 40k/biggest-batch amendments REVERTED by owner 20:08:53Z 'Let's do your original plan' after the wall-clock arithmetic — no training step had run): h256/d12 width-only contrast vs F (tap surface + adapters identical; depth structural), frozen 60k trunk (backbone sha e6ed783b verified), LOCAL 1xH100 unit fontaine-tiny10k, 10k steps @ eff-48 (48x1 vs F 12x4, same LR schedule), saves 1250 (F cadence). Launcher launch_local_fontaine_molmo2_flow_tiny_h256_10k_1xh100.sh: fit ladder b48c12 -&gt; b48c24 -&gt; 10k run -&gt; chained panel_v2 @10000. Frozen read: Delta_capacity@10k fully matched (tiny minus F, paired per-frame CI95) vs banked F@10k 9.4157 + state-copy execution oracle; bands |d|&lt;=0.3 prior confirmed / &gt;=1.0 capacity binds. LAUNCH HISTORY: 20:03Z rc2 (--zero1/--chunk-grad-allreduce DDP-only guards, dropped, pre-reg amended); 20:05Z b96 ladder started, owner re-scoped pre-step-1; 20:1xZ relaunch at final design.

</details>

---

**`owner-trajectory-datasets-survey-0809`** · `cpu`

OWNER STEERING 19:58:05Z 08-09: investigate additional trajectory datasets we could train on

**boundary:** CLOSED 2026-08-09T20:56Z work session (stale-close audit vs git: the work landed in the 19:49-20:3x session, commit beb8659, but the item was never flipped): survey post posts/2026-08-09-trajectory-datasets-survey.md shipped via 4 parallel research subagents, all links fetch-verified, Space 200-verified, in-channel summary posted 20:21:21Z with headline (855 in-scope SO-100/101 hub hours vs our 229, ~300h new 2026, sim-contamination hazard, MolmoAct2 curation diff = #1 recommendation); idea #9 fed. Follow-ups (corpus-delta re-crawl + MolmoAct2 diff, Bridge V2 pilot) are owner-decision items, deliberately NOT auto-queued.

<details><summary>full record</summary>

OWNER STEERING 19:58:05Z 08-09: investigate additional trajectory datasets we could train on — ideally SO-101, but also look more generally. Deliverable: a detailed blog post with links to the datasets, statistics (episodes/hours/tasks/embodiments/modalities), brief descriptions, and an assessment of what is actually usable for our training recipes (community_curated_v0 is the current substrate). Post link in-channel when it lands.

</details>

---

**`molmoact2-oob-panel-eval`** · `gpu-local`

OWNER STEERING 10:50Z+11:06Z 08-10: MolmoAct2 (allenai/MolmoAct2-SO100_101) out-of-band eval on the k4l2 panel

**boundary:** CLOSED 14:4xZ 08-10, DELIVERED END-TO-END IN ONE SESSION (~1.3/8 GPU-h): sweep rc=0 14:23:47Z (25,800 frames, 352 f/min sorted-index locality), frozen reads + contamination json banked (matched-window core, willnorris/bbox-2 excluded per owner amendment 13:14Z: snapflow top10tickets 3.90 / 60k-cont 4.46 / 40k 4.56 / stablekey 5.06 / er15k 5.89 / state-copy 8.32 / MolmoAct2 13.87 pooled = 16.97 clean-633 vs 7.00 contaminated-245; every paired read MOLMOACT2-WORSE, tight CIs; on their own training repos they beat state-copy -0.75 but trail snapflow +3.29 [+3.11,+3.48]), 3-policy HTML report rendered first try (32-frame gallery, 4 policies/joint, horizon visible) + uploaded to the NEW fontaine-reports static Space + reports.md section + numbers in-channel 14:37Z. HEADLINE FINDING: the released SO100_101 fine-tune does not transfer outside its 1,220-repo mixture (predicts sane joint-unit motion in the wrong workspace frame on unseen rigs). Owner threads answered same-session: 12:59Z inference-correctness challenge (contamination-split proof), 13:14Z exclusion amendment (applied + oracle branch), 13:48Z blog-storage question, 13:51Z reports-migration directive (fontaine-reports static Space live - dataset repo serves text/plain, tested; 72 blog links rewritten + 31 redirect stubs; squash queued behind HF GC via unit fontaine-blog-migrate), 14:13Z navbar bug (missing hashed toc js from the morning incident, re-uploaded, fixed). ORIGINAL EXECUTION RECORD: pre-reg finalized 00a9feb, smoke green 12:5xZ, sweep launched 13:2xZ after objection window. · [pre-reg](posts/2026-08-10-prereg-molmoact2-oob-panel.md)

<details><summary>full record</summary>

OWNER STEERING 10:50Z+11:06Z 08-10: MolmoAct2 (allenai/MolmoAct2-SO100_101) out-of-band eval on the k4l2 panel. Deep implementation read DONE + in-depth plan posted (posts/2026-08-10-molmoact2-oob-eval-plan.md, in-channel 11:38Z): 30-step/1.0s chunk at native fps vs our 50-step/1.67s -&gt; matched-window (steps 0-29) re-pool of banked npzs is the primary read; their predict_action end-to-end (q01/q99 norm from checkpoint norm_stats.json); contamination measured 245/878 panel repos = 31.0% core frames -&gt; pooled/clean/contaminated splits. EXECUTE: (1) FINALIZE the pre-reg (the plan post is the registered sketch; finalization adds frames, seeds, abort oracles, immutability stamp) BEFORE any GPU minute; (2) predictor script molmoact2_panel_predict.py + oracle-gated matched-window reads instrument; (3) 500-frame stratified smoke + scale sanity; (4) full 25,800 sweep (systemd unit) after smoke green + owner objection window; (5) OWNER GO 11:59Z 08-10 ('plan sounds good, let's eval the so101 checkpoint'): side-by-side HTML report REQUIRED — same frames, three policies per frame: snapflow 80k (bijou_flow_artrunk@80k banked panel npzs: top-10-tickets 5.1847 headline + stable-key single-draw 6.5997) vs MolmoAct2 vs state-copy, plus summary-stats block (pooled/clean-633/contaminated-245, paired CI95, chunk+first MAE, matched 30-step window primary, 50-step secondary); reports page + in-channel numbers. Gate &lt;= 8 GPU-h (est 2-5). Record-only: nothing gates or repoints our runs.

</details>

---

**`snapflow80k-draws10-panel-eval`** · `gpu-local`

OWNER REQUEST 14:33Z 08-10 (recovered 15:0xZ after the babysit-grep consume miss): full-panel mean-of-10-draws heun-30 eval for snapflow 80k (bijou_flow_artrunk_h1024_40k_ddp2 step_080000, local checkpoint on disk) -&gt; its row joi…

**boundary:** OWNER SKIP 15:01:04Z 08-10 ('Let's skip 2') — cancelled before launch, 0 GPU-h spent; the heun-30 single-draw row (zero-GPU) covers the single-draw reference in the report. Reopen only on explicit owner ask. · [pre-reg](posts/2026-08-10-prereg-molmoact2-oob-panel.md)

<details><summary>full record</summary>

OWNER REQUEST 14:33Z 08-10 (recovered 15:0xZ after the babysit-grep consume miss): full-panel mean-of-10-draws heun-30 eval for snapflow 80k (bijou_flow_artrunk_h1024_40k_ddp2 step_080000, local checkpoint on disk) -&gt; its row joins the MolmoAct2 3-policy report + reads. NOT banked at full panel (only the 2,458-frame drawsprobe_s7 subset exists; top-10-tickets is mean-of-10 over SEARCHED tickets, different config). EXECUTE: (1) pin the exact eval invocation from the goldenticket/stablekey launcher class (plain draws10, heun30, panel_curated_v0_k4l2 plan, stable noise seeds per the draws-fairness convention) as a short pre-reg note (record-only, owner-requested); (2) launch on local H100 via run_detached unit + babysit entry (est ~8-12 GPU-h at 10x decode of the ~1.3 GPU-h single-draw class; sanity-check the first-poll rate); (3) on rc=0 add BASELINES entry to molmoact2_panel_reads.py + DISPLAY to molmoact2_panel_report.py, rerun reads (oracle green) + report, re-upload, post updated numbers in-channel. Owner told 15:0xZ it launches tonight unless they object. PREREG NOTE: rides the molmoact2-oob pre-reg's report spec (owner-directed row add); a short launch note with the exact pinned invocation posts in-channel before the GPU minute, per charter.

</details>

---

**`molmoact2-rig-finetune-runbook`** · `cpu`

OWNER QUESTION 15:19:45Z 08-10: 'How could I -- out of band -- fine-tune molmo2act on my rig datasets and then do local rollouts? Happy to use their code.' Answered in-channel 15:23Z (4-step shape: rig repos are LeRobot-native -&gt;…

**boundary:** EXECUTED IN FULL 2026-08-10 16:1x-17:5xZ: codebase_version read (both repos v3.0) -&gt; pre-reg posts/2026-08-10-prereg-molmoact2-rig-finetune.md + param sheet in-channel 16:20Z (objection window to 17:50Z) -&gt; preflights P1-P4 green (P3 Amendment 1 posted in-window: offset tripwire record-only, measured posture-collapse via state-norm saturation [43.7,185.3] theirs vs [-103,+67] rig, 97% frames saturating; sign gate all-positive; anchors zero-shot 28.95 / state-copy 9.08 on 240 rig frames) -&gt; runbook posts/2026-08-10-molmoact2-rig-finetune-runbook.md (incl. SO-101 server adaptation, v3.0-end-to-end rollout rule, safety rails) -&gt; LAUNCHED 17:48:18Z unit fontaine-molmoact2-rig-ft (silence=launch honored), first-poll green 17:58Z (830 f/min, ~2.6 GPU-h projected vs 12 gate, vram 38.9). Successor: molmoact2-rig-ft-postprocess.

<details><summary>full record</summary>

OWNER QUESTION 15:19:45Z 08-10: 'How could I -- out of band -- fine-tune molmo2act on my rig datasets and then do local rollouts? Happy to use their code.' Answered in-channel 15:23Z (4-step shape: rig repos are LeRobot-native -&gt; their lerobot_wrapper; recompute q01/q99 stats over rig repos only via their stats.py; warm-start MolmoAct2-SO100_101 via their train_lerobot.py, trunk mostly frozen, 57 episodes short schedule; rollouts = adapt examples/droid/host_server_droid.py to SO-101 REPO_ID/NORM_TAG/state-dim-6 + lerobot client loop executing 30-step chunks w/ 0.5-1s replan; bf16/processor patches already ported in molmoact2_panel_predict.py). OWNER GO 15:24:16Z ('Yes, I want a runnable runbook and I want you to go ahead and do a fine-tune on the local GPU') — EXECUTE: runbook post (pinned commands, rig mixture file, rig-only q01/q99 stats, SO-101 server adaptation) + parameter sheet in-channel BEFORE any GPU minute (owner said treat-as-objection-window, silence=launch per my 15:2xZ ack), then LAUNCH the fine-tune on the free local H100 (systemd unit + babysit entry + first-poll rate/vram check + own gate). Warm-start MolmoAct2-SO100_101; 57 rig episodes (so101_pick_place_clean 7 + _v2 50); their train_lerobot.py; short schedule, trunk mostly frozen first rung. OFFERED a runnable runbook (pinned commands, rig mixture file, SO-101 server adaptation, box fine-tune pre-reg) — EXECUTE on owner yes, or fold into the next work session as CPU prep if they engage further. Motivating evidence: today's OOB eval (in-mixture repos beat state-copy -0.75, unseen 2x worse -&gt; rig fine-tune closes exactly that gap). CONVENTION THREAD (owner 15:48Z, github.com/irenegracekp/molmoact2-so101): MolmoAct2 trained on LeRobot v2.1 joint-angle convention, lerobot 0.5.x calibrates in v3.0 -&gt; live-inference danger (arm slams on wrong offsets); recorded-data eval UNaffected (repo says so itself + contaminated-split parity is the empirical check; answered in-channel 15:5xZ). RUNBOOK MUST INCLUDE: (1) read codebase_version off both rig repos' meta/info.json (never assume); (2) recommended path = convert rig actions/states v3.0-&gt;v2.1 (documented offsets/signs) so the model stays in its native convention; param sheet states the convention of every tensor; (3) first-step continuity oracle on held-out rig frames (pred step-0 ~ current state, loud fail); (4) rollout server: verify rig calibration lerobot version or recalibrate pinned 0.5.1, apply conversion, no-execute dry-run gate + command clamp before ANY motion.

</details>

---

**`blog-space-gc-tail`** · `cpu`

Blog Space storage tail (manual-only — the retry unit is STOPPED and must NOT be re-armed)

**boundary:** HF GC can take up to ~6h from the 13:5xZ 08-10 squash attempt; owner ask due ~08-11 morning if unchanged.

<details><summary>full record</summary>

Blog Space storage tail (manual-only — the retry unit is STOPPED and must NOT be re-armed). Each session: check usedStorage via huggingface_hub repo_info(expand). When it drops below ~500 MB: ONE upload_folder of the current book (delete_patterns searchindex-*.js/toc-*.js + reports redirect stubs per blog-space-push memory, NEVER ["**"]) + super_squash + curl-verify navbar/search/stubs + post the all-clear. If still capped (~998.6 MB) by 08-11 morning: ask the owner for the delete+recreate GO (offered 15:2xZ 08-10). CLOSED 03:2xZ 08-11: GC drained 543.6 -&gt; 403.9 MB (below the ~500 line); ONE upload_folder of the current book (delete_patterns searchindex-*.js/toc-*.js, scoped) + super_squash + curl-verify (index/now/style.css/reports/prereg all 200, fresh now.html content live) + all-clear posted in-channel. Retry unit remains STOPPED. Storage accounting lags squash (async GC) — next session may see a transient figure, record-only.

</details>

---

**`molmoact2-rig-ft-postprocess`** · `gpu-local`

Rig fine-tune rung reads + report + checkpoint upload (successor to molmoact2-rig-finetune-runbook)

**boundary:** CLOSED 21:0xZ 08-10 work session, all 7 steps: rc=0 20:27:44Z verified (2000 steps, ~2.7/12 GPU-h); step2000 converted (~/checkpoints/molmoact2-so101-rig-r1-step2000-hf, rig norm_stats verified); rung read MAE 3.2301 (PRE-REG PASS: monotone 6.76/4.66/3.59/3.23, beats zero-shot 28.95 + state-copy 9.08, all 6 corrs +0.885..+0.965, offsets &lt;=0.63, oracles green); results post posts/2026-08-10-molmoact2-rig-ft-results.md + HTML report eval__fontaine_so101_rig_ae_r1__anchor_rungs.html on fontaine-reports (curl 200) + 5 frozen jsons uploaded + reports.md section; weights delta to fontaine-checkpoints/molmoact2_so101_rig_r1_step2000 (AE 588 tensors + resized wte/lm_head, trunk dedup sha-verified 704/707 byte-identical); babysit entry pruned; runbook section 5 updated with measured numbers. Numbers + joint1-wording correction posted in-channel (owner 👍 on the 20:30Z results post). · [pre-reg](posts/2026-08-10-prereg-molmoact2-rig-finetune.md)

<details><summary>full record</summary>

Rig fine-tune rung reads + report + checkpoint upload (successor to molmoact2-rig-finetune-runbook). Train endpoint ~20:20Z 08-10 (2000 steps, unit fontaine-molmoact2-rig-ft, log ~/logs/molmoact2_rig_ft.log). IN-SESSION PROGRESS 18:3x-19:5xZ 08-10: rungs 500/1000/1500 converted + read (MAE 6.7561 / 4.66 / 3.5871 vs anchors zero-shot 28.95 &amp; state-copy 9.08 — expectation 2 MET, monotone, oracles green; HF dirs ~/checkpoints/molmoact2-so101-rig-r1-step{500,1000,1500}-hf). REMAINING: (1) verify rc=0 + final save; (2) convert step2000 via experiments/olmo/hf_model/convert_molmoact2_to_hf.py (molmoact2 venv, branch fontaine-so101-rig); (3) rung reads: uv run python fontaine/scripts/molmoact2_rig_preflight.py --model &lt;converted dir&gt; --out-stem analysis__molmoact2_rig_ft_step&lt;N&gt; — same 240 rows; pre-reg pass = beat BOTH anchors (zero-shot 28.95, state-copy 9.08 matched 1.0s window) + step-0 continuity + all motion corrs positive; (4) results post in-channel + blog results page + report on fontaine-reports (charts per house style, dark-mode); (5) best rung weights to fontaine-checkpoints same-session (standing rule); (6) prune rig_ft_r1 babysit entry; (7) runbook §5 updated with measured numbers. Contaminated-by-construction: label every read (train-frame sanity, not generalization; real eval = owner rig rollouts per runbook §3-4).

</details>

---

**`molmoact2-firstclass-port`** · `cpu`

OWNER GO 20:06:37Z 08-10 ('Let's do it, 1 through 4' on the 19:5xZ in-channel estimate): make MolmoAct2 first-class in-repo, rig-path-first

**boundary:** CLOSED 07:1xZ 08-11 work session — ITEM 4 G4-PASS ends the port (items 1-4 all closed): bijou/molmoact2/train.py (f9bc0ba) = first-class AE fine-tune trainer in the port package, their recipe verbatim (2000 steps, 64 global = 8x8 micro, AE-only 577,564,448 trainable = the G1-measured count, AdamW 5e-5/(0.9,0.95)/1e-6/wd0, warmup 200 -&gt; cosine 0.1x, clip 1.0, t=0.001+0.999*Beta(1,1.5), target actions-noise, valid-dim-mean MSE, rig-only q01/q99 pinned from the run-1 export, their img_aug=full op-for-op), TWO NAMED DELTAS pre-declared in the exec note (deterministic frozen trunk vs their live residual_dropout=0.1; per-frame sqrt-weighted sampling w/ replacement). Zero patches against their checkout — all three train_lerobot.py patches retired (our in-repo lerobot ingests the rig repos' language columns natively). Run molmoact2_ae_ours_r1 unit fontaine-molmoact2-ae-ours 05:19:51-06:56Z rc=0, ~1.9/6 GPU-h (port total ~2.6/8). G4 ALL FOUR CLAUSES PASS (fontaine/scripts/molmoact2_ours_ft_rung_read.py, 240 banked anchor rows, same per-row seeds): final rung 4.8846 &lt; both anchors (28.9454/9.0824); monotone 8.18@500 -&gt; 6.16@1000 -&gt; 5.21@1500 -&gt; 4.88@2000; loss corridor 96 matched steps ratios [0.63,1.33] zero violations (rule frozen pre-launch: 5-pt rolling median in [0.5x,2x] at steps &gt;=100); gate met. 8 CPU oracles tests/test_molmoact2_train.py, check.py 667 green. RECORD-ONLY FINDING: our rungs ~+1.65 above their-trainer run (4.88 vs 3.23@2000) while our LOSS ends 37% below (ratio 0.63) — fingerprint of the trunk-dropout delta; their stochastic-KV regularization buys anchor accuracy. Dropout-matched rung = named lever, NOT queued (needs own pre-reg). Artifacts: analysis__molmoact2_ours_ft_rung_read.json + corridor series on fontaine-reports (curl 200 x2); step_002000 AE-only bf16 predictor-consumable -&gt; fontaine-checkpoints/molmoact2_ae_ours_r1_step2000. Post-port unlocks live: panels score MolmoAct2 natively, SnapFlow 1-NFE distillation of their AE pre-registerable, rollout server can load either stack. · [pre-reg](posts/2026-08-10-prereg-molmoact2-firstclass-port.md)

<details><summary>full record</summary>

OWNER GO 20:06:37Z 08-10 ('Let's do it, 1 through 4' on the 19:5xZ in-channel estimate): make MolmoAct2 first-class in-repo, rig-path-first. Scope: (1) action expert port (their nn/action_expert.py 982 LOC + backbone-AE wiring molmoact2.py 1.3k LOC) + weight load; (2) action-side prompt/processing deltas (template, state encoding, q01/q99 norm_stats) on top of bijou/molmo2 processor; (3) parity harness vs their HF forward + banked 240-row anchors (zero-shot 28.95 / state-copy 9.08) + rig-ft rung checkpoints; (4) AE fine-tune in OUR trainer, retiring the 3 train_lerobot.py patches. Backbone reused from bijou/molmo2 (byte-verified); depth/trace/sim-eval stay out-of-band. Pre-reg post first (parity gates falsifiable), CPU-mostly, GPU only for parity checks.

</details>

---

**`er60k-events-oneoff-report`** · `gpu-local`

OWNER REQUEST 12:44:35Z + 12:45:13Z 08-11 (one-off, 'post a neat html report', 'I want to see many varied examples of events'): quantitative + qualitative investigation of the events the model generates vs ground truth (weak judg…

**boundary:** CLOSED 16:0xZ 08-11 work session, all 6 scope steps end-to-end (~1.55/4 GPU-h): (1) INSTRUMENT landed commit 7f43c54 (--dump-generations + main-arm retention under explicit --generate + aux metrics from any retained voice + Q3 predict_with_text guard; ShardResults.generation_identity shard-safe; closes the 35k aux-arm debt); (2) DUMP PASS rc=0 15:5xZ unit eval-er60k-events-dump (25,800 rows, single narrated all-fields arm, ~275 f/min; launch note + frozen spec in-channel 14:13Z pre-GPU; 24-frame smoke first). Instrument oracle: presence acc 0.8568 vs banked 0.8582 = delta 13/8,987 frames, INSIDE the documented cross-world-size bf16 batch-composition band (banked was a 4-way box shard) — near-reproduction reported with the caveat, not claimed exact; (3) QUANT analysis__er60k_events_confusion.json: both-none 7,238 / hits 333 / swaps 129 / MISSES 683 / false alarms 604; model speaks on 40% of the 1,145 gt-event frames, class-agrees 72% when it does; exact-string 3.6%; (4) QUAL report__er60k_events_oneoff.html — 136 image cards, repo-diverse galleries per bucket; (5) PROBE analysis__er60k_events_probe.json: 683 misses -&gt; 679 forced (replay oracle bit-exact 679/683, 0 none-variants post-ban): forced guess lands gt class 428/679 = 63% -&gt; dominant miss mode is saw-it-under-threshold (idle 86/release 80/occlusion 72/blur 62%; camera_view 10%, episode markers 0% = the genuinely-not-encoded tail); (6) HTML + all 5 artifacts on fontaine-reports curl-verified (html serves via 302-&gt;CDN-&gt;200 full 10.7 MB), reports.md section, numbers in-channel 16:0xZ. Named lever FED TO IDEAS (#23 event-none-calibration, on-ice w/ trigger): decode-time none-penalty, zero training. Babysit entry pruned. · [pre-reg](posts/2026-08-09-prereg-molmo2-er-60k.md)

<details><summary>full record</summary>

OWNER REQUEST 12:44:35Z + 12:45:13Z 08-11 (one-off, 'post a neat html report', 'I want to see many varied examples of events'): quantitative + qualitative investigation of the events the model generates vs ground truth (weak judge labels), on the fresh @60000 endpoint checkpoint. Scope: (1) INSTRUMENT: extend the narrated arm to dump per-frame (identity triple, generated event string, weak-label event) — the standard eval computes event acc in-memory and discards generations (bijou/eval/cli.py results.generations only fills from NarratedBijouPolicy and is never written; same gap class as the 35k aux arm); (2) DUMP PASS: one-off generation pass on step_060000 (local disk after the endpoint dl) over the ~8,987 labeled panel frames; (3) QUANT: full model-class x gt-class event confusion INCLUDING none/none — counts + per-class precision/recall + the (model none, gt event) miss bucket sized exactly; (4) QUAL: frame galleries per confusion bucket (hit / miss / false alarm / class swap) — camera image + generated line vs gt, MANY varied examples across repos/tasks; (5) CONSTRAINED PROBE: on (gt event, model none) frames re-decode the event slot with 'none' banned (1-step constrained decode, logit mask on the event slot) — does the forced guess match gt class ('didn't see it' vs 'saw it, under-threshold'); (6) neat standalone HTML -&gt; fontaine-reports (curl 200) + link in-channel. Plan acked in-channel 12:51Z. One-off: record-only, no run gating. PREREG NOTE: rides the er-60k pre-reg like the er15k/35k/55k owner-requested reads; a short launch note with the pinned invocation + confusion/probe spec posts in-channel before the GPU minute, per charter.

</details>

---

**`er60k-endpoint-postprocess`** · `gpu-box`

er_60k endpoint postprocess (pre-reg posts/2026-08-09-prereg-molmo2-er-60k.md): box hits @60000 ~12:3xZ 08-11 -&gt; chained panel_v2 k4l2 eval runs in-unit (--report + npz)

**boundary:** CLOSED 13:3xZ 08-11 work session (owning chained session rode endpoint + eval foreground): train hit @60000 12:36Z rc-clean (~153/155 GPU-h incl. chained eval); final async save published; chained panel_v2 k4l2 eval rc 13:28Z (4-way shard). THE ER DECISION READ: endpoint fast path 5.7782/1.9898 core (best banked trunk to date); vs 40k endpoint 6.0079 pooled -0.2297 [CI95 -0.281, -0.154] BELOW-BASELINE; vs 60k-cont 5.8602 pooled -0.0821 [CI95 -0.126, -0.025] BELOW-BASELINE CI excludes zero -&gt; ER INIT WINS BOTH LEGS, the ER trunk is the new reference trunk. Rung trajectory vs 40k endpoint: 15k +1.52 / 35k +0.28 / 55k -0.18 / 60k -0.23. Aux endpoint n~8,987: holding 0.915 / progress 0.060 / event 0.858 / visible 0.822; pairing +0.055 (45%). Rig-data effect read NOT split-compatible (panel repo_id identity contains no owner-rig repos) — skipped per the pre-reg if-clause. Artifacts on fontaine-reports (curl 200), reports.md endpoint section, chart-led in-channel post + owner ping 13:32Z (init-delta chart attached); step_060000 weights-only on fontaine-checkpoints (42.0s, commit 4ed3dd0, uploaded mid-eval); babysit entry er_60k pruned. Results blog post (chart-led consolidated ER-screen close) deliberately rolled to the next session — owner has headline + report; the events one-off (owner 12:44Z) takes priority next. · [pre-reg](posts/2026-08-09-prereg-molmo2-er-60k.md)

<details><summary>full record</summary>

er_60k endpoint postprocess (pre-reg posts/2026-08-09-prereg-molmo2-er-60k.md): box hits @60000 ~12:3xZ 08-11 -&gt; chained panel_v2 k4l2 eval runs in-unit (--report + npz). Owning session: ride the chained eval to rc in-turn (foreground polls, NEVER a Monitor), then paired CI95 reads vs banked 40k endpoint (6.0079) + 60k-continuation (5.8602) panels — THE ER decision read (in-run evidence so far: 20 straight negative matched legs to @40000, er − 40k endpoint-matched −0.67, run-best 5.10@44500 vs 40k best-ever 5.91; all record-only, panel decides) -&gt; HTML report + JSON + analysis to fontaine-reports (curl-audit every link 200) + reports.md + chart-led in-channel post + owner ping. Also: rig-data effect read at endpoint if split-compatible (held-out rig episodes, natural share 0.19%). Then prune the babysit entry and decide checkpoint upload (banked/consumable -&gt; fontaine-checkpoints same-session per standing rule; weights-only unless seeding training).

</details>

---

**`er35k-aux-panel-eval`** · `gpu-local`

OWNER REQUEST 20:47:38Z 08-10 (35k -&gt; hub + aux-enabled eval report on local GPU)

**boundary:** CLOSED 00:4xZ 08-11 work session: standard both-arms eval rc=0 00:41:21Z (~2.2/8 GPU-h total incl. aux arm). Class-matched reads (key bijou@35000, core 17,204): fast path 6.2892/2.3746; vs 40k endpoint 6.0079 pooled +0.2813 [CI95 +0.199, +0.337]; vs 60k-cont 5.8602 +0.4290 [+0.353, +0.467] — ABOVE-BASELINE at 58% training, the 15k gap (+1.52) ~82% closed; supersedes the aux-arm cross-class read as promised in-channel. Aux vs weak labels at full n~8,987, all four improved from 15k: holding 0.899-&gt;0.915, progress MAE 0.075-&gt;0.065, event 0.862-&gt;0.875, visible 0.704-&gt;0.823. Narration pairing measured at last: +fields costs +0.047 chunk (44% win) — matches the er15k class. Artifacts on fontaine-reports (curl 200 x2), reports.md standard-eval section supersedes, numbers in-channel, babysit entry pruned. Instrument-debt follow-up (retain main-policy generations under --generate) stays OPTIONAL, not queued. · [pre-reg](posts/2026-08-09-prereg-molmo2-er-60k.md)

<details><summary>full record</summary>

OWNER REQUEST 20:47:38Z 08-10 (35k -&gt; hub + aux-enabled eval report on local GPU). AUX ARM DONE 22:30:45Z rc=0 (~1.5/8 GPU-h): hub upload 42.4s + local dl 31.1s + eval unit eval-er35k-aux ~250 f/min; core 6.3425/2.3770 (aux-narrated decode); paired reads banked (vs 40k endpoint +0.335 [+0.247,+0.387] CROSS-CLASS narrated-vs-fastpath; vs 60k-cont +0.482); report+json+analysis on fontaine-reports; numbers in-channel 22:3xZ. HARNESS GAP FOUND: explicit --generate makes the MAIN arm narrate but discards its generations -&gt; per-field aux metrics (holding/progress/event/visible) empty + no base-vs-narrated pairing (bijou/eval/cli.py results.generations only fills from NarratedBijouPolicy). STANDARD EVAL RELAUNCHED 22:33:30Z unit eval-er35k-panel (both arms + full aux metrics, the er15k report shape, ETA ~01:0xZ; babysit entry er35k_panel). REMAINING (next session): rc=0 -&gt; class-matched reads via er15k_panel_reads.py (key bijou@35000, fast path vs banked 40k 6.0079 + 60k-cont 5.8602 — supersedes the cross-class read) -&gt; report/json/analysis to fontaine-reports + reports.md + in-channel numbers -&gt; prune babysit entry. OPTIONAL instrument-debt follow-up (own small item if pursued): retain main-policy generations under --generate so aux metrics survive without the second pass.

</details>

---

**`er55k-panel-eval`** · `gpu-local`

OWNER REQUEST 09:41:04Z 08-11 ('eval the 55000 step checkpoint that just landed on the box as before'): step_055000 -&gt; hub (42.9s, commit 99a1ae2, weights-only) + local dl (13.7s) + STANDARD both-arms panel eval LIVE local H100 u…

**boundary:** CLOSED 12:1xZ 08-11 work session (owning chained session rode it foreground to rc): eval rc=0 12:00:11Z (~2.2/8 GPU-h). Class-matched reads (key bijou@55000, core 17,204): fast path 5.8269/2.0172; vs 40k endpoint 6.0079 pooled -0.1810 [CI95 -0.232, -0.105] BELOW-BASELINE — first below-baseline read for the ER trunk (@35000 was +0.281 above); vs 60k-cont 5.8602 -0.0334 [-0.078, +0.024] CI-SPANS-0 = parity at 92% training; state-copy byte-match x3. Aux n~8,987: holding 0.920 / progress MAE 0.060 / event 0.858 / visible 0.822 (holding+progress improved from 35k, event -0.017); narration pairing +0.039 chunk, 46% win (er15k/er35k class). Artifacts on fontaine-reports (curl 200 x3) + reports.md @55000 section + numbers in-channel 12:0xZ; babysit entry er55k_panel pruned. Record-only as pre-registered; the @60000 endpoint panel (~12:3xZ) is the ER decision read. · [pre-reg](posts/2026-08-09-prereg-molmo2-er-60k.md)

<details><summary>full record</summary>

OWNER REQUEST 09:41:04Z 08-11 ('eval the 55000 step checkpoint that just landed on the box as before'): step_055000 -&gt; hub (42.9s, commit 99a1ae2, weights-only) + local dl (13.7s) + STANDARD both-arms panel eval LIVE local H100 unit eval-er55k-panel launched 09:48:27Z (er35k recipe verbatim, no --generate; first poll 98% util/30.5G). REMAINING (owning session, ride to rc ~11:5xZ with foreground polls, NEVER a Monitor): rc=0 -&gt; er15k_panel_reads.py --stem-cand reports/eval__fontaine_molmo2_er_60k_ddp4__step_055000__panel_curated_v0_k4l2 --out reports/analysis__er55k_panel_vs_banked_k4l2.json (fast-path key bijou@55000, CLASS-MATCHED vs banked 40k 6.0079 + 60k-cont 5.8602; 35k precedent 6.2892 / +0.2813) -&gt; report+json+analysis to fontaine-reports (curl 200) + reports.md + numbers in-channel -&gt; prune babysit entry er55k_panel. Record-only; nothing gates the box run. NOTE: the box endpoint @60000 ~12:4xZ lands right after — the same owning session likely takes er60k-endpoint-postprocess next.

</details>

---

**`er-screen-results-post`** · `cpu`

ER screen close: chart-led consolidated results post (deliberately rolled from the 10:00Z 08-11 session that closed er_60k)

**boundary:** CLOSED 16:2xZ 08-11 work session: posts/2026-08-11-er-init-screen-results.md landed with plain-words opener + 3 house-style charts (probe overlay from the salvaged box train logs, panel rung trajectory, decision CIs) + aux-heads table across rungs; chart script fontaine/scripts/er60k_screen_close_charts.py reads only banked files (regenerable, no live hosts); all artifact links curl-verified; SUMMARY + posts index registered (index drift fixed: four 08-10 posts were missing from posts/index.md); check.py green; Space pushed + link in-channel same session. · [pre-reg](posts/2026-08-09-prereg-molmo2-er-60k.md)

<details><summary>full record</summary>

ER screen close: chart-led consolidated results post (deliberately rolled from the 10:00Z 08-11 session that closed er_60k). The full ER-init story in one blog page, house chart style (dark-mode, eval-report scheme): rung trajectory vs 40k endpoint (+1.52 -&gt; +0.28 -&gt; -0.18 -&gt; -0.23), the decision read (endpoint 5.7782/1.9898; -0.2297 [-0.281,-0.154] vs 40k endpoint; -0.0821 [-0.126,-0.025] vs 60k-cont, both CI-excludes-zero), aux-heads table across rungs (holding er-better, event cont-better, rest tied), probe-curve overlay (shared seed), what-it-means-for-follow-ons (every new arm sits on er_60k/step_060000). Owner already has headline + report links (13:29Z post); this is the durable long-form. posts/ page + Papers-style plain-words opener per house rule; Space push; link in-channel.

</details>

---

**`rebase-fontaine-on-main-postreview`** · `cpu`

MAIN-AGENT DIRECTIVE (owner-relayed 14:34:48Z 08-11, message.txt attachment): rebase fontaine on main @36afff0 (owner-session correctness reviews of bijou/molmoact2 + bijou/molmo2 landed as code) and adapt

**boundary:** CLOSED 17:2xZ 08-11 work session (owner push 16:43Z seen mid-session, executed immediately): origin/main fetched fdd9aa3 -&gt; 36afff0, git rebase origin/main CLEAN — zero conflicts (only both-sides file bijou/train.py: main's wandb import move vs our parse_repeat_specs import; auto-merged + one ruff I001 fixup). All 7 directive items sentinel-verified at HEAD: released_so100_101 staticmethod, frozen no-default ActionExpertConfig, Gemma4Config.e2b()/.e4b() staticmethods, the 5 loud molmoact2 guards + tests (688 vs 683 pre-rebase), CPU-side patch alignment, main's authoritative versions of molmoact2_ae_parity.py (explicit dropout=0.0) + molmoact2_ours_ft_rung_read.py (n_obs_steps truthiness) + the 2 posts, tokenizer FileNotFoundError, styleguide adds. check.py green 688; pushed --force-with-lease 2a31981. Result posted in-channel same session.

<details><summary>full record</summary>

MAIN-AGENT DIRECTIVE (owner-relayed 14:34:48Z 08-11, message.txt attachment): rebase fontaine on main @36afff0 (owner-session correctness reviews of bijou/molmoact2 + bijou/molmo2 landed as code) and adapt — NOT merge-resolve in our favor: (1) ActionExpertConfig now frozen=True/slots=True, NO field defaults, all 15 fields explicit everywhere; released shape ONLY via ActionExpertConfig.released_so100_101(); (2) config factories moved to staticmethods: Gemma4Config.e2b()/.e4b(), Molmo2TextConfig.molmo2_4b() (module-level e2b_config()/e4b_config()/molmo2_4b_text_config() gone; styleguide: released shapes are staticmethod constructors, literals never restated); (3) new loud molmoact2 guards each with a test: --norm-stats must be named norm_stats.json; MolmoAct2Predictor.load refuses n_obs_steps missing or !=1 (require_single_obs); nonzero *dropout* keys in action_expert_config refused; extract_kv_states raises on unfilled cache layer; load_norm_stats requires non-empty setup_type/control_mode; (4) ensure_per_sample_patch_alignment now CPU-side in all three collators; (5) main edited two of our scripts (molmoact2_ae_parity.py build_ours passes dropout=0.0/attn_dropout=0.0 explicitly; molmoact2_ours_ft_rung_read.py n_obs_steps truthiness fix) and two posts (OOB plan Qwen3-8B-&gt;4B-class x2; deep dive parameter-accounting note: 621M and 577,564,448 are the same expert — 620,677,664 minus 36x frozen cross_attn.kv_proj 42,522,624 minus identity state_encoder 590,592, pinned in test_released_config_parameter_count) — main's versions authoritative; (6) styleguide adds: Shapes: docstring bullets on tensor-taking functions, intra-package imports relative, docs state present-truth (no corrected-on trails); (7) Molmo2TextTokenizer raises FileNotFoundError not SystemExit on missing tokenizer.json. Sister ask (check.py green on artifact-less clones) DONE d7b6864 same session (option a: frozen stage-01 analysis committed, oracle chain clone-verifiable).

</details>

---

**`sim100-postprocess`** · `cpu`

100-seed sim eval postprocess (successor, executable on unit fontaine-sim100b rc=0 ~03:2xZ 08-12): (1) verify ALL PHASE2 ARMS DONE + rc=0 in ~/logs/sim100_eval.log; (2) frozen reads sim100_reads.py --in-dir outputs/sim/eval100 --…

**boundary:** CLOSED 03:4xZ 08-12 work session, all 6 steps: (1) rc=0 + ALL PHASE2 ARMS DONE verified 03:16:37Z; (2) frozen reads analysis__sim100_seed_eval.json (5 arms, gates green, ordering auto-skipped as pre-declared); (3) 4 house charts (engagement split NEW) + report__sim100_seed_eval.html + 14-clip gallery (best/median/worst per arm + 2 er60k reach-but-miss) -&gt; fontaine-reports, curl 200 x5; (4) results post posts/2026-08-12-sim100-results.md (plain-words opener) + reports.md section + numbers in-channel; (5) babysit entry pruned; (6) sim-visual-matching named the lever + encoder-OOD-probe queued as its own item (owner 01:11Z ask). Per-arm numbers were posted in-channel as each arm landed (00:37Z ftrig4k, 01:30Z snap30k, 03:4xZ teacher80k) per the 23:41Z promise. wire-MolmoAct2-rig-ft-into-sim remains offered, not queued. · [pre-reg](posts/2026-08-11-prereg-sim-policy-eval-100seeds.md)

<details><summary>full record</summary>

100-seed sim eval postprocess (successor, executable on unit fontaine-sim100b rc=0 ~03:2xZ 08-12): (1) verify ALL PHASE2 ARMS DONE + rc=0 in ~/logs/sim100_eval.log; (2) frozen reads sim100_reads.py --in-dir outputs/sim/eval100 --out reports/analysis__sim100_seed_eval.json (arms er60k/hold/ftrig4k/snap30k/teacher80k; gates strikes+hold-floor; ordering read auto-skips - rungs killed); (3) charts via sim100_charts.py (phase-2 colors in) + HTML report + video gallery (per-arm best/median/worst; er60k reach-but-miss clips are the money shot) -&gt; fontaine-reports curl-200; (4) results post w/ plain-words opener + reports.md + numbers in-channel: headline is the phase-1 negative (er_60k 0/100, boat untouched 96/100) + whether ANY family engages the boat; (5) prune babysit entry sim100b_eval; (6) name sim-visual-matching as the unblocking lever; wire-MolmoAct2-rig-ft-into-sim is an offered follow-up (owner told 23:41Z it needs a closed-loop adapter + v2.1/v3.0 convention care).

</details>

---

**`sim-policy-eval-100seeds`** · `cpu`

GOAL (owner 17:07Z 08-11): evaluate one good policy (candidate er_60k/step_060000, the reference trunk) in sim on 100 fixed seeds; primary metric = boat-&gt;disk distance reduction (continuous), success rate secondary

**boundary:** CLOSED 03:2xZ 08-12 work session — BOTH PHASES COMPLETE, rc=0 03:16:37Z (phase 2 ~3.5/4 GPU-h; total ~5.5). FINAL: 0/500 successes; gates green (strikes 0/500, hold floor -0.0). Per-arm mean progress / moved&gt;=0.5cm: er60k -0.03 / 4 (96 untouched, reach-over-the-table fingerprint); snap30k -0.12 / 38; ftrig4k +0.08 / 47 (27 toward vs 20 away — the ONLY arm tilted toward the goal, best +3.64); teacher80k -0.73 / 56 (18 toward/38 away, the study's only CI-excludes-zero reads: vs hold -0.73 [-1.18,-0.34], vs er60k -0.70 — the strongest offline policy is measurably worse than doing nothing). THE FINDING: contact tracks family/capability, DIRECTION tracks visual familiarity — checkpoint-quality explanation dead (teacher engages most), visual gap confirmed as the lever. Owner question 01:11Z answered in-channel 01:30Z (diagnostics + SIMPLER recipe). · [pre-reg](posts/2026-08-11-prereg-sim-policy-eval-100seeds.md)

<details><summary>full record</summary>

GOAL (owner 17:07Z 08-11): evaluate one good policy (candidate er_60k/step_060000, the reference trunk) in sim on 100 fixed seeds; primary metric = boat-&gt;disk distance reduction (continuous), success rate secondary. Local GPU inference-only while sims run. Protocol (seed list, horizon, metric definition) pre-registered before running.

</details>

---

**`sim-lit-review`** · `cpu`

Sim + sim-to-real literature review (CPU, OWNER DIRECTIVE 17:07Z 08-11

**boundary:** CLOSED 18:5xZ 08-11 work session: 3 Papers pages landed same-session (papers/sim-as-eval.md, papers/so101-sim-landscape.md, papers/sim-contact-fidelity.md) via 3 parallel research agents, all links fetch-verified. Headlines: (1) SIMPLER lineage = the protocol design citations (sysid+visual-matching MMRV 0.056/r 0.924; controller gains FIRST-order for eval fidelity, friction values second-order; continuous progress metrics separate policies at up to 70% fewer trials than binary success = owner's distance metric vindicated; AutoEval 0/50-sim-vs-47/50-real caution: fidelity is per-policy-family; SureSim paired-rectification for when rig rollouts exist; 2026 head-to-head: simulator choice moves Spearman 0.400-0.700 on identical real evals). (2) Census: NO public SO-101 sim eval with continuous metric exists — 2026 SO-101 benchmarks all real-world; our substrate leads; steal LIBERO frozen init-states, so-frame REAL|SIM|OVERLAY; LeRobot EnvHub = later publication channel. (3) All 4 sim-review findings have documented mechanisms + named fixes (CoACD threshold-not-cap or native SDF — the SDF path also fixes the CC-BY-ND per-machine asset hazard; priority override is spec, fix = explicit contact pair + condim&gt;=4 + elliptic cones + impratio~10; BAM ships identified STS3215 model). (4) NEW FINDING from asset diff: our menagerie model kp 998.22/forcerange ±2.94 vs TheRobotStudio upstream kp 17.8/±3.35 for the same servo — ±2.94 is exactly the review's measured saturation; sysid question queued. ideas.md #16 fed; successor items sim-fixes-reset-contact + sim-servo-sysid queued.

<details><summary>full record</summary>

Sim + sim-to-real literature review (CPU, OWNER DIRECTIVE 17:07Z 08-11 — explicitly re-opens the paused lit lane for sim topics): (a) simulators/task suites usable for so101-class tabletop manipulation (existing so100/so101 ports, MuJoCo-family and other ecosystems) — final rig is so101; (b) sim-to-real transfer + the inverse (eval fidelity: sim success as a policy-quality metric alongside BC MAE). Papers pages with plain-words openers.

</details>

---

**`sim-review`** · `cpu`

Sim review (CPU, OWNER DIRECTIVE 17:07Z 08-11 — next-day focus is simulations): review sim/ (so101_sim.py, rollout_sim.py, convert_benchy.py, demo_scene.py, view.py, probe_*.py, fetch_assets.sh)

**boundary:** CLOSED 18:1xZ 08-11 work session (commit f14948f): findings post posts/2026-08-11-sim-review-findings.md + two committed probes (sim/probe_benchy_contact.py, sim/probe_phantom_volume.py). Contract seam ALL GREEN (camera kinds: judge stamped rig 'front' as kind top so sim 'top' naming matches training tags; kind-sorted image order identical; SO_MOTORS order + degrees match; er_60k stats table has both rig repos; AR-greedy + B=1 =&gt; deterministic, measured bit-identical qpos AND renders; 26.9 ms/tick =&gt; ~20 min sim-side per 100-seed eval). FOUR FINDINGS, fixes NOT executed (findings-first): (1) home pose UNREACHABLE - camera_box2 (wrist-cam mount) jams into a shoulder geom, elbow/wrist_flex/wrist_roll pinned at +-2.94 force limit, elbow -19deg steady-state, wrist_roll bimodal (0 or ~15deg) per seed = seed-dependent start state; (2) 2/20 seeds the arm STRIKES the boat during the reset settle (zero-qpos start lays the arm over the workspace), up to 30.4 mm pre-episode displacement; (3) phantom collision margin median 0.34 / p99 3.78 / max 5.39 mm, 74% of collision surface outside the visible boat, CoACD concavity 0.149 at the 16-hull cap (vs 0.05 asked); (4) grasp seam: 2.5 mm penetration acceptable BUT gripper priority=1 overrides the benchy drift-fix friction =&gt; torsional 5e-3 at the seam, 6.9deg in-grip spin + tilt to upright 0.84 on lift; rest drift 0.000 mm (drift fix healthy). INFRA: menagerie commit UNPINNED in fetch_assets.sh + CoACD assets regenerated per machine =&gt; cross-machine trajectory repro NOT guaranteed (pin one eval machine in the protocol); EGL libs installed on this box (libegl1 + libnvidia-gl-580); success() docstring claims a gripper-open check the code lacks. Protocol implications written into the post's final section.

<details><summary>full record</summary>

Sim review (CPU, OWNER DIRECTIVE 17:07Z 08-11 — next-day focus is simulations): review sim/ (so101_sim.py, rollout_sim.py, convert_benchy.py, demo_scene.py, view.py, probe_*.py, fetch_assets.sh) — map capabilities: observation surface vs the policy input contract (cameras, proprio), actuation/stepping model, seed &amp; determinism story, asset pipeline; investigate the owner-reported boat (benchy) contact-physics problem (findings first, fixes after). Output: findings report; feeds the 100-seed eval protocol pre-reg.

</details>

---

**`rig-mixture-instrument-prereg`** · `cpu`

Rig-mixture lever, step 1 (CPU, executable now): implement the per-root --dataset-repeat flag + oracle test exactly as pinned in the er-60k pre-reg mixture note (loader dedups repeated roots today, so there is no zero-code oversa…

**boundary:** CLOSED 17:0xZ 08-11 work session, both legs: (1) INSTRUMENT landed commit 1b1c314 — --dataset-repeat PATTERN=COUNT in bijou.train/bijou.data (fnmatch per-repo specs, first-match-wins, spec matching no selected dataset is FATAL, replicas share objects so no host-RAM cost; training-only, eval call sites never repeat; TrainArgs field defaulted for old-checkpoint replay); oracle test tests/test_dataset_repeat.py (16 tests: parse/precedence/no-match/concat expansion + the pinned 0.19%-&gt;4.97%@27x mixture arithmetic); check.py green 683. (2) PRE-REG DRAFT posted posts/2026-08-11-prereg-er60k-rig-mixture.md: single arm --init-from er_60k/step_060000 (weights-only suffices, optimizer state died with the box), explicit specs clean=27 v2=27 (~4.97% share, no wildcard), 10k steps eff-48 seed 3 warmup 500, primary read = paired CI95 on the deterministic rig holdout (1+5 episodes ~3.7k frames er_60k never trained on) vs the endpoint, guard = panel non-regression band +0.05 vs banked 5.7782; COMPUTE ASK named: (A) new 4x box ~28 GPU-h/7h wall gate 32, (B) local 1xH100 needs act-ckpt fit-preflight (full recipe measured OOM single-GPU 08-08), (C) defer. HOLDING for owner steering — successor item rig-mixture-screen-exec carries the hold. · [pre-reg](posts/2026-08-11-prereg-er60k-rig-mixture.md)

<details><summary>full record</summary>

Rig-mixture lever, step 1 (CPU, executable now): implement the per-root --dataset-repeat flag + oracle test exactly as pinned in the er-60k pre-reg mixture note (loader dedups repeated roots today, so there is no zero-code oversample), then draft the mixture-screen pre-reg on the new reference trunk er_60k/step_060000: rig at ~5% effective share (~27x repeat, inside the CL-triangle 2-20% replay band) vs the natural-share 0.19% passenger baseline this screen just banked. The ER results post names this lever explicitly unpriced. GPU leg NOT launchable without owner input: the 4x box is gone — the pre-reg draft must name the compute ask (local 1xH100 rung vs a new box) and hold for owner steering.

</details>

---

**`ftrig-eval20-flipped-parallel`** · `gpu-local`

OWNER PRIO (15:27Z 08-12): re-run the ftrig 20 episodes on flipped-mount physics (d5cf9fd) with parallel workers for speed

**boundary:** CLOSED 16:4xZ 08-12 work session, ridden end-to-end incl. an owner-caught CORRECTION (~0.36/0.5 GPU-h total): pre-reg + results + correction in posts/2026-08-12-prereg-ftrig-eval20-flipped-parallel.md. First readout (paired ~null, 18/20 bit-identical) measured only the collision-box half: owner spotted 16:07Z that videos showed the bracket unmoved -&gt; root cause MuJoCo geom_sameframe fast path silently ignoring runtime geom_pos/quat edits on the visual mesh (flag 2) -&gt; one-line fix (clear the flag, so101_sim.py) + postflip arm rerun. TRUE flip effect: 13/20 seeds changed (policy is vision-driven, bracket visible in top cam); knock-aways 6-&gt;2 (s4 -12.3 -&gt; -0.05, s5 -5.5 -&gt; +0.1), mean -1.21 -&gt; -0.46 cm, paired +0.75 cm CI95 [-0.33,+2.26] (n=20 rough, crosses zero); character shift toward freezing over shoving. Physics-side claims (control loss -62%, sweep) box-driven, stand. LESSON REGISTERED: any runtime geom_pos/quat edit must clear geom_sameframe. Also banked: lockstep-parallel bit-reproducibility; parallel-vs-seq outcome drift (11/20 &gt;0.1 cm, max 6.0). OWNER EXTENSION 16:37Z executed same session: step-500 checkpoint converted fresh (bijou.convert_molmoact2 -&gt; outputs/converted/molmoact2_rig_r1_step500) + same 20 seeds on the fixed sim: mean +0.02 vs step2000's -0.46 cm, paired +0.48 CI95 [-0.06,+1.13], 9 better/3 worse/8 tied, knock-aways 1 vs 2, best s0 +1.59 - extra 1500 ft steps buy no sim-side competence (overfit-to-rig-appearance consistent). Day total ~0.45/0.5 GPU-h, 5 arms. Rows + 80 videos + stills on fontaine-reports /ftrig_eval20_flip_parallel/ (curl 200); full exchange in-channel 16:02-16:5xZ. · [pre-reg](posts/2026-08-12-prereg-ftrig-eval20-flipped-parallel.md)

<details><summary>full record</summary>

OWNER PRIO (15:27Z 08-12): re-run the ftrig 20 episodes on flipped-mount physics (d5cf9fd) with parallel workers for speed. Design (confirmed in-channel 15:28Z): BOTH arms parallel workers=8 — pre-flip (flip disabled) + post-flip, same 20 seeds as molmoact2_ftrig_eval20, paired per-seed = the sanctioned within-parallel-path read (parallel oracle FAILED 14:37Z: rows are rough/exploratory, never registered-comparable to sequential banked rows; state the asterisk in the readout). Reads: paired progress_final_cm delta (flip effect), knock-away count change (4/20 pre-flip, bracket-collision hypothesis), videos. Est ~2x ~5 min wall, &lt;=0.5 GPU-h. Rough numbers + videos in-channel same session.

</details>

---
