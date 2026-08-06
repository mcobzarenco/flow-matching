# 2026-08-06 — Pre-registration: Molmo2-4B AR trunk, 40k × 4×DDP (`fontaine_molmo2_ar_40k_ddp4`)

*Status: PRE-REGISTERED before launch. Owner steering 2026-08-06
18:10Z "I want to run molmo2 tonight" + 18:12Z "Agreed" + 19:11Z
"Agreed on molmo2 on the 4x box … smoke test with ddp enabled" — the
[port plan](2026-08-06-molmo2-port-plan.md) §6 AR-first amendment's
phase-1 run. Code: the AR decoder arm landed this evening
(`c1119fb`+`f569f94`+`4ce9136`, check.py 337 green; keystone oracle:
prefill+continue ≡ monolithic multimodal forward under left padding).*

## 1. Question

Does a **video-grounded VLM trunk** (Molmo2-4B: spatio-temporal
pointing/tracking pretraining, Qwen3-4B decoder) beat our **text-first
Gemma-4 E2B** trunk as an AR VLA — FAST actions + aux text on the live
trunk — at matched data, matched steps, matched action tokenizer? The
grounding probes located E2B's error in frame-dependent level
mis-estimation and weak visual-token use (#11); Molmo2's pretraining
objective is the nearest open-weights match to "where is the gripper
and what is it doing". The untrained-gen probe (18:4xZ) showed the raw
trunk already reads our rig scenes accurately under our exact prompt —
night-and-day vs Gemma's refusals.

## 2. Recipe (one variable: the trunk; plus the scale-out it needs)

Mainline `arb_rcond` recipe verbatim where the trunk allows it:

- `--decoder ar_backbone --backbone allenai/Molmo2-4B --max-crops 1`
  (410 image tokens/camera — the smallest layout inside the shipped
  distribution; Amendment-2 pixel-math rationale)
- FAST tokenizer v2 (`vocab_total` 1,026), block anchored at
  **[152,064, 153,090)** — the second extension block
  (`fast_block_base`); trainable = `fast_embed` + fresh untied
  `fast_head` rows + decoder blocks + `ln_f` + `state_proj`;
  **frozen = `wte` (both matrices) + shipped `lm_head`** (the 18:1xZ
  freezing split; aux text reads the frozen head, grads flow through)
- aux fields `subgoal holding progress event visible`, aux-dropout 0,
  field-dropout 0.1; conditioning `subgoal outcome smoothness`
  (0.1/0.5 dropouts); instruction-augment 0.5; camera-kind-dropout 0.1
- `--decoder-lr 1e-4 --backbone-text-lr 2e-5 --grad-clip 100`
  (vision tower frozen — no `--backbone-vision-lr`; a vision-unfreeze
  rung is a follow-on, not this run)
- data: `community_curated_v0`, fps 30, camera-counts {1,2}, holdout
  0.1 @ split-seed 0 (E1 gate: **878 datasets / 38,571 episodes /
  18,636,749 frames / dims 6/6** — byte-identical to the arb_rcond
  mainline banner; any deviation aborts. Verified in the smoke ✓)
- **40k steps, 4×DDP, B12/rank (global 48)** — the e4b-screen scale-out
  rung; batch semantics FROZEN at launch. Declared confound vs the
  1×GPU B10 E2B mainline: global batch 48 vs 10 — the e4b screen
  carries the same one, and the comparison is a screen, not a paired
  ablation. Warmup 1000, seed 0, save 2500, eval 500.

## 3. Gates before launch

- F1 memory smoke (DDP4, 150 steps of the exact recipe, owner-asked):
  rc=0 AND peak ≤ ~75 GiB/GPU. Ladder: B12 direct → B12 chunked 2×6 →
  B8 chunked 2×4. **Rung 1 (B12 direct): OOM at 77.5 GiB in the
  forward (MLP intermediates; measured 19:5xZ) — REJECTED. Rung 2
  (B12 chunked 2×6, global 48 unchanged, gradient exactly equivalent):
  TODO_SMOKE_PEAK MiB peak, TODO_SMOKE_RATE s/step (last 5), rc=TODO_RC.**
- F2 wall: 40k × rate ≤ 30 h ⇒ full 40k; else this pre-reg SHRINKS to
  a 10k screen (label changes to `_10k`), no mid-run change.
  **Projected: TODO_WALL h.**
- Box pytest green at the launch commit (337 passed, run 19:4xZ ✓);
  the smoke also exercises eval/probe decode + the wandb metric path
  (offline) + the Molmo2 checkpoint writer at step 100.

## 4. In-run instruments and kill line

- Probe (eval-every 500): AR greedy decode chunk MAE on the probe set
  — the same instrument as every arb run. **New-trunk band: none
  declared** (no Molmo2 prior); anchors for READING (not killing):
  E2B arb family probes descend into ~10–11 @40k.
- **K1 kill line**: NaN/inf loss, OR the probe fails to descend below
  its own step-2500 value by step 10k, OR probe > 25 sustained for 3
  consecutive evals after step 5k. Kills wait for save boundaries.
- @10k, @20k, @30k: record-only milestone notes (Discord), no
  discretionary kills — the panel decides.

## 5. Endpoint reads (frozen before data, this section is the contract)

Chained after 40k, 4-GPU sharded, the family voice:
`panel_curated_v0_k4l2` (plans/holdout_curated_v0_k4l2.json, holdout
0.1 split-seed 0), stem
`eval__fontaine_molmo2_ar_40k_ddp4__step_040000__panel_curated_v0_k4l2`.

- **Read 1 (primary):** panel pooled chunk/first MAE vs the E2B AR
  anchors — A-s0 mainline **7.7966 / 3.9422** (same panel, same
  plan). Classification: BEATS if pooled chunk < 7.30 (−0.5, ~the
  family's seed spread); PARITY within ±0.5; WORSE beyond. Paired
  per-frame Δ + CI95 via the npz where row alignment holds (same
  plan ⇒ it should; a pairing failure is reported, not silently
  pooled).
- **Read 2:** state-copy fallback rows byte-match the banked panel
  values (11.7639/2.5851) — instrument integrity, not a result.
- **Read 3 (context):** vs e4b screen milestones (7.54@10k probe
  family) and arm C statedrop (10.50) — narrative only.
- Decision: BEATS ⇒ Molmo2 becomes the phase-2 flow-trunk candidate
  (frozen AR-adapted prefix — kills the −2.7 confound); PARITY ⇒
  grounding-probe follow-ups decide; WORSE ⇒ clean null banked, E2B
  stays mainline (VLM4VLA says nulls are modal — transferable either
  way).

## 6. Cost

~40k × TODO_SMOKE_RATE s/step ≈ TODO_WALL h on 4×H100 + ~2 h panel
eval. Checkpoints: full-trunk snapshots (~9.7 GB bf16) every 2500 —
~160 GB transient, pruned to milestones at the boundary (disk 6.6 T
free).

## Finalization amendment (pre-launch)

Filled from the smoke before `torchrun` fires; the launch commit hash
and the filled numbers are the launch record.
