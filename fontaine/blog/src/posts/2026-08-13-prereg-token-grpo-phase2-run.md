# Pre-reg (FINAL): token-GRPO phase 2 — R0 smoke + R1 on the MolmoAct2 discrete pathway

*2026-08-13 14:5xZ (frozen at commit `8548969`, 14:55:52Z; stamp
corrected from a mis-clocked 15:1xZ before first publication — no
content change). The finalization the [design
memo](2026-08-13-token-grpo-phase2-design.md) §9.4 promised: constants
frozen, HEAD + checkpoint pinned, launch follows the commit. Executed
under the owner delegation (11:07Z "you make the decisions, ensure we
make progress and GPU is always busy"; 11:18Z "don't wait for my
confirmations") with the memo's open forks resolved by the frozen
rules: **surface = §4 option B** (the veto window, open since ~06:0xZ,
passed unanswered; B was the recommendation), **pathway = the
molmoact2 discrete (AR) head** (owner 10:02Z steering; the
[ar100 gate](2026-08-13-prereg-molmoact2-ar100.md) proved it
success-capable, and arm B made grammar-masked decode the serving
mode). Instrument items 1–4 are all landed and oracle-gated
(`418715c`, `229d80f`, `a268046`, `fa739e9` — check.py 861 green);
this run is the first thing the loop harness executes.*

## Plain words

Everything before this was preparation: a probe showed the policy's
sampled attempts spread out enough to rank, a gate eval showed the
token pathway can actually succeed in our simulator, and three build
sessions produced the machinery that records every sampled action
token with its probability, replays them through the trainer, and
verifies — bit for bit — that the trainer is scoring exactly the
distribution the robot sampled from. This post freezes the first real
training run: the robot re-tries fresh scenarios 8 times each with
mild sampling noise, attempts that do better than their siblings get
pushed up, worse ones get pushed down, and a battery of tripwires
stops everything if the policy starts learning violence, collapsing
its diversity, or getting worse at the held-out test. It is deliberately
laddered — a 2-step smoke to measure the true pace and check the
signal survives on this model, then a 15-step read — with hard budget
gates and decision boundaries between rungs.

## Pinned identities

- **Code**: HEAD `fa739e9` (loop harness `sim/grpo_loop.py`; rollout
  + replay `a268046`; GRPO step `229d80f`; capture `418715c`).
- **Checkpoint**: `allenai/MolmoAct2-SO100_101` (hub release), served
  by the parity-gated first-class port; official SO-101 shim (signs
  `1,-1,1,1,1,1`, offsets `0,90,90,0,0,0`), norm tag
  `so100_so101_molmoact2`; FAST artifact
  `allenai/MolmoAct2-FAST-Tokenizer`.
- **Anchor**: the loaded checkpoint at step 0 (the KL reference; a CPU
  snapshot of the trainable tensors).
- **Frames**: v3 (production default at finalization; sim100
  amendment 5 — v4 shadows — remains an open owner ask and does NOT
  ride this run). Flipped mount, sim100 episode conventions,
  `--episode-seconds 30`, execute-horizon 30, workers 8.

## Frozen algorithm constants (memo §2/§3/§4)

- S=8 fresh spawn seeds/step from the dedicated stream (`1000 +
  8·step`; disjoint from sim100 0–99, probe 0–14, held-out 200–219),
  G=8 grammar-masked sampled draws each at **T=1.0** → 64
  episodes/step, RNG keyed `stable_sample_rng(run_seed=0,
  repo_id(draw), seed, replan, 0)`.
- Reward/episode: `progress_final_cm + 10·success − 2·[upright<0.9] −
  5·[strikes>0]`; advantages = within-group z-scores (ddof=0); groups
  with reward std < 0.05 cm dropped whole.
- One on-policy gradient pass/step (μ=1): advantage-weighted clipped
  token-CE over the action block, clip-higher **[0.8, 1.28]**, ratio
  under the SAME grammar-masked softmax the decode sampled (recorded
  per-token logprobs = π_old; the item-3 oracle bound 1e-5 on the CPU
  fixture; disk rows additionally carry the registered JPEG budget —
  the fixture measured ~1% of ratio at a fresh policy, R0 reads the
  real number). KL penalty OFF; k3 KL to the anchor recorded every
  step (one swapped reference forward, 32-row subsample).
- **Trainable surface (option B)**: the trunk TEXT stack (embeddings
  + transformer + lm_head) at **lr 5e-6** flat, AdamW(0.9, 0.95,
  eps 1e-6, wd 0), grad-clip 1.0, fp32 text (TF32 matmul), vision
  frozen bf16. Registered fallback on §7 instability: option A
  (patch-only) as a NEW pre-reg, not an in-run swap.
- Microbatch = 1 row; chunking is gradient-invariant (oracle-pinned),
  so raising it at a rung boundary is an execution note, not an
  amendment.
- Held-out eval: seeds **200–219, greedy grammar-masked**, at step 0
  (pre-update baseline), every 5 steps, and at the endpoint; paired
  Δ composite reward, seeded 10k-bootstrap CI95.

## Ladder + budget (measured-pace arithmetic)

Rollout pace measured on this exact serving stack (arm B: 100
episodes, 2,996 predicts, 63.1 min at workers 8 → 0.63 min/episode) →
64 episodes ≈ 40 min ≈ 0.67 GPU-h, ×1.0–1.3 for sampled streams and
the fp32 text stack ≈ **0.7–0.9 GPU-h/step rollouts**; gradient pass 1,920 batch-1
teacher-forced fwd+bwd ≈ 0.2–0.5 GPU-h; eval ~0.21 GPU-h per
occurrence → **~1.0–1.4 GPU-h/step estimate — R0 exists to measure
it**.

| rung | steps (cum) | boundary reads | budget |
|---|---|---|---|
| R0 smoke | 1–2 | plumbing rc 0; measured GPU-h/step; ON-SURFACE signal: median group std ≥ 0.25 cm AND ≥ 8/16 groups non-degenerate, else STOP (the probe's condition, transferred); step-1 mean_ratio ∈ [0.95, 1.05], clip_fraction < 0.2, else STOP (replay is not scoring the rollout distribution); KL telemetry sane → set the §7 KL numeric line; VRAM < 75 GiB | ≤ 3.5 GPU-h |
| R1 | 3–17 | memo §6 reads at the step-17 endpoint | re-priced at R0's measured pace; stop before launch of R1 if projection > 22 GPU-h cum |
| R2 (conditional) | +K steps | full frozen reads | K = what fits the **35 GPU-h total gate** at measured pace (memo cap unchanged) |

R1→R2 rule (memo §5 verbatim): extend iff (a) no tripwire fired and
(b) held-out greedy composite at the endpoint is not worse than
step 0 (paired CI not entirely below 0). Beyond R2 is a new pre-reg.

## Frozen reads (memo §6, unchanged)

1. **Primary**: paired Δ composite reward, held-out greedy, endpoint
   vs step 0 — IMPROVED iff CI95 entirely above 0; "phase 2
   promising" = CI-above-0 on this or read 2.
2. **Knock-away rate under sampling**: endpoint 5-step window below
   the transferred 10/120 baseline with binomial CI excluding it
   (recorded against R0's own measured baseline too — this surface's
   number lands at the R0 boundary).
3. **Success count** (record + headline if > 0).
4. Record-only: per-step median group std, non-degenerate fraction,
   KL-to-anchor curve, chosen-token NLL, tip rate, clip fraction,
   ratio extremes, per-seed traces.

## Tripwires (§7, mechanized in the loop — exit 3 + heartbeat row)

Any reset strike in training rollouts; non-finite loss; median group
std < 0.05 cm ×3 consecutive; knock-away rate > 2×(10/120) ×3
consecutive; held-out paired CI entirely below −1.0 cm. KL-to-anchor
runaway: recorded, numeric line set at the R0 boundary.

## Command (verbatim) + ops

R0:

```
MUJOCO_GL=egl fontaine/scripts/run_detached.sh fontaine-grpo-r0 \
  uv run python -m sim.grpo_loop \
  --checkpoint allenai/MolmoAct2-SO100_101 \
  --out-dir outputs/sim/grpo_phase2 --total-steps 2 \
  --eval-every 5 --save-every 1
```

(All other flags at their frozen defaults, which ARE the constants
above.) R1 resumes the R0 checkpoint: `--resume
outputs/sim/grpo_phase2/step_0002.pt --total-steps 17`. Babysit
registry entry at launch (`train-jsonl`,
`outputs/sim/grpo_phase2/train.jsonl`, probe `eval_reward_mean`, vram
key `vram_gib`, vram gate 75 GiB); heartbeat carries
reward/guard/ratio/KL facts per step. Training rows prune after each
gradient pass (~0.3 GB/step transient). Rung-boundary checkpoints
upload to `fontaine-checkpoints` if consumed by the next rung or a
result claim.

*Frozen at commit time; the launch immediately follows the push.
Amendments only via numbered addenda below.*

---

**Addendum 1 (2026-08-13 16:1xZ — plumbing fix + relaunch, no
constant changed).** Launch 1 (14:58:55Z) crashed rc 1 at 15:51:26Z
in the FIRST gradient step: `grpo_objective_sums` moved the caller's
advantages/rollout-logprob tensors to the training dtype but not the
training **device** (cuda/cpu mix — invisible to the CPU oracles; the
exact plumbing class R0 exists to catch). Everything before the step
was healthy: step-0 baseline banked (held-out greedy composite
**1.868, 2/20 successes**), wave 0 complete (64 sampled episodes,
~35 min ≈ 0.58 GPU-h — inside the estimate band), 1,889 training rows
written, mask verification passed, replay forward ran. Fix: normalize
`old_logprobs`/`advantages` with `.to(new_logprobs)` (device+dtype)
at one point in the surrogate — semantics unchanged, check.py 861
green. Relaunch rides the fix commit; the ~0.9 GPU-h of launch 1
counts against the R0 gate (honest accounting: R0 total budget may
land ~3.1 of the 3.5 gate).*

**Addendum 2 (2026-08-13 17:5xZ — memory fixes + relaunch, no frozen
constant changed; R0 ops gate raised 3.5 → 5.5 GPU-h).** Launch 2
(16:15:26Z) reproduced the step-0 baseline bit-identically (1.868,
2/20) and ran wave 0 + the FULL gradient accumulation cleanly, then
OOM'd at 17:12:17Z inside the first `optimizer.step()` — Adam state
init via the `_foreach` path materializes whole-surface temporaries.
The measured fact that matters: **77 GiB PyTorch-allocated at the
step** → the option-B text stack is ~15 GB fp32 (~3.9B params — this
checkpoint is a ~4B-class model, not the 2B the memo's 69.2-GiB
memory precedent was calibrated on; that precedent was measured on
the OLD er60k surface and did not survive the 10:02Z retarget).
Params+grads+2·Adam ≈ 62 GiB steady DOES fit; the two +P transients
did not. Fixes (allocation-shape only, semantics oracle-pinned
unchanged): (1) `AdamW(foreach=False)` — per-tensor step, no
whole-surface temporary; (2) the anchor-KL swap now stages live
weights to CPU for the reference forward instead of holding both
copies on GPU; (3) `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(launch 2 died with 1.36 GiB reserved-unallocated at a 96 MiB
request). Projected steady peak ~68–70 GiB vs the 75 vram gate —
tight; if launch 3 still OOMs, option B is measured-infeasible on one
H100 for this model and the fallback discussion (§4 option A) goes
in-channel as a new pre-reg, per the frozen rule. Gate accounting:
launches 1+2 spent ~1.85 GPU-h on the two plumbing crashes; the R0
**ops** gate rises to 5.5 GPU-h to cover them — the 35 GPU-h ladder
total is unchanged (R2 shrinks by whatever R0 overruns).*
