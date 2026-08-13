# Pre-reg (FINAL): token-GRPO phase 2 RE-SCOPE — R0-A smoke + R1-A on the patch-only surface

*2026-08-13 21:5xZ (real `date -u` at freeze: 21:52). The NEW
pre-registration the [R0 STOP
boundary](2026-08-13-prereg-token-grpo-phase2-run.md) registered as
its fallback rule: R0's frozen reads failed on VRAM (76.53 GiB ≥ 75,
option B measured-marginal on 1×H100) and on-surface signal (one
gradient step at lr 5e-6 collapsed sampling diversity and held-out
competence), and §4's named retreat arm — **option A, patch-only** —
goes in-channel as its own pre-reg, not an in-run swap. Executed under
the owner delegation (11:07Z "you make the decisions"; 11:18Z "don't
wait for my confirmations"). Instrument landed and oracle-gated this
session (`69b03e8` + the KL-line commit this post rides; 18 loop
oracles, check.py 866 green). The launch immediately follows the
push.*

## Plain words

The first real training run (R0) taught us two things the hard way:
the "train the whole language stack" variant barely fits on our one
GPU, and even a single update step at the standard learning rate was
enough to wreck the policy — its 8 samples per scenario started coming
out bit-identical (no diversity left to learn from) and its test score
collapsed. This re-scope is the registered plan B: train ONLY the tiny
slice of the network that owns the action vocabulary (~10.5M of ~4B
weights — the memory problem disappears outright), and fit the run
with four restraints priced off R0's own curves: a 5× lower learning
rate, a cap on how hard any single lucky rollout can push the update,
a penalty that actively pulls the policy back toward its starting
point, and a test-set check after EVERY step so damage is visible
immediately instead of at the end. The known risk flips direction:
this surface may be too weak to move behavior at all — that's exactly
what the 2-step smoke is priced to find out, and "inert" has its own
frozen stop rule.

## What changes vs R0 (and what doesn't)

Everything not listed here is **frozen unchanged from the [R0
pre-reg](2026-08-13-prereg-token-grpo-phase2-run.md)**: checkpoint
`allenai/MolmoAct2-SO100_101` fresh from the hub (R0's step_0001/0002
are collapsed-policy diagnostics, consumed by nothing), same shim/norm
tag/FAST artifact/frames v3, S=8 × G=8 at T=1.0, same reward and
z-score advantages (ddof=0), clip-higher [0.8, 1.28], same seed
streams (train `1000 + 8·step`, held-out 200–219, run_seed 0 — the
same-seed policy: a fresh run, frozen for wave-for-wave comparability
with R0), fp32 text stack for rollout AND replay, AdamW(0.9, 0.95,
eps 1e-6, wd 0, foreach=False), grad-clip 1.0, microbatch 1,
`release_cached_vram()` before every wave/eval, resume path
oracle-pinned.

The five deltas, each priced off an R0 measurement:

1. **Surface = option A (patch-only)** — trainable set is EXACTLY the
   FAST-block rows `[151934, 153982)` of the untied `wte.embedding` +
   `lm_head` (2×2048×2560 ≈ 10.5M params; enforced by a post-backward
   row mask, oracle: every row outside the span bit-identical through
   a real step). Dissolves the VRAM fail: no transformer grads/Adam
   (R0 measured those at ~45 of its 76.5 GiB). Priced risk: no
   published precedent this narrow; it cannot change the trunk's
   computation, only re-map action-token embeddings/logits — but
   logit re-mapping is exactly where R0's sharpening lived
   (chosen_nll 0.77→0.33 is a softmax-confidence move), so the
   surface can plausibly both learn and still collapse; the levers
   below assume it can.
2. **lr 1e-6** (5× down, the registered 5–10× band's conservative
   end for a first read on an unprecedented surface).
3. **Advantage clip ±2.0** — R0's wave-0 fed 4/64 successes as
   z ≈ +2.65 outliers into one overshooting update; the clip tempers
   the lone-success push ~25% and preserves ranking.
4. **KL penalty ON, β = 0.5, differentiable** — β·k3 to the step-0
   anchor per trained token, inside the objective (heartbeat
   `anchor_k3_pre`). The penalty compares two replay forwards on the
   SAME decoded frames, so the JPEG noise floor cancels (exactly 0 at
   an unmoved policy — oracle-pinned); the recorded `anchor_kl`
   telemetry does NOT cancel it (floor ≈ 0.0215, R0's own step-1
   reading at the anchor). Pricing: β·k3 reaches R0's surrogate scale
   (|loss| ≈ 0.032) at k3 ≈ 0.064 — between the floor and R0's
   post-collapse 0.0885 — so the penalty dominates the update before
   drift reaches the measured destructive scale. Cost: one extra
   no-grad replay pass per step (~0.1 GPU-h, priced in).
5. **eval-every 1 + the §7 KL numeric line mechanized** — held-out
   greedy paired eval after EVERY step (R0's damage took ONE step;
   the endpoint-only read saw it two steps late), and `--kl-stop
   0.06`: one `anchor_kl` reading above 0.06 (≈3× the 0.0215 noise
   floor, below the 0.0885 collapse reading) stops the loop — no
   streak, because the rollout-vs-anchor telemetry lags the update by
   a step. This closes R0's registered promise ("KL numeric line set
   at the boundary from the measured scale").

## Ladder + budget

R0 banked the pace book: 0.68–0.76 GPU-h/step (rollout-dominated) on
this exact stack. Per-step here ≈ rollout 0.7 + gradient ~0.15 +
anchor pass ~0.1 + eval 0.19 ≈ **~1.1 GPU-h/step**. ~31.2 GPU-h of
the 35 ladder total remain after R0's ~3.8.

| rung | steps (cum) | boundary reads | budget |
|---|---|---|---|
| R0-A smoke | 1–2 | GO to R1-A iff ALL: rc 0 with ratio ∈ [0.95, 1.05], clip < 0.2; wave signal alive (median group std ≥ 0.25 cm AND ≥ 8/16 groups pooled non-degenerate, NO R0-style wave-2 collapse); per-step held-out paired CI never entirely below −1.0 (in-loop) and endpoint Δ CI not entirely below 0; anchor_kl ≤ 0.06 every step (in-loop); VRAM peak < 75 GiB; R1-A projection ≤ 22 GPU-h cum. **Inert rule**: if the policy shows no measurable movement (anchor_kl within noise of the 0.0215 floor at BOTH steps AND endpoint paired Δ = 0.0 exactly AND chosen_nll drift < 0.005), R1-A does NOT auto-launch — the lr/β re-price goes in-channel as an addendum first (13 more GPU-h of inert steps is the new waste mode this surface makes possible). | ≤ 3.0 GPU-h ops gate |
| R1-A | 3–17 | the §6 frozen reads at the step-17 endpoint (primary: paired Δ CI95; knock-away rate; success count); resumes `step_0002.pt` via the R0-validated resume path (anchor snapshotted pre-restore) | ~16.5 at measured pace; stop before launch if projection > 22 cum |
| R2-A (conditional) | +K | full frozen reads; K = what fits **35 GPU-h total** at measured pace | remainder |

R1-A→R2-A rule unchanged from R0 (no tripwire fired; endpoint paired
CI not entirely below 0). Beyond R2-A is a new pre-reg. All R0
tripwires stay armed (strikes, non-finite loss, spread collapse ×3,
violence ×3, competence floor −1.0) plus the new KL line; the
knock-away watch item transfers (R0 step-1 measured 0.234 vs the
0.167 line — R0-A's own baseline lands at its boundary).

## Command (verbatim) + ops

```
MUJOCO_GL=egl fontaine/scripts/run_detached.sh fontaine-grpo-r0a \
  uv run python -m sim.grpo_loop \
  --checkpoint allenai/MolmoAct2-SO100_101 \
  --out-dir outputs/sim/grpo_phase2_a --total-steps 2 \
  --surface a --lr 1e-6 --kl-beta 0.5 --advantage-clip 2.0 \
  --kl-stop 0.06 --eval-every 1 --save-every 1
```

(All other flags at their frozen defaults, which ARE the constants
above; fresh out-dir keeps R0's diagnostics intact.) R1-A: `--resume
outputs/sim/grpo_phase2_a/step_0002.pt --total-steps 17` with the same
flags. Babysit registry entry at launch (`train-jsonl`, probe
`eval_reward_mean`, vram key `vram_gib`, gate 75 GiB); expected VRAM
~45–55 GiB peak (the read is recorded either way). Checkpoints are
~3.2 GB (two matrices), `--save-every 1`; a GO boundary uploads
`step_0002.pt` to `fontaine-checkpoints` only when R1-A consumes it
(weights-only rule).

*Frozen at commit time; the launch immediately follows the push.
Amendments only via numbered addenda below.*

---

**Addendum 1 (2026-08-13 21:5xZ — launch-env fix + relaunch, no
constant changed).** Launch 1 (21:55:48Z) died at 21:56:21Z in its
FIRST sim worker: `mujoco.FatalError: an OpenGL platform library has
not been loaded` — the command's `MUJOCO_GL=egl` prefix sets the env
for `run_detached.sh` itself, but the transient systemd unit gets the
USER MANAGER's clean environment (only PATH/HOME were forwarded), and
that manager env no longer carries `MUJOCO_GL` (R0's four launches
inherited it from a state that has since been lost). Zero GPU-h
consumed (died pre-load, ~1.6 min CPU). Fix, both belts:
`systemctl --user set-environment MUJOCO_GL=egl` (manager-durable),
and `run_detached.sh` now forwards `MUJOCO_GL` into the unit whenever
the caller sets it — the verbatim command's semantics, made real.
Launch 2: **21:58:04Z**, same command bit-for-bit.*
