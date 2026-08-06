# Pre-registration: SnapFlow 1-NFE self-distillation of flow-80k

*2026-08-06 ~00:3xZ. Immutable once posted. Ideas
[#12](../ideas.md) distillation leg (pairs with #1); source recipe:
SnapFlow ([arXiv:2604.05656](https://arxiv.org/abs/2604.05656)),
deep-read this session. Fills the local-GPU queue slot after the
draws chain + fairness probe (charter §3 queue depth ≥ 2).*

> **Amendment 1 (2026-08-06 ~05:5xZ):** σ_draw finalized at **0.0159**
> by the promised
> [finalization amendment](2026-08-06-sigma-draw-finalization.md) —
> 3σ_draw = 0.048 < 0.15, the floor binds: **endpoint adopt-signal iff
> chunk_mae ≤ 6.7732.** All other reads unchanged.

## Question

Can the best flow lineage reach **1-NFE decoding without losing its
panel position** — and does the mean-of-N draws win (run 2: draws-10
Heun-30 chunk_mae **5.365** / first_mae **1.424**, beating the
AR-100k anchor 5.8026/2.1431) survive distillation, turning the
"unconstrained class" caveat into a deployable config? Mean-of-10 at
1-NFE costs ~10 expert evals ≈ one Heun-5 draw — if it holds, the
charter §2 cost caveat on the draws result closes.

Owner alignment: the 2026-08-05 21:48Z exchange pre-stated this exact
branch — "flow's residual case = first_mae grounding edge + (if draws
close the gap) SnapFlow 1-NFE distill + small N." Draws run 2 closed
the gap; this is the follow-through.

## Subject & baseline anchors (all banked)

- Subject: `bijou_flow_artrunk_h1024_40k_ddp2/step_080000`
  (local + box copies; the draws chain's checkpoint).
- Teacher panel, Heun-30 single-draw: chunk_mae **6.6232** /
  first_mae **1.9331**.
- Draws (teacher, Heun-30): draws-10 **5.365 / 1.424**; draws-5 run 3
  in flight — σ_draw lands with runs 3–5.
- AR-100k panel: **5.8026 / 2.1431**.

## Method — SnapFlow recipe, mapped to bijou

Self-distillation, no external teacher, init `--init-from
step_080000`, trunk frozen (already the lineage protocol; trainable =
flow expert + the new target-time embedding φ_s).

1. **φ_s target-time embedding** (the only new parameters):
   zero-initialized two-layer MLP encoding the *target* time s, added
   to the existing sinusoidal time embedding where τ enters the
   adaRMS conditioning. Zero-init ⇒ at step 0 the extended model is
   exactly the teacher (s has no effect); s=t on standard forwards,
   s=0 flags one-step mode. Config-flagged, default absent —
   existing checkpoints load unchanged.
2. **Loss** (paper defaults; ablation-best α=0.5, λ=0.1):
   `L = α·L_FM + (1−α)·λ·L_shortcut`, aux text loss stack untouched
   from the teacher recipe. L_shortcut (paper Eq. 9–10, mapped to
   bijou's flow-time convention; noise end → data end):
   `x_mid = x_noise − ½·sg F(x_noise, s=t_noise, t=t_noise | c)`,
   `v_target = ½·[sg F(x_noise, t_noise, t_noise | c) + sg F(x_mid,
   t_mid, t_mid | c)]`, loss `‖F(x_noise, s=0, t=t_noise | c) −
   v_target‖²`. Stop-gradient targets only — **no EMA teacher**.
   Three expert forwards per consistency sample, one shared prefix
   encode.
3. **Training** — flow-80k recipe verbatim except the pre-registered
   deltas: 30k steps (paper), decoder_lr **2.5e-5** (paper; teacher
   trained at 1e-4 — this is a distillation refinement, not from
   scratch), native cosine + 500-step linear warmup, grad_clip 1.0
   (paper), batch 24 on 1×H100 (the teacher's per-GPU load; DDP2×24
   originally — distillation batch need not match the teacher's
   effective 48, stated not hidden). Everything else (data trio,
   fps 30, camera-counts 1 2, holdout 0.1/seed 0, adaRMS, chunk 50,
   aux weight 0.5) identical to `step_080000`'s `train_args`.
4. **1-NFE inference**: `x̂_data = x_noise − F(x_noise, s=0,
   t=t_noise | c)` — euler-1 with the s=0 switch via an explicit
   eval flag (loud, never inferred from step count).

## Gates & reads

- **Hard validation gate (pre-launch, blocks the run):** (a)
  zero-init identity oracle — teacher checkpoint loaded into the
  φ_s-extended model, s=t forward bit-identical to the unmodified
  model on the CPU fixture; (b) E1-style drift gate — step-0
  extended model, Heun-30 s=t, stride-7 probe subset (2,458 frames)
  reproduces the banked flow npz frame-MAE within 0.05.
- **@10k, record-only probe:** 1-NFE on the stride-7 subset. Kill
  only if 1-NFE probe chunk-MAE exceeds the teacher's own Heun-30
  probe read by > 3.0 (catastrophic non-convergence; SnapFlow's
  claim is endpoint near-parity, so mid-run reads don't kill inside
  that margin).
- **Endpoint (30k): full panel at 1-NFE, single draw** — primary.
  Adopt-signal iff chunk_mae ≤ 6.6232 + max(3σ_draw, 0.15) with
  first_mae co-read vs 1.9331. σ_draw pinned by **finalization
  amendment** from draws runs 3–5 before the endpoint eval is
  opened.
- **Deployment headline read:** mean-of-N at 1-NFE, N ∈ {5, 10}
  (draws machinery already landed). The decision read: does
  mean-of-10@1-NFE stay ≤ 5.8026 (beat AR) at ~10-expert-eval cost?
- Per-step horizon read (paired-analysis protocol) ships with the
  results post — a distill that fixes only late-horizon must not be
  misread at pooled chunk_mae alone.

## Numbered expectations

1. Both validation-gate oracles pass exactly (any miss is an
   instrument finding that blocks launch).
2. Endpoint 1-NFE panel chunk_mae within **+0.15** of 6.6232 —
   modal outcome parity-or-slightly-better (π0.5 1-NFE ≈ 10-step
   teacher on LIBERO; SmolVLA offline MSE −8.3%). **Falsified if**
   > +0.5: SnapFlow does not transfer to this lineage; banked as a
   negative with the @10k/endpoint probe curve explaining why.
3. Mean-of-10 at 1-NFE ≤ **5.8026** (retains the beat-AR read;
   modal ~5.4–5.6, i.e. most of the 5.365 draws win survives).
4. first_mae at 1-NFE ≤ 1.9331 + 0.05 — the grounding edge
   survives distillation.

## Cost & scheduling

~30k steps × (teacher step cost + 2 extra sg expert forwards) on
1×H100 — budget ~12–20 h wall (paper: 12 h/A800 at B4 on a 3B
expert; ours is a ~200M expert at B24 — record actual). Panel at
1-NFE is *cheaper* than Heun-30 (1 vs 60 expert evals/draw).
Launches at the first quiet local-GPU boundary after the draws
chain + fairness probe (~09–10Z-ish), never co-located with a live
run (charter §3).

## Pre-launch implementation checklist (CPU items, GPU-busy windows)

1. φ_s embedding behind a config flag + checkpoint-compat loading.
2. `bijou.train --distill snapflow` (α, λ frozen at 0.5/0.1):
   mixed loss, sg shortcut targets, shared prefix encode.
3. Eval 1-NFE switch (explicit flag) through `bijou.eval` + report
   `scoring semantics` fields.
4. Oracles: CPU mixed-loss fixture, zero-init identity, probe drift
   gate. `check.py` green before any launch.
5. Launcher staged + diff-verified vs the teacher recipe (E4B
   protocol).

Also read this session (lit slice, banked in ideas): OFP
([arXiv:2603.12480](https://arxiv.org/abs/2603.12480)) — from-scratch
one-step alternative, reserve if SnapFlow misses; GoldenStart
(2603.14245) — RL/online Q-guided priors, screened out for our
offline setting; **Golden Ticket
([arXiv:2603.15757](https://arxiv.org/html/2603.15757v1)) — a single
searched noise vector, inference-only, gains grow at fewer steps** —
banked in #1 as a cheap eval-side follow-up (our panel gives the
offline search criterion their rollout search lacks; pairs with
1-NFE + mean-of-N).
