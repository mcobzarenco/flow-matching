# FPO: the training loss as a stand-in for likelihood

*Read 2026-08-09 (lit slice `lit-radar-0814`, priority 4: #16
RL-pole roster completion). Paper:
[2510.09976](https://arxiv.org/abs/2510.09976) — "Reinforcement
Fine-Tuning of Flow-Matching Policies for Vision-Language-Action
Models" (Lyu, Sun, Lin, Li, Chen, Zhao, Zeng — Yi Zeng's
CASIA/BrainCog group; v1 2025-10-11, v2 2026-06-25, **ICRA 2026**).
Method name π₀-FPO. Priority flag up front: the name "Flow Policy
Optimization" and the core ratio trick trace to McAllister et al.'s
earlier "Flow Matching Policy Gradients" (~Jul 2025), which this
paper cites but never compares against — read this as the VLA-scale
packaging of that route, not its origin.*

**The paper in plain words.** PPO's central quantity is a ratio: how
much more likely did this action become under the new weights? For a
flow-matching policy that number is essentially uncomputable — you'd
have to integrate an ODE with a Jacobian-trace term per action. FPO's
move is to never compute it: use the *change in the flow-matching
training loss* on the action as a stand-in. If the new weights
reconstruct the action at lower CFM loss, call it "more likely,"
normalize across the batch, exponentiate, and drop the result into
the standard PPO clipped objective. Add a conservative Q-ensemble
critic for advantages and temporally-correlated exploration from
multi-step noisy Euler rollouts, fine-tune π₀ in simulation with
sparse rewards, and LIBERO average rises to 87.2% while ALOHA
Transfer Cube goes from ~40% to 65%+.

## What it contributes

- **The likelihood-free ratio.** Per action:
  Δℓ = ℓ_cfm(θ_old) − ℓ_cfm(θ), batch-normalized, mapped through
  exp(β·z) — this replaces the PPO probability ratio in the clipped
  surrogate. The justification is a stated "mild local monotonicity
  assumption" (CFM-loss decrease ⇔ conditional-density increase):
  order-preserving heuristic, no proof, free temperature β.
- **Critic and exploration machinery.** A conservative Q-ensemble
  (min over TD targets) with GAE for advantages — the "structure-aware
  credit assignment" of the abstract is credit to whole latent action
  chunks, *not* per-denoising-step decomposition. Exploration comes
  from multi-step noisy Euler integration in latent space
  (temporally correlated, no SDE machinery). On-policy, sliding
  buffer, clipping as the only trust region.
- **Trunk handling**: initialized from released π₀ with the decoder
  frozen — only the flow actor and critics update.

## The experiments it ran

Sim only: LIBERO (four suites) + ALOHA Transfer Cube. LIBERO average
**87.2%** (Spatial 97.2 / Object 97.3 / Goal 89.4 / Long 65.3) vs
π₀-FAST 85.5, VLA-RL 81.0, OpenVLA 76.5. ALOHA: π₀ ~40% → **65%+**
(the paper's only clean same-base before/after; learning curve runs
to ~1.6M steps — the sole compute-adjacent number disclosed). The
ablation (one LIBERO task, full method 78.5%): remove the CFM-ratio
proxy → **32.4%**; remove clipping → **45.1%**; single-step
exploration → 61.7%; single critic → 71.2%. Missing: env counts,
hyperparameters, GPUs, demo counts, reward definitions, any OOD
test, and any head-to-head against Flow-GRPO, ReinFlow, or
McAllister's FPO.

## What transfers to us

Roster entry 6 for the [#16](../ideas/16-rig-transfer-benchmark.md)
RL pole, and it fills the missing third answer to "how does a policy
gradient reach a flow policy":

1. **The gradient-route axis is now complete**: SDE log-probs (Z-1),
   SVGD density transport (RLDT), single-step critic-free
   (π-StepNFT), preference/implicit reward (FlowPRO) — and now
   **likelihood-free surrogate via CFM-loss difference**. No
   likelihoods, no SDE conversion, no BPTT; the training loss the
   policy already computes is the whole interface.
2. **The ablation's lesson: the gradient route carries the method,
   the critic barely matters.** Removing the ratio proxy costs 46
   points and removing clipping costs 33, while dropping the
   Q-ensemble to a single critic costs 7. For a roster deciding
   critic-vs-critic-free, that's a datum *against* paying for critic
   elaboration before the gradient route is right.
3. **Third independent vote for frozen-trunk RL** (decoder frozen,
   actor-only) — agreeing with Z-1's default and SA-VLA's
   protective-machinery lesson. And FPO's own degraded variants
   (no-clip 45.1%, no-ratio 32.4% — both plausibly below SFT level)
   are consistent with SA-VLA's negative sign: naive sparse-reward
   RL on a flow VLA is destructive without protective structure.
4. **The number to bank**: ALOHA ~40% → 65%+ own-baseline, sparse
   reward, sim. The LIBERO 87.2 headline is cross-base-model
   (baselines sit on OpenVLA/Octo/DP) and shouldn't be quoted as an
   RL-method comparison.

## What doesn't transfer

- **Sim-only, big-interaction-budget regime.** ~1.6M training steps
  on ALOHA implies an env budget a real rig can't pay; env count and
  compute are unreported, so the roster's open cost axis gets no new
  data.
- **Zero OOD/retention measurement** — nothing on π-StepNFT's
  IND-vs-OOD trade or FlowPRO's unmeasured retention, which is the
  axis the rig regime actually cares about.
- **Reproducibility is thin**: symbolic Algorithm 1, no
  hyperparameter values, no reward definition — even in the ICRA
  camera-ready.

## Which idea/arm it fed

[#16 rig-transfer benchmark](../ideas/16-rig-transfer-benchmark.md)
— RL-pole entry 6: the likelihood-surrogate route, critic-optional
(measured), frozen-trunk (again), sim-only with the cost axis
unreported. The pole's shape after six entries: gradient route and
protective structure decide the sign; critics are seasoning; nobody
has yet measured retention on a real rig. No new arm; the pole stays
sim-first and parked on #16's entry conditions.
