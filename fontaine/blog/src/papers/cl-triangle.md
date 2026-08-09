# The CL triangle: three papers "disagree" about VLA forgetting — until you read their tables

*Read 2026-08-09 (lit slice `lit-radar-0819`, priority hook: CL contradiction
triangle). Papers:
[2603.03818](https://arxiv.org/abs/2603.03818) — "Pretrained
Vision-Language-Action Models are Surprisingly Resistant to Forgetting in
Continual Learning" (Huihan Liu, Changyeon Kim, Bo Liu, Minghuan Liu, Yuke
Zhu; UT Austin RPL; submitted 2026-03-04, CC BY 4.0);
[2605.26820](https://arxiv.org/abs/2605.26820) — "Can Vision-Language-Action
Models Learn from Real-World Data Continually without Forgetting?" (Jiarun
Zhu et al.; Agentic Intelligence Lab; submitted 2026-05-26, v2 2026-07-28);
[2603.11653](https://arxiv.org/abs/2603.11653) — "Simple Recipe Works: VLAs
are Natural Continual Learners with Reinforcement Learning" (Jiaheng Hu, Jay
Shim, Chen Tang, Yoonchang Sung, Bo Liu, Peter Stone, Roberto Martin-Martin;
UT Austin RobIn; submitted 2026-03-12, CC BY 4.0; RLC 2026, best paper at the
ICRA26 RL4IL workshop).*

**The cluster in plain words.** When a robot policy learns task B after task
A, it tends to get worse at task A — "catastrophic forgetting." Three 2026
papers seem to disagree about whether big pretrained robot models
(vision-language-action models, VLAs) still have this problem. One says they
are "surprisingly resistant" to forgetting. One built a real-robot benchmark
and says naive sequential fine-tuning forgets catastrophically. One says a
simple recipe — small adapter weights (LoRA) plus on-policy reinforcement
learning — needs no anti-forgetting machinery at all. Read the tables instead
of the abstracts and the fight mostly evaporates: **all three find that plain
sequential supervised fine-tuning forgets badly, VLA or not** — the
"resistant" paper's own no-replay rows show it. And all three find the same
cheap fix works: mix a small amount of old data back in (experience replay —
as little as 2% of prior data covers it on a real robot). The only genuinely
replay-free regime is the third paper's, and it is bought by on-policy RL —
the model only trains on its own current behavior, which keeps updates close
to home — something an offline pipeline like ours cannot use. The
disagreement was rhetoric, not data.

## Paper 1 — 2603.03818: "Surprisingly Resistant" (sim, BC, replay in hand)

**What they ran.** LIBERO (sim), four suites × 10 sequential tasks, behavior
cloning throughout. Two pretrained VLAs — π0 (3B: SigLIP-So400m vision +
Gemma-2B LM + 300M flow-matching action expert) and GR00T N1.5 (3B: SigLIP +
Qwen3-1.7B + flow DiT) — against three from-scratch baselines
(BC-Transformer ~15M, BC-Diffusion ~26M, BC-ViT ~15M; ResNet-18 vision,
frozen BERT language). Default experience replay: M=1000 transitions per
task (~15–20% of a task's data), sampled 1:1 current-vs-all-past.

**Crucially, the VLA trunks are never fully fine-tuned.** π0 trains the
vision encoder fully but adapts LM and action expert with **LoRA**; GR00T
runs with vision and language **frozen**, full FT on the action head only.
The from-scratch baselines are full-FT-all-params. The headline comparison
is therefore pretrained-plus-constrained vs scratch-plus-unconstrained —
two axes moved at once.

**Numbers.** With the default buffer: π0 averages SR 76.8% at NBT −0.016
(negative = past tasks *improved*), GR00T SR 91.9% at NBT 0.027, vs
BC-Transformer SR 58.5% at NBT 0.245. At a 2% buffer the gap is the story:
VLA NBT ≈ 0.1–0.2 vs scratch ≈ 0.4–0.5 (figure-read, not tabulated).
**And their own Table 2, zero replay: π0 NBT 0.696 (LIBERO-Object) / 0.562
(LIBERO-10); GR00T 0.752 / 0.758.** That is catastrophic forgetting, in the
"resistant" paper, on pretrained VLAs. (The SR column in those sequential
rows — 0.91–0.96 — has to be a plasticity-side read; the paper never pins
the definition cleanly. The NBT column is the forgetting signal.)

Two findings worth keeping. **Pretraining ablation:** π0 initialized from
VL-only pretraining (no action pretraining) does as well as the full VLA
init (SR 89.9 vs 86.3, NBT 0.016 vs −0.032, at default buffer) — the CL
benefit comes from the *VLM* prior, not from robot-action pretraining.
**Rapid recovery:** after apparent forgetting, π0 recovers peak success on
an old task with <10% of the original training steps (recovery ratios
0.066–0.105); scratch baselines need 1.36–1.87×. Forgetting in pretrained
models is largely output-level and latently reversible.

Release: real. Project page (continual-vlas.github.io/forget-me-not) links
the `Continual-VLAs` GitHub org — `continual-GR00T` (41 MB),
`continual-openpi`, a LIBERO fork. All resolve.

## Paper 2 — 2605.26820: the real-robot benchmark (full FT, no safety net)

**What they ran.** The setting closest to what we will actually face. Full
fine-tuning of **π0.5 (~2.7B, all parameters trainable, nothing frozen, no
LoRA)** on two PiPER 6-DoF arms + four RealSense D435s. Ten tasks as two
5-task streams: single-arm (500 demos each) and bimanual (300 each); 4,000
trajectories, 35 GB, LeRobot format. 4,000 steps/task, lr 5e-5 cosine,
batch 128 on 8×H20. Scoring is a per-task stepwise rubric (partial credit
per checkpoint, penalties, normalized 0–100), not binary success.

**Numbers.** Naive sequential FT: single-arm average score collapses
**86.9 → 31.4 (BWT −81.0)** — task 1 falls to 20.0, task 2 to **0.0**;
bimanual 88.0 → 38.3 (BWT −68.6). That is a ~3B VLM-trunk policy losing a
task *completely* within 4×4,000 full-FT steps. Experience replay (episode
level, buffer ρ_B=0.2 of prior data, 20% of batches from replay): **AS 97.2
on both streams, BWT +1.5 / +1.9** — forgetting eliminated, and it beats
joint multi-task training (82.6 / 83.2) at matched compute. Their
sensitivity sweep is the most useful number in the cluster: **ρ_B=0.02 —
two percent of prior data — is "already highly effective."**

**Evidence-strength caveats, loud.** (i) ER at 97.2 *exceeds the single-task
baseline* (86.9/88.0) by ~10 points — some mix of the extra steps (5k vs 4k
per task), forward transfer, and a generous rubric; treat the +1.5 BWT as
the finding, not the 97.2. (ii) Evaluation trial counts per task are
**nowhere stated** — self-scored rubric, unknown N, so error bars are
unknowable. (iii) No LoRA / frozen-component arm at all (they name
parameter-efficient adapters as future work), so the benchmark shows full
FT forgets, not that nothing else would. They explicitly critique Papers 1
and 3: sim-benchmark results "may be confounded by overlap with pretraining
data," and standard CL protocols "violate causality" by precomputing
normalization statistics over the whole task stream before sequential
training. Fair hits; neither is demonstrated quantitatively.

Release: real. github.com/Agentic-Intelligence-Lab/ContinualVLA resolves
(code ~6 MB; the 35 GB dataset is claimed in-repo — we did not verify the
data download itself).

## Paper 3 — 2603.11653: LoRA + on-policy RL needs no CL machinery (sim, RL)

**What they ran.** OpenVLA-OFT (7B, autoregressive, action chunking) with
**LoRA rank 32 (~100M trainable)** trained by **GRPO** on sparse binary
reward, sequentially over 10-task sequences from LIBERO-spatial/object/long,
RoboCasa, ManiSkill (plus a 30-task extension). Compared against eight CL
methods (EWC, SLCA, ER, DER, DWE, RETAIN, …) and a joint multi-task oracle.

**Numbers.** Seq FT (LoRA+GRPO) forgetting is essentially zero: NBT 0.3±0.5%
(spatial), 1.0±0.7% (object), −2.4±1.0% (long — it *improves*). AVG 81.2 vs
oracle 85.8. Zero-shot on held-out tasks *beats the oracle* on 2 of 3 suites
(57.1 vs 51.2 on spatial). "Beats elaborate machinery" mostly means
*matches it at lower cost*: ER gets AVG 80.2 / NBT 0.6%, DWE gets NBT 0.0 —
the point is you paid nothing for parity.

**The ablation triangle is the real contribution.** Remove any leg and it
collapses (libero-spatial): swap GRPO for supervised FT on successful
rollouts (still LoRA) → **NBT 78.7%**, AVG 29.9. Keep GRPO but full-FT
instead of LoRA → **NBT 40.9%**, AVG 7.3. Shrink to a 12M from-scratch
policy → NBT 11.4%, AVG 13.1. Mechanism evidence: on-policy gradients are
weighted by the policy's own state-action distribution, so updates "move
gradually outward from the support" of the current policy — implicit
regularization SFT lacks; and LoRA keeps update geometry tame (effective
rank 27.5±5.7, nuclear norm 0.48) where full FT is wild (324.7±465.0,
nuclear norm 4.31). Fisher overlap: 0.02 on the 7B vs 0.16 on the 12M.

**Caveats.** Sim-only, needs a reward signal and rollouts; flow/diffusion
action heads are flagged as fragile ("require more careful constraints,
e.g. a lower LoRA rank") and π0-on-RoboCasa numbers are weak (Seq FT 29.5
vs oracle 31.4). Note also: **Bo Liu is an author of both this paper and
Paper 1** — the two "forgetting is manageable" corners of the triangle are
one UT Austin cluster; the real-robot dissent is independent.

Release: real. github.com/UT-Austin-RobIn/continual-vla-rl (240 MB, 65
stars).

## Adjudication: where forgetting actually bites

Line the conditions up and the three-way contradiction is almost entirely
abstract-writing:

| Regime | Forgetting | Evidence |
|---|---|---|
| Sequential **SFT/BC, no replay, full FT** | catastrophic | P2 real (BWT −81); P3 ablation (NBT 40.9 even *with* RL) |
| Sequential **SFT/BC, no replay, LoRA / part-frozen** | still catastrophic | P1's own Table 2 (NBT 0.56–0.76, π0 LoRA'd + GR00T trunk-frozen); P3 SFT ablation (NBT 78.7 with LoRA) |
| Sequential SFT/BC **+ small replay** (2–20% of prior data) | ~solved for pretrained VLAs | P1 (NBT ≈ 0 at 15–20%, ≈0.1–0.2 at 2%); P2 real (BWT +1.5 at 20%; 2% "already highly effective") |
| **LoRA + on-policy RL**, no replay | ~solved | P3 (NBT ≤ 1%) — sim-only, needs rewards+rollouts |

**What pretraining actually buys** (P1, the claim that survives): not
immunity — a much better replay exchange rate (2–4× less forgetting than
scratch models at a 2% buffer), fast recovery (<10% of original steps to
restore a "forgotten" task, vs >100% for scratch), and it is the *VLM*
pretraining that carries this, not action pretraining.

**Genuine residual conflicts.** (i) P2's confound charge — that sim
resistance partly reflects LIBERO-adjacent data in π0/GR00T pretraining —
is plausible, unproven, and untestable from the papers' tables; it
discounts the *magnitude* of P1's resistance, not the replay conclusion,
which P2 itself replicates on real hardware. (ii) P1's conclusion sentence
("pretraining fundamentally changes the dynamics") overreaches its own
Table 2; P2's framing ("can VLAs learn continually? severe forgetting")
buries that its own ER row answers "yes, trivially." Where the papers
actually overlap — sequential SFT, with and without replay — **all three
agree**. On the one real disagreement of substance (is anything replay-free
safe?), P3's evidence is strong but scoped: on-policy RL is the mechanism,
it is sim-only, and its own ablations prove the recipe does *not* license
replay-free SFT. For an offline BC programme, P2 is the binding precedent
and P1+P2 jointly price the fix.

**Cheapest sufficient mitigation, by regime:** offline BC (us): episode-level
replay of prior data at ρ≈0.02–0.2 of the old corpus, ~20% of batches —
real-robot verified at 3B full FT (P2). LoRA is a geometry-preserving rider
worth stacking (P3) but is proven *insufficient alone* under SFT. If you
have rewards and rollouts: LoRA-32 + GRPO, nothing else (P3, sim).

## What transfers to us

- **Our live unfreeze arms (#17, #4) are joint training, not task-sequential
  CL** — the 100k AdamC vision-unfreeze run trains on the fixed 229h mix, so
  none of these BWT numbers applies directly. What does apply: P2 shows a
  ~3B VLM trunk's prior competence can be erased in 4,000 full-FT steps at
  lr 5e-5, and P3 shows full FT makes large, uneven structural changes to
  pretrained weights (nuclear norm 4.31 vs LoRA's 0.48). Unfreezing the
  Molmo2 trunk risks *pretraining* forgetting (VLM generality we may need at
  rig time), even when panel MAE improves.
- **Vision-unfreeze is the published-normal component to unfreeze.** π0's
  default recipe — the well-behaved configuration in P1 — fully fine-tunes
  the vision encoder while constraining the LM with LoRA. Mild but real
  support for the pre-registered vision-unfreeze screen's design.
- **P1's VL-only ablation is good news for our stack:** the forgetting
  resistance and fast recovery come from VLM pretraining, no action
  pretraining needed. Molmo2-4B sits exactly in the covered class (2.7–7B
  VLM trunks).
- **Recovery cheapness bounds our downside.** If an unfreeze arm degrades
  something, P1 says restoring it costs <10% of the original steps of
  re-finetuning on the old mix. The F-then-joint rung's failure mode is
  cheap to undo.
- **Rig phase (#16) is literally P2's experiment** — real arm, new tasks,
  full FT of a pretrained flow-matching VLA — and P2 hands us the recipe
  and the failure numbers.

## What does NOT transfer

- **P3's replay-free recipe.** It is bought by on-policy sampling; we have
  no rollout loop, no reward, and our flow head is exactly the architecture
  P3 flags as needing extra constraint. Its SFT ablation (NBT 78.7 *with*
  LoRA) is the direct warning against porting the headline to our pipeline.
- **P1's absolute NBT levels** — sim, LIBERO, possible pretraining-data
  overlap (P2's critique). Keep the relative claims (replay exchange rate,
  recovery speed), discount the near-zero absolutes.
- **P2's 97.2 average score** — rubric-scored, unknown trial count, exceeds
  its own single-task baseline; use the BWT deltas, not the levels.

## Which ideas it feeds

- **Idea #17 (unfreeze recipes).** The live vision-unfreeze run stands —
  vision full FT inside an otherwise-constrained trunk is the field's
  default. Pre-register onto any *language/trunk* unfreeze arm the cheapest
  rider pair: LoRA on the LM (P3's geometry argument) **plus** a replay-like
  anchor — our banked Qwen-RobotManip 9:1 mix + λ0.1 LM aux already exceeds
  P2's sufficient dose (ρ=0.02–0.2 at 20% of batches).
- **Idea #4 (frozen-vs-joint; F-then-joint rung).** Rung survives, two
  add-ons: (1) run the joint phase's cheap drift instrument — per-layer
  weight-delta effective rank / norm vs the pretrained trunk (P3's
  statistic, ~free to log); (2) consider LoRA-joint as the first rung
  variant — P3 found full FT is the destructive axis, LoRA matched
  plasticity in their setting.
- **Idea #16 (rig fine-tuning).** Pre-register now, before the rig exists:
  rig adaptation carries an episode-level replay stream from the 229h
  corpus at ρ∈[0.02, 0.2] of prior data, ~20% of batches; naive rig-FT
  should be expected to wipe prior competence within a few thousand steps
  (P2: BWT −81 by 4×4k steps). P2 also found ER *beats* joint retraining at
  matched compute — replay-based adaptation, not full re-mix retraining, is
  the default rig plan. LoRA-based rig adaptation is the cheaper untested
  variant (P2 lists it as future work; P3 supports it if we ever get an
  on-rig reward signal).
