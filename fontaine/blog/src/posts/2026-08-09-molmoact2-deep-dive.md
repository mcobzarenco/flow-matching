# MolmoAct2, in depth: AI2 built a VLA on our trunk's family — here is everything in it

*2026-08-09, owner-requested deep dive (steering 20:49Z: "there's
already a molmo2 VLA — write a super in-depth piece on it").
Sources: the paper
([2605.02881](https://arxiv.org/abs/2605.02881), v2 2026-05-08,
51 pp), the [github.com/allenai/molmoact2](https://github.com/allenai/molmoact2)
repo (Apache-2.0, training + deployment code), the HF model/dataset
cards, and the [AI2 announcement](https://allenai.org/blog/molmoact2).
Four parallel research tracks, every claim below traced to one of
them. Written to be readable start-to-finish; the "what this means
for us" section at the end is the part that feeds our program.*

**In plain words.** MolmoAct2 is the Allen Institute's second-
generation open robot model: a ~5B-parameter system that looks at
cameras, reads an instruction, and produces one second of robot
motion at a time. Version 1 (2025) was a language model that
"thought out loud" about the scene — predicting depth and sketching
a trajectory as tokens before acting, slowly. Version 2 keeps the
thinking machinery but moves the actual motion generation into a
separate flow-matching "action expert" that reads the language
model's internal attention state at every layer, which makes it
~37× faster to act and noticeably better everywhere. It is trained
on three new open datasets AI2 built for it — including 720+ hours
of two-armed robot data and a cleaned-up 184-hour pool distilled
from 1,222 community SO-100/101 uploads on the LeRobot hub — and
everything is released: weights, data, code, even the action
tokenizer. For us this is the closest thing to a published
production version of our own stack: the backbone is literally
Molmo2, the trunk we train on.

## 1. Why this paper is personal

Our whole program runs flow-matching action experts on a frozen
**Molmo2-4B** trunk, trained on SO-101 LeRobot data. MolmoAct2 is
AI2 (the Molmo2 authors) building the same class of system at
production polish: a Molmo2-derived backbone, a flow-matching
action expert grafted onto it, trained on (among other things) a
curated version of the same community SO-100/101 pool our
`community_curated_v0` came from — and shipping a checkpoint for
our exact arm. Almost every design decision they made is a measured
answer to a question somewhere in our ledger: how big the expert,
where it reads the trunk, whether the trunk stays frozen, whether
knowledge insulation matters, what the discrete-token pathway is
still for.

## 2. The system at a glance

Total ≈ **5.06B parameters** in four pieces:

| Piece | Size | What it is |
|---|---|---|
| Vision encoder | 380M | SigLIP2 ViT, 384×384, patch 14 |
| V/L connector | 57M | attention-pooling 2×2 patches→token (3×3 for video) |
| LLM | 4.0B | Molmo2's language trunk (Qwen3-4B base): 36 layers, d 2560, 32Q/8KV heads |
| Action expert | 621M | non-causal DiT, 36 blocks, d 768, flow matching |

Released (all Apache-2.0 code, ~22 GB checkpoints on HF):
`MolmoAct2` (post-trained), `MolmoAct2-Think` (depth reasoning),
`MolmoAct2-Pretrain` (discrete-only), `Molmo2-ER` (the backbone),
per-embodiment finetunes (`-DROID`, `-BimanualYAM`,
**`-SO100_101`**, `-LIBERO`), the `MolmoAct2-FAST-Tokenizer`
("OpenFAST" — π's FAST tokenizer, reimplemented with open weights
*and* open training data), and all three new datasets in LeRobot
v3. Since v0.6.0, **upstream LeRobot ships MolmoAct2 as a
first-class policy** (finetune/eval/rollout lifecycle), and HF's
own port re-trained the LIBERO checkpoint to **98.25% vs the
paper's 97.20%** — the closest thing to an independent replication
that exists so far.

### What changed since MolmoAct 1

Five axes, per the paper: (1) backbone Molmo-7B → **Molmo2-ER**
(smaller, stronger, embodied-specialized); (2) three new open
datasets (v1 trained on ~22 h of robot data; v2 on ~30× that);
(3) the closed FAST tokenizer dependency replaced by **OpenFAST**;
(4) v1's purely autoregressive depth→trace→action pipeline
replaced by the **flow-matching expert** — depth tokens survive
only in the optional Think variant, and visual traces demote from
mandatory pipeline stage to optional steering input; (5) adaptive
depth re-prediction instead of re-predicting the full depth grid
every step. Net deployment effect: ~6,700 ms per action call →
~180 ms (base) / ~790 ms (Think).

## 3. Architecture

### 3.1 Molmo2-ER: the backbone is our trunk, specialized

The backbone is not a new model — it is **Molmo2-4B finetuned into
an embodied-reasoning specialist** ("Molmo2-ER") via a two-stage
recipe AI2 calls **specialize-then-rehearse**:

- *Specialize*: 20k steps on a 3.3M-sample embodied-reasoning
  corpus (spatial QA, pointing, detection, video QA, ego-exo
  multi-view, abstract spatial puzzles) + 8% Tulu-3 text.
- *Rehearse*: 1.5k steps re-interleaving Molmo2's original
  mid-training data with the ER corpus (best mix: 50/50 of the
  non-text share) at long context (16,384) — the stage that stops
  the specialist from forgetting it is a VLM.

The payoff is measured twice. As a VLM: **63.8% average over 13
embodied-reasoning benchmarks** — above GPT-5 (57.9), Gemini 2.5
Pro (57.1), and Gemini Robotics ER 1.5 Thinking (61.3), and a
+17.0 jump over base Molmo2. As a policy substrate: swapping
Molmo2 → Molmo2-ER under an otherwise identical discrete VLA lifts
LIBERO-Long **77.6 → 83.6%** — the reasoning specialization
transfers to action prediction. That second number is the cleanest
published evidence we've seen that *what the trunk was finetuned on
matters to the policy built on it*, at fixed architecture and data.

### 3.2 The token interface

Robot episodes are rendered into the LLM's world as text-plus-
special-tokens: a `<setup>` string ("bimanual yam robotic arms…"),
a `<control>` string ("absolute joint pose" / "delta end-effector
pose"), the robot state as **256 discretized state tokens embedded
in the prompt**, and — for the discrete pathway — 2,048 action
tokens produced by their reimplementation of π's FAST tokenizer
(DCT + BPE, trained on 1M action sequences across 5 embodiments,
weights and training data released). Heterogeneous action
"dialects" (absolute joint vs delta-EE) are deliberately **not
unified** — the prompt disambiguates, and the model learns both.

Actions are 1-second chunks at the dataset's native control rate
(30 steps @30 Hz for YAM and SO-100/101, 15 @15 Hz DROID), padded
to a shared 32-D width covering single-arm and bimanual layouts.

### 3.3 The action expert: per-layer KV conditioning

The continuous pathway — the deployment default — is a **621M
non-causal DiT** doing conditional flow matching (linear path,
masked MSE, 10 Euler steps at inference from Gaussian noise). The
architectural bet is *where it reads the trunk*: the expert has
**36 blocks, one per VLM layer**, and each block cross-attends to
that same-depth VLM layer's keys/values (through learned linear
adapters), between its own self-attention and MLP. Not final
hidden states — the trunk's full attention state at every depth.

Their conditioning ablation (LIBERO): final-hidden-state 94.0 <
per-head per-layer KV 94.8 < **flattened per-layer KV 95.9**. The
gap between "read the top" and "read every layer" is ~2 points at
the benchmark ceiling — real but not dramatic; the deeper read is
the architecture's identity.

*(Parameter accounting: the paper's "621M" and the HF export's
measured 577,564,448 are the same expert under two counting
conventions. Instantiated, the module is 620,677,664 params; the
exports omit the 36 frozen/inactive per-block ``cross_attn.kv_proj``
adapters (42,522,624) and the identity-injected ``state_encoder``
(590,592), both re-created by the loader. Measured on the first-class
port — ``outputs/probe_molmoact2_param_count.py``, pinned in
``tests/test_molmoact2_action_expert.py``.)*

Training the expert: flow loss with **K=4 noise samples per chunk
in post-training, K=8 in finetuning** (their K ablation is
monotone: 94.15/95.05/95.15/95.90 for K=1/2/4/8), and — in
post-training only — **knowledge insulation** à la π0.5: the KV
conditioning is detached so flow gradients never touch the VLM.
At finetune time they *drop* insulation ("no consistent gain from
detaching") and let flow gradients flow into the trunk. More on
what that means for us in §7.

### 3.4 MolmoAct2-Think: adaptive depth reasoning

The v1 inheritance. Before acting, Think autoregressively predicts
a **10×10 grid of depth codes** (128-entry VQ-VAE codebook over
Depth-Anything-V2 monocular depth) for the primary camera — then
the action. V2's twist is *adaptivity*: a grid cell is re-predicted
only if its RGB patch changed (cosine < 0.996 vs previous frame);
unchanged cells replay from cache. A learned per-layer gate
(sigmoid, bias init −4) controls how much depth-token K/V the
expert actually consumes. Think buys **+0.9 LIBERO average
(97.2 → 98.1, +2.2 on Long)** at **4.4× the latency** — a thin
margin at benchmark ceiling, unreported on real robots, and the
adaptive-vs-full-reprediction speedup is never measured
head-to-head against v1. The honest read: depth reasoning is now
an optional, gated accessory, not the engine.

## 4. Training pipeline

Three stages after Molmo2-ER exists (all AdamW, cosine to 10%,
LRs: ViT/connector 5e-6, LLM 1e-5, expert 5e-5 — constant across
stages):

| Stage | What trains | Steps | Data | Compute |
|---|---|---|---|---|
| Pre-train | discrete tokens only, no expert | 200k | 90% robot / 10% multimodal | 64×H100, 90 h (~5.8k GPU-h) |
| Post-train | + expert, knowledge-insulated | 100k | robot, K=4 flow samples | 64×H100, 36 h (~2.3k GPU-h) |
| Fine-tune | everything, insulation off | 100k | per embodiment, K=8 | 32–64×H100, ~36 h |

Robot mixture within pre-train: YAM 30%, SO-100/101 30%, DROID
30%, 10% legacy (Bridge/RT-1/BC-Z/v1 data). The whole main line is
order 10k H100-hours — production scale, but not extreme.

The finetune-design ablation is one of the paper's most useful
tables (LIBERO): **full FT + discrete co-training, no insulation =
97.20** (their recipe) > insulation on 97.05 > no co-training
96.95 > LoRA 96.25 (−2.8 on Long) > **expert-only 93.05** — the
one clear failure mode. Keeping the discrete-token loss as a rider
during continuous finetuning helps; freezing everything but the
expert costs 4 points.

## 5. The data story

Three new open datasets, all LeRobot v3:

- **BimanualYAM**: 34.5k demos, **720+ hours**, 28+ tasks
  (folding, cable untangling, table bussing, grocery scanning…),
  collected in 2 months by Cortex AI on a <$6,000 dual-YAM rig —
  the largest open bimanual dataset by an order of magnitude.
- **MolmoAct2-SO100/101**: the community-hub distillation —
  **1,660 candidate LeRobot repos → 1,222 kept (438 rejected),
  38,059 episodes / 19.8M frames / 183.6 h** from 377 contributors.
  Four-stage filter: structural validity → drop eval-style repos →
  license eligibility → a **learned quality gate** (TOPReward mean
  over final episodes must beat a threshold calibrated on
  human-audited sets). All 6-D action/state, mostly 30 fps. A
  detail the cards reveal that the paper doesn't: the released HF
  "dataset" is a **3.46 MB annotations manifest**, not raw data —
  `repo_list.json` names every kept `user/repo` verbatim, plus
  per-repo `tasks_annotated.parquet` files keyed by episode index.
  The hub repos themselves are the storage.
- **MolmoAct2-DROID**: 74.6k successful episodes, idle-frame
  filtered, with the community's 3-instructions-per-episode
  annotations.

Plus **language re-annotation at scale**: Qwen3.5-27B watches 12
frames per episode and writes instructions (5–25 words); unique
instructions across the mix go **71k (22%) → 146k (46%)**;
SO-100/101 goes 707 unique → 16,205. This is the "re-annotated
instructions we could port to episodes we already have" opportunity
flagged in yesterday's dataset survey, quantified.

## 6. Results, honestly grouped

**Zero-shot (the DROID-finetuned model, out of the box):** real
Franka kitchen, unseen objects, random camera poses: **87.1% avg
vs π0.5-DROID's 45.2** (15 trials/task — wide CIs, but the gap is
huge). Real SO-100 zero-shot: **56.7 vs 45.3** for their own π0
finetune (partial credit; SmolVLA scores 2.3). Sim (MolmoSpaces):
37.7 vs π0.5's 34.5 — with a glaring weak spot: articulated "Open"
tasks at **9.5%, worst of all baselines**.

**Finetuned:** LIBERO **97.2** (Think 98.1) — above GR00T N1.7
(97.0) and π0.5 (96.9). RoboEval bimanual: 44.3 vs π0.5's 40.5,
with ~2× shorter joint paths (their trajectories are notably less
wasteful). Real bimanual YAM, 8 tasks × 50 trials: **50.6 vs
OpenVLA-OFT 35.5, π0.5 32.2** — evaluated, caveat loudly, by the
same org that collected the training data. OOD on YAM: 50.7 vs
OpenVLA-OFT's 39.9, weakest axis spatial (26.3%).

**Latency (H100, LIBERO):** continuous expert **55.8 Hz effective
with CUDA graphs** (23.0 without caching tricks); the discrete
pathway 14.2 Hz; Think 12.7 Hz. Blog numbers for deployment: ~180
ms per action call (~16 GB bf16), vs 6,700 ms for v1 — the 37×.
Important framing from the paper's own limitations: these are
amortized chunk-throughput numbers, **open-loop within each
1-second chunk**, no cross-chunk continuity loss (they admit the
seam discontinuities), no within-chunk reactivity.

**The ablation ledger** (all LIBERO): backbone ER-ization +6.0 on
Long; per-layer KV over hidden-state +1.9; K=8 over K=1 +1.75;
full-FT over expert-only +4.15; co-training rider +0.25;
insulation at finetune −0.15 (i.e. nothing); Think +0.9 at 4.4×
cost.

**Independent signal: still thin.** Three months post-release the
outside evidence is HF's LeRobot LIBERO replication (98.25%), the
AI2-commissioned Cortex AI eval (0.51 vs OpenVLA-OFT 0.36 — run by
the org that collected the training data), and ~80 community
finetunes on the hub. A deliberate search found **no independent
hobbyist "I ran it on my SO-100" reports** and no adversarial
third-party eval; the press coverage relays AI2's claims. The
numbers deserve the standard treat-as-directional caveat until
someone with matching hardware reproduces the real-robot tables.

## 7. What this means for our program

**The trunk question got a production answer — and a new lever.**
They did not use base Molmo2: they spent ~20k steps making it an
embodied-reasoning specialist first, and measured +6.0 LIBERO-Long
from that alone at fixed everything-else. Molmo2-ER is *released*.
For us this opens a concrete, cheap arm: swap our frozen Molmo2-4B
trunk for frozen Molmo2-ER under the identical F recipe and read
the panel delta. It is the same class of question our tiny-expert
capacity rung answers for width — but on the axis their ablation
says matters more (+6.0 vs their +1.9 conditioning gain). This
slots straight into #17 as the highest-priority trunk arm we've
ever had externally priced.

**The attachment-seam ledger gets its most relevant entry yet
(#4).** Their staging is exactly our debate, at scale: post-train
the expert **with knowledge insulation on a working trunk** (=our
F, philosophically), then at finetune **unfreeze everything** —
and their ablation says expert-only finetuning costs 4.15 points
while insulation-at-finetune is a wash. Two readings, both
honest: (a) this is the strongest joint-pole vote in the ledger —
a production system measured frozen-vs-joint at its final stage
and chose joint; (b) the setting differs from our seam screen —
their "expert-only" starts from a *jointly post-trained* system,
not a converged frozen-trunk expert like our F, and their
benchmark is at ceiling. It does not overturn the frozen decision
memo; it sharpens what the fjoint rung is testing and predicts
fjoint > F2 if their result transfers.

**Conditioning surface (#4/#17):** per-layer KV cross-attention
beats final-hidden-state by ~2 points at ceiling. Our stride-3
12-tap residual surface is a coarser cousin of "read every layer."
Their result says the deep-read direction is right and prices the
remaining headroom as small-but-real.

**Expert capacity (the tiny10k rung, live tonight):** their expert
is **621M for a 4B trunk** (15.5%); our F is 367.5M (9.2%) and
tiny is 86.8M (2.2%). A production system landed near our F's
ratio, not our tiny's — a weak prior for tonight's Δ_capacity
read, and a useful anchor either way.

**The SO-101 shortcut is real and bigger than the survey knew.**
`MolmoAct2-SO100_101` is a released 5B checkpoint *already trained
on our embodiment* (absolute joint-pose, chunk 30 @30 Hz, 6-D
state, ~12.1 GiB bf16 with CUDA graphs, ~180 ms/call) — and
zero-shot rollout on SO-100/101 is an *officially documented
LeRobot path* (`lerobot-rollout`, 2 cameras), with one gotcha the
docs are explicit about: the checkpoint bakes in pre-0.5.0 LeRobot
joint conventions, so without the documented `joint_signs`/
`joint_offsets` remap the arm moves the wrong way. Their SO-100
zero-shot score (56.7% partial credit) sets expectations:
plausible demo, not a working product. Finetuning economics from
the LeRobot docs: **action-expert-only 16.5 GiB** (fits the local
H100 with room to spare), LoRA-VLM ~20 GiB, full FT 48–60 GiB —
the 8-GPU recipes in their own repo are not the floor. Separately,
their 183.6 h curated SO-100/101 pool + 16,205 re-annotated
instructions are the corpus-delta opportunity from yesterday's
survey, now *mechanized*: `repo_list.json` names every kept repo,
so the intersection with `community_curated_v0` is a set
operation, not a re-crawl; the per-episode instruction parquets
join onto our copies directly (verify episode counts per repo
first — re-uploads shift indices); and membership in their list is
a free external quality signal on our own corpus (their TOPReward
gate is the quality filter our re-crawl was missing).

**The AR side (#19) gets an open tokenizer.** Their FAST
reimplementation (weights + 1M-sequence training mix, released)
removes a dependency that made π-style discrete baselines awkward
to reproduce. And their measured discrete-vs-continuous deployment
gap (14.2 vs 55.8 Hz at equal quality ceiling) is the production
version of our flow-vs-AR panel gap.

**What we should NOT over-read.** Real-robot Ns are small (15
trials; 5/cell for OOD); the YAM eval was run by the data vendor;
the SO-100 baselines are their own finetunes plus a near-zero
SmolVLA; articulated objects and spatial OOD are genuinely weak;
Think's value is thin and un-validated on hardware; single model
size, no scaling study, no data-mixture ablation of the 90/10
robot ratio.

## 8. Cheat sheet: running it here

- Zero-shot on the rig: `allenai/MolmoAct2-SO100_101` via upstream
  LeRobot (`lerobot-rollout`, `norm_tag="so100_so101_molmoact2"`,
  continuous mode, 10 Euler steps, CUDA graphs for ~2×), ~12.1 GiB
  bf16 — fits the local H100 trivially, and **remember the
  joint_signs/joint_offsets remap** or the arm mirrors.
- Finetune on `community_curated_v0`: register a mixture entry
  (repo_ids, camera keys, `setup_type`/`control_mode` free-text,
  horizon 30), then `train_lerobot.py` from the SO checkpoint;
  their repo documents 8-GPU recipes but the LeRobot numbers say
  action-expert-only (`--ft_vlm=false`, LR 5e-5, 16.5 GiB) is a
  single-GPU job.
- Their depth-annotation generator
  (`scripts/generate_depth_annotation.py`) runs on any LeRobot
  dataset — the Think pathway is retrofittable to our data if we
  ever want the depth-reasoning rider.

*Everything above is from the four research tracks (paper PDF, repo
+ configs, HF cards, AI2 blog + community sweep); numbers are the
paper's own tables except where marked as blog claims. Follow-up
arms (Molmo2-ER frozen-trunk swap, SO100_101 corpus diff, zero-shot
rig eval) are owner-decision items — none are queued yet.*
