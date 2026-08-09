# Pre-registration + parameter sheet: molmo2 AdamC 100k from base 4B

*2026-08-09 13:1xZ. Status: **AWAITING OWNER APPROVAL — nothing
launches until explicit sign-off** (the owner's 12:37:56Z spec makes
the approval a hard gate). Implementation is landed and oracle-tested
(`401d6f7`, check.py 584 green); this post is the in-depth description
of every run parameter the spec asked for, plus the pre-registered
gates and reads.*

## 1. The owner spec, and how I read it

Owner (2026-08-09 12:37:56Z): a new molmo2 run from base 4B with
(1) 100k steps, (2) effective batch 32 (8 per rank), (3) the vision
encoder unfrozen from the beginning, `--{backbone,text}-vision-lr
2e-5`, (4) 1000 warmup steps, (5) AdamC per
[arXiv 2506.02285](https://arxiv.org/abs/2506.02285), implemented
efficiently and mindful of shared layers (tied lm_head in Gemma),
reading the shared conversation as part of implementing.

**One interpretation call** (please correct if wrong): our CLI has
`--backbone-text-lr` and `--backbone-vision-lr` (there is no
`--text-vision-lr`). I read the brace expansion as *both the text
stack and the vision tower train at 2e-5*:
`--backbone-text-lr 2e-5 --backbone-vision-lr 2e-5`. Text at 2e-5 is
also the 40k/60k lineage value, so the single genuinely new unfreeze
is the vision tower (+connector) from step 0. Note the vu5k
pre-registration had picked a more conservative 6e-6 for vision — but
that was a *warm-start* screen thawing a tower against an
already-action-trained trunk. This run is from base, where
tower-included finetuning at a uniform backbone LR with warmup is the
standard recipe; the spec's 2e-5 stands.

"From base 4B" = `--backbone allenai/Molmo2-4B` with **no**
`--init-from`: fresh decoder tables (`fast_embed`/`fast_head`
mean-initialized from the trunk's frozen tables, as in every phase-1
run), no action-trained checkpoint anywhere in the ancestry.

## 2. The optimizer: AdamC, and what landed

**The paper's claim.** With decoupled decay (AdamW), the steady-state
gradient-to-weight ratio of a normalized layer is
`‖g‖/‖x‖ = √(2λ/γt)`. A decaying LR schedule therefore *raises* the
equilibrium gradient norm as training ends — the paper's "gradients
rapidly increase near the end of training" — and the fix is to make
the decay coefficient track the schedule:

```
λ̂_t = λ · γt / γmax          (corrected, "normalized" = hidden layers)
x_{t+1} = (1 − γt²/γmax · λ) x_t − γt · m̂_t / (√v̂_t + ε)
```

The output layer keeps standard AdamW decay (Algorithm 1's exclusion);
1-D parameters stay undecayed as usual.

**The implementation** (per the shared conversation's recipe, which I
read): AdamC is *exactly* AdamW with a per-group, time-varying
`weight_decay`. `--optimizer adamc` partitions the parameters into
corrected / standard-decay / no-decay groups at construction, and one
O(#groups) Python loop writes `λ·γt/γmax` into the corrected groups'
`weight_decay` immediately before each `optimizer.step()` — the stock
**fused** AdamW kernel reads it fresh per call, so the update is
bit-exact AdamC with zero extra kernels. γmax per group is its
`initial_lr`: every branch of our `lr_lambda` (warmup, cosine, floor,
re-warmup) peaks at exactly 1.0, so the correction factor is the
schedule multiplier itself. During warmup λ̂_t < λ — the paper's
intended γt²/γmax behavior, not a bug. ZeRO-1 is covered: torch's
wrapper `step()` copies group attributes wrapper → local optimizer
before the sharded step, and a test now pins that contract against
torch upgrades.

**The shared/tied-layer care the spec demanded.** The group partition
for this run's composition (Molmo2 trunk + `ar_backbone` decoder):

| group | decay | contents |
|---|---|---|
| decoder (corrected) | λ·γt/γmax | `fast_embed` (untied input table), encoder prompt-side matrices (`state_proj`) |
| decoder head (standard) | λ·γt | **`fast_head`** — the fresh untied logit rows, the paper's excluded output layer |
| backbone_text (corrected) | λ·γt/γmax | every trunk block matrix |
| backbone_vision (corrected) | λ·γt/γmax | vision tower + connector matrices |
| no-decay groups | 0 | all 1-D params (RMSNorm scales incl. `ln_f`, biases) |

Molmo2's `wte` and `lm_head` are **untied and frozen by design** (they
never reach the optimizer), so no tied pair is trainable in this run.
The tied-head hazard is nonetheless real in-repo — the Gemma AR
decoder's `fast_embed` doubles as its logits head (`hidden @
fast_embedᵀ`), and Gemma's trunk lm_head is tied to its embedding — so
the implementation (a) routes the Gemma decoder's tied table to the
standard-decay group as one parameter object in one group, (b) refuses
unaudited decoder types loudly, and (c) now asserts in **both**
optimizer modes that the groups disjointly and exactly cover the
trainable set, so any future tied parameter appearing in two groups
dies at construction instead of being decayed twice.

**Oracles** (tests/test_adamc.py, 10 tests, in check.py): partition on
the tiny molmo2 composition with text+vision unfrozen (this run's
exact shape); λ̂ trajectory exact through warmup/peak/decay; **bitwise
equivalence with AdamW at γt = γmax**; bitwise equivalence with a
hand-set corrected AdamW under a decaying schedule; ZeRO-1
wrapper→local sync; tied-overlap and missing-head SystemExits. The
adamw path's group construction is pinned byte-identical to the
historical one.

## 3. Full parameter sheet

Everything the launcher will pass, with provenance. **Lineage** = the
40k/60k phase-1 recipe verbatim; **spec** = the owner's 12:37Z
message; **new** = proposed here, needs sign-off.

**Schedule and optimizer**

| flag | value | provenance / rationale |
|---|---|---|
| `--steps` | 100000 | spec |
| `--warmup-steps` | 1000 | spec (also lineage) |
| `--batch-size` | 8 (×4 ranks = eff. 32) | spec. Lineage ran 12/rank (eff. 48) — declared batch delta vs. all banked runs |
| `--optimizer` | **adamc** | spec item 5, implementation `401d6f7` |
| `--weight-decay` | **0.1 — DECISION POINT** | The paper's LLM experiments run at standard decay strength (0.1-class); our lineage default is 1e-5, at which decay ~does nothing and an AdamC-vs-AdamW comparison would measure noise. **Recommend 0.1** so the run actually evaluates AdamC; the conservative alternative is keeping 1e-5 (minimal-delta vs lineage, but then "evaluate AdamC" is mostly vacuous). Owner picks. |
| `--decoder-lr` | 1e-4 | lineage (spec silent; the decoder tables are fresh params and have always trained at 1e-4 peak) |
| `--backbone-text-lr` | 2e-5 | spec (per §1 interpretation; also lineage) |
| `--backbone-vision-lr` | 2e-5 | spec — vision unfrozen from step 0 |
| betas | (0.9, 0.95) | lineage, hardcoded; unchanged under adamc |
| `--grad-clip` | 100 | lineage |
| LR shape | cosine to 10% floor after warmup, all groups share the multiplier | lineage (`lr_lambda`); AdamC's λ̂ follows it by construction, flooring at 0.1λ |

**Data and objective (lineage verbatim — no spec deltas)**

| flag | value |
|---|---|
| `--train-data` | `/home/ubuntu/datasets/mcobzarenco/community_curated_v0` |
| `--fps 30 --camera-counts 1 2` | lineage |
| `--holdout-episodes 0.1 --split-seed 0` | lineage (same holdout as every banked panel) |
| `--decoder ar_backbone` + `--backbone allenai/Molmo2-4B` + `--max-crops 1` | lineage |
| `--fast-tokenizer` | `mcobzarenco/bijou-checkpoints/fast_tokenizer_v2` |
| aux/conditioning | `--aux-fields subgoal holding progress event visible --aux-dropout 0.0 --field-dropout 0.1 --condition-fields subgoal outcome smoothness --condition-dropout 0.1 --subgoal-dropout 0.5 --instruction-augment 0.5 --camera-kind-dropout 0.1` |
| E1 dataset gate | banner must read 878 datasets / 38,571 episodes / 18,636,749 frames / dims 6/6 — any deviation aborts before step 1 |

**Execution**

| flag | value | rationale |
|---|---|---|
| topology | 4×H100 box, torchrun DDP | lineage |
| `--zero1 --backward-chunks 4 --chunk-grad-allreduce` | B8c4 | per-chunk microbatch 2 preserved from B12c6; chunks must divide batch. Semantics exact (oracle-tested lineage machinery) |
| async saves | default on | validated live at the attach_F launch |
| `--num-workers 20 --prefetch-factor 4` | lineage |
| `--seed` | **1 — proposed** | from-base run, no resume, so the fresh-seed rule doesn't bind; but eff-32 restructures batching anyway, so no comparability is bought by reusing seed 0 — a fresh seed keeps the stream provenance unambiguous |
| `--eval-samples 256 --eval-every 500 --log-every 20` | lineage probe cadence |
| `--save-every` | **5000 — proposed** (lineage 2500) | 100k at 2500 = 40 checkpoints × ~37 GB ≈ 1.5 TB on the box; 5000 halves it and still gives 20 resume/analysis points. Owner call if 2500 preferred |
| run name / dirs | `fontaine_molmo2_adamc_100k_ddp4`, `outputs/train/…`, wandb project `fontaine` | convention |

## 4. Cost, memory, and the pre-launch smoke

No banked run has this exact shape (vision-unfrozen, B8). Anchors:
frozen-vision B12c6 measured **2.251 s/step** (perf-pass-1 box ladder,
A-arm); the vu5k pre-reg projected the tower backward at roughly
+10–20%; batch 8/12 scales activations ≈ ×⅔. Estimate: **~1.7–2.1
s/step ⇒ 47–58 h wall ⇒ ~190–235 GPU-h** train, + ~1.3 GPU-h endpoint
panel. Memory: B12 frozen-vision peaked 66.6 GiB and the vu5k B12
projection straddled 71; at B8 the projection clears comfortably, and
ZeRO-1 spreads the tower's extra moments (~430M params) across ranks.

**Launch gate (after approval, before the run): a 150-step smoke** on
the box with the full flag set — pins measured s/step and vram peak
(≤ 71 GiB gate), confirms the AdamC banner (partition param counts,
λ, γmax), the trainable-param banner counting the tower, and the first
async-save lines. ~0.2 GPU-h. The in-launcher rate gate then re-checks
the projection at the first jsonl window like every lineage launch.

## 5. Kill lines and monitoring (babysit entry at launch)

Judged at save boundaries, K1-style, frozen at launch:

- NaN/inf loss → immediate kill.
- Probe (eval action MAE) not below its own @2500 value by step 10k →
  kill (the "never learned" line).
- Probe > 25 sustained ×3 consecutive evals after step 5k → kill.
- vram_alloc_peak > 71 GiB → kill (memory creep).
- **Record-only AdamC watch:** grad-norm trajectory over the second
  half of training — the paper's signature claim is that the AdamW
  end-of-training grad-norm ramp disappears. Logged every 20 steps
  already; charted at the readout. Not a kill line (we have no matched
  AdamW twin at this batch to gate against).

## 6. Explicit decision points for the owner

1. **`--weight-decay`: 0.1 (recommended) or lineage 1e-5?** (§3 — at
   1e-5 the optimizer change is nearly a no-op.)
2. **Text LR = 2e-5** — confirm the §1 reading of
   `--{backbone,text}-vision-lr`.
3. **seed 1** (proposed) — or keep 0.
4. **save-every 5000** (proposed) — or lineage 2500 (~1.5 TB).
5. Everything else as tabled.

On sign-off (plus any overrides): launcher lands with the smoke gate +
babysit entry, smoke runs, then launch. Box GPUs are free and waiting.

## 7. What gets read at the end (frozen now)

- **Primary, record-only:** endpoint panel eval on the k4l2 holdout
  plan (greedy, `--report` HTML per the standing rule +
  `--dump-predictions` npz), quoted next to the banked 40k/60k/100k
  AR rows. **Directional only** — batch 32-vs-48 and 100k-vs-40k are
  confounded with the optimizer; no "AdamC beats AdamW" claim will be
  made from this run alone. A matched AdamW twin would need its own
  pre-reg.
- **AdamC-specific:** grad-norm + probe curves vs the 40k lineage
  curves at matched steps (charted, dark-mode, in the results post);
  the paper-shaped question is whether a late-training grad-norm rise
  appears here at all. (Whether our AdamW lineage runs even show the
  paper's ramp is itself unmeasured — the 40k jsonl grad-norm series
  gets charted as part of the same readout, no assumption baked in.)
- Kill-bar facts and cost actuals, as always.

*Amendments to this sheet (owner overrides on the decision points) will
be appended here before launch, not silently edited in.*

---

## Amendment 1 (2026-08-09 14:0xZ, pre-launch): owner decisions folded

Owner reply 13:19Z resolved all four decision points and changed the
launch procedure:

1. **`--weight-decay 0.01`** — owner pushed back on 0.1 as high and
   asked for the VLA-standard value; no AdamW-vs-AdamC comparison is
   wanted ("AdamC is the correct thing to do", not an ablation arm).
   Grounding: 0.1 is the from-scratch LLM-pretrain setting (the
   paper's own regime); VLA finetunes of pretrained VLM trunks run
   0–0.01 — openpi (π0/π0.5) uses ≈0 (torch config default 1e-10),
   OpenVLA finetune recipes use 0.01. λ=0.01 is the inside-practice
   value that still gives AdamC's correction something to shape.
2. **Text LR interpretation confirmed**: text 2e-5 AND vision 2e-5.
3. **seed 1** approved.
4. **save-every 5000** approved.
5. **NO pre-launch smoke** (owner: "go ahead with the real run, no
   smoke tests, if it dies, it dies and we restart it"). §4's
   150-step smoke gate is dropped. Consequences, declared:
   - The §5 vram kill bar (71 GiB, smoke-calibrated) is replaced by a
     **77 GiB near-OOM watch** — the memory projection (~70–73 GiB at
     microbatch 2, vs ~79 usable) was never smoke-confirmed, so 71
     would risk killing a healthy run on a 2-GiB estimate error.
     Actual peak gets recorded at the first monitoring poll.
   - **OOM policy**: relaunch at `BACKWARD_CHUNKS=8` (microbatch 1),
     same effective batch 32; second OOM escalates to owner.
   - Measured s/step and the wall-clock projection land at the first
     poll instead of pre-launch.

Launcher: `fontaine/scripts/box/launch_box_fontaine_molmo2_adamc_100k_ddp4.sh`
(diff vs the 40k lineage launcher = exactly the declared deltas).
Launched immediately after this amendment was pushed.

## Amendment 2 (2026-08-09 13:3xZ, pre-step-1): λ = 1e-5 (owner override)

Owner 13:24:10Z, overriding amendment 1's λ=0.01: weight decay stays
at **the 40k/60k lineage value, 1e-5** (the CLI default — confirmed
against both launchers, which never passed `--weight-decay`).
Amendment 1's launch was stopped at 13:29Z **before step 1** (the run
was still in model load / wandb init; no optimizer step was taken, no
checkpoint written; the save dir was removed) and relaunched with
λ=1e-5. Note for the readout: at λ=1e-5 the decay term is tiny, so
this run exercises AdamC's *mechanism* (schedule-tracking decay) at
lineage-equivalent regularization strength — grad-norm trajectory
remains the record-only watch, with no AdamW-vs-AdamC claim planned
either way (owner: no ablation wanted).
