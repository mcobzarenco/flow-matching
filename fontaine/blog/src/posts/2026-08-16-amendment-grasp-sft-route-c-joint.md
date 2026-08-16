# Registered amendment — route C: one `--objective joint` run merges the A+B pre-regs

*2026-08-16, ~01:0xZ. Owner steering 00:18Z: GPU released, **route C
picked** ("assess whether RAM suffices for joint; if not, optimize the
AR objective's memory to make it fit — route C either way"). This
amendment merges the
[flow-retrain pre-reg (A)](2026-08-15-prereg-grasp-sft-retrain-corrected-table.md)
and the
[token-SFT pre-reg (B)](2026-08-15-prereg-grasp-sft-token-sft-arm.md)
into the single joint run their §4/§4 route notes anticipated, records
the measured RAM feasibility work, and is posted **before launch**.
Launch is tonight per the owner's message; **init and λ are spelled
below for morning veto.***

**Plain words.** The owner picked the option that trains both of the
model's "mouths" at once: the *flow* head (continuous actions — the
one that already went 9 → 28/100 under the old, corrupt normalization
table) and the *token* head (discrete symbols — the one reinforcement
learning would push on, still at factory weights). One training run,
one data pass, both heads. The catch was memory: training the trunk
at full fp32 precision needs the optimizer's bookkeeping (34 GiB) to
sit next to the model, its gradients and its activations, and the
whole ensemble didn't fit on our 80 GiB card — we measured it
crashing. The fix that made it fit is exact, not approximate: the
bookkeeping now lives in the machine's main RAM (which has plenty of
room) and the update math runs on the CPU, provably bit-identical to
what the GPU would have computed. With that lever the run fits with
a sixth of the card to spare, and it launched tonight.

## §1 What merges, and what each parent contributes

| | from A (flow retrain) | from B (token-SFT) |
|---|---|---|
| init | **from-base, corrected table** (`molmoact2_base_corrected_stats_v0_vla`) | same artifact (its §1 requires the corrected table for the token stream too) |
| trunk | — (A froze it) | **`--backbone-text-lr 1e-5`** (B's measured smoke value; 2e-5 the registered alternative, judged at first poll) |
| flow decoder | **`--flow-decoder-init inherit`, `--decoder-lr 5e-5`** | — (B built no expert) |
| steps/batch | 2000 × gb64 (the stage-C endpoint both parents froze) | same |
| eval | flow head: euler-10, unseen 0–99 + train 1000–1099 | token head: grammar-masked greedy, unseen 0–99 |

**Insulation: ON (`--insulate-flow`).** The KI seam makes the merge
*exact*: flow gradients into every trunk parameter are literally zero
(A's frozen-trunk semantics for the expert), and the trunk learns from
CE alone (B's semantics). The two loss terms reach **disjoint
parameter sets**, so λ is an LR-relative knob, not a tuned constant —
**λ = 1.0**, the KI no-tuning default. Non-insulated joint would
confound both parents' recipes; it is NOT what launches.

**The A-confound carries over verbatim** (A §1, stated honestly): the
corrupt→corrected table fix and their-stack→bijou-stack move together.
The flow-head read inherits it unchanged; the token head has no
prior-stack row to confound (its released-weights floor is the
comparison).

## §2 RAM feasibility — measured, not estimated

The owner's expected peak (full-vocab CE logits over the suffix) is
**not** the binding term: at micro-batch 2 the `[B, T, 154624]`
logits cost well under 1 GiB. The measured constraint is **static
fp32 residency** — trunk masters ~20.3 GiB + gradients 16.9 GiB
(4.21B trainable: 3.63B text blocks + 0.58B flow expert) + **AdamW
moments 33.7 GiB** ≈ 71 GiB before a single activation.

Smokes on the real batch (gb64, real dataset, this checkpoint):

| config | result |
|---|---|
| chunks 8 (micro 8), act-ckpt | **OOM step 1**, 76.0 GiB |
| chunks 32 (micro 2), act-ckpt | step 1 green (67.8 GiB), **OOM step 2 backward** at 77.2 GiB — moments resident |
| chunks 8 + **`--offload-optim`** | **6/6 steps green, peak 58.0 GiB**, eval+save boundaries exercised, checkpoint validates |
| chunks 4 (micro 16) + offload | **peak 66.5 GiB, 11.2–11.4 s/step** ← launch config |

The make-it-fit change is `bijou.train --offload-optim`
(`8bb5b70`): AdamW moments live in host RAM, the step runs torch's
CPU reference kernels on pinned fp32 mirrors. AdamW is elementwise,
so this is **exact** — oracle-pinned bitwise to the CPU reference
(under a live LR schedule and AdamC-style group writes), fused-CUDA
closeness pinned, None-grad skip parity, resume round-trip bitwise
(`tests/test_offload_optim.py`, 5 tests; check.py 908 green). CE
health in-smoke: `loss_aux` 4.33 → 2.87 by step 6.

## §3 Frozen launch command

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python -m bijou.train \
  --train-data ~/datasets/fontaine/grasp_sft_demos_v0 \
  --init-from  ~/checkpoints/converted/molmoact2_base_corrected_stats_v0_vla \
  --objective  joint --joint-ce-weight 1.0 --insulate-flow \
  --flow-decoder-init inherit \
  --decoder-lr 5e-5 --backbone-text-lr 1e-5 \
  --steps 2000 --batch-size 64 --backward-chunks 4 \
  --activation-checkpointing --offload-optim \
  --save-every 500 \
  --save-dir ~/checkpoints/finetune/fontaine_grasp_sft_joint_corrected
```

Seed: bijou default (fresh run; no cross-stack seed comparability
exists — house policy n/a, per A §3). Family checkpoint-inferred →
`molmoact2_joint`; the collate/codec seam rides the checkpoint (no
`--fast-tokenizer`).

**Morning-veto items** (the run launches tonight per the 00:18Z
steering; any of these can be re-steered and the run relaunched
cheaply): **(1) init from-base** vs continue-from-2k
(`molmoact2_grasp_sft_stagec_ar_step2000_corrected_v1` — swaps one
path); **(2) λ = 1.0**; **(3) insulation ON**; **(4)
backbone-text-lr 1e-5** vs 2e-5.

## §4 Eval protocol + anchors (frozen; both parents' reads, one checkpoint)

At step 2000, `convert_legacy` n/a (new format native); probes on the
single joint endpoint:

- **Flow head** (A §4 verbatim): `rollout_sim` euler-10, 30 s
  episodes, bf16 flow decoder — unseen 0–99, then train band
  1000–1099. Anchors: released base **9**/100 (primary), stage-C
  step2000 corrupt-table **28**/100 (the floor to beat), ftrig4k/W0
  ~1/2 (context).
- **Token head** (B §3 verbatim): grammar-masked greedy decode,
  unseen 0–99. Primary bar: **≥ 20/100 → R2's competent-base premise
  holds**. Optional base-token anchor leg (+~1.3 GPU-h): default run
  it, per B §3.

Decision surface: A §5 for the flow read (>28 table-fix positive /
~28 data-limited / <25 seam investigation before pricing), B §3 for
the token read (≥20 R2 activates / 5–19 owner decision / <5 token-SFT
didn't transfer, the discrepancy is the finding). The two reads are
independent; neither gates the other's bank.

## §5 Budget

Train: measured 11.4 s/step × 2000 ≈ **~6.5 GPU-h** (ETA ~07:3xZ
from a ~01:1xZ launch). Evals: flow probe ~2.5 + token probe ~1.3 +
optional base-token anchor ~1.3 ≈ ~5. **Total ~11.5–12, gate ≤ 13**
(the grasp-SFT chain convention). Detached unit + babysit entry at
launch, first-poll util/rate/VRAM/host-RAM checks per standing rules.

## §6 Supersessions

This amendment **supersedes A §3 and B §2** (their standalone run
commands park; their eval protocols, anchors and decision surfaces
are inherited above unchanged). A §8–§10 and B §6 (format/flag
re-verifies) carry over — the launch command above is already spelled
in the current CLI surface. The A+B pre-reg pages stay up as the
protocol records; this page is the run's registration.
