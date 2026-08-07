# 16. Few-shot rig-transfer benchmark — `parked` for execution (owner 2026-08-05 21:43Z), instruments banked; **the north star** (owner 2026-08-05 17:20–17:23Z)

*Tag: `rig-benchmark` · idea #16 · [index](../ideas.md)*

- **OWNER STEER 2026-08-05 21:43Z — execution PARKED, priorities
  reweighted:** the rig datasets are small/noisy and a 12-ep fixed
  holdout is high-variance ("really depends on which episodes you
  choose"); owner will collect a better rig dataset later. **Short
  term: improve MAE on the comm holdout and/or attribute where the
  limit comes from** (bigger trunk? bigger image embeddings?
  video-trained trunk? is the flow expert needed at all vs pure AR?).
  Empirical anchor from the owner: lower comm-holdout MAE has always
  translated to good rig fine-tunes; current failure mode is gripper
  *placement* accuracy (grounding), motion is fine; aux tasks
  generalize strikingly (4k ft on AR-100k produced sensible subgoals
  for a fully-OOD instruction — "coiled USB-C cable", "glass pot").
  The instruments below stay banked: they are corpus-agnostic and
  re-run on the future dataset in minutes.

- **External anchor for the premise (deep read 2026-08-07,
  [post](../posts/2026-08-07-pi05-deep-read.md)):** π0.5's Fig. 8 is
  the diversity-buys-transfer bet measured at production scale —
  held-out-home performance scales monotonically with training
  locations (3→104), and at 104 locations MATCHES a control trained
  on the test homes; 97.6% of their phase-1 examples are not the
  target embodiment. Evidence, not proof (their scale: ~400 h,
  ~100 homes) — but the north-star premise now has a citable
  production-scale precedent.

- **Pre-reg draft posted 2026-08-05 ~21:2xZ**
  ([post](../posts/2026-08-05-prereg-rig-fewshot-benchmark.md)): design
  frozen — 12-ep holdout (SeedSequence(16)) + nested N ∈ {10,25,45}
  materialized derived corpora (leakage-checked, the #18.8 consumer);
  owner `run_ft_rig.sh` protocol constants; best-checkpoint-at-200
  selection; co-primary chunk_mae + first-4 pooled MAE; 3·σ_ft
  decision rule (σ_ft from N25 seed replicates); **eligibility gate:
  init pretrain corpus must certifiably exclude the rig repos —
  flow-80k is contaminated (rig data in its pretrain mix), rcond/box
  arms qualify.** Two slots (init selection rule + E5 noise scale)
  fill by finalization amendment after the box reads; execution at
  the first quiet GPU boundary after.
- **Instruments LANDED 2026-08-05 ~21:5xZ (Amendment 1 on the
  pre-reg):** plan frozen (`plans/rig_fewshot_v0_k4l2.json`, 12 eps /
  48 core + 24 labeled; holdout = native split 0.212/seed 16 → v2
  {1,2,3,6,11,15,20,24,25,30,41} + clean {2} — mechanism amendment:
  the draft's bespoke SeedSequence draw could not feed the leakage
  checker); subsets materialized + verified
  (`~/datasets/rig_fewshot_v0/`, n10 6,223 / n25 15,881 / n45 29,107
  frames, videos hardlinked bit-identical, judgments remapped, stats
  recomputed w/ oracle worst |Δ| 1.2e-4); **leakage certs PASSED ×3**
  (#18.8 provenance path, doctored-provenance negative control fails
  loud); loader smoke bit-exact incl. shifted mid-file video decode;
  **wrap census CLEAN on both rig repos** (hygiene gate 1). Remaining
  before launch: launcher gen + finalization amendment (slots 1–2)
  after the box reads.

- **Goal statement (owner):** "build a VLA for my rig… prove transfer
  so you can fine-tune a task on a new SO101 arm with tens of
  examples." Community-panel MAE is the proxy; **the
  sample-efficiency curve is the product metric.**
- **Design sketch (pre-reg to write after the box batch lands):**
  fine-tune the best lineage on N ∈ {10, 25, 50} episodes of a
  held-out rig task; measure panel-style MAE on that task's holdout
  (and eventually rollout success) vs N. Protocol precedent: the
  owner's ft-rig lineage (4–5k-step fine-tunes, `run_ft_rig*.sh` on
  the second box, both AR and flow variants).
- **Dependencies:** tonight's aux-off answer + seed-noise floor pick
  the trunk and set the minimum detectable effect for paired ft
  comparisons; sign/calibration hygiene (ideas #13, #14) bites
  hardest on a new arm — keep them warm.
- **Falsification:** the curve itself — if MAE at N=50 is no better
  than zero-shot, transfer is not proven and the pretraining recipe
  (not the ft protocol) is the suspect.
- Reweights the whole list: rig-transfer relevance now outranks
  community-panel micro-optimization at equal cost.
- **Metric note (2026-08-05 paired analysis):** the pre-reg must fix
  the deployment replan interval k and quote first-k pooled MAE next
  to chunk_mae — the flow-vs-AR ranking *flips* at k≤3 vs k≥5
  ([post](../posts/2026-08-05-flow-vs-ar-paired.md)); chunk_mae alone
  is the most AR-favorable point on that axis.
- **Literature (2026-08-05 slice): the ft-protocol arm should
  include LoRA-r32 + full vision-encoder ft** (arXiv:2607.10172,
  π0 on UR5e precision assembly): LoRA saturates at r=32 with no
  significant FFT advantage; **freezing or LoRA-restricting the
  vision encoder significantly degrades** (independent external
  support for our grounding-bottleneck reads, idea #11); static
  peak VRAM 36.2→10.8 GiB — on 1×H100 that headroom converts
  directly to batch for the few-shot fine-tunes. **Papers-page
  re-read 2026-08-07 ([page](../papers/data-and-trunks.md)):
  CONFIRMED on all counts, now numeric** — r=32 at 0.74 vs FFT 0.76
  (p=1.000); SigLIP frozen 0.14 / SigLIP-LoRA 0.43 vs 0.74 fully
  trainable; metric is ATP (sub-goal progress), not success rates;
  plateau beyond r=32 may partly be the α=r scaling rule.
