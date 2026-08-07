# 6. Aux attribution arms — `confirmed` (aux HELPS actions; results 2026-08-06)

**ANSWERED 2026-08-06 04:2xZ
([results post](../posts/2026-08-06-box-batch-results.md)): the
pre-registered decision rule fired REAL — aux-off costs +0.462 panel
chunk MAE (CI [0.387, 0.537], 7.5× the 0.061 replicate threshold,
leave-one-repo-out coherent). The mainline "within noise" expectation
is falsified: aux supervision shapes the action representation.**
Arms: A-s0/s1/s2 7.7966/7.8052/7.7355, B 8.2989; σ_seed(chunk) 0.038
→ E4B adopt band = 0.15 (floor binds). Twist held up: B's first_mae
3.5009 BEATS aux-on (3.94–4.11) with cond-sensitivity 1.13 vs
1.86–2.00 and predictions 8% closer to state-copy — consistent with
the #11 state-shortcut mechanism; the state-reliance probe is the
falsification instrument (all four npzs now banked). Standing rule:
**aux stays ON in every future recipe; an aux-off arm needs a new
pre-reg citing this result.**

The still-owed paired aux-on vs aux-off arms (does aux supervision
shape the representation, separate from "does narrating help" — the
100k run answered only the latter). Pre-registered mainline
expectation: within probe noise (±0.3). Promoted to arm B of the
paired 40k run after the wrap census killed unwrap-at-load:
[pre-reg](../posts/2026-08-05-prereg-paired-auxoff-40k.md). Primary read:
paired per-frame panel chunk_mae A@40k vs B@40k. **Executing on the
4×H100 box since 17:12Z** (parallel arms + 2 control seed replicates
for the noise floor, with a pre-registered decision rule:
[box batch pre-reg](../posts/2026-08-05-prereg-box-batch-4xh100.md)).
2026-08-06 01:3xZ: all four arms trained (A-s0 formal probe 7.0882@40k,
B 7.702@40k; s1/s2 at their boundary), panel evals chaining; **results
instrument `fontaine/scripts/box_batch_results.py` landed + oracled
before the data** — frozen decision rule, mechanical headline-column
matching vs report JSONs, σ_seed → the E4B adopt band and rig slot 2;
anchors/degenerate/synthetic-inflation oracles all passed. When the
four npz+report pairs land: one command produces the results-post
numbers and both finalization amendments.

- **External replication + a new rung-(a) probe (deep read
  2026-08-07, [post](../posts/2026-08-07-pi05-deep-read.md)):** our
  +0.462 aux-off cost is the same result class as π0.5's Fig. 13 —
  "Implicit HL" (subtask data in training, no runtime decoding) is
  their second-best config, i.e. semantic co-training shapes the
  action representation. Their further increment we have NEVER
  tested: **explicit runtime hierarchy** — decode a subtask first,
  condition actions on it. We own the seam: the `[subgoal|…]`
  conditioning slot (heavily dropped out; planner-less default
  well-trained). **Probe (zero training, quiet-GPU window):** have
  the AR model generate its own subgoal per panel frame, feed it
  back through `[subgoal|…]`, score panel vs no-hint baseline
  5.8026. Validity check first — eyes on a table of self-generated
  subgoals before any scalar (the never-generated-subgoal scar).
  Owner anchor in favor: the 21:43Z steer notes aux subgoals
  generalize strikingly OOD.
- **Lit slice 2026-08-07 03:2xZ — external support + two design
  constraints for rung (a):**
  [Hi-VLA systematic study](https://arxiv.org/html/2606.10267v1)
  (2606.10267) benchmarks hierarchy design and finds explicit
  language subgoals beat flat VLA **largest on long horizon** (flat
  25.30% → naive hierarchy 40.56% → best 67.08%; short-horizon gap
  near zero) — so the probe's per-step decomposition should expect
  the gain concentrated in LATE-horizon chunk_mae, mirroring the #1
  banked-prediction pattern. Two carried constraints: (i) their
  planner/controller are separate models — SELF-generated subgoals
  (our probe) are untested there, so ours is a genuine increment,
  not a replication; (ii) subgoal refresh granularity mattered a lot
  (4–8 s best; model-predicted horizons WORST) — our panel probe
  conditions per-frame, sidestepping refresh policy, but any later
  rollout arm must pre-register the refresh rule. Their hardest-task
  failure mode ("VLMs tend to ignore image inputs as task becomes
  harder") is the #11 state-dominant-bias story from the hierarchy
  side.
- **Rung (a) PRE-REGISTERED 2026-08-07 ~03:5xZ
  ([pre-reg](../posts/2026-08-07-prereg-selfsubgoal-probe.md)):** four
  arms on AR-100k (banked planner-less 5.8026 / oracle-truth
  `[subgoal|…]` / self-generated fed back through the slot /
  narrated-subgoal-only), validity table gated go/no-go BEFORE any
  scalar, frozen Δ-reads + horizon decomposition, ≤ 8 GPU-h with the
  q4-subset fallback. Execution at the first quiet local-GPU
  window ≥ the draws10_t1 boundary + its frozen reads.
- **Instrument LANDED 2026-08-07 ~04:3xZ (oracle-gated, this
  commit):** `bijou.eval --subgoal-mode {oracle,self}` — oracle mode
  renders per-frame TRUE labels through the trained slot (label-less
  frames decode the baseline context); self mode is the two-pass
  loop sharing one model load (pass 1 planner-less
  `[generate|subgoal actions]` = the `_narrsubgoal` arm free, pass 2
  feeds the text back through `[subgoal|…]` on the fast path =
  `_selfsubgoal`). `--dump-subgoals` retains per-frame generations
  (identity triple → text); `--selfsubgoal-force-empty` is the live
  oracle-(i) no-hint-limit run (`_emptyhint`, never a self-arm
  read); report JSON records the mode. Stage-1 validity table:
  `fontaine/scripts/selfsubgoal_stage1.py` (60 stratified frames,
  generation-only — NO scalars before the gate). The four
  pre-registered oracles' CPU halves are pinned in
  `tests/test_selfsubgoal.py` (prompt-byte equality of the no-hint
  limit and label-less oracle frames; one shared rendering path;
  pass 2's request set excludes subgoal); the real-checkpoint halves
  run pre-launch per the pre-reg. No semantic deviation from the
  pre-reg → no amendment needed.
- **Lit slice 2026-08-07 ~04:0xZ — two escalation anchors (radar
  only, no design change to rung (a)):** (i)
  [CAC-VLA](https://arxiv.org/html/2607.04816v1) (2607.04816)
  conditions the action head on VLM-predicted latent actions with a
  LEARNED GATE modulating conditioning strength — and trains on
  ground-truth-encoded conditioning while inferring on
  self-predicted, exactly the truth-vs-self asymmetry our
  Δ_oracle/Δ_self split diagnoses; if rung (a) lands in the
  "Δ_oracle < 0 but Δ_self ≥ 0" cell (generation quality is the
  gap), a gated-strength variant is a named escalation candidate
  (needs its own pre-reg). (ii) π0.7 (via the
  [NVIDIA WAM post](https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/))
  escalates explicit-HL beyond text: HL policy emits subtask
  instructions, a BAGEL-based world model renders them as subgoal
  IMAGES, the action expert conditions on obs+subgoal-image —
  reported "necessary for some dataset-bias-breaking tasks where
  no-subgoal variants fail", and subgoal images reportedly speed
  training by making action prediction near-inverse-dynamics. Our
  text-slot probe is the cheap first rung of exactly this ladder.
