# Pre-registration: molmo2 stage-2 attachment screen — frozen trunk vs KI-joint (#4)

*2026-08-07 ~05:1xZ. Immutable once posted. Ideas [#4](../ideas.md),
from the [π0.5 + KI deep read](2026-08-07-pi05-deep-read.md)
(arXiv:2504.16054 / 2505.23705), with the seam question's three-way
published map banked in #4:
[LabVLA](https://arxiv.org/abs/2606.13578) (a third group ships the
KI-joint recipe), [AEGIS](https://arxiv.org/abs/2604.16067) (names
"cross-modal gradient asymmetry", repairs it with orthogonal gradient
projection — the trained-repair middle path), and
[Wall-OSS-0.5](https://arxiv.org/abs/2605.30877) (discrete CE routes
action gradients into the backbone, flow as deployment interface —
structurally our sequential recipe, argued from the
multimodal-preservation side). The instrument does NOT exist yet: the
`bijou.train` molmo2 guard says so explicitly ("the flow phase lands
with its own pre-registration" — this is that post). It lands
oracle-gated before launch; if implementation forces any semantic
deviation from this post, an amendment posts before launch (the
[#19 amendment](2026-08-06-prereg-ar-sampled-draws.md) precedent).*

## Question

When the flow expert attaches to the Molmo2-4B AR trunk at its 40k
endpoint, does keeping the trunk adapting — its phase-1 CE objective
continuing during expert training, with stop-gradient on the
expert→trunk seam (the π0.5/KI production recipe, now also LabVLA's)
— beat our sequential hard-freeze default, at matched steps and
matched eff-batch? Our banked gemma-lineage stage-2 (6.62 panel @80k,
2.2× smaller expert than the h1536 lineage) is "extreme KI": the
trunk's discrete phase simply ended before the expert's flow phase
began. KI's frozen-backbone-0% result does not indict it (their
backbone was action-naive; ours is action-pretrained) — which is
exactly why this needs a measurement, not an adoption: the three
published camps all argue from trunks unlike ours.

## Arms — the seam is the ONLY contrast

| arm | trunk during expert training | expert sees | loss |
|---|---|---|---|
| **F — frozen** (default) | hard-frozen, encode without grad | residual taps | flow only |
| **K — KI-joint** | phase-1 CE continuing (`backbone-text-lr 2e-5`) | `sg(`residual taps`)` | CE + 1.0·flow |

Not run: **naive joint** (flow gradients into the trunk, no CE) — KI
measured that arm at production scale (~75%→5–10% language following,
7.5× slower convergence); we don't spend a rung re-measuring a
published collapse. α = 1.0 fixed, no tuning — KI's stop-grad result
is that the balance stops needing tuning (their α=1 vs π0.5's tuned
α=10); if K only works under a tuned α, that is a finding, not a
knob to fish.

K's CE branch is the phase-1 objective verbatim — same
`--aux-fields subgoal holding progress event visible`, same condition
fields and dropouts (`subgoal-dropout 0.5`), same
`backbone-text-lr 2e-5 --grad-clip 100`, same frozen
embeddings/lm_head (the molmo2 unfreeze surface) — so "trunk keeps
adapting" means *continuing the run it was in*, not a new objective.

## Shared recipe constants (identical across arms, NOT under test)

- **Start**: `fontaine_molmo2_ar_40k_ddp4/step_040000` via
  `--backbone-init-from` (trunk + prompt only, decoder fresh) —
  conditional on that run's own pre-registered gates having held
  (K1 probe ≤ 12.0944 by 10k crossed green; endpoint saved).
- **Conditioning surface**: residual taps (`--conditioning-streams
  residual`), the arch-batch arm-B recipe. The gemma tap rule (one
  tap per non-KV-shared prefix layer, 1:1 ascending) is
  Gemma-structural — molmo2's 36 uniform full-attention layers have
  no KV-share boundary — so the molmo2 rule is pinned here:
  **12 taps at uniform stride 3, trunk layers 2, 5, 8, …, 35**
  (last tap = final layer; expert layer i reads tap i, 1:1
  ascending; expert depth 12). Spanning the full stack is the
  π0.5-direction choice; the *depth-of-reads dial itself is NOT
  measured here* (#4 arm 1 stays open) — the surface is a constant
  held identical across F and K.
- **Expert**: h1024 / 8 heads / 4096 intermediate / 8 cross-heads
  (banked-lineage size), `--time-conditioning adarms
  --self-attention-mode bidirectional`, `--decoder-lr 1e-4`,
  warmup 500.
- **Data/topology**: identical to phase 1 — same dataset, split
  (`--holdout-episodes 0.1 --split-seed 0`), `--max-crops 1`,
  eff-48 (12/rank × 4×DDP, one eff-batch for both arms, never
  per-arm), `--zero1 --backward-chunks` as memory requires (same
  value both arms), seed 0, eval 256 @ every 500, save every 2500.

## Instrument (to land, oracle-gated, before launch)

Semantics frozen here; flag spellings are implementation's:

1. **Molmo2 residual exports** — `Molmo2Encoder` grows
   `residual_exports` + `stream_geometries` (the gemma
   `residual_sink` pattern); the `--decoder ar_backbone`-only guard
   lifts to admit `flow` + residual. Oracles: taps byte-match the
   trunk's post-layer hidden states at the pinned indices; an
   encode with taps enabled leaves the trunk's own output
   bit-identical to one without.
2. **Seam stop-grad flag** on the tap-consumption path. Oracle:
   with the flag, flow-loss gradients into every trunk parameter are
   exactly zero while CE gradients are nonzero; without it, nonzero
   (the naive-joint arm exists in code only as this oracle's
   negative control).
3. **Joint objective** (K only): one optimizer step over CE + flow.
   Oracles at the α-edges: flow-only with trunk frozen ≡ the F-arm
   step bit-for-bit; CE-only with the expert detached reproduces a
   phase-1 `ar_backbone` step's trunk gradients on the fixture
   family.
4. **#20 activation checkpointing** (its own queued item, keystone
   oracle: checkpointed ≡ plain forward/backward) — a **hard
   prerequisite for K**: phase 1 already sits at 67.07 GiB of the
   71 GiB gate with no expert riding; K adds expert
   params/grads/Adam + flow activations to a trunk-trainable step.
   An F1-style smoke memory ladder runs before launch; if K cannot
   fit eff-48 under 71 GiB even checkpointed, both arms downshift
   batch together (matched, loudly echoed) — never K alone.

## Gates (in-run, mechanized where precedent exists)

- **vram_alloc_peak ≤ 71 GiB** (standing box rule), both arms.
- **Sanity kill (K1-style, archB precedent verbatim)**: in-run
  256-frame probe > the phase-1 molmo2 trunk probe at the matched
  step + 3.0 at any eval ≥ 5k ⇒ kill that arm at the next save
  boundary (phase-1 curve: 9.64@5000 ⇒ bar ≈ 12.6@5000).
- **Cost gate, measured not judged**: first ~200 steps of each arm
  project the batch total (the `draws_rate_gate.py` mechanization
  pattern); projected total > **70 GPU-h** ⇒ both arms downshift to
  5k matched steps, the switch echoed loudly and the result labeled
  5k-screen.
- **K CE-health watch (record, not gate)**: K's CE-branch loss vs
  the phase-1 tail (~3.68 at 40k) at every eval — a rising CE under
  stop-grad is the drift signal AEGIS names; it feeds read 4.

## Frozen reads

Panel: `plans/holdout_curated_v0_k4l2_panel_v2.json`, flow keying
`--sample-draws 1 --sample-steps 30 --sample-method heun --noise-key
stable` (stem `__panel_v2_heun30_draws1_stable`), 4-GPU sharded,
sha256-pinned in the launcher. Paired per-frame reads from
`--dump-predictions` npz, seeded bootstrap 95% CI (seed 0, 10,000
resamples — the arch-batch conventions, `ci_excludes_zero`).

1. **Primary: Δ_seam = chunk_mae(K) − chunk_mae(F)** at matched 10k
   steps, paired per-frame, CI. This one number is the screen.
2. **Decision rule, frozen now**: Δ_seam < 0 with CI excluding zero
   AND read 4's drift band respected ⇒ KI-joint is the attachment
   recipe; the full-length attachment run pre-registers citing this.
   Δ_seam ≥ 0 or CI includes zero ⇒ **the frozen default stands**
   (ties go to cheaper + simpler), the KI-joint direction closes for
   this trunk class, and the Wall-OSS reading — phase-1 CE already
   routed the action gradients — is the recorded interpretation.
3. **Context anchors (quoted beside, never the decision)**: the
   molmo2 40k endpoint greedy AR panel number (same trunk,
   cross-decoder-class); gemma flow lineage **6.5997** @80k
   stable-key (cross-trunk, directional only, own-baseline rule);
   state-copy 11.785 as execution oracle (both arms must beat it
   decisively or the screen is void, not merely negative).
4. **Trunk-drift diagnostic (K only)**: greedy AR panel eval of K's
   trunk at the 10k screen end vs the 40k endpoint AR number —
   the KI "language following preserved" analog on our instrument.
   Band, frozen: |Δ_AR| ≤ 0.3 ⇒ drift acceptable. K wins on Δ_seam
   but breaks the band ⇒ KI-joint wins *with a named cost*; the
   AEGIS orthogonal-projection repair becomes the named escalation
   (banked, not built), and adoption waits for owner steer.
5. first_mae mirrors of 1 and 3; per-step-in-horizon curves from the
   npz for both arms (where in the chunk does the seam matter —
   record-only).

## Numbered expectations (banked before data)

1. Both arms beat state-copy decisively and land in flow-family
   range on their first screen — confidence medium-high (the gemma
   lineage did; a miss voids the screen rather than deciding it).
2. **Δ_seam < 0, modest — |Δ| in the 0.1–0.5 panel-MAE band** —
   confidence medium-low, the genuinely open number. Adoption
   evidence (π0.5/KI/LabVLA) says trunk adaptation helps; our
   trunk being action-pretrained says most of that help may already
   be banked in phase 1 (the Wall-OSS reading). This screen exists
   because the camps disagree about exactly our case.
3. K's trunk drift stays inside the band (|Δ_AR| ≤ 0.3): stop-grad
   plus the *same* CE data it was already training on should move
   the trunk gently — confidence medium.
4. **Falsified if Δ_seam ≥ 0**: KI-joint buys nothing on an
   action-pretrained trunk at screen scale. Recorded with the drift
   diagnostic either way; any escalation (all-layer reads — #4
   arm 1, AEGIS projection, α sweeps) needs a NEW pre-reg citing
   this result. No post-hoc α fishing, no rung extensions to chase
   a trend.

## Cost & scheduling

Estimates (measured gate above decides, not these): F ≈ 1.0–1.4
s/step → ~11–16 GPU-h; K ≈ 2.4–2.8 s/step (phase-1's 2.2 + expert)
→ ~27–31 GPU-h; panel evals ~3–5 GPU-h/arm + the K trunk-drift AR
panel ~2–4 GPU-h ⇒ **batch total ≈ 50–60 GPU-h, ceiling 70** with
the 5k downshift. Venue: the box 4×H100, **sequential, F first**
(cheaper, shakes out the shared instrument before K's memory-heavy
step), each 4×DDP. Run names
`fontaine_molmo2_flow_frozen_10k_ddp4` /
`fontaine_molmo2_flow_kijoint_10k_ddp4`; `babysit.toml` entries at
launch; first-poll util+rate check per standing rule.

Opens strictly after, in order: the molmo2_ar40k endpoint saves →
its chained greedy endpoint eval + the pre-registered #19
draws10_t1 arm (that pre-reg's box obligations come first) → the
instrument items above land oracle-gated (`check.py` green) → the
`molmo2-stage2-attachment-decision` queue item opens and the owner
has had the chance to steer it. This post makes that decision
executable, not automatic: the screen is the decision's first
measurement, and the full-length attachment run gets its own
pre-registration citing the screen's numbers.

## Amendment (2026-08-07 ~06:0xZ, pre-launch — instrument landing)

One under-specified corner surfaced at implementation and is pinned
here per the amendment rule above. The "Shared recipe constants"
section says the start is "`--backbone-init-from` (trunk + prompt
only, decoder fresh)". **"Decoder fresh" means the flow expert.** The
K arm's CE rider (the phase-1 `Molmo2ARDecoder` — FAST embedding/head
tables, decoder-owned) additionally loads its tables from the
endpoint checkpoint's `expert.safetensors`, strictly. Anything else
contradicts the arms table's definition of K — "phase-1 CE objective
continuing **verbatim**": a fresh-table CE branch would restart the
action head at row-mean init and the CE-health read against the
phase-1 tail (~3.68) would be meaningless. The F arm is untouched
(no rider). Also pinned: the rider's tables train at `--decoder-lr`
(their phase-1 routing), and the joint total logs the flow loss as
`loss_action` and the CE branch's per-token action CE as `loss_aux`
— the CE-health watch reads the latter against phase-1's
`loss_action` at the matched step. No other semantic changes; the
arms, gates, frozen reads and decision rule stand as posted.

## Amendment 2 (2026-08-07 ~17:0xZ, pre-launch — save cadence)

Operational, not a measurement change. The recipe constants above
say "save every 2500" — that number was chosen when a save stalled
stepping ~15.5 min on the 4×DDP box (the sync consolidate-and-write
path), balancing step-stall against recovery loss. Async checkpoint
saves landed after this post (`e3bdc93`, 08-07): capture is seconds
on the step path and the gather+write runs on a background thread,
so the stall side of that trade is gone. **Both arms save every
1250** (matched — the seam stays the only contrast). Every
save-boundary the post's judgment rules name is preserved: 1250
divides 2500, so the kill-rule boundaries (5000, 7500), the 10k
endpoint, and the 5k-downshift matched-read checkpoint
(`step_005000`) all remain save boundaries; the change only adds
midpoints. Motivation: three driver-kill incidents on 08-07 made
worst-case recovery loss concrete — halving the interval halves it
(~108 → ~54 min wall at the K arm's estimated rate) for seconds of
capture stall per extra save and ~40 GB disk per extra K save
against 6.3 T free (F saves are small: the frozen backbone is
hardlinked, only expert + optimizer are written fresh). Also
decided here: the DataStates pinned-buffer refinement stays banked
(#18.9) — the capture stall is seconds against a ≥26-minute save
interval (<0.2% overhead), and touching the oracle-gated save path
the day before a 50–70 GPU-h screen buys ~a minute total across a
run. Gates, arms, frozen reads and the decision rule stand as
posted.

## Amendment 3 (2026-08-09 ~00:5xZ, pre-launch — warm-start repoint to 60k)

Executes the 60k continuation pre-reg's frozen decision rule
(posted before that run launched): its read 1 came back **IMPROVED
— paired Δ(60k−40k) −0.1388, CI95 [−0.194, −0.090], 17,204 core
frames** (analysis banked,
`analysis__molmo2_60k_vs_40k_k4l2.json`), so **the screen's
warm-start checkpoint repoints from `step_040000` to
`fontaine_molmo2_ar_60k_ddp4/step_060000`**. The AR-100k bar was
NOT passed (5.8602 vs 5.8026, +0.058, cross-trunk unpaired) — noted
honestly, per that pre-reg the repoint does not depend on it.

Mechanics of the repoint, all landed in this amendment's commit:
both arm launchers' and the K-smoke ladder's `ENDPOINT` lines;
`attach_seam_results.py`'s read-4 drift comparator now pulls the
**60k** endpoint panel json (band 0.3 unchanged; oracle re-run
green). Consequences per the original boundary text: **the K-smoke
memory ladder must re-run GREEN at the 60k warm start before either
arm launches** (`k_mem_ready` is already absent — the ladder
deletes and re-earns it); phase-1 CE verbatim, seam, α, matched
steps/batch, gates, frozen reads and the decision rule all stand as
posted. The trunk-drift band's comparator value changes with the
repoint (6.0079 → 5.8602) because read 4 measures drift *from the
warm-start trunk*, which is now the 60k endpoint.
