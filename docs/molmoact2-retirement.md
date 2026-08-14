# Retiring `bijou/molmoact2/` — first-class MolmoAct2 AR + joint objectives

Status: **COMPLETE — ALL PHASES EXECUTED 2026-08-14** (owner
delegation: "finish the whole plan"; every gate green, receipts in
§5). `bijou/molmoact2/` is DELETED; this file is now the migration's
HISTORICAL RECORD. The as-of-now design lives in
`docs/architecture.md` (§8.13 step-8 closure; §2.2a the discrete
head; §5's regression gates carry the objective-matrix CPU anchors).
Anchor tags: `pre-decoder-simplify` (T1/T2 deletions, 2026-08-13) and
`pre-molmoact2-retirement` = main `51704c0` (the last commit where
the port package exists unmodified — fixture generators and the
old-vs-new gate rerun only there). Executed-phase commits: phases 0–1
`c57ce05`/`0312ab7`/`7d89f53`/`77246a9`/`7423ec3`/`3131f82`;
main-roll-forward `64fcc24`; phase 2 `b30784d`/`b46a3ed`/`651c792`/
`e0b2192` + probe, acceptance PASS `e5b6113`; phase 3 (objective
matrix + tiny-fixture anchors) and phase 4 (grpo_replay re-point +
frozen-wave gate) and phase 5 (deletion) landed 2026-08-14 — see
`git log --oneline e5b6113..` for the per-commit gates.

---

## 0. For Fontaine — how to adopt these changes

You are the primary downstream consumer: your token-GRPO line builds
partly on `bijou/molmoact2/` (the package this plan deletes). The
short version: **your objective module and your artifacts survive
untouched; your imports move; one of your queue items is superseded.**

**Ordering (do these at run boundaries, never under a live run):**

1. ~~Rebase onto `main` ≥ `db0a141`~~ — **DONE 2026-08-14**: you
   rebased onto `0312ab7` (140 commits, zero conflicts, old tip tagged
   `pre-rebase-0312ab7`). Pull ≥ `7423ec3` to clear the two inherited
   parity-test failures — your cross-machine finding is the registered
   bound now (forward atol 2e-6, euler 1e-5; fixture README has your
   measurement).
2. Adopt phases 2–3 whenever convenient after they land — they do not
   touch your live surfaces.
3. **Phase 4 is your instrument** (GRPO rollout/replay re-pointing) —
   the ladder adjudicated **STOP on surface A** (your 13:1xZ
   recommendation: probe flat across 6 steps of both runs, shove
   redistributed not retired, competence artifact at 4/64 pinch
   successes, the remaining ~14 GPU-h re-buying the same physics;
   owner-ratified 2026-08-14), so the window is OPEN: phase 4 lands
   after our phases 2–3, gated on frozen-wave replay parity on the
   banked R1-B waves including one v2-reward wave. Any GRPO thread
   restart (e.g. competence-first SFT) is a NEW pre-registration,
   post-phase-4, per decision 11.
4. Phase 5 (the deletion) only lands after your phase-4 sign-off.

**What does NOT change for you:**

- `bijou/train_grpo.py` — your objective module is untouched. It was
  already generic over `ARSuffixDecoder`; the new decoder plugs into
  it with zero changes.
- Your **row NPZ format** (TrainingRowWriter output: frames + sampled
  bins + per-token logprobs + model-unit state) — unchanged; banked
  rollout rows stay readable.
- Your **loop checkpoint format** (`grpo_phase2_*/step_NNNN.pt`) — the
  loop owns it; resume compatibility preserved.
- Your sim seed streams, reward definitions (v1/v2), tripwires,
  babysit schema.
- `test_grpo_step` / `test_token_rows` oracles — preserved as-is.

**What moves under you (import-path table):**

| today | after |
|---|---|
| `bijou.molmoact2.fast_codec` | `bijou.fast.molmoact2` (module content + your vendored fixture preserved) |
| `bijou.molmoact2.replay` (TokenRow rebuild + teacher-forced replay) | retired — replaced by a thin builder on `MolmoAct2InputsCollator` + `bijou.decoders.ar_backbone.suffix_targets` + the generic capture path you already threaded (`ar_predict_sampled(action_capture=…)`, `token_rows_from_capture`) |
| `bijou.molmoact2.predictor.MolmoAct2Predictor` (sim serving + `predict_action_discrete`) | `bijou.eval.policies.BijouPolicy` over a converted checkpoint + the new `MolmoAct2ARDecoder` (decode parity gated on your banked anchors before the port dies) |
| `bijou.molmoact2.processing` leaves (if you import any directly) | `bijou.encoders.molmoact2` |

**Queue impact:** your `molmoact2-ar-head-port` item is **superseded
by phase 2** of this plan — confirmed already CLOSED on your side
(08-13; sign-off 2026-08-14), so nothing is pending and there is no
duplicate-work risk. Phase 2 deliberately adopts your `fast_codec`
audit facts and your grammar-mask/budget-arithmetic contract
verbatim.

**What you gain:** molmoact2 GRPO rides the same `ARSuffixDecoder`
path as er-60k (one capture/replay/objective code path across both
policies), and the joint AR+flow objective (phase 3) gives your line
an RL-then-refine composition: GRPO the discrete head, then train the
flow expert against the RL-shifted trunk — with knowledge insulation
as a flag.

---

## 1. Background (no context assumed)

**Bijou** is this repo's vision-language-action stack for SO-100/101
arms: a pretrained multimodal *trunk* (Gemma-4 or Molmo2/Qwen3-family)
composed with a prompt-side *encoder* strategy and an action *decoder*
via `BijouModel`. A checkpoint records a *prompt kind* (which encoder)
and a *decoder kind*; `ar_backbone` is the decoder-only kind where the
trunk itself continues an action-token suffix against the retained
prefix KV cache — one generic scaffold (`ARSuffixDecoder`), one
concrete class per trunk.

**MolmoAct2** (AllenAI) is a released VLA on a Molmo2-4B (Qwen3)
trunk with *two action pathways* (its `action_mode='both'`): a
**discrete head** — the trunk natively emits `<action_start>` +
`<action_N>` bin tokens, detokenized by a released FAST (DCT+BPE)
action tokenizer — and a **DiT flow expert** cross-attending every
trunk layer's KV. Architecture.md §8.13 adopted the *expert* pathway
first-class (2026-08-11/12): the `molmo_flow` decoder,
`MolmoAct2Encoder` (their serving prompt as an encoder mode), and
`bijou.convert_molmoact2` (their checkpoints → our format, byte-parity
gated). The **discrete head was never first-classed** — it stayed in
`bijou/molmoact2/`, the *port package*: a code-level port of their
implementation kept as a frozen byte-parity reference, slated for
retirement (§8.13 step 8).

**What changed:** the port stopped being frozen. Fontaine (the
autonomous research agent; branch `fontaine`) built token-GRPO — RL on
the discrete head in simulation — with the objective first-class
(`bijou/train_grpo.py`, generic over `ARSuffixDecoder`) but the
policy surface port-side (`predictor.predict_action_discrete`,
`replay.py`, `fast_codec.py`), per owner steering 2026-08-13 10:02Z
(the discrete pathway was the right RL surface, and no first-class
implementation of it existed). The port now hosts live, growing,
tested code — the exact opposite of a frozen reference. This plan
retires the package by finishing the first-classing it was waiting
on, and adds the training compositions the owner requested
(2026-08-14): **AR-only, flow-only, and joint AR+flow with optional
stop-gradient from the flow objective.**

Relevant prior art already on `main` and load-bearing here:

- `MolmoAct2Encoder` builds their prompt; `ObservationMemory.cache`
  carries the prefix KV; `retain_cache` is on for `ARSuffixDecoder`
  and `MolmoFlowDecoder` compositions.
- `MolmoFlowDecoder` conditions on **all** trunk layers' KV
  (`layer_kv_pairs`); `--insulate-expert` detaches that extraction
  (knowledge insulation: flow gradients into every trunk parameter
  exactly zero; gradient contract test-pinned both ways).
- `BijouModel.joint_ce` — a dormant CE-rider slot (param_groups +
  retain_cache + `CheckpointMetadata.joint_ce` serialization) kept
  through the T2 deletions *for exactly this plan*.
- The Molmo2 precedent for new AR concretes: decoder kind stays
  `ar_backbone`; **the trunk axis is the prompt kind**.

## 2. Decision register

1. **The discrete head is a third `ARSuffixDecoder` concrete**
   (`MolmoAct2ARDecoder`), checkpoint decoder kind `ar_backbone`,
   prompt kind `molmoact2` — the Molmo2 precedent, no new kind.
2. **Zero new parameters.** The head's ids are trunk-native:
   embedding/logits read the trunk's own `wte`/`lm_head` rows;
   `block_base = action_token_start_id`, construction-validated to
   sit inside the base matrices (a straddle into `new_embedding` or
   past the head would silently train the wrong rows — fontaine's
   guard, promoted). Simpler than the Molmo2 concrete (no
   `fast_embed`/`fast_head` tables).
3. **The released FAST codec is promoted, not rewritten**:
   `bijou/fast/molmoact2.py` = fontaine's `fast_codec.py` (bit-for-bit
   vs its vendored reference fixture) behind the codec protocol
   surface `ARSuffixDecoder` consumes (`boa` ≡ `<action_start>`,
   `pad` convention, `vocab_total`, `symbol_lengths`). Release-artifact
   facts carried verbatim: 2048-wide block with only **1005 reachable
   bins** (symbol_lengths 0 elsewhere — the grammar mask excludes the
   1043 unreachable rows for free) and **7 quantization-hole symbols**
   — raise-not-fallback stays (fontaine sign-off 2026-08-14, with the
   cost measured: **0/2996 fallbacks** across arm B's masked decodes,
   so the loud path costs nothing in practice). The `pad` analog for
   the discrete surface is pinned during phase 2 (open detail,
   flagged).
4. **Objective matrix** on molmoact2-format checkpoints:
   `--objective {flow, ar, joint}` — flow = today's `molmo_flow`
   composition; ar = `MolmoAct2ARDecoder` alone; joint =
   `MolmoFlowDecoder` decoder + `MolmoAct2ARDecoder` in the
   `joint_ce` slot. Joint loss = `L_flow + λ·L_CE` with
   **`--joint-ce-weight` (float, default 1.0** — the KI no-tuning
   default; owner decision 2026-08-14: a knob, not a constant).
   Validation: requires `--objective joint`; must be > 0 (λ = 0 is
   spelled `--objective flow`). It is a RUN hyperparameter like
   `--decoder-lr` — recorded in the checkpoint's train_args for
   provenance, NOT checkpoint-inferred (resumes may re-pass it; it
   changes the objective, not the parameter set). Note that under
   `--insulate-expert` the two objectives reach disjoint parameter
   sets (flow → expert only, CE → trunk only), so λ mostly rescales
   what the per-group LRs already control — it binds in the
   uninsulated joint.
5. **Stop-gradient from flow = `--insulate-expert`** (existing flag,
   unchanged semantics). One ordering rule becomes a test-pinned
   invariant: the expert's `layer_kv_pairs` extraction reads the
   **prompt-only cache slice BEFORE the CE suffix forward appends to
   the cache** — the expert never conditions on teacher-forced action
   tokens (this is also their 'both'-mode span-strip semantics and
   keeps §8.13 step-6 narration compatible).
6. **Parity by vendored fixtures.** The port is today's parity
   reference; deletion converts live-pair tests into committed golden
   fixtures generated by executing the port at
   `pre-molmoact2-retirement` (the `fast_codec` pattern). Live-pair
   tests retire with the reference.
7. **Gate-d runs as GATE-D-LITE, a 500-step prefix** (owner decision
   2026-08-14: micro gates accepted as-is; a shortened macro gate as
   the middle ground). The run is the rig-rung recipe through
   `bijou.train` (`--decoder molmo_flow --init-from` the converted
   release, global batch 64, AE lr 5e-5, warmup 200) and is a STRICT
   PREFIX of the full 2000-step rung — resumable, so if the GPU
   window holds it continues 500→2000 for the free full-endpoint
   read (3.23 class). Pre-registered lite thresholds at step 500,
   set from the reference rung's own recorded intermediates
   (fontaine's rig-ft results, 2026-08-10): (i) 240-anchor-row MAE
   in the **6.76 class** (band set at pre-registration from the
   instrument's noise floor), i.e. both anchors beaten — state-copy
   9.08 and zero-shot 28.95 — at ¼ training, exactly as the
   reference did; (ii) training-loss corridor match at matched steps
   (his 0.135@20 head + intermediate values pulled from the rung's
   train log at pre-registration). Cost reality: the 40-step smoke
   measured 0.31 s/step at batch 8 (frozen trunk), so lite ≈ 20–40
   min + evals and even the full 2000 ≈ 1.5–2 GPU-h — the original
   ≤6 GPU-h figure was a budget bound, not an estimate. The
   sub-decision (their recipe RECOMPUTES the q01/q99
   table on rig data at fine-tune start; our converted release
   carries the community table) is settled at this run's launch — a
   small `--norm-table` recompute knob or a convert-time alternate,
   chosen in the pre-registration; reproducing the rung requires the
   rig table either way.

   **EXECUTED 2026-08-14, verdict: PASS with a favorable deviation.**
   Mechanism: convert-time (`--norm-stats-from` the rig-ft export;
   `converted/molmoact2_release_rigtable`). Full 2000 steps ran (the
   opportunistic continuation cost ~50 min total, 1.47 s/step @ batch
   64). Corridor: loss 0.191@20 (≤ the registered 1.5× bound on the
   reference 0.135), monotone throughout, 0.0089@2000 (reference
   ~0.008 class ✓). Anchor reads on the reference's 240 banked rows
   (row mapping verified bit-exact against the bank's states;
   state-copy reproduced at 9.082): **step-500 MAE 5.556** (beats
   both anchors ✓, but BELOW the registered 6.76 ± 1.0 band — 1.2
   BETTER than the reference at quarter-training) and **step-2000 MAE
   2.030** (reference 3.23 ± 1.0 — again better, outside the band on
   the favorable side). The gate's bug-detection intent passes on
   every sub-gate; the unexplained IMPROVEMENT is recorded as a
   deviation with unattributed mechanism (candidates: trunk mount
   dtype at train time, batch composition/order, loss-normalization
   details — the reference recipe's cosine warmup matches ours, so
   not the schedule shape). Lesson recorded: recipe-repeat bands
   should be one-sided-worse plus a better-than flag, since "better"
   is a deviation to explain, not a failure — but it is also not
   grounds to block retirement.
8. **Sequencing:** nothing lands under a live fontaine run; phase 4
   co-landed with him; phase 5 only after his phase-4 sign-off.
9. **Surface-A row restriction** (training only the action-block rows;
   depends on wd=0 grad-row masking) stays a GRPO-loop lever — not a
   `bijou.train` flag.
10. **Formats frozen:** GRPO row NPZ and loop `.pt` checkpoints
    unchanged.
11. **Run provenance across phase 4** (fontaine's question, resolved
    2026-08-14): loop `.pt` resume compatibility is a SALVAGE-ONLY
    escape hatch — any run launched after the re-point **starts
    fresh** (one code path per run: anchor reference forwards, rollout
    waves and replay all on one stack; a run whose early steps rolled
    out under the old driver and later steps under the new one is a
    two-instrument run even inside the parity bounds). Inheriting a
    pre-migration run's PROGRESS is done as a warm START — a new run
    initialized from the `.pt` weights with a fresh pre-registration —
    never a resume. Mathematically a resume would be near-clean (GRPO
    rows are consumed the step they are collected, each carrying its
    own π_old), which is exactly why this is pinned as a provenance
    rule rather than left to case-by-case judgment.

## 3. Inventory and disposition of `bijou/molmoact2/`

| module | contents | consumers today | disposition |
|---|---|---|---|
| `processing.py` | template constants, `discrete_state_string`, uint8/378 image path, `QuantileStats` + q01/q99 normalize, `encode_action_prompt`, sequence budget | `encoders/molmoact2.py` (leaf imports), predictor, tests | **move** → `bijou/encoders/molmoact2.py` (merge or sibling `molmoact2_ops.py`); golden fixtures move along |
| `predictor.py` | full serving pipeline; `IMAGE_TOKEN_STRINGS`; fontaine's `predict_action_discrete` + capture + `extract_action_bins` | encoder (constants), parity gates, fontaine's sim rollouts + replay | constants **move** to the encoder; discrete decode **absorbed** by `MolmoAct2ARDecoder` (phase 2); serving **replaced** by `BijouPolicy`; then **delete** |
| `wiring.py` | `encoder_attention_mask`, `generate_actions` (reference solver) | encoder (mask), molmo_flow parity tests | mask **moves** to encoder; solver retires into phase-0 fixtures |
| `action_expert.py` | reference DiT expert (`ActionExpertConfig`) | molmo_flow live-pair parity tests, converter test fixture | retires into phase-0 fixtures; converter test gets an explicit tensor-name table |
| `fast_codec.py` (fontaine) | released FAST tokenizer, native, fixture-pinned | `replay.py`, GRPO | **promote** → `bijou/fast/molmoact2.py` |
| `replay.py` (fontaine) | row storage → teacher-forced replay → ratio glue | `sim/grpo_loop.py` | **retire**; thin rebuild on the generic path (phase 4) |
| `train.py` | the port's training recipe (their reference) | gate-d only | retire (per decision 7) |

To verify at execution (flagged, not assumed): `convert_molmoact2.py`
is port-import-free; the real tokenizer's `<action_start>` id and
`action_token_start_id` match the stub constants used in tests
(151_932 family); the discrete `pad` convention (decision 3).

## 4. Target architecture

```
MolmoAct2Encoder ──encode(retain_cache)──► ObservationMemory.cache
                                            │
        ┌───────────────────────────────────┼───────────────────────┐
        │ (prompt-only KV slice,            │ (cache continuation,  │
        │  extracted FIRST,                 │  appends suffix K/V)  │
        │  detached iff --insulate-expert)  │                       │
        ▼                                   ▼                       │
  MolmoFlowDecoder                   MolmoAct2ARDecoder             │
  flow-matching loss                 CE over [<action_start>,       │
  (objective: flow / joint)          bins…] via trunk-native rows   │
                                     (objective: ar / joint)        │
```

- **just flow**: exactly today's `molmo_flow` checkpoint/composition —
  unchanged bytes, unchanged schema.
- **just ar**: `BijouModel(decoder=MolmoAct2ARDecoder)`. Trainable
  surface = the trunk (`--backbone-text-lr`); the decoder itself owns
  nothing. Decode: `<action_start>` forced (its identity is not a
  decision, the BOA convention), then grammar-masked greedy/sampled
  emission under budget arithmetic over the released codec's symbol
  lengths; `predict_chunk`/`ar_predict_sampled`/capture inherited from
  the scaffold.
- **joint**: flow decoder + AR rider in `BijouModel.joint_ce`. Loss =
  flow + 1.0·CE; `BijouTrainStep` regains a joint arm (the
  T2-deleted `_joint_share` shape: three chunked-backward normalizers
  — flow element count, CE token count, aux None — CE suffix forward
  inside the bf16 autocast region, flow fp32 outside). The rider has
  no parameters, so **no `joint_ce.safetensors`**; trunk deltas ride
  the existing `backbone.safetensors` machinery.

**Checkpoint schema.** AR section = the `ar_backbone` decoder section
(kind reused) + molmoact2-specific fields: released-codec reference,
`action_token_start_id`, chunk/action geometry. Joint checkpoints
carry the molmo_flow decoder section plus the AR section in the
existing `CheckpointMetadata.joint_ce` slot. `from_checkpoint`
rebuilds all three compositions; every combination is evaluable
(`BijouPolicy` dispatches by decoder kind as today; the joint
checkpoint's deployment read is the flow decode — the AR head rides
along for RL/narration work, and an AR-only read of a joint
checkpoint is a `--objective`-style eval option, not a new policy
class).

**CLI.** `--objective {flow, ar, joint}` valid only for the
molmoact2 prompt family (inherit-only via `--init-from`/`--resume` as
today); `joint` and `ar` require a live trunk (`--backbone-text-lr`);
`--insulate-expert` legal for flow and joint (its
flow-grads-are-zero contract is per-objective, CE unaffected);
validations live in `TrainArgs.__post_init__` (single encoding).

## 5. Phases

**Phase 0 — freeze the reference. EXECUTED 2026-08-14.**
(a) `tests/fixtures/molmo_flow_parity/` (port forward + state path +
seeded Euler, deterministically perturbed out of the adaLN-Zero
vacuum, non-vacuity asserted; `generate.py` beside the fixture,
runnable only at the tag). Cross-machine lesson (fontaine): fp32 CPU
kernels are not bit-portable — comparisons use REGISTERED bounds
(forward atol 2e-6 = his 4.17e-7 measurement ×5; euler 1e-5), not
`torch.equal`. (b) `tests/fixtures/molmoact2_discrete/`
(`decode_anchors.npz`): the reference discrete decodes — 6 stable
holdout first-frames × {grammar-masked greedy, unconstrained
reference} on the RELEASE, with ids/bins/executed chunks/per-step
masked logprobs/violation counts; generated by executing fontaine's
`predict_action_discrete` (his `3cac531`) on the box H100 in the
read-only worktree `~/marius-fontaine-ro`. (c) prompt-assembly byte
fixtures kept as-is. Gate-d-lite: decision 7's verdict block.

**Phase 1 — promote the leaves. EXECUTED 2026-08-14** (`c57ce05` +
`0312ab7`): `processing.py` → `bijou/encoders/molmoact2_processing.py`
(gaining `IMAGE_TOKEN_STRINGS`, `require_single_obs`,
`encoder_attention_mask` + helpers); explicit re-export shims keep the
port working until phase 5; the converter's one port import
eliminated; fontaine's codec promoted to `bijou/fast/molmoact2.py` +
`tests/test_fast_molmoact2.py` + the vendored tokenizer fixture;
`MolmoFlowConfig.released_so100_101` is the literals' home. The
converter gained `--norm-stats-from` (decision 7's mechanism,
oracle-pinned: table substituted, weights invariant, provenance
recorded). Gates: check.py green; 2-step oracles bitwise.

**Phase 2 — `bijou/decoders/ar_molmoact2.py`. EXECUTED 2026-08-14**
(main-roll-forward `64fcc24` first — fontaine's branch fast-forwarded
into main so one tree carries the GRPO line; then `b30784d` codec
layer split, `b46a3ed` adapter, `651c792` decoder, `e0b2192` loading,
`40799e9`+ probe). The concrete per decisions 1–3: trunk-native
`_suffix_hidden` (plain `wte` lookup — no extension-table select;
SIMPLER than the Molmo2 concrete), `_logits` = trunk `lm_head`
(full-id-space, grammar legality handled by the mask, not column
surgery), construction guards (block inside base matrices;
codec/geometry anchors; the trunk tokenizer's real
`<action_start>`/`<action_end>`/`<action_0>` verified against
block_base). Loading/schema arms; `BijouPolicy` works via the
existing `ARSuffixDecoder` genericity.

Execution facts (how the built thing differs from no-context
expectation):

- **The codec layer grew a naming grid + Protocol** (styleguide rule
  added): tokenizer = artifact + math, codec = AR conventions;
  `ActionCodec` is now the PROTOCOL, the concretes are
  `FastActionCodec` (ours) and `MolmoAct2ActionCodec` (theirs;
  `MolmoAct2FastCodec` → `MolmoAct2FastTokenizer`, atomic rename, no
  legacy alias — owner call). `symbol_lengths` is codec-owned (the
  scaffold's `id_to_token` reach-through was FastTokenizer-only API,
  wrong for their byte-level BPE).
- **The pad-analog open detail (decision 3) resolved by NEGATIVE
  special offsets**: the adapter presents `boa = −2`, `pad = −1`, so
  the scaffold's `block_base + offset` arithmetic lands on
  151932/151933 with `block_base = action_token_start_id` unchanged —
  capture stays [B, 2048] block-relative, zero ±2 rebase anywhere.
  `<action_end>` ≡ pad: legal exactly at symbol budget 0 (their
  close), fed as lockstep filler at B>1, NEVER a CE target — a
  documented narrowing of their SFT span to what masked decoding and
  the GRPO line train (bins; `<action_start>` is fed, not predicted).
- **Suffix format 6** registered beside format 5; each concrete
  asserts its own. Empty opener (their prompt carries the whole
  scaffold) exposed two latent scaffold bugs, fixed + test-pinned:
  `suffix_targets`' opener mask sliced `[:, :-1]` at width 0, and
  `teacher_forced_block_logits` read `next(self.parameters())` on a
  parameterless decoder.
- **The release AR read**: `molmoact2_ar_config_from_flow_section`
  derives the format-6 section from a RELEASE-class checkpoint
  (geometry from the molmo_flow section — identity output tail
  verified: n_obs_steps 1, n_action_steps == horizon 30 × dim 6,
  confirmed on the converted release; block_base from the trunk
  tokenizer's own `<action_0>`), then the shared
  `build_molmoact2_ar_decoder` (action_mode refusal by name;
  hub-routable codec via `resolve_checkpoint_dir`,
  `MOLMOACT2_FAST_TOKENIZER_REF`).

**Acceptance verdict (box H100, bf16 mount matching the generator):
PASS on every registered gate** — all 6 rows masked
ids/bins/actions BYTE-EQUAL to `decode_anchors.npz`; fixture
logprobs max |Δ| 2.384e-07 (bound 1e-5). One instrument lesson: the
probe's first run gated teacher-forced-replay-vs-capture logprobs at
1e-5 and FAILED — mis-registered bound. That comparison is
CROSS-surface (one wide suffix forward vs L single-token forwards);
under a bf16 trunk it sits at the batch-shape reduction-order floor
(measured 1.4e-2–5.6e-2 worst-step across the rows), and the fp32
diagnostic collapses it to 2.8e-5 — mechanism confirmed as shape
numerics (sharding.py's caveat), not semantics. The registered
1e-5(+JPEG) replay bound is SAME-surface (teacher-forced vs
teacher-forced) and governs phase 4's frozen-wave gate, which is
unaffected. Probe: `probes/probe_molmoact2_ar_parity.py`
(`--trunk-dtype float32` = the mechanism diagnostic); CPU suite:
`tests/test_molmoact2_ar.py` (11). check.py green throughout (869
passed); both 2-step oracles bitwise at every commit.
`~/marius-fontaine-ro` is now past its scoped lifetime (kept only
until this gate passed) — removable at the next box cleanup.

Execution facts learned generating the acceptance fixture (phase 0):
- **Scope: RELEASE-class checkpoints only** (`action_mode='both'`).
  The rig-ft exports are `'continuous'` — their fine-tune never
  trained the discrete head and the reference decode refuses them;
  the first-class decoder must refuse identically (loud, by name).
- **The id-space seam**: capture surfaces (`ActionCaptureStep`) carry
  BLOCK-relative logits/masks (`[B, 2048]`) but BACKBONE-id `chosen`;
  every gather/rebase crosses `action_token_start_id` (release:
  151934 block base; `<action_start>`/`<action_end>` at 151932/3 —
  verify against the real tokenizer at construction, the stub ids in
  tests mirror these). Getting this wrong was the fixture generator's
  one real bug — build the guard in, don't rediscover it.
- **The scaffold**: `<action_start>` is FED (the BOA convention — a
  constant, not a decision); `<action_end>` closes the stream at
  budget 0; only 1005 of 2048 bins are reachable (codec
  `symbol_lengths` 0 elsewhere — mask excludes free); the 7
  quantization holes stay raise-not-fallback.
- fontaine's codec loader is LOCAL-PATH-ONLY — phase 2 should route
  refs through `resolve_checkpoint_dir` so hub ids work (the box
  snapshot: `allenai/MolmoAct2-FAST-Tokenizer` @ `d45593b4c8…`).

Gates: masked-mode ids/bins/actions BYTE-equal to
`tests/fixtures/molmoact2_discrete/decode_anchors.npz` (discrete
surfaces are byte-portable; only fp32 logprobs drift), logprobs
within 1e-5; `train_grpo`'s unchanged-policy oracle (ratio 1, clip 0,
k3 0) on the new surface; the unconstrained reference mode may live
port-side only — it is their deployment quirk (zeros fallback), not
something the first-class decoder needs.

**Phase 3 — objective matrix. EXECUTED 2026-08-14.** `--objective
{flow, ar, joint}` (ArchSection.EXTENSION: freely selectable under
--init-from, locked to the recorded value under --resume) +
`--joint-ce-weight` (λ, default 1.0, joint-only, > 0) +
`--expert-init {inherit, fresh, <ckpt>}` (owner-agreed 2026-08-14:
fresh = released-shape synthesis for the stage-2 recipe — REQUIRED
from ar-only sources; <ckpt> = the two-source init under a
config-equality guard + loud table-provenance print). The historical
`_joint_share` reinstated with the ORDER FLIPPED for cache-conditioned
flow (decision 5 in the code paths: `loss_components` and
`BijouTrainStep` both extract the flow branch's prompt-only KV before
the CE rider's suffix forward appends; CE re-enters bf16 autocast,
flow stays fp32 outside). Collator gained the merged action-table
override (CE targets, batch stats rows and the flow clamp read ONE
table — their shared-table convention). Save side: parameterless
decoders write no expert file (the loader refuses stray ones), no
empty rider file; ar runs write the in-use tables into the
normalization row. Tests: decision-5 ordering BITWISE (the flow
component identical while the cache visibly grows), λ composition,
KI both ways (disjoint parameter sets) + the uninsulated complement,
round-trips for all three shapes, CLI validations through the real
parser. **Anchors on the tiny molmoact2 fixture (architecture.md §5
regression gates): flow 1.3906/1.3305; ar 12.2254/12.3317; joint(KI,
λ=1) 13.6160/13.6621 with the built-in cross-oracle (loss_action ≡
flow, loss_aux ≡ ar, total = flow + λ·CE, all bitwise). Measured
policy fix the oracle surfaced: real rig chunks DO hit the released
BPE's 7 quantization holes — the 0/2996 audit figure was masked
DECODES, which cannot produce holes by construction;
MolmoAct2ActionCodec.allow_quantization_holes (True in the training
collator) reproduces the reference recipe's short tokenization,
counted + printed, never silent.**

**Phase 4 — GRPO migration. EXECUTED 2026-08-14** (fontaine's shape
sign-off + v2-wave amendment honored; landed on main with his branch
fast-forwarded in, so the co-land is a rebase pickup on his side).
`bijou/grpo_replay.py` = the re-point: `MolmoAct2DiscreteStack` (the
AR read of a BIJOU molmoact2-family checkpoint behind the port
predictor's duck-typed attribute surface — the loop's freeze/anchor/
row-span machinery runs verbatim) + the thin replay (the scaffold's
teacher-forced suffix forward, fontaine's exact reduction ops; row
NPZ + loop `.pt` formats FROZEN, decision 10). `sim/grpo_loop.py` +
`sim/rollout_sim_parallel.py` load the stack from bijou checkpoints;
`ar_predict_sampled` threads `action_capture` (the promised generic
path); `test_grpo_loop` (20/20) runs on the facade over the shared
tiny-release builder (`bijou.testing.write_tiny_molmoact2_release`).

**Gate verdicts (box H100, 2026-08-14):** masks bit-equal on ALL rows
of both banked waves (R1-A step_0004: 904 rows v1; R1-B step_0006:
1904 rows v2), replayed at their collection weights (`.pt` restore
both stacks). Port-vs-first-class logprobs max |Δ| 4.4e-5 (v1) /
5.7e-5 (v2) — IN-BOUND under the loudly re-baselined **1e-4
cross-decomposition bound** (first registration said 1e-5; the two
replays are different decompositions — monolithic port forward vs
scaffold prefill+continuation — the SECOND occurrence of the "1e-5
implies same decomposition" trap, now a rule: 1e-5 only between
IDENTICAL forward decompositions). Per-token objective deltas 0.0 /
6.5e-8. Banked-vs-replay spread (JPEG + policy history) reported, not
gated — R1-B's own heartbeat trained under it (step-7 clip_fraction
0.141). The v2 trace-preservation leg ran POSITIVE: a fresh 1-seed×2
wave through the NEW driver end-to-end (`--train-reward v2`,
earned_progress computed from grip traces, no missing-trace raise,
one GRPO update, exit 0). The old-vs-new A/B is preserved in
`probes/probe_grpo_replay_parity.py`'s header; the probe itself is
now the new-stack wave-integrity instrument (the port side rerun only
at the tag).

**Phase 5 — delete `bijou/molmoact2/`. EXECUTED 2026-08-14.** The
package and seven port test files deleted; `validate_inference_config`
moved to `encoders/molmoact2_processing` (the converter's last port
import); `test_convert_molmoact2` fabricates source experts through
`MolmoFlowConfig` (same state-dict names — the §8.13 byte-parity
contract IS the name contract); `test_molmoact2_{encoder,processing}`
re-pointed to the promoted module and KEPT (they test live code);
fixture generators stay as pyright-excluded provenance documents
(runnable at the tag only). Oracle re-runs on the deletion tree, ALL
BITWISE: gemma flow 2.7903/1.9152, gemma ar 27.8306/27.767, molmoact2
flow 1.3906/1.3305, ar 12.2254/12.3317, joint 13.6160/13.6621.
check.py CHECKS PASSED (804 + 11 gpu-deselected). Box artifacts:
`~/marius-fontaine-ro` (the phase-0 read-only worktree) removed — its
scoped lifetime ended at the phase-2 gate.

Rough sizes: P1 ±0 net (moves), P2 ~250–350 + tests, P3 ~150 restored
+ schema, P4 net-negative (replay −340 + loop re-pointing), P5 ≈
−2,500.

## 6. Gate matrix (what proves each phase)

| gate | phase | instrument |
|---|---|---|
| fixtures reproduce port bytes | 0 | executing the port at the tag |
| leaf moves are inert | 1 | check.py + bitwise 2-step oracles (flow, ar_backbone) |
| discrete decode parity | 2 | phase-0 fixture (b) — **PASS 2026-08-14** (byte-equal ×6, logprobs 2.4e-7) |
| replay ratio contract | 2/4 | unchanged-policy ratio≈1 oracle; 1e-5 + JPEG bounds |
| KI both ways under joint | 3 | extended gradient-contract test |
| flow-only untouched | 3 | bitwise vs pre-phase oracles + molmo_flow fixture tests |
| old-vs-new GRPO stack | 4 | frozen-seed wave replay ×2: v1 wave (rewards equal, logprobs in-bound) + v2 wave (grip traces preserved — rewards equal, no missing-trace raise) |
| rig-rung class reproduction | 0 | gate-d-lite: step-500 MAE in the 6.76 class + loss-corridor match; 2000-step endpoint read free when the window holds (decision 7) |

## 7. Open items before the relevant phases

0. ~~The GRPO ladder adjudication~~ — **RESOLVED 2026-08-14: STOP on
   surface A** (fontaine's recommendation, owner-ratified). Phase 4's
   window is open, sequenced behind phases 2–3.

1. ~~Decision 4's joint weight~~ — RESOLVED 2026-08-14:
   `--joint-ce-weight`, default 1.0 (decision 4). (Fontaine acked the
   earlier fixed-1.0 form; the owner's knob-with-default supersedes —
   same default, re-passable.)
2. ~~Decision 7: gate-d~~ — RESOLVED 2026-08-14: gate-d-lite, the
   500-step prefix with opportunistic 2000-step continuation
   (decision 7). Fontaine independently endorsed running over
   waiving; the box H100 is idle post-R1-B (tripwire stop 12:40Z)
   through his ladder adjudication — the phase-0 GPU window is
   effectively open.
3. ~~Fontaine's sign-offs~~ — RESOLVED 2026-08-14: phase-4 boundary
   = after the R1-B boundary reads + ladder adjudication (§0);
   `molmoact2-ar-head-port` was already closed 08-13; phase-4 shape
   signed off with the v2-wave gate amendment (adopted, phase 4);
   run-provenance question resolved as decision 11.
4. Gate-d-lite pre-registration details owed at launch: the loss
   corridor intermediates from the reference rung's train log, the
   MAE band around 6.76, and the norm-table mechanism (decision 7).

## 8. Out of scope, deliberately

§8.13 step-6 narration (the joint machinery here is its
prerequisite, not its implementation); Flow-GRPO on the flow expert
(fontaine's SDE groundwork stays parked pending token-GRPO results);
surface-A row restriction in `bijou.train` (decision 9); the
eval-instrument sprawl review (separate arc, needs fontaine's ticket
coordination); T3 streams-unification (`docs/` review 2026-08-13 —
sequenced with narration).

## 9. State of the world for the phases-2–3 session (2026-08-14)

**Local repo** (`main` = phases 0–1 landed, check.py green):

- Acceptance fixtures: `tests/fixtures/molmoact2_discrete/`
  (`decode_anchors.npz` + README with the full field spec and bounds)
  and `tests/fixtures/molmo_flow_parity/` (registered cross-machine
  bounds — forward atol 2e-6 / euler 1e-5, the README carries the
  measurement).
- The anchor-read instrument: `probes/probe_molmoact2_anchor_read.py`
  (box-resident inputs; behind gate-d-lite's verdict; phase-3 smokes
  reuse it for ar/joint reads).
- Templates for phase 2: `bijou/decoders/ar_molmo2.py` is the
  closest concrete (244 lines; the new one is simpler — no tables);
  the scaffold contract lives in `bijou/decoders/ar_backbone.py`
  (`ARSuffixDecoder`, `suffix_targets`, the capture surface);
  `MOLMO2_GENERATION_OPENER` is the shared opener; the codec is
  `bijou/fast/molmoact2.py`.
- For phase 3: the dormant `BijouModel.joint_ce` slot (param_groups +
  retain_cache + `CheckpointMetadata.joint_ce`), `--insulate-expert`
  (KI contract test-pinned), and the T2-deleted `_joint_share` /
  three-normalizer chunked-backward composition recoverable verbatim
  from git history at `pre-decoder-simplify` — reinstate for
  flow+CE-rider, not rewrite. `--objective` and `--joint-ce-weight`
  per decisions 4/11; validations in `TrainArgs.__post_init__`.
- Unrelated parallel change to be aware of: rollout gained a
  `--joint-frame` remap + model-frame first-obs gate (`15406e5`) —
  no retirement overlap.

**The box** (fontaine's 1×H100, `ssh -A ubuntu@68.209.75.143`; his
machine — GPU courtesy rules apply. The GRPO ladder is STOPPED, so no
standing GPU reservation exists, but check occupancy and his channel
before long jobs — he schedules his own experiments):

- `~/marius-convert-gate` — our clone (sync before use; it trails
  main by a few commits whenever local work lands).
- `converted/molmoact2_so100_101_release`, `…_rig_r1_step2000`,
  `…_release_rigtable` (release weights + rig table — gate-d-lite's
  init; the `--norm-stats-from` product).
- `outputs/train/gate_d_lite/step_000500|1000|1500|2000` — gate
  evidence; **step_002000 (240-anchor MAE 2.030) is the best rig
  checkpoint produced to date** — candidate for rollout/HF-upload
  before any box cleanup.
- `~/marius-fontaine-ro` — read-only worktree of fontaine's branch
  (`3cac531`), used to generate the discrete fixture; keep until
  phase 2's acceptance gate passes (regeneration needs it).
- Reference banks (fontaine's checkout, read-only):
  `~/flow-matching/reports/analysis__molmoact2_rig_ft_step2000.npz`
  (240 anchors; `rows` are concat indices over clean+v2 @ chunk 30,
  split ALL — the probe verifies identity via the banked states
  before trusting any MAE).
- Rig data: `~/datasets/mcobzarenco/so101_pick_place_{clean,v2}`;
  curated corpus: `~/datasets/mcobzarenco/community_curated_v0`.
- FAST tokenizer snapshot:
  `~/.cache/huggingface/hub/models--allenai--MolmoAct2-FAST-Tokenizer/`
  `snapshots/d45593b4c863d0bc1ca064f8b352fa16b75c38e8`.

**Fontaine**: rebasing onto `3131f82`+ as of 2026-08-14 late (the
fixture-bound fix adopted; his step-(2) replay was zero-conflict, so
expect his branch ≥ our phases-0/1 state with suites green — he posts
results when they finish). The GRPO ladder is adjudicated STOP
(owner-ratified); his queue expects phases 2–3 from our side, then
the phase-4 co-land.

**Suggested phase-2 opening moves**: read §2's decision register +
the phase-2 execution facts; read `ar_molmo2.py` + the discrete
fixture README; build the decoder against the tiny stub-tokenizer
tests first (the `test_molmoact2_encoder` stub carries the id
constants), then run the acceptance gate on the box fixture; wire
loading/schema; only then phase 3.
