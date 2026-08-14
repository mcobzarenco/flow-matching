# Retiring `bijou/molmoact2/` — first-class MolmoAct2 AR + joint objectives

Status: **PLAN, pre-implementation** (owner-approved direction,
2026-08-14). Implemented phase by phase on `main`; each phase lands
with its gates green and this document's status lines updated. When
complete, the design record subsumes into `docs/architecture.md`
(§8.13 gets its step-8 closure; §2 gains the new decoder and the
objective matrix) and this file becomes the migration's historical
record. Anchor tags: `pre-decoder-simplify` (T1/T2 deletions,
2026-08-13) and `pre-molmoact2-retirement` (created in phase 0 — the
last commit where the port package exists).

---

## 0. For Fontaine — how to adopt these changes

You are the primary downstream consumer: your token-GRPO line builds
partly on `bijou/molmoact2/` (the package this plan deletes). The
short version: **your objective module and your artifacts survive
untouched; your imports move; one of your queue items is superseded.**

**Ordering (do these at run boundaries, never under a live run):**

1. **Rebase onto `main` ≥ `db0a141` first** (T1 retired `ar_fast`, T2
   retired residual conditioning + `--conditioning-streams` /
   `--seam-stop-grad` / `--joint-ce` wiring). Your conflict surface is
   small and disjoint from the deletions: `flow.py` (your
   `sample_actions_sde` vs our removals — different regions),
   `model.py` (your `action_capture` kwarg), `eval/policies.py` (your
   `TokenRow`/`stable_sde_step_noise` additions vs our `tile_memory`
   edit), `train.py` (3 lines). Expect trivial resolutions. Note your
   flow-SDE groundwork is unaffected by T2 (it has no residual
   dependence).
2. Adopt phases 1–3 whenever convenient after they land — they do not
   touch your live surfaces.
3. **Phase 4 is your instrument** (GRPO rollout/replay re-pointing) —
   it is co-landed with you at a boundary you pick, gated on a
   frozen-wave replay parity check (below).
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
by phase 2** of this plan — we are building the first-class discrete
AR head on main. Hold/close that item to avoid duplicate work; review
phase 2's design below instead (it deliberately adopts your
`fast_codec` audit facts and your grammar-mask/budget-arithmetic
contract verbatim).

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
   bins** (symbol_lengths 0 elsewhere — the grammar mask excludes them
   for free) and **7 quantization-hole symbols** (their pipeline
   silently zero-falls-back; ours raises). The `pad` analog for the
   discrete surface is pinned during phase 2 (open detail, flagged).
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
   norm-table sub-decision (their recipe RECOMPUTES the q01/q99
   table on rig data at fine-tune start; our converted release
   carries the community table) is settled at this run's launch — a
   small `--norm-table` recompute knob or a convert-time alternate,
   chosen in the pre-registration; reproducing the rung requires the
   rig table either way.
8. **Sequencing:** nothing lands under a live fontaine run; phase 4
   co-landed with him; phase 5 only after his phase-4 sign-off.
9. **Surface-A row restriction** (training only the action-block rows;
   depends on wd=0 grad-row masking) stays a GRPO-loop lever — not a
   `bijou.train` flag.
10. **Formats frozen:** GRPO row NPZ and loop `.pt` checkpoints
    unchanged.

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

**Phase 0 — freeze the reference.** Tag `pre-molmoact2-retirement`.
Generate + commit golden fixtures by executing the port: (a) tiny-pair
expert forward + Euler loop tensors (replaces `test_molmo_flow`'s
live-pair parity half), (b) `predict_action_discrete` decode bytes +
per-step logprobs on the banked anchor rows (phase 2's acceptance
gate), (c) keep the existing prompt-assembly byte fixtures (they pin
our collator, not the port). GPU window: gate-d-lite per decision 7
(500-step prefix, opportunistic continuation).

**Phase 1 — promote the leaves** (pure moves, CPU): the §3 table's
"move" rows. `MolmoFlowConfig.released_so100_101` becomes the
literals' single home; the port-mirror sync test retires as its own
comment pre-planned. Gate: `check.py` green; 2-step oracles bitwise
(the T1/T2 protocol).

**Phase 2 — `bijou/decoders/ar_molmoact2.py`.** The concrete per
decisions 1–3: trunk-native `_suffix_hidden` (plain `wte` lookup — no
extension-table select), `_logits` = trunk `lm_head` (full-id-space,
grammar legality handled by the mask, not column surgery),
construction guards (block inside base matrices; codec/geometry
anchors). Loading/schema arms; `BijouPolicy` works via the existing
`ARSuffixDecoder` genericity. Gates: decode bit-equal to phase-0
fixtures; replay logprobs within the registered bounds (1e-5
one-shot-vs-incremental + the JPEG budget); `train_grpo`'s
unchanged-policy oracle (ratio 1, clip 0, k3 0) on the new surface.

**Phase 3 — objective matrix.** `--objective` + validations; the
joint arm in `BijouTrainStep`/`BijouModel` (three-normalizer chunked
backward; per decision 4 weights); the decision-5 ordering invariant
test (expert KV extracted before CE suffix append; insulated ⇒ trunk
grads from the flow term exactly zero while CE grads flow — extend
the existing KI contract test to the joint composition); schema +
save/load round-trip tests for all three compositions. Gates: 2-step
corridor for `ar` and `joint` on the tiny fixture recorded as new
anchors; flow-only bitwise unchanged.

**Phase 4 — GRPO migration (co-landed with fontaine).** Replay
builder rewritten thin on `MolmoAct2InputsCollator` +
`suffix_targets` + generic capture (row NPZ unchanged);
`sim/grpo_loop.py` + sim rollout driver re-pointed from
`MolmoAct2Predictor` to `BijouPolicy` + `MolmoAct2ARDecoder`. Gate:
one frozen-seed sim wave replayed old-vs-new — episode rewards equal,
per-token logprobs within the registered bounds.

**Phase 5 — delete `bijou/molmoact2/`.** Remaining modules + their
tests go; loud refusals are NOT needed at the checkpoint layer (no
checkpoint records a "port" — converted artifacts already speak the
first-class schema), so the deletion is code-only; docs updated
(§8.13 step 8 closed; §2 decoder census; this file's status);
oracle re-runs bitwise for all surviving paths.

Rough sizes: P1 ±0 net (moves), P2 ~250–350 + tests, P3 ~150 restored
+ schema, P4 net-negative (replay −340 + loop re-pointing), P5 ≈
−2,500.

## 6. Gate matrix (what proves each phase)

| gate | phase | instrument |
|---|---|---|
| fixtures reproduce port bytes | 0 | executing the port at the tag |
| leaf moves are inert | 1 | check.py + bitwise 2-step oracles (flow, ar_backbone) |
| discrete decode parity | 2 | phase-0 fixture (b); banked-anchor rows |
| replay ratio contract | 2/4 | unchanged-policy ratio≈1 oracle; 1e-5 + JPEG bounds |
| KI both ways under joint | 3 | extended gradient-contract test |
| flow-only untouched | 3 | bitwise vs pre-phase oracles + molmo_flow fixture tests |
| old-vs-new GRPO stack | 4 | frozen-seed wave replay (rewards equal, logprobs in-bound) |
| rig-rung class reproduction | 0 | gate-d-lite: step-500 MAE in the 6.76 class + loss-corridor match; 2000-step endpoint read free when the window holds (decision 7) |

## 7. Open items before the relevant phases

1. ~~Decision 4's joint weight~~ — RESOLVED 2026-08-14:
   `--joint-ce-weight`, default 1.0 (decision 4).
2. ~~Decision 7: gate-d~~ — RESOLVED 2026-08-14: gate-d-lite, the
   500-step prefix with opportunistic 2000-step continuation
   (decision 7); needs a GPU window (owner arranging).
3. Fontaine's sign-offs: the phase-4 boundary, and holding his
   `molmoact2-ar-head-port` queue item (superseded).
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
