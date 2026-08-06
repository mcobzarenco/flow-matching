# 2026-08-06 — Molmo2-4B port plan (#17 rank 2, owner-promoted)

*Status: PLAN (not a pre-registration — no run is registered here;
each launch this plan enables gets its own pre-reg). Owner steering
2026-08-06 12:03Z: "get started on Molmo2-4B in the background too …
quite an involved implementation piece" — the port is promoted to the
CPU queue independent of the architecture-batch verdict; the
both-null branch of the [arch batch
pre-reg](2026-08-06-prereg-arch-batch-1.md) additionally promotes the
first Molmo2 run to next-in-line for multi-GPU time. First
deliverable per the steer: this plan, posted before any code.*

*Primary sources: fetched today from the
[HF repo](https://huggingface.co/allenai/Molmo2-4B) — `config.json`,
`preprocessor_config.json`, `chat_template.jinja`,
`modeling_molmo2.py` — distilled into `docs/molmo2.md` (charter §6
post-cutoff rule; the doc, not model memory, is ground truth).
Bijou-side facts below come from a full code-surface audit run this
session (file:line cited inline). Survey context:
[trunk survey](2026-08-05-trunk-survey.md) (rank 2 of 5).*

## 1. The question the port buys us

**Does a video-grounded VLM trunk (spatio-temporal
pointing/tracking pretraining) beat our text-first Gemma 4 E2B trunk
as a feature source for action prediction — at matched expert, data,
and steps?** The grounding probes located our error in
frame-dependent level mis-estimation and weak use of visual tokens
(#11); Molmo2's pretraining objective (pointing + tracking with
persistent IDs, "how many times does the robot grasp the red
block?") is the nearest open-weights objective to "where is the
gripper and what is it doing".

Framing caveat, banked from the lit slice (ideas #11, VLM4VLA ICLR
26): downstream VLA performance is **uncorrelated with VLM benchmark
rank** across porting studies — so Molmo2's 62.8-vs-58.1 tier win is
NOT the case for the port. The case is structural: video-native
grounding pretraining, clean single-injection architecture (decoder
hidden states are plain token streams — the residual-tap protocol
applies unchanged), and the shared-decoder amortization: Molmo2-4B,
InternVL3.5-4B, and Qwen3-VL-4B share the same Qwen3-4B decoder
geometry (36L / 2560 / GQA 32:8 / head_dim 128), so one decoder port
+ parity harness serves all three (InternVL3.5's `-Pretrained` base
SKU is the idea-#10 vehicle riding the same rails).

## 2. What exists today — the real port surface (audited)

The stage-2 pipeline is narrower than "port a VLM" suggests. Facts
that shape the plan:

- **Only a prefix of the trunk is mounted.** The cross-attention
  decoders mount E2B's non-KV-shared prefix — **layers 0..14, 15 of
  35** (`num_kv_shared_layers=20` ⇒ `first_kv_shared_layer_idx=15`;
  `bijou/gemma4/config.py:112`, `bijou/loading.py:643-648`). The
  expert is 15 layers deep to match.
- **Two stream modes, one contract.** K/V mode exports
  `kv4/kv9/kv14` (the FULL-attention layers of the prefix) as
  zero-copy `[B, kv_heads=1, P, head_dim=512]` views
  (`bijou/encoders/gemma4.py:457-465`); residual mode (arm B,
  landed today) taps raw post-layer hidden states `res0..res14`
  `[B, P, 1536]` and projects them through decoder-side learned
  adapters that mirror `TextAttention.project_kv` op-for-op
  (`bijou/decoders/flow.py:274-299`), producing contract-identical
  streams. Everything trainable lives decoder-side
  (`expert.safetensors`); the trunk stays under a no-grad encode.
- **Flow never touches action tokens.** The flow collator runs with
  `action_codec=None`; FAST ids live in their own space and only
  ar_backbone maps them into trunk vocab (`bijou/fast/codec.py:32`,
  `bijou/train.py:2500-2507`). A flow-first port has **no vocab
  surgery**.
- **The seam is designed but not extracted.** `docs/plan.md:176-200`
  already specifies the `ObservationEncoder` ABC and per-trunk
  `InputsCollator` (it even names a SigLIP trunk as the second
  cell), but `GemmaEncoder` is a concrete class, `BijouModel` is
  Gemma-typed (`bijou/model.py:63-77`), `interface.py:44` imports
  the Gemma `KVCache`, and `PromptKind` has exactly one member
  (`bijou/loading.py:96-101`). The port pays this refactor tax
  first (WP0).

**Design decision D1 — residual-only conditioning for Molmo2.** The
K/V-export path drags Gemma-specific machinery (KVCache, layer
types, `kv_stop_layer`, `project_kv`, KV-sharing depth inference)
that a uniform-attention Qwen3 doesn't have and doesn't need. The
residual path needs only "run N layers, record post-layer hidden
states" — and its adapters *learn* the projection, so the expert
keeps today's exact stream geometry (kv_heads 1 × head_dim 512)
regardless of Qwen3's 8 KV heads. This also sidesteps a live
landmine: `MemoryCrossAttention` computes
`num_key_value_groups = num_heads // stream_kv_heads` with no
validation on the K/V path (`bijou/blocks.py:108`) — 4 cross heads
over 8 trunk kv-heads would silently produce `n_rep=0`. Arm B's
five pre-launch oracles port as the correctness gates.

**Design decision D2 — mount depth 15 of 36.** Fractional depth
15/36 = 0.417 vs E2B's 15/35 = 0.429 — near-identical, so the
expert depth (15), the res0..res14 schedule, and the paired-
comparison story carry over unchanged. Independent support for
early-layers conditioning: SmolVLA (~L/2) and FLOWER (prunes ~50%
of deep layers) — ideas #11 lit slice. A deeper/full-depth mount is
a follow-on arm, not part of the port.

## 3. Work packages

### WP0 — extract the trunk seam (pure refactor, oracle-guarded)

Land `docs/plan.md`'s `ObservationEncoder` ABC
(`stream_geometries` / `inputs_collator` / `encode` /
`param_groups`); move `KVCache` out of `interface.py`; de-Gemma-type
`BijouModel` and the train loop's `CollatedBatch[GemmaInputs]`
signatures; add `PromptKind.MOLMO2` beside `GEMMA4` (the
`DecoderKind` enum is the pattern); decide whether `StreamGeometry`
grows a `scaling` field (Gemma hardcodes attention scaling 1.0
incl. expert cross-attention, `bijou/gemma4/text.py:310`,
`bijou/blocks.py:110`; under D1 the adapters absorb scale, so the
field is optional — decided at impl time, stated in the commit).
Gates: zero behavior change — the three CPU loss oracles bit-exact,
`check.py` green, no state-dict key changes (Gemma checkpoints load
strict before/after).

### WP1 — Qwen3 decoder port (`bijou/molmo2/text.py`)

Pure-torch, config-driven, in the `bijou/gemma4/` style. Under D1
the forward is *simpler* than Gemma's: no PLE (Gemma threads
`per_layer_inputs` through every layer — collapses away), no
sliding/global layer types, no KV sharing, no softcap, no cache, no
`kv_stop_layer` — just embed → N uniform layers with
`residual_taps`/`residual_sink` semantics identical to
`gemma4/text.py:709-711` (tap = post both residual adds). Qwen3
specifics, pinned from fetched files: GQA 32:8 head_dim 128 with
**1/√d scaling** (vs Gemma's 1.0 — implemented inside this trunk,
not shared code), per-head RMSNorm(128) on q,k before RoPE, RoPE
θ=5e6 (`RopeType.DEFAULT`), SwiGLU 9728 (silu — supported in
`bijou/nn.py:84-89`), RMSNorm eps 1e-6 in the `x*w` convention
(matches `bijou/nn.py:104-106`; this is the Gemma-2/3-incompatible
convention that happens to be Qwen3-compatible), **untied**
embeddings, **no** input-embedding scaling, vocab 151,936 + the
128-slot separate `Molmo2Embedding` extension matrix. Flagged:
confirm `rope_scaling_layers` (per-layer dynamic RoPE in
`modeling_molmo2.py`) is unused in the 4B SKU before assuming it
away. Truncated-mount loader keeps only layers 0..14 + embeddings
(analogue of `truncate_backbone_state`,
`bijou/gemma4/loading.py:108-130`, minus the FULL-layer/KV-prefix
constraints that don't exist here). Plus the tiny-checkpoint test
fixture (`write_tiny_checkpoint` analogue, `gemma4/testing.py:139`)
— half the test suite pattern depends on it.

### WP2 — SigLIP tower + connector (`bijou/molmo2/vision.py`)

27L / 1152 / patch 14 / 378² → 729 patches; taps at `vit_layers
[-3, -9]` **concatenated** (→2304); 2×2 attention pooling
(mean-of-group query); gated `ImageProjectorMLP` → 2560; features
**added at `image_patch` (id 151938) placeholder positions** in the
layer-0 embedding sequence (Molmo2 `+=` at placeholder ids — same
job as Gemma's `masked_scatter`, different mechanics; note Gemma's
placeholder ids sit *outside* its embedding vocab and get
pad-substituted, `bijou/gemma4/model.py:181-187` — Qwen's are real
ids, so that workaround must NOT be copied). Vision-block
bidirectional attention via the token-type mask (image↔image
unrestricted, text causal) — built into the port's mask
construction from the start. Input plumbing changes shape:
Gemma-4's encoder-free tower takes raw patch rows +
`image_position_ids`; SigLIP takes `[N,3,H,W]` crops — a new
`Molmo2Inputs` payload (the `SigLip2Inputs` sketch at
`docs/plan.md:160-175` is the template).

### WP3 — Processor / prompt assembly (`Molmo2InputsCollator`)

- Crops: `max_crops 8` @378², overlap margins [4,4], 2×2 pooling ⇒
  ~196 tokens per view. **Operating point: global view only (~196
  tokens/image), crops off** — 480p sources make high crop counts
  interpolation-heavy (the exact pixel math that moved arm A
  560→280, Amendment 2); the arch-batch arm A read informs any
  later crops rung. Token budget is a different dial than Gemma's
  `max_soft_tokens` {70,…,1120} — the collator maps our budget flag
  onto (global, +crops) explicitly rather than pretending the dials
  are the same (`max_soft_tokens` is recorded in checkpoints and
  drives length bucketing, `bijou/train.py:1326-1347` — the
  Molmo2 prompt config records its own crops field instead).
- Prompt: ChatML (`<|im_start|>user`…`<|im_end|>`) with `<|image|>`
  placeholders replaces Gemma's `<start_of_turn>`; the format-3
  semantic content (camera-kind tags, condition brackets,
  `[generate|…]`, trailing soft state token) is re-rendered in the
  new template. Three collator mechanisms re-derived and re-proved
  for the Qwen tokenizer: the turn-close probe (Gemma asserts a
  1–4-token close tail, `bijou/encoders/gemma4.py:176-197`), the
  state-slot splice just inside the close (`:264-279`,
  zero-init `state_proj` so the prompt starts undisturbed), and
  left-padding (load-bearing for Gemma's windowed masks; harmless
  but kept for uniformity). `PROMPT_FORMAT` is namespaced per
  trunk, not bumped (a new trunk's prompt is a different format
  space, `bijou/loading.py:169`).
- Tokenizer: Qwen2-family BPE from the checkpoint's own
  `tokenizer.json` — same "the checkpoint carries its tokenizer"
  rule as today; no Gemma byte constants
  (`camera_tag_text`, `GENERATION_OPENER`) may leak.

### WP4 — Stream export + expert wiring

`residual_expert_config` already derives `schedule =
res0..res{N-1}` from the mounted depth (`bijou/loading.py:462-511`)
— with D2's 15-layer mount this is res0..res14, expert depth 15,
**unchanged**. Adapters: one `ResidualStreamAdapter` per tap
projecting hidden 2560 (vs 1536) to the same kv_heads 1 ×
head_dim 512 streams, RoPE'd at logical positions — param count
≈ 23.62M × (2560/1536) ≈ **39M** (pre-reg-time exact count rule
applies). Cross-attention query positions, suffix positions, and
padding-mask semantics all live decoder-side and carry over
(`bijou/decoders/flow.py:557-563`, `:648-660`). One inherited
subtlety made explicit: the expert config *inherits the trunk's
activation and eps* (`bijou/loading.py:493-494`) — silu/1e-6 for
both trunks, so the expert architecture is unchanged, but the
equality is asserted, not assumed.

### WP5 — Schema, loaders, audit

`Molmo2PromptConfig` under `PromptKind.MOLMO2` (records crops
budget, exports, format, state_dim, condition fields);
`expert_config_from_architecture` consumes encoder-declared
geometry instead of reaching into `Gemma4Config.text.*`
(`bijou/loading.py:883-945` — the single biggest trunk-coupling
point on the load side); `backbone: {id, depth}` recording loses
its KV-sharing-specific depth inference (`bijou/train.py:1114-1119`);
`--backbone-init-from` / snapshot / `from_checkpoint` paths get
Molmo2 arms. Trainable-set audit (`_trainable_text_parameters`
analogue): under D1+frozen-trunk phase 1 the set is empty trunk-side
— but the surface is built correctly anyway because the AR-adaptation
phase (§6) will need it. Final audit pass over the report's
hardcoded-assumptions list (SDPA head-dim workaround comment,
`--offload-ple` N/A, audio-tower key skips, etc.).

**Explicit non-goal (phase 1): no AR-family port.** FAST + aux-text
would need Qwen-side vocab anchoring (Gemma tail-anchors FAST in a
3,259-id unused run; Qwen's layout differs and the 128-slot
extension matrix is too small) plus an AR adaptation run before a
flow stage-2 on an adapted trunk. Phase 1 trains the flow expert on
the **raw frozen Molmo2 prefix** — the stage-2 §8.11
controlled-phase baseline protocol, so the paired raw-trunk
comparison exists. The confound — our best lineage rides an
AR-*adapted* trunk (−2.7 MAE from adaptation), the Molmo2 arm won't
— **ships with any claim**, and the phase-1 comparison is declared
vs the matched raw-Gemma-trunk baseline, not vs the headline
lineage. If the raw read is promising, AR adaptation becomes its
own plan.

## 4. Parity & correctness harness (gates before any pre-reg)

Same discipline as `bijou/gemma4/verify_parity` + arm B's five
oracles:

1. **Weights parity**: HF fp32 shards → bijou state dict (mapping
   table committed); per-layer activation parity vs the
   `trust_remote_code` HF reference on fixed image+text inputs
   (tolerances stated per dtype), then greedy-decode agreement on a
   few VQA-style prompts through the full 36-layer stack (the
   truncated mount can't greedy-decode; parity runs full-depth,
   the mount is a separate strict-load test). Reference runs on an
   idle GPU at a run boundary — never under a live run.
2. **Stream contract** (arm B's gates, re-proved on the new trunk):
   tap semantics + padding-orientation invariance; trunk
   bitwise-frozen through a real optimizer step; grads reach every
   adapter param; checkpoint round-trip strict; Gemma paths
   untouched (no Molmo2 keys in Gemma state dicts and vice versa).
3. **CPU loss oracles**: the three banked oracles stay bit-exact on
   Gemma paths after WP0 and after every WP — the port must be
   purely additive; any legitimately moved anchor is re-baselined
   loudly per charter.
4. **Tiny-checkpoint fixture** so the whole suite runs CPU-only in
   CI (`check.py`), matching how the Gemma tests work.

## 5. Memory & cost budget (estimates, labeled as such)

- **Mounted trunk**: embeddings (151,936+128)×2560 ≈ 0.39B +
  15 × ~101M/layer ≈ 1.5B + SigLIP+connector ≈ 0.4B ⇒ **~2.3B
  params ≈ 4.7 GiB bf16**, frozen (no grads/Adam) — comparable to
  today's footprint, NOT the naive 9.7 GiB full-model number.
  Download 19.4 GB fp32 once → cast to bf16, cache the truncated
  mount.
- **Expert + optimizer**: unchanged (h1024 expert; adapters ~39M vs
  23.6M — noise). SnapFlow's whole stage-2 footprint was 22.4 GiB.
- **Prefix encode**: per-layer cost ~2.6–2.8× E2B's prefix layers
  (hidden 2560 vs 1536, MLP 9728 vs 6144); token counts comparable
  at the global-view operating point (196/image vs 280 soft
  tokens). No-grad encode; **B32 expected to hold with large
  margin on 80 GB — an estimate, not a budget**: the pre-launch
  memory smoke measures it (E4B scar: a "fits easily" prior OOM'd
  four rungs deep, so this line is a gate, not a formality).
- **Throughput**: prefix encode dominates (79% of step time
  measured on Gemma); expect ~0.6–1.0 s/step at matched batch —
  measured at first poll, starvation fixed before the run is left
  alone.
- **CPU-side cost**: WP0 ~1 session; WP1+parity ~1–2; WP2–WP3 ~1–2;
  WP4–WP5 ~1. **≈ 4–6 focused work sessions**, GPU only for parity
  bursts and the smoke. Fits the GPU-busy windows while arm A/B
  run (charter no-idle-pauses), which is exactly why the owner
  promoted it as background work.

## 6. Sequence & gates

1. ~~Plan post + `docs/molmo2.md` distilled doc~~ — DONE this
   session.
2. ~~WP0 seam refactor (oracle-guarded, zero behavior change)~~ —
   DONE 2026-08-06 (`7409df0`): `ObservationEncoder[I, B]` ABC at the
   seam, `KVCache` opaque there, `BijouModel`/train loop de-Gemma-typed,
   `PromptKind.MOLMO2` reserved (refuses to load until WP4). Decided at
   impl time: `StreamGeometry` grows NO `scaling` field — under D1 the
   adapters absorb scale; it would be dead config with one legal value.
   Gates held: check.py 294 green, loss oracles bit-exact, no
   state-dict key changes.
3. WP1 decoder port + weights parity.
4. WP2–WP3 vision/processor + end-to-end encode parity.
5. WP4–WP5 + the §4 suite green + `check.py` green.
6. **Then and only then**: pre-registration of the first run — flow
   stage-2 screen on the frozen raw Molmo2 15-layer mount,
   panel-v2/stable keying, vs a matched **raw-Gemma-prefix
   baseline arm** (whether the §8.11 controlled-phase artifact is
   directly usable on panel-v2 or a cheap control run is needed is
   decided in that pre-reg, not here). Multi-GPU slot per the
   arch-batch both-null promotion rule, else scheduled behind the
   batch. The pre-reg carries a real kill line so a null banks
   cleanly (VLM4VLA says nulls are the modal outcome for
   trunk-quality bets — a clean null here is transferable
   knowledge, charter §0).

Park criterion for the port itself (CPU work, low bar): park and
say so in ideas #17 if the parity harness surfaces a blocker >1
session deep (e.g. `rope_scaling_layers` turns out load-bearing and
gnarly).

## 7. Risks & flagged unknowns

- **`rope_scaling_layers`** (per-layer dynamic RoPE in the remote
  code) — confirm unused in the 4B SKU at WP1, first thing.
- **SigLIP-1 vs "SigLIP 2"** card/metadata discrepancy — inert for
  the port (weights ship in the repo); resolved in `docs/molmo2.md`
  when the paper is deep-read.
- **No base checkpoint** — instruct trunk only; base-vs-IT is
  InternVL3.5's job (idea #10) on the same decoder port.
- **Instruct-trunk prompt sensitivity** — our format-3 prompt is
  nothing like Molmo2's training distribution; nor was it like
  Gemma's, and stage-2 worked. Noted, not blocking; the screen
  measures it.
- **Research-use data caveat** — inert for us, flagged for any
  mainline adoption story.
- **WP0 regression risk** — the refactor touches the live Gemma
  path; it lands alone, oracle-guarded, never in the same commit
  as Molmo2 functionality.
