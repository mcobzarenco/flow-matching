# Plan: modular observation encoders × action decoders

> **Historical record** (2026-07-30 refactor plan; superseded by architecture.md §2/§5). The seam shipped as planned; since then: AR option C shipped as `ar_backbone` (the doc rules it out for the original seam — the seam grew instead), checkpoint format is 3, four decoder classes exist (incl. `Molmo2ARDecoder`), and the SigLIP2 encoder cell was never built (SigLIP arrived inside the Molmo2 trunk instead).

Status: **steps 1–4 IMPLEMENTED** (2026-07-30) — the refactor landed
without new encoder/decoder types: `bijou/nn.py` + `bijou/interface.py`
(seam types, ABCs, shared Collator), format-2 tagged checkpoint schema
with the permanent format-1 synthesizer, the collator split
(`encoders/gemma4.py` strategy + quantile-carrying NormStats), and the
GemmaEncoder/FlowDecoder composition under BijouModel. Gates held at
every step: CPU oracle EXACT 1.8896/1.7237, state_dict key-set fixtures,
grad-flow probe (flags-on oracle 1.5528), predict_chunk bitwise-equal to
the old policy path, real format-1 checkpoint loads. Remaining from this
doc: the new matrix cells (SigLIP2 encoder, AR-FAST decoder) and their
evidence ladders (§6 step 5).

## Brief

Support different architectural combinations of "VLM trunk" (observation
encoder) and action decoder, with existing checkpoints deserializing
without weight conversion (a one-off config conversion is acceptable if it
leaves the code cleaner — the design below needs none). Concretely, at
least:

- observation encoders: the current **Gemma4 E2B trunk**, and a **SigLIP2**
  version (camera frames + instruction text);
- action decoders: the current **flow-matching expert**, and an
  **autoregressive FAST token decoder**.

## 0. Settled decisions

- `ObservationMemory.streams` is a **name-keyed mapping** (`dict[str,
  MemoryStream]`, insertion-ordered as the encoder's exports); the
  decoder's **schedule references stream names** (`("kv4", "kv4", ...,
  "kv14")`). A positional variant (tuple + integer indices) was
  considered and dropped: a checkpoint whose schedule reads
  `["kv4", ...]` documents itself, while `[0, 0, ...]` requires
  cross-referencing the encoder's exports. The drift risk of string
  references is closed at composition time (schedule names must be a
  subset of exports; every export consumed; unknown name = loud error).
  Names are defined by each encoder (gemma: `"kv{layer}"`; siglip2:
  stage names) — decoder configs still never contain trunk internals.
- The cross-attention **schedule stays decoder-side**: its length IS
  the decoder depth, and future co-training (two decoders, one prefix)
  needs per-decoder schedules over shared exports.
- AR option C (full-VLM KV-shared deep half, `architecture.md` §8.3) does
  **not** fit this seam — it is the trunk, not a decoder attached to it.
  The matrix decoder is option B (fresh small AR decoder). C returns, if
  ever, as a gemma4-specific fused model on B-disappointing evidence.
- Per-stream static geometry (kv_heads, head_dim, RoPE-or-not) lives in
  the interface as `StreamGeometry`, derived by the encoder; the decoder
  sizes its projections from it and never learns what trunk it faces.
- `nn.py` primitives lift happens as part of this refactor (RMSNorm,
  attention dispatch, rope helpers move out of `gemma4/layers.py`, which
  re-imports them; parity tests untouched). Without it, `decoders/` would
  import `gemma4/` for generic math — the exact accidental coupling this
  plan exists to remove.
- New checkpoints write the new schema only; old checkpoints are read via
  a permanent ~30-line legacy synthesizer (no dual-writing, no file
  conversion).

## 1. The interface (`bijou/interface.py`)

```python
@dataclass(frozen=True, slots=True)
class StreamGeometry:
    """Static per-stream contract, known at construction time."""

    kv_heads: int
    head_dim: int
    rope: RopeParameters | None  # None => positions are baked into the
    # memory; decoder applies no query RoPE.
    # Set   => decoder RoPEs queries at
    # positions >= per-sample real prefix len
    # (the Gemma streams' contract today).


@dataclass(frozen=True, slots=True)
class MemoryStream:
    key: Tensor  # [B, kv_heads, P, head_dim]
    value: Tensor  # [B, kv_heads, P, head_dim]


@dataclass(frozen=True, slots=True)
class ObservationMemory:
    """The value crossing the encoder->decoder seam. Keys are the
    encoder's stream names, insertion-ordered as its `exports`."""

    streams: dict[str, MemoryStream]
    length: int  # padded P
    padding_mask: Tensor | None  # [B, P], True = real


@dataclass(frozen=True, slots=True)
class NormStats:
    """One modality's normalization stats, per sample: each tensor
    [B, dim] (dim = action_dim or state_dim). Every sample carries its
    OWN dataset's stats — per-dataset normalization; nothing here is
    aggregated across the batch.

    q01/q99 are None only when the stats were resolved from a checkpoint
    whose per_dataset_normalization predates quantiles (old-checkpoint
    rollout); batches built from datasets always carry them (selection
    requires backfilled stats). Consumers that need quantiles (AR/FAST)
    check for None and fail fast with the remedy; flow paths never read
    them."""

    mean: Tensor
    std: Tensor
    q01: Tensor | None
    q99: Tensor | None


@dataclass(frozen=True, slots=True)
class CollatedBatch(Generic[I]):
    """Trunk-agnostic core + typed encoder-specific inputs. pin_memory/to
    recurse into `encoder_inputs` and the NormStats fields (all
    implement the same two hooks)."""

    encoder_inputs: I  # GemmaInputs | SigLip2Inputs
    state: Tensor  # [B, state_dim]
    actions: Tensor  # [B, chunk, action_dim]
    action_is_pad: Tensor  # [B, chunk]
    action_stats: NormStats  # each [B, action_dim]
    state_stats: NormStats  # each [B, state_dim]
    # AR-only, filled by the collator when built with a FastTokenizer
    # (CPU-side in workers); None otherwise. The AR loss asserts loudly.
    # (Cannot be made mandatory like the quantiles: tokens depend on a
    # tokenizer artifact and cost real CPU per item — they are computed
    # only when an AR decoder will consume them.)
    # No separate mask field: with right padding + causal attention, PAD
    # is invisible to real positions, and the CE exclusion derives from
    # `action_tokens != PAD_ID` (targets built with ignore_index). PAD is
    # a reserved special, never a real token.
    action_tokens: Tensor | None  # [B, T_tok] PAD-padded token ids
```

### Encoder inputs in practice

```python
@dataclass(frozen=True, slots=True)
class GemmaInputs:
    """Today's prefix fields, verbatim — produced by the chat-template
    collator. The prompt layout (which tokens are images, where padding
    sits, P itself) is DECIDED AT COLLATE TIME and carried by input_ids;
    encode() just runs it."""

    input_ids: Tensor  # [B, P]
    attention_mask: Tensor  # [B, P]  (1 = real, 0 = right padding)
    pixel_values: Tensor  # [images, patches, 3·patch_size²]
    image_position_ids: Tensor  # [images, patches, 2]
    has_padding: bool  # CPU-side, avoids a device sync


@dataclass(frozen=True, slots=True)
class SigLip2Inputs:
    """Two towers, two preprocessors — and NO unified token sequence:
    there is no P at collate time. The encoder assembles the memory
    ([text tokens][cam_1 soft tokens][cam_2 ...]) at ENCODE time and
    derives P and the ObservationMemory padding mask itself. Gemma encodes
    the image->sample layout inside input_ids (placeholder tokens);
    SigLIP2 has no such carrier, so the mapping travels explicitly."""

    # vision tower (NaFlex: native aspect, ragged patch counts)
    pixel_values: Tensor  # [images, patches, 3·patch_size²]
    pixel_attention_mask: Tensor  # [images, patches]  (True = real patch)
    spatial_shapes: Tensor  # [images, 2]  ((h, w) in patches, for 2D pos)
    sample_index: Tensor  # [images]  which batch sample owns each image
    camera_slot: Tensor  # [images]  positional camera slot (sorted names)
    # text tower (SigLIP tokenizer, 64-token max)
    text_input_ids: Tensor  # [B, T_text]
    text_attention_mask: Tensor  # [B, T_text]
```

(Fixed-resolution SigLIP2 variants would carry `pixel_values`
[images, 3, H, W] with implicit patch geometry — the NaFlex form is
shown because 640×480-native is the point of choosing it.)

### The ABCs

```python
class ObservationEncoder(nn.Module, Generic[I]):
    """ABC. A trunk: inputs-collation strategy + encode + unfreeze surface."""

    def stream_geometries(self) -> dict[str, StreamGeometry]: ...  # keys =
    #   stream names, same order/keys as every ObservationMemory it produces
    def inputs_collator(self) -> InputsCollator[I]: ...
    def encode(self, inputs: I, *, with_grad: bool) -> ObservationMemory: ...
    def param_groups(self) -> dict[str, list[Parameter]]: ...

    # e.g. {"text": [...], "vision": [...]} — the --text-lr/--vision-lr
    # flags route here for ANY encoder.


class ActionDecoder(nn.Module):
    """ABC. Owns its objective and its chunk-space inference."""

    def action_tokenizer(self) -> FastTokenizer | None: ...  # AR: its artifact
    def loss(self, memory: ObservationMemory, batch: CollatedBatch[Any]) -> Tensor: ...
    def predict_chunk(
        self,
        memory: ObservationMemory,
        batch: CollatedBatch[Any],
        *,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor: ...

    # Returns RAW-unit [B, chunk, action_dim]; normalize/denormalize via
    # the batch's per-item stats happens INSIDE (this logic moves out of
    # BijouPolicy). Decoder-specific inference knobs (Heun steps, decode
    # temperature) are constructor parameters of the decoder/policy, not
    # ABC surface.
```

### Collation: one shared core, per-encoder strategies

There is exactly ONE `Collator` class — the trunk-agnostic core: stacks
state/actions/action_is_pad, attaches per-sample NormStats, tokenizes AR
targets in the workers when a tokenizer is present, owns camera-selection
policy (sorted keys, --camera filter, max_cameras), the instruction
override, and the worker rules (pickleable, lazy processor construction,
CPU-side has_padding decisions). Duplicating any of this per encoder is a
drift hazard (cf. the repeat-last-actions decision — one implementation
site, not N). Only the `encoder_inputs` production varies:

```python
@dataclass(frozen=True, slots=True)
class CameraFrame:
    """One camera's frame with its slot name (post-filter, sorted — e.g.
    "front", "wrist"; community datasets carry generic image/image2 names
    with no reliable semantics, rig datasets carry real ones). Encoders
    MAY render the name into the prompt ("front: <image> wrist: <image>")
    or ignore it (today's positional behavior)."""

    name: str
    image: Tensor  # [3, height, width], float, [0, 1]


@dataclass(frozen=True, slots=True)
class PromptInputs:
    """One sample's prompt-side payload, assembled by the shared Collator
    (instruction override + camera policy applied). Extension point for
    future prompt-side signals — e.g. π0-FAST-style discretized state as
    text would become an optional field here."""

    instruction: str
    cameras: tuple[CameraFrame, ...]


class InputsCollator(Protocol[I]):
    """Encoder-specific: a batch of PromptInputs -> I.
    Same pickling rules as today's PrefixCollator (lazy HF processor,
    __getstate__ drops the built one)."""

    def __call__(self, samples: list[PromptInputs]) -> I: ...


@dataclass
class Collator(Generic[I]):  # lives in interface.py (encoders sit
    inputs: InputsCollator[I]  # below data.py in the DAG, so the
    action_tokenizer: FastTokenizer | None  # shared core cannot)
    instruction: str | None
    camera_filter: tuple[str, ...] | None
    max_cameras: int | None

    def __call__(self, items: list[dict[str, Any]]) -> CollatedBatch[I]: ...
```

The composition root assembles it from both sides:
`Collator(inputs=encoder.inputs_collator(), action_tokenizer=
decoder.action_tokenizer(), ...)` — the encoder never learns about
tokenizers, the decoder never learns about pixels.

## 2. Configs (tagged unions, parse at the edge)

```python
class EncoderKind(StrEnum):
    GEMMA4 = "gemma4"
    SIGLIP2 = "siglip2"


class DecoderKind(StrEnum):
    FLOW = "flow"
    AR_FAST = "ar_fast"


@dataclass(frozen=True, slots=True)
class GemmaEncoderConfig:
    backbone: str  # HF id / local dir
    exports: tuple[int, ...]  # gemma layer indices, e.g. (4, 9, 14)
    # (trunk internals belong HERE — this is
    # the one config allowed to know them)
    max_soft_tokens: int


@dataclass(frozen=True, slots=True)
class SigLip2EncoderConfig:
    checkpoint: str  # e.g. google/siglip2-so400m-patch14-384
    exports: tuple[SigLip2Stage, ...]  # enum: TOP, MID, POOLED_TEXT, ...
    soft_token_budget: int  # NaFlex budget / pooling knob
    text_max_tokens: int


@dataclass(frozen=True, slots=True)
class FlowDecoderConfig:
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    hidden_activation: str
    rms_norm_eps: float
    self_attention_mode: SelfAttentionMode
    self_attention_rope_theta: float
    cross_attention_heads: int
    schedule: tuple[str, ...]  # stream NAMES; len = decoder depth
    action_dim: int
    state_dim: int
    chunk_size: int
    time_embed_dim: int
    time_conditioning: TimeConditioning
    # GONE vs today's ExpertConfig: cross_attention_head_dim,
    # cross_attention_rope — both now per-stream StreamGeometry.


@dataclass(frozen=True, slots=True)
class ARFastDecoderConfig:
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    rms_norm_eps: float
    cross_attention_heads: int
    schedule: tuple[str, ...]  # its own schedule over the same names
    tokenizer: str  # artifact ref, e.g. ".../fast_tokenizer_v1"
    vocab_total: int  # BPE vocab + BOA/PAD specials
    state_dim: int
    chunk_size: int
    action_dim: int
```

Checkpoint json (format 2):

```json
{
  "format": 2,
  "encoder": {"kind": "gemma4", "backbone": "google/gemma-4-e2b-it",
               "exports": [4, 9, 14], "max_soft_tokens": 140},
  "decoder": {"kind": "flow", "hidden_size": 1536, "...": "...",
               "schedule": ["kv4","kv4","kv4","kv4","kv9","kv9","kv9","kv9",
                            "kv14","kv14","kv14","kv14","kv14","kv14","kv14","kv14"],
               "time_conditioning": "adarms"},
  "step": 100000,
  "train_args": {"...": "unchanged, still recorded verbatim"},
  "normalization": {"...": ""},
  "per_dataset_normalization": {"...": ""}
}
```

Weight files: format 2 writes `encoder.safetensors` (adapted parts only,
when trained) + `decoder.safetensors`. Legacy files keep their names and
map to the same slots (`backbone.safetensors` -> encoder-adapted,
`expert.safetensors` -> flow decoder).

## 3. Legacy path (permanent, ~30 lines, no conversion)

Detection: `"encoder" not in meta` ⇒ legacy. Synthesis is a pure function
of fields every existing checkpoint already has:

```python
def legacy_configs(meta: dict) -> tuple[GemmaEncoderConfig, FlowDecoderConfig]:
    args = CheckpointTrainArgs.from_dict(meta["train_args"])
    streams = streams_from_counts(args.stream_counts)        # -> (4, 9, 14)
    schedule = tuple(
        f"kv{layer}"
        for layer, count in zip(streams, args.stream_counts, strict=True)
        for _ in range(count)
    )                                                        # ("kv4", ..., "kv14")
    encoder = GemmaEncoderConfig(
        backbone=meta["backbone"],
        exports=streams,
        max_soft_tokens=args.max_soft_tokens,
    )
    decoder = FlowDecoderConfig(..., schedule=schedule, ...)  # from train_args,
    return encoder, decoder                # exactly expert_config_from_train_args
```

`--init-from` an old checkpoint inside a new-format run compares
*synthesized* configs (the existing setdefault-backfill pattern
generalizes to: synthesize both sides, then diff).

## 4. Composition & construction order

```python
# loading.from_checkpoint / from_configs:
encoder = build_encoder(encoder_config, device=..., dtype=...)  # match on kind
geometries = encoder.stream_geometries()
validate_schedule(decoder_config.schedule, geometries)
#   - every schedule name exists in geometries (unknown name = loud error)
#   - every export consumed (unused export = config error, loud)
#   - len(schedule) == decoder depth (definitionally)
decoder = build_decoder(decoder_config, geometries, device=..., dtype=...)
model = BijouModel(encoder, decoder)
```

`BijouModel` shrinks to the composition root: `encode(batch)`,
`loss(batch)`, `predict_chunk(batch, ...)` — each one delegation plus the
frozen/live-trunk autocast policy (`BijouTrainStep` generalizes: encode
`with_grad` under bf16 autocast, decoder loss outside it, one DDP wrapper).

## 5. Module deltas (what actually moves)

- **`bijou/nn.py` (new)**: RMSNorm, attention dispatch (eager/SDPA),
  rotate_half/apply_rotary_pos_emb, rope_cos_sin, rope_inv_freq_from_params,
  MaskSpec. `gemma4/layers.py` re-imports (aliases) — gemma4 parity tests
  and checkpoint keys untouched.
- **`bijou/decoders/flow.py`**: today's `expert.py`, class renamed
  `FlowDecoder`, **attribute names frozen** (`layers`, `norm`,
  `action_out_proj`, `state_proj`, `time_in_proj`, `time_out_proj`,
  `cross_inv_freq`, `self_inv_freq`, per-layer names) so safetensors keys
  are byte-identical. Deltas: consumes `ObservationMemory` (per-layer lookup
  by schedule stream name, "kv4"/"kv9"/"kv14" under legacy configs);
  per-stream `StreamGeometry` sizes each layer's q-projection (legacy:
  all streams 512/1 — identical shapes); RoPE applied per stream geometry (None => skip); gains
  `loss()` (= `flow_matching_loss` moved in) and `predict_chunk()`
  (= `sample_actions` + the normalize/denormalize now in BijouPolicy).
  GATE: state_dict key-set equality vs today's ActionExpert under a
  legacy config, incl. buffer persistence.
- **`bijou/decoders/ar_fast.py` (new)**: token embedding (vocab +
  BOA/EOA/pad), state anchor token, N sandwich blocks (self-attn causal
  over `[state][BOA][t_1..t_k]`, cross-attn per its schedule), LM head.
  `loss` = CE over shifted `action_tokens` with ignore_index at PAD
  (mask derives from the reserved PAD id — no separate mask tensor;
  attention needs none: right padding + causality already hides PAD
  from real positions); `predict_chunk` = CONSTRAINED greedy decode
  under the FAST grammar (sequences are [BOA][t_1..t_k], no EOA — a
  valid generation expands to exactly chunk*dim coefficients, so each
  step masks to tokens fitting the remaining symbol budget and BOA/PAD
  are never sampled; every generation decodes by construction) ->
  FastTokenizer.decode -> denormalize via the batch's quantile stats. Requires threading q01/q99 through DatasetStats
  -> collator (dataset-owned quantiles, already backfilled corpus-wide;
  parse-edge None + loud AR failure for un-backfilled data).
- **`bijou/encoders/gemma4.py` (new, thin)**: wraps the owned gemma4
  backbone; `encode` = today's `BijouModel.encode_prefix` (KVCache,
  kv_stop_layer, stream extraction — now returning ordered tuple);
  `inputs_collator` = the prompt/pixel half of today's `PrefixCollator`
  (chat template + processor → GemmaInputs; the trunk-agnostic half
  moves into the shared `Collator`); `param_groups` = the text/vision
  partition from the unfreeze work; `stream_geometries` = (kv_heads=1,
  head_dim=512, rope=backbone global rope) × exports.
- **`bijou/encoders/siglip2.py` (new)**: HF-backed towers initially;
  learned adapter (LN + k/v projections + 2D patch positions + camera-slot
  embeddings, 1D text positions) producing MemoryStreams with
  `rope=None`. Text features cacheable across a rollout (instruction
  fixed) — a rig-latency win to exploit later.
- **`bijou/data.py`**: keeps dataset selection/stats/holdout (trunk-
  agnostic); `CollatedBatch`, `NormStats` and the shared `Collator` move
  to `interface.py`; `PrefixCollator`'s encoder-specific half becomes
  `encoders/gemma4.py`'s `GemmaInputsCollator`.
- **`bijou/train.py`**: objective-agnostic loop; CLI grows
  `--encoder {gemma4,siglip2}` / `--decoder {flow,ar_fast}` (defaults =
  today's pair) with per-kind arg groups; `train_args` recorded verbatim
  as now.
- **`bijou/eval/policies.py`**: BijouPolicy shrinks to collate -> encode ->
  `predict_chunk` (decoder-agnostic); the eval-level `ChunkPolicy`
  protocol is already the right upper interface and does not change.

Import DAG after: `train/eval/rollout -> loading -> data -> model ->
{encoders, decoders} -> interface -> {gemma4, siglip2, nn}`.

## 6. Sequencing (each step gated on the CPU loss oracle 1.8896/1.7237 EXACT)

1. `nn.py` lift + `interface.py` extraction; expert consumes
   ObservationMemory; schedule becomes positional internally. Key-set test.
2. Tagged config schema + legacy synthesizer + fixture tests (cont45k's
   real json committed as a fixture; `--init-from` old-into-new).
3. Collator split (`CollatedBatch[I]`); call sites in train/eval/rollout.
4. Encoder/decoder ABCs + BijouModel composition + param_groups routing;
   rerun the unfreeze grad-flow probe.
5. New cells, cheapest evidence first:
   a. SigLIP2 **acuity probe** (existing harness; kill criterion: not
      sharper than the 8.4 px tower / 10.8 px K4 anchors);
   b. SigLIP2 encoder + 20k mini-ablation vs matched gemma4 control;
   c. AR-FAST decoder (option B) + CE-convergence probes
      (`architecture.md` §8.3 ladder, minus option C).

## 7. Risks / open items

- Silent state_dict drift during the flow-decoder move — covered by the
  key-set equality gate + oracle.
- `CollatedBatch` generics vs pyright: verify `Generic` + frozen slots
  dataclass inference stays clean at call sites (fallback: per-encoder
  concrete batch dataclasses sharing the core by composition, no Generic).
- Quantile stats: **required on the data path, Optional at the
  old-checkpoint boundary** (`NormStats.q01/q99: Tensor | None`).
  DatasetStats gains `q01/q99` for action+state as `tuple[float, ...] |
  None` — and can mirror the per-modality grouping (`action:
  ChannelStats, state: ChannelStats` of float tuples), which its
  to/from_dict already uses on the wire. `from_lerobot_stats` (the data
  path) REQUIRES them: a dataset without backfilled quantiles fails at
  selection with the `ldtools.backfill_quantile_stats` command as the
  remedy (corpus + local dev copies verified backfilled).
  `from_state_dict` on OLD checkpoints (whose `per_dataset_normalization`
  predates quantiles) yields None; consumers that need quantiles check
  for None and FAIL FAST with an error naming the checkpoint and remedy
  — no sentinel values. Old-checkpoint flow rollout keeps working
  untouched (flow never reads quantiles); AR from an old stats table is
  impossible anyway (old checkpoints are flow models). A one-off
  converter (backfill quantiles into old checkpoints' stats tables from
  the hub datasets) can later retire the Optional entirely. Side benefit:
  in-batch quantiles make a `--normalization quantile` flow arm
  (π0-style [-1,1] scaling) a config-only future experiment.
- Two-sided learning risk for SigLIP2 streams (adapter k/v from scratch)
  — this is the stream-vs-KV question; the acuity probe and mini-ablation
  are the falsifiers before any big spend.
