# Plan: modular observation encoders × action decoders

Status: **design, not implemented** (2026-07-30).

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

- `EncodedPrefix.streams` is an **ordered tuple**, not a dict; stream
  identity is **positional**. Opaque string ids were considered and
  rejected (stringly-typed cross-references; their benefits — reorder
  safety, self-description — are covered by construction-time validation
  and the encoder config sitting in the same json).
- The cross-attention **schedule stays decoder-side but becomes
  positional**: indices into the encoder's declared `exports`. Decoder
  configs stop naming trunk internals (no more Gemma layer numbers in
  `ExpertConfig`). The schedule cannot move encoder-side: its length IS
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
    rope: RopeParameters | None   # None => positions are baked into the
                                  # memory; decoder applies no query RoPE.
                                  # Set   => decoder RoPEs queries at
                                  # positions >= per-sample real prefix len
                                  # (the Gemma streams' contract today).

@dataclass(frozen=True, slots=True)
class MemoryStream:
    key: Tensor      # [B, kv_heads, P, head_dim]
    value: Tensor    # [B, kv_heads, P, head_dim]

@dataclass(frozen=True, slots=True)
class EncodedPrefix:
    """The value crossing the encoder->decoder seam. Ordered exactly as
    the encoder config's `exports`."""
    streams: tuple[MemoryStream, ...]
    length: int                    # padded P
    padding_mask: Tensor | None    # [B, P], True = real


@dataclass(frozen=True, slots=True)
class BatchStats:
    """One modality's normalization stats, batched: each tensor [B, dim]
    (dim = action_dim or state_dim). Every sample carries its own
    dataset's stats — the per-dataset normalization mechanism."""
    mean: Tensor
    std: Tensor
    q01: Tensor
    q99: Tensor


@dataclass(frozen=True, slots=True)
class CollatedBatch(Generic[I]):
    """Trunk-agnostic core + typed encoder-specific inputs. pin_memory/to
    recurse into `encoder_inputs` and the BatchStats fields (all
    implement the same two hooks)."""
    encoder_inputs: I              # GemmaInputs | SigLip2Inputs
    state: Tensor                  # [B, state_dim]
    actions: Tensor                # [B, chunk, action_dim]
    action_is_pad: Tensor          # [B, chunk]
    action_stats: BatchStats       # each [B, action_dim]
    state_stats: BatchStats        # each [B, state_dim]
    # AR-only, filled by the collator when built with a FastTokenizer
    # (CPU-side in workers); None otherwise. The AR loss asserts loudly.
    # (Cannot be made mandatory like the quantiles: tokens depend on a
    # tokenizer artifact and cost real CPU per item — they are computed
    # only when an AR decoder will consume them.)
    action_tokens: Tensor | None       # [B, T_tok] padded token targets
    action_token_mask: Tensor | None   # [B, T_tok] True = real token


class ObservationEncoder(nn.Module, Generic[I]):
    """ABC. A trunk: collation + encode + unfreeze surface."""
    def stream_geometries(self) -> tuple[StreamGeometry, ...]: ...
    def build_collator(
        self, instruction: str | None, ...
    ) -> Collator[I]: ...                        # pickleable into workers
    def encode(self, inputs: I, *, with_grad: bool) -> EncodedPrefix: ...
    def param_groups(self) -> dict[str, list[Parameter]]: ...
    # e.g. {"text": [...], "vision": [...]} — the --text-lr/--vision-lr
    # flags route here for ANY encoder.


class ActionDecoder(nn.Module):
    """ABC. Owns its objective and its chunk-space inference."""
    def loss(self, prefix: EncodedPrefix, batch: CollatedBatch[Any]) -> Tensor: ...
    def predict_chunk(
        self, prefix: EncodedPrefix, batch: CollatedBatch[Any],
        *, generator: torch.Generator | None = None, noise: Tensor | None = None,
    ) -> Tensor: ...
    # Returns RAW-unit [B, chunk, action_dim]; normalize/denormalize via
    # the batch's per-item stats happens INSIDE (this logic moves out of
    # BijouPolicy). Decoder-specific inference knobs (Heun steps, decode
    # temperature) are constructor parameters of the decoder/policy, not
    # ABC surface.
```

## 2. Configs (tagged unions, parse at the edge)

```python
class EncoderKind(StrEnum):  GEMMA4 = "gemma4";  SIGLIP2 = "siglip2"
class DecoderKind(StrEnum):  FLOW = "flow";      AR_FAST = "ar_fast"

@dataclass(frozen=True, slots=True)
class GemmaEncoderConfig:
    backbone: str                  # HF id / local dir
    exports: tuple[int, ...]       # gemma layer indices, e.g. (4, 9, 14)
                                   # (trunk internals belong HERE — this is
                                   # the one config allowed to know them)
    max_soft_tokens: int

@dataclass(frozen=True, slots=True)
class SigLip2EncoderConfig:
    checkpoint: str                # e.g. google/siglip2-so400m-patch14-384
    exports: tuple[SigLip2Stage, ...]   # enum: TOP, MID, POOLED_TEXT, ...
    soft_token_budget: int         # NaFlex budget / pooling knob
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
    schedule: tuple[int, ...]      # POSITIONS into exports; len = depth
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
    schedule: tuple[int, ...]      # its own positional schedule
    tokenizer: str                 # artifact ref, e.g. ".../fast_tokenizer_v1"
    vocab_size: int                # BPE vocab + BOA/EOA/pad specials
    max_tokens: int                # decode budget (measured p99 + slack)
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
               "schedule": [0,0,0,0,1,1,1,1,2,2,2,2,2,2,2,2],
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
        position
        for position, count in enumerate(args.stream_counts)
        for _ in range(count)
    )                                                        # (0,0,0,0,1,...,2)
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
encoder = build_encoder(encoder_config, device=..., dtype=...)   # match on kind
geometries = encoder.stream_geometries()
validate_schedule(decoder_config.schedule, geometries)
#   - every index in bounds
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
  are byte-identical. Deltas: consumes `EncodedPrefix` (tuple indexing per
  schedule position instead of dict-by-layer); per-stream `StreamGeometry`
  sizes each layer's q-projection (legacy: all streams 512/1 — identical
  shapes); RoPE applied per stream geometry (None => skip); gains
  `loss()` (= `flow_matching_loss` moved in) and `predict_chunk()`
  (= `sample_actions` + the normalize/denormalize now in BijouPolicy).
  GATE: state_dict key-set equality vs today's ActionExpert under a
  legacy config, incl. buffer persistence.
- **`bijou/decoders/ar_fast.py` (new)**: token embedding (vocab +
  BOA/EOA/pad), state anchor token, N sandwich blocks (self-attn causal
  over `[state][BOA][t_1..t_k]`, cross-attn per its schedule), LM head.
  `loss` = CE over `action_tokens` masked by `action_token_mask`;
  `predict_chunk` = decode -> FastTokenizer.decode -> denormalize via the
  batch's quantile stats. Requires threading q01/q99 through DatasetStats
  -> collator (dataset-owned quantiles, already backfilled corpus-wide;
  parse-edge None + loud AR failure for un-backfilled data).
- **`bijou/encoders/gemma4.py` (new, thin)**: wraps the owned gemma4
  backbone; `encode` = today's `BijouModel.encode_prefix` (KVCache,
  kv_stop_layer, stream extraction — now returning ordered tuple);
  `build_collator` = today's `PrefixCollator`; `param_groups` = the
  text/vision partition from the unfreeze work; `stream_geometries` =
  (kv_heads=1, head_dim=512, rope=backbone global rope) × exports.
- **`bijou/encoders/siglip2.py` (new)**: HF-backed towers initially;
  learned adapter (LN + k/v projections + 2D patch positions + camera-slot
  embeddings, 1D text positions) producing MemoryStreams with
  `rope=None`. Text features cacheable across a rollout (instruction
  fixed) — a rig-latency win to exploit later.
- **`bijou/data.py`**: keeps dataset selection/stats/holdout (trunk-
  agnostic); `CollatedBatch` core moves to `interface.py`; `PrefixCollator`
  moves under `encoders/gemma4.py`.
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
   EncodedPrefix; schedule becomes positional internally. Key-set test.
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
- Quantile stats: **mandatory in the batch** (`BatchStats.q01/q99`).
  DatasetStats gains q01/q99 for action+state — and can mirror the
  per-modality grouping (`action: ChannelStats, state: ChannelStats` of
  float tuples), which its to/from_dict already uses on the wire;
  `from_lerobot_stats` (the data path) REQUIRES them — a dataset without backfilled quantiles fails at
  selection with the `ldtools.backfill_quantile_stats` command as the
  remedy (corpus + local dev copies verified backfilled 2026-07-30).
  The one place None survives is `from_state_dict` on OLD checkpoints
  (their `per_dataset_normalization` tables predate quantiles): there
  `item_tensors` emits NaN poison, never read by flow paths, and the AR
  decoder validates `isfinite` at its single consumption site with an
  error naming the checkpoint and remedy. Old-checkpoint flow rollout
  therefore keeps working untouched; AR from an old stats table is
  impossible anyway (old checkpoints are flow models). Side benefit:
  in-batch quantiles make a `--normalization quantile` flow arm
  (π0-style [-1,1] scaling) a config-only future experiment.
- Two-sided learning risk for SigLIP2 streams (adapter k/v from scratch)
  — this is the stream-vs-KV question; the acuity probe and mini-ablation
  are the falsifiers before any big spend.
