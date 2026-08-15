"""Phase-4 parity gates (docs/vla-architecture.md §10): the family
classes against the BijouModel world, on the SAME fabricated legacy
checkpoints — old path = ``bijou.loading.from_checkpoint`` +
``BijouTrainStep``'s sum-form protocol, new path = ``convert_legacy``
+ ``<Family>.from_checkpoint``. Everything numeric asserts BITWISE
equality (same box, same dtype, same batch — the tolerance discipline
the plan pins):

1. losses — two seeded steps of ``family.forward(counts)`` against the
   old ``loss_component_sums``/``count_normalizers`` composition;
2. predictions — ``predict``/``predict_flow``/``predict_ar`` (and the
   teacher-forced block logits) against the old ``predict_chunk`` and
   decoder-kernel paths;
3. ``param_groups`` — name → parameter-NAME sets identical (parameter
   ids cannot match across two separately-loaded models; names + the
   full state-dict bitwise check pin the same fact), and the loaded
   state dicts bitwise equal modulo the decoder attribute rename
   (``decoder`` → ``flow_decoder``/``ar_decoder``, §8 ledger);
4. ``checkpoint_components`` — exactly the metadata's ``weights: true``
   set, round-tripped through ``write_checkpoint`` +
   ``validate_checkpoint``.

Families covered: gemma_flow, gemma_ar, molmoact2_flow, molmoact2_ar,
molmoact2_joint. molmo2_ar has no hermetic prompt/processor fixture
(none exists in the old suite either); its construction shares
``build_molmo2_ar_decoder`` with the old path and its ops are the
gemma_ar-tested free functions.

Hermetic, CPU, synthetic data: tiny random trunks written per module
(the ``bijou.testing`` + ``gemma4.testing`` patterns), batches
fabricated directly (Gemma) or through the model's own inputs collator
(MolmoAct2)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, override

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from torch import Tensor, nn

from bijou.checkpoint import read_metadata, validate_checkpoint, write_checkpoint
from bijou.convert_legacy import convert
from bijou.fast.molmoact2 import MolmoAct2FastTokenizer
from bijou.loading import (
    PROMPT_FORMAT,
    ARDecoderConfig,
    FlowDecoderSection,
    GemmaPromptConfig,
    MolmoAct2PromptConfig,
    ar_backbone_config_to_dict,
    expert_config_from_architecture,
    from_checkpoint,
    load_vla,
    molmoact2_ar_config_from_flow_section,
    parse_prompt_config,
)
from bijou.model import BijouModel
from bijou.modelling.aux_text import SUFFIX_FORMAT
from bijou.modelling.codecs import FastActionCodec, MolmoAct2ActionCodec
from bijou.modelling.decoders.ar_gemma import GemmaARDecoder
from bijou.modelling.decoders.ar_molmoact2 import MolmoAct2ARDecoder
from bijou.modelling.decoders.ar_suffix import ARSuffixDecoder
from bijou.modelling.decoders.flow import (
    FlowDecoder,
    SelfAttentionMode,
    TimeConditioning,
)
from bijou.modelling.decoders.molmo_flow import MolmoFlowDecoder, molmo_flow_loss_sums
from bijou.modelling.encoders.gemma4 import GemmaInputs
from bijou.modelling.gemma4.config import Gemma4Config
from bijou.modelling.gemma4.loading import load_config
from bijou.modelling.gemma4.model import Gemma4Model
from bijou.modelling.gemma4.testing import tiny_config_json as gemma_tiny_config_json
from bijou.modelling.interface import (
    BatchInputs,
    CameraFrame,
    CollatedBatch,
    NormStats,
    ObservationMemory,
    PromptInputs,
    SamplingMethod,
)
from bijou.models.gemma_ar import GemmaARVLA
from bijou.models.gemma_flow import GemmaFlowVLA
from bijou.models.molmoact2_ar import MolmoAct2ARVLA
from bijou.models.molmoact2_flow import MolmoAct2FlowVLA
from bijou.models.molmoact2_joint import MolmoAct2JointVLA
from bijou.testing import (
    TINY_MOLMOACT2_D,
    TINY_MOLMOACT2_Q01,
    TINY_MOLMOACT2_Q99,
    TINY_MOLMOACT2_T,
    tiny_molmoact2_flow_section,
    write_tiny_molmoact2_release,
)
from bijou.vla import VLA, VLAFamily

TINY_FAST_FIXTURE = Path(__file__).parent / "fixtures" / "tiny_fast_tokenizer"
MOLMOACT2_FAST_FIXTURE = Path(__file__).parent / "fixtures" / "molmoact2_fast_tokenizer"

BATCH = 2
GEMMA_VOCAB = 2048
GEMMA_DIM = 6
GEMMA_CHUNK = 10
GEMMA_PROMPT_LEN = 12
JOINT_CE_WEIGHT = 0.5


# ---------------------------------------------------------------------------
# The old world's train-step composition, inlined: bijou.train now
# drives the family classes; the parity suite keeps the historical
# sum-form composition as its reference until phase 6 retires the suite
# with BijouModel.


class BijouTrainStep[I: BatchInputs](torch.nn.Module):
    """The historical one-module train forward (prefix encode + decoder
    objective), reduced to the sum-form protocol this suite exercises
    on CPU (the autocast contexts construct disabled there —
    byte-identical to the recorded math)."""

    def __init__(
        self,
        model: BijouModel[I, Any],
        *,
        backbone_trained: bool,
    ) -> None:
        super().__init__()
        self.model = model
        self.backbone_trained = backbone_trained

    @override
    def forward(
        self,
        batch: CollatedBatch[I],
        normalizers: (
            tuple[Tensor, Tensor | None] | tuple[Tensor, Tensor, Tensor | None] | None
        ) = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        inputs = batch.encoder_inputs
        device_type = next(iter(inputs.tensors().values())).device.type
        autocast_on = device_type == "cuda" and self.backbone_trained
        with torch.autocast(device_type, torch.bfloat16, enabled=autocast_on):
            memory = self.model.encode(inputs, with_grad=self.backbone_trained)
            if isinstance(self.model.decoder, ARSuffixDecoder):
                if normalizers is None:
                    return self.model.loss_components(memory, batch)
                assert len(normalizers) == 2  # AR runs: (action, aux)
                return self._chunk_share(memory, batch, normalizers)
        if self.model.joint_ce is not None:
            decoder = self.model.decoder
            assert isinstance(decoder, MolmoFlowDecoder)
            flow_sums = molmo_flow_loss_sums(
                decoder,
                memory,
                batch,
                insulate=self.model.insulate_expert,
            )
            with torch.autocast(device_type, torch.bfloat16, enabled=autocast_on):
                ce_sums = self.model.joint_ce_loss_sums(memory, batch)
            if normalizers is not None:
                assert len(normalizers) == 3  # joint runs: 3 normalizers
                return self._joint_share(flow_sums, ce_sums, normalizers)
            return self._joint_share(flow_sums, ce_sums, None)
        if normalizers is None:
            return self.model.loss_components(memory, batch)
        assert len(normalizers) == 2  # non-joint runs: (action, aux)
        return self._chunk_share(memory, batch, normalizers)

    def _joint_share(
        self,
        flow_sums: tuple[Tensor, Tensor],
        ce_sums: tuple[Tensor, Tensor, Tensor | None, Tensor | None],
        normalizers: tuple[Tensor, Tensor, Tensor | None] | None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        flow_sum, flow_count = flow_sums
        ce_action_sum, ce_action_count, ce_aux_sum, _ = ce_sums
        assert ce_aux_sum is None  # rider constructed aux-None
        if normalizers is None:
            flow_norm, ce_action_norm = flow_count, ce_action_count
        else:
            flow_norm, ce_action_norm, _ = normalizers
        loss = flow_sum / flow_norm + self.model.joint_ce_weight * (
            ce_action_sum / ce_action_norm
        )
        return (
            loss,
            (
                (flow_sum / flow_norm).detach()
                if normalizers is None
                else flow_sum.detach()
            ),
            ce_action_sum.detach(),
            ce_action_count,
        )

    def _chunk_share(
        self,
        memory: ObservationMemory,
        batch: CollatedBatch[I],
        normalizers: tuple[Tensor, Tensor | None],
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        action_norm, aux_norm = normalizers
        action_sum, _, aux_sum, aux_count = self.model.loss_component_sums(
            memory,
            batch,
        )
        loss = action_sum / action_norm
        if aux_sum is not None:
            decoder = self.model.decoder
            assert isinstance(decoder, ARSuffixDecoder)
            assert aux_norm is not None
            loss = loss + decoder.aux_loss_weight * (aux_sum / aux_norm.clamp(min=1))
        return (
            loss,
            action_sum.detach(),
            None if aux_sum is None else aux_sum.detach(),
            aux_count,
        )


# ---------------------------------------------------------------------------
# Shared assertion helpers


def old_sum_form(
    old_model: BijouModel[Any, Any],
    batch: CollatedBatch[Any],
    *,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    """The old world's two-phase composition: count normalizers (data
    only), then BijouTrainStep's sum-form forward — exactly what the
    family's ``forward(counts)`` must reproduce bitwise."""
    step = BijouTrainStep(old_model, backbone_trained=False)
    if old_model.joint_ce is not None:
        normalizers = old_model.joint_ce_count_normalizers(batch)
    else:
        normalizers = old_model.loss_count_normalizers(batch)
    torch.manual_seed(seed)
    return step(batch, normalizers=normalizers)


def assert_loss_parity(
    old_model: BijouModel[Any, Any],
    family: VLA[Any],
    batch: CollatedBatch[Any],
    *,
    seed: int,
) -> None:
    old_loss, old_action_sum, old_aux_sum, old_aux_count = old_sum_form(
        old_model,
        batch,
        seed=seed,
    )
    counts = family.loss_counts(batch)
    torch.manual_seed(seed)
    report = family(batch, counts=counts)
    assert torch.equal(report.objective, old_loss)
    assert torch.equal(report.components["action"].sum, old_action_sum)
    if old_aux_sum is None:
        assert set(report.components) == {"action"}
    else:
        assert old_aux_count is not None
        assert torch.equal(report.components["aux"].sum, old_aux_sum)
        assert torch.equal(report.components["aux"].count, old_aux_count)
    # The report's component keys are the loss_counts keys (the loop's
    # enforced invariant).
    assert set(report.components) == set(counts)


def named_parameter_ids(model: nn.Module) -> dict[int, str]:
    return {id(parameter): name for name, parameter in model.named_parameters()}


def rename_root(name: str, rename: dict[str, str]) -> str:
    head, dot, rest = name.partition(".")
    return rename.get(head, head) + dot + rest


def group_name_sets(
    model: nn.Module,
    groups: dict[str, list[nn.Parameter]],
    rename: dict[str, str],
) -> dict[str, set[str]]:
    by_id = named_parameter_ids(model)
    return {
        group: {rename_root(by_id[id(parameter)], rename) for parameter in group_params}
        for group, group_params in groups.items()
    }


def assert_param_group_parity(
    old_model: BijouModel[Any, Any],
    family: VLA[Any],
    rename: dict[str, str],
) -> None:
    """Group name → parameter-name sets identical, and every parameter
    (plus buffers — the whole state dict) bitwise equal between the two
    loads, modulo the decoder attribute rename."""
    old_groups = group_name_sets(old_model, old_model.param_groups(), rename)
    new_groups = group_name_sets(family, family.param_groups(), {})
    assert old_groups == new_groups
    old_state = {
        rename_root(name, rename): tensor
        for name, tensor in old_model.state_dict().items()
    }
    new_state = dict(family.state_dict())
    assert set(old_state) == set(new_state)
    for name, tensor in old_state.items():
        assert torch.equal(tensor, new_state[name]), f"state mismatch at {name}"


def assert_components_roundtrip(
    family: VLA[Any],
    converted: Path,
    target: Path,
) -> None:
    """checkpoint_components() must be exactly the metadata's
    ``weights: true`` set, and the family's live state must write back
    through the toolkit into a directory that validates (pristine-trunk
    fixtures: the converted checkpoint's own ``backbone/`` mirror)."""
    metadata = read_metadata(converted)
    weighted = {
        name for name, record in metadata.components.items() if record["weights"]
    }
    components = {
        name: {key: tensor.contiguous() for key, tensor in module.state_dict().items()}
        for name, module in family.checkpoint_components().items()
    }
    assert set(components) == weighted
    write_checkpoint(
        target,
        metadata=metadata,
        components=components,
        backbone=converted / "backbone",
    )
    validate_checkpoint(target)


# ---------------------------------------------------------------------------
# Gemma fixtures: a tiny trunk (small vocab — the real tokenizer never
# runs) + fabricated legacy checkpoints + hand-built GemmaInputs batches.


def write_gemma_trunk(directory: Path) -> Path:
    """A loadable tiny Gemma4 trunk: gemma4.testing's architecture at a
    2048 vocabulary (batches use ids < 1000; the real 262k vocab exists
    only to match the real tokenizer, which these tests never run),
    plus a WordLevel tokenizer so AutoTokenizer resolves hermetically
    (the ar_backbone construction path tokenizes its opener)."""
    config_json = gemma_tiny_config_json()
    config_json["text_config"]["vocab_size"] = GEMMA_VOCAB
    config_json["text_config"]["vocab_size_per_layer_input"] = GEMMA_VOCAB
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(config_json))
    torch.manual_seed(0)
    model = Gemma4Model(Gemma4Config.from_dict(config_json), device="cpu")
    state = {
        f"model.{name}": tensor.contiguous()
        for name, tensor in model.state_dict().items()
        # lm_head ties to the embedding at load, like the released
        # checkpoints.
        if name != "lm_head.weight"
    }
    save_file(state, str(directory / "model.safetensors"))
    vocab = {"<unk>": 0, "<pad>": 1, "<start_of_turn>": 3, "model": 4, "hello": 5}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}),
    )
    return directory


def gemma_prompt_config() -> GemmaPromptConfig:
    # Exports = the tiny prefix's global layers (1, 3, 5).
    return GemmaPromptConfig(
        exports=(1, 3, 5),
        max_soft_tokens=8,
        format=PROMPT_FORMAT,
        state_dim=GEMMA_DIM,
        condition_fields=(),
        generate_bracket=False,
    )


def gemma_flow_section() -> FlowDecoderSection:
    return FlowDecoderSection(
        hidden_size=32,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        self_attention_mode=SelfAttentionMode.CAUSAL_ACTIONS,
        self_attention_rope_theta=10_000.0,
        cross_attention_heads=2,
        schedule=("kv1", "kv3", "kv5"),
        action_dim=GEMMA_DIM,
        state_dim=GEMMA_DIM,
        chunk_size=GEMMA_CHUNK,
        time_embed_dim=16,
        time_conditioning=TimeConditioning.ADDITIVE,
    )


def gemma_stats_dict() -> dict[str, Any]:
    return {
        "action": {
            "mean": [0.0] * GEMMA_DIM,
            "std": [1.0] * GEMMA_DIM,
            "q01": [-1.0] * GEMMA_DIM,
            "q99": [1.0] * GEMMA_DIM,
        },
        "observation.state": {
            "mean": [0.0] * GEMMA_DIM,
            "std": [1.0] * GEMMA_DIM,
        },
    }


def gemma_train_args(**overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "decoder": "flow",
        "decoder_hidden": 32,
        "decoder_heads": 2,
        "decoder_intermediate": 64,
        "decoder_cross_heads": 2,
        "stream_counts": [1, 1, 1],
        "self_attention_mode": "causal_actions",
        "chunk_size": GEMMA_CHUNK,
        "max_soft_tokens": 8,
        "max_crops": 1,
        "time_conditioning": "additive",
        "target_time_embed": False,
        "fast_tokenizer": None,
        "aux_loss_weight": 1.0,
        "seed": 0,
    }
    args.update(overrides)
    return args


def write_gemma_flow_legacy(directory: Path, trunk: Path) -> Path:
    directory.mkdir(parents=True)
    expert_config = expert_config_from_architecture(
        gemma_prompt_config(),
        gemma_flow_section(),
        load_config(trunk),
    )
    torch.manual_seed(3)
    decoder = FlowDecoder(expert_config, device="cpu", dtype=torch.float32)
    save_file(
        {k: v.contiguous() for k, v in decoder.state_dict().items()},
        str(directory / "expert.safetensors"),
    )
    save_file(prompt_state(seed=4), str(directory / "prompt.safetensors"))
    config = {
        "format": 3,
        "backbone": {"id": str(trunk), "depth": "prefix"},
        "prompt": gemma_prompt_config().to_dict(),
        "decoder": gemma_flow_section().to_dict(),
        "step": 3,
        "train_args": gemma_train_args(),
        "normalization": gemma_stats_dict(),
        "per_dataset_normalization": {},
    }
    (directory / "bijou_config.json").write_text(json.dumps(config))
    return directory


def prompt_state(*, seed: int) -> dict[str, Tensor]:
    """Non-zero state_proj weights so the soft state token genuinely
    conditions the memory (zero-init would make state parity vacuous)."""
    generator = torch.Generator().manual_seed(seed)
    hidden = 64  # the tiny trunk's hidden size
    return {
        "state_proj.weight": torch.randn(hidden, GEMMA_DIM, generator=generator) * 0.05,
        "state_proj.bias": torch.randn(hidden, generator=generator) * 0.05,
    }


def gemma_ar_decoder_config() -> ARDecoderConfig:
    loaded = FastActionCodec.load(TINY_FAST_FIXTURE)
    return ARDecoderConfig(
        tokenizer=str(TINY_FAST_FIXTURE),
        vocab_total=loaded.vocab_total,
        block_base=GEMMA_VOCAB - loaded.vocab_total,  # tail placement, like E2B
        chunk_size=loaded.time_horizon,
        action_dim=loaded.action_dim,
        suffix_format=SUFFIX_FORMAT,
        aux=None,
    )


def write_gemma_ar_legacy(directory: Path, trunk: Path) -> Path:
    directory.mkdir(parents=True)
    torch.manual_seed(5)
    decoder = GemmaARDecoder(
        gemma_ar_decoder_config(),
        Gemma4Config.from_json(trunk / "config.json").text,
        FastActionCodec.load(TINY_FAST_FIXTURE),
        tokenizer=_gemma_text_tokenizer(trunk),
        device="cpu",
        dtype=torch.float32,
    )
    save_file(
        {k: v.contiguous() for k, v in decoder.state_dict().items()},
        str(directory / "expert.safetensors"),
    )
    save_file(prompt_state(seed=6), str(directory / "prompt.safetensors"))
    config = {
        "format": 3,
        "backbone": {"id": str(trunk), "depth": "full"},
        "prompt": gemma_prompt_config().to_dict(),
        "decoder": ar_backbone_config_to_dict(gemma_ar_decoder_config()),
        "step": 3,
        "train_args": gemma_train_args(
            decoder="ar_backbone",
            chunk_size=gemma_ar_decoder_config().chunk_size,
            fast_tokenizer=str(TINY_FAST_FIXTURE),
        ),
        "normalization": gemma_stats_dict(),
        "per_dataset_normalization": {},
    }
    (directory / "bijou_config.json").write_text(json.dumps(config))
    return directory


def _gemma_text_tokenizer(trunk: Path) -> Any:
    import transformers

    return transformers.AutoTokenizer.from_pretrained(str(trunk))


def gemma_batch(
    seed: int,
    *,
    chunk_size: int,
    with_tokens: bool,
) -> CollatedBatch[GemmaInputs]:
    """A hand-built Gemma batch: text-only prompt ids (no images — the
    encoder passes pixel_values=None through), the soft state slot just
    inside the sequence end, mean-0/std-1 stats (state and CollatedBatch
    carry the same values), and FAST action tokens when the AR family
    needs them."""
    generator = torch.Generator().manual_seed(seed)
    state = torch.randn(BATCH, GEMMA_DIM, generator=generator)
    inputs = GemmaInputs(
        input_ids=torch.randint(
            3,
            1000,
            (BATCH, GEMMA_PROMPT_LEN),
            generator=generator,
        ),
        attention_mask=torch.ones(BATCH, GEMMA_PROMPT_LEN, dtype=torch.long),
        pixel_values=None,  # pyright: ignore[reportArgumentType] — text-only parity batch; encode passes None through
        image_position_ids=None,  # pyright: ignore[reportArgumentType] — text-only parity batch
        state=state,
        state_slot=-2,
        has_padding=False,
    )
    actions = torch.cumsum(
        torch.randn(BATCH, chunk_size, GEMMA_DIM, generator=generator) * 0.05,
        dim=1,
    ).clamp(-1, 1)
    action_tokens: Tensor | None = None
    if with_tokens:
        codec = FastActionCodec.load(TINY_FAST_FIXTURE)
        bounds = np.full(GEMMA_DIM, 1.0)
        sequences = [
            codec.encode(actions[row].numpy(), -bounds, bounds) for row in range(BATCH)
        ]
        width = max(len(sequence) for sequence in sequences)
        action_tokens = torch.tensor(
            [
                sequence + [codec.pad] * (width - len(sequence))
                for sequence in sequences
            ],
            dtype=torch.long,
        )
    stats = NormStats(
        mean=torch.zeros(BATCH, GEMMA_DIM),
        std=torch.ones(BATCH, GEMMA_DIM),
        q01=torch.full((BATCH, GEMMA_DIM), -1.0),
        q99=torch.full((BATCH, GEMMA_DIM), 1.0),
    )
    return CollatedBatch(
        encoder_inputs=inputs,
        state=state,
        actions=actions,
        action_is_pad=torch.zeros(BATCH, chunk_size, dtype=torch.bool),
        action_stats=stats,
        state_stats=stats,
        action_tokens=action_tokens,
        suffix_tokens=None,
        suffix_is_aux=None,
    )


@pytest.fixture(scope="module")
def gemma_flow_world(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[BijouModel[Any, Any], GemmaFlowVLA, Path]:
    root = tmp_path_factory.mktemp("vla-parity-gemma-flow")
    trunk = write_gemma_trunk(root / "trunk")
    legacy = write_gemma_flow_legacy(root / "legacy", trunk)
    converted = root / "converted"
    metadata = convert(legacy, converted)
    assert metadata.family is VLAFamily.GEMMA_FLOW
    old_model, _ = from_checkpoint(legacy, device="cpu", dtype=torch.float32)
    family = GemmaFlowVLA.from_checkpoint(
        converted,
        device="cpu",
        dtype=torch.float32,
    )
    return old_model, family, converted


@pytest.fixture(scope="module")
def gemma_ar_world(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[BijouModel[Any, Any], GemmaARVLA, Path]:
    root = tmp_path_factory.mktemp("vla-parity-gemma-ar")
    trunk = write_gemma_trunk(root / "trunk")
    legacy = write_gemma_ar_legacy(root / "legacy", trunk)
    converted = root / "converted"
    metadata = convert(legacy, converted)
    assert metadata.family is VLAFamily.GEMMA_AR
    old_model, _ = from_checkpoint(legacy, device="cpu", dtype=torch.float32)
    family = GemmaARVLA.from_checkpoint(converted, device="cpu", dtype=torch.float32)
    return old_model, family, converted


# ---------------------------------------------------------------------------
# MolmoAct2 fixtures: the tiny release pair (bijou.testing) + derived
# joint/ar legacy layouts + collator-built batches.


@pytest.fixture(scope="module")
def molmoact2_world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("vla-parity-molmoact2")
    trunk, flow_legacy = write_tiny_molmoact2_release(root)
    flow_meta = json.loads((flow_legacy / "bijou_config.json").read_text())
    prompt = parse_prompt_config(flow_meta["prompt"])
    assert isinstance(prompt, MolmoAct2PromptConfig)
    ar_config = molmoact2_ar_config_from_flow_section(
        tiny_molmoact2_flow_section(),
        prompt,
        str(trunk),
        fast_tokenizer=str(MOLMOACT2_FAST_FIXTURE),
    )
    ar_section = ar_backbone_config_to_dict(ar_config)

    joint_legacy = root / "joint_legacy"
    joint_legacy.mkdir()
    joint_meta = json.loads(json.dumps(flow_meta))  # deep copy
    joint_meta["joint_ce"] = ar_section
    joint_meta["train_args"]["objective"] = "joint"
    joint_meta["train_args"]["joint_ce_weight"] = JOINT_CE_WEIGHT
    joint_meta["train_args"]["insulate_expert"] = True
    (joint_legacy / "bijou_config.json").write_text(json.dumps(joint_meta))
    shutil.copy2(
        flow_legacy / "expert.safetensors",
        joint_legacy / "expert.safetensors",
    )

    ar_legacy = root / "ar_legacy"
    ar_legacy.mkdir()
    ar_meta = json.loads(json.dumps(flow_meta))  # deep copy
    ar_meta["decoder"] = ar_section
    ar_meta["train_args"]["objective"] = "ar"
    ar_meta["train_args"]["fast_tokenizer"] = str(MOLMOACT2_FAST_FIXTURE)
    (ar_legacy / "bijou_config.json").write_text(json.dumps(ar_meta))

    worlds: dict[str, Any] = {"trunk": trunk}
    for name, legacy, family_type in (
        ("flow", flow_legacy, MolmoAct2FlowVLA),
        ("joint", joint_legacy, MolmoAct2JointVLA),
        ("ar", ar_legacy, MolmoAct2ARVLA),
    ):
        converted = root / f"{name}_converted"
        metadata = convert(legacy, converted)
        assert metadata.family.value == f"molmoact2_{name}"
        old_model, _ = from_checkpoint(legacy, device="cpu", dtype=torch.float32)
        family = family_type.from_checkpoint(
            converted,
            device="cpu",
            dtype=torch.float32,
        )
        worlds[name] = (old_model, family, converted)
    old_joint, _, _ = worlds["joint"]
    # Run properties the old world set from train flags (never
    # serialized on BijouModel); the family reads them from the
    # recorded objective instead.
    old_joint.joint_ce_weight = JOINT_CE_WEIGHT
    old_joint.insulate_expert = True
    return worlds


def molmoact2_batch(
    old_model: BijouModel[Any, Any],
    seed: int,
) -> CollatedBatch[Any]:
    """A batch through the model's OWN inputs collator (synthetic camera
    frames, normalized state), with raw actions inside the fixture's
    quantile rows and their codec token rows (the flow loss ignores
    them; the CE branches require them)."""
    generator = torch.Generator().manual_seed(seed)
    prompts = []
    for _ in range(BATCH):
        cameras = tuple(
            CameraFrame(
                name=name,
                kind="unknown",
                image=torch.rand(3, 96, 128, generator=generator),
            )
            for name in ("top", "wrist")
        )
        prompts.append(
            PromptInputs(
                instruction="pick the cube",
                cameras=cameras,
                condition_text="",
                state=torch.rand(TINY_MOLMOACT2_D, generator=generator) * 2 - 1,
            ),
        )
    inputs = old_model.encoder.inputs_collator()(prompts)
    normalized = torch.cumsum(
        torch.randn(
            BATCH,
            TINY_MOLMOACT2_T,
            TINY_MOLMOACT2_D,
            generator=generator,
        )
        * 0.04,
        dim=1,
    ).clamp(-0.9, 0.9)
    q01 = torch.tensor(TINY_MOLMOACT2_Q01, dtype=torch.float32)
    q99 = torch.tensor(TINY_MOLMOACT2_Q99, dtype=torch.float32)
    actions = (normalized + 1.0) * (q99 - q01) / 2.0 + q01
    codec = MolmoAct2ActionCodec(
        MolmoAct2FastTokenizer.load(MOLMOACT2_FAST_FIXTURE),
        time_horizon=TINY_MOLMOACT2_T,
        action_dim=TINY_MOLMOACT2_D,
    )
    sequences = [
        codec.encode(actions[row].double().numpy(), q01.numpy(), q99.numpy())
        for row in range(BATCH)
    ]
    width = max(len(sequence) for sequence in sequences)
    tokens = torch.tensor(
        [sequence + [codec.pad] * (width - len(sequence)) for sequence in sequences],
        dtype=torch.long,
    )
    stats = NormStats(
        mean=torch.zeros(BATCH, TINY_MOLMOACT2_D),
        std=torch.ones(BATCH, TINY_MOLMOACT2_D),
        q01=q01.expand(BATCH, -1).clone(),
        q99=q99.expand(BATCH, -1).clone(),
    )
    return CollatedBatch(
        encoder_inputs=inputs,
        state=torch.zeros(BATCH, TINY_MOLMOACT2_D),
        actions=actions.float(),
        action_is_pad=torch.zeros(BATCH, TINY_MOLMOACT2_T, dtype=torch.bool),
        action_stats=stats,
        state_stats=stats,
        action_tokens=tokens,
        suffix_tokens=None,
        suffix_is_aux=None,
    )


# ---------------------------------------------------------------------------
# gemma_flow


def test_gemma_flow_losses_bitwise(
    gemma_flow_world: tuple[BijouModel[Any, Any], GemmaFlowVLA, Path],
) -> None:
    old_model, family, _ = gemma_flow_world
    for step, seed in enumerate((11, 12)):
        batch = gemma_batch(100 + step, chunk_size=GEMMA_CHUNK, with_tokens=False)
        assert_loss_parity(old_model, family, batch, seed=seed)


def test_gemma_flow_predictions_bitwise(
    gemma_flow_world: tuple[BijouModel[Any, Any], GemmaFlowVLA, Path],
) -> None:
    old_model, family, _ = gemma_flow_world
    batch = gemma_batch(102, chunk_size=GEMMA_CHUNK, with_tokens=False)
    # predict at the recorded serving point (heun-5 — the historical
    # default the converter records) vs the old default predict_chunk.
    torch.manual_seed(21)
    old_prediction = old_model.predict_chunk(batch)
    torch.manual_seed(21)
    actions = family.predict(batch)
    assert torch.equal(actions, old_prediction.actions)
    # predict_flow at an explicit non-default operating point.
    old_generator = torch.Generator().manual_seed(7)
    new_generator = torch.Generator().manual_seed(7)
    old_euler = old_model.predict_chunk(
        batch,
        generator=old_generator,
        num_steps=3,
        method=SamplingMethod.EULER,
    )
    new_euler = family.predict_flow(
        batch,
        num_steps=3,
        method=SamplingMethod.EULER,
        generator=new_generator,
    )
    assert old_euler.noise is not None
    assert torch.equal(new_euler.actions, old_euler.actions)
    assert torch.equal(new_euler.noise, old_euler.noise)
    # Noise reuse (the paired-re-decode contract) short-circuits the draw.
    reused = family.predict_flow(
        batch,
        num_steps=3,
        method=SamplingMethod.EULER,
        noise=old_euler.noise,
    )
    assert torch.equal(reused.actions, old_euler.actions)


def test_gemma_flow_param_groups_and_state(
    gemma_flow_world: tuple[BijouModel[Any, Any], GemmaFlowVLA, Path],
) -> None:
    old_model, family, _ = gemma_flow_world
    assert_param_group_parity(old_model, family, {"decoder": "flow_decoder"})


def test_gemma_flow_components_roundtrip(
    gemma_flow_world: tuple[BijouModel[Any, Any], GemmaFlowVLA, Path],
    tmp_path: Path,
) -> None:
    _, family, converted = gemma_flow_world
    assert_components_roundtrip(family, converted, tmp_path / "roundtrip")


def test_gemma_flow_registry_dispatch(
    gemma_flow_world: tuple[BijouModel[Any, Any], GemmaFlowVLA, Path],
) -> None:
    _, family, converted = gemma_flow_world
    loaded = load_vla(converted, device="cpu", dtype=torch.float32)
    assert type(loaded) is GemmaFlowVLA
    assert loaded.spec == family.spec
    assert family.spec.chunk_size == GEMMA_CHUNK
    assert family.spec.action_dim == GEMMA_DIM


# ---------------------------------------------------------------------------
# gemma_ar


def test_gemma_ar_losses_bitwise(
    gemma_ar_world: tuple[BijouModel[Any, Any], GemmaARVLA, Path],
) -> None:
    old_model, family, _ = gemma_ar_world
    chunk = gemma_ar_decoder_config().chunk_size
    for step, seed in enumerate((13, 14)):
        batch = gemma_batch(200 + step, chunk_size=chunk, with_tokens=True)
        assert_loss_parity(old_model, family, batch, seed=seed)


def test_gemma_ar_predictions_bitwise(
    gemma_ar_world: tuple[BijouModel[Any, Any], GemmaARVLA, Path],
) -> None:
    old_model, family, _ = gemma_ar_world
    chunk = gemma_ar_decoder_config().chunk_size
    batch = gemma_batch(202, chunk_size=chunk, with_tokens=True)
    old_prediction = old_model.predict_chunk(batch)
    new_prediction = family.predict_ar(batch)
    assert torch.equal(new_prediction.actions, old_prediction.actions)
    assert torch.equal(family.predict(batch), old_prediction.actions)
    # Teacher-forced block logits: the family's tensor adapter over the
    # decoder kernel the old model dispatches to.
    tokens = batch.action_tokens
    assert tokens is not None
    action_ids = tokens[:, :8]
    memory = old_model.encode(batch.encoder_inputs, with_grad=False)
    old_rows = old_model.ar_teacher_forced_block_logits(
        memory,
        [[int(t) for t in row] for row in action_ids],
    )
    stacked = torch.stack([row for row in old_rows if row is not None])
    new_logits = family.teacher_forced_block_logits(batch, action_ids)
    assert torch.equal(new_logits, stacked)


def test_gemma_ar_param_groups_and_state(
    gemma_ar_world: tuple[BijouModel[Any, Any], GemmaARVLA, Path],
) -> None:
    old_model, family, _ = gemma_ar_world
    assert_param_group_parity(old_model, family, {"decoder": "ar_decoder"})


def test_gemma_ar_components_roundtrip(
    gemma_ar_world: tuple[BijouModel[Any, Any], GemmaARVLA, Path],
    tmp_path: Path,
) -> None:
    _, family, converted = gemma_ar_world
    assert_components_roundtrip(family, converted, tmp_path / "roundtrip")


def test_gemma_ar_narration_refusals_delegate(
    gemma_ar_world: tuple[BijouModel[Any, Any], GemmaARVLA, Path],
) -> None:
    """The aux-less checkpoint LOADS (the old path never refused it);
    narration refuses at decode time from the decoder's own
    trained-fields record — the same refusal point as the old world."""
    from bijou.modelling.aux_text import AuxField

    _, family, _ = gemma_ar_world
    chunk = gemma_ar_decoder_config().chunk_size
    batch = gemma_batch(203, chunk_size=chunk, with_tokens=True)
    with pytest.raises(ValueError, match="non-empty generate"):
        family.predict_narrated(batch, generate=())
    with pytest.raises(ValueError, match="trained aux fields"):
        family.predict_narrated(batch, generate=(AuxField.SUBGOAL,))


# ---------------------------------------------------------------------------
# molmoact2_flow / molmoact2_ar / molmoact2_joint


def test_molmoact2_flow_losses_bitwise(molmoact2_world: dict[str, Any]) -> None:
    old_model, family, _ = molmoact2_world["flow"]
    for step, seed in enumerate((31, 32)):
        batch = molmoact2_batch(old_model, 300 + step)
        assert_loss_parity(old_model, family, batch, seed=seed)


def test_molmoact2_flow_predictions_bitwise(molmoact2_world: dict[str, Any]) -> None:
    old_model, family, _ = molmoact2_world["flow"]
    batch = molmoact2_batch(old_model, 302)
    # The recorded serving point (euler at the checkpoint's
    # num_flow_steps) ≡ the old molmo_flow defaults.
    torch.manual_seed(41)
    old_prediction = old_model.predict_chunk(batch)
    torch.manual_seed(41)
    actions = family.predict(batch)
    assert torch.equal(actions, old_prediction.actions)
    old_generator = torch.Generator().manual_seed(9)
    new_generator = torch.Generator().manual_seed(9)
    old_two = old_model.predict_chunk(batch, generator=old_generator, num_steps=2)
    new_two = family.predict_flow(
        batch,
        num_steps=2,
        method=SamplingMethod.EULER,
        generator=new_generator,
    )
    assert old_two.noise is not None
    assert torch.equal(new_two.actions, old_two.actions)
    assert torch.equal(new_two.noise, old_two.noise)


def test_molmoact2_flow_param_groups_and_state(
    molmoact2_world: dict[str, Any],
) -> None:
    old_model, family, _ = molmoact2_world["flow"]
    assert_param_group_parity(old_model, family, {"decoder": "flow_decoder"})


def test_molmoact2_flow_components_roundtrip(
    molmoact2_world: dict[str, Any],
    tmp_path: Path,
) -> None:
    _, family, converted = molmoact2_world["flow"]
    assert_components_roundtrip(family, converted, tmp_path / "roundtrip")


def test_molmoact2_ar_losses_bitwise(molmoact2_world: dict[str, Any]) -> None:
    old_model, family, _ = molmoact2_world["ar"]
    for step, seed in enumerate((33, 34)):
        batch = molmoact2_batch(old_model, 310 + step)
        assert_loss_parity(old_model, family, batch, seed=seed)


def test_molmoact2_ar_predictions_bitwise(molmoact2_world: dict[str, Any]) -> None:
    old_model, family, _ = molmoact2_world["ar"]
    assert isinstance(family, MolmoAct2ARVLA)
    batch = molmoact2_batch(old_model, 312)
    old_prediction = old_model.predict_chunk(batch)
    new_prediction = family.predict_ar(batch)
    assert torch.equal(new_prediction.actions, old_prediction.actions)
    assert torch.equal(family.predict(batch), old_prediction.actions)
    tokens = batch.action_tokens
    assert tokens is not None
    action_ids = tokens[:, :8]
    memory = old_model.encode(batch.encoder_inputs, with_grad=False)
    old_rows = old_model.ar_teacher_forced_block_logits(
        memory,
        [[int(t) for t in row] for row in action_ids],
    )
    stacked = torch.stack([row for row in old_rows if row is not None])
    new_logits = family.teacher_forced_block_logits(batch, action_ids)
    assert torch.equal(new_logits, stacked)


def test_molmoact2_ar_param_groups_and_state(molmoact2_world: dict[str, Any]) -> None:
    old_model, family, _ = molmoact2_world["ar"]
    assert_param_group_parity(old_model, family, {"decoder": "ar_decoder"})


def test_molmoact2_ar_components_roundtrip(
    molmoact2_world: dict[str, Any],
    tmp_path: Path,
) -> None:
    """The all-parameterless family: checkpoint_components is EMPTY and
    the toolkit still writes/validates (metadata + backbone mirror)."""
    _, family, converted = molmoact2_world["ar"]
    assert family.checkpoint_components() == {}
    assert_components_roundtrip(family, converted, tmp_path / "roundtrip")


def test_molmoact2_joint_losses_bitwise(molmoact2_world: dict[str, Any]) -> None:
    old_model, family, _ = molmoact2_world["joint"]
    assert isinstance(family, MolmoAct2JointVLA)
    assert family.objective.ce_weight == JOINT_CE_WEIGHT
    assert family.objective.insulate_flow is True
    for step, seed in enumerate((35, 36)):
        batch = molmoact2_batch(old_model, 320 + step)
        assert_loss_parity(old_model, family, batch, seed=seed)


def test_molmoact2_joint_predictions_bitwise(molmoact2_world: dict[str, Any]) -> None:
    old_model, family, _ = molmoact2_world["joint"]
    assert isinstance(family, MolmoAct2JointVLA)
    batch = molmoact2_batch(old_model, 322)
    # The deployment path: the flow decoder at the recorded point.
    torch.manual_seed(43)
    old_prediction = old_model.predict_chunk(batch)
    torch.manual_seed(43)
    actions = family.predict(batch)
    assert torch.equal(actions, old_prediction.actions)
    # The AR capability: the family's decode ≡ the rider kernel the old
    # world reaches only by hand (predict_chunk dispatches on the flow
    # decoder; the rider has no old-world predict surface).
    rider = old_model.joint_ce
    assert isinstance(rider, MolmoAct2ARDecoder)
    memory = old_model.encode(batch.encoder_inputs, with_grad=False)
    old_ar = rider.predict_chunk(old_model.backbone, memory, batch)
    new_ar = family.predict_ar(batch)
    assert torch.equal(new_ar.actions, old_ar.actions)


def test_molmoact2_joint_param_groups_and_state(
    molmoact2_world: dict[str, Any],
) -> None:
    old_model, family, _ = molmoact2_world["joint"]
    assert_param_group_parity(
        old_model,
        family,
        {"decoder": "flow_decoder", "joint_ce": "ar_decoder"},
    )


def test_molmoact2_joint_components_roundtrip(
    molmoact2_world: dict[str, Any],
    tmp_path: Path,
) -> None:
    _, family, converted = molmoact2_world["joint"]
    assert_components_roundtrip(family, converted, tmp_path / "roundtrip")


def test_molmoact2_joint_cross_family_consistency(
    molmoact2_world: dict[str, Any],
) -> None:
    """The joint objective's components ARE the single-objective
    families' losses (the oracle cross-check, §11): its action
    component ≡ the flow family's loss and its aux component ≡ the ar
    family's, bitwise, on the same batch and seeds — all four models
    share one trunk and one expert initialization."""
    old_flow, _, _ = molmoact2_world["flow"]
    old_ar, _, _ = molmoact2_world["ar"]
    old_joint, joint_family, _ = molmoact2_world["joint"]
    batch = molmoact2_batch(old_flow, 330)
    counts = joint_family.loss_counts(batch)
    torch.manual_seed(51)
    report = joint_family(batch, counts=counts)
    _, flow_action_sum, _, _ = old_sum_form(old_flow, batch, seed=51)
    assert torch.equal(report.components["action"].sum, flow_action_sum)
    _, ce_action_sum, _, _ = old_sum_form(old_ar, batch, seed=51)
    assert torch.equal(report.components["aux"].sum, ce_action_sum)
    assert old_joint.joint_ce_weight == JOINT_CE_WEIGHT
