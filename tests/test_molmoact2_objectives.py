"""The molmoact2 objective matrix — phase-3 oracles
(docs/molmoact2-retirement.md: decisions 4–5, the gate matrix rows
"KI both ways under joint" and the save/load round-trips).

What this suite pins:

1. the DECISION-5 ordering invariant: the joint arm's flow component
   is BITWISE the flow-only loss — the CE rider's suffix forward
   appends to the shared cache, and the expert's prompt-only KV
   extraction must have happened first (the cache visibly grows, the
   flow term visibly doesn't move);
2. λ composition: total = flow + λ·CE (the logged CE-action read IS
   the CE total — the rider has no aux fields);
3. knowledge insulation under joint, both ways: the insulated flow
   term reaches ZERO trunk parameters while touching the expert; the
   CE term reaches the trunk while touching zero expert parameters
   (disjoint parameter sets — λ is an LR-relative knob only);
4. the joint checkpoint round-trip: a legacy directory with the
   molmo_flow section + the format-6 AR section in the joint_ce slot +
   expert weights → ``convert_legacy`` + ``load_vla`` remount decoder
   AND rider on the joint family (no joint_ce weights file — the rider
   owns nothing);
5. the fresh-section synthesizer (--expert-init fresh from ar-only
   sources): released shape + released serving/t-law constants +
   the ar section's geometry;
6. the Collator's merged action-table override: tokenization and the
   batch stats quantile rows both read the ONE table;
7. TrainArgs validations through the REAL parser (the
   test_train_args harness): family scoping, live-trunk requirements,
   λ rules, --expert-init explicitness and the ar-only-source matrix.

The flow decoder is deterministically PERTURBED out of the adaLN-Zero
vacuum before every gradient/parity read (the test_molmo_flow
lesson: zero-gated outputs make gradient tests pass vacuously).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch
from test_molmoact2_ar import (
    BATCH,
    FAST_FIXTURE,
    Q01,
    Q99,
    D,
    T,
    action_rows,
    batch,
    build_decoder,
    decoder_config,
    prompt_config,
    tiny_stats,
    write_tiny_molmoact2_trunk,
)
from test_train_args import _checkpoint, _parse

from bijou.convert_legacy import CheckpointMetadata, convert
from bijou.data import DatasetStats
from bijou.fast.molmoact2 import MolmoAct2FastTokenizer, QuantileStats
from bijou.loading import (
    BackboneConfig,
    BackboneDepth,
    MolmoFlowDecoderConfig,
    ar_backbone_config_to_dict,
    build_molmo_flow_decoder,
    load_vla,
    molmoact2_fresh_flow_section,
)
from bijou.modelling.codecs import MolmoAct2ActionCodec
from bijou.modelling.decoders.ar_suffix import (
    ar_backbone_loss_sums,
    ar_backbone_losses,
)
from bijou.modelling.decoders.flow import TimeConditioning
from bijou.modelling.decoders.molmo_flow import (
    MolmoFlowDecoder,
    molmo_flow_loss,
    molmo_flow_loss_sums,
)
from bijou.modelling.encoders.molmo2 import Molmo2Memory
from bijou.modelling.encoders.molmoact2 import MolmoAct2Encoder
from bijou.modelling.interface import Collator, SamplingMethod
from bijou.modelling.molmo2.cache import Molmo2KVCache
from bijou.modelling.molmo2.model import Molmo2Model, build_multimodal_mask, load_model
from bijou.models.molmoact2_joint import JointObjective, MolmoAct2JointVLA
from bijou.models.serving import FlowServing

PROMPT_LEN = 9


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_tiny_molmoact2_trunk(
        tmp_path_factory.mktemp("molmoact2-objectives") / "tiny-molmoact2",
    )


@pytest.fixture(scope="module")
def trunk(tiny_checkpoint: Path) -> Molmo2Model:
    return load_model(str(tiny_checkpoint), dtype=torch.float32)


def tiny_flow_section() -> MolmoFlowDecoderConfig:
    """A tiny expert whose num_layers matches the tiny trunk's 6 blocks
    (layer_kv_pairs pins one conditioning pair per expert block) and
    whose llm_kv_dim matches its KV geometry (2 heads × head_dim 16)."""
    return MolmoFlowDecoderConfig(
        max_horizon=T,
        max_action_dim=8,
        hidden_size=32,
        num_layers=6,
        num_heads=2,
        mlp_ratio=2.0,
        ffn_multiple_of=16,
        timestep_embed_dim=16,
        context_layer_norm=True,
        qk_norm=True,
        qk_norm_eps=1e-6,
        rope=True,
        causal_attn=False,
        llm_kv_dim=32,
        num_flow_steps=4,
        mask_action_dim_padding=True,
        action_dim=D,
        action_horizon=T,
        n_action_steps=T,
        normalization="q01q99",
        time_offset=0.001,
        time_scale=0.999,
        beta_alpha=1.0,
        beta_beta=1.5,
    )


def stats_row() -> DatasetStats:
    return dataclasses.replace(
        tiny_stats(),
        action_q01=tuple(Q01.tolist()),
        action_q99=tuple(Q99.tolist()),
    )


def build_flow_decoder() -> MolmoFlowDecoder:
    decoder = build_molmo_flow_decoder(
        tiny_flow_section(),
        device="cpu",
        dtype=torch.float32,
    )
    # Deterministic perturbation out of the adaLN-Zero vacuum — without
    # it the cross-attention outputs are zero-gated and every gradient
    # assertion below would pass vacuously (the test_molmo_flow lesson).
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.02)
    return decoder


def action_quantiles() -> QuantileStats:
    return QuantileStats(
        q01=torch.as_tensor(Q01, dtype=torch.float32),
        q99=torch.as_tensor(Q99, dtype=torch.float32),
    )


def joint_model(
    tiny_checkpoint: Path,
    trunk: Molmo2Model,
    *,
    insulate: bool,
    joint_ce_weight: float = 0.5,
) -> MolmoAct2JointVLA:
    return MolmoAct2JointVLA(
        trunk,
        MolmoAct2Encoder(
            str(tiny_checkpoint),
            setup_type="tabletop",
            control_mode="joint",
            num_state_tokens=32,
            action_mode="both",
            narration=False,
        ),
        build_flow_decoder(),
        build_decoder(tiny_checkpoint),
        action_quantiles=action_quantiles(),
        objective=JointObjective(
            ce_weight=joint_ce_weight,
            insulate_flow=insulate,
        ),
        serving=FlowServing(num_steps=4, method=SamplingMethod.EULER),
    )


def encode_memory_with_grad(trunk: Molmo2Model) -> Molmo2Memory:
    """The text-only prefill of test_molmoact2_ar's builder, WITHOUT the
    no_grad wrapper — the gradient contracts need a live graph from the
    cache back into the trunk."""
    generator = torch.Generator().manual_seed(11)
    prompt = torch.randint(1, 100, (BATCH, PROMPT_LEN), generator=generator)
    transformer = trunk.text.transformer
    embeds = transformer.wte(prompt)
    mask = build_multimodal_mask(
        image_type_mask=torch.zeros(BATCH, PROMPT_LEN, dtype=torch.bool),
        padding_mask=None,
        dtype=embeds.dtype,
        device=embeds.device,
    )
    cache = Molmo2KVCache(len(transformer.blocks))
    transformer(inputs_embeds=embeds, attention_mask=mask, cache=cache)
    return Molmo2Memory(
        cache=cache,
        length=PROMPT_LEN,
        padding_mask=None,
    )


def joint_terms_on(
    model: MolmoAct2JointVLA,
    memory: Molmo2Memory,
    sample: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The joint family's two loss terms in ITS forward's op order on an
    injected memory (the family forward encodes internally, so the
    memory-injection contracts compose its exact kernels): the flow
    branch extracts its prompt-only KV FIRST, then the CE rider's
    suffix forward appends to the same cache. Returns
    (flow_sum, flow_count, ce_sum, ce_count)."""
    flow_sum, flow_count = molmo_flow_loss_sums(
        model.flow_decoder,
        memory,
        actions_norm=model.action_quantiles.normalize(
            sample.actions,  # type: ignore[attr-defined]  # CollatedBatch[FakeInputs]
        ),
        insulate=model.objective.insulate_flow,
    )
    ce_sum, ce_count, ce_aux_sum, _ = ar_backbone_loss_sums(
        model.backbone,
        model.ar_decoder,
        memory,
        sample,  # type: ignore[arg-type]  # CollatedBatch[FakeInputs] — kernel-generic
    )
    assert ce_aux_sum is None  # rider constructed aux-None (format 6)
    return flow_sum, flow_count, ce_sum, ce_count


def test_decision5_ordering_and_lambda_composition(
    tiny_checkpoint: Path,
    trunk: Molmo2Model,
) -> None:
    """The flow component of the joint composition is BITWISE the
    flow-only loss (same seeded τ/ε draws, same prompt-only cache) even
    though the CE rider's forward visibly appends suffix K/V to the
    shared cache — the decision-5 ordering made observable through the
    family's own kernels in its forward's op order. And total =
    flow + λ·CE exactly. (End-to-end on real collation, the joint
    ORACLE's cross-check pins the same fact: loss_action_flow ≡ the flow
    anchor bitwise.)"""
    model = joint_model(tiny_checkpoint, trunk, insulate=False)
    raw, sequences = action_rows()
    sample = batch(raw, sequences)

    with torch.no_grad():
        torch.manual_seed(23)
        flow_only = molmo_flow_loss(
            model.flow_decoder,
            encode_memory_with_grad(trunk),
            actions_norm=model.action_quantiles.normalize(sample.actions),
        )
        memory = encode_memory_with_grad(trunk)
        cache = memory.cache
        assert isinstance(cache, Molmo2KVCache)
        assert cache.seen_tokens == PROMPT_LEN
        torch.manual_seed(23)
        flow_sum, flow_count, ce_sum, ce_count = joint_terms_on(model, memory, sample)
        # The CE forward APPENDED (teacher-forced suffix K/V in the
        # cache) — and the flow term did not move: it extracted first.
        assert cache.seen_tokens > PROMPT_LEN
        flow_component = flow_sum / flow_count
        assert torch.equal(flow_component, flow_only)
        # The CE branch on the SHARED (post-extraction) cache is bitwise
        # the CE branch on a fresh prefill — same-form (sum) comparison;
        # the mean-form reference below pins the λ composition across
        # forms at assert_close tolerance.
        reference_ce_sum, reference_ce_count, _, _ = ar_backbone_loss_sums(
            trunk,
            model.ar_decoder,
            encode_memory_with_grad(trunk),
            sample,
        )
        assert torch.equal(ce_sum, reference_ce_sum)
        assert torch.equal(ce_count, reference_ce_count)
        reference_ce, _, _, _ = ar_backbone_losses(
            trunk,
            model.ar_decoder,
            encode_memory_with_grad(trunk),
            sample,
        )
        total = flow_component + model.objective.ce_weight * (ce_sum / ce_count)
        torch.testing.assert_close(
            total,
            flow_only + model.objective.ce_weight * reference_ce,
        )


def test_ki_under_joint_reaches_disjoint_parameter_sets(
    tiny_checkpoint: Path,
    trunk: Molmo2Model,
) -> None:
    """KI both ways (the gate-matrix row): the insulated flow term
    backwards into ZERO trunk parameters and a NONZERO expert set; the
    CE term backwards into a NONZERO trunk set and zero expert
    parameters; the composed joint loss reaches both."""
    # load_model mounts frozen (the eval convention; train.py unfreezes
    # per the LR flags) — these are LIVE-trunk contracts.
    trunk.requires_grad_(True)
    model = joint_model(tiny_checkpoint, trunk, insulate=True)
    raw, sequences = action_rows()
    sample = batch(raw, sequences)
    flow_decoder = model.flow_decoder
    trunk_parameters = [p for p in trunk.parameters() if p.requires_grad]
    expert_parameters = [p for p in flow_decoder.parameters() if p.requires_grad]
    assert len(list(model.ar_decoder.parameters())) == 0  # the parameterless rider

    def zero_grads() -> None:
        for parameter in (*trunk_parameters, *expert_parameters):
            parameter.grad = None

    def gradient_norms(parameters: list[torch.nn.Parameter]) -> float:
        return sum(float(p.grad.abs().sum()) for p in parameters if p.grad is not None)

    # Flow term, insulated: expert-only.
    zero_grads()
    torch.manual_seed(31)
    flow = molmo_flow_loss(
        flow_decoder,
        encode_memory_with_grad(trunk),
        actions_norm=action_quantiles().normalize(sample.actions),
        insulate=True,
    )
    flow.backward()
    assert gradient_norms(trunk_parameters) == 0.0
    assert gradient_norms(expert_parameters) > 0.0

    # CE term: trunk-only (the rider owns nothing).
    zero_grads()
    ce, _, _, _ = ar_backbone_losses(
        trunk,
        model.ar_decoder,
        encode_memory_with_grad(trunk),
        sample,
    )
    ce.backward()
    assert gradient_norms(trunk_parameters) > 0.0
    assert gradient_norms(expert_parameters) == 0.0

    # The composed joint loss (the family forward's term composition)
    # reaches both sets.
    zero_grads()
    torch.manual_seed(31)
    flow_sum, flow_count, ce_sum, ce_count = joint_terms_on(
        model,
        encode_memory_with_grad(trunk),
        sample,
    )
    total = flow_sum / flow_count + model.objective.ce_weight * (ce_sum / ce_count)
    total.backward()
    assert gradient_norms(trunk_parameters) > 0.0
    assert gradient_norms(expert_parameters) > 0.0


def test_uninsulated_flow_reaches_the_trunk(
    tiny_checkpoint: Path,
    trunk: Molmo2Model,
) -> None:
    """The complement that keeps the KI assertion honest: with the seam
    OPEN, the flow term alone does reach trunk parameters through the
    extracted K/V."""
    trunk.requires_grad_(True)
    raw, sequences = action_rows()
    sample = batch(raw, sequences)
    flow_decoder = build_flow_decoder()
    trunk_parameters = [p for p in trunk.parameters() if p.requires_grad]
    for parameter in trunk_parameters:
        parameter.grad = None
    torch.manual_seed(37)
    flow = molmo_flow_loss(
        flow_decoder,
        encode_memory_with_grad(trunk),
        actions_norm=action_quantiles().normalize(sample.actions),
        insulate=False,
    )
    flow.backward()
    assert (
        sum(float(p.grad.abs().sum()) for p in trunk_parameters if p.grad is not None)
        > 0.0
    )


def test_joint_checkpoint_roundtrip(
    tiny_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """A POSSIBLE legacy joint checkpoint (molmo_flow section + the
    format-6 section in joint_ce + expert.safetensors, NO joint_ce
    weights file) converts and loads as the joint family: the converter
    infers molmoact2_joint off the recorded objective, and
    ``load_vla`` rebuilds the flow decoder BITWISE and mounts the
    parameterless rider with the recorded payload."""
    from safetensors.torch import save_file

    directory = tmp_path / "joint"
    directory.mkdir()
    source = build_flow_decoder()
    save_file(
        {k: v.contiguous() for k, v in source.state_dict().items()},
        str(directory / "expert.safetensors"),
    )
    metadata = CheckpointMetadata(
        backbone=BackboneConfig(id=str(tiny_checkpoint), depth=BackboneDepth.FULL),
        prompt=prompt_config(),
        decoder=tiny_flow_section().to_dict(),
        joint_ce=ar_backbone_config_to_dict(decoder_config()),
        normalization=stats_row(),
        per_dataset_normalization={"marius/rig": stats_row()},
        train_args={
            "decoder": "molmo_flow",
            "objective": "joint",
            "joint_ce_weight": 0.5,
            "decoder_hidden": 32,
            "decoder_heads": 2,
            "decoder_intermediate": 64,
            "decoder_cross_heads": 2,
            "stream_counts": [],
            "self_attention_mode": "bidirectional",
            "chunk_size": T,
            "max_soft_tokens": 140,
            "max_crops": 1,
            "time_conditioning": "additive",
            "target_time_embed": False,
            "fast_tokenizer": None,
        },
        step=0,
    )
    (directory / "bijou_config.json").write_text(json.dumps(metadata.to_json_dict()))
    converted = tmp_path / "converted"
    convert(directory, converted)
    model = load_vla(converted, device="cpu", dtype=torch.float32)
    assert isinstance(model, MolmoAct2JointVLA)
    assert model.ar_decoder.config == decoder_config()
    assert len(list(model.ar_decoder.parameters())) == 0
    assert model.objective == JointObjective(ce_weight=0.5, insulate_flow=False)
    loaded_state = model.flow_decoder.state_dict()
    for key, tensor in source.state_dict().items():
        assert torch.equal(loaded_state[key].cpu(), tensor)


def test_fresh_flow_section_synthesizer() -> None:
    """--expert-init fresh from an ar-only source: released shape +
    released serving/t-law constants, geometry from the format-6
    section."""
    section = molmoact2_fresh_flow_section(decoder_config())
    assert (section.hidden_size, section.num_layers) == (768, 36)
    assert (section.action_horizon, section.n_action_steps) == (T, T)
    assert section.action_dim == D
    assert section.num_flow_steps == 10
    assert (section.time_offset, section.time_scale) == (0.001, 0.999)
    assert (section.beta_alpha, section.beta_beta) == (1.0, 1.5)
    assert section.llm_kv_dim == 1024  # the release trunk's KV width
    with pytest.raises(SystemExit, match="exceeds the released expert"):
        molmoact2_fresh_flow_section(
            dataclasses.replace(decoder_config(), chunk_size=99),
        )


def test_collator_merged_action_table_override() -> None:
    """CE tokenization reads the ONE merged table; per-item quantiles
    are deliberately unused (items without them tokenize fine under the
    override) — and the batch stats stay HONEST per-item rows: the
    table never reaches them (decode-side tables are the family's
    quantile table, never batch state)."""
    codec = MolmoAct2ActionCodec(
        MolmoAct2FastTokenizer.load(FAST_FIXTURE),
        time_horizon=T,
        action_dim=D,
    )
    table_q01 = torch.tensor(Q01, dtype=torch.float32)
    table_q99 = torch.tensor(Q99, dtype=torch.float32)
    collator = Collator(
        inputs=None,  # type: ignore[arg-type]  # _action_tokens/_stats never touch it
        instruction=None,
        camera_filter=None,
        max_cameras=None,
        action_codec=codec,
        aux=None,
        generate_bracket=False,
        generate_override=None,
        camera_kind_dropout=0.0,
        instruction_augment=0.0,
        condition_fields=(),
        condition_dropout=0.0,
        subgoal_condition_dropout=0.0,
        action_q01=table_q01,
        action_q99=table_q99,
    )
    raw, sequences = action_rows()
    items = [
        {
            "repo_id": "marius/rig",
            "action": raw[i].float(),
            "action_mean": torch.zeros(D),
            "action_std": torch.ones(D),
            # NO per-item quantiles: the override is the table.
        }
        for i in range(BATCH)
    ]
    tokens = collator._action_tokens(items)
    assert tokens is not None
    for row, sequence in enumerate(sequences):
        assert tokens[row, : len(sequence)].tolist() == sequence
        assert bool((tokens[row, len(sequence) :] == codec.pad).all())
    stats = collator._stats(items, "action")
    # No per-item quantiles on the items ⇒ none on the batch: the merged
    # table is tokenization-side only and never masquerades as stats.
    assert stats.q01 is None and stats.q99 is None
    assert torch.equal(stats.mean, torch.zeros(BATCH, D))
    assert torch.equal(stats.std, torch.ones(BATCH, D))


def molmoact2_source(**overrides: object):  # noqa: ANN201  # test_train_args' fabricator type
    # The shared fabricator picks ADARMS/BIDIRECTIONAL to differ from
    # ARCH defaults; the molmoact2 families refuse the gemma_flow-only
    # τ knob — their sources record additive (the converter's
    # synthesized train_args).
    from bijou.modelling.decoders.flow import SelfAttentionMode
    from bijou.vla import VLAFamily

    facts: dict[str, object] = {
        "family": VLAFamily.MOLMOACT2_FLOW,
        "decoder": "molmo_flow",
        "chunk_size": T,
        "time_conditioning": TimeConditioning.ADDITIVE,
        "self_attention_mode": SelfAttentionMode.BIDIRECTIONAL,
        **overrides,
    }
    return _checkpoint(**facts)


def test_objective_cli_validations() -> None:
    """The pathway matrix through the REAL parser (new-CLI names): the
    --objective transition selects the family under --init-from, the
    live-trunk/λ/insulation rules ride the resolved family, and
    --flow-decoder-init keeps --expert-init's explicitness rules."""
    from bijou.vla import VLAFamily

    init = ["--init-from", "ckpt"]
    # Family scoping: --objective selects a molmoact2 pathway.
    with pytest.raises(SystemExit):
        _parse(["--family", "gemma_flow", "--objective", "ar"])  # fresh run
    # ar/joint require a live trunk (the head owns no parameters).
    with pytest.raises(SystemExit):
        _parse([*init, "--objective", "ar"], molmoact2_source())
    args = _parse(
        [*init, "--objective", "ar", "--backbone-text-lr", "1e-5"],
        molmoact2_source(),
    )
    assert args.family == "molmoact2_ar"
    # λ rules: joint-only, > 0.
    with pytest.raises(SystemExit):
        _parse(
            [*init, "--objective", "flow", "--joint-ce-weight", "0.5"],
            molmoact2_source(),
        )
    with pytest.raises(SystemExit):
        _parse(
            [
                *init,
                "--objective",
                "joint",
                "--backbone-text-lr",
                "1e-5",
                "--joint-ce-weight",
                "0",
            ],
            molmoact2_source(),
        )
    joint = _parse(
        [
            *init,
            "--objective",
            "joint",
            "--backbone-text-lr",
            "1e-5",
            "--joint-ce-weight",
            "0.25",
        ],
        molmoact2_source(),
    )
    assert (joint.family, joint.joint_ce_weight) == ("molmoact2_joint", 0.25)
    # --flow-decoder-init: --init-from only; never with ar (nothing to
    # initialize); refused on resume.
    with pytest.raises(SystemExit):
        _parse(
            [
                *init,
                "--objective",
                "ar",
                "--backbone-text-lr",
                "1e-5",
                "--flow-decoder-init",
                "fresh",
            ],
            molmoact2_source(),
        )
    with pytest.raises(SystemExit):
        _parse(
            ["--resume", "ckpt", "--flow-decoder-init", "fresh"],
            molmoact2_source(),
        )
    # Resume inherits the recorded family (the ARCH_FLAGS refusal
    # covers explicit passing).
    resumed = _parse(
        ["--resume", "ckpt", "--backbone-text-lr", "1e-5"],
        molmoact2_source(
            family=VLAFamily.MOLMOACT2_JOINT,
            objective={"kind": "joint", "ce_weight": 1.0, "insulate_flow": True},
        ),
    )
    assert resumed.family == "molmoact2_joint"
    with pytest.raises(SystemExit):
        _parse(["--resume", "ckpt", "--objective", "joint"], molmoact2_source())
    # KI matrix: flow+live trunk refused with the joint remedy; joint OK.
    with pytest.raises(SystemExit):
        _parse(
            [
                *init,
                "--objective",
                "flow",
                "--backbone-text-lr",
                "1e-5",
                "--insulate-flow",
            ],
            molmoact2_source(),
        )
    ki = _parse(
        [
            *init,
            "--objective",
            "joint",
            "--backbone-text-lr",
            "1e-5",
            "--insulate-flow",
        ],
        molmoact2_source(),
    )
    assert ki.insulate_flow and ki.family == "molmoact2_joint"
    with pytest.raises(SystemExit):
        _parse(
            [
                *init,
                "--objective",
                "ar",
                "--backbone-text-lr",
                "1e-5",
                "--insulate-flow",
            ],
            molmoact2_source(),
        )
