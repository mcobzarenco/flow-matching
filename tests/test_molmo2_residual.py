"""Molmo2 residual-tap conditioning + attachment-seam oracles (the
attach-screen pre-reg, 2026-08-07) — CPU, tiny molmo2 fixture:

(i)   the pinned tap rule (stride 3, last tap = final layer; 4B: 12 taps
      at 2, 5, …, 35) and its expert-config derivation;
(ii)  taps byte-match the trunk's post-layer hidden states at the pinned
      indices, and an encode with taps leaves the trunk's own product
      (the prefix KV cache) bit-identical to one without;
(iii) the attached streams obey the K/V contract (shapes, padding
      invariance at real columns) and raw taps are consumed and dropped;
(iv)  the seam stop-grad flag: flow-loss gradients into EVERY trunk
      parameter exactly zero with the flag, nonzero without (naive joint
      exists only as this negative control);
(v)   the joint CE+flow step at the α-edges: the flow half reproduces
      the F-arm step bit-for-bit (loss value + expert/adapter grads),
      and under stop-grad the trunk gradients reproduce a phase-1
      ar_backbone step's bitwise;
(vi)  the config schema round-trips (prompt + decoder → rebuilt expert
      config → strict state-dict load), with pre-field back-compat.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch
from test_molmo2_ar import (
    batch,
    build_decoder,
    codec,
    text_config,
    tiny_inputs,
)

from bijou.decoders.flow import ExpertConfig, FlowDecoder, TimeConditioning
from bijou.encoders.molmo2 import Molmo2Encoder, Molmo2Inputs
from bijou.interface import CollatedBatch
from bijou.loading import (
    FlowDecoderConfig,
    Molmo2PromptConfig,
    expert_config_from_architecture,
    flow_decoder_config_from_expert,
    molmo2_residual_expert_config,
    molmo2_residual_taps,
    parse_decoder_config,
)
from bijou.model import BijouModel
from bijou.molmo2.cache import Molmo2KVCache
from bijou.molmo2.model import Molmo2Model, build_multimodal_mask, load_model
from bijou.molmo2.testing import write_tiny_text_checkpoint
from bijou.train import BijouTrainStep

BATCH = 2
STATE_DIM = codec().action_dim  # the tiny FAST fixture's action space


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_tiny_text_checkpoint(
        tmp_path_factory.mktemp("molmo2-residual") / "tiny-molmo2",
    )


def fresh_model(tiny_checkpoint: Path) -> Molmo2Model:
    """A fresh trunk per test — the seam tests mutate requires_grad and
    accumulate gradients, so the module-scope sharing the AR suite uses
    would leak state between oracles."""
    return load_model(str(tiny_checkpoint), dtype=torch.float32)


def expert_config() -> ExpertConfig:
    loaded = codec()
    return molmo2_residual_expert_config(
        text_config(),
        action_dim=loaded.action_dim,
        state_dim=loaded.action_dim,
        hidden_size=32,
        num_attention_heads=2,
        intermediate_size=64,
        cross_attention_heads=2,
        chunk_size=loaded.time_horizon,
        time_embed_dim=8,
        time_conditioning=TimeConditioning.ADARMS,
    )


def build_residual_encoder(
    checkpoint: Path,
    *,
    taps: tuple[int, ...],
    seed: int = 1,
) -> Molmo2Encoder:
    torch.manual_seed(seed)
    return Molmo2Encoder(
        str(checkpoint),
        max_crops=1,
        state_dim=codec().action_dim,
        hidden_size=text_config().text.hidden_size,
        residual_exports=taps,
    )


def build_flow_model(
    checkpoint: Path,
    backbone: Molmo2Model,
    *,
    decoder_seed: int = 2,
) -> BijouModel:
    config = expert_config()
    encoder = build_residual_encoder(checkpoint, taps=config.streams)
    torch.manual_seed(decoder_seed)
    decoder = FlowDecoder(config, device="cpu", dtype=torch.float32)
    return BijouModel(backbone=backbone, encoder=encoder, decoder=decoder)


def perturb_zero_init_heads(decoder: FlowDecoder) -> None:
    """At TRUE init the zero-initialized heads (action_out_proj, adaRMS
    gates) make every upstream gradient exactly zero — perturb to a
    mid-training state so gradient PATHS are what the seam tests see
    (the gemma residual suite's convention)."""
    torch.manual_seed(7)
    torch.nn.init.normal_(decoder.action_out_proj.weight, std=0.02)
    for module in decoder.modules():
        if isinstance(module, torch.nn.Linear) and bool((module.weight == 0).all()):
            torch.nn.init.normal_(module.weight, std=0.02)


def flow_batch() -> CollatedBatch[Molmo2Inputs]:
    return batch(codec(), tiny_inputs())


# -- (i) the pinned tap rule --------------------------------------------------


def test_pinned_tap_rule() -> None:
    """The pre-registered molmo2 rule: uniform stride 3, last tap = the
    final layer. 4B (36 layers) ⇒ exactly 12 taps at 2, 5, …, 35; the
    tiny fixture (6 layers) ⇒ (2, 5)."""
    assert molmo2_residual_taps(36) == tuple(range(2, 36, 3))
    assert len(molmo2_residual_taps(36)) == 12
    assert molmo2_residual_taps(6) == (2, 5)


def test_expert_config_mirrors_trunk_geometry() -> None:
    config = expert_config()
    text = text_config().text
    assert config.residual_streams
    assert config.cross_attention_schedule == molmo2_residual_taps(
        text.num_hidden_layers,
    )
    assert config.residual_stream_dim == text.hidden_size
    assert config.cross_attention_kv_heads == text.num_key_value_heads
    assert config.cross_attention_head_dim == text.head_dim
    assert config.cross_attention_rope.rope_theta == text.rope_theta


# -- (ii) taps byte-match the trunk -------------------------------------------


def test_taps_byte_match_trunk_hidden_states(tiny_checkpoint: Path) -> None:
    """Encoder taps ≡ the transformer's own residual stream at the same
    indices: a reference forward with a sink over EVERY layer, fed the
    identical embeddings/mask/positions the encoder builds, must match
    the encoder's exported taps bitwise."""
    backbone = fresh_model(tiny_checkpoint)
    encoder = build_residual_encoder(tiny_checkpoint, taps=(2, 5))
    inputs = tiny_inputs()
    with torch.no_grad():
        memory = encoder.encode(backbone, inputs, with_grad=False)
    assert memory.residuals is not None
    assert sorted(memory.residuals) == ["res2", "res5"]

    with torch.no_grad():
        embeds = backbone.build_input_embeddings(
            inputs.input_ids,
            crops=inputs.crops,
            pooled_patches_idx=inputs.pooled_patches_idx,
        )
        embeds[:, inputs.state_slot, :] = encoder.state_proj(
            inputs.state.to(encoder.state_proj.weight.dtype),
        ).to(embeds.dtype)
        mask = build_multimodal_mask(
            image_type_mask=inputs.image_type_mask,
            padding_mask=inputs.attention_mask,
            dtype=embeds.dtype,
            device=embeds.device,
        )
        sink: dict[int, torch.Tensor] = {}
        backbone.text.transformer(
            inputs_embeds=embeds,
            position_ids=Molmo2Model.logical_positions(inputs.attention_mask),
            attention_mask=mask,
            residual_taps=range(text_config().text.num_hidden_layers),
            residual_sink=sink,
        )
    for tap in (2, 5):
        assert torch.equal(memory.residuals[f"res{tap}"], sink[tap]), (
            f"tap {tap} diverges from the trunk's own residual stream"
        )
    # The taps are pre-ln_f: the final tap must NOT equal the normed
    # output (which would indicate an off-by-one against the contract).
    assert not torch.equal(
        memory.residuals["res5"],
        backbone.text.transformer.ln_f(sink[5]),
    )


def test_taps_leave_trunk_product_bit_identical(tiny_checkpoint: Path) -> None:
    """Tap recording is a pure read: the prefix KV cache (the trunk's
    product for the suffix role) is bit-identical with and without taps
    enabled."""
    backbone = fresh_model(tiny_checkpoint)
    with_taps = build_residual_encoder(tiny_checkpoint, taps=(2, 5))
    without = build_residual_encoder(tiny_checkpoint, taps=())
    without.load_state_dict(with_taps.state_dict())
    inputs = tiny_inputs()
    with torch.no_grad():
        tapped = with_taps.encode(
            backbone,
            inputs,
            with_grad=False,
            retain_cache=True,
        )
        plain = without.encode(
            backbone,
            inputs,
            with_grad=False,
            retain_cache=True,
        )
    assert plain.residuals is None
    assert isinstance(tapped.cache, Molmo2KVCache)
    assert isinstance(plain.cache, Molmo2KVCache)
    for i, (a, b) in enumerate(
        zip(tapped.cache.layers, plain.cache.layers, strict=True),
    ):
        assert a.keys is not None and b.keys is not None
        assert a.values is not None and b.values is not None
        assert torch.equal(a.keys, b.keys), f"layer {i} keys drifted"
        assert torch.equal(a.values, b.values), f"layer {i} values drifted"


# -- (iii) stream contract ----------------------------------------------------


def test_attached_streams_carry_kv_contract(tiny_checkpoint: Path) -> None:
    model = build_flow_model(tiny_checkpoint, fresh_model(tiny_checkpoint))
    with torch.no_grad():
        memory = model.encode(tiny_inputs(), with_grad=False)
    assert memory.residuals is None  # attached and dropped
    assert sorted(memory.streams) == ["res2", "res5"]
    text = text_config().text
    length = tiny_inputs().input_ids.shape[1]
    for stream in memory.streams.values():
        assert stream.key.shape == (
            BATCH,
            text.num_key_value_heads,
            length,
            text.head_dim,
        )
        assert stream.value.shape == stream.key.shape


def test_attached_streams_padding_invariant_at_real_columns(
    tiny_checkpoint: Path,
) -> None:
    """Row 1 padded-in-batch vs solo-unpadded: attached K/V at REAL
    columns must agree — true only if adapter keys are RoPE'd at
    per-sample LOGICAL positions, mirroring the trunk's own K/V (the
    property the gemma streams are gated on, on molmo2's left-padded
    collation)."""
    model = build_flow_model(tiny_checkpoint, fresh_model(tiny_checkpoint))
    inputs = tiny_inputs()
    real = inputs.attention_mask[1].bool()
    padded = dataclasses.replace(
        inputs,
        input_ids=inputs.input_ids[1:],
        attention_mask=inputs.attention_mask[1:],
        image_type_mask=inputs.image_type_mask[1:],
        crops=inputs.crops[1:],
        pooled_patches_idx=inputs.pooled_patches_idx[1:],
        state=inputs.state[1:],
        has_padding=True,
    )
    solo = dataclasses.replace(
        padded,
        input_ids=inputs.input_ids[1:, real],
        attention_mask=inputs.attention_mask[1:, real],
        image_type_mask=inputs.image_type_mask[1:, real],
        has_padding=False,
    )
    with torch.no_grad():
        padded_memory = model.encode(padded, with_grad=False)
        solo_memory = model.encode(solo, with_grad=False)
    for name in padded_memory.streams:
        for field in ("key", "value"):
            a = getattr(padded_memory.streams[name], field)[0][:, real, :]
            b = getattr(solo_memory.streams[name], field)[0]
            delta = float((a - b).abs().max())
            assert delta < 1e-4, f"{name}.{field}: padding-dependent, |Δ|={delta}"


# -- (iv) the seam stop-grad flag ---------------------------------------------


def seam_flow_backward(
    tiny_checkpoint: Path,
    *,
    stop_grad: bool,
) -> BijouModel:
    """Live-trunk encode + flow-loss backward under the given seam mode;
    returns the model for gradient inspection."""
    backbone = fresh_model(tiny_checkpoint)
    model = build_flow_model(tiny_checkpoint, backbone)
    decoder = model.decoder
    assert isinstance(decoder, FlowDecoder)
    perturb_zero_init_heads(decoder)
    model.seam_stop_grad = stop_grad
    for parameter in model.encoder.param_groups(backbone)["text"]:
        parameter.requires_grad_(True)
    memory = model.encode(tiny_inputs(), with_grad=True)
    generator = torch.Generator().manual_seed(3)
    loaded = codec()
    state = torch.randn(BATCH, STATE_DIM, generator=generator)
    noisy = torch.randn(
        BATCH,
        loaded.time_horizon,
        loaded.action_dim,
        generator=generator,
    )
    time = torch.rand(BATCH, generator=generator)
    decoder(memory, state, noisy, time).square().mean().backward()
    return model


def test_seam_stop_grad_cuts_flow_gradients_to_trunk(
    tiny_checkpoint: Path,
) -> None:
    """WITH the flag: flow-loss gradients into every trunk parameter are
    exactly zero (absent), while every adapter still trains. WITHOUT it
    (the naive-joint negative control): flow gradients reach the trunk."""
    model = seam_flow_backward(tiny_checkpoint, stop_grad=True)
    for name, parameter in model.backbone.named_parameters():
        assert parameter.grad is None or not parameter.grad.any(), (
            f"flow gradient leaked through the stop-grad seam into {name}"
        )
    decoder = model.decoder
    assert isinstance(decoder, FlowDecoder)
    assert decoder.res_adapters is not None
    for name, adapter in decoder.res_adapters.items():
        for parameter_name, parameter in adapter.named_parameters():
            assert parameter.grad is not None, f"{name}.{parameter_name}: no grad"
            assert float(parameter.grad.abs().max()) > 0, (
                f"{name}.{parameter_name}: zero gradient under stop-grad — "
                "the adapters must keep training from the detached taps"
            )

    naive = seam_flow_backward(tiny_checkpoint, stop_grad=False)
    reached = [
        name
        for name, parameter in naive.backbone.named_parameters()
        if parameter.grad is not None and bool(parameter.grad.any())
    ]
    assert reached, (
        "naive joint (no stop-grad) must route flow gradients into the "
        "trunk — the negative control is what makes the flag's oracle real"
    )


# -- (v) the joint step at the α-edges ----------------------------------------


def build_joint_model(
    tiny_checkpoint: Path,
    *,
    stop_grad: bool = True,
) -> BijouModel:
    backbone = fresh_model(tiny_checkpoint)
    model = build_flow_model(tiny_checkpoint, backbone)
    decoder = model.decoder
    assert isinstance(decoder, FlowDecoder)
    perturb_zero_init_heads(decoder)
    rider, _ = build_decoder(backbone)
    model.joint_ce = rider
    model.seam_stop_grad = stop_grad
    for parameter in model.encoder.param_groups(backbone)["text"]:
        parameter.requires_grad_(True)
    return model


def test_joint_param_groups_route_the_rider_at_decoder_lr(
    tiny_checkpoint: Path,
) -> None:
    model = build_joint_model(tiny_checkpoint)
    assert model.joint_ce is not None
    rider_params = {id(p) for p in model.joint_ce.parameters()}
    decoder_group = {id(p) for p in model.param_groups()["decoder"]}
    assert rider_params <= decoder_group, (
        "the CE rider's tables must ride the decoder LR group — phase 1 "
        "trained them there, and 'continuing verbatim' includes routing"
    )


def test_joint_step_flow_half_is_the_f_arm_step(tiny_checkpoint: Path) -> None:
    """α-edge 1: the joint step's flow component (loss value AND the
    expert/adapter gradients, which only the flow term touches) is the
    F-arm step bit-for-bit on identical weights."""
    sample = flow_batch()

    f_model = build_flow_model(tiny_checkpoint, fresh_model(tiny_checkpoint))
    f_decoder = f_model.decoder
    assert isinstance(f_decoder, FlowDecoder)
    perturb_zero_init_heads(f_decoder)
    f_step = BijouTrainStep(f_model, backbone_trained=False)
    # Sum-form (the shape both arms run on the box under
    # --backward-chunks); a single chunk with its own counts as the
    # full-step normalizers is the unchunked step exactly.
    f_norm = f_model.loss_count_normalizers(sample)
    torch.manual_seed(11)  # the flow objective draws τ and ε
    f_loss, f_action_sum, _, _ = f_step(sample, normalizers=(f_norm[0], f_norm[1]))
    f_loss.backward()

    joint = build_joint_model(tiny_checkpoint, stop_grad=True)
    joint_decoder = joint.decoder
    assert isinstance(joint_decoder, FlowDecoder)
    joint_decoder.load_state_dict(f_decoder.state_dict())
    joint.encoder.load_state_dict(f_model.encoder.state_dict())
    joint_step = BijouTrainStep(joint, backbone_trained=True)
    torch.manual_seed(11)  # identical τ/ε draws
    joint_loss, joint_flow_sum, ce_action_sum, ce_action_count = joint_step(sample)
    joint_loss.backward()

    # Loss value: the unchunked joint returns the flow MEAN (sum/count);
    # normalize F's sum by the same count for the bitwise comparison.
    assert torch.equal(joint_flow_sum, f_action_sum / f_norm[0])
    assert ce_action_sum is not None and ce_action_count is not None
    # Expert + adapter gradients: only the flow term touches them, so
    # the joint step must reproduce the F step exactly.
    f_params = dict(f_decoder.named_parameters())
    for name, parameter in joint_decoder.named_parameters():
        f_grad, j_grad = f_params[name].grad, parameter.grad
        assert (f_grad is None) == (j_grad is None), name
        if f_grad is not None and j_grad is not None:
            assert torch.equal(f_grad, j_grad), (
                f"expert gradient {name} diverges between the joint step's "
                "flow half and the F-arm step"
            )


def test_joint_step_trunk_gradients_are_phase_1_ce(tiny_checkpoint: Path) -> None:
    """α-edge 2: under the stop-grad seam the joint step's TRUNK
    gradients come from the CE branch alone and reproduce a phase-1
    ar_backbone step's bitwise; without the seam (naive joint) they
    diverge — the negative control."""
    sample = flow_batch()

    reference_backbone = fresh_model(tiny_checkpoint)
    ref_decoder, _ = build_decoder(reference_backbone)
    ref_encoder = build_residual_encoder(tiny_checkpoint, taps=())
    reference = BijouModel(
        backbone=reference_backbone,
        encoder=ref_encoder,
        decoder=ref_decoder,
    )
    for parameter in ref_encoder.param_groups(reference_backbone)["text"]:
        parameter.requires_grad_(True)
    ref_step = BijouTrainStep(reference, backbone_trained=True)
    ref_norm = reference.loss_count_normalizers(sample)
    ref_loss, _, _, _ = ref_step(sample, normalizers=(ref_norm[0], ref_norm[1]))
    ref_loss.backward()
    ref_grads = {
        name: parameter.grad
        for name, parameter in reference_backbone.named_parameters()
    }

    joint = build_joint_model(tiny_checkpoint, stop_grad=True)
    joint.encoder.load_state_dict(ref_encoder.state_dict())
    torch.manual_seed(11)
    joint_loss, _, _, _ = BijouTrainStep(joint, backbone_trained=True)(sample)
    joint_loss.backward()
    for name, parameter in joint.backbone.named_parameters():
        ref_grad = ref_grads[name]
        assert (ref_grad is None) == (parameter.grad is None), name
        if ref_grad is not None and parameter.grad is not None:
            assert torch.equal(ref_grad, parameter.grad), (
                f"trunk gradient {name}: the joint step under stop-grad "
                "must reproduce the phase-1 CE step exactly"
            )

    naive = build_joint_model(tiny_checkpoint, stop_grad=False)
    naive.encoder.load_state_dict(ref_encoder.state_dict())
    torch.manual_seed(11)
    naive_loss, _, _, _ = BijouTrainStep(naive, backbone_trained=True)(sample)
    naive_loss.backward()
    diverged = any(
        parameter.grad is not None
        and (ref_grad := ref_grads[name]) is not None
        and not torch.equal(ref_grad, parameter.grad)
        for name, parameter in naive.backbone.named_parameters()
    )
    assert diverged, (
        "naive joint must add flow gradients on top of CE in the trunk — "
        "identical gradients would mean the seam flag is a no-op"
    )


# -- (vi) config round-trip ---------------------------------------------------


def test_molmo2_residual_config_roundtrips(tiny_checkpoint: Path) -> None:
    config = expert_config()
    prompt = Molmo2PromptConfig(
        max_crops=1,
        format=1,
        state_dim=STATE_DIM,
        condition_fields=(),
        generate_bracket=True,
        residual_exports=config.streams,
    )
    decoder_schema = flow_decoder_config_from_expert(config)
    assert list(decoder_schema.schedule) == ["res2", "res5"]
    prompt_back = Molmo2PromptConfig.from_dict(
        json.loads(json.dumps(prompt.to_dict())),
    )
    decoder_back = parse_decoder_config(
        json.loads(json.dumps(decoder_schema.to_dict())),
    )
    assert isinstance(decoder_back, FlowDecoderConfig)
    assert prompt_back == prompt
    rebuilt = expert_config_from_architecture(
        prompt_back,
        decoder_back,
        text_config(),
    )
    assert rebuilt == config
    torch.manual_seed(1)
    trained = FlowDecoder(config, device="cpu", dtype=torch.float32)
    fresh = FlowDecoder(rebuilt, device="cpu", dtype=torch.float32)
    fresh.load_state_dict(trained.state_dict(), strict=True)


def test_prompt_config_without_residual_field_defaults_empty() -> None:
    """AR-phase checkpoints written before the field existed load
    unchanged."""
    prompt = Molmo2PromptConfig(
        max_crops=1,
        format=1,
        state_dim=STATE_DIM,
        condition_fields=(),
        generate_bracket=True,
        residual_exports=(2, 5),
    )
    data = prompt.to_dict()
    del data["residual_exports"]
    assert Molmo2PromptConfig.from_dict(data).residual_exports == ()


def test_kv_schedule_names_are_refused() -> None:
    """Molmo2 is residual-only: a decoder schedule naming K/V streams is
    a config error, not a silent remap."""
    config = expert_config()
    prompt = Molmo2PromptConfig(
        max_crops=1,
        format=1,
        state_dim=STATE_DIM,
        condition_fields=(),
        generate_bracket=True,
        residual_exports=config.streams,
    )
    decoder_schema = flow_decoder_config_from_expert(config)
    mixed = type(decoder_schema).from_dict(
        {**decoder_schema.to_dict(), "schedule": ["kv2", "res5"]},
    )
    with pytest.raises(SystemExit, match="unknown stream"):
        expert_config_from_architecture(prompt, mixed, text_config())
