"""Oracles for the MolmoAct2 → bijou checkpoint converter (§8.13 step 2).

CPU tier over a fabricated tiny source in their exact layout (keys
measured on the real artifacts 2026-08-11: released SO-100/101 =
action_mode 'both', rig-ft exports = 'continuous'; norm_stats
metadata_by_tag rows carry mean/std/q01/q99 + setup/control/horizon).
The real-checkpoint gate (released + one rig-ft rung, 21 GB sources)
is a box run, pre-registered in architecture.md §8.13.

Pinned here:

- happy path: sections round-trip through bijou.loading
  (read_checkpoint_info), expert bytes verbatim (digest match source),
  deterministic + idempotent output;
- the stored tensor names are the PORT module's state_dict names minus
  the loader-injected compat tensors — the step-3 decoder adopts the
  same names, so this is the contract that makes expert.safetensors
  load strictly there;
- P2 guards at convert time: nonzero expert dropout, missing/≠1
  n_obs_steps, unsupported action_mode/state_format, missing stats
  rows, missing tokenizer;
- model assembly refuses loudly until step 5 (from_checkpoint guard).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from bijou.convert_molmoact2 import convert
from bijou.loading import (
    MolmoAct2PromptConfig,
    MolmoFlowDecoderConfig,
    checkpoint_sections,
    from_checkpoint,
    read_checkpoint_info,
)
from bijou.molmoact2.action_expert import ActionExpertConfig

_ACTION_DIM = 6
_STATE_DIM = 6
_MAX_ACTION_DIM = 8
_HORIZON = 4

_TINY_AE = ActionExpertConfig(
    max_horizon=_HORIZON,
    max_action_dim=_MAX_ACTION_DIM,
    hidden_size=16,
    num_layers=2,
    num_heads=2,
    mlp_ratio=4.0,
    ffn_multiple_of=16,
    timestep_embed_dim=8,
    dropout=0.0,
    attn_dropout=0.0,
    context_layer_norm=True,
    qk_norm=True,
    qk_norm_eps=1e-6,
    rope=True,
    causal_attn=False,
)


def _source_config() -> dict[str, Any]:
    return {
        "model_type": "molmoact2",
        "dtype": "float32",
        "tie_word_embeddings": False,
        "image_patch_id": 155_650,
        "add_action_expert": True,
        "action_expert_depth_gate": False,
        "action_mode": "both",
        "state_format": "discrete",
        "n_obs_steps": 1,
        "num_state_tokens": 256,
        "max_action_horizon": _HORIZON,
        "max_action_dim": _MAX_ACTION_DIM,
        "mask_action_dim_padding": True,
        "flow_matching_num_steps": 10,
        "flow_matching_time_offset": 0.001,
        "flow_matching_time_scale": 0.999,
        "flow_matching_beta_alpha": 1.0,
        "flow_matching_beta_beta": 1.5,
        "flow_matching_cutoff": 1.0,
        "add_setup_tokens": True,
        "add_control_tokens": True,
        "action_expert_config": {
            "model_type": "molmoact2_action_expert",
            "hidden_size": _TINY_AE.hidden_size,
            "num_layers": _TINY_AE.num_layers,
            "num_heads": _TINY_AE.num_heads,
            "mlp_ratio": _TINY_AE.mlp_ratio,
            "ffn_multiple_of": _TINY_AE.ffn_multiple_of,
            "timestep_embed_dim": _TINY_AE.timestep_embed_dim,
            "context_layer_norm": True,
            "qk_norm": True,
            "qk_norm_eps": 1e-6,
            "rope": True,
            "causal_attn": False,
            "dropout": 0.0,
            "attn_dropout": 0.0,
        },
        # A full tiny trunk config (the predictor-test fixture's shape):
        # the assembly test loads it for real.
        "text_config": {
            "model_type": "molmo2_text",
            "vocab_size": 151_936,
            "additional_vocab_size": 4_096,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "hidden_act": "silu",
            "layer_norm_eps": 1e-6,
            "rope_theta": 10_000.0,
            "use_qk_norm": True,
            "qk_norm_type": "qwen3",
            "qkv_bias": False,
            "norm_after": False,
            "rope_scaling": None,
            "rope_scaling_layers": None,
            "attention_dropout": 0.0,
            "embedding_dropout": 0.0,
            "residual_dropout": 0.0,
        },
        "vit_config": {
            "model_type": "molmo2",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "hidden_act": "gelu_pytorch_tanh",
            "layer_norm_eps": 1e-6,
            "image_patch_size": 14,
            "image_num_pos": 729,
            "float32_attention": True,
            "attention_dropout": 0.0,
            "residual_dropout": 0.0,
        },
        "adapter_config": {
            "model_type": "molmo2",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "hidden_act": "silu",
            "text_hidden_size": 32,
            "vit_layers": [-1, -2],
            "float32_attention": True,
            "pooling_attention_mask": True,
            "attention_dropout": 0.0,
            "residual_dropout": 0.0,
            "image_feature_dropout": 0.0,
        },
    }


def _norm_stats() -> dict[str, Any]:
    def rows(dim: int, pad_to: int | None = None) -> dict[str, list[float]]:
        width = pad_to if pad_to is not None else dim
        pad = width - dim
        return {
            "mean": [1.0] * dim + [0.0] * pad,
            "std": [2.0] * dim + [1.0] * pad,
            "q01": [-3.0] * dim + [0.0] * pad,
            "q99": [3.0] * dim + [0.0] * pad,
            "count": [100] * width,
        }

    return {
        "format": 1,
        "norm_mode": "q01_q99",
        "metadata_by_tag": {
            "tiny_tag": {
                "action_key": "action",
                "state_keys": ["observation.state"],
                "camera_keys": ["observation.images.front"],
                "normalize_gripper": True,
                "action_dim": _MAX_ACTION_DIM,
                "action_horizon": _HORIZON,
                "n_action_steps": _HORIZON,
                "setup_type": "tiny rig",
                "control_mode": "absolute joint pose",
                # Their vectors are max_action_dim padded; the stats
                # width the converter records is the q01 row's length,
                # so the fabricated rows keep the REAL dim only.
                "action_stats": rows(_ACTION_DIM),
                "state_stats": rows(_STATE_DIM),
            },
        },
    }


def _expert_state() -> dict[str, torch.Tensor]:
    """The HF-export tensor set: the port module's state_dict minus the
    loader-injected compat tensors, under the model.action_expert prefix."""
    torch.manual_seed(0)
    expert = _TINY_AE.build(llm_kv_dim=16)
    return {
        f"model.action_expert.{name}": tensor.clone()
        for name, tensor in expert.state_dict().items()
        if "kv_proj" not in name and not name.startswith("state_encoder")
    }


@pytest.fixture(scope="module")
def source_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny checkpoint in their exact layout: expert tensors + a REAL
    tiny trunk (text + vision, their key prefixes), so the step-5
    assembly path loads end-to-end with no hub access."""
    from bijou.molmo2.config import Molmo2Config
    from bijou.molmo2.model import Molmo2Model
    from bijou.molmo2.text import Molmo2TextModel
    from bijou.molmo2.vision import Molmo2VisionBackbone

    source = tmp_path_factory.mktemp("molmoact2-src") / "tiny-hf"
    source.mkdir()
    config = _source_config()
    (source / "config.json").write_text(json.dumps(config))
    (source / "norm_stats.json").write_text(json.dumps(_norm_stats()))
    (source / "tokenizer.json").write_text("{}")
    state = _expert_state()
    torch.manual_seed(1)
    parsed = Molmo2Config.from_dict(config)
    assert parsed.vit is not None and parsed.adapter is not None
    text = Molmo2TextModel(parsed.text, lm_head=True)
    vision = Molmo2VisionBackbone(parsed.vit, parsed.adapter)
    trunk = Molmo2Model(text, vision, image_patch_id=parsed.image_patch_id)
    for name, tensor in trunk.text.state_dict().items():
        key = name if name == "lm_head.weight" else f"model.{name}"
        state[key] = tensor.clone()
    for name, tensor in trunk.vision.state_dict().items():
        state[f"model.vision_backbone.{name}"] = tensor.clone()
    save_file(state, str(source / "model.safetensors"))
    return source


def _convert(source: Path, out: Path, backbone_ref: str = "user/tiny-hf") -> Path:
    return convert(str(source), out, norm_tag="tiny_tag", backbone_ref=backbone_ref)


def test_norm_stats_from_substitutes_the_table(
    source_dir: Path,
    tmp_path: Path,
) -> None:
    """--norm-stats-from: weights from --source, q01/q99 tables from the
    alternate artifact — their fine-tune recipe's table-recompute
    semantics (gate-d-lite reproduces the rig rung from RELEASED
    weights under the RIG table). Substitution recorded in provenance;
    expert bytes unchanged."""
    import shutil

    alternate = tmp_path / "alternate"
    shutil.copytree(source_dir, alternate)
    stats = json.loads((alternate / "norm_stats.json").read_text())
    tag = stats["metadata_by_tag"]["tiny_tag"]
    tag["action_stats"]["q01"] = [-7.0] * len(tag["action_stats"]["q01"])
    tag["action_stats"]["q99"] = [7.0] * len(tag["action_stats"]["q99"])
    tag["state_stats"]["q01"] = [-9.0] * len(tag["state_stats"]["q01"])
    tag["state_stats"]["q99"] = [9.0] * len(tag["state_stats"]["q99"])
    (alternate / "norm_stats.json").write_text(json.dumps(stats))

    plain = _convert(source_dir, tmp_path / "plain")
    substituted = convert(
        str(source_dir),
        tmp_path / "substituted",
        norm_tag="tiny_tag",
        backbone_ref="user/tiny-hf",
        norm_stats_from=str(alternate),
    )
    sub_info = read_checkpoint_info(substituted)
    assert sub_info.normalization.action_q01 == (-7.0,) * _ACTION_DIM
    assert sub_info.normalization.action_q99 == (7.0,) * _ACTION_DIM
    assert sub_info.normalization.state_q01 == (-9.0,) * _STATE_DIM
    assert sub_info.normalization.state_q99 == (9.0,) * _STATE_DIM
    plain_meta = json.loads((plain / "bijou_config.json").read_text())
    sub_meta = json.loads((substituted / "bijou_config.json").read_text())
    assert sub_meta["converted_from"]["norm_stats_from"] == str(alternate)
    assert "norm_stats_from" not in plain_meta["converted_from"]
    # Weights are untouched by the substitution.
    assert (
        sub_meta["converted_from"]["expert_sha256"]
        == plain_meta["converted_from"]["expert_sha256"]
    )


def test_happy_path_round_trips(source_dir: Path, tmp_path: Path) -> None:
    out = _convert(source_dir, tmp_path / "converted")
    info = read_checkpoint_info(out)
    assert info.backbone == "user/tiny-hf"
    assert info.step == 0
    assert info.train_args.decoder == "molmo_flow"
    assert info.train_args.chunk_size == _HORIZON

    sections = checkpoint_sections(json.loads((out / "bijou_config.json").read_text()))
    prompt = sections.prompt
    assert isinstance(prompt, MolmoAct2PromptConfig)
    assert prompt.action_mode == "both"
    assert prompt.setup_type == "tiny rig"
    assert prompt.norm_tag == "tiny_tag"
    assert prompt.state_dim == _STATE_DIM
    assert prompt.camera_keys == ("observation.images.front",)
    assert prompt.generate_bracket is False
    assert prompt.condition_fields == ()
    decoder = sections.decoder
    assert isinstance(decoder, MolmoFlowDecoderConfig)
    assert decoder.num_layers == _TINY_AE.num_layers
    assert decoder.llm_kv_dim == 16
    assert decoder.action_dim == _ACTION_DIM
    assert decoder.max_action_dim == _MAX_ACTION_DIM
    assert decoder.action_horizon == _HORIZON
    assert decoder.num_flow_steps == 10
    assert decoder.normalization == "q01q99"
    assert decoder.time_offset == 0.001
    assert decoder.time_scale == 0.999
    assert decoder.beta_alpha == 1.0
    assert decoder.beta_beta == 1.5

    # The q01/q99 fields of the normalization table ARE the clamp table.
    assert info.normalization.action_q01 == (-3.0,) * _ACTION_DIM
    assert info.normalization.action_q99 == (3.0,) * _ACTION_DIM
    assert info.normalization.action_std == (2.0,) * _ACTION_DIM


def test_expert_bytes_verbatim_and_names_match_port(
    source_dir: Path,
    tmp_path: Path,
) -> None:
    out = _convert(source_dir, tmp_path / "converted")
    from safetensors import safe_open

    with safe_open(out / "expert.safetensors", framework="pt", device="cpu") as f:
        written = {key: f.get_tensor(key) for key in f.keys()}  # noqa: SIM118
    source = {
        key.removeprefix("model.action_expert."): value
        for key, value in _expert_state().items()
        if key.startswith("model.action_expert.")
    }
    assert set(written) == set(source)
    for name, tensor in source.items():
        assert torch.equal(written[name], tensor), name
    # The names contract for step 3: the port module's state_dict minus
    # the loader-injected compat tensors.
    expert = _TINY_AE.build(llm_kv_dim=16)
    module_names = {
        name
        for name in expert.state_dict()
        if "kv_proj" not in name and not name.startswith("state_encoder")
    }
    assert set(written) == module_names


def test_idempotent_and_deterministic(source_dir: Path, tmp_path: Path) -> None:
    first = _convert(source_dir, tmp_path / "a")
    second = _convert(source_dir, tmp_path / "b")
    _convert(source_dir, tmp_path / "a")  # overwrite in place
    for name in ("bijou_config.json", "expert.safetensors"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    meta = json.loads((first / "bijou_config.json").read_text())
    digest = hashlib.sha256(
        (source_dir / "config.json").read_bytes(),
    ).hexdigest()
    assert meta["converted_from"]["source_config_sha256"] == digest


def test_from_checkpoint_assembles_molmo_flow(
    source_dir: Path,
    tmp_path: Path,
) -> None:
    """The step-5 assembly path end-to-end on the tiny converted
    checkpoint (backbone ref = the source dir, no hub access): decoder
    built + configured off the sections, expert weights byte-equal the
    source, compat tensors injected, encoder carries the prompt facts,
    q01/q99 table on the decoder buffers."""
    from bijou.decoders.molmo_flow import MolmoFlowDecoder
    from bijou.encoders.molmoact2 import MolmoAct2Encoder

    out = _convert(source_dir, tmp_path / "converted", backbone_ref=str(source_dir))
    model, info = from_checkpoint(out)
    decoder = model.decoder
    assert isinstance(decoder, MolmoFlowDecoder)
    assert decoder.config.num_layers == _TINY_AE.num_layers
    assert decoder.config.llm_kv_dim == 16
    runtime = decoder.runtime
    assert runtime is not None
    assert runtime.action_dim == _ACTION_DIM
    assert runtime.action_horizon == _HORIZON
    assert runtime.num_flow_steps == 10
    assert runtime.time_law.beta_beta == 1.5
    assert torch.equal(
        decoder.action_q01[:_ACTION_DIM].cpu(),
        torch.full((_ACTION_DIM,), -3.0),
    )
    assert torch.equal(
        decoder.action_q99[_ACTION_DIM:].cpu(),
        torch.ones(_MAX_ACTION_DIM - _ACTION_DIM),  # inert unit box on pads
    )
    source = {
        key.removeprefix("model.action_expert."): value
        for key, value in _expert_state().items()
        if key.startswith("model.action_expert.")
    }
    for name, tensor in source.items():
        loaded = decoder.state_dict()[name]
        assert torch.equal(loaded.cpu(), tensor), name
    assert torch.equal(
        decoder.state_dict()["state_encoder.weight"].cpu().float(),
        torch.eye(_TINY_AE.hidden_size),  # compat-injected
    )
    encoder = model.encoder
    assert isinstance(encoder, MolmoAct2Encoder)
    assert encoder.setup_type == "tiny rig"
    assert encoder.action_mode == "both"
    assert encoder.narration is False
    assert info.train_args.decoder == "molmo_flow"
    assert len(list(encoder.parameters())) == 0


def _broken_source(source_dir: Path, tmp_path: Path, mutate: Any) -> Path:
    import shutil

    broken = tmp_path / "broken"
    shutil.copytree(source_dir, broken)
    config = json.loads((broken / "config.json").read_text())
    stats = json.loads((broken / "norm_stats.json").read_text())
    mutate(config, stats)
    (broken / "config.json").write_text(json.dumps(config))
    (broken / "norm_stats.json").write_text(json.dumps(stats))
    return broken


def test_guards_refuse_bad_sources(source_dir: Path, tmp_path: Path) -> None:
    def nonzero_dropout(config: dict, _stats: dict) -> None:
        config["action_expert_config"]["dropout"] = 0.1

    def missing_n_obs(config: dict, _stats: dict) -> None:
        del config["n_obs_steps"]

    def discrete_mode(config: dict, _stats: dict) -> None:
        config["action_mode"] = "discrete"

    def continuous_state(config: dict, _stats: dict) -> None:
        config["state_format"] = "continuous"

    def missing_mean(_config: dict, stats: dict) -> None:
        del stats["metadata_by_tag"]["tiny_tag"]["action_stats"]["mean"]

    def truncated_cutoff(config: dict, _stats: dict) -> None:
        config["flow_matching_cutoff"] = 0.9

    for index, (mutate, error) in enumerate(
        (
            (nonzero_dropout, SystemExit),
            (missing_n_obs, NotImplementedError),
            (discrete_mode, NotImplementedError),
            (continuous_state, SystemExit),
            (missing_mean, SystemExit),
            (truncated_cutoff, SystemExit),
        ),
    ):
        broken = _broken_source(source_dir, tmp_path / f"case{index}", mutate)
        with pytest.raises(error):
            _convert(broken, tmp_path / f"out{index}")


def test_missing_tokenizer_refused(source_dir: Path, tmp_path: Path) -> None:
    import shutil

    broken = tmp_path / "no-tokenizer"
    shutil.copytree(source_dir, broken)
    (broken / "tokenizer.json").unlink()
    with pytest.raises(SystemExit, match=r"tokenizer\.json"):
        _convert(broken, tmp_path / "out")


def test_wrong_tag_refused(source_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="norm_tag"):
        convert(str(source_dir), tmp_path / "out", norm_tag="nope", backbone_ref=None)
