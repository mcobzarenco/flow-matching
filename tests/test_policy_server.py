"""Policy-server loopback oracles — the remote-inference seam, pure CPU
on the tiny molmoact2 release fixture.

What this suite pins:

1. the PARITY claim: RemotePolicy through a real loopback HTTP server
   is BITWISE equal to a local BijouPolicy on identical inputs — the
   JPEGs are encoded once and the local path scores the DECODED
   frames, so the equality is honest about the (lossy) wire codec;
2. /spec: schema version, family/chunk/action_dim, the recorded
   serving operating point, the capability facts the client narrows
   on, and stats tables that round-trip to the checkpoint's own;
3. options-vs-capability refusals as structured 400s naming the family
   (the loud-narrowing rule over the wire), malformed-state/JPEG/field
   refusals, and that the server SURVIVES all of them (a bad request
   never kills serving);
4. client-side loud failures: schema-version mismatch, connection
   errors (no silent retries), and capability refusals at
   construction — before a robot would connect;
5. the rollout parser's mutual-exclusion rules for --policy-server;
6. a client "restart" with a different camera set against the same
   server — per-camera collation state is never sticky.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

from bijou.checkpoint import read_metadata
from bijou.data import DatasetStats
from bijou.eval.policies import BijouPolicy
from bijou.modelling.aux_text import AuxField
from bijou.modelling.interface import SamplingMethod
from bijou.policy_server import PolicyServer, ServerRuntime, build_runtime
from bijou.remote_policy import RemotePolicy, decode_jpeg, encode_jpeg
from bijou.rollout import parse_args as rollout_parse_args
from bijou.testing import (
    TINY_MOLMOACT2_D,
    TINY_MOLMOACT2_T,
    write_tiny_molmoact2_release,
)

SEED = 11
STEPS = 3
INDEX = 5
TASK = "Pick up the cube."
# Inside the fixture's state q01/q99 band (bijou.testing constants).
STATE = [10.0, -20.0, 90.0, 5.0, -3.0, 50.0]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("policy-server") / "tiny"
    write_tiny_molmoact2_release(root)
    return root / "checkpoint_vla"


@pytest.fixture(scope="module")
def stats(checkpoint: Path) -> DatasetStats:
    return read_metadata(checkpoint).stats


@pytest.fixture(scope="module")
def runtime(checkpoint: Path) -> ServerRuntime:
    return build_runtime(
        checkpoint,
        device=torch.device("cpu"),
        flow_decoder_dtype=torch.float32,
        seed=SEED,
    )


@pytest.fixture(scope="module")
def server(runtime: ServerRuntime) -> Any:
    instance = PolicyServer(("127.0.0.1", 0), runtime)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{instance.server_address[1]}"
    instance.shutdown()
    instance.server_close()
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def local_policy(checkpoint: Path) -> BijouPolicy:
    return BijouPolicy(
        checkpoint,
        device=torch.device("cpu"),
        seed=SEED,
        sample_steps=STEPS,
        method=SamplingMethod.EULER,
        # The deployment keying (rollout.py and the server both pin it):
        # rig items carry no dataset identity for "stable" keying.
        noise_key="index",
    )


def raw_frames(names: tuple[str, ...] = ("top", "wrist")) -> dict[str, Tensor]:
    """Deterministic uint8-sourced camera frames in the rollout item
    convention.

    Shapes:
      - returns: name → [3, 48, 64] float [0, 1]
    """
    generator = np.random.default_rng(7)
    return {
        name: torch.from_numpy(
            generator.integers(0, 256, size=(48, 64, 3), dtype=np.uint8),
        )
        .permute(2, 0, 1)
        .float()
        / 255.0
        for name in names
    }


def rig_item(frames: dict[str, Tensor], stats: DatasetStats) -> dict[str, Any]:
    """The rollout ``observation_to_item`` shape (zero ground-truth
    stubs, per-item stats tensors, kinds and condition values riding
    the item)."""
    return {
        "task": TASK,
        "camera_kinds": {name: name for name in frames},
        "condition_outcome": "success",
        "observation.state": torch.tensor(STATE),
        "action": torch.zeros(TINY_MOLMOACT2_T, TINY_MOLMOACT2_D),
        "action_is_pad": torch.zeros(TINY_MOLMOACT2_T, dtype=torch.bool),
        **stats.item_tensors(),
        **{f"observation.images.{name}": frame for name, frame in frames.items()},
    }


def remote(server: str, **kwargs: Any) -> RemotePolicy:
    kwargs.setdefault("timeout", 30.0)
    kwargs.setdefault("sample_steps", STEPS)
    kwargs.setdefault("method", SamplingMethod.EULER)
    return RemotePolicy(server, **kwargs)


def base_request(stats: DatasetStats) -> dict[str, Any]:
    """A valid /predict body built by hand (the raw-HTTP tests mutate
    copies of it)."""
    import base64

    return {
        "task": TASK,
        "state": list(STATE),
        "images": {
            name: base64.b64encode(encode_jpeg(frame)).decode("ascii")
            for name, frame in raw_frames().items()
        },
        "stats": stats.state_dict(),
        "index": INDEX,
        "options": {"num_steps": STEPS, "method": "euler"},
    }


def post(server: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        f"{server}/predict",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


# ---------------------------------------------------------------------
# /spec


def test_spec_fields(server: str, checkpoint: Path, stats: DatasetStats) -> None:
    with urllib.request.urlopen(f"{server}/spec", timeout=30) as response:
        spec = json.loads(response.read())
    assert spec["schema_version"] == 1
    assert spec["family"] == "molmoact2_flow"
    assert spec["chunk_size"] == TINY_MOLMOACT2_T
    assert spec["action_dim"] == TINY_MOLMOACT2_D
    # The converter records the release's own operating point
    # (num_flow_steps, euler) — /spec must ship it verbatim.
    assert spec["serving"] == {"kind": "flow", "num_steps": 4, "method": "euler"}
    assert spec["checkpoint"] == str(checkpoint)
    assert spec["step"] == 0
    assert spec["narrating"] is False
    assert spec["draw_ensembling"] is False
    # molmo_flow decoder ⇒ the model-frame envelope gate applies.
    assert spec["global_state_table"] is True
    assert spec["condition_fields"] == []
    assert spec["generate_bracket"] is False
    assert DatasetStats.from_state_dict(spec["stats"]) == stats
    assert spec["per_dataset_stats"] == {}
    assert isinstance(spec["git_rev"], str) and len(spec["git_rev"]) > 0


def test_unknown_path_is_404(server: str) -> None:
    status, body = post(server, {})  # POST parses, but wrong path first:
    assert status == 400  # /predict with an empty body is a 400, not a crash
    request = urllib.request.Request(f"{server}/nope", method="GET")
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(request, timeout=30)
    assert excinfo.value.code == 404
    assert "error" in json.loads(excinfo.value.read())
    assert "error" in body


# ---------------------------------------------------------------------
# Parity


def test_loopback_parity_is_bitwise(
    server: str,
    local_policy: BijouPolicy,
    stats: DatasetStats,
) -> None:
    """RemotePolicy through the wire == local BijouPolicy, bitwise.

    The remote path JPEG-encodes the RAW frames once (client encode →
    wire → server decode); the local path is fed those decoded frames
    directly, so both inferences see identical pixels and the equality
    claim is honest about the lossy codec."""
    frames = raw_frames()
    client = remote(server)
    remote_chunks, remote_generations = client.predict_with_text(
        [rig_item(frames, stats)],
        [INDEX],
    )
    decoded = {name: decode_jpeg(encode_jpeg(frame)) for name, frame in frames.items()}
    local_chunks, local_generations = local_policy.predict_with_text(
        [rig_item(decoded, stats)],
        [INDEX],
    )
    assert remote_generations is None and local_generations is None
    assert tuple(remote_chunks[0].shape) == (TINY_MOLMOACT2_T, TINY_MOLMOACT2_D)
    assert remote_chunks[0].dtype == torch.float32
    assert torch.equal(remote_chunks[0], local_chunks[0])
    assert client.last_timings is not None
    assert set(client.last_timings) == {"decode_ms", "infer_ms", "total_ms"}


def test_client_restart_with_different_cameras(
    server: str,
    stats: DatasetStats,
) -> None:
    """The server survives a client coming back with a different camera
    set (no sticky per-camera collation state) — and with no options at
    all (defaults = the recorded serving point)."""
    one_camera = base_request(stats)
    one_camera["images"] = {
        name: encoded for name, encoded in one_camera["images"].items() if name == "top"
    }
    del one_camera["options"]
    status, body = post(server, one_camera)
    assert status == 200
    assert len(body["actions"]) == TINY_MOLMOACT2_T
    status, body = post(server, base_request(stats))
    assert status == 200
    assert len(body["actions"]) == TINY_MOLMOACT2_T
    assert len(body["actions"][0]) == TINY_MOLMOACT2_D
    assert {"decode_ms", "infer_ms", "total_ms"} == set(body["timings"])


# ---------------------------------------------------------------------
# Server-side refusals (structured 400s; the server never dies)


def test_narration_on_flow_family_is_400(server: str, stats: DatasetStats) -> None:
    payload = base_request(stats)
    payload["options"]["generate"] = ["subgoal"]
    status, body = post(server, payload)
    assert status == 400
    assert "molmoact2_flow" in body["error"]
    assert "narration" in body["error"]


def test_num_samples_on_molmo_flow_is_400(server: str, stats: DatasetStats) -> None:
    payload = base_request(stats)
    payload["options"]["num_samples"] = 2
    status, body = post(server, payload)
    assert status == 400
    assert "molmo_flow" in body["error"]


def test_unknown_option_is_400(server: str, stats: DatasetStats) -> None:
    payload = base_request(stats)
    payload["options"]["num_step"] = 3
    status, body = post(server, payload)
    assert status == 400
    assert "num_step" in body["error"]


def test_malformed_state_is_400(server: str, stats: DatasetStats) -> None:
    wrong_length = base_request(stats)
    wrong_length["state"] = [0.0, 1.0]
    status, body = post(server, wrong_length)
    assert status == 400
    assert "dims" in body["error"]

    not_numbers = base_request(stats)
    not_numbers["state"][2] = "ninety"
    status, body = post(server, not_numbers)
    assert status == 400
    assert "number" in body["error"]


def test_bad_jpeg_is_400(server: str, stats: DatasetStats) -> None:
    import base64

    payload = base_request(stats)
    payload["images"]["top"] = base64.b64encode(b"not a jpeg").decode("ascii")
    status, body = post(server, payload)
    assert status == 400
    assert "top" in body["error"]


def test_missing_field_is_400_and_server_survives(
    server: str,
    stats: DatasetStats,
) -> None:
    payload = base_request(stats)
    del payload["task"]
    status, body = post(server, payload)
    assert status == 400
    assert "task" in body["error"]
    # The never-dies claim: after every refusal above, a good request
    # still serves.
    status, body = post(server, base_request(stats))
    assert status == 200
    assert len(body["actions"]) == TINY_MOLMOACT2_T


# ---------------------------------------------------------------------
# Client-side loud failures


def test_client_refuses_schema_version_mismatch(
    server: str,
    runtime: ServerRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(runtime.spec, "schema_version", 99)
    with pytest.raises(SystemExit, match="schema_version"):
        remote(server)


def test_client_refuses_narration_at_construction(server: str) -> None:
    with pytest.raises(SystemExit, match="no narration surface"):
        remote(server, generate=(AuxField.SUBGOAL,))


def test_client_refuses_draw_ensembling_at_construction(server: str) -> None:
    with pytest.raises(SystemExit, match="single-draw"):
        remote(server, sample_draws=2)


def test_client_connection_error_is_loud() -> None:
    with pytest.raises(SystemExit, match="unreachable"):
        RemotePolicy("http://127.0.0.1:9", timeout=0.5)


def test_client_refuses_batch_prediction(server: str, stats: DatasetStats) -> None:
    client = remote(server)
    item = rig_item(raw_frames(), stats)
    with pytest.raises(SystemExit, match="single-observation"):
        client.predict_with_text([item, item], [0, 1])


# ---------------------------------------------------------------------
# Rollout parser mutual exclusion


def _parse_rollout(monkeypatch: pytest.MonkeyPatch, *extra: str) -> Any:
    monkeypatch.setattr("sys.argv", ["bijou.rollout", "--task", "t", *extra])
    return rollout_parse_args()


def test_rollout_requires_exactly_one_policy_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit):
        _parse_rollout(monkeypatch)
    with pytest.raises(SystemExit):
        _parse_rollout(
            monkeypatch,
            "--checkpoint",
            "ckpt",
            "--policy-server",
            "http://localhost:8143",
        )


def test_rollout_refuses_async_with_policy_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit):
        _parse_rollout(
            monkeypatch,
            "--policy-server",
            "http://localhost:8143",
            "--async-inference",
        )


@pytest.mark.parametrize(
    "flags",
    [
        ("--flow-decoder-dtype", "bfloat16"),
        ("--offload-ple",),
        ("--seed", "7"),
        ("--noise-ticket", "bank.npz"),
        ("--target-time", "zero"),
    ],
)
def test_rollout_refuses_server_side_flags_with_policy_server(
    monkeypatch: pytest.MonkeyPatch,
    flags: tuple[str, ...],
) -> None:
    with pytest.raises(SystemExit):
        _parse_rollout(
            monkeypatch,
            "--policy-server",
            "http://localhost:8143",
            *flags,
        )


def test_rollout_parses_valid_remote_and_local_invocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parse_rollout(monkeypatch, "--policy-server", "http://localhost:8143")
    assert args.policy_server == "http://localhost:8143"
    assert args.checkpoint is None
    assert args.policy_server_timeout == 30.0
    args = _parse_rollout(monkeypatch, "--checkpoint", "ckpt")
    assert args.checkpoint == Path("ckpt")
    assert args.policy_server is None
    # The local default resolves to float32 in main (None is the
    # explicit-vs-default sentinel the remote refusal needs).
    assert args.flow_decoder_dtype is None


# ---------------------------------------------------------------------
# Wire image codec


def test_jpeg_codec_round_trip() -> None:
    """encode → decode preserves geometry and stays close on smooth
    content (JPEG is lossy by design — the parity oracle feeds both
    paths the DECODED frames, so closeness here is a sanity bound, not
    a correctness requirement)."""
    ramp = torch.linspace(0, 1, 64).expand(48, 64)
    frame = torch.stack([ramp, ramp * 0.5, 1 - ramp])
    decoded = decode_jpeg(encode_jpeg(frame))
    assert decoded.shape == frame.shape
    assert decoded.dtype == torch.float32
    assert decoded.min() >= 0.0 and decoded.max() <= 1.0
    assert (decoded - frame).abs().mean() < 0.02
