"""HTTP policy server: BijouPolicy inference for a remote rollout loop.

Serves ONE checkpoint through ONE :class:`~bijou.eval.policies.BijouPolicy`
constructed at startup (a bad checkpoint dies loudly here, before the
socket opens). The rollout loop on the robot's laptop connects with
``bijou.remote_policy.RemotePolicy``: cameras and robot stay local, the
model runs here, and the sync loop's replan block grows by one round
trip — the servos hold their last goal while the loop waits, which is
the existing sync-rollout semantics with a longer block. Async /
latency-hiding over the WAN is deliberately out of scope.

Requests are serialized with one lock: the deployment shape is one
robot driven by one client, so concurrency would only interleave GPU
work that a single stream orders anyway — and per-request option knobs
(steps/method/draws/generate) are applied to the shared policy under
that lock, so they can never bleed between requests. Per-request
options are validated against the family's capabilities exactly like
the local CLI (narration on a family without a narration surface is a
structured 400 naming the family — the loud-narrowing rule); the
handler converts ``SystemExit`` from the policy internals into 400s,
so the server never dies on a bad request.

The only expensive request-dependent collation state is the AR-serving
families' ``[generate|…]`` prompt override — memoized per generate
tuple (the underlying inputs collator, with its lazily-built
tokenizer, is shared). Camera sets ride the items, never cached state:
a client may restart with different cameras without a server restart.

Transport is unauthenticated plain HTTP (stdlib ``http.server``, no
new dependencies): the server binds loopback by default and the
supported remote path is an SSH tunnel
(``ssh -N -L <port>:localhost:<port> <box>``); binding anything else
prints a loud warning. Images arrive base64-JPEG (measured ~0.2 ms CPU
to encode, ~33% wire inflation); the /predict response's ``timings``
(decode_ms/infer_ms/total_ms) exist so real rollouts measure the split
instead of assuming it.

Usage::

    uv run python -m bijou.policy_server \\
        --checkpoint <dir> --device cuda \\
        --flow-decoder-dtype bfloat16 --port 8143
"""

from __future__ import annotations

import argparse
import base64
import binascii
import dataclasses
import json
import math
import threading
import time
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch

from .checkpoint import read_metadata
from .data import DatasetStats
from .eval.policies import BijouPolicy
from .modelling.aux_text import AuxField
from .modelling.decoders.molmo_flow import MolmoFlowDecoder
from .modelling.interface import Collator, SamplingMethod
from .models.serving import FlowServing
from .remote_policy import WIRE_SCHEMA_VERSION, decode_jpeg, git_rev

LOOPBACK_BINDS = ("127.0.0.1", "localhost", "::1")


class RequestError(Exception):
    """A client-fixable /predict problem — becomes a structured 400
    (the server-side analogue of the CLI's SystemExit class)."""


@dataclass(frozen=True, slots=True)
class PredictRequest:
    """One /predict request, parsed and capability-validated ("parse,
    don't validate"): ``item`` is exactly the shape
    ``BijouPolicy.predict_with_text`` consumes (the rollout
    ``observation_to_item`` contract — state/action stubs/stats
    tensors/camera frames), ``index`` the client's replan counter (the
    "index" noise keying), and the option fields the per-request knobs
    applied under the server lock."""

    item: dict[str, Any]
    index: int
    generate: tuple[AuxField, ...]
    num_steps: int
    method: SamplingMethod
    num_samples: int


@dataclass
class ServerRuntime:
    """Everything the request handlers read: the policy and its lock,
    the precomputed /spec payload, the recorded flow serving point
    (request defaults; None for AR-serving checkpoints, whose greedy
    decode takes no knobs), and the per-generate-tuple collator cache
    (AR-serving families render the request into the prompt)."""

    policy: BijouPolicy
    spec: dict[str, Any]
    serving: FlowServing | None
    ar_serving: bool
    lock: threading.Lock
    collators: dict[tuple[AuxField, ...], Collator[Any]]

    def collator_for(self, generate: tuple[AuxField, ...]) -> Collator[Any]:
        """The collator serving one request's generate tuple. Only
        AR-serving families bake the request into the prompt
        ([generate|…] override); everyone else shares the base
        collator for every request."""
        if not self.ar_serving:
            return self.collators[()]
        if generate not in self.collators:
            self.collators[generate] = dataclasses.replace(
                self.collators[()],
                generate_override=generate,
            )
        return self.collators[generate]


def build_runtime(
    checkpoint: Path,
    *,
    device: torch.device,
    flow_decoder_dtype: torch.dtype,
    seed: int,
    offload_ple: bool = False,
) -> ServerRuntime:
    """Load the checkpoint into a BijouPolicy (loud on any problem —
    startup is the place to die) and precompute the /spec payload."""
    metadata = read_metadata(checkpoint)
    policy = BijouPolicy(
        checkpoint,
        device=device,
        seed=seed,
        # Index keying is the deployment semantics (rollout.py's
        # convention): fresh noise per replan via the client-sent
        # replan counter, reproducible under a fixed --seed.
        noise_key="index",
        flow_decoder_dtype=flow_decoder_dtype,
        offload_ple=offload_ple,
    )
    serving = (
        FlowServing.from_dict(metadata.serving)
        if metadata.serving.get("kind") == "flow"
        else None
    )
    spec: dict[str, Any] = {
        "schema_version": WIRE_SCHEMA_VERSION,
        "family": policy.spec.family.value,
        "chunk_size": policy.info.chunk_size,
        "action_dim": policy.spec.action_dim,
        "serving": metadata.serving,
        "checkpoint": str(checkpoint),
        "step": metadata.step,
        "git_rev": git_rev(Path(__file__).resolve().parent),
        # Capability facts the client narrows on at construction (the
        # loud-narrowing rule, before its robot connects).
        "narrating": policy.narrating is not None,
        "draw_ensembling": policy.gemma_flow is not None,
        # molmo_flow decoders normalize state with the checkpoint's ONE
        # baked-in table — drives rollout's model-frame envelope gate.
        "global_state_table": isinstance(policy.flow_decoder, MolmoFlowDecoder),
        "condition_fields": list(policy.info.condition_fields),
        "generate_bracket": policy.info.generate_bracket,
        "stats": policy.info.normalization.state_dict(),
        "per_dataset_stats": {
            repo_id: stats.state_dict()
            for repo_id, stats in sorted(
                policy.info.per_dataset_normalization.items(),
            )
        },
    }
    return ServerRuntime(
        policy=policy,
        spec=spec,
        serving=serving,
        ar_serving=policy.flow is None and policy.ar is not None,
        lock=threading.Lock(),
        collators={(): policy.collator},
    )


def _require(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise RequestError(f"request is missing the {key!r} field")
    return payload[key]


def _parse_state(raw: Any, expected_dim: int) -> list[float]:
    if not isinstance(raw, list) or len(raw) == 0:
        raise RequestError(
            f"state must be a non-empty list of numbers, got {type(raw).__name__}",
        )
    values: list[float] = []
    for position, entry in enumerate(raw):
        if isinstance(entry, bool) or not isinstance(entry, int | float):
            raise RequestError(
                f"state[{position}] is {entry!r} — every entry must be a number",
            )
        if not math.isfinite(float(entry)):
            raise RequestError(f"state[{position}] is non-finite ({entry!r})")
        values.append(float(entry))
    if len(values) != expected_dim:
        raise RequestError(
            f"state has {len(values)} dims, the shipped stats say "
            f"{expected_dim} (state_mean length) — wrong rig stats or a "
            "truncated state vector",
        )
    return values


def _parse_images(raw: Any) -> dict[str, torch.Tensor]:
    """Decode the request's base64-JPEG camera frames.

    Shapes:
      - returns: camera name → [3, H, W] float [0, 1] (the collator's
        frame convention)
    """
    if not isinstance(raw, dict) or len(raw) == 0:
        raise RequestError(
            "images must be a non-empty {camera_name: base64 jpeg} object",
        )
    frames: dict[str, torch.Tensor] = {}
    for name, encoded in raw.items():
        if not isinstance(encoded, str):
            raise RequestError(f"images[{name!r}] must be a base64 string")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as error:
            raise RequestError(
                f"images[{name!r}] is not valid base64: {error}",
            ) from error
        try:
            frames[str(name)] = decode_jpeg(data)
        except ValueError as error:
            raise RequestError(f"images[{name!r}]: {error}") from error
    return frames


def _parse_string_map(raw: Any, field: str) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RequestError(f"{field} must be a {{str: str}} object when present")
    result: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(value, str):
            raise RequestError(f"{field}[{key!r}] must be a string, got {value!r}")
        result[str(key)] = value
    return result


def _parse_positive_int(raw: Any, field: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise RequestError(f"options.{field} must be an integer >= 1, got {raw!r}")
    return raw


def _parse_options(
    raw: Any,
    runtime: ServerRuntime,
) -> tuple[tuple[AuxField, ...], int, SamplingMethod, int]:
    """(generate, num_steps, method, num_samples), validated against
    the loaded family's capabilities — unknown option keys, unknown aux
    fields and asks the family cannot back are all structured 400s."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise RequestError(f"options must be an object, got {type(raw).__name__}")
    known = {"generate", "num_steps", "method", "num_samples"}
    unknown = set(raw) - known
    if unknown:
        raise RequestError(
            f"unknown options {sorted(unknown)} — this server understands "
            f"{sorted(known)}",
        )
    policy = runtime.policy
    family = policy.spec.family.value
    generate_raw = raw.get("generate", [])
    if not isinstance(generate_raw, list):
        raise RequestError("options.generate must be a list of aux field names")
    valid_fields = [field.value for field in AuxField]
    generate: list[AuxField] = []
    for name in generate_raw:
        if not isinstance(name, str) or name not in valid_fields:
            raise RequestError(
                f"options.generate entry {name!r} is not an aux field "
                f"(choose from {valid_fields})",
            )
        generate.append(AuxField(name))
    if len(generate) > 0 and policy.narrating is None:
        raise RequestError(
            f"generate requested but {family} has no narration surface",
        )
    if runtime.serving is not None:
        num_steps = runtime.serving.num_steps
        method = runtime.serving.method
    else:
        # AR-serving checkpoints: the greedy block decode takes no
        # solver knobs; the values below are inert placeholders and
        # explicit num_steps/method are refused just under here.
        num_steps = 1
        method = SamplingMethod.EULER
    if "num_steps" in raw or "method" in raw:
        if runtime.serving is None:
            raise RequestError(
                f"options.num_steps/method drive a flow solver, but "
                f"{family} serves through its greedy discrete decode — "
                "drop them",
            )
        if "num_steps" in raw:
            num_steps = _parse_positive_int(raw["num_steps"], "num_steps")
        if "method" in raw:
            method_raw = raw["method"]
            values = [m.value for m in SamplingMethod]
            if not isinstance(method_raw, str) or method_raw not in values:
                raise RequestError(
                    f"options.method {method_raw!r} is not a sampling method "
                    f"(choose from {values})",
                )
            method = SamplingMethod(method_raw)
    num_samples = 1
    if "num_samples" in raw:
        num_samples = _parse_positive_int(raw["num_samples"], "num_samples")
    if num_samples > 1:
        # Mirror BijouPolicy's construction-time refusals (same
        # capability facts, structured 400 instead of SystemExit).
        if isinstance(policy.flow_decoder, MolmoFlowDecoder):
            raise RequestError(
                "options.num_samples > 1 is not implemented for molmo_flow "
                "decoders: draw ensembling tiles the prefix memory, and "
                "the molmo2 KV cache has no tile path — run single-sample",
            )
        if policy.gemma_flow is None:
            raise RequestError(
                f"options.num_samples > 1 needs a stochastic flow decode; "
                f"{family} decodes deterministically",
            )
        if len(generate) > 0:
            raise RequestError(
                "options.num_samples > 1 with options.generate: the "
                "ensembled mean decode emits no value lines — drop one",
            )
    return tuple(generate), num_steps, method, num_samples


def parse_request(payload: Any, runtime: ServerRuntime) -> PredictRequest:
    """The /predict request boundary: JSON in, a typed
    :class:`PredictRequest` out, every problem a :class:`RequestError`
    that names the field and the remedy."""
    if not isinstance(payload, dict):
        raise RequestError(
            f"request body must be a JSON object, got {type(payload).__name__}",
        )
    task = _require(payload, "task")
    if not isinstance(task, str) or task == "":
        raise RequestError(f"task must be a non-empty string, got {task!r}")
    try:
        stats = DatasetStats.from_state_dict(_require(payload, "stats"))
    except (KeyError, TypeError, ValueError) as error:
        raise RequestError(
            f"stats do not parse as a DatasetStats table "
            f"({type(error).__name__}: {error}) — send the "
            "DatasetStats.state_dict() shape the local rollout resolves "
            "from --stats-repo-id/--stats-dataset",
        ) from error
    state = _parse_state(_require(payload, "state"), len(stats.state_mean))
    frames = _parse_images(_require(payload, "images"))
    camera_kinds = _parse_string_map(payload.get("camera_kinds"), "camera_kinds")
    conditions = _parse_string_map(payload.get("conditions"), "conditions")
    index = payload.get("index", 0)
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise RequestError(f"index must be an integer >= 0, got {index!r}")
    generate, num_steps, method, num_samples = _parse_options(
        payload.get("options"),
        runtime,
    )
    policy = runtime.policy
    chunk = policy.info.chunk_size
    action_dim = policy.spec.action_dim
    # The rollout observation_to_item contract: ground-truth fields are
    # zero stubs (the collator requires them, the policy reads only
    # their shapes) and the stats travel as per-item tensors.
    item: dict[str, Any] = {
        "task": task,
        "camera_kinds": camera_kinds,
        **{f"condition_{field}": value for field, value in conditions.items()},
        "observation.state": torch.tensor(state),
        "action": torch.zeros(chunk, action_dim),
        "action_is_pad": torch.zeros(chunk, dtype=torch.bool),
        **stats.item_tensors(),
        **{f"observation.images.{name}": frame for name, frame in frames.items()},
    }
    return PredictRequest(
        item=item,
        index=index,
        generate=generate,
        num_steps=num_steps,
        method=method,
        num_samples=num_samples,
    )


def predict_response(runtime: ServerRuntime, request: PredictRequest) -> dict[str, Any]:
    """Run one prediction under the server lock and shape the response.
    The per-request knobs are plain BijouPolicy attributes; setting
    them under the lock (every request sets all of them) keeps the one
    shared policy consistent without re-loading anything."""
    policy = runtime.policy
    with runtime.lock:
        policy.sample_steps = request.num_steps
        policy.method = request.method
        policy.sample_draws = request.num_samples
        policy.generate = request.generate
        policy.collator = runtime.collator_for(request.generate)
        started = time.perf_counter()
        chunks, generations = policy.predict_with_text([request.item], [request.index])
        infer_ms = (time.perf_counter() - started) * 1000.0
    payload: dict[str, Any] = {
        "actions": chunks[0].tolist(),
        "timings": {"infer_ms": infer_ms},
    }
    if generations is not None:
        payload["generations"] = [
            dataclasses.asdict(generation) for generation in generations
        ]
    return payload


class PolicyServer(ThreadingHTTPServer):
    """ThreadingHTTPServer carrying the runtime for its handlers.
    Threads exist so a client can poll /spec while a slow /predict is
    in flight; predictions themselves serialize on the runtime lock."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], runtime: ServerRuntime) -> None:
        super().__init__(address, PolicyRequestHandler)
        self.runtime = runtime


class PolicyRequestHandler(BaseHTTPRequestHandler):
    """GET /spec and POST /predict; everything else 404. Every error
    path answers JSON ({error, traceback}) — the server survives any
    request; only startup may die."""

    def _runtime(self) -> ServerRuntime:
        assert isinstance(self.server, PolicyServer)  # constructed by it
        return self.server.runtime

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # http.server's dispatch convention names this
        if self.path == "/spec":
            self._send_json(200, self._runtime().spec)
            return
        self._send_json(
            404,
            {"error": f"unknown path {self.path!r} — GET /spec or POST /predict"},
        )

    def do_POST(self) -> None:  # http.server's dispatch convention names this
        if self.path != "/predict":
            self._send_json(
                404,
                {"error": f"unknown path {self.path!r} — GET /spec or POST /predict"},
            )
            return
        started = time.perf_counter()
        try:
            length_header = self.headers.get("Content-Length")
            length = int(length_header) if length_header is not None else 0
            payload = json.loads(self.rfile.read(length))
            request = parse_request(payload, self._runtime())
            decode_ms = (time.perf_counter() - started) * 1000.0
            response = predict_response(self._runtime(), request)
        except (RequestError, json.JSONDecodeError, ValueError) as error:
            self._send_json(400, {"error": str(error), "traceback": ""})
            return
        except SystemExit as error:
            # BijouPolicy's own loud refusals (user-fixable class) —
            # structured 400, never a dead server thread.
            self._send_json(
                400,
                {"error": str(error.code), "traceback": ""},
            )
            return
        except Exception as error:  # noqa: BLE001 — the server never dies on a request
            self._send_json(
                500,
                {
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                },
            )
            return
        total_ms = (time.perf_counter() - started) * 1000.0
        timings = response["timings"]
        timings["decode_ms"] = decode_ms
        timings["total_ms"] = total_ms
        self._send_json(200, response)
        print(
            f"[predict] index {request.index}: decode "
            f"{decode_ms:.0f} ms, infer {timings['infer_ms']:.0f} ms, "
            f"total {total_ms:.0f} ms",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--flow-decoder-dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="bfloat16 halves flow-decoder memory (post-load cast of the "
        "checkpoint's action decoder) — the rollout CLI's local flag, "
        "moved server-side for remote inference",
    )
    parser.add_argument(
        "--offload-ple",
        action="store_true",
        help="park the Gemma AR trunk's per-layer-embedding token table "
        "in host RAM (gemma_ar checkpoints only; other families refuse "
        "it loudly)",
    )
    parser.add_argument("--port", type=int, default=8143)
    parser.add_argument(
        "--bind",
        default="127.0.0.1",
        help="bind address; the default loopback is the supported "
        "posture (remote clients come in over an SSH tunnel). Anything "
        "else exposes an UNAUTHENTICATED model server and warns loudly",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="flow-noise seed (default: stochastic). Noise draws happen "
        "server-side, keyed by the client's replan index — reproducible "
        "under a fixed seed and the same request order",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # The rollout loop's serving-precision setting — remote inference
    # must run the numeric regime local rollouts run.
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    seed = args.seed if args.seed is not None else int(time.time())
    runtime = build_runtime(
        args.checkpoint,
        device=device,
        flow_decoder_dtype=getattr(torch, args.flow_decoder_dtype),
        seed=seed,
        offload_ple=args.offload_ple,
    )
    spec = runtime.spec
    print(
        f"policy: {runtime.policy.name} — family {spec['family']}, chunk "
        f"{spec['chunk_size']}, action_dim {spec['action_dim']}, recorded "
        f"serving {spec['serving']}",
    )
    print(
        f"checkpoint: {spec['checkpoint']} (step {spec['step']}), server "
        f"git rev {spec['git_rev']}, seed {seed}, "
        f"{args.flow_decoder_dtype} decoder on {args.device}",
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        allocated = torch.cuda.memory_allocated(device) / 2**30
        reserved = torch.cuda.memory_reserved(device) / 2**30
        print(
            f"GPU memory after load: {allocated:.2f} GiB allocated, "
            f"{reserved:.2f} GiB reserved",
        )
    if args.bind not in LOOPBACK_BINDS:
        print(
            f"WARNING: binding {args.bind} exposes an UNAUTHENTICATED "
            "plain-HTTP model server — anyone who can reach the port can "
            "drive inference. The supported remote path is a loopback "
            "bind plus an SSH tunnel: ssh -N -L "
            f"{args.port}:localhost:{args.port} <box>",
            flush=True,
        )
    server = PolicyServer((args.bind, args.port), runtime)
    print(
        f"serving on http://{args.bind}:{args.port} (GET /spec, POST "
        "/predict); first /predict pays kernel warmup; ctrl-c to stop",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping (keyboard interrupt)")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
