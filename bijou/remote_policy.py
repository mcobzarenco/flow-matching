"""Remote-inference client for the rollout loop, plus the wire contract
it shares with ``bijou.policy_server``.

:class:`RemotePolicy` exposes the surface ``bijou.rollout`` consumes
from a local ``BijouPolicy`` — ``info`` (:class:`~bijou.data.PolicyInfo`),
``name``/``generate``/``tickets`` for the banner, and
``predict_with_text`` — but ships each observation to a policy server
over HTTP instead of running the model: cameras and robot stay local,
the checkpoint lives on the GPU box, and the sync loop's replan block
simply gets longer by one round trip (async/latency-hiding over the WAN
is deliberately out of scope). The transport is unauthenticated plain
HTTP; the supported remote path is an SSH tunnel to a loopback-bound
server.

This module is deliberately import-light (stdlib + torch/cv2/numpy +
leaf bijou modules, no model/tokenizer imports), so it also hosts the
pieces of the wire contract both ends must agree on:
``WIRE_SCHEMA_VERSION``, the JPEG image codec pair
(:func:`encode_jpeg`/:func:`decode_jpeg` — the server imports its
half from here, so the parity oracle can compose the exact functions
both sides run), and the per-item stats payload
(:func:`stats_payload`, mirroring ``DatasetStats.state_dict``).

Failure policy: timeouts, connection errors, non-200 responses and a
schema-version mismatch all raise ``SystemExit`` with the remedy — no
silent retries; a dead tunnel must stop the control loop, not starve
it. Noise is drawn server-side (the server's ``--seed``); the request
carries the replan index so the "index" keying matches the local
path's semantics.
"""

from __future__ import annotations

import base64
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor

from .data import DatasetStats, PolicyInfo
from .modelling.aux_text import AuxField, AuxGeneration
from .modelling.interface import SamplingMethod

# The /spec + /predict JSON contract version. Bump on any change a
# deployed client/server pair could disagree about; the client refuses
# a mismatched server at construction (never mid-rollout).
WIRE_SCHEMA_VERSION = 1

# Fixed encoder setting so client and parity oracle produce identical
# bytes (cv2's own default is also 95; pinning it here removes the
# ambient dependency). Measured on 640x480 rig frames: ~30-60 KiB per
# frame, ~0.2 ms to encode.
JPEG_QUALITY = 95


def encode_jpeg(image: Tensor) -> bytes:
    """One camera frame → JPEG bytes (the wire image format).

    Exact inverse of the rollout item construction back to the camera's
    uint8 pixels (``round(x·255)`` recovers ``u/255`` bit-exactly), then
    cv2's JPEG encoder at :data:`JPEG_QUALITY` (lossy — the parity
    oracle feeds both inference paths the DECODED frames).

    Shapes:
      - ``image``: [3, H, W] float in [0, 1], RGB (the rollout item
        convention)
      - returns: JPEG bytes
    """
    pixels = (
        (image * 255.0).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
    )
    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
    )
    if not ok:
        raise ValueError(f"cv2.imencode refused a {tuple(image.shape)} frame")
    return encoded.tobytes()


def decode_jpeg(data: bytes) -> Tensor:
    """JPEG bytes → the collator's camera-frame tensor (the server's
    half of the image codec; lives beside :func:`encode_jpeg` so the
    pair is tested as one round trip).

    Shapes:
      - returns: [3, H, W] float in [0, 1], RGB
    """
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("not a decodable JPEG image")
    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def stats_payload(item: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    """The request's normalization block, read off the item's stats
    tensors — exactly the vectors the local path feeds the collator
    (``DatasetStats.item_tensors`` attached them), serialized in the
    ``DatasetStats.state_dict`` shape so the server round-trips through
    ``DatasetStats.from_state_dict`` without re-flooring. Quantile keys
    ride along iff the item carries them (old checkpoint tables do not).

    Shapes:
      - ``item['action_*']``: [action_dim]; ``item['state_*']``:
        [state_dim] (per-item stats tensors)
    """
    payload: dict[str, dict[str, list[float]]] = {
        "action": {
            "mean": item["action_mean"].tolist(),
            "std": item["action_std"].tolist(),
        },
        "observation.state": {
            "mean": item["state_mean"].tolist(),
            "std": item["state_std"].tolist(),
        },
    }
    if "action_q01" in item:
        payload["action"]["q01"] = item["action_q01"].tolist()
        payload["action"]["q99"] = item["action_q99"].tolist()
    if "state_q01" in item:
        payload["observation.state"]["q01"] = item["state_q01"].tolist()
        payload["observation.state"]["q99"] = item["state_q99"].tolist()
    return payload


def git_rev(directory: Path) -> str:
    """HEAD of the checkout containing ``directory`` (provenance for
    the banner/spec), or ``"unknown"`` when git or the .git directory
    is absent — a box scratch dir of rsync'd tracked files has none."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except OSError:
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def _transport_error(url: str, error: OSError, timeout: float) -> SystemExit:
    return SystemExit(
        f"policy server unreachable: {url} ({type(error).__name__}: {error}) "
        f"after timeout {timeout:g}s — no retries. Check that the server is "
        "up (tmux on the GPU box) and the SSH tunnel is alive "
        "(ssh -N -L <port>:localhost:<port> <box>), then rerun.",
    )


def _request_json(
    url: str,
    *,
    timeout: float,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """GET (payload None) or POST ``url`` and parse the JSON response.
    Non-200s raise SystemExit carrying the server's ``{error,
    traceback}`` body; transport errors raise SystemExit with the
    tunnel remedy — the client never retries silently."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"error": raw.decode(errors="replace"), "traceback": ""}
        if not isinstance(parsed, dict):
            parsed = {"error": raw.decode(errors="replace"), "traceback": ""}
        server_traceback = str(parsed.get("traceback", ""))
        detail = f"\nserver traceback:\n{server_traceback}" if server_traceback else ""
        raise SystemExit(
            f"policy server returned {error.code} for {url}: "
            f"{parsed.get('error', '<no error field>')}{detail}",
        ) from error
    except OSError as error:
        raise _transport_error(url, error, timeout) from error
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise SystemExit(f"policy server sent a non-object JSON response from {url}")
    return parsed


class RemotePolicy:
    """The rollout loop's policy handle for a remote checkpoint.

    Construction fetches ``/spec``, refuses a wire-schema mismatch,
    validates the requested options against the server's declared
    capabilities (the loud-narrowing rule — a --generate the family
    cannot back dies HERE, before the robot connects, exactly like the
    local constructor), builds :class:`~bijou.data.PolicyInfo` from the
    spec's stats tables (so ``--stats-repo-id`` keeps its local lookup
    behavior), and prints the banner — WARNING, not error, on a
    server-vs-local git-rev mismatch.

    ``predict_with_text`` mirrors ``BijouPolicy.predict_with_text`` for
    the single-observation batches the rollout loop sends: JPEG-encode
    the item's camera frames, POST, decode ``[chunk, action_dim]``
    chunks (float32, bit-identical to the server's tensors — JSON
    round-trips float32 exactly) and the narration generations. The
    server serializes requests (one robot, one client), so per-request
    options ride each POST.

    ``tickets``/``tickets_sha256`` are always None: the ticket bank is
    a server-side noise substitution the wire protocol does not carry.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float,
        generate: tuple[AuxField, ...] = (),
        sample_steps: int = 10,
        method: SamplingMethod = SamplingMethod.HEUN,
        sample_draws: int = 1,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.generate = generate
        self.sample_steps = sample_steps
        self.method = method
        self.sample_draws = sample_draws
        self.tickets: Tensor | None = None
        self.tickets_sha256: str | None = None
        # Server-side timings of the LAST predict (decode_ms/infer_ms/
        # total_ms) — the measurement surface for the replan-gap
        # arithmetic; None until the first predict.
        self.last_timings: dict[str, float] | None = None
        spec = _request_json(f"{self.base_url}/spec", timeout=timeout)
        version = spec.get("schema_version")
        if version != WIRE_SCHEMA_VERSION:
            raise SystemExit(
                f"policy server speaks wire schema_version {version!r}, this "
                f"client speaks {WIRE_SCHEMA_VERSION} — update the older "
                "side (server and laptop checkouts must agree on the "
                "protocol)",
            )
        self.family: str = str(spec["family"])
        self.serving: dict[str, Any] = dict(spec["serving"])
        self.checkpoint: str = str(spec["checkpoint"])
        self.step: int = int(spec["step"])
        self.server_git_rev: str = str(spec["git_rev"])
        # Whether the checkpoint normalizes state with its ONE baked-in
        # global table (molmo_flow decoders) — drives rollout's
        # model-frame envelope gate, which the client cannot infer
        # without model imports.
        self.global_state_table: bool = bool(spec["global_state_table"])
        narrating = bool(spec["narrating"])
        draw_ensembling = bool(spec["draw_ensembling"])
        self.info = PolicyInfo(
            chunk_size=int(spec["chunk_size"]),
            normalization=DatasetStats.from_state_dict(spec["stats"]),
            per_dataset_normalization={
                repo_id: DatasetStats.from_state_dict(table)
                for repo_id, table in spec["per_dataset_stats"].items()
            },
            condition_fields=tuple(spec["condition_fields"]),
            generate_bracket=bool(spec["generate_bracket"]),
        )
        self.action_dim: int = int(spec["action_dim"])
        # The BijouPolicy naming convention, with a _remote suffix so a
        # wire-served run is never mistakable for a local one in logs.
        self.name = f"bijou@{self.step}_remote"
        if sample_draws > 1:
            self.name += f"_draws{sample_draws}"
        # Loud narrowing at construction, mirroring the local
        # BijouPolicy (the server re-validates per request, but a bad
        # ask must die before the robot connects).
        if len(generate) > 0 and not narrating:
            raise SystemExit(
                f"--generate requested but {self.family} has no narration surface",
            )
        if sample_draws < 1:
            raise SystemExit(f"--sample-draws must be >= 1, got {sample_draws}")
        if sample_draws > 1 and not draw_ensembling:
            raise SystemExit(
                f"--sample-draws {sample_draws} needs a draw-ensembling "
                f"flow decode, which the server's {self.family} checkpoint "
                "does not offer (molmo_flow-decoder families run "
                "single-draw) — drop the flag",
            )
        serving_tag = "-".join(
            str(self.serving[key])
            for key in ("kind", "method", "num_steps")
            if key in self.serving
        )
        print(
            f"policy server: {self.base_url}\n"
            f"  family {self.family}, chunk {self.info.chunk_size}, "
            f"action_dim {self.action_dim}, recorded serving {serving_tag}\n"
            f"  checkpoint {self.checkpoint} (step {self.step}), "
            f"server git rev {self.server_git_rev}",
            flush=True,
        )
        local_rev = git_rev(Path(__file__).resolve().parent)
        if "unknown" in (local_rev, self.server_git_rev):
            print(
                f"  git-rev comparison unavailable (local {local_rev}, "
                f"server {self.server_git_rev})",
                flush=True,
            )
        elif local_rev != self.server_git_rev:
            print(
                f"WARNING: policy-server git rev {self.server_git_rev} != "
                f"local checkout {local_rev} — the box may run different "
                "inference code than what was scored offline",
                flush=True,
            )

    def predict_with_text(
        self,
        items: list[dict[str, Any]],
        indices: list[int],
    ) -> tuple[list[Tensor], list[AuxGeneration] | None]:
        """One observation's chunk (plus generations when narration was
        requested) through the wire — the exact surface the rollout
        loop consumes from the local policy.

        Shapes:
          - ``items[0]['observation.state']``: [state_dim]
          - ``items[0]['observation.images.*']``: [3, H, W] float [0, 1]
          - returns chunks: one [chunk, action_dim] float32 per item
        """
        if len(items) != 1 or len(indices) != 1:
            raise SystemExit(
                f"RemotePolicy serves single-observation batches (the "
                f"rollout loop's shape), got {len(items)} items — batch "
                "prediction stays local",
            )
        item = items[0]
        images = {
            key.removeprefix("observation.images."): base64.b64encode(
                encode_jpeg(value),
            ).decode("ascii")
            for key, value in item.items()
            if key.startswith("observation.images.")
        }
        request = {
            "task": item["task"],
            "state": [float(x) for x in item["observation.state"]],
            "images": images,
            "stats": stats_payload(item),
            "camera_kinds": dict(item.get("camera_kinds") or {}),
            "conditions": {
                key.removeprefix("condition_"): value
                for key, value in item.items()
                if key.startswith("condition_")
            },
            "index": int(indices[0]),
            "options": {
                "generate": [field.value for field in self.generate],
                "num_steps": self.sample_steps,
                "method": self.method.value,
                "num_samples": self.sample_draws,
            },
        }
        started = time.perf_counter()
        response = _request_json(
            f"{self.base_url}/predict",
            timeout=self.timeout,
            payload=request,
        )
        wall_ms = (time.perf_counter() - started) * 1000.0
        chunk = torch.tensor(response["actions"], dtype=torch.float32)
        expected = (self.info.chunk_size, self.action_dim)
        if tuple(chunk.shape) != expected:
            raise SystemExit(
                f"policy server sent actions shaped {tuple(chunk.shape)}, "
                f"spec says {expected} — server/client disagree about the "
                "checkpoint",
            )
        timings = {key: float(value) for key, value in response["timings"].items()}
        self.last_timings = timings
        print(
            f"  server | infer {timings['infer_ms']:.0f} ms, decode "
            f"{timings['decode_ms']:.0f} ms | wire+encode "
            f"{max(wall_ms - timings['total_ms'], 0.0):.0f} ms",
            flush=True,
        )
        generations: list[AuxGeneration] | None = None
        if "generations" in response:
            generations = [
                AuxGeneration(**generation) for generation in response["generations"]
            ]
        return [chunk], generations
