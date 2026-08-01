"""Minimal local stubs for mujoco (C-extension package: no py.typed, no
upstream .pyi, attributes assembled at import time — pyright sees
nothing). Everything is Any-typed on purpose: this marks the third-party
boundary so OUR sim code stays fully checked; it makes no claim about
mujoco's real signatures. Extend as call sites grow."""

from typing import Any

def __getattr__(name: str) -> Any: ...
