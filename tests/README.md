# Test tiers

Two tiers, one marker:

- **Default (CPU)** — everything unmarked. `uv run check.py` and bare
  `uv run pytest -m "not gpu" tests` run this tier; it is the commit
  gate and must stay fast (~22 s at 374 tests).
- **GPU** — mark with `@pytest.mark.gpu` any test that needs a CUDA
  device (real-kernel oracles, VRAM-shaped repros). Excluded from the
  default run; included by `uv run check.py --gpu` (run it on a GPU
  machine after math-adjacent changes to CUDA-path code).

Rules:

- A gpu-marked test must not be the only oracle for a change — keep a
  CPU-tier twin of the math (the chunked-backward oracle is the
  precedent: same contraction checked on CPU tensors).
- Markers are registered in `pyproject.toml` under
  `[tool.pytest.ini_options]`; `--strict-markers` is on, so a typo'd
  marker fails loudly instead of silently joining the default tier.
- `tests/test_check_tiers.py` is the oracle for the tiering itself.

Bare `uv run pytest tests` runs both tiers — fine on a GPU machine,
will fail on CPU-only ones once gpu-marked tests exist.
