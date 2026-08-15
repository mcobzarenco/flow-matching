"""Pure-torch Molmo2 (Qwen3-4B decoder geometry) — the second trunk cell.

The full stack: text decoder (``text``, with the truncated-mount
loader in ``loading``), vision tower + connector (``vision``), native
image/prompt processing (``processor``) and tokenizer (``tokenizer``),
the full-model assembly (``model``), and the per-layer KV cache for
the AR suffix role (``cache``).
Parity gates: CPU oracles in ``tests/test_molmo2_*``, golden processor
fixtures banked by ``bank_processor_goldens`` in the reference 4.x venv,
and ``verify_parity`` against HF on real weights.
"""
