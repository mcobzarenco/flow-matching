"""Pure-torch Molmo2 (Qwen3-4B decoder geometry) — the second trunk cell.

The full stack of the port plan
(`fontaine/blog/src/posts/2026-08-06-molmo2-port-plan.md`): text decoder
(``text``, with the truncated-mount loader in ``loading``), vision tower +
connector (``vision``), native image/prompt processing (``processor``) and
tokenizer (``tokenizer``), the full-model assembly (``model``), and the
per-layer KV cache for the AR suffix role (``cache``). The flow path stays
cache-free by design decision D1 — residual taps are its only export.
Parity gates: CPU oracles in ``tests/test_molmo2_*``, golden processor
fixtures banked by ``bank_processor_goldens`` in the reference 4.x venv,
and ``verify_parity`` against HF on real weights.
"""
