"""Re-export shim: the released-FAST tokenizer moved to
``bijou/fast/molmoact2.py`` (docs/molmoact2-retirement.md phase 1;
class renamed ``MolmoAct2FastCodec`` → ``MolmoAct2FastTokenizer``
2026-08-14 — it sits at the TOKENIZER layer of the codec/tokenizer
split, see docs/code-styleguide.md). Port-side call sites import
through here until the package is deleted (phase 5)."""

from __future__ import annotations

from ..fast.molmoact2 import MolmoAct2FastTokenizer

__all__ = ["MolmoAct2FastTokenizer"]
