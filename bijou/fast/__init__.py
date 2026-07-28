"""FAST action-chunk tokenization (DCT + BPE, arXiv:2501.09747).

Owned reimplementation of ``physical-intelligence/fast`` — algorithm and
artifact schema in ``tokenizer.py``, corpus-fit CLI in ``cli.py`` (run
with ``python -m bijou.fast``). The public API is re-exported here.
"""

from .tokenizer import (
    BPE_FILENAME,
    CONFIG_FILENAME,
    QUANTILE_FILENAME,
    FastDecodeError,
    FastTokenizer,
    QuantileEntry,
    dct_matrix,
    load_quantile_table,
    quantile_entry_for,
    save_quantile_table,
)

__all__ = [
    "BPE_FILENAME",
    "CONFIG_FILENAME",
    "QUANTILE_FILENAME",
    "FastDecodeError",
    "FastTokenizer",
    "QuantileEntry",
    "dct_matrix",
    "load_quantile_table",
    "quantile_entry_for",
    "save_quantile_table",
]
