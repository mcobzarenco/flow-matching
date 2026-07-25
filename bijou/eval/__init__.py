"""Open-loop checkpoint evaluation.

Run with ``python -m bijou.eval`` — see ``bijou/eval/__main__.py``. Scores
one or more policies (always the trivial state-copy baseline; optionally a
bijou checkpoint and/or a SmolVLA policy) on the SAME sampled frames of a
dataset selection, reporting pad-masked chunk metrics in raw action units
plus paired comparisons against the baseline.
"""
