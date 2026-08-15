"""Model families — the concrete :class:`~bijou.vla.VLA`
implementations, one class per (trunk, trained-surface set), plus the
shared objective payloads (`objectives.py`) and recorded serving
operating points (`serving.py`). Families compose the building blocks
in ``bijou/modelling/`` (trunks, encoders, decoders, codecs) and the
section builders in ``bijou.sections``, and own their assembly,
precision policy, and loss composition; shared machinery is free
functions (`ar_suffix_ops.py`, the molmoact2 assembly helpers), never
intermediate base classes; family-unique objective payloads co-locate
with their family module."""
