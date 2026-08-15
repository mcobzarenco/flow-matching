"""Model families — the concrete :class:`~bijou.vla.VLA`
implementations, one class per (trunk, trained-surface set), plus the
shared objective payloads (`objectives.py`). Families compose the
building blocks in ``bijou/modelling/`` (trunks, encoders, decoders,
codecs) and own their assembly, precision policy, and loss
composition; family-unique objective payloads co-locate with their
family module."""
