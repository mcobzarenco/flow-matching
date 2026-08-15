"""Modelling building blocks — trunks, encoders, decoders, codecs, and
the seam currencies families compose. Nothing in this package imports
`bijou.vla` or `bijou.models`; the import DAG runs strictly downward:
``models/* → vla → modelling/{encoders,decoders} →
modelling/{gemma4,molmo2,interface,aux_text,nn} → fast``."""
