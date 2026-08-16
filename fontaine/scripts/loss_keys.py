"""Read-time loss-key mapping for train-metrics jsonls (main 32149df).

Trunk 32149df re-keyed the per-component loss series by MECHANISM —
``loss_action_flow`` (flow-matching MSE), ``loss_action_ar`` (discrete
action-token CE), ``loss_narration`` (value-line text CE); the combined
``loss`` is unchanged — with NO write-side aliases: jsonls written
before the rename keep their historical keys and mean what they meant.
Overlay/read scripts that put series from both sides of the rename on
one chart translate OLD records into the new vocabulary at read time
via :func:`canonical_loss_record`. The mapping is run-family-dependent
because the historical ``loss_aux`` named two different quantities
(owner-pinned mapping, Discord message.txt 2026-08-16 09:19Z).
"""

# family -> {historical key -> mechanism-qualified key}. Families whose
# historical logs carried no component series (single-head runs predating
# 32149df logged only the combined `loss`) still get an entry so lookups
# never silently fall through to a wrong guess.
FAMILY_KEY_MAP: dict[str, dict[str, str]] = {
    # flow families: loss_action was the flow-matching MSE
    "gemma_flow": {"loss_action": "loss_action_flow"},
    "molmoact2_flow": {"loss_action": "loss_action_flow"},
    # joint: loss_aux was discrete ACTION-token CE, not narration
    "molmoact2_joint": {
        "loss_action": "loss_action_flow",
        "loss_aux": "loss_action_ar",
    },
    # AR families: loss_action was the action-token CE, loss_aux the
    # value-line narration CE
    "gemma_ar": {"loss_action": "loss_action_ar", "loss_aux": "loss_narration"},
    "molmo2_ar": {"loss_action": "loss_action_ar", "loss_aux": "loss_narration"},
    "molmoact2_ar": {"loss_action": "loss_action_ar", "loss_aux": "loss_narration"},
}

NEW_KEYS = ("loss_action_flow", "loss_action_ar", "loss_narration")


def canonical_loss_record(record: dict, family: str) -> dict:
    """Return ``record`` with historical component keys renamed to the
    mechanism-qualified vocabulary for ``family``. Records already in the
    new vocabulary pass through unchanged; ``loss`` is never touched.
    Unknown families raise — a wrong family silently mislabels a curve.
    """
    mapping = FAMILY_KEY_MAP.get(family)
    if mapping is None:
        raise ValueError(
            f"unknown family {family!r} — one of {sorted(FAMILY_KEY_MAP)}",
        )
    if any(k in record for k in NEW_KEYS):
        return dict(record)
    return {mapping.get(k, k): v for k, v in record.items()}
