# 8. Shortlist/output-vocab head for ar_backbone — `queued`

*Tag: `vocab-head` · idea #8 · [index](../ideas.md)*

The 262k-vocab CE softmax is the VRAM headroom eater; a shortlist
head raises feasible batch on 1×H100 (mainline queued it as the
structural fix after the B12 OOM). Cost: real code + an equivalence
check (loss oracle moves → loud re-baseline). Payoff multiplies every
future ar_backbone run on this box. **Design concretized (deep-dive
2026-08-05):** chunked/fused linear-CE (logsumexp vs `lm_head.weight`
+ the 1026-row patch; elementwise softcap fuses) — never materialize
the `[B·S, 262k]` fp32 logits (~1 GiB at B10,
`ar_backbone.py:743-748`). Decode-side: action-phase argmax over
block columns only is exact (grammar mask + monotone softcap).

- **Owner measurement (2026-08-05 21:52Z, in-channel + html
  attachment): FAST round-trip error is barely measurable** —
  quantization is not the binding limit of the AR approach, and AR
  "definitely trains faster". Consistent with the paired analysis
  (AR wins the late horizon — a codec-bound model wouldn't): the
  limit is upstream of the codec, in trunk/grounding. Strengthens
  the AR-side weighting of the attribution front (owner steer
  21:48Z).
