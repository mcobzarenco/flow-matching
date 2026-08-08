"""Frozen scorers for subgoal-draws selection (#6 rung (b)).

Pure arithmetic over the per-candidate decode stats that
``ARSuffixDecoder.decode_value_line`` records — no torch, no model
access, so the read script and the exact-arithmetic selftest fixtures
recompute every pick bit-for-bit from the ``--dump-subgoal-candidates``
JSON alone (pre-reg 2026-08-08-prereg-subgoal-draws.md, "Scorer").

Conventions frozen by the pre-reg:

- PRIMARY  self-certainty (2502.18581, argmax form): SC(y) = mean over
  the candidate's decode steps of KL(uniform ‖ p_step) computed on the
  decode's OWN next-token distributions — here the text-masked value
  softmax the decode actually chose/sampled from, with V = the number
  of legal text ids. Per step, KL(U ‖ p) = −log V − mean_j log p_j, so
  the dumped per-step mean log-prob over the legal vocabulary is the
  exact sufficient statistic. Zero extra forward passes, no label
  access anywhere.
- CEILING  token-level F1 vs the TRUE segment label (lowercase,
  whitespace tokens, multiset overlap) — record-only oracle similarity,
  bounding every scorer at this candidate width. Never reachable
  without an explicit label argument (provenance separation, oracle v).
- ALTERNATES (record-only, computed offline from the same dumps):
  mean chosen-token log-prob (likelihood) and the medoid by mean
  token-F1 to the other candidates. Their conditioned deltas are NOT
  measured; a future pre-reg may promote one.
- Every argmax breaks exact ties toward the LOWEST candidate index
  (greedy first). Duplicate strings decode identical distributions, so
  dedup is implicit in the tie rule.
- ELIGIBLE LIST (rung (b'), pre-reg …-cleanlist): under the clean-list
  filter every scorer sees only non-truncated candidates
  (``eligible_indices``); an all-truncated row falls back to the greedy
  candidate as decoded, recorded. Picks stay original-list indices.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence


def eligible_indices(truncated: Sequence[bool]) -> list[int]:
    """The rung-(b') frozen eligible-candidate rule (pre-reg
    2026-08-08-prereg-subgoal-draws-cleanlist): every scorer operates on
    the candidates with ``truncated == False``; when ALL candidates are
    truncated the list falls back to ``[0]`` (greedy as decoded — the
    caller records the fallback row). Returned indices are
    ORIGINAL-list indices in ascending order, so a sublist argmax
    mapped back through them preserves the greedy-first tie rule."""
    if not truncated:
        raise ValueError("eligibility over zero candidates")
    keep = [index for index, forced in enumerate(truncated) if not forced]
    return keep or [0]


def _argmax_lowest(scores: Sequence[float]) -> int:
    """Index of the maximum; exact ties break toward the lowest index
    (the pre-reg's greedy-first rule). Loud on an empty sequence."""
    if not scores:
        raise ValueError("argmax over zero candidates")
    best = 0
    for index, score in enumerate(scores):
        if score > scores[best]:
            best = index
    return best


def self_certainty(mean_logprob: Sequence[float], allowed_vocab: int) -> float:
    """SC of one candidate from its per-step mean log-probs over the
    legal text vocabulary: mean_i [ −log V − mean_logprob_i ]. Every
    candidate has ≥ 1 recorded step (its terminator step at minimum;
    empty candidates terminate immediately but still decode one
    distribution), so a zero-step record is instrument breakage."""
    if not mean_logprob:
        raise ValueError(
            "self-certainty over zero decode steps — even an empty "
            "candidate records its terminator step",
        )
    if allowed_vocab < 2:
        raise ValueError(f"allowed_vocab {allowed_vocab} is not a text vocabulary")
    return -sum(mean_logprob) / len(mean_logprob) - math.log(allowed_vocab)


def mean_chosen_logprob(chosen_logprob: Sequence[float]) -> float:
    """Length-normalized candidate log-likelihood (the record-only
    likelihood alternate): mean over decode steps of the emitted
    token's log-prob under the same masked value softmax."""
    if not chosen_logprob:
        raise ValueError("mean log-prob over zero decode steps")
    return sum(chosen_logprob) / len(chosen_logprob)


def token_f1(a: str, b: str) -> float:
    """Token-level F1: lowercase, whitespace tokenization, multiset
    overlap (2·|a ∩ b| / (|a| + |b|)). Degenerate cases frozen: both
    empty → 1.0 (identical), exactly one empty → 0.0."""
    tokens_a = a.lower().split()
    tokens_b = b.lower().split()
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = sum((Counter(tokens_a) & Counter(tokens_b)).values())
    return 2.0 * overlap / (len(tokens_a) + len(tokens_b))


def self_certainty_pick(
    per_candidate_mean_logprob: Sequence[Sequence[float]],
    allowed_vocab: int,
) -> int:
    """The PRIMARY selection: argmax self-certainty, ties → lowest
    index. Deliberately takes only distribution stats — no candidate
    text, no label (the deployment-honest scorer surface)."""
    return _argmax_lowest(
        [self_certainty(steps, allowed_vocab) for steps in per_candidate_mean_logprob],
    )


def likelihood_pick(per_candidate_chosen_logprob: Sequence[Sequence[float]]) -> int:
    """Record-only alternate: argmax mean chosen-token log-prob."""
    return _argmax_lowest(
        [mean_chosen_logprob(steps) for steps in per_candidate_chosen_logprob],
    )


def medoid_pick(texts: Sequence[str]) -> int:
    """Record-only alternate: the candidate maximizing mean token-F1 to
    the OTHER candidates (consensus without a vote to count). A single
    candidate is its own medoid."""
    if not texts:
        raise ValueError("medoid over zero candidates")
    if len(texts) == 1:
        return 0
    return _argmax_lowest(
        [
            sum(token_f1(text, other) for j, other in enumerate(texts) if j != i)
            / (len(texts) - 1)
            for i, text in enumerate(texts)
        ],
    )


def ceiling_pick(texts: Sequence[str], true_label: str) -> int:
    """The record-only ORACLE-similarity ceiling: argmax token-F1 vs
    the TRUE segment label. The explicit ``true_label`` argument is the
    provenance boundary — no deployment-named path may call this
    (pre-reg oracle v); label-less frames never reach it (their ceil
    row renders no subgoal, the rung-(a) oracle-arm convention)."""
    return _argmax_lowest([token_f1(text, true_label) for text in texts])
