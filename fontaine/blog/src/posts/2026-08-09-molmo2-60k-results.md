# Molmo2 60k continuation: IMPROVED — the attach screen repoints to step_060000

*2026-08-09 00:4x–01:0xZ. The +20k continuation
([pre-reg](2026-08-08-prereg-molmo2-ar-60k-continuation.md), owner
call 08:49Z 08-08) closed 23:21Z with its chained greedy panel eval
landing 23:49Z; this is the canonical frozen read — paired vs the
banked 40k endpoint npz via the new oracle-gated
`molmo2_60k_results.py` (planted-delta fixtures + 4 abort
branches), analysis banked in
`analysis__molmo2_60k_vs_40k_k4l2.json`. Instrument integrity
clean: identity and state-copy columns byte-match between the two
endpoint npzs; both npzs re-pool to their reports exactly.*

## The reads, in pre-reg order

**Read 1 (primary): was +20k worth it?** Paired per-frame
Δ(60k − 40k) on the 17,204 core frames, seeded bootstrap CI95:
**−0.1388 [−0.194, −0.090] — IMPROVED**, CI entirely below zero.
Pooled: 40k 6.0079/2.1871 → 60k **5.8602/2.0719** (chunk/first).
The fresh-data + restored-LR bet (expectation 3, confidence
medium-high) paid out at roughly −0.007/1k-steps.

**Read 2 (the owner bar):** 5.8602 vs AR-100k greedy **5.8026 —
NOT passed** (+0.058, cross-trunk unpaired, quoted with that
caveat). This was expectation 4, confidence honestly "open". Two
notes that keep it interesting: the *first_mae* side is already
past the bar (2.0719 vs 2.1431), and the Molmo2 trunk now sits
0.06 behind a run trained 1.67× longer on the Gemma trunk — at
40k the gap was 0.21 at 2.5× fewer steps.

**Read 3 (integrity):** state-copy / state-copy-norm byte-match
across both endpoint npzs (11.78475/2.62023 — the banked panel
values). Hard-abort oracle, passed silently.

**Read 4 (probe trajectory, record-only):** the rewarmed segment's
probe low is **6.0062@57000 — no new low** vs 5.91@26,500
(expectation 2 not met). The probe subset and the panel disagree
about the segment: the panel improved −0.139 paired while the probe
never beat its 26.5k low — a reminder the probe is a small-sample
kill-switch, not a headline.

**Read 5 (decision, frozen):** IMPROVED without the bar ⇒ **the
60k endpoint replaces the 40k endpoint as the phase-2 flow-trunk
candidate and the attach screen warm-starts from `step_060000`**,
with the bar miss noted honestly.

## What the repoint changed (amendment 3, landed with this post)

Per the attach pre-reg's
[amendment 3](2026-08-07-prereg-molmo2-attach-screen.md): both arm
launchers and the K-smoke ladder repoint their `ENDPOINT` to
`fontaine_molmo2_ar_60k_ddp4/step_060000`;
`attach_seam_results.py`'s read-4 trunk-drift comparator now pulls
the 60k panel json (band 0.3 unchanged, oracle re-run green — drift
is measured *from the warm-start trunk*, which moved). The K-smoke
memory ladder must re-run GREEN at the 60k warm start before either
arm launches. Weights already on the hub
(`fontaine-checkpoints/fontaine_molmo2_ar_60k_ddp4/step_060000`,
verified 00:0xZ).

Leaderboard: row 8 (60k greedy) added; the own-topology board row
moves to the 60k endpoint. The 40k rows stand as banked history.

Cost: the continuation ran ~49 GPU-h of the 60 ceiling; K1 never
armed; the async-save path validated in production across the run
(first-save lines checked at first babysit per the standing rule).
