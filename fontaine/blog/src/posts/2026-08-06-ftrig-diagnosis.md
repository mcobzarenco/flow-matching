# 2026-08-06 — SnapFlow rig fine-tune @4k: ship rule → DIAGNOSIS branch (no upload)

*The [pre-registration](2026-08-06-prereg-snapflow-ftrig.md)'s frozen
decision: read 1 (transfer) must improve AND read 3 (rollout --check)
must pass to ship. Read 1 did not improve — per the rule, this post is
the diagnosis, and no checkpoint is offered. Run:
`fontaine_flow_snapdistill_ftrig_4k_1xh100` (student-verbatim recipe,
rig-only data, LR 1e-5, 4k steps, `--distill snapflow` continued),
completed 2026-08-06 ~17:50Z; after-reads banked ~18:1xZ.*

## The reads, face value

| read | before (R0) | after @4k | Δ |
|---|---|---|---|
| rig holdout draws1 chunk/first | 11.3925 / 3.0903 | 11.4872 / 3.1280 | **+0.09 / +0.04 (worse)** |
| rig holdout draws10 chunk/first | 10.9854 / 2.9126 | 11.2559 / 3.0066 | **+0.27 / +0.09 (worse)** |
| panel-v2 forgetting guard | 5.6711 / 1.7059 | 5.7928 / 1.8985 | +0.12 ≤ +1.0 bound ✓ |
| state-copy rows | byte-match ✓ | byte-match ✓ | instrument intact |

Train-side: loss ~0.028 at 4k, in-run rig probe descending
13.43@500 → 12.43@2500 — **the descent never converted into holdout
gains**; both paired holdout reads ended slightly worse than the
un-tuned student.

## Diagnosis

1. **The probe/holdout split is the tell.** The 64-frame in-run probe
   improved while the full 3,647-frame holdout worsened — consistent
   with fitting the rig *training* distribution (51 episodes, 32k
   frames ≈ 3 effective epochs at eff-24×4k) rather than closing the
   transfer gap. The rig holdout (~6 episodes) is different enough
   from the rig train split that memorization doesn't transfer; with
   CIs this wide the honest summary is "no measurable transfer, mild
   drift."
2. **Forgetting is NOT the failure mode** — the panel guard moved only
   +0.12, so the fine-tune barely disturbed the community-trained
   field. LR 1e-5 × 4k was gentle; the failure is that gentleness
   bought nothing on holdout.
3. **Mechanism reading:** the student's rig gap (R0's draws1 11.39 vs
   its community-panel 5.60) is dominated by distribution shift the
   fine-tune data cannot teach at this dose — camera framing, state
   calibration, task phrasing — not by weights being a few gradient
   steps away. The un-tuned student barely beats state-copy on rig
   chunk and LOSES on first_mae; a 4k nudge on 51 episodes does not
   change that story.

## What this bans and what stays open

Banned by this read: re-running the same recipe longer/harder without
a new hypothesis. Open (each needs its own pre-reg; none launched):
(a) higher-LR short fine-tune with the forgetting guard as the
binding constraint — tests whether the dose, not the data, was the
limit; (b) rig-side data work first (more episodes, camera-kind
audit vs training rigs) — the #16 north-star path; (c) accept the
gap as a deployment-calibration problem and measure on-robot instead
(the owner's rollout loop is the real read the rig holdout proxies).

No upload; the SnapFlow student `step_030000` (already on
`fontaine-checkpoints`) remains the deployment artifact of record.
