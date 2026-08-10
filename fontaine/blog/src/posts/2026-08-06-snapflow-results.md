# SnapFlow results: 1-NFE distillation holds the panel — single draw beats AR at one expert eval

*2026-08-06, 15:2xZ. Results for the
[SnapFlow pre-registration](2026-08-06-prereg-snapflow-distill.md) +
Amendment 1 (σ_draw finalization: adopt-signal iff chunk_mae ≤
6.7732). Run `fontaine_flow_snapdistill_h1024_30k_1xh100` (seed 1),
self-distilled from `bijou_flow_artrunk_h1024_40k_ddp2/step_080000`
with the φ_s target-time extension, trunk frozen. All frozen reads
through `fontaine/scripts/snapflow_results.py` (banked
oracle-before-data 09:xxZ, five oracles green before any endpoint
byte existed); report
[`analysis__snapflow_distill_30k_k4l2.json`](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__snapflow_distill_30k_k4l2.json).
Keying: **v1 panel, index keying** — the registered comparators
predate the #18.2 stable-key adoption; in-flight reads finish as
registered (record-only, stated not hidden).*

## Gates (all passed, banked at launch)

- Gate (a) zero-init identity oracle: 6/6 bit-exact — step-0
  extended model ≡ teacher.
- Gate (b) E1-style drift: step-0 Heun-30 s=t on the stride-7 subset
  reproduced the banked flow npz, frame-MAE drift **0.01451** < 0.05.
- @10k record-only 1-NFE probe: **5.9222 / 1.8193** — kill line
  (teacher probe 6.6755 + 3.0 = 9.6755) passed by 3.75; at one-third
  training the 1-NFE student already beat the teacher's own Heun-30
  probe read (6.676/1.928).

## The endpoint reads (30k, full 25,800-frame panel)

| config | expert evals | chunk_mae | first_mae |
|---|---|---|---|
| teacher Heun-30, single draw (banked) | 30 | 6.6232 | 1.9331 |
| teacher Heun-30, mean-of-10 (banked) | 300 | 5.365 | 1.424 |
| AR-100k anchor | — | 5.8026 | 2.1431 |
| **student 1-NFE, single draw (primary)** | **1** | **5.6036** | **1.7039** |
| student 1-NFE, mean-of-5 | 5 | 5.3918 | 1.6056 |
| **student 1-NFE, mean-of-10** | **10** | **5.3675** | **1.5927** |

State-copy control rows byte-match the banked values on all three
evals (11.7848 / 2.6202) — same panel, same join, quotable.

Full HTML eval reports (per-repo tables + worst-frame galleries):
[single draw](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_flow_snapdistill_h1024_30k_1xh100__step_030000__panel_curated_v0_k4l2_1nfe_euler1.html) ·
[mean-of-5](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_flow_snapdistill_h1024_30k_1xh100__step_030000__panel_curated_v0_k4l2_1nfe_euler1_draws5.html) ·
[mean-of-10](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_flow_snapdistill_h1024_30k_1xh100__step_030000__panel_curated_v0_k4l2_1nfe_euler1_draws10.html)
— teacher comparators on the [reports index](../reports.md).

## Verdict against the pre-registered lines

- **Primary (expectation 2): PARITY-ADOPT.** 1-NFE single-draw
  chunk_mae 5.6036 ≤ 6.7732 — the adopt-signal fires, with **1.02 to
  spare vs the teacher's own Heun-30** (the pre-reg's modal outcome
  was "parity or slightly better"; this is better by 15%). Not
  falsified, not a miss.
- **Deployment headline (expectation 3): FIRES.** Mean-of-10 @1-NFE
  **5.3675 ≤ 5.8026** — the draws win survives distillation at
  one-thirtieth the compute (10 expert evals vs 300), landing just
  under the modal band [5.4, 5.6] and matching the teacher's own
  mean-of-10 (5.365) to 3 dp. **The charter §2 cost caveat on the
  draws result closes.**
- **Grounding edge (expectation 4): SURVIVES.** first_mae 1.7039 ≤
  1.9831 — and improves on the teacher's single-draw 1.9331.
- **Kill line: never threatened** (probe passed by 3.75).

## The sharper headline: single-draw 1-NFE already beats AR

The pre-reg asked whether mean-of-10@1-NFE could hold the beat-AR
read at ~10-expert-eval cost. The answer is stronger: **a single
1-NFE draw (ONE expert eval) scores 5.6036 — already below the AR
anchor 5.8026.** The mean-of-N machinery is now an optional +0.24
refinement, not the price of admission.

## Draw diversity: mostly collapsed into the mean — and that is fine

Averaging gains: teacher −1.258 (6.6232 → 5.365 over 10 draws);
student −0.236 (5.6036 → 5.3675). The mean-of-5 point makes the
shape unambiguous: student 5.3918 at 5 draws — ~90% of its total
averaging gain banked by draw 5, the same fractional shape as the
teacher (87% by draw 5: 6.6232 → 5.5235) but at **one-fifth the
amplitude**. The student's mean-of-5 already beats the teacher's
mean-of-5 (5.3918 vs 5.5235), and its mean-of-10 lands 0.03 *below*
the pre-registered modal band [5.4, 5.6] (`inside_modal_band: false`
in the report) — the band was projected assuming teacher-level
residual draw variance, and the student has much less.

This is exactly the fairness-probe finding operating in reverse: the
1-NFE endpoint approximates the posterior mean (chunk MAE rewards
mode non-commitment), so the distilled student banks most of the
ensembling gain in every single draw, leaving little residual draw
spread to average over. The distillation did not preserve the
teacher's draw distribution — it compiled the mean of it. For
deployment-style chunk-MAE that is the profitable direction; anyone
needing mode diversity (best-of-N search, multimodal planning) should
stay on the Heun teacher, whose best-of-10 bound (3.8597 on the
probe subset) has no student counterpart.

## Per-step horizon read (addendum npz, pre-registered)

The addendum npz eval re-ran the primary with `--dump-predictions`
(the chained stage-4 evals dumped JSON only); npz-pooled
5.6036/1.7039 matches the chained JSON to 2e-05 — same eval,
byte-consistent. The pre-registered question — does the 1-NFE
student degrade faster along the horizon than the Heun-30 teacher?
— answers **no, the opposite**: the student sits below the teacher
at every one of the 50 horizon steps (`crossover_step: null`), and
the per-step delta *widens* monotonically from −0.229 at step 1 to
−1.554 at step 50. The cumulative first-k curve never crosses
either. Face-value this is the mean-collapse reading again: later
horizon steps carry more draw spread, and a mean-valued prediction
profits most exactly where spread is largest — so distillation
compressed late-horizon error hardest. The 2026-08-05 paired
finding (flow diverges from AR along the horizon) does not
transfer to student-vs-teacher.

## Panel-v2 descriptive column

Descriptive only (v1 index keying stays the registered read, per
the transition convention): on panel-v2 stable keying the student
scores **5.6711 / 1.7059** vs the teacher's banked 6.7151 / 1.9453
— margin −1.04, matching the v1 margin. The parity read is
keying-robust.

## Cost (record actual, per pre-reg)

30k steps in **~4.5 h wall** on 1×H100 (08:43Z → 13:14Z, 0.48–0.51
s/step) — under half the 12–20 h budget. Endpoint evals: draws-1
~28 min, draws-10 ~25 min at ~1,100 f/min, draws-5 ~26 min; npz
addendum re-run ~27 min. Final in-run s=t drift 7.6601 —
the velocity-mode read stayed ~1σ high all run and stayed
deconfirmed as a 1-NFE proxy (the @10k probe called the endpoint
correctly).

## What adoption means (owner decision)

The instrument's assembly (verbatim from the report):
`parity-adopt` — *"ADOPT-SIGNAL + DEPLOYMENT HEADLINE: mean-of-10
@1-NFE beats the AR anchor at ~10-expert-eval cost — the charter §2
cost caveat on the draws win closes."* Concretely, the proposal on
the table: the 1-NFE student becomes the deployment-class inference
config for this lineage — single draw (1 expert eval) as the
latency floor, mean-of-10 (~10 expert evals, still 3× cheaper than
one Heun-30 draw) as the quality mode. Owner sign-off requested;
until then the numbers stand as measured. Proposed follow-ons
stay as queued ideas: #1 Golden-Ticket noise search now has a 1-NFE
substrate (panel-side search is 30× cheaper), and the stage-4 eval
default (`--noise-key`) flips per #18.2 now that the chain is done.
