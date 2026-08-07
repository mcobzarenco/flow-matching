# 18. Instrument & infra hardening — `screening` (items 1+8 done 2026-08-05, 2 flag-landed, 3+4+7 done 2026-08-06)

*Tag: `infra-hardening` · idea #18 · [index](../ideas.md)*

The [bijou deep-dive](../posts/2026-08-05-bijou-deep-dive.md)'s fix
queue, in leverage order (details + file:line in the post):

1. ~~Hardening pass~~ **DONE 2026-08-05 ~20:55Z**
   ([post](../posts/2026-08-05-hardening-pass.md)): aux-prompt-hash →
   probe/eval selection (`bijou.eval --aux-prompt-hash` new flag);
   `resolve_plan` bounds assert; `score_frame` n_valid assert;
   report JSON records full scoring semantics
   (exclude/aux_prompt_hash/sample_steps/method/draws/generate/
   condition_override/batch/world); npz gains episode_index/
   frame_index identity columns. Oracle: banked AR-100k panel
   recomputed bit-exact (12/12 cells, d=0) through the edited
   scoring path; 3 new unit tests; check.py green. NOT included:
   deep-dive finding 6b — now item 8 below.
2. Flow-noise stable-triple seeding — **implemented behind
   `--noise-key` 2026-08-05 ~21:20Z, break pre-registered**
   ([amendment](../posts/2026-08-05-noise-reseed-prereg.md)): `stable`
   keys noise to blake2b(repo_id, episode, frame) via numpy
   SeedSequence (128-bit, no torch 32-bit trap, no draw stride);
   default stays `index` (byte-identical, oracle 12/12 d=0) until the
   flip executes at the first anchor boundary after the box reads —
   one flow-80k panel re-bank, decision band pre-registered off the
   draws chain's empirical σ_draw. Until then flow anchors remain
   valid only at frozen corpus composition. **Band FINALIZED
   2026-08-06 ~05:5xZ
   ([amendment](../posts/2026-08-06-sigma-draw-finalization.md)):
   σ_draw = 0.0159 < 0.045 → floor binds, re-bank band
   [6.4882, 6.7582]; the flip eval is eligible now (box reads
   posted) and queues behind the probe work on the local GPU.**
   **FLIP EVAL LAUNCHED 2026-08-06 ~07:41Z** (tmux `stablekeyrebank`,
   `~/eval_flow80k_stablekey_rebank.sh`) after the fairness probe's
   direct σ_draw = 0.02367 kept the floors (`reopen_floors: false`
   asserted in-launcher); band [6.4882, 6.7582] + bitwise
   state-copy/AR controls read at the ~09:2xZ boundary.
   **DONE — ADOPTED 2026-08-06 08:3xZ
   ([results](../posts/2026-08-06-stablekey-rebank-results.md)): controls
   bitwise ✓, stable-key chunk 6.5997 INSIDE the band (Δ −0.0242 ≈
   1σ_draw), first 1.9355. `stable` is now the quoted keying for all
   new flow numbers; ledger anchor re-banked. The #18.2 chain is
   closed.** **DEFAULT FLIPPED 2026-08-06 ~15:5xZ:** the code default
   (`bijou.eval` CLI + `BijouPolicy`/`SmolVLAEvalPolicy` ctors) is now
   `stable` — the hold expired when the SnapFlow chain's index-keyed
   stage-4 endpoint evals + npz addendum completed (15:10Z). `index`
   retained permanently behind an explicit flag for historical
   reproduction; new default-pin regression test; check.py 295 green.
   Arm A/B launchers written at the box boundary inherit `stable`, as
   the arch-batch pre-reg requires.
3. ~~Q3 tripwire noise fix~~ **DONE 2026-08-06 ~02:4xZ** (deep-dive
   finding 3, closed before the SnapFlow distill launch — the next
   conditioned flow run): `FlowDecoder.predict_chunk` now returns the
   noise it integrated (`BijouPrediction.noise`; the fallback draw
   moved from `sample_actions` into `predict_chunk` — same randn
   call, proven bit-exact incl. generator consumption against a
   pre-edit banked reference), `validate()` captures it per rich row,
   and the Q3 override decode reuses each row's scalar-pass noise —
   |Δ| is now purely the conditioning effect (was floored at sampling
   variance for a conditioning-blind flow model, the exact state the
   alarm exists to catch). AR path byte-unchanged (noise None,
   greedy). Eval/panel paths untouched structurally: eval always
   supplies per-item noise explicitly. 3 new tests
   (`tests/test_condition_tripwire.py`); `check.py` 215 green.
   Semantics note: `condition_sensitivity` for flow runs is not
   comparable to mainline's historical values (which carried the
   variance floor).
4. ~~Resume hardening~~ **DONE 2026-08-06 ~01:1xZ** (deep-dive
   finding 2, all three traps): (a) fresh-seed-on-resume is now
   ENFORCED — `--resume` with the checkpoint's recorded
   `train_args.seed` dies loud at startup (before data/model build;
   the epoch-0 restart replays the same batches + τ/ε draws), with
   `--allow-same-seed-resume` as the explicit reproduction-only
   escape hatch and a warn-not-die path for pre-recording
   checkpoints; (b) live-backbone resume prints a WARNING that fp32
   masters restart snapped to the bf16 grid (the "lossless
   continuation" comment corrected — lossless only in the
   frozen-backbone regime); (c) the resume hyperparameter note now
   covers EVERY optimizer param group via CLI-intent capture at
   group construction (was group 0 only — a changed
   `--backbone-*-lr` on resume was silently ignored), reading
   `initial_lr` so schedule-decayed lr can't fake a mismatch. 11 new
   tests (`tests/test_resume_guards.py`), live oracle on the real
   flow-80k checkpoint: same-seed refused / fresh-seed proceeds;
   `snapflow_recipe_verify` extended (new field at inert default,
   stage 0 re-run green, 51 verbatim). **Unblocks idea #3, and lands
   before the E4B 100k launch opens its crash+resume risk window.**
5. ~~Rig-rollout safety gate~~ **DONE 2026-08-06 ~09:5xZ** (deep-dive
   findings 8+9, the first-physical-run blocker, closed while #16
   execution is parked so the gate exists before it is ever needed):
   new lerobot-free `bijou/rollout_safety.py` + wiring in
   `bijou.rollout`. (a) **Clamp mandatory** — `--max-relative-target`
   (positive, finite) required before the arm moves, `--unclamped` is
   the explicit opt-out, clamp+unclamped together die as
   contradictory; gate runs before the (slow) policy load and in
   `--check` mode too. (b) **First-obs envelope** — after connect, the
   first observation must lie inside per-joint bounds from the rig
   stats (q01..q99 widened by half-band, 15° absolute floor;
   mean±3σ fallback for quantile-less checkpoint tables; stats
   dim ≠ 6 joints dies as wrong-embodiment). Catches wrong
   `--stats-repo-id`, ticks-vs-degrees (~10³ ticks flags every
   joint), uncalibrated arms; `--skip-envelope-check` for deliberate
   unusual starts; per-joint table printed every run, envelope shown
   in `--check`. (c) **Camera kinds mirror training** — with
   `--stats-dataset`, kinds resolve through training's own path
   (`annotation_stamp` + `camera_kinds_of`: stamped+hash-matched
   file, else "unknown" — never the name heuristic, which stays only
   for the no-dataset case); `--camera-kind NAME=KIND` explicit
   override, validated against the vocabulary (deep-dive's wild case
   "front-named cam judged top" covered by test). 22 new CPU tests
   (`tests/test_rollout_safety.py`); `--check` exercised end-to-end
   on the real flow-80k checkpoint (CPU); `check.py` 274 green.
6. ~~Parity extension~~ **DONE 2026-08-06 ~03:4xZ** (deep-dive
   finding 7): `verify_parity` gains (a) a default-on **padded
   2-sample × 2-image batch check** — mixed-length prompts through
   the processor's `padding=True` path (natively LEFT-padded on
   transformers 5.14 — measured, not assumed), HF attention mask +
   per-sample logical position_ids (the `encode_tensors`
   convention, passed to HF too since its forward defaults to
   arange), gated per sample at its last REAL position against HF
   on the same padded batch AND against HF's unpadded per-sample
   forwards; **both padding orientations** run (the native side +
   the per-row roll to the other — the ar_backbone prompt path
   collates left, the token-identity of each row vs its solo
   tokenization is asserted). (b) **`--require-bitwise`** —
   escalates every same-shape HF comparison from tolerance to
   bitwise (the measured eager/H100 contract, previously printed
   but never enforced) and refuses near-tie token forks;
   cross-shape (padded-row vs unpadded) comparisons stay
   tolerance-only, labeled as such. Validated on the real E2B (CPU
   eager): full harness PASS — **ours-vs-HF-padded BITWISE on all
   real positions in both orientations**; solo cross-checks ≤0.44
   (GEMM-shape fp noise, tol 2.0). **Falsification oracle taught a
   scope lesson worth recording:** an arange-doctored run passes
   WITHIN TOL in any orientation, because in a single forward
   positions enter only through RoPE, which is *relative* — arange
   vs logical is a per-sample constant shift, visible only as
   fp-rotation noise (~0.6). So this check pins mask + padding +
   multi-image semantics vs HF; the position CHAIN (where the
   convention genuinely bites — cached continuation) stays pinned
   by `tests/test_backbone_continuation.py`. Oracle (corrected to
   a genuine corruption): real=PASS, zero-positions=FAIL,
   mask-dropped=FAIL. Remaining honest gap: state-token splice and
   15-layer truncation/`kv_stop_layer` have no HF counterpart
   (bijou self-consistency tests cover them).
7. ~~Duplicate-content census over curated_v0~~ **DONE 2026-08-06
   ~02:0xZ** ([results
   post](../posts/2026-08-06-dup-census-results.md)): the corpus is
   heavily forked — 6,935 of 52,507 episodes (2.67M frames) in 3,348
   cross-repo BYTE-EXACT clusters (action+state streams identical;
   quantized tier adds nothing). **The split is breached: 524 holdout
   episodes across 79 repos have byte-exact twins in train — 2,096 of
   17,204 core panel rows (12.2%) score on leaked episodes**, all via
   the cross-repo fork channel the repo-id dedup can't see. Anchor
   impact (validated partition, anchors reproduce exactly): leaked
   frames score ~1.3–1.6 better than clean on BOTH banked models —
   **clean-core anchors: AR-100k 5.9761/2.1695, flow-80k
   6.8137/1.9714** (published 5.8026/2.1431, 6.6232/1.9331 are
   ~0.17–0.19 optimistic in level; content-difficulty confound stated
   honestly). Paired within-corpus deltas (box batch, E4B, draws
   chain) UNAFFECTED — every model shares the same train corpus and
   the same leaked frames. Instruments:
   `fontaine/scripts/dup_content_census.py` (+`--oracle` 7-case
   suite, split mirror proven on all 878 plan repos, collision guard),
   `dup_census_anchor_impact.py` (join content-checked vs raw
   parquet). Exclusion list frozen in `~/dup_census_report.json`.
   **Panel-v2 amendment PROPOSED 2026-08-06 ~02:3xZ, awaiting owner
   steer** ([amendment](../posts/2026-08-06-panel-v2-amendment.md),
   instrument `fontaine/scripts/panel_v2.py`): v2 = v1 minus the 524
   leaked episodes minus the 3 wrap-census corrupt repos, strict
   row-subset (core 17,204→15,056, labeled 8,596→7,522) so every
   banked npz re-pools exactly with zero re-evals. v2 anchors
   derived + oracle-gated: **AR-100k 5.8894/2.1396, flow-80k
   6.7151/1.9453, state-copy 11.7639/2.5851** (frozen plan
   `plans/holdout_curated_v0_k4l2_panel_v2.json`, embeds exclusions).
   Transition proposal: in-flight pre-registered reads finish on v1;
   v2 for every new pre-reg on approval; bundle the #18.2 noise-key
   flip (+ optionally #14 shortest-arc) at the same re-bank boundary
   so the flow anchor re-banks once. Until steer, results posts quote
   full-panel (anchor convention) with the v2 column alongside.
8. ~~Leakage checker same-repo-id count/content assert~~ **DONE
   2026-08-05 ~21:20Z** (deep-dive finding 6b): the identity branch
   now VERIFIES the claim — episode-count assert plus per-episode
   length fingerprint (`meta/episodes.jsonl` v2 or `meta/episodes/`
   parquet v3; asymmetric metadata is fatal; same-directory shortcut
   for the literal identity case). Mismatch ⇒ SystemExit demanding
   `meta/source_provenance.json`, symmetric with the provenance
   branch's count assert. 4 new tests (179 green); full-corpus
   identity certification re-run PASSED with the new code
   (radioactive 5267 / checked 47240, 4.1 s); a mutated-count copy of
   `therarelab/so100_pick_place_2` fails loud in production. Unblocks
   derived-corpus training (ideas #9, #13 repair arm).
