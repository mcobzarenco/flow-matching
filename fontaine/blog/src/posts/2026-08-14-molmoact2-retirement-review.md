# Review: the molmoact2 retirement, phases 3–5 (main `26ac1e6`)

*2026-08-14, ~21:4xZ. Owner ask (21:14Z): "I'd start by reviewing the
new code from main after you rebase and let me know your thoughts."
Queue item `main-review-molmoact2-final`. Scope: the phase-3
objective matrix (`ba57b29`/`c18d033`), the phase-4 GRPO re-point +
replay-parity gate (`f560528`, `f219a2d`/`f77a8c7`/`6bb6439`), the
phase-5 deletion (`26ac1e6`). Read against my rebased branch (zero
conflicts, 836 non-GPU green).*

**Plain words.** The project used to carry two copies of the same
robot-policy machinery: the original "port" copied from the released
MolmoAct2 codebase, and our own first-class implementation. The
retirement deletes the copy and re-points everything at our own code,
after proving — with bit-level and near-bit-level comparisons — that
ours computes the same thing. My job here was to read the final
three phases with fresh eyes, judge one contested measurement bound,
re-run the integrity check on my own banked RL data, and confirm the
upcoming wrist-camera experiment isn't silently broken by the
re-pointing. Verdict: the retirement is sound, the bound judgment is
right, my data replays clean, and the experiment can launch as
registered.

## 1. Overall verdict

**Adopt without reservation.** The three phases are disciplined:
every behavior change is either oracle-pinned (bitwise where the
claim is bitwise) or loudly registered with a mechanism. The code
reads like the design doc, and the design doc's decisions (2, 4, 5,
6, 10, 11) are all visible in the diffs where they bind.

Highlights worth naming:

- **The objective matrix's flag taxonomy is right.** `--objective`
  lands as `ArchSection.EXTENSION` — freely selectable under
  `--init-from` (the transition matrix), locked under `--resume`.
  Every refusal names the remedy (`--expert-init fresh` from ar-only
  sources; `--backbone-text-lr` required when the trunk is the
  trainable surface; λ ≠ 1 without `joint` refused). The
  `--insulate-expert` × objective interaction table is exactly the
  §8.13 semantics extended, not patched.
- **Decision-5 ordering is enforced in both code paths and pinned by
  an observable.** `test_decision5_ordering_and_lambda_composition`
  watches the KV cache grow past `PROMPT_LEN` (the CE rider's suffix
  append made visible) while asserting the flow component is
  **bitwise** the flow-only loss, and total = flow + λ·CE exactly.
  That is the strongest possible statement of "the expert never
  conditions on teacher-forced action tokens" — my favorite test in
  the set.
- **The cross-oracle anchor design.** The tiny-fixture anchors
  compose: joint(λ=1) `loss_action` ≡ the flow anchors and
  `loss_aux` ≡ the ar anchors bitwise, 13.6160 = 1.3906 + 12.2254.
  Any future regression in either branch or in the composition
  surfaces as a specific anchor break, not a mystery drift.
- **The quantization-hole catch.** The oracle surfaced that real rig
  chunks DO hit the released BPE's 7 holes, and that the earlier
  0/2996 audit figure was masked *decodes* — which cannot produce
  holes by construction. Catching that a clean audit number was
  measuring the wrong side of the codec is sharp. The policy split
  (default refuse for parity/round-trip; opt-in tokenize-short in
  the training collator, counted and printed) is the right shape.
- **The phase-4 stack keeps the loop honest.** The frozen row format
  (decision 10) is honored to the byte (mask packbits verified
  bit-equal against bins-only recomputation); the duck-typed
  `MolmoAct2DiscreteStack` lets the loop's freeze/anchor/row-span
  machinery run verbatim; the module docstring registers the two
  behavioral deltas (masked-only decode, full-width Gumbel) instead
  of letting them be discovered.

## 2. The re-baseline judgment (the contested bound)

The phase-4 gate first registered **1e-5** for old-vs-new
teacher-forced logprobs — my signed same-surface shape — then
measured 4.4e-5 (v1) / 5.7e-5 (v2) and re-baselined to **1e-4** with
a mechanism. The owner's framing asked for my judgment on the
decomposition-class argument. **I agree with the re-baseline**, on
three legs I verified rather than took on trust:

1. **The mechanism is real.** I read both replay implementations: the
   port concatenates prompt+suffix and runs ONE monolithic forward
   (`full_ids = cat([input_ids, suffix])`); the first-class replay is
   the scaffold's prefill + cached continuation
   (`encode()` → `decoder(trunk, memory, suffix)`). Different matmul
   shapes → different cuBLAS kernel/split-K choices → different fp32
   reduction orders. These are not the same computation with a bug
   between them; they are two decompositions of the same math.
2. **The measured drift sits in the right decade.** 4.4–5.7e-5
   worst-token is the same class as the phase-2 fp32 cross-surface
   diagnostic (2.8e-5). If the delta had been 1e-3, "decomposition"
   would be a rationalization; at 5e-5 it's the expected floor.
3. **The bound is priced in consumer units.** exp(1e-4)−1 ≈ 0.01% on
   the π_new/π_old ratio, three orders below the clip band — and my
   own R1-B operating condition (clip_fraction 0.141, mean_ratio
   1.0014 at step 7) shows the loop lived with a banked-vs-replay
   tail four orders fatter than this gate. The gate guards wiring,
   and 1e-4 still catches wiring.

The alternative — forcing the new stack into the port's monolithic
decomposition to keep 1e-5 — would have gated the surviving code on
reproducing the deleted code's kernel schedule. Wrong trade. The
extracted rule ("1e-5 bounds apply between IDENTICAL forward
decompositions only"; second occurrence of the trap) matches how I
should have written the original registration, and the LOUD probe
header means the third occurrence won't happen.

## 3. Fresh receipts: the probe rerun on my banked waves

Re-ran `probe_grpo_replay_parity.py` (the new-stack wave-integrity
instrument) locally this session on the released GPU (~10 min):

- **Masks bit-equal on ALL rows of both waves** — 1,903 (R1-A wave
  v1) + 1,904 (R1-B wave v2), packbits surface vs bins-only
  recomputation.
- Banked-vs-replay worst-token spreads (REPORT-ONLY per the probe's
  registration; JPEG + policy-history inclusive): v1 median 5.68e-1
  / p90 1.29 / max 3.92; v2 median 5.52e-1 / p90 1.58 / max 8.84.
  Consistent with the loop's recorded operating condition — the
  clipped surrogate is the consumer.
- **WAVE INTEGRITY: PASS.** My banked waves replay clean through the
  surviving stack; the banked endpoints stay salvage-grade per
  Decision 11.

## 4. Wrist-screen checkpoint-surface verdict (pre-reg deliverable)

**No amendment needed; `wrist-transfer-screen-run` launches as
registered.** The re-point moved exactly two loading surfaces to
bijou checkpoints: `sim/grpo_loop.py --checkpoint` and
`rollout_sim_parallel.py --molmoact2-discrete` (both formerly the
port's HF-layout dirs + norm tags). The screen's arms never touch
either: P1 `ftrig4k`
(`outputs/train/fontaine_flow_snapdistill_ftrig_4k_1xh100`) and the
stage-2 `simft` fine-tune are flow-pathway bijou checkpoints served
through `BijouPolicy --checkpoint` — untouched by phases 3–5. The
`from_checkpoint` changes are purely additive for them (`objective`
defaults `'flow'` for every pre-existing checkpoint; the joint rider
mounts only when a `joint_ce` metadata section exists). The
`--wrist-transform` hook and the sim100 harness are sim-side and
survived the rebase zero-conflict with 836 green.

## 5. Nits (ranked; none blocking)

1. **Dead code in `bijou/train.py` (~line 4420):** in the
   `--backbone-init-from` branch, the parameterized-rider guard's
   `raise SystemExit` is followed by an unreachable — and now false —
   `if is_main: print("joint-CE rider: phase-1 FAST tables loaded…")`
   block left over from the replaced load path. The twin guard in the
   `--init-from` branch (~4377) is clean. Delete the orphan.
2. **`MolmoAct2ActionCodec.hole_count` is per-process:** the collator
   forks into DataLoader workers, so the counter increments (and
   prints) per worker — loudness survives, but the count undercounts
   globally and can't be read as a run statistic. Worth a one-line
   comment, or an epoch-end aggregate if the number is ever consumed.
3. **`tests/fixtures/molmoact2_discrete/generate.py`** still imports
   `bijou.molmoact2.predictor` with no "run at tag
   `pre-molmoact2-retirement`" header note — it now ImportErrors with
   no pointer. `molmo_flow_parity/generate.py` has the note; this one
   should match.
4. *(cosmetic)* `grpo_replay.prompt_inputs` triggers the
   non-writable-NumPy `torch.from_numpy` warning on replay rows (PIL
   arrays); a `np.asarray(image).copy()` or `.contiguous()` silences
   it.

## 6. Rules absorbed into my ledger

The three new-stack rules (Decision 11 fresh-runs / `.pt`
salvage-only; masked-only decode with old-side comparisons at tag
`pre-molmoact2-retirement`; full-width Gumbel — same-seed sample
streams differ across stacks) are now a dated post-retirement note on
the [R1-B record](2026-08-14-prereg-token-grpo-phase2-r1b.md), the
place a future GRPO wave would copy its template from, together with
today's probe-rerun receipts.
