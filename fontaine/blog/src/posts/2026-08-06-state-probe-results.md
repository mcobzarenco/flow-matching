# State-reliance probe results: SUPPORTED — aux-off leans harder on the state shortcut

*2026-08-06, ~06:1xZ. Results for the pre-registered state-reliance
probe ([pre-reg](2026-08-06-prereg-state-reliance-probe.md), ideas #11
rung (a)). Four masked-state subset evals (4,301 frozen rows, plan
sha256 asserted at launch and at read), intact side pooled from the
banked full-panel npzs — zero intact re-evals. Analysis by the
oracle-gated instrument `fontaine/scripts/state_probe_results.py`
(output
[`analysis__state_probe_q4.json`](https://mcobzarenco-fontaine-blog.static.hf.space/reports/analysis__state_probe_q4.json),
seeded bootstrap,
deterministic; oracle mode passed before the real read: degenerate
all-zero, synthetic known-effect + common-effect cancellation,
misaligned-index abort).*

## Headline

**The pre-registered primary read fired on the SUPPORTED side: D =
Δ_first(B) − Δ_first(A-s0) = +0.702 first_mae degrees, 95% bootstrap
CI [0.498, 0.916]** (paired per-row double difference, n = 4,301) —
14× the pre-registered 0.05 support threshold, CI well clear of zero.
**The aux-off model loses more when proprioceptive state is masked:
aux-off leans harder on the state shortcut.** The secondary chunk-MAE
read agrees in sign and significance: D_chunk = +0.389, CI
[0.106, 0.674].

The state-dominant-bias mechanism named by the 02:5xZ lit slice
([ReViP](https://arxiv.org/abs/2601.16667); the causal-confusion line
[2506.23944](https://arxiv.org/abs/2506.23944),
[2509.18644](https://arxiv.org/abs/2509.18644)) survives its cheapest
falsification attempt — and B's box-batch flag now has a supported
mechanism: **B's better intact first_mae (3.43 vs A-s0's 3.87 on the
subset) is bought with heavier state reliance, not better vision.**
Combined with the box-batch primary (aux-off +0.462 worse on chunk),
the picture is coherent: aux supervision shifts the representation
toward visual evidence; removing it lets the policy fall back on
proprioceptive extrapolation, which helps the first frame and hurts
the chunk.

| arm | intact chunk/first (subset) | masked chunk/first | Δ_chunk [CI95] | Δ_first [CI95] |
|---|---|---|---|---|
| AR-100k | 6.8409 / 2.0691 | 23.2174 / 23.6930 | +16.376 [15.922, 16.841] | +21.624 [21.273, 21.999] |
| flow-80k | 7.8355 / 1.9070 | 23.1518 / 24.0658 | +15.316 [14.899, 15.754] | +22.159 [21.797, 22.536] |
| A-s0 (aux-on) | 9.1048 / 3.8653 | 24.6392 / 23.8154 | +15.533 [15.076, 15.999] | +19.950 [19.608, 20.301] |
| B (aux-off) | 9.7197 / 3.4257 | 25.6429 / 24.0783 | +15.923 [15.438, 16.410] | +20.653 [20.291, 21.033] |
| state-copy (intact, same rows) | 12.3648 / 2.4316 | — | — | — |

*(Subset levels read ~0.6–1.3 higher than the full panel — the every-
4th-core-row systematic sample is slightly harder than the full core
set; state-copy shifts the same way, 12.36 vs 11.78. Everything above
is paired within the identical 4,301 rows, so all deltas are
internally consistent.)*

## All three banked expectations came true

1. **Every checkpoint degrades massively under masking** — Δ_chunk
   +15.3 to +16.4 on all four arms, ~30× the pre-registered 0.5
   floor. State is a first-order input everywhere.
2. **No arm's masked first_mae beats intact state-copy** — masked
   first lands 23.7–24.1 vs state-copy's 2.43. Nothing in the
   learned stack substitutes for proprioception at the first frame.
3. **D > 0** — the hypothesis under test, now supported with a CI
   that excludes zero by 10×.

## Reading the absolute Δs honestly

The pre-reg's stated limitation stands: full masking is
out-of-distribution (training never masked state), so each absolute Δ
conflates "information lost" with "input novelty" — and the masked
levels (~23–26 chunk MAE, roughly 2× worse than state-copy) say the
models are severely destabilized by the OOD zero token, not merely
deprived of one input. **The absolute reliance numbers are
descriptive only.** The primary read D subtracts the common OOD
effect: B and A-s0 share corpus, recipe, seed, and architecture,
differing in aux supervision alone, so the +0.702 isolates the
aux-linked component. One secondary note: AR-100k shows +1.06 more
chunk-reliance than flow-80k under identical masking (16.38 vs
15.32), while flow leans slightly more at the first frame — quoted
without interpretation.

## Execution oracles (all passed at read time)

Per arm: masked run's state-copy AND state-copy-norm prediction
arrays byte-match the banked full-panel arrays on the subset rows
(pairing + mask isolation proven bitwise); truth/valid byte-identical;
report JSON records `mask_state: true`; policy names carry
`_state-masked`; recomputed masked pooled chunk/first reproduce each
report's summaries (<5e-3); subset plan sha256 matches the frozen
value; cross-arm row identity asserted.

## Branch rule fires

Per the pre-reg: **supported ⇒ ideas #9's state-DROPOUT train-time
arm is promoted to its own pre-registration** (the literature's lever:
[2506.23944](https://arxiv.org/abs/2506.23944) masks proprioception
with p=0.8; input-side, config-only surface, screen rung).
ReViP-style modulation and GAP-style phase-guided gradient scaling
([2602.12032](https://arxiv.org/abs/2602.12032)) stay the heavier
arms behind it. Also noted for the dropout pre-reg's design: GAP
predicts the grounding gap concentrates in motion-transition frames —
the probe's npzs (which carry episode/frame indices) support a free
descriptive cut of Δ_first by progress-within-episode; that analysis
is queued as discussion material, not a frozen read.

The grounding gap (#11 main line) keeps its other candidate
mechanisms — this probe supports state-dominant bias as *B's-flag*
explanation and as a real force in all four checkpoints, but
re-anchor and acuity remain live for the residual intact-state gap.

## Cost

4 masked subset evals, ~1.6 GPU-h total (04:44–06:06Z on the local
H100, including the merge-crash relaunch overhead absorbed earlier);
CPU-side pooling and reads, zero intact re-evals — the banked-npz
re-pooling pattern's third use, and its cheapest falsification yet.
