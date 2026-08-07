# Main sync: batched draws + return-home — deep review, merge, measured speedup

*2026-08-07, work session 12:30Z. Owner steering 12:26Z ("incorporate
the missing changes from main") + 12:29Z ("deeply review the newly
added code; feel free to modify it").*

`main` was rebased onto our branch snapshot `42a202a` (our work
through mem-snapshot/vram-peaks is now mainline) with three commits on
top; we merged it back into `fontaine` after a line-by-line review.
The sync note (`docs/notes/2026-08-06-main-sync-for-fontaine.md`) is
the owner-side account; this is the review record.

## What came in

**`2ee2be5` — batched noise-draw ensembling.** Our `sample_draws`
eval path integrated draws sequentially: at rollout's B=1 that is
N×num_steps GPU-starved tiny forwards. Main batches all draws into
ONE solver call at `draws×B` via two new tilers (`tile_memory`,
`tile_stats` in `bijou/eval/policies.py`), draws-major so the
`collapse_draws` / `--dump-draws` layouts are byte-compatible.
Owner-side measurement on the rig laptop: **3,224 → 576 ms bf16
(5.6×)** for mean-of-10; fp32 seq-vs-batched max Δ 9.2e-5° (the same
math, reordered).

**`36570c0` — `--return-home`.** Ctrl-c (or duration end) glides the
arm back to its start-of-rollout pose — the envelope gate's
`first_state` — over ~1.5 s of cosine-eased interpolation
(`home_trajectory` in `rollout_safety`), `max_relative_target` still
clamping every step; a second ctrl-c cancels (arm holds), and glide
errors never mask the disconnect.

## Review verdict

The implementations are sound. What we verified, line by line:

- **Draws-major layout is consistent end to end**: the noise is built
  as a `cat` of per-draw stacks (row d·B+i = draw d, item i),
  `Tensor.repeat` tiles memory/stats the same way, and the final
  `reshape(draws, B, chunk, dim)` hands `collapse_draws` exactly the
  layout the sequential path produced.
- **`predict_chunk` reads only the tiled fields** (`state`,
  `state_stats`, `action_stats` — checked against the decoder
  source), and `FlowDecoder.forward` derives its per-sample RoPE
  position bases from the *tiled* padding mask, so unequal real
  lengths stay correct at draws×B.
- **`home_trajectory`** lands exactly at home (eased s(1)=1), is
  monotone, and its peak per-tick step is 1.57× the linear rate —
  matching its own comment; `first_state` is bound before the
  `try/finally` that runs the glide, so the finally can never see it
  unbound.
- **Semantic merge composition**: main's rewrite touches exactly the
  flow-draws block; our post-snapshot AR sampled-draws block (ideas
  #19) sits behind a disjoint guard (`ar_temperature` +
  `ARSuffixDecoder`) — no interaction.

Three gaps found, three fixes landed on top of the merge:

1. **`tile_memory` now refuses un-projected residual taps** the same
   way it refuses a live KV cache. `ObservationMemory.residuals`
   rides at [B, P, hidden]; tiling streams to draws·B while residuals
   stay at B is a silent inconsistency waiting for the first caller
   that tiles a raw-encoder memory. (In the current policy path
   `attach_residual_streams` has already consumed them — the guard
   costs nothing today and fails loud tomorrow.)
2. **The batched path now has an oracle** —
   `tests/test_batched_draws.py`: seq-vs-batched equivalence on a
   padded memory with *unequal* real lengths (the RoPE-base case) on
   a randomized tiny decoder, plus draws-major layout pins for both
   tilers and the two refusal guards. Main's commit shipped no tests;
   our convention is oracle-gated math changes.
3. **The `test_chunked_backward` tolerance call** (sync note §3,
   explicitly left to us): the aux-gradient oracle bound was 1e-5,
   calibrated on this box; the owner's RTX 3000 Ada measures
   1.0004e-4 with identical math (the module's own padding-width
   diagnostic puts same-math fp realizations at ~2e-4). Relaxed to
   5e-4 with both anchors in a comment — the failure mode the oracle
   guards (mean-of-chunk-means normalization) shows rel ≫ 1e-2, so
   the oracle stays sharp.

## Incidental find: the test suite could eat the repo (fixed)

Validating the merge in a scratch `git worktree` tripped a landmine
worth its own paragraph. Two harness tests (`test_refresh_ctrl`,
`test_session_driver`) drive *real* git against throwaway tmp repos.
Run them from a `git commit` pre-commit hook in a **linked worktree**
and git has exported an *absolute* `GIT_DIR` (plus `GIT_AUTHOR_*`) to
the hook — so the throwaway repo's `git init` silently
re-initializes the REAL repo's git dir, the fixture's config writes
land in the shared `.git/config` (`user = t@t`,
`core.worktree = <tmp path>` → "fatal: this operation must be run in
a work tree" once pytest cleans the tmp dir), and its `commit -qm c1`
lands on the real HEAD. In the main checkout this never fired
because there `GIT_DIR` is exported as *relative* `.git`, which
re-resolves against the tmp cwd — isolation by luck. Both tests now
scrub `GIT_*` from the subprocess environment (verified by running
the suite under a hostile absolute `GIT_DIR`); the damaged config
was repaired in place, and the orphaned `c1` commit is unreferenced
garbage.

## The speedup, measured on our leaderboard configs

The decode microbench (pre-reg
[2026-08-07](2026-08-07-prereg-leaderboard-decode-microbench.md))
ran its full 7-config pass on the *pre-merge sequential* code as the
baseline, then the three flow-draws configs re-ran post-merge under
the identical harness — same frames, same batch/workers, same clocks.

PLACEHOLDER_RESULTS_TABLE

The leaderboard's ⏱ column now carries the same-harness numbers for
every row, including the batch=1 single-stream latency (the
deployment-facing read, #16 hook).
