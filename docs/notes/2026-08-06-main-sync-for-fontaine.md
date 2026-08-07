# main ← fontaine sync note (2026-08-06)

For fontaine, from the owner's interactive session: **`main` was
rebased onto your branch snapshot at `42a202a`** (owner call — your
work through the mem-snapshot/vram-peaks commits is now mainline), and
three commits sit on top. Merge `main` into `research/fontaine` at
your convenience (your remote has moved past `42a202a`, so expect a
normal merge, not a fast-forward). What you're picking up:

## 1. `2ee2be5` — batched noise-draw ensembling

Your `sample_draws` path in `BijouPolicy.predict_with_text` integrated
draws **sequentially**; at rollout's B=1 that is N×num_steps
GPU-starved forwards. Now: one solver call at `draws×B` via
`tile_memory`/`tile_stats` (draws-major, so your `collapse_draws` and
`--dump-draws` layouts are byte-compatible; only the fields
`FlowDecoder.predict_chunk` reads are tiled — a KV cache refuses to
tile loudly).

Measured (RTX 3000 Ada 8 GiB, rig-ft flow 4k, mean-of-10, best of 3):

| | sequential | batched |
|---|---|---|
| bf16 | 3,224 ms | **576 ms** (5.6×) |
| fp32 | 3,441 ms | 1,804 ms |

Equivalence: fp32 seq-vs-batched **max Δ 9.2e-5°** (semantically
exact); the bf16 variants each sit ~1.3–1.5° from fp32 truth — 10-step
ODE integration noise, dtype-inherent, NOT batching (probe:
`outputs/probe_draws_equiv.py`, gitignored — rerun it after merging if
you touch this path). Eval-side implication for your charter item 1:
your unconstrained-class panel sweeps get the same speedup at B=24
(240-row expert batches fit H100 trivially).

`bijou.rollout` now exposes `--sample-draws` (decode tag `-meanN`;
your AR guard still rejects >1 on non-flow checkpoints). At 576 ms =
18 ticks @30 Hz, mean-of-10 + `--async-inference` are **sustainable
together** on the owner's laptop.

## 2. `36570c0` — `--return-home`

Ctrl-c (or duration end) glides the arm back to its start-of-rollout
pose (your envelope gate's `first_state`) over `--return-home-seconds`
(default 1.5 s): cosine-eased interpolation (`home_trajectory` in
YOUR `rollout_safety` module — pure, tested), driven at control fps,
`max_relative_target` clamping every step. A second ctrl-c cancels
(arm holds); glide errors never mask the disconnect.

## 3. Known issue on the snapshot, not fixed here

`tests/test_chunked_backward.py::
test_chunked_gradient_matches_unchunked_ar_backbone_aux` fails by a
hair on your branch as pulled: rel error **1.0004e-4** vs the 1e-4
bound. Pre-existing (fails with our changes stashed); a tolerance
call for you, not us — every other check is green (337 passed).

## 4. Incidental

`bijou/train.py`: import-group reorder only (`wandb` into the
third-party block — the shared ruff config's opinion), no behavior.
