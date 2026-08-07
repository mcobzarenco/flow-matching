# Pre-reg: leaderboard decode-cost micro-benchmark

*2026-08-07 ~11:0xZ. Record-only timing pass. Queue item
`leaderboard-decode-cost-microbench` (owner steering 2026-08-07
10:04Z: "would just time work, e.g. ms/sample?"). Instrument:
`fontaine/scripts/leaderboard_decode_microbench.py` (this pre-reg
lands with it, before any data exists).*

## Why

The [leaderboard](../leaderboard.md)'s `eval ms/frame` column mixes
two provenances: ⏱ rows timed cleanly, and ≈ rows mtime-bounded from
sequential launcher logs where batch size, workers and dump flags
drift across runs. One shared harness re-times **every decode config
on the board** under identical settings, and adds the number the rig
actually cares about: **batch=1 single-stream latency** (the #16
few-shot rig-transfer hook).

## What runs

One script, local 1×H100, the moment `draws10_t1` frees the GPU
(boundary ~12:3x–12:5xZ today) and **before any next local launch**.
Seven configs — decode flags byte-matched to the banked leaderboard
stems; checkpoints are the banked ones:

| config | checkpoint | decode flags |
|---|---|---|
| `ar_greedy` | AR-100k @100k | deployment fast path |
| `ar_draws10_t1` | AR-100k @100k | `--ar-temperature 1.0 --sample-draws 10` |
| `teacher_heun30_draws1` | flow teacher @80k | `--sample-method heun --sample-steps 30` |
| `teacher_heun30_draws10` | flow teacher @80k | … `--sample-draws 10` |
| `student_1nfe_draws1` | SnapFlow student @30k | `--sample-method euler --sample-steps 1 --target-time zero` |
| `student_1nfe_draws5` | SnapFlow student @30k | … `--sample-draws 5` |
| `student_1nfe_draws10` | SnapFlow student @30k | … `--sample-draws 10` |

Two modes, all configs identical settings within a mode:

- **batched** — batch 32 / workers 20 (the banked panel-eval
  config), `--num-samples 320 --seed 0`. Replaces the ≈ throughput
  entries.
- **single** — batch 1 / workers 4, `--num-samples 50 --seed 0`. One
  frame in flight at a time: per-frame wall-clock **is** single-stream
  latency (AR decodes stay token-serial inside it). New latency
  numbers, quoted alongside throughput, never mixed.

Frame choice is data-side and seeded, so **every config times the
same frames** within a mode (content mix matters — measured panel
rates ranged 16–40 f/min with content).

## Measurement rule (frozen)

The wrapper timestamps the eval's `scored N/M frames` progress lines
with its own monotonic clock (the `draws_rate_gate.py` pattern);
quoted rate = last line minus first line. This excludes model load,
dataset scan, and the first progress interval (CUDA warmup).
`bijou.eval` prints a line every 5 batches, so the registered frame
counts guarantee ≥ 2 lines per run (batched window = 160 frames;
single window ≥ 45 frames). Consistency anchor: `ar_greedy` batched
should land near the 88.7 ms/frame ⏱ row; a large disagreement is
reported as an instrument finding, not silently adopted.

## Guards & cost

GPU-quiet precondition; missing checkpoint, eval rc ≠ 0, < 2 progress
lines, non-increasing counts, or a 30-min per-run watchdog all abort
loudly. Oracle landed with the instrument (`--selftest`: exact rate
arithmetic, guard firing, parser fixtures). Budget: ~30 min GPU
projected (AR draws10 batched ~10 min dominates), **hard ceiling 1.5
GPU-h** via the watchdog; well under the charter's pre-reg threshold,
run in the boundary gap before the next pre-registered launch.

## Record-only clause

Timing only. No model-quality claims; the MAE column and leaderboard
order cannot change from this pass. The ≈ entries are replaced with
measured ms/frame (provenance flips to ⏱-bench); a latency column
note lands with the same numbers. If a number disagrees wildly with
its ≈ predecessor, both are shown until the discrepancy is explained.
