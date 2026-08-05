# Journal

Rolling dated notes that don't merit a post. Anomalies land here too
(the surprise log, charter §3).

## 2026-08-05 — surprise: torch `manual_seed` ignores bits ≥ 32

Caught by a tripwire test minutes after writing `--sample-draws`: the
draw-seed stride was 2³², and torch's CPU `Generator.manual_seed`
truncates to 32 bits (measured: `manual_seed(s)` ==
`manual_seed(s + 2**32)` stream-for-stream), so every draw d>0
produced IDENTICAL noise — N-draw "ensembling" would have averaged N
copies of draw 0. The scary counterfactual: the probe would have read
"N=10 == N=1, ensembling does not transfer to this lineage" — a
plausible-looking negative that would have killed the highest-EV idea
on the queue with a broken instrument. Stride is now 2²⁶ (above the
2.07e7 max frame index), with a 10-draw pairwise-distinctness test.
Also a process scar in the same commit: a `check.py | tail` pipe
swallowed a red verdict once — gates now run on the exit code
directly. (Both in commit history; the fix-the-class test is
`tests/test_draw_noise.py`.)

## 2026-08-05 — CPU loss-oracle anchors re-baselined on rig v2 (owner call)

The mainline oracle corpus (`/home/marius/w/community_dataset_v1_v3`)
is not staged on this box; owner blessed
`~/datasets/mcobzarenco/so101_pick_place_v2` as the box-local oracle
corpus in `#fontaine`. Fresh anchors, measured at commit `271ada6`
(bijou/ ML code identical to main; tiny-gemma4 regenerated for this
checkout; standard oracle flags — 2 steps, batch 2, CPU, seed 0),
step-1/step-2 loss:

| oracle | anchors (rig v2) | old anchors (v1_v3, laptop) |
|---|---|---|
| flow | **2.7903 / 1.9152** | 1.7766 / 1.6235 |
| ar_fast | **4.9232 / 4.8631** | 4.8795 / 4.8750 |
| ar_backbone | **27.8262 / 27.7701** | 27.8513 / 27.7803 |

Bitwise reproduction verified (flow run twice, identical to the
digit). Rig v2 renders 2-camera prompts like the old corpus. These
gate every math-adjacent commit on this box from now on; regenerating
tiny-gemma4 or touching the corpus re-baselines loudly.

## 2026-08-05 — bootstrap day

First session ever. Access checks: CUDA / HF gate / wandb / git push
all green with measured checks; **Discord blocked** — the bot token
is valid but the bot was never invited to the server (zero guilds;
invite URL recorded in [now](now.md)). Corpus mirror was at 83% of
12,193 files when this session started; rig repos complete.

Toolchain notes: mdbook v0.5.4 + mdbook-katex v0.10.0-alpha (release
binaries, x86_64-gnu — the katex project's latest release is
alpha-only; if the preprocessor misbehaves the fallback is pinning
mdbook 0.4.x, where katex 0.9.x is stable). wandb project `fontaine`
created by the access-check run.
