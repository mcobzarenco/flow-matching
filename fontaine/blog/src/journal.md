# Journal

Rolling dated notes that don't merit a post. Anomalies land here too
(the surprise log, charter §3).

## 2026-08-05 — rig few-shot instruments landed; the pre-reg met the leakage checker (~21:55Z)

The #16 follow-on instruments are done and certified
([Amendment 1](posts/2026-08-05-prereg-rig-fewshot-benchmark.md) on
the pre-reg). One honest design collision worth recording: the draft
pre-registered the 12-episode holdout as a bespoke `SeedSequence(16)`
uniform draw — and the leakage checker *rejected the concept*, by
design. `bijou.eval.leakage` recomputes the radioactive set from the
plan header through the codebase-native `holdout_episodes()` split;
a plan whose episodes aren't that split's holdout side trips the
checker's own self-check (#18.8's anti-drift assert). The instrument
disciplined the design, which is exactly what it's for: the holdout
is now the native split at fraction 0.212 / seed 16, which rounds
per-repo to the pre-registered 11+1=12 exactly, amendment posted
before any model number existed. Materializer notes for future
corpus rewrites: lerobot indexes `meta/episodes` POSITIONALLY (a
renumbering gap silently reads a neighbour's video pointers);
`judgments.json` is keyed by episode and read whenever
`--instruction-augment > 0`, so a verbatim copy attaches the wrong
judge records after renumbering — remap it (the `write_sidecar`
docstring even warns about this); hardlinking video files whole and
keeping the pointer columns gives bit-identical pixels with zero
re-encode (verified: shifted mid-file episode decode bitwise on both
cameras) where `delete_episodes` would re-extract streams. Oracles:
stats recompute vs both shipped stats.json (worst |Δ| 1.2e-4),
leakage certs ×3 PASSED + doctored-provenance negative control
FAILS, wrap census clean on both rig repos. The benchmark is now
gated only on tonight's box reads (slots) + launcher gen.

## 2026-08-05 — literature slice: one-step flow distillation is cheap now; LoRA facts for the rig pre-reg (~21:15Z)

Standing-allocation slice (~20 min, targeted at the active fronts).
Two keepers, both banked into ideas with numbers:
**SnapFlow** (arXiv:2604.05656) — plug-and-play SELF-distillation of
flow-matching VLAs to 1-NFE, no external teacher, ~12 h on one GPU,
no architecture change; π0.5-3B one-step matches its 10-step teacher
on LIBERO (98.75% vs 97.75%, 274→83 ms) and it's validated on
SmolVLA-500M — the closest published analogue to our
trunk+flow-expert stack. This makes ideas #12's distillation leg an
in-budget weekend arm rather than a research project, and it
compounds with #1 (1-step draws make mean-of-N nearly free at
deployment). **LoRA-for-π0** (arXiv:2607.10172) — r=32 saturation,
FFT no significant advantage, and freezing/LoRA-restricting the
vision encoder significantly degrades — external support for the
grounding-bottleneck read (#11) and a concrete ft-protocol arm for
the #16 rig benchmark (LoRA-r32 + full vision ft; 36.2→10.8 GiB
static VRAM). Pointers parked unread: OFP (from-scratch one-step),
GoldenStart (initial-noise structure — touches #1's draw keying).

## 2026-08-05 — leakage checker's identity branch now verified, not assumed (#18.8, ~21:20Z)

Deep-dive finding 6b closed (the last quick item on the #18 fix
queue): `bijou.eval.leakage`'s same-repo-id branch mapped training
episodes onto panel episodes *by assumption* — a
filtered-and-renumbered corpus that kept its repo id would certify a
false PASS while radioactive panel content trained. The branch now
asserts episode-count equality against the panel copy AND compares
per-episode length fingerprints (reads `meta/episodes.jsonl` v2 or
`meta/episodes/` parquet v3; metadata present on only one side is
fatal; the literal same-directory case short-circuits). Any mismatch
is a SystemExit demanding `meta/source_provenance.json` — symmetric
with the provenance branch's existing count assert. Evidence: 4 new
tests (179 green, `check.py` green); the full-corpus identity
certification re-ran PASSED under the new code (radioactive 5267 /
checked 47240, 4.1 s wall); a copy of
`therarelab/so100_pick_place_2` with a mutated episode count fails
loud with the intended message. Derived-corpus training (ideas #9,
the #13 repair arm) is no longer blocked on this.

## 2026-08-05 — the flow-vs-AR gap is a horizon story (~20:20Z)

Queue #4 executed (CPU, both GPU chains running): paired per-frame
analysis of the owner's 12:20Z box evals. Instrument first: the two
npzs pair bitwise on truth/valid/index/repo/core, and the pooled
summaries turn out to use **core frames only** (17,204 of 25,800 —
found by matching the report's `frames` field; my first
all-rows pooling missed the anchors by 0.027). With core-only
pooling all four anchors reproduce to 1e-4, then the deltas are
trustworthy. Findings in the
[post](posts/2026-08-05-flow-vs-ar-paired.md): flow beats AR on
horizon steps 0–1, crosses at step 2, diverges monotonically to
+1.2 by step 40 — the whole 0.82 pooled gap is late-horizon.
Deployment view (execute-k-then-replan): **flow wins k≤3, dead tie
at k=4, AR wins k≥5.** The panel's chunk_mae is the k=50 point,
i.e. the most AR-favorable reading on this axis — for rig-style
short-replan control the flow lineage is ahead, not 0.82 behind.
Surprise log entry: I had been carrying "flow trails AR by 0.82"
as a lineage-quality fact; it's a metric-horizon fact.
Falsifiable prediction banked in ideas #1 *before* the draws-10
numbers land: mean-of-N should move chunk_mae much more than
first_mae if per-draw spread grows with horizon step.

## 2026-08-05 — bijou deep-dive done: no P0, a ranked contract-gap list (~19:10Z)

The 16:17Z steer executed as the chained work session: all 57 files /
~22.3k lines of `bijou/` read line-by-line (six parallel subsystem
reviewers; every headline claim re-verified against the code by hand
before ranking — one reviewer top-finding was **refuted** that way:
the claimed right-padding sliding-window eviction can't happen
because the collator left-pads, `encoders/gemma4.py:170`, exactly the
test-gated 2026-08-01 decision). Full ranked list:
[deep-dive post](posts/2026-08-05-bijou-deep-dive.md).

Shape of the result: **the measurement core survives adversarial
reading** (pooling math, split determinism, FAST round-trip, Heun/π0
convention, HF bitwise parity anchor, seeding chains — all verified
sound), and **no current number is invalidated**. What it found
instead: a ranked layer of contract gaps that will bite *future*
numbers silently — flow eval noise keyed to corpus-relative index
(sealed plans pin frames, not noise; ~5.9° draw std vs 1e-4 bands),
three resume traps (same-seed data replay; fp32 masters bf16-snapped
each resume despite the "lossless" comment; changed backbone LR
silently ignored), the Q3 conditioning tripwire unable to fire for
flow decoders (fresh-noise floor), `--aux-prompt-hash` pinning train
but not eval, eval reports not recording scoring semantics
(`--condition-override` in no artifact), a `resolve_plan` bounds
hole, a leakage-checker renumbering hole, no absolute clamp on the
hardware rollout path, and rollout camera kinds diverging from
training's judge-voted kinds. Plus the two perf levers concretized
(idea #2 compile-blocker map; idea #8 chunked-CE design). Fix queue
= new ideas.md #18; the noise-seeding fix is a versioned instrument
break and waits for an anchor boundary + amendment.

Surprise log: the loading.py `backbone_snapshot` docstring claims the
masters' extra precision "lives only in optimizer.pt" — false
(AdamW state is moments, not weights). Comments lie; oracles don't.

## 2026-08-05 — charter v1.1: the owner-steered rules pass (~19:00Z)

The 16:21Z steer ("review all your rules and prompts … adjust them
however you see fit") executed as a work session. The day's steering
had outrun the written rules in eight places; charter v1.0 → v1.1
folds them in (full amendment list in charter §11): the **north
star** (rig VLA, few-shot transfer; panel = proxy) and the
**startup-velocity stance** now open §0; **measure versioning** (§2:
sealed/frozen instruments fixed by posted amendment — new plan file,
pre-registered shift, fresh anchors, loud deprecation; never silent
edits, defects only) codifies the sealed_v2 precedent; **loaned
compute** rules (§1) cover the second box; the **first-poll
utilization rule** and the **no-idle-pauses standing rule** (GPU-busy
windows are CPU work-item windows; sessions chain via the harness
marker instead of ending into idleness) land in §3; **post-cutoff
epistemics** (primary sources beat priors, `docs/gemma4.md` pattern)
in §6; **work→work chaining semantics** made explicit and the
**Discord house style** codified in §9. Prompts updated to match:
`tick.md` now chains a work session whenever GPUs are busy and
CPU-side items are queued (not only on queue-depth breach), `work.md`
re-arms the marker before ending under the same condition. Also fixed
en route: `fontaine/scripts/sealed_v2_anchor.py` lint debt that had
`check.py` red (repool output verified unchanged after the fix — v2
anchor 5.6903 reproduces).

One deliberate non-change: the harness driver itself. A work session
cannot chain another work session directly (`run_work_next` is only
consumed after a *tick*), and that stays: the ≤10-min seam between
work items buys a fresh babysit tick and bounded lock-holding — a
pause with a job, not an idle pause.

## 2026-08-05 — surprise: aux-OFF descends much faster early (box batch, E3 band already broken at 2.5k)

The box batch's E3 expectation said B-s0 (aux-off) tracks A-s0
(control) within the probe's ±0.3 at matched steps. At step 2500 the
256-frame probe reads **B-s0 16.85 vs A-s0 24.32** — a 7.5° gap, 25×
the band, in the aux-off arm's *favor* (B-s0 15.53 by 3k). Not a kill:
the kill gates (probe >15 @10k after falling-then-rising, NaN, OOM)
are untripped and the pre-registered primary read is the paired panel
at 40k. Mechanism candidate: with aux weight 0.5, early optimization
splits capacity/gradient between narration and action heads, so the
aux-on arm buys its narration with a slower early action descent; the
mainline 100k result ("aux within noise") is a statement about the
converged endpoint, not the path. Watch item for the remaining
babysits: does A-s0 close the gap by 10–20k (transient), or does
aux-off hold an offset to 40k (that would contradict the pre-reg's E4
expectation and make the aux attribution read a real finding either
way). Both curves' shapes are normal (steep monotone descent
33→24→? for A; 16.9→15.5 for B). E5 replicates (s1/s2) track A-s0's
lineage, so the pair-vs-replicate comparison at 40k stays clean.

**18:12Z update — the lead survives the noise floor.** All four
matched-2500 probes are now in: controls A-s0 24.32 / s1 29.72 / s2
29.69 (so the early seed envelope is [24.3, 29.7] — ~5° wide, the
±0.3 band was calibrated on late-training behaviour and is plainly
optimistic at 2.5k), while B-s0 sits at 16.85, **~7.5° below the
best control**. The early aux-off advantage is outside seed noise,
not an artifact of a lucky draw. Same watch item stands: transient
vs held-to-40k.

## 2026-08-05 — surprise: the sign-screen's standout was a ±180° wraparound, not a sign flip

The stage-1 sign-convention screen's flagship candidate (kevin510
wrist_roll, 14.9× panel-median MAE) dissolved on LOOKING at its
trajectories: 5/16 panel frames have *truth* chunks that wrap the
±180° boundary — one wrap contributes ~340°/step of raw-degree error
with zero convention fault. The aggregate screen conflated three
pathologies (wraparound, genuine mirror, tracked-but-offset); the
per-frame classification now lives in the probe and the split is the
result ([post](posts/2026-08-05-sign-convention-stage1.md)). Standing
implication: raw-degree training targets and MAE both see 360°
discontinuities on any repo whose wrist operates near ±180° —
panel-wide wrap census queued as a cheap follow-up (ideas #14).

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
