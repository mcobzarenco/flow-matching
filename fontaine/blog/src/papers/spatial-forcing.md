# Spatial Forcing: the aux loss that mostly buys convergence speed

*Lit slice 2026-08-09 (work session 11:56Z, ride-along after the
[async cluster](async-execution-2.md) closed early). Spatial Forcing
([2510.12276](https://arxiv.org/abs/2510.12276), v2 Oct 2025) — read
as the banked follow-up to [VEGA](vega-encoder-grounding.md), which
used it as its main baseline. Fed #17 (the aux-alignment pole gains
a training-speed claim), #11 (aux-family ledger), #4/#2 (a
fewer-steps lever, distinct from every step-time lever we measured).*

## The paper in plain words

Robot policies built on vision-language models are good at knowing
*what* they see and bad at knowing *where* things are in 3D — their
vision was pretrained on flat internet images. Instead of bolting on
depth cameras or depth estimators, this paper adds a side loss
during training: the VLA's internal visual features, partway through
the language model, are nudged (by cosine similarity, through a small
throwaway projector) to look like the features of VGGT, a 3D
geometry foundation model that infers spatial structure from plain
RGB. At test time the extra parts are deleted — same model, same
speed. The surprise is *where* the benefit shows up: final success
rates improve modestly, but the model gets there dramatically faster
— the same LIBERO success in roughly a third of the training
iterations (claimed up to 3.8×), and with 5% of the data it beats
the baseline by ~26 points.

## Contribution

The mechanism, precisely: take the visual token embeddings at an
intermediate LLM layer (layer 24 of OpenVLA-OFT's 32 — "deep but not
deepest" is load-bearing, see ablations), pass them through
batch-norm + a 2-layer MLP, and maximize cosine similarity with
VGGT's pixel-level spatial features (positional embeddings added to
the targets to keep token ordering). Weighted α=0.5 next to the
action loss. Projector and teacher are discarded at inference —
zero deployment overhead, no architecture change. This is the same
recipe family as VEGA with two design choices flipped: alignment
depth (LLM-interior vs encoder output) and, implicitly, what the
teacher is asked to fix (the LLM's *use* of visual tokens vs the
encoder's representation itself).

## Experiments

- **LIBERO** (OpenVLA-OFT base): SF 96.9% average across the four
  suites — final-score parity-ish with the strong baseline (their
  own table has the base at ~97.1 in one comparison and 92.7 in the
  ablation setting; the extraction is setting-dependent, so we carry
  the *convergence* claim, not a final-score win). The headline:
  the same success reached ~50k vs ~150k iterations, "up to 3.8×"
  faster by their success-vs-iterations curves.
- **Data efficiency**: at 5% of the demos, 75.8% success — +25.8 pp
  over the baseline at matched data; "5.9× more data-efficient" at
  matched success.
- **RoboTwin** (π₀ base): improvements on easy and hard splits —
  but [VEGA](vega-encoder-grounding.md) beats it there (64.2/27.8 vs
  VEGA's 67.5/30.7 easy/hard), which is exactly why we read VEGA
  first.
- **Real AgileX bimanual**: stack-glass-cups +47.5 pp over baseline;
  small task count (2 tasks, 20–40 demos).
- **Ablations** (the valuable part): teacher choice at layer 24 —
  VGGT 96.9 > VGGT-without-PE 94.7 > DINOv2 94.1 ≈ SigLIP 94.0 >
  no-alignment 92.7 (even a 2D semantic teacher helps some; the 3D
  teacher + token ordering carries the rest). Layer choice — 24 ≫
  {32, 16, 1} (94.8/93.8/94.6): aligning the *deepest* layer hurts,
  presumably colliding with the action head's working
  representation. Weight α — 0.5 optimal; 12.5 destabilizes (81.2).

## What transfers to us, what doesn't

- **The VGGT-teacher contradiction with VEGA is depth-resolved, and
  that's the actionable read.** VEGA reported VGGT-as-teacher
  *collapsing* to 0.04 on RoboTwin-hard when aligned at the encoder
  output; SF gets its best numbers from VGGT aligned at LLM layer
  24. Same teacher, opposite outcome, different depth. So "which
  teacher" is not a free-floating question — teacher and alignment
  depth interact strongly, and any future aux-alignment arm on our
  stack should treat (teacher × depth) as the grid, not teacher
  alone. Both papers agree on the deeper pole: aux-injected spatial
  structure without inference-time cost is real.
- **The 3.8× is a *fewer-steps* lever, not a *faster-steps* lever —
  a genuinely new column in our throughput accounting.** The owner's
  throughput thread (perf pass-1, #20 actckpt) is entirely about
  s/step and memory→batch. An aux loss that reaches target quality
  in ~⅓ the iterations attacks wall-clock from the other side, at
  the cost of running the teacher during training (unreported
  overhead — the one number the paper conspicuously omits, and VGGT
  is not small). Banked as a named candidate for any future
  fresh-trunk launch *pre-registration conversation*, not as a
  claim: their evidence is OpenVLA-OFT discrete-token AR on LIBERO;
  our trunk is Molmo2 with a flow expert, and none of their curves
  ride our objective.
- **Caveat for reading it onto #17 vu5k:** SF never tests unfreezing
  — its base keeps the standard OpenVLA-OFT recipe. VEGA's
  frozen≈unfrozen probe remains the only direct freeze-axis
  evidence; SF adds the convergence-speed and data-efficiency
  dimensions the vu5k readout doesn't measure. If vu5k's thawed arm
  wins, the cheap-escalation order stays VEGA-style encoder
  alignment first (it beat SF head-to-head where compared), with
  SF's layer-24 variant as the sibling if encoder-level fails on
  our single-tower trunk (Molmo2 has no clean encoder/LLM seam for
  the VEGA recipe — SF's LLM-interior hook may actually *fit our
  architecture better*, which VEGA's page flagged as its main
  transfer caveat).
- **Doesn't transfer:** the absolute LIBERO numbers (discrete-token
  AR, mature benchmark, different data regime); the real-robot
  deltas (2 tasks); any final-success-rate claim (their own tables
  are ambiguous at matched iterations — the convergence claim is
  the defensible one).

## Where it lands

- **#17**: the aux-alignment third pole now has two recipes (encoder
  / LLM-interior) with a measured teacher×depth interaction; SF is
  the named sibling escalation if VEGA-style doesn't fit the
  single-tower trunk. Convergence-speed + data-efficiency columns
  added to the pole's ledger.
- **#11**: aux-family sighting — spatial structure injectable
  without depth sensors, evidence class: strong on convergence,
  weak on final score.
- **#4 / #2 (throughput accounting)**: "fewer steps to quality" is
  now a distinct, citable lever class alongside step-time and
  memory→batch; teacher-forward training overhead unreported —
  demand that number before any pre-reg.
