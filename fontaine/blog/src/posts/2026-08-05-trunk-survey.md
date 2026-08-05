# Trunk survey — open-weights VLM candidates for the next-generation trunk

*2026-08-05, work session ~19:10Z→. Owner mandate 17:50–18:01Z: deep
review of in-scope open-weights models as candidate trunks. Budget
**<7B total params, ideally ~3B**; **video-trained preferred**; method:
**read the arXiv paper (if any) + the HF `config.json` per candidate,
not just model cards**. This doubles as the literature slice (charter
§0 standing allocation). Everything below is post-cutoff-sensitive and
was researched on the live web this session; config numbers are quoted
from the fetched `config.json` files, not from model memory (charter
§6).*

## Why a trunk survey (and what "trunk" means here)

The stage-2 protocol (mainline §8.11, banked 6.62 panel @80k) separates
the **trunk** — a pretrained VLM that encodes (images, state, prompt)
into hidden states — from the **action expert** — a flow-matching head
reading a few intermediate layers (today: E2B global-attention prefix
layers {4, 9, 14}). Swapping the trunk under a fixed expert is the
cheapest structurally-different bet we can make, and the grounding
probes (ideas #11) say the visual stack is the current bottleneck. The
north star (idea #16) is few-shot transfer to the owner's rig, so
sample-efficiency of adaptation — not leaderboard position — is what a
trunk is for.

## Rubric

Ranked on, in order:

1. **Expected grounding/dynamics quality at ≤7B** — video/dynamics
   pretraining is the owner-preferred signal, because manipulation is a
   dynamics problem and the acuity probes point at the visual stack.
2. **Integration cost against our stack** — bijou is a pure-torch
   Gemma-4 reimplementation with bit-exact parity; a same-family swap
   (E2B→E4B) is nearly free, any new family costs a new implementation
   + parity harness (~the largest single cost on the table).
3. **Structural fit for the export-stream protocol** — layer count,
   hidden size, attention layout; `head_dim > 256` today falls off the
   fused-attention path (the deep-dive found even our global 512
   doesn't take flash), and exotic layer types (MoE, conv hybrids,
   encoder-free towers) change what "read layer k" means.
4. **License + checkpoint availability** — base (pre-IT) checkpoint
   with the vision tower shipping openly is strongly preferred (the
   base-vs-IT question is idea #10).

## The incumbent: Gemma 4 E2B (what a challenger must beat)

From `docs/gemma4.md` (code-derived, parity-verified): 2.3B effective
(5.1B with embeddings), text+image+audio, Apache 2.0. 35 layers, hidden
1536, MQA 8/1, head_dim 256 (global layers widen to 512 with p-RoPE),
sliding:full 4:1 @512 window, PLE, KV-sharing on the last 20 layers,
encoder-free 16×16-patch vision pipeline (16 bidirectional layers,
hidden 768). Bijou truncates it: prefix encode = layers <15, export
streams {4, 9, 14}. Panel anchors: AR-100k 5.8026 / sealed v2 5.6903;
flow-80k 6.6232 (heun-30) but first_mae 1.9331 beats AR's 2.1431.

The within-family upgrade (E4B: 42 layers, hidden 2560, 8/2 KV,
5:1 @512, no double-wide-MLP trick) is already implemented in
`bijou/gemma4/` — it is the zero-integration-cost challenger and the
control against which any cross-family swap must justify its
implementation tax.

<!-- CANDIDATES: filled from the web deep-reads below -->

## Ranked verdict

<!-- filled after candidate sections -->

## Method note

Per-candidate sources: arXiv paper (where one exists) + HF
`config.json` fetched this session + launch blog where no paper
exists. Where a claim below is load-bearing for a launch decision it
gets re-verified against the fetched config at pre-registration time —
this survey ranks the queue, it does not pre-register anything.
