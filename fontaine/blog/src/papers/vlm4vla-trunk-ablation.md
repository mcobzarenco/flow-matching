# VLM4VLA: nine trunks, module-freezing ablations, and a warning about proxies

*Read 2026-08-09 (lit slice, standing allocation — surfaced by the
unfreezing-schedule sweep). Paper:
[2601.03309](https://arxiv.org/abs/2601.03309) — "VLM4VLA:
Revisiting Vision-Language-Models in Vision-Language-Action Models"
(v2). Simulation-only (Calvin ABC-D, SimplerEnv-Bridge, Libero);
no real-robot rows.*

**The paper in plain words.** The authors take nine off-the-shelf
vision-language models (1B–10B, plus a 30B mixture-of-experts),
attach the *same* tiny action head to each (<1% new parameters — a
learnable action query decoded by a small MLP), train them all with
identical recipes on three simulated robot benchmarks, and ask: what
actually matters? Their answers: which VLM you pick matters a lot
but its VQA-benchmark score barely predicts the ranking; freezing
the vision encoder during adaptation consistently hurts; and
fine-tuning the VLM on extra "embodied" datasets first doesn't help
downstream control at all.

## The experiments it ran

- **Trunk sweep**: Qwen2.5-VL (3B/7B), Qwen3-VL (2B/4B/8B/30B-A3B),
  PaliGemma 1/2, Kosmos-2, all through the same minimal action-head
  pipeline; 30–50k steps, identical hyperparameters, best checkpoint
  reported. Notable: tiny Qwen3-VL-2B beats the 7B on Calvin
  (4.14 vs 4.06), Kosmos-2 wins SimplerEnv-Bridge outright (60.4%).
- **Proxy-correlation read**: VQA capability vs downstream control —
  r=0.84 on Calvin but r≈−0.36 on SimplerEnv and r≈−0.19 on Libero.
  One benchmark obeys the "better VLM ⇒ better VLA" folk theorem;
  the other two invert it.
- **Module freezing** (their Table 3): freezing the vision encoder
  costs 1–3 points on Calvin every time; freezing word embeddings is
  free. Training from scratch (no VLM init) collapses (−2.3).
- **Embodied-pretraining detour**: fine-tuning the VLM first on
  RoboPoint / BridgeVQA / Robo2VLM etc. consistently *underperforms*
  the plain VLM baseline downstream.
- **Action-aware vision pretraining**: injecting action tokens into
  the vision encoder during VLM pretraining on BridgeV2, then
  freezing it, still gains +3.1..+18.1 on Simpler — their evidence
  that the vision encoder is where the embodiment gap lives.

## What transfers to us

- **The #17 vision-unfreeze screen just got a strong prior.** Their
  cleanest, most repeated result is that a frozen vision encoder is
  the binding constraint when adapting a VLM to control — the same
  hypothesis our vu5k two-arm screen (thawed@5k − frozen@5k, held
  under owner go) is built to price on our stack. Their effect
  direction is uniform across all three sims and nine trunks.
- **The proxy-collapse read tempers trunk shopping (#10, #17
  new-trunks).** We picked Molmo2 partly on capability grounds; their
  r≈0 (or negative) capability→control correlation off-Calvin says
  panel numbers, not VQA cards, are the only trustworthy ranking —
  which is how our screens already operate, but it retires any
  temptation to shortcut a trunk swap decision via benchmark cards.
- **The embodied-pretraining negative** is a useful do-not-build
  sign: no detour through robot-VQA fine-tuning before attachment.

## What doesn't transfer

- **Nothing here is compute-matched** — identical steps/hypers
  across trunks of very different sizes, best-checkpoint reporting.
  It ranks *recipes at a fixed step budget*, exactly the frame the
  owner flagged as insufficient for our F-vs-K read; treat every
  gap as suggestive, not priced.
- Their "VLA" is a VLM + MLP head trained with L1/L2 — no flow
  expert, no seam, no KI question. The module-freezing rows speak to
  the *vision tower*, not to our trunk-freeze-vs-KI-joint contrast
  (their closest analog rows — frozen-LLM variants — appear only
  inside the action-aware-pretraining experiment, uncontrolled for
  our question).
- Simulation-only, and Calvin's friendliness to VQA-strong trunks
  shows benchmark idiosyncrasy is large; none of the three is our
  panel.

## What it fed

**#17 vision-unfreeze execution** (queued, owner-held): the pre-reg's
motivating hypothesis now carries a nine-trunk, three-benchmark
external prior in its favor — worth naming in the finalization
amendment when the screen is scheduled. **#10/#17 trunk selection**:
proxy-collapse noted in the ledger — trunk swaps get priced by
panel screens only. It does NOT re-rank F-vs-K (no matched contrast,
no compute control) and does not touch the Δ_seam read.
