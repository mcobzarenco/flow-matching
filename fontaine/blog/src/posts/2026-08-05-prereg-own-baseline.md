# Pre-registration: own-baseline arm (`fontaine_arb_rcond_100k_1xh100`)

> **SUPERSEDED 2026-08-05 16:13Z, never launched** — owner steering
> replaced the standalone baseline with a
> [paired 40k design](2026-08-05-prereg-paired-auxoff-40k.md) whose
> arm A is this recipe at 40k and doubles as the own-topology control.
> Body below preserved verbatim per the immutability convention.

*2026-08-05. Immutable once posted; the launcher header carries the
same content. Launches tonight after the bootstrap eval bursts, gated
on the smoke run passing.*

## Question

What does the mainline-best recipe (`bijou_arb_rcond_100k_ddp4` —
request-conditioned ar_backbone, prompt fmt 3 / suffix fmt 5) do on
**this** topology — 1×H100, eff-batch 10, single process — at matched
steps? Charter §4's own-baseline rule: until this arm exists, every
training delta on this box is "vs mainline, cross-topology —
directional only". This run is the anchor later arms pair against.

## Exact command

`~/launch_fontaine_arb_rcond_100k_1xh100.sh` (committed convention:
launcher header = this post). Recipe flags verbatim from the mainline
100k report; differences, exhaustively: `--batch-size 10` (vs 12→10;
B10 is mainline's standing post-OOM setting — adopted from step 0, no
batch roulette), single process (no DDP wrapper), `--num-workers 16
--prefetch-factor 4`, `--save-every 5000`, wandb project `fontaine`.
Seed 0, fresh run, no resume.

## Numbered expectations

1. **Startup**: selection 878 datasets / 42,872 episodes (verified on
   this mirror today); model line = ~11M new decoder params + live
   text trunk at 2e-5; wandb run visible in `fontaine`.
2. **Throughput**: 0.4–0.6 s/step at B10 (mainline per-rank 0.42–0.47
   at B10–12); VRAM peak < 76 GiB (B11 measured 75.6); ~12–14 h wall
   for 100k.
3. **Curve** (256-frame in-run probe, ±0.3 floor): below **12 by
   10k**, below **9 by 30k**. The mainline eff-48 curve (7.54@10k,
   6.57@30k, 5.55@100k) is expected to be AHEAD of this arm at every
   matched step — eff-10 sees 1/4 the samples per step. This arm
   matching the mainline curve would itself be a finding (batch-size
   insensitivity at fixed steps).
4. **Kill gates**: probe > 15 at 10k with a falling-then-rising shape
   (divergence, not slowness); NaN loss; second OOM after the
   standing B−1 resume. Slowness alone is data, not a kill.

## Known seams and confounds

- eff-10 vs eff-40 means same-steps ≠ same-samples vs mainline: this
  arm anchors the topology precisely so that later comparisons are
  paired on it, not on mainline.
- LRs are kept verbatim (1e-4 / 2e-5) rather than batch-rescaled —
  deliberate: the arm measures the recipe as-is on this box; an
  LR-rescale arm is a separate follow-up if the curve is badly off.
- Expected fingerprint of the smaller batch: noisier probe series and
  a later-arriving plateau, not instability.

## Cost

~12–14 h of the standing GPU (overnight — charter §3), ~480 GB disk
(checked against `df`), zero API spend.
