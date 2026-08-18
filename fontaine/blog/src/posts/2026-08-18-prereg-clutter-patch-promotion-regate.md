# Pre-reg: clutter-patch promotion re-gate (pinned 20×5 probe, production path)

*2026-08-18 ~14:1xZ. Registered before any data. Queue item
`clutter-patch-promotion-regate`; the re-gate itself was registered
with the original gate (in-channel pre-reg 05:23Z 2026-08-13:
"re-gate on the pinned 20×5 probe before any behavioral eval moves").*

## What happened

The clutter-patch promotion landed (2026-08-18 work session):
`SO101Sim` v3/v4 now default to `clutter_appearance='patched'` — the
mined real crops pasted onto the drawn plate at the drawn poses
(`sim/clutter_patch.py`, moved verbatim from the gate instrument),
stand-in geoms parked off-frustum, wrist path untouched, zero extra
RNG draws. Render oracles are green (wrist bit-exact vs standins; top
bit-exact outside clutter-affected pixels; physics/stream identity).

The gate evidence (05:4xZ 08-13, PASS) was measured on a *hooked*
instrument: no_clutter mask over the pasted plate, inside
`sim_fg_appearance_fix.py`. This re-gate checks the *production*
implementation reproduces that read before any behavioral eval runs
on the patched substrate.

## Protocol (frozen)

The leg (a)/(b) harness's slot grid verbatim: 20 seeds × 5 appearance
draws (`appearance = 1000*draw + seed`), settled resets, numpy post
backend, top camera. Two production arms, no hooks:

- **patched** — `SO101Sim(render_style="v3")` (production default);
- **standins** — same with `clutter_appearance="standins"` (in-run
  anchor).

Embed with the same er_60k probe checkpoint + real_v2 A/B reference
as the gate run (`sim_encoder_ood_probe` machinery), knn5 AUROC vs
real.

## Bands (frozen)

- **standins anchor**: AUROC in **0.708–0.718** (the gate's
  registered v3 abort band). Outside → abort, no claims.
- **PASS**: patched AUROC within **±0.010 of 0.556** (the gate's
  patched read). The production path differs from the gate arm only
  at pixels where a stand-in occludes dynamic content — measured none
  on the oracle seeds — so the read should reproduce nearly exactly.
- **0.556 + (0.010, 0.030]** → record, inspect dumped frames before
  any behavioral move (implementation drift suspected, e.g. mask vs
  off-frustum occlusion differences).
- Worse than +0.030 → the promotion is suspect: revert the default to
  `standins` pending diagnosis.

## Cost and gating

~0.02 GPU-h (200 renders CPU/EGL + encoder embeds). Gates behavioral
evals **on the patched substrate only** (demo-gen v1.1, future sim100
cells). Tonight's pdnorm endpoint sim100 is pinned
`--clutter-appearance standins` (pdnorm prereg Amendment 1) and does
not wait on this.
