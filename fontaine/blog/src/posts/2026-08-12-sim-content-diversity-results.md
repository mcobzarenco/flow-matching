# Sim content diversity v3: spread 3× — bar MISSED — while the guard falls to 0.673, the best top-cam read yet

*2026-08-12 ~07:2xZ work session (in-channel 07:24:00Z per the
Discord timestamp; this stamp and the pre-reg's were corrected at
07:2xZ — the first draft carried times from an unchecked clock).
Executes the
[content-diversity pre-reg](2026-08-12-prereg-sim-content-diversity.md)
(registered same session, in-channel 06:35:39Z). Verdict up
front, both legs: **top sim k std/mean 0.038 → 0.114, short of the
registered ≥ 0.15 — the bar is MISSED as registered.** The AUROC leg
of the same bar did not just hold its ≤ 0.790 guard — it fell to
**0.673** (v2: 0.773), with k-ratio **1.02×** and centroid ratio
0.99×: the top camera's composites now sit *inside* the real
embedding spread, the strongest top-cam read this axis has ever
produced. Per the registered flip rule (bar met = flip), the shipped
default **stays v2**; v3 strictly dominates v2 on every measured
number, so the flip is put to the owner as a one-line call.*

## Plain words

We gave the simulator a deck of 26 real backgrounds — one per real
recording, each carrying that recording's actual daylight — and
taught it to scatter the desk clutter (mouse, mug, laptop) the way
the real operator actually scattered it between recordings,
including leaving things out of the scene at the measured
frequencies. Every simulated episode now looks like *a different
day at the table* instead of the same day repeated. We pre-promised
two numbers: the spread of the simulator's frames (as the policy's
own vision encoder sees them) should reach 15% where real
recordings sit at 45%, and the frames should get no easier to tell
from real. The spread tripled — 4% to 11% — but did not reach 15%,
so by our own pre-registered rule this is a miss and the new mode
does not ship as the default. The second number is the story
though: telling these frames from real got *harder* (0.773 → 0.673,
where 0.5 is "can't tell") — the closest the simulator's main
camera has ever been to passing for real. What's left of the
sameness is the part a fixed start-of-episode snapshot can never
have: mid-episode motion, human hands reaching into frame.

## Registered reads (reset-render probe, 100 seeds; 20×5 sensitivity)

| read | v2 (shipped) | v3 (this item) | registered line |
|---|---|---|---|
| top sim k std/mean (co-primary) | 0.038 | **0.114** (0.114 at 20×5) | ≥ 0.15 → **MISS** |
| top 5-NN AUROC (co-primary guard) | 0.773 | **0.673** (0.655 at 20×5) | ≤ 0.790 → **over-met** |
| top k-ratio sim/real | 1.16× | **1.02×** | — |
| top centroid ratio | 1.11× | **0.99×** | — |
| per-draw mean-k spread (20×5) | 0.005 | **0.025** | record-only |
| wrist (guard) | 0.548 | **0.548** — bit-identical | frames equal → **GREEN** |

- **Wrist guard**: v3 wrist frames bit-identical to v2 for the same
  (seed, appearance_seed) — asserted on renders
  (`fontaine/scripts/sim_v3_wrist_guard.py`, 5 pairs) before any
  read was credited; the probe's wrist numbers reproduce the
  periphery-fix close exactly.
- **Physics oracles green before any read**: settled qpos
  bit-identical across v0/v1/v2/v3 and across appearance seeds;
  spawn stream bit-matches banked sim100; new oracle pins v3
  clutter draws physics-inert (`tests/test_sim_appearance.py`, 6
  green).
- **Overfit tripwire** clear (0.673 ≫ 0.5).
- **Record-only breakdown**: all 26 plates drawn across the 100
  seeds; between-plate mean-k variation is most of the new spread
  (0.098 of 0.114); by drawn-clutter count, sparse resets sit
  closest to real (mean k 1.15e-5 at 1 object → 1.41e-5 at 4).
- **Iterations**: 2 of the registered ≤ 3 used. Iteration 2 composed
  each drawn plate's measured episode gain/bias onto the rendered
  foreground (coherent per-episode lighting) — encoder-null (0.114 /
  0.673 vs 0.114 / 0.671), kept for realism. Third confirmation on
  this axis that the er_60k encoder is invariant to global
  photometric shifts (v1 lighting jitter ~3%, v2 fixed-plate
  homogeneity null, now this): **content moves it, light does not.**

## Side by side

[REAL | v2 | v3 gallery](https://mcobzarenco-fontaine-reports.static.hf.space/chart__sim_content_diversity_top_gallery.png)
— the v3 row varies plate lighting and clutter per reset (absences
included; the mouse is genuinely gone from 73% of real A episodes).
[k-distance strips](https://mcobzarenco-fontaine-reports.static.hf.space/chart__sim_content_diversity_kdist.png)
— v3's cloud shifts onto the real held-out bulk and widens; the
remaining real tail (distances 2–4×10⁻⁵) is mid-episode content no
reset render can produce.

## What landed (ships as `render_style="v3"`, default STAYS `v2`)

1. **Plate bank miner** (`make_clean_plates.py --bank` →
   `assets/real_plates/bank/`): 26 per-episode top plates, inlier
   median vs the per-episode gain/bias-corrected global plate with
   feathered fallback — the naive median's baked boat-on-disk, arm
   rests and operator hands are excluded by construction (channel-MAX
   deviation; the channel-mean version let the operator's hand pass
   and smeared skin into a plate — caught by inspection, first
   candidate). Fallback fraction 3–23% per plate.
2. **Measured clutter spread** (same pass, `bank_manifest.json`):
   camera model verified by a displace-and-recover selfcheck through
   the sim's own segmentation renders (0.4 cm mouse / 1.7 cm pcb;
   per-object centroid-bias calibration — raw analytic error was up
   to 12 px on flat shapes). Mouse present 27% of A episodes
   (absolute box ≈ 7×16 cm), the white up-table item the mug stands
   in for 15%, laptop 77% (drawn as deltas about canonical — its
   real center is past the frame edge), pcb near-static (kept
   canonical). **Record-only: the real disk wanders 8–29 cm × ±19 cm
   across episodes** — banked as the baseline fact for the
   out-of-scope disk-position item.
3. **`render_style="v3"`**: v2 composite + per-reset plate draw +
   clutter presence/pose draws, all consuming the appearance RNG
   after every v2-era draw; the wrist render swaps clutter to
   canonical (data-side, physics never sees it) so the wrist path is
   v2 bit-identical.
4. Charts + probe JSONs pushed; guard script; oracle added.

## The flip question (owner call) — ANSWERED: flipped

By the registered rule the default stayed v2 — v3 dominates v2 on
every measured axis (spread 3×, AUROC −0.100, both ratios at ~1.0)
and costs nothing at runtime, so the flip went to the owner.
**Owner approved in-channel 07:29Z 08-12 ("should we swing to v3
then?") and the default is now `render_style="v3"`** (one line,
`so101_sim.py`; oracles + wrist guard re-run green). The sim100
rerun gate now reads GO-with-v3-frames on both cameras (top 0.673 +
wrist 0.548). The spot-check option from the v1 close stands.

## Why the spread bar was probably unreachable

Real held-out spread (0.447) is dominated by episode phase: arm
sweeps, the boat mid-carry, operator hands — the long right tail in
the strip chart. Reset renders sample none of that by construction.
The content levers this item registered (plates + clutter) moved
reset-frame spread 3× toward the measured ~0.10–0.12 that
*start-of-episode* real frames plausibly occupy; pushing further on
this instrument means diversifying *phase*, not appearance — i.e.,
probing rollout frames, not reset frames (a different registered
read).

## Artifacts

- Probe JSONs:
  [v3 primary](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_encoder_ood_probe_v3_content.json)
  · [20×5 homogeneity](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_encoder_ood_probe_v3_homog.json)
- Charts:
  [k-distance strips](https://mcobzarenco-fontaine-reports.static.hf.space/chart__sim_content_diversity_kdist.png)
  · [REAL | v2 | v3 gallery](https://mcobzarenco-fontaine-reports.static.hf.space/chart__sim_content_diversity_top_gallery.png)
- Total GPU spend this item: ~0.08 GPU-h of probe/guard reads (gate
  0.3); mining and iteration were CPU.
