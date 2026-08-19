# Now












*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-19 11:07–11:2xZ (real `date -u` at write: 11:20) —
work session (chained): **hf-evacuation-audit-v2-fleet EXECUTED +
CLOSED — the schema-2 fleet is fully recoverable off-box, ZERO weight
gaps, zero uploads needed; every claim verified by sha256 or a live
converter re-run. Owner surfaced mid-session ("What are my calls?") —
answered in-channel with both pending calls + recommendations.***

**Status**: no live runs — babysit registry empty, H100 idle (0 MiB /
0%) all session, policy-server not up; RAM 196 GiB available, disk
208 GB free. CPU + network only (0 GPU-h). The staged demos+one-rig
cell remains the only GPU item and pends the owner isolation call.

**Steering**: owner asked "What are my calls?" (11:12:53Z, id
…921930) — replied 11:19Z (post …522815) with the two open calls +
recs: (1) demos+one-rig isolation → **rec GO** (frozen cell, launcher
parse-green, H100 idle); (2) R2 band → **rec AMEND + ACTIVATE R2
from the 7% checkpoint** (decode diagnosis: greedy magnitude
attenuation, not calibration; R2 samples T=1.0), wave-0 abort bar
mixed <20%. Acked; awaiting the decisions — tight polls live.

**Done**: `hf-evacuation-audit-v2-fleet` EXECUTED + CLOSED. Method:
343 HF LFS sha256s enumerated + local `sha256sum` + two live
`convert_molmoact2` re-runs. Verified per `_v2`: er_60k trainer
source bitwise on HF (backbone/expert/prompt; `aux_loss_weight 0.5`
in the banked config reproduces `narration_weight 0.5` through the
committed rename fallback); joint_corrected v1 dir bitwise on HF
(schema_version 1); base + jointsurface experts re-extracted from
the public allenai snapshot **bitwise equal** (`4517d649…`); stage-C
pair re-extracted from the stagec-hf export **bitwise equal**
(`d840174d…`), with the HF model-delta bitwise = local staging
(`98f32f4d…`) and the overlay README banked. corrected_v1 ==
stagec expert bitwise (metadata-only variant). The two 2.2G legacy
expert-source dirs hold zero unique bytes — safe to retire. Public
allenai repo dependency flagged as a recorded acceptance. Mapping
table: posts/2026-08-19-hf-evacuation-audit-v2-fleet.md.

**Next**: `queue_cli.py next` → CPU items `token-probe-html-gallery`
(all inputs banked) and NEW refill
`disk-retirement-sweep-banked-sources` (er_60k trainer dir 37G
sha-audit + the two legacy dirs, ~41G payoff, disk 93% used);
`demos-plus-one-rig-exec` + R2 activation pend the owner replies to
…522815.*

*Updated 2026-08-19 11:04–11:0xZ (real `date -u` at write: 11:05) —
tick: **quiet tick right after the v1-fleet-upgrade close — no live
runs, H100 idle (0 MiB / 0%), channel quiet; `run_work_next` already
armed (11:03 close), so this tick closes fast to hand off to the
chained work session.***

**Status**: no live runs — babysit registry empty
(`no_live_runs_reason` current), GPU 0 MiB / 0% (H100 free;
policy-server not up). RAM 196 GiB available, disk 208 GB free. The
staged demos+one-rig cell remains the only GPU item and pends the
owner isolation call.

**Steering**: none — read empty, inbox empty, history (last 5) shows
no new reactions. OWNER CALLS PENDING: (1) demos+one-rig isolation
(draft 04:25:54Z, id …759115); (2) the B §3 R2 band (post
…840340, carrying the ACTIVATE-from-7% recommendation + receipts).

**Done**: boot (pull clean, queue validate green depth 2 / 16 open),
Discord read + history, standing free/df + GPU checks. No post
(quiet interval); no in-session hold — the marker was already armed
at the 11:03 work-session close, so the fastest path to resumed
polling + queue work is the chained session itself.

**Next**: chained work session takes CPU items
`token-probe-html-gallery` (all inputs banked) and
`hf-evacuation-audit-v2-fleet`, and keeps the owner-call polls;
`demos-plus-one-rig-exec` + R2 activation stay owner calls.*

*Updated 2026-08-19 10:46–11:0xZ (real `date -u` at write: 11:03) —
work session (chained): **v1-fleet-upgrade EXECUTED + CLOSED
(b1d1b27) — the schema-1 checkpoint fleet is retired: 6 v1 originals
(audit found 2 beyond the queue's 4), 4 fresh `_v2` conversions all
load-smoked, er_60k `_v2` regenerated with the trained
narration_weight 0.5, zero schema-1 dirs remain on disk.***

**Status**: no live runs — babysit registry empty, H100 idle (0 MiB /
0%) all session, policy-server not up; RAM 196 GiB available, disk
212 GB free (the materialized per-part trunks cost ~55 GB net; v1
weight files were hard-links so retirement freed little). CPU-only
session (0 GPU-h). The staged demos+one-rig cell remains the only GPU
item and pends the owner isolation call.

**Steering**: none — read empty at boot and at every stage poll,
inbox empty. OWNER CALLS STILL PENDING: (1) demos+one-rig isolation
(draft 04:25:54Z, id …759115); (2) the B §3 R2 band (post
…840340, carrying the ACTIVATE-from-7% recommendation +
receipts).

**Done**: `v1-fleet-upgrade` EXECUTED + CLOSED (b1d1b27). Disk audit
found SIX schema-1 dirs (queue named 4): the 5th was base `_vla`
whose `_v2` existed since edb8d4e with the v1 never retired, the 6th
the `step_002000` straggler in finetune/. Three pristine-trunk
conversions via `bijou.convert_v1` (jointsurface → Joint 5485M,
stage-C corrected_v1 + _vla → Flow 5490M), all load_vla-smoked.
er_60k `_v2` REGENERATED from v1 (regenerate-vs-annotate decided
regenerate): old-vs-regen all 4 weight files bitwise equal, tokenizer
identical, metadata diff exactly `narration_weight` 1.0→0.5; swapped
+ smoked (Molmo2ARVLA 4856M). Mounts repointed before retirement
(`sim_clutter_promotion_regate.py` default checkpoint, probes
launcher BASE+CKPT → `_v2`); no-blind-delete greps clean incl. the
policy-server checkout (no serving mounts). Six v1 originals retired
staggered with df checks; final sweep: zero schema-1 `metadata.json`
under ~/checkpoints + outputs. check.py 1066 green. Result post
…282587.

**Next**: `queue_cli.py next` → CPU items `token-probe-html-gallery`
(all inputs banked) and NEW refill `hf-evacuation-audit-v2-fleet`
(verify fontaine-checkpoints holds a recovery path per `_v2`; the two
legacy bijou_config expert-source dirs included before any retirement
decision); `demos-plus-one-rig-exec` + R2 activation stay owner
calls. `run_work_next` armed at close (CPU queue non-empty).*

## Utilization footer

Session 2026-08-19 11:07–11:2xZ (work, chained; 0 GPU-h — CPU +
network only, H100 idle throughout): **hf-evacuation-audit-v2-fleet
CLOSED — schema-2 fleet fully recoverable off-box, zero weight gaps,
zero uploads: er_60k + joint_corrected sources bitwise on HF, base +
stage-C experts proven byte-derivable by live converter re-runs,
delta bitwise-verified; the two 2.2G legacy dirs hold no unique
bytes** — exploit (integrity/infra debt); owner surfaced ("What are
my calls?") — both calls + recs posted in-channel 11:19Z, acked,
tight polls live; queue green depth 2 (16 open: refill
`disk-retirement-sweep-banked-sources`).

Session 2026-08-19 11:04–11:0xZ (tick; 0 GPU-h — no live runs, H100
idle): **quiet tick right after the v1-fleet-upgrade close — GPU
0 MiB / 0%, RAM 196 GiB available, disk 208 GB free; Discord fully
quiet (read empty, inbox empty, no new reactions); no in-session hold
— `run_work_next` was already armed at the 11:03 close, so the tick
closed fast to hand off** — the chained work session takes CPU items
`token-probe-html-gallery` + `hf-evacuation-audit-v2-fleet` and keeps
the owner-call polls (demos+one-rig isolation, R2 band — both still
unanswered); both GPU actions stay owner-gated. Queue green depth 2
(16 open).

Trailing-7-day GPU-hours on experiments / total (window 2026-08-12
00:00Z → 2026-08-19 08:45Z; rolled 08-19 from the 08-17 rebase +
prune records + archive session notes — receipts in
`fontaine/notes/util-window-roll-2026-08-19.md`, instrument
`fontaine/scripts/util_ledger_extract.py`): local **~84.1 / ~85.5**
(retained 08-12→stamp ~57.5 + post-stamp ~28.1: discriminator
roll-in ~4.8, pdnorm screen-wide ~15.9 train+battery, joint-probe
legs 3+4 ~3.9 incl. leg 4 live at stamp; ops/loss ~1.4 =
discriminator attempt-1 OOM + smokes). Local-only from this roll —
the box was killed 08-17 (~106 box GPU-h fall in-window for the
record; final box history in
`fontaine/notes/utilization-rebase-2026-08-17.md`). Older dated
snapshots and session notes: rolled verbatim to
the [now archive](archive/now-2026-08-07.md); the superseded 08-06
baseline + its accreted narrative: rolled verbatim to
[archive 08-17](archive/now-2026-08-17.md).
