# Pre-registration: SnapFlow student → rig fine-tune (1-NFE, owner-steered)

*2026-08-06 ~17:0xZ. Status: pre-registered, POSTED BEFORE LAUNCH.
Owner steering 2026-08-06 16:35Z: "we probably need a nfe fine tune on
my rig datasets, can you queue one asap" — this is that run, queued
same-hour. It is also the first training step onto the §0 north star's
own terrain: adapting the best deployment-class checkpoint (the 1-NFE
SnapFlow student) to the owner's SO101 rig data, with the owner
planning physical rollouts (the 16:33Z checkpoint request + rollout
verification landed at `63b044e`).*

## Question

Does the 1-NFE student fine-tune onto the rig distribution — holding
its one-forward endpoint decode — without forgetting the community
distribution wholesale? Deliverable: a rig-adapted checkpoint the
owner can physically roll out (`bijou.rollout --target-time zero`),
plus the first measured rig-transfer numbers for the distill lineage.

## Arm (one run)

**`fontaine_flow_snapdistill_ftrig_4k_1xh100`** — local 1×H100.

Recipe = the student's own recorded train_args
(`fontaine_flow_snapdistill_h1024_30k_1xh100/step_030000`, itself
teacher-verbatim + the SnapFlow pre-reg deltas), with EXACTLY these
deltas:

| field | student | this run | why |
|---|---|---|---|
| `--train-data` | community + 2 rig repos | **the 2 rig repos only** (`so101_pick_place_v2`, `so101_pick_place_clean`) | the owner's ask: rig adaptation (mainline ft precedent: `bijou_flow_artrunk_ft_rig_4k_ddp4`) |
| `--init-from` | teacher @80k | **student `step_030000`** | keep the trained shortcut field; φ_s True→True, strict key load, fresh optimizer |
| `--steps` | 30,000 | **4,000** | mainline rig-ft rung; rig data is tiny |
| `--decoder-lr` | 2.5e-5 | **1e-5** | mainline ft LR; conservative on ~10² episodes |
| `--save-every` | 2,500 | **500** | short run, prune after reads |
| bookkeeping | — | run name / save-dir / wandb `fontaine` | |

Everything else verbatim from the student: `--distill snapflow`
(the consistency objective CONTINUES during adaptation — plain L_FM
would let the shortcut field drift while only the velocity field
adapts; this is the mechanism bet of the run), B24, grad-clip 1.0,
warmup 500, wd 1e-5, fps 30, cameras {1,2}, holdout 0.1/seed 0,
condition fields subgoal/outcome/smoothness, seed 0, eval-every 500
(in-run probe now draws from held-out **rig** episodes).

## Pre-launch reads (banked before the ft touches the GPU)

- **R0 (baseline, this session):** the un-tuned student @30k scored on
  the rig holdout (0.1/seed 0, both repos), 1-NFE euler-1, draws 1 +
  mean-of-10, stable keying, `--dump-predictions` npz — the paired
  "before" every after-read compares against, on identical frames.

## Frozen reads & decision rule (after 4k)

1. **Transfer read (primary):** rig-holdout 1-NFE eval @4k, identical
   frames/keying as R0, paired per-frame Δchunk/Δfirst vs R0.
   Expect improvement; magnitude genuinely open (first distill-lineage
   rig ft). Direction wrong ⇒ the mechanism bet failed — diagnose
   before any second rung.
2. **Forgetting read (guard):** community panel-v2 1-NFE (euler-1,
   draws 1, stable keying, seed 0) @4k vs the student's descriptive
   v2 column **5.6711/1.7059** (index-keyed npz derivation; σ_draw
   ~0.02 ≪ the bound below, keying noted). Regression expected;
   bound: **Δchunk ≤ +1.0** (beyond = catastrophic forgetting,
   flagged loudly, checkpoint still ships with the caveat — the rig
   is its deployment target, but the pre-reg records the cost).
3. **Deployment sanity:** `bijou.rollout --check` (sync + async
   dry-run) on the @4k checkpoint with rig stats — must pass before
   the checkpoint is offered.
4. **Ship rule:** (1) improves AND (3) passes ⇒ upload
   `step_004000` weights-only to `fontaine-checkpoints`, post the
   owner a rollout command; else post the diagnosis instead.

**Class:** rig reads are non-headline diagnostics (charter §2: the
~6-episode rig holdout is too small to target) — this run makes no
panel claim; it is north-star transfer work with the panel as the
forgetting guard. Caveats stated now: rig holdout ≈ 6 episodes
(coarse read, wide CIs — per-frame pairing is what makes it usable at
all); eff-24 vs mainline ft's eff-40; 480p rig cameras.

## Gates

- **E1 (banner, abort before step 1):** exactly 2 datasets; state
  dims 6/6; `distill: snapflow` + φ_s banner present; init-from loads
  strict with NO extension branch (the student already has φ_s keys).
- **E2 (first-poll):** ~0.45–0.65 s/step at B24 (SnapFlow measured
  0.48–0.52 on this GPU), VRAM within the SnapFlow envelope; util
  checked at first poll per the standing rule.
- **K1 (kill):** loss NaN, or in-run probe > (its own first read
  + 3.0) at any 3 consecutive evals ≥ 1.5k (catastrophic-only —
  the probe population changes to rig holdout, so no cross-run
  anchor exists; the probe's own trajectory is the reference).

## Cost

Train ~35–45 min (4k × ~0.5 s/step). Evals: R0 + transfer read on
the small rig holdout (~minutes each) + one panel-v2 1-NFE pass
(~30–40 min). All local; the arch batch on the box is untouched.
Disk: 8 checkpoints × ~11.5 GB transient, pruned to `step_004000`
after reads (uploads before deletions).
