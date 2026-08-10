# The Curse of Precision: the demo bill diverges at a precision ceiling you can move but not predict — a sim-only fit whose scariest points are extrapolated

*Read 2026-08-10 (lit slice `lit-radar-0821`, priority 2). Paper:
[2607.23108](https://arxiv.org/abs/2607.23108) — "The Curse of
Precision: A Data Scaling Law for High-Precision Robotic
Manipulation" (Cuijie Xu, Yuanfan Xu, Min Xue, Jianjie Lin, Jian
Wang, Xudong Zhang, Yu Wang, Jincheng Yu; Dept. of Electronic
Engineering + Institute for Embodied Intelligence and Robotics,
Tsinghua University, and OpenMind (WuHu) Robotics Co.; arXiv cs.RO,
submitted 2026-07-25; arXiv nonexclusive distribution license;
accepted to ICRA 2026, 8 pages. Artifact status: **no code, no data,
no project page** — nothing linked in the paper body, footnotes, or
references, and a web sweep on 2026-08-10 finds only the arXiv page
and an ICRA poster PDF. The "experiments" are ~100 diffusion-policy
training runs in ManiSkill3; none of it is reproducible from a
release.)*

**The paper in plain words.** When you teach a robot by showing it
examples, some tasks forgive sloppiness — drop a block anywhere on a
plate — and some don't: a peg that must slide into a hole with a
millimeter to spare. This paper asks how the number of teaching
examples you need grows as the fit gets tighter. The answer is
brutal. It doesn't grow linearly, or even exponentially in any tame
sense: the required example count blows up toward infinity as the
tolerance approaches a hard wall. Their formula says the *logarithm*
of the example count grows like one over the distance to that wall —
so near the wall, doubling or tenfold-ing your data barely moves the
needle. The interesting twist is what the wall is made of. It is not
a property of the peg or the hole. When they removed the camera on
the robot's wrist, the wall moved to a looser tolerance — the system
got permanently worse, no data could fix it. When they swapped a
cautious demonstrator (who wiggled and corrected near the hole) for
a decisive one who went straight in — keeping only the successful
attempts, even though that demonstrator failed half the time — the
wall moved to a *tighter* tolerance. And when they made the scenario
less varied, the wall moved again. Cleaner examples and better
sensors buy you precision that no amount of extra data can. The
catch: this is all in simulation, on one robot, with one policy
architecture, across three tasks. The "law" is a curve fit — no
theory says it must be this shape — and the fit is a chain: the most
dramatic required-example counts at the tightest tolerances were
never measured, they were extrapolated from trends seen at 200–2,000
examples. And the wall's location cannot be computed ahead of time
from sensor spec sheets; you find it by running the robot, which we
cannot do until rig phase. What survives for us is a design
principle, not a number: tolerance is a dial, the ceiling is a
system property, and demo clarity beats demonstrator skill.

## What it contributes

- **Two stacked empirical laws.** First, at a fixed tolerance `P`,
  failure rate follows a power law in demo count:
  `ln(1−SR) = a·ln N + b` (Table I; `a` steepens as tolerance
  loosens — peg at 10 mm has `a = −0.72`, at 4 mm only
  `a = −0.19`, i.e. near the ceiling data hardly helps). Second,
  the headline law: the demo count needed to hit a target success
  rate satisfies `ln N = m/(P − c) + n` — super-exponential
  divergence as `P → c`. Both are fits, not derivations; the
  paper's own language is "we hypothesize this relationship is
  governed by" the form, and no competing functional forms are
  tested against it.
- **`c` as a system metric.** The ceiling `c` is fit per *system
  configuration*, shared across target success rates (grid search
  on `c` maximizing summed R² over the SR = 0.5/0.7/0.9 curves).
  The claim: `c` is "not a static physical constant of the task
  but an emergent property of the entire agent system, including
  its sensors and expert policy" — and, per their own third
  ablation, the task's randomization breadth too.
- **A cheap diagnostic protocol.** Since measuring `c` properly
  cost them ~100 runs × ~20 A100-hours, the practical pitch is the
  inverse: assume the form, fit `c` from a few tolerance levels,
  and use it two ways — predict the system's precision limit, and
  debug: evaluate the same task at relaxed tolerances; smooth
  degradation consistent with the law means you're at the
  system's intrinsic limit, erratic degradation means you have a
  bug.

## The experiments they actually ran

- **Setup:** Franka Panda in ManiSkill3 simulation, Diffusion
  Policy (ResNet-18 backbone, 1D U-Net denoiser, 100 DDPM steps),
  two 256×256 RGB-D cameras (third-person + wrist) plus
  proprioception, delta-EE-pose actions. Three tasks, tolerance =
  the precision axis: **peg insertion** (clearance 4–10 mm,
  scripted expert), **stack cuboid** (base half-side 4–10 mm,
  scripted expert), **roll ball** (target radius 35–200 mm,
  RL expert, state-based obs). N swept 200–2,000 demos per
  (task, tolerance) cell; ~100 training runs, tens of thousands of
  eval rollouts, SR from 100-episode evals with 95% Wilson
  intervals over 300 trials.
- **Fits:** precision law R² > 0.97 everywhere (Table II): peg
  `c = 2.35 mm`, stack `c = 2.75 mm`, roll ball `c = 20.3 mm`.
  Success-rate power law R² 0.84–0.99 (Table I).
- **The extrapolation caveat (my arithmetic, not their
  framing):** the precision-law data points are *required-N*
  values solved from the Table I power-law fits — and at the
  tightest tolerances those solutions sit far outside the tested
  range. Peg at 4 mm: `a = −0.19, b = 1.35` implies SR ≈ 0.09 at
  the largest tested N = 2,000, and reaching SR = 0.5 requires
  **N ≈ 47,000** — 23× beyond anything trained. Stack at 4 mm is
  worse (~65× beyond). So the "super-exponential blow-up" is
  anchored by measured points at loose tolerances and
  *extrapolated* points at tight ones. The R² > 0.97 is real but
  it is R² against partly synthetic targets.
- **Checkpoint selection is oracle-flavored:** they "report the
  mean of the top 3 highest SRs achieved during training" — peak
  selection on the eval metric. Fine for curve shape, inflates
  absolute SR.
- **The ablations (peg insertion only, refit `c` each time,
  Table III):** baseline `c = 2.35 mm`; **aggressive expert**
  (direct single-shot insertion, expert's own SR ~50% at 5 mm vs
  the conservative expert's ~98%, only successful trajectories
  kept) → `c = 1.27 mm`; **no wrist camera** → `c = 3.85 mm`;
  **low randomization** (only initial XY varied) → `c = 1.00 mm`.
  Their reading of the expert ablation: "the clarity and lack of
  ambiguity in demonstrations can be a more critical factor for
  achieving high precision than the expert's own raw success
  rate" — the conservative expert's corrective wiggles at the hole
  entrance create observational ambiguity BC can't resolve without
  history.
- **Stated limitations:** simulation only ("validating these laws
  on physical hardware is a crucial next step"); BC only (no
  RL/DAgger); three tasks; model capacity is a prerequisite (roll
  ball needed the high-capacity U-Net); full validation is
  data-intensive, diagnostic use requires assuming the form. Future
  work explicitly names "algorithmic data curation" to clean
  "large datasets of imperfect but plentiful demonstrations."

## What transfers to us — and what doesn't

- **`c` is not computable from our spec sheets.** The paper offers
  no decomposition of `c` into sensor noise, servo repeatability,
  or expert jitter — it is measured by fitting rollout sweeps, and
  the ablations only show *which knobs move it*, not by how much a
  priori. So the direct answer to "can we predict the SO-101
  ceiling from teleop sensor noise / servo backlash ahead of
  time" is **no**. STS3215 backlash and positions-only 30 fps
  logging put our plausible `c` in the multi-millimeter class, but
  that's physics intuition, not this paper.
- **Rollouts are the entry fee.** Every point on every curve is a
  real (simulated) rollout success rate. Our offline chunk-MAE
  panel cannot substitute: the law lives in binary
  success-vs-tolerance space, and nothing in the paper maps an
  action-MAE floor to `c`. Same verdict as the rollout-free-eval
  page — this whole instrument is rig-phase. (The one speculative
  bridge: a persistent chunk-MAE floor at ~ε in joint space bounds
  achievable tolerance from below; untested by anyone.)
- **The task-randomization result cuts both ways for us.**
  Narrowing the distribution moved `c` from 2.35 → 1.00 mm. Our
  community corpus is maximally broad (many rigs, operators,
  scenes); a narrow rig-collected fine-tune set is exactly the
  low-randomization regime. That is an argument that modest
  rig-day data can reach precision the 229 h corpus never will —
  consistent with the H2R-page finding that diverse pretraining +
  narrow adaptation is the winning stack.
- **Demo clarity is a curation axis we can act on now.** The
  aggressive-vs-conservative expert result is the paper's most
  transferable finding: hesitant, corrective, multi-retry
  demonstrations *raise* the ceiling even when they succeed more
  often. Community teleop is full of exactly that. Detecting
  retry/jitter signatures in `action`/`observation.state` traces
  is zero-GPU and corpus-feasible.

## Hook corrections

Banked hook: *"demos needed grow super-exponentially with target
precision, log N ∝ 1/(P−c); the ceiling c is a property of the
sensor+expert system, not the task — bounds what demo-scaling buys
on hobby-arm precision tasks; feeds #9/#16 bench design."*

1. **Functional form: as claimed, but it's a fit, not a law.**
   `log N ∝ 1/(P−c)` is exactly the paper's model, R² > 0.97 — but
   no derivation exists, no alternative forms were compared, and
   the tight-tolerance required-N points are extrapolations of the
   underlying power-law fits (peg@4 mm needs N ≈ 47k vs 2,000 max
   trained). Treat the *shape* as plausible and the constants as
   soft.
2. **"Not the task" is wrong as stated — loudly.** The paper's own
   third ablation moves `c` from 2.35 → 1.00 mm by *only reducing
   task randomization*. `c` is a property of the whole tuple
   (sensors, expert, policy capacity, **and task distribution**).
   The corrected slogan: `c` is not a physical constant of the
   *object tolerances*; everything else in the system, task
   breadth included, is inside it.
3. **"Bounds what demo-scaling buys on hobby-arm precision
   tasks" — directionally yes, quantitatively unearned.** Sim
   only, Franka only, diffusion policy only, scripted/RL experts
   only. No hardware `c` exists anywhere in the paper, and `c`
   cannot be forecast for our servo class without running the
   sweep. The bound is a design principle today, a number only
   after rig-phase rollouts.

## What it feeds

- **Idea #16 (few-shot rig-transfer bench) — three concrete design
  rules.** (1) *Tolerance is the placement dial*: build precision
  tasks as one task at 2–3 tolerance levels (e.g. insertion with
  ~10/7/4 mm clearance sleeves) instead of distinct tasks — this
  is the mechanism behind the already-banked "tasks in the 20–80%
  success band" rule, and multi-tolerance cells let the band be
  hit by re-sleeving rather than re-designing. (2) *Fit `c` as a
  headline rig metric*: with ≥50 trials/cell (their 300-trial
  Wilson protocol is the same family as our banked rule), the
  degradation curve across tolerance levels yields a single
  system-capability number independent of target SR — and doubles
  as their debug test: erratic degradation vs tolerance = bug,
  smooth = at system limit. (3) *Config deltas measured as
  Δc, not ΔSR at one tolerance*: wrist-cam on/off moved `c` by
  1.5 mm; a single-tolerance SR comparison would alias that as
  task-dependent noise.
- **Idea #9 (data levers) — a bound and a curation lever.** The
  bound: for any task whose tolerance sits near the
  corpus-system's `c`, more community hours buy ~nothing
  (`a = −0.19` at the tight end: 10× data ≈ halving failure odds
  slowly); prioritize levers that move `c` (wrist-cam-present
  episode filtering, cleaner demos) over raw volume for precision
  tasks. The lever: **clarity-filtering** — down-weight or drop
  episodes with retry/correction signatures (direction reversals
  near contact, dwell-and-jiggle patterns in `action` traces) for
  precision-task training mixes, per the c = 1.27 vs 2.35 expert
  result. This is the same slot as the queued Quality-over-Quantity
  influence-function read (2603.09056) — read that next with this
  page's "clarity beats expert SR" prior in hand, and note this
  paper's future-work section independently calls for exactly this
  curation.
- **No new triage ids.** The reference list is pre-cutoff heavy
  (Lin et al. 2410.18647 data-scaling, JUICER, Octo, ManiSkill3);
  nothing post-cutoff surfaced worth adding to the sweep.
