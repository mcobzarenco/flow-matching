# Pre-registration: parallel sim rollouts — N env workers, one batched policy

*Registered 2026-08-12 10:1xZ (work session; real `date -u` at
write: 10:10). Owner-approved 08:44Z, owner-sequenced 09:32Z: "Once
I relinquish the GPU, remember to do sim-parallel-rollouts before
any other experiments." The CPU scaffold landed this session
(commit `1e4e16f`, check.py 710 green); this note registers the GPU
leg that runs FIRST on release — an infrastructure oracle, not a
behavioral experiment.*

## Plain words

Evaluating a robot policy in our simulator is slow because we run
one episode at a time: the big GPU does a tiny bit of thinking per
step and then sits idle while the simulator draws the next pictures.
The fix is standard: run many simulator copies at once in separate
worker processes and have them share the one policy, which thinks
about all of their pictures in a single batch. A 100-episode
evaluation should drop from ~1.5 hours to ~20 minutes. The catch:
our evaluations are only trustworthy because every episode is
exactly reproducible. So before this faster path is allowed to
produce any registered number, it must prove — on real episodes —
that it gives *bit-for-bit the same rows* as the slow path it
replaces. That proof is what this note registers.

## What landed (CPU, commit `1e4e16f`)

- `sim/rollout_sim_parallel.py`: spawn worker processes each own a
  full `SO101Sim` (own EGL context — MuJoCo's EGL display is
  per-process global state); the parent holds the single checkpoint
  copy and serves batched `policy.predict` calls.
- **Deterministic lockstep rounds**: each round collects exactly one
  predict request per still-active worker in worker-index order,
  answers all with one batched forward. Batch membership is a pure
  function of (seed partition, worker count, policy outputs) — never
  of wall-clock timing. Seeds partition round-robin
  (`seeds[w::N]`).
- **Noise is untouched by construction**: rows carry the sequential
  driver's identity triple (`repo_id="sim/eval100"`,
  `episode_index=seed`, `frame_index=replan`), and stable-key flow
  noise is invariant to batch composition (oracle-pinned in
  `tests/test_stable_noise.py`).
- Both drivers now share one episode loop (`run_episode_loop`), and
  5 CPU-tier oracles (`tests/test_sim_parallel_rollouts.py`) pin
  harness equivalence with a fake sim whose dynamics depend on the
  commanded actions: rows bit-equal to the sequential loop's (minus
  `latency_ms`), per-seed predict counts, lockstep batch trace,
  hold-arm worker-local path, error propagation.

## The open question the GPU smoke answers

Batching N observations changes the GEMM shapes inside the decode;
floating-point reduction order may move with them, and heun-10
feeds any last-bit drift back through chaotic contact physics —
so batched-vs-batch-1 bit-identity is an empirical question, not a
provable one. It is exactly what
`fontaine/scripts/sim_parallel_oracle.py` measures.

## Registered protocol (on GPU release, before any other experiment)

- **Arm**: `er60k` (`er_60k/step_060000`, heun-10, bf16, policy
  seed 0) — the banked reference decode. Seeds 0–5, replans 30 ×
  horizon 30, `post_backend` auto (torch compositor) on both paths.
- **Run 1 (oracle at 2 workers)**: `sim_parallel_oracle.py
  --num-seeds 6 --workers 2` — sequential driver, then parallel,
  same seeds; compare every row field except `latency_ms`.
- **Run 2 (oracle at the target worker count)**: same at
  `--workers 8`. GREEN is only transferable to a registered eval run
  at the SAME worker count and decode settings; re-run the oracle
  when either changes.
- **Decision rule (frozen)**:
  - **GREEN both runs** (all fields bit-identical): the parallel
    path may substitute for the sequential driver in registered
    evals at the validated settings. The sim100 rerun (still
    owner-hold) would use it.
  - **FAIL**: the parallel path produces NO registered numbers as-is.
    Fallback (registered now): it may be used for arm-vs-arm
    *paired* comparisons only if every arm AND every baseline row in
    that comparison is regenerated under the identical parallel
    schedule (same worker count — no mixing with banked sequential
    rows), via a per-use amendment reporting the observed divergence
    (the oracle prints per-field max |diff|). Diagnosis knob:
    re-run at `--post-backend numpy` to separate compositor-context
    effects from decode-batching effects.
- **Throughput read (record-only, only if GREEN)**: one 20-seed
  er60k arm at 8 workers — prices the rerun afternoon (target from
  the queue item: 100-seed arm in ~20–30 min vs ~1.5 h sequential;
  spot20 measured ~5.4 min/episode render-bound at 3-process
  contention, ~94 ms/tick compositor solo).
- **Budget**: worker VRAM ≈ 0.5–1 GiB CUDA context each at
  `post_backend` auto (fine next to the policy on 80 GiB); 26 CPU
  cores bound the useful worker count well above 8. **Gate ≤ 1
  GPU-h** for oracle runs + throughput read combined.
- Launch mechanics: `run_detached.sh` unit + babysit registry entry
  at launch, per the charter; results post same session.

No behavioral claims ride on this. Its deliverable is a yes/no
gate fact (may the fast path produce registered numbers?) plus a
measured throughput number that prices every sim eval after it.
