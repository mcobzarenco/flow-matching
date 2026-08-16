# Working together

Mantra: **run the collaboration like a lab.** Pre-registered
expectations, measured claims, loud failures, reversible actions — and
the owner's running systems are part of the experiment, never the
agent's playground. Companion to `docs/code-styleguide.md` (code
conventions); this file is the *operating* conventions. Living document:
rules get added with the incident that taught them, so future sessions
inherit the scar tissue without the scars.

## Decisions

- **Design discussion before architecture code.** Interfaces, config
  schemas, training schemes get a written proposal (chat or `docs/`)
  and explicit agreement before implementation. Plans are subsumed into
  `docs/architecture.md` when built; docs always read as-of-now.
- **When the owner delegates a call ("up to you"), make it** — state
  the decision, the evidence threshold that would flip it, and act.
  Pre-committed rules beat vibes: "unless the 1.5k/2k evals turn
  sharply down, kill and restart at decoder 1e-4 / text 2.5e-5" turned
  a diverging-run debate into a mechanical check.
- **Challenge the owner's numbers with evidence, not deference** — and
  accept being challenged back. The owner's 6e-4/3e-4 fine-tune LRs
  destroyed a warm init within 1k steps (probe MAE 8.34 → 15.95, worse
  than from-scratch at matched steps); the counter-proposal carried
  precedents (native decoder LR, the validated unfreeze grid) and was
  adopted. Technical correctness over affirmation, receipts attached.
- **Speculative questions get a reasoned prediction AND a cheap
  falsification plan.** "I expect X because Y; here is the 20-minute
  probe that would prove me wrong."

## Measurement

- **Before/after numbers, with how they were measured.** "Should be
  fine" is banned. Estimates are labeled as estimates — and estimates
  are not budgets (a corpus BPE fit estimated at ~30 GB RSS measured
  132 GB and OOM-killed a co-located box; the fit had only ever run on
  an empty machine, so the co-location assumption had zero evidence
  behind it).
- **Bitwise oracles after any math-adjacent change**; re-baseline
  loudly when an oracle legitimately moves. Signature-only refactors
  say so instead of re-running them.
- **Long jobs carry pre-registered expectations in their launcher
  headers**, checked before the artifact is consumed (the tokenizer-v2
  fit shipped with "alphabet ~150, merges ~870, tokens/chunk 25–30,
  recon ~0.46" — all verified before any training run touched it).
- **Artifacts carry the numbers that would catch their own failure.**
  tokenizer v1's degenerate 53.3 tokens/chunk was *recorded* in its fit
  report and compared against nothing; now the fit prints compression
  vs expectation and hard-errors when the merge budget is eaten.
  A recorded-but-uncompared number is a tripwire nobody armed.
- **Paired experiments change one variable** (v1-vs-v2 tokenizer arms:
  same seed, data order, arch, schedule; only the artifact differs) —
  and when a result has a confound (params, objective, trunk), the
  caveat ships *with* the win, in the same message.
- **Know the instrument.** The 256-frame probe has a ±0.3 noise floor;
  a 5–6-episode rig holdout distinguishes arms only coarsely; the
  effective sample unit is episodes (even scenes), not frames — a
  train/holdout gap can open at 13% of a "frame epoch". Numbers are
  only comparable within identical settings (fps filter, split seed,
  frame set); token-level metrics never cross tokenizer versions.
- **When two instruments measure the same thing, check they agree**
  before trusting either (in-run probe 5.954 vs offline eval 5.956 —
  agreement to the third digit is what makes the cheap probe usable).

## Code changes

- `check.py` gates every commit; **trust only its final verdict line**
  — a buffering artifact once masked 3 pyright errors in piped output.
  The general rule: long tools print a terminal verdict; success is
  never inferred from the absence of visible errors in a pipe.
- **Probes rot silently.** Gitignored `outputs/` probes sit outside
  check.py's blast radius; after a refactor touches their imports,
  their last-known verdict is stale knowledge — re-run before citing
  (the unfreeze grad-flow probe had rotted three separate ways while
  "flags-on validated" was still being repeated).
- Commit and push freely to `main` with detailed messages (rebase if
  the remote moved). No meta-provenance in comments or docs ("owner
  decision", "as discussed") — the git log is the provenance.
- **Never sync box code under a live run** (spawned dataloader workers
  import whatever is on disk); concurrent work on a busy box gets its
  own clone. Never restart an in-flight run on new code mid-experiment.

## Long jobs and remote boxes

- Long jobs run in tmux via launcher scripts scp'd to `~` (heredocs
  over ssh break; `$(...)` is forbidden in local tool calls), with
  local copies in gitignored `outputs/`. The header documents intent,
  settings, expectations, and the follow-up commands.
- **No shell-substitution characters in generated one-liners — an
  UNCLOSED one hangs the terminal forever, silently.** Backticks,
  `$(...)`, `${VAR}` are banned outright; the deadly form is a stray
  backtick or quote inside a long pasted string (a grep pattern
  quoting markdown code spans, a multi-paragraph commit message): the
  shell sits at a continuation prompt awaiting the closing character,
  which reads as a hung command until someone checks htop. Literal
  searches use single-quoted `grep -F` (or the editor's search
  tools); multi-step text surgery goes through `python3 - <<'PYEOF'`
  heredocs; long commit messages avoid backticks. **Sub-agent briefs
  must carry this warning verbatim** — agents do not inherit the
  operator's scar tissue. (Added 2026-08-16: two sub-agents hung in
  one session, 45 min and ∼10 min, both at long-text steps — one grep
  pattern, one commit message.)
- **Liveness is `pgrep` (with the `[b]racket` trick — `pgrep -f`
  self-matches the ssh command carrying the pattern) or GPU memory,
  never a log tail**: an OOM-killed fit read as "still running" from
  its tail for two minutes; every fit/run poll now includes a process
  check.
- **Kills wait for save boundaries**: the eval line, the `saved …`
  line, and a training line past them, before any signal is sent.
- **Restarts need a training-semantics reason**: data, objective,
  batch/LR, or a format the checkpoints inherit — never eval-side
  cosmetics (metrics, tables, logging), which deploy at the next
  natural boundary instead. (2026-08-03: a 40-minute-old 100k run was
  killed to pick up a wandb-table column layout; the owner's call was
  that it did not warrant the restart.)
- **Machine deletion gets an explicit inventory** — what is preserved,
  where, verified — before the all-clear.
- `MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072` on every
  training/fit process. Host crashes corrupt HF dataset arrow caches
  (zero-byte `dataset_info.json` → delete the cache dirs).
- In-flight host RAM for dataloaders = workers × prefetch × batch;
  scale prefetch inversely with batch (64/8 and 128/4 proven).

## Launching a training run (checklist)

Run through this before every launch; each line is a past incident.

1. **Box code at the launch commit, check.py green, no live run**
   (sync-under-a-live-run corrupts workers; concurrent work gets its
   own clone).
2. **GPUs clear + no stale tmux session** — the launcher guards both;
   after a pkill, GPU memory takes ~20 s to drain before the guard
   passes.
3. **Launcher header pre-registers**: expectations with numbers, gates,
   known seams (batch/world-size changes, format changes, LR restarts),
   and the failure read that would stop the run.
4. **Resume semantics** — three separate traps:
   - `--resume` = continuation (Adam moments + scheduler + step;
     CLI LRs ignored); `--init-from` = warm start (fresh optimizer,
     step 0, new save-dir). Extensions re-raising LR off the cosine
     floor take `--rewarmup-steps` (ramp anchors at the resume step;
     `--warmup-steps` anchors at 0 and also sets the cosine phase).
   - **A resume with the unchanged `--seed` REPLAYS the original data
     order**: the train loop restarts at epoch 0, index 0 — it never
     fast-forwards to the resumed step — and one epoch of the curated
     corpus is ~97k steps at eff-192, so extension segments replay the
     original run's frames AND its τ/noise draws exactly. Pass a fresh
     `--seed` on every extension until sampler fast-forward exists.
     (Caught by the owner 2026-08-05, 3k steps into a replayed
     extension; the rcond B10 resume also replayed its first stretch,
     unknowingly.)
   - `--steps` counts TOTAL including resumed steps; "nothing to do"
     means it was left at the old total.
5. **Effective batch bookkeeping** when world size changes: per-rank
   batch × ranks, stated in the header (96×2 → 48×4 kept eff-192; the
   B12→B10 OOM resume changed it and is a recorded seam).
6. **Disk estimate**: checkpoint size × (steps / save-every) against
   `df` (24 GB × 20 saves fit; it was checked, not assumed).
7. **First-poll verification, not vibes**: the `resumed optimizer …
   (lr …)` line, the model line (param counts, LRs, frozen/live), the
   stage-2 init line when inheriting a trunk, data-selection counts
   against expectation, and — after an LR-schedule change — one
   closed-form LR check against a logged step.
8. **wandb naming**: resumes keep the display name (new attempt id);
   reports pin run IDs, never display names — a resumed attempt
   shadows its predecessor under name queries.

## Babysitting runs

- Poll every 30–60 min when stable; tighter around the first eval
  boundary and memory spikes. The owner interrupts long sleeps — poll
  in short increments, and **on interruption: ask, never silently
  retry**.
- Every poll: liveness, latest step, curve vs anchors, anomaly scan.
  Curves are reported **against anchors** (copy baseline, prior arms at
  matched steps), not as bare numbers.
- A rising loss gets a mechanism diagnosis (feature drift? warmup?
  divergence? — with matched-step comparisons), not reassurance.

## Ownership boundaries

- **The owner's transfers, sessions, and runs are theirs**: flag
  problems and propose alternatives; do not kill, clean, or "optimize"
  them unilaterally. Destructive cleanup of interrupted work needs
  sign-off. (2026-07-31: an owner-queued rsync at ~60% was killed and
  its partial `rm -rf`'d in favor of a genuinely faster route — the
  route was right, the unilateral deletion was not; the transfer
  restarted from zero.)
- `--partial` on every long transfer *from the start*, so interruption
  stays cheap for whoever decides.
- Check GPU occupancy before using the laptop GPU (the owner records
  and runs rollouts there).

## Mistakes

- **Own them with the damage stated plainly**, then the remedy — no
  over-apology, no burying the cost in the middle of a paragraph.
- **Fix the class, not the instance**: a bug earns a guard and a test
  that make the whole class unrepresentable (the eaten-merge-budget
  guard; the checkpoint backbone-invariant test), plus a loud line that
  would have caught it at creation time.
- **Two independent occurrences of the same misconception mean the
  invariant belongs in code**: the checkpoint bug's "frozen ⇒ pristine"
  conflation was also the owner's mental model when choosing which
  checkpoint to download — conventions that two people independently
  get wrong are not conventions, they are traps.

## Artifacts

- **Versioned immutably**: changed semantics = new name
  (`fast_tokenizer_v1` stays published; the refit is `_v2`). Models and
  artifact versions are permanently coupled; never mix.
- **Checkpoint directories are self-contained**: loading one must
  reconstruct the trained model with no reference to any other
  directory (both backbone part files + tokenizer/ always present;
  `backbone.text_trained`/`vision_trained` are the explicit facts —
  presence is never a signal; frozen parts hard-link their parent's
  files, so `rsync -H` on local transfers).
- Models record what they trained with (tokenizer ref, per-dataset
  stats); artifacts carry their own fit/fidelity reports.
- Uploads before deletions. `optimizer.pt` is kept for checkpoints that
seed future runs and pruned for the rest.
- **Eval reports are named after the checkpoint they scored**:
`reports/eval__<run_name>__step_<N>__<variant>.{html,json,log}` —
run-dir name and zero-padded step verbatim from the checkpoint path,
`<variant>` for the eval config (frame count, frozen frame-plan id,
`heun30`, …). The report records its provenance internally, but the
filename is what survives an `ls`, an scp, and a six-week-old
download folder. (Added 2026-08-05 after `eval_flow40k_1024.html`
needed the session log to disambiguate which flow-40k it scored;
legacy `report_*`/short-named files keep their names — the ledger
prose cites them, and renaming history would detach those citations.)
