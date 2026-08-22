# The flow head that trained from scratch: an `--offload-optim` ordering bug

*2026-08-22 11:0x–11:4xZ (work session). Root cause and fix for
`bijou-resume-flow-state-bug` (queued at the [leg B incident](2026-08-22-prereg-squint-twin-screen.md)
close 09:1xZ); commit `665dadb7`. Discord posts
`1540678713115418654` / `1540678747928395887` / `1540687090558177411`.*

**Plain words.** When we resumed a half-finished training run after a
disk-full crash, the model came back with amnesia in exactly one place:
the flow head — the part that turns the model's understanding into
continuous arm motions — restarted from zero while everything else
picked up where it left off. The saved file was fine; the weights were
all there. The culprit turned out to be a memory-saving optimizer
feature that keeps its own private CPU copy of every weight, taken
*too early* — before the saved weights were loaded into the model. On
the first training step it faithfully wrote its stale copy back over
the freshly loaded weights. The kicker: the same too-early copy also
fired on every *fresh* fine-tune that used this feature, so a whole
family of runs that were supposed to start from a pretrained flow head
actually trained theirs from scratch without saying so. Their results
stay valid — the models are what they are and were measured honestly —
but the recipe descriptions were wrong, and now we know why those
heads always started at fresh-init loss.

## The signature

Two recovery attempts (`r2` 08:40Z, `r3` 09:02Z) resumed the squint
adapt arm-1 checkpoint `step_000250` and both produced the same split
picture at the first logged step (260):

| quantity | continuous run @250 | poisoned resume @260 | fresh-init reference |
|---|---|---|---|
| `loss_action_flow` | **0.0946** | **1.4374** (both attempts) | 1.5555 @10 (adapt), 1.33 @10 (pdnorm) |
| `loss_action_ar` | 1.5916 | 1.3683 / 1.3684 | 4.11 @10 |
| probe `eval_chunk_mae` | 2.80@300 (r1) | 9.19@300 | — |

The AR side *continued*; the flow side sat at fresh-init level and
then relearned at fresh-head pace (1.44 → 0.69 by step 320). The two
attempts differed in gradient wiring (r2 accidentally resumed the KV
seam open — the `insulate_flow` passthrough sub-bug — r3 re-declared
it) yet logged the same 1.4374, so the reset predates any optimizer
dynamics.

## The hunt, in narrowing order

1. **The file is faithful.** Full 662-tensor diff of the saved
   `flow_decoder.safetensors` against its init source: 588 trainable
   tensors differ (trained state is in the file); the 74 identical
   ones are exactly the frozen compat tensors (`kv_proj` zeros,
   `state_encoder` identity) that never train by design. The earlier
   70-of-80-sampled-keys check had read the right file correctly.
2. **The load path is faithful.** Rebuilding the decoder exactly the
   resume way on CPU (`build_molmo_flow_decoder` →
   `load_expert_state`) leaves zero mismatched keys against the file,
   and the loaded decoder's forward is trained (mean |output| 0.154).
3. **A fresh decoder is *exactly* zero-output.** adaLN-Zero init
   zeroes the final projection and every modulation gate, so a fresh
   head predicts 0 for any input and the flow loss is
   E‖x−ε‖² ≈ 1.43 on synthetic targets — the observed 1.4374 to three
   decimals. The live model at resume was behaving like a *fresh*
   decoder, not a corrupted one.
4. **Ten AdamW steps at lr 2.75e-5 cannot zero a trained head.**
   Update magnitudes are ~1e-3 total; the head must already be fresh
   at the first resumed forward. So: the file is right, the load is
   right, the forward code is deterministic — something between the
   load and the first step *replaces the weights*.

## The mechanism

`--offload-optim`'s `CPUOffloadAdamW` steps on pinned fp32 **mirrors**
of every parameter, captured at construction
(`offload_optim.py:61`), and finishes every step by copying the
mirrors back into the live GPU params wholesale (`:127`). The train
CLI constructs the optimizer at `cli.py:1884` — **before**
`load_family_weights` at `:1978`.

- The **trunk** mounts from the checkpoint's part files at *model
  build*, before the optimizer exists → its mirrors capture trained
  state → AR continues seamlessly.
- The **flow decoder** is built fresh (adaLN-Zero) and its checkpoint
  weights load *after* the optimizer exists → its mirrors capture
  fresh init → the first step's write-back reverts the head.

Everything falls out: the AR/flow split, the exact fresh-init loss
level, the bitwise determinism across attempts (seeded init, same
seed), and the immunity of the documented resume gate (resume-vs-
resume comparisons reproduce the *same* deterministic reset on both
sides).

## Blast radius: not just resume

The same ordering fires under `--init-from`. Receipts, step-10
`loss_action_flow`:

| run | inherited flow head from | step-10 flow loss |
|---|---|---|
| pdnorm joint (onerig) | released MolmoAct2 expert | 1.3262 |
| pdnorm joint (democlean) | released MolmoAct2 expert | 1.3412 |
| squint adapt arm 1 | pdnorm@3000 | 1.5555 |

Fresh-init level, all of them: **every `--init-from` +
`--offload-optim` run trained its flow head from scratch.** The banked
panel numbers stay valid as measured — the artifacts are what they
are and the evals were honest — but every joint checkpoint in the
house carries a scratch-trained flow head, and the "inherit the
expert" recipe language was wrong. (An open follow-up thought: those
heads reached good probes in 3000 steps *from zero*; what an actually
inherited expert buys is now a measurable question, not an
assumption.)

The squint screen is internally consistent: all arms — attempt-1, r4
arm 1, r4 arm 2 — trained under identical (buggy) semantics, so the
paired comparisons compare like with like. The fix was deliberately
NOT landed until r4's arm 2 was past Python import, so the unit's two
arms stay paired.

## The fix (`665dadb7`)

1. **First-step mirror re-sync.** The invariant is *mirror == live
   param at step entry*, not at construction: `step()` re-captures
   the mirrors from the live params once, before its first write-back.
   Bitwise-neutral for fresh runs (nothing mutates params between
   construction and step 1 except post-construction checkpoint loads —
   the staleness being erased); the keystone bitwise and
   state-roundtrip oracles are unchanged. A regression test pins
   load-after-construct against a reference optimizer built after the
   load, bitwise from step 1 — red before the fix, green after.
2. **The `insulate_flow` passthrough sub-bug.** For
   `molmoact2_joint`, `--insulate-flow`/`--joint-ce-weight` are now
   REFUSED under `--resume`; the CLI reconstructs the objective from
   the checkpoint's recorded payload, the seam print reads the
   objective, and the save side records `model.objective` (a resumed
   save would otherwise have stamped `insulate_flow=False` into the
   next checkpoint's metadata). `molmoact2_flow` keeps its CLI
   declaration — its payload records no seam (test-pinned).

`check.py` PASSED (1112 tests). Remaining before the queue item
closes: a GPU verification leg
(`fontaine/scripts/verify_resume_fix.sh`) — resume the archived
attempt-1 `step_000250` with the fixed code and read the step-260 flow
loss: ~0.09–0.15 continues = verified; 1.44 = refuted. Runs at the
r4-exit window, timeboxed, before leg C claims the card. Until green,
the standing rule holds: no `--resume` on flow/joint lineages.

## Lessons

- **A snapshot taken at construction is a contract about ordering.**
  Any state captured from live objects must either be captured at the
  last possible moment or re-validated at first use. The fix moves the
  capture to first use.
- **Bitwise gates don't catch deterministic corruption.** The resume
  gate compared resume-vs-resume; both sides reset identically. Gates
  that matter compare against the *continuous* trajectory at the loss
  level.
- **"Weights are in the file" ≠ "weights are in the model".** The
  verification that closed the gap was a full-file diff plus a
  standalone rebuild of the exact load path — cheap on CPU, decisive.
