# Plan: adaRMS time conditioning for the flow expert

Status: **design, not implemented** (2026-07-28). Add an alternative way
to condition the action expert on flow time τ — DiT-style adaptive
RMSNorm modulation (adaLN-Zero, adapted to our RMSNorm sandwich blocks),
identity at init — selectable against the current input-additive scheme,
which stays the default and byte-identical.

## Why (Phase-0 evidence, handoff §2)

The per-τ / step-count diagnostic measured a real, in-domain integration
gap the current conditioning leaves on the table: Heun-5→30 = −0.99 MAE
(community pretrain) / −1.13 (rig ft), i.e. ~1 MAE of recoverable error
because the velocity field is rough in τ; and fine-tuning ROUGHENS mid-τ
transport (vel-MSE bump at τ≈0.2–0.5 on both sides). The marginal field
E[ε−a | x_τ, τ] changes character along τ (denoise-from-mostly-noise at
τ→1 vs refine-from-mostly-signal at τ→0); a single input-additive time
vector must carry that through 16 residual blocks, whereas per-layer
modulation lets each block reweight its own computation by τ — exactly
the DiT finding. **Pre-registered success signature**: adaRMS shrinks the
Heun-5→30 gap and the mid-τ ft bump; in-dist/holdout chunk-MAE moves
−0.1..−0.4; rig zero-shot unchanged (that's grounding, not conditioning).
Confounded by params (+~37%); attribution is a follow-up (a bottleneck
variant), not this plan's concern — owner chose DiT-faithful.

## Current conditioning (what stays as the default)

`ActionExpert.forward` builds a per-sample time vector and ADDS it to the
action tokens at the input:

    time_embeds = sinusoidal_time_embedding(time, time_embed_dim)   # [B, 256]
    time_embeds = time_out_proj(time_act(time_in_proj(time_embeds)))  # [B, hidden]
    action_embeds = action_in_proj(noisy_actions) + time_embeds[:, None, :]

The state token gets no time; each `ExpertLayer` is a Gemma sandwich —
per sublayer (cross-attn, self-attn, MLP): `residual; h = pre_ln(h);
h = sublayer(h); h = post_ln(h); h = residual + h`.

## The adaRMS variant

Reuse the same time MLP to produce a **conditioning vector** c = [B,
hidden]. Each layer owns a zero-initialized modulation head
`Sequential(SiLU, Linear(hidden → 9·hidden))` producing, per sublayer,
`(scale, shift, gate)`; the block becomes:

    residual = h
    h = pre_ln(h) * (1 + scale) + shift        # modulate the pre-norm output
    h = sublayer(h)
    h = post_ln(h)
    h = residual + gate * h                     # gate before the residual add

(scale/shift/gate broadcast [B, hidden] → [B, seq, hidden]; applied to all
suffix positions, so unlike additive the state token IS τ-modulated —
DiT-standard, and desirable: the state representation the actions attend
can be τ-dependent.) Optionally the final `self.norm` gets a scale+shift
head too (no gate — no residual there), DiT-faithful; zero-init keeps it
identity. τ enters ONLY through modulation in this mode: NOT added to
`action_embeds`.

Design choices, all as discussed (DiT-faithful, param-matching set aside):
- **All three sublayers modulated**, cross-attention included: τ-dependent
  prefix retrieval (coarse at high noise, fine near τ=0) is the most
  interesting capability this unlocks.
- **Keep the shift β** (not scale-only): with a scalar condition, β is
  what lets each layer inject a τ-dependent direction — it subsumes
  today's input-add at every depth rather than only layer 0.
- **Per-layer full heads** (9·hidden out): ~16·1024·9216 ≈ **+151M**
  (~+37%). Compute is elementwise + small matmuls; the expert is ~20% of
  step time, so a few % on total step time.

### Identity at init (required)

Zero-init each modulation head's weight AND bias ⇒ scale=shift=gate=0 ⇒
`modulate(x,0,0)=x` and `gate·sublayer=0` ⇒ every block is the exact
identity ⇒ the residual stream passes through untouched ⇒ `norm` →
`action_out_proj` (already zero-init) → velocity 0. So adaRMS at init is
both body-identity AND zero-field.

Gradient bootstrap: zero out_proj already blocks body gradients at step 1
(same as a fresh additive expert — see the unfreeze/AR discussions); zero
gates add one more ramp step (out_proj opens at step 1 → gates at step 2
→ sublayer weights at step 3). DiT zero-inits both the final linear and
the gates and trains fine; benign over 40k steps. Noted, not fixed.

## Config / CLI threading

- `TimeConditioning(StrEnum)` in `expert.py` next to `SelfAttentionMode`:
  `ADDITIVE = "additive"`, `ADARMS = "adarms"`.
- `ExpertConfig` gains `time_conditioning: TimeConditioning` (no default —
  the frozen config spells every field; the factory supplies the default).
- `ExpertLayer.__init__`: build `self.modulation` (zero-init) only when
  ADARMS, else `None`. `forward(..., condition: Tensor | None)`: when
  `self.modulation is None`, run today's path UNCHANGED (byte-identical);
  else apply the modulated path. `ActionExpert.forward`: compute c once;
  skip the input-add in ADARMS; pass c (or None) to each layer; optional
  final-norm modulation head.
- `loading.default_expert_config(..., time_conditioning=TimeConditioning.
  ADDITIVE)` — behavior default is fine here (factory, like the other
  knobs); thread into the `ExpertConfig(...)` construction.
- `CheckpointTrainArgs`: add `time_conditioning`; `from_dict` defaults to
  ADDITIVE (`TimeConditioning(data.get("time_conditioning", "additive"))`)
  so every existing checkpoint rebuilds as additive. `expert_config_
  from_train_args` passes it through.
- `train.py`: `--time-conditioning {additive,adarms}` (default additive);
  `TrainArgs` field; recorded in train_args → checkpoint.

## Backward compatibility (hard requirement — cont45k/mainline init-from)

Adding a field to `ExpertConfig` makes `dataclasses.asdict` include it,
which would break `ensure_matching_expert_config` for `--init-from` an OLD
checkpoint (its serialized `expert_config` lacks the key ⇒ dict mismatch).
Fix: before diffing, backfill new-since-v1 defaults into the SAVED dict —
`saved.setdefault("time_conditioning", "additive")`. Then init-from
cont45k (additive) into an additive run matches exactly, and init-from an
additive checkpoint into an ADARMS run correctly FAILS (different
architecture — adaRMS is always from-scratch or from another adarms
checkpoint; warm-starting additive→adarms is useless anyway, since zero
gates would ignore the loaded body). `from_checkpoint` already rebuilds
from train_args, so old checkpoints load unchanged.

## Validation ladder

1. **Oracle unchanged**: default (additive) 2-step tiny CPU run must
   reproduce 1.8896 / 1.7237 EXACTLY (additive path untouched). Establish
   and record a NEW adaRMS oracle (fresh 2-step tiny run) loudly.
2. **Identity-at-init unit test** (tiny, CPU): with a RANDOM-init
   action_out_proj, a fresh adaRMS expert's output equals
   `action_out_proj(norm(input_hidden))` — i.e. the body is the exact
   identity — and per-layer modulation outputs are all 0. (Additive has
   no such property; this is the adaRMS-specific guarantee.)
3. `check.py` (ruff/pyright/pytest) green; @override on the changed
   forwards; MaskSpec/config typing intact.
4. Matched A/B on a box: control (additive) vs adarms, same init recipe,
   40k, holdout 0.1/seed 0, same harness as the 4-arm round. Score the 3
   sides + the Phase-0 per-τ/step-count probe (the mechanism check).

## Risks

- Param confound (+37%): a win is "modulation OR capacity" until a
  bottleneck-variant follow-up; note it in results, don't block on it.
- adaRMS never warm-starts from additive checkpoints (guard enforces);
  it's a from-scratch arm, so it can't ride cont45k — budget a full
  pretrain, or accept it only proves out at ablation scale first.
- Forward-compat with the AR plan (`docs/plan_ar_fast.md`): once the AR
  suffix `[s_ar][t_*]` shares the expert (Option A), τ-modulation must be
  MASKED to the flow positions (`[s_flow][f_*]`) — the broadcast here
  assumes a flow-only suffix. Per-position gating, not per-sample
  broadcast, when that lands. Independent until then.

## Order of work

1. Enum + ExpertConfig field + `default_expert_config`/train-args
   threading + the `ensure_matching_expert_config` backfill (small;
   default-additive oracle must stay exact after this step alone).
2. `ExpertLayer`/`ActionExpert` adaRMS forward + zero-init heads +
   optional final-norm head.
3. CLI flag + checkpoint round-trip (incl. an OLD checkpoint) + oracles +
   identity-at-init test.
4. A/B on a box after the current queue (unfreeze A/B, AR work) clears.
