# 13. Sign-convention detection & repair (owner hypothesis) — `screening`

- **Hypothesis:** a small set of community repos encodes joint angles
  with flipped sign conventions (esp. wrist_roll on mirrored wrist-cam
  mounts); training on them injects contradictory supervision.
- **Status:** stage 1 (CPU screen over the panel npz) done 2026-08-05
  — 9 candidate (repo, dim) cells, three pathologies separated by
  per-frame classification; cleanest mirror lead
  kantine/domotic_dishTidyUp_anomaly wrist_flex (median frame corr
  −0.75). Instrument: `probes/probe_sign_convention_stage1.py`;
  [results post](../posts/2026-08-05-sign-convention-stage1.md).
- **Stage 2 PRE-REGISTERED (2026-08-05 ~23:3xZ,
  [pre-reg](../posts/2026-08-05-prereg-sign-stage2.md)):** optical-flow
  cross-check on the three mirror-signature cells (dishTidyUp_anomaly
  wrist_flex, groceriesSorting_expert wrist_roll, aractingi
  shoulder_lift), CPU-only (~20–40 min spare cores). Frozen: Farneback
  params, isolated-motion pair selection (|v_d| ≥ 0.5°/frame, 2×
  dominance), ego-cam identification rule (cams are unlabeled),
  15-repo so100 reference population with an 80% sign-consistency
  validity gate, MIRRORED/NORMAL/INCONCLUSIVE bootstrap rules,
  synthetic-flip hard validation gate before candidate cells open,
  Dongkkka + kevin510 as specificity controls, and the stream-
  consistency read (calibration-mirror vs action-only flip).
  Feasibility verified pre-post: all repos local, torchcodec decodes
  the AV1 videos, state+action parquet intact. Execution = a later
  work session; if ≥1 MIRRORED, the repair arm (flip-corrected
  derived corpus through #18.8 certs + paired screen) gets its own
  pre-reg.
- **Stage 2 EXECUTED 2026-08-05 ~23:5xZ — the escalation branch fired**
  ([results post](../posts/2026-08-05-sign-stage2-results.md), probe
  `probes/probe_sign_convention_stage2.py`): 3 of 4 reference
  populations FAILED the 80% sign-consistency gate (wrist_roll 9/15,
  wrist_flex 10/15, shoulder_lift 9/15; only shoulder_pan valid at
  13/15) ⇒ hard gate failed, **candidate cells never opened, no
  verdicts**. The t_x oracle PASSED end-to-end (mass 1.000 both
  directions) — the mechanism works where the population premise
  holds. Diagnosis: image-plane statistic signs follow *camera
  mounting* (cams sign-disagree in 11/15 shoulder_lift refs; ego-cam
  rule NO-MARGIN on ~half; ω underpowered off-wrist-cam) — not
  evidence that joint conventions vary corpus-wide. The three stage-1
  mirror cells remain unresolved leads; repair arm neither eligible
  nor dead. **Next (owner steer wanted): stage-2b amendment
  conditioning reference populations on `meta/camera_kinds.json`**
  (the 2026-08-02 VLM cam-labeling pass: wrist/front/side/top) — t_y
  from front cams, ω from wrist cams, label-gated ego rule; reuses
  the 38-repo flow cache, so it is cheap.
