# Baselines — lerobot-native environment

Independent uv project for running baseline VLAs (currently pi0.5) with
lerobot's own dependency pins. The main repo overrides
`transformers>=5.14` (Gemma-4 needs it) while lerobot 0.6 caps `<5.6`;
that skew already produced one inference breakage
(`create_causal_mask(cache_position=)`, removed in 5.x) which training
never hit — the next skew could be a silent numerical difference, which
a baseline cannot afford. This env resolves lerobot's declared matrix,
so results here are lerobot-native by construction.

Setup (laptop and box): `cd baselines && uv sync`. HF/wandb auth and the
model/dataset caches are shared with the main env (`~/.cache/huggingface`).

Scripts (hardware values match the owner's rig; box paths are absolute):

- `pi05_train_frozen.sh` — box. Frozen-trunk fine-tune
  (`--policy.train_expert_only=true`) of `lerobot/pi05_base` on
  `so101_pick_place_v2`.
- `pi05_server.sh` — box. Stock async policy server, localhost-only (it
  unpickles what it receives — reach it through an SSH tunnel, never a
  public bind).
- `pi05_client.sh` — laptop, arm + cameras attached. Robot cameras are
  named with pi0 slot names (`base_0_rgb`, `left_wrist_0_rgb`) because
  the async server resizes images by policy-feature key BEFORE the
  checkpoint's rename processor runs (lerobot 0.6
  `helpers.prepare_raw_observation`); training-time names would KeyError.
  Independent of the transformers version.
- `probe_pi05_infer.py` — box. Replays the exact server inference
  pipeline (raw obs → processors → `predict_action_chunk`) with a
  synthetic observation; debugs serving without robot round trips.

Dataset gotchas encoded in the scripts: `--dataset.root` must be
explicit (hub-download cache markers otherwise force a hub re-sync that
crashes on private repos) and `--tolerance_s=0.0167` (v3 concatenated
files break lerobot's 1e-4 default at fps 30).
