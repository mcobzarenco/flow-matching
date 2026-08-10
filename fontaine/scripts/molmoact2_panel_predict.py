"""MolmoAct2-SO100_101 out-of-band panel predictor — record-only.

Owner steering 2026-08-10 10:50Z/11:06Z + GO 11:59Z; pre-reg
posts/2026-08-10-prereg-molmoact2-oob-panel.md (plan sketch:
posts/2026-08-10-molmoact2-oob-eval-plan.md). Runs AllenAI's released
``allenai/MolmoAct2-SO100_101`` checkpoint end-to-end (their processor,
prompt, q01/q99 normalization, 10-step Euler flow expert — nothing of
ours in the model path) over the EXACT banked ``panel_curated_v0_k4l2``
rows and writes our standard eval npz contract:

  * identity columns (``index/repo_id/episode_index/frame_index/truth/
    valid/core``) + ``pred:state-copy``/``pred:state-copy-norm`` copied
    VERBATIM from the banked reference npz;
  * ``pred:molmoact2-so100@release`` (N, 50, 6) float32 — steps 0..29
    filled (their native 30-step / 1.0 s horizon), steps 30..49 NaN.

Execution oracles (hard abort): dataset row must match the banked npz
identity (repo/episode/frame) AND its raw action chunk must reproduce
the banked truth (atol 1e-6) — the same alignment oracle as
frame_mining.py; ``config.n_obs_steps == 1`` at load (HF-config default
of 30 would silently shift chunk slicing); prediction shape (30, 6).

Determinism: the flow expert's initial noise uses a fresh generator
seeded ``BASE_SEED + global concat index`` per frame, so any subset run
(smoke) is byte-reproducible inside the full sweep and row order never
matters.

Smoke mode (``--limit N``): evenly strided rows over the full panel
(deterministic), then a scale-sanity block per the pre-reg — per-dim
prediction range vs truth range and matched-window (steps 0..29) MAE vs
state-copy on the same rows; tripwires abort the smoke rc!=0 so the
sweep never launches on a broken harness.

Resume: progress is checkpointed to ``<out-stem>.partial.npz`` every
``--save-every`` frames; restart skips completed rows.

GPU: single local H100, batch-1 ``predict_action`` (their API is
per-frame), CUDA-graphed expert loop. Record-only: nothing gates or
repoints our runs.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("molmoact2_panel")

# The panel eval invocation, pinned (plans/holdout_curated_v0_k4l2.json;
# same constants as frame_mining.py / the box panel launchers).
DATA_ROOT = Path("~/datasets/mcobzarenco/community_curated_v0").expanduser()
FPS = (30.0,)
CAMERA_COUNTS = (1, 2)
HOLDOUT_FRACTION = 0.1
SPLIT_SEED = 0
CHUNK_SIZE = 50

MODEL_REPO = "allenai/MolmoAct2-SO100_101"
NORM_TAG = "so100_so101_molmoact2"
HORIZON = 30  # their SO-100/101 tag: 30 steps at native fps = 1.0 s
NUM_STEPS = 10  # their shipped default Euler step count
BASE_SEED = 0
PRED_KEY = "pred:molmoact2-so100@release"

REF_NPZ = (
    REPO_ROOT
    / "reports/eval__fontaine_molmo2_er_60k_ddp4__step_015000__panel_curated_v0_k4l2.npz"
)
COPY_KEYS = (
    "index",
    "repo_id",
    "episode_index",
    "frame_index",
    "truth",
    "valid",
    "core",
    "pred:state-copy",
    "pred:state-copy-norm",
)
OUT_STEM_DEFAULT = "reports/eval__molmoact2_so100_release__panel_curated_v0_k4l2_oob"

# Smoke tripwires (pre-registered): unit/sign bugs, not model quality.
RANGE_MARGIN = 1.5  # pred range may exceed truth span by at most this factor
MAE_TRIPWIRE = 3.0  # smoke matched-window MAE must be < 3x state-copy's


def _patch_modeling_for_bf16(local_dir: str) -> None:
    """Port of the upstream DROID example's idempotent bf16 patches
    (examples/droid/host_server_droid.py): the flow trajectory is
    hardcoded fp32 and ``_to_array`` calls ``.numpy()`` on bf16."""
    patches = [
        (
            (
                "device=device,\n            dtype=torch.float32,\n"
                "            generator=generator,"
            ),
            (
                "device=device,\n"
                "            dtype=source_tensor.dtype,  # patched_bf16_dtype\n"
                "            generator=generator,"
            ),
            "patched_bf16_dtype",
        ),
        (
            "return value.detach().cpu().numpy().astype(np.float32, copy=False)",
            (
                "return value.detach().cpu().float().numpy()"
                ".astype(np.float32, copy=False)  # patched_bf16_to_array"
            ),
            "patched_bf16_to_array",
        ),
    ]
    candidates = [Path(local_dir) / "modeling_molmoact2.py"]
    modules_root = Path(
        "~/.cache/huggingface/modules/transformers_modules",
    ).expanduser()
    if modules_root.is_dir():
        candidates += sorted(modules_root.glob("*/modeling_molmoact2.py"))
    for path in candidates:
        try:
            src = path.read_text(encoding="utf-8")
        except OSError:
            continue
        new_src = src
        for needle, replacement, marker in patches:
            if marker in new_src:
                continue
            if needle not in new_src:
                log.warning("patch %s: needle not found in %s", marker, path)
                continue
            new_src = new_src.replace(needle, replacement, 1)
        if new_src != src:
            path.write_text(new_src, encoding="utf-8")
            log.info("patched %s", path)


def load_model(device: str) -> tuple:
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # predict_action reads norm_stats.json via config._name_or_path — must
    # load from a resolved local dir, not the hub repo id.
    local_dir = snapshot_download(repo_id=MODEL_REPO)
    print(f"snapshot: {local_dir}", flush=True)
    _patch_modeling_for_bf16(local_dir)
    # tokenizer_config ships extra_special_tokens as a list; transformers
    # wants a dict. The model only resolves them via convert_tokens_to_ids.
    processor = AutoProcessor.from_pretrained(
        local_dir,
        trust_remote_code=True,
        extra_special_tokens={},
    )
    model = (
        AutoModelForImageTextToText.from_pretrained(
            local_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    n_obs = getattr(model.config, "n_obs_steps", None)
    if n_obs != 1:
        sys.exit(
            f"config.n_obs_steps == {n_obs!r}, training used 1 — the HF-config "
            f"default of 30 shifts chunk slicing to index 29; stop",
        )
    # _move_inputs_to_device only moves tensors; the processor emits fp32
    # pixel_values which must be cast to the bf16 weights' dtype.
    target_dtype = next(model.parameters()).dtype

    def _move_and_cast(inputs, dev, _target=target_dtype):  # noqa: ANN001, ANN202
        out = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                value = value.to(dev)
                if value.is_floating_point() and value.dtype != _target:
                    value = value.to(_target)
            out[key] = value
        return out

    model._move_inputs_to_device = _move_and_cast
    # Kwarg drift across snapshots: new signature takes
    # inference_action_mode, the old DROID server took action_mode.
    params = inspect.signature(model.predict_action).parameters
    mode_kwarg = (
        "inference_action_mode" if "inference_action_mode" in params else "action_mode"
    )
    print(f"predict_action mode kwarg: {mode_kwarg}", flush=True)
    return model, processor, mode_kwarg


def build_dataset():  # noqa: ANN201
    from bijou.data import EpisodeSplit, select_datasets

    selection = select_datasets(
        (DATA_ROOT,),
        (),
        CHUNK_SIZE,
        episode_split=EpisodeSplit.HOLDOUT,
        holdout_fraction=HOLDOUT_FRACTION,
        split_seed=SPLIT_SEED,
        allowed_fps=FPS,
        allowed_camera_counts=CAMERA_COUNTS,
    )
    dataset = selection.concat()
    print(
        f"selection: {len(selection.datasets)} datasets, {len(dataset)} frames",
        flush=True,
    )
    return dataset


def frame_images(item: dict) -> list:
    from PIL import Image

    pils = []
    for key in sorted(k for k in item if k.startswith("observation.images.")):
        image = item[key]
        array = (image.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
        pils.append(Image.fromarray(array))
    if not pils:
        sys.exit("frame has no observation.images.* keys — stop")
    return pils


def check_alignment(item: dict, ref: dict, row: int) -> None:
    """The frame_mining.py alignment oracle: the concat index must resolve
    to the exact banked frame, actions included."""
    if (
        str(item["repo_id"]) != str(ref["repo_id"][row])
        or int(item["episode_index"]) != int(ref["episode_index"][row])
        or int(item["frame_index"]) != int(ref["frame_index"][row])
    ):
        sys.exit(
            f"index misalignment at npz row {row}: item "
            f"{item['repo_id']}[{item['episode_index']}:{item['frame_index']}] "
            f"vs npz {ref['repo_id'][row]}[{ref['episode_index'][row]}:"
            f"{ref['frame_index'][row]}]",
        )
    action = item["action"].float().numpy()
    banked = ref["truth"][row][: action.shape[0]]
    if not np.allclose(action, banked, atol=1e-6):
        sys.exit(f"action chunk mismatch at npz row {row} — stop")


def smoke_sanity(pred: np.ndarray, ref: dict, rows: np.ndarray) -> dict:
    """Pre-registered smoke tripwires; any failure exits rc!=0."""
    truth = ref["truth"][rows][:, :HORIZON]
    valid = ref["valid"][rows][:, :HORIZON]
    window = pred[rows][:, :HORIZON]
    copy = ref["pred:state-copy"][rows][:, :HORIZON]
    m2 = valid & np.isfinite(truth).all(-1)
    w = m2[:, :, None].repeat(truth.shape[2], axis=2)

    per_dim = {}
    failures = []
    for d in range(truth.shape[2]):
        td = truth[:, :, d][m2]
        pd_ = window[:, :, d][m2]
        lo, hi = float(td.min()), float(td.max())
        span = max(hi - lo, 1e-6)
        p_lo, p_hi = float(pd_.min()), float(pd_.max())
        ok = (p_lo >= lo - RANGE_MARGIN * span) and (p_hi <= hi + RANGE_MARGIN * span)
        per_dim[f"dim{d}"] = {
            "truth_range": [round(lo, 3), round(hi, 3)],
            "pred_range": [round(p_lo, 3), round(p_hi, 3)],
            "ok": bool(ok),
        }
        if not ok:
            failures.append(f"dim{d} pred range [{p_lo:.2f},{p_hi:.2f}] outside band")

    mae = float(np.abs(window - truth)[w].mean())
    copy_mae = float(np.abs(copy - truth)[w].mean())
    if mae >= MAE_TRIPWIRE * copy_mae:
        failures.append(
            f"matched-window MAE {mae:.3f} >= {MAE_TRIPWIRE}x state-copy "
            f"{copy_mae:.3f} — unit/sign harness bug class",
        )
    out = {
        "n_rows": len(rows),
        "matched_window_mae": round(mae, 5),
        "state_copy_window_mae": round(copy_mae, 5),
        "per_dim_ranges": per_dim,
        "tripwires": failures or "none",
    }
    print(json.dumps(out, indent=1), flush=True)
    if failures:
        sys.exit(f"SMOKE TRIPWIRE: {failures} — no sweep on a broken harness")
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="smoke: N strided rows")
    parser.add_argument("--out-stem", default=OUT_STEM_DEFAULT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    with np.load(REF_NPZ, allow_pickle=True) as z:
        ref = {k: z[k] for k in z.files if k in COPY_KEYS}
    n_total = len(ref["index"])
    print(f"reference npz: {REF_NPZ.name}, {n_total} rows", flush=True)

    if args.limit:
        stride = max(n_total // args.limit, 1)
        rows = np.arange(n_total)[::stride][: args.limit]
        out_stem = f"{args.out_stem}_smoke{args.limit}"
    else:
        rows = np.arange(n_total)
        out_stem = args.out_stem
    partial_path = Path(f"{out_stem}.partial.npz")

    pred = np.full((n_total, CHUNK_SIZE, ref["truth"].shape[2]), np.nan, np.float32)
    done = np.zeros(n_total, dtype=bool)
    if partial_path.exists():
        with np.load(partial_path) as z:
            pred, done = z["pred"], z["done"]
        print(f"resume: {int(done.sum())} rows already done", flush=True)
    todo = rows[~done[rows]]
    print(f"predicting {len(todo)} of {len(rows)} selected rows", flush=True)

    model, processor, mode_kwarg = load_model(args.device)
    dataset = build_dataset()
    if len(dataset) <= int(ref["index"].max()):
        sys.exit(
            f"dataset has {len(dataset)} frames but npz index max is "
            f"{int(ref['index'].max())} — selection drift, stop",
        )

    subset = torch.utils.data.Subset(dataset, ref["index"][todo].tolist())
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=8,  # collate is identity; inference below is per-frame
        num_workers=args.num_workers,
        collate_fn=lambda items: items,
        prefetch_factor=4 if args.num_workers else None,
    )

    started = time.monotonic()
    count = 0
    since_save = 0
    for items in loader:
        for offset, item in enumerate(items):
            row = int(todo[count + offset])
            check_alignment(item, ref, row)
            state = np.asarray(item["observation.state"], dtype=np.float32).reshape(-1)
            generator = torch.Generator(device=args.device)
            generator.manual_seed(BASE_SEED + int(ref["index"][row]))
            with torch.inference_mode():
                out = model.predict_action(
                    processor=processor,
                    images=frame_images(item),
                    task=str(item["task"]),
                    state=state,
                    norm_tag=NORM_TAG,
                    enable_depth_reasoning=False,
                    num_steps=NUM_STEPS,
                    generator=generator,
                    normalize_language=True,
                    enable_cuda_graph=True,
                    **{mode_kwarg: "continuous"},
                )
            raw = out.actions if hasattr(out, "actions") else out
            if torch.is_tensor(raw):
                raw = raw.detach().to(dtype=torch.float32, device="cpu").numpy()
            actions = np.asarray(raw, dtype=np.float32)
            if actions.ndim == 3 and actions.shape[0] == 1:
                actions = actions[0]
            if actions.shape != (HORIZON, ref["truth"].shape[2]):
                sys.exit(f"prediction shape {actions.shape} != ({HORIZON}, 6) — stop")
            pred[row, :HORIZON] = actions
            done[row] = True
        count += len(items)
        since_save += len(items)
        if since_save >= args.save_every:
            np.savez_compressed(partial_path, pred=pred, done=done)
            since_save = 0
        rate = count / max(time.monotonic() - started, 1e-6) * 60
        print(f"progress: {count}/{len(todo)} ({rate:.1f} f/min)", flush=True)

    np.savez_compressed(partial_path, pred=pred, done=done)
    if not done[rows].all():
        sys.exit("loader ended with undone rows — stop")

    meta = {
        "model_repo": MODEL_REPO,
        "norm_tag": NORM_TAG,
        "mode": "continuous",
        "num_steps": NUM_STEPS,
        "horizon": HORIZON,
        "seed_rule": f"generator per frame = {BASE_SEED} + global concat index",
        "reference_npz": REF_NPZ.name,
        "rows": len(rows),
        "wall_minutes": round((time.monotonic() - started) / 60, 1),
    }
    if args.limit:
        meta["smoke_sanity"] = smoke_sanity(pred, ref, rows)
    else:
        payload = {k: ref[k] for k in COPY_KEYS}
        payload[PRED_KEY] = pred
        np.savez_compressed(f"{out_stem}.npz", allow_pickle=False, **payload)
        print(f"wrote {out_stem}.npz", flush=True)
        partial_path.unlink()
    Path(f"{out_stem}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_stem}.meta.json", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
