"""Live launcher-side oracles for the subgoal-swap content read (#6).

Pre-reg 2026-08-09-prereg-subgoal-swap.md. The CPU halves are pinned in
``tests/test_subgoal_swap.py``; this script is the real-data half the
launcher runs abort-on-red:

``--mode identity`` (oracle ii, AFTER the identity run, BEFORE the swap
arm): the identity-forced run must reproduce the banked oracle arm
BYTE-EXACTLY — identity columns, state-copy rows and the bijou
prediction array all byte-equal. Byte-exactness is a fair bar here
because the identity run renders byte-identical prompts on the same
plan at the same batching (the selfsubgoal amendment-1 lesson: decode
bit-exactness holds only at MATCHED batch composition — which this
comparison has by construction).

``--mode swap`` (oracles i + iv, AFTER the swap run): rebuild the
pinned map from the judgment sidecars (per-repo seeding makes each
repo's derangement independent of the rest of the selection) and check
the dump mechanically over EVERY row: the donor is the map's donor and
never the receiving episode, the rendered text equals the donor's
fraction-matched label, swapped-away truth is never rendered unless it
coincides textually (recorded, allowed — plausible-but-wrong may
coincide), skipped single-labeled-episode datasets rendered EMPTY.

Every failure is a hard abort (SystemExit). ``--selftest`` exercises
the pass path and the abort branches on synthetic fixtures — no data,
no GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bijou.eval.subgoal_swap import (
    build_swap_map,
    fraction_matched_label,
)

ORACLE_KEY = "pred:bijou@100000_oraclesubgoal"
IDENTITY_RUN_KEY = "pred:bijou@100000_swapidentity"
SHARED_KEYS = (
    "truth",
    "valid",
    "index",
    "repo_id",
    "episode_index",
    "frame_index",
    "core",
    "pred:state-copy",
    "pred:state-copy-norm",
)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def check_identity(identity_npz: Path, oracle_npz: Path) -> None:
    """Oracle (ii): the identity run byte-reproduces the banked oracle
    arm — abort-grade on any column."""
    identity = _load(identity_npz)
    oracle = _load(oracle_npz)
    for key in SHARED_KEYS:
        if key not in identity or key not in oracle:
            raise SystemExit(f"ORACLE (ii) RED: column {key!r} missing")
        if identity[key].dtype != oracle[key].dtype or not np.array_equal(
            identity[key],
            oracle[key],
        ):
            raise SystemExit(
                f"ORACLE (ii) RED: column {key!r} differs between "
                f"{identity_npz} and {oracle_npz}",
            )
    if IDENTITY_RUN_KEY not in identity:
        raise SystemExit(
            f"ORACLE (ii) RED: {IDENTITY_RUN_KEY!r} missing from "
            f"{identity_npz} (keys: {sorted(identity)})",
        )
    if ORACLE_KEY not in oracle:
        raise SystemExit(
            f"ORACLE (ii) RED: {ORACLE_KEY!r} missing from banked "
            f"{oracle_npz} (keys: {sorted(oracle)})",
        )
    ours = identity[IDENTITY_RUN_KEY]
    banked = oracle[ORACLE_KEY]
    if ours.dtype != banked.dtype or ours.shape != banked.shape:
        raise SystemExit(
            f"ORACLE (ii) RED: prediction dtype/shape mismatch "
            f"{ours.dtype}{ours.shape} vs {banked.dtype}{banked.shape}",
        )
    if ours.tobytes() != banked.tobytes():
        flips = int(np.sum(np.any(ours != banked, axis=(1, 2))))
        raise SystemExit(
            f"ORACLE (ii) RED: identity run does NOT byte-reproduce the "
            f"banked oracle arm — {flips}/{ours.shape[0]} rows differ. "
            "The swap plumbing changes the decode even with donor=self; "
            "NO swap launch until diagnosed",
        )
    print(
        f"oracle (ii) GREEN: identity run byte-reproduces the banked "
        f"oracle arm ({ours.shape[0]} rows, all shared columns byte-equal)",
    )


def check_swap_dump(dump_path: Path, data_root: Path) -> None:
    """Oracles (i) + (iv) over the real dump: mapping, rendered text,
    empty-slot rule — every row, abort-grade."""
    dump = json.loads(dump_path.read_text())
    seed = dump.get("subgoal_swap_seed")
    if dump.get("subgoal_swap_identity"):
        raise SystemExit(
            "ORACLE (iv) RED: dump is from an IDENTITY run — the swap "
            "arm's dump is required",
        )
    if seed is None:
        raise SystemExit("ORACLE (iv) RED: dump carries no swap seed")
    rows = dump.get("rows") or []
    if not rows:
        raise SystemExit("ORACLE (iv) RED: dump has no rows")
    skipped = set(dump.get("skipped_datasets") or [])
    repos = sorted({str(row["repo_id"]) for row in rows})
    maps = {}
    for repo in repos:
        dataset_dir = data_root / repo
        if not (dataset_dir / "meta").is_dir():
            raise SystemExit(
                f"ORACLE (iv) RED: {dataset_dir} has no meta/ — cannot "
                "rebuild the map for the dumped repo",
            )
        # Per-repo seeding: each repo's derangement is independent of
        # the rest of the selection, so a per-repo rebuild IS the map.
        maps[repo] = build_swap_map({repo: dataset_dir}, seed=int(seed))
    swapped = 0
    empties = 0
    coincidences = 0
    for row in rows:
        repo = str(row["repo_id"])
        episode = int(row["episode_index"])
        frame = int(row["frame_index"])
        rendered = str(row["rendered_subgoal"])
        truth = str(row["true_subgoal"])
        where = f"{repo} ep {episode} frame {frame}"
        if not truth.strip():
            raise SystemExit(
                f"ORACLE (iv) RED: {where} dumped with an EMPTY true "
                "label — only labeled frames may carry swap rows",
            )
        swap_map = maps[repo]
        episodes = swap_map.episodes.get(repo, {})
        if episode not in episodes:
            raise SystemExit(
                f"ORACLE (iv) RED: {where} not a labeled episode in the "
                "rebuilt map — sidecar/materialized disagreement",
            )
        donor_index = row["donor_episode_index"]
        expected_donor = swap_map.donors.get(repo, {}).get(episode)
        if donor_index != expected_donor:
            raise SystemExit(
                f"ORACLE (i) RED: {where} dumped donor {donor_index} != "
                f"rebuilt map donor {expected_donor}",
            )
        if donor_index is None:
            if repo not in skipped or repo not in swap_map.skipped:
                raise SystemExit(
                    f"ORACLE (i) RED: {where} has no donor but {repo} is "
                    "not a skipped single-labeled-episode dataset",
                )
            if rendered != "":
                raise SystemExit(
                    f"ORACLE (iv) RED: {where} unswappable but rendered "
                    f"{rendered!r} — the empty-slot rule is broken",
                )
            empties += 1
            continue
        if int(donor_index) == episode:
            raise SystemExit(
                f"ORACLE (i) RED: {where} maps to ITSELF in a swap run",
            )
        donor = episodes[int(donor_index)]
        receiving = episodes[episode]
        expected = fraction_matched_label(donor, frame / receiving.length)
        if rendered != expected:
            raise SystemExit(
                f"ORACLE (iv) RED: {where} rendered {rendered!r} != "
                f"fraction-matched donor label {expected!r}",
            )
        swapped += 1
        if rendered == truth:
            coincidences += 1
    print(
        f"oracles (i)+(iv) GREEN: {len(rows)} rows checked — "
        f"{swapped} swapped, {empties} empty-slot "
        f"({len(skipped)} skipped dataset(s)), "
        f"{coincidences} textual coincidences with the true label "
        "(allowed: plausible-but-wrong may coincide)",
    )


def _selftest() -> None:
    def expect_abort(label: str, fn: Callable[..., None], *args: object) -> None:
        try:
            fn(*args)
        except SystemExit as error:
            print(f"  abort branch ok [{label}]: {error}")
            return
        raise SystemExit(f"SELFTEST RED: {label} did not abort")

    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        rows = 4
        shared = {
            "truth": np.zeros((rows, 5, 3), dtype=np.float32),
            "valid": np.ones((rows, 5), dtype=bool),
            "index": np.arange(rows),
            "repo_id": np.array(["a/b"] * rows),
            "episode_index": np.zeros(rows, dtype=np.int64),
            "frame_index": np.arange(rows),
            "core": np.ones(rows, dtype=bool),
            "pred:state-copy": np.ones((rows, 5, 3), dtype=np.float32),
            "pred:state-copy-norm": np.ones((rows, 5, 3), dtype=np.float32),
        }
        prediction = np.random.default_rng(0).standard_normal((rows, 5, 3))
        prediction = prediction.astype(np.float32)
        np.savez(
            root / "oracle.npz",
            **shared,
            **{ORACLE_KEY: prediction},
        )
        np.savez(
            root / "identity.npz",
            **shared,
            **{IDENTITY_RUN_KEY: prediction},
        )
        check_identity(root / "identity.npz", root / "oracle.npz")
        flipped = prediction.copy()
        flipped[1, 0, 0] += 1e-3
        np.savez(
            root / "identity_bad.npz",
            **shared,
            **{IDENTITY_RUN_KEY: flipped},
        )
        expect_abort(
            "prediction flip",
            check_identity,
            root / "identity_bad.npz",
            root / "oracle.npz",
        )
        bad_shared = dict(shared)
        bad_shared["frame_index"] = shared["frame_index"] + 1
        np.savez(
            root / "identity_badcol.npz",
            **bad_shared,
            **{IDENTITY_RUN_KEY: prediction},
        )
        expect_abort(
            "identity column drift",
            check_identity,
            root / "identity_badcol.npz",
            root / "oracle.npz",
        )

        # Swap-dump fixtures: one two-episode dataset, seed 0.
        dataset = root / "acme" / "pick"
        meta = dataset / "meta"
        meta.mkdir(parents=True)
        (meta / "judge_annotations.json").write_text(
            json.dumps({"prompt_hash": "h", "model_filter": "m"}),
        )
        (meta / "judgments.json").write_text(
            json.dumps(
                {
                    "judgments": [
                        {
                            "episode_index": index,
                            "model": "m",
                            "prompt_hash": "h",
                            "judged_at": "2026-08-01 00:00:00",
                            "num_timesteps": 10,
                            "max_image_dim": 512,
                            "usage": {"input_tokens": 1, "output_tokens": 1},
                            "judgment": {
                                "subgoals": [
                                    {"until_frame": 100, "subgoal": text},
                                ],
                            },
                        }
                        for index, text in ((0, "reach"), (1, "grasp"))
                    ],
                },
            ),
        )
        (meta / "episodes.jsonl").write_text(
            "\n".join(
                json.dumps({"episode_index": index, "length": 100}) for index in (0, 1)
            ),
        )
        dump = {
            "subgoal_swap_seed": 0,
            "subgoal_swap_identity": False,
            "skipped_datasets": [],
            "rows": [
                {
                    "repo_id": "acme/pick",
                    "episode_index": 0,
                    "frame_index": 40,
                    "true_subgoal": "reach",
                    "donor_episode_index": 1,
                    "rendered_subgoal": "grasp",
                },
                {
                    "repo_id": "acme/pick",
                    "episode_index": 1,
                    "frame_index": 10,
                    "true_subgoal": "grasp",
                    "donor_episode_index": 0,
                    "rendered_subgoal": "reach",
                },
            ],
        }
        (root / "dump.json").write_text(json.dumps(dump))
        check_swap_dump(root / "dump.json", root)
        bad = json.loads(json.dumps(dump))
        bad["rows"][0]["rendered_subgoal"] = "polish the hull"
        (root / "dump_badtext.json").write_text(json.dumps(bad))
        expect_abort(
            "rendered text mismatch",
            check_swap_dump,
            root / "dump_badtext.json",
            root,
        )
        bad = json.loads(json.dumps(dump))
        bad["rows"][0]["donor_episode_index"] = 0
        (root / "dump_selfmap.json").write_text(json.dumps(bad))
        expect_abort(
            "self-mapping donor",
            check_swap_dump,
            root / "dump_selfmap.json",
            root,
        )
        bad = json.loads(json.dumps(dump))
        bad["subgoal_swap_identity"] = True
        (root / "dump_identityrun.json").write_text(json.dumps(bad))
        expect_abort(
            "identity-run dump",
            check_swap_dump,
            root / "dump_identityrun.json",
            root,
        )
    print("SELFTEST GREEN: pass paths + abort branches all fire")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["identity", "swap"])
    parser.add_argument("--identity-npz", type=Path)
    parser.add_argument("--oracle-npz", type=Path)
    parser.add_argument("--dump", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        _selftest()
        return
    if args.mode == "identity":
        if args.identity_npz is None or args.oracle_npz is None:
            raise SystemExit("--mode identity needs --identity-npz + --oracle-npz")
        check_identity(args.identity_npz, args.oracle_npz)
    elif args.mode == "swap":
        if args.dump is None or args.data_root is None:
            raise SystemExit("--mode swap needs --dump + --data-root")
        check_swap_dump(args.dump, args.data_root)
    else:
        raise SystemExit("pick --mode identity|swap or --selftest")


if __name__ == "__main__":
    main()
