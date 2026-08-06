"""Arch-batch-1 launcher recipe verify (stage-0 of the batch).

Extracts the ``-m bijou.train`` argv from each arm launcher AS WRITTEN
(``fontaine/scripts/box/launch_box_gpu123_fontaine_flow_arch*.sh``),
parses it through the real ``bijou.train.parse_args``, and demands the
resulting TrainArgs equal the teacher@40k control's recorded train_args
(``reports/teacher_artrunk40k_train_args.json``, banked from the box
checkpoint — Amendment 1's control) plus EXACTLY the pre-registered
deltas and run bookkeeping. Any other difference = recipe drift, refuse
to launch. The F1 smoke script shares its recipe body with the arm
launchers by construction (same flag block); the launchers are the
launch-path ground truth verified here.

PRE-REG: fontaine/blog/src/posts/2026-08-06-prereg-arch-batch-1.md
(+ Amendment 1: arm 0 dropped, control := teacher@40k; + Amendment 2:
arm A := img280).

Usage: uv run python fontaine/scripts/arch_recipe_verify.py
"""

from __future__ import annotations

import dataclasses
import json
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bijou.train import parse_args

REPO = Path(__file__).resolve().parents[2]
TEACHER_ARGS = REPO / "reports/teacher_artrunk40k_train_args.json"
BOX = Path(__file__).parent / "box"

# Deltas shared by both arms: topology + bookkeeping + the teacher's
# resume-lineage fields that must NOT ride along.
COMMON_DELTAS = {
    "batch_size": 32,  # x3 ranks = eff-96, matches teacher's ddp2x48
    "backbone_init_from": "outputs/train/bijou_arb_rcond_100k_ddp4/step_100000",
    "resume": None,  # teacher's own 2.5k-crash resume bookkeeping
    "rewarmup_steps": 0,  # rode on the teacher's resume
    "wandb_project": "fontaine",  # charter §1: never write bijou-dev
}

ARMS = {
    "launch_box_gpu123_fontaine_flow_archA_img280_40k_ddp3.sh": {
        "run": "fontaine_flow_archA_img280_40k_ddp3",
        "deltas": {"max_soft_tokens": 280},  # THE arm-A variable
    },
    "launch_box_gpu123_fontaine_flow_archB_fullresid_40k_ddp3.sh": {
        "run": "fontaine_flow_archB_fullresid_40k_ddp3",
        "deltas": {"conditioning_streams": "residual"},  # THE arm-B variable
    },
}


def launcher_train_argv(launcher: Path, run: str) -> list[str]:
    text = launcher.read_text()
    text = text.replace("${RUN}", run).replace("$RUN", run)
    text = text.replace('"$BATCH"', "32").replace("$BATCH", "32")
    start = text.index("-m bijou.train")
    end = text.index("2>&1", start)
    argv = shlex.split(text[start:end].replace("\\\n", " "))
    assert argv[:3] == ["-m", "bijou.train", "--train-data"]
    return argv[2:]


def normalize(value: object) -> object:
    if isinstance(value, (tuple, list)):
        return [normalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> None:
    teacher = json.loads(TEACHER_ARGS.read_text())["train_args"]
    failures: list[str] = []
    for launcher_name, spec in ARMS.items():
        run = spec["run"]
        argv = launcher_train_argv(BOX / launcher_name, run)
        sys.argv = ["bijou.train", *argv]
        ours = {
            key: normalize(value)
            for key, value in dataclasses.asdict(parse_args()).items()
        }
        expected_deltas = {
            **COMMON_DELTAS,
            **spec["deltas"],
            "save_dir": f"outputs/train/{run}",
            "wandb_run_name": run,
        }
        verbatim = 0
        for key, ours_value in sorted(ours.items()):
            teacher_value = normalize(teacher.get(key, ours_value))
            if key in expected_deltas:
                want = normalize(expected_deltas[key])
                if ours_value != want:
                    failures.append(
                        f"{run}: DELTA WRONG {key}: launcher={ours_value!r} "
                        f"pre-registered={want!r}",
                    )
                continue
            if ours_value != teacher_value:
                failures.append(
                    f"{run}: RECIPE DRIFT {key}: launcher={ours_value!r} "
                    f"teacher={teacher_value!r} (not a pre-registered delta)",
                )
            else:
                verbatim += 1
        print(
            f"{run}: {verbatim} fields teacher-verbatim, "
            f"{len(expected_deltas)} pre-registered deltas checked",
        )
    if failures:
        print("\nREFUSE TO LAUNCH:")
        for failure in failures:
            print(f"  {failure}")
        sys.exit(1)
    print("arch-batch recipe verify: OK (both arms)")


if __name__ == "__main__":
    main()
