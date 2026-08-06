"""SnapFlow launcher recipe verify (stage 0 of the launcher).

Extracts the ``-m bijou.train`` argv from the launcher script AS WRITTEN,
parses it through the real ``bijou.train.parse_args``, and demands the
resulting TrainArgs equal the teacher checkpoint's recorded train_args
plus EXACTLY the pre-registered deltas (2026-08-06 pre-registration) and
run bookkeeping. Any other difference = recipe drift, refuse to launch.

Usage: uv run python fontaine/scripts/snapflow_recipe_verify.py
"""

from __future__ import annotations

import dataclasses
import json
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bijou.train import parse_args

TEACHER = Path("outputs/train/bijou_flow_artrunk_h1024_40k_ddp2/step_080000")
LAUNCHER = Path(__file__).parent / "launch_local_snapflow_distill_30k_1xh100.sh"
RUN = "bijou_flow_snapdistill_h1024_30k_1xh100"

# The pre-registered science deltas + run bookkeeping. Everything else
# must match the teacher's recorded train_args exactly.
EXPECTED_DELTAS = {
    "steps": 30000,
    "decoder_lr": 2.5e-05,
    "grad_clip": 1.0,
    "batch_size": 24,
    "init_from": str(TEACHER),
    "resume": None,  # teacher's own 40k->80k resume bookkeeping
    "rewarmup_steps": 0,  # rode on the teacher's resume; ours is init-from
    "distill": "snapflow",
    "target_time_embed": True,
    "save_dir": f"outputs/train/{RUN}",
    "wandb_run_name": RUN,
}


def launcher_train_argv() -> list[str]:
    """The stage-3 train command of the launcher, shell-parsed, with the
    launcher's own variables expanded."""
    text = LAUNCHER.read_text()
    text = text.replace("${RUN}", RUN).replace("$RUN", RUN)
    text = text.replace("$TEACHER", str(TEACHER))
    start = text.index("-m bijou.train")
    # The command ends at the tee redirect (the argv proper).
    end = text.index("2>&1", start)
    argv = shlex.split(text[start:end].replace("\\\n", " "))
    assert argv[:3] == ["-m", "bijou.train", "--train-data"]
    return argv[2:]


# TrainArgs fields added AFTER the teacher run was recorded: the launcher
# must leave them at their inert defaults (verified below like any other
# field).
POST_TEACHER_DEFAULTS = {
    "bucket_by_length": False,
    "backward_chunks": 1,
    "target_time_embed": True,  # overridden in EXPECTED_DELTAS anyway
    "distill": "snapflow",
    "allow_same_seed_resume": False,  # resume-hardening flag, 2026-08-06
    "state_dropout": 0.0,  # #9 anti-shortcut regularizer flag, 2026-08-06
}


def normalize(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [normalize(item) for item in value]
    return value


def main() -> None:
    teacher = json.loads((TEACHER / "bijou_config.json").read_text())["train_args"]
    expected = {**teacher, **POST_TEACHER_DEFAULTS, **EXPECTED_DELTAS}

    sys.argv = ["bijou.train", *launcher_train_argv()]
    staged = {
        key: normalize(value) for key, value in dataclasses.asdict(parse_args()).items()
    }

    problems = []
    for key, want in expected.items():
        if key not in staged:
            problems.append(f"missing from TrainArgs: {key}")
        elif staged[key] != want:
            problems.append(f"{key}: staged {staged[key]!r} != expected {want!r}")
    extra = set(staged) - set(expected)
    if extra:
        problems.append(f"TrainArgs fields the teacher never recorded: {sorted(extra)}")
    if problems:
        print("RECIPE DRIFT — the launcher is NOT teacher+deltas:")
        for p in problems:
            print(f"  {p}")
        sys.exit(1)
    unchanged = len(expected) - len(EXPECTED_DELTAS)
    print(
        f"stage 0 OK: launcher == teacher train_args + pre-registered deltas "
        f"only ({unchanged} fields verbatim, {len(EXPECTED_DELTAS)} deltas)",
    )


if __name__ == "__main__":
    main()
