"""check.py test-tier oracle: the default step list must exclude gpu-marked
tests and --gpu must include them — a drift here silently drops a tier from
every commit gate."""

import subprocess
import sys

import check


def test_default_excludes_gpu_marked_tests() -> None:
    assert ("pytest", "-q", "-m", "not gpu", "tests") in check.steps(
        fix=False,
        gpu=False,
    )
    assert ("pytest", "-q", "-m", "not gpu", "tests") in check.steps(
        fix=True,
        gpu=False,
    )


def test_gpu_flag_runs_the_full_suite() -> None:
    assert ("pytest", "-q", "tests") in check.steps(fix=False, gpu=True)
    assert ("pytest", "-q", "tests") in check.steps(fix=True, gpu=True)


def test_pytest_is_the_last_step_in_every_mode() -> None:
    # The verdict line hinges on max() over all steps; pytest last keeps the
    # slow step's output adjacent to the verdict for piped tails.
    for fix in (False, True):
        for gpu in (False, True):
            assert check.steps(fix=fix, gpu=gpu)[-1][0] == "pytest"


def test_gpu_marker_is_registered() -> None:
    # --strict-markers (pyproject addopts) errors on unregistered markers;
    # this asserts the registration side so a pyproject edit can't orphan
    # the tier. Runs the REAL config via `pytest --markers`.
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--markers"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "@pytest.mark.gpu:" in out.stdout
