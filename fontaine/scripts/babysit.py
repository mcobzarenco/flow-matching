"""Babysit CLI (#21 P1, owner-signed 2026-08-07): one command per checkpoint.

Replaces the hand-run tick/checkpoint choreography (tmux + nvidia-smi +
ssh + rate arithmetic + Discord poll) with a single paste-ready block.

Output contract (owner constraints, sign-off 00:34Z):
- liveness by construction — pgrep + GPU-memory footprint, never a log
  tail; a dead trainer or vanished DDP rank flips the exit code;
- trajectories, not verdicts — last-k probe values, loss deltas since
  the previous sample, rate windows vs cumulative, pre-reg anchors
  printed alongside; no healthy/anomalous language anywhere;
- gate crossings are SURFACED as facts, never acted on — the
  healthy/anomalous/escalate call stays with the session (charter §6);
- the Discord poll runs LAST and unconditionally, so a babysit
  checkpoint cannot skip it (08-06 class fix, mechanized).

Registry: `fontaine/harness/babysit.toml`, one entry per live run,
updated at launch time. Previous samples cache in
`fontaine/harness/state/babysit_prev.json` so window rates are computed
against the last call (flush-lag illusions like the 23:57Z draws10
scare resolve via the cumulative line printed alongside).

Exit codes: 1 = liveness failure (or Discord poll failure), 3 = a gate
crossing was surfaced, 0 = neither. `babysit.py || escalate` works; the
script itself never kills anything.

Oracles: tests/test_babysit.py (rate/projection/gate arithmetic).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "fontaine" / "harness" / "babysit.toml"
PREV = REPO / "fontaine" / "harness" / "state" / "babysit_prev.json"
DISCORD = REPO / "fontaine" / "harness" / "discord.py"
INBOX = REPO / "fontaine" / "harness" / "state" / "discord_unreplied.jsonl"

EXIT_OK = 0
EXIT_LIVENESS = 1
EXIT_GATE_SURFACED = 3

SSH_OPTS = ["-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]
SENTINEL_PGREP = "@@PGREP"
SENTINEL_GPU = "@@GPU"
SENTINEL_PROBE = "@@PROBE"
SENTINEL_TAIL = "@@TAIL"
SENTINEL_CGROUP = "@@CGROUP"
# driver-background-task-guard (3 incidents 2026-08-07): a registered
# run whose processes live inside the driver unit's cgroup was launched
# as a session child — not teardown-safe (turn completion kills session
# tasks; unit stop used to kill the whole cgroup). Surface it at every
# poll, BEFORE the kill, so the session can relaunch via run_detached.sh.
DRIVER_UNIT = "fontaine-tick.service"


@dataclass
class Run:
    """One registry entry. `kind` is `train-jsonl` or `progress-log`."""

    name: str
    kind: str
    host: str  # "local" or an ssh host
    pgrep: str  # liveness pattern; count must be >= pgrep_min
    pgrep_min: int
    gpu_indices: list[int]
    gpu_mem_min_mib: int  # each listed GPU must hold at least this
    anchors: list[str]
    gates: list[dict[str, Any]]
    boundary: str
    # train-jsonl fields
    jsonl: str = ""
    total_steps: int = 0
    probe_key: str = ""
    vram_key: str = ""
    # progress-log fields
    log: str = ""
    progress_re: str = ""
    started_utc: str = ""
    gpu_hours_per_wall_hour: float = 1.0


@dataclass
class Report:
    lines: list[str] = field(default_factory=list)
    alive: bool = True
    gate_crossed: bool = False

    def add(self, line: str) -> None:
        self.lines.append(line)


def load_registry(path: Path) -> list[Run]:
    with path.open("rb") as f:
        data = tomllib.load(f)
    return [Run(**entry) for entry in data.get("run", [])]


def run_cmd(host: str, cmd: str, timeout: int = 60) -> tuple[int, str]:
    """Run locally or over ssh; returns (exit, stdout+stderr)."""
    argv = ["bash", "-c", cmd] if host == "local" else ["ssh", *SSH_OPTS, host, cmd]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 124, f"(timed out after {timeout}s)"
    return proc.returncode, proc.stdout + proc.stderr


def batched_probe_cmd(run: Run) -> str:
    """One shell round-trip per run: liveness facts + metric tails."""
    parts = [
        f"echo {SENTINEL_PGREP}",
        f"pgrep -fc '{run.pgrep}' || true",
        f"echo {SENTINEL_CGROUP}",
        # Three self-match exclusions, all measured live:
        # (1) 08-07 16:2xZ — the probe shell's ancestor chain: a
        # compound session command mentioning the run's stem (a log
        # grep on the same line) matches the pattern from inside the
        # driver's cgroup;
        # (2) 08-07 16:2xZ — NO pipeline around the for-loop: `| sort
        # -u` forks the loop into a subshell that inherits this bash
        # -c's cmdline (which contains the pattern) and is alive during
        # its own pgrep: a guaranteed false hit. Dedupe happens
        # Python-side.
        # (3) 08-08 05:4xZ — the probe shell's process group: a session
        # running `babysit | grep <stem>` matched its own grep — a
        # pipeline SIBLING of the babysit process, so never in the
        # ancestor chain, but in the tick cgroup with the stem in its
        # cmdline. Real runs launched via run_detached.sh get their own
        # session/pgid, and earlier session-child launches keep their
        # dead launch pipeline's pgid, so neither is masked.
        # (4) 08-08 19:0xZ — a driver-session WATCHER shell: a
        # background `until systemctl --user is-active <unit>` loop
        # (armed to launch the next arm at rc=0) carries the run stem
        # in its cmdline, lives in the tick cgroup, and is neither an
        # ancestor nor in the probe's pgid — a false DRIVER-CGROUP
        # exit 3. Monitoring shells are recognized by verb (systemctl/
        # journalctl/babysit.py in the cmdline) and skipped; a real
        # doomed workload never invokes those.
        (
            'anc=$$; ex=" "; while [ "$anc" -gt 1 ]; do ex="$ex$anc "; '
            "anc=$(awk '/^PPid:/{print $2}' /proc/$anc/status 2>/dev/null || echo 0);"
            " done; "
            "mypg=$(ps -o pgid= -p $$ | tr -d ' '); "
            f"for p in $(pgrep -f '{run.pgrep}'); do "
            'case "$ex" in *" $p "*) continue;; esac; '
            "ppg=$(ps -o pgid= -p $p 2>/dev/null | tr -d ' '); "
            '[ "$ppg" = "$mypg" ] && continue; '
            "tr '\\0' ' ' < /proc/$p/cmdline 2>/dev/null "
            "| grep -qE 'systemctl|journalctl|babysit\\.py' && continue; "
            "cat /proc/$p/cgroup 2>/dev/null; "
            "done || true"
        ),
        f"echo {SENTINEL_GPU}",
        (
            "nvidia-smi --query-gpu=index,memory.used,utilization.gpu"
            " --format=csv,noheader,nounits"
        ),
    ]
    if run.kind == "train-jsonl":
        parts += [
            f"echo {SENTINEL_PROBE}",
            f"grep -h '{run.probe_key}' {run.jsonl} | tail -12 || true",
            f"echo {SENTINEL_TAIL}",
            f"tail -n 3 {run.jsonl}",
        ]
    else:
        if run.jsonl and run.probe_key:
            # progress-log runs may still carry a jsonl (often the launch
            # log itself) — fetch the probe ladder from it (08-09 20:3xZ:
            # the adamc wiring was a silent no-op without this branch)
            parts += [
                f"echo {SENTINEL_PROBE}",
                f"grep -h '{run.probe_key}' {run.jsonl} | tail -12 || true",
            ]
        parts += [
            f"echo {SENTINEL_TAIL}",
            f"tail -c 100000 {run.log} | tr '\\r' '\\n' | grep -oE '{run.progress_re}' | tail -3 || true",
        ]
    return "; ".join(parts)


def split_sections(out: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = ""
    for line in out.splitlines():
        stripped = line.strip()
        if stripped in (
            SENTINEL_PGREP,
            SENTINEL_GPU,
            SENTINEL_PROBE,
            SENTINEL_TAIL,
            SENTINEL_CGROUP,
        ):
            current = stripped
            sections[current] = []
        elif current:
            sections[current].append(line)
    return sections


# ---------------------------------------------------------------- arithmetic
# Pure functions — under oracle in tests/test_babysit.py.


def per_minute_rate(
    prev_t: datetime,
    prev_count: float,
    now_t: datetime,
    now_count: float,
) -> float | None:
    """Units/minute between two samples; None when the window is degenerate."""
    dt_min = (now_t - prev_t).total_seconds() / 60.0
    if dt_min <= 0.5 or now_count < prev_count:
        return None
    return (now_count - prev_count) / dt_min


def projected_total_hours(
    started: datetime,
    now: datetime,
    done: float,
    total: float,
) -> float | None:
    """Wall-hours for the whole job at the cumulative rate so far."""
    elapsed_h = (now - started).total_seconds() / 3600.0
    if elapsed_h <= 0 or done <= 0:
        return None
    return elapsed_h * total / done


def gate_fact(
    gate: dict[str, Any],
    *,
    step: int | None = None,
    probe: float | None = None,
    vram_gib: float | None = None,
    projected_gpu_hours: float | None = None,
) -> tuple[str, bool]:
    """(fact line, crossed). States the numbers; never a verdict."""
    kind = gate["kind"]
    if kind == "probe_below_by_step":
        line = (
            f"gate (pre-reg): probe below {gate['value']} by step {gate['step']}"
            f" — latest probe {probe}@{step}"
        )
        crossed = (
            step is not None
            and probe is not None
            and step >= int(gate["step"])
            and probe >= float(gate["value"])
        )
        return line, crossed
    if kind == "vram_max_gib":
        line = (
            f"gate (rule): vram_alloc_peak <= {gate['value']} GiB — latest {vram_gib}"
        )
        return line, vram_gib is not None and vram_gib > float(gate["value"])
    if kind == "gpu_hours_max":
        line = (
            f"gate (pre-reg): total <= {gate['value']} GPU-h"
            f" — cumulative projection {projected_gpu_hours and round(projected_gpu_hours, 1)}"
        )
        return line, projected_gpu_hours is not None and projected_gpu_hours > float(
            gate["value"],
        )
    return f"gate (unknown kind {kind!r}): {gate}", False


# ---------------------------------------------------------------- per-run


def check_liveness(run: Run, sections: dict[str, list[str]], report: Report) -> None:
    """pgrep count + per-GPU memory floor. Never a log tail."""
    pgrep_lines = [ln for ln in sections.get(SENTINEL_PGREP, []) if ln.strip()]
    count = int(pgrep_lines[0]) if pgrep_lines and pgrep_lines[0].isdigit() else 0
    ok_procs = count >= run.pgrep_min
    gpus: dict[int, tuple[int, int]] = {}
    for ln in sections.get(SENTINEL_GPU, []):
        m = re.match(r"\s*(\d+),\s*(\d+),\s*(\d+)", ln)
        if m:
            gpus[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))
    bad_gpus = [
        i for i in run.gpu_indices if i not in gpus or gpus[i][0] < run.gpu_mem_min_mib
    ]
    mem_str = " ".join(
        f"gpu{i}:{gpus[i][0]}MiB/{gpus[i][1]}%" for i in sorted(gpus) if i in gpus
    )
    report.add(
        f"  liveness: {count} procs matching (need >={run.pgrep_min}); {mem_str}",
    )
    if not ok_procs:
        report.alive = False
        report.add(
            f"  LIVENESS FAILURE: {count} < {run.pgrep_min} procs match '{run.pgrep}'",
        )
    if bad_gpus:
        report.alive = False
        report.add(
            f"  LIVENESS FAILURE: gpu(s) {bad_gpus} below {run.gpu_mem_min_mib} MiB floor",
        )


def check_driver_cgroup(
    run: Run,
    sections: dict[str, list[str]],
    report: Report,
) -> None:
    """Surface run processes living inside the driver unit's cgroup —
    launched as session children, not teardown-safe (incidents 1-3,
    2026-08-07). Surfaced-fact semantics like a gate crossing: the
    relaunch call stays with the session (via run_detached.sh)."""
    doomed = sorted(
        {ln for ln in sections.get(SENTINEL_CGROUP, []) if DRIVER_UNIT in ln},
    )
    if doomed:
        report.gate_crossed = True
        report.add(
            f"  DRIVER-CGROUP SURFACED: {len(doomed)} cgroup line(s) under"
            f" {DRIVER_UNIT} — run was launched as a session child, not"
            " teardown-safe; relaunch via fontaine/scripts/run_detached.sh",
        )


def probe_trajectory(
    run: Run,
    sections: dict[str, list[str]],
) -> list[tuple[int, float]]:
    """(step, probe) pairs from the PROBE section. json.loads first;
    regex fallback for probe rows embedded in mixed log lines (a
    progress-log run's `jsonl` is often the launch log itself)."""
    probes: list[tuple[int, float]] = []
    step_re = re.compile(r'"step":\s*(\d+)')
    val_re = re.compile(rf'"{re.escape(run.probe_key)}":\s*([-+0-9.eE]+)')
    for ln in sections.get(SENTINEL_PROBE, []):
        try:
            row = json.loads(ln)
            if run.probe_key in row:
                probes.append((int(row["step"]), float(row[run.probe_key])))
            continue
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
        sm, vm = step_re.search(ln), val_re.search(ln)
        if sm and vm:
            probes.append((int(sm.group(1)), float(vm.group(1))))
    return probes


def check_train_jsonl(
    run: Run,
    sections: dict[str, list[str]],
    prev: dict[str, Any],
    now: datetime,
    report: Report,
) -> dict[str, Any]:
    rows = []
    for ln in sections.get(SENTINEL_TAIL, []):
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    if not rows:
        report.alive = False
        report.add("  LIVENESS FAILURE: no parseable rows in jsonl tail")
        return {}
    latest = rows[-1]
    step = int(latest.get("step", 0))
    loss = latest.get("loss")
    sps = latest.get("s_per_step")
    vram = latest.get(run.vram_key) if run.vram_key else None
    report.add(
        f"  step {step}/{run.total_steps}  loss {loss}  {sps} s/step  vram {vram} GiB",
    )
    # probe trajectory — the last-k eval rows, printed as a trajectory
    probes = probe_trajectory(run, sections)
    if probes:
        traj = " -> ".join(f"{v:.2f}@{s}" for s, v in probes[-8:])
        report.add(f"  probe {run.probe_key}: {traj}")
    # deltas vs previous babysit sample
    if prev:
        prev_t = datetime.fromisoformat(prev["t"])
        rate = per_minute_rate(prev_t, prev["count"], now, step)
        if rate is not None:
            report.add(
                f"  since last sample ({prev['t']}, step {prev['count']}):"
                f" +{step - int(prev['count'])} steps, {rate:.1f} steps/min",
            )
        if loss is not None and prev.get("loss") is not None:
            report.add(
                f"  loss delta since last sample: {prev['loss']} -> {loss}"
                f" ({loss - prev['loss']:+.4f})",
            )
    if sps and run.total_steps and step:
        eta_h = (run.total_steps - step) * float(sps) / 3600.0
        report.add(f"  at {sps} s/step: ~{eta_h:.1f} h to step {run.total_steps}")
    latest_probe = probes[-1] if probes else (None, None)
    for gate in run.gates:
        line, crossed = gate_fact(
            gate,
            step=latest_probe[0],
            probe=latest_probe[1],
            vram_gib=vram,
        )
        report.add(f"  {line}")
        if crossed:
            report.gate_crossed = True
            report.add(
                "  GATE CROSSING SURFACED (call stays with the session, charter §6)",
            )
    return {"t": now.isoformat(), "count": step, "loss": loss}


def check_progress_log(
    run: Run,
    sections: dict[str, list[str]],
    prev: dict[str, Any],
    now: datetime,
    report: Report,
) -> dict[str, Any]:
    done, total = None, None
    for ln in sections.get(SENTINEL_TAIL, []):
        m = re.search(r"(\d+)/(\d+)", ln)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
    if done is None:
        # step-style logs match progress_re but carry no N/M total. The
        # tail section holds grep -oE spans of progress_re itself, so the
        # FIRST integer is the progress counter; trailing integers are
        # per-item detail ("seed 15 replan 24" counted 24, not 15 — the
        # fabricated 3.3 seeds/min tick read of 08-16 09:57Z).
        for ln in sections.get(SENTINEL_TAIL, []):
            nums = re.findall(r"\d+", ln)
            if nums:
                done = int(nums[0])
    if done is None:
        report.alive = False
        report.add("  LIVENESS FAILURE: no progress line parsed from log")
        return {}
    if run.jsonl and run.probe_key:
        probes = probe_trajectory(run, sections)
        if probes:
            traj = " -> ".join(f"{v:.2f}@{s}" for s, v in probes[-8:])
            report.add(f"  probe {run.probe_key}: {traj}")
    started = datetime.fromisoformat(run.started_utc)
    # Multi-phase logs reset the counter per phase (subgoal_swap 03:13Z
    # 08-09: identity -> swap roll). Dividing the new phase's counter by
    # time-since-launch fabricates a slow rate and a false gpu-hours
    # crossing, so the projection anchors at the last observed reset
    # (persisted in the prev cache), falling back to launch time.
    phase_t = (
        datetime.fromisoformat(prev["phase_t0"]) if prev.get("phase_t0") else started
    )
    phase_c0 = float(prev.get("phase_c0", 0))
    if prev and done < prev["count"]:
        phase_t, phase_c0 = datetime.fromisoformat(prev["t"]), 0.0
        report.add(
            f"  counter reset ({prev['count']} -> {done}): phase roll —"
            f" projection re-anchored at {prev['t']}",
        )
    if total is not None:
        report.add(f"  progress: {done}/{total} frames")
        cum_rate = per_minute_rate(phase_t, phase_c0, now, done)
        elapsed_h = (now - started).total_seconds() / 3600.0
        if cum_rate is not None:
            # elapsed is a fact; only the remainder is projected, at the
            # current phase's rate (== old whole-job formula when the
            # anchor is launch itself)
            proj_h = elapsed_h + (total - done) / cum_rate / 60.0
        else:
            proj_h = None
        proj_gpu_h = proj_h * run.gpu_hours_per_wall_hour if proj_h else None
    else:
        report.add(f"  progress: count {done} (no total in log — bare-count mode)")
        cum_rate, proj_h = None, None
        elapsed_h = (now - started).total_seconds() / 3600.0
        # elapsed GPU-h is a floor of any projection; gate still fires once truly crossed
        proj_gpu_h = elapsed_h * run.gpu_hours_per_wall_hour
    if prev:
        prev_t = datetime.fromisoformat(prev["t"])
        rate = per_minute_rate(prev_t, prev["count"], now, done)
        window = f"window since {prev['t']} ({prev['count']} -> {done}): " + (
            f"{rate:.1f} f/min" if rate is not None else "degenerate window"
        )
        report.add(f"  {window}")
    if cum_rate is not None and proj_h is not None:
        eta_h = (total - done) / cum_rate / 60.0
        report.add(
            f"  cumulative since {phase_t.isoformat()}: {cum_rate:.1f} f/min"
            f" -> projected total ~{proj_h:.1f} h, ~{eta_h:.1f} h remaining",
        )
    for gate in run.gates:
        line, crossed = gate_fact(gate, projected_gpu_hours=proj_gpu_h)
        report.add(f"  {line}")
        if crossed:
            report.gate_crossed = True
            report.add(
                "  GATE CROSSING SURFACED (call stays with the session, charter §6)",
            )
    return {
        "t": now.isoformat(),
        "count": done,
        "phase_t0": phase_t.isoformat(),
        "phase_c0": phase_c0,
    }


def babysit_run(
    run: Run,
    prev: dict[str, Any],
    now: datetime,
) -> tuple[Report, dict[str, Any]]:
    report = Report()
    report.add(f"[{run.name}] host={run.host} kind={run.kind} boundary: {run.boundary}")
    code, out = run_cmd(run.host, batched_probe_cmd(run))
    # ssh transport failure (exit 255: kex reset, MaxStartups drop,
    # network blip) is NOT run death — without this guard the empty
    # probe output parses as 0 procs / no GPUs and fabricates a full
    # LIVENESS FAILURE block (false alarm measured 19:47Z 2026-08-16,
    # box sshd shedding connections under load). One spaced retry,
    # then report the transport failure as its own fact.
    if run.host != "local" and code == 255:
        time.sleep(25)
        code, out = run_cmd(run.host, batched_probe_cmd(run))
    if run.host != "local" and code == 255:
        report.alive = False
        report.add(
            f"  HOST UNREACHABLE: ssh transport failed twice (exit 255) — "
            f"run state UNKNOWN, not a liveness verdict: {out.strip()[:200]}",
        )
        return report, {}
    if code == 124 or (code != 0 and not out.strip()):
        report.alive = False
        report.add(
            f"  LIVENESS FAILURE: probe command failed (exit {code}) {out.strip()}",
        )
        return report, {}
    sections = split_sections(out)
    check_liveness(run, sections, report)
    check_driver_cgroup(run, sections, report)
    sample: dict[str, Any] = {}
    if run.kind == "train-jsonl":
        sample = check_train_jsonl(run, sections, prev, now, report)
    elif run.kind == "progress-log":
        sample = check_progress_log(run, sections, prev, now, report)
    else:
        report.add(f"  (unknown kind {run.kind!r} — liveness only)")
    for anchor in run.anchors:
        report.add(f"  anchor: {anchor}")
    return report, sample


# ---------------------------------------------------------------- main


def inbox_pending(path: Path = INBOX) -> int:
    """Count of unreplied owner messages in the discord inbox state file
    (fed by discord.py `read`, cleared only by an explicit `ack`)."""
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def print_inbox_banner(pending: int) -> None:
    """FIRST line of every babysit (2026-08-13 missed-reply class fix):
    the tail of a babysit's output has been truncated before — the
    Discord poll section vanished and two owner questions sat unseen
    ~2 h. A head-of-output banner cannot be truncated away."""
    if pending:
        print(
            f"!!! UNREPLIED DISCORD INBOX: {pending} owner message(s) pending"
            " an in-channel reply — `discord.py inbox` to reprint,"
            " `discord.py ack <id>` after replying",
        )


def discord_poll() -> tuple[bool, str]:
    """Mandatory, runs last: read (advances cursor) + history -n 5."""
    chunks = []
    ok = True
    for args in (["read"], ["history", "-n", "5"]):
        code, out = run_cmd(
            "local",
            f"cd {REPO} && uv run python {DISCORD} {' '.join(args)}",
            timeout=120,
        )
        chunks.append(f"$ discord.py {' '.join(args)}\n{out.strip()}")
        if code != 0:
            ok = False
            chunks.append(f"(discord.py {args[0]} exited {code})")
    return ok, "\n".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--prev", type=Path, default=PREV)
    parser.add_argument(
        "--no-discord",
        action="store_true",
        help="skip the poll (tests only)",
    )
    ns = parser.parse_args()

    runs = load_registry(ns.registry)
    prev_all: dict[str, Any] = {}
    if ns.prev.exists():
        prev_all = json.loads(ns.prev.read_text())
    now = datetime.now(UTC)

    exit_code = EXIT_OK
    print_inbox_banner(inbox_pending())
    print(
        f"babysit @ {now.isoformat(timespec='seconds')} — {len(runs)} registered run(s)",
    )
    for run in runs:
        report, sample = babysit_run(run, prev_all.get(run.name, {}), now)
        print("\n".join(report.lines))
        if sample:
            prev_all[run.name] = sample
        if not report.alive:
            exit_code = EXIT_LIVENESS
        elif report.gate_crossed and exit_code == EXIT_OK:
            exit_code = EXIT_GATE_SURFACED

    ns.prev.parent.mkdir(parents=True, exist_ok=True)
    ns.prev.write_text(json.dumps(prev_all, indent=2))

    if not ns.no_discord:
        ok, out = discord_poll()
        print(out)
        if not ok:
            print("DISCORD POLL FAILED — treat as a liveness-grade failure")
            exit_code = EXIT_LIVENESS
        # the poll's `read` may have just fed the inbox — surface the
        # updated count here too (the head banner covers the NEXT
        # invocation if this tail is truncated)
        print_inbox_banner(inbox_pending())
    print(
        f"babysit exit {exit_code} (exit codes: 0=ok, 1=liveness/poll failure, 3=gate crossing surfaced)",
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
