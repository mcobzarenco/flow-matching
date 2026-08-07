"""Oracles for fontaine/scripts/babysit.py (#21 P1 babysit CLI).

The arithmetic oracles are anchored to REAL measured numbers from the
2026-08-07 00:59Z babysit window (draws10_t1 re-acceleration read), so a
regression here means the CLI would misreport a rate the session already
verified by hand. The gate facts are checked for both sides of each
crossing; the output-contract test asserts the no-verdicts constraint
(owner sign-off 00:34Z): a crossed gate is SURFACED, never judged.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "fontaine" / "scripts" / "babysit.py"
spec = importlib.util.spec_from_file_location("babysit", SCRIPT)
assert spec is not None and spec.loader is not None
babysit = importlib.util.module_from_spec(spec)
# dataclass decoration resolves cls.__module__ via sys.modules at exec time
sys.modules["babysit"] = babysit
spec.loader.exec_module(babysit)


def _t(minute: float) -> datetime:
    base = datetime(2026, 8, 7, 0, 0, tzinfo=UTC)
    return datetime.fromtimestamp(base.timestamp() + minute * 60, tz=UTC)


class TestRates:
    def test_live_window_00_59z(self) -> None:
        # 1952 -> 2272 frames over 00:50:46 -> 00:59:22 measured 37.2 f/min
        t0 = datetime(2026, 8, 7, 0, 50, 46, tzinfo=UTC)
        t1 = datetime(2026, 8, 7, 0, 59, 22, tzinfo=UTC)
        rate = babysit.per_minute_rate(t0, 1952, t1, 2272)
        assert rate is not None
        assert abs(rate - 37.2) < 0.1

    def test_degenerate_windows(self) -> None:
        t0 = _t(0)
        assert babysit.per_minute_rate(t0, 100, _t(0.4), 120) is None  # <30 s
        assert babysit.per_minute_rate(t0, 100, _t(10), 90) is None  # regressed

    def test_cumulative_projection_matches_hand_read(self) -> None:
        # 2272 frames in 81.7 min -> 27.8 f/min -> ~15.5 h total (now.md 01:0xZ)
        start = datetime(2026, 8, 6, 23, 37, 42, tzinfo=UTC)
        now = start.fromtimestamp(start.timestamp() + 81.7 * 60, tz=UTC)
        cum = babysit.per_minute_rate(start, 0, now, 2272)
        assert cum is not None and abs(cum - 27.8) < 0.05
        proj = babysit.projected_total_hours(start, now, 2272, 25800)
        assert proj is not None and abs(proj - 15.46) < 0.05

    def test_projection_degenerate(self) -> None:
        t0 = _t(0)
        assert babysit.projected_total_hours(t0, t0, 100, 200) is None
        assert babysit.projected_total_hours(t0, _t(10), 0, 200) is None


class TestGateFacts:
    def test_probe_kill_line_both_sides(self) -> None:
        gate = {"kind": "probe_below_by_step", "value": 12.0944, "step": 10000}
        # before the gate step: never crossed, whatever the value
        _, crossed = babysit.gate_fact(gate, step=2500, probe=12.0944)
        assert not crossed
        # at/past the gate step, probe still >= reference: crossed
        line, crossed = babysit.gate_fact(gate, step=10000, probe=12.5)
        assert crossed
        assert "12.0944" in line and "12.5" in line
        # at/past the gate step, probe below: not crossed
        _, crossed = babysit.gate_fact(gate, step=10500, probe=11.0)
        assert not crossed
        # no probe data yet: not crossed (absence is surfaced elsewhere)
        _, crossed = babysit.gate_fact(gate, step=None, probe=None)
        assert not crossed

    def test_vram_rule(self) -> None:
        gate = {"kind": "vram_max_gib", "value": 71.0}
        assert not babysit.gate_fact(gate, vram_gib=67.07)[1]
        assert babysit.gate_fact(gate, vram_gib=71.5)[1]

    def test_gpu_hours_gate(self) -> None:
        gate = {"kind": "gpu_hours_max", "value": 24.0}
        assert not babysit.gate_fact(gate, projected_gpu_hours=15.5)[1]
        assert babysit.gate_fact(gate, projected_gpu_hours=25.0)[1]
        assert not babysit.gate_fact(gate, projected_gpu_hours=None)[1]

    def test_no_verdict_language(self) -> None:
        # Owner constraint: trajectories/facts only — the words that would
        # constitute a verdict must not appear in any gate fact line.
        for gate, kwargs in [
            (
                {"kind": "probe_below_by_step", "value": 12.0944, "step": 10000},
                {"step": 10000, "probe": 12.5},
            ),
            ({"kind": "vram_max_gib", "value": 71.0}, {"vram_gib": 71.5}),
            ({"kind": "gpu_hours_max", "value": 24.0}, {"projected_gpu_hours": 25.0}),
        ]:
            line, crossed = babysit.gate_fact(gate, **kwargs)
            assert crossed
            for verdict_word in ("healthy", "anomalous", "kill", "fail", "ok"):
                assert verdict_word not in line.lower()


class TestParsing:
    def test_split_sections(self) -> None:
        out = "@@PGREP\n4\n@@GPU\n0, 71617, 98\n1, 71721, 100\n@@TAIL\nrow1\n"
        sections = babysit.split_sections(out)
        assert sections["@@PGREP"] == ["4"]
        assert len(sections["@@GPU"]) == 2
        assert sections["@@TAIL"] == ["row1"]

    def test_real_registry_loads(self) -> None:
        runs = babysit.load_registry(babysit.REGISTRY)
        assert len(runs) >= 1
        for run in runs:
            assert run.kind in ("train-jsonl", "progress-log")
            assert run.pgrep and run.gpu_indices
            for gate in run.gates:
                assert gate["kind"] in (
                    "probe_below_by_step",
                    "vram_max_gib",
                    "gpu_hours_max",
                )


class TestPerRunChecks:
    def _train_sections(self, probe_rows: list[str], tail_rows: list[str]) -> dict:
        return {
            "@@PGREP": ["4"],
            "@@GPU": ["0, 71617, 98", "1, 71721, 100", "2, 71545, 98", "3, 71381, 97"],
            "@@PROBE": probe_rows,
            "@@TAIL": tail_rows,
        }

    def _train_run(self) -> object:
        return babysit.Run(
            name="t",
            kind="train-jsonl",
            host="local",
            pgrep="x",
            pgrep_min=4,
            gpu_indices=[0, 1, 2, 3],
            gpu_mem_min_mib=30000,
            anchors=[],
            gates=[{"kind": "probe_below_by_step", "value": 12.0944, "step": 10000}],
            boundary="-",
            jsonl="-",
            total_steps=40000,
            probe_key="eval_chunk_mae",
            vram_key="vram_alloc_peak_gib",
        )

    def test_train_trajectory_and_sample(self) -> None:
        run = self._train_run()
        probe = [
            json.dumps({"step": 2000, "eval_chunk_mae": 13.206}),
            json.dumps({"step": 2500, "eval_chunk_mae": 12.0944}),
        ]
        tail = [
            json.dumps(
                {
                    "step": 2740,
                    "loss": 4.3404,
                    "s_per_step": 2.176,
                    "vram_alloc_peak_gib": 67.07,
                },
            ),
        ]
        report = babysit.Report()
        prev = {"t": "2026-08-07T00:30:00+00:00", "count": 2140, "loss": 4.55}
        now = datetime(2026, 8, 7, 1, 0, tzinfo=UTC)
        sample = babysit.check_train_jsonl(
            run,
            self._train_sections(probe, tail),
            prev,
            now,
            report,
        )
        assert sample == {"t": now.isoformat(), "count": 2740, "loss": 4.3404}
        text = "\n".join(report.lines)
        assert "13.21@2000 -> 12.09@2500" in text  # trajectory, not a verdict
        assert "+600 steps, 20.0 steps/min" in text
        assert "-0.2096" in text  # loss delta since prev sample
        assert report.alive and not report.gate_crossed

    def test_train_gate_crossing_surfaced_not_fatal(self) -> None:
        run = self._train_run()
        probe = [json.dumps({"step": 10000, "eval_chunk_mae": 12.5})]
        tail = [
            json.dumps(
                {
                    "step": 10000,
                    "loss": 4.0,
                    "s_per_step": 2.2,
                    "vram_alloc_peak_gib": 67.0,
                },
            ),
        ]
        report = babysit.Report()
        now = datetime(2026, 8, 7, 6, 0, tzinfo=UTC)
        babysit.check_train_jsonl(
            run,
            self._train_sections(probe, tail),
            {},
            now,
            report,
        )
        assert report.gate_crossed
        assert report.alive  # a crossed gate is surfaced, never a liveness kill
        assert "SURFACED" in "\n".join(report.lines)

    def test_progress_log_projection(self) -> None:
        run = babysit.Run(
            name="p",
            kind="progress-log",
            host="local",
            pgrep="x",
            pgrep_min=2,
            gpu_indices=[0],
            gpu_mem_min_mib=8000,
            anchors=[],
            gates=[{"kind": "gpu_hours_max", "value": 24.0}],
            boundary="-",
            log="-",
            progress_re="scored [0-9]+/[0-9]+ frames",
            started_utc="2026-08-06T23:37:42Z",
        )
        sections = {
            "@@PGREP": ["4"],
            "@@GPU": ["0, 12675, 23"],
            "@@TAIL": ["  scored 2272/25800 frames"],
        }
        report = babysit.Report()
        start = datetime(2026, 8, 6, 23, 37, 42, tzinfo=UTC)
        now = start.fromtimestamp(start.timestamp() + 81.7 * 60, tz=UTC)
        sample = babysit.check_progress_log(run, sections, {}, now, report)
        assert sample["count"] == 2272
        text = "\n".join(report.lines)
        assert "27.8 f/min" in text
        assert "~15.5 h" in text
        assert not report.gate_crossed  # 15.5 < 24
