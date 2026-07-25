"""Self-contained static HTML eval report.

One file, no external assets: matplotlib charts and camera thumbnails are
embedded as base64 data URIs, so the report can be scp'd, archived next to
a checkpoint, and diffed across ablations. Sections: run config, the same
summary/paired/per-motor tables the terminal prints, then per-datapoint
blocks (repo id, global index, task, state, camera views) with one chart
overlaying every policy's predicted chunk against the ground truth.
"""

from __future__ import annotations

import base64
import html
import io
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import Tensor

from .metrics import PairedComparison, PolicySummary

_STYLE = """
body { font-family: -apple-system, system-ui, sans-serif; margin: 2em auto;
       max-width: 1100px; color: #222; }
table { border-collapse: collapse; margin: 0.8em 0; font-size: 14px;
        font-variant-numeric: tabular-nums; }
th, td { border: 1px solid #ccc; padding: 4px 10px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { background: #f0f0f0; }
pre { background: #f7f7f7; padding: 0.8em; font-size: 13px; overflow-x: auto; }
.sample { border-top: 2px solid #ddd; margin-top: 2em; padding-top: 1em; }
.cams img { margin-right: 8px; border: 1px solid #ccc; }
.meta { color: #555; font-size: 14px; }
img.chart { max-width: 100%; }
"""


@dataclass(frozen=True, slots=True)
class ReportSample:
    """Everything needed to render one datapoint's block."""

    index: int
    repo_id: str
    task: str
    state: Tensor
    cameras: dict[str, Tensor]
    truth: Tensor
    valid: Tensor
    predictions: dict[str, Tensor]


def _image_data_uri(image: Tensor, height: int = 220) -> str:
    """CHW float [0,1] tensor -> downscaled PNG data URI."""
    array = (image.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    pil = Image.fromarray(array)
    width = max(1, round(pil.width * height / pil.height))
    pil = pil.resize((width, height))
    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _chart_data_uri(sample: ReportSample, motor_names: list[str]) -> str:
    """Per-joint chart: ground truth (black) vs every policy's prediction."""
    matplotlib.use("Agg", force=True)
    dims = sample.truth.shape[-1]
    ncols = 3
    nrows = (dims + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 2.6 * nrows), squeeze=False
    )
    n_valid = int(sample.valid.sum())
    steps = range(n_valid)
    for dim in range(dims):
        ax = axes[dim // ncols][dim % ncols]
        ax.plot(
            steps,
            sample.truth[:n_valid, dim].tolist(),
            label="truth",
            color="black",
            linewidth=1.8,
        )
        for name, predicted in sample.predictions.items():
            ax.plot(
                steps,
                predicted[:n_valid, dim].tolist(),
                label=name,
                linestyle="--",
                linewidth=1.2,
            )
        name = motor_names[dim] if dim < len(motor_names) else f"dim {dim}"
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=90)
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(cell)}</th>" for cell in header)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def _sample_block(sample: ReportSample, motor_names: list[str]) -> str:
    per_policy_mae = {
        name: float((predicted - sample.truth).abs()[sample.valid].mean())
        for name, predicted in sample.predictions.items()
    }
    mae_line = "  |  ".join(
        f"{name}: {mae:.2f}" for name, mae in per_policy_mae.items()
    )
    state_line = ", ".join(f"{x:.1f}" for x in sample.state.tolist())
    cameras = "".join(
        f'<img src="{_image_data_uri(image)}" alt="{html.escape(name)}" '
        f'title="{html.escape(name)}">'
        for name, image in sorted(sample.cameras.items())
    )
    return (
        f'<div class="sample">'
        f"<h3>{html.escape(sample.repo_id)} &mdash; frame {sample.index}</h3>"
        f'<p class="meta">task: {html.escape(sample.task)}<br>'
        f"state: [{state_line}]<br>chunk MAE &mdash; {html.escape(mae_line)}</p>"
        f'<div class="cams">{cameras}</div>'
        f'<img class="chart" src="{_chart_data_uri(sample, motor_names)}">'
        f"</div>"
    )


def render_report(
    path: Path,
    config_lines: list[str],
    summaries: list[PolicySummary],
    comparisons: list[PairedComparison],
    motor_names: list[str],
    samples: list[ReportSample],
    total_scored: int,
) -> None:
    summary_table = _table(
        ["policy", "chunk_mae", "p50", "p90", "first_mae", "chunk_mse", "ms/frame"],
        [
            [
                s.name,
                f"{s.chunk_mae:.3f}",
                f"{s.mae_p50:.3f}",
                f"{s.mae_p90:.3f}",
                f"{s.first_mae:.3f}",
                f"{s.chunk_mse:.1f}",
                f"{s.seconds_per_frame * 1000:.0f}",
            ]
            for s in summaries
        ],
    )
    motor_table = _table(
        ["policy", *motor_names],
        [[s.name, *(f"{v:.2f}" for v in s.per_motor_mae)] for s in summaries],
    )
    paired_table = _table(
        ["policy", "mean_delta_mae", "delta_p50", "win_rate"],
        [
            [
                c.policy,
                f"{c.mean_delta:+.3f}",
                f"{c.delta_p50:+.3f}",
                f"{100 * c.win_rate:.0f}%",
            ]
            for c in comparisons
        ],
    )
    blocks = "\n".join(_sample_block(sample, motor_names) for sample in samples)
    config = html.escape("\n".join(config_lines))
    document = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>bijou eval report</title><style>{_STYLE}</style></head><body>"
        f"<h1>bijou eval report</h1><pre>{config}</pre>"
        f"<h2>Chunk metrics (raw action units, pad-masked)</h2>{summary_table}"
        + (f"<h2>Paired vs baseline</h2>{paired_table}" if comparisons else "")
        + f"<h2>Per-motor chunk MAE</h2>{motor_table}"
        f"<h2>Sample predictions ({len(samples)} of {total_scored} scored "
        f"frames)</h2>{blocks}"
        "</body></html>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document)
