"""Oracles for the eval report's paired-read section
(fontaine/scripts/grasp_sft_joint_unseen_report.py --paired-json): the
section renders the frozen sim100_paired_read.py numbers verbatim
(delta + CI wording, exact-p formatting, discordant counts), carries
the preset's ambiguous-band note, and stays absent when no paired json
is passed. Plus the ladder-embed section (--ladder-b64): renders the
pdnorm_panel_ladder_chart.py sidecar as an img below the meta line,
rejects non-PNG payloads, quiet on a missing preset default, loud on a
missing explicit flag. Plus the estimator-seam line (--truthfit-json):
renders pdnorm_endpoint_truthfit_rewear.py's ladder_read block
verbatim under the ladder figure, same quiet/loud behavior split."""

import base64
import json
import sys
from pathlib import Path

import pytest

from fontaine.scripts.grasp_sft_joint_unseen_report import (
    PRESETS,
    estimator_seam_line,
    ladder_section,
    main,
    paired_section,
)


def make_paired_payload(
    *,
    ci_excludes_zero: bool = True,
    prog_excludes_zero: bool = True,
) -> dict:
    return {
        "prereg": "posts/2026-08-xx-prereg-grasp-sft-v2-joint-pdnorm.md",
        "role": "recorded non-gating read (rides alongside the frozen absolute bands)",
        "arms": {
            "a": {"label": "pdnorm_endpoint", "path": "a.json", "config": {}},
            "b": {"label": "disc1000_demosonly", "path": "b.json", "config": {}},
        },
        "read": {
            "n_seeds": 100,
            "success": {
                "count_a": 18,
                "count_b": 11,
                "count_delta": 7,
                "count_delta_ci95": [2.0, 12.0],
                "ci_excludes_zero": ci_excludes_zero,
            },
            "discordant": {
                "both_succeed": 8,
                "a_only": 10,
                "b_only": 3,
                "both_fail": 79,
                "mcnemar_exact_p_two_sided": 0.0625,
            },
            "progress": {
                "mean_delta_cm": 1.2345,
                "ci95": [0.5, 1.9],
                "ci_excludes_zero": prog_excludes_zero,
                "win_rate": 0.8,
                "tie_rate": 0.05,
            },
            "reset_strikes": {"a": 0, "b": 2},
        },
    }


def write_leg(path: Path) -> None:
    episodes = [
        {
            "seed": s,
            "success_tick": 100 if s < 11 else None,
            "initial_cm": 8.0,
            "min_cm": 2.5 if s < 11 else 5.0,
            "final_cm": 2.5 if s < 11 else 6.0,
            "reset_strikes": 0,
        }
        for s in range(100)
    ]
    path.write_text(json.dumps({"config": {}, "episodes": episodes}))


def test_section_renders_frozen_numbers_verbatim() -> None:
    html = paired_section(make_paired_payload(), " BAND-NOTE.")
    assert "Paired read: pdnorm_endpoint vs disc1000_demosonly" in html
    assert "success-count delta, 18 vs 11 (CI95 [2, 12], excludes 0)" in html
    assert "+7" in html
    assert "p = 6.2e-02" in html
    assert "McNemar exact, 10 vs 3 discordant seeds" in html
    assert "mean progress delta (CI95 [0.5, 1.9], excludes 0)" in html
    assert "+1.23 cm" in html
    assert "per-seed progress win rate (5% ties)" in html
    assert "reset strikes 0/2" in html
    assert "BAND-NOTE." in html


def test_ci_spanning_zero_says_spans() -> None:
    html = paired_section(
        make_paired_payload(ci_excludes_zero=False, prog_excludes_zero=False),
        "",
    )
    assert html.count("spans 0") == 2
    assert "excludes 0" not in html


def test_main_disc1000_preset_wires_section_and_band_note(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leg = tmp_path / "flow_unseen.json"
    write_leg(leg)
    paired = tmp_path / "paired.json"
    paired.write_text(json.dumps(make_paired_payload()))
    out = tmp_path / "report.html"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "grasp_sft_joint_unseen_report.py",
            "--preset",
            "disc1000",
            "--leg-json",
            str(leg),
            "--video-dir",
            str(tmp_path / "novideos"),
            "--out-html",
            str(out),
            "--gallery-dir",
            str(tmp_path / "gallery"),
            "--paired-json",
            str(paired),
        ],
    )
    assert main() == 0
    html = out.read_text()
    assert "Paired read: pdnorm_endpoint vs disc1000_demosonly" in html
    # The disc1000 preset carries the pre-reg's ambiguous-band context.
    assert "11&ndash;19 ambiguous band" in html
    # Section sits between the anchors chart and the per-seed strip.
    assert (
        html.index("Against the anchors")
        < html.index("Paired read:")
        < html.index("Per-seed outcomes")
    )


def test_main_pdnormendpoint_preset_anchors_bands_and_paired_section(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leg = tmp_path / "flow_unseen.json"
    write_leg(leg)
    paired = tmp_path / "paired.json"
    paired.write_text(json.dumps(make_paired_payload()))
    out = tmp_path / "report.html"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "grasp_sft_joint_unseen_report.py",
            "--preset",
            "pdnormendpoint",
            "--leg-json",
            str(leg),
            "--video-dir",
            str(tmp_path / "novideos"),
            "--out-html",
            str(out),
            "--gallery-dir",
            str(tmp_path / "gallery"),
            "--paired-json",
            str(paired),
        ],
    )
    assert main() == 0
    html = out.read_text()
    # Anchor rows: base 9 / probe 44 / disc1000 baseline 11 — the paired
    # baseline arm gets its own row (labels render inside the chart PNG,
    # so the rows are asserted structurally; the tile joins the values).
    assert [(r[0], r[1]) for r in PRESETS["pdnormendpoint"]["anchor_rows"]] == [
        ("base (no SFT)", 9),
        ("joint probe step2000 (313 demos)", 44),
        ("disc1000 demosonly baseline (paired arm)", 11),
    ]
    assert "9 / 44 / 11" in html
    assert "anchors: base / probe / disc1000 baseline" in html
    # Meta line names the pre-reg's frozen absolute bands and the
    # wear-audit panel anchor ladder.
    assert "&le;10" in html and "broken-class band" in html
    assert "11&ndash;19 ambiguous band" in html
    assert "&ge;20 exonerates the mix" in html
    assert "27.40" in html and "25.15" in html and "8.37" in html
    # Released pre-SFT row joined the ladder once measured (08:22Z
    # 08-18, record-only): 25.89 at the null.
    assert "25.89 released pre-SFT" in html
    # Checkpoint/meta fields stay placeholders until the endpoint session
    # stamps them.
    assert "FILL-AT-ENDPOINT" in html
    # --paired-json composes: section present, band note carried over,
    # placed between the anchors chart and the per-seed strip.
    assert "Paired read: pdnorm_endpoint vs disc1000_demosonly" in html
    assert html.count("11&ndash;19 ambiguous band") == 2  # meta + band note
    assert (
        html.index("Against the anchors")
        < html.index("Paired read:")
        < html.index("Per-seed outcomes")
    )


def make_ladder_sidecar(path: Path) -> str:
    """A tiny valid-PNG-header payload standing in for the real sidecar."""
    payload = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"ladder-bytes").decode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
    return payload


def run_main_pdnormendpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *extra: str,
) -> Path:
    leg = tmp_path / "flow_unseen.json"
    write_leg(leg)
    out = tmp_path / "report.html"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "grasp_sft_joint_unseen_report.py",
            "--preset",
            "pdnormendpoint",
            "--leg-json",
            str(leg),
            "--video-dir",
            str(tmp_path / "novideos"),
            "--out-html",
            str(out),
            "--gallery-dir",
            str(tmp_path / "gallery"),
            *extra,
        ],
    )
    assert main() == 0
    return out


def test_ladder_section_embeds_sidecar_payload(tmp_path: Path) -> None:
    sidecar = tmp_path / "panel_ladder.b64"
    payload = make_ladder_sidecar(sidecar)
    section = ladder_section(sidecar)
    assert "<h2>Panel anchor ladder</h2>" in section
    assert f'<img src="data:image/png;base64,{payload}"' in section
    assert "panel_ladder.b64" in section
    # The figure reads against the wear-corrected class, never the raw row.
    assert "never the raw 58.14" in section


def test_ladder_section_rejects_non_png_payload(tmp_path: Path) -> None:
    sidecar = tmp_path / "panel_ladder.b64"
    sidecar.write_text(base64.b64encode(b"not a png").decode())
    with pytest.raises(AssertionError, match="base64-PNG"):
        ladder_section(sidecar)


def test_main_ladder_flag_renders_section_below_meta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sidecar = tmp_path / "panel_ladder.b64"
    payload = make_ladder_sidecar(sidecar)
    out = run_main_pdnormendpoint(
        tmp_path,
        monkeypatch,
        "--ladder-b64",
        str(sidecar),
    )
    html = out.read_text()
    assert f'<img src="data:image/png;base64,{payload}"' in html
    # The figure sits below the meta line's textual ladder, above the
    # tiles and the anchors chart.
    assert (
        html.index("27.40 re-worn")
        < html.index("Panel anchor ladder")
        < html.index("Against the anchors")
    )


def test_main_pdnormendpoint_defaults_to_chart_sidecar_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The preset default is the chart script's cwd-relative --out-b64
    # path; no flag needed once the endpoint session has stamped it.
    monkeypatch.chdir(tmp_path)
    assert PRESETS["pdnormendpoint"]["ladder_b64"] == Path(
        "reports/pdnorm_panel_ladder.b64",
    )
    payload = make_ladder_sidecar(tmp_path / "reports/pdnorm_panel_ladder.b64")
    out = run_main_pdnormendpoint(tmp_path, monkeypatch)
    html = out.read_text()
    assert "Panel anchor ladder" in html
    assert payload in html


def test_main_ladder_absent_when_sidecar_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Preset default missing → quiet skip (reports/ is gitignored, the
    # sidecar is regenerable); the section simply does not render.
    monkeypatch.chdir(tmp_path)
    out = run_main_pdnormendpoint(tmp_path, monkeypatch)
    assert "Panel anchor ladder" not in out.read_text()


def test_main_ladder_explicit_flag_missing_file_is_loud(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(FileNotFoundError):
        run_main_pdnormendpoint(
            tmp_path,
            monkeypatch,
            "--ladder-b64",
            str(tmp_path / "nope.b64"),
        )


def make_truthfit_payload(*, released: bool = True) -> dict:
    """A ladder_read block shaped like pdnorm_endpoint_truthfit_rewear.py
    output (values arbitrary; the line must render them verbatim)."""
    read = {
        "endpoint_native": 31.074,
        "endpoint_truthfit": 30.551,
        "estimator_seam_delta": 0.523,
        "sft_disc1000_truthfit": 27.398,
        "null_repo_midpoint": 25.153,
    }
    if released:
        read["released_truthfit"] = 27.141
    return {"npz": "endpoint.npz", "ladder_read": read}


def write_truthfit(path: Path, **kwargs: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_truthfit_payload(**kwargs)))


def test_seam_line_renders_ladder_read_verbatim() -> None:
    line = estimator_seam_line(make_truthfit_payload())
    assert "Estimator seam" in line
    assert "pdnorm_endpoint_truthfit_rewear.py" in line
    assert "<b>31.07</b> &rarr; truth-fit 30.55 (seam +0.52)" in line
    assert "disc-1000 27.40 / released 27.14 / repo-midpoint null 25.15" in line
    # The headline stays the deployment-honest native row.
    assert "native row stays the headline" in line


def test_seam_line_without_released_row_omits_it() -> None:
    line = estimator_seam_line(make_truthfit_payload(released=False))
    assert "disc-1000 27.40 / repo-midpoint null 25.15" in line
    assert "released" not in line


def test_seam_line_rejects_foreign_json() -> None:
    with pytest.raises(AssertionError, match="seam keys"):
        estimator_seam_line({"ladder_read": {"endpoint_native": 1.0}})


def test_main_truthfit_flag_renders_seam_under_ladder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sidecar = tmp_path / "panel_ladder.b64"
    make_ladder_sidecar(sidecar)
    truthfit = tmp_path / "truthfit.json"
    write_truthfit(truthfit)
    out = run_main_pdnormendpoint(
        tmp_path,
        monkeypatch,
        "--ladder-b64",
        str(sidecar),
        "--truthfit-json",
        str(truthfit),
    )
    html = out.read_text()
    # The seam line sits under the ladder figure, above the tiles and
    # the anchors chart.
    assert (
        html.index("Panel anchor ladder")
        < html.index("Estimator seam")
        < html.index("Against the anchors")
    )
    assert "<b>31.07</b> &rarr; truth-fit 30.55 (seam +0.52)" in html


def test_main_truthfit_renders_even_without_ladder_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The two embeds are independent: a missing (quiet-skipped) ladder
    # sidecar must not drop an explicitly passed seam json.
    monkeypatch.chdir(tmp_path)
    truthfit = tmp_path / "truthfit.json"
    write_truthfit(truthfit)
    out = run_main_pdnormendpoint(
        tmp_path,
        monkeypatch,
        "--truthfit-json",
        str(truthfit),
    )
    html = out.read_text()
    assert "Panel anchor ladder" not in html
    assert "Estimator seam" in html


def test_main_pdnormendpoint_defaults_to_truthfit_json_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The preset default is the cross-check script's cwd-relative --out
    # path; no flag needed once the ON-GO session has written it.
    monkeypatch.chdir(tmp_path)
    assert PRESETS["pdnormendpoint"]["truthfit_json"] == Path(
        "reports/analysis__pdnorm_endpoint_truthfit_wear.json",
    )
    write_truthfit(tmp_path / "reports/analysis__pdnorm_endpoint_truthfit_wear.json")
    out = run_main_pdnormendpoint(tmp_path, monkeypatch)
    assert "Estimator seam" in out.read_text()


def test_main_truthfit_absent_when_default_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Preset default missing → quiet skip (the json exists only once
    # the ON-GO endpoint npz does).
    monkeypatch.chdir(tmp_path)
    out = run_main_pdnormendpoint(tmp_path, monkeypatch)
    assert "Estimator seam" not in out.read_text()


def test_main_truthfit_explicit_flag_missing_file_is_loud(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(FileNotFoundError):
        run_main_pdnormendpoint(
            tmp_path,
            monkeypatch,
            "--truthfit-json",
            str(tmp_path / "nope.json"),
        )


def test_main_without_paired_json_has_no_section(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leg = tmp_path / "flow_unseen.json"
    write_leg(leg)
    out = tmp_path / "report.html"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "grasp_sft_joint_unseen_report.py",
            "--preset",
            "disc1000",
            "--leg-json",
            str(leg),
            "--video-dir",
            str(tmp_path / "novideos"),
            "--out-html",
            str(out),
            "--gallery-dir",
            str(tmp_path / "gallery"),
        ],
    )
    assert main() == 0
    html = out.read_text()
    assert "Paired read:" not in html
    assert "ambiguous band" not in html
