"""Golden-ticket stage-2 R2 read — ready before the data.

Implements exactly the stage-2 read of the golden-ticket pre-reg
(posts/2026-08-07-prereg-golden-ticket-screen.md, "Frozen reads" R2):
the stage-1 winner ticket's full-panel eval, judged ONLY on complement
core rows — panel core frames minus the probe plan's frame-identity
triples, because the probe rows selected the winner and must not judge
it.

  * R2 (frozen): paired per-frame Δ (winner − banked stable-key) on
    complement core rows, seeded frame bootstrap CI95. The ticket is
    REAL iff pooled Δ ≤ −0.05 AND CI95 excludes 0. Otherwise the
    screen CLOSES (stage 3 runs only on a stage-2 pass).
  * Alongside (board continuity, never the verdict): the full-panel
    pooled chunk MAE of both policies, and the PROBE-row Δ (the
    selection-biased number stage 1 already implied).
  * R4b/c (record-only): per-step horizon curve of the complement
    delta + first-step MAE mirror.

Row identity: the probe triple set comes from the stage-1 draws npz
(the rows that actually scored the tickets), keyed on
(repo_id, episode_index, frame_index). Full-panel npzs are joined the
draws10_t1_results way: identical length + byte-matching identity
columns, hard abort otherwise.

Provenance refusals (abort, never silent): the stage-2 npz must carry
the winner-only tickets sha (a392d630…) and a policy name carrying
"_ticket"; the banked npz must be the stable-key policy (no "_ticket");
the probe npz must carry the m64 bank sha; every probe triple must
exist in the panel; complement row count must be positive and match
panel_core − probe_core.

Oracle mode (--oracle, run before any stage-2 data exists):
  (a) pooling reproduction: the banked stable-key npz pooled on core
      rows through THIS file reproduces 6.5997 (4 dp);
  (b) planted deltas on synthetic pairs: winner = banked − 0.1 on
      complement rows only ⇒ complement Δ = −0.1 exactly, degenerate
      CI [−0.1, −0.1], verdict REAL; planted 0 ⇒ NOT REAL (line);
      planted −0.03 ⇒ NOT REAL (line, CI clear of 0 — the line is the
      binding clause); probe-only planted gain ⇒ complement Δ = 0
      (the leakage case the complement read exists to kill);
  (c) refusals fire: missing/mismatched tickets sha, banked npz
      carrying _ticket, identity mismatch, probe triple absent from
      the panel.

Pure CPU, read-only on inputs, deterministic (seeded bootstrap).

  uv run python fontaine/scripts/goldenticket_stage2_results.py \\
      --out reports/analysis__goldenticket_stage2.json
  uv run python fontaine/scripts/goldenticket_stage2_results.py --oracle
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
sys.path.insert(0, str(REPO))


def _sibling(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bbr = _sibling("box_batch_results")
fair = _sibling("draws_fairness")

RUN_STEM = "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000"
STAGE2_NPZ = f"{RUN_STEM}__panel_curated_v0_k4l2_ticket33_heun30.npz"
STAGE2_JSON = f"{RUN_STEM}__panel_curated_v0_k4l2_ticket33_heun30.json"
BANKED_NPZ = f"{RUN_STEM}__panel_curated_v0_k4l2_stablekey_heun30.npz"
PROBE_NPZ = f"{RUN_STEM}__drawsprobe_s7_ticket_draws64_heun30_draws.npz"
WINNER_SHA = "a392d630f264c3061ce7f0e246a8803ca8d1c50c64f112a6667a989fe4af1fa5"
BANK_SHA = "9bb13bc47a92f7cc764e81022a9a7b05dbb9ec391eb9ba8ab14d675c955cc7c0"
STABLEKEY_CHUNK_MAE = 6.5997  # banked anchor, 4 dp
R2_LINE = -0.05  # REAL iff pooled complement delta <= line AND CI95 excludes 0


def _fail(message: str) -> None:
    raise SystemExit(f"ABORT: {message}")


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(REPO / path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def policy_key(d: dict[str, np.ndarray], label: str) -> str:
    keys = [k for k in d if k.startswith("pred:bijou")]
    if len(keys) != 1:
        _fail(f"{label}: expected exactly one bijou prediction column, got {keys}")
    return keys[0]


def check_provenance(
    stage2: dict[str, np.ndarray],
    banked: dict[str, np.ndarray],
    probe: dict[str, np.ndarray],
    stage2_report: dict[str, Any],
) -> tuple[str, str]:
    s2_key = policy_key(stage2, "stage-2 npz")
    if "_ticket" not in s2_key:
        _fail(f"stage-2 policy '{s2_key}' does not carry _ticket — not a ticket read")
    # --dump-predictions carries no ticket fields (only --dump-draws
    # does); the winner-sha provenance lives in the run's report JSON.
    sha = stage2_report.get("tickets_sha256")
    if sha != WINNER_SHA:
        _fail(f"stage-2 report tickets_sha256 {sha!r} != winner sha {WINNER_SHA[:12]}…")
    bk_key = policy_key(banked, "banked npz")
    if "_ticket" in bk_key:
        _fail(f"banked policy '{bk_key}' carries _ticket — stable-key npz required")
    psha = probe.get("tickets_sha256")
    if psha is None or str(np.asarray(psha).ravel()[0]) != BANK_SHA:
        _fail("probe npz does not carry the m64 bank sha — wrong triple source")
    return s2_key, bk_key


def join_identity(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> None:
    """Full-panel npzs must byte-match on the identity columns."""
    for k in ("index", "repo_id", "episode_index", "frame_index", "core"):
        if not np.array_equal(a[k], b[k]):
            _fail(f"identity column '{k}' differs between the paired npzs")
    for k in ("truth", "valid"):
        if not np.array_equal(a[k], b[k]):
            _fail(f"'{k}' differs between the paired npzs — not the same rows")


def triples(d: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack(
        [
            np.asarray(d["repo_id"]).astype("U64"),
            np.asarray(d["episode_index"]).astype("U16"),
            np.asarray(d["frame_index"]).astype("U16"),
        ],
        axis=1,
    )


def complement_mask(
    panel: dict[str, np.ndarray],
    probe: dict[str, np.ndarray],
) -> np.ndarray:
    """Panel core rows minus the probe rows, keyed on the identity
    triple. Every probe triple must exist in the panel."""
    panel_t = ["|".join(row) for row in triples(panel)]
    probe_t = {"|".join(row) for row in triples(probe)}
    index = {t: i for i, t in enumerate(panel_t)}
    missing = [t for t in probe_t if t not in index]
    if missing:
        _fail(
            f"{len(missing)} probe triple(s) absent from the panel "
            f"(first: {sorted(missing)[:2]}) — wrong plan pairing",
        )
    in_probe = np.array([t in probe_t for t in panel_t], dtype=bool)
    core = panel["core"].astype(bool)
    comp = core & ~in_probe
    if comp.sum() == 0:
        _fail("complement is empty")
    expected = int(core.sum() - (in_probe & core).sum())
    if int(comp.sum()) != expected:
        _fail(f"complement count {int(comp.sum())} != core - probe_core {expected}")
    return comp


def paired_read(
    stage2: dict[str, np.ndarray],
    banked: dict[str, np.ndarray],
    probe: dict[str, np.ndarray],
    stage2_report: dict[str, Any],
) -> dict[str, Any]:
    s2_key, bk_key = check_provenance(stage2, banked, probe, stage2_report)
    join_identity(stage2, banked)
    mask = fair.element_mask(stage2["truth"], stage2["valid"])
    f_win = fair.frame_mae(stage2[s2_key], stage2["truth"], mask)
    f_bank = fair.frame_mae(banked[bk_key], banked["truth"], mask)
    comp = complement_mask(stage2, probe)
    core = stage2["core"].astype(bool)
    probe_core = core & ~comp

    deltas = (f_win - f_bank)[comp]
    lo, hi = bbr.bootstrap_ci(deltas)
    pooled = float(deltas.mean())
    real = bool(pooled <= R2_LINE and (hi < 0 or lo > 0) and hi < 0)

    err_w = np.abs(stage2[s2_key] - stage2["truth"]) * mask
    err_b = np.abs(banked[bk_key] - banked["truth"]) * mask
    horizon_w = fair.step_curve(err_w[comp], stage2["valid"][comp])
    horizon_b = fair.step_curve(err_b[comp], banked["valid"][comp])
    first_w = err_w[comp][:, 0, :].sum(-1) / np.maximum(
        mask[comp][:, 0, :].sum(-1),
        1,
    )
    first_b = err_b[comp][:, 0, :].sum(-1) / np.maximum(
        mask[comp][:, 0, :].sum(-1),
        1,
    )

    return {
        "inputs": {
            "stage2_npz": STAGE2_NPZ,
            "banked_npz": BANKED_NPZ,
            "probe_npz": PROBE_NPZ,
            "winner_tickets_sha256": WINNER_SHA,
            "policy": s2_key,
            "banked_policy": bk_key,
        },
        "rows": {
            "panel": len(core),
            "core": int(core.sum()),
            "probe_core_excluded": int(probe_core.sum()),
            "complement": int(comp.sum()),
        },
        "r2": {
            "delta_pooled": round(pooled, 5),
            "ci95": [round(lo, 5), round(hi, 5)],
            "line": R2_LINE,
            "verdict": "REAL" if real else "NOT-CONFIRMED",
        },
        "board_continuity": {
            "full_panel_winner_chunk": round(
                fair.pooled_mae(stage2[s2_key], stage2["truth"], mask),
                4,
            ),
            "full_panel_banked_chunk": round(
                fair.pooled_mae(banked[bk_key], banked["truth"], mask),
                4,
            ),
            "probe_row_delta_selection_biased": round(
                float((f_win - f_bank)[probe_core].mean()),
                5,
            ),
        },
        "record_only": {
            "complement_horizon_winner": [round(v, 4) for v in horizon_w],
            "complement_horizon_banked": [round(v, 4) for v in horizon_b],
            "first_mae_delta": round(float((first_w - first_b).mean()), 5),
        },
    }


# ----------------------------------------------------------------- oracle


def _expect_abort(fn: Any, tag: str) -> None:
    try:
        fn()
    except SystemExit as e:
        print(f"  oracle abort-branch '{tag}' fired: {e}")
        return
    raise AssertionError(f"abort branch '{tag}' did NOT fire")


def _tiny_pair(
    n: int = 12,
    probe_rows: int = 4,
    plant: float = 0.0,
    probe_plant: float = 0.0,
) -> tuple[dict, dict, dict]:
    rng = np.random.default_rng(7)
    truth = rng.normal(size=(n, 5, 3)).astype(np.float32)
    valid = np.ones((n, 5), dtype=bool)
    base = truth + 1.0  # banked |err| = 1 everywhere
    ident = {
        "index": np.arange(n),
        "repo_id": np.array([f"r{i % 3}" for i in range(n)]),
        "episode_index": np.arange(n) // 2,
        "frame_index": np.arange(n),
        "core": np.ones(n, dtype=bool),
        "truth": truth,
        "valid": valid,
    }
    win = base.copy()
    win[probe_rows:] -= plant  # complement rows
    win[:probe_rows] -= probe_plant  # probe rows
    stage2 = {**ident, "pred:bijou@80000_ticket": win}
    banked = {**ident, "pred:bijou@80000": base}
    probe = {
        "repo_id": ident["repo_id"][:probe_rows],
        "episode_index": ident["episode_index"][:probe_rows],
        "frame_index": ident["frame_index"][:probe_rows],
        "tickets_sha256": np.array(BANK_SHA),
    }
    return stage2, banked, probe


def run_oracles() -> None:
    print("oracle (a): banked stable-key pooled on core reproduces the anchor")
    banked = load_npz(BANKED_NPZ)
    key = policy_key(banked, "banked npz")
    mask = fair.element_mask(banked["truth"], banked["valid"])
    core = banked["core"].astype(bool)
    got = round(
        fair.pooled_mae(banked[key][core], banked["truth"][core], mask[core]),
        4,
    )
    assert got == STABLEKEY_CHUNK_MAE, f"pooled {got} != {STABLEKEY_CHUNK_MAE}"
    print(f"  reproduced {got}")

    print("oracle (b): planted deltas")
    report = {"tickets_sha256": WINNER_SHA}
    s2, bk, pr = _tiny_pair(plant=0.1)
    out = paired_read(s2, bk, pr, report)
    assert out["r2"]["delta_pooled"] == -0.1, out["r2"]
    assert out["r2"]["ci95"] == [-0.1, -0.1], out["r2"]
    assert out["r2"]["verdict"] == "REAL", out["r2"]
    assert out["rows"]["complement"] == 8 and out["rows"]["probe_core_excluded"] == 4
    print(f"  planted -0.1 -> {out['r2']}")

    s2, bk, pr = _tiny_pair(plant=0.0)
    out = paired_read(s2, bk, pr, report)
    assert out["r2"]["delta_pooled"] == 0.0 and out["r2"]["verdict"] == "NOT-CONFIRMED"
    print(f"  planted 0 -> {out['r2']['verdict']}")

    s2, bk, pr = _tiny_pair(plant=0.03)
    out = paired_read(s2, bk, pr, report)
    assert out["r2"]["delta_pooled"] == -0.03
    assert out["r2"]["verdict"] == "NOT-CONFIRMED", "line must bind at -0.03"
    print(f"  planted -0.03 -> {out['r2']['verdict']} (line binds, CI clear of 0)")

    s2, bk, pr = _tiny_pair(plant=0.0, probe_plant=0.5)
    out = paired_read(s2, bk, pr, report)
    assert out["r2"]["delta_pooled"] == 0.0 and out["r2"]["verdict"] == "NOT-CONFIRMED"
    assert out["board_continuity"]["probe_row_delta_selection_biased"] == -0.5
    print("  probe-only gain -> complement 0 (leakage killed), probe -0.5 recorded")

    print("oracle (c): refusals")
    s2, bk, pr = _tiny_pair()
    _expect_abort(
        lambda: paired_read(s2, bk, pr, {"tickets_sha256": "deadbeef"}),
        "wrong winner sha",
    )
    _expect_abort(lambda: paired_read(s2, bk, pr, {}), "missing winner sha")
    bad_bk = dict(bk)
    bad_bk["pred:bijou@80000_ticket"] = bad_bk.pop("pred:bijou@80000")
    _expect_abort(lambda: paired_read(s2, bad_bk, pr, report), "banked carries _ticket")
    bad_bk = dict(bk)
    bad_bk["frame_index"] = bad_bk["frame_index"] + 1
    _expect_abort(lambda: paired_read(s2, bad_bk, pr, report), "identity mismatch")
    bad_pr = dict(pr)
    bad_pr["frame_index"] = bad_pr["frame_index"] + 100
    _expect_abort(lambda: paired_read(s2, bk, bad_pr, report), "probe triple absent")

    print("ORACLES GREEN")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "reports/analysis__goldenticket_stage2.json",
    )
    args = parser.parse_args()
    if args.oracle:
        run_oracles()
        return
    out = paired_read(
        load_npz(STAGE2_NPZ),
        load_npz(BANKED_NPZ),
        load_npz(PROBE_NPZ),
        json.loads((REPO / STAGE2_JSON).read_text()),
    )
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    r2 = out["r2"]
    print(
        f"R2: complement delta {r2['delta_pooled']} CI95 {r2['ci95']} "
        f"(line {r2['line']}) -> {r2['verdict']}",
    )
    print(f"rows: {out['rows']}")
    print(f"board: {out['board_continuity']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
