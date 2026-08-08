"""Molmo2 AR 40k endpoint frozen reads — ready before the data.

Implements exactly §5 of the molmo2 pre-reg
(posts/2026-08-06-prereg-molmo2-ar-40k.md), no other numbers:

  * Read 1 (primary): panel pooled chunk/first MAE on core rows vs the
    E2B AR anchor A-s0 7.7966/3.9422 (same panel, same plan).
    Classification: BEATS iff pooled chunk < 7.30; PARITY within
    ±0.5 of 7.7966; WORSE beyond. Paired per-frame Δ + seeded
    bootstrap CI95 via the npz where row alignment holds — a pairing
    failure is REPORTED (paired block absent, pooled classification
    stands), never silently pooled.
  * Read 2 (instrument integrity, not a result): the molmo2 npz's
    state-copy / state-copy-norm columns byte-match the banked A-s0
    panel columns; their pooled values are quoted (banked
    11.7639/2.5851 chunk/first for state-copy).
  * Read 3 (context, narrative only): e4b screen milestone 7.54@10k
    probe family; arm C statedrop 10.50.
  * Decision text per the pre-reg: BEATS ⇒ phase-2 flow-trunk
    candidate; PARITY ⇒ grounding-probe follow-ups decide; WORSE ⇒
    clean null banked, E2B stays mainline.

Oracle mode (--oracle, run before any endpoint data exists):
  (a) anchor reproduction: the banked A-s0 npz pooled on core rows
      through THIS file reproduces 7.7966/3.9422 (4 dp) and its
      state-copy column pools to 11.7847/2.6202 (see constants note);
  (b) planted deltas on synthetic pairs: molmo2 = anchor − 1.0
      everywhere ⇒ paired Δ −1.0 exactly, degenerate CI, BEATS;
      planted +0.2 ⇒ PARITY; planted +0.8 ⇒ WORSE;
  (c) abort/report branches: corrupted state-copy column aborts;
      identity mismatch reports pairing failure without pooling a
      paired Δ (classification still rendered from pooled numbers).

Pure CPU, read-only on inputs, deterministic (seeded bootstrap).

  uv run python fontaine/scripts/molmo2_endpoint_results.py \\
      --out reports/analysis__molmo2_endpoint_k4l2.json
  uv run python fontaine/scripts/molmo2_endpoint_results.py --oracle
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

MOLMO2_NPZ = (
    "reports/eval__fontaine_molmo2_ar_40k_ddp4__step_040000__panel_curated_v0_k4l2.npz"
)
ANCHOR_NPZ = (
    "reports/eval__fontaine_arb_rcond_40k_1xh100__step_040000"
    "__panel_curated_v0_k4l2.npz"
)
ANCHOR_CHUNK = 7.7966  # A-s0 mainline, banked
ANCHOR_FIRST = 3.9422
BEATS_LINE = 7.30  # anchor - 0.5, frozen in the pre-reg
PARITY_BAND = 0.5
# Same-plan state-copy pooled under the panel convention (matches the
# leaderboard control row 11.785/2.620). NOTE, recorded not silent: the
# pre-reg's parenthetical quoted "11.7639/2.5851", which reproduces
# under NO pooling of this plan (core/all-rows/norm all checked) — a
# drafting slip. The operative Read-2 oracle is the BYTE-MATCH of the
# state-copy columns vs the banked same-plan npz (strictly stronger
# than pooled equality); these constants pin the pooled quote.
STATE_COPY_CHUNK = 11.7847
STATE_COPY_FIRST = 2.6202
E4B_CONTEXT = 7.54  # read 3, narrative only
STATEDROP_CONTEXT = 10.50


def _fail(message: str) -> None:
    raise SystemExit(f"ABORT: {message}")


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(REPO / path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def policy_key(d: dict[str, np.ndarray], label: str) -> str:
    """The BARE model column (leaderboard/mainline convention); panel
    evals may also dump a '+fields' conditioned variant — recorded in
    inputs, never read 1."""
    keys = [
        k for k in d if k.startswith("pred:") and "state-copy" not in k and "+" not in k
    ]
    if len(keys) != 1:
        _fail(f"{label}: expected exactly one bare model column, got {keys}")
    return keys[0]


def pooled(d: dict[str, np.ndarray], key: str) -> tuple[float, float]:
    """(chunk, first) pooled on core rows, the panel convention."""
    core = d["core"].astype(bool)
    mask = fair.element_mask(d["truth"], d["valid"])[core]
    pred, truth = d[key][core], d["truth"][core]
    chunk = fair.pooled_mae(pred, truth, mask)
    first = np.abs(pred - truth)[:, 0, :][mask[:, 0, :]].mean()
    return round(float(chunk), 4), round(float(first), 4)


def classify(chunk: float) -> str:
    if chunk < BEATS_LINE:
        return "BEATS"
    if abs(chunk - ANCHOR_CHUNK) <= PARITY_BAND:
        return "PARITY"
    return "WORSE"


DECISION = {
    "BEATS": "Molmo2 becomes the phase-2 flow-trunk candidate (frozen "
    "AR-adapted prefix — kills the -2.7 confound)",
    "PARITY": "grounding-probe follow-ups decide",
    "WORSE": "clean null banked; E2B stays mainline (VLM4VLA: nulls are modal)",
}


def endpoint_reads(
    molmo2: dict[str, np.ndarray],
    anchor: dict[str, np.ndarray],
) -> dict[str, Any]:
    m_key = policy_key(molmo2, "molmo2 npz")
    a_key = policy_key(anchor, "anchor npz")

    # Read 2 first — instrument integrity gates everything else.
    for k in ("pred:state-copy", "pred:state-copy-norm"):
        if k not in molmo2 or k not in anchor:
            _fail(f"'{k}' missing from a npz — not a panel eval dump")
        if not np.array_equal(molmo2[k], anchor[k]):
            _fail(f"'{k}' does NOT byte-match the banked panel — instrument drift")
    sc_chunk, sc_first = pooled(molmo2, "pred:state-copy")

    m_chunk, m_first = pooled(molmo2, m_key)
    verdict = classify(m_chunk)

    # Read 1 paired block — only where row alignment holds. The A-s0
    # npz predates the episode/frame identity columns (added to
    # dump_identity later); alignment uses the columns BOTH carry —
    # index (the concat index under this exact selection), repo_id,
    # core, truth and valid together still pin identical rows.
    shared = [
        k
        for k in ("index", "repo_id", "episode_index", "frame_index", "core")
        if k in molmo2 and k in anchor
    ]
    aligned = (
        all(np.array_equal(molmo2[k], anchor[k]) for k in shared)
        and np.array_equal(molmo2["truth"], anchor["truth"])
        and np.array_equal(molmo2["valid"], anchor["valid"])
    )
    paired: dict[str, Any] | None = None
    if aligned:
        core = molmo2["core"].astype(bool)
        mask = fair.element_mask(molmo2["truth"], molmo2["valid"])
        deltas = (
            fair.frame_mae(molmo2[m_key], molmo2["truth"], mask)
            - fair.frame_mae(anchor[a_key], anchor["truth"], mask)
        )[core]
        lo, hi = bbr.bootstrap_ci(deltas)
        paired = {
            "delta_pooled": round(float(deltas.mean()), 5),
            "ci95": [round(lo, 5), round(hi, 5)],
            "frames": int(core.sum()),
        }

    return {
        "inputs": {
            "molmo2_npz": MOLMO2_NPZ,
            "anchor_npz": ANCHOR_NPZ,
            "policy": m_key,
            "anchor_policy": a_key,
        },
        "read1": {
            "molmo2_chunk": m_chunk,
            "molmo2_first": m_first,
            "anchor_chunk": ANCHOR_CHUNK,
            "anchor_first": ANCHOR_FIRST,
            "beats_line": BEATS_LINE,
            "parity_band": PARITY_BAND,
            "classification": verdict,
            "paired": paired,
            "pairing_failure": not aligned,
            "alignment_columns": [*shared, "truth", "valid"],
            "decision": DECISION[verdict],
        },
        "read2": {
            "state_copy_byte_match": True,
            "state_copy_norm_byte_match": True,
            "state_copy_pooled": [sc_chunk, sc_first],
            "expected_pooled": [STATE_COPY_CHUNK, STATE_COPY_FIRST],
            "prereg_parenthetical_nonreproducing": [11.7639, 2.5851],
        },
        "read3_context": {
            "e4b_probe_10k": E4B_CONTEXT,
            "statedrop_arm_c": STATEDROP_CONTEXT,
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


def _tiny_pair(offset: float) -> tuple[dict, dict]:
    rng = np.random.default_rng(11)
    n = 10
    truth = rng.normal(size=(n, 4, 2)).astype(np.float32)
    valid = np.ones((n, 4), dtype=bool)
    ident = {
        "index": np.arange(n),
        "repo_id": np.array([f"r{i % 2}" for i in range(n)]),
        "episode_index": np.arange(n) // 3,
        "frame_index": np.arange(n),
        "core": np.ones(n, dtype=bool),
        "truth": truth,
        "valid": valid,
        "pred:state-copy": truth + 3.0,
        "pred:state-copy-norm": truth + 2.9,
    }
    anchor = {**ident, "pred:bijou@40000": truth + 2.0}
    molmo2 = {**ident, "pred:molmo2@40000": truth + 2.0 + offset}
    return molmo2, anchor


def run_oracles() -> None:
    print("oracle (a): banked anchor npz reproduces its numbers through this file")
    anchor = load_npz(ANCHOR_NPZ)
    a_chunk, a_first = pooled(anchor, policy_key(anchor, "anchor"))
    assert (a_chunk, a_first) == (ANCHOR_CHUNK, ANCHOR_FIRST), (a_chunk, a_first)
    sc = pooled(anchor, "pred:state-copy")
    assert sc == (STATE_COPY_CHUNK, STATE_COPY_FIRST), sc
    print(f"  reproduced {a_chunk}/{a_first} and state-copy {sc[0]}/{sc[1]}")

    print("oracle (b): planted classifications")
    out = endpoint_reads(*_tiny_pair(-1.0))
    r1 = out["read1"]
    assert r1["paired"] and r1["paired"]["delta_pooled"] == -1.0
    assert r1["paired"]["ci95"] == [-1.0, -1.0]
    assert r1["classification"] == ("BEATS" if r1["molmo2_chunk"] < BEATS_LINE else "?")
    print(f"  planted -1.0 -> chunk {r1['molmo2_chunk']} {r1['classification']}")

    # Classification thresholds on the frozen constants (pure classify).
    assert classify(7.2999) == "BEATS" and classify(7.30) == "PARITY"
    assert classify(8.2966) == "PARITY" and classify(8.2967) == "WORSE"
    print("  classify boundaries exact at 7.30 / 7.7966±0.5")

    print("oracle (c): abort/report branches")
    m, a = _tiny_pair(0.0)
    bad = dict(m)
    bad["pred:state-copy"] = bad["pred:state-copy"] + 1.0
    _expect_abort(lambda: endpoint_reads(bad, a), "state-copy drift")
    m, a = _tiny_pair(0.0)
    m["frame_index"] = m["frame_index"] + 1  # breaks alignment, not integrity
    m["pred:state-copy"] = a["pred:state-copy"]  # keep integrity check green
    out = endpoint_reads(m, a)
    assert out["read1"]["pairing_failure"] and out["read1"]["paired"] is None
    assert out["read1"]["classification"] in ("BEATS", "PARITY", "WORSE")
    print("  pairing failure reported; pooled classification still rendered")

    print("ORACLES GREEN")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "reports/analysis__molmo2_endpoint_k4l2.json",
    )
    args = parser.parse_args()
    if args.oracle:
        run_oracles()
        return
    out = endpoint_reads(load_npz(MOLMO2_NPZ), load_npz(ANCHOR_NPZ))
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    r1 = out["read1"]
    print(
        f"READ 1: molmo2 {r1['molmo2_chunk']}/{r1['molmo2_first']} vs anchor "
        f"{ANCHOR_CHUNK}/{ANCHOR_FIRST} -> {r1['classification']}",
    )
    if r1["paired"]:
        print(f"  paired: {r1['paired']}")
    else:
        print("  PAIRING FAILURE reported — pooled classification only")
    print(f"  decision: {r1['decision']}")
    print(
        f"READ 2: state-copy byte-match OK, pooled {out['read2']['state_copy_pooled']}",
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
