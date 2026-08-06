"""State-reliance probe results — the pre-registered reads, oracled before use.

Implements exactly the frozen reads of the state-reliance probe pre-reg
(posts/2026-08-06-prereg-state-reliance-probe.md, ideas #11 rung (a)):

  * per arm, per column (chunk_mae, first_mae): Delta = masked - intact,
    paired per-row over the 4,301 subset rows, seeded bootstrap 95% CI
    (seed 0, 10,000 resamples)
  * PRIMARY: D = Delta_first(B) - Delta_first(A-s0), paired per-row
    (the per-row double difference), bootstrap CI. *Supported* iff the
    CI excludes 0 AND D >= 0.05 first_mae degrees.
  * secondary: the same double difference on chunk_mae; per-arm absolute
    reliance (all four Deltas with CIs); AR-100k vs flow-80k reliance;
    masked levels vs the intact state-copy / state-copy-norm baselines
    pooled on the subset rows.

Masked side: each arm's ``--mask-state`` subset npz (headline = the bare
``pred:bijou@STEP_state-masked`` column; ``+fields`` descriptive only).
Intact side: the banked FULL-PANEL npzs pooled over exactly the subset
rows (the panel-v2 re-pooling precedent) — matched by npz ``index`` and
verified on (repo_id, episode_index, frame_index).

Execution oracles (pre-registered, abort on failure):
  (1) each masked run's state-copy AND state-copy-norm prediction arrays
      byte-match the banked arrays on the subset rows (proves row pairing
      and that masking touched only the bijou policy), and the masked
      report JSON's baseline summaries reproduce this file's pooling
      (|d| < 5e-3, the box_batch_results drift-oracle convention);
  (2) the masked report JSON records ``mask_state: true`` and the policy
      name carries ``_state-masked``;
  (3) the subset plan file matches the frozen sha256.

Oracle mode (--oracle, run before the real read):
  (a) degenerate no-masking: synthetic masked = banked subset rows ->
      every Delta exactly 0, CI [0, 0], D = 0, verdict not supported;
  (b) synthetic known effect: 1.10x error inflation on B's masked arm
      -> Delta_first(B) = 0.10 * intact first level, D > 0.05, CI
      excludes 0, verdict SUPPORTED; the SAME inflation applied to both
      B and A-s0 -> D = 0 again (the double difference subtracts the
      common effect) — not supported;
  (c) pairing break: row-shuffled masked npz -> hard abort.

Pooling semantics are byte-identical to box_batch_results.py (validated
against the AR-100k 5.8026/2.1431 and flow-80k 6.6232/1.9331 anchors).
Pure CPU, read-only on inputs. Deterministic (seeded bootstrap).

Usage:
  python fontaine/scripts/state_probe_results.py \
      --arm AR-100k MASKED_NPZ MASKED_JSON BANKED_NPZ \
      --arm flow-80k ... --arm A-s0 ... --arm B ... \
      [--plan plans/holdout_curated_v0_k4l2_stateprobe_q4.json] \
      [--out reports/analysis__state_probe_q4.json]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PLAN_DEFAULT = "plans/holdout_curated_v0_k4l2_stateprobe_q4.json"
PLAN_SHA256 = "876c39c8fe2b3cb16945a40c35ec157c907b4f7417e7dfd0b6cf46dd47355ef5"
SUBSET_ROWS = 4301
BOOT_N = 10_000
BOOT_SEED = 0
D_MIN = 0.05  # pre-registered support threshold on D (first_mae degrees)
BASELINES = ["state-copy", "state-copy-norm"]
SUMMARY_TOL = 5e-3


def masks(d: dict) -> tuple:
    truth, valid = d["truth"], d["valid"]
    m3 = valid[:, :, None] & np.isfinite(truth).all(-1, keepdims=True)
    w = m3.repeat(truth.shape[2], axis=2).astype(np.float64)
    return truth, valid, w


def pooled_chunk(err: np.ndarray, w: np.ndarray) -> float:
    return float(err[w.astype(bool)].mean())


def pooled_first(err: np.ndarray, valid: np.ndarray) -> float:
    v0 = valid[:, 0]
    return float(err[v0, 0, :].mean())


def row_chunk_mae(err: np.ndarray, w: np.ndarray) -> np.ndarray:
    nvalid = w.sum(axis=(1, 2))
    return (err * w).sum(axis=(1, 2)) / np.maximum(nvalid, 1)


def row_first_mae(err: np.ndarray) -> np.ndarray:
    return err[:, 0, :].mean(axis=1)


def bootstrap_ci(rows: np.ndarray, n: int = BOOT_N, seed: int = BOOT_SEED) -> tuple:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rows), size=(n, len(rows)))
    means = rows[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def assert_plan(plan_path: str) -> None:
    got = hashlib.sha256(Path(plan_path).read_bytes()).hexdigest()
    if got != PLAN_SHA256:
        sys.exit(f"subset plan sha256 mismatch: {got} != frozen {PLAN_SHA256}")
    print(f"plan sha256 OK: {plan_path}")


def load_masked(npz_path: str, json_path: str | None, label: str) -> tuple:
    d = np.load(npz_path, allow_pickle=True)
    if d["truth"].shape[0] != SUBSET_ROWS:
        sys.exit(
            f"{label}: masked npz has {d['truth'].shape[0]} rows, want {SUBSET_ROWS}",
        )
    if not d["core"].all():
        sys.exit(f"{label}: masked npz has non-core rows")
    bare = [
        k
        for k in d.files
        if k.startswith("pred:bijou@") and k.endswith("_state-masked")
    ]
    if len(bare) != 1:
        sys.exit(
            f"{label}: expected exactly one bare *_state-masked column, got {bare}",
        )
    key = bare[0]
    if json_path is not None:
        rep = json.loads(Path(json_path).read_text())
        if rep.get("mask_state") is not True:
            sys.exit(f"{label}: report JSON does not record mask_state=true")
        policy = key.removeprefix("pred:")
        summ = {s["policy"]: s for s in rep.get("summaries", [])}
        if policy not in summ:
            sys.exit(f"{label}: report has no summary for {policy!r}")
        truth, valid, w = masks(d)
        err = np.abs(d[key] - truth)
        gc, gf = pooled_chunk(err, w), pooled_first(err, valid)
        wc, wf = summ[policy]["chunk_mae"], summ[policy]["first_mae"]
        if abs(gc - wc) >= SUMMARY_TOL or abs(gf - wf) >= SUMMARY_TOL:
            sys.exit(
                f"{label}: recomputed masked chunk/first {gc:.4f}/{gf:.4f} do not "
                f"reproduce the report's {wc:.4f}/{wf:.4f} — scoring drift, stop",
            )
        for pol in BASELINES:
            if pol not in summ:
                sys.exit(f"{label}: report missing baseline summary {pol!r}")
    return d, key


def pair_banked(masked: dict, banked: dict, label: str) -> np.ndarray:
    """Positions of the masked subset rows inside the banked full-panel npz."""
    bi = {int(v): i for i, v in enumerate(banked["index"])}
    try:
        pos = np.array([bi[int(v)] for v in masked["index"]])
    except KeyError as e:
        sys.exit(f"{label}: subset index {e} missing from banked npz")
    if not np.array_equal(banked["repo_id"][pos], masked["repo_id"]):
        sys.exit(f"{label}: banked/masked row identity mismatch on repo_id")
    for k in ["truth", "valid"]:
        if not np.array_equal(banked[k][pos], masked[k]):
            sys.exit(f"{label}: banked/masked {k} not byte-identical — pairing broken")
    for pol in BASELINES:
        if not np.array_equal(banked[f"pred:{pol}"][pos], masked[f"pred:{pol}"]):
            sys.exit(
                f"{label}: {pol} not byte-identical to banked on subset rows — "
                f"masking touched more than the bijou policy, stop",
            )
    print(f"{label}: pairing + baseline byte-match oracles OK")
    return pos


def banked_bare_key(banked: dict, label: str) -> str:
    bare = [
        k
        for k in banked.files
        if k.startswith("pred:bijou@")
        and not k.endswith("+fields")
        and "masked" not in k
    ]
    if len(bare) != 1:
        sys.exit(f"{label}: expected one bare banked pred:bijou@ column, got {bare}")
    return bare[0]


def analyze(arms: dict, out_path: str | None) -> dict:
    """arms: label -> dict(masked npz, masked key, banked npz, banked pos).

    Order must be AR-100k, flow-80k, A-s0, B (primary uses the last two).
    """
    labels = list(arms)
    base = arms[labels[0]]["masked"]
    for lab in labels[1:]:
        m = arms[lab]["masked"]
        for k in ["index", "repo_id", "valid"]:
            if not np.array_equal(base[k], m[k]):
                sys.exit(f"cross-arm pairing broken on {k} for {lab}")
    truth, valid, w = masks(base)
    keep_chunk = w.sum(axis=(1, 2)) > 0
    keep_first = valid[:, 0]

    rows, pooled = {}, {}
    for lab, a in arms.items():
        m_err = np.abs(a["masked"][a["mkey"]] - truth)
        i_err = np.abs(a["banked"][a["bkey"]][a["pos"]] - truth)
        rows[lab] = {
            "chunk": (row_chunk_mae(m_err, w) - row_chunk_mae(i_err, w))[keep_chunk],
            "first": (row_first_mae(m_err) - row_first_mae(i_err))[keep_first],
        }
        pooled[lab] = {
            "masked_chunk": round(pooled_chunk(m_err, w), 4),
            "masked_first": round(pooled_first(m_err, valid), 4),
            "intact_chunk": round(pooled_chunk(i_err, w), 4),
            "intact_first": round(pooled_first(i_err, valid), 4),
            "masked_key": a["mkey"],
            "banked_key": a["bkey"],
        }

    deltas = {}
    for lab in labels:
        deltas[lab] = {}
        for col in ["chunk", "first"]:
            r = rows[lab][col]
            lo, hi = bootstrap_ci(r)
            deltas[lab][col] = {
                "mean": round(float(r.mean()), 5),
                "ci95": [round(lo, 5), round(hi, 5)],
                "n_rows": len(r),
            }

    # PRIMARY: D = Delta_first(B) - Delta_first(A-s0), per-row double difference
    a0, b = labels[2], labels[3]
    reads = {}
    for col in ["chunk", "first"]:
        d_rows = rows[b][col] - rows[a0][col]
        lo, hi = bootstrap_ci(d_rows)
        reads[col] = {
            "D": round(float(d_rows.mean()), 5),
            "ci95": [round(lo, 5), round(hi, 5)],
            "n_rows": len(d_rows),
        }
    d_first, (ci_lo, ci_hi) = reads["first"]["D"], reads["first"]["ci95"]
    excludes0 = ci_lo > 0 or ci_hi < 0
    supported = bool(excludes0 and d_first >= D_MIN)
    primary = {
        "definition": "D = Delta_first(B) - Delta_first(A-s0), paired per-row",
        **reads["first"],
        "ci_excludes_0": bool(excludes0),
        "threshold": D_MIN,
        "verdict": (
            "SUPPORTED: aux-off leans harder on state (D >= 0.05, CI excludes 0)"
            if supported
            else "NOT SUPPORTED: state-dominant bias dropped as B's-flag explanation"
        ),
        "supported": supported,
    }

    # secondary: intact baselines pooled on the subset (identical across arms
    # by the byte-match oracle; quote from the first arm's masked npz)
    baselines = {}
    for pol in BASELINES:
        err = np.abs(base[f"pred:{pol}"] - truth)
        baselines[pol] = {
            "chunk_mae": round(pooled_chunk(err, w), 4),
            "first_mae": round(pooled_first(err, valid), 4),
        }
    vision_only = {
        lab: {
            "masked_first_vs_state_copy_first": round(
                pooled[lab]["masked_first"] - baselines["state-copy"]["first_mae"],
                4,
            ),
            "masked_beats_state_copy_first": bool(
                pooled[lab]["masked_first"] < baselines["state-copy"]["first_mae"],
            ),
        }
        for lab in labels
    }

    expectations = {
        "1_all_delta_chunk_gt_0.5": bool(
            all(deltas[lab]["chunk"]["mean"] > 0.5 for lab in labels),
        ),
        "2_masked_first_above_intact_state_copy_all_arms": bool(
            all(
                not vision_only[lab]["masked_beats_state_copy_first"] for lab in labels
            ),
        ),
        "3_D_positive": bool(d_first > 0),
    }

    out = {
        "subset_rows": SUBSET_ROWS,
        "pooled": pooled,
        "deltas_masked_minus_intact": deltas,
        "primary_D_first": primary,
        "secondary_D_chunk": reads["chunk"],
        "ar_vs_flow_delta_chunk": round(
            deltas[labels[0]]["chunk"]["mean"] - deltas[labels[1]]["chunk"]["mean"],
            5,
        ),
        "intact_baselines_on_subset": baselines,
        "masked_vs_baselines": vision_only,
        "expectations": expectations,
    }
    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=1))
        print(f"wrote {out_path}")
    print(json.dumps(out, indent=1))
    print(f"\nPRIMARY: D = {d_first} CI95 [{ci_lo}, {ci_hi}] -> {primary['verdict']}")
    return out


def make_synthetic_masked(
    banked: dict,
    pos: np.ndarray,
    bkey: str,
    scale: float = 1.0,
) -> "_DictNpz":
    """Banked subset rows repackaged as a fake masked npz (oracle mode)."""
    d = {k: banked[k][pos] for k in ["truth", "valid", "index", "repo_id", "core"]}
    for pol in BASELINES:
        d[f"pred:{pol}"] = banked[f"pred:{pol}"][pos]
    pred = banked[bkey][pos]
    if scale != 1.0:
        pred = d["truth"] + scale * (pred - d["truth"])
    d[bkey + "_state-masked"] = pred
    d["files"] = list(d)  # np.load-like access shim
    return _DictNpz(d)


class _DictNpz(dict):
    @property
    def files(self) -> list:
        return [k for k in self if k != "files"]


def _arm(masked: dict, mkey: str, banked: dict, bkey: str, pos: np.ndarray) -> dict:
    return {"masked": masked, "mkey": mkey, "banked": banked, "bkey": bkey, "pos": pos}


def oracle() -> None:
    ar_p = "reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2.npz"
    ar = np.load(ar_p, allow_pickle=True)
    bkey = banked_bare_key(ar, "oracle-ar")
    # subset = every 4th core row of the frozen plan (positions == 0 mod 4),
    # mirrored here on the banked npz's row order (state_probe_subset_plan.py)
    plan = json.loads(Path(PLAN_DEFAULT).read_text())
    assert len(plan["core"]) == SUBSET_ROWS
    pos = np.where(ar["core"])[0][::4]
    assert len(pos) == SUBSET_ROWS and ar["core"][pos].all()
    mkey = bkey + "_state-masked"

    # (a) degenerate: masked == banked subset rows -> everything exactly 0
    syn = make_synthetic_masked(ar, pos, bkey)
    p2 = pair_banked(syn, ar, "oracle-degenerate")
    assert np.array_equal(p2, pos)
    arms = {
        lab: _arm(syn, mkey, ar, bkey, pos)
        for lab in ["AR-100k", "flow-80k", "A-s0", "B"]
    }
    res = analyze(arms, None)
    for lab in arms:
        for col in ["chunk", "first"]:
            assert res["deltas_masked_minus_intact"][lab][col]["mean"] == 0.0
            assert res["deltas_masked_minus_intact"][lab][col]["ci95"] == [0.0, 0.0]
    assert res["primary_D_first"]["D"] == 0.0
    assert res["primary_D_first"]["ci95"] == [0.0, 0.0]
    assert not res["primary_D_first"]["supported"]
    print("oracle (a) degenerate: all-zero deltas, CI [0,0], not supported OK\n")

    # (b) synthetic known effect: 1.10x error inflation on B only.
    # Delta_first(B) = 0.10 * intact first level (~0.21 >= 0.05), CI > 0.
    infl = make_synthetic_masked(ar, pos, bkey, scale=1.10)
    arms["B"] = _arm(infl, mkey, ar, bkey, pos)
    res = analyze(arms, None)
    want = 0.10 * res["pooled"]["B"]["intact_first"]
    got = res["primary_D_first"]["D"]
    assert abs(got - want) < 5e-3, f"synthetic D {got} != 0.10*intact {want:.5f}"
    assert res["primary_D_first"]["supported"], "synthetic effect not detected"
    # same inflation on BOTH arms -> the double difference cancels exactly
    arms["A-s0"] = _arm(infl, mkey, ar, bkey, pos)
    res = analyze(arms, None)
    assert res["primary_D_first"]["D"] == 0.0, "common-effect subtraction failed"
    assert not res["primary_D_first"]["supported"]
    print("oracle (b) synthetic: known effect detected, common effect cancels OK\n")

    # (c) pairing break: a misaligned index column (rows no longer the frames
    # their index claims) must hard-abort. NB a coherent whole-row shuffle is
    # NOT a violation — pairing is by index, so it still maps correctly.
    perm = np.random.default_rng(1).permutation(SUBSET_ROWS)
    broken = _DictNpz({k: syn[k] for k in syn.files})
    broken["index"] = syn["index"][perm]
    try:
        pair_banked(broken, ar, "oracle-misaligned")
    except SystemExit:
        print("oracle (c) pairing break: misaligned index aborted OK")
    else:
        sys.exit("oracle (c) FAILED: misaligned index was not caught")
    print("\nORACLE: all three checks PASSED")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--arm",
        nargs=4,
        action="append",
        metavar=("LABEL", "MASKED_NPZ", "MASKED_JSON", "BANKED_NPZ"),
    )
    p.add_argument("--plan", default=PLAN_DEFAULT)
    p.add_argument("--out", default="reports/analysis__state_probe_q4.json")
    p.add_argument("--oracle", action="store_true")
    a = p.parse_args()
    if a.oracle:
        oracle()
        return
    if not a.arm or len(a.arm) != 4:
        sys.exit(
            "need exactly four --arm LABEL MASKED_NPZ MASKED_JSON BANKED_NPZ "
            "(AR-100k, flow-80k, A-s0, B — in that order)",
        )
    assert_plan(a.plan)
    arms = {}
    for label, masked_npz, masked_json, banked_npz in a.arm:
        masked, mkey = load_masked(masked_npz, masked_json, label)
        banked = np.load(banked_npz, allow_pickle=True)
        bkey = banked_bare_key(banked, label)
        pos = pair_banked(masked, banked, label)
        arms[label] = _arm(masked, mkey, banked, bkey, pos)
        print(f"{label}: masked {mkey} vs banked {bkey} over {len(pos)} rows")
    analyze(arms, a.out)


if __name__ == "__main__":
    main()
