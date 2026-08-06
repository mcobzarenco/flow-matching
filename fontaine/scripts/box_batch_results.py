"""Box-batch 40k results analysis — the pre-registered reads, ready before the data.

Implements exactly the frozen reads of the box batch pre-reg
(posts/2026-08-05-prereg-box-batch-4xh100.md) + the aux-off pre-reg
(posts/2026-08-05-prereg-paired-auxoff-40k.md):

  * per-arm panel chunk_mae / first_mae (headline numbers for the results post)
  * PRIMARY: paired per-frame chunk_mae delta B-s0 - A-s0 on core frames,
    with a seeded frame-level bootstrap CI (the pairing-noise measure)
  * SECONDARY / E5: pairwise per-frame deltas among {A-s0, A-s1, A-s2};
    pooled pairwise |dchunk_mae| vs the pre-registered 0.2 soft / 0.3
    headline bands
  * DECISION RULE (frozen): the aux-off effect |A-s0 - B-s0| is real only
    if it exceeds the LARGEST pairwise replicate delta AND the per-frame
    delta distribution is coherent (not driven by a single repo -
    checked by leave-one-repo-out: sign and threshold must survive every
    single-repo exclusion)
  * sigma_seed = sample std (ddof=1) of the three replicate pooled
    chunk_mae -> feeds the two finalization amendments: E4B adopt band
    max(3*sigma_seed, 0.15) and rig-benchmark slot 2

Headline column = the bare ``pred:bijou@STEP`` (the anchor-convention
column that 5.8026/6.6232 validate); ``+fields`` is reported as a
secondary descriptive only. Each arm's ``--output-json`` report is a
cross-check oracle, not a selector: the npz-recomputed pooled chunk_mae
and first_mae must reproduce that report's ``summaries`` entry for the
policy (|d| < 5e-3) or the run aborts — catching plan/scoring drift
between the box eval and this analysis.

Pooling semantics are byte-identical to flow_vs_ar_paired.py (validated
against the AR-100k 5.8026/2.1431 and flow-80k 6.6232/1.9331 anchors).

Oracle mode (run before any real data existed, recorded in the session
ledger):
  --oracle  runs the machinery on the two banked panel npzs: (a) anchor
            reproduction 5.8026/6.6232 through this file's own pooling,
            (b) degenerate same-npz pairing -> every delta exactly 0,
            bootstrap CI [0, 0], decision "within noise", (c) synthetic
            1.05x error inflation on B -> known-sign, known-magnitude
            delta detected as real against a zero replicate threshold
            (a flat +c prediction shift is NOT a valid synthetic: with
            balanced error signs the MAE shift cancels).

Usage:
  python fontaine/scripts/box_batch_results.py \
      --arm A-s0 NPZ REPORT_JSON --arm A-s1 NPZ REPORT_JSON \
      --arm A-s2 NPZ REPORT_JSON --arm B NPZ REPORT_JSON \
      [--out reports/analysis__box_batch_40k_k4l2.json]

Pure CPU, read-only on inputs. Bootstrap is seeded (0) - deterministic.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PAIR_KEYS = ["truth", "valid", "index", "repo_id", "core"]
ANCHORS = {"ar": (5.8026, 2.1431), "flow": (6.6232, 1.9331)}
BOOT_N = 10_000
BOOT_SEED = 0
BAND_FLOOR = 0.15  # E4B adopt band floor, pre-registered
E5_SOFT = 0.2
E5_HEADLINE = 0.3


def load_arm(npz_path: str, report_path: str | None) -> tuple:
    d = np.load(npz_path, allow_pickle=True)
    pred_keys = [k for k in d.files if k.startswith("pred:bijou@")]
    if not pred_keys:
        sys.exit(f"{npz_path}: no pred:bijou@ key")
    return d, pred_keys, report_path


def masks(d: dict) -> tuple:
    truth, valid, core = d["truth"], d["valid"], d["core"]
    m3 = valid[:, :, None] & np.isfinite(truth).all(-1, keepdims=True)
    w = m3.repeat(truth.shape[2], axis=2).astype(np.float64)
    return truth, valid, core, w


def pooled_chunk(err: np.ndarray, core: np.ndarray, w: np.ndarray) -> float:
    return float(err[core][w.astype(bool)[core]].mean())


def pooled_first(err: np.ndarray, valid: np.ndarray, core: np.ndarray) -> float:
    v0 = valid[:, 0] & core
    return float(err[v0, 0, :].mean())


def frame_mae(err: np.ndarray, w: np.ndarray) -> tuple:
    nvalid = w.sum(axis=(1, 2))
    return (err * w).sum(axis=(1, 2)) / np.maximum(nvalid, 1), nvalid


def pick_headline(
    d: dict,
    pred_keys: list,
    report_path: str | None,
    truth: np.ndarray,
    valid: np.ndarray,
    core: np.ndarray,
    w: np.ndarray,
    label: str,
) -> tuple:
    """Pick the bare (anchor-convention) pred column; cross-check vs the report.

    The headline is the bare ``pred:bijou@STEP`` column — the same column the
    AR-100k 5.8026 / flow-80k 6.6232 anchors validate; ``+fields`` stays a
    secondary descriptive. The arm's ``--output-json`` report is used as an
    oracle, not a selector: our npz-recomputed pooled chunk_mae AND first_mae
    must reproduce the report's ``summaries`` entry for that policy (|d| <
    5e-3) or the run aborts — this catches plan/scoring drift between the box
    eval and this analysis.
    """
    scores = {}
    for k in pred_keys:
        err = np.abs(d[k] - truth)
        scores[k] = (pooled_chunk(err, core, w), pooled_first(err, valid, core))
    bare = [k for k in pred_keys if not k.endswith("+fields")]
    if len(bare) != 1:
        sys.exit(f"{label}: expected exactly one bare pred:bijou@ column, got {bare}")
    key = bare[0]
    if report_path is None:
        # oracle mode: no report to cross-check against
        return scores, key
    policy = key.removeprefix("pred:")
    rep = json.loads(Path(report_path).read_text())
    summ = [s for s in rep.get("summaries", []) if s.get("policy") == policy]
    if len(summ) != 1:
        sys.exit(
            f"{label}: report has {len(summ)} summaries for policy {policy!r} "
            f"(have: {[s.get('policy') for s in rep.get('summaries', [])]})",
        )
    gc, gf = scores[key]
    wc, wf = summ[0]["chunk_mae"], summ[0]["first_mae"]
    if abs(gc - wc) >= 5e-3 or abs(gf - wf) >= 5e-3:
        sys.exit(
            f"{label}: npz-recomputed chunk/first {gc:.4f}/{gf:.4f} do not "
            f"reproduce the report's {wc:.4f}/{wf:.4f} for {policy} — "
            f"plan/scoring drift, stop",
        )
    print(f"{label}: report cross-check OK ({policy} chunk {gc:.4f} first {gf:.4f})")
    return scores, key


def bootstrap_ci(deltas: np.ndarray, n: int = BOOT_N, seed: int = BOOT_SEED) -> tuple:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(deltas), size=(n, len(deltas)))
    means = deltas[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def loro(deltas: np.ndarray, repo: np.ndarray) -> dict:
    """Leave-one-repo-out means + top-repo contribution share."""
    total = deltas.sum()
    n = len(deltas)
    out = {}
    for r in np.unique(repo):
        s = repo == r
        m = float((total - deltas[s].sum()) / max(n - s.sum(), 1))
        out[str(r)] = {"n_excl": int(s.sum()), "mean_without": round(m, 5)}
    return out


def analyze(arms: dict, out_path: str | None) -> dict:
    """arms: label -> (npz dict, headline pred key). Order: A-s0, A-s1, A-s2, B."""
    labels = list(arms)
    base = arms[labels[0]][0]
    for lab, (d, _) in arms.items():
        for k in PAIR_KEYS:
            if not np.array_equal(base[k], d[k]):
                sys.exit(f"pairing broken on {k} for arm {lab}")
    truth, valid, core, w = masks(base)
    repo = base["repo_id"]

    frame, pooled = {}, {}
    for lab, (d, key) in arms.items():
        err = np.abs(d[key] - truth)
        f, _nvalid = frame_mae(err, w)
        frame[lab] = f
        pooled[lab] = {
            "chunk_mae": round(pooled_chunk(err, core, w), 4),
            "first_mae": round(pooled_first(err, valid, core), 4),
            "pred_key": key,
        }
    keep = (w.sum(axis=(1, 2)) > 0) & core

    # PRIMARY: B - A-s0 paired per-frame on core frames
    a0, b = labels[0], labels[-1]
    d_ab = (frame[b] - frame[a0])[keep]
    ci = bootstrap_ci(d_ab)
    primary = {
        "mean": round(float(d_ab.mean()), 5),
        "median": round(float(np.median(d_ab)), 5),
        "ci95": [round(ci[0], 5), round(ci[1], 5)],
        "b_win_rate": round(float((d_ab < 0).mean()), 4),
        "n_frames": int(keep.sum()),
    }

    # SECONDARY / E5: pairwise replicate deltas among the three controls
    reps = labels[:3]
    pairwise = {}
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            dd = (frame[reps[j]] - frame[reps[i]])[keep]
            pairwise[f"{reps[i]}~{reps[j]}"] = {
                "mean": round(float(dd.mean()), 5),
                "abs_mean": round(abs(float(dd.mean())), 5),
                "pooled_abs_dchunk": round(
                    abs(pooled[reps[j]]["chunk_mae"] - pooled[reps[i]]["chunk_mae"]),
                    5,
                ),
            }
    max_rep_delta = max(v["abs_mean"] for v in pairwise.values())
    max_pooled_rep = max(v["pooled_abs_dchunk"] for v in pairwise.values())
    e5 = {
        "max_pooled_pairwise_abs_dchunk": max_pooled_rep,
        "soft_expectation_le": E5_SOFT,
        "headline_if_gt": E5_HEADLINE,
        "verdict": (
            "within soft expectation"
            if max_pooled_rep <= E5_SOFT
            else (
                "HEADLINE: seed noise larger than assumed"
                if max_pooled_rep > E5_HEADLINE
                else "between bands"
            )
        ),
    }

    # sigma_seed for the finalization amendments
    rep_chunks = [pooled[r]["chunk_mae"] for r in reps]
    rep_firsts = [pooled[r]["first_mae"] for r in reps]
    sigma_seed = float(np.std(rep_chunks, ddof=1))
    amendments = {
        "replicate_chunk_maes": rep_chunks,
        "replicate_first_maes": rep_firsts,
        "sigma_seed_chunk": round(sigma_seed, 5),
        "sigma_seed_first": round(float(np.std(rep_firsts, ddof=1)), 5),
        "e4b_adopt_band": round(max(3 * sigma_seed, BAND_FLOOR), 5),
        "e4b_adopt_band_rule": "max(3*sigma_seed, 0.15) below AR anchor 5.8026",
    }

    # DECISION RULE (frozen)
    effect = abs(primary["mean"])
    exceeds = effect > max_rep_delta
    lo = loro(d_ab, repo[keep])
    sign = np.sign(primary["mean"])
    coherent = (
        all(
            np.sign(v["mean_without"]) == sign
            and abs(v["mean_without"]) > max_rep_delta
            for v in lo.values()
        )
        if exceeds and sign != 0
        else False
    )
    if not exceeds:
        verdict = "within seed noise at 40k/eff-10 — idea #6's 40k rung closes as action-neutral"
    elif not coherent:
        verdict = "exceeds replicate deltas but NOT repo-coherent — no real-effect call"
    elif primary["mean"] > 0:
        verdict = "REAL: aux-off WORSE — aux supervision helps actions (representation shaping)"
    else:
        verdict = "REAL: aux-off BETTER — the aux term taxes actions at weight 0.5"
    influential = sorted(
        lo.items(),
        key=lambda kv: abs(kv[1]["mean_without"] - primary["mean"]),
        reverse=True,
    )[:5]
    decision = {
        "effect_abs": round(effect, 5),
        "threshold_max_pairwise_replicate": max_rep_delta,
        "exceeds_threshold": bool(exceeds),
        "repo_coherent": bool(coherent),
        "most_influential_repos": [{"repo": k, **v} for k, v in influential],
        "verdict": verdict,
    }

    out = {
        "pooled": pooled,
        "primary_B_minus_As0": primary,
        "pairwise_replicates": pairwise,
        "E5_noise_floor": e5,
        "finalization_amendments": amendments,
        "decision": decision,
    }
    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=1))
        print(f"wrote {out_path}")
    print(
        json.dumps(
            {
                k: out[k]
                for k in [
                    "pooled",
                    "primary_B_minus_As0",
                    "E5_noise_floor",
                    "finalization_amendments",
                ]
            },
            indent=1,
        ),
    )
    print(f"\nDECISION: {decision['verdict']}")
    print(
        f"  effect {decision['effect_abs']} vs replicate threshold {max_rep_delta}; coherent={coherent}",
    )
    return out


def oracle() -> None:
    ar_p = "reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2.npz"
    fl_p = "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_k4l2_heun30.npz"
    ar = np.load(ar_p, allow_pickle=True)
    fl = np.load(fl_p, allow_pickle=True)

    # (a) anchors through THIS file's pooling
    for lab, d, key, (wc, wf) in [
        ("ar", ar, "pred:bijou@100000", ANCHORS["ar"]),
        ("flow", fl, "pred:bijou@80000", ANCHORS["flow"]),
    ]:
        truth, valid, core, w = masks(d)
        err = np.abs(d[key] - truth)
        gc, gf = pooled_chunk(err, core, w), pooled_first(err, valid, core)
        ok = abs(gc - wc) < 5e-3 and abs(gf - wf) < 5e-3
        print(
            f"oracle anchor {lab}: chunk {gc:.4f} (want {wc}) first {gf:.4f} (want {wf}) {'OK' if ok else 'FAIL'}",
        )
        if not ok:
            sys.exit("anchor FAIL — pooling semantics wrong, stop")

    # (b) degenerate: same npz as all four arms -> all deltas exactly 0
    arms = dict.fromkeys(["A-s0", "A-s1", "A-s2", "B"], (ar, "pred:bijou@100000"))
    res = analyze(arms, None)
    assert res["primary_B_minus_As0"]["mean"] == 0.0, "degenerate delta nonzero"
    assert res["primary_B_minus_As0"]["ci95"] == [0.0, 0.0], "degenerate CI nonzero"
    assert res["finalization_amendments"]["sigma_seed_chunk"] == 0.0
    assert res["finalization_amendments"]["e4b_adopt_band"] == BAND_FLOOR
    assert not res["decision"]["exceeds_threshold"]
    assert "within seed noise" in res["decision"]["verdict"]
    print(
        "oracle degenerate: all-zero deltas, CI [0,0], band floor 0.15, within-noise verdict OK",
    )

    # (c) synthetic error inflation on B: pred' = truth + 1.05*(pred - truth)
    # scales every valid element's abs error by exactly 1.05, so the paired
    # frame-level delta mean is 0.05 * mean frame MAE (~0.29 here) and the
    # pooled chunk_mae is exactly 1.05 * 5.8026. (A flat +c on predictions is
    # NOT a valid synthetic — balanced error signs cancel the MAE shift.)
    sh = {k: ar[k] for k in ar.files}
    sh["pred:bijou@100000"] = ar["truth"] + 1.05 * (
        ar["pred:bijou@100000"] - ar["truth"]
    )
    arms = {
        "A-s0": (ar, "pred:bijou@100000"),
        "A-s1": (ar, "pred:bijou@100000"),
        "A-s2": (ar, "pred:bijou@100000"),
        "B": (sh, "pred:bijou@100000"),
    }
    res = analyze(arms, None)
    m = res["primary_B_minus_As0"]["mean"]
    bc = res["pooled"]["B"]["chunk_mae"]
    assert abs(bc - 1.05 * 5.8026) < 5e-3, f"B pooled {bc} != 1.05*anchor"
    assert 0.2 < m < 0.4, f"synthetic inflation mean {m} outside expectation"
    assert res["decision"]["exceeds_threshold"] and res["decision"]["repo_coherent"]
    assert "aux-off WORSE" in res["decision"]["verdict"]
    print(
        f"oracle synthetic: 1.05x error inflation -> delta +{m:.5f}, B chunk {bc:.4f} = 1.05*anchor, real + coherent, correct sign OK",
    )

    # (d) report cross-check path on the real AR pair: bare column picked,
    # npz-recomputed numbers must reproduce the report's summaries entry.
    ar_json = "reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2.json"
    truth, valid, core, w = masks(ar)
    pred_keys = [k for k in ar.files if k.startswith("pred:bijou@")]
    _scores, key = pick_headline(
        ar,
        pred_keys,
        ar_json,
        truth,
        valid,
        core,
        w,
        "oracle-ar",
    )
    assert key == "pred:bijou@100000", f"bare-column pick wrong: {key}"
    print("oracle report cross-check: bare column picked + report reproduced OK")
    print("\nORACLE: all four checks PASSED")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--arm",
        nargs=3,
        action="append",
        metavar=("LABEL", "NPZ", "REPORT_JSON"),
    )
    p.add_argument("--out", default="reports/analysis__box_batch_40k_k4l2.json")
    p.add_argument("--oracle", action="store_true")
    a = p.parse_args()
    if a.oracle:
        oracle()
        return
    if not a.arm or len(a.arm) != 4:
        sys.exit(
            "need exactly four --arm LABEL NPZ REPORT_JSON (A-s0, A-s1, A-s2, B — in that order)",
        )
    arms = {}
    for label, npz_path, report_path in a.arm:
        d, pred_keys, _ = load_arm(npz_path, report_path)
        truth, valid, core, w = masks(d)
        scores, key = pick_headline(
            d,
            pred_keys,
            report_path,
            truth,
            valid,
            core,
            w,
            label,
        )
        arms[label] = (d, key)
        print(
            f"{label}: headline column {key} (all columns: { {k: round(c, 4) for k, (c, _) in scores.items()} })",
        )
    analyze(arms, a.out)


if __name__ == "__main__":
    main()
