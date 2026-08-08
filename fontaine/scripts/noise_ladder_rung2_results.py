"""Noise-ladder rung-2 stage-2 frozen reads — ready before the data.

Implements exactly the stage-2 reads of the per-dataset golden-tickets
pre-reg (posts/2026-08-08-prereg-noise-ladder-perdataset.md, "Stage 2 —
one confirm eval (GPU), paired frozen reads" + amendment 1). All reads
are paired per-frame with a seeded bootstrap CI95 (seed 0, 10,000
resamples) **clustered by dataset** — frames within a dataset share the
routing decision, so the resampling unit is the dataset, never the
frame (an unclustered CI would overstate precision).

  1. PRIMARY Δ_route = routed map vs ticket 33, on qualifying datasets'
     held-out COMPLEMENT core rows only (probe rows selected the
     tickets and never judge). Pass = CI95 entirely below 0; anything
     else is the pre-reg falsifier (routing buys nothing at panel
     scale) and read 3 adjudicates why.
  2. Δ_route vs the banked stable-key run on the same rows
     (record-only context; rung 1 banked shared-vs-stable).
  3. Per-dataset win table: fraction of qualifying datasets whose
     routed ticket beats 33 on their held-out rows, vs the 50% null
     (exact two-sided sign test, ties dropped).
  4. Mirrors of read 1 (record-only): per-step horizon curves, and the
     R4b dispersion-quartile geometry. Dispersion source, pinned here
     because complement rows carry no draw stack by construction:
     per-frame valid-weighted std across the stage-1 probe stack
     restricted to the TOP-10 ticket axis (the routing's choice set,
     stage-3 R4b convention), averaged over each qualifying dataset's
     probe rows -> one dispersion per dataset; quartiles are over the
     97 qualifying datasets; each quartile pools Δ_route over its
     members' complement rows; Spearman is dataset-level
     (dispersion vs mean Δ_route).
  5. Execution oracles (abort, never silent): routed npz + report
     carry the m64 bank sha AND ticket_map_sha256 == the amendment-1
     extended map sha (27858421…) whose restriction to the committed
     792 reproduces the pre-registered map sha (15d92935…) exactly;
     added datasets route to 33 only; map covers every panel dataset;
     image ⊆ top-10 + {33}; policy carries _ticketmap (never pools as
     a plain _ticket read); sample_draws == 1; identity + state-copy
     columns byte-match across all three panels; every probe triple
     exists in the panel; rows of datasets mapped to 33 byte-match the
     banked ticket33 npz (matched composition — same plan, same batch
     size); qualifying complement row count == the stage-01 committed
     6,014.

Oracle mode (--oracle, run before any stage-2 data exists):
  (a) banked reproductions through THIS file's pooling: ticket33 /
      stable-key full-panel chunk == 5.6524 / 6.6750 (4 dp), panel
      complement == 14,746 rows, qualifying complement == the
      stage-01 json's 6,014;
  (b) planted worlds on synthetic multi-dataset fixtures: −0.1 on
      qualifying complement rows ⇒ Δ_route = −0.1 exactly, degenerate
      CI, CONFIRMED; planted 0 ⇒ NOT-CONFIRMED; probe-only gain ⇒
      complement Δ = 0 (the leakage case the complement read kills);
      the cluster CI binds (±1-per-dataset world: clustered CI ≫
      frame-bootstrap CI); win-table + exact sign-test arithmetic
      (4/8 ⇒ p = 1; 8/8 ⇒ p = 2⁻⁷);
  (c) refusal branches fire (wrong/missing map shas, restriction
      drift, added-dataset image ≠ {33}, missing _ticketmap, banked
      npz carrying _ticketmap, identity mismatch, non-qualifying rows
      not byte-matching, sample_draws ≠ 1, complement count drift,
      probe triple absent);
  (d) R4b planted geometry: gain ∝ −dispersion recovers Spearman −1
      and monotone quartile means.

Pure CPU, read-only on inputs, deterministic (seeded bootstrap).

  uv run python fontaine/scripts/noise_ladder_rung2_results.py \\
      --out reports/analysis__noise_ladder_rung2.json
  uv run python fontaine/scripts/noise_ladder_rung2_results.py --oracle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(_HERE))

from box_batch_results import BOOT_N, BOOT_SEED, bootstrap_ci
from draws_fairness import element_mask, frame_mae, pooled_mae, spearman, step_curve
from goldenticket_stage2_results import complement_mask, load_npz, policy_key

RUN_STEM = "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000"
ROUTED_NPZ = f"{RUN_STEM}__panel_curated_v0_k4l2_ticketmap_heun30.npz"
ROUTED_JSON = f"{RUN_STEM}__panel_curated_v0_k4l2_ticketmap_heun30.json"
T33_NPZ = f"{RUN_STEM}__panel_curated_v0_k4l2_ticket33_heun30.npz"
STABLE_NPZ = f"{RUN_STEM}__panel_curated_v0_k4l2_stablekey_heun30.npz"
PROBE_NPZ = f"{RUN_STEM}__drawsprobe_s7_ticket_draws64_heun30_draws.npz"
STAGE01_JSON = "reports/analysis__noise_ladder_stage01.json"
EXT_MAP_JSON = "plans/noise_ladder_ticketmap_panel.json"

BANK_SHA = "9bb13bc47a92f7cc764e81022a9a7b05dbb9ec391eb9ba8ab14d675c955cc7c0"
MAP_SHA = "15d9293553ac1a8878e0b7b0c385f03127a518d96e408bc1f496f5d8c4ec2173"
EXT_MAP_SHA = "27858421c6293ccaf4d98405a9e8b1f2182480bc63459fea6e27d1e36e0ec6b7"
TOP10 = [33, 2, 0, 51, 10, 59, 38, 28, 15, 36]
GLOBAL_WINNER = 33

# Banked anchors for oracle (a) — analysis__goldenticket_stage2.json
# board_continuity, full-panel pooling (all rows, element mask).
T33_FULL_PANEL_CHUNK = 5.6524
STABLE_FULL_PANEL_CHUNK = 6.675
PANEL_COMPLEMENT_ROWS = 14746


def _fail(message: str) -> None:
    raise SystemExit(f"ABORT: {message}")


def canonical_map_sha(mapping: dict[str, int]) -> str:
    """Byte-identical to bijou.eval.policies.load_ticket_map's form."""
    return hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()


def clustered_ci(
    deltas: np.ndarray,
    clusters: np.ndarray,
    n: int = BOOT_N,
    seed: int = BOOT_SEED,
) -> tuple[float, float]:
    """Dataset-clustered bootstrap CI95 of the pooled per-frame mean:
    resample whole datasets with replacement, pool every frame of the
    drawn datasets (frames within a dataset share the routing
    decision — the pre-reg's clustering clause)."""
    labels, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=deltas.astype(np.float64))
    counts = np.bincount(inv).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(labels), size=(n, len(labels)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided sign test against the 50% null (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    pmf = [math.comb(n, k) * 0.5**n for k in range(n + 1)]
    p_obs = pmf[wins]
    return float(min(1.0, sum(p for p in pmf if p <= p_obs + 1e-12)))


def _scalar(value: Any) -> str:
    return str(np.asarray(value).ravel()[0])


def check_execution_oracles(
    routed: dict[str, np.ndarray],
    t33: dict[str, np.ndarray],
    stable: dict[str, np.ndarray],
    probe: dict[str, np.ndarray],
    routed_report: dict[str, Any],
    stage01: dict[str, Any],
    extended_map: dict[str, int],
    expected_map_sha: str,
    expected_ext_sha: str,
    bank_sha: str,
    top10: list[int],
) -> tuple[str, str, str, dict[str, int], list[str]]:
    """The pre-reg's item-5 oracles. Returns (routed_key, t33_key,
    stable_key, committed_map, qualifying_datasets)."""
    routed_key = policy_key(routed, "routed npz")
    if not routed_key.endswith("_ticketmap"):
        _fail(
            f"routed policy '{routed_key}' lacks _ticketmap — never pool a "
            "routed read as a plain ticket read",
        )
    t33_key = policy_key(t33, "ticket33 npz")
    if t33_key.endswith("_ticketmap") or not t33_key.endswith("_ticket"):
        _fail(f"ticket33 policy '{t33_key}' is not a plain _ticket read")
    stable_key = policy_key(stable, "stable-key npz")
    if "_ticket" in stable_key:
        _fail(f"stable-key policy '{stable_key}' carries _ticket")

    if _scalar(routed["tickets_sha256"]) != bank_sha:
        _fail(f"routed npz bank sha != m64 {bank_sha[:12]}…")
    if _scalar(routed["ticket_map_sha256"]) != expected_ext_sha:
        _fail(
            f"routed npz map sha != extended {expected_ext_sha[:12]}… "
            "(amendment 1: routed runs carry the panel-total enumeration)",
        )
    if "ticket_map_sha256" in t33 and _scalar(t33["ticket_map_sha256"]) != "":
        _fail("ticket33 npz unexpectedly carries a ticket map sha")
    if routed_report.get("ticket_map_sha256") != expected_ext_sha:
        _fail(
            f"routed report map sha {routed_report.get('ticket_map_sha256')!r} "
            f"!= {expected_ext_sha[:12]}…",
        )
    if routed_report.get("sample_draws") != 1:
        _fail(f"routed report sample_draws {routed_report.get('sample_draws')} != 1")
    if _scalar(probe["tickets_sha256"]) != bank_sha:
        _fail("probe npz does not carry the m64 bank sha — wrong triple source")
    if probe["draws"].ndim != 4 or probe["draws"].shape[1] != 64:
        _fail(f"probe draws stack {probe['draws'].shape} is not the 64-ticket axis")

    state_copy = [k for k in routed if k.startswith("pred:state-copy")]
    for other, label in ((t33, "ticket33"), (stable, "stable-key")):
        for k in (
            "index",
            "repo_id",
            "episode_index",
            "frame_index",
            "core",
            "truth",
            "valid",
            *state_copy,
        ):
            if k not in other:
                _fail(f"{label} npz lacks column '{k}'")
            if not np.array_equal(routed[k], other[k]):
                _fail(
                    f"column '{k}' differs routed vs {label} — not matched composition",
                )

    committed_map = stage01["stage1"]["routing_map"]
    committed_map = {str(k): int(v) for k, v in committed_map.items()}
    if canonical_map_sha(committed_map) != expected_map_sha:
        _fail(f"committed map canonical sha != pre-registered {expected_map_sha[:12]}…")
    if stage01["stage1"].get("routing_map_sha256") != expected_map_sha:
        _fail("stage-01 json's own routing_map_sha256 drifted")
    if canonical_map_sha(extended_map) != expected_ext_sha:
        _fail(f"extended map canonical sha != pinned {expected_ext_sha[:12]}…")
    restriction = {k: v for k, v in extended_map.items() if k in committed_map}
    if restriction != committed_map:
        _fail(
            "extended map restricted to the committed 792 does NOT reproduce "
            "the pre-registered map — the selection drifted",
        )
    added_image = {v for k, v in extended_map.items() if k not in committed_map}
    if not added_image <= {GLOBAL_WINNER}:
        _fail(
            f"amendment datasets route to {sorted(added_image)} — the "
            f"non-qualifying fallback is {GLOBAL_WINNER} only",
        )
    if not set(extended_map.values()) <= set(top10):
        _fail(
            f"map image escapes top-10 + {{33}}: "
            f"{sorted(set(extended_map.values()) - set(top10))}",
        )
    panel_datasets = {str(r) for r in routed["repo_id"]}
    uncovered = sorted(panel_datasets - set(extended_map))
    if uncovered:
        _fail(f"map misses {len(uncovered)} panel dataset(s), first {uncovered[:3]}")

    qualifying = [str(d) for d in stage01["stage1"]["qualifying"]]
    if len(qualifying) != int(stage01["stage1"]["qualifying_datasets"]):
        _fail("stage-01 qualifying list length != its own qualifying_datasets count")
    for ds in qualifying:
        if ds not in committed_map:
            _fail(f"qualifying dataset {ds!r} absent from the committed map")
        if committed_map[ds] not in top10:
            _fail(
                f"qualifying dataset {ds!r} routes to {committed_map[ds]} "
                "outside the top-10",
            )

    # Rows of datasets mapped to 33 must byte-match the banked ticket33
    # run — same plan + batch size = same composition, and ticket-33
    # noise is per-item, so any mismatch means a different run.
    to33 = np.array(
        [extended_map[str(r)] == GLOBAL_WINNER for r in routed["repo_id"]],
        dtype=bool,
    )
    if to33.any() and not np.array_equal(routed[routed_key][to33], t33[t33_key][to33]):
        _fail(
            "rows mapped to ticket 33 do NOT byte-match the banked ticket33 "
            "npz — composition or lineage drifted",
        )

    return routed_key, t33_key, stable_key, committed_map, qualifying


def rung2_reads(
    routed: dict[str, np.ndarray],
    t33: dict[str, np.ndarray],
    stable: dict[str, np.ndarray],
    probe: dict[str, np.ndarray],
    routed_report: dict[str, Any],
    stage01: dict[str, Any],
    extended_map: dict[str, int],
    expected_map_sha: str = MAP_SHA,
    expected_ext_sha: str = EXT_MAP_SHA,
    bank_sha: str = BANK_SHA,
    top10: list[int] | None = None,
) -> dict[str, Any]:
    top10 = TOP10 if top10 is None else top10
    routed_key, t33_key, stable_key, committed_map, qualifying = (
        check_execution_oracles(
            routed,
            t33,
            stable,
            probe,
            routed_report,
            stage01,
            extended_map,
            expected_map_sha,
            expected_ext_sha,
            bank_sha,
            top10,
        )
    )

    mask = element_mask(routed["truth"], routed["valid"])
    f_routed = frame_mae(routed[routed_key], routed["truth"], mask)
    f_t33 = frame_mae(t33[t33_key], t33["truth"], mask)
    f_stable = frame_mae(stable[stable_key], stable["truth"], mask)

    comp = complement_mask(routed, probe)  # panel core minus probe triples
    repo = np.asarray([str(r) for r in routed["repo_id"]])
    qual_set = set(qualifying)
    qual = np.array([r in qual_set for r in repo], dtype=bool)
    qcomp = comp & qual
    expected_rows = int(stage01["stage1"]["qualifying_complement_rows"])
    if int(qcomp.sum()) != expected_rows:
        _fail(
            f"qualifying complement rows {int(qcomp.sum())} != stage-01 "
            f"committed {expected_rows}",
        )

    core = routed["core"].astype(bool)
    qual_probe_core = core & ~comp & qual

    # ---- read 1 (PRIMARY): Δ_route = routed − ticket33, clustered CI
    d_route = (f_routed - f_t33)[qcomp]
    clusters = repo[qcomp]
    pooled1 = float(d_route.mean())
    lo1, hi1 = clustered_ci(d_route, clusters)
    confirmed = bool(hi1 < 0.0)

    # ---- read 2: Δ vs stable-key (record-only)
    d_stable = (f_routed - f_stable)[qcomp]
    pooled2 = float(d_stable.mean())
    lo2, hi2 = clustered_ci(d_stable, clusters)

    # ---- read 3: per-dataset win table + exact sign test
    per_dataset: dict[str, dict[str, Any]] = {}
    means: dict[str, float] = {}
    for ds in sorted(qual_set):
        rows = qcomp & (repo == ds)
        mean_delta = float((f_routed - f_t33)[rows].mean())
        means[ds] = mean_delta
        per_dataset[ds] = {
            "ticket": committed_map[ds],
            "rows": int(rows.sum()),
            "delta_route": round(mean_delta, 5),
        }
    wins = sum(1 for v in means.values() if v < 0)
    losses = sum(1 for v in means.values() if v > 0)
    ties = len(means) - wins - losses
    win_rate = wins / max(wins + losses, 1)
    p_sign = sign_test_p(wins, losses)

    # ---- read 4: horizon + dispersion-quartile mirrors (record-only)
    err_routed = np.abs(routed[routed_key] - routed["truth"]) * mask
    err_t33 = np.abs(t33[t33_key] - t33["truth"]) * mask
    horizon_routed = step_curve(err_routed[qcomp], routed["valid"][qcomp])
    horizon_t33 = step_curve(err_t33[qcomp], t33["valid"][qcomp])

    probe_repo = np.asarray([str(r) for r in probe["repo_id"]])
    probe_mask = element_mask(probe["truth"], probe["valid"])
    stack = probe["draws"][:, np.asarray(top10)]  # the routing's choice set
    nvalid = probe_mask.sum(axis=(1, 2))
    frame_disp = (stack.astype(np.float64).std(axis=1) * probe_mask).sum(
        axis=(1, 2),
    ) / np.maximum(nvalid, 1)
    disp: dict[str, float] = {}
    for ds in sorted(qual_set):
        rows = probe_repo == ds
        if not rows.any():
            _fail(f"qualifying dataset {ds!r} has no probe rows — dispersion undefined")
        disp[ds] = float(frame_disp[rows].mean())
    ds_order = sorted(qual_set)
    disp_arr = np.array([disp[ds] for ds in ds_order])
    gain_arr = np.array([means[ds] for ds in ds_order])
    qs = np.quantile(disp_arr, [0.25, 0.5, 0.75])
    bins = np.digitize(disp_arr, qs)
    quartiles: dict[str, dict[str, Any]] = {}
    for b, label in enumerate(["q1_tight", "q2", "q3", "q4_dispersed"]):
        members = [ds for ds, bi in zip(ds_order, bins, strict=True) if bi == b]
        rows = qcomp & np.isin(repo, members)
        quartiles[label] = {
            "datasets": len(members),
            "rows": int(rows.sum()),
            "dispersion": round(float(disp_arr[bins == b].mean()), 4),
            "delta_route": round(float((f_routed - f_t33)[rows].mean()), 5)
            if rows.any()
            else None,
        }

    return {
        "inputs": {
            "routed_npz": ROUTED_NPZ,
            "ticket33_npz": T33_NPZ,
            "stablekey_npz": STABLE_NPZ,
            "probe_npz": PROBE_NPZ,
            "policy": routed_key,
            "committed_map_sha256": expected_map_sha,
            "ticket_map_sha256": expected_ext_sha,
            "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED, "cluster": "dataset"},
        },
        "rows": {
            "panel": len(core),
            "core": int(core.sum()),
            "qualifying_datasets": len(qualifying),
            "qualifying_complement": int(qcomp.sum()),
            "qualifying_probe_core_excluded": int(qual_probe_core.sum()),
        },
        "read1_primary": {
            "delta_route_pooled": round(pooled1, 5),
            "ci95_clustered": [round(lo1, 5), round(hi1, 5)],
            "pass_rule": "CI95 entirely below 0",
            "verdict": "CONFIRMED" if confirmed else "NOT-CONFIRMED",
        },
        "read2_vs_stablekey": {
            "record_only": True,
            "delta_pooled": round(pooled2, 5),
            "ci95_clustered": [round(lo2, 5), round(hi2, 5)],
        },
        "read3_win_table": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": round(win_rate, 4),
            "p_sign_two_sided": round(p_sign, 6),
            "per_dataset": per_dataset,
        },
        "read4_mirrors": {
            "record_only": True,
            "complement_horizon_routed": [round(v, 4) for v in horizon_routed],
            "complement_horizon_ticket33": [round(v, 4) for v in horizon_t33],
            "dispersion_source": (
                "stage-1 probe stack restricted to the top-10 ticket axis, "
                "valid-weighted per-frame std (stage-3 R4b convention), "
                "averaged per qualifying dataset over its probe rows"
            ),
            "quartiles": quartiles,
            "spearman_dispersion_vs_delta": round(spearman(disp_arr, gain_arr), 4),
        },
        "board_continuity": {
            "full_panel_routed_chunk": round(
                pooled_mae(routed[routed_key], routed["truth"], mask),
                4,
            ),
            "full_panel_ticket33_chunk": round(
                pooled_mae(t33[t33_key], t33["truth"], mask),
                4,
            ),
            "core_routed_chunk": round(
                pooled_mae(routed[routed_key][core], routed["truth"][core], mask[core]),
                4,
            ),
            "core_routed_first": round(
                pooled_mae(
                    routed[routed_key][core][:, :1],
                    routed["truth"][core][:, :1],
                    mask[core][:, :1],
                ),
                4,
            ),
            "probe_row_delta_selection_biased": round(
                float((f_routed - f_t33)[qual_probe_core].mean()),
                5,
            )
            if qual_probe_core.any()
            else None,
        },
    }


# ----------------------------------------------------------------- oracle


def _expect_abort(fn: Any, tag: str) -> None:
    try:
        fn()
    except SystemExit as e:
        print(f"  refusal '{tag}' fired: {e}")
        return
    raise AssertionError(f"refusal '{tag}' did NOT fire")


class _World:
    """Synthetic multi-dataset fixture: n_ds qualifying datasets (p
    probe + c complement core rows each), one non-qualifying probe
    dataset routed to 33, one amendment-added dataset (no probe rows)
    routed to 33."""

    def __init__(
        self,
        n_ds: int = 8,
        p: int = 6,
        c: int = 25,
        plant: float = 0.0,
        probe_plant: float = 0.0,
        per_ds_plant: list[float] | None = None,
        disp_scale: list[float] | None = None,
    ) -> None:
        rng = np.random.default_rng(11)
        self.qualifying = [f"d{i:02d}" for i in range(n_ds)]
        datasets = [*self.qualifying, "nonqual", "added"]
        repo, is_probe = [], []
        for ds in datasets:
            n_probe = 0 if ds == "added" else p
            for _ in range(n_probe):
                repo.append(ds)
                is_probe.append(True)
            for _ in range(c):
                repo.append(ds)
                is_probe.append(False)
        n = len(repo)
        repo = np.array(repo)
        is_probe = np.array(is_probe, dtype=bool)
        truth = rng.normal(size=(n, 5, 3)).astype(np.float32)
        valid = np.ones((n, 5), dtype=bool)
        base = truth + 1.0  # ticket33 |err| = 1 everywhere
        routed_pred = base.copy()
        for i, ds in enumerate(self.qualifying):
            ds_plant = per_ds_plant[i] if per_ds_plant is not None else plant
            rows = (repo == ds) & ~is_probe
            routed_pred[rows] -= ds_plant
            rows = (repo == ds) & is_probe
            routed_pred[rows] -= probe_plant
        ident = {
            "index": np.arange(n),
            "repo_id": repo,
            "episode_index": np.arange(n) // 3,
            "frame_index": np.arange(n),
            "core": np.ones(n, dtype=bool),
            "truth": truth,
            "valid": valid,
            "pred:state-copy": truth + 0.5,
        }
        prov = {
            "tickets_sha256": np.array(BANK_SHA),
        }
        self.committed = dict.fromkeys(self.qualifying, 2)
        self.committed["nonqual"] = 33
        self.extended = {**self.committed, "added": 33}
        self.map_sha = canonical_map_sha(self.committed)
        self.ext_sha = canonical_map_sha(self.extended)
        self.routed = {
            **ident,
            **prov,
            "pred:bijou@80000_ticketmap": routed_pred,
            "ticket_map_sha256": np.array(self.ext_sha),
        }
        self.t33 = {**ident, **prov, "pred:bijou@80000_ticket": base}
        self.stable = {**ident, "pred:bijou@80000": base + 0.2}
        probe_rows = np.flatnonzero(is_probe)
        scales = dict.fromkeys(datasets, 1.0)
        if disp_scale is not None:
            scales.update(dict(zip(self.qualifying, disp_scale, strict=True)))
        noise = rng.normal(size=(len(probe_rows), 64, 5, 3)).astype(np.float32)
        scale_vec = np.array(
            [scales[ds] for ds in repo[probe_rows]],
            dtype=np.float32,
        )
        draws = truth[probe_rows][:, None] + noise * scale_vec[:, None, None, None]
        self.probe = {
            "repo_id": repo[probe_rows],
            "episode_index": ident["episode_index"][probe_rows],
            "frame_index": ident["frame_index"][probe_rows],
            "truth": truth[probe_rows],
            "valid": valid[probe_rows],
            "draws": draws,
            "tickets_sha256": np.array(BANK_SHA),
        }
        self.report = {"ticket_map_sha256": self.ext_sha, "sample_draws": 1}
        self.stage01 = {
            "stage1": {
                "qualifying": self.qualifying,
                "qualifying_datasets": n_ds,
                "qualifying_complement_rows": n_ds * c,
                "routing_map": self.committed,
                "routing_map_sha256": self.map_sha,
            },
        }

    def read(self, **overrides: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "routed": self.routed,
            "t33": self.t33,
            "stable": self.stable,
            "probe": self.probe,
            "routed_report": self.report,
            "stage01": self.stage01,
            "extended_map": self.extended,
            "expected_map_sha": self.map_sha,
            "expected_ext_sha": self.ext_sha,
        }
        kwargs.update(overrides)
        return rung2_reads(**kwargs)


def run_oracles() -> None:
    print("oracle (a): banked reproductions through this file's pooling")
    t33 = load_npz(T33_NPZ)
    stable = load_npz(STABLE_NPZ)
    probe = load_npz(PROBE_NPZ)
    stage01 = json.loads((REPO / STAGE01_JSON).read_text())
    mask = element_mask(t33["truth"], t33["valid"])
    got33 = round(pooled_mae(t33[policy_key(t33, "t33")], t33["truth"], mask), 4)
    assert got33 == T33_FULL_PANEL_CHUNK, f"{got33} != {T33_FULL_PANEL_CHUNK}"
    gots = round(
        pooled_mae(stable[policy_key(stable, "stable")], stable["truth"], mask),
        4,
    )
    assert gots == STABLE_FULL_PANEL_CHUNK, f"{gots} != {STABLE_FULL_PANEL_CHUNK}"
    comp = complement_mask(t33, probe)
    assert int(comp.sum()) == PANEL_COMPLEMENT_ROWS, int(comp.sum())
    qual = {str(d) for d in stage01["stage1"]["qualifying"]}
    repo = np.asarray([str(r) for r in t33["repo_id"]])
    qcomp = comp & np.array([r in qual for r in repo], dtype=bool)
    expected = int(stage01["stage1"]["qualifying_complement_rows"])
    assert int(qcomp.sum()) == expected, f"{int(qcomp.sum())} != {expected}"
    print(
        f"  ticket33 {got33}, stable {gots}, complement {int(comp.sum())}, "
        f"qualifying complement {int(qcomp.sum())}",
    )

    print("oracle (b): planted worlds")
    out = _World(plant=0.1).read()
    r1 = out["read1_primary"]
    assert r1["delta_route_pooled"] == -0.1, r1
    assert r1["ci95_clustered"] == [-0.1, -0.1], r1
    assert r1["verdict"] == "CONFIRMED", r1
    assert out["rows"]["qualifying_complement"] == 8 * 25
    assert out["read3_win_table"]["wins"] == 8
    assert out["read3_win_table"]["p_sign_two_sided"] == round(2 * 0.5**8, 6)
    print(
        f"  planted -0.1 -> {r1['delta_route_pooled']} CI {r1['ci95_clustered']} "
        f"{r1['verdict']}; 8/8 wins p={out['read3_win_table']['p_sign_two_sided']}",
    )

    out = _World(plant=0.0).read()
    assert out["read1_primary"]["delta_route_pooled"] == 0.0
    assert out["read1_primary"]["verdict"] == "NOT-CONFIRMED"
    assert out["read3_win_table"]["ties"] == 8
    print("  planted 0 -> NOT-CONFIRMED (8 ties)")

    out = _World(plant=0.0, probe_plant=0.5).read()
    assert out["read1_primary"]["delta_route_pooled"] == 0.0
    assert out["read1_primary"]["verdict"] == "NOT-CONFIRMED"
    assert out["board_continuity"]["probe_row_delta_selection_biased"] == -0.5
    print("  probe-only gain -> complement 0 (leakage killed), probe -0.5 recorded")

    world = _World(per_ds_plant=[1, 1, 1, 1, -1, -1, -1, -1])
    out = world.read()
    r1 = out["read1_primary"]
    lo, hi = r1["ci95_clustered"]
    frame_lo, frame_hi = bootstrap_ci(
        np.repeat([-1.0, 1.0], 4 * 25),
    )
    assert r1["delta_route_pooled"] == 0.0
    assert (hi - lo) > 3 * (frame_hi - frame_lo), (
        f"clustered CI [{lo},{hi}] not ≫ frame CI [{frame_lo},{frame_hi}]"
    )
    wt = out["read3_win_table"]
    assert wt["wins"] == 4 and wt["losses"] == 4
    assert wt["p_sign_two_sided"] == 1.0
    print(
        f"  ±1-per-dataset world: clustered CI [{lo}, {hi}] ≫ frame CI "
        f"[{round(frame_lo, 3)}, {round(frame_hi, 3)}]; 4/8 wins p=1.0",
    )

    print("oracle (d): R4b planted geometry")
    scales = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    out = _World(
        per_ds_plant=[0.05 * s for s in scales],
        disp_scale=scales,
    ).read()
    r4 = out["read4_mirrors"]
    assert r4["spearman_dispersion_vs_delta"] == -1.0, r4
    q = [
        r4["quartiles"][k]["delta_route"]
        for k in ("q1_tight", "q2", "q3", "q4_dispersed")
    ]
    assert q == sorted(q, reverse=True), f"quartile deltas not monotone: {q}"
    print(f"  gain ∝ -dispersion -> Spearman -1, monotone quartiles {q}")

    print("oracle (c): refusals")
    w = _World(plant=0.1)
    _expect_abort(
        lambda: w.read(routed_report={**w.report, "ticket_map_sha256": "deadbeef"}),
        "wrong report map sha",
    )
    _expect_abort(
        lambda: w.read(routed_report={**w.report, "sample_draws": 10}),
        "sample_draws != 1",
    )
    bad = dict(w.routed)
    bad["ticket_map_sha256"] = np.array("deadbeef")
    _expect_abort(lambda: w.read(routed=bad), "wrong npz map sha")
    bad = dict(w.routed)
    bad["pred:bijou@80000_ticket"] = bad.pop("pred:bijou@80000_ticketmap")
    _expect_abort(lambda: w.read(routed=bad), "routed lacks _ticketmap")
    bad = dict(w.t33)
    bad["pred:bijou@80000_ticketmap"] = bad.pop("pred:bijou@80000_ticket")
    _expect_abort(lambda: w.read(t33=bad), "ticket33 carries _ticketmap")
    bad = dict(w.t33)
    bad["frame_index"] = bad["frame_index"] + 1
    _expect_abort(lambda: w.read(t33=bad), "identity mismatch")

    def with_map(mapping: dict[str, int]) -> dict[str, Any]:
        # Provenance (npz + report shas) made CONSISTENT with the bad
        # map so the map-STRUCTURE oracle itself must be the one that
        # fires — never the sha check (fixture-blindness guard).
        sha = canonical_map_sha(mapping)
        return {
            "extended_map": mapping,
            "expected_ext_sha": sha,
            "routed": {**w.routed, "ticket_map_sha256": np.array(sha)},
            "routed_report": {**w.report, "ticket_map_sha256": sha},
        }

    _expect_abort(
        lambda: w.read(**with_map({**w.extended, w.qualifying[0]: 0})),
        "restriction drift",
    )
    _expect_abort(
        lambda: w.read(**with_map({**w.extended, "added": 2})),
        "added dataset routes off 33",
    )
    bad = dict(w.routed)
    pred = bad["pred:bijou@80000_ticketmap"].copy()
    pred[np.asarray(bad["repo_id"]) == "nonqual"] += 0.25
    bad["pred:bijou@80000_ticketmap"] = pred
    _expect_abort(lambda: w.read(routed=bad), "non-qualifying rows not byte-equal")
    bad_stage01 = json.loads(json.dumps(w.stage01))
    bad_stage01["stage1"]["qualifying_complement_rows"] = 999
    _expect_abort(lambda: w.read(stage01=bad_stage01), "complement count drift")
    bad = dict(w.probe)
    bad["frame_index"] = bad["frame_index"] + 10_000
    _expect_abort(lambda: w.read(probe=bad), "probe triple absent")

    print("ORACLES GREEN")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "reports/analysis__noise_ladder_rung2.json",
    )
    args = parser.parse_args()
    if args.oracle:
        run_oracles()
        return
    from bijou.eval.policies import load_ticket_map

    bank_count = np.load(REPO / "plans/tickets_goldenticket_m64.npz")["tickets"].shape[
        0
    ]
    extended_map, _ = load_ticket_map(REPO / EXT_MAP_JSON, bank_count)
    out = rung2_reads(
        load_npz(ROUTED_NPZ),
        load_npz(T33_NPZ),
        load_npz(STABLE_NPZ),
        load_npz(PROBE_NPZ),
        json.loads((REPO / ROUTED_JSON).read_text()),
        json.loads((REPO / STAGE01_JSON).read_text()),
        extended_map,
    )
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    r1 = out["read1_primary"]
    wt = out["read3_win_table"]
    print(
        f"read 1 (PRIMARY): Δ_route {r1['delta_route_pooled']} "
        f"clustered CI95 {r1['ci95_clustered']} -> {r1['verdict']}",
    )
    print(
        f"read 2 (record): vs stable-key {out['read2_vs_stablekey']['delta_pooled']} "
        f"CI {out['read2_vs_stablekey']['ci95_clustered']}",
    )
    print(
        f"read 3: {wt['wins']}W/{wt['losses']}L/{wt['ties']}T "
        f"win rate {wt['win_rate']} (sign p {wt['p_sign_two_sided']})",
    )
    print(
        f"read 4: Spearman(dispersion, Δ) "
        f"{out['read4_mirrors']['spearman_dispersion_vs_delta']}",
    )
    print(f"rows: {out['rows']}")
    print(f"board: {out['board_continuity']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
