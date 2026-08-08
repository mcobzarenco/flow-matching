"""Stage-1 re-adjudication for clean-list subgoal-draws (#6 rung (b')).

The rung-(b') pre-reg (2026-08-08-prereg-subgoal-draws-cleanlist,
"Stage 1") makes stage 1 CPU-free: pass 1 is byte-identical to rung
(b)'s (checkpoint/plan/seeds/T unchanged, the filter is selection-side
only), so the BANKED 60-row stage-1 table IS rung (b')'s stage-1 data.
This script re-scores that table through the frozen eligible-list rule
(``bijou.eval.subgoal_scoring.eligible_indices``) and the committed
scorers, and gates stage 2 on the re-scoped bars:

  (a') rows with ≥ 1 eligible SAMPLED candidate      ≥ 90%   (prior 60/60)
  (b') rows with ≥ 2 unique eligible strings         ≥ 50%   (prior 57/60)
  (c') top pooled eligible sampled string            ≤ 50%   (prior 5.4%)
  (d)  eligible candidates subgoal-shaped (eyes)     carried from the
       stage-1 close post's qualitative block.

Because every number here was computed from banked data and written
into the pre-reg BEFORE it froze, the script does not adjudicate a
mismatch — it ABORTS loudly (pre-reg: "a failed bar here would mean
instrument breakage, not new evidence"). The pinned priors double as
the pre-reg's oracles (vii) banked-table pick-invariance — the filter
changes 0/60 self-certainty picks and 0/60 ceiling picks on the real
table — and (x) exact reproduction of every written prior
(60/60, 57/60, 23/425, 0/60 + 0/60).

On green it writes the stage-2 gate JSON (the arms launcher for the
(b') execution refuses to run without it).

``--oracle`` exercises the pass path on the REAL banked table plus
every abort branch via planted mutations of it: a filter-binds world
(the full-list argmax made truncated ⇒ pick-invariance breaks, both
scorers), an all-truncated row (greedy fallback recorded ⇒ bar (a')
prior breaks), banked-pick drift, provenance drift, and a missing
table file. No GPU, no network.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bijou.eval.subgoal_scoring import (
    ceiling_pick,
    eligible_indices,
    self_certainty_pick,
)

TABLE_DEFAULT = "reports/analysis__subgoal_draws_stage1_table.json"
OUT_DEFAULT = "reports/analysis__subgoal_draws_cleanlist_stage1.json"
PREREG = "fontaine/blog/src/posts/2026-08-08-prereg-subgoal-draws-cleanlist.md"

# Written priors — computed from the banked table and frozen in the
# pre-reg BEFORE it posted. Any mismatch is instrument breakage.
PRIORS = {
    "rows": 60,
    "rows_with_truncated_candidate": 40,
    "sc_pick_changed": 0,
    "ceil_pick_changed": 0,
    "a_eligible_sampled_rows": 60,
    "b_diverse_eligible_rows": 57,
    "c_top_eligible_sampled": {
        "text": "retract the arm to the home pose",
        "count": 23,
    },
    "c_eligible_pool": 425,
    "fallback_rows": 0,
}

# Pass-1 identity: the frozen decode inputs the byte-identity argument
# rests on (pre-reg "Stage 1"; table header fields).
PROVENANCE = {
    "checkpoint_suffix": "bijou_arb_rcond_100k_ddp4/step_100000",
    "sample_plan": "plans/holdout_curated_v0_k4l2.json",
    "seed": 0,
    "subgoal_draws": 8,
    "subgoal_temperature": 1.0,
}

A_BAR, B_BAR, C_BAR = 0.90, 0.50, 0.50


def recompute(table: dict) -> dict:
    """Provenance checks + the filtered re-score of the banked table.
    Every abort is a hard SystemExit — this runs pre-gate."""
    checkpoint = str(table.get("checkpoint") or "")
    if not checkpoint.endswith(PROVENANCE["checkpoint_suffix"]):
        sys.exit(
            f"table checkpoint {checkpoint!r} does not end with the frozen "
            f"{PROVENANCE['checkpoint_suffix']!r} — wrong table, stop",
        )
    for key in ("sample_plan", "seed", "subgoal_draws", "subgoal_temperature"):
        if table.get(key) != PROVENANCE[key]:
            sys.exit(
                f"table {key} {table.get(key)!r} != frozen "
                f"{PROVENANCE[key]!r} — pass-1 byte-identity argument "
                "does not hold, stop",
            )
    rows = table.get("rows")
    if not isinstance(rows, list) or len(rows) != PRIORS["rows"]:
        sys.exit(
            f"table has {len(rows) if isinstance(rows, list) else 'no'} rows "
            f"— expected the banked {PRIORS['rows']}, stop",
        )
    binds = sc_changed = ceil_changed = a_rows = b_rows = 0
    fallback_rows = 0
    pick_ne_greedy = 0
    eligible_sizes: list[int] = []
    pool: dict[str, int] = {}
    pool_n = 0
    for row in rows:
        cands = row["candidates"]
        trunc = [bool(c["truncated"]) for c in cands]
        vocab = cands[0]["allowed_vocab"]
        # The banked table's own pick must reproduce from the dumped
        # stats before any filtered claim is made off them.
        sc_full = self_certainty_pick([c["mean_logprob"] for c in cands], vocab)
        if sc_full != row["sc_pick"]:
            sys.exit(
                f"row {row['index']}: banked sc_pick {row['sc_pick']} does "
                f"not reproduce from the dumped stats ({sc_full}) — table "
                "integrity broken, stop",
            )
        elig = eligible_indices(trunc)
        eligible_sizes.append(len(elig))
        if any(trunc):
            binds += 1
        if all(trunc):
            fallback_rows += 1
        sc_filtered = elig[
            self_certainty_pick([cands[i]["mean_logprob"] for i in elig], vocab)
        ]
        sc_changed += sc_filtered != sc_full
        label = row.get("true_subgoal")
        if label is not None:
            ceil_full = ceiling_pick([c["text"] for c in cands], label)
            ceil_filtered = elig[ceiling_pick([cands[i]["text"] for i in elig], label)]
            ceil_changed += ceil_filtered != ceil_full
        if cands[sc_filtered]["text"] != cands[0]["text"]:
            pick_ne_greedy += 1
        # (a') ≥ 1 eligible SAMPLED candidate (greedy never counts).
        if any(i > 0 for i in elig if not trunc[i]):
            a_rows += 1
        # (b') ≥ 2 unique strings on the eligible list.
        if len({cands[i]["text"] for i in elig}) >= 2:
            b_rows += 1
        # (c') pool: every non-truncated SAMPLED candidate.
        for i in range(1, len(cands)):
            if not trunc[i]:
                pool_n += 1
                pool[cands[i]["text"]] = pool.get(cands[i]["text"], 0) + 1
    if not pool:
        sys.exit("eligible sampled pool is EMPTY — not the banked table, stop")
    top_text, top_count = max(pool.items(), key=lambda kv: (kv[1], kv[0]))
    sizes = sorted(eligible_sizes)
    return {
        "rows": len(rows),
        "rows_with_truncated_candidate": binds,
        "sc_pick_changed": sc_changed,
        "ceil_pick_changed": ceil_changed,
        "a_eligible_sampled_rows": a_rows,
        "b_diverse_eligible_rows": b_rows,
        "c_top_eligible_sampled": {"text": top_text, "count": top_count},
        "c_eligible_pool": pool_n,
        "fallback_rows": fallback_rows,
        "pick_ne_greedy_filtered": pick_ne_greedy,
        "eligible_list_size": {
            "min": sizes[0],
            "max": sizes[-1],
            "mean": round(sum(sizes) / len(sizes), 5),
        },
    }


def adjudicate(table: dict, out_path: str | None) -> dict:
    numbers = recompute(table)
    for key, want in PRIORS.items():
        if numbers[key] != want:
            sys.exit(
                f"INSTRUMENT BREAKAGE: {key} recomputed as {numbers[key]!r}, "
                f"the pre-reg's written prior is {want!r} — the priors were "
                "computed from this same banked table, so a mismatch means "
                "the instrument (not the data) changed; stop, no "
                "adjudication (pre-reg 'Stage 1')",
            )
    print(
        "priors reproduced exactly (oracles vii + x): "
        f"{numbers['rows_with_truncated_candidate']}/60 rows carry a "
        f"truncated candidate; filter changes {numbers['sc_pick_changed']}/60 "
        f"SC picks + {numbers['ceil_pick_changed']}/60 ceiling picks; "
        f"a' {numbers['a_eligible_sampled_rows']}/60, "
        f"b' {numbers['b_diverse_eligible_rows']}/60, "
        f"c' {numbers['c_top_eligible_sampled']['count']}/"
        f"{numbers['c_eligible_pool']}",
    )
    bars = {
        "a_prime": {
            "line": f">= {A_BAR:.0%} of rows with >= 1 eligible sampled candidate",
            "value": numbers["a_eligible_sampled_rows"] / numbers["rows"],
            "pass": numbers["a_eligible_sampled_rows"] / numbers["rows"] >= A_BAR,
        },
        "b_prime": {
            "line": f">= {B_BAR:.0%} of rows with >= 2 unique eligible strings",
            "value": numbers["b_diverse_eligible_rows"] / numbers["rows"],
            "pass": numbers["b_diverse_eligible_rows"] / numbers["rows"] >= B_BAR,
        },
        "c_prime": {
            "line": f"top pooled eligible sampled string <= {C_BAR:.0%}",
            "value": numbers["c_top_eligible_sampled"]["count"]
            / numbers["c_eligible_pool"],
            "pass": numbers["c_top_eligible_sampled"]["count"]
            / numbers["c_eligible_pool"]
            <= C_BAR,
        },
        "d_eyes": {
            "line": "eligible candidates subgoal-shaped (eyes)",
            "value": "carried from the stage-1 close post's qualitative "
            "block (every inspected clean candidate subgoal-shaped and "
            "phase-relevant)",
            "pass": True,
        },
    }
    failed = [name for name, bar in bars.items() if not bar["pass"]]
    if failed:
        sys.exit(
            f"INSTRUMENT BREAKAGE: bar(s) {failed} fail although the priors "
            "reproduced — bar arithmetic drifted from the pre-reg, stop",
        )
    out = {
        "prereg": PREREG,
        "table": TABLE_DEFAULT,
        "provenance": PROVENANCE,
        "numbers": numbers,
        "bars": bars,
        "stage2_gate": "OPEN",
        "note": "stage 1 closed CPU-free on the banked rung-(b) table "
        "(pass-1 byte-identity; the eligible-list filter is "
        "selection-side only). Stage 2 = the frozen (b) arms with the "
        "clean-list rule, per the (b') pre-reg.",
    }
    for name, bar in bars.items():
        value = bar["value"]
        shown = f"{value:.1%}" if isinstance(value, float) else "eyes"
        print(f"  bar {name}: PASS ({shown})")
    print("STAGE-1 GATE: OPEN — the (b') stage-2 arms may launch")
    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=2))
        print(f"wrote {out_path}")
    return out


# ------------------------------------------------------------------ oracle


def oracle(table_path: Path) -> None:
    def expect_exit(fn: Callable[[], object], needle: str, label: str) -> None:
        try:
            fn()
        except SystemExit as err:
            if needle not in str(err):
                raise AssertionError(
                    f"{label}: aborted with {err!r}, wanted {needle!r}",
                ) from None
            print(f"  abort branch OK: {label}")
            return
        raise AssertionError(f"{label}: did not abort")

    def mutated(mutate: Callable[[dict], object]) -> dict:
        table = json.loads(table_path.read_text())
        mutate(table)
        return table

    if not table_path.exists():
        sys.exit(
            f"banked stage-1 table {table_path} is MISSING — the oracle "
            "adjudicates real data, stop",
        )
    table = json.loads(table_path.read_text())
    out = adjudicate(table, None)
    assert out["stage2_gate"] == "OPEN"
    assert out["numbers"] == {**out["numbers"], **PRIORS}
    print("  pass path OK (real banked table, priors exact, gate OPEN)")

    # Planted filter-binds world ON THE REAL DATA: make a row's
    # full-list SC argmax truncated — the filtered pick must move, so
    # pick-invariance (prior 0/60) breaks and the script refuses.
    def plant_binds(t: dict) -> None:
        for row in t["rows"]:
            pick = row["sc_pick"]
            if not row["candidates"][pick]["truncated"] and any(
                not c["truncated"] for i, c in enumerate(row["candidates"]) if i != pick
            ):
                row["candidates"][pick]["truncated"] = True
                return
        raise AssertionError("no plantable row found")

    expect_exit(
        lambda: adjudicate(mutated(plant_binds), None),
        "sc_pick_changed",
        "planted filter-binds world (SC pick moves, invariance breaks)",
    )

    # Planted all-truncated row: greedy fallback recorded — and bar
    # (a')'s prior (60/60) breaks, so the script refuses.
    def plant_fallback(t: dict) -> None:
        for c in t["rows"][0]["candidates"]:
            c["truncated"] = True

    expect_exit(
        lambda: adjudicate(mutated(plant_fallback), None),
        "INSTRUMENT BREAKAGE",
        "planted all-truncated row (fallback counted, priors break)",
    )
    # …and the fallback row really is recorded as one by the recompute
    # (recompute never gates on priors — adjudicate does).
    numbers = recompute(mutated(plant_fallback))
    assert numbers["fallback_rows"] == 1, numbers
    assert numbers["a_eligible_sampled_rows"] == 59, numbers
    print("  fallback recording OK (eligible == [greedy], counted)")

    expect_exit(
        lambda: adjudicate(
            mutated(lambda t: t["rows"][3].update(sc_pick=0)),
            None,
        ),
        "does not reproduce from the dumped stats",
        "banked-pick drift",
    )
    expect_exit(
        lambda: adjudicate(mutated(lambda t: t.update(seed=1)), None),
        "byte-identity argument",
        "provenance drift (seed)",
    )
    expect_exit(
        lambda: adjudicate(
            mutated(lambda t: t.update(checkpoint="/other/step_090000")),
            None,
        ),
        "wrong table",
        "provenance drift (checkpoint)",
    )
    expect_exit(
        lambda: adjudicate(
            mutated(lambda t: t.update(rows=t["rows"][:10])),
            None,
        ),
        "expected the banked",
        "row-count drift",
    )
    print("oracle: ALL branches OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--table", default=TABLE_DEFAULT)
    parser.add_argument("--out", default=OUT_DEFAULT)
    args = parser.parse_args()
    table_path = Path(args.table)
    if args.oracle:
        oracle(table_path)
        return
    if not table_path.exists():
        sys.exit(
            f"banked stage-1 table {table_path} is MISSING — stage 1 is a "
            "re-adjudication of banked data, nothing to score; stop",
        )
    adjudicate(json.loads(table_path.read_text()), args.out)


if __name__ == "__main__":
    main()
