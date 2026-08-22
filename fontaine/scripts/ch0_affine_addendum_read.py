"""ch0 affine addendum read (CPU, record-only).

Addendum to ch0_shift_constant_read.py, spec frozen in-channel
06:47Z 08-22 (post 1540613448264712243) BEFORE compute: the shift
candidates left KS unmoved (0.295 -> 0.286/0.308 vs ref 0.161), so
the only remaining one-channel one-family edit is an affine. Two
candidates, both targeting demos (the shared-convention choice):

- moment: x' = mu_demos + (x - mu_clean) * (sigma_demos/sigma_clean)
- robust: x' = med_demos + (x - med_clean) * (range95_demos/range95_clean)

Frozen decision rule: if neither affine lands the clean<->demos KS
at or under the demos<->v2 reference band, NO viable one-channel ch0
edit exists and the ch0 edit-cell rung closes pre-launch (amendment
on the gripfix pre-reg's <=10 branch; ladder advances to
content-level slicing). Appends "affine_addendum" to
reports/analysis__ch0_shift_constant_read.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from clean_content_manifold_probe import (
    DATASETS,
    EXPECTED_FRAMES,
    ks_distance,
    load_dataset,
)

CH = 0
REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "reports/analysis__ch0_shift_constant_read.json"


def main() -> None:
    report = json.loads(REPORT.read_text())
    stats = report["location_stats"]

    data = {}
    for name, path in DATASETS.items():
        d = load_dataset(path)
        n = d["action"].shape[0]
        assert n == EXPECTED_FRAMES[name], (name, n)
        data[name] = d
    ch0 = {
        kind: {name: data[name][kind][:, CH].astype(np.float64) for name in DATASETS}
        for kind in ("action", "state")
    }

    # Oracle: reproduce the banked unshifted KS from the first read.
    for kind in ("action", "state"):
        got = ks_distance(ch0[kind]["clean"], ch0[kind]["demos"])
        banked = report["banked_ks_reproduced"][kind]["clean_vs_demos"]
        assert abs(got - banked) < 1e-12, (kind, got, banked)
    print("oracle green: first-read KS reproduced")

    # Affine constants from the banked action stats (one transform,
    # both columns - same rule as the shift candidates).
    a_demos, a_clean = stats["action"]["demos"], stats["action"]["clean"]
    candidates = {
        "affine_moment": {
            "scale": a_demos["std"] / a_clean["std"],
            "center_src": a_clean["mean"],
            "center_dst": a_demos["mean"],
        },
        "affine_robust": {
            "scale": (a_demos["q95"] - a_demos["q05"])
            / (a_clean["q95"] - a_clean["q05"]),
            "center_src": a_clean["median"],
            "center_dst": a_demos["median"],
        },
    }

    out = {}
    for cand, c in candidates.items():
        out[cand] = {"constants": c}
        for kind in ("action", "state"):
            x = ch0[kind]["clean"]
            xt = c["center_dst"] + (x - c["center_src"]) * c["scale"]
            ref = report["banked_ks_reproduced"][kind]["demos_vs_v2"]
            out[cand][kind] = {
                "vs_demos": ks_distance(xt, ch0[kind]["demos"]),
                "vs_v2": ks_distance(xt, ch0[kind]["v2"]),
                "reference": ref,
                "transformed_min": float(xt.min()),
                "transformed_max": float(xt.max()),
            }

    in_band = {
        cand: all(
            out[cand][k]["vs_demos"] <= out[cand][k]["reference"]
            for k in ("action", "state")
        )
        for cand in candidates
    }
    out["verdict"] = {
        "in_band": in_band,
        "any_viable": any(in_band.values()),
    }

    report["affine_addendum"] = out
    report["addendum_spec_post"] = "1540613448264712243"
    REPORT.write_text(json.dumps(report, indent=1))
    print(f"report updated -> {REPORT}")

    for cand in candidates:
        c = out[cand]["constants"]
        print(
            f"\n{cand}: scale {c['scale']:.4f}, {c['center_src']:+.3f} -> {c['center_dst']:+.3f}",
        )
        for kind in ("action", "state"):
            v = out[cand][kind]
            print(
                f"  {kind:6s} vs_demos {v['vs_demos']:.4f} vs_v2 {v['vs_v2']:.4f} "
                f"(ref {v['reference']:.4f}) range [{v['transformed_min']:+.1f}, {v['transformed_max']:+.1f}]",
            )
    print(f"\nverdict: in_band {in_band} -> any_viable {out['verdict']['any_viable']}")


if __name__ == "__main__":
    main()
