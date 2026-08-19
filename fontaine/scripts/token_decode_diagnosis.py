"""Token-head decode diagnosis (CPU, banked artifacts only) — feeds the
owner's R2 band call (post 1539564065414840340: activate token-GRPO from
7% / token-focused SFT variant first / park).

Dissects the route-C joint-probe episode JSONs + videos: the joint
checkpoint reads 7/100 grammar-greedy while its flow sibling reads
44/100 on the SAME trunk (which the CE stream alone trained — flow was
insulated), and the base-token anchor reads 0/100.

Instruments (all from banked artifacts, no GPU):
  1. Contact funnel — grip is CONTACT-coded in rollout_sim (jaw touches
     benchy; 3 = two-sided pinch), so touch -> pinch -> success
     conversion rates read directly off the traces.
  2. Frozen-arm check — per-video motion energy (frame differencing).
     The 08-13 ar100 read found 6.8% zeros-fallback chunks; grammar
     masking repairs decodability by construction, this checks no
     residual no-op collapse survives.
  3. Reach envelope — contact/success rate vs spawn distance bin.
  4. Carry dynamics — cm/s toward the goal while pinched; pinch-fail
     taxonomy (wrong-way / stalled carry / timed-out-still-holding).
  5. Stereotypy — cross-seed frame dissimilarity at fixed ticks (a
     one-canonical-trajectory decode would read LOW).

Outputs:
  reports/analysis__token_decode_diagnosis.json
  fontaine/blog/src/img/grasp_sft/token_decode_diagnosis.png

Usage:
  uv run python fontaine/scripts/token_decode_diagnosis.py \
      [--probe-dir outputs/sim/grasp_sft/joint_probes] [--skip-videos]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Any

REPO = Path("/home/ubuntu/flow-matching")
ARMS = ("flow_unseen", "token_unseen", "token_base")
TICK_HZ = 30.0
DIST_BINS = ((6.0, 8.0), (8.0, 9.0), (9.0, 10.0), (10.0, 11.0), (11.0, 13.0))
# Motion-energy floor: a rendered-but-frozen scene reads ~0 mean |diff|
# (deterministic renderer); the least-active banked episode reads 1.4.
FROZEN_MOTION_BAR = 0.15


def touched(e: dict[str, Any]) -> bool:
    return any(g > 0 for g in e["grip"])


def pinched(e: dict[str, Any]) -> bool:
    return any(g == 3 for g in e["grip"])


def succeeded(e: dict[str, Any]) -> bool:
    return e["success_tick"] is not None


def funnel(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Instrument 1: the contact -> pinch -> success conversion ladder."""
    n = len(episodes)
    touch = [e for e in episodes if touched(e)]
    pinch = [e for e in episodes if pinched(e)]
    succ = [e for e in episodes if succeeded(e)]
    knock = [e for e in episodes if (e["initial_cm"] - e["final_cm"]) < -1.0]
    return {
        "n": n,
        "touch": len(touch),
        "pinch": len(pinch),
        "success": len(succ),
        "success_seeds": sorted(e["seed"] for e in succ),
        "knock_aways_gt1cm": len(knock),
        "touch_to_pinch": round(len(pinch) / len(touch), 3) if touch else None,
        "pinch_to_success": round(len(succ) / len(pinch), 3) if pinch else None,
    }


def pinch_fail_taxonomy(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Instrument 4b: how episodes that achieve the grasp predicate die.

    wrong_way: net object motion AWAY from the goal over the pinch span.
    stalled_carry: net carry toward the goal but released/lost short.
    timeout_holding: still pinched in the final 3 ticks — ran out the
      episode clock mid-carry (a carry-speed failure by construction).
    """
    rows = []
    for e in episodes:
        if not pinched(e) or succeeded(e):
            continue
        g, dist = e["grip"], e["distance_cm"]
        ticks = [i for i, v in enumerate(g) if v == 3]
        t0, t1 = ticks[0], ticks[-1]
        carry_cm = dist[t0] - dist[min(t1, len(dist) - 1)]
        ends_pinched = t1 >= len(g) - 3
        if ends_pinched:
            klass = "timeout_holding"
        elif carry_cm > 0.5:
            klass = "stalled_carry"
        else:
            klass = "wrong_way"
        rows.append(
            {
                "seed": e["seed"],
                "class": klass,
                "pinch_ticks": len(ticks),
                "carry_cm": round(carry_cm, 2),
                "min_cm": round(e["min_cm"], 2),
            },
        )
    counts = {
        k: sum(1 for r in rows if r["class"] == k)
        for k in ("wrong_way", "stalled_carry", "timeout_holding")
    }
    return {"n": len(rows), "classes": counts, "rows": rows}


def carry_speeds(episodes: list[dict[str, Any]], min_ticks: int = 30) -> dict[str, Any]:
    """Instrument 4a: cm/s toward the goal over the pinched span
    (episodes with >= min_ticks pinch ticks — one full replan chunk)."""
    rows = []
    for e in episodes:
        g, dist = e["grip"], e["distance_cm"]
        ticks = [i for i, v in enumerate(g) if v == 3]
        if len(ticks) < min_ticks:
            continue
        t0, t1 = ticks[0], ticks[-1]
        dt = (t1 - t0) / TICK_HZ
        if dt <= 0:
            continue
        speed = (dist[t0] - dist[min(t1, len(dist) - 1)]) / dt
        rows.append(
            {"seed": e["seed"], "cm_per_s": round(speed, 3), "success": succeeded(e)},
        )
    sp = [r["cm_per_s"] for r in rows]
    ss = [r["cm_per_s"] for r in rows if r["success"]]
    return {
        "n": len(rows),
        "median_cm_per_s": round(st.median(sp), 3) if sp else None,
        "success_median_cm_per_s": round(st.median(ss), 3) if ss else None,
        "rows": rows,
    }


def reach_envelope(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Instrument 3: contact + success by spawn-distance bin."""
    out = []
    for lo, hi in DIST_BINS:
        grp = [e for e in episodes if lo <= e["initial_cm"] < hi]
        out.append(
            {
                "bin": [lo, hi],
                "n": len(grp),
                "touch": sum(1 for e in grp if touched(e)),
                "success": sum(1 for e in grp if succeeded(e)),
            },
        )
    return out


def load_arms(probe_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        a: json.loads((probe_dir / f"{a}.json").read_text())["episodes"] for a in ARMS
    }


# ---------------------------------------------------------------- videos


def motion_energy(
    probe_dir: Path,
    arms: dict[str, list[dict[str, Any]]],
    cache: Path,
) -> dict[str, dict[str, Any]]:
    """Instrument 2: mean |frame diff| per episode (0.5 s sampling,
    160x120 gray). Cached — the sweep costs ~20 s over 300 videos."""
    if cache.exists():
        raw = json.loads(cache.read_text())
        return {a: {int(k): v for k, v in raw[a].items()} for a in raw}

    # Threads, not processes: `one` is a closure (unpicklable), and cv2
    # decode releases the GIL — 300 videos still finish in ~20 s.
    from concurrent.futures import ThreadPoolExecutor

    import cv2  # local import: the JSON-only path stays cv2-free

    jobs = [
        (a, e["seed"], str(probe_dir / a / f"rollout_seed{e['seed']:03d}.mp4"))
        for a in arms
        for e in arms[a]
        if (probe_dir / a / f"rollout_seed{e['seed']:03d}.mp4").exists()
    ]

    def one(job: tuple[str, int, str]) -> tuple[str, int, dict[str, float] | None]:
        arm, seed, path = job
        cap = cv2.VideoCapture(path)
        prev, diffs, i = None, [], 0
        while True:
            if not cap.grab():
                break
            if i % 15 == 0:
                ok, fr = cap.retrieve()
                if not ok:
                    break
                g = cv2.cvtColor(cv2.resize(fr, (320, 120)), cv2.COLOR_BGR2GRAY).astype(
                    "int16",
                )
                if prev is not None:
                    diffs.append(float(abs(g - prev).mean()))
                prev = g
            i += 1
        cap.release()
        if not diffs:
            return arm, seed, None
        return (
            arm,
            seed,
            {"mean": round(sum(diffs) / len(diffs), 4), "max": round(max(diffs), 4)},
        )

    out: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(8) as ex:
        for arm, seed, m in ex.map(one, jobs):
            out.setdefault(arm, {})[seed] = m
    cache.write_text(json.dumps(out))
    return out


# ----------------------------------------------------------------- chart

PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
MAGENTA = "#dc267f"  # token head (joint ckpt) — the entity under diagnosis
BLUE = "#648fff"  # flow head (joint ckpt), same trunk
GRAY = "#9aa0a8"  # token base anchor


def style_axis(ax: Any) -> None:
    ax.set_facecolor(PAGE)
    ax.tick_params(colors=META, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for lab in (ax.xaxis.label, ax.yaxis.label, ax.title):
        lab.set_color(TEXT)


def render_chart(
    res: dict[str, Any],
    arms: dict[str, list[dict[str, Any]]],
    out_png: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), facecolor=PAGE)
    fig.subplots_adjust(hspace=0.42, wspace=0.26, top=0.90)
    for ax in axes.flat:
        style_axis(ax)

    entity = {
        "flow_unseen": ("flow head", BLUE),
        "token_unseen": ("token head", MAGENTA),
        "token_base": ("token base", GRAY),
    }

    # A — the contact funnel
    ax = axes[0][0]
    stages = ("touched", "pinched", "success")
    x = np.arange(len(stages))
    for k, arm in enumerate(ARMS):
        f = res["funnel"][arm]
        vals = [f["touch"], f["pinch"], f["success"]]
        label, color = entity[arm]
        bars = ax.bar(
            x + (k - 1) * 0.27,
            vals,
            width=0.25,
            color=color,
            zorder=3,
            label=label,
        )
        for b, v in zip(bars, vals, strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 1.5,
                str(v),
                ha="center",
                color=TEXT,
                fontsize=9,
            )
    ax.set_xticks(x, stages)
    ax.set_ylim(0, 100)
    ax.set_ylabel("episodes / 100")
    ax.set_title("A · the funnel: same trunk, three decodes")
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(TEXT)

    # B — reach envelope by spawn distance
    ax = axes[0][1]
    for arm in ("flow_unseen", "token_unseen"):
        env = res["reach_envelope"][arm]
        centers = [(b["bin"][0] + b["bin"][1]) / 2 for b in env]
        touch_rate = [100 * b["touch"] / b["n"] if b["n"] else np.nan for b in env]
        succ_rate = [100 * b["success"] / b["n"] if b["n"] else np.nan for b in env]
        label, color = entity[arm]
        ax.plot(
            centers,
            touch_rate,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=5,
            zorder=3,
        )
        ax.plot(
            centers,
            succ_rate,
            color=color,
            linewidth=1.4,
            marker="o",
            markersize=4,
            linestyle="--",
            zorder=3,
        )
        ax.annotate(
            f"{label} · touched",
            (centers[-1], touch_rate[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            fontsize=8,
            va="center",
        )
        ax.annotate(
            "success",
            (centers[-1], succ_rate[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            fontsize=8,
            va="center",
            alpha=0.8,
        )
    ax.set_xlabel("spawn distance, boat → disk (cm)")
    ax.set_ylabel("% of episodes in bin")
    ax.set_ylim(-3, 105)
    ax.set_title("B · reach envelope: token contact dies with distance")

    # C — carry speed while pinched
    ax = axes[1][0]
    rng = np.random.default_rng(0)
    for k, arm in enumerate(("flow_unseen", "token_unseen")):
        cs = res["carry_speed"][arm]
        label, color = entity[arm]
        for r in cs["rows"]:
            xj = k + rng.uniform(-0.14, 0.14)
            if r["success"]:
                ax.plot(xj, r["cm_per_s"], "o", color=color, markersize=5, zorder=3)
            else:
                ax.plot(
                    xj,
                    r["cm_per_s"],
                    "o",
                    markerfacecolor=PAGE,
                    markeredgecolor=color,
                    markersize=5,
                    zorder=3,
                )
        med = cs["median_cm_per_s"]
        ax.plot([k - 0.2, k + 0.2], [med, med], color=TEXT, linewidth=1.6, zorder=4)
        ax.text(
            k - 0.24,
            med,
            f"median {med:.2f}",
            color=TEXT,
            fontsize=9,
            va="center",
            ha="right",
        )
    ax.axhline(0, color=GRID, linewidth=1.0)
    ax.set_xlim(-0.75, 1.55)
    ax.set_xticks([0, 1], ["flow head", "token head"])
    ax.set_ylabel("carry speed while pinched (cm/s)")
    ax.set_title("C · greedy carries at ~40% of flow speed")
    ax.text(
        0.02,
        0.03,
        "filled = success · open = failed carry",
        transform=ax.transAxes,
        color=META,
        fontsize=8,
    )

    # D — carry traces (time vs boat→disk distance)
    ax = axes[1][1]

    def trace(e: dict[str, Any], color: str, lw: float, alpha: float) -> None:
        t = np.arange(len(e["distance_cm"])) / TICK_HZ
        ax.plot(t, e["distance_cm"], color=color, linewidth=lw, alpha=alpha, zorder=3)

    flow_succ = [e for e in arms["flow_unseen"] if succeeded(e)][:12]
    for e in flow_succ:
        trace(e, BLUE, 1.0, 0.55)
    tok = {
        r["seed"]: r["class"]
        for r in res["pinch_fail_taxonomy"]["token_unseen"]["rows"]
    }
    for e in arms["token_unseen"]:
        if tok.get(e["seed"]) in ("stalled_carry", "timeout_holding"):
            trace(e, MAGENTA, 1.4, 0.9)
        elif succeeded(e):
            trace(e, MAGENTA, 1.0, 0.45)
    ax.set_xlabel("episode time (s)")
    ax.set_ylabel("boat → disk (cm)")
    ax.set_ylim(0, 14.5)
    ax.set_title("D · flow dives, greedy token crawls + times out")
    ax.text(
        0.03,
        0.95,
        "blue = flow successes (12)",
        transform=ax.transAxes,
        ha="left",
        color=BLUE,
        fontsize=8,
    )
    ax.text(
        0.03,
        0.88,
        "magenta = token stalled/timeout carries + successes",
        transform=ax.transAxes,
        ha="left",
        color=MAGENTA,
        fontsize=8,
    )

    fig.suptitle(
        "Token-head decode diagnosis — joint ckpt step 2000, unseen seeds 0–99\n"
        "greedy decode is success-capable but magnitude-attenuated: "
        "reach envelope truncated, carry speed halved",
        color=TEXT,
        fontsize=12,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=PAGE, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ main


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--probe-dir",
        type=Path,
        default=REPO / "outputs/sim/grasp_sft/joint_probes",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO / "reports/analysis__token_decode_diagnosis.json",
    )
    p.add_argument(
        "--chart",
        type=Path,
        default=REPO / "fontaine/blog/src/img/grasp_sft/token_decode_diagnosis.png",
    )
    p.add_argument("--skip-videos", action="store_true")
    args = p.parse_args()

    arms = load_arms(args.probe_dir)
    res: dict[str, Any] = {
        "funnel": {a: funnel(arms[a]) for a in ARMS},
        "pinch_fail_taxonomy": {a: pinch_fail_taxonomy(arms[a]) for a in ARMS},
        "carry_speed": {a: carry_speeds(arms[a]) for a in ARMS},
        "reach_envelope": {a: reach_envelope(arms[a]) for a in ARMS},
    }

    if not args.skip_videos:
        me = motion_energy(args.probe_dir, arms, args.probe_dir / "motion_energy.json")
        res["motion"] = {
            a: {
                "n": len(me.get(a, {})),
                "min_mean_motion": min(v["mean"] for v in me[a].values() if v),
                "near_frozen_lt_bar": sum(
                    1 for v in me[a].values() if v and v["mean"] < FROZEN_MOTION_BAR
                ),
            }
            for a in ARMS
            if a in me
        }

    tu, fu = res["funnel"]["token_unseen"], res["funnel"]["flow_unseen"]
    overlap = sorted(set(tu["success_seeds"]) & set(fu["success_seeds"]))
    res["success_overlap_token_vs_flow"] = {
        "overlap_seeds": overlap,
        "token_only": sorted(set(tu["success_seeds"]) - set(fu["success_seeds"])),
    }
    res["headline"] = {
        "no_frozen_arm_episodes": (
            res.get("motion") is None
            or all(v["near_frozen_lt_bar"] == 0 for v in res["motion"].values())
        ),
        "token_touch_pct": tu["touch"],
        "token_success_beyond_10cm": sum(
            b["success"]
            for b in res["reach_envelope"]["token_unseen"]
            if b["bin"][0] >= 10.0
        ),
        "carry_speed_ratio_token_over_flow": round(
            res["carry_speed"]["token_unseen"]["median_cm_per_s"]
            / res["carry_speed"]["flow_unseen"]["median_cm_per_s"],
            3,
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"wrote {args.out}")

    render_chart(res, arms, args.chart)
    print(f"wrote {args.chart}")

    h = res["headline"]
    print(
        f"\nheadline: frozen-arm episodes: "
        f"{'NONE' if h['no_frozen_arm_episodes'] else 'PRESENT'}; "
        f"token touched {h['token_touch_pct']}/100; "
        f"successes beyond 10 cm: {h['token_success_beyond_10cm']}; "
        f"carry-speed ratio token/flow {h['carry_speed_ratio_token_over_flow']}",
    )


if __name__ == "__main__":
    main()
