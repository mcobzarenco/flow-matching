"""er_60k @60000 events one-off: confusion quant + galleries + probe + HTML.

Owner request 12:44:35Z + 12:45:13Z 2026-08-11 (record-only, rides the
er-60k pre-reg; launch note with the frozen confusion/probe spec posted
in-channel 14:1xZ before the GPU minute). Consumes the dump pass's
``--dump-generations`` JSON (instrument commit 7f43c54) and produces:

quant  (CPU): 13-class model x gt event confusion INCLUDING none/none,
       per-class precision/recall, the (model: none, gt: event) miss
       bucket sized exactly, presence-detection cross-check vs the
       banked event acc → analysis__er60k_events_confusion.json
probe  (GPU): on miss frames, replay each frame's OWN generated prefix
       (subgoal/holding/progress value lines teacher-forced through the
       suffix scaffold) and re-decode the event slot greedy with the
       exact "none" tokenization banned at its first step. Per-frame
       oracle: the UNBANNED replay must reproduce "none" bit-exact,
       else the frame is EXCLUDED and counted. Also extracts the
       gallery frames' camera images (same dataset selection as the
       eval — concat indices from the dump stay valid).
html   (CPU): standalone dark-theme report (raw strings everywhere;
       the class binning is quant/display only) → reports/.

Phases run independently: quant → probe → html.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

STEM = "eval__fontaine_molmo2_er_60k_ddp4__step_060000__panel_curated_v0_k4l2_events"
DUMP = REPO / "reports" / f"{STEM}_generations.json"
EVAL_JSON = REPO / "reports" / f"{STEM}.json"
CONFUSION_OUT = REPO / "reports" / "analysis__er60k_events_confusion.json"
PROBE_OUT = REPO / "reports" / "analysis__er60k_events_probe.json"
GALLERY_DIR = REPO / "reports" / "er60k_events_gallery"
HTML_OUT = REPO / "reports" / "report__er60k_events_oneoff.html"
CHECKPOINT = REPO / "outputs/train/fontaine_molmo2_er_60k_ddp4/step_060000"
DATA_ROOT = "/home/ubuntu/datasets/mcobzarenco/community_curated_v0"
# Banked endpoint aux read (eval__..._k4l2.json, rc 13:28Z 08-11).
# NEAR-reproduction is the oracle, not byte-identity: the banked run
# was a 4-way box shard, this dump is world-size 1 — sharding.py
# documents the cross-world-size bf16 batch-composition drift (near-tie
# greedy argmax flips; the narrated value phase adds a designed
# composition dependence). Measured delta on this dump: 13/8,987
# frames (0.8568 vs 0.8582).
BANKED_EVENT_ACC = 0.8582396795371091
ORACLE_TOLERANCE = 0.003

# ---------------------------------------------------------------- taxonomy
# FROZEN at the 14:1xZ 08-11 launch note (measured gt-vocab coverage
# 92.6%, other 7.4%). First match wins; "; "-joined multi-event strings
# classify by their FIRST part. Applied identically to gt and generated
# strings — the binning is quant/display only, raw strings are shown
# everywhere they matter.
RULES: list[tuple[str, re.Pattern[str]]] = [
    ("episode_marker", re.compile(r"episode (reset|ends|begins|starts)")),
    (
        "human",
        re.compile(
            r"human|person |person'|hand (enters|reaches|receives|approaches"
            r"|takes|holds)|handed|hands? off|bite",
        ),
    ),
    (
        "idle",
        re.compile(
            r"idle|stationar|no motion|not moving|motionless|no visible motion"
            r"|barely mov|little to no|little visible motion|little .*motion"
            r"|remains still|essentially still|nearly still|arm still"
            r"|robot still|^still|static|scene unchanged|unchanged from start"
            r"|no manipulation",
        ),
    ),
    ("blur", re.compile(r"blur")),
    (
        "occlusion",
        re.compile(
            r"occlud|obscur|hidden|blocked|out of view|out of frame"
            r"|off.?screen|not visible|leaves the .*view|exits? the"
            r"|out of the camera",
        ),
    ),
    (
        "miss_fail",
        re.compile(
            r"miss|fail|unsuccessful|unable|attempt|retry|struggl|hover"
            r"|without (closing|grasping|touching)|never closes|does not"
            r"|doesn't",
        ),
    ),
    (
        "release_place",
        re.compile(
            r"releas|placed|places|deposit|dropped (in|into|onto|to)|inserted"
            r"|stacked|set down|put down|lowered (in|into|onto)|pour",
        ),
    ),
    (
        "drop_slip",
        re.compile(
            r"drop|slip|fell|fall|lost grip|tumbl|topple|spill"
            r"|loses (the |its )?(grip|hold)",
        ),
    ),
    (
        "grasp_pickup",
        re.compile(
            r"grasp|pick|lift|grab|gripped|grips|pulled (open|free|out)"
            r"|pulls (open|free|out)|opens the|closes the",
        ),
    ),
    (
        "contact_collision",
        re.compile(
            r"collid|collision|bump|knock|hit |hits |pushed|pushes|nudg"
            r"|drag|swept|sweep|contact",
        ),
    ),
    (
        "camera_view",
        re.compile(
            r"camera|view|exposure|lighting|glare|shadow|reflect"
            r"|frame (jump|shift)",
        ),
    ),
]
CLASSES = ["none", *[name for name, _ in RULES], "other"]


def classify(text: str) -> str:
    head = text.split(";")[0].lower().strip()
    if head == "none":
        return "none"
    for name, pattern in RULES:
        if pattern.search(head):
            return name
    return "other"


def event_rows(dump: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows with BOTH an event label and a parsed generated event —
    exactly the population the standard eval's event acc scores."""
    return [
        row
        for row in dump["rows"]
        if row["labels"].get("event") is not None
        and row["generated"].get("event") is not None
    ]


def bucket(row: dict[str, Any]) -> str:
    gt = row["labels"]["event"].strip()
    model = row["generated"]["event"].strip()
    if gt == "none" and model == "none":
        return "both_none"
    if gt != "none" and model == "none":
        return "miss"
    if gt == "none" and model != "none":
        return "false_alarm"
    return "hit" if classify(gt) == classify(model) else "class_swap"


def spread(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Up to ``limit`` rows maximizing repo diversity (round-robin over
    repos, deterministic seed) — the owner asked for MANY VARIED
    examples, not the first N of one dataset."""
    by_repo: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_repo[row["repo_id"]].append(row)
    rng = random.Random(0)
    queues = [rng.sample(v, len(v)) for v in by_repo.values()]
    rng.shuffle(queues)
    picked: list[dict[str, Any]] = []
    while queues and len(picked) < limit:
        for queue in list(queues):
            picked.append(queue.pop())
            if not queue:
                queues.remove(queue)
            if len(picked) >= limit:
                break
    return picked


# ------------------------------------------------------------------ quant
def phase_quant() -> None:
    dump = json.loads(DUMP.read_text())
    rows = event_rows(dump)
    unparsed = sum(
        1
        for row in dump["rows"]
        if row["labels"].get("event") is not None
        and row["generated"].get("event") is None
    )
    matrix: dict[str, dict[str, int]] = {
        gt: dict.fromkeys(CLASSES, 0) for gt in CLASSES
    }
    buckets = collections.Counter()
    exact_hits = 0
    gt_event_rows = 0
    presence_hits = 0
    other_gt = collections.Counter()
    other_model = collections.Counter()
    for row in rows:
        gt_text = row["labels"]["event"].strip()
        model_text = row["generated"]["event"].strip()
        gt = classify(gt_text)
        model = classify(model_text)
        matrix[gt][model] += 1
        buckets[bucket(row)] += 1
        presence_hits += int((gt_text == "none") == (model_text == "none"))
        if gt_text != "none":
            gt_event_rows += 1
            exact_hits += int(gt_text.lower() == model_text.lower())
        if gt == "other":
            other_gt[gt_text] += 1
        if model == "other":
            other_model[model_text] += 1

    per_class = {}
    for name in CLASSES:
        gt_total = sum(matrix[name].values())
        model_total = sum(matrix[gt][name] for gt in CLASSES)
        correct = matrix[name][name]
        per_class[name] = {
            "gt_frames": gt_total,
            "model_frames": model_total,
            "precision": correct / model_total if model_total else None,
            "recall": correct / gt_total if gt_total else None,
        }

    payload = {
        "stem": STEM,
        "frames_scored": len(rows),
        "unparsed_generations": unparsed,
        "presence_accuracy": presence_hits / len(rows),
        "banked_event_acc": BANKED_EVENT_ACC,
        "oracle_tolerance": ORACLE_TOLERANCE,
        "exact_string_match_on_gt_event_frames": (
            exact_hits / gt_event_rows if gt_event_rows else None
        ),
        "buckets": dict(buckets),
        "classes": CLASSES,
        "matrix": matrix,
        "per_class": per_class,
        "other_residue": {
            "gt_top": other_gt.most_common(15),
            "model_top": other_model.most_common(15),
        },
    }
    CONFUSION_OUT.write_text(json.dumps(payload, indent=1))
    acc = payload["presence_accuracy"]
    print(f"quant: {len(rows)} event-scored frames, buckets {dict(buckets)}")
    print(
        f"presence acc {acc:.4f} vs banked {BANKED_EVENT_ACC} "
        f"({'ORACLE PASS' if abs(acc - BANKED_EVENT_ACC) < ORACLE_TOLERANCE else 'ORACLE FAIL'})",
    )
    print(f"wrote {CONFUSION_OUT}")


# ------------------------------------------------------------------ probe
GALLERY_LIMIT = 28
PROBE_LIMIT = 800


def value_lines(text: str) -> dict[str, str]:
    """Display-form generation text → field: value (verbatim strings —
    generated values cannot contain newlines, the terminator ends them)."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        field, sep, value = line.partition(": ")
        if sep:
            values[field] = value
    return values


def phase_probe() -> None:
    import torch

    from bijou.data import EpisodeSplit, select_datasets
    from bijou.eval.policies import BijouPolicy
    from bijou.eval.report import _image_data_uri
    from bijou.modelling.aux_text import AuxField

    dump = json.loads(DUMP.read_text())
    rows = event_rows(dump)
    by_bucket: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_bucket[bucket(row)].append(row)

    gallery = {
        name: spread(by_bucket.get(name, []), GALLERY_LIMIT)
        for name in ("hit", "class_swap", "miss", "false_alarm")
    }
    miss_rows = by_bucket.get("miss", [])
    probe_rows = miss_rows
    if len(probe_rows) > PROBE_LIMIT:
        probe_rows = spread(probe_rows, PROBE_LIMIT)
        print(
            f"probe CAPPED at {PROBE_LIMIT}/{len(miss_rows)} miss frames "
            "(repo-diverse spread; the cap is reported, never silent)",
        )

    # Same selection args as the eval — the dump's concat indices are
    # valid only under this exact selection.
    selection = select_datasets(
        (Path(DATA_ROOT),),
        (),
        50,
        episode_split=EpisodeSplit("holdout"),
        holdout_fraction=0.1,
        split_seed=0,
        allowed_fps=(30,),
        allowed_camera_counts=(1, 2),
        load_episode_annotations=True,
    )
    dataset = selection.concat()
    device = torch.device("cuda")
    policy = BijouPolicy(
        CHECKPOINT,
        device=device,
        seed=0,
        generate=tuple(AuxField),
    )
    fields = policy.aux_fields
    event_position = fields.index(AuxField.EVENT)
    decoder = policy.model.decoder
    backbone = policy.model._molmo2_backbone()
    runtime = decoder.aux_runtime
    assert runtime is not None
    tokenizer = runtime.tokenizer
    terminator = runtime.terminator_id
    none_ids = tokenizer.encode("none", add_special_tokens=False)
    config = decoder.config
    vocab = config.block_base + config.vocab_total
    min_value = float(torch.finfo(torch.float32).min)
    text_allowed = torch.zeros(vocab, dtype=torch.bool, device=device)
    text_allowed[: config.block_base] = True
    if decoder.newline_carrier_ids:
        text_allowed[
            torch.tensor(sorted(decoder.newline_carrier_ids), device=device)
        ] = False
    from bijou.modelling.aux_text import VALUE_BUDGETS

    budget = VALUE_BUDGETS[AuxField.EVENT]

    def replay_decode(memory: Any, prefix: list[int], *, ban_first: bool) -> str:
        """Feed the teacher-forced prefix, then greedy-decode one value
        line — decode_value_line's loop with an optional first-step ban
        of the 'none' head token."""
        feed = torch.tensor([prefix], dtype=torch.long, device=device)
        fed = 0
        ids: list[int] = []
        for step in range(budget + 1):
            logits, fed = decoder._step(backbone, memory, feed, fed)
            logits = logits[:, :vocab].masked_fill(~text_allowed, min_value)
            if step == 0 and ban_first:
                logits[:, none_ids[0]] = min_value
            next_id = int(logits.argmax(dim=-1))
            if step == budget and next_id != terminator:
                break  # budget exhausted — forced terminator, text stands
            if next_id == terminator:
                break
            ids.append(next_id)
            feed = torch.tensor([[next_id]], dtype=torch.long, device=device)
        return tokenizer.decode(ids).strip()

    def frame_prefix(row: dict[str, Any]) -> list[int] | None:
        values = value_lines(row["text"])
        prefix = list(decoder.opener_ids)
        for field in fields[:event_position]:
            value = values.get(field.value)
            if value is None:
                return None
            prefix += tokenizer.encode(value, add_special_tokens=False)
            prefix.append(terminator)
        return prefix

    def encode_item(index: int) -> tuple[Any, Any]:
        item = dataset[index]
        items = policy.apply_overrides([item])
        batch = policy.collator(items).to(policy.device)
        memory = policy.model.encode(batch.encoder_inputs, with_grad=False)
        return item, memory

    probe_results: list[dict[str, Any]] = []
    excluded = 0
    still_none = 0
    with torch.no_grad():
        for n, row in enumerate(probe_rows, 1):
            prefix = frame_prefix(row)
            if prefix is None:
                excluded += 1
                continue
            _, memory = encode_item(row["index"])
            snapshot = decoder.cache_snapshot(memory)
            unbanned = replay_decode(memory, prefix, ban_first=False)
            if unbanned != "none":
                # Replay drifted from the dump pass — the oracle the
                # launch note promised; excluded, counted, shown.
                excluded += 1
                continue
            decoder.cache_restore(memory, snapshot)
            forced = replay_decode(memory, prefix, ban_first=True)
            if forced == "none":
                still_none += 1
                continue
            gt_text = row["labels"]["event"].strip()
            probe_results.append(
                {
                    "index": row["index"],
                    "repo_id": row["repo_id"],
                    "episode_index": row["episode_index"],
                    "frame_index": row["frame_index"],
                    "instruction": row["instruction"],
                    "gt_event": gt_text,
                    "forced_event": forced,
                    "gt_class": classify(gt_text),
                    "forced_class": classify(forced),
                    "class_match": classify(gt_text) == classify(forced),
                },
            )
            if n % 50 == 0:
                print(f"  probed {n}/{len(probe_rows)} miss frames", flush=True)

    matches = sum(r["class_match"] for r in probe_results)
    payload = {
        "stem": STEM,
        "miss_frames_total": len(miss_rows),
        "probed": len(probe_rows),
        "excluded_replay_oracle": excluded,
        "still_none_after_ban": still_none,
        "forced_guesses": len(probe_results),
        "class_match": matches,
        "class_match_rate": matches / len(probe_results) if probe_results else None,
        "results": probe_results,
    }
    PROBE_OUT.write_text(json.dumps(payload, indent=1))
    print(
        f"probe: {len(probe_results)} forced guesses from {len(probe_rows)} "
        f"miss frames ({excluded} excluded by the replay oracle, "
        f"{still_none} still-none after the ban); class match "
        f"{matches}/{len(probe_results)}",
    )
    print(f"wrote {PROBE_OUT}")

    # Gallery images: one camera frame per selected example (+ probe
    # examples), saved as data-URI text files the html phase embeds.
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    probe_indices = {r["index"] for r in probe_results}
    wanted: dict[int, dict[str, Any]] = {}
    for rows_ in gallery.values():
        for row in rows_:
            wanted[row["index"]] = row
    for row in rows:
        if row["index"] in probe_indices and row["index"] not in wanted:
            wanted[row["index"]] = row
    manifest = {}
    for n, (index, _row) in enumerate(sorted(wanted.items()), 1):
        item = dataset[index]
        cameras = {
            key: value
            for key, value in item.items()
            if key.startswith("observation.images.")
        }
        key = min(cameras)
        uri = _image_data_uri(cameras[key], height=200)
        (GALLERY_DIR / f"{index}.uri").write_text(uri)
        manifest[index] = {"camera": key}
        if n % 25 == 0:
            print(f"  extracted {n}/{len(wanted)} gallery frames", flush=True)
    (GALLERY_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "gallery": {
                    name: [row["index"] for row in rows_]
                    for name, rows_ in gallery.items()
                },
                "frames": manifest,
            },
            indent=1,
        ),
    )
    print(f"gallery: {len(wanted)} frames → {GALLERY_DIR}")


# ------------------------------------------------------------------- html
BUCKET_TITLES = {
    "hit": "Hits — model and gt agree an event happened, same class",
    "class_swap": "Class swaps — both saw an event, different class",
    "miss": "Misses — gt has an event, model said none",
    "false_alarm": "False alarms — model reports an event, gt says none",
}
DARK = {
    "bg": "#121417",
    "text": "#d8dade",
    "heading": "#eceef1",
    "border": "#3a3f46",
    "th": "#1f242b",
    "pre": "#1a1e24",
    "meta": "#9aa0a8",
    "accent": "#648fff",
    "good": "#42be65",
    "bad": "#dc267f",
    "warn": "#ffb000",
}


def esc(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def phase_html() -> None:
    dump = json.loads(DUMP.read_text())
    confusion = json.loads(CONFUSION_OUT.read_text())
    probe = json.loads(PROBE_OUT.read_text())
    manifest = json.loads((GALLERY_DIR / "manifest.json").read_text())
    rows_by_index = {row["index"]: row for row in dump["rows"]}

    def uri(index: int) -> str | None:
        path = GALLERY_DIR / f"{index}.uri"
        return path.read_text() if path.exists() else None

    def card(row: dict[str, Any], extra: str = "") -> str:
        image = uri(row["index"])
        img = f'<img src="{image}">' if image else ""
        gt = esc(row["labels"].get("event", "—"))
        model = esc(row["generated"].get("event", "—"))
        return f"""<div class="card">{img}
<div class="lines">
<div class="meta">{esc(row["repo_id"])} · ep {row["episode_index"]} · frame {row["frame_index"]}</div>
<div class="meta">“{esc(row["instruction"])}”</div>
<div><span class="tag model">model</span> {model}</div>
<div><span class="tag gt">gt</span> {gt}</div>
{extra}</div></div>"""

    matrix = confusion["matrix"]
    classes = confusion["classes"]
    live = [
        c
        for c in classes
        if sum(matrix[c].values()) or sum(matrix[g][c] for g in classes)
    ]
    header = "".join(f"<th>{esc(c)}</th>" for c in live)
    body_rows = []
    max_off = (
        max(
            (matrix[g][m] for g in live for m in live if g != m),
            default=1,
        )
        or 1
    )
    for gt_class in live:
        cells = []
        for model_class in live:
            count = matrix[gt_class][model_class]
            if gt_class == model_class:
                style = 'style="background:rgba(66,190,101,0.25)"' if count else ""
            elif count:
                alpha = 0.08 + 0.5 * min(count / max_off, 1.0)
                style = f'style="background:rgba(220,38,127,{alpha:.2f})"'
            else:
                style = ""
            cells.append(f"<td {style}>{count or ''}</td>")
        total = sum(matrix[gt_class].values())
        body_rows.append(
            f"<tr><th>{esc(gt_class)}</th>{''.join(cells)}<td>{total}</td></tr>",
        )
    model_totals = "".join(f"<td>{sum(matrix[g][m] for g in live)}</td>" for m in live)

    per_class = confusion["per_class"]
    pr_rows = "".join(
        f"<tr><td>{esc(name)}</td>"
        f"<td>{per_class[name]['gt_frames']}</td>"
        f"<td>{per_class[name]['model_frames']}</td>"
        f"<td>{per_class[name]['precision']:.2f}</td>"
        f"<td>{per_class[name]['recall']:.2f}</td></tr>"
        for name in live
        if per_class[name]["precision"] is not None
        and per_class[name]["recall"] is not None
    )

    buckets = confusion["buckets"]
    n = confusion["frames_scored"]
    acc = confusion["presence_accuracy"]
    oracle_ok = abs(acc - confusion["banked_event_acc"]) < confusion["oracle_tolerance"]
    oracle_color = DARK["good"] if oracle_ok else DARK["bad"]

    tiles = "".join(
        f'<div class="tile"><div class="big">{value}</div><div>{esc(label)}</div></div>'
        for label, value in [
            ("event-scored frames", f"{n:,}"),
            ("both none", f"{buckets.get('both_none', 0):,}"),
            ("hits (same class)", f"{buckets.get('hit', 0):,}"),
            ("class swaps", f"{buckets.get('class_swap', 0):,}"),
            ("misses (model: none)", f"{buckets.get('miss', 0):,}"),
            ("false alarms", f"{buckets.get('false_alarm', 0):,}"),
        ]
    )

    gallery_sections = []
    for name in ("hit", "class_swap", "miss", "false_alarm"):
        indices = manifest["gallery"].get(name, [])
        cards = "".join(card(rows_by_index[i]) for i in indices if i in rows_by_index)
        gallery_sections.append(
            f"<h2>{esc(BUCKET_TITLES[name])} "
            f'<span class="meta">({buckets.get(name, 0):,} frames, '
            f"{len(indices)} shown)</span></h2>"
            f'<div class="cards">{cards}</div>',
        )

    probe_cards = []
    for result in probe["results"][:24]:
        row = rows_by_index.get(result["index"])
        if row is None:
            continue
        match = result["class_match"]
        color = DARK["good"] if match else DARK["warn"]
        extra = (
            f'<div><span class="tag" style="background:{color}">forced</span> '
            f"{esc(result['forced_event'])} "
            f'<span class="meta">({esc(result["forced_class"])} vs gt '
            f"{esc(result['gt_class'])} — "
            f"{'match' if match else 'no match'})</span></div>"
        )
        probe_cards.append(card(row, extra))

    match_rate = probe["class_match_rate"]
    residue_rows = "".join(
        f"<tr><td>{esc(text)}</td><td>{count}</td></tr>"
        for text, count in confusion["other_residue"]["gt_top"]
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>er_60k @60000 — events one-off</title><style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 2em auto;
  max-width: 1200px; color: {DARK["text"]}; background: {DARK["bg"]};
  padding: 0 1em; }}
h1, h2, h3 {{ color: {DARK["heading"]}; }}
table {{ border-collapse: collapse; margin: 0.8em 0; font-size: 13px;
  font-variant-numeric: tabular-nums; }}
th, td {{ border: 1px solid {DARK["border"]}; padding: 3px 8px;
  text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: {DARK["th"]}; }}
.meta {{ color: {DARK["meta"]}; font-size: 12.5px; }}
.tiles {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 1em 0; }}
.tile {{ background: {DARK["th"]}; border: 1px solid {DARK["border"]};
  border-radius: 8px; padding: 10px 16px; font-size: 13px;
  color: {DARK["meta"]}; }}
.tile .big {{ font-size: 22px; color: {DARK["heading"]};
  font-variant-numeric: tabular-nums; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fill,
  minmax(260px, 1fr)); gap: 12px; }}
.card {{ background: {DARK["pre"]}; border: 1px solid {DARK["border"]};
  border-radius: 8px; overflow: hidden; font-size: 13px; }}
.card img {{ width: 100%; display: block; }}
.card .lines {{ padding: 8px 10px; display: flex; flex-direction: column;
  gap: 4px; }}
.tag {{ display: inline-block; border-radius: 4px; padding: 0 6px;
  font-size: 11px; color: #121417; font-weight: 600; }}
.tag.model {{ background: {DARK["accent"]}; }}
.tag.gt {{ background: {DARK["meta"]}; }}
.oracle {{ color: {oracle_color}; font-weight: 600; }}
</style></head><body>
<h1>er_60k @60000 — what events does the model see?</h1>
<p class="meta">One-off, owner-requested (2026-08-11 12:44Z) ·
checkpoint fontaine_molmo2_er_60k_ddp4/step_060000 (the new reference
trunk, ER decision read 13:28Z) · narrated all-fields greedy voice over
the k4l2 panel's judge-labeled frames · record-only, rides the er-60k
pre-reg · raw strings shown everywhere, the 13-class binning (frozen at
the launch note, 92.6% gt-vocab coverage) is quant/display only.</p>

<div class="tiles">{tiles}</div>
<p>Presence accuracy (none vs any event, the standard eval's metric):
<b>{acc:.4f}</b> on {n:,} frames — <span class="oracle">
{"within" if oracle_ok else "OUTSIDE"} the cross-world-size numerics
band of the banked endpoint event acc
{confusion["banked_event_acc"]:.4f}</span> (instrument oracle: the
banked run was a 4-way box shard; bf16 batch-composition drift flips
near-tie greedy argmaxes across world sizes — sharding.py — so
near-reproduction, not byte-identity, is the expected relation). Exact-string match on gt-event frames:
{confusion["exact_string_match_on_gt_event_frames"]:.3f}.
Unparsed generations: {confusion["unparsed_generations"]}.</p>

<h2>Model &times; gt event-class confusion <span class="meta">(rows = gt,
cols = model; green diagonal = class agreement, magenta = mass)</span></h2>
<table><tr><th>gt \\ model</th>{header}<th>gt total</th></tr>
{"".join(body_rows)}
<tr><th>model total</th>{model_totals}<td>{n:,}</td></tr></table>

<h2>Per-class precision / recall <span class="meta">(classes with
support on either side)</span></h2>
<table><tr><th>class</th><th>gt frames</th><th>model frames</th>
<th>precision</th><th>recall</th></tr>{pr_rows}</table>

<details><summary class="meta">taxonomy residue — top gt strings the
rules file under "other" ({confusion["per_class"]["other"]["gt_frames"]}
frames)</summary>
<table><tr><th>gt string</th><th>n</th></tr>{residue_rows}</table>
</details>

<h2>Constrained probe — force a guess on the misses</h2>
<p>On the {probe["miss_frames_total"]:,} miss frames (gt event, model
said <i>none</i>): replay the frame's own generated
subgoal/holding/progress lines teacher-forced, then re-decode the event
slot greedy with the exact "none" tokenization banned at its first
step. Replay oracle: the unbanned replay must reproduce "none"
bit-exact ({probe["excluded_replay_oracle"]} excluded).
{probe["still_none_after_ban"]} frames still decoded a none-variant
after the ban.</p>
<p><b>Forced guess matches the gt class on
{probe["class_match"]}/{probe["forced_guesses"]}
({100 * match_rate:.1f}%)</b> — a high rate means the model often
<i>saw</i> the event but scored it under "none"'s threshold; a low rate
means it genuinely didn't encode it.</p>
<div class="cards">{"".join(probe_cards)}</div>

{"".join(gallery_sections)}

<p class="meta">Generated by fontaine/scripts/er60k_events_report.py ·
dump {esc(dump["policy"])} on plan holdout_curated_v0_k4l2 ·
confusion analysis analysis__er60k_events_confusion.json · probe
analysis__er60k_events_probe.json</p>
</body></html>"""
    HTML_OUT.write_text(html)
    print(f"wrote {HTML_OUT} ({HTML_OUT.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["quant", "probe", "html"])
    args = parser.parse_args()
    {"quant": phase_quant, "probe": phase_probe, "html": phase_html}[args.phase]()


if __name__ == "__main__":
    main()
