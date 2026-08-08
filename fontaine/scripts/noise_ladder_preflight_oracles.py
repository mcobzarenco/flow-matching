"""Noise-ladder rung-2 PREFLIGHT adjudicator (#1, pre-reg
2026-08-08-prereg-noise-ladder-perdataset.md, stage-2 oracle item 5):
abort-on-red byte-match between the routed (--noise-ticket-map) decode
and the plain single-ticket decode on the committed 2-dataset ticket-2
preflight plan — the rung-(b) preflight pattern. On green, writes
``reports/analysis__noise_ladder_preflight_oracles.json``; the stage-2
launcher refuses to run without it. NO scalars are read or printed —
this adjudicates the instrument, never the question.

Checks (all abort, never silent):
  (i)   identity columns byte-match between the two npzs (same plan,
        same batch size => same row order — matched composition);
  (ii)  the routed pred stack is BYTE-IDENTICAL to the plain ticket-2
        pred stack (rows of a dataset mapped to ticket t decode exactly
        as a plain ticket-t run of those rows);
  (iii) policy-name provenance: routed npz's single pred key carries
        _ticketmap (never plain _ticket); the plain npz's carries
        _ticket without _ticketmap — a routed read must never pool as a
        single-ticket read;
  (iv)  npz + report provenance: routed run carries the m64 bank sha
        AND ticket_map_sha256 == the amendment-1 extended map sha
        `27858421c6293cca…` at sample_draws 1 (the pre-registered
        `15d9293553ac1a88…` is enforced via the restriction, check vi);
        the plain run carries the t2-only bank's file sha and NO map;
  (v)   bank lineage: the t2-only bank byte-equals m64[2:3];
  (vi)  map coverage (CPU, amendment 1): the committed map reproduces
        the pre-registered sha; the panel-total EXTENDED enumeration
        reproduces its pinned sha, restricts to the committed map
        EXACTLY (selection unchanged), routes every added dataset to
        33 (the non-qualifying fallback), and covers every dataset the
        panel plan decodes (core + labeled); image within top-10+{33};
  (vii) the preflight rows come from exactly the two committed
        datasets, both routed to ticket 2 by the committed map.

--selftest builds synthetic worlds (green + every red class) in a temp
dir and must pass before the real adjudication is trusted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bijou.eval.policies import load_ticket_map

RUN_STEM = "eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000"
ROUTED = f"reports/{RUN_STEM}__noiseladder_preflight_t2_ticketmap_heun30"
PLAIN = f"reports/{RUN_STEM}__noiseladder_preflight_t2_ticket2_heun30"
ANALYSIS = "reports/analysis__noise_ladder_stage01.json"
# Amendment 1 (2026-08-08 16:4xZ, found by THIS adjudicator's first real
# run): the committed map enumerates the probe universe (792 datasets);
# the panel plan carries 86 more with zero probe rows. The pre-reg's own
# rule routes them (non-qualifying → 33); the extended ENUMERATION is
# materialized as its own committed artifact, and the restriction to the
# original 792 must reproduce the pre-registered sha exactly — the
# selection is unchanged, only made total.
EXT_MAP = "plans/noise_ladder_ticketmap_panel.json"
PANEL_PLAN = "plans/holdout_curated_v0_k4l2.json"
M64 = "plans/tickets_goldenticket_m64.npz"
T2 = "plans/tickets_goldenticket_t2.npz"
OUT = "reports/analysis__noise_ladder_preflight_oracles.json"

M64_SHA = "9bb13bc47a92f7cc764e81022a9a7b05dbb9ec391eb9ba8ab14d675c955cc7c0"
MAP_SHA = "15d9293553ac1a8878e0b7b0c385f03127a518d96e408bc1f496f5d8c4ec2173"
EXT_MAP_SHA = "27858421c6293ccaf4d98405a9e8b1f2182480bc63459fea6e27d1e36e0ec6b7"
TOP10 = [33, 2, 0, 51, 10, 59, 38, 28, 15, 36]
PREFLIGHT_DATASETS = ["dragon-95/so100_sorting_3", "shin1107/koch_base_episodes"]
PREFLIGHT_TICKET = 2

IDENTITY = (
    "truth",
    "valid",
    "index",
    "repo_id",
    "episode_index",
    "frame_index",
    "core",
)


def _fail(message: str) -> None:
    raise SystemExit(f"PREFLIGHT ORACLE RED: {message}")


def _pred_key(npz: np.lib.npyio.NpzFile, path: str) -> str:
    """The single bijou policy key — every panel npz also carries the
    state-copy baseline columns (pred:state-copy[-norm]), which are
    byte-matched separately, never adjudicated as the policy."""
    keys = [k for k in npz.files if k.startswith("pred:bijou")]
    if len(keys) != 1:
        _fail(f"{path}: expected exactly one pred:bijou key, got {keys}")
    return keys[0]


def adjudicate(
    routed_npz_path: Path,
    routed_json_path: Path,
    plain_npz_path: Path,
    plain_json_path: Path,
    analysis_path: Path,
    ext_map_path: Path,
    panel_plan_path: Path,
    m64_path: Path,
    t2_path: Path,
    map_sha: str = MAP_SHA,
    ext_map_sha: str = EXT_MAP_SHA,
    m64_sha: str = M64_SHA,
) -> dict[str, object]:
    """Run every check; return the green record (never writes)."""
    for path in (
        routed_npz_path,
        routed_json_path,
        plain_npz_path,
        plain_json_path,
        analysis_path,
        ext_map_path,
        panel_plan_path,
        m64_path,
        t2_path,
    ):
        if not Path(path).exists():
            _fail(f"missing input {path}")

    routed = np.load(routed_npz_path, allow_pickle=False)
    plain = np.load(plain_npz_path, allow_pickle=False)

    # (i) identity — matched composition. State-copy baseline columns
    # are noise-free functions of the rows, so they byte-match too when
    # (and only when) the two runs scored the same rows the same way.
    state_copy = [k for k in routed.files if k.startswith("pred:state-copy")]
    for column in (*IDENTITY, *state_copy):
        if column not in routed.files or column not in plain.files:
            _fail(f"identity column {column!r} missing from an npz")
        if not np.array_equal(routed[column], plain[column]):
            _fail(f"identity column {column!r} differs between npzs")

    # (iii) policy-name provenance.
    routed_key = _pred_key(routed, str(routed_npz_path))
    plain_key = _pred_key(plain, str(plain_npz_path))
    if not routed_key.endswith("_ticketmap"):
        _fail(f"routed pred key {routed_key!r} lacks the _ticketmap suffix")
    if plain_key.endswith("_ticketmap") or not plain_key.endswith("_ticket"):
        _fail(f"plain pred key {plain_key!r} is not a plain _ticket read")

    # (ii) the byte-match itself.
    a, b = routed[routed_key], plain[plain_key]
    if a.dtype != b.dtype or a.shape != b.shape:
        _fail(
            f"pred stacks disagree in dtype/shape: {a.dtype}{a.shape} vs {b.dtype}{b.shape}",
        )
    if not np.array_equal(a, b):
        first = int(np.argwhere(~np.isclose(a, b, rtol=0, atol=0))[0][0])
        _fail(
            f"routed decode != plain ticket-{PREFLIGHT_TICKET} decode (first row {first})",
        )

    # (iv) npz provenance.
    t2_file_sha = hashlib.sha256(Path(t2_path).read_bytes()).hexdigest()
    if str(routed["tickets_sha256"]) != m64_sha:
        _fail(f"routed npz bank sha {routed['tickets_sha256']} != m64 {m64_sha}")
    if str(routed["ticket_map_sha256"]) != ext_map_sha:
        _fail(
            f"routed npz map sha {routed['ticket_map_sha256']} != extended "
            f"{ext_map_sha} (amendment 1: routed runs carry the panel-total "
            "enumeration)",
        )
    if str(plain["tickets_sha256"]) != t2_file_sha:
        _fail(f"plain npz bank sha {plain['tickets_sha256']} != t2 file {t2_file_sha}")
    if str(plain["ticket_map_sha256"]) != "":
        _fail("plain npz unexpectedly carries a ticket map sha")
    routed_report = json.loads(Path(routed_json_path).read_text())
    if routed_report.get("ticket_map_sha256") != ext_map_sha:
        _fail(
            f"routed report map sha {routed_report.get('ticket_map_sha256')} "
            f"!= {ext_map_sha}",
        )
    if not routed_report.get("noise_ticket_map"):
        _fail("routed report carries no noise_ticket_map path")
    if routed_report.get("sample_draws") != 1:
        _fail(f"routed report sample_draws {routed_report.get('sample_draws')} != 1")
    plain_report = json.loads(Path(plain_json_path).read_text())
    if plain_report.get("ticket_map_sha256") is not None:
        _fail("plain report unexpectedly carries a ticket map sha")

    # (v) bank lineage.
    m64_bank = np.load(m64_path, allow_pickle=False)["tickets"]
    t2_bank = np.load(t2_path, allow_pickle=False)["tickets"]
    if hashlib.sha256(Path(m64_path).read_bytes()).hexdigest() != m64_sha:
        _fail(f"m64 bank file sha drifted from {m64_sha}")
    if not np.array_equal(t2_bank, m64_bank[PREFLIGHT_TICKET : PREFLIGHT_TICKET + 1]):
        _fail(
            f"t2-only bank is not m64[{PREFLIGHT_TICKET}:{PREFLIGHT_TICKET + 1}] byte-for-byte",
        )

    # (vi) map coverage on the FULL panel plan (amendment 1): the
    # extended enumeration must (a) reproduce its pinned sha, (b)
    # restrict to the committed 792-dataset map EXACTLY — reproducing
    # the pre-registered sha, so the routed selection is unchanged —
    # (c) route every added dataset to 33 (the pre-reg's non-qualifying
    # fallback), and (d) cover every dataset the panel plan decodes
    # (core AND labeled rows all take noise).
    committed, sha = load_ticket_map(Path(analysis_path), m64_bank.shape[0])
    if sha != map_sha:
        _fail(f"committed map canonical sha {sha} != pre-registered {map_sha}")
    extended, ext_sha = load_ticket_map(Path(ext_map_path), m64_bank.shape[0])
    if ext_sha != ext_map_sha:
        _fail(f"extended map canonical sha {ext_sha} != pinned {ext_map_sha}")
    restriction = {k: v for k, v in extended.items() if k in committed}
    if restriction != committed:
        _fail(
            "extended map restricted to the committed 792 does NOT "
            "reproduce the pre-registered map — the selection drifted",
        )
    added_image = {v for k, v in extended.items() if k not in committed}
    if not added_image <= {33}:
        _fail(
            f"amendment datasets route to {sorted(added_image)} — the "
            "non-qualifying fallback is 33 only",
        )
    panel = json.loads(Path(panel_plan_path).read_text())
    panel_datasets = {row[0] for row in panel["core"]} | {
        row[0] for row in panel.get("labeled", [])
    }
    uncovered = sorted(panel_datasets - set(extended))
    if uncovered:
        _fail(f"map misses {len(uncovered)} panel dataset(s), first {uncovered[:5]}")
    if not set(extended.values()) <= set(TOP10):
        _fail(f"map image {sorted(set(extended.values()))} escapes top-10 + {{33}}")

    # (vii) preflight rows come from the two committed ticket-2 datasets.
    rows = {str(r) for r in routed["repo_id"]}
    if rows != set(PREFLIGHT_DATASETS):
        _fail(f"preflight rows from {sorted(rows)}, expected {PREFLIGHT_DATASETS}")
    for repo in PREFLIGHT_DATASETS:
        if extended[repo] != PREFLIGHT_TICKET or committed[repo] != PREFLIGHT_TICKET:
            _fail(f"{repo} does not route to {PREFLIGHT_TICKET} in both maps")

    return {
        "verdict": "GREEN",
        "rows": int(a.shape[0]),
        "routed_pred_key": routed_key,
        "plain_pred_key": plain_key,
        "tickets_sha256": m64_sha,
        "committed_map_sha256": map_sha,
        "ticket_map_sha256": ext_map_sha,
        "amendment1_added_datasets": len(extended) - len(committed),
        "t2_bank_sha256": t2_file_sha,
        "preflight_datasets": PREFLIGHT_DATASETS,
        "preflight_ticket": PREFLIGHT_TICKET,
        "routed_report": str(routed_json_path),
        "plain_report": str(plain_json_path),
    }


# --- selftest ----------------------------------------------------------


def _selftest() -> None:
    def expect_red(label: str, **kwargs: object) -> None:
        try:
            adjudicate(**kwargs)  # type: ignore[arg-type]
        except SystemExit as err:
            assert "PREFLIGHT ORACLE RED" in str(err), (label, err)
            return
        raise AssertionError(f"selftest world {label!r} unexpectedly green")

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        rng = np.random.default_rng(0)
        bank = rng.standard_normal((64, 5, 3)).astype(np.float32)
        m64_path = tmp / "m64.npz"
        np.savez_compressed(m64_path, tickets=bank)
        m64_sha = hashlib.sha256(m64_path.read_bytes()).hexdigest()
        t2_path = tmp / "t2.npz"
        np.savez_compressed(t2_path, tickets=bank[2:3])
        t2_sha = hashlib.sha256(t2_path.read_bytes()).hexdigest()

        mapping = dict.fromkeys(PREFLIGHT_DATASETS, 2)
        mapping["other/dataset"] = 33
        analysis = tmp / "analysis.json"
        analysis.write_text(json.dumps({"stage1": {"routing_map": mapping}}))
        map_sha = hashlib.sha256(
            json.dumps(mapping, sort_keys=True).encode(),
        ).hexdigest()
        # Amendment-1 shape: the panel plan carries a dataset the
        # committed map never saw; the extended enumeration adds it → 33.
        extended = dict(mapping)
        extended["panelonly/dataset"] = 33
        ext_map = tmp / "ext_map.json"
        ext_map.write_text(json.dumps(extended, sort_keys=True))
        ext_sha = hashlib.sha256(
            json.dumps(extended, sort_keys=True).encode(),
        ).hexdigest()
        panel = tmp / "panel.json"
        panel.write_text(
            json.dumps(
                {
                    "core": [
                        [repo, 0, 0] for repo in [*PREFLIGHT_DATASETS, "other/dataset"]
                    ],
                    "labeled": [["panelonly/dataset", 0, 0]],
                },
            ),
        )

        pred = rng.standard_normal((6, 5, 3)).astype(np.float32)
        identity = {
            "truth": rng.standard_normal((6, 5, 3)).astype(np.float32),
            "valid": np.ones((6, 5), dtype=bool),
            "index": np.arange(6),
            "repo_id": np.array(PREFLIGHT_DATASETS * 3),
            "episode_index": np.arange(6),
            "frame_index": np.arange(6),
            "core": np.ones(6, dtype=bool),
            # Real panel npzs carry the state-copy baseline columns —
            # the fixture models them (the 16:37Z real-data red was
            # this class: an adjudicator blind to extra pred: keys).
            "pred:state-copy": rng.standard_normal((6, 5, 3)).astype(np.float32),
            "pred:state-copy-norm": rng.standard_normal((6, 5, 3)).astype(
                np.float32,
            ),
        }

        def write_world(
            name: str,
            *,
            routed_pred: np.ndarray = pred,
            routed_suffix: str = "_ticketmap",
            plain_suffix: str = "_ticket",
            routed_map_sha: str = ext_sha,
            routed_repo: np.ndarray | None = None,
        ) -> dict[str, Path]:
            routed_npz = tmp / f"{name}_routed.npz"
            plain_npz = tmp / f"{name}_plain.npz"
            routed_id = dict(identity)
            if routed_repo is not None:
                routed_id["repo_id"] = routed_repo
            np.savez_compressed(
                routed_npz,
                **routed_id,
                **{f"pred:bijou@80000{routed_suffix}": routed_pred},
                noise_tickets=np.array(str(m64_path)),
                tickets_sha256=np.array(m64_sha),
                noise_ticket_map=np.array(str(analysis)),
                ticket_map_sha256=np.array(routed_map_sha),
            )
            np.savez_compressed(
                plain_npz,
                **identity,
                **{f"pred:bijou@80000{plain_suffix}": pred},
                noise_tickets=np.array(str(t2_path)),
                tickets_sha256=np.array(t2_sha),
                noise_ticket_map=np.array(""),
                ticket_map_sha256=np.array(""),
            )
            routed_json = tmp / f"{name}_routed.json"
            routed_json.write_text(
                json.dumps(
                    {
                        "noise_ticket_map": str(analysis),
                        "ticket_map_sha256": routed_map_sha,
                        "sample_draws": 1,
                    },
                ),
            )
            plain_json = tmp / f"{name}_plain.json"
            plain_json.write_text(
                json.dumps({"noise_ticket_map": None, "ticket_map_sha256": None}),
            )
            return {
                "routed_npz_path": routed_npz,
                "routed_json_path": routed_json,
                "plain_npz_path": plain_npz,
                "plain_json_path": plain_json,
                "analysis_path": analysis,
                "ext_map_path": ext_map,
                "panel_plan_path": panel,
                "m64_path": m64_path,
                "t2_path": t2_path,
                "map_sha": map_sha,
                "ext_map_sha": ext_sha,
                "m64_sha": m64_sha,
            }

        # Green world.
        record = adjudicate(**write_world("green"))  # type: ignore[arg-type]
        assert record["verdict"] == "GREEN" and record["rows"] == 6
        print("selftest GREEN world: pass")

        # Red worlds.
        flipped = pred.copy()
        flipped[3, 2, 1] = np.nextafter(flipped[3, 2, 1], np.float32(np.inf))
        expect_red("pred byte flip", **write_world("flip", routed_pred=flipped))
        expect_red(
            "routed missing _ticketmap",
            **write_world("suffix", routed_suffix="_ticket"),
        )
        expect_red(
            "map sha mismatch",
            **write_world("mapsha", routed_map_sha="0" * 64),
        )
        expect_red(
            "identity mismatch",
            **write_world(
                "identity",
                routed_repo=np.array([*PREFLIGHT_DATASETS * 2, "x/y", "x/y"]),
            ),
        )
        # Amendment-1 drift: an extension that CHANGES a committed
        # route must abort — the enumeration may only be made total.
        drifted = dict(extended)
        drifted[PREFLIGHT_DATASETS[0]] = 33
        drift_map = tmp / "drift_map.json"
        drift_map.write_text(json.dumps(drifted, sort_keys=True))
        drift_sha = hashlib.sha256(
            json.dumps(drifted, sort_keys=True).encode(),
        ).hexdigest()
        world = write_world("drift", routed_map_sha=drift_sha)
        world["ext_map_path"] = drift_map
        world["ext_map_sha"] = drift_sha
        expect_red("restriction drift", **world)
        print("selftest 5 RED worlds: all abort as required")
        print("SELFTEST PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--routed-npz", type=Path, default=Path(f"{ROUTED}.npz"))
    parser.add_argument("--routed-json", type=Path, default=Path(f"{ROUTED}.json"))
    parser.add_argument("--plain-npz", type=Path, default=Path(f"{PLAIN}.npz"))
    parser.add_argument("--plain-json", type=Path, default=Path(f"{PLAIN}.json"))
    parser.add_argument("--analysis", type=Path, default=Path(ANALYSIS))
    parser.add_argument("--ext-map", type=Path, default=Path(EXT_MAP))
    parser.add_argument("--panel-plan", type=Path, default=Path(PANEL_PLAN))
    parser.add_argument("--m64", type=Path, default=Path(M64))
    parser.add_argument("--t2", type=Path, default=Path(T2))
    parser.add_argument("--out", type=Path, default=Path(OUT))
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return

    record = adjudicate(
        args.routed_npz,
        args.routed_json,
        args.plain_npz,
        args.plain_json,
        args.analysis,
        args.ext_map,
        args.panel_plan,
        args.m64,
        args.t2,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2))
    print(f"PREFLIGHT ORACLES ALL GREEN — wrote {args.out}")


if __name__ == "__main__":
    main()
