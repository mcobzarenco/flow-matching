# Code styleguide

Mantra: **write Python like it's Rust.** Make illegal states hard to
represent, make fallibility visible, and let the type checker prove as much
as it can. Where Python can't enforce something, we enforce it by
convention — and write the convention down here. This is a living document;
rules get added or amended as we hit new cases.

Enforcement lives in `pyproject.toml` (ruff + pyright + pytest, run via
`uv run python check.py`; `--fix` applies formatting and lint fixes).
Anything the linters can check, they do — this file explains *why*, and
covers what they can't. Tests live in `tests/` (pytest, added 2026-07-28
with bijou.fast): pure-CPU, synthetic-data, fast — they run inside
check.py's verdict, so they must stay in the milliseconds-to-seconds
range; anything needing real data or a GPU is a probe in `outputs/`,
not a test. A probe whose numbers the docs cite graduates to the
committed `probes/` (ruff + pyright cover it; run as `python -m
probes.<name>`): gitignored instruments rot silently — the unfreeze
grad-flow probe rotted three ways while its anchors were being
cited, and the doc-cited `probe_rollout_vram.py` is lost outright
(added 2026-08-05).

## Types

- **Everything is annotated.** All function arguments and returns (ruff
  `ANN`, pyright `reportMissingParameterType`). Local annotations where
  inference would widen to `Any`/`Unknown`.
- **`Any` marks a boundary, not a shortcut.** It is allowed only where a
  third-party API is genuinely untyped (transformers, lerobot items,
  wandb). Inside our own code, data crossing a boundary gets parsed into a
  typed structure as early as possible ("parse, don't validate").
- **`@override` on every inherited method** (pyright
  `reportImplicitOverride`): silent overrides are how signature drift
  ships. This includes every `nn.Module.forward`.
- **`# type: ignore` and `cast` are debts.** Each needs a comment naming
  the wrong stub or upstream gap it papers over. Stale ignores are errors
  (`reportUnnecessaryTypeIgnoreComment`).
- **Dead logic is an error**: comparisons/isinstance/casts/contains the
  types make impossible (`reportUnnecessary*`) mean either the code or the
  annotation is wrong. Non-exhaustive `match` over enums is an error
  (`reportMatchNotExhaustive`) — cover every variant, like a Rust `match`.
- **Truthiness is reserved for `bool`.** `if x:` / `x or y` only when
  `x: bool`; everything else compares explicitly (`is None`,
  `== ""`, `len(x) == 0`) — truthiness conflates None with the empty
  string/list/0, which are usually DIFFERENT states in our data
  ("unlabeled" vs "labeled empty"). Exception: idiomatic
  collection-emptiness checks (`if parts:`) where None is impossible
  by type. No linter enforces this (ruff has no such rule as of 0.16;
  the closest families — pylint C1802/1803, refurb FURB110 — push the
  OPPOSITE way, toward implicit truthiness; `FURB` stays off the
  select list for that reason), so it's a review convention. (Added
  2026-08-03: `item.get("condition_subgoal") or subgoal_text(item)`
  sent an operator's empty-string subgoal override to the frame-label
  fallback — the exact state the override existed to suppress.)

## Data: dataclasses over dicts

- **A dict with fixed keys is a struct** — write the `@dataclass`. Dicts
  are for genuinely dynamic key sets (caches, per-repo tables, third-party
  payloads). If you find yourself writing `payload["step"]` for a key that
  always exists, the type is missing.
- Dataclasses are **`frozen=True, slots=True`** unless mutation is the
  point. Immutability by default; opting into mutation is a visible choice.
- **No field defaults on config dataclasses.** Every construction site
  spells out every field — configuration is data, and silent defaults are
  how two runs differ without anyone noticing. (Defaults are fine for
  genuinely optional *behavior* parameters of small value types.)
- Serialization is explicit at the edge: `dataclasses.asdict` /
  `to_dict()` on the way out, a `from_dict()` classmethod that validates
  and converts on the way in (see `bijou/gemma4/config.py`).
- Prefer enums (`EpisodeSplit`, `SamplingMethod`) over string constants;
  parse strings into enums at the CLI/JSON boundary.
- **Released/standard model shapes are staticmethod constructors on the
  config dataclass**, named for the artifact and stating what they match:
  `Gemma4Config.e2b()`, `Molmo2TextConfig.molmo2_4b()`,
  `ActionExpertConfig.released_so100_101()`. The literals for a released
  architecture live in exactly ONE place — probes and tests that need the
  shape call the constructor, never restate the numbers. (Added
  2026-08-11: the MolmoAct2 released expert's 15 fields were spelled out
  in a probe AND a test; module-level `*_config()` factories were the
  older form and were migrated the same day.)
- **One word per concept: the pretrained trunk network (Gemma or
  Molmo2) is the `backbone`** — in
  both senses (the pretrained artifact: `--backbone`,
  `backbone_text.safetensors`/`backbone_vision.safetensors`; and the
  mounted module: `model.backbone`, `backbone_text`/`backbone_vision`
  groups, `backbone_text_trained`/`backbone_vision_trained`). "Trunk"
  is allowed in prose as an informal
  synonym but never in identifiers or schema keys. (Added 2026-08-01:
  the seam refactor briefly introduced a two-sense trunk/backbone
  convention; it produced `trunk.backbone` in the metadata and was
  reverted the same day — a naming rule that needs a judgment call at
  every site is a trap.)
- **One word per layer: a *tokenizer* is the artifact + its math**
  (normalized chunk ↔ body token ids — DCT/BPE, fit and load I/O); **a
  *codec* is the AR conventions around one** (specials and their
  placement, raw-unit quantile glue, per-id symbol lengths — what
  decoders and collators agree on). The unqualified `ActionCodec` is
  the Protocol; implementations are named after the tokenizer they
  wrap (`FastActionCodec` wraps `FastTokenizer`,
  `MolmoAct2ActionCodec` wraps `MolmoAct2FastTokenizer`) — a new
  artifact family lands on this grid without a naming debate. (Added
  2026-08-14: the released-artifact port shipped a tokenizer-layer
  class named `…FastCodec`, and the scaffold reached through the codec
  into `tokenizer.bpe` for symbol lengths — a layer boundary that
  exists only by convention gets violated silently.)

## Functions and arguments

- **Booleans are keyword-only** in our own signatures (ruff `FBT001/002`):
  `f(x, keep_rich=True)`, never `f(x, True)`. Positional bools in calls to
  third-party APIs (`requires_grad_(False)`) are tolerated (`FBT003` off).
- **Explicit `device=` and `dtype=` keywords** on anything that creates
  tensors or modules. No silent inheritance of ambient defaults.
- Behavior-selecting parameters after the data parameters, keyword-only
  (`*,`) when there are more than a couple.
- Unused-but-required parameters (protocol conformance, callbacks) keep
  their public name when the protocol requires it; use a leading
  underscore (`_worker_id`) when the name is ours to choose.

## Classes and inheritance

- **Composition over inheritance.** Subclassing to reuse another
  class's code is banned: share code through free functions or a
  composed helper, and pass collaborators in. In Rust terms: traits
  and enums exist, `extends` does not.
- **Abstract bases are traits**: abstract methods plus, at most,
  algorithm skeletons written purely against them (`ARSuffixDecoder`'s
  suffix scaffold). **Inheriting data is the trap** — a stateful base
  couples every concrete to assumptions no contract states. Whatever a
  base method needs arrives through `__init__` parameters or abstract
  methods, never by assuming what subclasses define; state shared
  between variants is a composed struct, not a parent.
- **Polymorphism, in order of preference**: a `Protocol` when
  implementations only share a surface (`ActionCodec`); a closed union
  with exhaustive `match` at the composition root when the variants'
  signatures legitimately differ (the decoder-module union in
  `bijou.sections.decoder_schema_dict`, the objective payload unions
  in `bijou/models/`); an `nn.Module + abc.ABC` trait only when
  implementations must BE modules (parameters, module tree — the `VLA`
  trait lattice).
- Concretes override only abstract methods — needing to override a
  concrete one means the base's contract is wrong; fix the base.
  Stateful hierarchies are one level below the framework base; mixins
  (stateful behavior injection) are banned. Stateless traits are not
  depth-bounded: a trait may refine another and a class may implement
  several at once — that adds contracts, not state, like Rust trait
  bounds. Extending a concrete class is tolerated only for third-party
  classes where the reference implementation does the same
  (`ScaledEmbedding`).

## Errors and fallibility

- **Loud over lenient.** Anything skipped, dropped, substituted or
  defaulted is printed with the reason (see `select_datasets`' dropped
  list, `StatsAttachedDataset`'s substitution warnings). Silent fallbacks
  are bugs waiting to be discovered in a wandb chart.
- Fail **early**: validate CLI combinations at the parse boundary —
  `parser.error` in `parse_args`, or for a large CLI a `from_namespace`
  classmethod on the args dataclass that receives the parser
  (`bijou.train.TrainArgs`: explicitness rules and checkpoint
  resolution live there; value invariants of the RESOLVED config live
  once in `__post_init__`, whose ValueError `from_namespace` translates
  to `parser.error` — the CLI keeps its usage-line UX and direct
  construction can never build an invalid config). Config compatibility
  before loading weights (`ensure_matching_expert_config`), stats
  before training on them.
- **Checkpoint-inferred flags are refused, not re-validated**: under
  `--resume` every architecture-determining flag errors at the door and
  resolves from the checkpoint (`bijou.train.ARCH_FLAGS`, the write
  side; `loading.CheckpointTrainArgs`, the read side — a sync test pins
  the two). Under `--init-from`, inherited sections refuse their flags;
  `--decoder` is the section-replacement declarator (stage-2). "Flag
  says X, checkpoint says Y, code silently prefers one" is the drift
  class this kills. (Added 2026-08-11, molmo_flow plan step 1.)
- Error messages carry the *values* that failed and, where possible, the
  remedy (`"--eval-samples is required when --holdout-episodes > 0"`).
- Exceptions for programming errors; `SystemExit` with a message for
  user-fixable CLI/environment problems. Catch-and-continue only with a
  bounded retry and a printed reason.

## Modules and imports

- **All imports at the top.** No lazy imports, no `TYPE_CHECKING` blocks —
  if importing something is slow or cyclic, that's an architecture smell
  to fix, not hide.
- **Imports are sorted and grouped** (stdlib / third-party / first-party;
  ruff `I`, `known-first-party = ["bijou"]`). This matches the editor's
  `source.organizeImports.ruff` code action, so on-save organize-imports
  and `check.py` can never disagree.
- **Intra-package imports are relative** (`from .schema import …`,
  `from ..data import repo_id_of`), never `from bijou.…`: the package is
  self-contained under moves/renames, and the relative form makes the DAG
  depth visible at the import site (one dot = sibling, two = parent
  package). Convention, not lint — ruff's only rule here (`TID252`)
  enforces the opposite and stays off. (Added 2026-07-30 after bijou.judge
  landed with absolute self-imports.)
- **The import DAG is strict** and reviewed:
  `train`/`eval`/`rollout`/`judge` → `loading` → `models/*` →
  `vla`/`sections` → `modelling/*` (`…/decoders` → `…/encoders` →
  `modelling/interface` → `modelling/gemma4`/`modelling/molmo2` —
  decoders import the encoders' per-trunk memory types: the producer
  owns its product's type) →
  `fast`, importing downward only; `data` sits beside the model stack
  (loading imports both; `judge` touches only `data`);
  `modelling/aux_text` and `annotations` are leaves.
  `annotations` is the judge-annotation ARTIFACT contract
  (verdict schema, sidecar record + I/O, camera-kind vocabulary, the
  lerobot "event" style registration): the judge WRITES artifacts and
  training READS them, so the shapes live below both —
  `judge/schema.py`/`store.py` re-export the moved names (judge-side
  call sites unaffected), while `SYSTEM_PROMPT`/`PROMPT_HASH` (how
  verdicts are produced) stay judge-side. This is what keeps
  judge → data from ever reversing. No module imports its importer.
  (Added 2026-08-02: `data` needed typed sidecar parsing for
  instruction augmentation; groping the JSON untyped to dodge a DAG
  edge was the smell — moving the contract was the fix.)
  (`checkpoint.py` owns the VLA checkpoint format — schema + IO
  toolkit — because `train` and `eval` both sit above it; `sections.py`
  owns the tagged section schemas/builders every reader shares; the
  LEGACY `bijou_config.json` layout lives only in `convert_legacy.py`,
  reader and writer both.)
- Package CLIs live in `cli.py`, not `__main__.py` (spawn-based
  multiprocessing cannot unpickle objects defined in a package
  `__main__`).

## Formatting

- `ruff format` settles all formatting arguments.
- **One argument per line when a call or signature doesn't fit on one
  line.** Enforced by `COM812`: it inserts the trailing comma on split
  constructs, and the magic trailing comma makes the formatter explode
  them one-per-line. (Verified stable: fix → format reaches a fixpoint;
  `check.py --fix` runs lint fixes before the formatter for this reason.)
  COM812 is selected on `check.py`'s command line (`--extend-select`,
  additive), NOT in `pyproject.toml`: `ruff format` hardcodes a
  conflict advisory whenever the CONFIG selects the rule (no
  suppression knob as of 0.16), and a warning that fires on every
  clean run trains people to skim output — the exact habit the
  verdict-line rule exists to kill. Costs the editor's COM812
  squiggles; acceptable for an autofix-only rule the gate lands
  mechanically. Editor and gate still agree on everything they BOTH
  check — the rule moved gates, it didn't fork.
- Comments explain *why*, not *what*. A comment restating the code is
  noise; a comment recording a measurement, an upstream bug, or a rejected
  alternative is documentation.
- **Docs state the up-to-date truth, never their own edit history.** No
  "corrected on X", "originally said Y", or strike-through trails in any
  markdown — fix the text as if it had always been right; git is the
  history. Dated notes that explain why a RULE or DESIGN exists (the
  incident annotations in this file and working-together.md) are fine:
  they document the world, not the document.
- Docstrings state contracts: shapes, units, device expectations,
  collective behavior (e.g. "every rank must call this at the same step").
  Math typography (τ, ε, σ, −) is welcome; the confusables whitelist in
  `pyproject.toml` covers it.
- **Docstrings are EVERGREEN and self-contained.** They describe what
  the code IS — contract, invariants, semantics — never how it got
  here: no dates, no decision/phase numbers, no sign-offs, no "owner
  call", no author or agent names, no pointers into planning or
  historical documents. A reader must not need the project's history
  to understand a module, and narrative provenance in docstrings rots
  silently (git log and the historical docs in `docs/` are the
  provenance). Pointers to AS-OF-NOW docs (architecture.md sections)
  are fine. The boundary with the neighbouring rules: `#` comments MAY
  carry dated measurements and incident notes (that convention
  stands), and probe/launcher module HEADERS MUST (pre-registered
  expectations are dated by nature) — the docstring rule is about
  library code's contract surface. (Added after the molmoact2
  retirement repeatedly landed module docstrings narrating decisions,
  dates and sign-offs — a recurring class; the fix is the rule plus a
  sweep, not another cleanup commit.)
- **Every function that takes tensors documents their shapes as
  docstring bullets** — a `Shapes:` block, one line per tensor argument
  (and the return): ``- ``x``: [B, T, hidden] role``. Inline `# [B, S]`
  comments may stay, but the docstring is the contract a caller reads.
  (Added 2026-08-11 with the molmoact2 port review; `bijou/molmoact2`
  is the reference example.)

## Determinism and measurement

- Every stochastic path takes an explicit seed or `torch.Generator`;
  "seeded by default from global state" does not count. Distinct concerns
  get distinct seeds (`--seed` for training, `--eval-seed` for probes,
  `--split-seed` for the episode holdout) so changing one cannot silently
  change another.
- Numbers in docs/commit messages are measured, not estimated — and say
  *how* they were measured. Regressions are checked against oracles
  (the 2-step tiny-backbone loss reproduction) before claiming
  "no behavior change".

## Not enforced (yet), by choice

- pyright strict mode / `reportUnknown*`: drowned by untyped lerobot and
  transformers at every call site. Revisit if/when stubs improve.
- `reportUninitializedInstanceVariable`: incompatible with torch's
  `register_buffer` pattern (class-level `Tensor` annotation, assignment
  via `register_buffer` in `__init__`).
- ruff `ARG` (unused arguments): protocol conformance requires
  name-stable parameters; the underscore convention covers the rest.
- ruff `PLR` complexity metrics, `TRY003` (long exception messages are a
  feature here; `TRY004` *is* enforced), `T20` (print *is* the logging
  system), `ISC001/002` (formatter conflict / every argparse help string;
  `ISC004` *is* enforced).

## Toolchain lockstep

- The dev-dependency ruff version tracks the binary Zed's extension
  bundles (editor diagnostics come from that binary, resolving the same
  `pyproject.toml`). Version skew = default-rule skew: ruff 0.16 added
  ISC004/DTZ005/TRY004 to the *defaults* that `extend-select` builds on,
  which made the editor flag code the older CLI accepted. When the editor
  and `check.py` disagree, compare `ruff --version` first.
