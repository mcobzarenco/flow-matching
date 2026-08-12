"""Auxiliary text targets from judge annotations (ar_backbone suffix).

Draws a judged frame's REQUEST SET (which fields the prompt's
``[generate|…]`` conditioning asks for) and renders the corresponding
HEADERLESS value lines that precede BOA in the ar_backbone suffix —
for ``[generate|subgoal holding progress event actions]`` e.g.:

    reach toward the toy boat\n
    no\n
    30%\n
    none\n

Which field a line answers is pinned by request order (template order
always), not by generated header bytes. Requested ⊆ labeled, always:
subgoal exists on every frame of a judged episode;
holding/progress/visible only on judge-sampled frames — the finite
mask IS the sampled-frame set; event wherever its status is KNOWN
(firing frames → the text; sampled no-event frames → the explicit
``none`` — unsampled frames are unknown, never requested; per
docs/episode-annotations.md). The model therefore learns
p(value | observation, asked) — label PRESENCE is conditioning, never
a prediction target (no field ever asks "was this frame judged").
Unjudged samples request nothing and train the ``[generate|actions]``
fast path — an aux-enabled fine-tune extends the base rather than
fighting it.

Label provenance is the dataset's own ``meta/judge_annotations.json``
stamp — the blessed materialization, consumed as-is (selection records
each dataset's stamp; the checkpoint carries the distinct set). There
is deliberately NO code-level prompt-hash pin: the judge prompt
advances with the judging code, not with materialized labels — pin
per-run via ``--aux-prompt-hash`` when a sweep must fail loudly on a
mid-sweep re-materialization.

The template is VERSIONED (:data:`AUX_TEMPLATE_VERSION`) and recorded in
the checkpoint's decoder section: inference decodes values under this
line protocol (\\n-terminated, request-ordered), so a byte-level
convention change on an existing checkpoint silently breaks decoding —
version it instead.

Field order is fixed (subgoal, holding, progress, event); ``fields``
selects a subset but never reorders. Aux token ids are ordinary
text-vocabulary ids (the full-vocab head was chosen for exactly this);
action ids live in the FAST block. The collator assembles both into one
suffix tensor in BACKBONE id space.

The artifact vocabulary (EVENT_STYLE + its lerobot registration, the
camera-kind vocabulary, the verdict schema) lives one module lower, in
``bijou.annotations`` — the contract leaf both the judge (writer) and
the training readers import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, override

import torch
import transformers
from lerobot.datasets.language_render import active_at
from torch import Tensor

from .annotations import EVENT_STYLE

# v4 (2026-08-03): HEADERLESS values. The suffix carries only the
# requested fields' VALUES, ``\n``-terminated, in request order — which
# field a line answers is pinned by the prompt's ``[generate|…]`` list,
# not by generated header bytes (they were fully predictable given the
# request: pure padding). Values are \n-sanitized at render — a stray
# newline inside a judge string would shift every later line's
# supervision (with v3 headers that was a local parse failure;
# headerless it would be silent misalignment). Training assembles the
# suffix from PER-FIELD encodings (never one joint string), so
# cross-line merges cannot exist by construction; the remaining
# tokenizer contract — ``enc(value) + enc(\n) == enc(value + \n)`` — is
# asserted by :func:`build_aux_runtime` (measured to hold on the real
# E2B tokenizer for every field's value class, 2026-08-03).
AUX_TEMPLATE_VERSION = 4
# Suffix format 5: ``[GENERATION_OPENER][value\n per requested][BOA]
# [t_1..t_k]``. The opener is Gemma-4's OWN generation prompt — the
# exact string apply_chat_template appends with
# add_generation_prompt=True, `<|turn>model\n` = ids [105, 4368, 107]
# on E2B (verified against the real checkpoint 2026-08-02). What speaks
# is commanded by the PROMPT's ``[generate|…]`` conditioning (always
# present, ``actions`` always terminal): label presence is explained
# away per FIELD — the model learns p(value | obs, asked), never
# "was this frame judged". BOA survives as the action block's own
# begin-marker (single-id constrained-decode anchor; codec output
# consumed verbatim). Formats < 5 (fed
# [ACT]/[AUX] mode tokens, state as suffix position 0, v≤3 header
# bytes) are REFUSED — no trained artifact worth loading exists
# (owner call, 2026-08-03).
SUFFIX_FORMAT = 5
GENERATION_OPENER = "<|turn>model\n"
FIELD_TERMINATOR = "\n"
HOLDING_VALUES = ("no", "yes")  # indexed by the 0/1 label
# The explicit event negative: requested on judge-sampled no-event
# frames (a TRUE "nothing happened" — unsampled frames are unknown and
# never requested), so event presence is read from the VALUE, not from
# whether the model chose to emit a line.
EVENT_NONE = "none"
# The generate-list key and its terminal word ([generate|subgoal …
# actions]): every sample's prompt carries it — actions are always
# requested and always last, so the list fully describes the suffix.
GENERATE_KEY = "generate"
ACTIONS_WORD = "actions"
# LeRobot's camera-frame key convention — single-sourced here (the DAG
# leaf); the collator imports it.
IMAGE_KEY_PREFIX = "observation.images."


class AuxField(StrEnum):
    """Aux fields in their fixed template order (new fields append —
    existing checkpoints' trained prefixes stay stable)."""

    SUBGOAL = "subgoal"
    HOLDING = "holding"
    PROGRESS = "progress"
    EVENT = "event"
    VISIBLE = "visible"


# Per-field VALUE token budgets at decode (terminator excluded):
# free-text fields match the render-side truncation caps; constrained/
# short fields get their worst case with slack. Exhaustion forces the
# terminator and counts a fallback.
VALUE_BUDGETS: dict[AuxField, int] = {
    AuxField.SUBGOAL: 20,
    AuxField.HOLDING: 2,
    AuxField.PROGRESS: 6,
    AuxField.EVENT: 24,
    AuxField.VISIBLE: 16,
}


def generate_text(request: tuple[AuxField, ...]) -> str:
    """The prompt's ``[generate|…]`` conditioning bracket for one
    sample: requested fields in template order, ``actions`` terminal
    and unconditional — ``[generate|actions]`` is the deployment fast
    path."""
    words = [f.value for f in request] + [ACTIONS_WORD]
    return f"[{GENERATE_KEY}|{' '.join(words)}]"


class TextTokenizer(Protocol):
    """The slice of a HF tokenizer the aux renderer/decoder uses (tests
    inject a stub; runtime uses AutoTokenizer)."""

    def encode(self, text: str, *, add_special_tokens: bool = ...) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


@dataclass(frozen=True, slots=True)
class AuxDecodeConfig:
    """The aux record a checkpoint carries in its decoder section —
    everything inference needs to elicit the fields the model trained on,
    plus label provenance. ``template_version`` pins the value/terminator
    byte conventions (AUX_TEMPLATE_VERSION when written); a version this
    code does not know is a loud error, never a guess."""

    template_version: int
    fields: tuple[AuxField, ...]
    prompt_hash: str
    judge_model: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_version": self.template_version,
            "fields": [f.value for f in self.fields],
            "prompt_hash": self.prompt_hash,
            "judge_model": self.judge_model,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuxDecodeConfig:
        version = int(data["template_version"])
        if version != AUX_TEMPLATE_VERSION:
            raise SystemExit(
                f"checkpoint aux template_version {version} != this code's "
                f"{AUX_TEMPLATE_VERSION} — decoding would elicit a format "
                "the model never trained on",
            )
        return cls(
            template_version=version,
            fields=tuple(AuxField(f) for f in data["fields"]),
            prompt_hash=str(data["prompt_hash"]),
            judge_model=str(data["judge_model"]),
        )


@dataclass(frozen=True, slots=True)
class AuxRuntime:
    """Tokenized decode constants, built once from the checkpoint's text
    tokenizer (:func:`build_aux_runtime`). Value ids are decoded under
    per-field budgets/constraints; ``terminator_id`` ends every field's
    value line."""

    config: AuxDecodeConfig
    tokenizer: TextTokenizer
    value_candidates: dict[AuxField, tuple[tuple[int, ...], ...]]
    terminator_id: int


# Representative values per field for the boundary tripwire below
# (HOLDING checks its REAL candidates, EVENT its explicit negative —
# a letter-initial phrase, a digit, positional indices).
_BOUNDARY_PROBES: dict[AuxField, tuple[str, ...]] = {
    AuxField.SUBGOAL: ("reach toward the object",),
    AuxField.HOLDING: HOLDING_VALUES,
    AuxField.PROGRESS: ("85%",),
    AuxField.EVENT: ("object dropped", EVENT_NONE),
    AuxField.VISIBLE: ("object 0,1; gripper none",),
}


def build_aux_runtime(
    config: AuxDecodeConfig,
    tokenizer: TextTokenizer,
    *,
    newline_carrier_ban: bool = False,
) -> AuxRuntime:
    """Tokenize the versioned template's decode constants. The field
    terminator must be a single token (true for \\n under the Gemma
    tokenizer and the test stub) — anything else is a loud error, since
    value decoding detects termination by one id.

    Construction also asserts the headerless template's one remaining
    tokenizer contract on every configured field:
    ``encode(value) + encode(\\n) == encode(value + \\n)`` for
    representative values (the real candidates for constrained fields).
    Training assembles suffixes from per-field encodings, so a violation
    means the decode-side forced terminator would sit off the training
    manifold — the v2 header template broke the equivalent property
    (space merged into " yes") and shipped a silently-wrong holding
    metric.

    ``newline_carrier_ban``: the caller's decoder bans every text id
    whose decoded bytes carry a newline during value decoding
    (ARSuffixDecoder.newline_carrier_ids), which restores the trained
    split-form termination by construction — the boundary probe then
    downgrades to a note (Qwen's BPE merges ``'%\\n'`` and fails the
    probe; Gemma passes it and keeps the strict contract)."""

    def encode(text: str) -> tuple[int, ...]:
        return tuple(tokenizer.encode(text, add_special_tokens=False))

    terminator = encode(FIELD_TERMINATOR)
    if len(terminator) != 1:
        raise SystemExit(
            f"aux field terminator {FIELD_TERMINATOR!r} tokenizes to "
            f"{len(terminator)} ids — single-token terminator required",
        )
    for aux_field in config.fields:
        for value in _BOUNDARY_PROBES[aux_field]:
            split = encode(value) + terminator
            joint = encode(value + FIELD_TERMINATOR)
            if split != tuple(joint):
                if newline_carrier_ban:
                    print(
                        f"[aux] boundary probe differs for "
                        f"{aux_field.value!r} (part {list(split)} != joint "
                        f"{list(joint)}) — tolerated: the decoder bans "
                        "newline-carrier ids, so decode terminates on the "
                        "trained split-form terminator",
                        flush=True,
                    )
                    continue
                raise SystemExit(
                    f"aux template boundary broken for {aux_field.value!r}: "
                    f"part-encoding {list(split)} != joint {list(joint)} — "
                    "this tokenizer merges values into the terminator, so "
                    "decode-forced terminators would sit off the training "
                    "manifold; the template must change with the tokenizer "
                    "(AUX_TEMPLATE_VERSION bump)",
                )
    candidates: dict[AuxField, tuple[tuple[int, ...], ...]] = {
        # Constrained value set; first-token argmax picks the candidate,
        # its remaining ids are forced.
        AuxField.HOLDING: tuple(encode(value) for value in HOLDING_VALUES),
    }
    return AuxRuntime(
        config=config,
        tokenizer=tokenizer,
        value_candidates={f: candidates[f] for f in config.fields if f in candidates},
        terminator_id=terminator[0],
    )


@dataclass(frozen=True, slots=True)
class AuxGeneration:
    """One sample's generated aux fields (raw text + lenient parses;
    None = field not elicited or unparseable — the raw text is the
    ground truth for reports)."""

    text: str
    subgoal: str | None
    holding: bool | None
    progress: float | None
    event: str | None
    visible: str | None


@dataclass
class AuxSpec:
    """Aux-rendering configuration + the lazy text tokenizer.

    Pickled into spawned dataloader workers — the tokenizer and the
    dropout generator are built lazily and dropped in ``__getstate__``
    (the GemmaInputsCollator convention). ``annotated_repos`` are the
    repo ids whose annotation stamps were verified at selection time
    (stale-hash datasets are treated as unjudged, loudly, by selection —
    not here). ``block_base`` maps codec ids into backbone id space for
    the assembled suffix.

    ``dropout``: probability a LABELED sample's request set collapses to
    ``{actions}`` (the deployment fast path stays trained under dense
    annotation). ``field_dropout``: per labeled field, independent
    probability of dropping it from the request set — request and
    target always move together, so all SUBSETS of the labeled set
    appear in the data distribution (inference-time partial requests
    stay in-distribution). Draws come from a per-process generator
    seeded from ``torch.initial_seed()`` (in a dataloader worker: a
    pure function of --seed, rank and worker id — same seed, same
    dropout pattern; probe collators run dropout-0 clones so eval
    always sees the true labels)."""

    tokenizer_dir: str
    fields: tuple[AuxField, ...]
    annotated_repos: frozenset[str]
    block_base: int
    dropout: float
    field_dropout: float
    # Build the text tokenizer from the checkpoint's own tokenizer.json
    # (tokenizers backend) instead of AutoTokenizer — required for
    # trust_remote_code-pinned checkpoints (Molmo2), whose AutoTokenizer
    # load prompts interactively and cannot run in dataloader workers.
    native_backend: bool = False
    # 16 truncated frequently on the curated corpus's judge subgoals
    # (observed in the first full-recipe run's logs); 20 = the decode
    # VALUE_BUDGETS cap.
    max_subgoal_tokens: int = 20
    # Wider than subgoal: multi-event frames join with "; ".
    max_event_tokens: int = 24
    _tokenizer: Any = field(default=None, repr=False, compare=False)
    _generator: torch.Generator | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        ordered = tuple(f for f in AuxField if f in self.fields)
        if ordered != self.fields:
            raise ValueError(
                f"aux fields must keep template order {[f.value for f in AuxField]}; "
                f"got {[f.value for f in self.fields]}",
            )
        if not self.fields:
            raise ValueError("aux enabled with no fields — pass aux=None instead")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"aux dropout {self.dropout} outside [0, 1)")
        if not 0.0 <= self.field_dropout < 1.0:
            raise ValueError(
                f"aux field dropout {self.field_dropout} outside [0, 1)",
            )

    @override
    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_tokenizer": None, "_generator": None}

    def tokenizer(self) -> TextTokenizer:
        if self._tokenizer is None:
            # Built lazily worker-side (the instance is dropped by
            # __getstate__); the tokenizer files are the processor dir's.
            if self.native_backend:
                from .molmo2.tokenizer import Molmo2TextTokenizer

                self._tokenizer = Molmo2TextTokenizer(self.tokenizer_dir)
            else:
                self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                    self.tokenizer_dir,
                )
        return self._tokenizer

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer().encode(text, add_special_tokens=False)

    def _rng(self) -> torch.Generator:
        if self._generator is None:
            self._generator = torch.Generator().manual_seed(torch.initial_seed())
        return self._generator

    def field_value(
        self,
        aux_field: AuxField,
        item: dict[str, Any],
    ) -> str | None:
        """One field's VALUE text at this frame, or None when its label
        does not exist (the field is then never requested). Values are
        \\n-sanitized — headerless lines mean a stray newline would
        shift every later line's supervision."""
        match aux_field:
            case AuxField.SUBGOAL:
                text = subgoal_text(item)
                return None if text is None else _sanitize(text)
            case AuxField.HOLDING:
                value = item.get("annotation.holding")
                if value is None or not bool(torch.isfinite(value)):
                    return None
                return HOLDING_VALUES[int(value)]
            case AuxField.PROGRESS:
                value = item.get("annotation.progress")
                if value is None or not bool(torch.isfinite(value)):
                    return None
                return f"{round(float(value) * 100)}%"
            case AuxField.EVENT:
                text = events_text(item)
                if text is not None:
                    return _sanitize(text)
                # Explicit negative on judge-sampled frames only: the
                # finite progress mask IS the sampled-frame set; event
                # status elsewhere is unknown, never "none".
                sampled = item.get("annotation.progress")
                if sampled is not None and bool(torch.isfinite(sampled)):
                    return EVENT_NONE
                return None
            case AuxField.VISIBLE:
                return visibility_text(item)

    def value_ids(self, aux_field: AuxField, value: str) -> list[int]:
        """``value\\n`` token ids, truncation-capped for the free-text
        fields (loudly — a capped target is a shortened prefix of the
        label)."""
        body = self._encode(value)
        cap = {
            AuxField.SUBGOAL: self.max_subgoal_tokens,
            AuxField.EVENT: self.max_event_tokens,
        }.get(aux_field)
        if cap is not None and len(body) > cap:
            print(
                f"[aux] truncating {aux_field.value} ({len(body)} > "
                f"{cap} tokens): {value!r}",
                flush=True,
            )
            body = body[:cap]
        return body + self._encode(FIELD_TERMINATOR)

    def draw(
        self,
        item: dict[str, Any],
        *,
        suppress_subgoal: bool = False,
    ) -> tuple[tuple[AuxField, ...], list[int]]:
        """One sample's (request set, target value ids) — always
        consistent by construction: a field is requested iff its label
        exists AND it survives the dropout draws, and exactly the
        requested fields' ``value\\n`` ids are supervised, in template
        order. Empty request for unjudged frames, datasets whose stamp
        failed verification, and — with probability ``dropout`` — for
        labeled samples (the sample then trains the ``[generate|
        actions]`` fast path). ``suppress_subgoal`` excludes the subgoal
        FIELD (anti-copy coupling: when the collator put the subgoal in
        the PROMPT, requesting it would train copying —
        prompt-conditioning and prediction are exact complements)."""
        if item.get("repo_id") not in self.annotated_repos:
            return (), []
        request: list[AuxField] = []
        ids: list[int] = []
        for aux_field in self.fields:
            if suppress_subgoal and aux_field is AuxField.SUBGOAL:
                continue
            value = self.field_value(aux_field, item)
            if value is None:
                continue
            if self.field_dropout > 0.0 and (
                float(torch.rand((), generator=self._rng())) < self.field_dropout
            ):
                continue
            request.append(aux_field)
            ids.extend(self.value_ids(aux_field, value))
        if (
            request
            and self.dropout > 0.0
            and float(torch.rand((), generator=self._rng())) < self.dropout
        ):
            return (), []
        return tuple(request), ids


def _sanitize(value: str) -> str:
    """Newlines inside a value would break the headerless line protocol
    (every later line shifts one field) — collapse them to spaces."""
    return " ".join(value.split())


def subgoal_text(item: dict[str, Any]) -> str | None:
    """The frame's current subgoal segment label (piecewise-constant
    coverage on judged episodes), or None. Shared by the aux renderer
    (prediction target) and the collator's prompt conditioning (C2).
    Rollout/policy items carry no timestamp (they are not dataset
    frames) — that is "no label", not an error."""
    timestamp = item.get("timestamp")
    if timestamp is None:
        return None
    row = active_at(
        float(timestamp),
        persistent=item.get("language_persistent") or [],
        style="subtask",
    )
    if row is None or not row.get("content"):
        return None
    return str(row["content"])


def camera_prompt_order(kinds: dict[str, str], names: list[str]) -> list[str]:
    """Camera short names in PROMPT order: sorted by (semantic kind,
    short name), kinds RAW from the dataset's map (missing ⇒
    "unknown") — never the dropout-applied kind, so a kind-dropout
    draw retags but can never reorder images. The single source of
    camera order for the prompt (Collator.cameras_of) and the
    positional ``visible`` indices; untagged datasets (empty map)
    degenerate to plain short-name order — the pre-tag behavior."""
    return sorted(names, key=lambda name: (kinds.get(name, "unknown"), name))


# Loud-once bookkeeping for visibility surface disagreements (per
# worker process): a broken dataset prints one reason, not one line per
# frame — the StatsAttachedDataset substitution-print precedent.
_VISIBILITY_MISMATCH_WARNED: set[str] = set()


def _visibility_mismatch(repo_id: str, reason: str) -> None:
    if repo_id not in _VISIBILITY_MISMATCH_WARNED:
        _VISIBILITY_MISMATCH_WARNED.add(repo_id)
        print(
            f"[aux] {repo_id}: {reason} — the 'visible' field is skipped "
            "for this dataset's frames until its camera kinds and "
            "annotation columns are re-materialized in sync (printed "
            "once per dataset per worker)",
            flush=True,
        )


def visibility_text(item: dict[str, Any]) -> str | None:
    """``object 0,1; gripper 1`` — which cameras can see the task object
    and the gripper on this frame, or None when the frame wasn't
    judge-sampled (NaN mask) or no camera map travels with the item.
    Cameras are referenced by their PROMPT POSITION (ascending indices
    into :func:`camera_prompt_order`) — positional on purpose: kind
    names collide (two "unknown" cameras), short names are
    dataset-internal vocabulary the prompt never shows, and indices
    stay invariant under camera-kind dropout. Slots are read in the
    feature's storage order (sorted short names). Cameras seeing
    nothing render "none" — a TRUE negative on sampled frames
    (occlusion is signal). Surface disagreements (kinds map ≠ vector
    slots ≠ the item's cameras) render nothing, LOUDLY once per
    dataset: guessing through misaligned slots would label the wrong
    cameras."""
    kinds = item.get("camera_kinds") or {}
    if len(kinds) == 0:
        return None
    names = sorted(kinds)
    item_cameras = sorted(
        key.removeprefix(IMAGE_KEY_PREFIX)
        for key in item
        if key.startswith(IMAGE_KEY_PREFIX)
    )
    if len(item_cameras) > 0 and item_cameras != names:
        _visibility_mismatch(
            str(item.get("repo_id", "<unknown>")),
            f"camera kinds map covers {names} but the item carries "
            f"cameras {item_cameras}",
        )
        return None
    order = camera_prompt_order(kinds, names)
    index_of = {name: i for i, name in enumerate(order)}
    parts: list[str] = []
    for label, key in (
        ("object", "annotation.visible_object"),
        ("gripper", "annotation.visible_gripper"),
    ):
        value = item.get(key)
        if value is None:
            return None
        # Single-camera datasets store shape-(1,) features as scalars
        # (lerobot convention) — normalize before the slot walk.
        vector = torch.atleast_1d(value)
        if vector.numel() != len(names):
            _visibility_mismatch(
                str(item.get("repo_id", "<unknown>")),
                f"{key} has {vector.numel()} slot(s) for {len(names)} camera kind(s)",
            )
            return None
        if not bool(torch.isfinite(vector).all()):
            return None  # not judge-sampled — the normal sparse case
        seen = sorted(
            index_of[name]
            for name, slot in zip(names, vector.tolist(), strict=True)
            if slot >= 0.5
        )
        cameras = ",".join(str(i) for i in seen) if len(seen) > 0 else "none"
        parts.append(f"{label} {cameras}")
    return "; ".join(parts)


def parse_visibility(text: str) -> tuple[frozenset[int], frozenset[int]] | None:
    """Lenient inverse of :func:`visibility_text`: ``object 0,1; gripper
    none`` -> ({0, 1}, {}) as (object slots, gripper slots) in prompt
    positions, or None when the text doesn't parse (a malformed
    generation — callers skip, they never guess). Order-insensitive:
    ``0,1`` and ``1,0`` parse equal, so set equality is the comparison."""
    parts = [part.strip() for part in text.split(";")]
    if len(parts) != 2:
        return None
    slots: list[frozenset[int]] = []
    for prefix, part in zip(("object", "gripper"), parts, strict=True):
        if not part.startswith(prefix):
            return None
        rest = part.removeprefix(prefix).strip()
        if rest == "none":
            slots.append(frozenset())
            continue
        try:
            slots.append(frozenset(int(token) for token in rest.split(",")))
        except ValueError:
            return None
    return slots[0], slots[1]


def events_text(item: dict[str, Any]) -> str | None:
    """All events that fired on this frame, "; "-joined, or None.
    ``language_events`` is FRAME-LOCAL in lerobot items, so every
    event-style row in it belongs to this exact frame — read directly
    rather than through ``emitted_at``, whose single-row resolver raises
    on multi-event frames (drop + regression on one frame is real data;
    it killed a corpus run on 2026-08-02)."""
    contents = [
        str(row["content"])
        for row in item.get("language_events") or []
        if row.get("style") == EVENT_STYLE and row.get("content")
    ]
    if not contents:
        return None
    return "; ".join(contents)


def label_values(
    item: dict[str, Any],
    fields: tuple[AuxField, ...],
) -> dict[AuxField, str]:
    """The LABEL value per configured field where one exists at this
    frame — presence rules identical to :meth:`AuxSpec.field_value`
    minus tokenization concerns (pure strings, no truncation caps, no
    \\n sanitization — on capped fields the trained target may be a
    shortened prefix of what this shows)."""
    values: dict[AuxField, str] = {}
    for aux_field in fields:
        match aux_field:
            case AuxField.SUBGOAL:
                text = subgoal_text(item)
                if text is not None:
                    values[aux_field] = text
            case AuxField.HOLDING:
                value = item.get("annotation.holding")
                if value is not None and bool(torch.isfinite(value)):
                    values[aux_field] = HOLDING_VALUES[int(value)]
            case AuxField.PROGRESS:
                value = item.get("annotation.progress")
                if value is not None and bool(torch.isfinite(value)):
                    values[aux_field] = f"{round(float(value) * 100)}%"
            case AuxField.EVENT:
                text = events_text(item)
                if text is not None:
                    values[aux_field] = text
                else:
                    sampled = item.get("annotation.progress")
                    if sampled is not None and bool(torch.isfinite(sampled)):
                        values[aux_field] = EVENT_NONE
            case AuxField.VISIBLE:
                text = visibility_text(item)
                if text is not None:
                    values[aux_field] = text
    return values


def display_text(values: dict[AuxField, str]) -> str:
    """Human-readable ``field: value`` lines for report tables — the
    DISPLAY layer re-attaches field names; the model's suffix bytes are
    headerless values only."""
    return "".join(f"{f.value}: {v}\n" for f, v in values.items())


def aux_label_text(item: dict[str, Any], fields: tuple[AuxField, ...]) -> str:
    """The display-form label reference column next to generations in
    eval tables (see :func:`label_values`/:func:`display_text`)."""
    return display_text(label_values(item, fields))


def assemble_suffix(
    aux_ids: list[list[int]],
    action_tokens: Tensor,
    *,
    block_base: int,
    codec_pad: int,
) -> tuple[Tensor, Tensor]:
    """Per-sample ``[aux text ids][block_base+BOA..actions]`` rows, padded
    to the batch max with the block PAD id.

    ``action_tokens``: the collator's ``[BOA, t_1..t_k]`` codec-id rows,
    already PAD-padded to their own max (PAD-padding is preserved — the
    loss ignores PAD wherever it sits). Returns (suffix_tokens [B, W]
    long, suffix_is_aux [B, W] bool), both in BACKBONE id space.
    """
    batch = action_tokens.shape[0]
    if len(aux_ids) != batch:
        raise ValueError(f"{len(aux_ids)} aux rows for batch {batch}")
    blocks = action_tokens + block_base
    pad_id = block_base + codec_pad
    widths = [len(aux) + blocks.shape[1] for aux in aux_ids]
    width = max(widths)
    suffix = torch.full((batch, width), pad_id, dtype=torch.long)
    is_aux = torch.zeros((batch, width), dtype=torch.bool)
    for i, aux in enumerate(aux_ids):
        if aux:
            # Text ids must stay below the block: an id inside the
            # reserved run would silently alias an action token in the
            # loss/decode routing (never produced by a real tokenizer,
            # but "never" is what asserts are for).
            if max(aux) >= block_base:
                raise ValueError(
                    f"aux row {i} contains id {max(aux)} >= block_base "
                    f"{block_base} — aux ids must be text-vocabulary ids",
                )
            suffix[i, : len(aux)] = torch.tensor(aux, dtype=torch.long)
            is_aux[i, : len(aux)] = True
        suffix[i, len(aux) : len(aux) + blocks.shape[1]] = blocks[i]
    return suffix, is_aux
