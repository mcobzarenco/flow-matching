"""Auxiliary text targets from judge annotations (ar_backbone suffix).

Renders a judged frame's annotations into the aux text segment that
precedes BOA in the ar_backbone suffix:

    subgoal: reach toward the toy boat\n
    holding: no\n
    progress: 30%\n
    event: object dropped from gripper\n

Presence-based: a field appears iff its label exists at this frame
(subgoal on every frame of a judged episode; holding/progress only on
judge-sampled frames — the finite mask IS the sampled-frame set;
event only on the exact firing frame, positives-only — the implicit
negative is the trained transition past it wherever labels exist; per
docs/episode-annotations.md). Unjudged samples produce no aux at all
and train mode [ACT] — an aux-enabled fine-tune extends the base
rather than fighting it.

Label provenance is the dataset's own ``meta/judge_annotations.json``
stamp — the blessed materialization, consumed as-is (selection records
each dataset's stamp; the checkpoint carries the distinct set). There
is deliberately NO code-level prompt-hash pin: the judge prompt
advances with the judging code, not with materialized labels — pin
per-run via ``--aux-prompt-hash`` when a sweep must fail loudly on a
mid-sweep re-materialization.

The template is VERSIONED (:data:`AUX_TEMPLATE_VERSION`) and recorded in
the checkpoint's decoder section: inference elicits fields by forcing
these exact header strings, so a byte-level template change on an
existing checkpoint silently breaks decoding — version it instead.

Field order is fixed (subgoal, holding, progress, event); ``fields``
selects a subset but never reorders. Aux token ids are ordinary
text-vocabulary ids (the full-vocab head was chosen for exactly this);
action ids live in the FAST block. The collator assembles both into one
suffix tensor in BACKBONE id space.

This module also owns the project-local lerobot "event" language style
registration (idempotent set-adds on lerobot's documented import-time
hook) — it is the DAG leaf both the judge (writer) and training
(reader) sit above, so importing either side makes event rows
resolvable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, override

import torch
import transformers
from lerobot.datasets.language import (
    EVENT_ONLY_STYLES,
    EXTENDED_STYLES,
    STYLE_REGISTRY,
)
from lerobot.datasets.language_render import active_at, emitted_at
from torch import Tensor

# The project-local lerobot language style event rows are stored under.
# bijou.judge.materialize writes rows with its own equal constant —
# test-gated in tests/test_aux_text.py.
EVENT_STYLE = "event"
EXTENDED_STYLES.add(EVENT_STYLE)
EVENT_ONLY_STYLES.add(EVENT_STYLE)
STYLE_REGISTRY.add(EVENT_STYLE)

AUX_TEMPLATE_VERSION = 2
# Suffix format 3 (the only trained format going forward): every
# ar_backbone suffix — aux or not — is [state][GENERATION_OPENER][MODE]
# [aux text when labeled][BOA][actions]. The opener is the IT chat
# template's own generation prompt (introduced by format 2, which
# OPENER_SUFFIX_FORMAT marks for per-feature legacy gating); the MODE
# token is FED, never predicted — [AUX] on samples that carry aux
# supervision, [ACT] otherwise — so "speak vs act" is commanded by the
# caller instead of learned as a marginal over whichever frames a judge
# happened to label (label presence is explained away; vision features
# are never asked to predict judged-ness). Aux-less runs feed [ACT] on
# every sample.
SUFFIX_FORMAT = 3
OPENER_SUFFIX_FORMAT = 2
GENERATION_OPENER = "<start_of_turn>model\n"
# Mode token offsets: backbone id = (block_base − NUM_MODES) + offset —
# directly below the FAST block, inside the same reserved-unused tail,
# embedded through the decoder's trainable mode tables.
ACT_MODE = 0
AUX_MODE = 1
NUM_MODES = 2
# Free-phase token budget at decode: worst-case configured template
# (headers ~14 + subgoal 16 + holding 4 + progress 5 + event 16) with
# slack; the fallback (force BOA, count it) fires past this.
MAX_FREE_TOKENS = 72
FIELD_TERMINATOR = "\n"
SUBGOAL_HEADER = "subgoal: "
HOLDING_HEADER = "holding: "
PROGRESS_HEADER = "progress: "
EVENT_HEADER = "event: "
HOLDING_VALUES = ("no", "yes")  # indexed by the 0/1 label


class AuxField(StrEnum):
    """Aux fields in their fixed template order."""

    SUBGOAL = "subgoal"
    HOLDING = "holding"
    PROGRESS = "progress"
    EVENT = "event"


class AuxDecodeMode(StrEnum):
    """Inference-time mode selection for the format-3 decode: which mode
    token is fed after the opener. ACT — feed [ACT][BOA], straight to
    grammar-constrained actions (the deployment fast path). FREE — feed
    [AUX], free-until-BOA text generation first (requires an
    aux-trained checkpoint: [AUX] is untrained otherwise)."""

    ACT = "act"
    FREE = "free"


class TextTokenizer(Protocol):
    """The slice of a HF tokenizer the aux renderer/decoder uses (tests
    inject a stub; runtime uses AutoTokenizer)."""

    def encode(self, text: str, *, add_special_tokens: bool = ...) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


@dataclass(frozen=True, slots=True)
class AuxDecodeConfig:
    """The aux record a checkpoint carries in its decoder section —
    everything inference needs to elicit the fields the model trained on,
    plus label provenance. ``template_version`` pins the exact header
    bytes (AUX_TEMPLATE_VERSION when written); a version this code does
    not know is a loud error, never a guess."""

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
                f"{AUX_TEMPLATE_VERSION} — forced-scaffold decoding would "
                "elicit a format the model never trained on",
            )
        return cls(
            template_version=version,
            fields=tuple(AuxField(f) for f in data["fields"]),
            prompt_hash=str(data["prompt_hash"]),
            judge_model=str(data["judge_model"]),
        )


@dataclass(frozen=True, slots=True)
class AuxRuntime:
    """Tokenized scaffold for forced-field decoding, built once from the
    checkpoint's text tokenizer (:func:`build_aux_runtime`). Header ids
    are FED (never predicted); value ids are decoded under per-field
    constraints; ``terminator_id`` ends every field's value."""

    config: AuxDecodeConfig
    tokenizer: TextTokenizer
    header_ids: dict[AuxField, tuple[int, ...]]
    value_candidates: dict[AuxField, tuple[tuple[int, ...], ...]]
    terminator_id: int


def build_aux_runtime(
    config: AuxDecodeConfig,
    tokenizer: TextTokenizer,
) -> AuxRuntime:
    """Tokenize the versioned template's scaffold. The field terminator
    must be a single token (true for \\n under the Gemma tokenizer and
    the test stub) — anything else is a loud error, since value decoding
    detects termination by one id."""

    def encode(text: str) -> tuple[int, ...]:
        return tuple(tokenizer.encode(text, add_special_tokens=False))

    terminator = encode(FIELD_TERMINATOR)
    if len(terminator) != 1:
        raise SystemExit(
            f"aux field terminator {FIELD_TERMINATOR!r} tokenizes to "
            f"{len(terminator)} ids — single-token terminator required",
        )
    headers = {
        AuxField.SUBGOAL: encode(SUBGOAL_HEADER),
        AuxField.HOLDING: encode(HOLDING_HEADER),
        AuxField.PROGRESS: encode(PROGRESS_HEADER),
        AuxField.EVENT: encode(EVENT_HEADER),
    }
    candidates: dict[AuxField, tuple[tuple[int, ...], ...]] = {
        # Constrained value set; first-token argmax picks the candidate,
        # its remaining ids are forced.
        AuxField.HOLDING: tuple(encode(value) for value in HOLDING_VALUES),
    }
    return AuxRuntime(
        config=config,
        tokenizer=tokenizer,
        header_ids={f: headers[f] for f in config.fields},
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

    ``dropout``: probability a LABELED sample renders as unlabeled
    (trains [ACT] + BOA with its aux text dropped) — keeps the fast
    path trained under dense annotation. Draws come from a per-process
    generator seeded from ``torch.initial_seed()`` (in a dataloader
    worker: a pure function of --seed, rank and worker id — same seed,
    same dropout pattern; probe collators run a dropout-0 clone so eval
    always sees the true labels)."""

    tokenizer_dir: str
    fields: tuple[AuxField, ...]
    annotated_repos: frozenset[str]
    block_base: int
    dropout: float
    max_subgoal_tokens: int = 16
    max_event_tokens: int = 16
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

    @override
    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_tokenizer": None, "_generator": None}

    def tokenizer(self) -> TextTokenizer:
        if self._tokenizer is None:
            # Built lazily worker-side (the instance is dropped by
            # __getstate__); the tokenizer files are the processor dir's.
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.tokenizer_dir,
            )
        return self._tokenizer

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer().encode(text, add_special_tokens=False)

    def render_field(
        self,
        aux_field: AuxField,
        item: dict[str, Any],
    ) -> list[int] | None:
        """Token ids of one field's ``header value\\n`` string, or None
        when the label does not exist at this frame."""
        match aux_field:
            case AuxField.SUBGOAL:
                row = active_at(
                    float(item["timestamp"]),
                    persistent=item.get("language_persistent") or [],
                    style="subtask",
                )
                if row is None or not row.get("content"):
                    return None
                header = self._encode(SUBGOAL_HEADER)
                body = self._encode(str(row["content"]))
                if len(body) > self.max_subgoal_tokens:
                    print(
                        f"[aux] truncating subgoal ({len(body)} > "
                        f"{self.max_subgoal_tokens} tokens): {row['content']!r}",
                        flush=True,
                    )
                    body = body[: self.max_subgoal_tokens]
                return header + body + self._encode(FIELD_TERMINATOR)
            case AuxField.HOLDING:
                value = item.get("annotation.holding")
                if value is None or not bool(torch.isfinite(value)):
                    return None
                text = HOLDING_HEADER + HOLDING_VALUES[int(value)] + FIELD_TERMINATOR
                return self._encode(text)
            case AuxField.PROGRESS:
                value = item.get("annotation.progress")
                if value is None or not bool(torch.isfinite(value)):
                    return None
                percent = round(float(value) * 100)
                return self._encode(f"{PROGRESS_HEADER}{percent}%{FIELD_TERMINATOR}")
            case AuxField.EVENT:
                row = emitted_at(
                    float(item["timestamp"]),
                    persistent=item.get("language_persistent") or [],
                    events=item.get("language_events") or [],
                    style=EVENT_STYLE,
                )
                if row is None or not row.get("content"):
                    return None
                header = self._encode(EVENT_HEADER)
                body = self._encode(str(row["content"]))
                if len(body) > self.max_event_tokens:
                    print(
                        f"[aux] truncating event ({len(body)} > "
                        f"{self.max_event_tokens} tokens): {row['content']!r}",
                        flush=True,
                    )
                    body = body[: self.max_event_tokens]
                return header + body + self._encode(FIELD_TERMINATOR)

    def render(self, item: dict[str, Any]) -> list[int]:
        """All present fields' token ids, template order. Empty for
        unjudged frames, for datasets whose stamp failed verification,
        and — with probability ``dropout`` — for labeled samples (mode
        dropout: the sample then trains as [ACT])."""
        if item.get("repo_id") not in self.annotated_repos:
            return []
        ids: list[int] = []
        for aux_field in self.fields:
            rendered = self.render_field(aux_field, item)
            if rendered is not None:
                ids.extend(rendered)
        if ids and self.dropout > 0.0:
            if self._generator is None:
                self._generator = torch.Generator().manual_seed(torch.initial_seed())
            if float(torch.rand((), generator=self._generator)) < self.dropout:
                return []
        return ids


def aux_label_text(item: dict[str, Any], fields: tuple[AuxField, ...]) -> str:
    """The template text the LABELS would render for this item — the
    reference column next to generations in eval tables. Pure strings
    (no tokenizer); presence rules identical to :meth:`AuxSpec.render`."""
    parts: list[str] = []
    for aux_field in fields:
        match aux_field:
            case AuxField.SUBGOAL:
                row = active_at(
                    float(item["timestamp"]),
                    persistent=item.get("language_persistent") or [],
                    style="subtask",
                )
                if row is not None and row.get("content"):
                    parts.append(f"{SUBGOAL_HEADER}{row['content']}{FIELD_TERMINATOR}")
            case AuxField.HOLDING:
                value = item.get("annotation.holding")
                if value is not None and bool(torch.isfinite(value)):
                    parts.append(
                        HOLDING_HEADER + HOLDING_VALUES[int(value)] + FIELD_TERMINATOR,
                    )
            case AuxField.PROGRESS:
                value = item.get("annotation.progress")
                if value is not None and bool(torch.isfinite(value)):
                    parts.append(
                        f"{PROGRESS_HEADER}{round(float(value) * 100)}%"
                        f"{FIELD_TERMINATOR}",
                    )
            case AuxField.EVENT:
                row = emitted_at(
                    float(item["timestamp"]),
                    persistent=item.get("language_persistent") or [],
                    events=item.get("language_events") or [],
                    style=EVENT_STYLE,
                )
                if row is not None and row.get("content"):
                    parts.append(f"{EVENT_HEADER}{row['content']}{FIELD_TERMINATOR}")
    return "".join(parts)


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
