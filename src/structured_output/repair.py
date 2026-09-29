"""Shared structured-output repair helper for the typed-decision plane (TD-21).

Every consumer audited in ``artifacts/audits/td-json-consumer-audit-20260924.md``
that still fishes JSON out of free text follows the same three-step shape once
converted: fish deterministically, and only on a miss spend ONE constrained
completion turn back to a real backend. This module is that shape, factored
out so ~30 call sites (see ``handoffs/active/typed-decision-plane.md`` TD-21.1
through TD-21.29) can share one implementation instead of thirty near-copies.

Reference implementation and the designs it refuted (do NOT re-derive them):
``epyc-inference-research:scripts/kernel_rnd/autokernel/loop/actors.py``
``_schema_repair`` / ``_parse_reply`` (commit ``ad2b89ff``, measured
2026-09-24 on a live 27B at :8083, five report shapes recovered in 2-12s
each). Three single-turn designs were tried and refuted there before landing
on the two-stage shape this module implements:

  1. An optional ``abstain`` property inside the extraction schema gets
     filled in BESIDE a full, real answer -- the model answers AND abstains.
  2. An ``anyOf`` branch that lets the model choose "abstain" vs "answer"
     wins the branch even for reports that plainly do answer.
  3. An in-band marker value (e.g. ``"mechanism_id": "abstain"``) inside the
     answer schema over-abstains on reports that merely hedge.

Hence: single-branch extraction schemas ONLY (never a schema with a judgment
branch baked in), and a SEPARATE stage-1 boolean "does the reply EXPLICITLY
decline?" turn -- constrained to ``{explicitly_declines, reason}`` and
nothing else -- only spent when the caller says a decline is a legitimate
outcome for this site (``decline_question`` is given).

This module also closes three residuals the reference left open
(TD-21.30(b)/(d)/(e) in the handoff):

  (b) The reference accepts any object where ``"abstain" in body or
      required-keys-present``, so a partial or hallucinated object that
      happens to carry the required keys short-circuits repair even when a
      required value has the WRONG TYPE. Here, "parsed"/"repaired" both
      require the fished/extracted value to validate against the FULL
      schema (``jsonschema.Draft202012Validator``, matching the drafts
      already used at ``src/graph/helpers.py:1336`` and
      ``src/typed_decisions/schema.py``) -- type errors, not just missing
      keys, are caught.
  (d) The reference's own ``REVIEW_SCHEMA`` leaves ``additionalProperties``
      unset, so a repair turn may legally return extra keys. Here, any
      object-typed schema that does not set ``additionalProperties`` gets it
      defaulted to ``False`` for both the wire schema and the validator, so
      an extraction turn cannot smuggle in fields the caller never asked for.
  (e) The reference's repair body carries no ``model`` field, so a
      multi-model local endpoint would 400 on it. ``http_chat_completer``
      takes an optional ``model`` and includes it on the wire when given.

TD-21.35 (2026-09-24): a ``required`` field in a JSON-schema-to-GBNF grammar
does not mean "the caller wants this" -- it means "the grammar FORCES a
value, real or not". Live, this forced a ``deep_eval`` draft with no stated
tier to repair to a fabricated ``tier=2``. ``require_evidence`` above catches
that AFTER the fact; ``parse_with_repair``'s ``relax_required`` (default
``True``) stops it at the source by sending a WIRE schema with ``required``
stripped at every object level for the extraction turn only -- except a
``const``/single-``enum`` discriminator key, which stays required so a
``oneOf``/``anyOf`` union stays disambiguable. The RESULT is still validated
against the caller's original, unrelaxed schema, so an omitted required
field now fails honestly (``"failed"``, reason names the field) instead of
being invented. See ``_relax_required_for_wire`` for the full mechanism and
why this is orthogonal to the three abstain-signalling designs refuted
above (those are about a judgment channel winning over a real answer; this
never touches branch selection or a judgment property, only whether an
ordinary data field is forced).

Public API: ``fish_json``, ``parse_with_repair``, ``parse_with_repair_async``
(TD-21.26, for an injected coroutine-function completer), ``RepairResult``,
``http_chat_completer``, ``primitives_completer``,
``STRUCTURED_OUTPUT_REPAIR_COUNTS``.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, best_match

from src.prompt_builders.code_utils import _repair_json_text

_log = logging.getLogger(__name__)

Kind = Literal["object", "array", "any"]
Status = Literal["parsed", "repaired", "declined", "failed"]

#: A completion function: given the messages for ONE turn and the JSON schema
#: that constrains it, return the model's raw content string. Must RAISE on a
#: transport failure (network, HTTP status, malformed envelope) -- never
#: return a sentinel -- ``parse_with_repair`` turns any exception into a typed
#: ``"failed"`` result and records the exception's summary as the reason.
CompleteFn = Callable[[Sequence[Mapping[str, Any]], Mapping[str, Any]], str]

#: TD-21.26: the async twin of ``CompleteFn`` for call sites whose injected
#: model callable is itself a coroutine function (e.g. the env_synth
#: species's ``LLMCall = Callable[[str, str], Awaitable[str]]``). Same
#: contract: raise on transport failure, never return a sentinel.
AsyncCompleteFn = Callable[[Sequence[Mapping[str, Any]], Mapping[str, Any]], Awaitable[str]]

SCHEMA_REPAIR_TIMEOUT_S = 300
SCHEMA_REPAIR_TAIL_CHARS = 16000
DEFAULT_MAX_TOKENS = 2048

_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_OPENERS = {"{": "}", "[": "]"}
_CLOSERS_TO_OPENER = {"}": "{", "]": "["}


# --------------------------------------------------------------------------- telemetry

#: Keyed by (site, status); mirrors the style of
#: ``src.prompt_builders.code_utils.TOOL_CALL_JSON_REPAIR_COUNTS`` but keeps
#: the site dimension so ~30 call sites share one counter without colliding.
STRUCTURED_OUTPUT_REPAIR_COUNTS: dict[tuple[str, str], int] = {}
_counts_lock = threading.Lock()


def _record(site: str, status: str, *, repair_calls: int = 0, reason: str = "") -> None:
    key = (site, status)
    with _counts_lock:
        STRUCTURED_OUTPUT_REPAIR_COUNTS[key] = STRUCTURED_OUTPUT_REPAIR_COUNTS.get(key, 0) + 1
    if status == "parsed":
        _log.debug("structured_output_repair site=%s status=%s", site, status)
    else:
        _log.info(
            "structured_output_repair site=%s status=%s calls=%d reason=%r",
            site, status, repair_calls, reason[:200],
        )


def reset_counts_for_tests() -> None:
    """Test-only: clear the module-level counter between test cases."""
    with _counts_lock:
        STRUCTURED_OUTPUT_REPAIR_COUNTS.clear()


# --------------------------------------------------------------------------- fishing


def _kind_matches(value: Any, kind: Kind) -> bool:
    if kind == "object":
        return isinstance(value, dict)
    if kind == "array":
        return isinstance(value, list)
    return isinstance(value, (dict, list))


def _try_parse(text: str) -> Any:
    """``json.loads`` gated by the existing deterministic repair idiom."""
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    fixed = _repair_json_text(text)
    if fixed is None:
        return None
    try:
        return json.loads(fixed)
    except (json.JSONDecodeError, ValueError):
        return None


def _fish_from_fences(text: str, kind: Kind) -> Any:
    """The LAST fenced ```json block that parses to a value of `kind`."""
    for block in reversed(_FENCE_RE.findall(text)):
        value = _try_parse(block)
        if value is not None and _kind_matches(value, kind):
            return value
    return None


def _fish_balanced(text: str, kind: Kind) -> Any:
    """The LAST top-level balanced object/array of `kind`, string-aware.

    Single left-to-right pass. A quote toggles string-mode so a brace or
    bracket character inside a JSON string never perturbs the depth count
    (the reference's ``_extract_json`` does not track strings at all). Only
    TOP-LEVEL spans -- where the bracket stack is empty both before the open
    and after the matching close -- are tried as candidates; nested
    structures ride along inside their parent's span. A closer that does not
    match the type of bracket currently open abandons that candidate (the
    stack resets) rather than raising or mis-tracking depth.
    """
    wanted_opener = {"object": "{", "array": "["}.get(kind)  # None for "any" == either
    best = None
    stack: list[str] = []
    start: int | None = None
    in_str = False
    escape = False
    for index, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch in _OPENERS:
            if not stack:
                start = index
            stack.append(ch)
            continue
        if ch in _CLOSERS_TO_OPENER:
            if stack and stack[-1] == _CLOSERS_TO_OPENER[ch]:
                stack.pop()
                if not stack and start is not None:
                    span = text[start : index + 1]
                    if wanted_opener is None or span[0] == wanted_opener:
                        value = _try_parse(span)
                        if value is not None and _kind_matches(value, kind):
                            best = value
                    start = None
            else:
                stack = []
                start = None
    return best


def fish_json(text: str, *, kind: Kind = "object") -> Any:
    """Deterministic extraction of a JSON value from free text. Never raises.

    Tries, in order:

      1. The LAST fenced ```json (or bare ```) block whose content parses
         (tolerant of trailing commas / stray closers via the existing
         ``_repair_json_text`` idiom) to a value matching `kind`.
      2. The LAST top-level balanced object/array of `kind` found anywhere
         in the text, using string-aware brace matching.

    `kind="object"` / `"array"` require exactly that JSON type at the top
    level; `kind="any"` accepts either. Returns the parsed value, or `None`
    if nothing of the requested kind can be extracted -- never raises on
    malformed or empty input.
    """
    if not text:
        return None
    fenced = _fish_from_fences(text, kind)
    if fenced is not None:
        return fenced
    return _fish_balanced(text, kind)


# --------------------------------------------------------------------------- schema closure + validation


def _closed_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """TD-21.30(d): default ``additionalProperties: False`` on an object
    schema that does not set it, so neither the wire request nor the
    validator invites the model to add fields the caller never asked for.
    Only the top level is closed -- a caller that deliberately wants an open
    nested object keeps that by setting ``additionalProperties`` there
    explicitly; this module does not recurse into nested schemas."""
    closed = dict(schema)
    if closed.get("type") == "object" and "additionalProperties" not in closed:
        closed["additionalProperties"] = False
    return closed


def _build_validator(schema: Mapping[str, Any], *, site: str) -> Draft202012Validator | None:
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        _log.error("structured_output_repair site=%s invalid_schema=%r", site, exc)
        return None
    return Draft202012Validator(schema)


def _validation_failure_reason(validator: Draft202012Validator, value: Any) -> str:
    """TD-21.35: name what actually failed, instead of the flat generic
    string this used to be. Most relevant with `relax_required=True`: a
    missing required field no longer gets invented by the grammar, so THIS
    is now the path that turns that honest omission into a typed `"failed"`
    -- `jsonschema`'s own "'count' is a required property" (or a type-error
    message for any other validation failure) tells the caller exactly what
    to look at instead of a bare "failed schema validation"."""
    error = best_match(validator.iter_errors(value))
    if error is None:  # defensive -- `is_valid` already returned False
        return "extraction result failed schema validation"
    return f"extraction result failed schema validation: {error.message}"


# --------------------------------------------------------------------------- TD-21.35: wire-schema relaxation
#
# `required` in a JSON-schema-to-GBNF grammar does not mean "the caller
# wants this field" -- it means "the grammar FORCES the model to emit a
# value for this key, real or not" (llama.cpp
# common/json-schema-to-grammar.cpp:873-880 builds an empty `required` set
# when the schema has none, and `_build_object_rule` at :640-731 buckets
# every property NOT in that set as optional; :659-663 is the bucketing
# itself). Live, 2026-09-24: a `deep_eval` draft that named no tier forced a
# `tier` grammar slot, so the model filled `tier=2` whole cloth -- a
# schema-VALID, fully invented value `require_evidence` then has to catch
# after the fact. This relaxation stops the invention at the source for the
# REPAIR/EXTRACTION turn only: the WIRE schema sent as `response_format`
# drops `required` at every object level so the grammar can no longer force
# an unstated field, while `parse_with_repair` still validates the result
# against the ORIGINAL (unrelaxed) schema -- a field the model still omits
# now fails validation honestly (`"failed"`, reason names the field) instead
# of being invented to satisfy the grammar.
#
# EXCEPTION: a key whose own property schema is a `const` (or a
# single-value `enum`, which is equivalent) is a DISCRIMINATOR, not caller
# data -- e.g. an autopilot action union's `type: {"const": "deep_eval"}`
# (`scripts/autopilot/controller_io.py:_action_type_json_schema`). Relaxing
# a discriminator's `required` would let the grammar accept an object that
# satisfies no branch unambiguously (or none at all when every branch's
# `type` is optional), so any key detected as `const`/single-`enum`-pinned
# stays in `required` on the wire -- relaxation only ever removes a forced
# slot for a value the model could otherwise invent, never a slot the
# caller already pinned to one legal value.
#
# This is orthogonal to the three abstain-signalling designs refuted in
# `epyc-inference-research:scripts/kernel_rnd/autokernel/loop/actors.py`
# (optional `abstain` property filled in BESIDE a real answer; an `anyOf`
# judgment branch winning over a real answer; an in-band marker
# over-abstaining) -- those are about a SEPARATE judgment channel (does the
# model decline?) riding inside or beside the answer schema and winning when
# it should not. This relaxation touches neither `oneOf`/`anyOf` branch
# selection nor any judgment property; it only changes whether an ordinary
# DATA field inside whichever branch/object already applies is forced to
# carry a value. A discriminator branch is still selected exactly as before
# (its `type` const is still required), and nothing here adds, removes, or
# reweights a branch.


def _is_wire_discriminator(prop_schema: Any) -> bool:
    """True for a property schema that pins the property to exactly one
    legal value -- a `const`, or a single-element `enum` (the model has no
    choice either way, so keeping it `required` on the wire forces nothing
    the caller didn't already pin). Used to decide which `required` entries
    survive relaxation; see the TD-21.35 block above."""
    if not isinstance(prop_schema, Mapping):
        return False
    if "const" in prop_schema:
        return True
    enum = prop_schema.get("enum")
    return isinstance(enum, list) and len(enum) == 1


def _relax_required_for_wire(schema: Any) -> Any:
    """TD-21.35: return a copy of `schema` with `required` dropped at every
    object level, recursively through `properties`, `oneOf`/`anyOf`/`allOf`
    branches, `items` (list or single-schema form), and `$defs`/
    `definitions` -- EXCEPT a key detected as a discriminator by
    `_is_wire_discriminator`, which stays required so a `oneOf`/`anyOf`
    union stays disambiguable. Everything else (`type`, `enum`, `const`,
    numeric bounds, `additionalProperties`, property ORDER) passes through
    unchanged -- this only ever touches `required`. Never mutates its input;
    always returns a new structure. This is the schema sent to `complete()`
    for the extraction/repair turn ONLY -- `parse_with_repair` validates the
    result against the original, unrelaxed schema regardless."""
    if isinstance(schema, list):
        return [_relax_required_for_wire(item) for item in schema]
    if not isinstance(schema, Mapping):
        return schema

    relaxed: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "required":
            continue  # rebuilt below, once `properties` is known
        if key in ("properties", "$defs", "definitions") and isinstance(value, Mapping):
            relaxed[key] = {k: _relax_required_for_wire(v) for k, v in value.items()}
        elif key in ("oneOf", "anyOf", "allOf") and isinstance(value, list):
            relaxed[key] = [_relax_required_for_wire(v) for v in value]
        elif key == "items":
            relaxed[key] = _relax_required_for_wire(value)
        elif key == "additionalProperties" and isinstance(value, Mapping):
            relaxed[key] = _relax_required_for_wire(value)
        else:
            relaxed[key] = value

    required = schema.get("required")
    properties = schema.get("properties")
    if isinstance(required, list) and isinstance(properties, Mapping):
        kept = [key for key in required if _is_wire_discriminator(properties.get(key))]
        if kept:
            relaxed["required"] = kept
    # A `required` list with no sibling `properties` map has nothing to check
    # a discriminator against -- dropped entirely (nothing here could pin a
    # value the grammar can point at).

    return relaxed


# --------------------------------------------------------------------------- evidence check
#
# `require_evidence` (parse_with_repair): a schema-VALID repaired value can
# still be fabricated -- a required field forces a fill even when the raw
# reply never states one. This walks the repaired value against the schema
# in parallel (resolving which oneOf/anyOf branch matched so per-property
# `const`s are visible) and flags any number/short-string leaf that does not
# literally appear in the raw reply.

_MAX_EVIDENCE_LEAF_CHARS = 40


def _normalize_for_evidence(text: str) -> str:
    """Case-insensitive, whitespace-normalised form used on BOTH sides of an
    evidence comparison."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _select_evidence_branch(schema: Mapping[str, Any], value: Any) -> Mapping[str, Any]:
    """Resolve which `oneOf`/`anyOf` alternative `value` validated against,
    so `_evidence_failures` can see that branch's per-property `const`s. The
    caller already proved `value` is valid against the WHOLE schema before
    calling this; if (defensively) no single alternative matches on its own,
    the original schema is returned and every leaf is evidence-checked --
    fails safe toward MORE scrutiny, never less."""
    branches = schema.get("oneOf") or schema.get("anyOf")
    if not isinstance(branches, list):
        return schema
    for branch in branches:
        if not isinstance(branch, Mapping):
            continue
        try:
            if Draft202012Validator(branch).is_valid(value):
                return branch
        except SchemaError:
            continue
    return schema


#: Boundaries for a "standalone" number token. A plain `\w` boundary alone
#: would wrongly reject a number immediately followed by a sentence-ending
#: period ("tier 1." -- "1" IS standalone there), so only a digit run
#: separated by a `.` from more digits counts as "embedded" (blocks "10"
#: matching inside "10.5" or "2" inside "2.5"); a bare trailing/leading `.`
#: with no adjoining digit is a normal sentence boundary, not a decimal.
_NUM_LEFT_BOUNDARY = r"(?<!\w)(?<!\d\.)"
_NUM_RIGHT_BOUNDARY = r"(?!\w)(?!\.\d)"


def _number_has_evidence(value: float, normalized_raw: str) -> bool:
    """`value` must appear as a standalone token (not embedded in a longer
    number or identifier, and not merely the integer/fraction half of a
    different decimal number) in the normalised raw reply."""
    candidates = {str(value)}
    if isinstance(value, float) and value.is_integer():
        candidates.add(str(int(value)))
    elif isinstance(value, int):
        candidates.add(str(float(value)))
    alternation = "|".join(re.escape(c) for c in candidates)
    pattern = f"{_NUM_LEFT_BOUNDARY}(?:{alternation}){_NUM_RIGHT_BOUNDARY}"
    return re.search(pattern, normalized_raw) is not None


def _string_has_evidence(value: str, normalized_raw: str) -> bool:
    needle = _normalize_for_evidence(value)
    return (not needle) or (needle in normalized_raw)


def _evidence_failures(
    value: Any,
    schema: Any,
    raw: str,
    *,
    path: str = "",
    evidence_exempt: Collection[str] = (),
) -> list[str]:
    """Return the dotted/bracketed paths of every leaf in `value` that is a
    number or a string of at most `_MAX_EVIDENCE_LEAF_CHARS` characters and
    does NOT appear in `raw` -- empty when every such leaf is evidenced (or
    exempt: `const`-pinned, boolean, `None`, a longer string, or named in
    `evidence_exempt` -- see `parse_with_repair`'s docstring)."""
    normalized_raw = _normalize_for_evidence(raw)
    return _evidence_failures_normalized(
        value, schema, normalized_raw, path=path, evidence_exempt=evidence_exempt
    )


def _evidence_failures_normalized(
    value: Any,
    schema: Any,
    normalized_raw: str,
    *,
    path: str,
    evidence_exempt: Collection[str] = (),
) -> list[str]:
    if isinstance(schema, Mapping) and ("oneOf" in schema or "anyOf" in schema):
        schema = _select_evidence_branch(schema, value)

    if isinstance(value, dict):
        properties = schema.get("properties") if isinstance(schema, Mapping) else None
        properties = properties if isinstance(properties, Mapping) else {}
        failures: list[str] = []
        for key, sub_value in value.items():
            sub_path = f"{path}.{key}" if path else str(key)
            if key in evidence_exempt or sub_path in evidence_exempt:
                continue
            sub_schema = properties.get(key) if isinstance(properties.get(key), Mapping) else {}
            failures.extend(
                _evidence_failures_normalized(
                    sub_value, sub_schema, normalized_raw, path=sub_path,
                    evidence_exempt=evidence_exempt,
                )
            )
        return failures

    if isinstance(value, list):
        items_schema = schema.get("items") if isinstance(schema, Mapping) else None
        items_schema = items_schema if isinstance(items_schema, Mapping) else {}
        failures = []
        for index, item in enumerate(value):
            failures.extend(
                _evidence_failures_normalized(
                    item, items_schema, normalized_raw, path=f"{path}[{index}]",
                    evidence_exempt=evidence_exempt,
                )
            )
        return failures

    # Scalar leaf.
    if isinstance(schema, Mapping) and "const" in schema:
        return []  # caller-pinned; not something the model could invent
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [] if _number_has_evidence(value, normalized_raw) else [path or "<root>"]
    if isinstance(value, str):
        if len(value) > _MAX_EVIDENCE_LEAF_CHARS:
            return []  # a longer string may be a legitimate paraphrase
        return [] if _string_has_evidence(value, normalized_raw) else [path or "<root>"]
    return []


# --------------------------------------------------------------------------- result


@dataclass(frozen=True)
class RepairResult:
    """Outcome of one ``parse_with_repair`` call.

    ``value`` is the typed, schema-valid payload on "parsed"/"repaired", or
    `None` on "declined"/"failed" -- this module NEVER fabricates or
    defaults a value; a typed failure is always the caller's to interpret.
    ``reason`` carries the decline's stated reason, or the failure detail;
    empty on "parsed". ``repair_calls`` is 0 (fished clean), 1 (one decline
    probe, or one extraction turn) or 2 (both).
    """

    value: Any
    status: Status
    reason: str
    site: str
    repair_calls: int


# --------------------------------------------------------------------------- repair turns

_DECLINE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "explicitly_declines": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["explicitly_declines", "reason"],
    "additionalProperties": False,
}
_DECLINE_VALIDATOR = Draft202012Validator(_DECLINE_SCHEMA)

_DECLINE_INSTRUCTION_TEMPLATE = (
    "You classify one reply, given as the user message, against exactly one "
    "question. {question} Set explicitly_declines accordingly and quote the "
    "reply's own stated reason in `reason`, or an empty string if it gives "
    "none. Never answer the reply's own question yourself -- only classify "
    "whether it explicitly declines."
)

_DEFAULT_EXTRACT_INSTRUCTION = (
    "Convert the reply given as the user message into exactly one JSON "
    "value matching the given schema. Copy the reply's own wording and "
    "values faithfully -- do not invent, judge, or improve anything that is "
    "not already present in the reply."
)


def _decline_probe(
    raw: str, *, question: str, complete: CompleteFn, site: str, tail_chars: int
) -> tuple[bool | None, str]:
    """ONE constrained boolean turn. Returns (declined, reason);
    `declined=None` means the probe itself was inconclusive (transport
    failure, unparseable reply, or an off-schema value) and the caller
    should fall through to the extraction turn rather than treat it as a
    decline."""
    messages = [
        {"role": "system", "content": _DECLINE_INSTRUCTION_TEMPLATE.format(question=question)},
        {"role": "user", "content": raw[-tail_chars:]},
    ]
    try:
        content = complete(messages, _DECLINE_SCHEMA)
    except Exception as exc:  # noqa: BLE001 - transport is caller-defined
        _log.info("structured_output_repair site=%s stage=decline transport_error=%r", site, exc)
        return None, ""
    value = fish_json(content, kind="object")
    if value is None or not _DECLINE_VALIDATOR.is_valid(value):
        return None, ""
    declined = value.get("explicitly_declines")
    reason = str(value.get("reason") or "")
    if declined is True:
        return True, reason
    if declined is False:
        return False, reason
    return None, ""


def _extraction_turn(
    raw: str,
    *,
    schema: Mapping[str, Any],
    complete: CompleteFn,
    instruction: str | None,
    site: str,
    tail_chars: int,
    relax_required: bool = True,
) -> tuple[Any, str | None]:
    """ONE constrained extraction turn. Returns (value_or_None, error_or_None).

    TD-21.35: the schema sent on the wire (as `response_format`, so it
    becomes the GBNF grammar) is `_relax_required_for_wire(schema)` unless
    `relax_required=False` -- `schema` itself (what the RESULT still gets
    validated against, by the caller) is never mutated."""
    wire_schema = _relax_required_for_wire(schema) if relax_required else schema
    messages = [
        {"role": "system", "content": instruction or _DEFAULT_EXTRACT_INSTRUCTION},
        {"role": "user", "content": raw[-tail_chars:]},
    ]
    try:
        content = complete(messages, wire_schema)
    except Exception as exc:  # noqa: BLE001 - transport is caller-defined
        msg = f"transport_error: {type(exc).__name__}: {exc}"
        _log.info("structured_output_repair site=%s stage=extraction %s", site, msg)
        return None, msg
    value = fish_json(content, kind="any")
    if value is None:
        return None, "extraction reply carried no parseable JSON"
    return value, None


def parse_with_repair(
    raw: str,
    *,
    schema: Mapping[str, Any],
    complete: CompleteFn,
    decline_question: str | None = None,
    instruction: str | None = None,
    site: str,
    kind: Kind | None = None,
    tail_chars: int = SCHEMA_REPAIR_TAIL_CHARS,
    require_evidence: bool = False,
    evidence_exempt: Collection[str] = (),
    relax_required: bool = True,
) -> RepairResult:
    """Fish first; repair with at most two constrained turns on a miss.

    1. ``fish_json(raw)`` (kind inferred from ``schema["type"]`` unless
       ``kind`` overrides it). If the fished value validates against the
       FULL schema (Draft 2020-12, TD-21.30(b): type errors on required
       fields are caught, not just their presence) -> ``"parsed"``, 0 calls.
    2. Else, if ``decline_question`` is given: ONE constrained boolean turn.
       An explicit decline -> ``"declined"``, value `None`, the stated
       reason carried, extraction is NEVER attempted (a decline is not a
       parse failure to recover from).
    3. Else: ONE constrained extraction turn, temperature-0 by convention of
       the injected `complete`, asking the model to copy -- never invent or
       judge -- the reply's own content into the schema. Valid -> `"repaired"`,
       UNLESS `require_evidence` rejects it (see below).
    4. Otherwise -> `"failed"` with a reason; value is always `None`.

    `complete` is caller-injected (see `http_chat_completer` /
    `primitives_completer`) and must raise on transport failure; this
    function never lets that exception escape to its own caller.

    `require_evidence` (default `False`, so every already-landed conversion
    is byte-identical): a schema-VALID repaired value can still be a
    fabrication -- a grammar that forces a required field forces the model
    to fill it even when the raw reply never states one (observed live,
    2026-09-24: a `deep_eval` draft with no stated tier repaired to
    `tier=2`, invented whole). When `True`, a REPAIRED value (never a
    cleanly FISHED one -- that came out of `raw` by construction) must pass
    `_evidence_failures`: every scalar leaf that is a number, or a string of
    at most `_MAX_EVIDENCE_LEAF_CHARS` characters, must appear in `raw`
    (case-insensitive, whitespace-normalised; a number as a standalone
    token), except a leaf whose schema position pins it via `const` (e.g. an
    action-type discriminator -- the caller already chose it, it is not
    something the model could invent) and booleans (a yes/no judgement is
    not evidence-checkable the same way a copied fact is). Longer strings
    are exempt -- a faithful paraphrase is legitimate re-expression, not
    invention. A failure -> `"failed"`, reason names the unevidenced
    field(s), never a fabricated value.

    `evidence_exempt` (default empty, only meaningful with `require_evidence
    =True`): property names -- a bare key (matched at ANY nesting depth,
    e.g. `"mode"`) or a dotted/bracketed path (matched exactly, e.g.
    `"verifier.type"`) -- whose ENTIRE subtree is exempt from the evidence
    check regardless of type or length. For a schema that mixes literal
    facts the model must copy (numbers, ids, short extracted strings) with
    CLASSIFICATIONS the model legitimately maps prose onto (an `enum` the
    reply rarely spells verbatim -- a review `decision`, a `delegate_to`
    role name, a plan step's `actor`) turning on `require_evidence` for the
    whole schema would spuriously fail the classification leaves (`enum`,
    unlike `const`, is NOT auto-exempt: the model chose that value, it did
    not copy it). Name those keys here instead of leaving the whole site
    off. Small and call-site-specific by design -- see the TD-21.34 call
    sites in `chat_delegation_decision.py` / `proactive_stage.py` /
    `review_service.py` for the mixed-schema idiom in practice.

    `relax_required` (default `True`, TD-21.35): the schema sent on the wire
    for the EXTRACTION turn only (never the decline probe, never what the
    result is validated against) has `required` dropped at every object
    level, so the grammar can no longer FORCE a value for a field the raw
    reply never mentioned -- the root cause the `require_evidence` backstop
    above was catching after the fact (see `_relax_required_for_wire`'s
    docstring for the full mechanism and the `const`-discriminator
    exception). Validation is still against the ORIGINAL, unrelaxed schema,
    so a field the model still omits now fails validation honestly
    (`"failed"`, naming the field) rather than being invented -- the
    contract callers see (`"parsed"`/`"repaired"`/`"declined"`/`"failed"`,
    `value` typed or `None`) is unchanged; only what the grammar can force
    during the extraction turn changes. Set `False` only for a site whose
    instruction legitimately asks the model to DERIVE a required field
    (rather than copy one already present in `raw`) and that depends on the
    grammar forcing it to actually try -- as of TD-21.35 no audited call
    site does this (every schema-required field across the ~30 sites is
    copied/classified from the reply, never derived from nothing); this
    parameter exists so a future site with that shape has an explicit,
    documented way to opt out instead of silently losing the force.
    """
    working_schema = _closed_schema(schema)
    validator = _build_validator(working_schema, site=site)
    if validator is None:
        result = RepairResult(None, "failed", "schema itself is not a valid JSON Schema", site, 0)
        _record(site, result.status, reason=result.reason)
        return result

    effective_kind: Kind = kind or ("array" if working_schema.get("type") == "array" else "object")

    fished = fish_json(raw, kind=effective_kind)
    if fished is not None and validator.is_valid(fished):
        result = RepairResult(fished, "parsed", "", site, 0)
        _record(site, result.status)
        return result

    calls = 0
    if decline_question:
        calls += 1
        declined, decline_reason = _decline_probe(
            raw, question=decline_question, complete=complete, site=site, tail_chars=tail_chars
        )
        if declined is True:
            result = RepairResult(None, "declined", decline_reason, site, calls)
            _record(site, result.status, repair_calls=calls, reason=decline_reason)
            return result

    calls += 1
    extracted, err = _extraction_turn(
        raw, schema=working_schema, complete=complete, instruction=instruction,
        site=site, tail_chars=tail_chars, relax_required=relax_required,
    )
    if extracted is not None and validator.is_valid(extracted):
        if require_evidence:
            unevidenced = _evidence_failures(
                extracted, working_schema, raw, evidence_exempt=evidence_exempt
            )
            if unevidenced:
                reason = (
                    "repaired value has unevidenced field(s) not present in the "
                    f"raw reply: {', '.join(unevidenced)}"
                )
                result = RepairResult(None, "failed", reason, site, calls)
                _record(site, result.status, repair_calls=calls, reason=reason)
                return result
        result = RepairResult(extracted, "repaired", "", site, calls)
        _record(site, result.status, repair_calls=calls)
        return result

    if extracted is not None:
        reason = _validation_failure_reason(validator, extracted)
    else:
        reason = err or "extraction result failed schema validation"
    result = RepairResult(None, "failed", reason, site, calls)
    _record(site, result.status, repair_calls=calls, reason=reason)
    return result


# --------------------------------------------------------------------------- async repair turns (TD-21.26)


async def _decline_probe_async(
    raw: str, *, question: str, complete: AsyncCompleteFn, site: str, tail_chars: int
) -> tuple[bool | None, str]:
    """Async twin of ``_decline_probe`` -- see its docstring for semantics."""
    messages = [
        {"role": "system", "content": _DECLINE_INSTRUCTION_TEMPLATE.format(question=question)},
        {"role": "user", "content": raw[-tail_chars:]},
    ]
    try:
        content = await complete(messages, _DECLINE_SCHEMA)
    except Exception as exc:  # noqa: BLE001 - transport is caller-defined
        _log.info("structured_output_repair site=%s stage=decline transport_error=%r", site, exc)
        return None, ""
    value = fish_json(content, kind="object")
    if value is None or not _DECLINE_VALIDATOR.is_valid(value):
        return None, ""
    declined = value.get("explicitly_declines")
    reason = str(value.get("reason") or "")
    if declined is True:
        return True, reason
    if declined is False:
        return False, reason
    return None, ""


async def _extraction_turn_async(
    raw: str,
    *,
    schema: Mapping[str, Any],
    complete: AsyncCompleteFn,
    instruction: str | None,
    site: str,
    tail_chars: int,
    relax_required: bool = True,
) -> tuple[Any, str | None]:
    """Async twin of ``_extraction_turn`` -- see its docstring for semantics,
    including the TD-21.35 wire-schema relaxation."""
    wire_schema = _relax_required_for_wire(schema) if relax_required else schema
    messages = [
        {"role": "system", "content": instruction or _DEFAULT_EXTRACT_INSTRUCTION},
        {"role": "user", "content": raw[-tail_chars:]},
    ]
    try:
        content = await complete(messages, wire_schema)
    except Exception as exc:  # noqa: BLE001 - transport is caller-defined
        msg = f"transport_error: {type(exc).__name__}: {exc}"
        _log.info("structured_output_repair site=%s stage=extraction %s", site, msg)
        return None, msg
    value = fish_json(content, kind="any")
    if value is None:
        return None, "extraction reply carried no parseable JSON"
    return value, None


async def parse_with_repair_async(
    raw: str,
    *,
    schema: Mapping[str, Any],
    complete: AsyncCompleteFn,
    decline_question: str | None = None,
    instruction: str | None = None,
    site: str,
    kind: Kind | None = None,
    tail_chars: int = SCHEMA_REPAIR_TAIL_CHARS,
    require_evidence: bool = False,
    evidence_exempt: Collection[str] = (),
    relax_required: bool = True,
) -> RepairResult:
    """Async twin of :func:`parse_with_repair` (TD-21.26) for call sites whose
    injected model callable is itself a coroutine function -- e.g. the
    env_synth species's ``LLMCall = Callable[[str, str], Awaitable[str]]``.
    Identical fish-then-repair semantics and identical telemetry
    (``STRUCTURED_OUTPUT_REPAIR_COUNTS``); see :func:`parse_with_repair` for
    the full contract, INCLUDING ``require_evidence``/``evidence_exempt``
    (TD-21.34: this twin originally lacked both -- every async call site
    that needs the evidence guard was unable to get it) and ``relax_required``
    (TD-21.35: same wire-schema relaxation for the extraction turn, same
    default ``True``, same `const`-discriminator exception). Kept as a
    literal parallel implementation rather than a wrapper so neither path
    pays an event-loop indirection for the other.
    """
    working_schema = _closed_schema(schema)
    validator = _build_validator(working_schema, site=site)
    if validator is None:
        result = RepairResult(None, "failed", "schema itself is not a valid JSON Schema", site, 0)
        _record(site, result.status, reason=result.reason)
        return result

    effective_kind: Kind = kind or ("array" if working_schema.get("type") == "array" else "object")

    fished = fish_json(raw, kind=effective_kind)
    if fished is not None and validator.is_valid(fished):
        result = RepairResult(fished, "parsed", "", site, 0)
        _record(site, result.status)
        return result

    calls = 0
    if decline_question:
        calls += 1
        declined, decline_reason = await _decline_probe_async(
            raw, question=decline_question, complete=complete, site=site, tail_chars=tail_chars
        )
        if declined is True:
            result = RepairResult(None, "declined", decline_reason, site, calls)
            _record(site, result.status, repair_calls=calls, reason=decline_reason)
            return result

    calls += 1
    extracted, err = await _extraction_turn_async(
        raw, schema=working_schema, complete=complete, instruction=instruction,
        site=site, tail_chars=tail_chars, relax_required=relax_required,
    )
    if extracted is not None and validator.is_valid(extracted):
        if require_evidence:
            unevidenced = _evidence_failures(
                extracted, working_schema, raw, evidence_exempt=evidence_exempt
            )
            if unevidenced:
                reason = (
                    "repaired value has unevidenced field(s) not present in the "
                    f"raw reply: {', '.join(unevidenced)}"
                )
                result = RepairResult(None, "failed", reason, site, calls)
                _record(site, result.status, repair_calls=calls, reason=reason)
                return result
        result = RepairResult(extracted, "repaired", "", site, calls)
        _record(site, result.status, repair_calls=calls)
        return result

    if extracted is not None:
        reason = _validation_failure_reason(validator, extracted)
    else:
        reason = err or "extraction result failed schema validation"
    result = RepairResult(None, "failed", reason, site, calls)
    _record(site, result.status, repair_calls=calls, reason=reason)
    return result


# --------------------------------------------------------------------------- complete() adapters


def http_chat_completer(
    base_url: str,
    *,
    model: str | None = None,
    timeout_s: int = SCHEMA_REPAIR_TIMEOUT_S,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> CompleteFn:
    """A `CompleteFn` that POSTs one turn to `{base_url}/chat/completions`.

    Mirrors the AutoKernel `_schema_repair` wire shape exactly: `messages`,
    `response_format={type: json_schema, json_schema: {name, schema}}`,
    `temperature: 0`, `chat_template_kwargs: {enable_thinking: False}`. Adds
    `model` to the body when given (TD-21.30(e): the reference's repair body
    carries no `model` field, so a multi-model local endpoint would 400 on
    it and silently degrade to `None`). Uses stdlib `urllib` only, matching
    the reference. Raises `urllib.error.URLError` / `OSError` / `KeyError` /
    `IndexError` / `TypeError` / `ValueError` on any transport, HTTP or
    envelope failure -- `parse_with_repair` is the layer that turns that
    into a typed `"failed"` result.
    """
    url = base_url.rstrip("/") + "/chat/completions"

    def complete(messages: Sequence[Mapping[str, Any]], schema: Mapping[str, Any]) -> str:
        body: dict[str, Any] = {
            "messages": [dict(m) for m in messages],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "structured_output_repair", "schema": dict(schema)},
            },
            "temperature": 0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if model:
            body["model"] = model
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
        return payload["choices"][0]["message"]["content"]

    return complete


def primitives_completer(
    primitives: Any,
    role: str,
    *,
    temperature: float = 0.0,
    n_tokens: int | None = None,
) -> CompleteFn:
    """A `CompleteFn` over `LLMPrimitives.llm_call`.

    `llm_call(prompt, ..., role=, json_schema=, temperature=, ...)` takes a
    prompt STRING, not a chat `messages` list (unlike `chat_completion_call`,
    which takes messages but has no `json_schema` parameter at all). The
    `messages` this helper builds (one system + one user turn) are therefore
    rendered into a single prompt, in role order, faithfully -- this is a
    deliberate compromise, not a hidden loss: `skip_suffix=True` is passed so
    the role's registry `system_prompt_suffix` does not get appended after
    the rendered turns.

    `temperature` IS forwarded explicitly (default 0, matching the reference
    idiom). `llm_call` has NO `enable_thinking` parameter: thinking is
    suppressed only when the resolved role's registry entry already sets
    `chat_template_kwargs.enable_thinking=false`
    (`src/backends/llama_server.py:634`, `feedback_qwen3x_enable_thinking_false`)
    -- this adapter cannot force it off for a role whose registry entry does
    not already declare it, so callers repairing against a role without that
    entry should expect (and may need to fence out) a thinking preamble in
    `content`.

    RI-23: with ``thinking_roles_chat_lane`` on, the repair turn runs under
    ``chat_completions_roles.thinking_off()`` — a per-call
    ``enable_thinking: false`` override on the chat lane, which now also carries the
    thinking-on roles. A repair turn is a schema-shaped emission, not a reasoning
    budget. Flag off: unchanged (the override is a no-op).

    Since TD-21.0 (orch `b284ede3`, 2026-09-24) `json_schema` reaches the wire on
    BOTH lanes: as `json_schema` on `/completion` and as an OpenAI
    `response_format` on `/v1/chat/completions`.
    """

    def complete(messages: Sequence[Mapping[str, Any]], schema: Mapping[str, Any]) -> str:
        from src.chat_completions_roles import thinking_off

        prompt = "\n\n".join(
            f"[{m.get('role', 'user')}]\n{m.get('content', '')}" for m in messages
        )
        with thinking_off():
            return primitives.llm_call(
                prompt,
                role=role,
                json_schema=dict(schema),
                temperature=temperature,
                n_tokens=n_tokens,
                skip_suffix=True,
            )

    return complete
