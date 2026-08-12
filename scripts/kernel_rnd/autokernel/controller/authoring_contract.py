"""AutoKernel authoring prompt, context, and external-number contracts.

This is deliberately an off-campaign-path adapter: the lean AutoKernel loop
does not own a model client, but any planner/actor adapter must cross this seam
before it can emit a prompt or proposal. It implements AK-PL-1, AK-LE-4 and
AK-LE-5 without restoring the deleted controller plane.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable, Optional

from .. import schemas

__all__ = [
    "FORBIDDEN_PROMPT_MARKERS", "PromptLeakError", "ContextBudgetError",
    "CompactionError", "ContextBudget", "ContextItem", "PricedContext",
    "price_context", "GitRecoveryRecipe", "CompactionRecord",
    "ExternalNumber", "EvaluationCase", "CasePopulation", "DiagnosticRecord",
    "DiagnosticDisclosure", "filter_refine_diagnostics", "assert_prompt_hygiene",
    "assemble_authoring_prompt", "c4_profile_context_item",
]


FORBIDDEN_PROMPT_MARKERS = (
    "test-backend-ops",
    "max_nmse_err",
    "init_tensor_uniform",
)


class PromptLeakError(ValueError):
    """A fully rendered authoring prompt discloses sealed evaluator internals."""


class ContextBudgetError(ValueError):
    """A round asks for unpriced, bulk, duplicate, or over-budget context."""


class CompactionError(ValueError):
    """A research-log compaction is not recoverable from an immutable source."""


VISIBILITY_PUBLIC = "PUBLIC"
VISIBILITY_SEALED = "SEALED"


@dataclass(frozen=True)
class ContextBudget:
    """Pre-priced per-round context ceiling."""

    max_total_tokens: int
    max_item_tokens: int
    max_items: int

    def __post_init__(self) -> None:
        for name in ("max_total_tokens", "max_item_tokens", "max_items"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ContextBudgetError(f"{name} must be a positive integer")
        if self.max_item_tokens > self.max_total_tokens:
            raise ContextBudgetError(
                "max_item_tokens cannot exceed the round's total token budget")


@dataclass(frozen=True)
class ContextItem:
    """One selected excerpt; a whole-file read is structurally inadmissible."""

    source_ref: str
    purpose: str
    content: str
    bulk_read: bool = False

    def __post_init__(self) -> None:
        for name in ("source_ref", "purpose", "content"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ContextBudgetError(f"context item {name} must be non-empty")
            if "\0" in value:
                raise ContextBudgetError(f"context item {name} contains NUL")
        if not isinstance(self.bulk_read, bool):
            raise ContextBudgetError("bulk_read must be boolean")

    @property
    def estimated_tokens(self) -> int:
        # Deterministic, provider-independent pre-price. Realized provider usage
        # is evidence, never permission to exceed this round's declared ceiling.
        return max(1, math.ceil(len(self.content.encode("utf-8")) / 4))


@dataclass(frozen=True)
class PricedContext:
    round_id: str
    budget: ContextBudget
    items: tuple[ContextItem, ...]
    estimated_tokens: int

    def render_budget_table(self) -> str:
        rows = [
            "| Source | Purpose | Estimated tokens | Bulk read |",
            "|---|---|---:|---|",
        ]
        for item in self.items:
            source = item.source_ref.replace("|", "\\|").replace("\n", " ")
            purpose = item.purpose.replace("|", "\\|").replace("\n", " ")
            rows.append(
                f"| {source} | {purpose} | {item.estimated_tokens} | no |")
        rows.append(
            f"| TOTAL | round {self.round_id} | {self.estimated_tokens} / "
            f"{self.budget.max_total_tokens} | NEVER |")
        return "\n".join(rows)


def price_context(*, round_id: str, budget: ContextBudget,
                  items: Iterable[ContextItem]) -> PricedContext:
    """Price and admit one round's selected excerpts; never bulk-read."""
    if not isinstance(round_id, str) or not round_id.strip():
        raise ContextBudgetError("round_id must be non-empty")
    if not isinstance(budget, ContextBudget):
        raise TypeError("budget must be a ContextBudget")
    rows = tuple(items)
    if len(rows) > budget.max_items:
        raise ContextBudgetError(
            f"round has {len(rows)} context items; budget permits {budget.max_items}")
    seen: set[str] = set()
    total = 0
    for item in rows:
        if not isinstance(item, ContextItem):
            raise TypeError("items must contain ContextItem values")
        if item.bulk_read:
            raise ContextBudgetError(
                f"{item.source_ref}: bulk reads are never admissible; select a priced excerpt")
        if item.source_ref in seen:
            raise ContextBudgetError(
                f"{item.source_ref}: duplicate context source in one round")
        seen.add(item.source_ref)
        if item.estimated_tokens > budget.max_item_tokens:
            raise ContextBudgetError(
                f"{item.source_ref}: estimated {item.estimated_tokens} tokens exceeds "
                f"per-item ceiling {budget.max_item_tokens}")
        total += item.estimated_tokens
    if total > budget.max_total_tokens:
        raise ContextBudgetError(
            f"round context estimates {total} tokens; ceiling is "
            f"{budget.max_total_tokens}")
    return PricedContext(round_id=round_id, budget=budget, items=rows,
                         estimated_tokens=total)


def c4_profile_context_item(report_path: str, *, expected_sha256: str) -> ContextItem:
    """Project a hash-bound C4 report into the priced authoring-context seam."""
    from .. import profile_context

    context = profile_context.load_profile_context(
        report_path, expected_sha256=expected_sha256)
    content = json.dumps(
        context.discovery_context(), sort_keys=True, separators=(",", ":"))
    # This is a contractual check, not a caller convenience: a new C4 field may
    # never silently disclose a sealed evaluator marker to the authoring model.
    assert_prompt_hygiene(content)
    return ContextItem(
        source_ref=f"c4-profile://{context.report_sha256}",
        purpose=f"C4 {context.stage} mechanism and wall-share evidence",
        content=content,
    )


@dataclass(frozen=True)
class GitRecoveryRecipe:
    """Immutable source for recovering a pre-compaction research log."""

    repo: str
    commit: str
    path: str

    def __post_init__(self) -> None:
        if not isinstance(self.repo, str) or not os.path.isabs(self.repo):
            raise CompactionError("recovery repo must be an absolute path")
        try:
            schemas.require.commit(self.commit, "recovery.commit")
        except ValueError as exc:
            raise CompactionError(str(exc)) from exc
        if not isinstance(self.path, str) or not self.path.strip():
            raise CompactionError("recovery path must be a non-empty repository-relative path")
        parsed = PurePosixPath(self.path)
        if parsed.is_absolute() or ".." in parsed.parts or self.path.startswith("-"):
            raise CompactionError(
                "recovery path must be relative, option-safe, and contain no '..'")

    @property
    def argv(self) -> tuple[str, ...]:
        return ("git", "-C", self.repo, "show", f"{self.commit}:{self.path}")

    @property
    def command(self) -> str:
        return " ".join(shlex.quote(part) for part in self.argv)


@dataclass(frozen=True)
class CompactionRecord:
    """Only admissible compacted-log payload: header, recovery, then body."""

    kept: tuple[str, ...]
    dropped: tuple[str, ...]
    recovery: GitRecoveryRecipe
    compacted_body: str

    def __post_init__(self) -> None:
        for name in ("kept", "dropped"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values:
                raise CompactionError(f"{name} must be a non-empty tuple")
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise CompactionError(f"{name} entries must be non-empty strings")
        overlap = set(self.kept) & set(self.dropped)
        if overlap:
            raise CompactionError(f"sections cannot be both kept and dropped: {sorted(overlap)}")
        if not isinstance(self.recovery, GitRecoveryRecipe):
            raise TypeError("recovery must be a GitRecoveryRecipe")
        if not isinstance(self.compacted_body, str) or not self.compacted_body.strip():
            raise CompactionError("compacted_body must be non-empty")

    def render(self) -> str:
        kept = "\n".join(f"- {item}" for item in self.kept)
        dropped = "\n".join(f"- {item}" for item in self.dropped)
        return (
            "<!-- AUTOKERNEL REVERSIBLE COMPACTION\n"
            "WHAT WAS KEPT:\n" + kept + "\n"
            "WHAT WAS DROPPED:\n" + dropped + "\n"
            "RECOVERY RECIPE:\n" + self.recovery.command + "\n"
            "END AUTOKERNEL REVERSIBLE COMPACTION -->\n\n"
            + self.compacted_body.rstrip() + "\n"
        )


@dataclass(frozen=True)
class ExternalNumber:
    """AK-LE-5 structured external number; never an unreceipted scalar."""

    external_number_id: str
    label: str
    observed_value: float
    unit: str
    source_ref: str
    quant: str
    basis: str
    denominator_value: float
    denominator_source_ref: str
    retrieved_at: Optional[str] = None
    source_commit: Optional[str] = None

    def __post_init__(self) -> None:
        if (isinstance(self.denominator_value, bool)
                or not isinstance(self.denominator_value, (int, float))
                or not math.isfinite(float(self.denominator_value))
                or self.denominator_value <= 0):
            raise ValueError("denominator_value must be a positive finite number")
        errors = schemas.validate_external_number(self.to_dict())
        if errors:
            raise ValueError("invalid external number: " + "; ".join(errors))

    @property
    def normalized_roofline_utilization(self) -> float:
        return float(self.observed_value) / float(self.denominator_value)

    def to_dict(self) -> dict:
        return {
            "external_number_id": self.external_number_id,
            "label": self.label,
            "observed_value": self.observed_value,
            "unit": self.unit,
            "source_ref": self.source_ref,
            "retrieved_at": self.retrieved_at,
            "source_commit": self.source_commit,
            "quant": self.quant,
            "basis": self.basis,
            "roofline_denominator": {
                "value": self.denominator_value,
                "unit": self.unit,
                "quant": self.quant,
                "basis": self.basis,
                "source_ref": self.denominator_source_ref,
            },
            "normalized_roofline_utilization": self.normalized_roofline_utilization,
        }

    def render_prompt_line(self) -> str:
        revision = self.source_commit or self.retrieved_at
        return (
            f"{self.label}: {self.observed_value:g} {self.unit}; quant={self.quant}; "
            f"basis={self.basis}; roofline_utilization="
            f"{self.normalized_roofline_utilization:.6f}; source={self.source_ref}; "
            f"source_revision={revision}; denominator_source="
            f"{self.denominator_source_ref}"
        )


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    family: str
    visibility: str

    def __post_init__(self) -> None:
        for name in ("case_id", "family"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"evaluation case {name} must be non-empty")
        if self.visibility not in {VISIBILITY_PUBLIC, VISIBILITY_SEALED}:
            raise ValueError("evaluation case visibility must be PUBLIC or SEALED")


@dataclass(frozen=True)
class CasePopulation:
    """Predeclared PUBLIC/SEALED split; sealed ids have no prompt projection."""

    population_id: str
    selection_seed_sha256: str
    cases: tuple[EvaluationCase, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.population_id, str) or not self.population_id.strip():
            raise ValueError("population_id must be non-empty")
        schemas.require.sha256(self.selection_seed_sha256,
                               "case_population.selection_seed_sha256")
        if not isinstance(self.cases, tuple) or not self.cases:
            raise ValueError("case population must be a non-empty tuple")
        if any(not isinstance(case, EvaluationCase) for case in self.cases):
            raise TypeError("case population contains a non-EvaluationCase")
        ids = tuple(case.case_id for case in self.cases)
        if len(set(ids)) != len(ids):
            raise ValueError("case ids must be unique across PUBLIC and SEALED populations")
        visibility = {case.visibility for case in self.cases}
        if visibility != {VISIBILITY_PUBLIC, VISIBILITY_SEALED}:
            raise ValueError("case population must predeclare both PUBLIC and SEALED cases")

    @property
    def public_ids(self) -> frozenset[str]:
        return frozenset(case.case_id for case in self.cases
                         if case.visibility == VISIBILITY_PUBLIC)

    @property
    def sealed_ids(self) -> frozenset[str]:
        return frozenset(case.case_id for case in self.cases
                         if case.visibility == VISIBILITY_SEALED)


@dataclass(frozen=True)
class DiagnosticRecord:
    case_id: str
    summary: str
    detail: str

    def __post_init__(self) -> None:
        for name in ("case_id", "summary", "detail"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"diagnostic {name} must be non-empty")


@dataclass(frozen=True)
class DiagnosticDisclosure:
    public: tuple[str, ...]
    sealed: tuple[DiagnosticRecord, ...]
    population_id: str

    def render_public(self) -> str:
        rows = "\n".join(f"- {message}" for message in self.public)
        return rows or "(no public repair diagnostics)"


def filter_refine_diagnostics(*, population: CasePopulation,
                              diagnostics: Iterable[DiagnosticRecord]
                              ) -> DiagnosticDisclosure:
    """Project only safe PUBLIC summaries into a refine prompt.

    SEALED ids and details remain in the returned evaluator-side record but
    have no public string projection. Public summaries containing evaluator
    implementation names or exact ``ERR = value > tolerance`` disclosures are
    replaced by a generic mismatch class.
    """
    if not isinstance(population, CasePopulation):
        raise TypeError("population must be a CasePopulation")
    public: list[str] = []
    sealed: list[DiagnosticRecord] = []
    known = population.public_ids | population.sealed_ids
    for record in diagnostics:
        if not isinstance(record, DiagnosticRecord):
            raise TypeError("diagnostics must contain DiagnosticRecord values")
        if record.case_id not in known:
            raise ValueError(f"diagnostic names undeclared case {record.case_id!r}")
        if record.case_id in population.sealed_ids:
            sealed.append(record)
            continue
        candidate = record.summary.strip()
        folded = candidate.casefold()
        if (any(marker in folded for marker in FORBIDDEN_PROMPT_MARKERS)
                or re.search(r"\bERR\s*=.*?>", candidate, re.IGNORECASE)):
            candidate = "numerical mismatch in a public case; inspect kernel semantics"
        public.append(candidate)
    return DiagnosticDisclosure(
        public=tuple(public), sealed=tuple(sealed),
        population_id=population.population_id)


def assert_prompt_hygiene(rendered_prompt: str) -> None:
    """Scan the fully rendered prompt, after every context line is assembled."""
    if not isinstance(rendered_prompt, str) or not rendered_prompt.strip():
        raise PromptLeakError("rendered authoring prompt must be non-empty")
    folded = rendered_prompt.casefold()
    leaked = tuple(marker for marker in FORBIDDEN_PROMPT_MARKERS if marker in folded)
    if leaked:
        raise PromptLeakError(
            "assembled authoring prompt exposes sealed evaluator internals: "
            + ", ".join(leaked))
    if re.search(r"\bERR\s*=\s*[-+0-9.eE]+\s*>\s*[-+0-9.eE]+", rendered_prompt,
                 re.IGNORECASE):
        raise PromptLeakError(
            "assembled authoring prompt exposes an exact evaluator error/tolerance pair")


def assemble_authoring_prompt(*, role: str, task: str, context: PricedContext,
                              external_numbers: Iterable[ExternalNumber] = (),
                              diagnostics: Optional[DiagnosticDisclosure] = None) -> str:
    """The reviewed planner/actor prompt assembly seam.

    The leak scan happens last, over the complete string. External numbers can
    enter only through validated typed records; context has already passed the
    per-round budget and never-bulk-read gate.
    """
    if role not in {"planner", "actor", "critic", "implement", "exploit"}:
        raise ValueError(f"unknown authoring role {role!r}")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be non-empty")
    if not isinstance(context, PricedContext):
        raise TypeError("context must be a PricedContext")
    external = tuple(external_numbers)
    if any(not isinstance(value, ExternalNumber) for value in external):
        raise TypeError("external_numbers must contain ExternalNumber values")
    excerpts = "\n\n".join(
        f"[{item.source_ref}] purpose={item.purpose}\n{item.content}"
        for item in context.items
    ) or "(no selected context)"
    numbers = "\n".join(value.render_prompt_line() for value in external) \
        or "(no external numeric design priors)"
    if diagnostics is not None and not isinstance(diagnostics, DiagnosticDisclosure):
        raise TypeError("diagnostics must be a DiagnosticDisclosure or None")
    public_diagnostics = ("(no evaluator feedback)" if diagnostics is None
                          else diagnostics.render_public())
    rendered = (
        f"AUTOKERNEL AUTHORING ROLE: {role}\n"
        f"TASK:\n{task.strip()}\n\n"
        "CONTEXT BUDGET (never bulk-read):\n"
        f"{context.render_budget_table()}\n\n"
        f"SELECTED CONTEXT:\n{excerpts}\n\n"
        f"ADMISSIBLE EXTERNAL NUMBERS:\n{numbers}\n"
        f"PUBLIC REPAIR DIAGNOSTICS:\n{public_diagnostics}\n"
    )
    assert_prompt_hygiene(rendered)
    return rendered
