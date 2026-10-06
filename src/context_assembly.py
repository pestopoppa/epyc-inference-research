"""DCP-1: Context bundle data model for budget-bounded delegation context pre-assembly.

Per `handoffs/active/delegation-context-preassembly.md` (intake-605). This is the *assemble*
side of context engineering (`context_compression.py` / context folding own the *evict* side):
a `ContextBundle` is a curated, sliced, codemap-augmented set of file references that provably
fits a token budget, assembled *before* a delegated role runs.

This module is the **data model + merge/accounting/policy logic only** (DCP-1). The discovery
+ budget-bounded assembly loop (DCP-2), the GitNexus codemap producer (DCP-3), and dispatcher
wiring (DCP-4) build on top of these types.

Layout note: placed as a top-level `context_assembly` module to match the existing
`context_manager.py` / `context_compression.py` convention rather than a new `src/context/`
package (per the handoff's gap-fix layout note). Carries no orchestrator-runtime imports so it
stays unit-testable in isolation.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

# ─── enums (string constants for storage/serialization simplicity) ───────────────


class InclusionMode:
    FULL = "full"  # whole file body
    SLICES = "slices"  # selected line ranges only
    CODEMAP_ONLY = "codemap_only"  # signature-only API skeleton, no bodies
    EXCLUDED = "excluded"  # deliberately left out (with a reason)
    ALL = ("full", "slices", "codemap_only", "excluded")


class SourceKind:
    GITNEXUS = "gitnexus"
    COLGREP = "colgrep"
    DIRECT_READ = "direct_read"
    MANUAL_SEED = "manual_seed"
    ALL = ("gitnexus", "colgrep", "direct_read", "manual_seed")


# ─── line ranges (1-indexed, inclusive) ──────────────────────────────────────────


@dataclass(frozen=True)
class LineRange:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 1 or self.end < self.start:
            raise ValueError(f"invalid LineRange({self.start}, {self.end})")

    @property
    def n_lines(self) -> int:
        return self.end - self.start + 1


def merge_line_ranges(ranges: Iterable[LineRange], *, adjacency_gap: int = 0) -> list[LineRange]:
    """Merge overlapping / adjacent ranges into a sorted, minimal set.

    Two ranges merge when the gap between them is <= `adjacency_gap` (gap of 0 merges
    touching/overlapping ranges, e.g. 1-10 and 11-20 → 1-20). Used so a file that picks up
    several nearby hunks contributes one contiguous slice rather than many fragments.
    """
    items = sorted(ranges, key=lambda r: (r.start, r.end))
    if not items:
        return []
    merged: list[LineRange] = [items[0]]
    for r in items[1:]:
        last = merged[-1]
        if r.start <= last.end + 1 + adjacency_gap:
            merged[-1] = LineRange(last.start, max(last.end, r.end))
        else:
            merged.append(r)
    return merged


# ─── token estimation (audit note #1: model-calibrated, conservative) ─────────────


#: Default estimator. Conservative (tends to *over*-estimate) so the packer fails closed
#: rather than overflowing the target window. Replace with a real tokenizer via
#: ContextBundle(token_estimator=...) once the target role's tokenizer is wired.
def conservative_char_estimator(text: str) -> int:
    if not text:
        return 0
    # ~3.5 chars/token for English+code, rounded up; +1 floor so non-empty text never costs 0.
    return max(1, math.ceil(len(text) / 3.5))


TokenEstimator = Callable[[str], int]


# ─── exclusion policy (audit note #5: filters belong in DCP-1, not as an afterthought) ──

_EXCLUDE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"(^|/)(node_modules|vendor|\.venv|venv|site-packages|dist|build|target)/"),
        "vendored/build directory",
    ),
    (
        re.compile(
            r"\.(gguf|bin|so|dylib|dll|o|a|png|jpg|jpeg|gif|pdf|zip|tar|gz|whl|faiss|npy|npz|safetensors|onnx)$"
        ),
        "binary/artifact",
    ),
    (
        re.compile(
            r"(^|/)(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|poetry\.lock|Cargo\.lock|uv\.lock)$"
        ),
        "lockfile",
    ),
    (re.compile(r"(\.pb\.go|_pb2\.py|\.generated\.|\.min\.js)$"), "generated code"),
    (re.compile(r"(^|/)(\.env(\..*)?|id_rsa|.*\.pem|.*\.key)$"), "secrets-like file"),
]


def default_exclusion_reason(path: str) -> str | None:
    """Return a reason string if `path` should be excluded by policy, else None.

    Callers may still force-include by passing the file explicitly with
    `source=MANUAL_SEED`; the reason is recorded either way so the worker can request an override.
    """
    for pat, reason in _EXCLUDE_PATTERNS:
        if pat.search(path):
            return reason
    return None


# ─── bundle entry + budget bands ─────────────────────────────────────────────────


@dataclass
class BundleEntry:
    path: str
    mode: str = InclusionMode.CODEMAP_ONLY
    line_ranges: list[LineRange] = field(default_factory=list)
    symbol_ids: list[str] = field(default_factory=list)
    content_sha256: str | None = None
    source: str = SourceKind.DIRECT_READ
    reason_included: str | None = None
    reason_downgraded_or_excluded: str | None = None
    estimated_tokens: int = 0

    def __post_init__(self) -> None:
        if self.mode not in InclusionMode.ALL:
            raise ValueError(f"invalid mode: {self.mode!r}")
        if self.source not in SourceKind.ALL:
            raise ValueError(f"invalid source: {self.source!r}")
        self.normalize()

    def normalize(self) -> None:
        """Merge line ranges (slices mode only)."""
        if self.line_ranges:
            self.line_ranges = merge_line_ranges(self.line_ranges)

    @property
    def counts_toward_budget(self) -> bool:
        return self.mode != InclusionMode.EXCLUDED


@dataclass
class BudgetBands:
    """Explicit budget reservations (DCP-2 packs into these before free packing)."""

    task: int = 0
    codemap: int = 0
    editable: int = 0
    tests: int = 0
    output_reserve: int = 0

    def total_reserved(self) -> int:
        return self.task + self.codemap + self.editable + self.tests + self.output_reserve


# ─── the bundle ──────────────────────────────────────────────────────────────────


@dataclass
class ContextBundle:
    """A budget-bounded, manifest-tracked context bundle for one delegated sub-task.

    The *manifest* (what was included/excluded and why) is kept separate from any rendered
    prompt text so a downstream role can request top-ups against stable IDs (path +
    content_sha256) instead of re-asking for "that file again."
    """

    budget: int
    bundle_id: str | None = None
    repo_sha: str | None = None
    gitnexus_index_commit: str | None = None
    bands: BudgetBands | None = None
    token_estimator: TokenEstimator = conservative_char_estimator
    entries: list[BundleEntry] = field(default_factory=list)

    # ─ mutation ─
    def add_entry(self, entry: BundleEntry, *, body_text: str | None = None) -> BundleEntry:
        """Add an entry. If `body_text` is given and estimated_tokens is unset, estimate it."""
        if entry.estimated_tokens == 0 and body_text:
            entry.estimated_tokens = self.token_estimator(body_text)
        # apply exclusion policy when no explicit reason and not a manual seed
        if entry.source != SourceKind.MANUAL_SEED and entry.mode != InclusionMode.EXCLUDED:
            reason = default_exclusion_reason(entry.path)
            if reason is not None:
                entry.mode = InclusionMode.EXCLUDED
                entry.reason_downgraded_or_excluded = reason
        self.entries.append(entry)
        return entry

    def downgrade(self, path: str, new_mode: str, reason: str) -> bool:
        """Downgrade an entry's mode (e.g. full → codemap_only) to save budget. Returns hit."""
        if new_mode not in InclusionMode.ALL:
            raise ValueError(f"invalid mode: {new_mode!r}")
        for e in self.entries:
            if e.path == path:
                e.mode = new_mode
                e.reason_downgraded_or_excluded = reason
                return True
        return False

    def exclude(self, path: str, reason: str) -> bool:
        return self.downgrade(path, InclusionMode.EXCLUDED, reason)

    # ─ accounting ─
    def total_tokens(self) -> int:
        return sum(e.estimated_tokens for e in self.entries if e.counts_toward_budget)

    def remaining(self) -> int:
        return self.budget - self.total_tokens()

    def fits(self) -> bool:
        return self.total_tokens() <= self.budget

    def included(self) -> list[BundleEntry]:
        return [e for e in self.entries if e.counts_toward_budget]

    def excluded(self) -> list[BundleEntry]:
        return [e for e in self.entries if e.mode == InclusionMode.EXCLUDED]

    # ─ manifest (separate from rendered text) ─
    def manifest(self) -> dict:
        """Machine-readable manifest for the worker prompt + top-up requests + eval."""
        return {
            "bundle_id": self.bundle_id,
            "repo_sha": self.repo_sha,
            "gitnexus_index_commit": self.gitnexus_index_commit,
            "budget": self.budget,
            "total_tokens": self.total_tokens(),
            "remaining": self.remaining(),
            "fits": self.fits(),
            "bands": None
            if self.bands is None
            else {
                "task": self.bands.task,
                "codemap": self.bands.codemap,
                "editable": self.bands.editable,
                "tests": self.bands.tests,
                "output_reserve": self.bands.output_reserve,
            },
            "entries": [
                {
                    "path": e.path,
                    "mode": e.mode,
                    "line_ranges": [[r.start, r.end] for r in e.line_ranges],
                    "symbol_ids": e.symbol_ids,
                    "content_sha256": e.content_sha256,
                    "source": e.source,
                    "reason_included": e.reason_included,
                    "reason_downgraded_or_excluded": e.reason_downgraded_or_excluded,
                    "estimated_tokens": e.estimated_tokens,
                }
                for e in self.entries
            ],
        }


# ─── DCP-2: budget-bounded packing (pure core; discovery is the live part) ────────

# Downgrade ladder: try the richest mode that fits, fall back toward cheaper ones.
_DOWNGRADE_LADDER = [InclusionMode.FULL, InclusionMode.SLICES, InclusionMode.CODEMAP_ONLY]


@dataclass
class Candidate:
    """A discovered file with per-mode token costs and a ranking priority (DCP-2 input).

    Discovery (ColGREP top-k + GitNexus caller/callee neighborhoods) produces these; the
    packing loop below is pure so it is unit-testable without inference or live indexes.
    """

    path: str
    priority: float  # higher = include first
    cost_full: int
    cost_slices: int
    cost_codemap: int
    desired_mode: str = InclusionMode.FULL  # best mode we'd like for this file
    line_ranges: list[LineRange] = field(default_factory=list)
    symbol_ids: list[str] = field(default_factory=list)
    content_sha256: str | None = None
    source: str = SourceKind.COLGREP

    def cost_for(self, mode: str) -> int:
        return {
            InclusionMode.FULL: self.cost_full,
            InclusionMode.SLICES: self.cost_slices,
            InclusionMode.CODEMAP_ONLY: self.cost_codemap,
        }[mode]


def _ladder_from(desired: str) -> list[str]:
    """Modes to try, richest-first, starting no richer than `desired`."""
    if desired not in _DOWNGRADE_LADDER:
        return [InclusionMode.CODEMAP_ONLY]
    start = _DOWNGRADE_LADDER.index(desired)
    return _DOWNGRADE_LADDER[start:]


def pack_to_budget(
    candidates: Iterable[Candidate],
    budget: int,
    *,
    bands: BudgetBands | None = None,
    bundle_id: str | None = None,
    repo_sha: str | None = None,
    gitnexus_index_commit: str | None = None,
) -> ContextBundle:
    """Greedily pack candidates into a token budget, downgrading mode to fit (DCP-2 core).

    By descending priority, include each candidate at the richest mode (no richer than its
    `desired_mode`) whose cost fits the remaining budget; otherwise exclude it with a reason
    ("fail closed" rather than overflow). A ColGREP candidate with hit line ranges cannot fall
    back to signature-only CODEMAP_ONLY: if its allowed FULL/SLICES modes do not fit, it is
    excluded with a missing-evidence reason. Policy-excluded paths (binaries/secrets/...) are
    recorded as EXCLUDED and never consume budget. Pure — no I/O.
    """
    effective_budget = budget if bands is None else budget - bands.output_reserve
    bundle = ContextBundle(
        budget=effective_budget,
        bundle_id=bundle_id,
        repo_sha=repo_sha,
        gitnexus_index_commit=gitnexus_index_commit,
        bands=bands,
    )
    for cand in sorted(candidates, key=lambda c: (-c.priority, c.path)):
        policy_reason = (
            default_exclusion_reason(cand.path) if cand.source != SourceKind.MANUAL_SEED else None
        )
        if policy_reason is not None:
            bundle.entries.append(
                BundleEntry(
                    path=cand.path,
                    mode=InclusionMode.EXCLUDED,
                    source=cand.source,
                    content_sha256=cand.content_sha256,
                    reason_downgraded_or_excluded=policy_reason,
                )
            )
            continue

        placed = False
        has_colgrep_spans = cand.source == SourceKind.COLGREP and bool(cand.line_ranges)
        mode_ladder = _ladder_from(cand.desired_mode)
        if has_colgrep_spans:
            mode_ladder = [mode for mode in mode_ladder if mode != InclusionMode.CODEMAP_ONLY]
        for mode in mode_ladder:
            cost = cand.cost_for(mode)
            if bundle.total_tokens() + cost <= effective_budget:
                downgraded = mode != cand.desired_mode
                entry = BundleEntry(
                    path=cand.path,
                    mode=mode,
                    line_ranges=list(cand.line_ranges) if mode == InclusionMode.SLICES else [],
                    symbol_ids=list(cand.symbol_ids),
                    content_sha256=cand.content_sha256,
                    source=cand.source,
                    estimated_tokens=cost,
                    reason_included=f"priority={cand.priority:g}",
                    reason_downgraded_or_excluded=(
                        f"downgraded {cand.desired_mode}->{mode} to fit budget"
                        if downgraded
                        else None
                    ),
                )
                bundle.entries.append(entry)
                placed = True
                break
        if not placed:
            bundle.entries.append(
                BundleEntry(
                    path=cand.path,
                    mode=InclusionMode.EXCLUDED,
                    source=cand.source,
                    line_ranges=list(cand.line_ranges),
                    symbol_ids=list(cand.symbol_ids),
                    content_sha256=cand.content_sha256,
                    reason_downgraded_or_excluded=(
                        "missing evidence: ColGREP hit spans cannot fit without codemap-only fallback"
                        if has_colgrep_spans
                        else "no mode fits remaining budget (fail-closed)"
                    ),
                )
            )
    return bundle
