"""oracles.py — the ONE §6.5 oracle registry both AK4 consumers read.

WHY THIS MODULE EXISTS
----------------------
`context.py` renders the oracle registry INTO the planner brief; `critic.py`
gates `oracle_reference.oracle` AGAINST the oracle registry. They were built in
parallel and each transcribed §6.5's table independently, so they ended up
sharing exactly ONE oracle id out of nineteen:

    context: 'upstream llama.cpp / ggml'   critic: 'llama.cpp_upstream'
    context: 'AMD AITER'                   critic: 'AITER'
    context: 'AMD composable_kernel / hipBLASLt / rocBLAS'
                                           critic: 'composable_kernel', 'hipBLASLt', 'rocBLAS'

A planner that read the registry from its own context and cited what it read was
therefore rejected by the critic with *"not in the declared registry; new oracles
enter through research-intake, not by an agent adding a row"* — a refusal that
blamed the planner for the controller's own disagreement, and one that no
single-module test could see. The two also disagreed on the CLASS vocabulary:
critic had no `conditional`, so §6.5's FlashAttention/FlashInfer row (*"portable
_source where a HIP path exists, else reimplement"*) was inexpressible there.

So the facts live here once, and both consumers derive. This module is data plus
resolution; it renders nothing and gates nothing.

TWO GRANULARITIES, ONE TABLE
----------------------------
§6.5's table groups trees that share a class and a reason (*"AMD
composable_kernel / hipBLASLt / rocBLAS"* is one row). A PROPOSAL, though, names
the single tree it ported from. Both are legitimate ids, so a row declares its
`group_id` (the §6.5 row, verbatim — what a reader of the design sees) and its
`members` (the trees a port may name), and BOTH resolve here. That is what makes
"cite what you were shown" and "name the tree you copied" the same registry.

New oracles enter through `research-intake` and never by an agent adding a row
(AK-D34), which is why this is a frozen tuple and not a registry with `add()`.

Owning design: `epyc-root/handoffs/active/autokernel-research-loop.md` §6.5,
AK-D16, AK-D34. Governing instrument: `measurement/protocols/kernel-research.md`
(P-AK-SEARCH-1) — an oracle tree is read-only reference material and denial 2
forbids building or measuring a production claim from one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from .. import schemas

__all__ = [
    "OracleRegistryError",
    "ORACLE_ACTIVE", "ORACLE_RETIRED", "ORACLE_STATUSES",
    "HARVEST_PORTABLE_SOURCE", "HARVEST_REIMPLEMENT", "HARVEST_MIXED",
    "HARVEST_CONDITIONAL", "HARVEST_CLASSES", "SPLIT_HARVEST_CLASSES",
    "OracleFact", "REGISTRY",
    "resolve", "known_ids", "group_ids", "member_ids", "active_facts",
    "retired_facts", "harvest_class_of", "is_retired",
    "audit_registry_well_formed", "audit_consumer_registry",
]


class OracleRegistryError(Exception):
    """A registry that cannot be read is a registry nobody may gate on."""


ORACLE_ACTIVE = "active"
ORACLE_RETIRED = "retired"
ORACLE_STATUSES = (ORACLE_ACTIVE, ORACLE_RETIRED)

HARVEST_PORTABLE_SOURCE = "portable_source"
HARVEST_REIMPLEMENT = "reimplement"
HARVEST_MIXED = "mixed"
HARVEST_CONDITIONAL = "conditional"

#: AK-D34's axis is architectural portability, NOT licensing. `mixed` and
#: `conditional` are both in §6.5's table and are NOT synonyms: `mixed` splits by
#: PART (Marlin's CUDA cores reimplement, its layouts port), `conditional` splits
#: by AVAILABILITY (FlashAttention ports where a HIP path exists, else it does
#: not). Collapsing them loses which question to ask before scheduling the work.
HARVEST_CLASSES = (
    HARVEST_PORTABLE_SOURCE, HARVEST_REIMPLEMENT, HARVEST_MIXED,
    HARVEST_CONDITIONAL,
)

#: Classes whose row does not fix ONE class for the whole tree. A port off such a
#: row declares the class it actually relied on, and that declaration may
#: legitimately differ from the row's own label — which is why a consumer must
#: not require equality for these two.
SPLIT_HARVEST_CLASSES = frozenset({HARVEST_MIXED, HARVEST_CONDITIONAL})


@dataclass(frozen=True)
class OracleFact:
    """One §6.5 table row. `group_id` is the row; `members` are its trees.

    `covers` is the surface-family vocabulary `context.oracle_coverage()` matches
    a target against. `class_note` is mandatory for a split class because the
    class is a SCHEDULE input — "mixed" without saying which part ports tells a
    planner nothing it can act on.
    """

    group_id: str
    members: tuple
    harvest_class: str
    why: str
    covers: tuple
    class_note: str = ""
    status: str = ORACLE_ACTIVE
    retired_on: str = ""
    correction: str = ""
    constraint_ref: str = ""
    locator_note: str = "§6.5 oracle registry"

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, str) or not self.group_id.strip():
            raise OracleRegistryError("group_id: required and non-empty")
        if not isinstance(self.members, tuple) or not self.members:
            raise OracleRegistryError(
                f"{self.group_id}: members must be a non-empty tuple; a row a "
                "proposal cannot name is a row that gates nothing"
            )
        for member in self.members:
            if not isinstance(member, str) or not member.strip():
                raise OracleRegistryError(f"{self.group_id}: member ids must be non-empty")
        if self.harvest_class not in HARVEST_CLASSES:
            raise OracleRegistryError(
                f"{self.group_id}: harvest_class {self.harvest_class!r} not in "
                f"{list(HARVEST_CLASSES)}; AK-D34 — an oracle whose class cannot be "
                "established does not enter"
            )
        if self.harvest_class in SPLIT_HARVEST_CLASSES and not self.class_note.strip():
            raise OracleRegistryError(
                f"{self.group_id}: harvest_class {self.harvest_class!r} must carry a "
                "class_note saying which part ports and which must be reimplemented"
            )
        if self.status not in ORACLE_STATUSES:
            raise OracleRegistryError(f"{self.group_id}: unknown status {self.status!r}")
        if self.status == ORACLE_RETIRED and not (
            self.correction.strip() and self.retired_on.strip()
        ):
            raise OracleRegistryError(
                f"{self.group_id}: a retired row must carry its correction and the "
                "date it was retired (§6.5 keeps the row VISIBLE precisely so the "
                "correction is met instead of the row being re-added)"
            )
        if not isinstance(self.covers, tuple):
            raise OracleRegistryError(f"{self.group_id}: covers must be a tuple")

    @property
    def retired(self) -> bool:
        return self.status == ORACLE_RETIRED

    def ids(self) -> tuple:
        """Every id that names this row: the §6.5 group, then its trees.

        Deduplicated because a single-tree row (`ik_llama.cpp`, `CUTLASS`) names
        its group and its member with one string, and that is an identity, not
        the ambiguity `_build_index()` refuses.
        """
        return tuple(dict.fromkeys((self.group_id,) + tuple(self.members)))

    def retirement_note(self) -> Optional[str]:
        if not self.retired:
            return None
        return (
            f"RETIRED {self.retired_on}: {self.correction}"
            + (f" See the {self.constraint_ref!r} HARD_CONSTRAINT in §19.2."
               if self.constraint_ref else "")
        )


#: §6.5's table, verbatim in its groupings, expanded in its members.
REGISTRY: tuple = (
    OracleFact(
        group_id="ik_llama.cpp",
        members=("ik_llama.cpp",),
        harvest_class=HARVEST_PORTABLE_SOURCE,
        why="source of the iqk lineage; the single largest banked gain this project has",
        covers=("quant_gemv", "iqk_quant_kernels", "cpu_repack"),
    ),
    OracleFact(
        group_id="upstream llama.cpp / ggml",
        members=("llama.cpp_upstream", "ggml"),
        harvest_class=HARVEST_PORTABLE_SOURCE,
        why="fixes and optimizations the fork has not taken; the fork diverges continuously",
        covers=("op_coverage", "upstream_fixes", "dispatch"),
    ),
    OracleFact(
        group_id="AMD composable_kernel / hipBLASLt / rocBLAS",
        members=("composable_kernel", "hipBLASLt", "rocBLAS"),
        harvest_class=HARVEST_PORTABLE_SOURCE,
        why=("the most directly relevant unexploited source for gfx90a — CDNA2 "
             "GEMM/attention tiling written by the vendor for this exact architecture"),
        covers=("gemm_tiling", "attention_tiling", "pipelining"),
    ),
    OracleFact(
        group_id="AMD AITER",
        members=("AITER",),
        harvest_class=HARVEST_REIMPLEMENT,
        why=("kept visible only to carry its correction: the row claimed AITER was "
             "'AMD's own inference kernel work, same target hardware', and that was wrong"),
        covers=("gpu_inference_kernels",),
        status=ORACLE_RETIRED,
        retired_on="2026-08-03",
        correction=(
            "AITER's supported-hardware table lists NO MI210/MI250/gfx90a, not even "
            "experimental — consumer RDNA parts rank ahead of our datacenter card."
        ),
        constraint_ref="cdna2-abandoned-by-vendor-and-quant-schools",
        locator_note="§6.5 oracle registry — AITER row, RETIRED 2026-08-03",
    ),
    OracleFact(
        group_id="FlashAttention / FlashInfer",
        members=("FlashAttention", "FlashInfer"),
        harvest_class=HARVEST_CONDITIONAL,
        why="attention tiling, KV layout, paged-attention kernels",
        covers=("attention_tiling", "kv_layout", "paged_attention"),
        class_note="portable_source where a HIP path exists, else reimplement",
    ),
    OracleFact(
        group_id="CUTLASS",
        members=("CUTLASS",),
        harvest_class=HARVEST_REIMPLEMENT,
        why="tiling, pipelining and epilogue DESIGN; the instructions do not port to gfx90a",
        covers=("gemm_tiling", "pipelining", "epilogue"),
    ),
    OracleFact(
        group_id="vLLM / SGLang / TensorRT-LLM",
        members=("vLLM", "SGLang", "TensorRT-LLM"),
        harvest_class=HARVEST_REIMPLEMENT,
        why="scheduling, paged KV and continuous batching as design oracles",
        covers=("scheduling", "paged_kv", "continuous_batching"),
    ),
    OracleFact(
        group_id="Marlin / EXL2 / AWQ / GPTQ kernels",
        members=("Marlin", "EXL2", "AWQ", "GPTQ"),
        harvest_class=HARVEST_MIXED,
        why="low-bit GEMV and dequant layouts — directly adjacent to the G2/G3 seed families",
        covers=("quant_gemv", "dequant_layout", "weight_packing"),
        class_note="CUDA cores reimplement; layout/packing portable_source",
    ),
    OracleFact(
        group_id="Triton / MLC / TVM kernel corpora",
        members=("Triton", "MLC", "TVM"),
        harvest_class=HARVEST_REIMPLEMENT,
        why="autotuning and layout search strategies",
        covers=("autotuning", "layout_search"),
    ),
)


def _build_index() -> dict:
    index: dict = {}
    for fact in REGISTRY:
        for oracle_id in fact.ids():
            if oracle_id in index:
                raise OracleRegistryError(
                    f"oracle id {oracle_id!r} names two rows; an ambiguous id is a "
                    "gate that answers differently depending on which row it found"
                )
            index[oracle_id] = fact
    return index


#: Built at import. An ambiguous registry raises HERE rather than at the first
#: gate that consults it, because a gate is not where a data defect should surface.
_BY_ID: Mapping[str, OracleFact] = _build_index()


def resolve(oracle_id: Any) -> Optional[OracleFact]:
    """The row an id names, by §6.5 group OR by member tree. `None` = unknown.

    `None` is a REJECT for a port and never a shrug: new oracles enter through
    `research-intake`, which verifies real gfx90a/EPYC support and assigns the
    harvest class (AK-D34).
    """
    if not isinstance(oracle_id, str):
        return None
    return _BY_ID.get(oracle_id)


def known_ids() -> frozenset:
    """Every id that resolves — groups and members together."""
    return frozenset(_BY_ID)


def group_ids() -> tuple:
    return tuple(fact.group_id for fact in REGISTRY)


def member_ids() -> tuple:
    return tuple(member for fact in REGISTRY for member in fact.members)


def active_facts() -> tuple:
    return tuple(fact for fact in REGISTRY if not fact.retired)


def retired_facts() -> tuple:
    """Rows kept visible ONLY to carry their correction (§6.5)."""
    return tuple(fact for fact in REGISTRY if fact.retired)


def harvest_class_of(oracle_id: Any) -> Optional[str]:
    fact = resolve(oracle_id)
    return None if fact is None else fact.harvest_class


def is_retired(oracle_id: Any) -> Optional[bool]:
    """`None` when the id does not resolve — "unknown" is not "not retired"."""
    fact = resolve(oracle_id)
    return None if fact is None else fact.retired


def audit_registry_well_formed() -> schemas.Check:
    """Every row resolves, no id is ambiguous, every retired row carries a note.

    Cheap and total, so a consumer may assert it at import instead of trusting
    that this file was reviewed.
    """
    reasons: list = []
    if not REGISTRY:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "the registry is empty; there is nothing to check and an empty registry "
            "rejects every port",
        ))
    for fact in REGISTRY:
        for oracle_id in fact.ids():
            if _BY_ID.get(oracle_id) is not fact:
                reasons.append(f"{oracle_id!r} does not resolve back to its own row")
        if fact.retired and not fact.retirement_note():
            reasons.append(f"{fact.group_id}: retired with no retirement note")
    if not retired_facts():
        # §6.5 deletes no wrong row. If the retired set is empty, either the AITER
        # correction was deleted or this file was rewritten from the pre-2026-08-03
        # table, and both are the failure the row exists to prevent.
        reasons.append(
            "no retired row is present; §6.5 keeps a wrong row VISIBLE with its "
            "correction, so an empty retired set means a correction was deleted"
        )
    if reasons:
        return schemas.Check(schemas.FAIL, tuple(reasons))
    return schemas.Check(schemas.PASS)


def audit_consumer_registry(
    rows: Sequence[Any],
    *,
    id_of,
    harvest_class_of_row,
    retired_of,
    what: str,
) -> schemas.Check:
    """Does a consumer's derived registry still agree with this one?

    The property that matters end to end is not "the tables look alike" but:
    every id the consumer exposes RESOLVES here, with the SAME class and the SAME
    retirement. A consumer that dropped a retired row, renamed an id, or
    reclassified one has silently re-created the fork this module closed.
    """
    if not rows:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"{what}: no rows were supplied; an empty consumer registry cannot be "
            "compared, and it is not evidence of agreement",
        ))
    reasons: list = []
    seen: set = set()
    for row in rows:
        oracle_id = id_of(row)
        seen.add(oracle_id)
        fact = resolve(oracle_id)
        if fact is None:
            reasons.append(
                f"{what}: {oracle_id!r} is not in the §6.5 registry; the consumer "
                "invented or renamed a row"
            )
            continue
        declared = harvest_class_of_row(row)
        if declared != fact.harvest_class:
            reasons.append(
                f"{what}: {oracle_id!r} is classified {declared!r} but §6.5 says "
                f"{fact.harvest_class!r}"
            )
        if bool(retired_of(row)) != fact.retired:
            reasons.append(
                f"{what}: {oracle_id!r} retirement disagrees — consumer says "
                f"retired={bool(retired_of(row))}, §6.5 says retired={fact.retired}"
            )
    for fact in retired_facts():
        if not (set(fact.ids()) & seen):
            reasons.append(
                f"{what}: the retired row {fact.group_id!r} reaches this consumer "
                "under none of its ids; a dropped correction is how the wrong row "
                "gets re-added"
            )
    if reasons:
        return schemas.Check(schemas.FAIL, tuple(reasons))
    return schemas.Check(schemas.PASS)


_WELL_FORMED = audit_registry_well_formed()
if _WELL_FORMED.outcome != schemas.PASS:  # pragma: no cover - import-time invariant
    raise OracleRegistryError(
        "the §6.5 registry is not well formed: " + "; ".join(_WELL_FORMED.reasons)
    )
