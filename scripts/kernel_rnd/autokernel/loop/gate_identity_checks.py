"""CPU GDN and model-output identity gate checks, split verbatim from gates.py (code-budget split)."""
from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Verdict:
    """Passed, or refused with the reason the actor needs to fix it."""
    gate: str
    passed: bool
    reason: str = ""
    detail: str = ""
    #: AKX-ALL-29.4 write side: one compact record per gated `test-backend-ops` case
    #: (see `gate_case_records`). Nullable and ADDITIVE: it takes no part in equality, is
    #: never read by any gate, and `to_dict` omits the key when it is None so rows
    #: written before this field existed, and verdicts that carry none, keep their shape.
    case_records: "tuple | None" = dataclasses.field(default=None, compare=False)
    elapsed_s: float | None = dataclasses.field(default=None, compare=False)

    def to_dict(self) -> dict:
        row = {"gate": self.gate, "passed": self.passed,
               "reason": self.reason or None, "detail": self.detail[:2000] or None}
        if self.elapsed_s is not None:
            row["elapsed_s"] = self.elapsed_s
        if self.case_records is not None:
            row["case_records"] = list(self.case_records)
        return row


def check_cpu_gdn_reference(build_dir: Path, source_root: Path, *,
                            resolved_recipe=None) -> Verdict:
    """Independent scalar fixture, after the native CPU suite's host/unit check."""
    from . import gdn_reference

    options = ({"launch_env": resolved_recipe.launch_env,
                "topology_prefix": tuple(resolved_recipe.topology_prefix)}
               if resolved_recipe is not None else {})
    result = gdn_reference.check_cpu_gdn(build_dir, source_root, **options)
    return Verdict("reference_comparison" if result.status == "wrong" else
                   "oracle_unavailable" if result.status == "unavailable" else
                   "reference_comparison", result.status == "pass",
                   result.reason, result.detail)


def check_model_output_identity(*, anchor_recipe, candidate_recipe, requests,
                                window=None, repeats: int = 1, record_dir=None) -> Verdict:
    """Whole-model gate for a `model_identity` route (`cpu_weight_placement`)."""
    from . import model_identity

    result = model_identity.check(anchor_recipe=anchor_recipe,
                                  candidate_recipe=candidate_recipe,
                                  requests=tuple(requests or ()), window=window,
                                  repeats=repeats, record_dir=record_dir)
    return Verdict("reference_comparison" if result.status != "unavailable" else
                   "oracle_unavailable", result.status == "pass",
                   result.reason, result.detail)
