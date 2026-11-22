"""Graph-KLD gate policy loader, split verbatim from gates.py (code-budget split)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json


# --- AK-FUSE-GRAPH-KLD-1 keep-path policy (operator ruling 2026-11-18) ---------------------
# The per-backend clauses live in GRAPH_KLD_POLICY_FILE, written by the operator's ratify
# script (P-AK-SEARCH-1-A5 amendment); the values are human-only. Absent or malformed file
# -> `threshold_unset` (a reported refusal). Clauses: GPU decides on 27B Q8_0 (KLD and |dPPL|)
# with Q38FN report-only except a min same-top refusing clause (operator 2026-11-19, run hybrid
# via the clause's n_cpu_moe); CPU adds Q38FN |dPPL| and same-top. |dPPL| % = |exp(ln_ratio)-1|*100.
GRAPH_KLD_POLICY_SCHEMA = "epyc.autokernel.graph_kld_policy.v1"


GRAPH_KLD_POLICY_FILE = Path(__file__).with_name("graph_kld_policy.json")


_GRAPH_KLD_LIMITS = ("max_mean_kld", "max_abs_dppl_pct", "min_same_top_pct")


@dataclass(frozen=True)
class GraphKldClause:
    model: str
    decisive: bool
    max_mean_kld: float | None = None
    max_abs_dppl_pct: float | None = None
    min_same_top_pct: float | None = None
    #: GPU gate vehicle for models not served on GPU (Q38FN): -ngl 999 plus --n-cpu-moe N.
    n_cpu_moe: int | None = None


@dataclass(frozen=True)
class GraphKldPolicy:
    backend: str
    corpus: str
    chunks: int
    models: dict
    clauses: tuple


def load_graph_kld_policy(backend: str, path: "Path | None" = None) -> "GraphKldPolicy | None":
    """The backend's ratified clauses, or None (-> threshold_unset) on any defect: missing
    file, wrong schema, unknown model, a decisive clause without a limit, a non-numeric limit."""
    try:
        cfg = json.loads(Path(path or GRAPH_KLD_POLICY_FILE).read_text(encoding="utf-8"))
        if cfg.get("schema") != GRAPH_KLD_POLICY_SCHEMA:
            return None
        models = {str(k): str(v) for k, v in cfg["models"].items()}
        clauses = []
        for row in cfg["backends"][backend]:
            limits = {k: float(row[k]) for k in _GRAPH_KLD_LIMITS if row.get(k) is not None}
            decisive = bool(row["decisive"])
            if row["model"] not in models or (decisive and not limits):
                return None
            ncm = row.get("n_cpu_moe")
            clauses.append(GraphKldClause(str(row["model"]), decisive, **limits,
                                          n_cpu_moe=None if ncm is None else int(ncm)))
        if not any(c.decisive for c in clauses):
            return None
        return GraphKldPolicy(backend, str(cfg["corpus"]), int(cfg.get("chunks", 16)),
                              models, tuple(clauses))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
