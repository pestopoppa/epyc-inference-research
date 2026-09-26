"""Embedding-pool policy: the knobs, loaded from ``orchestration/embedding_pool_policy.yaml``.

Policy is data. Placement (which embedder is on which NUMA node, which serving instance
occupies which nodes, how many slots an embedder has) is NOT here: it is derived from the
stack's own declarations in :mod:`src.embedding_pool.topology`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

POLICY_PATH = Path(__file__).resolve().parents[2] / "orchestration" / "embedding_pool_policy.yaml"

BUSY_SOURCES = frozenset({"ledger", "slots"})


@dataclass(frozen=True)
class ClientPolicy:
    connect_timeout_s: float = 1.0
    request_timeout_s: float = 15.0
    call_timeout_s: float = 30.0
    admission_wait_s: float = 0.0
    failure_backoff_s: float = 2.0
    min_norm: float = 1e-6
    embedding_dim: int | None = 1024
    cache_max_entries: int = 20000
    poll_interval_s: float = 0.02


@dataclass(frozen=True)
class NeighbourCapPolicy:
    enabled: bool = True
    guarded_roles: tuple[str, ...] = ("frontdoor",)
    max_in_flight: int = 1
    busy_sources: tuple[str, ...] = ("ledger", "slots")
    busy_ttl_s: float = 0.25
    slots_timeout_s: float = 0.5
    unknown_is_busy: bool = True
    use_capped_foreign_instances: bool = True


@dataclass(frozen=True)
class EmbeddingPoolPolicy:
    client: ClientPolicy = field(default_factory=ClientPolicy)
    neighbour_cap: NeighbourCapPolicy = field(default_factory=NeighbourCapPolicy)

    def with_cap(self, **changes: Any) -> "EmbeddingPoolPolicy":
        return replace(self, neighbour_cap=replace(self.neighbour_cap, **changes))

    def with_client(self, **changes: Any) -> "EmbeddingPoolPolicy":
        return replace(self, client=replace(self.client, **changes))

    def as_dict(self) -> dict[str, Any]:
        return {
            "client": {f.name: getattr(self.client, f.name) for f in fields(ClientPolicy)},
            "neighbour_cap": {
                f.name: (
                    list(v) if isinstance(v := getattr(self.neighbour_cap, f.name), tuple) else v
                )
                for f in fields(NeighbourCapPolicy)
            },
        }


def _section(cls: type, raw: Any, name: str) -> Any:
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise ValueError(f"embedding_pool_policy: `{name}` must be a mapping")
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(raw) - set(known))
    if unknown:
        raise ValueError(f"embedding_pool_policy: unknown key(s) in `{name}`: {unknown}")
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        default = getattr(cls(), key)
        if isinstance(default, tuple):
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ValueError(f"embedding_pool_policy: `{name}.{key}` must be a list of strings")
            value = tuple(value)
        elif isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"embedding_pool_policy: `{name}.{key}` must be a boolean")
        elif key == "embedding_dim":
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
            ):
                raise ValueError(
                    "embedding_pool_policy: `client.embedding_dim` must be a positive int or null"
                )
        elif isinstance(default, int):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"embedding_pool_policy: `{name}.{key}` must be a non-negative int"
                )
        elif isinstance(default, float):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"embedding_pool_policy: `{name}.{key}` must be a non-negative number"
                )
            value = float(value)
        kwargs[key] = value
    return cls(**kwargs)


def parse_policy(document: Any) -> EmbeddingPoolPolicy:
    """Validate a parsed policy document. Raises ValueError listing the first problem."""
    if not isinstance(document, dict):
        raise ValueError("embedding_pool_policy: document must be a mapping")
    unknown = sorted(set(document) - {"version", "client", "neighbour_cap"})
    if unknown:
        raise ValueError(f"embedding_pool_policy: unknown top-level key(s): {unknown}")
    if document.get("version") != 1:
        raise ValueError("embedding_pool_policy: version must be 1")
    client = _section(ClientPolicy, document.get("client"), "client")
    cap = _section(NeighbourCapPolicy, document.get("neighbour_cap"), "neighbour_cap")
    bad = sorted(set(cap.busy_sources) - BUSY_SOURCES)
    if bad:
        raise ValueError(
            f"embedding_pool_policy: unknown busy source(s) {bad}; known {sorted(BUSY_SOURCES)}"
        )
    if client.request_timeout_s <= 0 or client.call_timeout_s <= 0 or client.connect_timeout_s <= 0:
        raise ValueError("embedding_pool_policy: timeouts must be > 0")
    if client.cache_max_entries < 0:
        raise ValueError("embedding_pool_policy: cache_max_entries must be >= 0")
    if client.poll_interval_s <= 0:
        raise ValueError("embedding_pool_policy: poll_interval_s must be > 0")
    return EmbeddingPoolPolicy(client=client, neighbour_cap=cap)


def load_policy(path: Path | None = None) -> EmbeddingPoolPolicy:
    """Load and validate the policy file. A missing or invalid file raises: no silent defaults."""
    policy_path = POLICY_PATH if path is None else Path(path)
    return parse_policy(yaml.safe_load(policy_path.read_text()))


__all__ = [
    "BUSY_SOURCES",
    "ClientPolicy",
    "EmbeddingPoolPolicy",
    "NeighbourCapPolicy",
    "POLICY_PATH",
    "load_policy",
    "parse_policy",
]
