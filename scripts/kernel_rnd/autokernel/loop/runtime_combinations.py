"""Installed runtime options share one eligibility check at both loop gates."""
from __future__ import annotations


_ESTABLISHED = frozenset({"threads", "threads_batch", "cpu_list", "numa_policy",
                          "load_threads", "env"})


def established(dimension, allowed_env) -> bool:
    def member(kind, value):
        return (kind in _ESTABLISHED and
                (kind != "env" or value["key"] in allowed_env))
    if dimension.kind == "combination":
        return all(member(item["kind"], item["value"]) for item in dimension.candidate)
    return member(dimension.kind, dimension.candidate)
