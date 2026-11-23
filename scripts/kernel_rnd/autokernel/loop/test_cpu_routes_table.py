"""AK-TRIM-OR-SPLIT-GATES golden: the CPU route table moved out of gates.py unchanged."""
from __future__ import annotations

import hashlib

from . import cpu_routes_table, gates

#: sha256 of repr(routes, paths, ops) recorded from gates.py BEFORE the move.
TABLE_DIGEST = "7acaeade7c1bbbd8c23ecec2c66350598b58526800052dd6b2f9b6cec47c6d7b"


def test_route_table_is_unchanged():
    got = repr(([repr(r) for r in gates.CPU_SOURCE_ROUTES],
                [repr(r) for r in gates.CPU_MULTI_FILE_ROUTES],
                gates.CPU_SOURCE_ROUTE_PATHS, gates.CPU_ROUTE_PATHS_ALL,
                gates._DS41_SYNC_OPS, gates._SOLO_ELIGIBLE_OPS))
    assert (len(gates.CPU_SOURCE_ROUTES), len(gates.CPU_MULTI_FILE_ROUTES)) == (18, 2)
    assert hashlib.sha256(got.encode()).hexdigest() == TABLE_DIGEST


def test_gates_reexports_the_table_objects():
    assert gates.CPU_SOURCE_ROUTES is cpu_routes_table.CPU_SOURCE_ROUTES
    assert gates.CpuSourceRoute is cpu_routes_table.CpuSourceRoute
    assert gates.CPU_MULTI_FILE_ROUTES is cpu_routes_table.CPU_MULTI_FILE_ROUTES


def test_generated_route_refusal_wording_is_unchanged():
    patch = "@@ -1,0 +2 @@\n+  #define X 1\n"
    why = gates._generated_route_refusal(patch, "a\nb\n", "a\n")
    assert why is not None and why.startswith("an added line is refused lexically (")
    assert "#define X 1" in why
