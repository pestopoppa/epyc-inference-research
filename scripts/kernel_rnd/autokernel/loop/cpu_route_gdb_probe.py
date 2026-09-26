"""Trusted GDB-side record for a stopped, single-case CPU source-route probe.

Executed by GDB after ``run``/``continue`` stops at a requested breakpoint.  The
inferior has no copy of the write descriptor; its stdout/stderr cannot produce
this record.  The expected symbol pattern arrives through the debugger's own
environment and is unset for the inferior before ``run``.  This file lives with
the loop, never in the actor's kernel tree.
"""
import json
import os
import re

import gdb


def _record():
    expected = os.environ["AK_ROUTE_WITNESS_DSO"]
    pattern = os.environ["AK_ROUTE_WITNESS_SYMBOL"]
    result = {"schema": "epyc.autokernel.cpu_route_hit.v1", "status": "unavailable"}
    try:
        frame = gdb.selected_frame()
        symbol = frame.name() or ""
        result["symbol"] = symbol
        actual = gdb.solib_name(frame.pc())
        hit_once = [bp for bp in (gdb.breakpoints() or []) if bp.hit_count == 1]
        if not hit_once or not actual or not os.path.samefile(actual, expected):
            result["reason"] = "breakpoint or loaded DSO did not match"
        elif symbol == "ggml_backend_cpu_set_use_ref":
            # SysV AMD64: the bool second argument is passed in SIL.
            if int(gdb.parse_and_eval("$rsi")) & 0xff == 1:
                result.update(status="hit", role="independent_reference",
                              dso=os.path.realpath(actual))
            else:
                result["reason"] = "CPU reference was not set to use_ref=true"
        elif re.search(pattern, symbol) is not None:
            result.update(status="hit", role="candidate_route",
                          dso=os.path.realpath(actual))
        else:
            result["reason"] = "unexpected breakpoint symbol"
    except (gdb.error, OSError, ValueError, KeyError, re.error) as exc:
        result["reason"] = str(exc)[:200]
    os.write(int(os.environ["AK_ROUTE_WITNESS_FD"]),
             (json.dumps(result, sort_keys=True) + "\n").encode("utf-8"))


_record()
