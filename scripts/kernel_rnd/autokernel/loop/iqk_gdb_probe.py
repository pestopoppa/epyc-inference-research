"""Trusted GDB-side record for a stopped, single-case CPU IQK probe.

Executed by GDB after ``run`` stops at the one requested breakpoint.  The
inferior has no copy of the write descriptor; its stdout/stderr cannot produce
this record.  This file lives with the loop, never in the actor's kernel tree.
"""
import json
import os

import gdb


def _record():
    expected = os.environ["AK_IQK_WITNESS_DSO"]
    result = {"schema": "epyc.autokernel.iqk_case_hit.v1",
              "status": "unavailable"}
    try:
        frame = gdb.selected_frame()
        symbol = frame.name()
        result["symbol"] = symbol
        matches = [bp for bp in (gdb.breakpoints() or [])
                   if bp.location == symbol and bp.hit_count == 1]
        actual = gdb.solib_name(frame.pc())
        if len(matches) == 1 and actual and os.path.samefile(actual, expected):
            if symbol == "ggml_backend_cpu_set_use_ref":
                # SysV AMD64: the bool second argument is passed in SIL.
                if int(gdb.parse_and_eval("$rsi")) & 0xff == 1:
                    result.update(status="hit", role="independent_reference",
                                  dso=os.path.realpath(actual))
                else:
                    result["reason"] = "CPU reference was not set to use_ref=true"
            elif symbol == "iqk_mul_mat_moe_rows":
                result.update(status="hit", role="candidate_helper",
                              dso=os.path.realpath(actual))
            else:
                result["reason"] = "unexpected breakpoint symbol"
        else:
            result["reason"] = "breakpoint, symbol or loaded DSO did not match"
    except (gdb.error, OSError, ValueError) as exc:
        result["reason"] = str(exc)[:200]
    os.write(int(os.environ["AK_IQK_WITNESS_FD"]),
             (json.dumps(result, sort_keys=True) + "\n").encode("utf-8"))


_record()
