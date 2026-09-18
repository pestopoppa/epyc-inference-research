"""Trusted GDB-side record for a stopped, single-case CPU IQK probe.

Executed by GDB after ``run`` stops at the one requested breakpoint.  The
inferior has no copy of the write descriptor; its stdout/stderr cannot produce
this record.  This file lives with the loop, never in the actor's kernel tree.
"""
import json
import os
import re

import gdb


def _record():
    expected = os.environ["AK_IQK_WITNESS_DSO"]
    result = {"schema": "epyc.autokernel.iqk_case_hit.v1",
              "status": "unavailable"}
    try:
        frame = gdb.selected_frame()
        symbol = frame.name() or ""
        result["symbol"] = symbol
        dot_quant = os.environ.get("AK_IQK_DOT_QUANT")
        dot_width = os.environ.get("AK_IQK_DOT_WIDTH")
        dot_dequant = {"Q4_K": "DequantizerQ4K_AVX2",
                       "Q5_K": "DequantizerQ5K_AVX2"}.get(dot_quant)
        dot_hit = (dot_dequant is not None and dot_width in {str(i) for i in range(1, 9)}
                   and re.search(r"mul_mat_qX_K_q8_2_X4_T<.*" + dot_dequant +
                                 r",\s*" + dot_width + r">", symbol) is not None)
        matches = [bp for bp in (gdb.breakpoints() or [])
                   if bp.hit_count == 1 and
                   (bp.location == symbol or
                    (dot_hit and "mul_mat_qX_K_q8_2_X4_T" in (bp.location or "") and
                     dot_dequant in (bp.location or "")))]
        actual = gdb.solib_name(frame.pc())
        if len(matches) == 1 and actual and os.path.samefile(actual, expected):
            if symbol == "ggml_backend_cpu_set_use_ref":
                # SysV AMD64: the bool second argument is passed in SIL.
                if int(gdb.parse_and_eval("$rsi")) & 0xff == 1:
                    result.update(status="hit", role="independent_reference",
                                  dso=os.path.realpath(actual))
                else:
                    result["reason"] = "CPU reference was not set to use_ref=true"
            elif symbol in {"iqk_mul_mat_moe_rows", "iqk_moe_fused_up_gate"}:
                result.update(status="hit", role="candidate_helper",
                              dso=os.path.realpath(actual))
            elif dot_hit:
                result.update(status="hit", role="candidate_dot",
                              symbol="mul_mat_qX_K_q8_2_X4_T",
                              quant=dot_quant, width=int(dot_width),
                              frame_symbol=symbol,
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
