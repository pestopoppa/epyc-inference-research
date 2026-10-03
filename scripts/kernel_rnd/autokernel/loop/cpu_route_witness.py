"""Engagement witness plus independent scalar reference for the widened CPU routes.

Same contract as `iqk_witness`: an oracle gate, not a benchmark.  One selected
backend-ops case runs under a one-shot debugger breakpoint pair: the use_ref=true
reference setter, then the edited route's entry symbol.  Both hits must resolve in
the CANDIDATE DSO, the case must be the sole selected one and pass, and (for the
iqk dispatch routes) the engagement marker for the expected type must print.  The
numerical verdict comes from `cpu_quant_reference`, which decodes stored quant bytes
without ggml -- the only independent reference for llamafile (use_ref does not bypass
it) and for scheduling edits (the scalar result does not run through the candidate's
barrier).  The RMS_NORM route instead uses `cpu_norm_reference` (bit identity with
HEAD's arithmetic plus a float64 bound): use_ref runs the same rms_norm body, so the
native suite is not independent there either.  A witness may name further engagement
cases (`also`), each of which must hit in its own debugger run.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess

from .iqk_witness import Result

TIMEOUT_S = 180
SCRIPT = Path(__file__).with_name("cpu_route_gdb_probe.py")
_DENSE = "bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1"
_F16_TINYBLAS_GEMM = ("tinyBLAS<16, float __vector.16., float __vector.16., "
                      "unsigned short, unsigned short, float>::gemm<")
# test_rms_norm(f32, {64,5,4,3}, v=false, 1e-6f) and test_rms_norm_mul_add(f32, {64,5,4,3},
# 1e-6f, broadcast=false): each registered exactly once in the anchor's test-backend-ops.
_RMS_NORM_CASE = "type=f32,ne=[64,5,4,3],v=0,eps=0.000001,inplace=0"
_RMS_NORM_MUL_ADD_CASE = "type=f32,ne=[64,5,4,3],eps=0.000001,broadcast=0,multi_add=0"


@dataclass(frozen=True)
class RouteWitness:
    op: str | None                 # backend-ops op for the engagement case; None = no case
    case: str | None               # exact vars() string of the single selected case
    breakpoint: tuple[str, str] | None  # ("break"|"rbreak", location)
    symbol_pattern: str | None     # regex the hit frame name must match
    active: str | None             # regex the case output must contain (engagement)
    quants: tuple[str, ...]
    ops: tuple[str, ...]
    expert_modes: tuple[str, ...] = ("alternating",)
    # Further engagement cases that must ALSO hit (e.g. a fused entry of the same body).
    also: tuple["RouteWitness", ...] = ()
    # Numerical reference: "quant" = cpu_quant_reference (MUL_MAT/MUL_MAT_ID fixture),
    # "rms_norm" = cpu_norm_reference (bit identity with HEAD plus a float64 bound),
    # "rms_norm_tol" = cpu_norm_reference in tolerance mode (float64 bound 2^-16,
    # deterministic repetitions, no bit identity with HEAD), "model_identity" = gated
    # by `model_identity` in the loop, never by this witness.
    reference: str = "quant"
    # Also run cpu_fusion_reference (the DS41 hc_mixes RMS_NORM -> MUL_MAT(F16) graph):
    # the route can change in-backend fusion, and no native case builds that pair.
    norm_mulmat: bool = False


# Case strings are the reviewed backend-ops fixture's vars(): test_mul_mat(type, f32,
# 16, 16, 256, {1,1}, {1,1}) and test_mul_mat_id(q4_K, f32, 4, 2, false, 512, 4, 256).
# A drifted fixture selects 0 cases and the witness refuses (fails closed).
WITNESSES = {
    "dense_q8_tinyblas": RouteWitness(
        op="MUL_MAT", case=f"type_a=q8_0,type_b=f32,m=16,n=16,k=256,{_DENSE}",
        breakpoint=("rbreak", "tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>::mnpack"),
        symbol_pattern=r"tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>::mnpack",
        active=None, quants=("Q8_0",), ops=("MUL_MAT",)),
    "iqk_mmid_dispatch": RouteWitness(
        op="MUL_MAT_ID", case="type_a=q4_K,type_b=f32,n_mats=4,n_used=2,b=0,m=512,n=4,k=256",
        breakpoint=("break", "ggml_iqk_try_mul_mat_id"),
        symbol_pattern=r"^ggml_iqk_try_mul_mat_id$",
        active=r"\[iqk\] ACTIVE: MoE mul_mat_id via ik kernels \(type=12 ",
        quants=("Q4_K", "Q5_K"), ops=("MUL_MAT_ID",),
        expert_modes=("alternating", "single")),
    "iqk_dense_dispatch": RouteWitness(
        op="MUL_MAT", case=f"type_a=q4_K,type_b=f32,m=16,n=16,k=256,{_DENSE}",
        breakpoint=("break", "ggml_iqk_try_mul_mat"),
        symbol_pattern=r"^ggml_iqk_try_mul_mat$",
        active=r"\[iqk\] ACTIVE: ik_llama GEMM kernels engaged \(first mul_mat type=12 ",
        quants=("Q4_K", "Q5_K", "Q8_0"), ops=("MUL_MAT", "MUL_MAT_ID")),
    # tinyBLAS::matmul is inlined into llamafile_sgemm; the NOINLINE gemm it dispatches to
    # is the class's entry with a symbol. Any RM/RN/BM instantiation (or constprop clone)
    # counts, so a plan change still proves engagement. `.` stands for the parentheses
    # because GDB's rbreak regex and Python's disagree on whether `\(` is literal. The
    # float scalar fixture runs on an 8-thread team with 40 rows (5 y-tiles of 8 under
    # HEAD's plan), i.e. inside the narrow-M regime the route exists to re-plan; branch
    # coverage is still not claimed.
    "float_tinyblas_plan": RouteWitness(
        op="MUL_MAT", case=f"type_a=f16,type_b=f32,m=16,n=16,k=256,{_DENSE}",
        breakpoint=("rbreak", _F16_TINYBLAS_GEMM),
        symbol_pattern=_F16_TINYBLAS_GEMM,
        active=None, quants=("F16", "BF16", "F32"), ops=("MUL_MAT",)),
    # The rms_norm template is inlined into its two exported dispatchers, so each entry
    # proves its instantiation ran in the candidate DSO: the plain case enters
    # ggml_compute_forward_rms_norm, and the fused RMS_NORM_MUL_ADD case (verified to fuse
    # on anchor-gen-001; use_ref disables fusion, so only the candidate reaches it) enters
    # ggml_compute_forward_rms_norm_mul_fused. The numerical verdict is cpu_norm_reference:
    # bit identity with HEAD's arithmetic on an 8-thread team, rows < 8 for the split.
    "cpu_norm_rowsplit": RouteWitness(
        op="RMS_NORM", case=_RMS_NORM_CASE,
        breakpoint=("break", "ggml_compute_forward_rms_norm"),
        symbol_pattern=r"^ggml_compute_forward_rms_norm$",
        active=None, quants=("F32",), ops=("RMS_NORM", "RMS_NORM_MUL_ADD"),
        also=(RouteWitness(
            op="RMS_NORM_MUL_ADD", case=_RMS_NORM_MUL_ADD_CASE,
            breakpoint=("break", "ggml_compute_forward_rms_norm_mul_fused"),
            symbol_pattern=r"^ggml_compute_forward_rms_norm_mul_fused$",
            active=None, quants=("F32",), ops=("RMS_NORM_MUL_ADD",)),),
        reference="rms_norm"),
    # Every node passes the barrier, so entry proves nothing; the numerical suite over
    # every quant, op and width is the witness that publish-before-consume still holds.
    # 2026-10-03: plus the hc_mixes norm->mul_mat graph, so a fusion added to
    # ggml_cpu_try_fuse_ops (seed 3) is run and judged, not merely compiled.
    "cpu_graph_sync": RouteWitness(
        op=None, case=None, breakpoint=None, symbol_pattern=None, active=None,
        quants=("Q4_K", "Q5_K", "Q8_0"), ops=("MUL_MAT", "MUL_MAT_ID"),
        expert_modes=("alternating", "single"), norm_mulmat=True),
    # Same entries as cpu_norm_rowsplit; the numbers are judged in tolerance mode.
    "cpu_norm_numerics": RouteWitness(
        op="RMS_NORM", case=_RMS_NORM_CASE,
        breakpoint=("break", "ggml_compute_forward_rms_norm"),
        symbol_pattern=r"^ggml_compute_forward_rms_norm$",
        active=None, quants=("F32",), ops=("RMS_NORM", "RMS_NORM_MUL_ADD"),
        also=(RouteWitness(
            op="RMS_NORM_MUL_ADD", case=_RMS_NORM_MUL_ADD_CASE,
            breakpoint=("break", "ggml_compute_forward_rms_norm_mul_fused"),
            symbol_pattern=r"^ggml_compute_forward_rms_norm_mul_fused$",
            active=None, quants=("F32",), ops=("RMS_NORM_MUL_ADD",)),),
        reference="rms_norm_tol"),
    # The exported MUL_MAT entry (the iqk dispatch is called from inside it, so every
    # MUL_MAT case enters it); the scalar fixture decodes stored bytes without ggml.
    "cpu_mul_mat_body": RouteWitness(
        op="MUL_MAT", case=f"type_a=q8_0,type_b=f32,m=16,n=16,k=256,{_DENSE}",
        breakpoint=("break", "ggml_compute_forward_mul_mat"),
        symbol_pattern=r"^ggml_compute_forward_mul_mat$",
        active=None, quants=("Q8_0", "Q4_K", "Q5_K", "F16"), ops=("MUL_MAT",)),
    "cpu_weight_placement": RouteWitness(
        op=None, case=None, breakpoint=None, symbol_pattern=None, active=None,
        quants=(), ops=(), reference="model_identity"),
}


def assess_case(witness: RouteWitness, records: list[dict], output: str,
                returncode: int, dso: Path) -> Result:
    """Only debugger-pipe records can establish path engagement."""
    expected = (("independent_reference", "ggml_backend_cpu_set_use_ref"),
                ("candidate_route", None))
    if len(records) != 2 or any(
            record.get("schema") != "epyc.autokernel.cpu_route_hit.v1" or
            record.get("status") != "hit" or record.get("role") != role or
            (symbol is not None and record.get("symbol") != symbol) or
            record.get("dso") != str(dso.resolve())
            for record, (role, symbol) in zip(records, expected)):
        return Result("unavailable", "use_ref=true reference and the edited route entry "
                      "were not both proven in the candidate DSO", str(records)[:1000])
    if not re.search(witness.symbol_pattern or "$^", records[1].get("symbol", "")):
        return Result("unavailable", "route hit symbol does not match the admitted entry",
                      str(records)[:1000])
    plain = re.sub(r"\x1b\[[0-9;]*m", "", output)
    case_line = f"{witness.op}({witness.case}): OK"
    if (returncode != 0 or plain.count(case_line) != 1 or
            "  1/1 tests passed" not in plain or "  Backend CPU: OK" not in plain or
            (witness.active is not None and not re.search(witness.active, plain)) or
            "exited normally" not in plain):
        return Result("wrong" if f"{witness.op}(" in plain and "FAIL" in plain
                      else "unavailable",
                      "route entry hit, but the sole selected case/reference did not pass",
                      plain[-1000:])
    return Result("pass", f"single {witness.op} case passed with route-entry hit and "
                  "use_ref=true reference", str(records))


def _engagement(build_dir: Path, witness: RouteWitness, *, resolved_recipe) -> Result:
    binary = build_dir / "bin/test-backend-ops"
    dso = build_dir / "bin/libggml-cpu.so.0"
    gdb = shutil.which("gdb")
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        return Result("unavailable", "use_ref register witness requires SysV AMD64")
    if not gdb or not binary.is_file() or not dso.is_file() or not SCRIPT.is_file():
        return Result("unavailable", "route witness binary, DSO, GDB or trusted script missing")
    kind, location = witness.breakpoint
    read_fd, write_fd = os.pipe()
    env = dict(resolved_recipe.launch_env)
    env.update(AK_ROUTE_WITNESS_FD=str(write_fd), AK_ROUTE_WITNESS_DSO=str(dso.resolve()),
               AK_ROUTE_WITNESS_SYMBOL=witness.symbol_pattern, DEBUGINFOD_URLS="")
    argv = [*resolved_recipe.topology_prefix, gdb, "-nx", "--return-child-result",
            "-batch", "-q", "-ex", "set pagination off", "-ex", "set confirm off",
            "-ex", "set debuginfod enabled off", "-ex", "set auto-load off",
            "-ex", "set print thread-events off", "-ex", "set breakpoint pending on",
            "-ex", "break ggml_backend_cpu_set_use_ref",
            "-ex", f"python import os; os.set_inheritable({write_fd}, False)",
            "-ex", "unset environment AK_ROUTE_WITNESS_FD",
            "-ex", "unset environment AK_ROUTE_WITNESS_DSO",
            "-ex", "unset environment AK_ROUTE_WITNESS_SYMBOL",
            # The use_ref stop is after the candidate DSO loaded, so a regex
            # breakpoint on a local template member resolves reliably from here.
            "-ex", "run", "-x", str(SCRIPT), "-ex", "disable 1",
            "-ex", f"{kind} {location}", "-ex", "continue", "-x", str(SCRIPT),
            "-ex", "disable", "-ex", "continue",
            "--args", str(binary), "test", "-o", witness.op, "-b", "CPU",
            "-p", "^" + re.escape(witness.case) + "$", "-j", "1"]
    try:
        child = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 pass_fds=(write_fd,), start_new_session=True, text=True)
        os.close(write_fd)
        write_fd = -1
        try:
            stdout, stderr = child.communicate(timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.communicate()
            return Result("unavailable", "route witness debugger case timed out")
        with os.fdopen(read_fd, "r", encoding="utf-8") as pipe:
            read_fd = -1
            text = pipe.read(8192)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return Result("unavailable", f"route witness debugger case could not run: {exc}")
    finally:
        if write_fd >= 0:
            os.close(write_fd)
        if read_fd >= 0:
            os.close(read_fd)
    try:
        records = [json.loads(line) for line in text.splitlines()]
    except (json.JSONDecodeError, UnicodeDecodeError):
        records = []
    return assess_case(witness, records, stdout + stderr, child.returncode, dso)


def check(build_dir: Path, *, resolved_recipe, source_root: Path, route: str,
          source_path: str) -> Result:
    """Engagement (where an entry proves anything) then the independent scalar suite."""
    witness = WITNESSES.get(route)
    if witness is None:
        return Result("unavailable", f"route {route!r} has no reviewed witness")
    if resolved_recipe.backend != "cpu" or \
            dict(resolved_recipe.launch_env).get("GGML_IQK") != "1":
        return Result("unavailable", "CPU route witness requires a resolved CPU GGML_IQK=1 recipe")
    resolved_recipe.validate_launch(resolved_recipe.template, build_dir, resolved_recipe.port)
    if not (Path(source_root) / source_path).is_file():
        return Result("unavailable", "candidate route source is missing")
    details = []
    for engaged in (witness, *witness.also):
        if engaged.breakpoint is None:
            continue
        hit = _engagement(Path(build_dir), engaged, resolved_recipe=resolved_recipe)
        if hit.status != "pass":
            return hit
        details.append(hit.detail)
    if witness.reference == "model_identity":
        return Result("unavailable", f"route {route!r} is gated by the whole-model "
                      "identity check, not by a route witness")
    if witness.reference in ("rms_norm", "rms_norm_tol"):
        from . import cpu_norm_reference
        norm = cpu_norm_reference.check_rms_norm_suite(
            build_dir, source_root, launch_env=resolved_recipe.launch_env,
            topology_prefix=tuple(resolved_recipe.topology_prefix),
            bit_exact=witness.reference == "rms_norm")
        if norm.status != "pass":
            return Result(norm.status, norm.reason, norm.detail)
        details.append(norm.detail)
        return Result("pass", f"{route}: route-entry witnesses ("
                      f"{', '.join(w.op for w in (witness, *witness.also))}) and "
                      f"{norm.reason}; branch coverage not claimed", "\n".join(details))
    if witness.reference != "quant":
        return Result("unavailable", f"route {route!r} names an unknown reference")
    from . import cpu_quant_reference
    for mode in witness.expert_modes:
        scalar = cpu_quant_reference.check_cpu_quant_suite(
            build_dir, source_root, launch_env=resolved_recipe.launch_env,
            topology_prefix=tuple(resolved_recipe.topology_prefix),
            quants=witness.quants, ops=witness.ops, widths=tuple(range(1, 9)),
            expert_mode=mode)
        if scalar.status != "pass":
            return Result(scalar.status, scalar.reason, scalar.detail)
        details.append(scalar.detail)
    fusion_note = ""
    if witness.norm_mulmat:
        from . import cpu_fusion_reference
        fused = cpu_fusion_reference.check_norm_mulmat_suite(
            build_dir, source_root, launch_env=resolved_recipe.launch_env,
            topology_prefix=tuple(resolved_recipe.topology_prefix))
        if fused.status != "pass":
            return Result(fused.status, fused.reason, fused.detail)
        details.append(fused.detail)
        fusion_note = f" and {fused.reason}"
    return Result("pass", f"{route}: " + ("route-entry witness and " if witness.breakpoint
                                          else "") +
                  f"independent scalar {'/'.join(witness.quants)} "
                  f"{'/'.join(witness.ops)} widths 1-8 passed{fusion_note}; branch "
                  "coverage not claimed", "\n".join(details))
