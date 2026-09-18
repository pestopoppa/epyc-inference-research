"""Single-case, no-rebuild CPU IQK source engagement witness.

This is an oracle gate, not a benchmark.  The trusted backend-ops executable
compares candidate CPU with its use_ref=true CPU reference; a one-shot debugger
breakpoint proves an allowlisted inner helper ran in the candidate DSO.  The
candidate cannot supply the debugger script, case selector or witness pipe.
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


# These two cases are enumerated by the reviewed backend-ops fixture.  The
# exact regex plus a 1/1 tally prevents a broad filtered suite from lending a
# different quant's ACTIVE marker to this helper witness.
CASES = (("Q4_K", "q4_K", 12, 1), ("Q5_K", "q5_K", 13, 2))
SYMBOL = "iqk_mul_mat_moe_rows"
SOURCE = "ggml/src/ggml-cpu/iqk/iqk_mul_mat.cpp"
TIMEOUT_S = 180


@dataclass(frozen=True)
class Result:
    status: str  # pass | wrong | unavailable
    reason: str
    detail: str = ""


def _assess_case(witness: str, stdout: str, stderr: str, returncode: int,
                 dso: Path, case_text: str, type_id: int, quant: str) -> Result:
    """Only debugger-pipe records can establish path engagement."""
    try:
        records = [json.loads(line) for line in witness.splitlines()]
    except (json.JSONDecodeError, UnicodeDecodeError):
        return Result("unavailable", "trusted IQK debugger hit record absent or malformed",
                      (stdout + stderr)[-1000:])
    expected_hits = (("independent_reference", "ggml_backend_cpu_set_use_ref"),
                     ("candidate_helper", SYMBOL))
    if (len(records) != 2 or any(
            record.get("schema") != "epyc.autokernel.iqk_case_hit.v1" or
            record.get("status") != "hit" or record.get("role") != role or
            record.get("symbol") != symbol or
            record.get("dso") != str(dso.resolve())
            for record, (role, symbol) in zip(records, expected_hits))):
        return Result("unavailable", "use_ref=true reference and selected IQK helper "
                      "were not both proven in the candidate DSO", str(records)[:1000])
    plain = re.sub(r"\x1b\[[0-9;]*m", "", stdout + stderr)
    case_line = f"MUL_MAT_ID({case_text}): OK"
    active = re.compile(r"\[iqk\] ACTIVE: MoE mul_mat_id via ik kernels "
                        rf"\(type={type_id} activation=\d+ n_as=4\)")
    if (returncode != 0 or plain.count(case_line) != 1 or
            "  1/1 tests passed" not in plain or
            "  Backend CPU: OK" not in plain or
            not active.search(plain) or
            "exited normally" not in plain):
        return Result("wrong" if "MUL_MAT_ID(" in plain and "FAIL" in plain else "unavailable",
                      "IQK helper hit, but the sole selected case/reference did not pass",
                      plain[-1000:])
    return Result("pass", f"single {quant} MUL_MAT_ID case passed with inner-helper "
                  "hit and use_ref=true reference", str(records))


def _check_one(build_dir: Path, *, resolved_recipe, source_root: Path,
               quant: str, type_name: str, type_id: int, n_used: int) -> Result:
    """Prove one exact quant case and reference in one process."""
    binary = build_dir / "bin/test-backend-ops"
    dso = build_dir / "bin/libggml-cpu.so.0"
    gdb = shutil.which("gdb")
    script = Path(__file__).with_name("iqk_gdb_probe.py")
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        return Result("unavailable", "IQK use_ref register witness requires SysV AMD64")
    if not gdb or not binary.is_file() or not dso.is_file() or not script.is_file():
        return Result("unavailable", "IQK case witness binary, DSO, GDB or trusted script missing")
    resolved_recipe.validate_launch(resolved_recipe.template, build_dir,
                                    resolved_recipe.port)
    if resolved_recipe.backend != "cpu" or dict(resolved_recipe.launch_env).get("GGML_IQK") != "1":
        return Result("unavailable", "IQK case witness requires resolved CPU recipe with GGML_IQK=1")
    if not (source_root / SOURCE).is_file():
        return Result("unavailable", "candidate IQK source is missing")

    read_fd, write_fd = os.pipe()
    env = dict(resolved_recipe.launch_env)
    env.update(AK_IQK_WITNESS_FD=str(write_fd),
               AK_IQK_WITNESS_DSO=str(dso.resolve()), DEBUGINFOD_URLS="")
    case_text = (f"type_a={type_name},type_b=f32,n_mats=4,n_used={n_used},"
                 "b=0,m=512,n=1,k=256")
    argv = [*resolved_recipe.topology_prefix, gdb, "-nx", "--return-child-result",
            "-batch", "-q", "-ex", "set pagination off", "-ex", "set confirm off",
            "-ex", "set debuginfod enabled off", "-ex", "set auto-load off",
            "-ex", "set print thread-events off", "-ex", "set breakpoint pending on",
            "-ex", "break ggml_backend_cpu_set_use_ref",
            "-ex", f"break {SYMBOL}",
            "-ex", f"python import os; os.set_inheritable({write_fd}, False)",
            "-ex", "unset environment AK_IQK_WITNESS_FD",
            "-ex", "unset environment AK_IQK_WITNESS_DSO",
            "-ex", "run", "-x", str(script), "-ex", "disable 1", "-ex", "continue",
            "-x", str(script), "-ex", "disable 2", "-ex", "continue",
            "--args", str(binary), "test", "-o", "MUL_MAT_ID", "-b", "CPU",
            "-p", "^" + case_text + "$", "-j", "1"]
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
            return Result("unavailable", "IQK debugger case timed out")
        with os.fdopen(read_fd, "r", encoding="utf-8") as pipe:
            read_fd = -1
            witness = pipe.read(4096)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return Result("unavailable", f"IQK debugger case could not run: {exc}")
    finally:
        if write_fd >= 0:
            os.close(write_fd)
        if read_fd >= 0:
            os.close(read_fd)

    return _assess_case(witness, stdout, stderr, child.returncode,
                        dso, case_text, type_id, quant)


def check(build_dir: Path, *, resolved_recipe, source_root: Path) -> Result:
    """Require both quant-path hits and the independent 40-row scalar fixture."""
    hits = []
    for quant, type_name, type_id, n_used in CASES:
        result = _check_one(build_dir, resolved_recipe=resolved_recipe,
                            source_root=source_root, quant=quant,
                            type_name=type_name, type_id=type_id, n_used=n_used)
        if result.status != "pass":
            return result
        hits.append(result.detail)
    from . import cpu_quant_reference
    scalar = cpu_quant_reference.check_cpu_quant_suite(
        build_dir, source_root, launch_env=resolved_recipe.launch_env,
        topology_prefix=tuple(resolved_recipe.topology_prefix),
        quants=("Q4_K", "Q5_K"), ops=("MUL_MAT_ID",))
    if scalar.status != "pass":
        return Result(scalar.status, scalar.reason, scalar.detail)
    return Result("pass", "Q4_K/Q5_K helper-entry witnesses and independent "
                  "40-row scalar output comparisons passed; branch-level coverage "
                  "is not claimed", "\n".join((scalar.detail, *hits)))


def _assess_fused_hit_records(records: list[dict], dso: Path, *,
                              dot_quant: str | None, width: int) -> Result:
    expected = [{"schema": "epyc.autokernel.iqk_case_hit.v1", "status": "hit",
                 "role": "candidate_helper", "symbol": "iqk_moe_fused_up_gate",
                 "dso": str(dso.resolve())}]
    if dot_quant is not None:
        expected.append({"schema": "epyc.autokernel.iqk_case_hit.v1",
                         "status": "hit", "role": "candidate_dot",
                         "symbol": "mul_mat_qX_K_q8_2_X4_T",
                         "quant": dot_quant, "width": width,
                         "dso": str(dso.resolve())})
    if len(records) != len(expected) or any(
            any(record.get(key) != value for key, value in wanted.items())
            for record, wanted in zip(records, expected)):
        return Result("unavailable", "exact fused helper and dot specialization "
                      "hits in candidate DSO not proven", str(records)[:1000])
    return Result("pass", "trusted candidate DSO hits matched", str(records))


def run_fused_probe(binary: Path, quant: str, *, dso: Path,
                    launch_env: dict[str, str],
                    topology_prefix: tuple[str, ...],
                    expected_width: int = 2,
                    dot_quant: str | None = None,
                    single_expert: bool = False) -> tuple[Result, str]:
    """Run the exact fused graph under a trusted one-shot helper breakpoint."""
    if expected_width not in range(1, 9) or \
            (dot_quant is not None and (dot_quant != quant or not single_expert)) or \
            (dot_quant is not None and dot_quant not in {"Q4_K", "Q5_K"}):
        return Result("unavailable", "unsupported IQK fused dot witness request"), ""
    gdb = shutil.which("gdb")
    script = Path(__file__).with_name("iqk_gdb_probe.py")
    if not gdb or not dso.is_file() or not script.is_file():
        return Result("unavailable", "fused IQK debugger or candidate DSO missing"), ""
    read_fd, write_fd = os.pipe()
    env = dict(launch_env)
    env.update(AK_IQK_WITNESS_FD=str(write_fd),
               AK_IQK_WITNESS_DSO=str(dso.resolve()), DEBUGINFOD_URLS="")
    if dot_quant is not None:
        env.update(AK_IQK_DOT_QUANT=dot_quant,
                   AK_IQK_DOT_WIDTH=str(expected_width))
    argv = [*topology_prefix, gdb, "-nx", "--return-child-result", "-batch", "-q",
            "-ex", "set pagination off", "-ex", "set confirm off",
            "-ex", "set debuginfod enabled off", "-ex", "set auto-load off",
            "-ex", "set print thread-events off", "-ex", "set breakpoint pending on",
            "-ex", "break iqk_moe_fused_up_gate",
            "-ex", f"python import os; os.set_inheritable({write_fd}, False)",
            "-ex", "unset environment AK_IQK_WITNESS_FD",
            "-ex", "unset environment AK_IQK_WITNESS_DSO",
            "-ex", "unset environment AK_IQK_DOT_QUANT",
            "-ex", "unset environment AK_IQK_DOT_WIDTH",
            "-ex", "run", "-x", str(script), "-ex", "disable 1"]
    if dot_quant is not None:
        # At this point the fused helper breakpoint has loaded the candidate
        # CPU DSO.  rbreak can now resolve its local template specialization;
        # before DSO load a regex breakpoint would not be pending reliably.
        # Match all instantiations, then fail closed unless the *first* actual
        # dot hit is exactly the expected quant and width.  This detects a
        # fixture that unexpectedly takes a row-exact or dequantized route.
        argv += ["-ex", "rbreak mul_mat_qX_K_q8_2_X4_T",
                 "-ex", "continue", "-x", str(script), "-ex", "disable"]
    argv += ["-ex", "continue", "--args", str(binary), quant,
             "FUSED_UP_GATE", str(expected_width)]
    if single_expert:
        argv.append("--single-expert")
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
            return Result("unavailable", "fused IQK debugger timed out"), ""
        with os.fdopen(read_fd, "r", encoding="utf-8") as pipe:
            read_fd = -1
            witness = pipe.read(4096)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return Result("unavailable", f"fused IQK debugger failed: {exc}"), ""
    finally:
        if write_fd >= 0:
            os.close(write_fd)
        if read_fd >= 0:
            os.close(read_fd)
    try:
        records = [json.loads(line) for line in witness.splitlines()]
    except json.JSONDecodeError:
        records = []
    assessed = _assess_fused_hit_records(records, dso,
                                         dot_quant=dot_quant,
                                         width=expected_width)
    if assessed.status != "pass":
        return assessed, ""
    if child.returncode != 0 or "exited normally" not in stdout + stderr:
        return Result("unavailable", "fused graph did not finish after helper hit",
                      (stdout + stderr)[-1000:]), ""
    probe_lines = [line for line in stdout.splitlines()
                   if line.startswith(("AK_CPU_QUANT_REFERENCE_V1 ", "A ", "G ", "O "))]
    return Result("pass", f"{quant} fused helper" +
                  (f" and dot width {expected_width}" if dot_quant else "") +
                  " executed in candidate DSO",
                  str(records)), "\n".join(probe_lines)


def check_fused(build_dir: Path, *, resolved_recipe, source_root: Path) -> Result:
    """Both quantized fused graph cases need exact hits and scalar agreement."""
    if resolved_recipe.backend != "cpu" or \
            dict(resolved_recipe.launch_env).get("GGML_IQK") != "1":
        return Result("unavailable", "fused IQK route requires resolved CPU GGML_IQK=1")
    resolved_recipe.validate_launch(resolved_recipe.template, build_dir,
                                    resolved_recipe.port)
    if not (source_root / SOURCE).is_file():
        return Result("unavailable", "candidate fused IQK source is missing")
    from . import cpu_quant_reference
    result = cpu_quant_reference.check_cpu_quant_suite(
        build_dir, source_root, launch_env=resolved_recipe.launch_env,
        topology_prefix=tuple(resolved_recipe.topology_prefix),
        quants=("Q4_K", "Q5_K"), ops=("FUSED_UP_GATE",),
        require_fused_hit=True)
    reason = ("Q4_K/Q5_K exact fused up-gate graph scalar comparison and trusted "
              "candidate-helper hits passed; native MUL_MAT_ID is a separate "
              "preceding host/op gate, and GLU 0/0 is not claimed"
              if result.status == "pass" else result.reason)
    return Result(result.status, reason, result.detail)


def check_q45_dot(build_dir: Path, *, resolved_recipe, source_root: Path) -> Result:
    """Require direct and fused scalar agreement plus exact Q4/Q5 dot hits."""
    if resolved_recipe.backend != "cpu" or \
            dict(resolved_recipe.launch_env).get("GGML_IQK") != "1":
        return Result("unavailable", "Q4/Q5 dot route requires resolved CPU GGML_IQK=1")
    resolved_recipe.validate_launch(resolved_recipe.template, build_dir,
                                    resolved_recipe.port)
    if not (source_root / "ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp").is_file():
        return Result("unavailable", "candidate Q4/Q5 dot source is missing")
    from . import cpu_quant_reference
    widths = tuple(range(1, 9))
    # A fused product can obscure a wrong gate or up intermediate. Check both
    # public plain and expert matmul outputs independently before the fused
    # consumer; the same template serves all three routes.
    direct = cpu_quant_reference.check_cpu_quant_suite(
        build_dir, source_root, launch_env=resolved_recipe.launch_env,
        topology_prefix=tuple(resolved_recipe.topology_prefix),
        quants=("Q4_K", "Q5_K"), ops=("MUL_MAT", "MUL_MAT_ID"), widths=widths,
        expert_mode="single")
    if direct.status != "pass":
        return Result(direct.status, direct.reason, direct.detail)
    fused = cpu_quant_reference.check_cpu_quant_suite(
        build_dir, source_root, launch_env=resolved_recipe.launch_env,
        topology_prefix=tuple(resolved_recipe.topology_prefix),
        quants=("Q4_K", "Q5_K"), ops=("FUSED_UP_GATE",), widths=widths,
        expert_mode="single", require_fused_hit=True, require_dot_hit=True)
    if fused.status != "pass":
        return Result(fused.status, fused.reason, fused.detail)
    return Result("pass", "Q4_K/Q5_K widths 1-8 plain, expert, and fused scalar cases "
                  "passed with exact dot-specialization hits in candidate DSO",
                  "\n".join((direct.detail, fused.detail)))
