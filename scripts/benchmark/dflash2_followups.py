#!/usr/bin/env python3
"""Governed DF2-5 concurrency and DF2-6 greedy-parity campaign harness.

No action occurs on import or validation.  Execution uses one serialized official
MI210 claim and one inference-call window per fresh cell, seals KFD/VRAM samples,
and delegates prospective throughput/acceptance carriers to ``dflash2_beliefs``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from scripts.benchmark import dflash2_beliefs as beliefs
from scripts.kernel_rnd.autokernel.execution.inference_window import InferenceCallWindow
from scripts.kernel_rnd.autokernel.resource.device_claim import ClaimJournal, acquire_device_claim


RESEARCH = Path(__file__).resolve().parents[2]
SOURCE = Path("/mnt/raid0/llm/llama.cpp-dflash2-qwen38-20260820")
BUILD = SOURCE / "build-hip-dflash2"
BINARY = BUILD / "bin/llama-server"
TARGET = Path("/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf")
DRAFT = Path("/mnt/raid0/llm/models/Qwen3.8-27B-DFlash2-Q8_0.gguf")
QUESTIONS = Path("/workspace/tmp/questions_mtp_ab.json")
RUNNER = RESEARCH / "scripts/benchmark/v7_quality_gate_runner.py"
PARITY = RESEARCH / "scripts/benchmark/dflash2_parity_client.py"
PORT = 18072
ROOTS = {
    "concurrency": Path("/workspace/repos/epyc-inference-research/artifacts/architect-bench-gpu-20260814/dflash2_concurrency_20260820"),
    "parity": Path("/workspace/repos/epyc-inference-research/artifacts/architect-bench-gpu-20260814/dflash2_greedy_parity_20260820"),
}
CELLS = {
    "concurrency": beliefs.ARMS,
    "parity": ("plain_greedy", "dflash2_greedy"),
}
EXPECTED = {
    "source_commit": "2046c64e9948671c7557428b198acebc6f416575",
    "binary_sha256": "d09f65568501192a291c5dd0904b29aa3617e4e20bc271e99f66b5c3472fe0dc",
    "target_model_sha256": "a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348",
    "draft_model_sha256": "7f1c9a31a6ed40044c69f6508b50fd63b87abd8e1fb7fe4290303df549153751",
    "questions_sha256": "2088d2c0bf2c66a4a76d67359d4cfcebdbeb19eefd6161b5a309e94ce6a5476d",
    "runner_sha256": "6dea92dd9e374f79691f5df502fa11035ffd484906754f20190a4189111ae7dc",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def np_for(cell: str) -> int:
    match = re.search(r"_np([248])$", cell)
    return int(match.group(1)) if match else 1


def protocol(campaign: str) -> dict:
    if campaign == "concurrency":
        return {"np": [2, 4, 8], "context": 32768, "threads": 8, "batch": 2048,
                "ubatch": 2048, "kv": "f16/f16", "questions": 12, "max_tokens": 2048,
                "seed": 42, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                "enable_thinking": False, "endpoint": "chat", "repeats": 1,
                "mtp_n_max": 8, "dflash_requested_n_max": 8}
    return {"np": 1, "context": 32768, "threads": 8, "batch": 2048, "ubatch": 2048,
            "kv": "f16/f16", "questions": 12, "max_tokens": 256, "seed": 42,
            "temperature": 0.0, "top_p": 1.0, "top_k": 1, "min_p": 0.0,
            "typical_p": 1.0, "repeat_penalty": 1.0, "cache_prompt": False,
            "endpoint": "completion", "return_tokens": True, "repeats": 1}


def static_route_authority() -> dict:
    mmvq = (SOURCE / "ggml/src/ggml-cuda/mmvq.cu").read_text(encoding="utf-8")
    mmq = (SOURCE / "ggml/src/ggml-cuda/mmq.cu").read_text(encoding="utf-8")
    mmvf = (SOURCE / "ggml/src/ggml-cuda/mmvf.cu").read_text(encoding="utf-8")
    mmf = (SOURCE / "ggml/src/ggml-cuda/mmf.cu").read_text(encoding="utf-8")
    dispatch = (SOURCE / "ggml/src/ggml-cuda/ggml-cuda.cu").read_text(encoding="utf-8")
    speculative = (SOURCE / "common/speculative.cpp").read_text(encoding="utf-8")
    checks = {
        "q8_mmvf_excluded": "GGML_TYPE_Q8_0" not in mmvf,
        "quantized_mmf_excluded": "if (ggml_is_quantized(type)) {\n        return false;" in mmf,
        "q8_mmvq_boundary": re.search(
            r"case\s+GGML_TYPE_Q8_0\s*:(?:(?!case\s+GGML_TYPE_).){0,500}?"
            r"return\s+log_decision\(ne11\s*<=\s*1\)\s*;", mmvq, re.S) is not None,
        "q8_mmq_supported": "case GGML_TYPE_Q8_0:" in mmq,
        "gfx90a_small_batch_mmq": "if (n_experts > 64 || ne11 <= 128)" in mmq,
        "dispatch_order": dispatch.index("ggml_cuda_should_use_mmvq") <
                          dispatch.index("ggml_cuda_should_use_mmq") <
                          dispatch.index('ggml_cuda_log_mul_mat_route("CUBLAS"'),
        "dflash_block_minus_target":
            "const int32_t n_draft_max = is_dspark ? block_size : block_size - 1;" in speculative,
    }
    if not all(checks.values()):
        raise RuntimeError(f"DF2 route authority incomplete: {checks}")
    strings = subprocess.check_output(
        ["strings", str(BUILD / "bin/libggml-hip.so.0.16.0")], text=True, errors="replace")
    if "GGML_CUDA_MUL_MAT_ROUTE" not in strings:
        raise RuntimeError("candidate lacks the bounded live route instrument")
    return {"schema": "epyc.df2.static_route_authority.v1", "status": "expected_route_only",
            "block_verify_shape": {"trained_block_size": 8, "effective_draft_tokens": 7,
                                   "target_verify_columns": 8},
            "q8_dense_route": {"ne11_1": "MMVQ", "ne11_2_through_128_on_gfx90a": "MMQ",
                               "dflash_ne11_8_expected": "MMQ"},
            "checks": checks, "live_route_diagnostic_required": True,
            "limitation": "runtime CUBLAS_PRECHECK cannot be discharged from source alone"}


def validate(*, require_roots_absent: bool) -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=SOURCE, text=True)
    if commit != EXPECTED["source_commit"] or dirty:
        raise RuntimeError("candidate source identity drift")
    identities = {"binary_sha256": sha256(BINARY), "target_model_sha256": sha256(TARGET),
                  "draft_model_sha256": sha256(DRAFT), "questions_sha256": sha256(QUESTIONS),
                  "runner_sha256": sha256(RUNNER)}
    for key, value in identities.items():
        if value != EXPECTED[key]:
            raise RuntimeError(f"{key} drift: {value}")
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    if len(questions) != 12 or {row.get("suite") for row in questions} != {"olympiadbench_hard"}:
        raise RuntimeError("question contract drift")
    if require_roots_absent:
        present = [str(root) for root in ROOTS.values() if root.exists()]
        if present:
            raise RuntimeError(f"future roots are not absent: {present}")
    return {"schema": "epyc.df2.followups_validation.v2", "status": "ready",
            "validated_at": now(), "gpu_action": False, "source_commit": commit,
            **identities, "route_authority": static_route_authority()}


def initialize(campaign: str) -> Path:
    root = ROOTS[campaign]
    if root.exists():
        raise RuntimeError(f"campaign root already exists: {root}")
    check = validate(require_roots_absent=True)
    root.mkdir(parents=True)
    write_json(root / "preflight.json", {
        "schema": "epyc.df2.followups_preflight.v2",
        "campaign_id": f"df2-{'5' if campaign == 'concurrency' else '6'}-qwen38-{campaign}-20260820",
        "campaign_kind": "experimental_runtime", "authority": beliefs.AUTHORITY,
        "created_at": now(), "source_root": str(SOURCE), "source_commit": check["source_commit"],
        "binary": str(BINARY), "binary_sha256": check["binary_sha256"],
        "target_model": str(TARGET), "target_model_sha256": check["target_model_sha256"],
        "draft_model": str(DRAFT), "draft_model_sha256": check["draft_model_sha256"],
        "questions": str(QUESTIONS), "questions_sha256": check["questions_sha256"],
        "runner": str(RUNNER), "runner_sha256": check["runner_sha256"],
        "parity_client": str(PARITY), "parity_client_sha256": sha256(PARITY),
        "protocol": protocol(campaign), "route_authority": check["route_authority"],
    })
    return root


def server_command(cell: str) -> list[str]:
    np = np_for(cell)
    cmd = [str(BINARY), "-m", str(TARGET), "-np", str(np), "-c", "32768",
           "-t", "8", "-tb", "8", "-b", "2048", "-ub", "2048",
           "-ctk", "f16", "-ctv", "f16", "--host", "127.0.0.1", "--port", str(PORT)]
    if cell.startswith("plain"):
        cmd += ["--spec-type", "none"]
    elif cell.startswith("mtp"):
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "8"]
    elif cell.startswith("dflash2"):
        cmd += ["-md", str(DRAFT), "--spec-type", "draft-dflash", "--spec-draft-n-max", "8"]
    else:
        raise RuntimeError(f"unknown cell {cell}")
    return cmd


def runner_command(campaign: str, cell: str, arm_dir: Path) -> list[str]:
    models = str(TARGET) if not cell.startswith("dflash2") else f"{TARGET};{DRAFT}"
    common = ["--host", "127.0.0.1", "--port", str(PORT), "--output", str(arm_dir / "result.json"),
              "--per-question-out", str(arm_dir / "pq.jsonl"),
              "--live-status-out", str(arm_dir / "pq.live-status.json"),
              "--questions-in", str(QUESTIONS), "--arm", cell,
              "--kernel", "dflash2-forward-port-2046c64e", "--binary", str(BINARY),
              "--models", models, "--timeout", "300"]
    if campaign == "parity":
        return [sys.executable, str(PARITY), *common]
    return [sys.executable, str(RUNNER), *common, "--suites", "olympiadbench_hard", "--n", "12",
            "--seed", "42", "--endpoint", "chat", "--max-tokens", "2048",
            "--temperature", "0.6", "--top-p", "0.95", "--top-k", "20",
            "--no-enable-thinking", "--repeats", "1", "--concurrency", str(np_for(cell))]


def healthy() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def sample_resource(server_pid: int, baseline: int) -> dict:
    try:
        vram = int(Path("/sys/class/drm/card2/device/mem_info_vram_used").read_text().strip())
        vram_error = None
    except Exception as exc:
        vram, vram_error = None, f"{type(exc).__name__}: {exc}"
    try:
        pids = sorted(int(path.name) for path in Path("/sys/class/kfd/kfd/proc").iterdir()
                      if path.name.isdigit())
        kfd_error = None
    except Exception as exc:
        pids, kfd_error = [], f"{type(exc).__name__}: {exc}"
    return {"ts": now(), "server_pid": server_pid, "kfd_pids": pids,
            "server_kfd_resident": server_pid in pids, "vram_used_bytes": vram,
            "vram_delta_bytes": None if vram is None else vram - baseline,
            "vram_error": vram_error, "kfd_error": kfd_error}


def run_cell(campaign: str, cell: str, root: Path) -> None:
    arm_dir = root / cell
    if arm_dir.exists():
        raise RuntimeError(f"incomplete/unvalidated cell root exists: {arm_dir}")
    arm_dir.mkdir()
    server_argv = server_command(cell)
    runner_argv = runner_command(campaign, cell, arm_dir)
    write_json(arm_dir / "commands.json", {"server": server_argv, "runner": runner_argv,
                                           "environment": {"LD_LIBRARY_PATH": f"{BUILD / 'bin'}:/opt/rocm/lib"}})
    baseline = int(Path("/sys/class/drm/card2/device/mem_info_vram_used").read_text().strip())
    preflight = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
    claim = acquire_device_claim(
        "mi210_0", purpose=f"{preflight['campaign_id']} {cell}",
        campaign_id=preflight["campaign_id"], journal=ClaimJournal(arm_dir / "claim-journal.jsonl"),
        holder_label=f"df2-{cell}", timeout_s=600, max_hold_s=2400)
    write_json(arm_dir / "claim-open.json", claim.receipt().to_dict())
    window = InferenceCallWindow(timeout_s=600).acquire()
    env = os.environ.copy(); env["LD_LIBRARY_PATH"] = f"{BUILD / 'bin'}:/opt/rocm/lib"
    samples, server, runner = [], None, None
    runner_rc, failure, released = None, None, None
    started = now()
    handles = [(arm_dir / name).open("wb") for name in
               ("server.stdout", "server.stderr", "runner.stdout", "runner.stderr")]
    try:
        if healthy():
            raise RuntimeError(f"port {PORT} is already occupied by a healthy server")
        server = subprocess.Popen(server_argv, stdout=handles[0], stderr=handles[1], env=env,
                                  cwd=SOURCE, start_new_session=True)
        deadline = time.monotonic() + 300
        while not healthy():
            samples.append(sample_resource(server.pid, baseline))
            if server.poll() is not None:
                raise RuntimeError(f"server exited during startup rc={server.returncode}")
            if time.monotonic() >= deadline:
                raise RuntimeError("server health timeout")
            time.sleep(0.25)
        runner = subprocess.Popen(runner_argv, stdout=handles[2], stderr=handles[3], env=env, cwd=RESEARCH)
        write_json(arm_dir / "processes.json", {"holder_pid": os.getpid(), "server_pid": server.pid,
                                                "runner_pid": runner.pid, "started_at": started})
        deadline = time.monotonic() + 2100
        while runner.poll() is None:
            samples.append(sample_resource(server.pid, baseline))
            if server.poll() is not None:
                raise RuntimeError(f"server exited during runner rc={server.returncode}")
            if time.monotonic() >= deadline:
                runner.terminate(); raise RuntimeError("runner exceeded 2100 seconds")
            time.sleep(0.25)
        runner_rc = runner.returncode
        samples.append(sample_resource(server.pid, baseline))
        if runner_rc != 0:
            raise RuntimeError(f"runner exited {runner_rc}")
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if runner is not None and runner.poll() is None:
            runner.terminate()
            try: runner.wait(timeout=10)
            except subprocess.TimeoutExpired: runner.kill(); runner.wait(timeout=10)
        if server is not None and server.poll() is None:
            os.kill(server.pid, signal.SIGTERM)
            try: server.wait(timeout=30)
            except subprocess.TimeoutExpired: os.kill(server.pid, signal.SIGKILL); server.wait(timeout=10)
        for handle in handles: handle.close()
        window.release()
        released = claim.release().to_dict()
        write_json(arm_dir / "claim-released.json", released)
        write_json(arm_dir / "resource-samples.json", {
            "schema": "epyc.df2.resource_samples.v1", "baseline_vram_bytes": baseline,
            "interval_s": 0.25, "samples": samples})
        write_json(arm_dir / "transport.json", {
            "schema": "epyc.df2.arm_transport.v1", "arm": cell, "started_at": started,
            "finished_at": now(), "runner_returncode": runner_rc, "failure": failure,
            "claim_id": claim.claim_id, "claim_released": released is not None,
            "inference_window_released": not window.held,
            "server_pid": None if server is None else server.pid,
            "server_returncode": None if server is None else server.returncode,
            "runner_pid": None if runner is None else runner.pid})
    result = json.loads((arm_dir / "result.json").read_text(encoding="utf-8"))
    throughput = result["suites"][0]["throughput"]
    stderr = (arm_dir / "server.stderr").read_text(encoding="utf-8")
    accept = [line for line in stderr.splitlines() if "draft acceptance =" in line]
    (arm_dir / "acceptance.txt").write_text("".join(line + "\n" for line in accept), encoding="utf-8")
    resident = [row for row in samples if row["server_kfd_resident"]]
    positive = [row for row in resident if isinstance(row["vram_delta_bytes"], int) and row["vram_delta_bytes"] > 0]
    if len(resident) < 2 or len(positive) < 2:
        raise RuntimeError("cell lacks two in-window KFD+VRAM samples")
    write_json(arm_dir / "summary.json", {
        "schema": "epyc.df2.followup_arm.v2", "arm": cell,
        "aggregate_decode_tok_s": throughput["aggregate_decode_tok_s"],
        "aggregate_total_tok_s": throughput["aggregate_total_tok_s"],
        "completion_tokens": throughput["completion_tokens"], "wall_s": throughput["wall_s"],
        "acceptance_lines": len(accept), "resource_sample_count": len(samples),
        "kfd_resident_samples": len(resident), "positive_vram_samples": len(positive),
        "peak_vram_used_bytes": max(row["vram_used_bytes"] for row in positive),
        "claim_id": claim.claim_id, "claim_released": True})


def compare_parity(root: Path) -> dict:
    def rows(cell):
        return [json.loads(line) for line in (root / cell / "pq.jsonl").read_text().splitlines()]
    plain, draft = rows("plain_greedy"), rows("dflash2_greedy")
    if len(plain) != 12 or len(draft) != 12:
        raise RuntimeError("parity row count mismatch")
    comparisons = []
    for left, right in zip(plain, draft):
        same = (left["id"], left["prompt_fingerprint"], left["request_sha256"]) == (
            right["id"], right["prompt_fingerprint"], right["request_sha256"])
        comparisons.append({"id": left["id"], "same_identity": same,
                            "token_parity": left["tokens"] == right["tokens"],
                            "content_parity": left["content"] == right["content"],
                            "plain_tokens_sha256": left["tokens_sha256"],
                            "dflash2_tokens_sha256": right["tokens_sha256"]})
    passed = all(row["same_identity"] and row["token_parity"] and row["content_parity"]
                 for row in comparisons)
    out = {"schema": "epyc.df2.greedy_parity.v1", "status": "pass" if passed else "fail",
           "exact_token_parity": passed, "belief_measurements": [],
           "belief_reason": "categorical parity is not coerced into a metric", "comparisons": comparisons}
    write_json(root / "parity-summary.json", out)
    if not passed:
        raise RuntimeError("DFlash2 exact greedy token parity failed")
    return out


def execute(campaign: str) -> None:
    root = ROOTS[campaign]
    if not (root / "preflight.json").is_file():
        raise RuntimeError("initialize campaign exactly once before execute")
    preflight = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
    for cell in CELLS[campaign]:
        if (root / cell / "summary.json").is_file():
            if campaign == "concurrency":
                beliefs._verify_arm(root, cell, preflight)
            continue
        if (root / cell).exists():
            raise RuntimeError(f"incomplete cell retained; use a fresh campaign attempt: {root / cell}")
        run_cell(campaign, cell, root)
    if campaign == "concurrency":
        beliefs.finalize_concurrency(root)
    else:
        compare_parity(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--init", choices=("concurrency", "parity"))
    parser.add_argument("--execute", choices=("concurrency", "parity"))
    args = parser.parse_args()
    if sum((args.validate_only, bool(args.init), bool(args.execute))) != 1:
        parser.error("choose exactly one mode")
    if args.validate_only:
        print(json.dumps(validate(require_roots_absent=True), sort_keys=True))
    elif args.init:
        print(json.dumps({"initialized": str(initialize(args.init)), "at": now()}, sort_keys=True))
    else:
        execute(args.execute)
