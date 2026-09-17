#!/usr/bin/env python3
"""INF-61 -- re-collect Qwen3.8-27B's 24-cell np x depth grid at MTP `--spec-draft-n-max 8`.

Handoff: `handoffs/active/gpu-candidates-surface-qwen38-update.md` ("Re-collect the 24-cell
np x depth grid at n-max 8"). The withdrawn 2026-08-15 grid
(`artifacts/np_context_study_v8_20260727/q38/`, untracked) was captured at MTP n-max 4 on v9.

WHAT THIS REPRODUCES. `artifacts/np_context_study_v8_20260727/driver/run_model_block.sh::cell()`
-- the codified np x context cell -- with the one intended change (`--spec-draft-n-max 8`).
That driver cannot run as-is on v9: it hard-asserts the v8 head/binary digest and a v8 cgroup
sidecar. The cell semantics are carried over verbatim:
  * server: frozen production v9 HIP binary (RUN only, never built), `GGML_IQK=1`,
    `LD_LIBRARY_PATH=<build-hip>/bin`, `taskset -c 184-191`, `--jinja -ngl all -fa on
    -np NP -c NP*L -t 8 -tb 8 -b 2048 -ub 2048 -ctk f16 -ctv f16 --reasoning off
    --spec-type draft-mtp --spec-draft-n-max 8`; one fresh server per cell;
  * client: `v7_quality_gate_runner.py --suites olympiadbench_hard --n 155 --limit NP
    --seed 42 --max-tokens L --concurrency NP --temperature 0.6 --top-p 0.95 --top-k 20
    --no-enable-thinking --endpoint chat --questions-in <pinned olympiadbench_hard>`;
  * capacity: a startup OOM signature, n_ctx_slot < L, or VRAM > 61 GiB => SKIP (recorded).
  * metric: the runner's `throughput.aggregate_decode_tok_s` (completion tokens / suite wall).
Deltas, stated: host-thread containment is `taskset` + a per-thread affinity check rather
than the v8 cgroup; GPU residency is SAMPLED across each launch (autokernel
`residency.Sampler`) and a cell read non-resident is refused; the GPU claim is held.

n: `--passes` launches per cell (pass order alternates ascending/descending) so each cell
carries a between-launch spread, unit = launch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts/kernel_rnd"))
from autokernel.loop import claim, residency  # noqa: E402

BUILD = Path("/mnt/raid0/llm/llama.cpp/build-hip/bin")
MODEL = Path("/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf")
PIN = Path("/mnt/raid0/llm/epyc-inference-research/artifacts/architect-bench-gpu-20260720/questions_olympiadbench_hard.json")
RUNNER = REPO / "scripts/benchmark/v7_quality_gate_runner.py"
PY = Path("/mnt/raid0/llm/epyc-inference-research/.venv/bin/python")
CORES = "184-191"
PORT = 18072
NMAX = 8
NPS = (1, 2, 4, 8, 16, 32)
LENGTHS = (2048, 8192, 16384, 32768)
VRAM_SKIP_GIB = 61
OOM_RE = re.compile(r"hipErrorOutOfMemory|failed to allocate .*\b(HIP|ROCm|VRAM|GPU|device|KV|buffer)\b"
                    r"|\b(HIP|ROCm)\b.*\b(out of memory|memory allocation)\b", re.I)


def cpu_list_set(text: str) -> set[int]:
    out = set()
    for part in text.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def affinity_check(pid: int) -> dict:
    rows = []
    for task in sorted(Path(f"/proc/{pid}/task").iterdir()):
        status = (task / "status").read_text()
        aff = next(l.split(":", 1)[1].strip() for l in status.splitlines() if l.startswith("Cpus_allowed_list:"))
        rows.append(aff)
    ok = bool(rows) and all(cpu_list_set(a) == cpu_list_set(CORES) for a in rows)
    return {"threads": len(rows), "all_pinned": ok}


def stop(proc: subprocess.Popen) -> str:
    if proc.poll() is not None:
        return "exited"
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
        return "terminated"
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)
        return "killed"


def run_cell(out: Path, np_: int, L: int, pas: int, pin: Path = PIN) -> dict:
    d = out / f"np{np_}_L{L}" / f"pass{pas}"
    d.mkdir(parents=True, exist_ok=True)
    argv = ["taskset", "-c", CORES, str(BUILD / "llama-server"), "-m", str(MODEL),
            "--host", "127.0.0.1", "--port", str(PORT), "--metrics", "--slots", "--jinja",
            "--device", "ROCm0", "-ngl", "all", "-fa", "on", "-np", str(np_), "-c", str(np_ * L),
            "-t", "8", "-tb", "8", "-b", "2048", "-ub", "2048", "-ctk", "f16", "-ctv", "f16",
            "--reasoning", "off", "--spec-type", "draft-mtp", "--spec-draft-n-max", str(NMAX)]
    env = dict(os.environ)
    env.pop("HSA_OVERRIDE_GFX_VERSION", None)
    env.pop("GGML_NOHUGEPAGE_PROCESS", None)
    env.update(GGML_IQK="1", LD_LIBRARY_PATH=str(BUILD))
    (d / "server.argv").write_text(" ".join(argv) + "\n")
    row = {"np": np_, "L": L, "ctx": np_ * L, "pass": pas, "argv": argv, "dir": str(d),
           "started_at": time.time()}
    sampler = residency.Sampler(interval=0.5)
    request_start = request_end = None
    with sampler, (d / "server.stderr").open("wb") as err:
        proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=err, env=env)
        row["server_pid"] = proc.pid
        try:
            deadline = time.time() + 600
            healthy = False
            while time.time() < deadline:
                if proc.poll() is not None:
                    break
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                        if r.status == 200:
                            healthy = True
                            break
                except Exception:
                    pass
                time.sleep(3)
            text = (d / "server.stderr").read_text(errors="replace")
            if not healthy:
                m = OOM_RE.search(text)
                row.update(status="skip" if m else "failed",
                           reason=f"capacity_start:{m.group(0)[:80]}" if m else "server not healthy")
                return row
            row["affinity"] = affinity_check(proc.pid)
            if not row["affinity"]["all_pinned"]:
                row.update(status="failed", reason="host threads not pinned to " + CORES)
                return row
            slot = re.findall(r"(?:n_ctx_per_seq|n_ctx_slot)\s*=\s*(\d+)", text)
            nslot = int(slot[-1]) if slot else 0
            vram_gib = residency.vram_bytes() / 2**30
            row.update(n_ctx_slot=nslot, vram_gib_after_load=round(vram_gib, 2))
            if nslot < L or vram_gib > VRAM_SKIP_GIB:
                row.update(status="skip", reason=f"n_ctx_slot={nslot} vram={vram_gib:.1f}G")
                return row
            cmd = [str(PY), str(RUNNER), "--host", "127.0.0.1", "--port", str(PORT),
                   "--output", str(d / "results.json"), "--suites", "olympiadbench_hard",
                   "--n", "155", "--limit", str(np_), "--seed", "42", "--max-tokens", str(L),
                   "--repeats", "1", "--concurrency", str(np_), "--temperature", "0.6",
                   "--top-p", "0.95", "--top-k", "20", "--no-enable-thinking", "--endpoint", "chat",
                   "--kernel", "production-consolidated-v9",
                   "--arm", f"q38_mtp8_np{np_}_L{L}_p{pas}", "--binary", str(BUILD / "llama-server"),
                   "--models", str(MODEL), "--questions-in", str(pin),
                   "--per-question-out", str(d / "per_question.jsonl")]
            renv = dict(os.environ, HF_HOME="/mnt/raid0/llm/cache/huggingface",
                        RUNNER_REQUEST_TIMEOUT_S="5400")
            request_start = time.time()
            with (d / "runner.stdout").open("wb") as o, (d / "runner.stderr").open("wb") as e:
                rc = subprocess.run(cmd, stdout=o, stderr=e, env=renv, cwd=str(REPO)).returncode
            request_end = time.time()
            row["runner_rc"] = rc
            if rc != 0 or not (d / "results.json").exists():
                row.update(status="failed", reason=f"runner rc={rc}")
                return row
            res = json.loads((d / "results.json").read_text())
            suite = res["suites"][0]
            row.update(status="ok", throughput=suite["throughput"],
                       aggregate_decode_tok_s=suite["throughput"]["aggregate_decode_tok_s"],
                       accuracy=suite["accuracy"], n_questions=suite["n"],
                       truncated=suite.get("truncated"), errors=suite.get("errors"),
                       runner_source_sha256=res["meta"].get("runner_source_sha256"))
            acc = re.findall(r"draft acceptance = ([0-9.]+) \(\s*(\d+) accepted / \s*(\d+) generated\), mean len =\s*([0-9.]+)",
                             (d / "server.stderr").read_text(errors="replace"))
            row["mtp_acceptance"] = {"slots_reported": len(acc),
                                     "accepted": sum(int(a[1]) for a in acc),
                                     "generated": sum(int(a[2]) for a in acc)}
        finally:
            row["teardown"] = stop(proc)
    proof = sampler.proof
    covers = (request_start is not None and request_end is not None)
    row["residency"] = {**proof, "request_start": request_start, "request_end": request_end,
                        "status": "proven" if (proof["resident"] and covers and proof["vram_reads"]) else
                        ("unproven" if row.get("status") != "ok" else "NOT_RESIDENT")}
    if row.get("status") == "ok" and row["residency"]["status"] != "proven":
        row["status"] = "refused_not_resident"
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--questions-in", type=Path, default=PIN,
                    help="pinned olympiadbench_hard question file (untracked artifact; sha256 is recorded)")
    args = ap.parse_args()
    if not args.questions_in.is_file():
        print(f"refusing: pinned question file missing: {args.questions_in}", file=sys.stderr)
        return 2
    pin_sha = hashlib.sha256(args.questions_in.read_bytes()).hexdigest()
    cells = [(np_, L) for L in LENGTHS for np_ in NPS]
    plan = []
    for p in range(args.passes):
        plan += [(np_, L, p) for np_, L in (cells if p % 2 == 0 else list(reversed(cells)))]
    args.out.mkdir(parents=True, exist_ok=True)
    rows_path = args.out / "cells.jsonl"
    done = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            if r["status"] in ("ok", "skip"):
                done.add((r["np"], r["L"], r["pass"]))
    skip_cells = set()  # a capacity skip is a property of the cell; do not relaunch it
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            if r["status"] == "skip":
                skip_cells.add((r["np"], r["L"]))
    sha = hashlib.sha256((BUILD / "llama-server").read_bytes()).hexdigest()
    print(f"plan {len(plan)} launches ({len(done)} done); v9 llama-server sha256 {sha[:16]}", flush=True)
    if not args.execute:
        return 0
    (args.out / "meta.json").write_text(json.dumps({
        "schema": "epyc.inf61.q38_np_depth_grid.v1", "binary": str(BUILD / "llama-server"),
        "llama_server_sha256": sha, "kernel": "production-consolidated-v9 (0db32c06e, 10125)",
        "model": str(MODEL), "spec": f"draft-mtp n-max {NMAX}", "nps": NPS, "lengths": LENGTHS,
        "passes": args.passes, "unit": "launch", "questions": str(args.questions_in),
        "questions_sha256": pin_sha,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
    with claim.hold() as receipt:
        print(f"GPU claim held: {dict(receipt)}", flush=True)
        for np_, L, p in plan:
            if (np_, L, p) in done:
                continue
            if (np_, L) in skip_cells:
                row = {"np": np_, "L": L, "pass": p, "status": "skip", "reason": "capacity skip in an earlier pass"}
            else:
                t0 = time.time()
                try:
                    row = run_cell(args.out, np_, L, p, args.questions_in)
                except Exception as exc:
                    row = {"np": np_, "L": L, "pass": p, "status": "failed",
                           "reason": f"{type(exc).__name__}: {exc}"[:400]}
                row["seconds"] = round(time.time() - t0, 1)
                if row["status"] == "skip":
                    skip_cells.add((np_, L))
            with rows_path.open("a") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
            print(f"np{np_:<2} L{L:<5} pass{p} {row['status']:8s} "
                  f"{row.get('aggregate_decode_tok_s', row.get('reason', ''))} "
                  f"res={row.get('residency', {}).get('status')} [{row.get('seconds')}s]", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
