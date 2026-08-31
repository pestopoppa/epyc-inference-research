#!/usr/bin/env python3
"""DF2-5 concurrency grid — the G2 runner for `epyc.g2_df25_draft_grid.v1`.

The reader for this schema already exists (`scripts/vidya/adapters/research_sweeps.py`,
whose docstring states "G2-G4 runners do not exist yet"). This is that runner.

WHY A NEW RUNNER AND NOT `dflash2_followups.py`
-----------------------------------------------
`dflash2_followups.py` seals DF2-4's protocol in `dflash2_beliefs.EXPECTED_PROTOCOL`
and `REQUIRED_ARM_FILES`. DF2-5's REVISED design (handoff 2026-08-21, intake-1277)
needs four things that contract forbids or omits: a `none` arm at every concurrency,
paired `--kv-unified`, per-SLOT acceptance rather than aggregate, and
`GGML_CUDA_LOG_MMVQ_ROUTE=1` capture. Extending the sealed module would invalidate
DF2-4's receipts. This runner is additive and leaves them intact.

THE FIVE REVISED DESIGN RULES, AND WHERE EACH IS ENFORCED
---------------------------------------------------------
1. Sweep IN-FLIGHT REQUESTS, not `-np`. `-np` is held at the request count so it is
   never the binding constraint (upstream #27117 shows a `-np 16` server carrying 4
   concurrent requests is bit-identical to `-np 4`, so a `-np` sweep sits entirely in
   the healthy region).                                    -> `cell_server_argv`
2. Per-slot acceptance and mean accepted length. The `id N` field is present in every
   `slot print_timing` line; every existing parser drops it.  -> `SLOT_TIMING_RE`
3. Three arms at every point: none / MTP / DFlash2. Without `none` a regression cannot
   be attributed to speculation at all; without MTP it cannot be attributed to DFlash.
   #27117 is DFlash-1 and predates PR #27342 by three days.   -> `ARMS`
4. `--spec-draft-n-max` held FIXED across the sweep, because `accepted/generated` is
   structurally n_max-dependent.                              -> `N_MAX`
5. Every cell twice, with and without `--kv-unified` -- the single discriminating
   control nobody upstream has run on any backend for any drafter. -> `KV_UNIFIED`

MTP TAKES NO `-md`: the head is in-file (`nextn`) in the Q8_0 target; the
`mtp-*.gguf` sidecar is redundant for this base (model_registry.yaml:2500-2503).

GPU RESIDENCY IS PROVEN, NEVER ASSUMED. `llama.cpp` *dlopens* `libggml-hip.so`, so a
binary shows zero HIP linkage either way and `ldd` cannot prove a HIP run. A cell whose
VRAM never rises is refused rather than recorded -- a CPU-only fallback silently
produces plausible-looking numbers roughly 20x too slow.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

G2_SCHEMA = "epyc.g2_df25_draft_grid.v1"

TARGET = Path("/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf")
DFLASH2 = Path("/mnt/raid0/llm/models/Qwen3.8-27B-DFlash2-Q8_0.gguf")
QUESTIONS = Path("/workspace/tmp/questions_mtp_ab.json")
RUNNER = Path("/mnt/raid0/llm/epyc-inference-research/scripts/benchmark/v7_quality_gate_runner.py")

#: Rule 4 -- fixed across every cell. DFlash2's server clamps an n_max of 8 to
#: block_size-1 = 7 (the target token occupies one block position); MTP uses all 8.
#: That asymmetry is inherent to the drafters, not a protocol choice, and is why
#: `mean_accepted_length` is reported beside every acceptance ratio.
N_MAX = 8

#: Rule 1 -- in-flight request counts. Onset of the #27117 phenomenon is reported at
#: 8 concurrent and 4 is healthy in every report, so a sweep stopping at 4 cannot see
#: it. Our production role runs np up to 8.
CONCURRENCIES = (1, 2, 4, 8)

#: Rule 3 -- `none` is not optional. It is the attribution arm.
ARMS = ("none", "mtp", "dflash2")

#: Rule 5 -- the paired control.
KV_UNIFIED = (False, True)

#: Per-slot context held constant at 4096 so total KV scales with concurrency the way
#: production does. Varying per-slot context across a concurrency sweep would confound
#: cache pressure with scheduling.
CTX_PER_SLOT = 4096

PORT = 18099
HOST = "127.0.0.1"

#: Rule 2 -- the `id (\d+)` group is the whole point; every existing parser drops it.
SLOT_TIMING_RE = re.compile(
    r"slot print_timing:\s*id\s+(\d+)\s*\|\s*task\s+(\d+)\s*\|\s*"
    r"draft acceptance\s*=\s*([0-9.]+)\s*\(\s*(\d+) accepted /\s*(\d+) generated\),\s*"
    r"mean len\s*=\s*([0-9.]+)"
)

VRAM_SYSFS = Path("/sys/class/drm/card2/device/mem_info_vram_used")
#: A loaded 27B Q8_0 sits near 36.4 GiB; anything under ~8 GiB means the weights are
#: not resident and the cell ran on CPU.
VRAM_RESIDENT_FLOOR = 8 * 1024**3


class CellRefused(RuntimeError):
    """A cell could not be measured under conditions that make it interpretable."""


def read_vram() -> int:
    try:
        return int(VRAM_SYSFS.read_text().strip())
    except (OSError, ValueError):
        return -1


def cell_server_argv(build_bin: Path, arm: str, conc: int, kvu: bool,
                     pin_host_cores: str | None = None) -> list[str]:
    # Optional because the 2026-08-27 originals ran UNPINNED (both arms equally, so
    # their comparison is internally consistent). The standing GPU recipe pins
    # llama-server host threads to the codified list (`evaluator/recipes.py:
    # gpu_host_cpu_list()`, sourced from architect_bench_gpu_lib.sh -- 184-191,
    # node-3 SMT siblings, NOT 88-95). A refresh passes it; deltas against the
    # unpinned 2026-08-27 absolute numbers are then cross-protocol and only
    # within-bundle comparisons are claim-grade.
    argv = ["taskset", "-c", pin_host_cores] if pin_host_cores else []
    argv += [
        str(build_bin / "llama-server"),
        "-m", str(TARGET),
        # Rule 1: -np is pinned TO the in-flight count, never above it, so the sweep
        # actually varies what upstream showed to be the only load-bearing quantity.
        "-np", str(conc),
        "-c", str(CTX_PER_SLOT * conc),
        "-t", "8", "-tb", "8", "-b", "2048", "-ub", "2048",
        "-ctk", "f16", "-ctv", "f16",
        "--device", "ROCm0", "-ngl", "99", "-fa", "on",
        "--host", HOST, "--port", str(PORT),
        "--metrics", "--slots",
    ]
    if arm == "none":
        argv += ["--spec-type", "none"]
    elif arm == "mtp":
        # No -md: the MTP head is in-file for this base.
        argv += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(N_MAX)]
    elif arm == "dflash2":
        argv += ["-md", str(DFLASH2), "-ngld", "99",
                 "--spec-type", "draft-dflash", "--spec-draft-n-max", str(N_MAX)]
    else:
        raise CellRefused(f"unknown arm {arm!r}")
    # Rule 5.
    argv.append("--kv-unified" if kvu else "--no-kv-unified")
    return argv


def wait_for_ready(proc: subprocess.Popen, log: Path, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise CellRefused(f"server exited early rc={proc.returncode}; see {log}")
        try:
            with urllib.request.urlopen(f"http://{HOST}:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(3)
    raise CellRefused(f"server not ready within {timeout_s}s; see {log}")


def stop_server(proc: subprocess.Popen) -> None:
    """Terminate ONLY the pid we spawned, then verify it is actually gone.

    Never a pattern kill: this is a shared host and any name pattern is a wildcard
    over other sessions' processes.
    """
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=60)


def run_cell(build_bin: Path, out_root: Path, arm: str, conc: int, kvu: bool,
             run_id: str, max_tokens: int, n_questions: int,
             pin_host_cores: str | None = None) -> dict:
    cell = f"{arm}_c{conc}_kvu{int(kvu)}"
    cell_dir = out_root / cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    log = cell_dir / "server.stderr"
    argv = cell_server_argv(build_bin, arm, conc, kvu, pin_host_cores)
    (cell_dir / "server_command.txt").write_text(" ".join(argv) + "\n", encoding="utf-8")

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{build_bin}:/opt/rocm/lib"
    # Runtime-only instrument (ggml-cuda.cu:1812-1814); needs no rebuild. Required by
    # DF2-6 and free to capture here: our local a6b4b5263 deliberately routes Q8_0 to a
    # different kernel at ne11>=2, so knowing which kernel each verify batch took is
    # the difference between attributing a result to DFlash2 and to our own patch.
    env["GGML_CUDA_LOG_MMVQ_ROUTE"] = "1"
    env.pop("HSA_OVERRIDE_GFX_VERSION", None)

    started = time.time()
    with log.open("wb") as errf:
        proc = subprocess.Popen(argv, stdout=errf, stderr=subprocess.STDOUT,
                                env=env, cwd=str(build_bin.parent.parent))
        try:
            wait_for_ready(proc, log, timeout_s=900)
            vram_loaded = read_vram()
            if vram_loaded < VRAM_RESIDENT_FLOOR:
                raise CellRefused(
                    f"{cell}: VRAM {vram_loaded} below residency floor "
                    f"{VRAM_RESIDENT_FLOOR} -- weights are not GPU-resident, refusing "
                    "to record a CPU-fallback number as a GPU measurement")

            result_json = cell_dir / "result.json"
            run_argv = [
                sys.executable, str(RUNNER),
                "--host", HOST, "--port", str(PORT),
                "--output", str(result_json),
                "--suites", "olympiadbench_hard", "--n", str(n_questions),
                "--seed", "42", "--endpoint", "chat", "--max-tokens", str(max_tokens),
                "--kernel", f"champion-5c278648a-{arm}",
                "--binary", str(build_bin / "llama-server"),
                "--models", str(TARGET) + ("" if arm != "dflash2" else f";{DFLASH2}"),
                "--timeout", "600", "--temperature", "0.6",
                "--top-p", "0.95", "--top-k", "20", "--no-enable-thinking",
                "--repeats", "1",
                "--per-question-out", str(cell_dir / "pq.jsonl"),
                "--live-status-out", str(cell_dir / "pq.live-status.json"),
                "--questions-in", str(QUESTIONS),
                "--arm", cell,
                "--concurrency", str(conc),
            ]
            (cell_dir / "runner_command.txt").write_text(" ".join(run_argv) + "\n",
                                                         encoding="utf-8")
            vram_mid = read_vram()
            rc = subprocess.run(run_argv, capture_output=True, text=True, timeout=14400)
            (cell_dir / "runner.stdout").write_text(rc.stdout or "", encoding="utf-8")
            (cell_dir / "runner.stderr").write_text(rc.stderr or "", encoding="utf-8")
            if rc.returncode != 0:
                raise CellRefused(f"{cell}: runner rc={rc.returncode}")
        finally:
            stop_server(proc)

    throughput = None
    if (cell_dir / "result.json").exists():
        data = json.loads((cell_dir / "result.json").read_text())
        suites = data.get("suites") or []
        if suites:
            throughput = (suites[0].get("throughput") or {}).get("aggregate_decode_tok_s")

    text = log.read_text(errors="replace")
    slots = [
        {"slot_index": int(m.group(1)), "task": int(m.group(2)),
         "fraction": float(m.group(3)), "accepted_n": int(m.group(4)),
         "generated_n": int(m.group(5)), "mean_accepted_length": float(m.group(6))}
        for m in SLOT_TIMING_RE.finditer(text)
    ]
    mmvq_routes = len(re.findall(r"GGML_CUDA_MUL_MAT_ROUTE|mul_mat_q|mul_mat_vec_q", text))

    return {
        "cell": cell, "arm": arm, "concurrency": conc, "kv_unified": kvu,
        "n_max": N_MAX, "aggregate_decode_tok_s": throughput,
        "slot_rows": slots, "distinct_slots": sorted({s["slot_index"] for s in slots}),
        "vram_loaded_bytes": vram_loaded, "vram_mid_bytes": vram_mid,
        "mmvq_route_log_lines": mmvq_routes,
        "wall_s": round(time.time() - started, 1),
        "run_id": run_id,
    }


def g2_rows(cell: dict, run_id: str, ts: str) -> list[dict]:
    """Project a cell into native G2 rows.

    The `none` arm yields NO rows by design: it drafts nothing, so there is no
    per-slot acceptance to report. Its value is the attribution baseline in the cell
    summary. Emitting zero rows is correct -- a row invented here would claim a
    warrant the run never captured.
    """
    if cell["arm"] == "none":
        return []
    drafter = "draft-mtp" if cell["arm"] == "mtp" else "draft-dflash"
    return [{
        "schema": G2_SCHEMA,
        "run_id": run_id,
        "trial_ts_utc": ts,
        "n_max": cell["n_max"],
        "slot_index": s["slot_index"],
        "drafter_arm": drafter,
        "kv_unified": cell["kv_unified"],
        "accepted": s["accepted_n"] > 0,
        "mean_accepted_length": s["mean_accepted_length"],
        "acceptance_fraction": s["fraction"],
        "accepted_n": s["accepted_n"],
        "generated_n": s["generated_n"],
        "concurrency": cell["concurrency"],
        "aggregate_decode_tok_s": cell["aggregate_decode_tok_s"],
    } for s in cell["slot_rows"]]


def main() -> int:
    ap = argparse.ArgumentParser(description="DF2-5 concurrency grid (G2 runner)")
    ap.add_argument("--build-bin", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--n-questions", type=int, default=12)
    ap.add_argument("--only-arm", action="append", default=None)
    ap.add_argument("--only-conc", action="append", type=int, default=None)
    ap.add_argument("--only-kvu", type=int, choices=(0, 1), default=None,
                    help="restrict rule 5's paired control to one side (the "
                         "operator-gate bundle consumes only kvu=0; a minimal "
                         "refresh may skip kvu=1 and say so)")
    ap.add_argument("--pin-host-cores", default=None,
                    help="taskset the server to this cpu list (pass the codified "
                         "GPU host-thread list; the 2026-08-27 originals ran "
                         "unpinned)")
    args = ap.parse_args()

    for path in (TARGET, DFLASH2, QUESTIONS, RUNNER):
        if not path.exists():
            print(f"REFUSED: missing {path}", file=sys.stderr)
            return 2
    if not (args.build_bin / "llama-server").is_file():
        print(f"REFUSED: no llama-server under {args.build_bin}", file=sys.stderr)
        return 2

    arms = tuple(args.only_arm) if args.only_arm else ARMS
    concs = tuple(args.only_conc) if args.only_conc else CONCURRENCIES
    kvus = (KV_UNIFIED if args.only_kvu is None else (bool(args.only_kvu),))

    args.out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cells, rows, refusals = [], [], []

    for kvu in kvus:
        for arm in arms:
            for conc in concs:
                label = f"{arm}_c{conc}_kvu{int(kvu)}"
                print(f"[{time.strftime('%H:%M:%S')}] cell {label}", flush=True)
                try:
                    cell = run_cell(args.build_bin, args.out, arm, conc, kvu,
                                    args.run_id, args.max_tokens, args.n_questions,
                                    args.pin_host_cores)
                except CellRefused as exc:
                    print(f"  REFUSED: {exc}", flush=True)
                    refusals.append({"cell": label, "reason": str(exc)})
                    continue
                cells.append(cell)
                rows.extend(g2_rows(cell, args.run_id, ts))
                print(f"  {cell['aggregate_decode_tok_s']} tok/s  "
                      f"slots={cell['distinct_slots']}  wall={cell['wall_s']}s",
                      flush=True)
                (args.out / "cells.json").write_text(
                    json.dumps(cells, indent=2), encoding="utf-8")
                with (args.out / "g2_rows.jsonl").open("w", encoding="utf-8") as fh:
                    for row in rows:
                        fh.write(json.dumps(row, sort_keys=True) + "\n")
                (args.out / "refusals.json").write_text(
                    json.dumps(refusals, indent=2), encoding="utf-8")

    print(f"\ncells={len(cells)} g2_rows={len(rows)} refused={len(refusals)}")
    return 0 if cells and not refusals else (1 if refusals else 0)


if __name__ == "__main__":
    raise SystemExit(main())
