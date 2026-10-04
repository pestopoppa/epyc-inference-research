#!/usr/bin/env python3
"""G5 capacity dimension: peak footprint over a MIXED request sequence.

A steady single-shape measurement cannot see allocation that grows per shape change.
v10's AutoKernel keep `mmvq_q8_1_graph_cache` (GPU-POOL-1, stack owner 2026-10-04)
held shared-pool buffers across HIP-graph captures and grew :8083 by ~+0.29 GiB per
`speculative.n_max: 0` alternation event (51.69 -> 59.77 GiB, KVU-16h) while every
steady decode A/B looked clean. So the capacity dimension launches the candidate once
at the recipe's own context and slots and drives a mixed sequence per cycle:

    decode with drafting disabled (`speculative.n_max: 0`)   -- batch-1 graph shape
    decode with the recipe's drafting                          -- verify-batch shape
    long-prompt prefill (prompt repeated past the ubatch)      -- large-batch buffers
    short decode with drafting disabled again                  -- alternation back

sampling the launched PID's own footprint (KFD VRAM on GPU, VmHWM on CPU) with the
residency sampler, and records the high-water mark after every cycle. A keep fails
capacity when the peak exceeds the ceiling OR the footprint keeps growing after the
first (warm-up) cycle by more than `GROWTH_TOLERANCE_BYTES`.

Process discipline as everywhere in the loop: only the PID started here is signalled.
"""
from __future__ import annotations

import json
import signal
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping, Sequence

SCHEMA = "epyc.autokernel.capacity_probe.v1"
SEQUENCE = "mixed:nmax0-alternation+varying-batch"
DEFAULT_CYCLES = 4
#: Growth allowed from the end of cycle 1 (allocator warm-up) to the end of the last
#: cycle. One mmvq_q8_1_graph_cache alternation event alone is ~0.29 GiB.
GROWTH_TOLERANCE_BYTES = 128 << 20
LONG_PROMPT_REPEAT = 12


def _with(body: bytes, **fields: Any) -> bytes:
    row = json.loads(body)
    row.update(fields)
    return json.dumps(row, sort_keys=True, separators=(",", ":")).encode()


def _long(body: bytes, repeat: int) -> bytes:
    row = json.loads(body)
    prompt = row.get("prompt")
    if isinstance(prompt, str):
        row["prompt"] = "\n\n".join([prompt] * repeat)
    elif isinstance(prompt, list):
        row["prompt"] = list(prompt) * repeat
    row.update(n_predict=1, cache_prompt=False)
    return json.dumps(row, sort_keys=True, separators=(",", ":")).encode()


def mixed_sequence(frozen_requests: Sequence[tuple[str, bytes]], *, cycles: int = DEFAULT_CYCLES,
                   long_repeat: int = LONG_PROMPT_REPEAT) -> list[list[tuple[str, bytes]]]:
    """Per cycle, the (label, body) steps of the mixed sequence (pure)."""
    if not frozen_requests:
        raise ValueError("capacity probe needs at least one frozen request")
    if cycles < 3:
        raise ValueError("capacity probe needs >= 3 cycles to separate warm-up from growth")
    body = frozen_requests[0][1]
    steps = [("decode_nmax0", _with(body, **{"speculative.n_max": 0})),
             ("decode_draft", body),
             ("prefill_long", _long(body, long_repeat)),
             ("decode_nmax0_short", _with(body, **{"speculative.n_max": 0, "n_predict": 16}))]
    return [list(steps) for _ in range(cycles)]


def evaluate(cycle_peaks: Sequence[int], *, limit_bytes: int | None, backend: str,
             ctx: int | None, np: int | None,
             tolerance_bytes: int = GROWTH_TOLERANCE_BYTES) -> dict:
    """The capacity record `surface_validation.keep_dimensions` grades (pure)."""
    peaks = [int(p) for p in cycle_peaks]
    peak = max(peaks) if peaks else None
    growth = (peaks[-1] - peaks[0]) if len(peaks) >= 2 else None
    return {"schema": SCHEMA, "sequence": SEQUENCE, "backend": backend,
            "peak_bytes": peak or None, "limit_bytes": limit_bytes,
            "cycle_peaks_bytes": peaks, "growth_bytes": growth,
            "growth_tolerance_bytes": tolerance_bytes, "ctx": ctx, "np": np,
            "source": "own_pid_kfd_vram" if backend == "gpu" else "own_pid_vmhwm"}


def _post(url: str, body: bytes, timeout_s: float) -> None:
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        response.read()


def _healthy(port: int, proc, timeout_s: float) -> None:
    started = time.time()
    while time.time() - started < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"capacity probe server exited {proc.returncode} during load")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                return
        except (OSError, urllib.error.URLError):
            time.sleep(2)
    raise RuntimeError("capacity probe server not healthy within the boot timeout")


def run(*, argv: Sequence[str], env: Mapping[str, str], port: int,
        sequence: Sequence[Sequence[tuple[str, bytes]]], backend: str,
        popen: Callable = subprocess.Popen, post: Callable = _post,
        wait_healthy: Callable = _healthy, sampler_factory: Callable | None = None,
        settle: Callable[[], None] = lambda: time.sleep(1.0),
        boot_timeout_s: float = 600.0, request_timeout_s: float = 900.0,
        term_grace_s: float = 180.0) -> list[int]:
    """Launch once, drive the cycles, return the own-PID high-water mark per cycle."""
    from . import residency
    sampler = (sampler_factory or residency.Sampler)()
    key = "own_pid_peak_vram_bytes" if backend == "gpu" else "own_pid_peak_rss_bytes"
    peaks: list[int] = []
    proc = None
    with sampler:
        try:
            proc = popen(list(argv), env=dict(env), stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
            sampler.watch_pid(proc.pid)
            wait_healthy(port, proc, boot_timeout_s)
            for steps in sequence:
                for _label, body in steps:
                    post(f"http://127.0.0.1:{port}/completion", body, request_timeout_s)
                settle()  # let the sampler observe the cycle's high-water mark
                peaks.append(int(sampler.proof.get(key) or 0))
        finally:
            if proc is not None and proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(term_grace_s)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(30)
    return peaks


__all__ = ["DEFAULT_CYCLES", "GROWTH_TOLERANCE_BYTES", "SCHEMA", "SEQUENCE", "evaluate",
           "mixed_sequence", "run"]
