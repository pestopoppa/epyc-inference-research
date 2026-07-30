#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import server_np_sweep as sweep


def test_host_health_accepts_canonical_numa_balancing_disabled() -> None:
    warnings = sweep.host_health_warnings(
        {
            "uptime_seconds": 3600,
            "numa_balancing": "0",
            "existing_llama_processes": [],
        }
    )

    assert warnings == []


def test_host_health_flags_numa_balancing_enabled() -> None:
    warnings = sweep.host_health_warnings(
        {
            "uptime_seconds": 3600,
            "numa_balancing": "1",
            "existing_llama_processes": [],
        }
    )

    assert warnings == [
        "kernel.numa_balancing='1'; expected '0' for canonical NUMA-interleave CPU benchmarking"
    ]



def _result(stream_id: int, index: int, predicted_tokens: int, predicted_ms: float):
    return sweep.RequestResult(
        model="testmodel",
        np_level=2,
        request_index=index,
        qid=f"q{index}",
        suite="s",
        success=True,
        latency_ms=predicted_ms,
        predicted_tokens=predicted_tokens,
        prompt_tokens=10,
        predicted_tps=(
            predicted_tokens / (predicted_ms / 1000.0) if predicted_ms > 0 else 0.0
        ),
        predicted_ms=predicted_ms,
        stream_id=stream_id,
    )


def test_summarize_cell_reports_aggregate_and_per_stream_decode_tps() -> None:
    # Operator ruling 2026-07-30: the PRIMARY metric is aggregate decode tok/s,
    # from llama.cpp's predicted_n/predicted_ms — never wall clock. Stream 0:
    # 100 tok / 10s = 10 tok/s. Stream 1: 300 tok / 10s = 30 tok/s. Aggregate
    # (system-wide) = 40 tok/s; per-stream (token-weighted mean per slot) =
    # 400/20 = 20 tok/s. Wall time is 1000s so wall-clock contamination shows.
    results = [
        _result(0, 0, 50, 5000.0),
        _result(0, 1, 50, 5000.0),
        _result(1, 2, 150, 5000.0),
        _result(1, 3, 150, 5000.0),
    ]
    row = sweep.summarize_cell(
        model=sweep.ModelSpec(label="testmodel", path=Path("/nonexistent.gguf")),
        np_level=2,
        results=results,
        wall_s=1000.0,
        ttft_ms=123.0,
        server_pid=4242,
        server_command=["llama-server"],
    )
    assert abs(row["aggregate_decode_tps"] - 40.0) < 1e-9
    assert abs(row["per_stream_decode_tps"] - 20.0) < 1e-9
    assert row["decode_tokens_total"] == 400
    assert abs(row["decode_seconds_total"] - 20.0) < 1e-9
    assert row["decode_stream_count"] == 2
    # NEVER wall clock: the wall-clock token rate is a different, smaller number
    assert abs(row["aggregate_wallclock_tps"] - 0.4) < 1e-9
    # tasks/hour is still computed and persisted, as a secondary diagnostic
    assert abs(row["tasks_per_hour"] - (4 / 1000.0 * 3600.0)) < 1e-9
    # a request with no server-side decode duration drops from BOTH sums
    broken = list(results) + [_result(2, 4, 999, 0.0)]
    broken[-1].predicted_ms = 0.0
    row2 = sweep.summarize_cell(
        model=sweep.ModelSpec(label="testmodel", path=Path("/nonexistent.gguf")),
        np_level=2,
        results=broken,
        wall_s=1000.0,
        ttft_ms=123.0,
        server_pid=4242,
        server_command=["llama-server"],
    )
    assert abs(row2["aggregate_decode_tps"] - 40.0) < 1e-9
    assert row2["decode_tokens_total"] == 400


def test_build_recommendations_ranks_on_decode_tps_not_tasks_per_hour() -> None:
    # ANTI-SHORT-CIRCUIT GUARD: the two orderings DISAGREE on purpose. np=4
    # wins on tasks/hour (900 > 500); np=16 wins on aggregate decode tok/s
    # (400 > 120). Operator ruling 2026-07-30 makes tok/s the ranked metric, so
    # best_np must be 16. A version ranking on the demoted metric — or a
    # fixture that stopped writing the key the code reads — fails here instead
    # of passing by short-circuit. The wall-clock fallback is set to follow the
    # tasks/hour ordering so it cannot rescue a dropped primary key either.
    rows = [
        {
            "model": "m", "np": 4, "success_count": 43,
            "aggregate_decode_tps": 120.0, "per_stream_decode_tps": 30.0,
            "aggregate_wallclock_tps": 90.0,
            "tasks_per_hour": 900.0, "p95_latency_ms": 9000.0,
        },
        {
            "model": "m", "np": 16, "success_count": 43,
            "aggregate_decode_tps": 400.0, "per_stream_decode_tps": 25.0,
            "aggregate_wallclock_tps": 50.0,
            "tasks_per_hour": 500.0, "p95_latency_ms": 30000.0,
        },
    ]
    rec = sweep.build_recommendations(rows)[0]
    assert rec["metric"] == "aggregate_decode_tps"
    assert rec["best_np"] == 16                 # tok/s; tasks/hour would say 4
    assert rec["best_decode_tps"] == 400.0
    assert rec["best_per_stream_decode_tps"] == 25.0
    assert rec["throughput_basis"]["best"] == "decode"
    assert rec["mixed_throughput_basis"] is False
    # the 95%-of-peak saturation knee is a tok/s knee too: np=4 holds only 30%
    assert rec["saturation_np_95pct"] == 16
    assert rec["saturation_decode_tps"] == 400.0
    # tasks/hour is NOT deleted — kept beside the verdict, labelled secondary
    assert rec["best_tasks_per_hour"] == 500.0
    assert rec["saturation_tasks_per_hour"] == 500.0
    # latency is untouched by the ruling and remains the second axis
    assert rec["best_p95_latency_ms"] == 30000.0


def test_row_throughput_tps_prefers_decode_then_falls_back() -> None:
    # Reader tolerance for append-only summaries: prefer the new key, then the
    # renamed wall-clock key, then its pre-2026-07-29 name.
    assert sweep.row_throughput_tps({"aggregate_decode_tps": 42.0}) == 42.0
    assert sweep.row_throughput_basis({"aggregate_decode_tps": 42.0}) == "decode"
    renamed = {"aggregate_wallclock_tps": 11.0}
    assert sweep.row_throughput_tps(renamed) == 11.0
    assert sweep.row_throughput_basis(renamed) == "wallclock_fallback"
    legacy = {"aggregate_predicted_tps": 9.0}
    assert sweep.row_throughput_tps(legacy) == 9.0
    assert sweep.row_throughput_basis(legacy) == "wallclock_fallback"
    # tasks/hour never feeds the primary metric, however large
    assert sweep.row_throughput_tps({"tasks_per_hour": 5000.0}) == 0.0
    assert sweep.row_throughput_basis({"tasks_per_hour": 5000.0}) == "none"


def test_run_prompt_batch_stamps_stable_stream_ids(tmp_path) -> None:
    # The aggregate is the SUM of per-stream decode rates, so summarize_cell
    # needs each request attributed to the client stream that issued it. The
    # pool has exactly np_level workers and each runs its requests serially, so
    # a worker IS a stream. Without this stamping every request would land in
    # one bucket and the aggregate would silently collapse to the per-slot mean.
    import threading

    seen_idents: set[int] = set()

    def fake_send(*, port, prompt, request_index, model_label, np_level, n_predict, timeout_s):
        seen_idents.add(threading.get_ident())
        return _result(-1, request_index, 10, 1000.0)

    original = sweep.send_completion
    sweep.send_completion = fake_send
    try:
        results, wall_s = sweep.run_prompt_batch(
            port=18070,
            model=sweep.ModelSpec(label="testmodel", path=Path("/nonexistent.gguf")),
            np_level=3,
            prompts=[
                sweep.PromptSpec(qid=f"q{i}", suite="s", tier=1, prompt="p")
                for i in range(12)
            ],
            n_predict=16,
            request_timeout_s=5.0,
            requests_path=tmp_path / "requests.jsonl",
        )
    finally:
        sweep.send_completion = original

    assert len(results) == 12
    assert wall_s > 0
    stream_ids = {result.stream_id for result in results}
    assert stream_ids  # every request attributed
    assert -1 not in stream_ids
    # ids are dense 0..n-1 and never exceed the pool width
    assert stream_ids == set(range(len(stream_ids)))
    assert len(stream_ids) <= 3 and len(stream_ids) == len(seen_idents)
    # the on-disk request records carry both the stream id and predicted_ms
    lines = (tmp_path / "requests.jsonl").read_text().splitlines()
    assert len(lines) == 12
    import json as _json

    record = _json.loads(lines[0])
    assert "predicted_ms" in record and "stream_id" in record
