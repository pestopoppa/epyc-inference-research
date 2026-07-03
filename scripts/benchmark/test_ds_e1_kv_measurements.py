#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def test_ds_e1_kv_measurement_dry_run_writes_clean_window_plan(tmp_path):
    script = Path(__file__).with_name("ds_e1_kv_measurements.sh")
    output_dir = tmp_path / "ds_e1_plan"

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--write-plan",
            "--role",
            "worker_general",
            "--ctx",
            "2048",
        ],
        cwd="/mnt/raid0/llm/epyc-inference-research",
        env={**os.environ, "OUTPUT_DIR": str(output_dir)},
        check=True,
        capture_output=True,
        text=True,
    )

    assert "mode: dry-run" in result.stdout
    assert "Wrote DS-E1 plan artifacts:" in result.stdout
    assert "Dry-run only. Re-run with --execute in a clean window" in result.stdout

    plan = json.loads((output_dir / "measurement_plan.json").read_text())
    assert plan["schema"] == "ds_e1_kv_measurement_plan.v1"
    assert plan["clean_window_required"] is True
    assert plan["role_filter"] == "worker_general"
    assert plan["ctx_filter"] == 2048
    assert plan["results_file"] == str(output_dir / "kv_measurements.csv")
    assert plan["contamination_overrides_excluded"] == [
        "--allow-active-autopilot",
        "--allow-live-llama",
    ]
    assert plan["rows"] == [
        {
            "context_length": 2048,
            "max_context": 16384,
            "model_id": "gemma4-26b-a4b-q4_k_m",
            "model_path": "/mnt/raid0/llm/models/gemma-4-26B-A4B-it-Q4_K_M.gguf",
            "role": "worker_general",
        }
    ]

    runner = output_dir / "run_clean_window.sh"
    assert os.access(runner, os.X_OK)
    runner_text = runner.read_text()
    assert "--execute --role worker_general --ctx 2048" in runner_text
    assert "--allow-active-autopilot" not in runner_text
    assert "--allow-live-llama" not in runner_text
