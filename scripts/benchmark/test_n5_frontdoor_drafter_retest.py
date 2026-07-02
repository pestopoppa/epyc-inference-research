#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def test_n5_frontdoor_drafter_retest_dry_run_emits_clear_footer(tmp_path):
    script = Path(__file__).with_name("n5_frontdoor_drafter_retest.sh")
    output_dir = tmp_path / "n5_retest"

    result = subprocess.run(
        [
            "bash",
            str(script),
        ],
        cwd="/mnt/raid0/llm/epyc-inference-research",
        env={**os.environ, "OUTPUT_DIR": str(output_dir)},
        check=True,
        capture_output=True,
        text=True,
    )

    assert "N5 frontdoor drafter retest preflight:" in result.stdout
    assert "mode: dry_run" in result.stdout
    assert "purpose: N5 qwen35-compatible frontdoor drafter alpha retest preflight" in result.stdout
    assert "Dry-run only. No inference was launched." in result.stdout
    assert (
        f"Review {output_dir / 'preflight.json'} and {output_dir / 'commands.sh'} for the clean-window launch package."
        in result.stdout
    )
    assert "Re-run with --strict --execute in a coordinated clean window to launch the smoke." in result.stdout

    preflight = json.loads((output_dir / "preflight.json").read_text())
    assert preflight["execution_mode"] == "dry_run"
    assert preflight["purpose"] == "N5 qwen35-compatible frontdoor drafter alpha retest preflight"
    assert (output_dir / "commands.sh").exists()
