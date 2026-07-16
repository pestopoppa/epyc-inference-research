#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def test_n5_frontdoor_drafter_retest_dry_run_emits_clear_footer(tmp_path):
    script = Path(__file__).with_name("n5_frontdoor_drafter_retest.sh")
    output_dir = tmp_path / "n5_retest"
    llama_tree = tmp_path / "llama.cpp"
    llama_tree.mkdir()
    subprocess.run(["git", "init", str(llama_tree)], check=True, capture_output=True, text=True)
    fake_server = llama_tree / "build" / "bin" / "llama-server"
    fake_server.parent.mkdir(parents=True)
    fake_server.write_text(
        """#!/bin/bash
set -euo pipefail
case "${1:-}" in
  --version) echo "llama build a6c793fc6" ;;
  --help) echo "--spec-type [none|ngram-mod]"; echo "-md, --model-draft FNAME" ;;
  *) exit 0 ;;
esac
""",
        encoding="utf-8",
    )
    fake_server.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            str(script),
        ],
        cwd="/mnt/raid0/llm/epyc-inference-research",
        env={
            **os.environ,
            "OUTPUT_DIR": str(output_dir),
            "LLAMA_CPP_DIR": str(llama_tree),
            "LLAMA_SERVER": str(fake_server),
            "PRODUCTION_LLAMA_CPP_DIR": str(llama_tree),
        },
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
    assert any(
        "production llama.cpp tree selected" in blocker
        for blocker in preflight["blockers"]
    )
    assert any(
        "speculative flag surface is missing required N5/v7 tokens" in blocker
        for blocker in preflight["blockers"]
    )
    assert set(preflight["binary"]["missing_spec_tokens"]) == {
        "draft-tree",
        "draft-mtp",
        "--spec-draft-n-max",
        "--spec-draft-p-split",
    }
    assert preflight["binary"]["ld_library_path_prefix"] == str(fake_server.parent)
    assert preflight["required_arms"] == ["positive_mtp", "spec_off", "n5_spec_on"]
    assert set(preflight["server_commands"]) == {"positive_mtp", "spec_off", "n5_spec_on"}
    n5_argv = preflight["server_commands"]["n5_spec_on"]["argv"]
    assert f"LD_LIBRARY_PATH={fake_server.parent}" in n5_argv
    assert "/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf" in n5_argv
    assert "--spec-type" in n5_argv
    assert "draft-tree" in n5_argv
    assert "--spec-draft-n-max" in n5_argv
    assert "--draft-max" not in n5_argv
    assert "/mnt/raid0/llm/scratch/n5/Qwen3.5-0.8B-Q8_0.frontdoor-mtp-specials.gguf" in n5_argv
    positive_argv = preflight["server_commands"]["positive_mtp"]["argv"]
    assert "draft-mtp" in positive_argv
    assert "-md" not in positive_argv
    spec_off_argv = preflight["server_commands"]["spec_off"]["argv"]
    assert "none" in spec_off_argv
    assert (output_dir / "commands.sh").exists()
