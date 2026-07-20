from pathlib import Path

from scripts.benchmark import streamingllm_floor_sweep as runner


def test_parse_perf_extracts_llama_timings() -> None:
    stderr = """
common_perf_print: prompt eval time =      28.82 ms /    42 tokens (    0.69 ms per token,  1457.12 tokens per second)
common_perf_print:        eval time =    3021.57 ms /   319 runs   (    9.47 ms per token,   105.57 tokens per second)
common_perf_print:       total time =    3083.51 ms /   361 tokens
Maximum resident set size (kbytes): 967100
Exit status: 0
"""
    perf = runner.parse_perf(stderr)

    assert perf["prompt_tokens"] == 42
    assert perf["prompt_tps"] == 1457.12
    assert perf["decode_runs"] == 319
    assert perf["decode_tps"] == 105.57
    assert perf["total_tokens"] == 361
    assert perf["max_rss_kib"] == 967100


def test_score_output_checks_anchor_retention() -> None:
    body = "\n".join(
        f"{idx:03d} ANCHOR_ALPHA_17 retained"
        for idx in range(1, 22)
    )
    body += "\nSTREAMINGLLM_DONE ANCHOR_ALPHA_17 ANCHOR_BRAVO_29 ANCHOR_CHARLIE_41\n"

    score = runner.score_output(body)

    assert score["pass"]
    assert score["final_marker_present"]
    assert score["numbered_line_count"] == 21


def test_dry_run_writes_manifest(tmp_path: Path) -> None:
    exit_code = runner.main(
        [
            "--output-dir",
            str(tmp_path),
            "--context",
            "128",
            "--tokens",
            "256",
            "--cluster",
            "4:64",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "prompt.txt").exists()


def test_command_uses_absolute_prompt_path(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("hello\n", encoding="utf-8")
    args = runner.parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--context",
            "128",
            "--tokens",
            "256",
        ]
    )

    cmd = runner.command_for(args, runner.ArmSpec("baseline", 0, 0), prompt_path)

    assert str(prompt_path.resolve()) in cmd


def test_normalize_capture_strips_trailing_whitespace() -> None:
    assert runner.normalize_capture("alpha  \n\nbeta\t\n\n") == "alpha\n\nbeta\n"
    assert runner.normalize_capture(" \n\t\n") == ""
