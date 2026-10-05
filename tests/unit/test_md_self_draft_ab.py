"""Unit coverage for the CPU embedded NEXTN self-draft A/B harness."""

from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.benchmark import md_self_draft_ab as ab


@pytest.mark.parametrize("mode", ["explicit", "help", "missing-default"])
def test_fresh_cli_import_honors_binary_override_without_kernel_store(tmp_path, mode):
    missing_store = tmp_path / "absent-kernel-store"
    code = """
import sys
from pathlib import Path
from src.registry import kernel_paths
kernel_paths.PRODUCTION_ROOT = Path(sys.argv[1])
assert not kernel_paths.PRODUCTION_ROOT.exists()
from scripts.benchmark import md_self_draft_ab as ab
mode = sys.argv[2]
if mode == 'explicit':
    assert ab.parse_args(['--binary', '/tmp/explicit-unused-server']).binary == Path('/tmp/explicit-unused-server')
elif mode == 'help':
    try:
        ab.parse_args(['--help'])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError('help must exit')
else:
    try:
        ab.parse_args([])
    except kernel_paths.KernelPathError as exc:
        assert "kernel backend 'cpu'" in str(exc)
    else:
        raise AssertionError('missing default store must fail closed')
assert not kernel_paths.PRODUCTION_ROOT.exists()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(missing_store), mode],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        binary=Path("/bin/llama-server"),
        model=Path("/models/qwen-mtp.gguf"),
        host="127.0.0.1",
        port=18070,
        output_dir=tmp_path,
        prompt="hello",
        prompt_file="",
        n_predict=16,
        repetitions=2,
        warmups=0,
        seed=1234,
        temperature=0.0,
        context_tokens=4096,
        threads=8,
        ubatch=512,
        slots=1,
        kv_type_k="q8_0",
        kv_type_v="q8_0",
        spec_type="draft-mtp",
        draft_max=4,
        draft_p_min=None,
        threads_draft=None,
        request_timeout=30,
        startup_timeout=30,
        min_speedup_ratio=1.05,
        server_arg=[],
        env=[],
        mlock=True,
        no_mmap=False,
        flash_attn=True,
        jinja=True,
        skip_autopilot_idle_check=False,
        dry_run=False,
    )


def test_build_server_command_compares_same_file_md_vs_embedded(tmp_path: Path) -> None:
    args = _args(tmp_path)

    same = ab.build_server_command(args, ab.Arm("same_file_md", True))
    embedded = ab.build_server_command(args, ab.Arm("embedded_self_draft", False))

    assert same[same.index("-md") + 1] == "/models/qwen-mtp.gguf"
    assert "-md" not in embedded
    assert same[same.index("--spec-type") + 1] == "draft-mtp"
    assert embedded[embedded.index("--spec-draft-n-max") + 1] == "4"
    assert "--mlock" in same
    assert "--flash-attn" in embedded


def test_build_report_marks_embedded_faster_from_median_tps(tmp_path: Path) -> None:
    args = _args(tmp_path)
    same = ab.ArmResult(
        name="same_file_md",
        include_same_file_md=True,
        command=["llama-server", "-md", "/models/qwen-mtp.gguf"],
        log_path=str(tmp_path / "same.log"),
        pid=123,
        load_memory=ab.MemorySample(rss_kib=100 * 1024, pss_kib=90 * 1024),
        post_run_memory=ab.MemorySample(rss_kib=101 * 1024, pss_kib=91 * 1024),
        runs=[
            ab.CompletionRun(1, 1.0, 30.0, 30, 10, 120),
            ab.CompletionRun(2, 1.0, 31.0, 31, 10, 121),
        ],
        acceptance={"task_line_count": 1, "cumulative_line_count": 0},
    )
    embedded = ab.ArmResult(
        name="embedded_self_draft",
        include_same_file_md=False,
        command=["llama-server"],
        log_path=str(tmp_path / "embedded.log"),
        pid=124,
        load_memory=ab.MemorySample(rss_kib=80 * 1024, pss_kib=70 * 1024),
        post_run_memory=ab.MemorySample(rss_kib=81 * 1024, pss_kib=71 * 1024),
        runs=[
            ab.CompletionRun(1, 1.0, 36.0, 36, 10, 120),
            ab.CompletionRun(2, 1.0, 37.0, 37, 10, 121),
        ],
        acceptance={"task_line_count": 1, "cumulative_line_count": 0},
    )

    report = ab.build_report(args, [same, embedded])

    assert report["decision"] == "embedded_self_draft_faster"
    assert report["speedup_ratio_median_tps_embedded_over_same_file_md"] == pytest.approx(36.5 / 30.5)
    assert report["pss_delta_mib_embedded_minus_same_file_md"] == -20.0


def test_main_dry_run_prints_both_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = ab.main(
        [
            "--binary",
            "/bin/llama-server",
            "--model",
            "/models/qwen-mtp.gguf",
            "--output-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "-md" in payload["commands"]["same_file_md"]
    assert "-md" not in payload["commands"]["embedded_self_draft"]


def test_main_refuses_when_autopilot_not_idle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        ab,
        "autopilot_quiet",
        lambda _root: (False, {"pid": 123, "phase": "dispatch_action"}),
    )

    rc = ab.main(
        [
            "--binary",
            "/bin/llama-server",
            "--model",
            "/models/qwen-mtp.gguf",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert rc == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"] == "autopilot_not_idle"
