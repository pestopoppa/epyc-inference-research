#!/usr/bin/env python3
"""SS-BENCH-GATE-c — the running API's OWN spawn layer is bench-guarded too.

SS-BENCH-GATE-b guarded every CLI-launcher spawn (start/reload/aux/sidecar);
this suite covers the named residual: llama-servers spawned BY THE RUNNING API
process with default affinity, i.e. the WorkerPoolManager warm-start path whose
`numactl --interleave=all` prefix is a DEFAULT-affinity shape — the kernel may
schedule the worker's threads on any core, exactly like the 2026-07-27
incident sidecar that destroyed 1h09m of decision-gating measurement.

The API has no CLI flags, so `--allow-during-bench` becomes the
ORCHESTRATOR_ALLOW_DURING_BENCH=1 env knob, evaluated at spawn time and logged
loudly when it bypasses a refusal.

The claim is read by the REAL reader against a fake /proc tree (same seam as
the -b suite); the spawn itself is asserted through the Popen seam — no real
process is ever started.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Skip entire module if aiohttp is not available (used by worker_pool)
pytest.importorskip("aiohttp", reason="aiohttp required for worker_pool tests")

import scripts.server.bench_core_claim as bcc
from scripts.server.bench_core_claim import (
    API_BENCH_ALLOW_ENV,
    EMPTY_BENCH_CLAIM,
    BenchClaim,
    read_bench_claim,
)
from src.services.worker_pool import (
    WorkerConfig,
    WorkerInstance,
    WorkerPoolConfig,
    WorkerPoolManager,
    WorkerTier,
)

BENCH_PROC = ((4242, "python laguna_q4_cpu_bench_runner.py --run"),)

HOST_CORES = frozenset(range(192))


def _fake_proc_tree(tmp_path: Path, pid: int, main: str, threads: dict[str, str]) -> Path:
    """A /proc-like tree for one pid: status + task/<tid>/status per thread."""
    pdir = tmp_path / str(pid)
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "status").write_text(f"Cpus_allowed_list:\t{main}\n")
    task = pdir / "task"
    for tid, cpu_list in threads.items():
        (task / tid).mkdir(parents=True)
        (task / tid / "status").write_text(f"Cpus_allowed_list:\t{cpu_list}\n")
    return tmp_path


def _claim_from_fake_proc(tmp_path: Path, cpu_list: str) -> BenchClaim:
    """The real claim reader against a fake /proc tree + fake bench detection."""
    root = _fake_proc_tree(tmp_path, 4242, cpu_list, {"4243": cpu_list, "4244": cpu_list})
    return read_bench_claim(proc_root=root, detect=lambda: list(BENCH_PROC))


@pytest.fixture
def pool_manager(tmp_path: Path) -> WorkerPoolManager:
    config = WorkerPoolConfig(
        llama_server_path="/usr/local/bin/llama-server",
        log_dir=str(tmp_path / "worker-logs"),
        workers={
            "fast": WorkerConfig(
                name="fast",
                port=8102,
                model_path="/models/fast.gguf",
                tier=WorkerTier.WARM,
                threads=16,
                slots=4,
                task_types=["boilerplate", "transform"],
            ),
        },
    )
    return WorkerPoolManager(config=config)


@pytest.fixture
def worker(pool_manager: WorkerPoolManager) -> WorkerInstance:
    return WorkerInstance(config=pool_manager.config.workers["fast"])


def _wire_seams(
    monkeypatch: pytest.MonkeyPatch,
    manager: WorkerPoolManager,
    *,
    claim: BenchClaim,
    numactl: bool = True,
) -> MagicMock:
    """Fake the guard seams + the spawn; return the Popen mock capturing argv.

    `read_bench_claim` is the "fake bench detection" seam (the claim itself is
    built by the REAL reader against a fake /proc tree); `host_core_set` is
    the /sys seam; Popen is the spawn seam (nothing is ever executed).
    """
    if not claim.empty:
        monkeypatch.setattr(bcc, "host_core_set", lambda *_a, **_k: HOST_CORES)
    monkeypatch.setattr(bcc, "read_bench_claim", lambda *_a, **_k: claim)
    monkeypatch.setattr(
        "src.services.worker_pool.shutil.which",
        lambda _name: "/usr/bin/numactl" if numactl else None,
    )
    monkeypatch.setattr(manager, "_check_port_in_use", AsyncMock(return_value=False))
    monkeypatch.setattr(manager, "_wait_for_health", AsyncMock(return_value=True))
    popen = MagicMock()
    popen.poll.return_value = None
    monkeypatch.setattr("src.services.worker_pool.subprocess.Popen", popen)
    return popen


def _spawn_argv(popen: MagicMock) -> list[str]:
    return list(popen.call_args[0][0])


# --------------------------------------------------------------------------- #
# The /proc seam: the claim the guard uses comes from the REAL reader over a
# fake /proc tree (bench "detected" via the injectable detector).
# --------------------------------------------------------------------------- #


def test_fake_proc_seam_yields_claimed_cores(tmp_path: Path) -> None:
    claim = _claim_from_fake_proc(tmp_path, "0-95")
    assert not claim.unobservable
    assert claim.cores == frozenset(range(96))
    assert claim.procs == BENCH_PROC


# --------------------------------------------------------------------------- #
# Worker pool spawn: pin / refuse / unchanged
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_no_bench_spawn_is_byte_identical(
    monkeypatch, pool_manager: WorkerPoolManager, worker: WorkerInstance
) -> None:
    """No bench live -> the exact legacy argv (numactl prefix preserved)."""
    monkeypatch.delenv(API_BENCH_ALLOW_ENV, raising=False)
    popen = _wire_seams(monkeypatch, pool_manager, claim=EMPTY_BENCH_CLAIM)
    expected_cmd = pool_manager._build_launch_command(worker.config)

    assert await pool_manager._start_worker(worker) is True

    assert _spawn_argv(popen) == ["numactl", "--interleave=all", *expected_cmd]


@pytest.mark.asyncio
async def test_no_bench_spawn_unchanged_without_numactl(
    monkeypatch, pool_manager: WorkerPoolManager, worker: WorkerInstance
) -> None:
    """No bench + no numactl -> bare cmd, exactly as today."""
    monkeypatch.delenv(API_BENCH_ALLOW_ENV, raising=False)
    popen = _wire_seams(monkeypatch, pool_manager, claim=EMPTY_BENCH_CLAIM, numactl=False)
    expected_cmd = pool_manager._build_launch_command(worker.config)

    assert await pool_manager._start_worker(worker) is True

    assert _spawn_argv(popen) == expected_cmd


@pytest.mark.asyncio
async def test_bench_claiming_0_95_pins_worker_to_host_minus_claim(
    monkeypatch, tmp_path: Path, pool_manager: WorkerPoolManager, worker: WorkerInstance
) -> None:
    """Bench claims 0-95 on a 192-core host -> worker pinned to 96-191."""
    monkeypatch.delenv(API_BENCH_ALLOW_ENV, raising=False)
    claim = _claim_from_fake_proc(tmp_path, "0-95")
    popen = _wire_seams(monkeypatch, pool_manager, claim=claim)
    expected_cmd = pool_manager._build_launch_command(worker.config)

    assert await pool_manager._start_worker(worker) is True

    assert _spawn_argv(popen) == ["taskset", "-c", "96-191", *expected_cmd]


@pytest.mark.asyncio
async def test_bench_claiming_middle_range_pins_to_complement(
    monkeypatch, tmp_path: Path, pool_manager: WorkerPoolManager, worker: WorkerInstance
) -> None:
    """Bench claims 48-95 -> pinned to the folded complement 0-47,96-191."""
    monkeypatch.delenv(API_BENCH_ALLOW_ENV, raising=False)
    claim = _claim_from_fake_proc(tmp_path, "48-95")
    popen = _wire_seams(monkeypatch, pool_manager, claim=claim)

    assert await pool_manager._start_worker(worker) is True

    assert _spawn_argv(popen)[:3] == ["taskset", "-c", "0-47,96-191"]


@pytest.mark.asyncio
async def test_bench_claiming_every_core_refuses_spawn(
    monkeypatch, pool_manager: WorkerPoolManager, worker: WorkerInstance
) -> None:
    """No non-overlapping subset exists -> the spawn is refused (no Popen)."""
    monkeypatch.delenv(API_BENCH_ALLOW_ENV, raising=False)
    claim = BenchClaim(cores=frozenset(HOST_CORES), procs=BENCH_PROC)
    popen = _wire_seams(monkeypatch, pool_manager, claim=claim)

    assert await pool_manager._start_worker(worker) is False
    assert popen.call_count == 0


@pytest.mark.asyncio
async def test_unobservable_claim_refuses_spawn(
    monkeypatch, pool_manager: WorkerPoolManager, worker: WorkerInstance
) -> None:
    """Unknown must mean busy: an unreadable claim refuses the spawn."""
    monkeypatch.delenv(API_BENCH_ALLOW_ENV, raising=False)
    claim = BenchClaim(unobservable=True, procs=BENCH_PROC)
    popen = _wire_seams(monkeypatch, pool_manager, claim=claim)

    assert await pool_manager._start_worker(worker) is False
    assert popen.call_count == 0


# --------------------------------------------------------------------------- #
# Allow knob: ORCHESTRATOR_ALLOW_DURING_BENCH=1 bypasses refusals, loudly
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_allow_during_bench_env_bypasses_refusal_and_logs_loudly(
    monkeypatch, caplog, pool_manager: WorkerPoolManager, worker: WorkerInstance
) -> None:
    """Knob set -> an unobservable claim no longer refuses; the bypass is loud."""
    monkeypatch.setenv(API_BENCH_ALLOW_ENV, "1")
    claim = BenchClaim(unobservable=True, procs=BENCH_PROC)
    popen = _wire_seams(monkeypatch, pool_manager, claim=claim)
    expected_cmd = pool_manager._build_launch_command(worker.config)

    with caplog.at_level(logging.WARNING, logger="scripts.server.bench_core_claim"):
        assert await pool_manager._start_worker(worker) is True

    assert _spawn_argv(popen) == ["numactl", "--interleave=all", *expected_cmd]
    assert API_BENCH_ALLOW_ENV in caplog.text
    assert "may invalidate the run" in caplog.text


@pytest.mark.asyncio
async def test_allow_knob_set_without_bench_stays_silent(
    monkeypatch, caplog, pool_manager: WorkerPoolManager, worker: WorkerInstance
) -> None:
    """Knob set but no bench live -> nothing to bypass, no warning."""
    monkeypatch.setenv(API_BENCH_ALLOW_ENV, "1")
    popen = _wire_seams(monkeypatch, pool_manager, claim=EMPTY_BENCH_CLAIM)
    expected_cmd = pool_manager._build_launch_command(worker.config)

    with caplog.at_level(logging.WARNING, logger="scripts.server.bench_core_claim"):
        assert await pool_manager._start_worker(worker) is True

    assert _spawn_argv(popen) == ["numactl", "--interleave=all", *expected_cmd]
    assert "ORCHESTRATOR_ALLOW_DURING_BENCH" not in caplog.text


# ── the two named -c residuals: LlamaCppBackend + lightonocr ────────────────


def test_llamacppbackend_cmd_pinned_off_bench_claim(monkeypatch) -> None:
    """Legacy per-inference llama-completion (numactl default-affinity) is pinned
    off a live bench claim instead of tripping its continuity gate."""
    from src.inference.model_server import LlamaCppBackend

    monkeypatch.setattr(
        bcc, "read_bench_claim", lambda proc_root=Path("/proc"): BenchClaim(cores=frozenset(range(96)))
    )
    monkeypatch.setattr(bcc, "host_core_set", lambda *_a, **_k: HOST_CORES)
    backend = LlamaCppBackend(_FakeRegistry())
    cmd = backend._build_command(
        _FakeRoleConfig(), _FakeRequest(timeout=60)
    )
    assert "taskset -c" in cmd
    assert "numactl" not in cmd


def test_llamacppbackend_cmd_quiet_path_byte_identical(monkeypatch) -> None:
    """No bench live -> the legacy numactl-prefixed command is unchanged."""
    from src.inference.model_server import LlamaCppBackend

    monkeypatch.setattr(bcc, "read_bench_claim", lambda proc_root=Path("/proc"): EMPTY_BENCH_CLAIM)
    backend = LlamaCppBackend(_FakeRegistry())
    cmd = backend._build_command(
        _FakeRoleConfig(), _FakeRequest(timeout=60)
    )
    assert cmd.startswith("timeout 60 env OMP_NUM_THREADS=1 numactl --interleave=all")


def test_llamacppbackend_cmd_refuses_unobservable_claim(monkeypatch) -> None:
    """Unknown must mean busy: an unobservable claim refuses the spawn."""
    from src.inference.model_server import LlamaCppBackend

    monkeypatch.setattr(
        bcc, "read_bench_claim", lambda proc_root=Path("/proc"): BenchClaim(unobservable=True)
    )
    with pytest.raises(RuntimeError, match="refusing to spawn llama-completion"):
        LlamaCppBackend(_FakeRegistry())._build_command(_FakeRoleConfig(), _FakeRequest(timeout=60))


@pytest.mark.asyncio
async def test_lightonocr_spawn_pinned_off_bench_claim(monkeypatch, tmp_path: Path) -> None:
    """Per-request llama-mtmd-cli spawns get taskset-pinned off a live bench."""
    import src.services.lightonocr_llama_server as ocr_mod

    monkeypatch.setattr(
        bcc, "read_bench_claim", lambda proc_root=Path("/proc"): BenchClaim(cores=frozenset(range(96)))
    )
    configured_cli = Path(tmp_path) / "ni18-unavailable-mtmd-bin" / "llama-mtmd-cli"
    assert configured_cli.name == "llama-mtmd-cli"
    assert not configured_cli.exists() and not configured_cli.is_symlink()
    monkeypatch.setattr(ocr_mod, "CLI_PATH", str(configured_cli))
    monkeypatch.setattr(bcc, "host_core_set", lambda *_a, **_k: HOST_CORES)
    captured: list[list[str]] = []

    async def fake_exec(*cmd, **kw):
        captured.append(list(cmd))
        proc = asyncio.subprocess.Process  # type: ignore[attr-defined]
        return MagicMock(spec=proc, communicate=AsyncMock(return_value=(b"text", b"stats")))

    monkeypatch.setattr(ocr_mod.asyncio, "create_subprocess_exec", fake_exec)
    worker = ocr_mod.LlamaOCRWorker(worker_id=1, threads=8)
    await worker._run_inference("/tmp/fake.png")
    assert captured and captured[0][:3] == ["taskset", "-c", "96-191"]
    assert captured[0][3] == str(configured_cli)


@pytest.mark.asyncio
async def test_lightonocr_spawn_quiet_path_unchanged(monkeypatch, tmp_path: Path) -> None:
    """No bench -> the spawn argv is the CLI + flags, no taskset prefix."""
    import src.services.lightonocr_llama_server as ocr_mod

    monkeypatch.setattr(bcc, "read_bench_claim", lambda proc_root=Path("/proc"): EMPTY_BENCH_CLAIM)
    configured_cli = tmp_path / "ni18-unavailable-mtmd-bin" / "llama-mtmd-cli"
    assert configured_cli.name == "llama-mtmd-cli"
    assert not configured_cli.exists() and not configured_cli.is_symlink()
    monkeypatch.setattr(ocr_mod, "CLI_PATH", str(configured_cli))
    captured: list[list[str]] = []

    async def fake_exec(*cmd, **kw):
        captured.append(list(cmd))
        return MagicMock(
            spec=asyncio.subprocess.Process,
            communicate=AsyncMock(return_value=(b"text", b"stats")),
        )

    monkeypatch.setattr(ocr_mod.asyncio, "create_subprocess_exec", fake_exec)
    worker = ocr_mod.LlamaOCRWorker(worker_id=1, threads=8)
    await worker._run_inference("/tmp/fake.png")
    assert captured and captured[0][0] == str(configured_cli)


class _FakeRoleConfig:
    acceleration = type("A", (), {"type": "cpu"})()
    model = type("M", (), {"full_path": "/mnt/raid0/llm/models/fake.gguf"})()


class _FakeRequest:
    def __init__(self, timeout: int):
        self.timeout = timeout
        self.n_tokens = 8
        self.temperature = 0.0
        self.top_p = 1.0
        self.top_k = 40
        self.seed = None
        self.prompt_file = ""
        self.prompt = "hello"
        self.negative_prompt = None
        self.n_predict = None


class _FakeRegistry:
    _runtime_defaults: dict = {}


    def get_draft_for_role(self, name: str):
        return None
