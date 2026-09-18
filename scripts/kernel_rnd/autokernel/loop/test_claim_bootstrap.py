"""Standalone import regression for the orchestrator-owned CPU claim."""
from pathlib import Path
import os
import subprocess
import sys


def test_hold_cpu_bootstraps_root_worktree_owner_without_pythonpath_glue(tmp_path):
    shared_root = tmp_path / "root-main"
    lane_root = tmp_path / "root-lane"
    git_dir = shared_root / ".git" / "worktrees" / "root-lane"
    runtime = shared_root / "repos" / "epyc-orchestrator" / "src" / "runtime"
    git_dir.mkdir(parents=True)
    lane_root.mkdir()
    runtime.mkdir(parents=True)
    (lane_root / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")
    (git_dir / "commondir").write_text("../..\n", encoding="utf-8")
    (runtime / "cpu_region_lock.py").write_text(
        "def cpu_region_lock(*args, **kwargs): raise AssertionError('preflight must run first')\n"
        "def global_region_lock_path(region): raise AssertionError(region)\n",
        encoding="utf-8",
    )
    (runtime / "instance_topology.py").write_text(
        "ATOMIC_REGIONS = ('fixture',)\n"
        "def cpu_list_to_regions(cpu_list): return ['fixture']\n",
        encoding="utf-8",
    )
    (runtime / "region_lock_cli.py").write_text(
        "def _preflight(*, strict): return 'fixture preflight reached'\n",
        encoding="utf-8",
    )

    research_root = Path(__file__).parents[4]
    environment = os.environ.copy()
    environment.update(EPYC_ROOT_REPO=str(lane_root), PYTHONPATH=".")
    result = subprocess.run(
        [sys.executable, "-c", (
            "from scripts.kernel_rnd.autokernel.loop import claim\n"
            "try:\n"
            "    with claim.hold_cpu('0'):\n"
            "        raise AssertionError('claim unexpectedly yielded')\n"
            "except claim.ClaimRefused as exc:\n"
            "    assert str(exc) == 'fixture preflight reached', exc\n"
        )],
        cwd=research_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
