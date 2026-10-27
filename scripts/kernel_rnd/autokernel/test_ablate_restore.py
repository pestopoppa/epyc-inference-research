"""Model-free regressions for the actual HARNESS-1 entrypoint."""
import os
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).with_name("ablate.sh")


def git(work, *args):
    return subprocess.check_output(["git", "-C", str(work), *args], text=True)


def fixture(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    git(work, "init", "-q")
    git(work, "config", "user.email", "fixture@example.invalid")
    git(work, "config", "user.name", "Fixture")
    (work / "lever").write_text("champion\n")
    (work / "obsolete").write_text("retained champion file\n")
    git(work, "add", "lever", "obsolete")
    git(work, "commit", "-qm", "champion")
    # No real build, region claim, model or object inspection is invoked.
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    for name, content in {
        "taskset": '#!/bin/bash\nshift 2\nexec "$@"\n',
        "cmake": "#!/bin/bash\nexit 0\n",
        "python3": "#!/bin/bash\nexit ${FAKE_DIFF_RC:-0}\n",
    }.items():
        p = fakebin / name
        p.write_text(content)
        p.chmod(0o755)
    # region-lock's absolute path is deliberately preserved in production;
    # replace only that executor in a fixture copy, never the cleanup logic.
    runner = tmp_path / "ablate.sh"
    original = SCRIPT.read_text()
    executor = "/workspace/repos/epyc-orchestrator/scripts/region-lock run --cpu-list 48-95 --role build --tag harness1-abl-$NAME -- "
    assert original.count(executor) == 1, "refuse fixture if real build executor changed"
    runner.write_text(original.replace(executor, "env "))
    env = dict(os.environ, ABL_WORKTREE=str(work), ABL_HARNESS=str(tmp_path / "harness"),
               PATH=str(fakebin) + os.pathsep + os.environ["PATH"])
    return work, runner, env


def test_staged_arm_is_restored_and_preexisting_index_is_refused(tmp_path):
    work, runner, env = fixture(tmp_path)
    sentinel = work / "untracked-operator-artifact"
    sentinel.write_text("retain")
    result = subprocess.run(["bash", str(runner), "staged",
                             "printf mutated > lever; printf added > new-tracked; "
                             "git add lever new-tracked; git rm obsolete"], env=env)
    assert result.returncode == 0
    assert (work / "lever").read_text() == "champion\n"
    assert git(work, "diff", "HEAD", "--") == ""
    assert sentinel.read_text() == "retain"
    assert (work / "obsolete").read_text() == "retained champion file\n"
    assert not (work / "new-tracked").exists()
    (work / "lever").write_text("operator change")
    git(work, "add", "lever")
    before = git(work, "diff", "--cached")
    # HEAD-only dirt checks miss an index change whose worktree is back at HEAD.
    (work / "lever").write_text("champion\n")
    result = subprocess.run(["bash", str(runner), "refused", "exit 99"], env=env)
    assert result.returncode == 1
    assert git(work, "diff", "--cached") == before


def test_failed_diff_pipeline_preserves_status_and_restores_staged_arm(tmp_path):
    work, runner, env = fixture(tmp_path)
    env["FAKE_DIFF_RC"] = "37"
    result = subprocess.run(["bash", str(runner), "diff-fails", "printf mutated > lever; git add lever"], env=env)
    assert result.returncode == 37
    assert (work / "lever").read_text() == "champion\n"
    assert git(work, "diff", "HEAD", "--") == ""
