"""Tests for agentic_swe_harness — ReplayClient + FakeEnv only.

No network, no docker, no model, no subprocess. Runnable directly
(`python3 test_agentic_swe_harness.py`, test_answer_scoring.py pattern) or
via pytest if available.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import agentic_swe_harness as h

FILES = {"pkg/mod.py": "import os\n\n\ndef f():\n    return 1\n"}
INSTANCE = {"instance_id": "demo__demo-1", "repo": "demo/demo",
            "problem_statement": "f() returns 1 but should return 2."}


def _cfg(**kw):
    return h.AgentConfig(**{"max_turns": 10, "max_wall_s": 300, **kw})


def _edit(search: str, replace: str, path: str = "pkg/mod.py") -> str:
    return (f"ACTION: edit\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n"
            f">>>>>>> REPLACE {path}")


# --------------------------------------------------------------------------- #
# Unit: parsing / truncation / paths
# --------------------------------------------------------------------------- #
def test_parse_action_variants():
    a = h.parse_action("Let me look around.\nACTION: bash\nls -la src/")
    assert a.kind == "bash" and a.command == "ls -la src/"
    # fenced command
    a = h.parse_action("ACTION: bash\n```bash\ngrep -n 'def f' pkg/mod.py\n```")
    assert a.kind == "bash" and a.command == "grep -n 'def f' pkg/mod.py"
    # command on the ACTION line itself (lenient)
    a = h.parse_action("ACTION: bash ls")
    assert a.kind == "bash" and a.command == "ls"
    # done
    assert h.parse_action("All set.\nACTION: done").kind == "done"
    # edit
    a = h.parse_action(_edit("    return 1", "    return 2"))
    assert a.kind == "edit" and len(a.blocks) == 1
    assert a.blocks[0][2] == "pkg/mod.py"
    # malformed: no action
    assert h.parse_action("I think I should look at the files.").kind is None
    # malformed: two actions
    two = "ACTION: bash\nls\nACTION: done"
    assert h.parse_action(two).kind is None
    # malformed: empty bash
    assert h.parse_action("ACTION: bash\n\n").kind is None
    # malformed: edit without SR block
    assert h.parse_action("ACTION: edit\nplease change return 1 to 2").kind is None


def test_truncate_output_head_tail():
    text = "A" * 3000 + "MIDDLE" + "B" * 3000
    out = h.truncate_output(text, limit=4000)
    assert len(out) < len(text)
    assert out.startswith("A" * 100) and out.endswith("B" * 100)
    assert "chars truncated" in out and "MIDDLE" not in out
    assert h.truncate_output("short", limit=4000) == "short"


def test_norm_rel_path():
    assert h.norm_rel_path("a/django/db/models.py") == "django/db/models.py"
    assert h.norm_rel_path("b/pkg/mod.py") == "pkg/mod.py"
    # regression vs convert_sr_to_patch.py's lstrip("ab/") char-set bug:
    assert h.norm_rel_path("astropy/units/core.py") == "astropy/units/core.py"
    assert h.norm_rel_path("bench/x.py") == "bench/x.py"
    assert h.norm_rel_path("/testbed/pkg/mod.py") == "pkg/mod.py"
    assert h.norm_rel_path("./pkg/mod.py") == "pkg/mod.py"
    assert h.norm_rel_path("/etc/passwd") is None
    assert h.norm_rel_path("../escape.py") is None
    assert h.norm_rel_path("") is None


def test_no_oracle_prompt():
    inst = dict(INSTANCE)
    inst["patch"] = "--- a/secret/gold_file.py\n+++ b/secret/gold_file.py\n"
    task = h.build_task_prompt(inst)
    assert inst["problem_statement"] in task
    assert "/testbed" in task and inst["repo"] in task
    assert "gold_file" not in task and "Relevant files" not in task


# --------------------------------------------------------------------------- #
# Loop scenarios
# --------------------------------------------------------------------------- #
def _run(responses, files=FILES, canned=None, cfg=None, traj=None):
    client = h.ReplayClient(responses)
    env = h.FakeEnv(files=files, canned=canned or {})
    res = h.run_instance(client, env, INSTANCE, cfg or _cfg(), traj_path=traj)
    return res, client, env


def test_happy_path_explore_edit_done():
    with tempfile.TemporaryDirectory() as td:
        traj = Path(td) / "demo.jsonl"
        res, client, env = _run(
            ["I'll explore first.\nACTION: bash\nls",
             "Found it.\n" + _edit("    return 1", "    return 2"),
             "Fixed.\nACTION: done"],
            canned={"ls": (0, "pkg\nsetup.py\n")},
            traj=traj)
        assert res["status"] == "done"
        assert res["turns_used"] == 3
        assert res["edits_applied"] == 1 and res["edits_failed"] == 0
        assert res["model_patch"], "expected a non-empty diff"
        assert "-    return 1" in res["model_patch"]
        assert "+    return 2" in res["model_patch"]
        assert "a/pkg/mod.py" in res["model_patch"]
        # the bash observation was fed back to the model
        second_call = client.calls[1]
        assert any("exit code 0" in m["content"] and "setup.py" in m["content"]
                   for m in second_call if m["role"] == "user")
        # trajectory: 3 turn records + 1 summary
        lines = [json.loads(l) for l in traj.read_text().splitlines()]
        assert len(lines) == 4
        assert [l["turn"] for l in lines[:3]] == [1, 2, 3]
        assert [l["action"] for l in lines[:3]] == ["bash", "edit", "done"]
        assert all(k in lines[0] for k in ("command", "exit", "obs_len", "wall"))
        assert lines[3]["summary"]["status"] == "done"
        assert lines[3]["summary"]["patch_chars"] == len(res["model_patch"])


def test_failed_search_then_corrected_retry():
    res, client, env = _run(
        [_edit("    return 42", "    return 2"),      # wrong SEARCH
         _edit("    return 1", "    return 2"),       # corrected
         "ACTION: done"])
    assert res["status"] == "done"
    assert res["edits_failed"] == 1 and res["edits_applied"] == 1
    assert "+    return 2" in res["model_patch"]
    # the failure observation reached the model before its retry
    second_call = client.calls[1]
    fail_obs = [m for m in second_call if m["role"] == "user"
                and "FAILED" in m.get("content", "")]
    assert fail_obs and "SEARCH block not found" in fail_obs[-1]["content"]


def test_ws_normalized_match_applies():
    # file has trailing spaces the model won't reproduce -> exact match fails,
    # whitespace-normalized fallback (rstrip line-sequence) applies
    files = {"pkg/mod.py": "def f():   \n    return 1  \n"}
    res, client, env = _run(
        [_edit("def f():\n    return 1", "def f():\n    return 2"),
         "ACTION: done"],
        files=files)
    assert res["edits_applied"] == 1 and res["edits_failed"] == 0
    assert "+    return 2" in res["model_patch"]
    obs = [m for m in client.calls[1] if m["role"] == "user"][-1]["content"]
    assert "whitespace-normalized match" in obs


def test_malformed_then_nudge_recovers():
    with tempfile.TemporaryDirectory() as td:
        traj = Path(td) / "t.jsonl"
        res, client, env = _run(
            ["I think I should look at the files first.",   # no action
             "ACTION: bash\nls",                             # recovery
             _edit("    return 1", "    return 2"),
             "ACTION: done"],
            canned={"ls": (0, "pkg\n")},
            traj=traj)
        assert res["status"] == "done"
        assert res["malformed"] == 1
        # nudge was injected before the second model call
        second_call = client.calls[1]
        assert second_call[-1]["role"] == "user"
        assert "no valid action" in second_call[-1]["content"]
        # nudge retry happens INSIDE turn 1 (4 model calls, 3 turns)
        assert len(client.calls) == 4 and res["turns_used"] == 3
        lines = [json.loads(l) for l in traj.read_text().splitlines()]
        assert lines[0]["action"] == "bash" and lines[0]["nudged"] is True
        assert res["model_patch"]


def test_double_malformed_wastes_turn():
    res, client, env = _run(
        ["no action here",           # malformed -> nudge
         "still just prose",         # malformed again -> wasted turn
         "ACTION: done"])
    assert res["status"] == "done"
    assert res["malformed"] == 2
    assert res["turns_used"] == 2          # wasted turn 1 + done turn 2
    # the wasted-turn observation reached the model
    third_call = client.calls[2]
    assert any("wasted" in m["content"] for m in third_call if m["role"] == "user")
    assert res["model_patch"] == ""        # nothing was edited


def test_turn_budget_exhaustion_partial_diff():
    cfg = _cfg(max_turns=3)
    res, client, env = _run(
        ["ACTION: bash\nls",
         _edit("    return 1", "    return 2"),
         "ACTION: bash\ngrep -n return pkg/mod.py"],   # never says done
        canned={"ls": (0, "pkg\n"),
                "grep -n return pkg/mod.py": (0, "5:    return 2\n")},
        cfg=cfg)
    assert res["status"] == "turns_exhausted"
    assert res["turns_used"] == 3
    assert len(client.calls) == 3          # clean exit, no extra model calls
    assert "+    return 2" in res["model_patch"], "partial diff must survive"


def test_command_timeout_observation_continues():
    res, client, env = _run(
        ["ACTION: bash\npython -m pytest -x",
         "ACTION: bash\nls",
         "ACTION: done"],
        canned={"python -m pytest -x": "TIMEOUT",
                "ls": (0, "pkg\n")})
    assert res["status"] == "done"
    # the timeout surfaced as an observation with the timeout exit code...
    second_call = client.calls[1]
    obs = [m for m in second_call if m["role"] == "user"][-1]["content"]
    assert "timed out" in obs and f"exit code {h.TIMEOUT_EXIT}" in obs
    # ...and the loop continued (second bash ran)
    assert "ls" in env.calls


def test_new_file_creation_empty_search():
    res, client, env = _run(
        [_edit("", "def helper():\n    return 2\n", path="pkg/helper.py"),
         "ACTION: done"])
    assert res["edits_applied"] == 1
    assert env.read_file("pkg/helper.py") is not None
    # created file was intent-to-add staged, then appeared in the diff
    assert any(c.startswith(h.GIT_ADD_N_PREFIX) and "pkg/helper.py" in c
               for c in env.calls)
    assert "b/pkg/helper.py" in res["model_patch"]
    assert "+def helper():" in res["model_patch"]


def test_multi_block_edit_single_action():
    multi = ("Fixing both spots.\nACTION: edit\n"
             "<<<<<<< SEARCH\n    return 1\n=======\n    return 2\n"
             ">>>>>>> REPLACE pkg/mod.py\n"
             "<<<<<<< SEARCH\nimport os\n=======\nimport os\nimport sys\n"
             ">>>>>>> REPLACE pkg/mod.py")
    res, client, env = _run([multi, "ACTION: done"])
    assert res["edits_applied"] == 2 and res["edits_failed"] == 0
    assert res["turns_used"] == 2
    patch = res["model_patch"]
    assert "+    return 2" in patch and "+import sys" in patch


def test_wall_budget_exhaustion():
    # fake clock: first 3 reads (t0, turn-1 gate, turn-1 record) stay small,
    # every later read is way past the budget -> turn-2 gate trips
    reads = [0.0, 1.0, 2.0]

    def clock():
        return reads.pop(0) if reads else 9999.0

    client = h.ReplayClient(["ACTION: bash\nls", "ACTION: bash\nls"])
    env = h.FakeEnv(files=FILES, canned={"ls": (0, "pkg\n")})
    res = h.run_instance(client, env, INSTANCE, _cfg(max_wall_s=100), clock=clock)
    assert res["status"] == "wall_exhausted"
    assert len(client.calls) == 1          # stopped before the second model call


def test_history_compaction_preserves_task_and_recent():
    msgs = [{"role": "system", "content": "SYS"},
            {"role": "user", "content": "TASK"}]
    for i in range(8):
        msgs.append({"role": "assistant", "content": f"resp {i}"})
        msgs.append({"role": "user", "content": f"OBS-{i} " + "x" * 500})
    elided = h.compact_history(msgs, max_chars=2000, keep_recent=4)
    assert elided > 0
    assert msgs[0]["content"] == "SYS" and msgs[1]["content"] == "TASK"
    # last 4 messages untouched
    assert all("x" * 100 in m["content"] for m in msgs[-4:] if m["role"] == "user")
    # an old observation was elided
    assert any(m.get("_elided") for m in msgs[2:-4])


def test_predictions_file_shape():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "predictions.json"
        h.write_predictions(p, [{"instance_id": "demo__demo-1",
                                 "model_name_or_path": "agentic_A4_35b_a3b",
                                 "model_patch": "--- a/x\n+++ b/x\n"}])
        rows = json.loads(p.read_text())
        assert isinstance(rows, list) and len(rows) == 1
        assert set(rows[0]) == {"instance_id", "model_name_or_path", "model_patch"}
        assert not p.with_suffix(".json.tmp").exists()


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except AssertionError as e:
                failed += 1
                print("FAIL", name, "-", e)
    if failed:
        raise SystemExit(f"{failed} test(s) failed")
    print("ALL TESTS PASSED")
