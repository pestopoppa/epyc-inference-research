"""KPF-64.3 readiness: arms E and F are declared in arms.json, launch with the PREDECLARED driver args, and the
adapter's families sample works without codex (childless by construction)."""
import json
import sys
import subprocess
from pathlib import Path

import adapter
import pytest

HERE = Path(__file__).resolve().parents[1]
ARMS = json.loads((HERE / "arms.json").read_text())["arms"]


def test_arms_json_matches_predeclared_text():
    e, f = ARMS["E"], ARMS["F"]
    assert (e["ctx_checkpoints"], e["cache_ram_mib"], e["sessions_mode"]) == (32, 32768, "flat")
    assert e["driver_args"] == ["--active", "8", "--think-scale", "0.1", "--think-cap-s", "60"]
    assert (f["ctx_checkpoints"], f["cache_ram_mib"], f["sessions_mode"]) == (32, 32768, "families")
    assert f["driver_args"][:4] == ["--children", "bursts", "--parent-during-burst", "pause"]
    assert "--think-scale" in f["driver_args"] and "--think-cap-s" in f["driver_args"]


@pytest.mark.parametrize("arm,needles", [
    ("E", ["--active 8", "--think-scale 0.1", "--think-cap-s 60", "adapter.py --mode flat", "--ctx-checkpoints 32", "--cache-ram 32768"]),
    ("F", ["--children bursts", "--parent-during-burst pause", "adapter.py --mode families", "--cache-ram 32768"])])
def test_launcher_dry_run_for_follow_up_arms(tmp_path, arm, needles):
    import os
    fake = tmp_path / "cpu" / "llama-server"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    env = {**os.environ, "WINDOW_CONFIRMED": "1", "KPF64_SERVER_BIN": str(fake), "KPF64_PORT": "18971"}
    r = subprocess.run(["bash", str(HERE / "arm_launcher.sh"), "--arm", arm, "--dry-run"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    for n in needles:
        assert n in r.stdout, (n, r.stdout)


def test_families_default_stores_drop_codex():
    assert adapter.default_stores("flat") == ("claude", "codex", "opencode")
    assert adapter.default_stores("families") == ("claude", "opencode")


def test_driver_warns_when_think_scale_has_no_gaps(tmp_path):
    import os
    s = tmp_path / "s"
    s.mkdir()
    turns = [{"pre": "a" * 400, "ref": "r", "reset": False}, {"pre": "b" * 400, "ref": "r", "reset": False}]
    (s / "x-00.json").write_text(json.dumps({"id": "x-00", "store": "claude", "turns": turns}))
    (s / "manifest.json").write_text(json.dumps({"schema": "kpf64.sessions.v1", "sessions": [{"id": "x-00"}], "totals": {}}))
    env = {**os.environ, "WINDOW_CONFIRMED": "1"}
    r = subprocess.run([sys.executable, str(HERE / "replay_driver.py"), "--sessions", str(s), "--out", str(tmp_path / "o"),
                        "--dry-run", "--think-scale", "0.1", "--history", "live"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "no turn carries gap_s" in r.stderr
    assert json.loads((tmp_path / "o" / "run_meta.json").read_text())["turns_with_gap_s"] == 0


def test_b_config_twins_differ_only_in_cache_ram():
    for tw, base in (("EB", "E"), ("FB", "F")):
        a, b = ARMS[tw], ARMS[base]
        assert (a["ctx_checkpoints"], a["cache_ram_mib"]) == (32, 0)
        assert {k: v for k, v in a.items() if k not in ("label", "cache_ram_mib")} == \
               {k: v for k, v in b.items() if k not in ("label", "cache_ram_mib")}
