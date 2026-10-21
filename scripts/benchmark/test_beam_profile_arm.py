#!/usr/bin/env python3
"""HS-PROF-1: BEAM profile arm wiring + snapshot builder (offline; no model)."""

from __future__ import annotations

import pytest

import hs_prof1_profile_arms as hp
import long_context_adapters as lca
from long_context_adapters import BEAMAdapter, beam_context_kind_of_prompt
from test_beam_adapter import write_repo_fixture

ABS = ("preference_following", "instruction_following")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in (lca.BEAM_CONTEXT_MODE_ENV, lca.BEAM_PROFILE_SNAPSHOTS_ENV, lca.BEAM_PROFILE_ARM_ENV,
              lca.BEAM_ABILITIES_ENV):
        monkeypatch.delenv(k, raising=False)


def test_profile_mode_emits_profile_prompt_only_for_requested_abilities(tmp_path):
    ad = BEAMAdapter(data_dir=write_repo_fixture(tmp_path), context_mode="profile",
                     profile_snapshots={"1": "[format] Wants tables", "2": ""}, profile_arm="admission",
                     abilities=ABS)
    ps = ad.extract_all()
    assert {p["scoring_config"]["ability"] for p in ps} == set(ABS)
    assert len(ps) == 2 * 2 * 2
    p = next(x for x in ps if x["metadata"]["conversation_id"] == "1")
    assert p["prompt"].startswith(lca.BEAM_PROFILE_HEADER) and "Wants tables" in p["prompt"]
    assert beam_context_kind_of_prompt(p["prompt"]) == "profile"
    assert p["metadata"]["profile_arm"] == "admission" and p["provenance"]["profile_arm"] == "admission"
    assert "Assistant" not in p["prompt"]            # no transcript leaks into the arm
    empty = next(x for x in ps if x["metadata"]["conversation_id"] == "2")
    assert lca.BEAM_NO_PROFILE in empty["prompt"]


def test_profile_mode_refuses_missing_snapshot_and_missing_config(tmp_path):
    ad = BEAMAdapter(data_dir=write_repo_fixture(tmp_path), context_mode="profile",
                     profile_snapshots={"1": "x"}, profile_arm="a")
    with pytest.raises(lca.BEAMLoadError):
        ad.extract_all()
    with pytest.raises(ValueError):
        BEAMAdapter(data_dir=tmp_path, context_mode="profile")
    with pytest.raises(ValueError):
        BEAMAdapter(data_dir=tmp_path, context_mode="profile", profile_snapshots={}, profile_arm=None)


def test_default_modes_unchanged(tmp_path):
    ps = BEAMAdapter(data_dir=write_repo_fixture(tmp_path)).extract_all()
    assert len(ps) == 40 and "profile_arm" not in ps[0]["metadata"]


def test_snapshot_builder_arms_differ_on_inferred_text(tmp_path):
    """Runs in a subprocess: the orchestrator's `src` package must not leak into this process."""
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    raw = ("PREF [format] Prefers box-drawing Unicode tables\n"
           "PREF [style] Tends to ask for concise answers\n"
           "PREF [domain] Is currently debugging a kernel\n"
           "PREF [general] Likes things")
    (tmp_path / "d.json").write_text(json.dumps({"c1": raw}))
    root = os.environ.get("EPYC_ORCH_ROOT", "/workspace/repos/epyc-orchestrator")
    if not (Path(root) / "src" / "user_modeling" / "admission.py").is_file():
        pytest.skip("orchestrator checkout with src/user_modeling/admission.py not present")
    r = subprocess.run([sys.executable, str(Path(hp.__file__)), "build", "--derived", str(tmp_path / "d.json"),
                        "--out-dir", str(tmp_path)], capture_output=True, text=True,
                       env={**os.environ, "EPYC_ORCH_ROOT": root})
    assert r.returncode == 0, r.stderr[-800:]
    cur = json.loads((tmp_path / "snapshots_pref_current.json").read_text())["c1"]
    adm = json.loads((tmp_path / "snapshots_admission.json").read_text())["c1"]
    assert "Tends to ask" in cur and "currently debugging" in cur
    assert "box-drawing" in adm and "Tends to" not in adm and "currently" not in adm and "Likes things" not in adm


def test_derive_refuses_without_window(monkeypatch):
    monkeypatch.delenv("WINDOW_CONFIRMED", raising=False)
    with pytest.raises(SystemExit):
        hp.derive("http://127.0.0.1:9/v1", "m", None, "100K")
