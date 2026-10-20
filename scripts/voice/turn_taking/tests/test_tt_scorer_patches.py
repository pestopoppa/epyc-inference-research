"""CS-45 pin gate + patched-scorer known answer. Needs the (ephemeral) pinned clones; skipped if absent.

  /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python -I -m pytest -q test_tt_scorer_patches.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
TT = HERE.parent
sys.path.insert(0, str(TT))
import tt_audio as A  # noqa: E402
import tt_pins as PINS  # noqa: E402

FDB = Path(os.environ.get("TT_FDB_CHECKOUT", "/mnt/raid0/llm/tmp/dive-w3-fdb-repo/repo"))
TB = Path(os.environ.get("TT_TB_CHECKOUT", "/mnt/raid0/llm/tmp/dive-w3-turnbench/repo"))
LS = Path("/mnt/raid0/llm/data/asr-eval/librispeech-test-clean-300/wav")


def test_static_pins_pass():
    assert PINS.check_static() == []


@pytest.mark.skipif(not FDB.exists(), reason="FDB clone absent")
def test_patches_apply_and_scrub_to_recorded_hashes(tmp_path):
    assert PINS.apply("fdb", FDB, tmp_path / "f") == []


@pytest.mark.skipif(not TB.exists(), reason="TurnBench clone absent")
def test_turnbench_patch_applies(tmp_path):
    assert PINS.apply("tb", TB, tmp_path / "t") == []


@pytest.mark.skipif(not FDB.exists(), reason="FDB clone absent")
def test_wrong_base_hash_is_refused(tmp_path, monkeypatch):
    e = PINS.era()
    for c in e["changes"]:
        if c["scorer"].startswith("Full"):
            c["base_sha256"] = "0" * 64
    monkeypatch.setattr(PINS, "era", lambda: e)
    bad = PINS.apply("fdb", FDB, tmp_path / "f")
    assert bad and "base sha256" in bad[0]


@pytest.mark.skipif(not FDB.exists(), reason="FDB clone absent")
def test_wrong_commit_is_refused(tmp_path, monkeypatch):
    monkeypatch.setitem(PINS.COMMIT, "fdb", "0" * 40)
    assert "pinned" in PINS.apply("fdb", FDB, tmp_path / "f")[0]


@pytest.mark.skipif(not FDB.exists() or not LS.exists(), reason="FDB clone or LibriSpeech absent")
def test_patched_get_timing_known_answer(tmp_path):
    dest = tmp_path / "f"
    assert PINS.apply("fdb", FDB, dest) == []
    SR = A.SR
    a = A.read_wav(LS / "6930-75918-0000.wav").samples[:, 0]  # speech 0.546-3.505 s (pinned Silero, faster-whisper parity)
    b = A.read_wav(LS / "1320-122612-0000.wav").samples[:, 0]
    u = np.zeros(14 * SR, dtype=np.int16)
    u[: len(a)] = a
    m = np.zeros(14 * SR, dtype=np.int16)
    s = int(5.0 * SR)
    m[s : s + len(b)] = b[: 14 * SR - s]
    d = tmp_path / "toy" / "c1"
    d.mkdir(parents=True)
    A.write_wav(d / "input.wav", A.Pcm(u[:, None], SR))
    A.write_wav(d / "output.wav", A.Pcm(m[:, None], SR))
    env = {**os.environ, "TT_PINS_DIR": str(TT)}
    r = subprocess.run([sys.executable, "-I", str(dest / "v1_v1.5/evaluation/get_timing.py"), "--root_dir", str(tmp_path / "toy")],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    iv = json.loads((d / "latency_intervals.json").read_text())
    (u0, u1), = iv["latency_resp_list"]
    assert 3.45 <= u0 <= 3.62          # user speech end (3.505 s + pad/merge)
    assert 5.0 <= u1 <= 5.6            # model first speech after its 5.0 s start
    assert iv["latency_stop_list"] == []  # no overlap in the stimulus
