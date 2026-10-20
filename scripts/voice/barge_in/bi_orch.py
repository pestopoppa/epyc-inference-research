#!/usr/bin/env python3
"""CS-48 access to the accepted epyc-orchestrator ``src/voice`` source, READ-ONLY.

The harness imports the controller/cascade as they are; it never edits, patches or monkey-patches them. The sha256 of
every ``src/voice/*.py`` file is recorded in each run's ``run.json`` so a run states exactly which controller it drove.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

DEFAULT_SRC = "/workspace/repos/epyc-orchestrator"
VOICE_FILES = ("__init__.py", "contracts.py", "controller.py", "cascade.py", "http_clients.py")


def orch_root() -> Path:
    return Path(os.environ.get("EPYC_ORCH_SRC", DEFAULT_SRC))


def voice_hashes() -> dict[str, str]:
    d = orch_root() / "src" / "voice"
    return {f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in VOICE_FILES if (d / f).exists()}


def load_voice():
    """-> (VoiceController, CascadeBackend, VoiceTurn, VoiceEvent). Import path is added, nothing is written."""
    sys.dont_write_bytecode = True  # never leave __pycache__ in the (read-only) orchestrator tree
    root = str(orch_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.voice.cascade import CascadeBackend
    from src.voice.contracts import VoiceEvent, VoiceTurn
    from src.voice.controller import VoiceController

    return VoiceController, CascadeBackend, VoiceTurn, VoiceEvent
