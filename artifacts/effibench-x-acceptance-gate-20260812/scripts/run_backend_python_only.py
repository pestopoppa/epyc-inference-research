#!/usr/bin/env python
"""Python-only EffiBench-X sandbox backend for the canonical-solutions acceptance gate.

Operational wrapper — NOT an upstream patch:
- prunes EFFIBENCH_REGISTRY in-place to python3 (avoids pulling 5 unneeded images;
  upstream setup() iterates all six languages unconditionally),
- binds the server main process to physical core 11 instead of the last core
  (upstream binds to core 95, whose SMT sibling 191 is a GPU host thread on this host),
- serves on 127.0.0.1:8999 (host port 8000 is the orchestrator API).
"""
import logging
import sys

import effibench.utils as u

# Prune registry in place (same dict/list objects are imported elsewhere).
for k in list(u.EFFIBENCH_REGISTRY):
    if k != "python3":
        del u.EFFIBENCH_REGISTRY[k]
u.EFFIBENCH_LANGS[:] = ["python3"]

import effibench.backends.backend_utils as bu

def _bind_main_to_core11():
    import os
    topo = bu.get_cpu_topology()
    core = 11 if 11 in topo else max(topo.keys())
    bu.set_cpu_affinity(os.getpid(), topo[core])

bu.bind_main_process_to_last_core = _bind_main_to_core11

from effibench.utils import setup_logger
from effibench.backends import get_backend

setup_logger()
NUM_WORKERS = 10
manager, app = get_backend(backend_type="docker", num_workers=NUM_WORKERS, setup_logging=False, skip_setup=False)
logging.info(f"Python-only backend ready with {manager.num_workers} workers")

import types
temp_module = types.ModuleType("temp_app_module")
temp_module.app = app
sys.modules["temp_app_module"] = temp_module

import uvicorn
uvicorn.run("temp_app_module:app", host="127.0.0.1", port=8999, reload=False)
