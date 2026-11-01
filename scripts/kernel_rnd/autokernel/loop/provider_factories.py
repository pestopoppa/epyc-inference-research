"""Closed registry for restartable provider constructors.

No production provider is registered. Enrollment is a source change: entries
bind a local Python file, its SHA-256 and a top-level no-argument function.
The supervisor starts a fresh interpreter before importing this file's target.
"""
from __future__ import annotations

import hashlib
import sys
import types
import inspect
from pathlib import Path

REGISTERED_FACTORIES = {}


def source(factory_id):
    try:
        relative, name, expected = REGISTERED_FACTORIES[factory_id]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('provider factory is not registered') from exc
    root = Path(__file__).resolve().parent
    path = root / relative
    if (not isinstance(relative, str) or path.is_symlink()
            or path.resolve().parent != root or not name.isidentifier()
            or len(expected) != 64 or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
        raise ValueError('provider factory source identity differs')
    return path, name


def construct(factory_id):
    path, name = source(factory_id)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != REGISTERED_FACTORIES[factory_id][2]:
        raise ValueError('provider factory changed while loading')
    module = types.ModuleType('_autokernel_registered_provider')
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(payload, str(path), 'exec'), module.__dict__)
    factory = getattr(module, name, None)
    if (not inspect.isfunction(factory) or factory.__module__ != module.__name__
            or factory.__qualname__ != name or factory.__closure__ is not None
            or inspect.signature(factory).parameters):
        raise ValueError('provider constructor must be a top-level no-argument function')
    return factory()
