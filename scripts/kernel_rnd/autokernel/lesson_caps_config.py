"""AK-PM-12-S4: the per-workload cap table and the keep path's cap wiring.

Off by default. With ``AUTOKERNEL_LESSON_CAPS=1`` the loop's keep path
(``loop/run.py:submit_keep_lesson``) forwards ``load_caps()`` to
``keep_lesson.submit_keep``; without it the keep path is exactly as before
(no caps -> UNKNOWN / cap_missing). Every cap is re-derived from its committed
native receipt by ``lesson_cap`` at load time; nothing here is a constant cap.

``AUTOKERNEL_LESSON_CAPS_RECEIPT`` overrides the receipt directory (absolute).
"""
from __future__ import annotations

import os

from .lesson_cap import receipt_ceiling, receipt_provenance

FLAG = 'AUTOKERNEL_LESSON_CAPS'
RECEIPT_ENV = 'AUTOKERNEL_LESSON_CAPS_RECEIPT'
# Stable copy of epyc-root artifacts/autokernel/ak-pm12-native-receipt-gpu-20261122 (root 0dead048e).
DEFAULT_RECEIPT = '/mnt/raid0/llm/tmp/ak-pm12-receipt-stable-20261122'
# task_id -> GPU part. AK-PM-12-S1 workload: Qwen3.8-27B Q8_0 dense MUL_MAT verify n=16, C17, MI210.
WORKLOADS = {'AK-PM-12:27b-q8-dense-n16': 'gfx90a'}


def enabled(environ=None):
    return (os.environ if environ is None else environ).get(FLAG) == '1'


def build_caps(receipt_dir=None, workloads=None):
    """{task_id: {upper_cap, cap_provenance}} re-derived from the receipt; raises on bad evidence."""
    receipt = receipt_dir or os.environ.get(RECEIPT_ENV) or DEFAULT_RECEIPT
    return {task: {'upper_cap': receipt_ceiling(receipt, part),
                   'cap_provenance': receipt_provenance(receipt, part, task)}
            for task, part in (WORKLOADS if workloads is None else workloads).items()}


def load_caps(environ=None):
    """None unless the flag is on; never raises (bad evidence -> None -> UNKNOWN, as today)."""
    if not enabled(environ):
        return None
    try:
        return build_caps()
    except Exception:  # noqa: BLE001 - observe-only write side
        return None


def submit_keep_with_caps(journal, attempt, *, producer=None):
    """The keep path's call: keep_lesson.submit_keep with caps only when enabled."""
    from .loop import keep_lesson
    return keep_lesson.submit_keep(journal, attempt, producer=producer, caps=load_caps())
