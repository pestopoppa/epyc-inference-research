"""AK-SPLIT-SCHEDULING: the operation preview/cancel code lives in operation_preview.py."""
import subprocess
import sys
from pathlib import Path

from . import operation_preview, scheduling

ROOT = Path(__file__).resolve().parents[4]


def test_engine_methods_are_the_moved_functions():
    engine = scheduling.SchedulerEngine
    assert engine.cancel_operation_before_launch is operation_preview.cancel_before_launch
    assert engine.preview_operation_accounting is operation_preview.preview_accounting
    assert engine.preview_operation_cancellation is operation_preview.preview_cancellation
    assert engine.apply_operation_preview is operation_preview.apply_preview


def test_scheduling_no_longer_defines_preview_code():
    source = Path(scheduling.__file__).read_text()
    for name in ("class OperationPreview", "def preview_operation_", "def apply_operation_preview"):
        assert name not in source
    assert "OperationPreview" not in scheduling.__all__


def test_import_order_independent():
    for first in ("operation_preview", "scheduling"):
        code = (f"import scripts.kernel_rnd.autokernel.loop.{first}\n"
                "from scripts.kernel_rnd.autokernel.loop import scheduling as s, operation_preview as o\n"
                "assert s.SchedulerEngine.apply_operation_preview is o.apply_preview\n")
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
