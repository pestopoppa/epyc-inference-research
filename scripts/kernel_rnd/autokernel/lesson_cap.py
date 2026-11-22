"""AK-PM-12-S2: receipt-derived speedup ceiling and its evidence verifier.

speedup_ceiling = baseline_ns / (bytes_moved / peak_bw)

Every input is read back from evidence, never asserted:
  * baseline_ns  -- the native receipt's baseline capture stdout, after the
                    capture.json digests recorded in RESULT.json are re-checked;
  * bytes_moved  -- RESULT.json ``bytes_moved_per_forward`` of the same receipt;
  * peak_bw      -- the measured table (peak_bw_measured.json). The table value
                    must equal the best of its own embedded measured lines, and an
                    unknown part refuses (AK-PM-15: no spec-sheet constant).

A ``workload_receipt`` provenance is accepted only when the receipt re-derives
the cap and, if a cap is given, the cap equals the derived one. A
``workload_roofline`` provenance (a paper derivation) is still refused.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

PEAK_TABLE = Path(__file__).with_name('peak_bw_measured.json')
RECEIPT_KIND = 'workload_receipt'
_SOURCE_KEYS = {'receipt', 'part', 'task_id'}


class CapEvidenceError(ValueError):
    pass


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _positive(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def speedup_ceiling(baseline_ns, bytes_moved, peak_bw_bytes_per_s):
    """Pure roofline ceiling: baseline time over the bandwidth-bound floor."""
    if not (_positive(baseline_ns) and _positive(bytes_moved) and _positive(peak_bw_bytes_per_s)):
        raise CapEvidenceError('ceiling inputs must be finite and positive')
    floor_ns = bytes_moved / peak_bw_bytes_per_s * 1e9
    return baseline_ns / floor_ns


def peak_bw(part, table_path=PEAK_TABLE):
    """Measured peak bandwidth (bytes/s) for an exact part; refuse anything else."""
    table = json.loads(Path(table_path).read_text())
    if table.get('schema') != 'autokernel.peak_bw_measured.v1':
        raise CapEvidenceError('peak table schema mismatch')
    entry = table.get('parts', {}).get(part)
    if not isinstance(entry, dict):
        raise CapEvidenceError(f'unknown part {part!r}: no measured peak; refusing to estimate')
    lines = entry.get('lines')
    if not isinstance(lines, list) or not lines or not all(_positive(l.get('TBps')) for l in lines):
        raise CapEvidenceError('peak entry has no measured lines')
    value = entry.get('peak_bw_bytes_per_s')
    if not _positive(value) or round(max(l['TBps'] for l in lines) * 1e12) != value:
        raise CapEvidenceError('peak value is not the best of its measured lines')
    if not entry.get('source_row') or len(entry.get('source_file_sha256', '')) != 64:
        raise CapEvidenceError('peak entry lacks its measurement source')
    return float(value)


def receipt_ceiling(receipt_dir, part, table_path=PEAK_TABLE):
    """Re-derive the cap from a committed native receipt directory."""
    root = Path(receipt_dir)
    if not root.is_absolute():
        raise CapEvidenceError('receipt path must be absolute')
    result = json.loads((root / 'RESULT.json').read_text())
    if result.get('verdict') != 'RECEIPT_OK' or result.get('complete') is not True or result.get('smoke'):
        raise CapEvidenceError('receipt is not a complete RECEIPT_OK run')
    captures = result.get('captures') or {}
    for name in ('baseline', 'verification'):
        path = root / 'receipt' / name / 'capture.json'
        if path.is_symlink() or captures.get(name) != _sha(path.read_bytes()):
            raise CapEvidenceError(f'{name} capture digest mismatch')
    meta = json.loads((root / 'receipt/baseline/capture.json').read_text())
    stdout = (root / 'receipt/baseline/stdout.txt').read_bytes()
    if meta.get('stdout_sha256') != _sha(stdout) or meta.get('returncode') != 0:
        raise CapEvidenceError('baseline stdout not bound to its capture')
    out = json.loads(stdout)
    if out.get('correctness') is not True or out.get('protocol_id') != result.get('protocol_id'):
        raise CapEvidenceError('baseline capture is not a correct run of the receipt protocol')
    baseline_ns = out.get('baseline_ns')
    if baseline_ns != result.get('baseline_ns'):
        raise CapEvidenceError('RESULT baseline differs from the captured baseline')
    bytes_moved = result.get('bytes_moved_per_forward')
    if not isinstance(bytes_moved, int) or isinstance(bytes_moved, bool):
        raise CapEvidenceError('receipt lacks an integer bytes_moved_per_forward')
    return speedup_ceiling(baseline_ns, bytes_moved, peak_bw(part, table_path))


def verify_cap_provenance(provenance, task_id, upper_cap=None, table_path=PEAK_TABLE):
    """True only for a receipt-derived cap bound to this task (and cap, if given)."""
    try:
        if not isinstance(provenance, dict) or provenance.get('kind') != RECEIPT_KIND:
            return False
        source = json.loads(provenance['source'])
        if not isinstance(source, dict) or set(source) != _SOURCE_KEYS or source['task_id'] != task_id:
            return False
        derived = receipt_ceiling(source['receipt'], source['part'], table_path)
    except (CapEvidenceError, OSError, ValueError, KeyError, TypeError):
        return False
    return upper_cap is None or (_positive(upper_cap) and math.isclose(upper_cap, derived, rel_tol=1e-9))


def receipt_provenance(receipt_dir, part, task_id):
    """Build the provenance a caller passes alongside ``upper_cap``."""
    return {'kind': RECEIPT_KIND, 'source': json.dumps(
        {'receipt': str(receipt_dir), 'part': part, 'task_id': task_id}, sort_keys=True)}
