"""RVP-C5-ISA-1: archive a candidate kernel's AMDGCN ISA next to its source diff and
refuse a candidate whose claimed mechanism instruction is absent (static) or whose
instruction counter reads zero on the measured path (dynamic).

Input is llvm-objdump text of the built code object (the same disassembly
``gfx90a_isa_audit.py`` consumes), so the audit's parser is reused unchanged.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gfx90a_isa_audit as ga  # noqa: E402

from .counters import CounterReading, SignalError  # noqa: E402

SCHEMA = "epyc.gfx90a.isa_claim_check.v1"
_LDS_DMA = re.compile(r"(^|\s)lds(\s|$)")


def _lds_dma(i) -> bool:
    return i.op.startswith("buffer_load") and bool(_LDS_DMA.search(i.args))


MECHANISMS = {
    # name: (predicate over ga.Inst, rocprof counter that proves execution, or None)
    "mfma": (lambda i: i.op.startswith("v_mfma"), "SQ_INSTS_MFMA"),
    "dot": (lambda i: i.op.startswith("v_dot"), None),
    "lds_staging": (lambda i: i.op.startswith(("ds_read", "ds_write", "ds_load", "ds_store")), "SQ_INSTS_LDS"),
    "lds_dma": (_lds_dma, None),
    "buffer_load": (lambda i: i.op.startswith("buffer_load") and not _lds_dma(i), "SQ_INSTS_VMEM"),
}


def kernel_text(disasm: str, symbol: str) -> str:
    out, on = [], False
    for line in disasm.splitlines():
        m = ga._SYM_RE.match(line)
        if m:
            on = m.group(2) == symbol
        if on:
            out.append(line)
    if not out:
        raise SignalError(f"symbol not in disassembly: {symbol}")
    return "\n".join(out) + "\n"


def archive(disasm: str, symbol: str, out_dir: Path, source_diff_sha256: str) -> dict:
    """Write ``<isa_sha256>.s`` and return the record to store beside the source diff."""
    if not re.fullmatch(r"[0-9a-f]{64}", source_diff_sha256):
        raise SignalError("source_diff_sha256 must be a lowercase SHA-256")
    text = kernel_text(disasm, symbol)
    digest = hashlib.sha256(text.encode()).hexdigest()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{digest}.s").write_text(text, encoding="utf-8")
    insts = ga.split_disassembly(text, {symbol})[symbol]
    return {"schema": SCHEMA, "symbol": symbol, "isa_sha256": digest, "isa_file": f"{digest}.s",
            "source_diff_sha256": source_diff_sha256, "n_insts": len(insts),
            "mechanism_counts": {k: sum(1 for i in insts if p(i)) for k, (p, _) in MECHANISMS.items()}}


def check_claims(claims: Sequence[str], insts: Sequence, readings: Sequence[CounterReading] = ()) -> dict:
    """Per claim: static presence, then the counter reading when one exists.

    accepted = every claim statically present and no claimed counter reading zero.
    A claim with no reading is accepted ``static_only``; None is never treated as zero.
    """
    have = {r.name: r.value for r in readings}
    rows, accepted = [], True
    for claim in claims:
        if claim not in MECHANISMS:
            raise SignalError(f"unknown mechanism claim {claim!r}; known: {sorted(MECHANISMS)}")
        pred, counter = MECHANISMS[claim]
        n = sum(1 for i in insts if pred(i))
        value = have.get(counter) if counter else None
        if n == 0:
            status = "refused_absent_in_isa"
        elif value is not None and value == 0:
            status = "refused_counter_zero"
        elif value is not None:
            status = "confirmed_dynamic"
        else:
            status = "static_only"
        accepted &= not status.startswith("refused")
        rows.append({"claim": claim, "static_count": n, "counter": counter, "counter_value": value,
                     "status": status})
    return {"schema": SCHEMA, "accepted": accepted, "claims": rows}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="archive ISA and check claimed mechanisms")
    ap.add_argument("disasm", type=Path)
    ap.add_argument("symbol")
    ap.add_argument("--diff-sha256", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--claim", action="append", default=[])
    a = ap.parse_args(argv)
    text = a.disasm.read_text(encoding="utf-8")
    rec = archive(text, a.symbol, a.out_dir, a.diff_sha256)
    res = check_claims(a.claim, ga.split_disassembly(text, {a.symbol})[a.symbol])
    print(json.dumps({"archive": rec, "check": res}, indent=1, sort_keys=True))
    return 0 if res["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
