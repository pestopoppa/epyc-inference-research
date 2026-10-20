"""AKX-ALL-29.5: poisoned guard bands around dst and KV views.

`test-backend-ops` compares a node's dst only, so a kernel that writes OUTSIDE its
declared extent (a stray row, an off-by-one tail, a KV slot that is not its own)
can pass it. This module is the harness side of the fix: a guarded arena that places
every view between two poison-filled bands, and a checker that names which band of
which view was touched and where.

Pure Python over host bytes, so the logic runs on CPU buffers. A GPU producer copies
the arena to the device, runs the kernel with each view's pointer at `offset(name)`,
copies the arena back and calls `check()`; the receipt it emits is judged by
`verdict_from_receipt`, which is opt-in (not part of the default gate chain) and
treats an absent or incomplete receipt as a failure of THAT call, never as a pass.

The band pattern is position-dependent, so a stray write of 0x00, 0xFF, a repeated
byte or a copy of neighbouring data is still a difference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from . import gates

SCHEMA = "epyc.autokernel.guard_bands.v1"
GATE = "guard_bands"
DEFAULT_BAND_BYTES = 4096        # one page: also catches a whole stray row or cache line
ALIGN = 256                      # device allocation alignment each view starts on
MAX_REPORTED = 8


def poison_byte(index: int) -> int:
    """Deterministic and position-dependent (167 is odd: 256 distinct values per period)."""
    return (index * 167 + 91) & 0xFF


@dataclass(frozen=True)
class Violation:
    view: str
    band: str                    # "head" | "tail"
    first_offset: int            # byte offset from the start of the band
    n_bytes: int                 # bytes in the band that differ from the poison
    expected: int
    observed: int

    def reason(self) -> str:
        where = ("before" if self.band == "head" else "after") + f" view '{self.view}'"
        return (f"guard_band_violation: {self.n_bytes} byte(s) written {where} "
                f"({self.band} band, first at +{self.first_offset}: "
                f"0x{self.observed:02x} != poison 0x{self.expected:02x})")


class GuardedArena:
    """One buffer holding named views, each wrapped by head and tail poison bands."""

    def __init__(self, band_bytes: int = DEFAULT_BAND_BYTES):
        if band_bytes <= 0 or band_bytes % ALIGN:
            raise ValueError(f"band_bytes must be a positive multiple of {ALIGN}")
        self.band_bytes = band_bytes
        self._views: dict[str, tuple[int, int]] = {}      # name -> (offset, extent)
        self._size = 0
        self.buf: bytearray | None = None

    def add_view(self, name: str, extent: int) -> int:
        """Reserve `extent` bytes; returns the view's offset in the arena."""
        if self.buf is not None:
            raise RuntimeError("arena already sealed")
        if not name or name in self._views:
            raise ValueError(f"view name '{name}' is empty or duplicated")
        if extent <= 0:
            raise ValueError("view extent must be positive")
        start = self._size + self.band_bytes
        self._views[name] = (start, extent)
        padded = -(-extent // ALIGN) * ALIGN
        # the tail band starts at the view's exact end; padding up to ALIGN belongs to it
        self._size = start + padded + self.band_bytes
        return start

    def seal(self) -> bytearray:
        """Allocate and fill every band (and the alignment pad) with poison."""
        self.buf = bytearray(self._size)
        for name in self._views:
            for band, (lo, hi) in self._band_ranges(name).items():
                for i in range(lo, hi):
                    self.buf[i] = poison_byte(i - lo)
        return self.buf

    def offset(self, name: str) -> int:
        return self._views[name][0]

    def extent(self, name: str) -> int:
        return self._views[name][1]

    def view(self, name: str) -> memoryview:
        off, ext = self._views[name]
        return memoryview(self._need_buf())[off:off + ext]

    def _need_buf(self) -> bytearray:
        if self.buf is None:
            raise RuntimeError("arena not sealed")
        return self.buf

    def _band_ranges(self, name: str) -> dict[str, tuple[int, int]]:
        off, ext = self._views[name]
        padded = -(-ext // ALIGN) * ALIGN
        return {"head": (off - self.band_bytes, off),
                "tail": (off + ext, off + padded + self.band_bytes)}

    def check(self, buf: bytes | bytearray | memoryview | None = None) -> list[Violation]:
        """Every band byte that is no longer poison, grouped per band. `buf` is the
        arena read back from the device (defaults to the host buffer)."""
        data = self._need_buf() if buf is None else buf
        if len(data) != self._size:
            raise ValueError(f"arena readback is {len(data)} bytes, expected {self._size}")
        found = []
        for name in self._views:
            for band, (lo, hi) in self._band_ranges(name).items():
                bad = [i for i in range(lo, hi) if data[i] != poison_byte(i - lo)]
                if bad:
                    found.append(Violation(name, band, bad[0] - lo, len(bad),
                                           poison_byte(bad[0] - lo), data[bad[0]]))
        return found

    def receipt(self, violations: Sequence[Violation]) -> dict[str, Any]:
        return {"schema": SCHEMA, "band_bytes": self.band_bytes,
                "views": [{"name": n, "extent": e, "offset": o} for n, (o, e) in self._views.items()],
                "violations": [vars(v).copy() for v in violations]}


def verdict_from_receipt(receipt: Mapping[str, Any] | None, *,
                         required_views: Iterable[str] = ("dst",)) -> gates.Verdict:
    """Pass only when the receipt covers every required view and no band was touched."""
    if not isinstance(receipt, Mapping) or receipt.get("schema") != SCHEMA:
        return gates.Verdict(GATE, False, "guard_band_receipt_missing: no "
                             f"{SCHEMA} receipt; absence is not a pass")
    covered = {v.get("name") for v in receipt.get("views") or () if isinstance(v, Mapping)}
    missing = [n for n in required_views if n not in covered]
    if missing:
        return gates.Verdict(GATE, False, "guard_band_receipt_incomplete: no guard band "
                             f"around required view(s) {', '.join(missing)}")
    rows = [r for r in receipt.get("violations") or () if isinstance(r, Mapping)]
    if rows:
        shown = [Violation(**{k: r[k] for k in Violation.__dataclass_fields__}) for r in rows]
        return gates.Verdict(GATE, False, shown[0].reason()
                             + (f" (+{len(shown) - 1} more band(s))" if len(shown) > 1 else ""),
                             "\n".join(v.reason() for v in shown[:MAX_REPORTED]))
    return gates.Verdict(GATE, True, f"{len(covered)} view(s) guarded, all bands intact")
