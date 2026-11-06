#!/usr/bin/env python3
"""Bit-level CPU check of the R7 CAS loop arithmetic: f32(bf16 a) + f32(bf16 b) (IEEE f32 add, RNE) then the emitted
v_bfe/v_add/v_cmp_u/v_or/v_cndmask rounding, vs an exact rational-to-bf16 round-to-nearest-even reference."""
import random, struct, math, sys
from fractions import Fraction
def f32(bits): return struct.unpack('<f', struct.pack('<I', bits & 0xffffffff))[0]
def bits(f): return struct.unpack('<I', struct.pack('<f', f))[0]
def r32(x):  # double -> f32 RNE (C cast in struct.pack); overflow -> inf
    try: return struct.unpack('<f', struct.pack('<f', x))[0]
    except OverflowError: return math.copysign(math.inf, x)
def emit(x):  # x: f32 bits of the sum
    t = ((x >> 16) & 1) + 0x7fff
    t = (t + x) & 0xffffffff
    if math.isnan(f32(x)): r = x | 0x400000
    else: r = t
    return r >> 16
def ref(a16, b16):
    fa, fb = f32(a16 << 16), f32(b16 << 16)
    if math.isnan(fa) or math.isnan(fb): return 'nan'
    if math.isinf(fa) or math.isinf(fb):
        s = fa + fb
        return 'nan' if math.isnan(s) else bits(s) >> 16
    q = Fraction(fa) + Fraction(fb)
    # exact RNE to bf16 via enumeration of neighbours: use float64 exact sum then RNE on bits (sum of two bf16 is exact in f64)
    s64 = fa + fb
    pass  # q is the exact sum; s64 only seeds the candidate search
    b64 = struct.unpack('<Q', struct.pack('<d', s64))[0]
    # convert f64 -> bf16 RNE: go through exact rounding using numpy with ml_dtypes-free method
    best = None
    s32 = r32(s64)
    cand = []
    base = bits(s32) >> 16 if not math.isinf(s32) else 0x7f7f | (0x8000 if s32 < 0 else 0)
    for c in (base - 1, base, base + 1, base + 2):
        c &= 0xffff
        v = f32(c << 16)
        if math.isnan(v): continue
        cand.append((abs(Fraction(v) - q) if not math.isinf(v) else Fraction(10**400), c & 1, c))
    cand.sort()
    d0 = cand[0][0]
    ties = [c for c in cand if c[0] == d0]
    pick = min(ties, key=lambda c: c[1])[2] if len(ties) > 1 else ties[0][2]
    # overflow: RNE beyond max finite -> inf
    mx = Fraction(f32(0x7f7f << 16)); half_ulp = Fraction(2) ** (127 - 8)
    if abs(q) >= mx + half_ulp: pick = 0x7f80 | (0x8000 if q < 0 else 0)
    if q == 0: pick = 0x8000 if (a16 & 0x8000 and b16 & 0x8000) else 0
    return pick
random.seed(1); n = int(sys.argv[1]) if len(sys.argv) > 1 else 200000; bad = 0
special = [0, 0x8000, 0x7f7f, 0xff7f, 0x0001, 0x8001, 0x007f, 0x0080, 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fc0]
for i in range(n):
    a = random.choice(special) if i % 10 == 0 else random.getrandbits(16)
    b = random.choice(special) if i % 7 == 0 else (a ^ random.getrandbits(9)) if i % 3 == 0 else random.getrandbits(16)
    if ((a >> 7) & 0xff) == 0 and a & 0x7f or ((b >> 7) & 0xff) == 0 and b & 0x7f: continue  # denormal inputs: depend on MODE, see report
    s = bits(r32(f32(a << 16) + f32(b << 16)))
    got = emit(s); r = ref(a, b)
    if r == 'nan':
        ok = (got & 0x7f80) == 0x7f80 and (got & 0x7f) != 0
    else:
        ok = got == r
    if not ok:
        bad += 1
        if bad < 10: print('MISMATCH', hex(a), hex(b), hex(s), hex(got), r)
print({'samples': n, 'mismatches': bad})
