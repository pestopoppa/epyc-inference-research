import re, sys

log, s, e = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
txt = open(log).read()
# 'prompt eval time' also contains 'eval time' -> must exclude, or prefill rates
# (100-200 tok/s) contaminate the decode distribution.
lines = [l for l in txt.splitlines() if 'eval time =' in l and 'prompt eval' not in l]
rates = [float(m.group(1)) for l in lines for m in [re.search(r'([\d.]+) tokens per second', l)] if m]
toks = [int(m.group(1)) for l in lines for m in [re.search(r'/\s+(\d+) tokens', l)] if m]
acc = re.findall(r'draft acceptance\s*=\s*([\d.]+)', txt)
if rates:
    r = sorted(rates)
    print(f"  per-stream tok/s : n={len(r)} min={r[0]:.2f} median={r[len(r)//2]:.2f} max={r[-1]:.2f}")
    print(f"  AGGREGATE tok/s  : {sum(rates):.2f}")
    print(f"  wall={e-s:.1f}s  tokens={sum(toks)}  wall-derived agg={sum(toks)/(e-s):.2f}")
    if acc:
        a = [float(x) for x in acc]
        print(f"  draft acceptance : mean={sum(a)/len(a):.3f}")
else:
    print("  NO TIMINGS FOUND")
