import re, glob, os, sys

# Line-by-line, so "prompt eval time =" can never be mistaken for "eval time =".
# The earlier parser used a look-back window that cut between "prompt " and
# "eval", which let every prefill line masquerade as a decode line.
def parse(path):
    prefill = decode = acc = None
    ptok = dtok = None
    for l in open(path, errors="ignore"):
        if "prompt eval time =" in l:
            m = re.search(r"/\s+(\d+) tokens \([^)]*?([\d.]+) tokens per second", l)
            if m: ptok, prefill = int(m.group(1)), float(m.group(2))
        elif "eval time =" in l:
            m = re.search(r"/\s+(\d+) tokens \([^)]*?([\d.]+) tokens per second", l)
            if m: dtok, decode = int(m.group(1)), float(m.group(2))
        elif "draft acceptance" in l:
            m = re.search(r"draft acceptance\s*=\s*([\d.]+)", l)
            if m: acc = float(m.group(1))
    return ptok, prefill, dtok, decode, acc

# Resident context depth. NOT the "prompt eval time" token count: that is the
# per-request *incremental* prefill, so when rep 1 times out before logging one,
# a later cache-warm request reports a tiny count (INF-45: prompt=2926 where the
# true resident context was 27,586). llama-server logs the slot's resident
# n_tokens at "stop processing: n_tokens = N". Returns None (never the prompt
# eval count) when no such line exists.
STOP_RE = re.compile(r"stop processing:\s*n_tokens\s*=\s*(\d+)")

def parse_depth(path):
    vals = []
    for l in open(path, errors="ignore"):
        m = STOP_RE.search(l)
        if m: vals.append(int(m.group(1)))
    return max(vals) if vals else None

ORDER = ["p0k5", "p8k", "p32k", "p128k"]
def main():
    roles = {}
    for f in glob.glob("/mnt/raid0/llm/tmp/cc_*.log"):
        b = os.path.basename(f)[3:-4]
        for p in ORDER:
            if b.endswith("_" + p):
                roles.setdefault(b[: -(len(p) + 1)], {})[p] = f

    for role in sorted(roles):
        print(f"##### {role} #####")
        base = None
        for p in ORDER:
            f = roles[role].get(p)
            if not f: continue
            ptok, pre, dtok, dec, acc = parse(f)
            if dec is None:
                print(f"  {p:6}: no decode timing (cell incomplete)"); continue
            if base is None: base = dec
            depth = parse_depth(f)
            depth_s = str(depth) if depth is not None else "UNKNOWN"
            drop = f"{100*(dec/base-1):+5.0f}%" if base else "    —"
            print(f"  {p:6}: depth={depth_s:>7} tok  prefill={pre:7.2f}  "
                  f"DECODE={dec:6.2f} tok/s  {drop}"
                  + (f"  accept={acc:.3f}" if acc is not None else "  spec=off"))

if __name__ == "__main__":
    main()
