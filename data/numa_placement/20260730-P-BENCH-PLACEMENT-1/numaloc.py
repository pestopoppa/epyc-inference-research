import re, sys

# /proc/<pid>/numa_maps lines carry per-node page counts as N<node>=<pages>.
# Sum them across all mappings, then report the share sitting on `own`.
pid, own = sys.argv[1], int(sys.argv[2])
tot = {}
for line in open(f"/proc/{pid}/numa_maps"):
    for node, pages in re.findall(r"\bN(\d+)=(\d+)", line):
        tot[int(node)] = tot.get(int(node), 0) + int(pages)
s = sum(tot.values())
if not s:
    print("  no pages"); sys.exit()
dist = " ".join(f"N{k}={100*v/s:5.1f}%" for k, v in sorted(tot.items()))
gb = s * 4096 / 1e9
print(f"  own=N{own} local={100*tot.get(own,0)/s:5.1f}%  [{dist}]  total={gb:.1f} GB")
