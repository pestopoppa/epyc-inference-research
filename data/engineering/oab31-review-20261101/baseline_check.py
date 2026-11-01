"""Run original 94b1dd64 scratch and code checks against git-object bytes."""
import subprocess
import traceback
from pathlib import Path
from scripts.kernel_rnd.autokernel import check_regrowth_guards as growth

REV = "94b1dd64c163d915eb1347b118f57e44e981bf78"
ROOT = "scripts/kernel_rnd/autokernel/loop/"
SCRATCH = Path("/mnt/raid0/llm/tmp/oab-actor-backend/oab31-discard-codex3-20261101/baseline-scan")
SCRATCH.mkdir(exist_ok=True)

def source(path):
    return subprocess.check_output(["git", "show", f"{REV}:{path}"])

source_list = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", REV, ROOT], text=True).splitlines()
test_source = source(ROOT + "test_scratch.py")
namespace = {"__name__": "autokernel.loop.test_scratch_baseline", "__package__": "autokernel.loop",
             "__file__": str(SCRATCH / "test_scratch.py")}
exec(compile(test_source, namespace["__file__"], "exec"), namespace)
sites = {}
for path in source_list:
    relative = path.removeprefix(ROOT)
    if "/" in relative or not relative.endswith(".py") or relative.startswith("test_") or relative == "scratch.py":
        continue
    temp = SCRATCH / relative
    temp.write_bytes(source(path))
    for key, line in namespace["creation_sites"](temp):
        sites.setdefault(key, []).append(line)
    temp.unlink()
fixture = namespace["ScratchInventory"]()
fixture.maxDiff = None
fixture._sites = lambda: sites
for test in ("test_allowlist_has_no_stale_entries_and_every_reason_is_one_line",
             "test_every_creating_call_is_the_registry_or_reviewed_evidence"):
    print("\nBASELINE EXACT TEST", test, flush=True)
    try:
        getattr(fixture, test)()
    except AssertionError:
        traceback.print_exc()
    else:
        print("PASS")

baseline = SCRATCH / "baseline.json"
baseline.write_bytes(source("scripts/kernel_rnd/autokernel/loop_code_budget_baseline.json"))
counts = {}
for path in source_list:
    relative = path.removeprefix(ROOT)
    if not relative.endswith(".py") or Path(relative).name.startswith("test_"):
        continue
    temp = SCRATCH / "one.py"
    temp.write_bytes(source(path))
    counts[relative] = growth.file_composition(temp)[0]
    temp.unlink()
print("\nBASELINE GROWTH ERRORS", flush=True)
for error in growth.check_code_baseline(baseline, counts):
    print(error, flush=True)
print("BASELINE CODE TOTAL", sum(counts.values()), flush=True)
