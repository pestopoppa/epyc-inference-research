import hashlib, json, subprocess
from pathlib import Path
from scripts.kernel_rnd.autokernel import check_regrowth_guards as guard
p=Path("scripts/kernel_rnd/autokernel/loop_code_budget_baseline.json")
old=json.loads(subprocess.check_output(["git","show","d41be316c941097f0e54a3ec4147ef67d07afe36:"+str(p)]));new=json.loads(p.read_text())
ref="operator 2026-11-01 via ws58 AskUserQuestion: OAB-31 six snapshot ceilings"
limits={"actors.py":3037,"bestof.py":1135,"loop.py":1427,"pipeline.py":355,"pool.py":488,"run.py":6588}
assert old["exceptions"]=={}
assert {k:v for k,v in old.items() if k!="exceptions"}=={k:v for k,v in new.items() if k!="exceptions"}
assert new["exceptions"]=={k:{"ceiling":v,"approval":ref} for k,v in limits.items()}
e=json.loads(Path("data/engineering/oab31-review-20261101/budget-decision-evidence.json").read_text())
for f,h in e["sha256_by_changed_file"].items(): assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h, f
counts={f.name:guard.file_composition(f)[0] for f in Path("scripts/kernel_rnd/autokernel/loop").glob("*.py") if not f.name.startswith("test_")}
for name,limit in limits.items(): assert counts[name]<=limit,(name,counts[name],limit)
errors=guard.check_code_baseline(p,counts)
for error in errors: print(error)
assert len(errors)==31,len(errors)
assert all(not any(error.startswith(name+":") for name in limits) for error in errors)
print("approved entries exact; source snapshot verified; residual baseline guard categories:",len(errors))
