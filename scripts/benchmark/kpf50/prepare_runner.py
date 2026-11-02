#!/usr/bin/env python3
"""Reproduce prospective KPF50 runner from an exact retained KPF18 image."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-dir',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    here=Path(__file__).resolve().parent; source=Path(a.source_dir); out=Path(a.out)
    cfg=json.loads((here/'runner-source.json').read_text()); inp=source/'kpf18_gate.py'
    if digest(inp)!=cfg['source_sha256'] or digest(source/'kpf18_arms.py')!=cfg['arms_sha256']:
        ap.exit(2,'source runner or arms hash mismatch\n')
    out.mkdir(parents=True,exist_ok=True); target=out/'kpf18_gate.py'
    if target.exists(): ap.exit(2,'output already exists\n')
    try:
        subprocess.run(['patch','--silent','--output',str(target),str(inp),str(here/'runner.patch')],check=True)
        if digest(target)!=cfg['result_sha256']: raise ValueError('prepared runner hash mismatch')
        shutil.copy2(source/'kpf18_arms.py',out/'kpf18_arms.py')
        shutil.copy2(here/'hsa_clock.py',out/'hsa_clock.py')
        print(json.dumps({'runner_sha256':digest(target),'arms_sha256':digest(out/'kpf18_arms.py'),
                          'clock_sha256':digest(out/'hsa_clock.py')},sort_keys=True))
    except (subprocess.CalledProcessError,ValueError) as e:
        target.unlink(missing_ok=True);ap.exit(2,f'runner preparation failed: {e}\n')
if __name__=='__main__': main()
