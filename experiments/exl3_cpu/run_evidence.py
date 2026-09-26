#!/usr/bin/env python3
"""Prospective EXL3 CPU verification/measurement using the shared native writer.

Build first; this wrapper loads and pins the writer, records the immutable read
set, executes the checker, then writes the resulting proposition. It cannot turn
an old stdout log into an acceptance record.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import struct
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--writer', type=Path, required=True)
    ap.add_argument('--writer-sha256', required=True)
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--blas-library', type=Path, required=True)
    ap.add_argument('--canonical-bindings', type=Path, required=True,
                    help='mapping of derived fixture filename to canonical artifact/fixture path and sha256')
    ap.add_argument('--canonical-root', type=Path, default=HERE.parents[1])
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--bench', action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    writer = args.writer.resolve()
    if sha(writer) != args.writer_sha256:
        raise SystemExit('writer digest mismatch')
    spec = importlib.util.spec_from_file_location('exl3_evidence', writer)
    e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e)
    package_spec=importlib.util.spec_from_file_location('exl3_cpu_shared',writer.parent/'__init__.py',submodule_search_locations=[str(writer.parent)])
    package=importlib.util.module_from_spec(package_spec);sys.modules[package_spec.name]=package;package_spec.loader.exec_module(package)
    contract=importlib.import_module('exl3_cpu_shared.contract')
    fixtures=importlib.import_module('exl3_cpu_shared.fixtures')
    binary, library = args.binary.resolve(), args.blas_library.resolve()
    manifest_path = HERE/'fixtures/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    bindings = json.loads(args.canonical_bindings.read_text())
    required = {'mul1_k3.bin', 'mul1_k4.bin', 'mcg_k4.bin'}
    if set(bindings) != required:
        raise SystemExit('all three derived real fixtures require canonical bindings')
    reads = [writer, writer.parent/'contract.py', writer.parent/'fixtures.py', writer.parent/'__init__.py', Path(__file__).resolve(), HERE/'exl3_cpu.cpp', HERE/'exl3_cpu.h',
             HERE/'test_cpu.cpp', manifest_path, args.canonical_bindings.resolve(), library]
    for name, digest in manifest['files'].items():
        path=HERE/'fixtures'/name
        if sha(path)!=digest: raise SystemExit('derived fixture digest mismatch: '+name)
        reads.append(path)
    for name, binding in bindings.items():
        if binding['derived_sha256'] != manifest['files'][name]:
            raise SystemExit('canonical binding has wrong derived fixture digest')
        for kind in ('artifact', 'fixture'):
            path=(args.canonical_root/binding[kind+'_path']).resolve()
            if sha(path)!=binding[kind+'_sha256']:raise SystemExit('canonical file digest mismatch')
            reads.append(path)
        artifact=(args.canonical_root/binding['artifact_path']).resolve()
        native,payload,_=fixtures.admit_real(artifact,args.canonical_root/binding['fixture_path'])
        if native['artifact_sha256']!=binding['canonical_artifact_sha256']:raise SystemExit('canonical artifact identity mismatch')
        if len(native['matrices'])!=1:raise SystemExit('one canonical matrix required')
        matrix=native['matrices'][0]
        raw=(HERE/'fixtures'/name).read_bytes()
        magic,bits,cb,n=struct.unpack_from('<4I',raw)
        if magic!=0x334c5845 or n!=128 or matrix['padded_shape']!=[128,128] or matrix['shape']!=[128,128] or matrix['K']!=bits or matrix['codebook']!=('mcg' if cb else 'mul1'):
            raise SystemExit('derived/canonical geometry or codebook mismatch')
        size=128*128*bits//8
        if payload[matrix['tensors']['trellis']['path']]!=raw[16:16+size]:raise SystemExit('derived trellis mismatch')
        for i,key in enumerate(('suh','svh')):
            floats=struct.unpack_from('<128f',raw,16+size+i*512)
            half=struct.pack('<128e',*floats)
            if payload[matrix['tensors'][key]['path']]!=half:raise SystemExit('derived scale mismatch')
        fixture=json.loads((args.canonical_root/binding['fixture_path']).read_text())
        weight=struct.unpack_from('<16384f',raw,16+size+1024+65536)
        if hashlib.sha256(struct.pack('<16384e',*weight)).hexdigest()!=fixture['reference']['transformed_fp16_sha256']:
            raise SystemExit('canonical golden differs from C++ fixture')
        reads.extend(artifact.parent/path for path in payload)
        reads.extend(artifact.parent.glob('*.json'))
    metadata = dict(platform=platform.platform(), cpu=Path('/proc/cpuinfo').read_text().split('\n\n')[0],
                    affinity=sorted(os.sched_getaffinity(0)), compiler=subprocess.check_output([os.environ.get('CXX','g++'),'--version'],text=True),
                    binary=str(binary), binary_sha256=sha(binary), library=str(library), library_sha256=sha(library),
                    writer_sha256=sha(writer), benchmark=args.bench, env={k:v for k,v in os.environ.items() if k.startswith('CPU_REGION_') or k.startswith('REGION_LOCK_')})
    if args.bench and len(metadata['affinity'])!=1:
        raise SystemExit('microbenchmark must run on exactly one explicitly claimed CPU')
    meta_path=args.output/'environment.json';meta_path.write_text(json.dumps(metadata,indent=2)+'\n');reads.append(meta_path.resolve())
    read_set=[{'path':str(p),'sha256':sha(p)} for p in sorted(set(reads))]
    checker_sha=sha(binary)
    cmd=[str(binary),str(HERE/'fixtures'),str(library)]+(['--bench'] if args.bench else [])
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1')
    start=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (args.output/'stdout.txt').open('w') as out,(args.output/'stderr.txt').open('w') as err:
        child=subprocess.Popen(cmd,stdout=out,stderr=err,env=env)
        samples=[]
        while child.poll() is None:
            # Observe only the child PID we own, during its actual execution window.
            try:
                samples.append({'monotonic_ns':time.monotonic_ns(),'pid':child.pid,
                                'stat':Path(f'/proc/{child.pid}/stat').read_text(),
                                'status':Path(f'/proc/{child.pid}/status').read_text()})
            except FileNotFoundError: pass
            time.sleep(.01)
        rc=child.returncode
    sample_path=args.output/'during-process.json';sample_path.write_text(json.dumps(samples)+'\n')
    stdout_path=args.output/'stdout.txt';stdout=stdout_path.read_text()
    # Refuse source, fixture, checker, library or writer drift across the run.
    if sha(binary)!=checker_sha or any(sha(d['path'])!=d['sha256'] for d in read_set):
        raise SystemExit('read set changed during execution')
    read_set += [{'path':str(p.resolve()),'sha256':sha(p)} for p in (stdout_path,args.output/'stderr.txt',sample_path)]
    identity=lambda name,obj:dict(id=name,sha256=e.digest(obj))
    identities=dict(model=identity('three real expert tile slices',manifest['fixtures']),
                    artifact=identity('canonical EXL3-1 bindings',bindings),source=identity('CPU implementation read set',read_set),
                    binary={'id':str(binary),'sha256':checker_sha},library={'id':str(library),'sha256':sha(library)},
                    toolchain=identity('C++17 baseline + target ISA functions',metadata['compiler']),
                    hardware=identity('host CPU/affinity',metadata),residency=identity('owned CPU process during execution',samples))
    common=dict(run_id='exl3-cpu-'+start,date=start,category='CANDIDATE',protocol_id='',protocol_eligible=False,
                arm='standalone-cpu',comparator='independent-codebooks-and-materialized-FP32-FMA-oracle',identities=identities,backend='cpu')
    proposition='Mandatory real MUL1 K3/K4 and MCG K4 reconstruction is exact; scalar/BW/VNNI/VBMI synthetic K1-K8 operators meet the checked parity, padding, routing, bias, guard and provider envelopes'
    passed=rc==0 and re.search(r'^PASS checks=\d+$',stdout,re.M) is not None and all(f'isa {i} supported=1' in stdout for i in range(4))
    fields=dict(common,schema=e.VERIFIER,claim=proposition,fixture='cpu-mandatory-real-and-synthetic-suite',
                path='standalone-experimental',decided_proposition=proposition,verdict='pass' if passed else 'fail',
                checker={'id':'exl3-cpu-test/v1','path':str(binary),'sha256':checker_sha},
                fixture_sha256=sha(manifest_path),read_set=read_set,read_set_sha256=e.digest(read_set))
    row=e.write(args.output/'native',fields);e.project(row)
    print('verifier',row['row_id'],row['verdict'])
    if args.bench and passed:
        begin=re.search(r'microbench_window_begin_ns=(\d+)',stdout)
        end=re.search(r'microbench_window_end_ns=(\d+)',stdout)
        if not begin or not end or not any(int(begin[1])<=s['monotonic_ns']<=int(end[1]) for s in samples):
            raise SystemExit('no residency sample overlaps the timed microbenchmark window')
        vectors={}
        for isa,grouped,rep,value in re.findall(r'isa=(\d+) grouped=(\d+) rep=(\d+) batch_us=([0-9.e+-]+)',stdout):
            vectors.setdefault((isa,grouped),[]).append(float(value))
        if len(vectors)!=8 or any(len(v)!=5 for v in vectors.values()):raise SystemExit('incomplete benchmark matrix')
        for (isa,grouped),raw in vectors.items():
            fields=dict(common,schema=e.MEASUREMENT,arm=f'isa{isa}-grouped{grouped}',operator='indexed-mul1-q8',
                        shape=[8,1,128,128],metric='batch_latency',value=math.fsum(raw)/len(raw),unit='us_per_eight_expert_batch',
                        metric_direction='lower_better',repetitions=len(raw),reps_basis='five batches of four timed invocations after one warmup',
                        raw_vector=raw,aggregation='arithmetic_mean',claim='Cache-resident experimental operator latency observation; no inference or promotion claim')
            row=e.write(args.output/'native',fields);e.project(row);print('measurement',row['row_id'])
    raise SystemExit(0 if passed else 1)

if __name__=='__main__':main()
