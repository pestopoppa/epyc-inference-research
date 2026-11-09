#!/usr/bin/python3
"""Seal the C9 W2 GPU census packet: rebuild-provenance, recipe, store binding, pins, row spec.

Usage: seal.py <packet_dir> <c9_worktree>. The packet already holds the copied probe/producer
files and the built instrument/libcensus-native.so. Model digests are hashed here, once,
outside any grant, and must equal the box's pins.
"""
import hashlib, json, subprocess, sys
from pathlib import Path

D = Path(sys.argv[1]).resolve()
SRC = Path(sys.argv[2]).resolve()
HERE = Path(__file__).resolve().parent
ROW = 'inf75-w2-gpu-census-c9-20261109'
COMMIT = 'd210f85c2cc8abb1c67cf79ed12582efd9e0026c'
STORE = Path('/mnt/raid0/llm/kernels/builds/gpu-20261109-d210f85c2-c9fold')
MDIR = Path('/mnt/raid0/llm/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF')
MODEL, MMPROJ = MDIR / 'Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf', MDIR / 'mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf'
BOX = {'model': '7bd52989f39ddca3dde36edb902fef72c5fde81e9211ca0f5ceca0119bd16d9c',
       'projector': 'a31d2c298f28735f6d27847851b8e3662ceb29cd6a1e66aa13d54b17f45aefca'}
HEADERS = ['include/llama.h', 'ggml/include/ggml.h', 'ggml/include/ggml-backend.h', 'tools/mtmd/mtmd.h']


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def dump(p, v):
    Path(p).write_text(json.dumps(v, indent=1, sort_keys=True) + '\n')


assert subprocess.check_output(['git', '-C', str(SRC), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
assert not subprocess.check_output(['git', '-C', str(SRC), 'status', '--porcelain', '--', *HEADERS], text=True).strip()
for name in ('job.py', 'work.py'):
    (D / name).write_bytes((HERE / name).read_bytes())
assert sha(D / 'instrument/native_trace_mm.c') == sha(HERE / 'native_trace_mm.c')
lib = D / 'instrument/libcensus-native.so'  # gpu_probe matches this exact basename in /proc/<pid>/maps
digests = {k: sha(p) for k, p in (('model', MODEL), ('projector', MMPROJ))}
assert digests == BOX, digests
dump(D / 'instrument/BUILD-PROVENANCE.json', {
    'schema': 'inf75.w2.c9.instrument-build.v1', 'source_commit': COMMIT, 'header_source_tree': str(SRC),
    'source_sha256': sha(D / 'instrument/native_trace_mm.c'), 'compiled_library_sha256': sha(lib),
    'compiler': subprocess.check_output(['gcc-15', '--version'], text=True).splitlines()[0],
    'compile_flags': ['-shared', '-fPIC', '-O2', '-Wall', '-Wextra', '-Werror', '-ldl', '-pthread'],
    'headers': {h: sha(SRC / h) for h in HEADERS},
    'note': 'all four headers byte-identical to the C5 headers of the 2026-10-29 tracer; compiled fresh against C9.'})
bind = D / 'store-binding'
bind.mkdir(exist_ok=True)
elf = [p for p in sorted((STORE / 'bin').iterdir()) if p.is_file() and not p.is_symlink() and p.name.startswith(('llama-server', 'libllama', 'libggml', 'libmtmd'))]
(bind / 'ELF.sha256').write_text(''.join(f'{sha(p)}  {p}\n' for p in elf))
meta = [STORE / n for n in ('BUILD-IDENTITY.json', 'BUILD-OK.json', 'CMakeCache.txt', 'PROVENANCE.md', 'SHA256SUMS', 'VERSION.txt')]
(bind / 'build-sources.sha256').write_text(''.join(f'{sha(p)}  {p}\n' for p in meta))
(bind / 'projector.sha256').write_text(f"{digests['projector']}  {MMPROJ}\n")  # 1 GiB projector: capture retains its digest, not a copy
dump(bind / 'BINDING.json', {'schema': 'inf75.w2.c9.store-binding.v1', 'source_commit': COMMIT, 'store_build': str(STORE),
                             'llama_server_sha256': sha(STORE / 'bin/llama-server'), 'no_build': True, 'instrument_sha256': sha(lib)})
(bind / 'BINDING_COMPLETE').write_text(f'BINDING_COMPLETE source={COMMIT} store={STORE} no-build=true\n')
dump(D / 'recipe-provenance.json', {'schema': 'inf75.w2.c9.recipe-binding.v1', 'carried_from': '/mnt/raid0/llm/tmp/codex-ni-inf75-w2-gpu-census-20261029/recipe.json',
                                    'changed': ['binary/build_dir/LD_LIBRARY_PATH to C9 store', 'instrument path', 'HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0 (ROCr thread affinity, compute-fillers.md)'], 'model_sha256': digests['model'], 'projector_sha256': digests['projector']})
env = {'PATH': '/usr/bin:/bin', 'LD_LIBRARY_PATH': f'{STORE}/bin:/opt/rocm/lib:/opt/AMD/aocc-compiler-5.0.0/lib', 'LD_PRELOAD': str(lib),
       'OMP_NUM_THREADS': '2', 'OMP_DYNAMIC': 'false', 'OMP_PROC_BIND': 'spread', 'OMP_PLACES': 'cores', 'KMP_BLOCKTIME': '10',
       'GGML_IQK': '1', 'LLAMA_MEDIA_MARKER': '<__media__>', 'HSA_OVERRIDE_CPU_AFFINITY_DEBUG': '0'}
binary = f'{STORE}/bin/llama-server'
P = {'id': ROW, 'scope': 'AKX-P0d W2 ROCm0 single-slot exact model+projector structural census on C9; prefill512/decode1, no speculation, no benchmark/quality claim',
     'binary': binary, 'build_dir': str(STORE), 'model': str(MODEL), 'projector': str(MMPROJ), 'port': 25182, 'physical_cpus': '72-95', 'timeout_s': 300,
     'request_files': {'prefill': str(D / 'requests/prefill.json'), 'decode': str(D / 'requests/decode.json')},
     'argv': [binary, '-m', str(MODEL), '--mmproj', str(MMPROJ), '--host', '127.0.0.1', '--port', '25182', '-np', '1', '-c', '16384', '-t', '2', '-ub', '512',
              '--flash-attn', 'on', '-ctk', 'q8_0', '-ctv', 'q8_0', '--device', 'ROCm0', '-ngl', 'all', '--image-min-tokens', '1024', '--cache-ram', '0',
              '--log-verbosity', '4', '--slot-save-path', '__RUN_ROOT__/runtime-slots'],
     'env': env, 'expected_ops': ['MUL_MAT', 'MUL_MAT_ID'], 'kernel_commit': COMMIT, 'instrument_library': str(lib),
     'foreign_load': str(D / 'foreign_load.py'),
     'capture_inputs': {
         'binary': binary, 'build_manifest': str(STORE / 'SHA256SUMS'), 'model': str(MODEL), 'projector_digest': str(bind / 'projector.sha256'), 'recipe': str(D / 'recipe.json'),
         'binding': str(D / 'recipe-provenance.json'), 'producer': str(D / 'producer/autokernel/loop/census.py'), 'driver': str(D / 'work.py'),
         'capture_module': str(D / 'producer/autokernel/loop/census_capture.py'), 'residency': str(D / 'producer/autokernel/loop/residency.py'),
         'workload_contract': str(D / 'producer/autokernel/controller/workload_contract.py'), 'build_receipt_binding': str(bind / 'BINDING.json'),
         'build_complete': str(bind / 'BINDING_COMPLETE'), 'build_log': str(STORE / 'PROVENANCE.md'), 'elf_manifest': str(bind / 'ELF.sha256'),
         'build_job': str(bind / 'BINDING.json'), 'build_sources': str(bind / 'build-sources.sha256'), 'source_manifest': str(D / 'CAPTURE-SOURCES.sha256'),
         'native_instrument': str(lib), 'instrument_provenance': str(D / 'instrument/BUILD-PROVENANCE.json')}}
dump(D / 'recipe.json', P)
dump(D / 'pins.json', {'binary_sha256': sha(binary), 'build_manifest_sha256': sha(STORE / 'SHA256SUMS'), 'model_sha256': digests['model'], 'projector_sha256': digests['projector'],
                       'model_size': MODEL.stat().st_size, 'projector_size': MMPROJ.stat().st_size, 'instrument_sha256': sha(lib)})
launch = D / 'launch.sh'
launch.write_text(f'''#!/bin/bash
# AKX-P0d W2 GPU structural census on champion C9 {COMMIT[:9]}, kernel-store GPU build.
set -euo pipefail
D={D}
cd "$D"
/usr/bin/sha256sum --check --status "$D/SOURCES.sha256"
exec /bin/bash /mnt/raid0/llm/tmp/compute-flags/bin/rowflow/filler_run15.sh --id {ROW} \\
  --class gpu --role bench-gpu --gpu-threads 184-185 --min-window 900 --port 25182 \\
  --cmd "/usr/bin/python3 -B $D/job.py" \\
  --complete-cmd "test -s $D/RESULT.json" \\
  --stop "$D/STOP" --workdir "$D"
''')
launch.chmod(0o755)
core = ['job.py', 'work.py', 'gpu_probe.py', 'gpu_mm_evidence.py', 'server_evidence_full.py', 'affinity_scope.py', 'interval_guard.py', 'w2_microbatch.py', 'foreign_load.py', 'resource_watch.py',
        'recipe.json', 'recipe-provenance.json', 'pins.json', 'requests/prefill.json', 'requests/decode.json', 'requests-prefill.json', 'requests-decode.json', 'instrument/native_trace_mm.c',
        'instrument/libcensus-native.so', 'instrument/BUILD-PROVENANCE.json', 'store-binding/BINDING.json', 'store-binding/BINDING_COMPLETE', 'store-binding/ELF.sha256', 'store-binding/build-sources.sha256', 'store-binding/projector.sha256']
prod = [f'producer/autokernel/{x}' for x in ('loop/census.py', 'loop/census_capture.py', 'loop/residency.py', 'loop/serving_server_owner.py', 'controller/workload_contract.py')]
(D / 'CAPTURE-SOURCES.sha256').write_text(''.join(f'{sha(D / f)}  {D / f}\n' for f in core + prod))
everything = sorted(p for p in D.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'SOURCES.sha256' and not p.name.endswith('.pyc')
                    and p.relative_to(D).parts[0].split('-')[0] not in ('smoke', 'run') and p.name not in ('STOP', 'RESULT.json', 'LAST-RESULT.json', '.smoke-ok'))
hdr = [SRC / h for h in HEADERS]
(D / 'SOURCES.sha256').write_text(''.join(f'{sha(p)}  {p}\n' for p in everything + hdr))
dump(D / 'spec.json', {
    'id': ROW, 'owner': 'autokernel', 'launcher': str(launch), 'lane': 'gpu', 'regions': 'mi210_0,184-185', 'result_path': str(D / 'RESULT.json'), 'result_format': 'json',
    'decision_rule': 'INF-75 AKX-P0d W2 structural census re-pinned to champion C9 d210f85c2 (store build gpu-20261109-d210f85c2-c9fold, llama-server 10578), Qwen3-VL-30B-A3B-Instruct Q4_K_M + F16 mmproj (full digests verified against the box pins at seal), np1, no speculation, real multimodal path, ROCm0, tracer libcensus-native.so rebuilt against C9 headers. Predeclared: accept only rc=0 with OBSERVED native prefill512 and decode1 facets, each with >=2 request-contained same-PID VRAM/KFD/foreign-load samples, verified capture journals. Anything else is INCONCLUSIVE. Structural witness only: no rate, quality, promotion, full-W2 or full-P0d claim. Stop on answer, or on seal/custody/disk/device refusal, STOP, or deadline.',
    'estimate_s': 600, 'min_window_s': 900, 'ram_gib': 64, 'handoff_task': 'handoffs/active/autokernel-akx-program.md:AKX-P0d-W2', 'supersedes': 'inf75-w2-gpu-prospective-20261102', 'priority_rank': 38,
    'smoke_predicate': f"/usr/bin/python3 -I -c 'import glob,json,os,sys; t=int(os.environ[\"FILLER_SMOKE_T0\"]); ok=[p for p in glob.glob(\"{D}/smoke-*/SMOKE-RESULT.json\") if os.path.getmtime(p)>t and json.load(open(p)).get(\"all_phases_observed\") and json.load(open(p)).get(\"cleanup_verified\")]; sys.exit(0 if ok else 1)'",
    'first_progress_s': 420, 'keep': ['*.py', '*.sh', '*.json', '*.sha256', 'producer/**', 'instrument/**', 'store-binding/**', 'requests/**', 'smoke-*/**', 'run-*/**'], 'gpu_use': 'exclusive'})
print('sealed', sha(D / 'SOURCES.sha256'))
