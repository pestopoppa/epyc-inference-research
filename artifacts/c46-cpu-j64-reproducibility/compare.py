import sys, json
sys.path.insert(0, '/mnt/raid0/llm/epyc-inference-research/scripts/kernel_rnd/autokernel/controller')
import anchor_integrity as A
J1 = '/mnt/raid0/llm/autokernel/campaigns/ak-ds41-cpu-decode-20260923/store/anchor-gen-002'
R = '/mnt/raid0/llm/tmp/c46-repro'
builds = {'j1-anchor-gen-002': J1, 'j64-a': f'{R}/j64-a', 'j64-b': f'{R}/j64-b'}
out = {'object_digest': {k: A.object_digest(v) for k, v in builds.items()}}
out['object_diff'] = {f'{a}__vs__{b}': A.object_diff(builds[a], builds[b]) for a, b in
                      [('j64-a', 'j64-b'), ('j1-anchor-gen-002', 'j64-a'), ('j1-anchor-gen-002', 'j64-b')]}
libs = ['libggml-cpu.so', 'libggml-base.so', 'libggml.so', 'libllama.so']
out['code_digest'] = {lib: {k: A.code_digest(f'{v}/bin/{lib}') for k, v in builds.items()} for lib in libs}
print(json.dumps(out, indent=1))
