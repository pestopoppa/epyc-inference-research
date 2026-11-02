import importlib.util, json, os, subprocess
from pathlib import Path
p='/mnt/raid0/llm/worktrees/rtg57-fa4b-accounting-20261102/scripts/benchmark/kvu19a2_immutable_build.py'
s=importlib.util.spec_from_file_location('custody',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
src=Path('/dev/shm/kvu19a2-sync-source-20261102'); out=Path('/dev/shm/kvu19a2-sync-build-20261102')
assert os.geteuid()!=0 and m.file_sha(src/'ggml/src/ggml-cuda/fattn-tile.cuh')=='e66f5dc8c7a1ff8efd1bad30c0154a8741dbf204833e0f53ca172ab21809062a'
assert m.file_sha('/dev/shm/kvu19a2-sync-source-20261102.tar')=='34251fa02113cbccc5a2d911bc6d0274ef2730bd31474075aa1129772fb65559'
out.mkdir(exist_ok=False); before=m.inventory(src); (out/'inventory.json').write_text(json.dumps(before,sort_keys=True,separators=(',',':'))+'\n')
assert m.file_sha(m.TOOLS)==m.TOOLS_SHA256
subprocess.run(['/usr/bin/sha256sum','-c','--quiet',str(m.TOOLS)],check=True)
env={'PATH':'/opt/rocm/bin:/usr/bin:/bin','ROCM_PATH':'/opt/rocm','HIP_PATH':'/opt/rocm','CXX':'/usr/bin/g++','CC':'/usr/bin/gcc','HIPCXX':'/opt/rocm/llvm/bin/clang++','HOME':str(out)}
cmd=['/usr/bin/cmake','-S',str(src),'-B',str(out/'build'),'-DCMAKE_BUILD_TYPE=Release','-DGGML_HIP=ON','-DAMDGPU_TARGETS=gfx90a','-DLLAMA_BUILD_TESTS=ON','-DLLAMA_BUILD_SERVER=OFF','-DLLAMA_BUILD_TOOLS=OFF']
with (out/'configure.log').open('w') as f: subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
cache=(out/'build/CMakeCache.txt').read_text(); assert f'CMAKE_HOME_DIRECTORY:INTERNAL={src}' in cache and 'GGML_HIP:BOOL=ON' in cache
with (out/'build.log').open('w') as f: subprocess.run(['/usr/bin/cmake','--build',str(out/'build'),'--target','test-backend-ops','--parallel','24'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
assert before==m.inventory(src) and m.file_sha(m.TOOLS)==m.TOOLS_SHA256
subprocess.run(['/usr/bin/sha256sum','-c','--quiet',str(m.TOOLS)],check=True)
files=['build/bin/test-backend-ops','build/bin/libggml-hip.so','build/CMakeCache.txt','configure.log','build.log','inventory.json']
receipt={'commit':'18d030309a57071c32ffd0db42994ae473d2722a','source_archive_sha256':m.file_sha('/dev/shm/kvu19a2-sync-source-20261102.tar'),'tile_sha256':m.file_sha(src/'ggml/src/ggml-cuda/fattn-tile.cuh'),'toolchain_catalog_sha256':m.file_sha(m.TOOLS),'outputs':{f:m.file_sha(out/f) for f in files},'configure':cmd,'scope':'root owned, read-only to build user; transitive toolchain not fully captured'}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt,indent=2))
