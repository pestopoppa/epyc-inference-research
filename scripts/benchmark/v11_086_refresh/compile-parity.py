"""Granted-build check: generated bench flags/link order equal pinned originals."""
import json
from pathlib import Path
import re
import shlex
import sys

original, generated, src = map(Path, sys.argv[1:4])
receipt = []
for target in ('llama-bench-impl', 'llama-bench'):
    old_dir = original / 'tools/llama-bench/CMakeFiles' / (target + '.dir')
    new_dir = generated / 'CMakeFiles' / (target + '.dir')
    def assignments(path):
        out = {}
        for line in path.read_text().splitlines():
            if ' = ' in line:
                k, v = line.split(' = ', 1); out[k] = shlex.split(v)
        return out
    old, new = assignments(old_dir / 'flags.make'), assignments(new_dir / 'flags.make')
    for field in ('CXX_DEFINES', 'CXX_FLAGS'):
        # Explicit GNU++17 is semantically identical to the archived computed
        # default17/extensionsON; no other added/missing option is accepted.
        a, b = old[field], new[field]
        if field == 'CXX_FLAGS':
            b = [x for x in b if x != '-std=gnu++17']
        assert a == b, (target, field, a, b)
    old_sources = re.search(r'CMAKE_HOME_DIRECTORY:INTERNAL=(.+)', (original / 'CMakeCache.txt').read_text()).group(1)
    def includes(items, root):
        return [str(Path(x[2:]).resolve().relative_to(Path(root).resolve())) for x in items]
    assert includes(old['CXX_INCLUDES'], old_sources) == includes(new['CXX_INCLUDES'], src)
    a, b = shlex.split((old_dir / 'link.txt').read_text()), shlex.split((new_dir / 'link.txt').read_text())
    assert a[0] == b[0] == '/usr/bin/g++-15'
    def dependencies(tokens):
        return [Path(x).name for x in tokens if re.search(r'lib.*\.(?:so(?:\.[0-9.]+)?|a)$', x)]
    # The -o output libbench is deliberately excluded from dependency sequence.
    def without_output(tokens):
        tokens = list(tokens); i = tokens.index('-o'); del tokens[i:i+2]; return tokens
    assert dependencies(without_output(a)) == dependencies(without_output(b)), (target, a, b)
    for tokens in (a, b):
        assert any('$ORIGIN' in x and 'rpath' in x for x in tokens), tokens
        if target == 'llama-bench-impl':
            assert '-shared' in tokens and '-fPIC' in tokens
            assert '-Wl,-soname,libllama-bench-impl.so' in tokens
    receipt.append({'target': target, 'defines': new['CXX_DEFINES'], 'flags': new['CXX_FLAGS'],
                    'includes': includes(new['CXX_INCLUDES'], src), 'link_dependencies': dependencies(without_output(b)), 'pass': True})
# Computed defaults are build-time evidence, not an assumption about compiler versions.
for root in (original, generated):
    files = list(root.glob('CMakeFiles/*/CMakeCXXCompiler.cmake'))
    assert len(files) == 1, files
    text = files[0].read_text()
    for expected in ('set(CMAKE_CXX_COMPILER "/usr/bin/g++-15")', 'set(CMAKE_CXX_COMPILER_VERSION "15.2.0")',
                     'set(CMAKE_CXX_STANDARD_COMPUTED_DEFAULT "17")', 'set(CMAKE_CXX_EXTENSIONS_COMPUTED_DEFAULT "ON")'):
        assert expected in text, (str(files[0]), expected)
print(json.dumps({'compile_and_link_parity': receipt, 'cxx_standard': 17, 'extensions': True, 'compiler': '/usr/bin/g++-15 15.2.0'}, indent=2))
