#!/usr/bin/env python3
"""Compare complete object manifests, failing closed on missing or malformed inputs."""
import hashlib
import json
from pathlib import Path
import sys


def manifest(root):
    root = Path(root)
    result = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.rglob('*.o')) if p.is_file()}
    if not result:
        raise ValueError('build has no objects')
    return result


def main():
    reference, build, expected, output = sys.argv[1:]
    original = json.loads(Path(reference).read_text())
    names = json.loads(Path(expected).read_text())
    if (not isinstance(original, dict) or not original or
        any(not isinstance(k, str) or not isinstance(v, str) or len(v) != 64 or
            any(c not in '0123456789abcdef' for c in v) for k, v in original.items())):
        raise ValueError('invalid reference manifest')
    if (not isinstance(names, list) or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names)):
        raise ValueError('expected object set must be a unique list of paths')
    current = manifest(build)
    missing = sorted(set(original) - set(current))
    added = sorted(set(current) - set(original))
    changed = sorted(k for k in original.keys() & current.keys() if original[k] != current[k])
    record = {'changed': changed, 'missing': missing, 'added': added,
              'expected': sorted(names), 'n_objects': len(current)}
    Path(output).write_text(json.dumps(record, indent=2) + '\n')
    return 0 if not missing and not added and changed == sorted(names) else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f'object identity failure: {exc}', file=sys.stderr)
        sys.exit(2)
