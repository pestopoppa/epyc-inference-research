"""Fresh interpreter child of the lane supervisor; not a user entry point."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

_ALLOWED = {
    'scripts.kernel_rnd.autokernel.loop.serial_run',
    'scripts.kernel_rnd.autokernel.loop.run',
    'scripts.kernel_rnd.autokernel.loop.unified_driver',
    'scripts.kernel_rnd.autokernel.loop.campaign_service',
}


def _verify():
    module = sys.argv[1]
    if module not in _ALLOWED:
        raise RuntimeError('unsupported supervised entry')
    fd = int(os.environ.pop('AUTOKERNEL_WATCHDOG_FD'))
    state_path = os.environ.pop('AUTOKERNEL_WATCHDOG_STATE')
    expected_manifest = os.environ.pop('AUTOKERNEL_WATCHDOG_MANIFEST')
    original = json.loads(os.environ.pop('AUTOKERNEL_WATCHDOG_ORIGINAL_PYTHONPATH'))
    if original is None:
        os.environ.pop('PYTHONPATH', None)
    else:
        os.environ['PYTHONPATH'] = original
    try:
        if os.read(fd, 1) != b'1':
            raise RuntimeError('supervisor did not release child')
    finally:
        os.close(fd)
    state = json.loads(Path(state_path).read_text())
    manifest = state['manifest']
    if [str(Path(name).resolve()) for name in state['named_stores']] != state['resolved_stores']:
        raise RuntimeError('target store alias changed after release')
    body = dict(manifest)
    digest = body.pop('sha256')
    actual = hashlib.sha256(json.dumps(body, sort_keys=True,
                                       separators=(',', ':')).encode()).hexdigest()
    argv_digest = hashlib.sha256(json.dumps(sys.argv[2:], separators=(',', ':')).encode()).hexdigest()
    env_digest = hashlib.sha256(json.dumps(sorted(os.environ.items()),
                                           separators=(',', ':')).encode()).hexdigest()
    if (digest != expected_manifest or digest != actual or manifest['cwd'] != os.getcwd()
            or manifest['argv_sha256'] != argv_digest or manifest['environment_sha256'] != env_digest):
        raise RuntimeError('supervisor binding differs')
    own_path = str(Path(__file__).resolve())
    if own_path not in manifest['files']:
        raise RuntimeError('supervisor child source is not in the input manifest')
    for name, expected in manifest['files'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'supervisor source/input changed: {name}')
    return module, manifest, state


def main():
    try:
        module, manifest, state = _verify()
    except (OSError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as exc:
        print(f'supervised child refused: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 78
    from . import lane_watchdog
    lane_watchdog._SUPERVISED = True
    if module.endswith('.serial_run'):
        lane_watchdog.install_serial_capability(state)
    target = importlib.import_module(module)
    source = str(Path(target.__file__).resolve())
    if source not in manifest['files'] or hashlib.sha256(Path(source).read_bytes()).hexdigest() != manifest['files'][source]:
        print('supervised entry source is not bound', file=sys.stderr)
        return 78
    return target._main(sys.argv[2:])


if __name__ == '__main__':
    raise SystemExit(main())
