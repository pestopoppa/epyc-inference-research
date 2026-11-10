"""Model-free controls exercise real fresh-interpreter children in private fixtures."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

HERE = Path(__file__).parent
MODULE = 'scripts.kernel_rnd.autokernel.loop.watchdog_test_entry'


@pytest.fixture
def fixture(tmp_path):
    pkg = tmp_path / 'scripts/kernel_rnd/autokernel/loop'
    pkg.mkdir(parents=True)
    for directory in (pkg, *list(pkg.parents)[:3]):
        (directory / '__init__.py').touch()
    # Follow statically named internal imports from the true serial entry and
    # lifecycle helper; copy Python source only, with explicit size bounds.
    source_root = HERE.parent
    prefix = 'scripts.kernel_rnd.autokernel'
    queue = [prefix + '.loop.' + name for name in
             ('serial_run', 'lane_watchdog', 'lane_watchdog_child',
              'provider_factories', 'worker_lifecycle')]
    seen = set()
    copied_bytes = 0
    while queue:
        name = queue.pop()
        if name in seen or not (name == prefix or name.startswith(prefix + '.')):
            continue
        seen.add(name)
        relative = name[len(prefix):].lstrip('.').replace('.', '/')
        source = source_root / (relative + '.py') if relative else source_root / '__init__.py'
        if not source.is_file():
            source = source_root / relative / '__init__.py'
        if not source.is_file() or source.name.startswith('test_'):
            continue
        data = source.read_bytes()
        copied_bytes += len(data)
        if len(seen) > 1500 or copied_bytes > 64 * 1024 * 1024:
            raise AssertionError('native Python dependency closure exceeded fixture bound')
        destination = pkg.parent / source.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        package = name if source.name == '__init__.py' else name.rpartition('.')[0]
        for node in ast.walk(ast.parse(data, filename=str(source))):
            if isinstance(node, ast.Import):
                queue.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                try:
                    base = (importlib.util.resolve_name('.' * node.level + (node.module or ''), package)
                            if node.level else (node.module or ''))
                except (ImportError, ValueError):
                    continue
                queue.append(base)
                queue.extend(base + '.' + alias.name for alias in node.names if alias.name != '*')
    # Absolute package imports outside autokernel are required by the real
    # serial closure. Copy the exact tracked files; do not substitute policy.
    for relative in ("scripts/benchmark/belief_capture.py", "scripts/lib/canonical_recipe.py"):
        source = HERE.parents[3] / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    child = (HERE / 'lane_watchdog_child.py').read_text()
    def replace_once(text, needle, replacement):
        assert text.count(needle) == 1, f'fixture replacement expected exactly one: {needle!r}'
        return text.replace(needle, replacement, 1)
    child = replace_once(child, "'scripts.kernel_rnd.autokernel.loop.serial_run',",
                          "'scripts.kernel_rnd.autokernel.loop.serial_run',\n    "
                          "'scripts.kernel_rnd.autokernel.loop.watchdog_test_entry',")
    # In the private copy serial execution has no route to the native build or
    # model leaf. Fail before the release pipe unless the bound inert fixture and
    # exact copied serial source were both attested.
    child = replace_once(child,
        "    fd = int(os.environ.pop('AUTOKERNEL_WATCHDOG_FD'))",
        "    if module.endswith('.serial_run'):\n"
        "        if not os.environ.get('WATCHDOG_TEST_INERT'):\n"
        "            raise RuntimeError('inert serial leaf is mandatory')\n"
        "        here = Path(__file__).resolve()\n"
        "        serial = here.parent / 'serial_run.py'\n"
        "        if (str(here.parent) != os.environ['WATCHDOG_TEST_PACKAGE'] or\n"
        "            hashlib.sha256(here.read_bytes()).hexdigest() != os.environ['WATCHDOG_TEST_CHILD_SHA'] or\n"
        "            hashlib.sha256(serial.read_bytes()).hexdigest() != os.environ['WATCHDOG_TEST_SERIAL_SHA']):\n"
        "            raise RuntimeError('inert fixture identity differs')\n"
        "        Path(os.environ['WATCHDOG_TEST_TRACE']).write_text('inert-armed')\n"
        "    fd = int(os.environ.pop('AUTOKERNEL_WATCHDOG_FD'))")
    child = replace_once(child,
        "    target = importlib.import_module(module)",
        "    if os.environ.get('WATCHDOG_TEST_INERT') and module.endswith('.serial_run'):\n"
        "        return importlib.import_module('scripts.kernel_rnd.autokernel.loop.watchdog_test_entry')._main("
        "['signal', os.environ['WATCHDOG_TEST_INERT']])\n"
        "    target = importlib.import_module(module)")
    assert child.count("raise RuntimeError('inert serial leaf is mandatory')") == 1
    assert child.count("write_text('inert-armed')") == 1
    assert child.count("['signal', os.environ['WATCHDOG_TEST_INERT']])") == 1
    (pkg / 'lane_watchdog_child.py').write_text(child)
    (pkg / 'watchdog_test_entry.py').write_text('''\nimport os, signal, sys, time\nfrom pathlib import Path\ndef _main(argv):\n mode, marker = argv[:2]\n p = Path(marker)\n if mode == "alwayscrash":\n  p.write_text((p.read_text() if p.exists() else "") + "x")\n  signal.pidfd_send_signal(os.pidfd_open(os.getpid()), signal.SIGKILL)\n if mode == "crash" and not p.exists():\n  p.write_text("first")\n  signal.pidfd_send_signal(os.pidfd_open(os.getpid()), signal.SIGKILL)\n if mode == "changed":\n  p.write_text("changed")\n  signal.pidfd_send_signal(os.pidfd_open(os.getpid()), signal.SIGKILL)\n if mode == "orphan":\n  pid = os.fork()\n  if pid == 0:\n   os.setsid()\n   p.write_text(str(os.getpid()))\n   time.sleep(3)\n   os._exit(0)\n  while not p.exists(): time.sleep(.01)\n  signal.pidfd_send_signal(os.pidfd_open(os.getpid()), signal.SIGKILL)\n if mode == "factory":\n  from . import provider_factories\n  p.write_text(str(provider_factories.construct("fixture")))\n  return 0\n if mode == "signal" and os.environ.get("WATCHDOG_TEST_CAPABILITY_STORE"):\n  from . import lane_watchdog\n  import subprocess\n  store = os.environ["WATCHDOG_TEST_CAPABILITY_STORE"]\n  cap_env, fds = lane_watchdog.serial_child_capability(store)\n  command = "from scripts.kernel_rnd.autokernel.loop import lane_watchdog,run; import os,sys; assert os.environ.get(\\"PYTHONPATH\\") == os.environ[\\"WATCHDOG_EXPECT_PATH\\"]; run._main=lambda args: 0; lane_watchdog.supervise=lambda *a,**k: sys.exit(77); assert run.main([\\"--store\\",sys.argv[1]]) == 0"\n  subprocess.run([sys.executable, "-c", command, store], env=dict(os.environ, **cap_env), pass_fds=fds, check=True)\n if mode in ("signal", "ignore"):\n  signal.signal(signal.SIGTERM, signal.SIG_IGN if mode == "ignore" else lambda *_: sys.exit(0))\n  p.write_text(str(os.getpid()))\n  while True: time.sleep(.1)\n p.write_text((p.read_text() if p.exists() else "") + ":done")\n return 0\n''')
    (tmp_path / 'recovery.checkpoint').write_text('fixture checkpoint; native recovery is tested separately')
    script = tmp_path / 'run.py'
    script.write_text('''\nimport sys, tempfile\nfrom scripts.kernel_rnd.autokernel.loop import lane_watchdog\ntempfile.tempdir = sys.argv[1]\nlane_watchdog.STOP_GRACE_SECONDS = .1\nlane_watchdog.STOP_STUCK_NOTICE_SECONDS = .3\nmode, marker, key = sys.argv[2:5]\nif mode == "bad_uid":\n import os\n lane_watchdog._process_uid = lambda pid: os.getuid() + 1\nif mode == "bad_birth":\n from dataclasses import replace\n from scripts.kernel_rnd.autokernel.loop import worker_lifecycle\n original_identity = worker_lifecycle.process_identity\n count = [0]\n def changed_identity(pid):\n  count[0] += 1\n  value = original_identity(pid)\n  return replace(value, start_ticks=value.start_ticks + 1) if count[0] > 1 else value\n worker_lifecycle.process_identity = changed_identity\nif mode == "pre_release":\n import os, signal\n original = lane_watchdog._write\n def interrupted(*args):\n  original(*args)\n  signal.pidfd_send_signal(os.pidfd_open(os.getpid()), signal.SIGTERM)\n lane_watchdog._write = interrupted\ntry:\n rc = lane_watchdog.supervise("scripts.kernel_rnd.autokernel.loop.watchdog_test_entry",\n     [mode, marker], key=key, extra_files=([marker] if mode == "changed" else []),
     recovery_checkpoint=sys.argv[1] + '/recovery.checkpoint')\nexcept lane_watchdog.WatchdogRefused as exc:\n print("HELD:", exc, flush=True)\n sys.exit(75)\nsys.exit(rc)\n''')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(tmp_path), str(tmp_path / 'scripts/kernel_rnd'))), EPYC_ROOT_REPO=str(tmp_path))
    return tmp_path, script, env


def launch(fixture, mode, marker, key):
    root, script, env = fixture
    return subprocess.Popen([sys.executable, str(script), str(root), mode, str(marker), str(key)],
                            cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)


def signal_owned(process, signum):
    fd = os.pidfd_open(process.pid, 0)
    try:
        signal.pidfd_send_signal(fd, signum)
    finally:
        os.close(fd)


def test_normal_exit_does_not_restart(fixture):
    root = fixture[0]
    process = launch(fixture, 'normal', root / 'marker', root / 'store')
    assert process.wait(timeout=30) == 0
    assert (root / 'marker').read_text() == ':done'


def test_signal_crash_restarts_once(fixture):
    root = fixture[0]
    process = launch(fixture, 'crash', root / 'marker', root / 'store')
    assert process.wait(timeout=30) == 0
    assert (root / 'marker').read_text() == 'first:done'


def test_changed_original_document_holds(fixture):
    root = fixture[0]
    marker = root / 'config'
    marker.write_text('original')
    process = launch(fixture, 'changed', marker, root / 'store')
    stdout, _ = process.communicate(timeout=30)
    assert process.returncode == 75
    assert 'inputs changed' in stdout


def test_stop_is_terminal_and_same_store_is_exclusive(fixture):
    root = fixture[0]
    marker = root / 'marker'
    owner = launch(fixture, 'signal', marker, root / 'store')
    try:
        deadline = time.monotonic() + 20
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.05)
        assert marker.exists()
        import json
        import hashlib
        from .worker_lifecycle import process_identity
        store = str((root / 'store').resolve())
        digest = hashlib.sha256(store.encode()).hexdigest()
        state = json.loads((root / f'autokernel-lane-watchdog-{os.getuid()}' /
                            (digest + '.json')).read_text())
        assert state['process'] == process_identity(int(marker.read_text())).to_dict()
        assert state['uid'] == os.getuid()
        contender = launch(fixture, 'normal', root / 'other', root / 'store')
        stdout, _ = contender.communicate(timeout=30)
        assert contender.returncode == 75 and 'already supervised' in stdout
        signal_owned(owner, signal.SIGTERM)
        assert owner.wait(timeout=30) == 128 + signal.SIGTERM
        assert marker.read_text().isdigit()
    finally:
        if owner.poll() is None:
            signal_owned(owner, signal.SIGKILL)
            owner.wait(timeout=30)


def test_initialized_provider_is_refused_before_fork(monkeypatch):
    from . import unified_driver
    with pytest.raises(unified_driver.DriverRefused, match='HELD:.*registered trusted provider factory'):
        unified_driver.main(['--config', '/unused', '--listen', '127.0.0.1:8500'],
                            provider_registry=object())


def test_registered_factory_constructed_in_fresh_process(fixture):
    import hashlib
    root = fixture[0]
    pkg = root / 'scripts/kernel_rnd/autokernel/loop'
    source = pkg / 'factory_fixture.py'
    source.write_text('import os\ndef make():\n return os.getpid()\n')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    registry = pkg / 'provider_factories.py'
    registry.write_text(registry.read_text().replace(
        'REGISTERED_FACTORIES = {}',
        f"REGISTERED_FACTORIES = {{'fixture': ('factory_fixture.py', 'make', '{digest}')}}"))
    marker = root / 'factory-pid'
    process = launch(fixture, 'factory', marker, root / 'store')
    assert process.wait(timeout=30) == 0
    assert int(marker.read_text()) not in {os.getpid(), process.pid}


def test_detached_descendant_holds_recovery(fixture):
    root = fixture[0]
    marker = root / 'descendant'
    process = launch(fixture, 'orphan', marker, root / 'store')
    stdout, _ = process.communicate(timeout=30)
    assert process.returncode == 75
    assert 'descendants remain' in stdout
    assert marker.read_text().isdigit()


def _state(root, store):
    import hashlib
    digest = hashlib.sha256(str(store.resolve()).encode()).hexdigest()
    return root / f'autokernel-lane-watchdog-{os.getuid()}' / (digest + '.json')


def test_dead_adopted_orphan_is_not_a_survivor(fixture):
    root = fixture[0]
    pkg = root / 'scripts/kernel_rnd/autokernel/loop'
    entry = pkg / 'watchdog_test_entry.py'
    entry.write_text(entry.read_text().replace(
        ' if mode == "factory":',
        ' if mode == "deadorphan":\n'
        '  if os.fork() == 0:\n'
        '   if os.fork() == 0:\n'
        '    p.write_text("grandchild")\n'
        '    os._exit(0)\n'
        '   os._exit(0)\n'
        '  os.wait()\n'
        '  while not p.exists(): time.sleep(.01)\n'
        '  time.sleep(.5)\n'
        '  return 0\n'
        ' if mode == "factory":', 1))
    process = launch(fixture, 'deadorphan', root / 'marker', root / 'store')
    stdout, stderr = process.communicate(timeout=30)
    assert process.returncode == 0, stdout + stderr
    assert not _state(root, root / 'store').exists()


def test_held_state_is_archived_once_recorded_processes_are_gone(fixture):
    root = fixture[0]
    held = launch(fixture, 'orphan', root / 'descendant', root / 'store')
    stdout, _ = held.communicate(timeout=30)
    assert held.returncode == 75 and 'descendants remain' in stdout
    state = _state(root, root / 'store')
    original = state.read_text()
    assert '"adopted_live"' in original
    process = launch(fixture, 'normal', root / 'marker', root / 'store')
    stdout, stderr = process.communicate(timeout=30)
    assert process.returncode == 0, stdout + stderr
    assert 'prior state resolved' in stderr and not state.exists()
    archived = list(state.parent.glob(state.name + '.resolved-*'))
    assert len(archived) == 1 and archived[0].read_text() == original


def test_live_recorded_process_keeps_refusal_and_is_named(monkeypatch, tmp_path):
    import json
    from . import lane_watchdog
    from .worker_lifecycle import process_identity
    me = process_identity(os.getppid()).to_dict()
    gone = dict(me, pid=2 ** 22 - 3)
    state = tmp_path / 'state.json'
    state.write_text(json.dumps({'uid': os.getuid(), 'pgid': gone['pid'], 'process': gone,
                                 'supervisor': me, 'resolved_stores': [str(tmp_path / 's')]}))
    refusal = lane_watchdog._prior_state_refusal([state])
    assert refusal.startswith(lane_watchdog._UNRESOLVED_STATE)
    assert f"supervisor pid {me['pid']}" in refusal and state.exists()


def test_recycled_pid_is_not_a_survivor(tmp_path):
    import json
    from . import lane_watchdog
    from .worker_lifecycle import process_identity
    recycled = process_identity(os.getpid()).to_dict()
    recycled['start_ticks'] -= 1
    state = tmp_path / 'state.json'
    state.write_text(json.dumps({'uid': os.getuid(), 'pgid': recycled['pid'], 'process': recycled,
                                 'supervisor': recycled, 'resolved_stores': [str(tmp_path / 's')]}))
    assert lane_watchdog._prior_state_refusal([state]) is None


def test_malformed_state_keeps_refusing(tmp_path):
    from . import lane_watchdog
    state = tmp_path / 'state.json'
    state.write_text('{}')
    assert lane_watchdog._prior_state_refusal([state]) == lane_watchdog._UNRESOLVED_STATE


def test_restart_storm_is_bounded(fixture):
    root = fixture[0]
    marker = root / 'attempts'
    process = launch(fixture, 'alwayscrash', marker, root / 'store')
    stdout, _ = process.communicate(timeout=30)
    assert process.returncode == 75 and 'restart limit' in stdout
    assert marker.read_text() == 'xxx'


def test_factory_rejects_unregistered_and_closure(tmp_path, monkeypatch):
    from . import provider_factories
    with pytest.raises(ValueError, match='not registered'):
        provider_factories.source('absent')
    import hashlib
    source = tmp_path / 'invalid_factory.py'
    source.write_text('def outer():\n def make(): return 1\n return make\nmake = outer()\n')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(provider_factories, '__file__', str(tmp_path / 'provider_factories.py'))
    monkeypatch.setattr(provider_factories, 'REGISTERED_FACTORIES',
                        {'bad': ('invalid_factory.py', 'make', digest)})
    with pytest.raises(ValueError, match='top-level'):
        provider_factories.construct('bad')


def test_native_serial_and_standalone_share_the_real_target_store(fixture, monkeypatch):
    """Both real entrypoints resolve the same roster, then contend on that store."""
    import json
    import tempfile
    from . import lane_watchdog, serial_roster, serial_run, unified_driver
    from .test_serial_roster import _inputs
    from .test_standalone_inputs import _document

    root = fixture[0]
    roster_dir = root / 'roster'
    roster_dir.mkdir()
    resolved, owners, serial_args = _inputs(roster_dir, backends=('cpu',))
    owned_path = Path(serial_run.option(serial_args, '--owned-targets'))
    target_root = Path(serial_run.option(serial_args, '--state-dir')) / 'targets'
    backing_store = root / 'actual-target-store'
    backing_store.mkdir()
    alias = root / 'store-alias'
    alias.symlink_to(backing_store, target_is_directory=True)
    next(iter(owners.values()))['store'] = str(alias)
    owned_path.write_text(json.dumps(owners))
    native_targets, _, _ = serial_roster.build_targets(
        serial_run.option(serial_args, '--resolved-campaign'), owned_path,
        target_root=target_root, state_root=serial_run.option(serial_args, '--state-dir'))
    assert Path(serial_run.option(native_targets[0], '--store')).resolve() == backing_store.resolve()

    calls = []
    original = lane_watchdog.supervise
    monkeypatch.setattr(lane_watchdog, 'supervise',
                        lambda module, argv, **kwargs: calls.append((module, kwargs)) or 0)
    assert serial_run.main(serial_args) == 0
    config_dir = root / 'config'
    config_dir.mkdir()
    config = _document(config_dir)['driver_config']
    config.update(resolved_campaign_path=serial_run.option(serial_args, '--resolved-campaign'),
                  store_path=str(root / 'separate-campaign-store'), runtime_anchors={},
                  runtime_dimensions={}, profiles={}, experiment_plans={},
                  profile_requests={}, execution_inputs={})
    config_path = config_dir / 'driver.json'
    config_path.write_text(json.dumps(config))
    standalone_args = ['--config', str(config_path), '--listen', '127.0.0.1:8500',
                       '--owned-targets', str(owned_path), '--target-root', str(target_root)]
    assert unified_driver.main(standalone_args) == 0
    serial_keys = {Path(item).resolve() for item in calls[0][1]['keys']}
    standalone_keys = {Path(item).resolve() for item in calls[1][1]['keys']}
    assert backing_store.resolve() in serial_keys & standalone_keys
    assert (root / 'separate-campaign-store').resolve() in standalone_keys

    marker = root / 'native-lock-holder'
    runner = root / 'native-serial-runner.py'
    package = root / 'scripts/kernel_rnd/autokernel/loop'
    assert (package / 'serial_run.py').read_bytes() == (HERE / 'serial_run.py').read_bytes()
    assert (package / 'worker_lifecycle.py').read_bytes() == (HERE / 'worker_lifecycle.py').read_bytes()
    assert (package / 'lane_watchdog.py').read_bytes() == (HERE / 'lane_watchdog.py').read_bytes()
    trace = root / 'inert-pre-release.trace'
    runner.write_text("""
import hashlib, os, sys, tempfile
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from scripts.kernel_rnd.autokernel.loop import lane_watchdog, serial_run, worker_lifecycle
package = Path(sys.argv[1]) / 'scripts/kernel_rnd/autokernel/loop'
assert Path(serial_run.__file__).resolve() == (package / 'serial_run.py').resolve()
assert Path(worker_lifecycle.__file__).resolve() == (package / 'worker_lifecycle.py').resolve()
assert Path(lane_watchdog.__file__).resolve() == (package / 'lane_watchdog.py').resolve()
assert hashlib.sha256(Path(serial_run.__file__).read_bytes()).hexdigest() == sys.argv[4]
assert hashlib.sha256(Path(worker_lifecycle.__file__).read_bytes()).hexdigest() == sys.argv[5]
assert hashlib.sha256((package / 'lane_watchdog_child.py').read_bytes()).hexdigest() == sys.argv[6]
assert hashlib.sha256(Path(lane_watchdog.__file__).read_bytes()).hexdigest() == sys.argv[7]
tempfile.tempdir = sys.argv[1]
os.environ['WATCHDOG_TEST_INERT'] = sys.argv[2]
os.environ['WATCHDOG_TEST_TRACE'] = sys.argv[3]
os.environ['WATCHDOG_TEST_PACKAGE'] = str(package.resolve())
os.environ['WATCHDOG_TEST_SERIAL_SHA'] = sys.argv[4]
os.environ['WATCHDOG_TEST_CHILD_SHA'] = sys.argv[6]
raise SystemExit(serial_run.main(sys.argv[8:]))
""")
    env = dict(fixture[2], WATCHDOG_TEST_CAPABILITY_STORE=str(backing_store), WATCHDOG_EXPECT_PATH=fixture[2]['PYTHONPATH'])
    serial_sha = hashlib.sha256((HERE / 'serial_run.py').read_bytes()).hexdigest()
    lifecycle_sha = hashlib.sha256((HERE / 'worker_lifecycle.py').read_bytes()).hexdigest()
    child_sha = hashlib.sha256((package / 'lane_watchdog_child.py').read_bytes()).hexdigest()
    watchdog_sha = hashlib.sha256((HERE / 'lane_watchdog.py').read_bytes()).hexdigest()
    holder = subprocess.Popen([sys.executable, str(runner), str(root), str(marker), str(trace),
                               serial_sha, lifecycle_sha, child_sha, watchdog_sha, *serial_args], cwd=root, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 20
        while not marker.exists() and time.monotonic() < deadline:
            if holder.poll() is not None:
                stdout, stderr = holder.communicate()
                pytest.fail(f'serial owner exited before its inert leaf: stdout={stdout} stderr={stderr}')
            time.sleep(.05)
        assert marker.exists()
        assert trace.read_text() == 'inert-armed'
        monkeypatch.setattr(tempfile, 'tempdir', str(root))
        monkeypatch.setattr(lane_watchdog, 'supervise', original)
        with pytest.raises(lane_watchdog.WatchdogRefused, match='already supervised'):
            unified_driver.main(standalone_args)
        signal_owned(holder, signal.SIGTERM)
        assert holder.wait(timeout=30) == 128 + signal.SIGTERM
    finally:
        if holder.poll() is None:
            signal_owned(holder, signal.SIGKILL)
            holder.wait(timeout=30)


def test_signal_before_release_never_runs_child(fixture):
    root = fixture[0]
    marker = root / 'not-started'
    process = launch(fixture, 'pre_release', marker, root / 'store')
    assert process.wait(timeout=30) == 128 + signal.SIGTERM
    assert not marker.exists()


def test_alias_retarget_between_attempts_holds(fixture):
    root = fixture[0]
    first = root / 'first-store'
    second = root / 'second-store'
    first.mkdir()
    second.mkdir()
    alias = root / 'alias'
    alias.symlink_to(first, target_is_directory=True)
    marker = root / 'active-pid'
    owner = launch(fixture, 'signal', marker, alias)
    try:
        deadline = time.monotonic() + 20
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.05)
        assert marker.exists()
        child_fd = os.pidfd_open(int(marker.read_text()), 0)
        try:
            alias.unlink()
            alias.symlink_to(second, target_is_directory=True)
            signal.pidfd_send_signal(child_fd, signal.SIGKILL)
        finally:
            os.close(child_fd)
        stdout, _ = owner.communicate(timeout=30)
        assert owner.returncode == 75 and 'alias changed' in stdout
    finally:
        if owner.poll() is None:
            signal_owned(owner, signal.SIGKILL)
            owner.wait(timeout=30)


def test_unresponsive_owned_child_is_killed_and_reaped(fixture):
    root = fixture[0]
    marker = root / 'ignoring-pid'
    owner = launch(fixture, 'ignore', marker, root / 'store')
    try:
        deadline = time.monotonic() + 20
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.05)
        assert marker.exists()
        pid = int(marker.read_text())
        child_fd = os.pidfd_open(pid, 0)
        try:
            signal_owned(owner, signal.SIGTERM)
            assert owner.wait(timeout=60) == 128 + signal.SIGTERM
            record = next((root / f'autokernel-lane-watchdog-{os.getuid()}').glob('*.termination.json'))
            body = json.loads(record.read_text())
            assert body['reason'] == 'drain_grace_expired' and body['stop_clean'] is True
            assert body['kill_sent_at'] - body['term_sent_at'] >= body['drain_grace_s'] > 0
            assert body['exited_at'] >= body['kill_sent_at']
            import select
            poller = select.poll()
            poller.register(child_fd, select.POLLIN)
            assert poller.poll(1000), 'captured child pidfd is not terminal'
        finally:
            os.close(child_fd)
    finally:
        if owner.poll() is None:
            signal_owned(owner, signal.SIGKILL)
            owner.wait(timeout=30)


def test_missing_native_checkpoint_holds_after_first_crash(fixture):
    root = fixture[0]
    (root / 'recovery.checkpoint').unlink()
    marker = root / 'attempts-without-checkpoint'
    process = launch(fixture, 'alwayscrash', marker, root / 'store')
    stdout, _ = process.communicate(timeout=30)
    assert process.returncode == 75 and 'checkpoint unavailable' in stdout
    assert marker.read_text() == 'x'


@pytest.mark.parametrize('mode,reason', [('bad_uid', 'UID differs'),
                                         ('bad_birth', 'identity changed')])
def test_owned_process_identity_mismatch_refuses_before_release(fixture, mode, reason):
    root = fixture[0]
    marker = root / 'identity-mismatch-child'
    process = launch(fixture, mode, marker, root / 'store')
    stdout, _ = process.communicate(timeout=30)
    assert process.returncode == 75 and reason in stdout
    assert not marker.exists()


def test_initialized_provider_refuses_before_any_lane_action(monkeypatch):
    from . import lane_watchdog, unified_driver
    def forbidden(*args, **kwargs):
        raise AssertionError("lane action happened before migration refusal")
    monkeypatch.setattr(unified_driver, "load_config_document", forbidden)
    monkeypatch.setattr(lane_watchdog, "supervise", forbidden)
    with pytest.raises(unified_driver.DriverRefused, match="HELD:.*registered trusted provider factory"):
        unified_driver.main(["--config", "/unused", "--listen", "127.0.0.1:8500"], provider_registry=object())


def test_nonlisten_none_provider_preserves_existing_main(monkeypatch):
    from . import lane_watchdog, unified_driver
    monkeypatch.setattr(lane_watchdog, "supervise", lambda *a, **k: pytest.fail("non-listen was supervised"))
    seen = []
    monkeypatch.setattr(unified_driver, "_main", lambda argv, **kw: seen.append((argv, kw)) or 37)
    argv = ["--config", "/unused", "--dry-run"]
    assert unified_driver.main(argv) == 37
    assert seen == [(argv, {"provider_registry": None})]


def test_service_factory_refuses_before_any_lane_action(monkeypatch):
    from . import campaign_service, lane_watchdog
    monkeypatch.setattr(lane_watchdog, "supervise", lambda *a, **k: pytest.fail("lane started before refusal"))
    monkeypatch.setattr(campaign_service, "_main", lambda *a, **k: pytest.fail("lane started before refusal"))
    with pytest.raises(campaign_service.ControlRefused, match="HELD:.*registered trusted provider factory"):
        campaign_service.main(["--resolved-campaign", "/unused", "--store", "/unused", "--listen", "127.0.0.1:8500"], runtime_factory=object())


def test_direct_run_is_supervised_and_rejects_forged_marker(monkeypatch, tmp_path):
    from . import lane_watchdog, run
    calls = []
    monkeypatch.setattr(lane_watchdog, "_SUPERVISED", False)
    monkeypatch.setattr(lane_watchdog, "supervise", lambda *a, **k: calls.append((a, k)) or 23)
    monkeypatch.setattr(run, "_main", lambda *a, **k: pytest.fail("unowned run entered"))
    argv = ["--store", str(tmp_path / "store")]
    assert run.main(argv) == 23
    assert calls[0][0] == ("scripts.kernel_rnd.autokernel.loop.run", argv)
    assert calls[0][1]["max_restarts"] == 0
    monkeypatch.setenv("AUTOKERNEL_SERIAL_CAPABILITY", "1")
    with pytest.raises(lane_watchdog.WatchdogRefused, match="capability"):
        run.main(argv)
    assert len(calls) == 1


def test_serial_capability_requires_live_owner_and_inherited_held_lock(monkeypatch, tmp_path):
    import fcntl
    import hashlib
    import json
    import os
    from . import lane_watchdog, worker_lifecycle

    root = tmp_path / f"autokernel-lane-watchdog-{os.getuid()}"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(lane_watchdog.tempfile, "gettempdir", lambda: str(tmp_path))
    store = str((tmp_path / "store").resolve())
    digest = hashlib.sha256(store.encode()).hexdigest()
    state_path = root / (digest + ".json")
    lock_path = root / (digest + ".lock")
    serial = {"pid": 12341, "start_ticks": 1, "boot_id": "boot"}
    supervisor = {"pid": 12340, "start_ticks": 2, "boot_id": "boot"}
    class Identity:
        def __init__(self, row):
            self.row = row
        def to_dict(self):
            return self.row
    monkeypatch.setattr(worker_lifecycle, "process_identity",
                        lambda pid: Identity({12341: serial, 12340: supervisor}[pid]))
    monkeypatch.setattr(lane_watchdog, "_parent_pid", lambda pid: 12340)
    monkeypatch.setattr(lane_watchdog, "_process_uid", lambda pid: os.getuid())
    with lock_path.open("a+") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"uid": os.getuid(), "state_path": str(state_path), "process": serial,
                 "supervisor": supervisor, "stores": [store], "named_stores": [store],
                 "resolved_stores": [store], "lock_fds": {store: held.fileno()}}
        state_path.write_text(json.dumps(state))
        assert lane_watchdog._check_serial_capability(state, store, 12341) == held.fileno()
        with pytest.raises(lane_watchdog.WatchdogRefused, match="does not cover"):
            lane_watchdog._check_serial_capability(state, tmp_path / "other", 12341)
        monkeypatch.setattr(lane_watchdog, "_parent_pid", lambda pid: 99999)
        with pytest.raises(lane_watchdog.WatchdogRefused, match="not live"):
            lane_watchdog._check_serial_capability(state, store, 12341)
        monkeypatch.setattr(lane_watchdog, "_parent_pid", lambda pid: 12340)
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)
        with pytest.raises(lane_watchdog.WatchdogRefused, match="not held"):
            lane_watchdog._check_serial_capability(state, store, 12341)


def test_supervised_native_provider_factory_reaches_runtime_owner(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from . import campaign_service, lane_watchdog, provider_factories, standalone_inputs, unified_driver

    manifest = SimpleNamespace(driver_config=SimpleNamespace(
        resolved_campaign_path=str(tmp_path / "campaign.json"),
        store_path=str(tmp_path / "store"), config_generation=3))
    materialized = object()
    registry = object()
    runtime = object()
    seen = []
    monkeypatch.setattr(lane_watchdog, "_SUPERVISED", True)
    monkeypatch.setattr(unified_driver, "load_config_document", lambda path: {"schema": "fixture-startup"})
    monkeypatch.setattr(standalone_inputs.StartupManifest, "from_dict", lambda document: manifest)
    monkeypatch.setattr(standalone_inputs, "materialize", lambda value: materialized if value is manifest else None)
    monkeypatch.setattr(provider_factories, "construct", lambda name: registry if name == "fixture" else None)
    def native_factory(inputs, supplied_registry):
        assert inputs is materialized and supplied_registry is registry
        return runtime
    monkeypatch.setattr(standalone_inputs, "runtime_factory", native_factory)
    monkeypatch.setattr(campaign_service, "_main",
                        lambda argv, runtime_factory: seen.append((argv, runtime_factory)) or 31)
    result = unified_driver._main(["--config", str(tmp_path / "config.json"),
                                   "--provider-factory", "fixture", "--listen", "127.0.0.1:8500"])
    assert result == 31 and seen[0][1] is runtime
    assert seen[0][0][-2:] == ["--listen", "127.0.0.1:8500"]



def test_foreign_session_after_excludes_only_a_younger_foreign_led_session(monkeypatch):
    """2026-11-10: a non-dumpable operator sshd child (our uid, root-led session started after
    the recorded supervisor) is not a survivor; anything else unreadable still is."""
    import types
    from pathlib import Path as _P
    from autokernel.loop import lane_watchdog as lw
    stats = {"/proc/50/stat": "50 (sshd) S 40 40 40 0", "/proc/60/stat": "60 (x) S 1 60 60 0",
             "/proc/70/stat": "70 (y) S 41 41 41 0"}
    uids = {"/proc/40": 0, "/proc/41": 0}
    ticks = {40: 500, 41: 50}
    monkeypatch.setattr(lw, "_proc_stat", lambda pid: ("S", 1, pid, ticks[pid]) if pid in ticks else None)
    real_read, real_stat = _P.read_text, _P.stat
    monkeypatch.setattr(_P, "read_text", lambda self, *a, **k: stats[str(self)] if str(self) in stats else real_read(self, *a, **k))
    monkeypatch.setattr(_P, "stat", lambda self, *a, **k: types.SimpleNamespace(st_uid=uids[str(self)]) if str(self) in uids else real_stat(self, *a, **k))
    assert lw._foreign_session_after(50, 100) is True     # root leader younger than the supervisor
    assert lw._foreign_session_after(60, 100) is False    # its own session leader: unproven
    assert lw._foreign_session_after(70, 100) is False    # foreign leader OLDER: supervisor may share it
    uids["/proc/40"] = __import__("os").getuid()
    assert lw._foreign_session_after(50, 100) is False    # our-uid leader: could be a descendant
