"""Direct actual-test runner retaining every fixture directory; no test deletion."""
import hashlib
import ast
import inspect
import traceback
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest import mock

D = Path(__file__).resolve().parent
OWNER = Path(os.environ['AK_TEST_OWNER'])
sys.path.insert(0, str(OWNER / 'scripts/kernel_rnd'))
source = D / 'final/scripts/kernel_rnd/autokernel/loop/test_store_anchor_dry_run.py'
if not source.is_file():
    source = D.parents[2] / 'scripts/kernel_rnd/autokernel/loop/test_store_anchor_dry_run.py'
assert source.is_file(), 'run from the private packet or committed artifact directory'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '51650790754941835fae30ed7fa7e85f43615fdfd98c6fac38c716dccf60ec64', 'test source differs from reviewed pin'
spec = importlib.util.spec_from_file_location('autokernel.loop.test_store_anchor_dry_run', source)
tests = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tests
spec.loader.exec_module(tests)
assert hashlib.sha256(Path(tests.run.__file__).read_bytes()).hexdigest() == '357b3551f135c82451f7fc96ebd3ec751417719406b9a38a34b7503eaf14f761', 'actual main differs from reviewed source pin'
retain_name = os.environ.get('AK_TEST_RETAIN_ROOT')
if not retain_name or not Path(retain_name).is_absolute():
    raise SystemExit('AK_TEST_RETAIN_ROOT must name a fresh absolute caller-owned non-Git directory')
RETAIN_ROOT = Path(retain_name).resolve()
if RETAIN_ROOT.exists() or RETAIN_ROOT.is_symlink():
    raise SystemExit('retained root already exists; choose a fresh path, never delete it')
if not RETAIN_ROOT.parent.is_dir() or RETAIN_ROOT.parent.stat().st_uid != os.getuid():
    raise SystemExit('retained root parent must already exist and be owned by this caller')
for ancestor in (RETAIN_ROOT.parent, *RETAIN_ROOT.parent.parents):
    marker = ancestor / '.git'
    if marker.exists() or marker.is_symlink():
        raise SystemExit('retained fixtures must be outside every Git worktree')
RETAIN_ROOT.mkdir(mode=0o700)
retained = []
real_mkdtemp = tempfile.mkdtemp
class RetainedDirectory:
    def __init__(self, *args, **kwargs):
        self.name = real_mkdtemp(prefix='fixture-', dir=RETAIN_ROOT)
        retained.append(self.name)
    def cleanup(self):
        pass
    def __enter__(self):
        return self.name
    def __exit__(self, *args):
        pass

deletion_attempts = []
def deny_delete(*args, **kwargs):
    deletion_attempts.append((args, kwargs))
    raise AssertionError('no-delete source-control runner refused Python deletion')

def run_cases(cases):
    with mock.patch.object(tempfile, 'TemporaryDirectory', RetainedDirectory), mock.patch.object(os, 'unlink', side_effect=deny_delete), mock.patch.object(os, 'remove', side_effect=deny_delete), mock.patch.object(os, 'rmdir', side_effect=deny_delete):
        for attested in cases:
            print('ACTUAL TEST START attested=', attested, flush=True)
            tests.test_actual_main_store_anchor_dry_run_is_read_only_and_attested(attested)
            print('ACTUAL TEST PASS attested=', attested, flush=True)

if os.environ.get('AK_TEST_MUTATE_STARTUP') == '1':
    module = ast.parse(inspect.getsource(tests.run.main))
    count = 0
    old = ast.parse("""verified_head = champion.verify_startup(
        worktree=args.worktree, branch=args.champion_branch,
        anchor_build=args.anchor_build,
        allow_unverified_anchor=args.allow_unverified_anchor or store_anchor,
        experimental_identity=experimental and not source_authoring and not store_anchor)
""").body[0]
    class OriginalStartup(ast.NodeTransformer):
        def visit_If(self, node):
            global count
            if (isinstance(node.test, ast.Name) and node.test.id == 'store_anchor'
                    and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                            and child.func.attr == 'warn_divergence' for child in ast.walk(node))
                    and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                            and child.func.attr == 'verify_startup' for child in ast.walk(node))):
                count += 1
                return old
            return self.generic_visit(node)
    module = OriginalStartup().visit(module)
    assert count == 1
    ast.fix_missing_locations(module)
    exec(compile(module, '834bef12-parent-startup-mutation', 'exec'), tests.run.__dict__)
    try:
        run_cases([True])
    except AssertionError as error:
        last = traceback.extract_tb(error.__traceback__)[-1]
        assert last.line and "'UNATTESTED' not in text" in last.line, (last, error)
        traceback.print_exc()
        print('EXPECTED SOURCE MUTATION RED: original store startup emits UNATTESTED under exact flag')
    else:
        raise AssertionError('original startup mutation escaped the exact-flag regression')
else:
    run_cases([True, False])

assert not deletion_attempts, deletion_attempts
print(json.dumps({'actual_main': True, 'exact_flag_both_cases': '--allow-unverified-anchor',
                  'cases': 1 if os.environ.get('AK_TEST_MUTATE_STARTUP') == '1' else 2,
                  'fixture_kind': 'synthetic-model-free-store',
                  'retained_directories': retained, 'python_deletion_guard': 'no_attempts',
                  'test_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}, indent=2))
