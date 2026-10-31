"""Model-free actual-main startup: retained store bytes and strict store ancestry."""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
import sys
import io
import json
from pathlib import Path
from unittest import mock

import pytest

from . import actor_wire_gate, floor_registry, run
from .test_existing_cpu_run import test_existing_main_cpu_five_iterations_preserves_canonical_champion as cpu_fixture
from .test_floor_registry import MASTER_TEXT
from . import test_promotion_targets as promotion_fixture


def snapshot(root):
    result = {'.': ('directory', root.lstat().st_mtime_ns)}
    for path in sorted(root.rglob('*')):
        stat = path.lstat()
        if path.is_symlink():
            value = ('symlink', str(path.readlink()))
        elif path.is_dir():
            value = ('directory', stat.st_mtime_ns)
        else:
            value = ('file', stat.st_mtime_ns, path.read_bytes())
        result[str(path.relative_to(root))] = value
    return result


@pytest.mark.parametrize('attested', [True, False])
def test_actual_main_store_anchor_dry_run_is_read_only_and_attested(attested):
    real_main = run.main
    original_backend = run.actors.backend_for
    receipts = []
    original_fixture = promotion_fixture.TheKeepBuildsAProductionCompleteAnchor

    class StoreImageFixture(original_fixture):
        def setUp(self):
            super().setUp()
            # A store image has a cpu-* name, not the pool's anchor-gen-* contract.
            # Construct it directly, preserving the fixture's legacy anchor too.
            previous = self.startup_anchor
            self.startup_anchor = self.store / 'cpu-synthetic-store'
            (self.startup_anchor / 'bin').mkdir(parents=True)
            (self.startup_anchor / 'bin/libggml-cpu.so').write_bytes(
                (previous / 'bin/libggml-cpu.so').read_bytes())
            assert not (self.startup_anchor / 'provenance.json').exists()

    class AttestationRefused(Exception):
        pass

    def invoke(argv):
        def arg(name):
            return Path(argv[argv.index(name) + 1])
        store, anchor, tree = arg('--store'), arg('--anchor-build'), arg('--worktree')
        # Convert the fixture's synthetic build into a store image. All model,
        # kernel, timing and actor observations remain the fixture's doubles.
        assert anchor.name.startswith('cpu-')
        assert not (anchor / 'provenance.json').exists()
        head = run._git(tree, 'rev-parse', 'HEAD')
        (anchor / 'VERSION.txt').write_text(f'version: 1 ({head[:9] if attested else "0" * 9})\nbuilt with GNU 15.2.0 for Linux x86_64\n')
        (anchor / 'SHA256SUMS').write_text('fixture build inventory\n')
        (anchor / 'PROVENANCE.md').write_text('synthetic store fixture\n')
        master = store.parent / 'registry.yaml'
        master.write_text(MASTER_TEXT)
        (store / 'floor-registry-lookups.jsonl').write_text('{"previous":"retained"}\n')
        before = snapshot(store)
        output, errors = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(output), redirect_stderr(errors), mock.patch.object(
                    floor_registry, 'equivalence_for_build', return_value={'equivalence_key': 'e' * 64}), mock.patch.object(
                    run._FloorRegistryContext, 'ensure_preflight', return_value={'status': 'skipped'}), mock.patch.object(
                    run.actors, 'backend_for', side_effect=lambda model, effort:
                        replace(original_backend(model, effort), binary=sys.executable)):
                extra = ['--floor-registry', str(master), '--floor-registry-role', 'ds41_test_role',
                         '--serving-instrument', 'matched_process_v2', '--serving-pairs', '24',
                         '--cpu-calibrate-serving', '24', '--allow-unverified-anchor']
                actual_argv = [*argv, *extra]
                assert '--dry-run' in actual_argv
                assert actual_argv.count('--allow-unverified-anchor') == 1
                print('ACTUAL_MAIN_EXACT_FLAG_ARGV', json.dumps(actual_argv))
                if attested:
                    try:
                        assert real_main(actual_argv) == 0
                    except SystemExit as exc:
                        raise AssertionError(errors.getvalue().splitlines()[-1]) from exc
                else:
                    with pytest.raises(SystemExit) as refused:
                        real_main(actual_argv)
                    assert refused.value.code == 2
                    assert 'REFUSED store anchor' in errors.getvalue()
        finally:
            assert snapshot(store) == before
            print(output.getvalue())  # preserve the enclosing legacy fixture contract
            print("ACTUAL_MAIN_STDOUT attested=", attested, file=sys.__stdout__)
            print(output.getvalue(), file=sys.__stdout__)
            print("ACTUAL_MAIN_STDERR attested=", attested, file=sys.__stderr__)
            print(errors.getvalue(), file=sys.__stderr__)
        text = output.getvalue()
        assert 'UNATTESTED' not in text
        if attested:
            assert 'attested: commit' in text
            receipt = run._floor_registry_receipt()['floor_registry']
            assert receipt['lookup_count'] >= 1
            assert {r['status'] for r in receipt['lookups']} == {floor_registry.STATUS_MISS}
            assert 'floor registry lookup: miss' in text
            requests = []
            for line in text.splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    requests.append(row)
            wire = next(row for row in requests if row.get('schema') == actor_wire_gate.SCHEMA)
            actor_wire_gate.validate_request(wire)
            assert 'stdout only' in text
            receipts.append(receipt)
        if not attested:
            raise AttestationRefused
        return 0

    old_registry = run._FLOOR_REGISTRY
    try:
        with mock.patch.object(promotion_fixture, 'TheKeepBuildsAProductionCompleteAnchor', StoreImageFixture), \
                mock.patch.object(run, 'main', side_effect=invoke):
            if attested:
                cpu_fixture(dry_run=True, runtime_declaration=False)
            else:
                with pytest.raises(AttestationRefused):
                    cpu_fixture(dry_run=True, runtime_declaration=False)
        assert bool(receipts) is attested
    finally:
        run._FLOOR_REGISTRY = old_registry
