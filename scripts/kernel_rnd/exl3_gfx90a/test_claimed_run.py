"""Authority and co-residency tests: no physical claim or HIP process is started."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import yaml

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('exl3_claimed_run',HERE/'claimed_run.py')
launcher=importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)

class AuthorityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='exl3-owner-test-',dir='/mnt/raid0/llm/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        (self.root/'coordination/session-bus').mkdir(parents=True)
        (self.root/'agents').mkdir()
        (self.root/'agents/inference-main.md').write_text('The Inference Main can still execute its own GPU work under the existing inference rules.\n')
        self.owner={'id':'inference','role':'inference-main','lanes':['cpu','gpu','none'],
                    'resource_owner':['cpu','gpu'],'schedulable':True,'role_policy':'agents/inference-main.md'}
        self.config={'roster':[self.owner],'resource_claims':{'gpu':{'enabled':False,'provider':None}}}
        self.write()
        self.lease={'state':'ACTIVE','holder':'mainB','resources':{'gpu_devices':['mi210_0']},
                    'expires_ts':(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()}
        self.fold=mock.Mock(return_value={'lease-1':self.lease})
        self.bus=mock.patch.dict('sys.modules',{'session_bus':types.SimpleNamespace(fold_resource_leases=self.fold)})
        self.bus.start();self.addCleanup(self.bus.stop)

    def write(self):
        (self.root/'coordination/session-bus/config.yaml').write_text(yaml.safe_dump(self.config))

    def owner_preflight(self,**kwargs):
        return launcher.preflight(self.root,None,'inference',owner_run=True,task_id='EXL3-3',**kwargs)

    def test_owner_disabled_provider_has_explicit_authority_without_lease(self):
        result=self.owner_preflight()
        self.assertEqual(result['mode'],'inference_owner')
        self.assertIsNone(result['lease'])
        self.assertEqual(result['campaign_id'],'owner:inference:EXL3-3')
        self.assertFalse(result['delegated_provider']['enabled'])
        self.assertEqual(result['roster_owner']['id'],'inference')
        self.assertEqual(result['role_policy']['sha256'],launcher.sha(self.root/'agents/inference-main.md'))
        self.fold.assert_not_called()

    def test_owner_is_not_an_automatic_holder_bypass(self):
        with self.assertRaisesRegex(launcher.Refusal,'lease ID'):
            launcher.preflight(self.root,None,'inference')
        with self.assertRaisesRegex(launcher.Refusal,'disabled'):
            launcher.preflight(self.root,'lease-1','inference')

    def _assert_owner_refusal(self, holder, task, lease):
        with self.assertRaises(launcher.Refusal):
            launcher.preflight(self.root,lease,holder,owner_run=True,task_id=task)

    def test_owner_refuses_delegated_holder_without_owner_mode(self):
        self._assert_owner_refusal('mainB','EXL3-3',None)

    def test_owner_refuses_missing_task_id(self):
        self._assert_owner_refusal('inference',None,None)

    def test_owner_refuses_blank_task_id(self):
        self._assert_owner_refusal('inference',' ',None)

    def test_owner_refuses_owner_mode_with_lease(self):
        self._assert_owner_refusal('inference','EXL3-3','lease-1')

    def _assert_roster_change_refused(self, change):
        owner=self.config['roster'][0]
        if change=='missing':self.config['roster']=[]
        elif change=='duplicate':self.config['roster'].append(dict(owner))
        elif change=='another_owner':self.config['roster'].append({'id':'other','resource_owner':['gpu']})
        elif change=='role':owner['role']='retired'
        elif change=='gpu_lane':owner['lanes']=['cpu']
        elif change=='resource_owner':owner['resource_owner']=['cpu']
        elif change=='unschedulable':owner['schedulable']=False
        elif change=='policy':owner['role_policy']='agents/other.md'
        self.write()
        with self.assertRaises(launcher.Refusal):self.owner_preflight()

    def test_roster_missing_owner_refuses(self):
        self._assert_roster_change_refused('missing')

    def test_roster_duplicate_owner_refuses(self):
        self._assert_roster_change_refused('duplicate')

    def test_roster_another_owner_refuses(self):
        self._assert_roster_change_refused('another_owner')

    def test_roster_retired_role_refuses(self):
        self._assert_roster_change_refused('role')

    def test_roster_missing_gpu_lane_refuses(self):
        self._assert_roster_change_refused('gpu_lane')

    def test_roster_missing_gpu_resource_owner_refuses(self):
        self._assert_roster_change_refused('resource_owner')

    def test_roster_unschedulable_owner_refuses(self):
        self._assert_roster_change_refused('unschedulable')

    def test_roster_wrong_policy_refuses(self):
        self._assert_roster_change_refused('policy')

    def test_owner_missing_policy_refuses(self):
        (self.root/'agents/inference-main.md').unlink()
        with self.assertRaisesRegex(launcher.Refusal,'policy is missing'):self.owner_preflight()

    def _assert_delegation_provider_refused(self, gpu):
        self.config['resource_claims']['gpu']=gpu;self.write()
        with self.assertRaises(launcher.Refusal):launcher.preflight(self.root,'lease-1','mainB')
        self.fold.assert_not_called()

    def test_delegation_refuses_disabled_provider_without_name(self):
        self._assert_delegation_provider_refused({'enabled':False,'provider':None})

    def test_delegation_refuses_enabled_provider_without_name(self):
        self._assert_delegation_provider_refused({'enabled':True,'provider':None})

    def test_delegation_refuses_disabled_named_provider(self):
        self._assert_delegation_provider_refused({'enabled':False,'provider':'gpu-device-claim'})

    def test_delegation_refuses_unknown_provider(self):
        self._assert_delegation_provider_refused({'enabled':True,'provider':'unknown'})

    def test_delegated_active_lease_passes(self):
        self.config['resource_claims']['gpu']={'enabled':True,'provider':'gpu-device-claim'};self.write()
        result=launcher.preflight(self.root,'lease-1','mainB')
        self.assertEqual(result['mode'],'delegated');self.assertEqual(result['lease'],self.lease)
        self.assertEqual(result['campaign_id'],'lease-1')

    def _assert_delegated_lease_refused(self, change):
        self.config['resource_claims']['gpu']={'enabled':True,'provider':'gpu-device-claim'};self.write()
        lease=dict(self.lease)
        if change=='reserved':lease['state']='RESERVED'
        elif change=='holder':lease['holder']='someone_else'
        elif change=='device':lease['resources']={'gpu_devices':['other_gpu']}
        elif change=='expired':lease['expires_ts']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
        self.fold.return_value={} if change=='missing' else {'lease-1':lease}
        with self.assertRaises(launcher.Refusal):launcher.preflight(self.root,'lease-1','mainB')

    def test_delegated_missing_lease_refuses(self):
        self._assert_delegated_lease_refused('missing')

    def test_delegated_reserved_lease_refuses(self):
        self._assert_delegated_lease_refused('reserved')

    def test_delegated_wrong_holder_refuses(self):
        self._assert_delegated_lease_refused('holder')

    def test_delegated_wrong_device_refuses(self):
        self._assert_delegated_lease_refused('device')

    def test_delegated_expired_lease_refuses(self):
        self._assert_delegated_lease_refused('expired')

    def test_delegated_mode_refuses_owner_only_task_argument(self):
        with self.assertRaisesRegex(launcher.Refusal,'task-id'):
            launcher.preflight(self.root,'lease-1','mainB',task_id='EXL3-3')

    def test_unknown_kfd_census_refuses(self):
        with self.assertRaises(launcher.Refusal):
            launcher.require_exclusive_kfd({'kfd_pids':None})

    def test_foreign_kfd_pid_refuses(self):
        with self.assertRaises(launcher.Refusal):
            launcher.require_exclusive_kfd({'kfd_pids':['1234']})

    def test_empty_kfd_census_is_accepted(self):
        self.assertIsNone(launcher.require_exclusive_kfd({'kfd_pids':[]}))

    def test_cli_owner_preflight_never_starts_a_process_or_claim(self):
        unused=self.root/'unused'
        args=['claimed_run.py','--root',str(self.root),'--contract-root','/unused',
              '--build','/unused','--output',str(unused),'--holder','inference',
              '--owner-run','--task-id','EXL3-3','--preflight-only']
        output=io.StringIO()
        with mock.patch('sys.argv',args),mock.patch.object(launcher.subprocess,'Popen') as popen,redirect_stdout(output):
            launcher.main()
        popen.assert_not_called();self.fold.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())['mode'],'inference_owner')
        self.assertFalse(unused.exists())

    def test_cli_owner_and_delegated_modes_are_mutually_exclusive(self):
        args=['claimed_run.py','--contract-root','/unused','--build','/unused','--output','/unused',
              '--holder','inference','--owner-run','--lease-id','lease-1','--task-id','EXL3-3','--preflight-only']
        with mock.patch('sys.argv',args),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as raised:
            launcher.main()
        self.assertEqual(raised.exception.code,2)

    def _assert_physical_claim_checks(self, owner, boundary):
        @dataclass
        class Receipt:
            device: str = 'mi210_0'
        build=self.root/'build';build.mkdir()
        binary=build/'test_runtime';binary.write_text('not executable; mocked child only')
        sources={name:launcher.sha(HERE/name) for name in ('test_runtime.hip','kernels.hip','contract.hpp')}
        (build/'build.json').write_text(json.dumps({'artifact_sha256':{'test_runtime':launcher.sha(binary)},'source_sha256':sources}))
        self.config['resource_claims']['gpu']={'enabled':True,'provider':'gpu-device-claim'};self.write()
        out=self.root/f'run-{owner}-{boundary}'
        args=['claimed_run.py','--root',str(self.root),'--contract-root',str(self.root),
              '--build',str(build),'--output',str(out)]
        args+=['--owner-run','--holder','inference','--task-id','EXL3-3'] if owner else ['--lease-id','lease-1','--holder','mainB']
        claim=mock.Mock(held=boundary!='unheld_claim',_fd=123)
        claim.receipt.return_value=Receipt();claim.revocation.return_value=None
        context=mock.MagicMock();context.__enter__.return_value=claim
        acquire=mock.Mock(return_value=context)
        provider=types.SimpleNamespace(gpu_device_claim=acquire,ClaimJournal=mock.Mock())
        evidence=types.SimpleNamespace()
        modules={'scripts.kernel_rnd.exl3':types.SimpleNamespace(evidence=evidence),
                 'scripts.kernel_rnd.autokernel.resource.device_claim':provider}
        sample={'kfd_pids':['4321'] if boundary=='foreign_kfd' else [],'vram_used_bytes':{}}
        proc=mock.Mock(pid=12345);proc.poll.side_effect=[None,0];proc.wait.return_value=0
        with mock.patch.dict('sys.modules',modules),mock.patch('sys.argv',args), \
             mock.patch.object(launcher.subprocess,'check_output',return_value=b''), \
             mock.patch.object(launcher.subprocess,'Popen',return_value=proc) as popen, \
             mock.patch.object(launcher,'residency_sample',return_value=sample), \
             mock.patch.object(launcher.time,'sleep'), \
             mock.patch.object(launcher,'seal_verifier',return_value={'row_id':'mock'}) as seal,redirect_stdout(io.StringIO()):
            if boundary=='no_residency':
                with self.assertRaises(SystemExit) as raised:launcher.main()
                self.assertEqual(raised.exception.code,2)
                self.assertEqual(popen.call_args.kwargs['pass_fds'],(123,))
                self.assertEqual(popen.call_args.kwargs['env']['EXL3_PHYSICAL_CLAIM_FD'],'123')
                self.assertEqual(seal.call_args.args[6],2)
            else:
                with self.assertRaises(launcher.Refusal):launcher.main()
                popen.assert_not_called();seal.assert_not_called()
        self.assertEqual(acquire.call_args.args,('mi210_0',))
        self.assertEqual(acquire.call_args.kwargs['timeout_s'],0)
        self.assertEqual(acquire.call_args.kwargs['campaign_id'],'owner:inference:EXL3-3' if owner else 'lease-1')

    def test_delegated_unheld_claim_refuses_before_launch(self):
        self._assert_physical_claim_checks(False,'unheld_claim')

    def test_owner_unheld_claim_refuses_before_launch(self):
        self._assert_physical_claim_checks(True,'unheld_claim')

    def test_delegated_foreign_kfd_refuses_before_launch(self):
        self._assert_physical_claim_checks(False,'foreign_kfd')

    def test_owner_foreign_kfd_refuses_before_launch(self):
        self._assert_physical_claim_checks(True,'foreign_kfd')

    def test_delegated_missing_residency_stops_child(self):
        self._assert_physical_claim_checks(False,'no_residency')

    def test_owner_missing_residency_stops_child(self):
        self._assert_physical_claim_checks(True,'no_residency')

class KfdTeardownPollingTests(unittest.TestCase):
    class FakeClock:
        def __init__(self):self.now=0.0;self.sleeps=[]
        def monotonic(self):return self.now
        def sleep(self,seconds):self.sleeps.append(seconds);self.now+=seconds

    def test_reaped_pid_waits_for_kfd_bookkeeping_and_retains_samples(self):
        clock=self.FakeClock()
        samples=[{'pid':1234,'pid_alive':False,'kfd_pids':['1234'],'vram_used_bytes':{}},
                 {'pid':1234,'pid_alive':False,'kfd_pids':['1234'],'vram_used_bytes':{}},
                 {'pid':1234,'pid_alive':False,'kfd_pids':[],'vram_used_bytes':{}}]
        with mock.patch.object(launcher.time,'monotonic',side_effect=clock.monotonic), \
             mock.patch.object(launcher.time,'sleep',side_effect=clock.sleep), \
             mock.patch.object(launcher,'residency_sample',side_effect=samples) as sample:
            teardown,cleared=launcher.poll_kfd_pid_exit(1234,timeout_s=0.1,poll_interval_s=0.02)
        self.assertTrue(cleared)
        self.assertEqual(teardown,samples)
        self.assertEqual(sample.call_count,3)
        self.assertEqual(clock.sleeps,[0.02,0.02])

    def test_unknown_or_persistent_kfd_census_fails_only_at_deadline(self):
        clock=self.FakeClock()
        lingering={'pid':5678,'pid_alive':False,'kfd_pids':['5678'],'vram_used_bytes':{}}
        unknown={'pid':5678,'pid_alive':False,'kfd_pids':None,'vram_used_bytes':{}}
        with mock.patch.object(launcher.time,'monotonic',side_effect=clock.monotonic), \
             mock.patch.object(launcher.time,'sleep',side_effect=clock.sleep), \
             mock.patch.object(launcher,'residency_sample',side_effect=[unknown,lingering,lingering,lingering]):
            teardown,cleared=launcher.poll_kfd_pid_exit(5678,timeout_s=0.05,poll_interval_s=0.02)
        self.assertFalse(cleared)
        self.assertEqual(teardown,[unknown,lingering,lingering,lingering])
        self.assertEqual(len(clock.sleeps),3)
        self.assertAlmostEqual(sum(clock.sleeps),0.05)

if __name__=='__main__':unittest.main(verbosity=2)
