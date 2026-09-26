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

    def test_owner_refuses_another_holder_missing_task_or_lease_claim(self):
        for holder,task,lease in [('mainB','EXL3-3',None),('inference',None,None),
                                  ('inference',' ',None),('inference','EXL3-3','lease-1')]:
            with self.subTest(holder=holder,task=task,lease=lease),self.assertRaises(launcher.Refusal):
                launcher.preflight(self.root,lease,holder,owner_run=True,task_id=task)

    def test_owner_roster_must_be_unique_and_current(self):
        for change in ('missing','duplicate','another_owner','role','gpu_lane','resource_owner','unschedulable','policy'):
            with self.subTest(change=change):
                saved=json.loads(json.dumps(self.config))
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
                self.config=saved
        self.write()

    def test_owner_missing_policy_refuses(self):
        (self.root/'agents/inference-main.md').unlink()
        with self.assertRaisesRegex(launcher.Refusal,'policy is missing'):self.owner_preflight()

    def test_delegation_still_requires_enabled_known_provider(self):
        for gpu in ({'enabled':False,'provider':None},{'enabled':True,'provider':None},
                    {'enabled':False,'provider':'gpu-device-claim'},{'enabled':True,'provider':'unknown'}):
            with self.subTest(gpu=gpu):
                self.config['resource_claims']['gpu']=gpu;self.write()
                with self.assertRaises(launcher.Refusal):launcher.preflight(self.root,'lease-1','mainB')
        self.fold.assert_not_called()

    def test_delegated_active_lease_passes(self):
        self.config['resource_claims']['gpu']={'enabled':True,'provider':'gpu-device-claim'};self.write()
        result=launcher.preflight(self.root,'lease-1','mainB')
        self.assertEqual(result['mode'],'delegated');self.assertEqual(result['lease'],self.lease)
        self.assertEqual(result['campaign_id'],'lease-1')

    def test_delegated_lease_checks_remain_fail_closed(self):
        self.config['resource_claims']['gpu']={'enabled':True,'provider':'gpu-device-claim'};self.write()
        for change in ('missing','reserved','holder','device','expired'):
            with self.subTest(change=change):
                lease=dict(self.lease)
                if change=='reserved':lease['state']='RESERVED'
                elif change=='holder':lease['holder']='someone_else'
                elif change=='device':lease['resources']={'gpu_devices':['other_gpu']}
                elif change=='expired':lease['expires_ts']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
                self.fold.return_value={} if change=='missing' else {'lease-1':lease}
                with self.assertRaises(launcher.Refusal):launcher.preflight(self.root,'lease-1','mainB')

    def test_delegated_mode_refuses_owner_only_task_argument(self):
        with self.assertRaisesRegex(launcher.Refusal,'task-id'):
            launcher.preflight(self.root,'lease-1','mainB',task_id='EXL3-3')

    def test_live_or_unreadable_kfd_refuses_both_modes(self):
        for pids in (None,['1234']):
            with self.subTest(pids=pids),self.assertRaises(launcher.Refusal):
                launcher.require_exclusive_kfd({'kfd_pids':pids})
        launcher.require_exclusive_kfd({'kfd_pids':[]})

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

    def test_both_modes_keep_physical_claim_kfd_and_residency_checks(self):
        @dataclass
        class Receipt:
            device: str = 'mi210_0'
        build=self.root/'build';build.mkdir()
        binary=build/'test_runtime';binary.write_text('not executable; mocked child only')
        sources={name:launcher.sha(HERE/name) for name in ('test_runtime.hip','kernels.hip','contract.hpp')}
        (build/'build.json').write_text(json.dumps({'artifact_sha256':{'test_runtime':launcher.sha(binary)},'source_sha256':sources}))
        self.config['resource_claims']['gpu']={'enabled':True,'provider':'gpu-device-claim'};self.write()
        for owner in (False,True):
            for boundary in ('unheld_claim','foreign_kfd','no_residency'):
                with self.subTest(owner=owner,boundary=boundary):
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

if __name__=='__main__':unittest.main(verbosity=2)
