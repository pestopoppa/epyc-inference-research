import os, signal, subprocess, sys, tempfile, threading, time, unittest
from pathlib import Path
from stt8_engine import engine_for, run_process
from stt8_producer import REQUIRED, run_case
BIND={k:'pinned' for k in REQUIRED}
class FakeEngineTests(unittest.TestCase):
    def test_same_entry_normal_and_cutoff(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); side=p/'native.sidecar'
            code="import os;open(os.environ['CS8_TRANSCRIPT_PATH'],'xb').write(b'CS8TRACE 1 chat\\nT 7 2\\nhi\\nE complete eog\\n')"
            def normal(cmd,env,binding):
                q=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                so,se=q.communicate(timeout=5)
                return {'rc':q.returncode,'reaped':q.poll() is not None,'interrupted':False,'stdout':so,'stderr':se}
            r=run_case(p/'records','normal',BIND,engine_for([sys.executable,'-c',code],'/isolated',side,normal))
            self.assertEqual((r['status'],r['text']),('complete','hi'))
            self.assertIsNone(run_case(p/'records','normal',BIND,lambda _:None))
            def cutoff(cmd,env,binding):
                q=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                self.assertEqual(q.stdout.readline(),b'READY\n')
                self.assertIsNone(q.poll())
                q.terminate(); q.communicate(timeout=5)
                self.assertIsNotNone(q.poll())
                raise InterruptedError('test STOP; child reaped')
            r=run_case(p/'records','cut',BIND,engine_for([sys.executable,'-c','import time;print("READY",flush=True);time.sleep(60)'],'/isolated',p/'cut.sidecar',cutoff))
            self.assertEqual(r['status'],'unknown')
            self.assertIsNone(run_case(p/'records','cut',BIND,lambda _:None))
    def test_existing_sidecar_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'trace';p.write_bytes(b'old')
            with self.assertRaises(FileExistsError): engine_for(['x'],'/isolated',p,lambda *_:None)
    def test_actual_job_local_stop_reaps_and_retains_partial(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); stop=p/'STOP'
            timer=threading.Timer(.3,stop.touch); timer.start()
            try:
                cmd=[sys.executable,'-c','import time;print("READY",flush=True);time.sleep(60)']
                def runner(cmd,env,binding):
                    return run_process(cmd,env,binding,deadline=time.monotonic()+5,stop_path=stop)
                r=run_case(p/'records','cut',BIND,engine_for(cmd,'/isolated',p/'trace',runner))
                self.assertEqual(r['status'],'unknown')
                self.assertTrue(r['interrupted']); self.assertTrue(r['reaped'])
                self.assertIn(b'READY',(p/'records'/'cut.stdout').read_bytes())
                self.assertIsNone(run_case(p/'records','cut',BIND,lambda _:None))
            finally: timer.cancel()
if __name__=='__main__': unittest.main()
