"""Large payload refs remain external; source retention cannot be bypassed."""
from pathlib import Path
import json,tempfile,unittest
from autokernel.loop import census_capture as capture

class LargeRefs(unittest.TestCase):
    def fixture(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        root=Path(tmp.name)
        inputs={}
        for name in ('binary','model','recipe','binding','producer','driver','grant',
            'capture_module','residency','workload_contract','build_receipt_binding',
            'build_complete','build_log','elf_manifest','build_job'):
            path=root/name;path.write_text(name+'\n');inputs[name]=str(path)
        sources=root/'build_sources'
        sources.write_text(f"{capture.file_sha256(Path(inputs['producer']))}  {inputs['producer']}\n")
        inputs['build_sources']=str(sources)
        return root,inputs
    def begin(self,path,inputs,large):
        return capture.begin(path,attempt_id='fixture',started_ns=1,argv=[inputs['binary']],
            env={'PATH':'/fixture'},shape={'phase':'verify','n_tokens':4,'n_seq':1},
            grant=Path(inputs['grant']).read_text(),deadline='30',kernel_commit='a'*40,
            inputs=inputs,large_refs=large)
    def test_large_drafter_is_hash_reference_not_retained_copy(self):
        root,inputs=self.fixture();payload=root/'drafter.gguf'
        with payload.open('wb') as stream:stream.truncate((64<<20)+1)
        inputs['drafter']=str(payload);journal=root/'large.jsonl'
        self.begin(journal,inputs,('drafter',))
        ref=json.loads(journal.read_text())['inputs']['drafter']
        self.assertEqual(ref['path'],str(payload))
        self.assertEqual(ref['sha256'],capture.file_sha256(payload))
        self.assertNotIn('origin_path',ref)
        self.assertFalse((root/'evidence'/ref['sha256']).exists())
    def test_source_files_missing_and_zero_shards_cannot_bypass_retention(self):
        for key in ('driver','missing','model_shard_0'):
            with self.subTest(key=key):
                root,inputs=self.fixture();journal=root/'refused.jsonl'
                with self.assertRaises(ValueError):self.begin(journal,inputs,(key,))
                self.assertEqual(json.loads(journal.read_text())['kind'],'refusal')

if __name__=='__main__':unittest.main()
