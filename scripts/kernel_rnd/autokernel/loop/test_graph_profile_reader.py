"""Synthetic prospective SC55 decoder controls; no grade or runtime claim."""
import copy,hashlib,json,unittest
from pathlib import Path
from . import graph_profile_reader as r
from . import graph_profile_capture as c
from . import test_graph_profile_capture as capture_fixtures

class ReaderTests(unittest.TestCase):
    phases=capture_fixtures.CaptureTests.phases
    def setUp(self):
        capture_fixtures.CaptureTests.setUp(self)
        self.p['source']['profiler_source_sha256']=r.SOURCE_SHA256
        self.graph={'schema':'epyc.graph_structural_receipt.v1','source':self.p['source'],
                    'nodes':[{'idx':0,'op':'MUL_MAT','name':'test','src0_type':'Q4_K','src0_ne':[4,4,1],
                              'dst_ne':[4,1,1],'src_ids':['weight'],'op_params_hex':'00'}],
                    'external_tensors':[{'id':'weight','type':'Q4_K','ne':[4,4,1,1],'strides':[1,4,16,16]}]}
        Path(self.raw['graph_identity']).write_text(json.dumps(self.graph))
        self.p['graph']['identity_sha256']=hashlib.sha256(Path(self.raw['graph_identity']).read_bytes()).hexdigest()
        self.o['graph_identity_sha256']=self.p['graph']['identity_sha256']
        self.run={'schema':'epyc.graph_profile_run_receipt.v1','capture_id':self.p['capture_id'],'owner':self.p['owner'],'source':self.p['source'],
                  'graph_identity_sha256':self.p['graph']['identity_sha256'],'counter_reset':False,
                  'eval_events':[{'global_idx':0,'n_nodes':1,'structural_sha256':self.p['graph']['identity_sha256'],'phase':'decode','accumulated':True}],
                  'expected_pernode_evals':[{'idx':0,'evals':1}]}
        self.node_cells=['0','MUL_MAT','test','Q4_K','4','4','1','4','1','1','1.000','2.000','1','2','1','0','0.000','0.000','0.000']
        self.path_cells=['dense_mul_mat','1.0','0.0010','0.0020','16','0.02','0.01']
        self.write_native()
    def write_native(self):
        Path(self.raw['pernode']).write_text('\t'.join(r.NODE_COLUMNS)+'\n'+'\t'.join(self.node_cells)+'\n')
        Path(self.raw['log']).write_text('[cpu_prof] PATHTABLE\t'+'\t'.join(r.PATH_COLUMNS)+'\n[cpu_prof] PATHROW\t'+'\t'.join(self.path_cells)+'\n')
        Path(self.raw['run_receipt']).write_text(json.dumps(self.run))
    def seal_native(self):
        self.write_native();self.phases();c.finalize(self.seal,self.pre,self.during,self.closed)
        return c.receipt(self.seal)
    def test_valid_native_roundtrip_thread_zero_unavailable(self):
        decoded=r.read(self.seal_native())
        self.assertEqual(decoded['thread_availability'],'unavailable')
        self.assertEqual(decoded['semantics']['accumulated_evals'],1)
        self.assertNotIn('grade',decoded)
    def test_node_header_exact_19_not_old_alias(self):
        self.node_cells[14]='1'
        raw=('\t'.join(r.NODE_COLUMNS).replace('wall_max_ev','argmax_ev')+'\n'+'\t'.join(self.node_cells)+'\n').encode()
        with self.assertRaises(c.Refusal):r.parse_nodes(raw,'unavailable')
    def test_node_identity_eval_numeric_refusals(self):
        for index,value in ((0,'-1'),(12,'0'),(12,'1.5'),(10,'nan'),(11,'-1'),(16,'1')):
            cells=list(self.node_cells);cells[index]=value
            raw=('\t'.join(r.NODE_COLUMNS)+'\n'+'\t'.join(cells)+'\n').encode()
            with self.subTest(index=index,value=value):
                with self.assertRaises(c.Refusal):r.parse_nodes(raw,'unavailable')
        raw=('\t'.join(r.NODE_COLUMNS)+'\n'+('\t'.join(self.node_cells)+'\n')*2).encode()
        with self.assertRaises(c.Refusal):r.parse_nodes(raw,'unavailable')
    def test_path_exact_header_duplicate_and_nonfinite_refusals(self):
        good=('[cpu_prof] PATHTABLE\t'+'\t'.join(r.PATH_COLUMNS)+'\n[cpu_prof] PATHROW\t'+'\t'.join(self.path_cells)+'\n')
        for bad in (good.replace('calls_per_eval','ne_calls'),good+good,good.replace('0.0010','inf'),good.replace('dense_mul_mat','unknown')):
            with self.assertRaises(c.Refusal):r.parse_paths(bad.encode())
    def test_raw_digest_and_size_refusal(self):
        ref=self.seal_native();Path(self.raw['log']).write_text('changed')
        with self.assertRaises(c.Refusal):r.read(ref)
    def test_opaque_graph_receipt_refuses_invention(self):
        with self.assertRaises(c.Refusal):r.interpret({'text':'opaque'},self.run,self.p,[])
    def test_structural_edges_unknown_or_unbound_refuse(self):
        bad=copy.deepcopy(self.graph);bad['nodes'][0]['src_ids']=['missing']
        rows=r.parse_nodes(Path(self.raw['pernode']).read_bytes(),'unavailable')
        with self.assertRaises(c.Refusal):r.interpret(bad,self.run,self.p,rows)
    def test_run_phase_graph_and_filter_refusals(self):
        rows=r.parse_nodes(Path(self.raw['pernode']).read_bytes(),'unavailable')
        for key,value in (('phase','unknown'),('phase','prefill'),('structural_sha256','a'*64),('n_nodes',2),('accumulated',False),('global_idx',1)):
            bad=copy.deepcopy(self.run);bad['eval_events'][0][key]=value
            with self.subTest(key=key):
                with self.assertRaises(c.Refusal):r.interpret(self.graph,bad,self.p,rows)
    def test_node_cardinality_and_native_metadata_refusals(self):
        rows=r.parse_nodes(Path(self.raw['pernode']).read_bytes(),'unavailable')
        for field,value in (('op','ADD'),('wall_max_ev',2),('evals',2)):
            bad=copy.deepcopy(rows);bad[0][field]=value
            with self.assertRaises(c.Refusal):r.interpret(self.graph,self.run,self.p,bad)
        bad=copy.deepcopy(self.run);bad['expected_pernode_evals']=[]
        with self.assertRaises(c.Refusal):r.interpret(self.graph,bad,self.p,rows)
    def test_carrier_unknown_and_phase_clock_refuse(self):
        ref=self.seal_native();envelope=json.loads(self.seal.read_text());envelope['writer_observed_at']='2025-01-01T00:00:00Z'
        self.seal.write_text(json.dumps(envelope))
        with self.assertRaises(c.Refusal):r.read(c.receipt(self.seal))
    def test_source_grammar_unknown_refuses(self):
        self.p['source']['profiler_source_sha256']='a'*64
        with self.assertRaises(c.Refusal):r.read(self.seal_native())

    def test_native_label_selector_absent_refuses(self):
        self.p['measurement_metadata']['metrics'][0]['selector']['identity']=9
        with self.assertRaises(c.Refusal):r.read(self.seal_native())

    def test_unavailable_thread_label_never_projects_zero(self):
        self.p['measurement_metadata']['metrics'][0]['selector']['field']='thr_mean_us'
        with self.assertRaises(c.Refusal):r.read(self.seal_native())

    def test_sealed_native_date_cannot_label_another_year(self):
        ref=self.seal_native();seal=json.loads(self.seal.read_text());pre=json.loads(self.pre.read_text())
        seal['provenance']['measurement_metadata']['date']='2030-01-01'
        pre['provenance']['measurement_metadata']['date']='2030-01-01'
        self.pre.write_text(json.dumps(pre));new_pre=c.receipt(self.pre)
        during=json.loads(self.during.read_text());during['pre']=new_pre;self.during.write_text(json.dumps(during))
        seal['pre']=new_pre;seal['during']=c.receipt(self.during);self.seal.write_text(json.dumps(seal))
        with self.assertRaises(c.Refusal):r.read(c.receipt(self.seal))
