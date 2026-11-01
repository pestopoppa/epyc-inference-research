import unittest
import hashlib
from stt8_compare import compare, compare_fleurs_validation, score_utterance_chunks

def row(i,errors=0,pcm='p'):
    return {'utterance_id':i,'errors':errors,'reference_tokens':10,'reference_normalised':'a','pcm_sha256':pcm,'class':'ok'}
class CompareTests(unittest.TestCase):
    def test_paired_direction_and_noninferiority(self):
        ids={'en':['e'],'it':['i']}; a={'en':[row('e')],'it':[row('i')]}
        b={'en':[row('e',1)],'it':[row('i',0)]}
        r=compare(ids,a,b); self.assertEqual(r['languages']['en']['point_delta_pp'],10.0); self.assertEqual(r['stt8_gate'],'FAIL')
        self.assertEqual(compare(ids,a,a)['stt8_gate'],'PASS')
    def test_refuse_mismatch_unknown_and_duplicate(self):
        ids={'en':['e'],'it':['i']}; a={'en':[row('e')],'it':[row('i')]}
        for bad in ({'en':[row('e',pcm='z')],'it':[row('i')]}, {'en':[],'it':[row('i')]}, {'en':[row('e'),row('e')],'it':[row('i')]}):
            with self.assertRaises(ValueError): compare(ids,a,bad)
        a['en'][0]['class']='VOID'
        with self.assertRaises(ValueError): compare(ids,a,a)
    def test_frozen_validation_cannot_accept_a_subset(self):
        a={'en':[row('e')],'it':[row('i')]}
        with self.assertRaises(ValueError): compare_fleurs_validation(a,a)
    def test_canonical_scoring_failure_and_unknown(self):
        row={'clip_id':'x','pcm_sha256':hashlib.sha256(b'fixture pcm').hexdigest(),'reference_raw':'one two three','frames':16000,'language':'en_us'}
        ok={'chunk_index':0,'pcm_sha256':'c','status':'complete','text':'one two three'}
        self.assertEqual(score_utterance_chunks(row,[ok],['c'])['errors'],0)
        failed={**ok,'status':'failure'}
        score=score_utterance_chunks(row,[failed],['c'])
        self.assertEqual(score['errors'],score['reference_tokens'])
        with self.assertRaises(ValueError): score_utterance_chunks(row,[{**ok,'status':'unknown'}],['c'])
        with self.assertRaises(ValueError): score_utterance_chunks(row,[ok],['other'])
if __name__=='__main__':unittest.main()
