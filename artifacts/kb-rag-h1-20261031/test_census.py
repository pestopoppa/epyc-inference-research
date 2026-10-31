import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from tokenizers import Tokenizer
import census

TOKENIZER = '/mnt/raid0/llm/models/gte-moderncolbert-v1-onnx/tokenizer.json'


class NativeControls(unittest.TestCase):
    def test_native_prepend_specials_and_boundary(self):
        original = Tokenizer.from_file(TOKENIZER)
        full = Tokenizer.from_str(original.to_str()); full.no_padding(); full.no_truncation()
        capped = Tokenizer.from_str(original.to_str()); capped.no_padding(); capped.enable_truncation(max_length=256)
        for text in ('short', ' café 漢字 😊 ', '[D] [CLS] text', 'word ' * 400):
            result = census.tokens(text, full, capped)
            native_full = full.encode('[D] ' + text)
            native_cap = capped.encode('[D] ' + text)
            self.assertEqual(result['full_tokens'], len(native_full.ids))
            self.assertEqual(result['kept_tokens'], len(native_cap.ids))
            self.assertLessEqual(result['kept_covered_codepoints'], result['full_covered_codepoints'])
        self.assertGreater(census.tokens('word ' * 400, full, capped)['full_tokens'], 256)
        self.assertEqual(census.tokens('word ' * 400, full, capped)['kept_tokens'], 256)
        self.assertEqual(len(full.encode('[D] ', add_special_tokens=False).ids), 1)
        boundary = {}
        for n in range(300):
            text = 'word ' * n
            size = len(full.encode('[D] ' + text).ids)
            if size in (255, 256, 257):
                boundary[size] = text
        self.assertEqual(set(boundary), {255, 256, 257})
        for size, text in boundary.items():
            measurement = census.tokens(text, full, capped)
            self.assertEqual(measurement['full_tokens'], size)
            self.assertEqual(measurement['kept_tokens'], min(size, 256))

    def test_denominator_includes_bad_or_missing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / 'fixture.sqlite'; output = Path(tmp) / 'out.jsonl'
            c = sqlite3.connect(db)
            c.execute('create table chunk(chunk_id integer primary key,file_path text,heading_path text,content_hash text,text_preview text)')
            c.execute('create virtual table chunk_fts using fts5(file_path,heading_path,text)')
            text = 'café 漢字'
            digest = hashlib.sha256(text.encode()).hexdigest()[:16]
            c.execute('insert into chunk values(1,?,?,?,?)', ('file','[]',digest,text))
            c.execute('insert into chunk_fts(rowid,file_path,heading_path,text) values(1,?,?,?)', ('file','[]',text))
            c.execute('insert into chunk values(2,?,?,?,?)', ('file','[]',digest,text))
            c.execute('insert into chunk values(3,?,?,?,?)', ('file','[]','wrong',text))
            c.execute('insert into chunk_fts(rowid,file_path,heading_path,text) values(3,?,?,?)', ('file','[]',text))
            c.commit(); c.close()
            summary = census.run(str(db), TOKENIZER, str(output))
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual((summary['cohort_rows'], summary['known_rows'], summary['unknown_rows']), (3,1,2))
            self.assertEqual([r['status'] for r in rows], ['known','unknown','unknown'])
            self.assertNotIn(text, output.read_text())


if __name__ == '__main__':
    unittest.main()
