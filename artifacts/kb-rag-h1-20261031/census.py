#!/usr/bin/env python3
"""Private-snapshot qd-v1 text census; output contains hashes and counts only."""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from tokenizers import Tokenizer

CAP = 256
PREFIX = '[D] '
SNAPSHOT_SHA256 = 'cb7067b30cebe0260e3432abec3f1a1d85498b732f58b9f7432240bb4d6fd7fb'
TOKENIZER_SHA256 = '23abe2a8f5640f8836c24cead7b76e77613b846824d7540151c6d90d7c2f4869'


def covered(offsets, start, end):
    intervals = sorted((max(a, start), min(b, end)) for a, b in offsets if b > start and a < end)
    total = 0
    cursor = start
    for a, b in intervals:
        if b > cursor:
            total += b - max(a, cursor)
            cursor = b
    return total


def tokens(text, full, capped):
    source = PREFIX + text
    a, b = full.encode(source), capped.encode(source)
    lo, hi = len(PREFIX), len(source)
    return {'full_tokens': len(a.ids), 'kept_tokens': len(b.ids),
            'full_covered_codepoints': covered(a.offsets, lo, hi),
            'kept_covered_codepoints': covered(b.offsets, lo, hi),
            'raw_codepoints': len(text)}


def percentile(values, p):
    # Nearest rank, 1-based; full-token lengths only among known rows.
    from math import ceil
    return sorted(values)[ceil(p * len(values)) - 1] if values else None


def run(snapshot, tokenizer, output):
    src = sqlite3.connect(f'{Path(snapshot).resolve().as_uri()}?mode=ro&immutable=1', uri=True)
    src.row_factory = sqlite3.Row
    original = Tokenizer.from_file(tokenizer)
    full = Tokenizer.from_str(original.to_str())
    capped = Tokenizer.from_str(original.to_str())
    full.no_truncation(); full.no_padding()
    capped.enable_truncation(max_length=CAP); capped.no_padding()
    known = []
    unknown = 0
    with open(output, 'x', encoding='utf-8') as dest:
        for row in src.execute('select chunk_id,file_path,heading_path,content_hash,text_preview from chunk order by chunk_id'):
            record = {'chunk_id': row['chunk_id'], 'status': 'unknown'}
            try:
                f = src.execute('select file_path,heading_path,text from chunk_fts where rowid=?', (row['chunk_id'],)).fetchone()
                if f is None or not isinstance(f['text'], str):
                    raise ValueError('missing_text')
                text = f['text']
                digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
                if (f['file_path'] != row['file_path'] or f['heading_path'] != row['heading_path']
                    or digest[:16] != row['content_hash'] or text.strip()[:240] != row['text_preview']):
                    raise ValueError('metadata_or_digest_mismatch')
                record.update(tokens(text, full, capped), input_sha256=digest, status='known')
                known.append(record)
            except (ValueError, UnicodeError, sqlite3.Error) as e:
                record['reason'] = str(e) if str(e) in ('missing_text', 'metadata_or_digest_mismatch') else type(e).__name__
                unknown += 1
            dest.write(json.dumps(record, sort_keys=True) + '\n')
    lengths = [r['full_tokens'] for r in known]
    sums = {k: sum(r[k] for r in known) for k in ('full_tokens', 'kept_tokens', 'full_covered_codepoints', 'kept_covered_codepoints', 'raw_codepoints')}
    return {'cohort_rows': len(known)+unknown, 'known_rows': len(known), 'unknown_rows': unknown,
            'full_tokens_p50': percentile(lengths, .50), 'full_tokens_p95': percentile(lengths, .95),
            'full_tokens_max': max(lengths, default=None), 'at_cap_or_more': sum(n >= CAP for n in lengths),
            'strictly_over_cap': sum(n > CAP for n in lengths), 'sums_known_only': sums}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--snapshot', required=True); p.add_argument('--tokenizer', required=True); p.add_argument('--output', required=True)
    args = p.parse_args()
    for path, expected in ((args.snapshot, SNAPSHOT_SHA256), (args.tokenizer, TOKENIZER_SHA256)):
        with open(path, 'rb') as source:
            if hashlib.file_digest(source, 'sha256').hexdigest() != expected:
                raise SystemExit('enrolled input digest mismatch')
    print(json.dumps(run(args.snapshot, args.tokenizer, args.output), sort_keys=True))
