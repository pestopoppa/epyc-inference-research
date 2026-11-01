"""R11 smoke admission draft. Fail closed; never used by production grants."""
import datetime
import os
import re
import time
from pathlib import Path

FLAGS = Path('/mnt/raid0/llm/tmp/compute-flags')

def check(cpus):
    cpus = set(cpus)
    if len(cpus) != 2 or any(c < 96 or c > 183 for c in cpus):
        raise RuntimeError('R11 requires two allowed SMT sibling CPUs')
    quarters = {(c - 96) // 24 for c in cpus}
    if len(quarters) != 1 or os.sched_getaffinity(0) != cpus:
        raise RuntimeError('R11 smoke affinity/quarter mismatch')
    if list(FLAGS.glob('HOLD_*')):
        raise RuntimeError('global HOLD')
    q = next(iter(quarters))
    now = time.time()
    rows=(FLAGS/'FILLERS.md').read_text().splitlines()
    for path in sorted((FLAGS / 'grants').glob('*.grant')):
        fields = {}
        for token in path.read_text().split():
            k, sep, value = token.partition('=')
            if not sep or k in fields:
                raise RuntimeError('malformed grant')
            fields[k] = value
        if set(fields) != {'regions', 'start', 'end'}:
            raise RuntimeError('unknown grant schema')
        def parse(value):
            parsed = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
            if parsed.tzinfo is None:
                raise RuntimeError('grant time lacks timezone')
            return parsed.timestamp()
        start, end = parse(fields['start']), parse(fields['end'])
        if start >= end:
            raise RuntimeError('invalid grant interval')
        if end <= now or start >= now + 300:
            continue
        matches=[line.split('|') for line in rows if line.startswith('| '+path.stem+' |')]
        if len(matches)!=1 or len(matches[0])<=9:
            raise RuntimeError('unknown grant class')
        grant_class=matches[0][9].strip()
        if grant_class in ('block-correctness','diagnostic'):
            continue
        raise RuntimeError('timing or unclassified grant exists anywhere')
    return {'kind': 'r11-unlocked-smoke', 'quarter': q, 'cpus': sorted(cpus), 'checked_epoch': now}
