#!/usr/bin/env python3
"""Score captured, paired generations; performs no inference."""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def looped(text):
    # Detect >=4 adjacent repetitions of a 3..80-word phrase, case insensitive.
    words = re.findall(r"\w+|[^\w\s]", text.casefold())
    for size in range(3, min(80, len(words) // 4) + 1):
        for start in range(len(words) - 4 * size + 1):
            block = words[start:start + size]
            if block * 4 == words[start:start + 4 * size]:
                return True
    return False


def score(prompts_path, captures_path, cap):
    if cap <= 0:
        raise ValueError('cap must be positive')
    raw = Path(prompts_path).read_bytes()
    prompts = json.loads(raw)
    ids = [p['id'] for p in prompts]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('prompt IDs must be nonempty and unique')
    captures = json.loads(Path(captures_path).read_text())
    if set(captures) != {'reference', 'candidate'}:
        raise ValueError('exactly reference and candidate arms required')
    result = {'prompt_sha256': hashlib.sha256(raw).hexdigest(), 'cap': cap, 'arms': {}}
    for arm, records in captures.items():
        seen = [r['id'] for r in records]
        if len(seen) != len(set(seen)) or set(seen) != set(ids):
            raise ValueError(f'{arm}: prompt coverage mismatch')
        labels = []
        for r in records:
            if r['cap'] != cap or r['finish_reason'] not in ('stop', 'length') or not isinstance(r['text'], str):
                raise ValueError(f'{arm}/{r["id"]}: invalid cap, finish reason or text')
            if type(r['wrong']) is not bool:
                raise ValueError(f'{arm}/{r["id"]}: wrong requires independent human review')
            labels.append({'id': r['id'], 'wrong': r['wrong'], 'looped': looped(r['text']),
                           'cut_off': r['finish_reason'] == 'length'})
        counts = Counter(k for label in labels for k in ('wrong', 'looped', 'cut_off') if label[k])
        result['arms'][arm] = {'labels': labels, 'counts': {k: counts[k] for k in ('wrong', 'looped', 'cut_off')}}
    ref, cand = (result['arms'][a]['counts'] for a in ('reference', 'candidate'))
    result['reference_futility'] = ref['cut_off'] / len(ids) > .20
    result['flags'] = {k: cand[k] > 2 * ref[k] for k in ('looped', 'cut_off')}
    result['verdict'] = 'redeclare-cap' if result['reference_futility'] else ('flag' if any(result['flags'].values()) else 'not-flagged')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('prompts')
    ap.add_argument('captures')
    ap.add_argument('--cap', type=int, required=True)
    args = ap.parse_args()
    print(json.dumps(score(args.prompts, args.captures, args.cap), indent=2))


if __name__ == '__main__':
    main()
