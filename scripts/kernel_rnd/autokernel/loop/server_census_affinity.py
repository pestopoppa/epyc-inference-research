"""Strict Linux CPU-list containment; thread pinning may narrow a held claim."""
import re


def cpus(text):
    if not isinstance(text,str) or not re.fullmatch(r'\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*',text):
        raise ValueError('nonempty Linux CPU list required')
    result = set()
    for part in text.split(','):
        lo,_,hi = part.partition('-')
        lo,hi = int(lo),int(hi or lo)
        if lo > hi or hi > 191:
            raise ValueError('CPU range outside host')
        result.update(range(lo,hi+1))
    return result


def within(affinity, claimed):
    try:
        allowed = cpus(claimed)
        return bool(affinity) and all(cpus(mask) <= allowed for mask in affinity.values())
    except ValueError:
        return False
