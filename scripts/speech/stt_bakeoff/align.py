"""align.py — token-level minimum-edit alignment shared by every STT bake-off scorer.

One primitive, used by the WER scorer (CS-8.STT1, P-STT-1 §1.4) and the keyword
scorer (CS-8.LEX1, Contextual Earnings-22 definition), so the two never disagree
about which hypothesis token sits at which reference position.

Contract (P-STT-1 §1.4): a token is a whitespace-delimited unit of the normalised
string; the utterance error count is the Levenshtein distance with unit cost for
substitution, deletion and insertion (S + D + I, no weighting). No fuzzy or
edit-distance-tolerant token matching: two tokens match iff they are equal strings.

The backtrace is deterministic: on equal cost it prefers match/substitution, then
deletion, then insertion, so the same pair of sequences always yields the same
alignment (the determinism property of §1.3 carried over to the reduction).

Pure function module: no I/O, no process, no inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

MATCH = "match"
SUB = "sub"
DEL = "del"  # reference token with no hypothesis token
INS = "ins"  # hypothesis token with no reference token


@dataclass(frozen=True)
class AlignedPair:
    """One alignment column. `ref_idx`/`hyp_idx` are None for INS/DEL respectively."""
    op: str
    ref_idx: Optional[int]
    hyp_idx: Optional[int]
    ref_tok: Optional[str]
    hyp_tok: Optional[str]


@dataclass(frozen=True)
class Alignment:
    pairs: Tuple[AlignedPair, ...]
    substitutions: int
    deletions: int
    insertions: int
    n_ref: int
    n_hyp: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    """Levenshtein distance over tokens (unit costs). O(len(ref)*len(hyp)) time, O(len(hyp)) space."""
    ref = list(ref)
    hyp = list(hyp)
    prev = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        cur = [i] + [0] * len(hyp)
        r = ref[i - 1]
        for j in range(1, len(hyp) + 1):
            cost = 0 if r == hyp[j - 1] else 1
            cur[j] = min(prev[j - 1] + cost, prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return prev[-1]


def align(ref: Sequence[str], hyp: Sequence[str]) -> Alignment:
    """Minimum-edit alignment with a deterministic backtrace.

    Primary objective: minimum S+D+I (the P-STT-1 error count). Secondary objective,
    among equal-cost alignments: maximum number of exact matches, so a correctly
    recognised token is aligned to its reference position whenever an optimal
    alignment allows it (the keyword scorer counts a true positive only at the
    aligned position). Remaining ties: match/sub, then deletion, then insertion.
    """
    ref = list(ref)
    hyp = list(hyp)
    n, m = len(ref), len(hyp)
    # cell = (errors, -matches); compared lexicographically.
    d: List[List[Tuple[int, int]]] = [[(0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = (i, 0)
    for j in range(m + 1):
        d[0][j] = (j, 0)
    for i in range(1, n + 1):
        r = ref[i - 1]
        row, up = d[i], d[i - 1]
        for j in range(1, m + 1):
            if r == hyp[j - 1]:
                diag = (up[j - 1][0], up[j - 1][1] - 1)
            else:
                diag = (up[j - 1][0] + 1, up[j - 1][1])
            dele = (up[j][0] + 1, up[j][1])
            inse = (row[j - 1][0] + 1, row[j - 1][1])
            row[j] = min(diag, dele, inse)

    pairs: List[AlignedPair] = []
    s = de = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            eq = ref[i - 1] == hyp[j - 1]
            prev = d[i - 1][j - 1]
            diag = (prev[0], prev[1] - 1) if eq else (prev[0] + 1, prev[1])
            if d[i][j] == diag:
                if not eq:
                    s += 1
                pairs.append(AlignedPair(MATCH if eq else SUB, i - 1, j - 1, ref[i - 1], hyp[j - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and d[i][j] == (d[i - 1][j][0] + 1, d[i - 1][j][1]):
            de += 1
            pairs.append(AlignedPair(DEL, i - 1, None, ref[i - 1], None))
            i -= 1
            continue
        ins += 1
        pairs.append(AlignedPair(INS, None, j - 1, None, hyp[j - 1]))
        j -= 1
    pairs.reverse()
    out = Alignment(tuple(pairs), s, de, ins, n, m)
    assert out.errors == d[n][m][0] == edit_distance(ref, hyp)
    return out
