#!/usr/bin/env python3
"""CS-45 clip assembly: ONE function turns a recorded recipe into stereo PCM.

Builder, materializer and verifier all call ``assemble_clip``; the verifier additionally checks the
pair invariants on the resulting arrays, so a recipe that no longer reproduces its recorded
PCM-SHA-256 (or whose pair stops being an exact splice of one recording) is refused.

Channel layout (16 kHz, int16, 2 channels): ch0 = USER, ch1 = ASSISTANT (our TTS output channel).

Timeline (all sample counts at 16 kHz, constants below are part of the instrument version):

  ended context  : [LEAD_IN zeros][assistant utterance][GAP_AFTER_ASSIST zeros][user segment][TRAIL zeros]
  continuing ctx : [LEAD_IN zeros][assistant monologue, COMPLETE, to its natural end][END_PAD zeros]
                                   user segment starts BC_ONSET_IN_ASSIST after the assistant starts;
                                   the assistant keeps speaking through the user segment and finishes on its own
                                   (v1 truncated it at user_end + TRAIL, which the operator heard as the assistant
                                   stopping mid-utterance during a backchannel: the defect this version fixes).

`tail` = digital silence on the user channel after the user segment. CUT pairs: tail = TRAIL for both members.
CTX pairs: one tail shared by both members, long enough that the continuing context finishes (ctx_tail()).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SR = 16000
LEAD_IN = 4800  # 0.30 s of silence on both channels before anything happens
GAP_AFTER_ASSIST = 6400  # 0.40 s between the assistant ending and the user starting (ended context)
BC_ONSET_IN_ASSIST = 52800  # 3.30 s into a continuing assistant monologue
TRAIL = 32000  # 2.0 s of digital silence after the user's last sample (matched across a pair)
FADE = 160  # 10 ms raised-cosine fade-out at the end of every user segment (cut or natural)
ASSIST_FADE = 160  # 10 ms fade-out on the final samples of a continuing assistant context (it ends on its own)
N_CONT_MIN = 48000  # a continuing assistant must still be speaking >= 3.0 s after the user segment ENDS

END_PAD = 8000  # 0.5 s of silence after a continuing assistant context ends naturally
INSTRUMENT_VERSION = "tt-corpus-2"  # v2 (2026-10-20): continuing assistant contexts run to their natural end; per-pair tail; speech-relative trim


@dataclass(frozen=True)
class Layout:
    user_onset: int  # first sample of the user segment on the clip clock
    user_len: int  # samples in the (cut + faded) user segment
    user_end: int  # user_onset + user_len
    total: int
    assistant_start: int | None
    assistant_end: int | None  # last+1 sample of assistant speech actually present in the clip
    tail: int = TRAIL  # silence on the user channel after the user segment (total == user_end + tail)


def _fade_out(x: np.ndarray, n: int) -> np.ndarray:
    y = x.astype(np.float64).copy()
    n = min(n, len(y))
    if n > 0:
        ramp = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, n, endpoint=True)))
        y[-n:] *= ramp
    return np.clip(np.rint(y), -32768, 32767).astype(np.int16)


def user_segment(rec: np.ndarray, trim_start: int, trim_end: int, cut: int | None) -> np.ndarray:
    """rec: mono int16 source recording. cut: sample offset INSIDE the trimmed segment (None = natural end)."""
    seg = rec[trim_start:trim_end]
    if cut is not None:
        if not (0 < cut < len(seg)):
            raise ValueError(f"cut {cut} outside trimmed segment of {len(seg)} samples")
        seg = seg[:cut]
    return _fade_out(seg, FADE)


def ctx_tail(assistant_len: int, seg_len: int) -> int:
    """Tail shared by both members of a context pair: long enough for the continuing assistant to finish + END_PAD."""
    return max(TRAIL, assistant_len + END_PAD - BC_ONSET_IN_ASSIST - seg_len)


def assemble_clip(
    user_rec: np.ndarray,
    trim: tuple[int, int],
    cut: int | None,
    assistant_rec: np.ndarray | None,
    assistant_mode: str,
    tail: int | None = None,
) -> tuple[np.ndarray, Layout]:
    """Return (int16 array [n,2], Layout). assistant_mode in {'ended','continuing'}."""
    tail = TRAIL if tail is None else int(tail)
    if tail < TRAIL:
        raise ValueError("tail shorter than TRAIL")
    seg = user_segment(user_rec, trim[0], trim[1], cut)
    if assistant_rec is None:
        raise ValueError("every CS-45 clip carries an assistant context channel")
    if assistant_mode == "ended":
        a_start = LEAD_IN
        a_end = a_start + len(assistant_rec)
        onset = a_end + GAP_AFTER_ASSIST
        total = onset + len(seg) + tail
        out = np.zeros((total, 2), dtype=np.int16)
        out[a_start:a_end, 1] = assistant_rec
        out[onset : onset + len(seg), 0] = seg
        lay = Layout(onset, len(seg), onset + len(seg), total, a_start, a_end, tail)
    elif assistant_mode == "continuing":
        a_start = LEAD_IN
        onset = a_start + BC_ONSET_IN_ASSIST
        total = onset + len(seg) + tail
        a_end = a_start + len(assistant_rec)
        if a_end + END_PAD > total:
            raise ValueError(f"tail {tail} too short: the continuing assistant context would be truncated ({a_end + END_PAD} > {total})")
        if onset + len(seg) + N_CONT_MIN > a_end:
            raise ValueError("continuing assistant context ends less than N_CONT_MIN after the user segment")
        out = np.zeros((total, 2), dtype=np.int16)
        out[a_start:a_end, 1] = _fade_out(assistant_rec, ASSIST_FADE)
        out[onset : onset + len(seg), 0] = seg
        lay = Layout(onset, len(seg), onset + len(seg), total, a_start, a_end, tail)
    else:
        raise ValueError(f"unknown assistant_mode {assistant_mode!r}")
    return out, lay


def check_pair_invariants(
    family: str,
    a_pcm: np.ndarray,
    a_lay: Layout,
    a_cut: int | None,
    b_pcm: np.ndarray,
    b_lay: Layout,
    b_cut: int | None,
) -> list[str]:
    """Independent check on the ASSEMBLED arrays. Returns a list of violations (empty = pair is a clean splice).

    CUT_* : members are the SAME recording; user channels agree sample-for-sample from the onset up to
            (shorter segment - FADE); each user channel is exact digital silence for TRAIL samples after
            its own end; both members end exactly TRAIL after the user's last sample; assistant channels
            are identical when shifted to the same origin.
    CTX_BC: the faded user segment is sample-identical in both members (it merely sits at a different
            clock position); trailing silence on the user channel is TRAIL in both; the assistant
            channel is the member-specific context and must differ.
    """
    v: list[str] = []
    ua, ub = a_pcm[:, 0], b_pcm[:, 0]
    if a_lay.tail != b_lay.tail:
        v.append("members do not share the same tail")
    if family in ("CUT_IC", "CUT_WAIT") and a_lay.tail != TRAIL:
        v.append("cut pairs must use tail == TRAIL")
    for tag, u, lay in (("a", ua, a_lay), ("b", ub, b_lay)):
        tail = u[lay.user_end : lay.user_end + lay.tail]
        if lay.tail < TRAIL or len(tail) != lay.tail or np.any(tail != 0):
            v.append(f"member {tag}: user channel is not exactly {lay.tail} samples of digital silence after the user segment")
        if lay.total != lay.user_end + lay.tail:
            v.append(f"member {tag}: clip does not end exactly tail after the user segment")
        if np.any(u[: lay.user_onset] != 0):
            v.append(f"member {tag}: user channel is non-zero before the user onset")
    if family in ("CUT_IC", "CUT_WAIT"):
        n = min(a_lay.user_len, b_lay.user_len) - FADE
        if n <= 0:
            v.append("cut segments shorter than the fade")
        else:
            if not np.array_equal(ua[a_lay.user_onset : a_lay.user_onset + n], ub[b_lay.user_onset : b_lay.user_onset + n]):
                v.append("user channels are not sample-identical over the shared prefix")
        if a_cut is not None and b_cut is not None and a_cut == b_cut:
            v.append("both members cut at the same point")
        if (a_cut is None) == (b_cut is None):
            v.append("exactly one member must run to the natural end")
        # assistant context identical (aligned at its own start)
        aa = a_pcm[a_lay.assistant_start : a_lay.assistant_end, 1]
        bb = b_pcm[b_lay.assistant_start : b_lay.assistant_end, 1]
        if not np.array_equal(aa, bb):
            v.append("assistant context differs between members of a cut-point pair")
    elif family == "CTX_BC":
        sa = ua[a_lay.user_onset : a_lay.user_end]
        sb = ub[b_lay.user_onset : b_lay.user_end]
        if not np.array_equal(sa, sb):
            v.append("user segment is not sample-identical between members of a context pair")
        if a_cut is not None or b_cut is not None:
            v.append("context pairs are never cut")
        if np.array_equal(a_pcm[:, 1][: min(len(a_pcm), len(b_pcm))], b_pcm[:, 1][: min(len(a_pcm), len(b_pcm))]):
            v.append("assistant channel is identical between members of a context pair")
        # the continuing member's assistant must finish on its own AFTER the user segment, and stay silent-padded to the end
        if b_lay.assistant_end is None or b_lay.assistant_end - b_lay.user_end < N_CONT_MIN:
            v.append("continuing assistant does not run at least N_CONT_MIN past the user segment")
        if b_lay.assistant_end is not None and b_lay.total - b_lay.assistant_end < END_PAD:
            v.append("continuing assistant context is truncated by the clip end")
    else:
        v.append(f"unknown family {family!r}")
    return v
