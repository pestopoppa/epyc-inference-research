"""128-mel streaming frontend for Parakeet-Realtime-EOU-120M (CS-46 (b)).

Constants follow the parakeet-rs @6ad1501 ``parakeet_eou.rs`` contract:
16 kHz, n_fft 512, win 400 (symmetric Hann, as ``torch.hann_window(periodic=False)``),
hop 160, pre-emphasis 0.97, power spectrum, 128 mel bins, natural log with an
ADDITIVE guard of 2**-24 (NeMo ``log_zero_guard_type="add"``), no per-feature
normalisation and no dither (the model card says "raw log-mel").

``StreamingMel`` featurises only NEW frames per fed block.  With centred STFT
(torch ``center=True``, constant zero padding) frame ``i`` is the window of
the pre-emphasised signal at samples ``[i*hop-200, i*hop+200)`` (400-sample
window centred in the 512 FFT buffer), so it is final as soon as sample
``i*hop+200`` has arrived.  ``flush()`` emits the remaining frames with zero
right padding, which reproduces the offline whole-clip features exactly.

The mel filterbank flavour is a parameter: ``slaney`` (librosa/NeMo default,
what NeMo's ``AudioToMelSpectrogramPreprocessor`` builds) or ``htk`` (what
parakeet-rs ``create_mel_filterbank_htk`` builds).  The CS-46 feature diff
decides which one NeMo's EOU config actually uses.
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 128
PREEMPH = 0.97
LOG_ZERO_GUARD = 2.0 ** -24  # 5.9604645e-8 == parakeet-rs LOG_ZERO_GUARD
FMAX = 8000.0


def hann_symmetric(n: int = WIN_LENGTH) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * i / (n - 1))).astype(np.float32)


def _hz_to_mel_htk(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_to_hz_htk(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def _hz_to_mel_slaney(hz):
    hz = np.asarray(hz, dtype=np.float64)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(hz >= min_log_hz,
                    min_log_mel + np.log(np.maximum(hz, 1e-12) / min_log_hz) / logstep,
                    hz / f_sp)


def _mel_to_hz_slaney(mel):
    mel = np.asarray(mel, dtype=np.float64)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(mel >= min_log_mel,
                    min_log_hz * np.exp(logstep * (mel - min_log_mel)),
                    f_sp * mel)


def mel_filterbank(kind: str = "slaney", n_mels: int = N_MELS, n_fft: int = N_FFT,
                   sr: int = SAMPLE_RATE, fmax: float = FMAX) -> np.ndarray:
    """[n_mels, n_fft//2+1] float32 triangular filterbank with area (slaney) norm.

    ``slaney`` == librosa.filters.mel(htk=False, norm='slaney'); ``htk`` ==
    parakeet-rs ``create_mel_filterbank_htk`` (HTK mel points, 2/(f[i+2]-f[i]) norm).
    """
    if kind == "slaney":
        h2m, m2h = _hz_to_mel_slaney, _mel_to_hz_slaney
    elif kind == "htk":
        h2m, m2h = _hz_to_mel_htk, _mel_to_hz_htk
    else:
        raise ValueError(kind)
    n_freqs = n_fft // 2 + 1
    mel_pts = np.linspace(float(h2m(0.0)), float(h2m(fmax)), n_mels + 2)
    hz_pts = m2h(mel_pts)
    fft_freqs = np.arange(n_freqs, dtype=np.float64) * (sr / n_fft)
    w = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for i in range(n_mels):
        left, center, right = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        up = (fft_freqs - left) / (center - left)
        down = (right - fft_freqs) / (right - center)
        w[i] = np.maximum(0.0, np.minimum(up, down))
        w[i] *= 2.0 / (right - left)
    return w.astype(np.float32)


class StreamingMel:
    """Incremental log-mel extractor: ``push(pcm) -> new frames [n, 128]``."""

    def __init__(self, mel: str = "slaney", preemph: float = PREEMPH,
                 guard: float = LOG_ZERO_GUARD, fb: np.ndarray | None = None):
        self.fb = mel_filterbank(mel) if fb is None else fb
        self.window = hann_symmetric()
        self.preemph = float(preemph)
        self.guard = float(guard)
        self.reset()

    def reset(self) -> None:
        self._prev_raw = None            # last raw sample seen (pre-emphasis memory)
        self._tail = np.zeros(N_FFT // 2, dtype=np.float32)  # left zero pad (centre=True)
        self._tail_start = -(N_FFT // 2)  # signal index of _tail[0]
        self._next_frame = 0
        self._n_samples = 0

    @property
    def frames_emitted(self) -> int:
        return self._next_frame

    @property
    def samples_seen(self) -> int:
        return self._n_samples

    def _preemph(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        if self._prev_raw is None:
            y[0] = x[0]
            y[1:] = x[1:] - self.preemph * x[:-1]
        else:
            y[0] = x[0] - self.preemph * self._prev_raw
            y[1:] = x[1:] - self.preemph * x[:-1]
        self._prev_raw = x[-1]
        return y

    def _frames(self, upto_sample: int) -> np.ndarray:
        """Compute every not-yet-emitted frame i with i*hop+WIN//2 <= upto_sample."""
        half = WIN_LENGTH // 2
        # frame i is final iff i*hop + half <= available samples (signal idx < upto_sample+... )
        last = (upto_sample - half) // HOP_LENGTH  # largest final frame index
        n_new = last - self._next_frame + 1
        if n_new <= 0:
            return np.zeros((0, N_MELS), dtype=np.float32)
        return self._compute(self._next_frame, n_new)

    def _compute(self, first: int, n: int) -> np.ndarray:
        off = (N_FFT - WIN_LENGTH) // 2
        buf = np.zeros((n, N_FFT), dtype=np.float32)
        for k in range(n):
            centre = (first + k) * HOP_LENGTH
            s = centre - WIN_LENGTH // 2 - self._tail_start  # index into _tail of window start
            seg = self._tail[s:s + WIN_LENGTH]
            if seg.shape[0] < WIN_LENGTH:  # beyond known signal (flush): zero right pad
                seg = np.concatenate([seg, np.zeros(WIN_LENGTH - seg.shape[0], np.float32)])
            buf[k, off:off + WIN_LENGTH] = seg * self.window
        spec = np.fft.rfft(buf.astype(np.float64), axis=1)
        power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
        mel = power @ self.fb.T
        out = np.log(mel + np.float32(self.guard)).astype(np.float32)
        self._next_frame = first + n
        # drop signal no longer needed: next frame's window starts at nf*hop-200
        keep_from = self._next_frame * HOP_LENGTH - WIN_LENGTH // 2
        drop = keep_from - self._tail_start
        if drop > 0:
            self._tail = self._tail[drop:]
            self._tail_start += drop
        return out

    def push(self, pcm: np.ndarray) -> np.ndarray:
        x = np.asarray(pcm, dtype=np.float32).ravel()
        if x.size == 0:
            return np.zeros((0, N_MELS), dtype=np.float32)
        y = self._preemph(x)
        self._tail = np.concatenate([self._tail, y])
        self._n_samples += x.size
        return self._frames(self._n_samples)

    def flush(self) -> np.ndarray:
        """End of stream: emit frames up to floor(n_samples/hop) (NeMo's valid length)."""
        total = self._n_samples // HOP_LENGTH
        n = total - self._next_frame
        if n <= 0:
            return np.zeros((0, N_MELS), dtype=np.float32)
        # right pad with zeros, as torch.stft center=True does
        pad = np.zeros(N_FFT // 2, dtype=np.float32)
        self._tail = np.concatenate([self._tail, pad])
        return self._compute(self._next_frame, n)


def offline_mel(pcm: np.ndarray, mel: str = "slaney", **kw) -> np.ndarray:
    """Whole-clip features [T, 128], T = floor(len/hop) (NeMo valid-length masking)."""
    sm = StreamingMel(mel=mel, **kw)
    a = sm.push(pcm)
    b = sm.flush()
    return np.concatenate([a, b], axis=0)
