#!/usr/bin/env python3
"""ALSA digital PCM output sink; reports only verified local PCM-drop timing.

SND_PCM_STATE_SETUP after snd_pcm_drop witnesses the ALSA queue is stopped.
It does not assert acoustic silence or downstream HDMI/DAC buffer state.
A short/nonblocking write fails closed; no unbounded background playback.
"""
import ctypes
import time

class ALSADigitalSink:
    def __init__(self, device, rate, *, api=None, clock=time.monotonic_ns):
        self.clock=clock; self.api=api or ctypes.CDLL('libasound.so.2'); self.handle=ctypes.c_void_p(); self.closed=False
        a=self.api
        a.snd_pcm_open.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_char_p,ctypes.c_int,ctypes.c_int]; a.snd_pcm_open.restype=ctypes.c_int
        a.snd_pcm_set_params.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_uint,ctypes.c_uint,ctypes.c_int,ctypes.c_uint]; a.snd_pcm_set_params.restype=ctypes.c_int
        a.snd_pcm_writei.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]; a.snd_pcm_writei.restype=ctypes.c_long
        for name in ('snd_pcm_drop','snd_pcm_state','snd_pcm_close'):
            f=getattr(a,name); f.argtypes=[ctypes.c_void_p]; f.restype=ctypes.c_int
        # stream=PLAYBACK(0), NONBLOCK(1); S16_LE(2), RW_INTERLEAVED(3)
        rc=a.snd_pcm_open(ctypes.byref(self.handle),device.encode(),0,1)
        if rc<0: raise RuntimeError(f'ALSA open failed: {rc}')
        rc=a.snd_pcm_set_params(self.handle,2,3,1,rate,1,500000)
        if rc<0:
            a.snd_pcm_close(self.handle); self.closed=True
            raise RuntimeError(f'ALSA configure failed: {rc}')
    def push(self, pcm, ns):
        if self.closed or not pcm or len(pcm)%2: raise ValueError('invalid PCM sink input')
        buf=ctypes.create_string_buffer(pcm); frames=len(pcm)//2
        written=self.api.snd_pcm_writei(self.handle,buf,frames)
        if written != frames: raise RuntimeError(f'ALSA partial/nonblocking write: {written}/{frames}')
        return self.clock()
    async def cancel_and_drain(self, ns):
        if self.closed: raise RuntimeError('ALSA sink already closed')
        rc=self.api.snd_pcm_drop(self.handle); state=self.api.snd_pcm_state(self.handle)
        if rc<0 or state != 1: raise RuntimeError(f'ALSA drop unverified: {rc}, state {state}')
        return self.clock()
    def close(self):
        if not self.closed:
            rc=self.api.snd_pcm_close(self.handle); self.closed=True
            if rc<0: raise RuntimeError(f'ALSA close failed: {rc}')
