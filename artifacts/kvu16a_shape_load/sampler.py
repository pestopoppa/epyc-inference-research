#!/usr/bin/env python3
"""Candidate replay own-PID KFD sampler; no process creation or termination.

The caller starts this only after recording a captured child PID and birth identity.
Read failures set a sticky refusal event. The owning runtime must interrupt the
request and drain its own child on refusal; it must never reattach to a reused PID.
"""
import json
import os
from pathlib import Path
import threading
import time
import device_identity

GIB = 1024 ** 3
HARD = int(62.5 * GIB)
SOFT = int(60.5 * GIB)


def start_ticks(procdir):
    # comm may contain spaces/parentheses; the final ')' precedes state and fields.
    text = (procdir / 'stat').read_text()
    return int(text[text.rfind(')') + 2:].split()[19])


def read_sample(pid, birth, hip_library, proc_root=Path('/proc'), kfd_root=Path('/sys/class/kfd/kfd/proc'),
                card=None, allow_uninitialized=False, device=None):
    proc = proc_root / str(pid)
    before = start_ticks(proc)
    if before != birth:
        raise RuntimeError('captured server birth identity changed')
    maps = (proc / 'maps').read_text()
    hip = any(line.split()[-1] == hip_library for line in maps.splitlines() if line.split() and line.split()[-1].startswith('/'))
    if device is None:
        raise RuntimeError('validated MI210 DRM/KFD binding is required')
    device_identity.revalidate(device)
    gpu = kfd_root / str(pid)
    files = list(gpu.glob('vram_*'))
    target = gpu / f"vram_{device['gpu_id']}"
    # Only the bound KFD gpu_id can be charged as own residency. An allocation
    # on any other GPU refuses even before our target reaches positive VRAM.
    for f in files:
        if f.name != target.name and int(f.read_text().strip()) > 0:
            raise RuntimeError('captured process has residency on an unbound GPU')
    if not target.is_file() and not allow_uninitialized:
        raise RuntimeError('own process has no target KFD VRAM entry')
    own = int(target.read_text().strip()) if target.is_file() else None
    card_used = int(Path(device['card_vram_path']).read_text().strip())
    device_identity.revalidate(device)
    if start_ticks(proc) != birth:
        raise RuntimeError('captured server identity changed while sampling')
    if (own is not None and own < 0) or card_used < 0:
        raise RuntimeError('invalid VRAM value')
    return {'t': time.monotonic(), 'pid': pid, 'birth_ticks': birth, 'own_vram_b': own,
            'card_used_b': card_used, 'hip_resident': hip, 'gpu_id':device['gpu_id'],
            'gpu_unique_id':device['unique_id'], 'gpu_bdf':device['bdf']}


def ceiling(sample):
    if (sample['own_vram_b'] is not None and sample['own_vram_b'] > HARD) or sample['card_used_b'] > HARD:
        return 'hard'
    if sample['own_vram_b'] is not None and sample['own_vram_b'] > SOFT:
        return 'soft'
    return None


class Sampler:
    def __init__(self, pid, birth, hip_library, sink, interval=0.5, reader=read_sample, startup=False, phase=None, device=None):
        self.pid, self.birth, self.hip_library = pid, birth, hip_library
        self.sink, self.interval, self.reader = Path(sink), interval, reader
        self.startup = startup
        self.device = device
        self.phase = phase or (lambda: 'unknown')
        self.residency_seen = False
        self.abort, self.stop_event = threading.Event(), threading.Event()
        self.reason = None
        self.last = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            with self.sink.open('x', encoding='utf-8') as out:
                while not self.stop_event.is_set():
                    t_start = time.monotonic()
                    kwargs = {}
                    if self.device is not None: kwargs['device'] = self.device
                    if self.startup and not self.residency_seen: kwargs['allow_uninitialized'] = True
                    record = self.reader(self.pid, self.birth, self.hip_library, **kwargs)
                    record['sample_start_monotonic_s'] = t_start
                    record['sample_end_monotonic_s'] = time.monotonic()
                    record['status'] = 'observed'
                    if record.get('hip_resident') is True and (record.get('own_vram_b') or 0) > 0:
                        self.residency_seen = True
                    record['phase'] = self.phase()
                    self.last = record
                    out.write(json.dumps(record, sort_keys=True) + '\n')
                    out.flush()
                    os.fsync(out.fileno())
                    level = ceiling(record)
                    if level:
                        self.reason = level + ' VRAM ceiling exceeded'
                        self.abort.set()
                        break
                    self.stop_event.wait(self.interval)
        except Exception as exc:
            self.reason = f'sampler refusal: {type(exc).__name__}: {exc}'
            self.abort.set()
            try:
                with self.sink.open('a', encoding='utf-8') as out:
                    out.write(json.dumps({'status': 'missing', 'phase': self.phase(),
                                          't': time.monotonic(), 'error': self.reason}) + '\n')
            except OSError:
                pass

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            self.reason = 'sampler failed to drain'
            self.abort.set()
        return not self.thread.is_alive()
