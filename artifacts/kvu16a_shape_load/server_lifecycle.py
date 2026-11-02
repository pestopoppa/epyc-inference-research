#!/usr/bin/env python3
"""Captured-server lifecycle component for keeper job; never signals by name.

This module does not claim a GPU or invoke a job. The caller must run inside a
keeper-granted engine process group and supply a fresh PID file and output dir.
"""
import os
from pathlib import Path
import subprocess
import time
import sampler


class Server:
    def __init__(self, cmd, env, log, pidfile, *, factory=subprocess.Popen, birth_reader=None):
        self.cmd=list(cmd); self.env=dict(env); self.log=Path(log); self.pidfile=Path(pidfile)
        self.factory=factory
        self.birth_reader=birth_reader or (lambda p: sampler.start_ticks(Path('/proc') / str(p)))
        self.proc=None;self.birth=None;self.stream=None
    def start(self):
        if self.proc is not None: raise RuntimeError('server already started')
        self.stream=self.log.open('xb')
        try:
            # Inherit the engine's killable process group; no setsid/detachment.
            self.proc=self.factory(self.cmd, env=self.env, stdin=subprocess.DEVNULL,
                                   stdout=self.stream, stderr=subprocess.STDOUT, start_new_session=False)
            self.birth=self.birth_reader(self.proc.pid)
            with self.pidfile.open('a') as stream:
                stream.write(str(self.proc.pid)+'\n');stream.flush();os.fsync(stream.fileno())
            return self.proc.pid,self.birth
        except BaseException:
            # The caller must run cleanup for a captured child even if recording fails.
            if self.proc is None:
                self.stream.close();self.stream=None
            raise
    def alive(self):
        if self.proc is None or self.proc.poll() is not None: return False
        if self.birth_reader(self.proc.pid) != self.birth:
            raise RuntimeError('captured server identity changed')
        return True
    def close(self, term_grace=30, kill_grace=10):
        if self.proc is None: return {'captured':False,'verified_dead':True}
        sent=[]
        try:
            if self.alive():
                self.proc.terminate(); sent.append('TERM')
                try: self.proc.wait(timeout=term_grace)
                except subprocess.TimeoutExpired:
                    if self.alive():
                        self.proc.kill();sent.append('KILL')
                        self.proc.wait(timeout=kill_grace)
            dead=self.proc.poll() is not None
            return {'captured':True,'pid':self.proc.pid,'birth_ticks':self.birth,
                    'signals':sent,'verified_dead':dead,'rc':self.proc.returncode}
        finally:
            if self.stream is not None: self.stream.close();self.stream=None
