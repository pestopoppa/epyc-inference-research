#!/usr/bin/env python3
"""Native-sidecar adapter; actual process runner must be supplied by canonical engine."""
import os
import subprocess
import tempfile
import time
from pathlib import Path

def run_process(command, env, binding, *, deadline, stop_path):
    """Job-local native child; canonical launcher still owns admission and grants."""
    if time.monotonic() >= deadline or Path(stop_path).exists():
        raise InterruptedError('stopped before child launch')
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        proc=subprocess.Popen(command,env=env,stdout=out,stderr=err)
        proc_path=Path('/proc')/str(proc.pid)
        stat=(proc_path/'stat').read_text().rsplit(') ',1)[1].split()
        identity={'pid':proc.pid,'uid':proc_path.stat().st_uid,'birth_ticks':int(stat[19])}
        if identity['uid'] != os.getuid():
            proc.terminate(); proc.wait()
            raise RuntimeError('native child ownership mismatch')
        mappings=set()
        kfd_seen=False
        interrupted=False
        try:
            while proc.poll() is None:
                try:
                    current=(proc_path/'stat').read_text().rsplit(') ',1)[1].split()
                    if int(current[19]) != identity['birth_ticks'] or proc_path.stat().st_uid != identity['uid']:
                        raise RuntimeError('native child identity changed')
                    for line in (proc_path/'maps').read_text().splitlines():
                        if any(name in line for name in ('libggml','libllama','libmtmd')):
                            mappings.add(line)
                    for fd in (proc_path/'fd').iterdir():
                        try:
                            if os.readlink(fd).startswith('/dev/kfd'): kfd_seen=True
                        except FileNotFoundError:
                            pass
                except FileNotFoundError:
                    if proc.poll() is None: raise RuntimeError('live native mappings unavailable')
                if time.monotonic() >= deadline or Path(stop_path).exists():
                    interrupted=True
                    proc.terminate()
                    try: proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill(); proc.wait()
                    break
                time.sleep(.05)
            rc=proc.wait()
            out.seek(0,os.SEEK_END); err.seek(0,os.SEEK_END)
            if out.tell()>64*1024*1024 or err.tell()>64*1024*1024:
                raise ValueError('native output cap exceeded')
            out.seek(0); err.seek(0)
            return {'rc':rc,'reaped':proc.poll() is not None,'interrupted':interrupted,
                    'stdout':out.read(),'stderr':err.read(),'process':identity,'mappings':sorted(mappings),
                    'kfd_fd_seen':kfd_seen}
        finally:
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()

def engine_for(command, library_path, sidecar_path, runner):
    path=Path(sidecar_path)
    if path.exists() or path.is_symlink(): raise FileExistsError('sidecar path already occupied')
    if not command or not library_path: raise ValueError('missing command or linkage path')
    def engine(binding):
        # canonical runner must own process, enforce STOP/deadline, and prove reaping
        env={**os.environ,'LD_LIBRARY_PATH':library_path,'CS8_TRANSCRIPT_PATH':str(path)}
        raw=runner(command,env,binding)
        if path.is_symlink(): raise ValueError('sidecar symlink refused')
        if path.exists():
            if not path.is_file() or path.stat().st_size>64*1024*1024: raise ValueError('invalid sidecar')
            raw['sidecar']=path.read_bytes()
        return raw
    return engine
