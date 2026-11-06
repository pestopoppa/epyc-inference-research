"""Shared GPU residency proof for the Q38P packets: VRAM + KFD processes sampled DURING a step, attributed to OUR
process (own session id) or OUR container (cgroup contains the container id). Host PID namespace is shared with
this container (verified 2026-11-04: /proc/1 is host systemd)."""
import os, re, signal, subprocess, threading, time

KFD = '/sys/class/kfd/kfd/proc'


def gpu_card():
    for c in sorted(os.listdir('/sys/class/drm')):
        p = f'/sys/class/drm/{c}/device'
        if re.fullmatch(r'card\d+', c) and os.path.exists(p + '/mem_info_vram_used'):
            return p
    return None


def rd(p, d=None):
    try:
        return int(open(p).read().strip())
    except (OSError, ValueError):
        return d


FOREIGN_VRAM_MIN_B = 64 << 20   # a foreign KFD pid below this holds no model: a monitor, not a competitor


def kfd_pids():
    try:
        return sorted(int(p) for p in os.listdir(KFD) if p.isdigit())
    except OSError:
        return []


class Sampler(threading.Thread):
    """Every 0.5 s: device VRAM used, busy %, all KFD pids, and the KFD VRAM of pids that are OURS
    (getsid == sid, or /proc/<pid>/cgroup mentions container_id)."""
    def __init__(self, sid=None, container_id=None, exe_suffix=None):
        super().__init__(daemon=True)
        self.sid, self.cid, self.exe_suffix = sid, container_id, exe_suffix
        self.stop = threading.Event(); self.card = gpu_card()
        self.vram, self.busy, self.all_kfd, self.ours, self.maps = [], [], set(), {}, {}
        self.foreign = {}

    def mine(self, pid):
        try:
            if self.sid is not None and os.getsid(pid) == self.sid:
                return True
            if self.cid and self.cid[:12] in open(f'/proc/{pid}/cgroup').read():
                return True
        except OSError:
            pass
        return False

    def run(self):
        while not self.stop.is_set():
            if self.card:
                self.vram.append(rd(self.card + '/mem_info_vram_used', 0)); self.busy.append(rd(self.card + '/gpu_busy_percent', 0))
            for p in kfd_pids():
                self.all_kfd.add(p)
                if not self.mine(p):
                    # A foreign KFD client counts only if it holds VRAM: the keeper's ~5 s `rocm-smi` sampler and other
                    # monitors open KFD with 0 VRAM (q38p4 smoke 2026-11-05: 4 such pids in 26 s, all transient).
                    d = os.path.join(KFD, str(p))
                    try:
                        v = sum(rd(os.path.join(d, f), 0) for f in os.listdir(d) if f.startswith('vram_'))
                    except OSError:
                        v = 0
                    try:
                        comm = open(f'/proc/{p}/comm').read().strip()
                    except OSError:
                        comm = '?'
                    prev = self.foreign.get(p, (0, comm))
                    self.foreign[p] = (max(prev[0], v), prev[1] if prev[1] != '?' else comm)
                if self.mine(p):
                    d = os.path.join(KFD, str(p))
                    try:
                        v = sum(rd(os.path.join(d, f), 0) for f in os.listdir(d) if f.startswith('vram_'))
                    except OSError:
                        v = 0
                    self.ours[p] = max(self.ours.get(p, 0), v)
                    if p not in self.maps:
                        try:
                            self.maps[p] = {'exe': os.readlink(f'/proc/{p}/exe'),
                                            'libs': sorted({os.path.realpath(l.split()[-1]) for l in open(f'/proc/{p}/maps')
                                                            if re.search(r'libggml|libamdhip64|libhsa-runtime', l)})}
                        except OSError:
                            pass
            self.stop.wait(0.5)

    def record(self, v0):
        vmax = max(self.vram) if self.vram else None
        return {'vram_before_b': v0, 'vram_max_b': vmax, 'vram_delta_b': (vmax - v0) if vmax is not None and v0 is not None else None,
                'samples': len(self.vram), 'busy_max_pct': max(self.busy) if self.busy else None,
                'kfd_pids_seen_all': sorted(self.all_kfd), 'kfd_pids_ours': {str(k): v for k, v in self.ours.items()},
                'kfd_foreign': sorted(p for p in self.all_kfd if p not in self.ours and self.foreign.get(p, (0,))[0] >= FOREIGN_VRAM_MIN_B),
                'kfd_foreign_all': {str(p): {'vram_b': v, 'comm': c} for p, (v, c) in sorted(self.foreign.items())}, 'maps': {str(k): v for k, v in self.maps.items()}}


def run_sampled(argv, log, env=None, timeout_at=None, stopfile=None, container_name=None, cwd=None, errlog=None):
    """Run argv in its own session, sampling during. If container_name is given, the container id is resolved as soon as it
    exists and KFD pids inside that container count as ours. Returns (rc, error, proof)."""
    card = gpu_card(); v0 = rd(card + '/mem_info_vram_used', 0) if card else None
    err = None
    with open(log, 'wb') as f, open(errlog or os.devnull, 'wb') as fe:
        p = subprocess.Popen(argv, stdout=f, stderr=fe if errlog else subprocess.STDOUT, env=env, cwd=cwd, start_new_session=True)
        s = Sampler(sid=p.pid); s.start()
        while p.poll() is None:
            if container_name and not s.cid:
                r = subprocess.run(['docker', 'inspect', '-f', '{{.Id}}', container_name], capture_output=True, text=True)
                if r.returncode == 0 and r.stdout.strip():
                    s.cid = r.stdout.strip()
            if (stopfile and os.path.exists(stopfile)) or (timeout_at and time.time() >= timeout_at):
                err = 'stop' if stopfile and os.path.exists(stopfile) else 'deadline'
                if container_name:
                    subprocess.run(['docker', 'kill', container_name], capture_output=True)
                for sig, w in ((signal.SIGTERM, 20), (signal.SIGKILL, 10)):
                    try:
                        os.killpg(p.pid, sig)
                    except ProcessLookupError:
                        break
                    try:
                        p.wait(w); break
                    except subprocess.TimeoutExpired:
                        continue
                break
            time.sleep(0.5)
        rc = p.wait(); s.stop.set(); s.join(5)
    proof = s.record(v0); proof['container_id'] = s.cid
    return rc, err, proof
