"""Sample owned descendant RSS during a bounded, externally CPU-pinned command."""
import os
import signal
import subprocess
import sys
import time

p = subprocess.Popen(sys.argv[1:], start_new_session=True)
peak = 0
samples = 0
start = time.monotonic()
while p.poll() is None:
    todo = [p.pid]
    seen = set()
    total = 0
    while todo:
        pid = todo.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1]) * 1024
            with open(f"/proc/{pid}/task/{pid}/children") as f:
                todo.extend(map(int, f.read().split()))
        except (FileNotFoundError, ProcessLookupError):
            pass
    samples += 1
    peak = max(peak, total)
    if total > 8 * 1024**3 or time.monotonic() - start > 900:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        print("guard terminated owned process group", file=sys.stderr)
        sys.exit(1)
    time.sleep(0.05)
print(f"guard samples={samples} peak_tree_rss_bytes={peak} exit={p.returncode}", file=sys.stderr)
sys.exit(p.returncode)
