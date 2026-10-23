"""GPS-PIN-2: dflash2_capability_smoke.sh wiring (syntax + structure; nothing launched)."""
import re
import subprocess
from pathlib import Path

SH = Path(__file__).with_name("dflash2_capability_smoke.sh")


def test_syntax_and_wiring():
    assert subprocess.run(["bash", "-n", str(SH)]).returncode == 0
    t = SH.read_text()
    assert t.index("export HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0") < t.index("start_server() {")
    assert 'taskset -c "$PIN_CPUS"' in t
    assert re.search(r'--enforce "\$SRV_PID" "\$PIN_CPUS" --json\s*\\\n\s*> "\$OUT/affinity-\$arm\.json"', t)
    assert "REFUSE($arm): tasks escape" in t and "return 6" in t
    assert "pkill" not in t and "pgrep" not in t
    assert (SH.parent.parent / "kernel_rnd/autokernel/loop/launch_affinity.py").is_file()
