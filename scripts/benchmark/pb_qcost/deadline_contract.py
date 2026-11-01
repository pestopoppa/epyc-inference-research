"""Pure deadline serialization shared by PB-QCOST wrappers and driver tests."""
import math
import time


def deadline_arg_utc(epoch: float) -> str:
    """Serialize an epoch using the PB-QCOST CLI's documented UTC-Z contract."""
    if not math.isfinite(epoch):
        raise ValueError("deadline epoch must be finite")
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))
