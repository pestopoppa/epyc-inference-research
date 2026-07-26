# Attempt 01 Root Cause

The sidecar did not fail because of the Laguna model or v8 HIP binary.

The first launcher invocation performed a full 37 GB SHA-256 before starting
the server. The tool returned while that shell was still progressing. A second
launcher then observed the port as free before the delayed first launcher
bound it. The second server logged `couldn't bind HTTP server socket`; the
first server loaded successfully, but was later reaped with the original
tool-parent shell. The SWE runner consequently waited on a dead endpoint.

No request was sent to Laguna. The waiting runner was terminated and verified
dead. Before retry, ROCm reported no KFD PID and 13,094,912 bytes used.

Retry contract: one launcher only, a new port, no full model hash in the
launch path (the prior pinned SHA and file size are recorded), and two health
checks separated by 30 seconds after the launch shell exits.
