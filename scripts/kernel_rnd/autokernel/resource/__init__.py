"""autokernel.resource — the resource plane (§5.7, §6, §14 AK2).

Two halves live here, and the split is load-bearing:

* **Witness modules OBSERVE** host resource state — region-lock and device-claim
  witness, owned-process enumeration, host-health facts. They acquire nothing,
  launch nothing, terminate nothing, and signal nothing.
* **`device_claim.py` ACQUIRES.** It is the one exception, and it exists because
  invariant 9 demands it: *"Resources are acquired, not observed. Every CPU/GPU
  benchmark or profiler run holds the appropriate region/device claim. Idle
  sensing is never a claim."* §2.6 lists the cross-process GPU device claim as
  the first row of substrate that does not exist anywhere in this project, so
  observation alone cannot satisfy the invariant — someone has to hold the lock.
  It shares the CPU sibling's on-disk lock root (`cpu_region.*.lock` lives beside
  `gpu_device.*.lock`) so the two claims exclude across repositories, and it
  still signals no process: revocation is quiesce-and-drain, never a kill.

Process termination remains behind the orchestrator's auditable wrappers, which
are a different repository and a different trust boundary.

NAMING HAZARD (deliberate, documented): this package is named `resource`, which
is also a stdlib module name. Nothing in this package may `import resource`, and
no module here may be imported in a way that puts its PARENT directory on
`sys.path` ahead of the stdlib. AutoPilot's item-12 scar was exactly ambient
import identity — *"which code scores your eval depends on which eval ran first
in the process"* (§2.5) — so the shadowing risk is called out rather than left to
be discovered by a wrong-module import at 3am.
"""
