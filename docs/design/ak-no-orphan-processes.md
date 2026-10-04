# AutoKernel: no orphaned processes (DS41-C84)

Status: implemented on branch `ak-ds41-c84-no-orphans` (research repo), 2026-09-30.
Code: `scripts/kernel_rnd/autokernel/loop/procguard.py`, plus hooks in `ak_check.py`,
`actors.py`, `scratch.py`, `serving.py` and `run.py`. Tests: `loop/test_procguard.py`.

## 1. Incident

PID 930091, `anchor-gen-005/bin/test-backend-ops test -b CPU -o MUL_MAT,MUL_MAT_ID -p
type_a=(q4_K|q5_K|q8_0), -j 2`, ran from 2026-09-29 01:05Z for about 25 h. It ran at
nice 19, pinned to the ak-check op-test cores 88-95, which lie inside DS41's measured
set 0-95. It used 4-6 cores and accumulated about 6 days of CPU time. Its cwd was a
lane-0 author's ak-check scratch dir, and its exe was a pruned anchor generation. Both
were already deleted. It had its own PGID and SID and had been reparented to the
container subreaper. It ran through two 48-launch serving-floor calibrations and every
A/B of that day. Decode is memory-bandwidth-bound, so nice 19 shielded nothing.

The argv is exactly what `ak_check.op_test` builds (`DEFAULT_TYPES`, `-j 2`,
`ak_check.py:994`). `run_bounded` has not changed since before the incident. The live
loop runs `bf07f204`, which is this branch's base.

The same class of leak appears in the pytest fixtures. Two fake `llama-server` scripts
(`pytest-6516`/`6517`, `test_actual_loop_runtime_http_0`) had been running for 1.7 days
(`lstart` 09-28 07:32, parent = containerd-shim). The pytest processes that started them
were killed before the `finally` that terminates their fakes could run.

## 2. Root cause (file:line at `origin/main` = `bf07f204`)

Four facts together allow a process to escape:

1. **ak-check puts its child in a new session, and only its own timeout ends that
   session.** `ak_check.py:646-668` `run_bounded` starts
   `nice -n 19 taskset -c ... <argv>` with `start_new_session=True` (`:653`). The group
   is killed only in `except subprocess.TimeoutExpired` (`:657-668`). ak-check installs
   no signal handler: `main()` (`:1215`) and `__main__` (`:1342`) take the default
   action. If ak-check is SIGTERMed or SIGKILLed while waiting in `communicate()`, the
   Python process dies on the spot and the child keeps running with no deadline.
2. **opencode kills only its own detached group, and much earlier than ak-check's
   budget.** opencode 1.18.31 (`opencode.exe`, bundled JS) runs every shell-tool
   command with `detached:!0` (setsid) and `forceKillAfter: 3s`. On timeout it sends
   `process.kill(-pid, "SIGTERM")` and then SIGKILL to that group (`killTree`). The
   default timeout is `bashDefaultTimeoutMs ?? 120000`, overridable only by
   `OPENCODE_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS`, which the loop never sets. An
   `ak-check --op-test` can legitimately take 600 s of peer wait
   (`DEFAULT_PEER_WAIT_S`, `:121`; raised to 2700 s on 2026-10-04), plus 120 s per compiled TU, 120 s of link and 300 s
   of op test (`:112-113`). So at 120 s opencode TERMs the shim/ak-check group
   (bash execs python, so it is one group). ak-check dies from rule 1, and
   test-backend-ops, which sits in yet another session, is untouched. The same thing
   happens if opencode aborts the tool because the loop ended the actor.
3. **The loop's actor kill reaches only the actor's own group, and a normal return
   kills nothing.** `actors.py:604` starts the actor with `start_new_session=True`, and
   `_end_group` (`:567`, `killpg` at `:562`) ends only that group, on stop, budget or
   timeout. On normal exit (`:637`) no group is ended at all. Neither reaches
   opencode's detached tool sessions. The non-stoppable branch
   `subprocess.run(..., timeout=)` (`:762`) kills only the direct child. The best-of
   ak-check validator (`bestof.py:557`) goes through the same `_run_stoppable`, and a
   panel stop TERMs ak-check's group: the same escape as rule 1.
4. **The fence was held by the wrong process.** ak-check's shared fence slot
   (`sandbox_slot`, `:198-227`, `flock` at `:217`) exists only on the Python process's
   fd, and `Popen` closes fds by default. When ak-check died, the slot was released
   while its test-backend-ops kept running. The tail then took the fence and measured
   beside it, which is the "lock, not observation" guarantee failing silently.

The measurement side had no census that would notice. `residency.CpuLifecycleSampler`
records `non_target_activity` as raw facts. By design (its docstring) it never assigns
campaign identity, and `cpu_lifecycle_invalidity` vetoes only on affinity
contradictions. A 25 h, 5-core campaign process therefore stayed "unproven" and was
never a refusal.

The harness faults "test-backend-ops did not prove a nonempty CPU suite" (experiments
rows 108, 109 and 115, 09-29 09:15-11:06Z, lane 1, `gate_refused`) are ak-check results
from `ak_check.py:1012`. They were recorded while the orphan was running and are
consistent with contention on the op-test cores 88-95. They do not prove a mechanism:
ak-check prints its own verdict when its child exits non-zero without an `OK` line,
which the orphan does not directly cause. Treat them as related, not as root cause.

## 3. Options evaluated

| | Mechanism | Works here? | Verdict |
|---|---|---|---|
| (a) | `PR_SET_CHILD_SUBREAPER` on the serial parent / run.py | yes (prctl is unprivileged) | **Rejected.** Orphans would reparent to the loop, but the loop must then reap them. `waitpid(-1)` steals the exit status of Popen children, and CPython turns the resulting ECHILD into `returncode = 0`. A failing gate or compiler would then read as a pass. Reaping only "adopted" PIDs needs a registry of every Popen in a large, threaded process. Changes to the serial parent take effect only when it restarts. The cookie gives the same identity without adoption. |
| (b) | cgroup v2 scope per call, `cgroup.kill` | **partly.** No `systemd-run` (pid 1 is the host's systemd; this container has no user manager). `/sys/fs/cgroup` is cgroup2, `rw,nsdelegate`, owned by `node`, `subtree_control` empty, and `cgroup.kill` exists (kernel 6.14). So `mkdir` of a child cgroup plus `cgroup.kill` should work (not exercised: operator filesystem-containment rule). | **Deferred (recommended follow-up for actor calls only).** It is the strongest mechanism, because a process cannot leave a cgroup by setsid. But it needs the child moved in before it forks: either a `preexec_fn` (unsafe in this threaded loop) or an argv wrapper (changes recorded argv). It depends on the devcontainer config, which a rebuild can change. It also writes into the container's shared cgroup tree, where leaked empty cgroups need their own cleanup. `worker_lifecycle.OwnedContainer` already defines the protocol (create/add/kill/wait_empty) behind an operator-owned provider, but the DS41 serving/actor path does not use it. A provider built on the writable subtree is the natural home for this follow-up. |
| (c) | `PR_SET_PDEATHSIG` | yes | **Adopted for ak-check's children** via `/usr/bin/setpriv --pdeathsig KILL` in front of `nice` (it survives the nice/taskset execs). It is not used for actors or fixtures, because the signal follows the spawning **thread**, and the loop spawns from worker threads. |
| (d) | group/session kill on every timeout, with verify-dead and TERM→KILL | yes | **Adopted in ak-check** (`run_bounded` ends the group on every exit path and verifies it empty via `killpg(0)` plus a `/proc` pgrp scan; TERM/HUP/INT handlers end live groups before exiting). For actors a group kill is insufficient (rule 3), so the call-scope sweep below generalizes it. |
| (e) | measurement census: refuse or flag a contended sample; sweep what is provably ours | yes | **Adopted** (`MeasurementWatch`). |
| (f) | scratch release refuses or vacates a dir with a process inside | yes | **Adopted** (`ScratchRegistry._vacate`). |
| (g) | **provenance cookie** in the environment, swept by exact value | yes | **Adopted: the backbone.** This is the Jenkins ProcessTreeKiller technique. |

## 4. Design: the recommended minimal robust set

**Backbone: the provenance cookie (`procguard.Guard.call_scope`).** Every actor or
validator process gets
`AK_PROC_SCOPE=<campaign-key>.<owner-pid>.<owner-start-ticks>.<uuid>` in its env.
Environments survive fork, exec, setsid and double-fork, so opencode's detached tool
sessions, ak-check, the compiler and test-backend-ops all carry it in
`/proc/<pid>/environ`. When the call ends, on every path (return, stop, budget, timeout,
exception), each process still carrying that exact value is ended. The name contains no
`KEY`, `TOKEN` or `SECRET`, so codex's default env-var excludes keep it. The recorded
seat env is unchanged; the cookie sits next to the per-attempt `TMPDIR`, which is
already unrecorded.

**Signalling discipline.** Only processes selected by provenance, only of this uid,
never by name, never as a group. Each one is pinned with `pidfd_open`, and its start
ticks and selection reason are re-read after pinning. It gets SIGTERM, a 5 s grace,
SIGKILL, and is then verified dead through pidfd readiness. Survivors are reported,
never assumed gone. The codex app-server daemon is exempt by exe prefix: the managed
daemon outlives every `codex exec` by design and serves the operator's own sessions.
Exempting is the safe direction of a path rule.

**Ownership classes (one function, `Guard.classify`).**

| class | meaning | action |
|---|---|---|
| self / ancestor / descendant | this loop's own tree | never touched |
| live | cookie of this owner, call still open | never touched |
| leak | cookie of this owner, call closed | ended |
| stale | cookie of this campaign, owner (pid + start ticks) dead | ended |
| peer | cookie of this campaign, owner alive and not an ancestor | flagged, never killed |
| orphan_scratch | no cookie, cwd/exe under this campaign's scratch, **orphaned** (parent is init or one of this loop's own ancestors, i.e. the shared container subreaper), no controlling tty | ended (the operator's rule: under campaign scratch is provably ours; the C84 orphan predates the cookie). A child of some other live process (a peer loop's work) or a process with a tty (somebody's shell) is never swept without a cookie |
| campaign_foreign | cwd/exe under the campaign store, not provably ours | flagged, never killed |
| exempt | shared daemon | never touched, never flagged |

**Where it runs.**

1. `actors._run_stoppable` (the actor path and the best-of ak-check validator) and the
   `subprocess.run` actor branch: the call scope.
2. `ak_check.run_bounded`: `setpriv --pdeathsig KILL`, a group end with verification on
   every exit path, TERM/HUP/INT handlers (script entry only, never an importer), and
   the fence slot and lane-lock fds passed to the child. The lock now follows the work:
   a child that somehow survives keeps the fence, so the tail refuses with
   `TailFenceTimeout` instead of measuring beside it.
3. `run.py` installs the guard before the run scope. It sweeps stale processes at run
   start (a batch boundary: the previous child's leftovers), keeps the guard installed
   while scopes release, and sweeps again at run end.
4. `serving._measure_once` (CPU backend: A/B, floor calibration and runtime
   calibration all pass through it) runs a `MeasurementWatch`.
   - **Before:** it ends provably-ours orphans.
   - **During:** a census every 3 s. A FOREIGN-class process (leak, stale,
     orphan_scratch, campaign_foreign, peer) that burns at least 10% of a core in two
     consecutive intervals becomes a `campaign_process_contention` condition, and the
     launch raises `MeasurementInvalid` through the existing invalid-arm path. No kill
     happens during timed rounds.
   - Static facts (cwd, exe, cookie) are cached per (pid, start ticks), so the
     steady-state cost is one `stat` read per process, the same order as the existing
     `CpuLifecycleSampler`.
5. `ScratchRegistry._release_dir` and `remove_worktree`: `_vacate` ends provably-ours
   processes inside the path first. If anything survives, including a live or peer
   process that is not ours to end, the release is refused and journalled
   (`release_failed`), and the dir stays for the next sweep.

**Fixtures.** Every out-of-process fake (`test_legacy_cpu_serving._server`,
`test_cpu_profile_runtime` server and perf, and `test_native_server_response`'s
contained HTTP server) starts with `FIXTURE_PARENT_WATCH`. This is a daemon thread that
exits the fake when `getppid()` changes, with a hard cap of 900 s
(`AK_FIXTURE_MAX_S`). It uses getppid, not PDEATHSIG, because fixtures spawn from
worker threads.

## 5. Tradeoffs and residual risk

- **Scrubbed environments.** A tool that runs `env -i` drops the cookie. Such a process
  is still caught by `orphan_scratch` (scratch cwd), by the release vacate, and by the
  measurement flag. It is not caught if it lives entirely outside the campaign roots.
- **Loop-spawned children when run.py is SIGKILLed** (llama-server, builds, witness
  probes) carry no cookie, because we deliberately do not modify `os.environ`: that
  could alter recipe or launch-env digests. A leaked llama-server whose exe is under
  the store is flagged by the next measurement (campaign_foreign, CPU-active) and
  holds the port, so it is visible, not silent. A cgroup per run (option b) would
  close this gap.
- **False-positive refusal.** A long CPU job run by an operator from inside the
  campaign store would refuse measurements until it ends. That is correct: it
  contends.
- **Cost.** Each call end and each scratch release does one `/proc` scan (a few ms).
  The watch adds about one `stat` per process every 3 s.

## 6. Follow-ups (not in this change)

- `OPENCODE_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS` for author seats. The 120 s default is
  below ak-check's worst case, so today every slow `--op-test` is cut off and the author
  never sees its result. This is a seat-behaviour change (it affects measured author
  efficiency and seat A/B comparability), so it needs an operator decision.
  Recommendation: set it to 1,200,000 for author seats only.
- The op-test suite appears far larger than its 300 s budget. The orphan computed for
  25 h at about 5.8 cores. Once ak-check's own deadline is enforced again, expect
  "ran out of the op-test budget" results until the suite is narrowed (for example,
  fewer shapes per type).
- cgroup-per-actor-call (option b), if the devcontainer config is made part of the
  contract.

## 7. Deploy

No epoch, fingerprint or resume-binding input changes. Recipe, launch env, recorded seat
env, observer schemas and stored evidence are untouched. A new invalid-arm condition
applies only to NEW launches, and `_reopen_launch` re-judges stored bodies with the
unchanged `cpu_lifecycle_invalidity`. The change is safe to take at a batch boundary:
the serial parent keeps running old code, and each new run.py child installs the guard.
At its first start it ends the leftovers of the previous child (dead-owner cookies,
no-cookie orphans in `store/scratch`).
