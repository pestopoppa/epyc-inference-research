## ggml-cpu: `GGML_OP_MAP_CUSTOM1/2/3` and `GGML_OP_CUSTOM` ignore the caller's `n_tasks`

### Class of problem

Public-API contract violation on the CPU backend: `ggml_map_custom1/2/3()` and
`ggml_custom_4d()`/friends (`ggml/include/ggml.h`) let a caller request a specific
task count via their `n_tasks` parameter (`GGML_N_TASKS_MAX` meaning "use the graph's
full thread count" — 1 being the explicit, documented opposite: run the callback
once, serially). The CPU backend's scheduler (`ggml_get_n_tasks()`,
`ggml/src/ggml-cpu/ggml-cpu.c`) honours that request when sizing the graph, but the
four forward functions that actually invoke the user's callback ignore it and hand the
callback the *graph's* thread indices instead. This is not itself a memory-safety bug
inside ggml — it becomes one in **any caller that follows the documented contract**,
e.g. by sizing its own per-thread scratch buffer to the `n_tasks` it registered the op
with. Such a caller will overrun its own buffer, or otherwise race on state it believed
was single-threaded, the first time that op runs in a graph with `n_threads >
n_tasks`.

### Background: `n_tasks` is advisory on this backend

`ggml_graph_compute_thread()` sets `params.nth = n_threads` for every node
(`ggml-cpu.c`); `ggml_get_n_tasks()` is consumed only by `ggml_graph_plan()` for
scheduling/sizing and never reaches a kernel's `params->ith`/`params->nth` directly —
each kernel decides for itself how to interpret those. `MAP_CUSTOM1/2/3`/`CUSTOM` are
the one place a *caller-supplied* task count is part of the public contract (as opposed
to an internal planning heuristic), which is what makes ignoring it a contract
violation rather than an internal scheduling detail.

### The bug

Planner (`ggml_get_n_tasks()`, `ggml-cpu.c`) honours the caller's request:

```c
case GGML_OP_MAP_CUSTOM1:
    {
        struct ggml_map_custom1_op_params p;
        memcpy(&p, node->op_params, sizeof(p));
        if (p.n_tasks == GGML_N_TASKS_MAX) {
            n_tasks = n_threads;
        } else {
            n_tasks = MIN(p.n_tasks, n_threads);
        }
    } break;
```

(identical for `MAP_CUSTOM2`, `MAP_CUSTOM3`, `CUSTOM`). But the forward
(`ggml_compute_forward_map_custom1`, `ggml/src/ggml-cpu/ops.cpp`) passes the graph's
actual thread indices straight through, not the planned `n_tasks`:

```c
void ggml_compute_forward_map_custom1(const ggml_compute_params * params, ggml_tensor * dst) {
    ...
    p.fun(dst, a, params->ith, params->nth, p.userdata);   // params->nth == n_threads, always
}
```

A callback registered with `n_tasks == 1` — the documented way to request
single-threaded execution — is invoked once per worker thread, each seeing
`nth == n_threads`, not the `nth == 1` it was told to expect.

### Fix

Recompute the effective `n_tasks` in each of the four forwards (mirroring the
planner's own `GGML_N_TASKS_MAX` handling) and skip the callback on threads beyond
that count:

```c
static inline int ggml_custom_op_effective_n_tasks(int requested_n_tasks, int nth) {
    return requested_n_tasks == GGML_N_TASKS_MAX ? nth : MIN(requested_n_tasks, nth);
}

void ggml_compute_forward_map_custom1(const ggml_compute_params * params, ggml_tensor * dst) {
    ...
    const int n_tasks = ggml_custom_op_effective_n_tasks(p.n_tasks, params->nth);
    if (params->ith >= n_tasks) {
        return;
    }
    p.fun(dst, a, params->ith, n_tasks, p.userdata);
}
```

applied identically to `MAP_CUSTOM2`, `MAP_CUSTOM3`, and `CUSTOM`. This is the smallest
diff that restores the documented contract without changing behaviour for the (much
more common) `GGML_N_TASKS_MAX` case.

### Reproduction and regression check (attached)

`repro_map_custom_ntasks.c` registers a `MAP_CUSTOM1` op with `n_tasks = 1` (plus the
same unrelated padding `ADD1` node used in the other reproducer in this batch, for the
same reason — otherwise the graph's own thread count gets capped to 1 and the bug
can't be exercised) and counts how many times the callback runs and what `nth` it
observes. On unmodified upstream `master` (verified at `e71b805`, 2026-09-07), with
`n_threads = 8`:

```
running with n_threads=8, MAP_CUSTOM1 registered n_tasks=1 ...
  callback invoked: ith=0 nth=8
  callback invoked: ith=1 nth=8
  ... (8 total)
callback invocations: 8 (expected 1 if n_tasks were honoured)
max nth seen by any invocation: 8 (expected 1 if n_tasks were honoured)
CONTRACT VIOLATION CONFIRMED
```

With the fix applied, the same program reports exactly 1 invocation with `nth = 1`. A
second check (`regression_check.c`, attached) registers a `MAP_CUSTOM1` op with
`GGML_N_TASKS_MAX` and confirms all 8 threads still run after the fix (calls=8) — the
common, fully-parallel case is unaffected.

There is no existing `test-backend-ops` coverage for `MAP_CUSTOM*`/`CUSTOM` (`grep -rl
map_custom tests/` on upstream `master` returns nothing) — these two small programs are
the only regression coverage this contract currently has, before or after this PR.

Per policy, this is presented as a contract-violation fix with a counting
demonstration, not a working exploit against a hypothetical vulnerable caller.
