## ggml-cpu: `GGML_OP_SET_ROWS` work buffer is sized for 1 thread but written by `n_threads`

### Class of problem

Heap buffer overflow (out-of-bounds heap write) in the CPU backend's `SET_ROWS`
scratch buffer, reachable through the public `ggml_set_rows()` API with ordinary,
well-formed inputs (no aliasing, no adversarial shapes) — a sizing bug, not a
memory-safety issue in the caller. This is a plan/kernel mismatch: the planner
believes only one thread will touch the scratch buffer; the kernel actually lets every
worker thread touch it.

### Background: `n_tasks` is advisory on this backend

`ggml_graph_compute_thread()` sets `params.nth = n_threads` for **every** node
(`ggml/src/ggml-cpu/ggml-cpu.c`, the thread-pool dispatch loop). `ggml_get_n_tasks()`
is consumed only by `ggml_graph_plan()` — to size the shared `wdata` scratch buffer and
to compute the graph-wide `cplan.n_threads = MIN(max_tasks, n_threads)` cap. **It never
reaches a kernel.** A kernel that assumes "my planner said `n_tasks = 1`, so I only run
on one thread" is wrong: nothing enforces that at execution time, and the
`SET_ROWS` kernel does not make that assumption — it always row-splits over
`params->nth == n_threads`.

### The bug

Planner (`ggml_graph_plan()`, `ggml-cpu.c`):

```c
case GGML_OP_SET_ROWS:
    {
        if (node->src[0]->type == GGML_TYPE_F16 && node->type != GGML_TYPE_F16) {
            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;   // n_tasks == 1 for this op
        }
    } break;
```

with `n_tasks` fixed at 1 for `SET_ROWS` elsewhere in the same function (the
`GGML_OP_GET_ROWS`/`GGML_OP_SET_ROWS` case of the scheduling switch).

Kernel (`ggml_compute_forward_set_rows_impl`, `ggml/src/ggml-cpu/ops.cpp`):

```c
const int ith = params->ith;
const int nth = params->nth;                       // == n_threads, not n_tasks
const int64_t dr = (nr + nth - 1)/nth;
const int64_t ir0 = dr*ith;
const int64_t ir1 = std::min(ir0 + dr, nr);
...
float * wdata = (float *) params->wdata + (nc + CACHE_LINE_SIZE_F32) * ith;   // per-thread slice
```

reached on the **F16-source → non-F16-destination** branch (any destination type other
than F16 — F32, BF16, or a quantised type — written from an F16 source; the usual case
is a quantised or BF16 KV cache written from an F16 source tensor).

Every thread `ith` with a non-empty row range `[ir0, ir1)` dereferences its own slice of
`wdata`. The planner allocates room for exactly **one** such slice; the kernel can use
up to `min(nr, n_threads)` of them concurrently. With `nr >= n_threads` (e.g. prefill of
N tokens, `nr == N`), **every** thread's slice overruns the buffer, for any row width
`nc`. It is masked in the common case only because `ggml_graph_plan()`'s work buffer is
sized as a `MAX` over every node in the graph, so a co-resident `MUL_MAT` or
`FLASH_ATTN_EXT` node's much larger requirement usually reserves enough headroom by
accident — it is not a correctness guarantee.

### Fix

Size the buffer by `n_threads`, the same way every other `n_tasks`-sized `wdata` site
in `ggml_graph_plan()` already does (`CPY`/`DUP`, `ADD`/`ADD1`, `ACC`, `MUL_MAT`, ...) —
this is the one site that used the advisory `n_tasks` instead:

```c
case GGML_OP_SET_ROWS:
    {
        if (node->src[0]->type == GGML_TYPE_F16 && node->type != GGML_TYPE_F16) {
            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_threads;
        }
    } break;
```

The kernel is already correct and already parallel — sizing is the only thing wrong.
(An alternative of clamping `ith` in the kernel to the previously-assumed single-thread
range was considered and rejected: threads with `ith >= 1` own real, non-overlapping
row ranges, so clamping would silently drop rows and turn a heap overflow into wrong
output — strictly worse.)

### Reproduction (attached)

`repro_set_rows_overflow.c` builds a minimal graph — one `SET_ROWS` node (F16 source
`[64,8]`, F32 destination `[64,8]`, 8 unique row indices — no destination aliasing) plus
an unrelated `ADD1` node (needed only so `ggml_graph_plan()` doesn't cap the whole
graph's thread count to 1 for a graph made entirely of `n_tasks == 1` ops — a real
decode/prefill graph always has other node types alongside `SET_ROWS`, so this is
representative, not contrived) — plans it for 8 threads, allocates exactly
`cplan.work_size` bytes, and runs it.

Built and run with `-fsanitize=address,undefined` against a from-scratch `ggml`-only
build (`cmake -S . -B build -DCMAKE_C_FLAGS=... -DCMAKE_CXX_FLAGS=...
-DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF`, target `ggml`), on unmodified upstream
`master` (verified at `e71b805`, 2026-09-07):

```
planned work_size = 768 bytes (n_threads=8, nr=8, nc=64)
kernel needs        >= 2560 bytes
==...==ERROR: AddressSanitizer: heap-buffer-overflow ...
    #0 ... in ggml_fp16_to_fp32_row ggml/src/ggml.c:471
    #1 ... in ggml_compute_forward_set_rows_impl<unsigned short, long> ops.cpp:5277
    #2 ... in ggml_compute_forward_set_rows ops.cpp:5312
    ...
0x... is located 0 bytes after 768-byte region [...]
```

With the one-line fix applied, the same reproducer reports `work_size = 2560 bytes`
(exactly the proven minimum) and exits cleanly with no ASan report. A regression check
confirms `test-backend-ops -b CPU -o "SET_ROWS" -p "type_src=f16"` still passes 280/280
with the fix (this suite's largest `nr` in that filter is 11, `hardware_concurrency()`
on the machine used was 192, so `nr < n_threads` in every one of its shapes — it never
exercises `nr >= n_threads` and would not have caught this on its own).

Per policy, no destination-aliasing / write-primitive exploit is included — this
reproduces the out-of-bounds access with ASan, nothing more.
