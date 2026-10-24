# akx_lb3_attribution — AKX-ALL-LB3.2 attribution instrument

The 2026-10-22 LB3 premeasure ended `SPLIT_UNRESOLVED` because its F could not decide anything. It had three defects:

1. The perf window stayed open through llama-bench's cached-state restores. A restore runs **before** its `depth run (cached)` line, so it always fell inside the window.
2. The non-greedy line regex cut templated symbols at their first space.
3. A CPU-sample share cannot charge barrier spin to the op the threads wait on.

This package replaces that instrument. Code only: nothing here starts a model. The run lives in the launcher directory named in the handoff (`/mnt/raid0/llm/tmp/f8-launchers-20261024/akx-lb3-2/`), with its own predeclaration file.

| Module | Item | What it does |
|---|---|---|
| `perfscript.py` | (b), (a) | Parses `perf script -F pid,tid,time,ip,sym,dso` lines. The DSO is split off from the right with `rpartition(" (")`. Classification checks the DSO first (libgomp, libc, libllama state/host, libstdc++, ggml), then the symbol. |
| `phases.py` | (a) | Tracks llama-bench progress lines and holds the perf `--control` policy. Events stay enabled from a test's `starting` line and are disabled only on a non-cached `depth run`. Timed intervals are `[t_line(prompt run k), + samples_ns[k-1] - 2 ms]`, taken from llama-bench's own clock. A main-thread restore run that straddles an interval end is trimmed. |
| `perfagg.py` | (a) | G arm: CPU-sample class shares inside the intervals only. A repetition with a `state_restore` sample inside its interval is dropped. So is the first repetition after a prefill, because perf was not yet enabled when it started. Reported, not gating: this is a lower bound. |
| `evalprof.py` | (c) | M arm: per-node critical-path wall (thread 0, through the graph barrier). It comes from the measurement build's per-evaluation record (`GGML_CPU_PROF_EVAL_FILE`) and is gated by stderr order. It also reads the host phases `[phase_prof]` (apply_ubatch, set_inputs, graph_compute) and `[input_prof]` (set_input per input class: the KQ mask, and the PLE input that calls `get_prev_tokens`). |
| `rule.py` | rule | The S/F rule with F = FA op wall share. It also applies the INVALID gates (measurement-build overhead, coverage, repetitions) and picks the seed class. |
| `lb32_run.py`, `analyze.py`, `verdict.py` | run | Runs one arm, post-processes one arm, and writes the verdict. |

## Measurement build

The measurement build is llama.cpp fork branch `experimental/akx-lb3-2a-optiming-20261024`. It is cut from champion `dcc853b55` and must be built with `-DGGML_CPU_PROF=ON`.

All its edits are same-line appends inside `#ifdef GGML_CPU_PROF`, or an end-of-file block. A build without the define therefore preprocesses byte-identically to the champion: `gcc -E -P` with the store-image flags gives the same output.

The champion store image stays the timing anchor (G arm). The M arm only attributes, and it is valid only if `|tM/tG - 1| <= 5%` at every gating cell.

## Tests

```bash
cd <research worktree> && /usr/bin/python3 -m pytest -q scripts/benchmark/akx_lb3_attribution
```

The fixtures are 62 perf-script lines recorded by the 2026-10-22 run, plus its llama-bench progress lines (symbols and timestamps only). The tests cover:

- a templated tinyBLAS symbol, plus heap and dequantizer templates;
- a libc memcpy inside a restore (excluded) next to a worker-thread libc copy inside the step (kept);
- a restore run straddling a late interval (trimmed) and a long straddle (repetition dropped);
- per-evaluation gating and node classes;
- every rule outcome.
