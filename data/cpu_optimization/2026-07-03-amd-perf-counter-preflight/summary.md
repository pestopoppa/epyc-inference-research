# AMD Perf Counter Preflight

Generated: `2026-07-03T17:35:37.942398+00:00`
Status: **blocked**

## Host

- Model: `AMD EPYC 9655 96-Core Processor`
- Vendor: `AuthenticAMD`
- Kernel: `6.14.0-37-generic`
- perf_event_paranoid: `1`

## Perf

- Binary: `None`
- Found: `False`
- perf list ok: `None`

## Canonical Events

| Event | Status |
|---|---|
| `fp_ops_retired_by_type.vector_mac` | missing |
| `fp_ops_retired_by_type.vector_all` | missing |
| `fp_ops_retired_by_type.scalar_all` | missing |
| `ls_dmnd_fills_from_sys.dram_io_all` | missing |
| `ls_hw_pf_dc_fills.dram_io_all` | missing |
| `cycles` | missing |
| `instructions` | missing |
| `task-clock` | missing |

## Probe

- Attempted: `False`
- OK: `None`

## Recommendation

Install or expose linux-tools/perf for the running kernel before using bench_canonical.sh --perf or accepting roofline evidence.
