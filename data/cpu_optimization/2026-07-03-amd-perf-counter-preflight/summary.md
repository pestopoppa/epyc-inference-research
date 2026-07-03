# AMD Perf Counter Preflight

Generated: `2026-07-03T18:15:48.051414+00:00`
Status: **ok**

## Host

- Model: `AMD EPYC 9655 96-Core Processor`
- Vendor: `AuthenticAMD`
- Kernel: `6.14.0-37-generic`
- perf_event_paranoid: `1`

## Perf

- Binary: `/usr/bin/perf`
- Found: `True`
- perf list ok: `True`

## Canonical Events

| Event | Status |
|---|---|
| `fp_ops_retired_by_type.vector_mac` | present |
| `fp_ops_retired_by_type.vector_all` | present |
| `fp_ops_retired_by_type.scalar_all` | present |
| `ls_dmnd_fills_from_sys.dram_io_all` | present |
| `ls_hw_pf_dc_fills.dram_io_all` | present |
| `cycles` | present |
| `instructions` | present |
| `task-clock` | present |

## Probe

- Attempted: `True`
- OK: `True`

## Recommendation

Canonical AMD perf events are visible and the smoke probe passed.
