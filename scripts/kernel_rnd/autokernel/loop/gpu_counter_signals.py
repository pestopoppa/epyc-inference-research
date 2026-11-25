"""RVP-CTR-1: counter-grounded planner signals for the GPU AutoKernel lane.

A fixed, declared counter subset is captured with rocprofv3 6.2 ``-i`` pmc input files (one profiler
pass per group, because the SQ/TCC/TCP blocks cannot all be read in one pass) for the
top hotspot kernels of a GPU screen. This module

* declares the subset (``PASSES``) and the derived metrics (``METRICS``) with their
  formulas, taken from rocprofiler-sdk 6.2 ``derived_counters.xml`` for gfx90a unless
  the formula notes otherwise;
* builds the rocprofv3 6.2 input file and argv for one pass (``pass_input``,
  ``pass_argv``) -- this module never runs it;
* parses the ``*counter_collection.csv`` of each pass and folds it into an observation
  document (``build_observation``), schema ``epyc.autokernel.gpu_counter_observation.v1``;
* renders the planner block (``render``) and enforces the citation rule
  (``check_citations``): every GPU proposal must cite at least one PRESENT metric as
  ``ctr:<metric>``, and may not cite an ABSENT one. A metric whose inputs were not all
  captured is ``None`` -- never zero.

The A/B ("2 planner rounds with and without counters on the same target and budget")
is a run.py flag: ``--gpu-counters PATH`` puts the observation in the planner context;
omitting it is the control arm.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "epyc.autokernel.gpu_counter_observation.v1"
CITE_RE = re.compile(r"\bctr:([a-z_][a-z0-9_]*)")

# Per-dispatch constants: rocprofv3 reports them per dispatch, so they are taken once
# (max), never summed across dispatches.
CONSTANTS = ("CU_NUM", "SE_NUM", "SIMD_NUM")

# One rocprofv3 pass per group. GRBM_GUI_ACTIVE is in every pass so each derived
# metric divides by the active cycles of the pass that produced its numerator.
PASSES: dict[str, tuple[str, ...]] = {
    "occupancy": ("GRBM_GUI_ACTIVE", "SQ_WAVES", "SQ_BUSY_CYCLES", "SQ_LEVEL_WAVES",
                  "SQ_ACCUM_PREV_HIRES", "CU_NUM", "SE_NUM", "SIMD_NUM"),
    "alu": ("GRBM_GUI_ACTIVE", "SQ_ACTIVE_INST_VALU", "SQ_VALU_MFMA_BUSY_CYCLES",
            "SQ_INSTS_MFMA", "SQ_INSTS_VALU", "CU_NUM", "SIMD_NUM"),
    "lds": ("GRBM_GUI_ACTIVE", "SQ_LDS_BANK_CONFLICT", "SQ_ACTIVE_INST_LDS",
            "SQ_INSTS_LDS", "CU_NUM"),
    "memstall": ("GRBM_GUI_ACTIVE", "TCP_TCP_TA_DATA_STALL_CYCLES",
                 "TA_ADDR_STALLED_BY_TC_CYCLES", "CU_NUM", "SE_NUM"),
    "bandwidth": ("GRBM_GUI_ACTIVE", "TCC_EA_RDREQ", "TCC_EA_RDREQ_32B", "TCC_EA_WRREQ",
                  "TCC_EA_WRREQ_64B"),
}


def _ratio(num: float, den: float) -> float | None:
    return None if not den else num / den


# name -> (pass, inputs, function(values, duration_ns), formula text)
METRICS: dict[str, tuple[str, tuple[str, ...], Any, str]] = {
    "occupancy_waves_per_cu": (
        "occupancy", ("SQ_ACCUM_PREV_HIRES", "GRBM_GUI_ACTIVE", "CU_NUM"),
        lambda v, d: _ratio(v["SQ_ACCUM_PREV_HIRES"], v["GRBM_GUI_ACTIVE"] * v["CU_NUM"]),
        "SQ_ACCUM_PREV_HIRES/GRBM_GUI_ACTIVE/CU_NUM (MeanOccupancyPerCU)"),
    "valu_busy_pct": (
        "alu", ("SQ_ACTIVE_INST_VALU", "GRBM_GUI_ACTIVE", "SIMD_NUM"),
        lambda v, d: _ratio(100 * v["SQ_ACTIVE_INST_VALU"] * 4,
                            v["SIMD_NUM"] * v["GRBM_GUI_ACTIVE"]),
        "100*SQ_ACTIVE_INST_VALU*4/SIMD_NUM/GRBM_GUI_ACTIVE (VALUBusy)"),
    "mfma_busy_pct": (
        "alu", ("SQ_VALU_MFMA_BUSY_CYCLES", "GRBM_GUI_ACTIVE", "CU_NUM"),
        lambda v, d: _ratio(100 * v["SQ_VALU_MFMA_BUSY_CYCLES"],
                            v["GRBM_GUI_ACTIVE"] * v["CU_NUM"] * 4),
        "100*SQ_VALU_MFMA_BUSY_CYCLES/(GRBM_GUI_ACTIVE*CU_NUM*4) (omniperf MFMA util; "
        "not in derived_counters.xml)"),
    "mfma_insts": (
        "alu", ("SQ_INSTS_MFMA",), lambda v, d: v["SQ_INSTS_MFMA"],
        "SQ_INSTS_MFMA (executed MFMA wave-instructions; C5-ISA-1 execution witness)"),
    "lds_bank_conflict_pct": (
        "lds", ("SQ_LDS_BANK_CONFLICT", "GRBM_GUI_ACTIVE", "CU_NUM"),
        lambda v, d: _ratio(100 * v["SQ_LDS_BANK_CONFLICT"],
                            v["GRBM_GUI_ACTIVE"] * v["CU_NUM"]),
        "100*SQ_LDS_BANK_CONFLICT/GRBM_GUI_ACTIVE/CU_NUM (LDSBankConflict)"),
    "lds_insts": (
        "lds", ("SQ_INSTS_LDS",), lambda v, d: v["SQ_INSTS_LDS"],
        "SQ_INSTS_LDS (executed LDS wave-instructions; C5-ISA-1 execution witness)"),
    "mem_unit_stalled_pct": (
        "memstall", ("TCP_TCP_TA_DATA_STALL_CYCLES", "GRBM_GUI_ACTIVE", "CU_NUM"),
        lambda v, d: _ratio(100 * v["TCP_TCP_TA_DATA_STALL_CYCLES"],
                            v["GRBM_GUI_ACTIVE"] * v["CU_NUM"]),
        "100*sum(TCP_TCP_TA_DATA_STALL_CYCLES)/GRBM_GUI_ACTIVE/CU_NUM: MEAN over TCPs, "
        "whereas rocprof MemUnitStalled uses the max instance over SE_NUM"),
    "achieved_bw_gbps": (
        "bandwidth", ("TCC_EA_RDREQ", "TCC_EA_RDREQ_32B", "TCC_EA_WRREQ", "TCC_EA_WRREQ_64B"),
        lambda v, d: _ratio(
            v["TCC_EA_RDREQ_32B"] * 32 + (v["TCC_EA_RDREQ"] - v["TCC_EA_RDREQ_32B"]) * 64
            + (v["TCC_EA_WRREQ"] - v["TCC_EA_WRREQ_64B"]) * 32 + v["TCC_EA_WRREQ_64B"] * 64,
            d),
        "(FETCH_SIZE+WRITE_SIZE) bytes of the counter run / device ns of the same kernel in the "
        "kernel-trace run of the same workload (6.2 counter CSVs carry no timestamps)"),
}

ROCPROFV3 = "/mnt/raid0/llm/tools/rocprofiler-sdk-6.2.0-66/opt/rocm-6.2.0/bin/rocprofv3"

# AK-BNSIG: the bottleneck-signature counter set (bottleneck_signature.py), declared here so the loop has ONE
# counter-declaration surface. Kept separate from PASSES/METRICS on purpose: folding it in would change
# counter_set_sha256() and make load_observation refuse every stored RVP-CTR-1 observation. Pass C3
# (MemUnitBusy/MemUnitStalled) is dropped: both read 0 per dispatch on this host, so mem_busy/mem_stalled come
# from the raw TA counters of C7. NOTE the name: `mem_stalled_ta` (TA_ADDR_STALLED_BY_TC_CYCLES_sum) is a
# different counter from METRICS' `mem_unit_stalled_pct` (TCP_TCP_TA_DATA_STALL_CYCLES).
# The pass table lives in the pure, vendorable bottleneck_signature.py (row job dirs copy it); it is re-exported here.
from .bottleneck_signature import PASSES_SIG as SIGNATURE_PASSES  # noqa: E402
SIGNATURE_AXES: dict[str, str] = {
    "fetch_over_bytes": "FETCH_SIZE*1024/bytes_moved",
    "ea_req_per_kb": "TCC_EA_RDREQ/(bytes_moved/1024)",
    "mem_stalled_ta": "TA_ADDR_STALLED_BY_TC_CYCLES_sum/(CU_NUM*C7.GRBM_GUI_ACTIVE) (MemUnitStalled reads 0 here)",
    "mem_busy_ta": "TA_BUSY_avr/C7.GRBM_GUI_ACTIVE (MemUnitBusy reads 0 here)",
    "wait_any": "SQ_WAIT_ANY/SQ_WAVE_CYCLES",
    "wait_inst": "SQ_WAIT_INST_ANY/SQ_WAVE_CYCLES",
    "issue_busy": "4*SQ_ACTIVE_INST_ANY/(SIMD_NUM*C5.GRBM_GUI_ACTIVE)",
    "valu_per_kb": "SQ_INSTS_VALU/(bytes_moved/1024)",
    "lds_per_kb": "SQ_INSTS_LDS/(bytes_moved/1024)",
}


def signature_counter_set_sha256() -> str:
    body = {"passes": SIGNATURE_PASSES, "axes": SIGNATURE_AXES}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def signature_pass_input(name: str) -> str:
    """rocprofv3 6.2 ``-i`` input text for one bottleneck-signature pass."""
    if name not in SIGNATURE_PASSES:
        raise KeyError(f"undeclared signature pass {name!r}")
    return "pmc: " + " ".join(SIGNATURE_PASSES[name]) + "\n"


def counter_set_sha256() -> str:
    body = {"passes": PASSES, "metrics": {k: [m[0], list(m[1]), m[3]] for k, m in METRICS.items()}}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def pass_input(name: str) -> str:
    """rocprofv3 6.2 input-file text for one declared pass (6.2 has no --pmc flag;
    counters are read from ``-i FILE`` with a ``pmc:`` line)."""
    if name not in PASSES:
        raise KeyError(f"undeclared counter pass {name!r}")
    return "pmc: " + " ".join(PASSES[name]) + "\n"


def pass_argv(name: str, out_dir: Path, command: Sequence[str], *,
              kernel_names: Sequence[str] = (), rocprof: str = ROCPROFV3) -> list[str]:
    """rocprofv3 6.2 argv for one declared pass; the caller writes ``pass_input(name)``
    to ``out_dir/<name>.pmc.txt`` first. Mangled names (-M) so rows join the ISA audit.
    Never executed here."""
    pass_input(name)  # rejects an undeclared pass
    pmc = out_dir / f"{name}.pmc.txt"
    argv = [rocprof, "-M", "-i", str(pmc), "--output-format", "csv",
            "-d", str(out_dir / name), "-o", name]
    if kernel_names:
        argv += ["--kernel-names", *kernel_names]
    return [*argv, "--", *command]


def kernel_key(name: str) -> str:
    """Join key for a kernel name. rocprofv3 6.2 -M writes the code-object symbol in
    counter_collection.csv WITH the ``.kd`` descriptor suffix, while the kernel trace
    (and the ISA audit) name the function without it. Without this the first r1 row
    (ak-gpu-signals-ctr-20261116) joined 0 of 10 hotspots: every metric came back None."""
    name = name.strip()
    return name[:-3] if name.endswith(".kd") else name


def agent_constants(text: str, *, gfx: str = "gfx90a") -> dict[str, float]:
    """CU_NUM / SIMD_NUM / SE_NUM from rocprofv3 ``*_agent_info.csv`` (the GPU agent row).

    On gfx90a with rocprofiler-sdk 6.2 the constant COUNTERS are wrong: SIMD_NUM reads
    0 (every VALUBusy was a zero-denominator None in row r1) and CU_NUM reads 112 (the
    8x14 physical slots, not the 104 enabled CUs). The agent table carries the true
    Cu_Count 104 / Simd_Count 416 / Array_Count 8, so it overrides the counters."""
    for row in csv.DictReader(io.StringIO(text)):
        if row.get("Name") == gfx:
            return {"CU_NUM": float(row["Cu_Count"]), "SIMD_NUM": float(row["Simd_Count"]),
                    "SE_NUM": float(row["Array_Count"])}
    raise ValueError(f"no {gfx} agent in agent_info")


def trace_durations(text: str) -> dict[str, float]:
    """rocprofv3 kernel_trace.csv -> {kernel_key: summed device ns}."""
    out: dict[str, float] = {}
    for row in csv.DictReader(io.StringIO(text)):
        try:
            ns = int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
        except (KeyError, TypeError, ValueError):
            continue
        key = kernel_key(row.get("Kernel_Name") or "")
        out[key] = out.get(key, 0.0) + max(ns, 0)
    return out


def parse_counter_csv(text: str) -> dict[str, dict[str, Any]]:
    """rocprofv3 counter_collection.csv -> {kernel: {counter: value, _dispatches, _duration_ns}}.

    Counters are summed over dispatches (and over the dimension rows rocprofv3 may
    emit); CONSTANTS are taken once. Duration is summed per distinct dispatch."""
    out: dict[str, dict[str, Any]] = {}
    seen: dict[str, dict[str, tuple[int, int]]] = {}
    reader = csv.DictReader(io.StringIO(text))
    need = {"Kernel_Name", "Counter_Name", "Counter_Value"}
    if not reader.fieldnames or need - set(reader.fieldnames):
        raise ValueError(f"counter csv lacks columns {sorted(need - set(reader.fieldnames or []))}")
    for row in reader:
        kernel = kernel_key(row["Kernel_Name"])
        name = row["Counter_Name"]
        value = float(row["Counter_Value"])
        entry = out.setdefault(kernel, {})
        if name in CONSTANTS:
            entry[name] = max(entry.get(name, 0.0), value)
        else:
            entry[name] = entry.get(name, 0.0) + value
        # rocprofv3 6.2 counter CSVs carry NO timestamps; dispatches are counted by id
        # and the duration comes from the kernel trace (build_observation durations=).
        dispatch = row.get("Dispatch_Id") or row.get("Correlation_Id") or ""
        start, end = row.get("Start_Timestamp"), row.get("End_Timestamp")
        span = (int(start), int(end)) if start and end else None
        if dispatch:
            seen.setdefault(kernel, {})[dispatch] = span
    for kernel, entry in out.items():
        dispatches = seen.get(kernel, {})
        entry["_dispatches"] = len(dispatches)
        spans = [v for v in dispatches.values() if v]
        entry["_duration_ns"] = float(sum(e - s for s, e in spans)) or None
    return out


def build_observation(pass_csvs: Mapping[str, str], *, hotspots: Sequence[str] = (),
                      top: int = 5, source: Mapping[str, str] | None = None,
                      durations: Mapping[str, float] | None = None,
                      agent: Mapping[str, float] | None = None) -> dict:
    """Fold the per-pass CSV texts into the planner observation.

    ``hotspots``: kernel names in device-time order (from the kernel-trace hotspot
    view); only those (first ``top``) are kept. Without it, kernels are ranked by the
    summed duration of the ``alu`` pass (or any pass carrying timestamps)."""
    unknown = set(pass_csvs) - set(PASSES)
    if unknown:
        raise KeyError(f"undeclared passes {sorted(unknown)}")
    parsed = {name: parse_counter_csv(text) for name, text in pass_csvs.items()}
    kernels: list[str]
    if hotspots:
        kernels = [kernel_key(k) for k in list(hotspots)[:top]]
    else:
        durations: dict[str, float] = {}
        for table in parsed.values():
            for kernel, entry in table.items():
                durations[kernel] = max(durations.get(kernel, 0.0), entry["_duration_ns"] or 0.0)
        kernels = sorted(durations, key=lambda k: -durations[k])[:top]
    durations = {kernel_key(k): v for k, v in (durations or {}).items()}
    for table in parsed.values():   # a counter run has no timestamps: use the trace's device time
        for kernel, entry in table.items():
            if not entry["_duration_ns"] and durations.get(kernel):
                entry["_duration_ns"] = float(durations[kernel])
            for name, value in (agent or {}).items():   # agent table beats 6.2 constant counters
                if name in entry:
                    entry["_counter_" + name] = entry[name]
                entry[name] = float(value)
    rows = []
    for kernel in kernels:
        metrics: dict[str, float | None] = {}
        missing: dict[str, list[str]] = {}
        duration = None
        for name, (pass_name, inputs, fn, _formula) in METRICS.items():
            values = parsed.get(pass_name, {}).get(kernel)
            absent = [c for c in inputs if values is None or c not in values]
            if pass_name == "bandwidth" and values is not None and not values["_duration_ns"]:
                absent.append("_duration_ns")
            if absent:
                metrics[name] = None
                missing[name] = absent
                continue
            duration = values["_duration_ns"] or duration
            result = fn(values, values["_duration_ns"])
            metrics[name] = None if result is None else round(float(result), 6)
            if result is None:
                missing[name] = ["zero denominator"]
        rows.append({"kernel": kernel, "metrics": metrics, "missing": missing,
                     "duration_ns": duration})
    present = any(v is not None for row in rows for v in row["metrics"].values())
    joined = sum(1 for k in kernels if any(k in t for t in parsed.values()))
    return {"schema": SCHEMA, "status": "present" if present else "absent",
            "counter_set_sha256": counter_set_sha256(),
            "passes_captured": sorted(pass_csvs), "kernels": rows,
            "joined_kernels": joined, "agent_constants": dict(agent or {}),
            **({} if present else {"reason": f"0 metrics resolved; {joined}/{len(kernels)} "
                                             "hotspot kernels found in the counter CSVs"}),
            "source": dict(source or {})}


def present_metrics(observation: Mapping[str, Any]) -> set[str]:
    return {name for row in observation.get("kernels", ())
            for name, value in row.get("metrics", {}).items() if value is not None}


def absent_metrics(observation: Mapping[str, Any]) -> set[str]:
    return set(METRICS) - present_metrics(observation)


def render(observation: Mapping[str, Any], *, limit: int = 5) -> list[str]:
    lines = ["", "## GPU hardware counters (RVP-CTR-1, declared subset)"]
    if observation.get("status") != "present":
        lines.append("Counters ABSENT for this target: no counter may be cited and none is "
                     "zero. Reason: " + str(observation.get("reason") or "not captured"))
        return lines
    names = list(METRICS)
    lines.append("| kernel | " + " | ".join(f"ctr:{n}" for n in names) + " |")
    lines.append("|---|" + "---|" * len(names))
    for row in list(observation.get("kernels", ()))[:limit]:
        cells = ["ABSENT" if row["metrics"].get(n) is None else f"{row['metrics'][n]:.4g}"
                 for n in names]
        lines.append(f"| `{row['kernel'][:80]}` | " + " | ".join(cells) + " |")
    lines += ["", "Formulas: " + "; ".join(f"{n} = {METRICS[n][3]}" for n in names), "",
              "RULE (enforced on your reply): the hypothesis `statement` or `falsifier` must "
              "cite at least one counter shown above as `ctr:<name>` with a value, and the "
              "mechanism must follow from it. A cell reading ABSENT was not measured: it is "
              "NOT zero, and citing it refuses the proposal."]
    return lines


class CitationRefused(ValueError):
    pass


def check_citations(texts: Iterable[str], observation: Mapping[str, Any] | None) -> list[str]:
    """Return the present metrics cited; raise CitationRefused on a rule breach.

    No observation (control arm) -> no rule, returns []."""
    if not observation:
        return []
    cited = {m for text in texts for m in CITE_RE.findall(text or "")}
    if observation.get("status") != "present":
        if cited:
            raise CitationRefused(f"counters are absent for this target; cited {sorted(cited)}")
        return []
    unknown = cited - set(METRICS)
    if unknown:
        raise CitationRefused(f"cited undeclared counters {sorted(unknown)}")
    absent = cited & absent_metrics(observation)
    if absent:
        raise CitationRefused(f"cited ABSENT counters {sorted(absent)} (absent is not zero)")
    good = sorted(cited & present_metrics(observation))
    if not good:
        raise CitationRefused("no present counter cited as ctr:<name>")
    return good


def load_observation(path: Path) -> dict:
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r} != {SCHEMA}")
    if doc.get("counter_set_sha256") != counter_set_sha256():
        raise ValueError(f"{path}: counter set differs from the declared subset")
    return doc


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("argv", help="print the rocprofv3 argv of one pass (JSON)")
    a.add_argument("pass_name", choices=sorted(PASSES))
    a.add_argument("--out-dir", type=Path, required=True)
    a.add_argument("--kernel-name", action="append", default=[])
    a.add_argument("command", nargs="+")
    b = sub.add_parser("build", help="fold pass CSVs into an observation")
    b.add_argument("--pass", dest="passes", action="append", default=[], metavar="NAME=CSV")
    b.add_argument("--hotspots", type=Path, help="JSON list of kernel names, device-time order")
    b.add_argument("--top", type=int, default=5)
    b.add_argument("--trace", type=Path, help="kernel_trace.csv of the same workload (durations)")
    b.add_argument("--agent-info", type=Path, help="rocprofv3 *_agent_info.csv (CU/SIMD/SE counts)")
    b.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "argv":
        print(json.dumps({"input": pass_input(args.pass_name),
                          "argv": pass_argv(args.pass_name, args.out_dir, args.command,
                                            kernel_names=args.kernel_name)}))
        return 0
    texts, source = {}, {}
    for item in args.passes:
        name, _, path = item.partition("=")
        data = Path(path).read_bytes()
        texts[name] = data.decode()
        source[name] = hashlib.sha256(data).hexdigest()
    hot = json.loads(args.hotspots.read_text()) if args.hotspots else ()
    dur = trace_durations(args.trace.read_text()) if args.trace else None
    agent = agent_constants(args.agent_info.read_text()) if args.agent_info else None
    doc = build_observation(texts, hotspots=hot, top=args.top, source=source, durations=dur,
                            agent=agent)
    args.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"status": doc["status"], "kernels": len(doc["kernels"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
