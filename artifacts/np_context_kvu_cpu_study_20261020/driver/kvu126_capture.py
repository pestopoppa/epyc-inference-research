"""Opt-in unique-cell gauge sampling for KVU-12.6; no inferred occupancy."""
import hashlib
import json
import math
import re
import threading
import time
import urllib.request
from pathlib import Path


class CaptureError(Exception):
    pass


UNIQUE_CELLS_METRIC = "llamacpp:kv_cache_used_cells"
UNIQUE_CELLS_HELP = "Number of occupied unique KV cache cells."


def read_gauge(port, name, timeout=2):
    """Read one explicitly named unlabelled Prometheus gauge, retaining native text."""
    if name != UNIQUE_CELLS_METRIC:
        raise CaptureError("unsupported metric; a source-verified unique KV-cell gauge is required")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=timeout) as response:
            text = response.read().decode("utf-8")
    except Exception as exc:
        raise CaptureError(f"metrics unavailable: {exc}") from exc
    typ = re.findall(rf"^# TYPE {re.escape(name)} (\w+)\s*$", text, re.M)
    help_text = re.findall(rf"^# HELP {re.escape(name)} (.+)\s*$", text, re.M)
    values = re.findall(rf"^{re.escape(name)}\s+(\S+)\s*$", text, re.M)
    if typ != ["gauge"] or help_text != [UNIQUE_CELLS_HELP] or len(values) != 1:
        raise CaptureError(f"native unique-cell gauge {name} unavailable or ambiguous")
    try:
        value = float(values[0])
        if not math.isfinite(value) or not value.is_integer() or value < 0:
            raise ValueError(values[0])
    except ValueError as exc:
        raise CaptureError(f"invalid unique-cell value: {values[0]}") from exc
    return int(value), text


class Sampler:
    def __init__(self, port, metric, interval_s=0.25, identity_check=None):
        self.port, self.metric, self.interval_s = port, metric, interval_s
        self.identity_check = identity_check
        self.samples, self.errors = [], []
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.done.is_set():
            start = time.monotonic()
            try:
                before = self.identity_check() if self.identity_check else None
                value, raw = read_gauge(self.port, self.metric)
                after = self.identity_check() if self.identity_check else None
                if before != after:
                    raise CaptureError("process identity changed during metric request")
                self.samples.append({"request_start_monotonic_s": start, "monotonic_s": time.monotonic(),
                                     "unique_cells": value, "metrics_text": raw, "pid_birth_identity": after})
            except Exception as exc:
                self.errors.append(str(exc))
            self.done.wait(self.interval_s)

    def start(self):
        self.thread.start()

    def finish(self, records, pool):
        self.done.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive() or self.errors:
            raise CaptureError("metric sampling failed or did not stop: " + str(self.errors))
        if not records or any(r.get("t_first_monotonic") is None or r.get("t_end_monotonic") is None for r in records):
            raise CaptureError("missing per-stream decode window")
        start = max(r["t_first_monotonic"] for r in records)
        end = min(r["t_end_monotonic"] for r in records)
        samples = [s for s in self.samples if start <= s["request_start_monotonic_s"] <= s["monotonic_s"] <= end]
        if end <= start or len(samples) < 2 or any(s["unique_cells"] > pool for s in samples):
            raise CaptureError("insufficient in-window unique-cell samples or value exceeds pool")
        return {"decode_start_monotonic_s": start, "decode_end_monotonic_s": end,
                "metric_name": self.metric, "samples": samples}


def write_bound_record(directory, wave, row_sha256, capture):
    """Bind original timed samples to a completed producer wave without rewriting either."""
    directory = Path(directory)
    raw = dict(capture, row_sha256=row_sha256)
    raw_path = directory / f"wave-{wave}-native.json"
    raw_bytes = json.dumps(raw, sort_keys=True, allow_nan=False).encode()
    with raw_path.open("xb") as out:
        out.write(raw_bytes)
    record = dict(raw, schema="epyc.kvu126.native_unique_cells.v1", native_source=raw_path.name,
                  native_source_sha256=hashlib.sha256(raw_bytes).hexdigest())
    receipt_path = directory / f"wave-{wave}-occupancy.json"
    with receipt_path.open("x") as out:
        out.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return receipt_path
