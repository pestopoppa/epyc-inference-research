"""AK-HR-1: offline, read-only census and exporter over the closed AutoKernel history.

Stdlib only.  No GPU, no timing, no server, no process control.  Every SQLite store is
opened with `mode=ro&immutable=1` and the run asserts that no store file mtime moved.
"""
