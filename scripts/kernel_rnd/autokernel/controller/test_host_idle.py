"""AK-BENCH-IDLE-GUARD: the observe-only reading, offline."""
from autokernel.controller import host_idle


def _proc(tmp_path, stat):
    (tmp_path / "stat").write_text(stat)
    (tmp_path / "loadavg").write_text("46.5 1 1 1/1 1\n")
    return tmp_path


def test_busy_is_scoped_to_the_granted_cpus_and_observe_only(tmp_path):
    root = _proc(tmp_path, "cpu  1 1 1 1 1 1 1 1\ncpu2 100 0 0 0 0 0 0 0\ncpu3 0 0 0 100 0 0 0 0\n")

    def tick(_):  # cpu2 goes fully busy, cpu3 stays idle
        (root / "stat").write_text(
            "cpu  2 2 2 2 2 2 2 2\ncpu2 200 0 0 0 0 0 0 0\ncpu3 0 0 0 200 0 0 0 0\n")
    row = host_idle.observe("pre", "2-3", proc_root=root, sleep=tick)
    assert row["busy_pct"] == 50.0 and row["loadavg1"] == 46.5
    assert row["busy_threshold_pct"] is None and row["error"] is None


def test_unreadable_host_is_recorded_not_raised(tmp_path):
    row = host_idle.observe("post", "0", proc_root=tmp_path / "absent", sleep=lambda _: None)
    assert row["busy_pct"] is None and "FileNotFoundError" in row["error"]
