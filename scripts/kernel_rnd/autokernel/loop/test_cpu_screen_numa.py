"""AK-EVAL-QUARTERS validity fix: a quarter screen binds its pages to its own NPS4 node."""
import pytest

from . import cpu_screen, gates
from .test_glm_frozen_requests import _canonical_launch


@pytest.mark.parametrize("region,node", [(None, 0), ("q1", 1), ("q2", 2)])
def test_quarter_screen_binds_its_own_node_and_gets_its_own_floor_name(region, node):
    _template, full = _canonical_launch(18311)
    result = cpu_screen.prepare_launch(full, "quarter", range(96), region=region)
    prefix = list(result["launch"].topology_prefix)
    assert "--interleave=all" not in prefix
    assert f"--membind={node}" in prefix
    assert result["numa_policy"] == f"membind_node{node}"
    assert result["regions"] == (f"q{node}",)
    assert f"quarter-q{node}-membind_node{node}" in result["launch"].template.name
    assert result["launch"].template.cpu_list == ",".join(map(str, range(24 * node, 24 * node + 24)))


def test_half_screen_keeps_interleave_and_region_needs_a_quarter():
    _template, full = _canonical_launch(18311)
    half = cpu_screen.prepare_launch(full, "half", range(96))
    assert half["numa_policy"] == "interleave_all"
    assert "--interleave=all" in half["launch"].topology_prefix
    with pytest.raises(cpu_screen.ScreenRefused, match="explicit screen region"):
        cpu_screen.prepare_launch(full, "half", range(96), region="q1")
    with pytest.raises(cpu_screen.ScreenRefused, match="explicit screen region"):
        cpu_screen.prepare_launch(full, "quarter", range(96), region="q9")


def test_tool_default_binds_a_single_quarter_and_interleaves_a_wider_list():
    prefix, _threads, _args = gates._launch_argv(None, "24-47", 24, batch_ok=True)
    assert prefix[-1] == "--membind=1"
    prefix, _threads, _args = gates._launch_argv(None, "0-95", 96, batch_ok=True)
    assert prefix[-1] == "--interleave=all"
