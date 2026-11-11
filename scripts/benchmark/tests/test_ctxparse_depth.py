"""INF-45: ctxparse depth must come from `stop processing: n_tokens`, never the
incremental `prompt eval time` count. Offline."""
import importlib.util
from pathlib import Path

_P = Path(__file__).resolve().parents[3] / "data/numa_placement/20260730-P-BENCH-PLACEMENT-1/ctxparse.py"
_spec = importlib.util.spec_from_file_location("ctxparse", _P)
ctx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ctx)

STOP = "0.56.379.902 I slot      release: id  0 | task 0 | stop processing: n_tokens = {n}, truncated = 0\n"
PE = "prompt eval time =    1000.00 ms /  {n} tokens (    0.34 ms per token,   2926.00 tokens per second)\n"


def test_depth_from_n_tokens_not_prompt_eval(tmp_path):
    f = tmp_path / "cc_x_p32k.log"
    f.write_text(PE.format(n=2926) + STOP.format(n=27586))
    assert ctx.parse_depth(str(f)) == 27586
    assert ctx.parse(str(f))[0] == 2926  # the incremental count is NOT the depth


def test_depth_takes_max_resident(tmp_path):
    f = tmp_path / "a.log"
    f.write_text(STOP.format(n=618) + STOP.format(n=8821))
    assert ctx.parse_depth(str(f)) == 8821


def test_depth_none_when_absent_no_fallback(tmp_path):
    f = tmp_path / "b.log"
    f.write_text(PE.format(n=2926))
    assert ctx.parse_depth(str(f)) is None
