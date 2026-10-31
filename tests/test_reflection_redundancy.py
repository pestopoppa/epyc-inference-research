import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/analysis/reflection_redundancy.py"
spec = importlib.util.spec_from_file_location("reflection_redundancy", SCRIPT)
rrr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rrr)


def test_nonadjacent_prior_match_counts_once_and_excludes_first():
    result = rrr.reflection_redundancy(["take the mug", "look inside cabinet", "take the mug"])
    assert result["rrr"] == 0.5
    assert result["redundant_count"] == 1 and result["eligible_count"] == 2
    assert result["comparisons"][1]["matching_earlier_indices"] == [0]


def test_many_matches_do_not_count_the_same_reflection_twice():
    result = rrr.reflection_redundancy(["repeat", "repeat", "repeat", "repeat"])
    assert result["rrr"] == 1 and result["redundant_count"] == 3
    assert result["comparisons"][-1]["matching_earlier_indices"] == [0, 1, 2]


def test_inclusive_threshold_and_just_below_boundary():
    base = "".join(chr(0x100 + i) for i in range(100))
    for common, expected in [(85, 1), (84, 0)]:
        changed = base[:common] + "".join(chr(0x500 + i) for i in range(100 - common))
        result = rrr.reflection_redundancy([base, changed])
        assert result["rrr"] == expected


@pytest.mark.parametrize("reflections", [[], ["one reflection"]])
def test_insufficient_trace_is_unknown(reflections):
    result = rrr.reflection_redundancy(reflections)
    assert result["rrr"] is None and result["eligible_count"] == 0
    assert result["unknown_reason"] == "fewer_than_two_recorded_reflections"


@pytest.mark.parametrize("reflections", [None, "text", {}, [None], ["text", 1]])
def test_missing_or_wrong_input_refuses(reflections):
    with pytest.raises(ValueError):
        rrr.reflection_redundancy(reflections)


def test_recorded_empty_text_is_preserved_without_filtering():
    result = rrr.reflection_redundancy(["", ""])
    assert result["reflection_count"] == 2 and result["rrr"] == 1


def test_input_is_not_mutated_and_text_is_not_normalized():
    values = ["a", "a ", "\t"]
    before = list(values)
    result = rrr.reflection_redundancy(values)
    assert values == before and result["rrr"] == 0


def test_cli_pins_actual_input_bytes_and_preserves_existing_output(tmp_path):
    source, output = tmp_path / "reflections.json", tmp_path / "report.json"
    source.write_text('["same", "same"]\n')
    raw = source.read_bytes()
    command = [
        sys.executable,
        "-I",
        str(SCRIPT),
        "--reflections-json",
        str(source),
        "--output",
        str(output),
    ]
    assert subprocess.run(command, capture_output=True).returncode == 0
    result = json.loads(output.read_bytes())
    assert result["rrr"] == 1 and result["input"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["producer"] == {
        "path": str(SCRIPT.resolve()),
        "sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
    }
    original = output.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert output.read_bytes() == original and source.read_bytes() == raw


def test_cli_producer_pin_tracks_actual_executed_copy(tmp_path):
    script = tmp_path / "copied_reducer.py"
    script.write_bytes(SCRIPT.read_bytes() + b"\n# distinct private source fixture\n")
    source, output = tmp_path / "reflections.json", tmp_path / "report.json"
    source.write_text('["same", "same"]')
    command = [
        sys.executable,
        "-I",
        str(script),
        "--reflections-json",
        str(source),
        "--output",
        str(output),
    ]
    assert subprocess.run(command, capture_output=True).returncode == 0
    producer = json.loads(output.read_bytes())["producer"]
    assert producer["path"] == str(script.resolve())
    assert producer["sha256"] == hashlib.sha256(script.read_bytes()).hexdigest()
    assert producer["sha256"] != hashlib.sha256(SCRIPT.read_bytes()).hexdigest()


def test_cli_malformed_or_null_input_never_emits_a_report(tmp_path):
    source, output = tmp_path / "reflections.json", tmp_path / "report.json"
    for data in ['["first", null]', '{"not":"a sequence"}', "broken JSON"]:
        source.write_text(data)
        command = [
            sys.executable,
            "-I",
            str(SCRIPT),
            "--reflections-json",
            str(source),
            "--output",
            str(output),
        ]
        assert subprocess.run(command, capture_output=True).returncode == 2
        assert not output.exists()
