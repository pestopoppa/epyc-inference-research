"""Pinned-source attestation; run only in the owner's approved compute window."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

HERE = Path(__file__).parent
FIXTURES = HERE / "fixtures" / "beam_judge"


def fixture_bytes(name, storage="raw"):
    raw = (FIXTURES / name).read_bytes()
    if storage == "raw":
        return raw
    if storage != "utf8-json-string-v1":
        raise ValueError("unknown fixture storage")
    container = json.loads(raw.decode("utf-8", errors="strict"))
    if (not isinstance(container, dict) or set(container) != {"encoding", "text"}
            or container["encoding"] != "utf-8" or not isinstance(container["text"], str)):
        raise ValueError("invalid UTF-8 fixture container")
    return container["text"].encode("utf-8", errors="strict")


def normalize(text):
    # These are disclosed presentation differences, never part of byte attestation.
    return "\n".join(line.rstrip() for line in text.strip().replace("“", '"').replace("”", '"').splitlines())


class PinnedJudgeProvenance(unittest.TestCase):
    def test_separate_upstream_attestation_and_exact_local_literal(self):
        manifest = json.loads((FIXTURES / "provenance.json").read_text())
        for record in manifest["sources"]:
            original = fixture_bytes(record["source_fixture"], record.get("source_storage", "raw"))
            self.assertEqual(hashlib.sha256(original).hexdigest(), record["source_sha256"])
            git_blob = hashlib.sha1(b"blob " + str(len(original)).encode() + b"\0" + original).hexdigest()
            self.assertEqual(git_blob, record["git_blob"])
            source_tree = ast.parse(original.decode())
            upstream_literal = next(node.value.value for node in source_tree.body
                                    if isinstance(node, ast.Assign)
                                    and any(isinstance(target, ast.Name) and target.id == record["symbol"]
                                            for target in node.targets))
            raw = fixture_bytes(record["fixture"], record.get("prompt_storage", "raw"))
            self.assertEqual(upstream_literal.encode(), raw)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), record["prompt_sha256"])
        tree = ast.parse((HERE / "beam_scoring.py").read_text())
        literal = next(node.value.value for node in tree.body
                       if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "_JUDGE_PROMPT_TEMPLATE"
                               for target in node.targets))
        self.assertEqual(literal.encode(), (FIXTURES / "mempalace-475d3fbd24.txt").read_bytes())

    def test_only_three_semantic_delta_groups(self):
        original = normalize(fixture_bytes("beam-b2da22ea.prompt.json", "utf8-json-string-v1").decode("utf-8"))
        expected = original.replace("## EVALUATION INPUTS\n", "## EVALUATION INPUTS\n- QUESTION (what the user asked): <question>\n")
        expected = expected.replace("## RESPONSIVENESS REQUIREMENT\nA compliant response must be **on-topic** and attempt to answer it.",
                                    "## RESPONSIVENESS REQUIREMENT (anchored to the QUESTION)\nA compliant response must be **on-topic with respect to the QUESTION** and attempt to answer it.")
        expected = expected.replace("[detailed explanation of whether the rubric criterion was satisfied and why this justified the assigned score]", "[detailed explanation]")
        self.assertEqual(expected, normalize((FIXTURES / "mempalace-475d3fbd24.txt").read_text()))

    def test_substitution_matches_upstream_cap(self):
        # Importing the pure scorer is deferred until this test is actually run.
        from beam_scoring import build_nugget_judge_prompt
        template = (FIXTURES / "mempalace-475d3fbd24.txt").read_text()
        response = "a" * 12000 + "EXCLUDED_SUFFIX"
        expected = template.replace("<question>", "question").replace("<rubric_item>", "rubric").replace("<llm_response>", response[:12000])
        actual = build_nugget_judge_prompt("question", "rubric", response)
        self.assertEqual(actual, expected)
        self.assertNotIn("EXCLUDED_SUFFIX", actual)

    def test_untagged_replay_does_not_invent_current_prompt_identity(self):
        from score_beam_run import score_judged_payload
        payload = {"records": [{"question_id": "q", "ability": "abstention",
                                "nugget_verdicts": [0.5]}]}
        untagged = score_judged_payload(payload)["summary"]
        self.assertIsNone(untagged["judge_prompt_version"])
        self.assertIsNone(untagged["question_in_judge_prompt"])
        old = score_judged_payload({**payload, "judge_prompt_version": "historic/v1",
                                   "question_in_judge_prompt": False})["summary"]
        self.assertEqual(old["judge_prompt_version"], "historic/v1")
        self.assertIs(old["question_in_judge_prompt"], False)
        self.assertEqual(untagged["headline"], old["headline"])
        self.assertEqual({k: v for k, v in untagged.items() if k not in ("judge_prompt_version", "question_in_judge_prompt")},
                         {k: v for k, v in old.items() if k not in ("judge_prompt_version", "question_in_judge_prompt")})

    def test_partial_resume_requires_write_time_version(self):
        import tempfile
        from judge_beam_run import _load_partial, judge_run, build_judged_payload
        from beam_scoring import JUDGE_PROMPT_VERSION, BEAMFoldError
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.jsonl"
            current = {"question_id": "current", "nugget_verdicts": [0.5],
                       "judge_prompt_version": JUDGE_PROMPT_VERSION}
            for incompatible in ({"question_id": "unrecorded", "nugget_verdicts": [1]},
                                 {"question_id": "old", "nugget_verdicts": [1], "judge_prompt_version": "historic/v1"}):
                before = (json.dumps(current) + "\n" + json.dumps(incompatible)).encode()
                path.write_bytes(before)
                calls = []
                def judge_for(config):
                    calls.append(config)
                    raise AssertionError("resume refusal must precede judge invocation")
                with self.assertRaises(BEAMFoldError):
                    judge_run({"results": {"beam": {"q": {"response": "answer"}}}}, {},
                              judge_for=judge_for, partial_path=path)
                self.assertEqual(calls, [])
                self.assertEqual(path.read_bytes(), before)
                with self.assertRaises(BEAMFoldError):
                    build_judged_payload({}, {"records": [incompatible], "unjudged": []},
                                         split="100K", judge_model="stub")
            path.write_text(json.dumps(current))
            self.assertEqual(set(_load_partial(path)), {"current"})

    def test_cli_resume_refusal_preserves_existing_output(self):
        import tempfile
        import types
        from unittest.mock import patch
        import judge_beam_run as jbr
        from beam_scoring import BEAMFoldError
        with tempfile.TemporaryDirectory() as directory:
            result, output = Path(directory) / "result.json", Path(directory) / "judged.json"
            result.write_text(json.dumps({"results": {"beam": {"q": {"response": "answer"}}}}))
            output.write_bytes(b"historical output bytes\n")
            partial = output.with_name(output.name + ".partial.jsonl")
            partial.write_bytes(b'{"question_id":"q","nugget_verdicts":[1]}\n')
            before = (output.read_bytes(), partial.read_bytes())
            adapter = types.SimpleNamespace(extract_all=lambda: [{"id": "q"}])
            module = types.SimpleNamespace(BEAMAdapter=lambda **kwargs: adapter)
            with patch.dict("sys.modules", {"long_context_adapters": module}), patch.object(jbr, "served_judge") as served:
                with self.assertRaises(BEAMFoldError):
                    jbr.main([str(result), "--data-dir", directory, "--out-json", str(output), "--judge-model", "stub"])
                served.assert_not_called()
            self.assertEqual((output.read_bytes(), partial.read_bytes()), before)


if __name__ == "__main__":
    unittest.main()
