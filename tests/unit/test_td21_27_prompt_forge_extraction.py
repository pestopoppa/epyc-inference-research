"""Tests for TD-21.27: PromptForge mutation-extraction failure handling.

Before: `_extract_mutation`/`_extract_code_mutation` fished a fenced block
out of the Claude CLI reply and, on a miss, silently returned `original` --
the mutation became an invisible no-op (`mutated_content == original_content`)
while `propose_mutation` still ran the full integrity/transfer-safety/effect
pipeline over it and `_action_prompt_mutation` still burned an eval cycle
before discovering nothing had changed. `_extract_code_mutation` also used a
weaker, non-line-anchored fence split than `_extract_mutation`, the exact bug
class `_extract_mutation`'s own docstring already warned about.

After: both extractors share one line-anchored `_fenced_blocks` fisher and
return `(content, ok)`. `propose_mutation`/`propose_code_mutation` short-
circuit on `ok=False`, marking the mutation `safety_valid=False` with an
explicit `mutation_extraction_failed` reason -- reusing the EXISTING gate
`_action_prompt_mutation`/`_action_code_mutation` already check before
running an eval, so a Claude CLI miss is now visible and free instead of
silent and wasteful. The Claude CLI backend has no HTTP endpoint for the
shared `src.structured_output.repair` idiom to reach, so this is a
deterministic-fish-only fix (`MUTATION_EXTRACTION_COUNTS`), per the TD-21
dispatch's HARD RULES.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = ROOT / "scripts" / "autopilot"
sys.path.insert(0, str(AUTOPILOT_DIR))

import species.prompt_forge as prompt_forge_mod  # noqa: E402
from species.prompt_forge import (  # noqa: E402
    EvalIdVocabulary,
    MUTATION_EXTRACTION_COUNTS,
    PromptForge,
)


def _reset_counts():
    MUTATION_EXTRACTION_COUNTS.clear()


class TestExtractMutationUnit:
    def setup_method(self):
        _reset_counts()

    def test_fenced_markdown_block_is_extracted(self):
        forge = PromptForge(prompts_dir=Path("/tmp/unused"), auto_commit=False)
        content, ok = forge._extract_mutation(
            "```markdown\nBase prompt\nBe careful.\n```", "Base prompt\n"
        )
        assert ok is True
        assert content == "Base prompt\nBe careful."
        assert MUTATION_EXTRACTION_COUNTS[("prompt_forge.mutation_extraction", "parsed")] == 1

    def test_no_fence_at_all_is_a_typed_failure_not_silent(self):
        forge = PromptForge(prompts_dir=Path("/tmp/unused"), auto_commit=False)
        content, ok = forge._extract_mutation("Sure, I'll improve it somehow.", "Base prompt\n")
        assert ok is False
        assert content == "Base prompt\n"  # safe no-op, never fabricated
        assert MUTATION_EXTRACTION_COUNTS[("prompt_forge.mutation_extraction", "failed")] == 1

    def test_inline_fence_mention_in_prose_never_mis_captured(self):
        # The exact bug class `_extract_mutation` already guarded against:
        # an inline ``` mention inside a sentence must not be treated as a
        # fence delimiter.
        forge = PromptForge(prompts_dir=Path("/tmp/unused"), auto_commit=False)
        prose = (
            "I looked at the code and noticed the call to ```result.index(...)``` "
            "inside the loop, but I did not produce a replacement prompt."
        )
        content, ok = forge._extract_mutation(prose, "Base prompt\n")
        assert ok is False
        assert content == "Base prompt\n"


class TestExtractCodeMutationUnit:
    def setup_method(self):
        _reset_counts()

    def test_fenced_python_block_is_extracted(self):
        forge = PromptForge(prompts_dir=Path("/tmp/unused"), auto_commit=False)
        content, ok = forge._extract_code_mutation("```python\nVALUE = 1\n```", "")
        assert ok is True
        assert content == "VALUE = 1"
        assert MUTATION_EXTRACTION_COUNTS[("prompt_forge.code_mutation_extraction", "parsed")] == 1

    def test_no_fence_at_all_is_a_typed_failure(self):
        forge = PromptForge(prompts_dir=Path("/tmp/unused"), auto_commit=False)
        content, ok = forge._extract_code_mutation("I decided not to change anything.", "VALUE = 1")
        assert ok is False
        assert content == "VALUE = 1"
        assert MUTATION_EXTRACTION_COUNTS[("prompt_forge.code_mutation_extraction", "failed")] == 1

    def test_inline_triple_backtick_in_prose_never_mis_captured(self):
        # The weaker pre-TD-21.27 `result.split("```")` treated ANY triple
        # backtick as a delimiter with no line-anchoring; this must not
        # regress once `_extract_code_mutation` shares `_fenced_blocks`.
        forge = PromptForge(prompts_dir=Path("/tmp/unused"), auto_commit=False)
        prose = "Calling ```obj.method()``` here is fine, no code block follows."
        content, ok = forge._extract_code_mutation(prose, "VALUE = 1")
        assert ok is False
        assert content == "VALUE = 1"


class TestProposeMutationSkipsOnExtractionFailure:
    def setup_method(self):
        _reset_counts()

    def test_propose_mutation_marks_extraction_failure_explicitly(self, tmp_path, monkeypatch):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "worker_math.md").write_text("Base prompt\n")
        forge = PromptForge(prompts_dir=prompts_dir, auto_commit=False)
        monkeypatch.setattr(forge, "_invoke_claude", lambda _p: "No fenced block in this reply.")

        mutation = forge.propose_mutation("worker_math.md")

        assert mutation.safety_valid is False
        assert mutation.safety_reason.startswith("mutation_extraction_failed")
        assert mutation.mutated_content == mutation.original_content
        assert MUTATION_EXTRACTION_COUNTS[("prompt_forge.mutation_extraction", "failed")] == 1

    def test_propose_mutation_happy_path_unaffected(self, tmp_path, monkeypatch):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "worker_math.md").write_text("Base prompt\n")
        vocabulary = EvalIdVocabulary.from_rows([
            {"suite": "synthetic_suite", "id": "synthetic_eval_001", "prompt": "synthetic eval prompt"},
        ])
        forge = PromptForge(
            prompts_dir=prompts_dir,
            auto_commit=False,
            eval_id_vocabulary=vocabulary,
        )
        monkeypatch.setattr(
            forge, "_invoke_claude", lambda _p: "```markdown\nBase prompt\nBe careful.\n```"
        )

        mutation = forge.propose_mutation("worker_math.md")

        assert mutation.safety_valid is True, mutation.safety_reason
        assert mutation.mutated_content == "Base prompt\nBe careful."


class TestProposeCodeMutationSkipsOnExtractionFailure:
    def setup_method(self):
        _reset_counts()

    def test_propose_code_mutation_marks_extraction_failure_explicitly(self, tmp_path, monkeypatch):
        target = "src/tool_policy.py"
        forge = PromptForge(prompts_dir=tmp_path, auto_commit=False)
        monkeypatch.setattr(forge, "_invoke_claude", lambda _p: "I chose not to change anything.")

        mutation = forge.propose_code_mutation(target_file=target)

        assert mutation.safety_valid is False
        assert mutation.safety_reason.startswith("mutation_extraction_failed")
        assert mutation.mutated_content == mutation.original_content
        # Routed through the transfer_safety gate ordering (syntax_valid=True
        # is a routing signal here, not a claim anything was screened).
        assert mutation.syntax_valid is True
        assert MUTATION_EXTRACTION_COUNTS[("prompt_forge.code_mutation_extraction", "failed")] == 1
