"""The rebuilt livecodebench oracle: RUN the answer, don't grep it.

Origin: epyc-root `artifacts/audit/unscoreable-rows-livecodebench-cruxeval-mah-20260812.md`.
All 2,360 livecodebench rows in the live pool carry `expected == "def "` — one distinct
value across the whole suite — and 2,349 score by `substring`. Measured through the real
scorer over the upstream corpus: `def solve():\n    pass` passes 100% of rows and echoing
the prompt passes 100% of rows. The other 11 rows set `code_execution` with a `test_code`
whose every assert line is commented out, and whose arguments were scraped out of English.

Widening `expected` is not a fix and is not tested here: no string test can separate a
correct program from a plausible one. The oracle under test executes the answer against
LeetCode's own worked examples.

Every test below asserts one of the two directions that matter:
  * an answer that solves NOTHING must not pass, and
  * the known-correct reference solution must pass.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "lcb_under_test", REPO / "scripts" / "benchmark" / "livecodebench_oracle.py")
lcb = importlib.util.module_from_spec(_spec)
sys.modules["lcb_under_test"] = lcb
_spec.loader.exec_module(lcb)

_scorer_spec = importlib.util.spec_from_file_location(
    "lcb_scorer_under_test", REPO / "scripts" / "benchmark" / "debug_scorer.py")
scorer = importlib.util.module_from_spec(_scorer_spec)
sys.modules["lcb_scorer_under_test"] = scorer
_scorer_spec.loader.exec_module(scorer)


# A LeetCode-shaped row, escaped exactly the way upstream escapes it.
ROW = {
    "slug": "two-sum",
    "title": "Two Sum",
    "difficulty": "Easy",
    "content": (
        "Given an array of integers `nums` and an integer `target`, return "
        "_indices of the two numbers_.\n\n"
        "**Example 1:**\n\n"
        "**Input:** nums = \\[2,7,11,15\\], target = 9\n"
        "**Output:** \\[0,1\\]\n\n"
        "**Example 2:**\n\n"
        "**Input:** nums = \\[3,2,4\\], target = 6\n"
        "**Output:** \\[1,2\\]\n"
    ),
    "python": (
        "\n    ```python\n"
        "def twoSum(nums, target):\n"
        "    seen = {}\n"
        "\n"
        "    for i, num in enumerate(nums):\n"
        "        if target - num in seen:\n"
        "            return [seen[target - num], i]\n"
        "        seen[num] = i\n"
        "    return []\n"
        "```\n"
    ),
}


def _oracle(row: dict = ROW) -> dict:
    built = lcb.build_oracle(row)
    assert built is not None, "fixture must be buildable or the test proves nothing"
    return built


def _score(answer: str, oracle: dict) -> bool:
    return bool(
        scorer.score_answer(answer, oracle["entry_point"], "code_execution", oracle)
    )


def _fenced(code: str) -> str:
    return f"```python\n{code}\n```"


CORRECT = (
    "def twoSum(nums, target):\n"
    "    for i in range(len(nums)):\n"
    "        for j in range(i + 1, len(nums)):\n"
    "            if nums[i] + nums[j] == target:\n"
    "                return [i, j]\n"
    "    return []\n"
)


# ── the two directions ───────────────────────────────────────────────────


@pytest.mark.usefixtures("_owned_synthetic_scorer_scratch")
def test_echoing_the_whole_prompt_fails() -> None:
    """THE decisive test. Under the shipped `substring "def "` oracle this passed."""
    prompt = lcb.default_prompt(ROW, "twoSum", ["nums", "target"])
    assert _score(prompt, _oracle()) is False


@pytest.mark.usefixtures("_owned_synthetic_scorer_scratch")
def test_a_signature_correct_stub_that_does_nothing_fails() -> None:
    """`def f(): pass` passed 100% of the shipped suite. It must not pass here."""
    assert _score(_fenced("def twoSum(nums, target):\n    pass"), _oracle()) is False


@pytest.mark.usefixtures("_owned_synthetic_scorer_scratch")
def test_returning_the_first_example_output_verbatim_fails() -> None:
    """The example outputs are printed IN the prompt, so a constant is an echo."""
    stub = "def twoSum(nums, target):\n    return [0, 1]"
    assert _score(_fenced(stub), _oracle()) is False


@pytest.mark.usefixtures("_owned_synthetic_scorer_scratch")
def test_returning_the_first_argument_fails() -> None:
    assert _score(_fenced("def twoSum(nums, target):\n    return nums"), _oracle()) is False


@pytest.mark.usefixtures("_owned_synthetic_scorer_scratch")
def test_a_different_but_correct_solution_passes() -> None:
    """The other direction: an oracle a correct answer fails is equally useless.

    This is deliberately NOT the reference — brute force, not a hash map. The
    oracle must test behaviour, not reproduction of an implementation.
    """
    assert _score(_fenced(CORRECT), _oracle()) is True


@pytest.mark.usefixtures("_owned_synthetic_scorer_scratch")
def test_a_correct_answer_with_different_parameter_names_passes() -> None:
    """Arguments are bound POSITIONALLY; the reference's parameter NAMES are not the test."""
    renamed = CORRECT.replace("nums", "values").replace("target", "goal")
    assert "values" in renamed
    assert _score(_fenced(renamed), _oracle()) is True


@pytest.mark.usefixtures("_owned_synthetic_scorer_scratch")
def test_a_correct_answer_wrapped_in_prose_passes() -> None:
    answer = f"Sure — here's an O(n) approach.\n\n{_fenced(CORRECT)}\n\nIt uses a dict."
    assert _score(answer, _oracle()) is True


@pytest.mark.usefixtures("_owned_synthetic_scorer_scratch")
def test_a_solution_correct_on_the_first_example_only_fails() -> None:
    """Why MIN_CASES exists: one case is satisfiable by memorising one answer."""
    partial = (
        "def twoSum(nums, target):\n"
        "    if nums == [2, 7, 11, 15]:\n"
        "        return [0, 1]\n"
        "    return []\n"
    )
    assert _score(_fenced(partial), _oracle()) is False


@pytest.mark.usefixtures("_owned_synthetic_scorer_scratch")
def test_an_answer_defining_the_wrong_function_name_fails() -> None:
    """Which is exactly why `prompt_contract` must ship in the prompt."""
    assert _score(_fenced(CORRECT.replace("twoSum", "two_sum")), _oracle()) is False


# ── the build gate ───────────────────────────────────────────────────────


@pytest.mark.usefixtures("_owned_synthetic_scorer_scratch")
def test_the_gate_emits_a_row_that_discriminates() -> None:
    oracle, diagnostics = lcb.build_validated_oracle(ROW)
    assert oracle is not None
    assert diagnostics["reason"] == "ok"
    assert diagnostics["any_echo_passes"] is False
    assert diagnostics["all_correct_pass"] is True


@pytest.mark.usefixtures("_owned_synthetic_scorer_scratch")
def test_the_gate_drops_a_row_whose_examples_all_share_one_answer() -> None:
    """A constant is guessable from the prompt, so such a row measures nothing.

    Discrimination is a GATE, not a report: this row is not emitted-with-a-warning,
    it is not emitted.
    """
    row = dict(ROW)
    row["content"] = ROW["content"].replace("**Output:** \\[1,2\\]", "**Output:** \\[0,1\\]")
    row["python"] = (
        "```python\ndef twoSum(nums, target):\n    return [0, 1]\n```"
    )
    oracle, diagnostics = lcb.build_validated_oracle(row)
    assert oracle is None
    assert diagnostics["reason"] == "failed_validation"
    assert diagnostics["echo"]["stub_returns_first_example_output"] is True


def test_the_gate_drops_a_row_whose_reference_fails_its_own_cases() -> None:
    """Upstream ships solutions to the WRONG problem; measured, 53.26% of buildable rows.

    An oracle no correct answer can satisfy is as useless as one everything
    satisfies, so a reference that cannot pass is a refusal, not a warning.
    """
    row = dict(ROW)
    row["python"] = "```python\ndef twoSum(nums, target):\n    return [9, 9]\n```"
    oracle, diagnostics = lcb.build_validated_oracle(row)
    assert oracle is None
    assert diagnostics["reason"] == "failed_validation"
    assert diagnostics["all_correct_pass"] is False


def test_the_gate_drops_a_row_with_only_one_worked_example() -> None:
    row = dict(ROW)
    row["content"] = ROW["content"].split("**Example 2:**")[0]
    oracle, diagnostics = lcb.build_validated_oracle(row)
    assert oracle is None
    assert diagnostics["reason"] == "no_buildable_cases"


def test_a_class_design_row_with_no_module_level_function_is_dropped() -> None:
    row = dict(ROW)
    row["python"] = "```python\nclass Solution:\n    def twoSum(self, nums, target):\n        return []\n```"
    assert lcb.build_cases(row) is None


# ── parsing upstream's prose ─────────────────────────────────────────────


def test_markdown_escaping_is_undone_before_values_are_parsed() -> None:
    """Without this every list argument parses as a string of escaped brackets."""
    cases = lcb.build_oracle(ROW)["entry_point_cases"]
    assert cases[0]["args"] == [[2, 7, 11, 15], 9]
    assert cases[0]["expected"] == [0, 1]


def test_the_export_space_before_a_closing_quote_is_removed() -> None:
    """Upstream renders `s = "abcabcbb"` as `s =  "abcabcbb "`."""
    ok, value = lcb.parse_value('"abcabcbb "')
    assert (ok, value) == (True, "abcabcbb")


def test_leading_whitespace_inside_a_string_is_preserved() -> None:
    """Only the export's trailing space goes; `s = "   -42 "` really does lead with spaces."""
    ok, value = lcb.parse_value('"   -42 "')
    assert (ok, value) == (True, "   -42")


def test_javascript_spelled_literals_are_parsed() -> None:
    assert lcb.parse_value("true") == (True, True)
    assert lcb.parse_value("false") == (True, False)
    assert lcb.parse_value("[1,null,3]") == (True, [1, None, 3])


def test_an_unparseable_output_is_refused_rather_than_guessed() -> None:
    """The shipped adapter's defect was guessing: `assert reverse_bits(and) == ...`."""
    ok, _ = lcb.parse_value("the answer is at most 3 because of the constraints")
    assert ok is False


def test_commas_inside_nested_brackets_do_not_split_arguments() -> None:
    parts = lcb.split_top_level('grid = [[1,2],[3,4]], k = "a,b"')
    assert parts == ['grid = [[1,2],[3,4]]', 'k = "a,b"']


def test_prose_that_is_not_a_binding_is_refused() -> None:
    ok, named, positional = lcb.parse_bindings("a 3 x 3 board with values")
    assert ok is False


# ── entry point resolution ───────────────────────────────────────────────


def test_the_entry_point_matching_the_slug_wins_over_the_first_definition() -> None:
    """The `happy-number` defect, pinned.

    Upstream writes `def get_next(n)` and THEN `def is_happy(n)`. Both take one
    argument named `n`, so every other criterion ties. Ranking by position picked
    `get_next`, silently pointing the oracle at a helper — which then looked like
    an upstream data defect rather than a bug in this module.
    """
    code = (
        "def get_next(n):\n    return n\n\n"
        "def is_happy(n):\n    return n == 1\n"
    )
    assert lcb.entry_point_of(code, ["n"], 1, slug="happy-number")[0] == "is_happy"


def test_the_entry_point_is_chosen_by_arity_when_the_example_names_nothing() -> None:
    """MUTATION-DISCOVERED. Replacing `arg_count` with `len(arg_names)` left all 35
    tests green, so the argument-count criterion was unpinned.

    `additive-number` renders its example as a bare value (`**Input:** "112358"`),
    so there are no parameter names at all. With no count to rank by, every
    candidate ties at zero and position alone decides — and upstream writes its
    recursion helper first often enough that position is a coin flip.
    """
    code = (
        "def check(num1, num2, remaining):\n    return False\n\n"
        "def evaluate(value):\n    return True\n"
    )
    assert lcb.entry_point_of(code, [], 1, slug="some-unrelated-slug")[0] == "evaluate"
    # And with two arguments in the example, the two-parameter helper is the one
    # that can actually be called.
    assert lcb.entry_point_of(code, [], 2, slug="some-unrelated-slug")[0] == "check"


def test_an_orphaned_self_parameter_is_removed() -> None:
    """Upstream dedented `class Solution` methods and left `self` behind."""
    code = "def trimBST(self, root, low, high):\n    return root\n"
    normalized = lcb.normalize_reference(code)
    assert lcb.entry_point_of(normalized, ["root", "low", "high"], 3, slug="trim")[1] == [
        "root", "low", "high"
    ]


def test_a_self_that_the_body_actually_uses_is_left_alone() -> None:
    """Semantics-preserving only when checked. A real method keeps its receiver."""
    code = "def f(self, x):\n    return self.y + x\n"
    assert lcb.normalize_reference(code) == code


def test_trailing_parameters_with_defaults_may_be_omitted() -> None:
    """Upstream's recursion accumulators (`def sumNumbers(root, cur=0)`)."""
    code = "def sumNumbers(root, cur=0):\n    return cur\n"
    name, params, required = lcb.entry_point_of(code, ["root"], 1, slug="sum-root")
    assert (name, params, required) == ("sumNumbers", ["root", "cur"], 1)
    assert lcb.order_args({"root": [1]}, [[1]], params, required) == [[1]]


def test_an_argument_count_the_signature_cannot_accept_is_refused() -> None:
    """`first-bad-version` names `n` and `bad`; the reference takes only `n`."""
    assert lcb.order_args({"n": 5, "bad": 4}, [5, 4], ["n"], 1) is None


@pytest.mark.usefixtures("_owned_synthetic_scorer_scratch")
def test_arguments_bind_by_source_order_when_the_prose_names_differ() -> None:
    """MUTATION-DISCOVERED. Deleting the positional fallback in `order_args` left
    all 35 tests green, so nothing pinned it.

    Upstream routinely names the prose and the reference differently (`def
    maxProfit(prices)` for `**Input:** stock = [7,1,5]`). Refusing on a name
    mismatch alone throws away rows whose ARGUMENTS are perfectly well determined,
    and getting the order wrong cannot produce a WRONG oracle — it produces a
    reference that fails its own cases, and the gate drops the row.
    """
    assert lcb.order_args({"stock": [7, 1, 5]}, [[7, 1, 5]], ["prices"], 1) == [[7, 1, 5]]
    row = dict(ROW)
    row["content"] = ROW["content"].replace("nums = ", "values = ").replace(
        "target = ", "goal = ")
    oracle, diagnostics = lcb.build_validated_oracle(row)
    assert oracle is not None, diagnostics
    assert oracle["entry_point_cases"][0]["args"] == [[2, 7, 11, 15], 9]


def test_a_named_argument_that_is_not_a_leading_parameter_is_refused() -> None:
    """MUTATION-DISCOVERED. Deleting the prefix/order guard left all 35 tests green.

    Here the example names only `cur`, the SECOND parameter. Without the guard the
    value is handed to the function as `root` — a silently mis-bound oracle, which
    is the shipped adapter's defect (assert on whatever was scraped) in a new
    costume. `required` alone does not catch it: `cur` is a defaulted parameter, so
    a one-argument call is perfectly legal.
    """
    assert lcb.order_args({"cur": 3}, [3], ["root", "cur"], 1) is None
    assert lcb.order_args({"root": [1]}, [[1]], ["root", "cur"], 1) == [[1]]


def test_the_emitted_signature_omits_defaulted_parameters_the_grader_never_passes() -> None:
    """Showing a model `def f(root, cur)` when the grader calls `f(root)` is a trick question."""
    row = dict(ROW)
    row["python"] = (
        "```python\ndef twoSum(nums, target, memo=None):\n"
        "    return [0, 1] if target == 9 else [1, 2]\n```"
    )
    _cases, name, params = lcb.build_cases(row)
    assert (name, params) == ("twoSum", ["nums", "target"])


# ── the prompt contract the oracle depends on ────────────────────────────


def test_the_prompt_states_the_exact_function_name_the_grader_will_call() -> None:
    """Without this the oracle measures name-guessing, not problem-solving."""
    contract = lcb.prompt_contract("twoSum", ["nums", "target"])
    assert "def twoSum(nums, target):" in contract
    assert "def twoSum(nums, target):" in lcb.default_prompt(ROW, "twoSum", ["nums", "target"])


def test_the_prompt_does_not_leak_the_reference_solution() -> None:
    prompt = lcb.default_prompt(ROW, "twoSum", ["nums", "target"])
    assert "seen[num] = i" not in prompt


# ── why the unfenced reference variant is measured but does not gate ─────


def test_the_scorer_truncates_unfenced_code_at_the_first_blank_line() -> None:
    """Evidence for `GATING_CORRECT_VARIANTS`, not an assertion about it.

    `_extract_code_block`'s unfenced fallback stops at `\\n\\n`. A raw answer that
    defines a helper, a blank line, then the entry point loses the entry point.
    Gating on that variant dropped 344 of 2,360 rows (14.58%) for a reason with
    nothing to do with their oracle — coverage 15.85% instead of 29.83%.
    """
    raw = "def helper(n):\n    return n + 1\n\ndef solve(n):\n    return helper(n)\n"
    assert scorer._extract_code_block(raw, "python") == "def helper(n):\n    return n + 1"
    assert "def solve" in scorer._extract_code_block(f"```python\n{raw}```", "python")


def test_the_unfenced_reference_is_still_scored_and_reported() -> None:
    """Excluded from the gate is not excluded from measurement."""
    assert "reference_raw" not in lcb.GATING_CORRECT_VARIANTS
    _oracle_, diagnostics = lcb.build_validated_oracle(ROW)
    assert "reference_raw" in diagnostics["correct"]
    assert "unfenced_reference_passes" in diagnostics


# ── the defect this replaces, pinned so it cannot come back ──────────────


def test_the_shipped_substring_oracle_passes_a_stub_that_solves_nothing() -> None:
    """Regression pin on the defect itself: `expected="def "`, `substring`."""
    config = {"case_sensitive": True, "substring": "def "}
    assert scorer.score_answer("def f(): pass", "def ", "substring", config) is True


def test_the_shipped_oracle_is_constant_across_the_suite() -> None:
    """An oracle identical on every question carries zero per-question information."""
    assert lcb.build_oracle(ROW)["entry_point_cases"] != lcb.build_oracle(
        {
            **ROW,
            "slug": "add-two",
            "content": ROW["content"].replace("target = 9", "target = 5"),
        }
    )["entry_point_cases"]


# ── integration against the real upstream snapshot ───────────────────────


_SNAPSHOT_MISSING = not lcb.DEFAULT_LEETCODE_JSONL.exists()


@pytest.mark.skipif(_SNAPSHOT_MISSING, reason="upstream leetcode snapshot not cached")
def test_upstream_ships_no_test_cases_only_prose_and_reference_solutions() -> None:
    """The premise of the whole module: an oracle must be manufactured, not transcribed."""
    with lcb.DEFAULT_LEETCODE_JSONL.open() as handle:
        row = json.loads(handle.readline())
    assert set(row) == {
        "id", "slug", "title", "difficulty", "content", "python", "java",
        "javascript", "c++",
    }


# ── the adapter that consumes the manifest ───────────────────────────────


_coding_spec = importlib.util.spec_from_file_location(
    "lcb_coding_adapters_under_test",
    REPO / "scripts" / "benchmark" / "dataset_adapter_modules" / "coding.py",
    submodule_search_locations=[
        str(REPO / "scripts" / "benchmark" / "dataset_adapter_modules")],
)
sys.path[:0] = [str(REPO / "scripts" / "benchmark")]
coding = importlib.util.module_from_spec(_coding_spec)
sys.modules["lcb_coding_adapters_under_test"] = coding
_coding_spec.loader.exec_module(coding)


def _upstream_two_sum() -> dict:
    with lcb.DEFAULT_LEETCODE_JSONL.open() as handle:
        return json.loads(handle.readline())


def test_the_manifest_path_resolves_to_a_built_manifest() -> None:
    """A path off by one directory yields an EMPTY manifest and a silent empty suite."""
    assert coding.LIVECODEBENCH_MANIFEST_PATH.exists(), (
        f"{coding.LIVECODEBENCH_MANIFEST_PATH} missing — regenerate with "
        "scripts/benchmark/livecodebench_oracle.py --manifest"
    )
    assert coding.livecodebench_manifest()["oracles"]


@pytest.mark.skipif(_SNAPSHOT_MISSING, reason="upstream leetcode snapshot not cached")
def test_the_adapter_emits_an_executable_oracle_not_the_string_def() -> None:
    question = coding.LiveCodeBenchAdapter()._row_to_prompt(0, _upstream_two_sum())
    assert question["scoring_method"] == "code_execution"
    assert question["expected"] != "def "
    assert question["scoring_config"]["entry_point"] == question["expected"]
    assert question["scoring_config"]["entry_point_cases"]
    assert question["scoring_config"].get("test_code") is None


@pytest.mark.skipif(_SNAPSHOT_MISSING, reason="upstream leetcode snapshot not cached")
def test_the_adapter_emits_the_signature_contract_in_the_prompt() -> None:
    question = coding.LiveCodeBenchAdapter()._row_to_prompt(0, _upstream_two_sum())
    assert "def twoSum(nums, target):" in question["prompt"]


@pytest.mark.skipif(_SNAPSHOT_MISSING, reason="upstream leetcode snapshot not cached")
def test_the_adapter_question_scores_the_two_directions_end_to_end() -> None:
    """The whole chain: manifest -> adapter row -> real scorer."""
    question = coding.LiveCodeBenchAdapter()._row_to_prompt(0, _upstream_two_sum())

    def _score_it(answer: str) -> bool:
        return bool(scorer.score_answer(
            answer, question["expected"], question["scoring_method"],
            question["scoring_config"]))

    assert _score_it(question["prompt"]) is False
    assert _score_it(_fenced("def twoSum(nums, target):\n    pass")) is False
    assert _score_it(_fenced(CORRECT)) is True


def test_the_manifest_oracle_varies_per_question() -> None:
    """The direct inverse of the defect: 2,360 rows shared ONE `expected`, `"def "`.

    An oracle identical on every question carries zero per-question information
    however elaborate it looks, so this asserts variety in the shipped artifact
    rather than in a fixture.
    """
    oracles = coding.livecodebench_manifest()["oracles"]
    entry_points = {entry["entry_point"] for entry in oracles.values()}
    assert len(entry_points) > len(oracles) / 2
    shared: dict[str, list[str]] = {}
    for slug, entry in oracles.items():
        key = json.dumps(entry["scoring_config"]["entry_point_cases"], sort_keys=True)
        shared.setdefault(key, []).append(slug)
    collisions = [slugs for slugs in shared.values() if len(slugs) > 1]
    # Exactly one collision, and it is upstream's, not ours: LeetCode's
    # `maximum-average-subarray-i` and `-ii` are printed with identical worked
    # examples. Two questions sharing an oracle is a duplicate; a suite sharing
    # one is the defect this replaces.
    assert collisions == [["maximum-average-subarray-i", "maximum-average-subarray-ii"]]


@pytest.mark.skipif(_SNAPSHOT_MISSING, reason="upstream leetcode snapshot not cached")
def test_sampling_draws_only_from_rows_that_have_an_oracle() -> None:
    """`sample(n)` must return n scoreable questions, not n minus the drops."""
    adapter = coding.LiveCodeBenchAdapter()
    adapter._dataset = [
        json.loads(line)
        for line in lcb.DEFAULT_LEETCODE_JSONL.read_text().splitlines()
        if line.strip()
    ]
    drawn = adapter.sample(20)
    assert len(drawn) == 20
    assert {q["scoring_method"] for q in drawn} == {"code_execution"}
    assert all(q["scoring_config"]["entry_point_cases"] for q in drawn)
    stratified = adapter.sample(30, stratify=True)
    assert len(stratified) == 30
    assert len({q["tier"] for q in stratified}) == 3


def test_the_adapter_drops_a_row_with_no_validated_oracle() -> None:
    """Dropped, never downgraded back to a string oracle."""
    row = {"slug": "no-such-problem-in-the-manifest", "title": "X", "content": "Y"}
    assert coding.LiveCodeBenchAdapter()._row_to_prompt(0, row) is None


def test_a_missing_manifest_empties_the_suite_rather_than_scoring_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-closed. A fail-open default here would resume scoring `"def "` forever."""
    monkeypatch.setattr(coding, "_LIVECODEBENCH_MANIFEST", None)
    monkeypatch.setattr(coding, "LIVECODEBENCH_MANIFEST_PATH", Path("/nonexistent/x.json"))
    assert coding.livecodebench_manifest() == {"oracles": {}}
    assert coding.LiveCodeBenchAdapter()._row_to_prompt(0, dict(ROW)) is None
    monkeypatch.setattr(coding, "_LIVECODEBENCH_MANIFEST", None)


@pytest.mark.skipif(_SNAPSHOT_MISSING, reason="upstream leetcode snapshot not cached")
def test_every_manifest_row_is_scoreable_by_the_eval_tower_predicate() -> None:
    """A row the tower books as UNSCOREABLE is a row this remediation did not fix.

    `_has_code_execution_oracle` accepts an `entry_point` oracle only when
    `expected` is also non-empty — the exact pair the adapter now emits.
    """
    tower_spec = importlib.util.spec_from_file_location(
        "lcb_eval_tower_under_test", REPO / "scripts" / "autopilot" / "eval_tower.py")
    tower = importlib.util.module_from_spec(tower_spec)
    sys.modules["lcb_eval_tower_under_test"] = tower
    tower_spec.loader.exec_module(tower)
    question = coding.LiveCodeBenchAdapter()._row_to_prompt(0, _upstream_two_sum())
    assert tower._is_scoreable_question(question) is True


@pytest.mark.skipif(_SNAPSHOT_MISSING, reason="upstream leetcode snapshot not cached")
def test_a_real_upstream_row_survives_the_gate_end_to_end() -> None:
    rows = {
        json.loads(line)["slug"]: json.loads(line)
        for line in lcb.DEFAULT_LEETCODE_JSONL.read_text().splitlines()[:40]
        if line.strip()
    }
    oracle, diagnostics = lcb.build_validated_oracle(rows["two-sum"])
    assert oracle is not None, diagnostics
    assert oracle["entry_point"] == "twoSum"
    assert len(oracle["entry_point_cases"]) >= lcb.MIN_CASES


@pytest.fixture
def _owned_synthetic_scorer_scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Real local synthetic scorer subprocesses; no model, dataset or network access."""
    import os
    import subprocess

    scratch = tmp_path / "synthetic-scorer"
    scratch.mkdir(mode=0o700)
    assert scratch.stat().st_uid == os.getuid()
    assert not list(scratch.iterdir())
    actual_oracle_scorer = lcb._scorer()
    monkeypatch.setattr(scorer, "_SCORER_TMP_ROOT", scratch)
    monkeypatch.setattr(actual_oracle_scorer, "_SCORER_TMP_ROOT", scratch)
    original_popen = subprocess.Popen
    children = []

    def bounded_local_child(argv, *args, **kwargs):
        assert isinstance(argv, list) and len(argv) == 2
        assert Path(argv[0]).resolve() == Path(sys.executable).resolve()
        solution = Path(argv[1]).resolve()
        cwd = Path(kwargs["cwd"]).resolve()
        assert solution.name == "solution.py" and solution.parent == cwd
        assert cwd.is_relative_to(scratch.resolve()) and cwd != scratch.resolve()
        assert not args
        assert kwargs == {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE,
                          "text": True, "cwd": Path(kwargs["cwd"]),
                          "start_new_session": True}
        child = original_popen(argv, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", bounded_local_child)
    yield scratch
    assert children, "selected synthetic scorer case never reached its real execution seam"
    assert all(child.poll() is not None for child in children)
    assert not list(scratch.iterdir())
