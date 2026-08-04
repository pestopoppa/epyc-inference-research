#!/usr/bin/env python3
"""test_schemas_require.py — `schemas.require` is a field type, and it STAYS one.

WHY THIS FILE EXISTS, AND WHY IT IS NOT A STYLE TEST
----------------------------------------------------
`_req_sha256` existed in three modules. Two matched `^[0-9a-f]{64}$` and stopped
there; the third also refused `schemas.is_placeholder_digest`. The bodies were
otherwise byte-identical, so nothing about reading either one told you that
`BuildProvenance.output_binary_sha256` — the identity of the built candidate —
accepted sixty-four zeros while the T0 provider next door refused them.

Fixing the two copies was one commit and it bought nothing durable: the copies
were not a mistake anyone made once, they were what this package does. The layer
that could have held the behaviour (`schemas.py`) held only predicates, so every
module that needed a validator wrote one, and writing one costs a `re.compile`
and four lines.

So the hoist is only half the work. The other half is this file, and its whole
job is to make the FOURTH copy impossible to write quietly. A module that wants
its own digest validator needs the shape and the predicate. Both live in
`schemas`, both are public, and there are exactly two ways to get them:

  * import them — `schemas.SHA256_RE`, `schemas.is_placeholder_digest` — which is
    a line on the diff that says what is about to happen; or
  * compile `^[0-9a-f]{64}$` locally, which `TestNoKeepSetModuleReDerivesAScalarValidator`
    refuses **by name and by module**.

There is no third route, and neither of the two is quiet. That is the property
this file defends. It is not defending line count: the hoist itself removed 57
lines, which is nothing.

SCOPE
-----
The walk covers the KEEP SET only — the 19 modules the simplification review
leaves standing. `evaluator/integrity.py`, `evaluator/surface.py`, `release/`,
`adapters/`, `controller/` and `surface/` are condemned; enforcing a hoist on
code that is about to be tagged and removed is the exact waste the refactor plan
exists to avoid, and a rule that fires on a file nobody will fix is a rule people
learn to skip.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

try:
    from . import schemas
except ImportError:                                                # pragma: no cover
    import schemas


PKG_ROOT = Path(__file__).resolve().parent

#: The modules the simplification review keeps. Relative to `PKG_ROOT`. Written
#: out rather than globbed: a glob would silently pick up a condemned module the
#: moment somebody moved it, and silently drop a keep-set one the moment somebody
#: renamed it — and a conformance test whose SCOPE can drift is not one.
KEEP_SET = (
    "schemas.py",
    "journal.py",
    "storage.py",
    "campaign.py",
    "evaluator/api.py",
    "evaluator/correctness.py",
    "evaluator/recipes.py",
    "evaluator/devices.py",
    "evaluator/controls.py",
    "evaluator/statistics.py",
    "execution/worktree.py",
    "execution/microbench.py",
    "execution/t0_provider.py",
    "execution/chain.py",
    "execution/cpu_region_claim.py",
    "execution/control_runner.py",
    "resource/device_claim.py",
    "resource/preflight.py",
    "resource/claim_witness.py",
)

#: `schemas.py` is where the field types live, so it is the one module allowed to
#: compile the digest shape and to define the bodies.
HOME = "schemas.py"

#: A bare, anchored, whole-string hex-digest pattern — the shape of an IDENTITY.
#: Deliberately not a substring rule: `storage._SHA256SUMS_LINE_RE` and
#: `t0_provider._BUILD_LOG_REF_RE` embed `[0-9a-f]{64}` inside a larger pattern
#: because they PARSE a line that contains a digest. Parsing a digest out of text
#: is not validating a field, and forbidding it would be a rule about characters
#: rather than about the thing that actually drifted.
FORBIDDEN_DIGEST_PATTERNS = frozenset({
    r"^[0-9a-f]{64}$",
    r"^[0-9a-f]{40}$",
    r"^[0-9a-f]{40,64}$",
})

#: Function names that ARE a promoted field type. A module may keep the name — 40
#: call sites read `_req_sha256(...)` and should keep reading it — but the name
#: must resolve to `schemas.require.<family>`, either by assignment or by a body
#: that calls it.
PROMOTED = {
    "sha256": ("_req_sha256", "_require_sha256"),
    "commit": ("_req_commit", "_require_commit"),
    "str": ("_req_str", "_require_str", "_require_nonempty_str", "_req_str_arg"),
    "int": ("_req_int", "_require_int"),
    "abs_path": ("_req_abs", "_require_abs_path"),
    "producer": ("_req_producer", "_require_producer"),
    "bool": ("_req_bool", "_require_bool"),
    "tuple": ("_req_tuple", "_require_tuple"),
}

#: Promoted NAMES that legitimately do NOT delegate, each with the reason it is
#: not the same predicate. Two kinds, and the distinction is the whole point:
#:
#:   * a DIFFERENT POLICY that happens to share a name — forcing it into the
#:     field type would be the category error §5 of the refactor plan warns about
#:     for the denylists ("unioning them ... would break the execution plane"); and
#:   * a module the plan puts on the DELETION list — refactoring code that is
#:     about to be tagged and removed is the exact waste this exercise exists to
#:     avoid, and a rule that fires on a file nobody will fix teaches people to
#:     skip the rule.
#:
#: An entry here is a claim, and `TestTheExemptionsAreDifferencesAndNotLaziness`
#: makes each one pay for itself with an observable behavioural difference.
NOT_THE_SAME_PREDICATE = {
    ("evaluator/recipes.py", "_require_int"):
        "an argv-token int: it carries a MAXIMUM and raises RecipeParameterError, "
        "and its message names the bound that was crossed",
    ("evaluator/recipes.py", "_require_abs_path"):
        "not `require.abs_path`. It is an argv token first (no leading '-'), then "
        "absolute, then '..'-free, then suffix-matched — four rules about what can "
        "be reproduced from a recorded argv, not one about a path",
    ("evaluator/controls.py", "_require_nonempty_str"):
        "on the deletion list (refactor plan §6: controls.py + control_runner.py, "
        "3,951 lines, superseded by campaign.anchor_drift)",
    ("execution/control_runner.py", "_require_nonempty_str"):
        "on the deletion list (refactor plan §6, same pair)",
}

#: `schemas.py` is the bottom layer, and this is the ONE non-stdlib name it
#: reaches for — deferred inside a function, wrapped in `except ImportError`, and
#: degrading to an UNREADABLE trust boundary rather than to an empty one. It is a
#: leaf format dependency: it creates no cycle, and no autokernel module is on
#: the other side of it. Enumerated so that the second one is a test failure.
#:
#: (The refactor plan calls this module "the only one importing nothing but
#: stdlib". At module level that is exactly true. At function level it is one
#: import short, and this list is where that stops being invisible.)
DEFERRED_THIRD_PARTY_IMPORTS = {
    "yaml": "parse_trust_boundary, function-level, optional, fail-closed",
}

#: THE ENUMERATED DEBT, and the only one. `evaluator/api._require_sha256` still
#: checks shape and not placeholder-ness, so `AnchorIdentity.binary_sha256`
#: accepts `"0" * 64`. Delegating it is a behaviour change, not a hoist: 152
#: tests build an `AnchorIdentity` or an `ArtifactIdentity` out of filler, and 8
#: of those live in `evaluator/test_surface.py`, a test module for condemned code
#: this refactor may not edit. The tightening waits for the deletion.
#:
#: The entry is what makes that a DECISION instead of a difference nobody sees:
#: `test_the_known_weaker_digest_validator_list_does_not_grow` fails if a second
#: module joins, and `test_the_known_weaker_entry_is_still_earned` fails if this
#: one is fixed and the entry is left behind to authorise the next copy.
KNOWN_WEAKER_DIGEST_VALIDATORS = {
    "evaluator/api.py": "_require_sha256",
}


def _parse(rel: str) -> ast.Module:
    return ast.parse((PKG_ROOT / rel).read_text(encoding="utf-8"), filename=rel)


def _existing_keep_set() -> tuple:
    """Keep-set modules that are on disk.

    The deletion of the condemned half is in flight in this tree. A keep-set
    module is never expected to be missing, and `test_the_keep_set_is_all_there`
    asserts exactly that — this helper exists so the OTHER tests report a real
    finding instead of a `FileNotFoundError` if it ever is.
    """
    return tuple(rel for rel in KEEP_SET if (PKG_ROOT / rel).is_file())


def _string_constants(tree: ast.Module) -> list:
    return [node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)]


def _dotted(node: ast.AST) -> str:
    """`a.b.c` for an attribute/name chain, else ''."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def _require_families_referenced(node: ast.AST) -> set:
    """Every `…require.<family>` reached from `node`, at any depth."""
    found = set()
    for sub in ast.walk(node):
        dotted = _dotted(sub)
        if not dotted:
            continue
        parts = dotted.split(".")
        if len(parts) >= 2 and parts[-2] == "require":
            found.add(parts[-1])
    return found


def _promoted_bindings(tree: ast.Module) -> list:
    """`(family, name, lineno, families_it_reaches)` for every promoted name bound.

    Covers both spellings a delegation can take:
      * `_req_sha256 = schemas.require.sha256`  (an assignment), and
      * `def _req_str_arg(...): return schemas.require.str(..., error=X)`
    """
    by_name = {name: family for family, names in PROMOTED.items() for name in names}
    out = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in by_name:
                    out.append((by_name[target.id], target.id, node.lineno,
                                _require_families_referenced(node.value)))
        elif isinstance(node, ast.FunctionDef) and node.name in by_name:
            out.append((by_name[node.name], node.name, node.lineno,
                        _require_families_referenced(node)))
    return out


class TestTheFieldTypesAreReal(unittest.TestCase):
    """`require` behaves, before anything asks whether it is used."""

    def test_a_placeholder_digest_is_not_a_measured_identity(self):
        for filler in ("0" * 64, "f" * 64, "a" * 64,
                       schemas.content_hash and "e3b0c44298fc1c149afbf4c8996fb924"
                       "27ae41e4649b934ca495991b7852b855"):
            with self.subTest(filler=filler):
                with self.assertRaises(ValueError) as ctx:
                    schemas.require.sha256(filler, "probe")
                self.assertIn("placeholder digest", str(ctx.exception))

    def test_a_real_digest_passes_and_is_returned(self):
        real = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
        self.assertEqual(schemas.require.sha256(real, "probe"), real)

    def test_a_malformed_digest_is_refused_before_the_placeholder_test(self):
        for bad in ("", "zz", "0" * 63, "A" * 64, None, 0, ["0" * 64]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError) as ctx:
                    schemas.require.sha256(bad, "probe")
                self.assertIn("lowercase sha256 hex digest", str(ctx.exception))

    def test_the_error_type_is_the_callers_and_not_this_modules(self):
        """`error=` shares the predicate without exporting `schemas`' vocabulary."""
        class SeamError(Exception):
            pass

        for call in (lambda: schemas.require.str("", "p", error=SeamError),
                     lambda: schemas.require.sha256("0" * 64, "p", error=SeamError),
                     lambda: schemas.require.commit("x", "p", error=SeamError),
                     lambda: schemas.require.abs_path("rel", "p", error=SeamError),
                     lambda: schemas.require.int(-1, "p", error=SeamError),
                     lambda: schemas.require.bool(1, "p", error=SeamError),
                     lambda: schemas.require.tuple([], "p", error=SeamError),
                     lambda: schemas.require.producer("nobody", "p", error=SeamError)):
            with self.subTest(call=call):
                with self.assertRaises(SeamError):
                    call()

    def test_a_bool_is_not_an_int(self):
        with self.assertRaises(ValueError):
            schemas.require.int(True, "probe")

    def test_a_list_is_refused_and_never_converted(self):
        with self.assertRaises(TypeError):
            schemas.require.tuple([1, 2], "probe")

    def test_a_relative_path_is_not_an_absolute_one(self):
        self.assertEqual(schemas.require.abs_path("/x/y", "p"), "/x/y")
        for bad in ("x/y", "", "  ", None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    schemas.require.abs_path(bad, "p")

    def test_the_producer_domain_is_the_one_the_evaluator_publishes(self):
        from .evaluator import correctness
        self.assertIs(correctness.EVIDENCE_PRODUCERS, schemas.EVIDENCE_PRODUCERS)
        for producer in schemas.EVIDENCE_PRODUCERS:
            self.assertEqual(schemas.require.producer(producer, "p"), producer)
        with self.assertRaises(ValueError):
            schemas.require.producer("planner", "p")

    def test_require_is_a_namespace_and_not_a_class_to_instantiate(self):
        with self.assertRaises(TypeError):
            schemas.require()

    def test_every_promoted_family_is_actually_exposed(self):
        for family in PROMOTED:
            with self.subTest(family=family):
                self.assertTrue(callable(getattr(schemas.require, family, None)))


class TestSchemasStaysTheBottomLayer(unittest.TestCase):
    """`require` may only live here while `schemas.py` imports nothing but stdlib.

    That is not tidiness. `schemas.py` is the one module all the others import,
    and it can only stay that way while it imports none of them. The moment it
    reaches for `evaluator/…` the import graph acquires a cycle and the next
    concern has nowhere to be hoisted TO — which is the mechanism that produced
    the copies in the first place.
    """

    STDLIB_ONLY = {"hashlib", "json", "math", "posixpath", "re", "dataclasses",
                   "datetime", "fnmatch", "typing", "__future__", "collections",
                   "itertools", "functools", "os", "sys", "enum", "abc", "types"}

    @staticmethod
    def _imports(tree, *, top_level_only):
        """`{root_package: lineno}` for every import, or only the module-level ones."""
        nodes = tree.body if top_level_only else list(ast.walk(tree))
        found = {}
        for node in nodes:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found.setdefault(alias.name.split(".")[0], node.lineno)
            elif isinstance(node, ast.ImportFrom):
                name = "." * node.level + (node.module or "").split(".")[0]
                found.setdefault(name, node.lineno)
        return found

    def test_the_module_level_imports_are_stdlib_only(self):
        """What every other module gets when it does `from .. import schemas`."""
        tree = _parse(HOME)
        unexpected = sorted(root for root in self._imports(tree, top_level_only=True)
                            if root not in self.STDLIB_ONLY)
        self.assertEqual(unexpected, [],
                         f"schemas.py imported something outside the stdlib at module "
                         f"level: {unexpected}")

    def test_schemas_imports_no_sibling_at_any_depth(self):
        """The property that makes it the bottom layer, and it is not about stdlib.

        A relative import — or an `autokernel.…` one — inside a function is still
        a cycle; it just fails later. This is the assertion that keeps `require`
        hoistable INTO here, which is the whole reason the copies existed.
        """
        offenders = sorted(
            f"{root} (line {lineno})"
            for root, lineno in self._imports(_parse(HOME), top_level_only=False).items()
            if root.startswith(".") or root in ("autokernel", "scripts"))
        self.assertEqual(offenders, [],
                         f"schemas.py reached for a sibling module: {offenders}")

    def test_the_one_deferred_third_party_import_is_the_one_on_the_list(self):
        """`yaml`, function-level and optional — enumerated so a second is a failure.

        Found by this test rather than asserted from the plan, which says this
        module imports "nothing but stdlib". At module level that is true; at
        function level it was one short, and nothing said so.
        """
        deferred = {root for root in self._imports(_parse(HOME), top_level_only=False)
                    if root not in self.STDLIB_ONLY
                    and root not in self._imports(_parse(HOME), top_level_only=True)}
        self.assertEqual(deferred, set(DEFERRED_THIRD_PARTY_IMPORTS),
                         "the deferred non-stdlib imports of schemas.py changed; each "
                         "one is a dependency the bottom layer takes on, so it is "
                         "listed with its reason or it is not there")

    def test_the_deferred_import_still_degrades_instead_of_raising(self):
        """It is only tolerable because its absence is a REFUSAL, not a soft pass."""
        boundary = schemas.parse_trust_boundary("not: {this: schema}")
        self.assertEqual(boundary.globs, ())
        self.assertFalse(boundary.readable)

    def test_the_producer_domain_did_not_come_back_from_the_evaluator(self):
        """`require.producer` needs its domain, so the domain moved HERE.

        Had it stayed in `evaluator/correctness.py`, `schemas` would have had to
        import the evaluator to type the field — the cycle above, arriving by the
        smallest possible door.
        """
        self.assertIn("EVIDENCE_PRODUCERS", schemas.__all__)
        self.assertEqual(schemas.EVIDENCE_PRODUCERS,
                         ("evaluator", "candidate", "actor", "unknown"))


class TestNoKeepSetModuleReDerivesAScalarValidator(unittest.TestCase):
    """The teeth. A fourth `_req_sha256` cannot be written without failing here."""

    def test_the_keep_set_is_all_there(self):
        missing = [rel for rel in KEEP_SET if not (PKG_ROOT / rel).is_file()]
        self.assertEqual(missing, [],
                         "a keep-set module is gone; either the deletion took one it "
                         "should not have, or KEEP_SET names a module that was renamed")

    def test_only_schemas_compiles_the_digest_shape(self):
        """The re-derivation route that needs no import, closed by name.

        This is the load-bearing assertion in the file. Everything else here
        checks that the hoist happened; this checks that it cannot be undone by
        someone who simply does not import `schemas.require` at all.
        """
        offenders = []
        for rel in _existing_keep_set():
            if rel == HOME:
                continue
            for text in _string_constants(_parse(rel)):
                if text in FORBIDDEN_DIGEST_PATTERNS:
                    offenders.append(f"{rel}: {text!r}")
        self.assertEqual(sorted(offenders), [],
                         "a keep-set module compiled its own digest shape. Use "
                         "`schemas.SHA256_RE` / `schemas.COMMIT_RE`, and validate with "
                         "`schemas.require.sha256`, which also refuses a placeholder.")

    def test_every_promoted_name_resolves_to_the_field_type(self):
        offenders = []
        for rel in _existing_keep_set():
            if rel == HOME:
                continue
            for family, name, lineno, reached in _promoted_bindings(_parse(rel)):
                if KNOWN_WEAKER_DIGEST_VALIDATORS.get(rel) == name:
                    continue
                if (rel, name) in NOT_THE_SAME_PREDICATE:
                    continue
                if family not in reached:
                    offenders.append(f"{rel}:{lineno} {name} does not reach "
                                     f"schemas.require.{family}")
        self.assertEqual(sorted(offenders), [],
                         "a module re-implemented a promoted field type instead of "
                         "delegating to it. Keep the name; delete the body — or, if it "
                         "genuinely is a different predicate, add it to "
                         "NOT_THE_SAME_PREDICATE with the reason and a test that shows "
                         "the difference.")

    def test_every_exemption_names_a_function_that_exists(self):
        """An exemption for a function nobody has any more is a licence lying around."""
        stale = []
        for (rel, name), _reason in NOT_THE_SAME_PREDICATE.items():
            if not (PKG_ROOT / rel).is_file():
                continue
            names = {node.name for node in _parse(rel).body
                     if isinstance(node, ast.FunctionDef)}
            names |= {t.id for node in _parse(rel).body if isinstance(node, ast.Assign)
                      for t in node.targets if isinstance(t, ast.Name)}
            if name not in names:
                stale.append(f"{rel}:{name}")
        self.assertEqual(sorted(stale), [],
                         "an exemption names a function that is gone; delete the entry")

    def test_every_exemption_states_a_reason(self):
        for key, reason in NOT_THE_SAME_PREDICATE.items():
            with self.subTest(key=key):
                self.assertGreater(len(reason.strip()), 30,
                                   "an exemption without a reason is a silent fork")

    def test_the_known_weaker_digest_validator_list_does_not_grow(self):
        """One entry, and the point of the list is that a second one is a failure."""
        self.assertEqual(KNOWN_WEAKER_DIGEST_VALIDATORS,
                         {"evaluator/api.py": "_require_sha256"})

    def test_the_known_weaker_entry_is_still_earned(self):
        """If it were fixed and the entry stayed, the entry would authorise the next copy.

        A permanent exception list is how a gate rots. This fails the day
        `api._require_sha256` starts refusing a placeholder, so the entry has to
        be deleted in the same commit that earns its deletion.
        """
        from .evaluator import api
        try:
            api._require_sha256("0" * 64, "probe")       # the debt, executed
        except ValueError:
            self.fail("api._require_sha256 now refuses a placeholder — good. Delete its "
                      "entry from KNOWN_WEAKER_DIGEST_VALIDATORS and the paragraph in "
                      "its docstring, or the entry stays behind as a licence for the "
                      "next copy.")
        self.assertIn("KNOWN WEAKER GATE", api._require_sha256.__doc__ or "",
                      "the deviation must say so where a reader of the function is")

    def test_no_keep_set_module_duplicates_the_placeholder_predicate(self):
        """`is_placeholder_digest` has one body too — a second would be a fork.

        The check it performs is a policy ("what counts as filler"), and
        `controls.py:2392` already predicted in writing what a second copy of a
        policy table does: *both copies still return PASS*.
        """
        offenders = []
        for rel in _existing_keep_set():
            if rel == HOME:
                continue
            for node in ast.walk(_parse(rel)):
                if isinstance(node, ast.FunctionDef) and "placeholder" in node.name:
                    offenders.append(f"{rel}:{node.lineno} {node.name}")
        self.assertEqual(sorted(offenders), [],
                         "call `schemas.is_placeholder_digest`; do not re-decide what "
                         "counts as a fabricated digest")


class TestTheExemptionsAreDifferencesAndNotLaziness(unittest.TestCase):
    """Each `NOT_THE_SAME_PREDICATE` entry, made to show the difference it claims.

    Without this, the exemption dict is a place to put anything inconvenient —
    which is how the denylists forked in the first place: both copies still said
    PASS, and nothing ever asked either of them to prove it was the one it meant
    to be.
    """

    def test_the_recipes_int_carries_a_maximum_the_field_type_has_no_word_for(self):
        from .evaluator import recipes
        with self.assertRaises(recipes.RecipeParameterError):
            recipes._require_int(9, "threads", minimum=1, maximum=8)
        self.assertEqual(schemas.require.int(9, "threads"), 9)     # no upper bound

    def test_the_recipes_abs_path_refuses_paths_the_field_type_accepts(self):
        from .evaluator import recipes
        for token, why in (("-/etc/passwd", "an argv token may not start with '-'"),
                           ("/a/../b", "a '..' segment does not identify what ran"),
                           ("/a/model.txt", "the suffix is part of the contract")):
            with self.subTest(why=why):
                with self.assertRaises(recipes.RecipeParameterError):
                    recipes._require_abs_path(token, "model", suffix=".gguf")
        self.assertEqual(schemas.require.abs_path("/a/../b", "model"), "/a/../b")

    def test_the_recipes_error_type_is_the_recipes_one(self):
        """A caller that catches `RecipeParameterError` must keep catching it."""
        from .evaluator import recipes
        with self.assertRaises(recipes.RecipeParameterError):
            recipes._require_str("", "flag")
        with self.assertRaises(recipes.RecipeParameterError):
            recipes._require_str("a\x00b", "flag")

    def test_the_two_deletion_list_exemptions_name_deletion_and_nothing_else(self):
        """The other exemption kind is temporal, and it must say so.

        `controls.py` and `control_runner.py` are 3,951 lines the plan routes to
        the deletion list because `campaign.anchor_drift` measures the same
        quantity in-run. When they go, these two entries go with them, and
        `test_every_exemption_names_a_function_that_exists` is what notices.
        """
        for rel in ("evaluator/controls.py", "execution/control_runner.py"):
            with self.subTest(rel=rel):
                self.assertIn("deletion list",
                              NOT_THE_SAME_PREDICATE[(rel, "_require_nonempty_str")])


class TestTheDelegationIsRealAndNotAReimplementation(unittest.TestCase):
    """Each migrated name IS the field type, checked by identity where it can be.

    An AST test can be satisfied by a body that calls `require.sha256` and then
    ignores it. These do not go through the AST.
    """

    def test_correctness_and_the_execution_plane_share_one_digest_validator(self):
        from .evaluator import correctness
        from .execution import t0_provider, worktree
        for module in (correctness, t0_provider, worktree):
            with self.subTest(module=module.__name__):
                self.assertIs(module._req_sha256, schemas.require.sha256)

    def test_the_names_that_stayed_are_the_same_object_everywhere(self):
        from .evaluator import api, correctness
        from .execution import t0_provider, worktree
        self.assertIs(correctness._req_str, schemas.require.str)
        self.assertIs(t0_provider._req_str, schemas.require.str)
        self.assertIs(worktree._req_str, schemas.require.str)
        self.assertIs(api._require_nonempty_str, schemas.require.str)
        self.assertIs(correctness._req_commit, schemas.require.commit)
        self.assertIs(t0_provider._req_commit, schemas.require.commit)
        self.assertIs(worktree._req_commit, schemas.require.commit)
        self.assertIs(t0_provider._req_abs, schemas.require.abs_path)
        self.assertIs(correctness._req_producer, schemas.require.producer)
        self.assertIs(correctness._req_bool, schemas.require.bool)
        self.assertIs(correctness._req_tuple, schemas.require.tuple)
        self.assertIs(correctness._req_int, schemas.require.int)
        self.assertIs(t0_provider._req_int, schemas.require.int)

    def test_the_optional_wrappers_inherit_the_rejection(self):
        """`_opt_sha256` is not a second policy; it is `None`-or-the-field-type."""
        from .evaluator import correctness
        self.assertIsNone(correctness._opt_sha256(None, "p"))
        with self.assertRaises(ValueError):
            correctness._opt_sha256("0" * 64, "p")
        self.assertIsNone(correctness._opt_commit(None, "p"))
        with self.assertRaises(ValueError):
            correctness._opt_commit("nope", "p")

    def test_the_worktree_tree_digest_exemption_is_still_exactly_one_value(self):
        """The one admitted filler, and it is admitted because it is a MEASUREMENT.

        `sha256` of an empty manifest is what a clean build directory hashes to,
        and `run_build(require_fresh_build_dir=True)` requires exactly that value.
        Delegating `_req_sha256` must not have widened this back out.
        """
        from .evaluator import integrity
        from .execution import worktree
        self.assertEqual(worktree._req_tree_digest(integrity.EMPTY_TREE_SHA256, "p"),
                         integrity.EMPTY_TREE_SHA256)
        for still_refused in ("0" * 64, "f" * 64, "a" * 64):
            with self.subTest(value=still_refused):
                with self.assertRaises(ValueError):
                    worktree._req_tree_digest(still_refused, "p")

    def test_the_chain_seam_keeps_raising_its_own_error(self):
        """`error=` shares the predicate and not the vocabulary. Verified on the seam.

        `execution/test_execution_chain.py` cannot load in this tree — it imports
        `controller.state_machine`, which the in-flight deletion removed — so the
        one behaviour this refactor changed in `chain.py` is asserted here instead
        of nowhere.
        """
        from .execution import chain
        self.assertEqual(chain._req_str_arg("branch", "branch_name"), "branch")
        for bad in ("", "   ", None, 7):
            with self.subTest(bad=bad):
                with self.assertRaises(chain.ChainSeamError):
                    chain._req_str_arg(bad, "branch_name")


if __name__ == "__main__":                                          # pragma: no cover
    unittest.main()
