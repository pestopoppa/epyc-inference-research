"""Create the four pinned CJ-7a tasks without executing source or candidates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

SOURCE_PATH = Path(__file__).parent / "fixtures" / "source_snapshot.py"
SOURCE_SHA256 = "844301cb9ac6a7b4344266a8a5025e7abcf47f06da27df54d6a1e675a02ad86a"
SOURCE_REPOSITORY = "https://github.com/pestopoppa/epyc-inference-research"
SOURCE_COMMIT = "ba4a037de79e282940dbdc1901646e774e9939ba"
SOURCE_ORIGINAL_PATH = "scripts/benchmark/e6_budget_analyze.py"
SOURCE_LICENSE = "MIT (repository LICENSE at pinned source commit)"


@dataclass(frozen=True)
class Fixture:
    fixture_id: str
    mutation: str
    instruction: str
    source: str
    mutated_source: str
    expected_source: str
    provenance: dict[str, str]

    def task_payload(self) -> dict[str, str]:
        """Return only actor-visible data; references and provenance stay private."""
        return {
            "id": self.fixture_id,
            "instruction": self.instruction,
            "source": self.mutated_source,
        }


_MUTATIONS = (
    (
        "operator-swap",
        "b + c",
        "b - c",
        "Restore the original arithmetic operator in the McNemar count.",
    ),
    (
        "boolean-flip",
        "if not d:",
        "if d:",
        "Restore the original condition that skips empty result data.",
    ),
    (
        "off-by-one",
        "range(k + 1)",
        "range(k)",
        "Restore the inclusive upper endpoint of the exact binomial sum.",
    ),
    (
        "removed-guard",
        "if n else 1.0",
        "",
        "Restore the empty-discordance guard so the zero case returns 1.0.",
    ),
)


def _source_bytes() -> bytes:
    raw = SOURCE_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("pinned first-party source snapshot hash mismatch")
    return raw


def generate_fixtures() -> tuple[Fixture, ...]:
    """Generate known-inverse tasks from exact, unique source substitutions."""
    source = _source_bytes().decode("utf-8")
    provenance = {
        "repository": SOURCE_REPOSITORY,
        "commit": SOURCE_COMMIT,
        "path": SOURCE_ORIGINAL_PATH,
        "source_sha256": SOURCE_SHA256,
        "license": SOURCE_LICENSE,
        "scope": "mechanical edit application only; not software engineering",
    }
    result = []
    for name, original, mutated, instruction in _MUTATIONS:
        if source.count(original) != 1:
            raise ValueError(f"fixture {name}: source anchor must occur exactly once")
        changed = source.replace(original, mutated, 1)
        if changed == source:
            raise ValueError(f"fixture {name}: mutation did not change the source")
        result.append(
            Fixture(
                fixture_id=f"cj7a-{name}-001",
                mutation=name,
                instruction=instruction,
                source=source,
                mutated_source=changed,
                expected_source=source,
                provenance=provenance.copy(),
            )
        )
    return tuple(result)


def get_fixture(fixture_id: str) -> Fixture:
    for fixture in generate_fixtures():
        if fixture.fixture_id == fixture_id:
            return fixture
    raise KeyError(f"unknown CJ-7a fixture: {fixture_id}")
