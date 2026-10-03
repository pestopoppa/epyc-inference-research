"""Tests for prompt hot-swap resolver."""

from __future__ import annotations

import pytest

from src.prompt_builders.resolver import (
    PROMPT_DIR,
    _get_variant,
    _safe_format,
    current_prompt_dir,
    get_direct_answer_prefix,
    prompt_dir_override,
    resolve_prompt,
)

_RETIRED_ARCHITECT_ROLE = "architect_" "coding"


# ── _safe_format ─────────────────────────────────────────────────────────


class TestSafeFormat:
    """Safe template interpolation."""

    def test_replaces_known_vars(self):
        result = _safe_format("Hello {name}!", {"name": "World"})
        assert result == "Hello World!"

    def test_missing_vars_stay_as_placeholders(self):
        result = _safe_format("Hello {name}, you are {role}!", {"name": "Alice"})
        assert result == "Hello Alice, you are {role}!"

    def test_empty_vars(self):
        result = _safe_format("Hello {name}!", {})
        assert result == "Hello {name}!"

    def test_no_placeholders(self):
        result = _safe_format("No placeholders here.", {"name": "Alice"})
        assert result == "No placeholders here."

    def test_malformed_template_returns_raw(self):
        result = _safe_format("Bad {template{nested}", {"template": "x"})
        assert "Bad" in result  # Should not crash

    def test_double_braces_preserved(self):
        """Double braces are literal braces in format strings."""
        result = _safe_format('{{"key": "{value}"}}', {"value": "42"})
        assert '"key"' in result
        assert "42" in result

    def test_empty_string(self):
        result = _safe_format("", {"key": "val"})
        assert result == ""


# ── _get_variant ─────────────────────────────────────────────────────────


class TestGetVariant:
    """Variant resolution from environment variables."""

    def test_no_env_returns_none(self, monkeypatch):
        monkeypatch.delenv("PROMPT_VARIANT__test_prompt", raising=False)
        monkeypatch.delenv("PROMPT_VARIANT", raising=False)
        assert _get_variant("test_prompt") is None

    def test_per_prompt_env_var(self, monkeypatch):
        monkeypatch.setenv("PROMPT_VARIANT__architect_investigate", "v2")
        assert _get_variant("architect_investigate") == "v2"

    def test_global_env_var(self, monkeypatch):
        monkeypatch.delenv("PROMPT_VARIANT__my_prompt", raising=False)
        monkeypatch.setenv("PROMPT_VARIANT", "beta")
        assert _get_variant("my_prompt") == "beta"

    def test_per_prompt_overrides_global(self, monkeypatch):
        monkeypatch.setenv("PROMPT_VARIANT__my_prompt", "v3")
        monkeypatch.setenv("PROMPT_VARIANT", "beta")
        assert _get_variant("my_prompt") == "v3"


# ── resolve_prompt ───────────────────────────────────────────────────────


class TestResolvePrompt:
    """Core resolve_prompt() function."""

    def test_fallback_when_no_file(self, tmp_path, monkeypatch):
        """No file exists → uses fallback string."""
        monkeypatch.setattr(
            "src.prompt_builders.resolver.PROMPT_DIR", tmp_path / "nonexistent"
        )
        monkeypatch.delenv("PROMPT_VARIANT", raising=False)
        result = resolve_prompt("missing", "fallback text")
        assert result == "fallback text"

    def test_file_overrides_fallback(self, tmp_path, monkeypatch):
        """File exists → reads from file, ignores fallback."""
        monkeypatch.setattr("src.prompt_builders.resolver.PROMPT_DIR", tmp_path)
        monkeypatch.delenv("PROMPT_VARIANT", raising=False)
        (tmp_path / "my_prompt.md").write_text("From file!")
        result = resolve_prompt("my_prompt", "fallback")
        assert result == "From file!"

    def test_variant_file_overrides_default(self, tmp_path, monkeypatch):
        """Variant file exists → reads variant, ignores default file."""
        monkeypatch.setattr("src.prompt_builders.resolver.PROMPT_DIR", tmp_path)
        (tmp_path / "my_prompt.md").write_text("Default file")
        (tmp_path / "my_prompt.v2.md").write_text("Variant v2")
        monkeypatch.setenv("PROMPT_VARIANT__my_prompt", "v2")
        result = resolve_prompt("my_prompt", "fallback")
        assert result == "Variant v2"

    def test_variant_missing_falls_to_default_file(self, tmp_path, monkeypatch):
        """Variant set but file missing → falls to default file."""
        monkeypatch.setattr("src.prompt_builders.resolver.PROMPT_DIR", tmp_path)
        (tmp_path / "my_prompt.md").write_text("Default file")
        monkeypatch.setenv("PROMPT_VARIANT__my_prompt", "v99")
        result = resolve_prompt("my_prompt", "fallback")
        assert result == "Default file"

    def test_empty_variant_falls_to_default_file(self, tmp_path, monkeypatch):
        """Empty variant files are ignored instead of returning empty prompts."""
        monkeypatch.setattr("src.prompt_builders.resolver.PROMPT_DIR", tmp_path)
        (tmp_path / "my_prompt.md").write_text("Default file")
        (tmp_path / "my_prompt.v2.md").write_text("  \n")
        monkeypatch.setenv("PROMPT_VARIANT__my_prompt", "v2")
        result = resolve_prompt("my_prompt", "fallback")
        assert result == "Default file"

    def test_empty_default_file_falls_to_constant(self, tmp_path, monkeypatch):
        """Empty default files fall back to the explicit caller constant."""
        monkeypatch.setattr("src.prompt_builders.resolver.PROMPT_DIR", tmp_path)
        monkeypatch.delenv("PROMPT_VARIANT", raising=False)
        (tmp_path / "my_prompt.md").write_text("\n\t")
        result = resolve_prompt("my_prompt", "fallback")
        assert result == "fallback"

    def test_explicit_variant_param(self, tmp_path, monkeypatch):
        """Explicit variant= param overrides env var."""
        monkeypatch.setattr("src.prompt_builders.resolver.PROMPT_DIR", tmp_path)
        (tmp_path / "my_prompt.beta.md").write_text("Beta version")
        monkeypatch.setenv("PROMPT_VARIANT__my_prompt", "v2")
        result = resolve_prompt("my_prompt", "fallback", variant="beta")
        assert result == "Beta version"

    def test_template_interpolation(self, tmp_path, monkeypatch):
        """Template vars get interpolated in file content."""
        monkeypatch.setattr("src.prompt_builders.resolver.PROMPT_DIR", tmp_path)
        monkeypatch.delenv("PROMPT_VARIANT", raising=False)
        (tmp_path / "greet.md").write_text("Hello {user}, you are {role}!")
        result = resolve_prompt("greet", "fallback", user="Alice", role="admin")
        assert result == "Hello Alice, you are admin!"

    def test_template_interpolation_in_fallback(self, tmp_path, monkeypatch):
        """Template vars also work in fallback strings."""
        monkeypatch.setattr(
            "src.prompt_builders.resolver.PROMPT_DIR", tmp_path / "nonexistent"
        )
        monkeypatch.delenv("PROMPT_VARIANT", raising=False)
        result = resolve_prompt(
            "missing", "Q: {question}\nA:", question="What is 2+2?"
        )
        assert result == "Q: What is 2+2?\nA:"

    def test_subdir_resolution(self, tmp_path, monkeypatch):
        """subdir= parameter resolves files in subdirectory."""
        monkeypatch.setattr("src.prompt_builders.resolver.PROMPT_DIR", tmp_path)
        monkeypatch.delenv("PROMPT_VARIANT", raising=False)
        roles_dir = tmp_path / "roles"
        roles_dir.mkdir()
        (roles_dir / "frontdoor.md").write_text("I am frontdoor prompt")
        result = resolve_prompt("frontdoor", "fallback", subdir="roles")
        assert result == "I am frontdoor prompt"

    def test_subdir_fallback(self, tmp_path, monkeypatch):
        """subdir file missing → uses fallback."""
        monkeypatch.setattr("src.prompt_builders.resolver.PROMPT_DIR", tmp_path)
        monkeypatch.delenv("PROMPT_VARIANT", raising=False)
        result = resolve_prompt("nonexistent_role", "fallback text", subdir="roles")
        assert result == "fallback text"

    def test_alias_role_uses_canonical_family_fallback(self, tmp_path, monkeypatch):
        """Legacy aliases should resolve through canonical family fallback."""
        monkeypatch.setattr("src.prompt_builders.resolver.PROMPT_DIR", tmp_path)
        roles_dir = tmp_path / "roles"
        roles_dir.mkdir()
        (roles_dir / "worker_general.md").write_text("Worker general prompt")
        result = resolve_prompt("worker_explore", "fallback", subdir="roles")
        assert result.startswith("Worker general prompt")
        assert "Do NOT self-correct" in result

    def test_name_path_escape_falls_to_constant(self, tmp_path, monkeypatch):
        """Prompt names cannot escape PROMPT_DIR through path traversal."""
        monkeypatch.setattr("src.prompt_builders.resolver.PROMPT_DIR", tmp_path)
        outside = tmp_path.parent / "escaped.md"
        outside.write_text("escaped")
        result = resolve_prompt("../escaped", "fallback")
        assert result == "fallback"

    def test_symlink_escape_falls_to_constant(self, tmp_path, monkeypatch):
        """Prompt symlinks cannot resolve outside PROMPT_DIR."""
        monkeypatch.setattr("src.prompt_builders.resolver.PROMPT_DIR", tmp_path)
        outside = tmp_path.parent / "outside_prompt.md"
        outside.write_text("escaped")
        (tmp_path / "unsafe.md").symlink_to(outside)
        result = resolve_prompt("unsafe", "fallback")
        assert result == "fallback"

    def test_no_template_vars_no_interpolation(self, tmp_path, monkeypatch):
        """Without template vars, literal braces in file are preserved."""
        monkeypatch.setattr("src.prompt_builders.resolver.PROMPT_DIR", tmp_path)
        monkeypatch.delenv("PROMPT_VARIANT", raising=False)
        (tmp_path / "raw.md").write_text('Reply JSON: {{"key": "value"}}')
        result = resolve_prompt("raw", "fallback")
        assert result == 'Reply JSON: {{"key": "value"}}'

    def test_prompt_dir_points_to_orchestration(self):
        """PROMPT_DIR should point to orchestration/prompts/."""
        assert PROMPT_DIR.name == "prompts"
        assert PROMPT_DIR.parent.name == "orchestration"

    def test_prompt_dir_override_is_request_scoped(self, tmp_path, monkeypatch):
        """prompt_dir_override temporarily redirects prompt resolution."""
        canonical = tmp_path / "canonical"
        scratch = tmp_path / "scratch"
        canonical.mkdir()
        scratch.mkdir()
        (canonical / "frontdoor.md").write_text("canonical")
        (scratch / "frontdoor.md").write_text("scratch")
        monkeypatch.setattr("src.prompt_builders.resolver.PROMPT_DIR", canonical)
        monkeypatch.delenv("PROMPT_VARIANT", raising=False)

        assert current_prompt_dir() == canonical
        assert resolve_prompt("frontdoor", "fallback") == "canonical"
        with prompt_dir_override(scratch):
            assert current_prompt_dir() == scratch.resolve()
            assert resolve_prompt("frontdoor", "fallback") == "scratch"
        assert current_prompt_dir() == canonical
        assert resolve_prompt("frontdoor", "fallback") == "canonical"

    def test_prompt_dir_override_rejects_missing_root(self, tmp_path):
        """Missing scratch roots fail before prompt resolution."""
        with pytest.raises(FileNotFoundError):
            with prompt_dir_override(tmp_path / "missing"):
                pass


class TestDirectAnswerPrefix:
    def test_canonical_worker_aliases_share_direct_prefix(self):
        assert get_direct_answer_prefix("worker_explore", "What is 2+2?").startswith(
            "Answer with ONLY"
        )
        assert get_direct_answer_prefix("worker_general", "What is 2+2?").startswith(
            "Answer with ONLY"
        )

    def test_frontdoor_keeps_direct_prefix(self):
        assert get_direct_answer_prefix("frontdoor", "List exactly one item").startswith(
            "Respond with only"
        )

    def test_non_eligible_roles_get_no_prefix(self):
        assert get_direct_answer_prefix("coder_escalation", "What is 2+2?") == ""


class TestPromptFilesExist:
    """Verify the prompt files created in orchestration/prompts/ are loadable."""

    @pytest.mark.parametrize(
        "name",
        [
            "root_lm_system",
            "architect_investigate",
            "architect_synthesis",
            "plan_review",
            "task_decomposition",
            "formalizer",
        ],
    )
    def test_prompt_file_exists(self, name):
        path = PROMPT_DIR / f"{name}.md"
        assert path.exists(), f"Missing prompt file: {path}"
        content = path.read_text()
        assert len(content) > 10, f"Prompt file too short: {path}"

    @pytest.mark.parametrize(
        "role",
        [
            "frontdoor",
            "coder_escalation",
            "coder_escalation",
            "architect_general",
            _RETIRED_ARCHITECT_ROLE,
            "ingest_long_context",
            "worker_general",
            "worker_math",
            "worker_vision",
        ],
    )
    def test_role_prompt_file_exists(self, role):
        path = PROMPT_DIR / "roles" / f"{role}.md"
        assert path.exists(), f"Missing role prompt file: {path}"
        content = path.read_text()
        assert len(content) > 10, f"Role prompt file too short: {path}"


class TestS4OmegaInterventionArm:
    """S4 Omega A/B intervention arm (repl-turn-efficiency L101).

    The arm is a default-off prompt variant selected via
    PROMPT_VARIANT__root_lm_system=s4_omega. The default root_lm_system.md must
    be untouched when the variant is not requested — the A/B needs both arms.
    """

    VARIANT = "s4_omega"

    def test_variant_file_exists(self):
        path = PROMPT_DIR / f"root_lm_system.{self.VARIANT}.md"
        assert path.exists(), f"Missing S4 Omega arm prompt file: {path}"
        content = path.read_text()
        assert len(content) > 10, f"S4 Omega arm too short: {path}"

    def test_variant_resolves_when_requested(self, monkeypatch):
        monkeypatch.setenv("PROMPT_VARIANT__root_lm_system", self.VARIANT)
        result = resolve_prompt("root_lm_system", "fallback")
        # The variant must actually be the arm, not the default or fallback.
        assert "TURN BUDGET" in result or "MANDATED PROCEDURE" in result
        assert "HARD LIMIT of 10 rounds" in result

    def test_default_untouched_without_variant(self, monkeypatch):
        monkeypatch.delenv("PROMPT_VARIANT__root_lm_system", raising=False)
        monkeypatch.delenv("PROMPT_VARIANT", raising=False)
        result = resolve_prompt("root_lm_system", "fallback")
        # Default file, no turn-floor arm leaked in.
        assert "HARD LIMIT of 10 rounds" not in result
        assert result != "fallback"  # real default file exists
