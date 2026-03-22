# tests/test_entity_resolver.py

"""
Tests for EntityResolver (post-compilation transform — Phase 9C).

Tests cover:
- Direct resolution (app_id, display name)
- Alias resolution ("browser" → "chrome")
- Ambiguity handling (never silent)
- Not-found → structured error
- resolve_plan() transform (adds app_id alongside app_name)
- Multi-node plans (only entity-param nodes processed)
- IRReference values skipped (runtime-resolved)
- Structured error messages (user-facing clarification)
"""

import pytest
from unittest.mock import MagicMock

from infrastructure.application_registry import (
    ApplicationEntity,
    ApplicationRegistry,
    LaunchStrategy,
    ResolutionMethod,
)
from cortex.entity_resolver import (
    EntityResolver,
    EntityResolutionError,
    ResolutionResult,
    ResolutionType,
)
from ir.mission import (
    MissionPlan,
    MissionNode,
    OutputReference,
    IR_VERSION,
    ExecutionMode,
)
from skills.contract import SkillContract, FailurePolicy
from skills.base import Skill
from skills.skill_result import SkillResult
from execution.registry import SkillRegistry


# ─────────────────────────────────────────────────────────────
# Test fixtures
# ─────────────────────────────────────────────────────────────

class StubOpenAppSkill(Skill):
    contract = SkillContract(
        name="system.open_app",
        action="open_app",
        target_type="app",
        description="Open an application",
        inputs={"app_name": "application_name"},
        entity_params=["app_name"],
        outputs={"opened": "application_name"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
    )
    def execute(self, inputs, world, snapshot=None):
        return SkillResult(outputs={"opened": inputs["app_name"]})


class StubCreateFolderSkill(Skill):
    contract = SkillContract(
        name="fs.create_folder",
        action="create_folder",
        target_type="folder",
        description="Create a folder",
        inputs={"name": "folder_name"},
        outputs={},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
    )
    def execute(self, inputs, world, snapshot=None):
        return SkillResult(outputs={})


class StubCloseAppSkill(Skill):
    contract = SkillContract(
        name="system.close_app",
        action="close_app",
        target_type="app",
        description="Close an application",
        inputs={"app_name": "application_name"},
        entity_params=["app_name"],
        outputs={"closed": "application_name"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
    )
    def execute(self, inputs, world, snapshot=None):
        return SkillResult(outputs={"closed": inputs["app_name"]})


@pytest.fixture
def app_registry():
    """Registry with common test entities."""
    reg = ApplicationRegistry()
    reg.register(ApplicationEntity(
        app_id="chrome",
        display_names=["Google Chrome", "Chrome"],
        canonical_process_names=["chrome.exe"],
        launch_strategies=[
            LaunchStrategy(type="executable", value="C:\\chrome.exe",
                           method=ResolutionMethod.APP_PATHS, priority=100),
        ],
    ))
    reg.register(ApplicationEntity(
        app_id="spotify",
        display_names=["Spotify"],
        canonical_process_names=["spotify.exe"],
        protocols=["spotify:"],
    ))
    reg.register(ApplicationEntity(
        app_id="vscode",
        display_names=["Visual Studio Code", "VS Code"],
        canonical_process_names=["Code.exe"],
    ))
    reg.register(ApplicationEntity(
        app_id="chromium",
        display_names=["Chromium"],
        canonical_process_names=["chromium.exe"],
    ))
    reg.register(ApplicationEntity(
        app_id="notepad",
        display_names=["Notepad"],
        canonical_process_names=["notepad.exe"],
    ))
    return reg


@pytest.fixture
def skill_registry():
    """SkillRegistry with test stubs."""
    reg = SkillRegistry()
    reg.register(StubOpenAppSkill())
    reg.register(StubCreateFolderSkill())
    reg.register(StubCloseAppSkill())
    return reg


@pytest.fixture
def aliases():
    return {
        "browser": "chrome",
        "music": "spotify",
        "text editor": "notepad",
        "ide": "vscode",
    }


@pytest.fixture
def resolver(app_registry, skill_registry, aliases):
    return EntityResolver(
        registry=app_registry,
        skill_registry=skill_registry,
        alias_map=aliases,
    )


def _make_plan(*nodes):
    """Helper to build a test MissionPlan."""
    return MissionPlan(
        id="test_plan",
        nodes=list(nodes),
        metadata={"ir_version": IR_VERSION},
    )


def _open_node(app_name, node_id="node_0"):
    """Helper to build a system.open_app node."""
    return MissionNode(
        id=node_id,
        skill="system.open_app",
        inputs={"app_name": app_name},
    )


def _close_node(app_name, node_id="node_0"):
    return MissionNode(
        id=node_id,
        skill="system.close_app",
        inputs={"app_name": app_name},
    )


def _folder_node(name, node_id="node_0"):
    return MissionNode(
        id=node_id,
        skill="fs.create_folder",
        inputs={"name": name},
    )


# ─────────────────────────────────────────────────────────────
# Direct resolution (resolve() method)
# ─────────────────────────────────────────────────────────────

class TestDirectResolution:

    def test_resolve_by_app_id(self, resolver):
        result = resolver.resolve("chrome")
        assert result.type == ResolutionType.RESOLVED
        assert result.app_id == "chrome"
        assert result.score == 1.0

    def test_resolve_by_display_name(self, resolver):
        result = resolver.resolve("Google Chrome")
        assert result.type == ResolutionType.RESOLVED
        assert result.app_id == "chrome"

    def test_resolve_case_insensitive(self, resolver):
        result = resolver.resolve("SPOTIFY")
        assert result.type == ResolutionType.RESOLVED
        assert result.app_id == "spotify"

    def test_resolve_with_whitespace(self, resolver):
        result = resolver.resolve("  spotify  ")
        assert result.type == ResolutionType.RESOLVED
        assert result.app_id == "spotify"

    def test_resolve_not_found(self, resolver):
        result = resolver.resolve("nonexistent_app")
        assert result.type == ResolutionType.NOT_FOUND
        assert result.app_id is None

    def test_resolve_empty_string(self, resolver):
        result = resolver.resolve("")
        assert result.type == ResolutionType.NOT_FOUND

    def test_result_preserves_original_term(self, resolver):
        result = resolver.resolve("Spotify")
        assert result.term == "Spotify"


# ─────────────────────────────────────────────────────────────
# Alias resolution
# ─────────────────────────────────────────────────────────────

class TestAliasResolution:

    def test_alias_browser(self, resolver):
        result = resolver.resolve("browser")
        assert result.type == ResolutionType.RESOLVED
        assert result.app_id == "chrome"
        assert result.score == 0.9  # Alias score

    def test_alias_music(self, resolver):
        result = resolver.resolve("music")
        assert result.type == ResolutionType.RESOLVED
        assert result.app_id == "spotify"

    def test_alias_ide(self, resolver):
        result = resolver.resolve("ide")
        assert result.type == ResolutionType.RESOLVED
        assert result.app_id == "vscode"

    def test_alias_case_insensitive(self, resolver):
        result = resolver.resolve("Browser")
        assert result.type == ResolutionType.RESOLVED
        assert result.app_id == "chrome"


# ─────────────────────────────────────────────────────────────
# Ambiguity handling
# ─────────────────────────────────────────────────────────────

class TestAmbiguityHandling:

    def test_exact_match_not_ambiguous(self, resolver):
        """'chrome' exactly matches app_id — should NOT be ambiguous."""
        result = resolver.resolve("chrome")
        assert result.type == ResolutionType.RESOLVED
        assert result.app_id == "chrome"


# ─────────────────────────────────────────────────────────────
# resolve_plan() — Post-compilation transform
# ─────────────────────────────────────────────────────────────

class TestResolvePlan:
    """Test resolve_plan() adds app_id alongside app_name."""

    def test_resolved_adds_app_id(self, resolver):
        """Resolved app adds app_id to inputs without overwriting app_name."""
        plan = _make_plan(_open_node("spotify"))
        resolved = resolver.resolve_plan(plan)
        inputs = resolved.nodes[0].inputs

        assert inputs["app_name"] == "spotify"  # preserved
        assert inputs["app_id"] == "spotify"     # added

    def test_original_plan_not_mutated(self, resolver):
        """resolve_plan never mutates the original."""
        plan = _make_plan(_open_node("chrome"))
        original_inputs = dict(plan.nodes[0].inputs)
        resolver.resolve_plan(plan)
        assert plan.nodes[0].inputs == original_inputs

    def test_display_name_resolves(self, resolver):
        """Display name 'Google Chrome' resolves to app_id 'chrome'."""
        plan = _make_plan(_open_node("Google Chrome"))
        resolved = resolver.resolve_plan(plan)
        assert resolved.nodes[0].inputs["app_id"] == "chrome"
        assert resolved.nodes[0].inputs["app_name"] == "Google Chrome"

    def test_alias_resolves_in_plan(self, resolver):
        """Alias 'browser' resolves to app_id 'chrome' in plan."""
        plan = _make_plan(_open_node("browser"))
        resolved = resolver.resolve_plan(plan)
        assert resolved.nodes[0].inputs["app_id"] == "chrome"
        assert resolved.nodes[0].inputs["app_name"] == "browser"

    def test_not_found_raises_error(self, resolver):
        """Not-found app raises EntityResolutionError."""
        plan = _make_plan(_open_node("nonexistent"))
        with pytest.raises(EntityResolutionError) as exc_info:
            resolver.resolve_plan(plan)
        msg = exc_info.value.user_message()
        assert "nonexistent" in msg

    def test_near_match_resolves_fuzzy(self, resolver):
        """Near-match 'spotif' resolves via fuzzy (substring)."""
        plan = _make_plan(_open_node("spotif"))
        resolved = resolver.resolve_plan(plan)
        assert resolved.nodes[0].inputs["app_id"] == "spotify"
        assert resolved.nodes[0].inputs["app_name"] == "spotif"

    def test_close_app_resolved(self, resolver):
        """close_app skill also resolves entity params."""
        plan = _make_plan(_close_node("chrome"))
        resolved = resolver.resolve_plan(plan)
        assert resolved.nodes[0].inputs["app_id"] == "chrome"


# ─────────────────────────────────────────────────────────────
# Multi-node plans
# ─────────────────────────────────────────────────────────────

class TestMultiNodePlans:
    """Only entity-param nodes are processed."""

    def test_mixed_plan_only_app_nodes_resolved(self, resolver):
        """Only system.open_app gets app_id, fs.create_folder untouched."""
        plan = _make_plan(
            _open_node("chrome", "node_0"),
            _folder_node("logs", "node_1"),
        )
        resolved = resolver.resolve_plan(plan)

        # open_app node has app_id
        assert resolved.nodes[0].inputs["app_id"] == "chrome"
        # create_folder node untouched
        assert "app_id" not in resolved.nodes[1].inputs
        assert resolved.nodes[1].inputs["name"] == "logs"

    def test_multiple_app_nodes_all_resolved(self, resolver):
        """Multiple app nodes all get resolved."""
        plan = _make_plan(
            _open_node("spotify", "node_0"),
            _close_node("chrome", "node_1"),
        )
        resolved = resolver.resolve_plan(plan)
        assert resolved.nodes[0].inputs["app_id"] == "spotify"
        assert resolved.nodes[1].inputs["app_id"] == "chrome"


# ─────────────────────────────────────────────────────────────
# IRReference skipping
# ─────────────────────────────────────────────────────────────

class TestIRReferenceSkipping:
    """IRReference values are runtime-resolved — skip them."""

    def test_output_reference_skipped(self, resolver):
        """OutputReference in app_name is skipped (runtime-resolved pipe)."""
        ref = OutputReference(node="node_0", output="apps", index=1, field="name")
        node = MissionNode(
            id="node_1",
            skill="system.open_app",
            inputs={"app_name": ref},
            depends_on=["node_0"],
        )
        plan = _make_plan(node)
        resolved = resolver.resolve_plan(plan)
        # Should pass through unchanged — no app_id added
        assert isinstance(resolved.nodes[0].inputs["app_name"], OutputReference)
        assert "app_id" not in resolved.nodes[0].inputs


# ─────────────────────────────────────────────────────────────
# Structured error messages
# ─────────────────────────────────────────────────────────────

class TestErrorMessages:

    def test_not_found_user_message(self, resolver):
        """Not-found error includes the original term."""
        plan = _make_plan(_open_node("xyzabc"))
        with pytest.raises(EntityResolutionError) as exc_info:
            resolver.resolve_plan(plan)
        msg = exc_info.value.user_message()
        assert "xyzabc" in msg
        assert "couldn't find" in msg.lower()

    def test_error_str(self, resolver):
        """Error __str__ includes structured violation info."""
        plan = _make_plan(_open_node("xyzabc"))
        with pytest.raises(EntityResolutionError) as exc_info:
            resolver.resolve_plan(plan)
        s = str(exc_info.value)
        assert "xyzabc" in s
        assert "not found" in s.lower() or "not_found" in s.lower()


# ─────────────────────────────────────────────────────────────
# Phase 9E: File entity resolution
# ─────────────────────────────────────────────────────────────

class StubReadFileSkill(Skill):
    """Stub for fs.read_file — has file_path_input semantic type."""
    contract = SkillContract(
        name="fs.read_file",
        action="read_file",
        target_type="file",
        description="Read a file",
        inputs={"path": "file_path_input"},
        optional_inputs={"anchor": "anchor_name"},
        outputs={"content": "file_content"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
    )
    def execute(self, inputs, world, snapshot=None):
        return SkillResult(outputs={"content": ""})


def _read_node(path, node_id="node_0"):
    return MissionNode(id=node_id, skill="fs.read_file", inputs={"path": path})


def _make_file_resolver(file_index=None, location_config=None):
    """Build a resolver with file entity resolution support."""
    app_reg = ApplicationRegistry()
    skill_reg = SkillRegistry()
    skill_reg.register(StubReadFileSkill())
    skill_reg.register(StubOpenAppSkill())
    return EntityResolver(
        registry=app_reg,
        skill_registry=skill_reg,
        file_index=file_index,
        location_config=location_config,
    )


def _mock_file_index(matches):
    """Build a mock FileIndex that returns given matches."""
    from world.file_ref import FileRef
    idx = MagicMock()
    idx.__bool__ = lambda self: True
    refs = [
        FileRef(
            ref_id=m["ref_id"],
            name=m["name"],
            anchor=m.get("anchor", "WORKSPACE"),
            relative_path=m["relative_path"],
            size_bytes=m.get("size_bytes", 1024),
            confidence=m.get("confidence", 1.0),
        )
        for m in matches
    ]
    idx.search.return_value = refs
    return idx


class TestFileEntityResolution:
    """Phase 9E: bare file names → anchor-qualified paths via FileIndex."""

    def test_single_match_replaces_path(self):
        """1 match → path replaced with relative_path, metadata stored."""
        fi = _mock_file_index([{
            "ref_id": "fref_abc123",
            "name": "resume.pdf",
            "anchor": "WORKSPACE",
            "relative_path": "documents/resume.pdf",
            "confidence": 1.0,
        }])
        resolver = _make_file_resolver(file_index=fi)
        plan = _make_plan(_read_node("resume"))
        resolved = resolver.resolve_plan(plan)
        inputs = resolved.nodes[0].inputs
        assert inputs["path"] == "documents/resume.pdf"
        assert inputs["_resolved_file_ref_id"] == "fref_abc123"
        assert inputs["_resolved_file_ref"]["name"] == "resume.pdf"

    def test_single_match_sets_anchor(self):
        """Non-WORKSPACE anchor → anchor set in inputs."""
        fi = _mock_file_index([{
            "ref_id": "fref_desk01",
            "name": "report.docx",
            "anchor": "DESKTOP",
            "relative_path": "report.docx",
            "confidence": 1.0,
        }])
        resolver = _make_file_resolver(file_index=fi)
        plan = _make_plan(_read_node("report"))
        resolved = resolver.resolve_plan(plan)
        inputs = resolved.nodes[0].inputs
        assert inputs["path"] == "report.docx"
        assert inputs["anchor"] == "DESKTOP"

    def test_ambiguous_raises_with_options(self):
        """N matches with close confidence → EntityResolutionError with options."""
        fi = _mock_file_index([
            {"ref_id": "fref_a", "name": "notes.txt", "relative_path": "a/notes.txt", "confidence": 0.8},
            {"ref_id": "fref_b", "name": "notes.md", "relative_path": "b/notes.md", "confidence": 0.8},
        ])
        resolver = _make_file_resolver(file_index=fi)
        plan = _make_plan(_read_node("notes"))
        with pytest.raises(EntityResolutionError) as exc_info:
            resolver.resolve_plan(plan)
        violations = exc_info.value.violations
        assert len(violations) == 1
        assert violations[0].resolution_type == "ambiguous_file"
        assert len(violations[0].options) == 2
        assert violations[0].options[0]["ref_id"] == "fref_a"
        # User message includes file options
        msg = exc_info.value.user_message()
        assert "notes" in msg.lower()

    def test_not_found_raises(self):
        """0 matches → EntityResolutionError with not_found_file."""
        fi = _mock_file_index([])
        resolver = _make_file_resolver(file_index=fi)
        plan = _make_plan(_read_node("nonexistent"))
        with pytest.raises(EntityResolutionError) as exc_info:
            resolver.resolve_plan(plan)
        v = exc_info.value.violations[0]
        assert v.resolution_type == "not_found_file"
        assert v.raw_value == "nonexistent"
        msg = exc_info.value.user_message()
        assert "nonexistent" in msg

    def test_explicit_path_skipped(self):
        """Values with / or \\ pass through unchanged (explicit paths)."""
        fi = _mock_file_index([])
        resolver = _make_file_resolver(file_index=fi)
        plan = _make_plan(_read_node("docs/report.txt"))
        resolved = resolver.resolve_plan(plan)
        # Should NOT call file_index.search — path has separator
        assert resolved.nodes[0].inputs["path"] == "docs/report.txt"
        fi.search.assert_not_called()

    def test_ir_reference_skipped(self):
        """IRReference values pass through unchanged (runtime pipes)."""
        fi = _mock_file_index([])
        resolver = _make_file_resolver(file_index=fi)
        ref = OutputReference(node="search_1", output="matches", index=0, field="relative_path")
        node = MissionNode(
            id="node_0",
            skill="fs.read_file",
            inputs={"path": ref},
            depends_on=["search_1"],
        )
        plan = _make_plan(node)
        resolved = resolver.resolve_plan(plan)
        assert isinstance(resolved.nodes[0].inputs["path"], OutputReference)
        fi.search.assert_not_called()

    def test_no_file_index_defers(self):
        """No file_index → Phase 9E is a no-op (defers to recovery)."""
        resolver = _make_file_resolver(file_index=None)
        plan = _make_plan(_read_node("resume"))
        resolved = resolver.resolve_plan(plan)
        # Should pass through unchanged — no violations
        assert resolved.nodes[0].inputs["path"] == "resume"
