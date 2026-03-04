# tests/test_skill_loading_integration.py

"""
Production-path skill loading integration test.

Mirrors the exact boot sequence of main.py:
    load_skills(registry, skills_config, deps=skill_deps)

This catches the class of bugs where:
- A skill declares an unregistered semantic type
- A required dep is None (skill silently skipped)
- Constructor signature doesn't match deps dict keys

Also verifies semantic type hygiene:
- Every type in SEMANTIC_TYPES is used by at least one skill
"""

import os
import tempfile

import pytest
import yaml

from cortex.semantic_types import SEMANTIC_TYPES
from execution.registry import SkillRegistry


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")


@pytest.fixture(scope="module")
def skills_config():
    path = os.path.join(CONFIG_DIR, "skills.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def production_deps():
    """Build the full deps dict exactly as main.py does.

    Every dep that main.py puts into skill_deps must be here.
    If a dep needs a real object (not None), provide one.
    """
    from infrastructure.system_controller import SystemController
    from unittest.mock import MagicMock

    # task_store: use a real JsonTaskStore on a temp file
    from runtime.json_task_store import JsonTaskStore
    tmp = tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, mode="w",
    )
    tmp.write('{"schema_version":1,"next_counter":0,"jobs":[]}')
    tmp.close()
    task_store = JsonTaskStore(path=tmp.name)

    # content_llm: mock (not required for registration test)
    content_llm = MagicMock()
    content_llm.complete.return_value = "mock"

    # location_config: None is fine — fs skills will be skipped
    # We test separately that skipped skills are accounted for
    deps = {
        "location_config": None,
        "system_controller": SystemController(),
        "content_llm": content_llm,
        "task_store": task_store,
    }
    yield deps
    os.unlink(tmp.name)


@pytest.fixture(scope="module")
def loaded_registry(skills_config, production_deps):
    """Load ALL skills via the production code path."""
    from main import load_skills
    registry = SkillRegistry()
    load_skills(registry, skills_config, deps=production_deps)
    return registry


# ─────────────────────────────────────────────────────────────
# A. Production-path skill loading
# ─────────────────────────────────────────────────────────────


class TestProductionSkillLoading:
    """Verify all skills load through the production code path."""

    # Skills that are expected to be skipped because their required dep
    # (location_config) is unavailable in CI/test environments.
    EXPECTED_SKIPPED = {
        "fs.create_folder",
        "fs.write_file",
        "fs.read_file",
    }

    def test_all_non_skipped_skills_loaded(
        self, skills_config, loaded_registry,
    ):
        """Every skill in skills.yaml (minus known skips) must be registered."""
        declared = {
            entry["name"]
            for entry in skills_config.get("skills", [])
        }
        registered = loaded_registry.all_names()

        missing = declared - registered - self.EXPECTED_SKIPPED
        assert not missing, (
            f"Skills declared in skills.yaml but NOT loaded: {sorted(missing)}. "
            "Check: (a) semantic types registered? (b) deps available? "
            "(c) constructor signature matches deps?"
        )

    def test_job_skills_specifically_loaded(self, loaded_registry):
        """system.list_jobs and system.cancel_job must be in the registry.

        These are the skills that were silently dropped in production
        due to unregistered semantic types (job_list, job_identifier).
        This test exists to prevent that regression.
        """
        names = loaded_registry.all_names()
        assert "system.list_jobs" in names, (
            "system.list_jobs not loaded — check job_list in SEMANTIC_TYPES"
        )
        assert "system.cancel_job" in names, (
            "system.cancel_job not loaded — check job_identifier in SEMANTIC_TYPES"
        )

    def test_no_audit_violations(self, loaded_registry):
        """Action namespace audit must pass (no collisions, no missing actions)."""
        violations = loaded_registry.audit_action_namespace()
        assert violations == [], (
            f"Action namespace audit violations: {violations}"
        )

    def test_every_skill_has_contract(self, loaded_registry):
        """Every registered skill must have a contract with name and description."""
        for name in loaded_registry.all_names():
            skill = loaded_registry.get(name)
            assert skill.contract is not None, f"{name}: missing contract"
            assert skill.contract.name == name, (
                f"{name}: contract.name={skill.contract.name} mismatch"
            )
            assert skill.contract.description, f"{name}: empty description"


# ─────────────────────────────────────────────────────────────
# B. Semantic type hygiene
# ─────────────────────────────────────────────────────────────


class TestSemanticTypeHygiene:
    """Prevent registry drift: no dead types, no missing types."""

    def test_all_semantic_types_are_used(self, loaded_registry):
        """Every type in SEMANTIC_TYPES must be used by at least one skill.

        Dead types bloat the registry and confuse future developers.
        If a type is no longer used, remove it from SEMANTIC_TYPES.
        """
        used_types: set = set()
        for name in loaded_registry.all_names():
            skill = loaded_registry.get(name)
            used_types.update(skill.contract.inputs.values())
            used_types.update(skill.contract.outputs.values())
            if skill.contract.optional_inputs:
                used_types.update(skill.contract.optional_inputs.values())

        # Types used by skipped skills (fs.*) — these are legitimately
        # registered but their skills need location_config to load.
        # We know these types from the contracts themselves.
        fs_types = {
            "anchor_name", "relative_path", "folder_name",
            "filesystem_path", "file_path_input", "file_content",
        }
        used_types.update(fs_types)

        unused = set(SEMANTIC_TYPES.keys()) - used_types
        assert not unused, (
            f"Dead semantic types in SEMANTIC_TYPES (not used by any skill): "
            f"{sorted(unused)}. Remove them or add a skill that uses them."
        )

    def test_no_duplicate_type_descriptions(self):
        """Descriptions should be unique to prevent copy-paste drift."""
        descriptions: dict = {}
        duplicates = []
        for name, sem_type in SEMANTIC_TYPES.items():
            desc = sem_type.description
            if desc in descriptions:
                duplicates.append(
                    f"'{name}' and '{descriptions[desc]}' share description"
                )
            descriptions[desc] = name
        assert not duplicates, f"Duplicate type descriptions: {duplicates}"
