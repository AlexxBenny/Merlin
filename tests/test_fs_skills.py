# tests/test_fs_skills.py

"""
Tests for fs.write_file and fs.read_file skills.

Covers:
- Contract validation (semantic types, action namespace, domain)
- Write execution (content written, events emitted, outputs correct)
- Read execution (content returned, missing file error, non-file error)
- Type-aware report builder (long text truncation, generated content block)
- Mixed-mode chain: generate_text → write_file via OutputReference
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from skills.fs.write_file import WriteFileSkill
from skills.fs.read_file import ReadFileSkill
from skills.contract import SkillContract
from cortex.semantic_types import SEMANTIC_TYPES, assert_types_registered


# ──────────────────────────────────────────────
# Contract validation
# ──────────────────────────────────────────────

class TestWriteFileContract(unittest.TestCase):
    """Contract compliance for fs.write_file."""

    def test_contract_is_skill_contract(self):
        assert isinstance(WriteFileSkill.contract, SkillContract)

    def test_name(self):
        assert WriteFileSkill.contract.name == "fs.write_file"

    def test_action(self):
        assert WriteFileSkill.contract.action == "write_file"

    def test_domain(self):
        assert WriteFileSkill.contract.domain == "fs"

    def test_action_namespaced(self):
        """Action must be prefixed by domain."""
        name = WriteFileSkill.contract.name
        domain = WriteFileSkill.contract.domain
        assert name.startswith(f"{domain}.")

    def test_inputs(self):
        inputs = WriteFileSkill.contract.inputs
        assert "path" in inputs
        assert "content" in inputs

    def test_outputs(self):
        outputs = WriteFileSkill.contract.outputs
        assert "written" in outputs

    def test_semantic_types_registered(self):
        assert_types_registered(
            WriteFileSkill.contract.name,
            WriteFileSkill.contract.inputs,
            WriteFileSkill.contract.outputs,
        )

    def test_mutates_world(self):
        assert WriteFileSkill.contract.mutates_world is True

    def test_emits_events(self):
        assert "file_written" in WriteFileSkill.contract.emits_events


class TestReadFileContract(unittest.TestCase):
    """Contract compliance for fs.read_file."""

    def test_contract_is_skill_contract(self):
        assert isinstance(ReadFileSkill.contract, SkillContract)

    def test_name(self):
        assert ReadFileSkill.contract.name == "fs.read_file"

    def test_action(self):
        assert ReadFileSkill.contract.action == "read_file"

    def test_domain(self):
        assert ReadFileSkill.contract.domain == "fs"

    def test_inputs(self):
        inputs = ReadFileSkill.contract.inputs
        assert "path" in inputs

    def test_outputs(self):
        outputs = ReadFileSkill.contract.outputs
        assert "content" in outputs

    def test_semantic_types_registered(self):
        assert_types_registered(
            ReadFileSkill.contract.name,
            ReadFileSkill.contract.inputs,
            ReadFileSkill.contract.outputs,
        )

    def test_does_not_mutate_world(self):
        assert ReadFileSkill.contract.mutates_world is False


# ──────────────────────────────────────────────
# Execution — write_file
# ──────────────────────────────────────────────

class TestWriteFileExecution(unittest.TestCase):
    """Write file execution logic."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.loc_config = MagicMock()
        self.loc_config.resolve.return_value = Path(self.tmp_dir)
        self.timeline = MagicMock()
        self.skill = WriteFileSkill(location_config=self.loc_config)

    def test_writes_content_to_file(self):
        result = self.skill.execute(
            {"path": "test.txt", "content": "Hello, world!"},
            self.timeline,
        )
        written_path = Path(self.tmp_dir) / "test.txt"
        assert written_path.exists()
        assert written_path.read_text(encoding="utf-8") == "Hello, world!"

    def test_returns_written_path(self):
        result = self.skill.execute(
            {"path": "out.txt", "content": "data"},
            self.timeline,
        )
        assert "written" in result.outputs
        assert "out.txt" in result.outputs["written"]

    def test_creates_parent_directories(self):
        result = self.skill.execute(
            {"path": "sub/dir/file.txt", "content": "nested"},
            self.timeline,
        )
        written_path = Path(self.tmp_dir) / "sub" / "dir" / "file.txt"
        assert written_path.exists()
        assert written_path.read_text(encoding="utf-8") == "nested"

    def test_emits_event(self):
        self.skill.execute(
            {"path": "test.txt", "content": "Hello"},
            self.timeline,
        )
        self.timeline.emit.assert_called_once()
        args = self.timeline.emit.call_args
        assert args[0][0] == "skill.fs"
        assert args[0][1] == "file_written"

    def test_overwrites_existing_file(self):
        (Path(self.tmp_dir) / "exist.txt").write_text("old")
        self.skill.execute(
            {"path": "exist.txt", "content": "new"},
            self.timeline,
        )
        assert (Path(self.tmp_dir) / "exist.txt").read_text() == "new"

    def test_uses_anchor(self):
        self.skill.execute(
            {"path": "test.txt", "content": "data", "anchor": "DESKTOP"},
            self.timeline,
        )
        self.loc_config.resolve.assert_called_with("DESKTOP")

    def test_default_anchor_is_workspace(self):
        self.skill.execute(
            {"path": "test.txt", "content": "data"},
            self.timeline,
        )
        self.loc_config.resolve.assert_called_with("WORKSPACE")


# ──────────────────────────────────────────────
# Execution — read_file
# ──────────────────────────────────────────────

class TestReadFileExecution(unittest.TestCase):
    """Read file execution logic."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.loc_config = MagicMock()
        self.loc_config.resolve.return_value = Path(self.tmp_dir)
        self.timeline = MagicMock()
        self.skill = ReadFileSkill(location_config=self.loc_config)

    def test_reads_file_content(self):
        (Path(self.tmp_dir) / "hello.txt").write_text("Hello!", encoding="utf-8")
        result = self.skill.execute(
            {"path": "hello.txt"},
            self.timeline,
        )
        assert result.outputs["content"] == "Hello!"

    def test_raises_for_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            self.skill.execute(
                {"path": "nonexistent.txt"},
                self.timeline,
            )

    def test_raises_for_directory(self):
        (Path(self.tmp_dir) / "subdir").mkdir()
        with self.assertRaises(ValueError):
            self.skill.execute(
                {"path": "subdir"},
                self.timeline,
            )

    def test_emits_event_on_success(self):
        (Path(self.tmp_dir) / "data.txt").write_text("content")
        self.skill.execute(
            {"path": "data.txt"},
            self.timeline,
        )
        self.timeline.emit.assert_called_once()
        args = self.timeline.emit.call_args
        assert args[0][1] == "file_read"

    def test_no_event_on_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            self.skill.execute(
                {"path": "missing.txt"},
                self.timeline,
            )
        self.timeline.emit.assert_not_called()


# ──────────────────────────────────────────────
# Type-aware report builder
# ──────────────────────────────────────────────

class TestTypeAwareReporting(unittest.TestCase):
    """Report builder truncates long outputs and adds content blocks."""

    def test_short_output_not_truncated(self):
        from reporting.report_builder import ReportBuilder, ActionRecord
        from execution.executor import NodeStatus

        action = ActionRecord(
            node_id="n0",
            skill="fs.write_file",
            status=NodeStatus.COMPLETED.value,
            inputs={"path": "test.txt"},
            outputs={"written": "/tmp/test.txt"},
        )
        desc = ReportBuilder._describe_action(action)
        assert '→ written="/tmp/test.txt"' in desc

    def test_long_output_truncated(self):
        from reporting.report_builder import ReportBuilder, ActionRecord
        from execution.executor import NodeStatus

        long_text = "A" * 500
        action = ActionRecord(
            node_id="n0",
            skill="reasoning.generate_text",
            status=NodeStatus.COMPLETED.value,
            inputs={"prompt": "tell me a story"},
            outputs={"text": long_text},
        )
        desc = ReportBuilder._describe_action(action)
        assert "..." in desc
        assert "500 chars" in desc
        # Should NOT contain full 500-char string
        assert ("A" * 200) not in desc

    def test_generated_content_block_in_prompt(self):
        from reporting.report_builder import ReportBuilder, StructuredReport, ReportType, ActionRecord
        from execution.executor import NodeStatus
        from conversation.frame import ConversationFrame

        long_text = "Once upon a time " * 50  # ~850 chars
        report = StructuredReport(
            report_type=ReportType.SUCCESS,
            actions=[ActionRecord(
                node_id="n0",
                skill="reasoning.generate_text",
                status=NodeStatus.COMPLETED.value,
                inputs={"prompt": "story"},
                outputs={"text": long_text},
            )],
            user_query="tell me a story",
            mission_id="test_001",
        )

        builder = ReportBuilder()
        prompt = builder._build_llm_prompt(
            report, ConversationFrame(),
        )
        assert "Generated content" in prompt
        assert "present this to the user" in prompt
        assert long_text in prompt  # Full content in dedicated block

    def test_no_content_block_for_short_outputs(self):
        from reporting.report_builder import ReportBuilder, StructuredReport, ReportType, ActionRecord
        from execution.executor import NodeStatus
        from conversation.frame import ConversationFrame

        report = StructuredReport(
            report_type=ReportType.SUCCESS,
            actions=[ActionRecord(
                node_id="n0",
                skill="fs.create_folder",
                status=NodeStatus.COMPLETED.value,
                inputs={"name": "projects"},
                outputs={"created": "/tmp/projects"},
            )],
            user_query="create folder projects",
            mission_id="test_002",
        )

        builder = ReportBuilder()
        prompt = builder._build_llm_prompt(
            report, ConversationFrame(),
        )
        assert "Generated content" not in prompt


# ──────────────────────────────────────────────
# Mixed-mode chain: generate → write (via $ref)
# ──────────────────────────────────────────────

class TestMixedModeChain(unittest.TestCase):
    """End-to-end: generate_text output feeds into write_file input via OutputReference."""

    def test_generate_then_write_chain(self):
        """Simulate the executor resolving $ref from generate_text → write_file."""
        from ir.mission import MissionPlan, MissionNode, OutputReference, IR_VERSION

        # Build the plan: node_0 = generate_text, node_1 = write_file($ref node_0.text)
        plan = MissionPlan(
            id="chain_001",
            nodes=[
                MissionNode(
                    id="node_0",
                    skill="reasoning.generate_text",
                    inputs={"prompt": "write a haiku about rain"},
                ),
                MissionNode(
                    id="node_1",
                    skill="fs.write_file",
                    inputs={
                        "path": "haiku.txt",
                        "content": OutputReference(node="node_0", output="text"),
                    },
                    depends_on=["node_0"],
                ),
            ],
            metadata={"ir_version": IR_VERSION},
        )

        # Verify plan structure
        assert len(plan.nodes) == 2
        assert plan.nodes[1].depends_on == ["node_0"]

        # Verify the OutputReference
        ref = plan.nodes[1].inputs["content"]
        assert isinstance(ref, OutputReference)
        assert ref.node == "node_0"
        assert ref.output == "text"

    def test_write_file_with_resolved_content(self):
        """After $ref resolution, write_file receives plain string content."""
        tmp_dir = tempfile.mkdtemp()
        loc_config = MagicMock()
        loc_config.resolve.return_value = Path(tmp_dir)
        timeline = MagicMock()

        skill = WriteFileSkill(location_config=loc_config)
        # Simulate resolved $ref — content is now a plain string
        generated_poem = "Rain falls on leaves\nSilent drops on ancient stone\nNature breathes again"
        result = skill.execute(
            {"path": "haiku.txt", "content": generated_poem},
            timeline,
        )
        written = Path(tmp_dir) / "haiku.txt"
        assert written.exists()
        assert written.read_text(encoding="utf-8") == generated_poem
        assert result.outputs["written"] == str(written)

    def test_read_after_write_chain(self):
        """Write then read: content roundtrips correctly."""
        tmp_dir = tempfile.mkdtemp()
        loc_config = MagicMock()
        loc_config.resolve.return_value = Path(tmp_dir)
        timeline = MagicMock()

        content = "Hello from the chain!"

        # Write
        writer = WriteFileSkill(location_config=loc_config)
        writer.execute(
            {"path": "chain.txt", "content": content},
            timeline,
        )

        # Read
        reader = ReadFileSkill(location_config=loc_config)
        result = reader.execute(
            {"path": "chain.txt"},
            timeline,
        )
        assert result.outputs["content"] == content


# ──────────────────────────────────────────────
# Registry audit
# ──────────────────────────────────────────────

class TestRegistryAudit(unittest.TestCase):
    """Verify skills pass audit_action_namespace if registry is available."""

    def test_write_file_passes_audit(self):
        from execution.registry import SkillRegistry
        registry = SkillRegistry()

        loc_config = MagicMock()
        skill = WriteFileSkill(location_config=loc_config)
        registry.register(skill)

        # Should not raise
        registry.audit_action_namespace()

    def test_read_file_passes_audit(self):
        from execution.registry import SkillRegistry
        registry = SkillRegistry()

        loc_config = MagicMock()
        skill = ReadFileSkill(location_config=loc_config)
        registry.register(skill)

        registry.audit_action_namespace()


if __name__ == "__main__":
    unittest.main()
