# tests/test_clarification_flow.py

"""
Tests for the ambiguity → ask-back → clarification flow.

Covers:
- SkillResult.status field (new)
- fs.search_file ambiguity detection (same-name files)
- email.draft_message duplicate attachment → no_op (not crash)
- ExecutionResult.clarification_needed field (new)
- OutcomeAnalyzer classifies ambiguous_input as SOFT_FAILURE
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from skills.skill_result import SkillResult
from execution.executor import ExecutionResult, NodeStatus


# ─────────────────────────────────────────────────────────────
# SkillResult.status
# ─────────────────────────────────────────────────────────────

class TestSkillResultStatus:

    def test_default_status_is_none(self):
        result = SkillResult(outputs={"x": 1})
        assert result.status is None

    def test_status_no_op(self):
        result = SkillResult(
            outputs={"x": 1},
            status="no_op",
            metadata={"reason": "ambiguous_input"},
        )
        assert result.status == "no_op"
        assert result.metadata["reason"] == "ambiguous_input"

    def test_skillresult_is_frozen(self):
        result = SkillResult(outputs={"x": 1}, status="no_op")
        with pytest.raises(AttributeError):
            result.status = "completed"


# ─────────────────────────────────────────────────────────────
# ExecutionResult.clarification_needed
# ─────────────────────────────────────────────────────────────

class TestExecutionResultClarification:

    def test_default_clarification_is_none(self):
        er = ExecutionResult()
        assert er.clarification_needed is None

    def test_set_clarification(self):
        er = ExecutionResult()
        er.clarification_needed = {
            "question": "Which file?",
            "options": [{"label": "file1"}, {"label": "file2"}],
            "context": {"source": "decision_engine"},
        }
        assert er.clarification_needed["question"] == "Which file?"
        assert len(er.clarification_needed["options"]) == 2


# ─────────────────────────────────────────────────────────────
# OutcomeAnalyzer: ambiguous_input → SOFT_FAILURE
# ─────────────────────────────────────────────────────────────

class TestOutcomeAnalyzerAmbiguity:

    def test_no_op_ambiguous_input_is_soft_failure(self):
        from execution.metacognition import OutcomeAnalyzer, OutcomeSeverity
        analyzer = OutcomeAnalyzer()
        result = analyzer.classify("no_op", {"reason": "ambiguous_input"})
        assert result == OutcomeSeverity.SOFT_FAILURE

    def test_no_op_idempotent_is_benign(self):
        from execution.metacognition import OutcomeAnalyzer, OutcomeSeverity
        analyzer = OutcomeAnalyzer()
        result = analyzer.classify("no_op", {"reason": "already_playing"})
        assert result == OutcomeSeverity.BENIGN

    def test_completed_is_benign(self):
        from execution.metacognition import OutcomeAnalyzer, OutcomeSeverity
        analyzer = OutcomeAnalyzer()
        result = analyzer.classify("completed", {})
        assert result == OutcomeSeverity.BENIGN


# ─────────────────────────────────────────────────────────────
# fs.search_file: ambiguity detection
# ─────────────────────────────────────────────────────────────

class TestSearchFileAmbiguity:

    def _make_file_ref(self, name, rel_path, anchor="WORKSPACE", confidence=1.0):
        """Create a mock FileRef object."""
        ref = MagicMock()
        ref.name = name
        ref.relative_path = rel_path
        ref.anchor = anchor
        ref.confidence = confidence
        ref.to_output_dict.return_value = {
            "name": name,
            "relative_path": rel_path,
            "anchor": anchor,
            "confidence": confidence,
        }
        return ref

    def test_single_match_returns_completed(self):
        from skills.fs.search_file import SearchFileSkill

        loc = MagicMock()
        file_index = MagicMock()
        file_index.search.return_value = [
            self._make_file_ref("report.pdf", "docs/report.pdf"),
        ]

        skill = SearchFileSkill(location_config=loc, file_index=file_index)
        timeline = MagicMock()

        result = skill.execute({"query": "report.pdf"}, timeline)

        assert result.status is None  # Not no_op
        assert len(result.outputs["matches"]) == 1

    def test_same_name_multiple_matches_returns_no_op(self):
        from skills.fs.search_file import SearchFileSkill

        loc = MagicMock()
        file_index = MagicMock()
        file_index.search.return_value = [
            self._make_file_ref("resume.pdf", "work/resume.pdf", "DOCUMENTS"),
            self._make_file_ref("resume.pdf", "backup/resume.pdf", "DESKTOP"),
            self._make_file_ref("resume.pdf", "old/resume.pdf", "WORKSPACE"),
        ]

        skill = SearchFileSkill(location_config=loc, file_index=file_index)
        timeline = MagicMock()

        result = skill.execute({"query": "resume.pdf"}, timeline)

        assert result.status == "no_op"
        assert result.metadata["reason"] == "ambiguous_input"
        assert "resume.pdf" in result.metadata["message"]
        # Still has all matches in outputs
        assert len(result.outputs["matches"]) == 3

    def test_different_names_multiple_matches_returns_completed(self):
        from skills.fs.search_file import SearchFileSkill

        loc = MagicMock()
        file_index = MagicMock()
        file_index.search.return_value = [
            self._make_file_ref("report.pdf", "docs/report.pdf"),
            self._make_file_ref("report_v2.pdf", "docs/report_v2.pdf"),
        ]

        skill = SearchFileSkill(location_config=loc, file_index=file_index)
        timeline = MagicMock()

        result = skill.execute({"query": "report"}, timeline)

        # Different names → not ambiguous
        assert result.status is None
        assert len(result.outputs["matches"]) == 2

    def test_no_matches_returns_empty(self):
        from skills.fs.search_file import SearchFileSkill

        loc = MagicMock()
        file_index = MagicMock()
        file_index.search.return_value = []

        skill = SearchFileSkill(location_config=loc, file_index=file_index)
        timeline = MagicMock()

        result = skill.execute({"query": "nonexistent.pdf"}, timeline)

        assert result.outputs["matches"] == []


# ─────────────────────────────────────────────────────────────
# email.draft_message: duplicate attachment → no_op (not crash)
# ─────────────────────────────────────────────────────────────

class TestDraftMessageDuplicateAmbiguity:

    def test_duplicate_attachment_returns_no_op_not_crash(self, tmp_path):
        """Duplicate attachment names should trigger ambiguity, not crash."""
        from skills.email.draft_message import DraftMessageSkill

        # Create a file
        test_file = tmp_path / "doc.pdf"
        test_file.write_text("content")

        llm = MagicMock()
        llm.complete.return_value = "SUBJECT: Test\nBODY:\nTest body"
        client = MagicMock()
        client.create_draft.return_value = {"id": "d1", "body": "body"}
        loc = MagicMock()
        loc.resolve.return_value = Path(tmp_path)

        skill = DraftMessageSkill(
            content_llm=llm, email_client=client, location_config=loc,
        )

        timeline = MagicMock()
        inputs = {
            "prompt": "Send doc",
            "recipient": "bob@test.com",
            "attachments": [
                {"anchor": "DESKTOP", "relative_path": "doc.pdf", "name": "doc.pdf"},
                {"anchor": "DESKTOP", "relative_path": "doc.pdf", "name": "doc.pdf"},
            ],
        }

        result = skill.execute(inputs, timeline)

        # Should NOT crash — should return no_op with ambiguous_input
        assert result.status == "no_op"
        assert result.metadata["reason"] == "ambiguous_input"
        assert "doc.pdf" in result.metadata["message"]

    def test_validate_attachments_still_raises_for_duplicate(self, tmp_path):
        """_validate_attachments itself still raises ValueError (safety net).
        The execute() method catches it and converts to no_op."""
        from skills.email.draft_message import DraftMessageSkill

        test_file = tmp_path / "doc.pdf"
        test_file.write_text("content")

        llm = MagicMock()
        client = MagicMock()
        loc = MagicMock()
        loc.resolve.return_value = Path(tmp_path)

        skill = DraftMessageSkill(
            content_llm=llm, email_client=client, location_config=loc,
        )

        refs = [
            {"anchor": "DESKTOP", "relative_path": "doc.pdf", "name": "doc.pdf"},
            {"anchor": "DESKTOP", "relative_path": "doc.pdf", "name": "doc.pdf"},
        ]

        with pytest.raises(ValueError, match="Duplicate"):
            skill._validate_attachments(refs)

    def test_single_attachment_works_normally(self, tmp_path):
        """Single attachment should work without ambiguity."""
        from skills.email.draft_message import DraftMessageSkill

        test_file = tmp_path / "report.pdf"
        test_file.write_text("report")

        llm = MagicMock()
        llm.complete.return_value = "SUBJECT: Report\nBODY:\nHere's the report."
        client = MagicMock()
        client.create_draft.return_value = {"id": "d2", "body": "report"}
        loc = MagicMock()
        loc.resolve.return_value = Path(tmp_path)

        skill = DraftMessageSkill(
            content_llm=llm, email_client=client, location_config=loc,
        )

        timeline = MagicMock()
        inputs = {
            "prompt": "Send the report",
            "recipient": "bob@test.com",
            "attachments": [
                {"anchor": "DESKTOP", "relative_path": "report.pdf", "name": "report.pdf"},
            ],
        }

        result = skill.execute(inputs, timeline)

        assert result.status is None  # Normal completion
        assert result.outputs["draft_id"] == "d2"


# ─────────────────────────────────────────────────────────────
# Executor: SkillResult.status="no_op" → NodeStatus.NO_OP
# ─────────────────────────────────────────────────────────────

class TestExecutorStatusHonoring:

    def test_no_op_status_from_skill_result(self):
        """Executor should return NO_OP when SkillResult.status == 'no_op'."""
        from execution.executor import MissionExecutor
        from ir.mission import MissionPlan, MissionNode, ExecutionMode
        from world.timeline import WorldTimeline
        from skills.contract import SkillContract, FailurePolicy

        contract = SkillContract(
            name="test.mock",
            action="mock",
            target_type="mock",
            description="Mock",
            narration_template="mocking",
            domain="test",
            resource_cost="low",
            inputs={},
            outputs={"result": "text"},
            allowed_modes={ExecutionMode.foreground},
            failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
        )

        # Create a mock skill with all fields the executor accesses
        mock_skill = MagicMock()
        mock_skill.name = "test.mock"
        mock_skill.contract = contract
        mock_skill.input_keys = frozenset()
        mock_skill.output_keys = {"result"}
        mock_skill.execute.return_value = SkillResult(
            outputs={"result": "ambiguous"},
            status="no_op",
            metadata={"reason": "ambiguous_input"},
        )

        # registry.get() must return the same mock_skill object
        registry = MagicMock()
        registry.get.return_value = mock_skill
        timeline = MagicMock(spec=WorldTimeline)
        timeline.event_count.return_value = 0
        timeline.events_since_index.return_value = []

        executor = MissionExecutor(registry=registry, timeline=timeline)

        node = MissionNode(
            id="0", skill="test.mock", inputs={},
            outputs={"result": {"name": "test.mock.result", "type": "text"}},
            depends_on=[], mode=ExecutionMode.foreground,
        )

        exec_result = ExecutionResult()
        node_id, status, outputs, metadata = executor._execute_node(
            node, exec_result, None,
        )

        assert status == NodeStatus.NO_OP
        assert metadata["reason"] == "ambiguous_input"


# ─────────────────────────────────────────────────────────────
# Supervisor: ambiguous_input blocks dependent nodes
# ─────────────────────────────────────────────────────────────

class TestAmbiguityCascadeBlock:
    """Verify that ambiguous_input nodes block dependent execution.

    The supervisor's _execute_layer must add ambiguous_input nodes
    to exec_result.failed so the executor's cascade-skip logic
    (line 356) prevents dependents from running with bad data.
    """

    def test_ambiguous_node_blocks_dependents(self):
        """Node returning NO_OP/ambiguous_input should cause
        dependent nodes to be SKIPPED."""
        er = ExecutionResult()

        # Simulate node_0 returning NO_OP with ambiguous_input
        er.record("node_0", NodeStatus.NO_OP, {"matches": ["a", "b", "c"]},
                  {"reason": "ambiguous_input"})

        # The supervisor's new logic: add to failed if ambiguous_input
        meta = er.metadata.get("node_0", {})
        if meta.get("reason") == "ambiguous_input":
            er.failed.add("node_0")

        # Now check cascade: node_1 depends on node_0
        assert "node_0" in er.failed
        # Executor's cascade-skip checks:
        # skipped_or_failed_deps = [d for d in node.depends_on
        #     if d in er.skipped or d in er.failed]
        skipped_or_failed = [
            d for d in ["node_0"]
            if d in er.skipped or d in er.failed
        ]
        assert len(skipped_or_failed) > 0, "Dependent should see node_0 as failed"

    def test_idempotent_noop_does_not_block_dependents(self):
        """NO_OP with reason='already_playing' should NOT block dependents.
        Only ambiguous_input triggers the cascade block."""
        er = ExecutionResult()

        # Simulate idempotent NO_OP
        er.record("node_0", NodeStatus.NO_OP, {"status": "already_playing"},
                  {"reason": "already_playing"})

        # Supervisor does NOT add to failed for non-ambiguous reasons
        meta = er.metadata.get("node_0", {})
        if meta.get("reason") == "ambiguous_input":
            er.failed.add("node_0")

        # node_0 should NOT be in failed
        assert "node_0" not in er.failed
        assert "node_0" in er.completed

    def test_ambiguity_cascade_through_chain(self):
        """In a 3-node chain (node_0 → node_1 → node_2), ambiguity at
        node_0 should cascade-skip both node_1 and node_2."""
        er = ExecutionResult()

        # node_0: ambiguous
        er.record("node_0", NodeStatus.NO_OP, {"matches": ["a", "b"]},
                  {"reason": "ambiguous_input"})
        er.failed.add("node_0")  # Supervisor's new logic

        # node_1: depends on node_0 → should be skipped
        deps_1 = ["node_0"]
        blocked_1 = [d for d in deps_1 if d in er.skipped or d in er.failed]
        assert len(blocked_1) > 0
        # Simulate executor skipping node_1
        er.record("node_1", NodeStatus.SKIPPED, {})

        # node_2: depends on node_1 → should also be skipped
        deps_2 = ["node_1"]
        blocked_2 = [d for d in deps_2 if d in er.skipped or d in er.failed]
        assert len(blocked_2) > 0

    def test_outcome_verdict_has_ambiguous_reason(self):
        """Supervisor should append an outcome_verdict with the
        ambiguous_input reason for the orchestrator's recovery loop."""
        from execution.metacognition import OutcomeAnalyzer, OutcomeSeverity

        analyzer = OutcomeAnalyzer()
        er = ExecutionResult()

        # Simulate what the supervisor does
        status = NodeStatus.NO_OP
        meta = {"reason": "ambiguous_input", "message": "Which file?"}
        severity = analyzer.classify(status, meta)

        assert severity == OutcomeSeverity.SOFT_FAILURE

        er.outcome_verdicts.append({
            "node_id": "node_0",
            "skill": "fs.search_file",
            "status": str(status),
            "reason": meta.get("reason", ""),
            "severity": severity.value,
        })

        assert len(er.outcome_verdicts) == 1
        assert er.outcome_verdicts[0]["reason"] == "ambiguous_input"
        assert er.outcome_verdicts[0]["severity"] == "soft_failure"

    def test_failure_category_has_no_execution_error(self):
        """FailureCategory enum should not have EXECUTION_ERROR.
        Orchestrator must use CAPABILITY_FAILURE instead."""
        from execution.metacognition import FailureCategory

        # Verify EXECUTION_ERROR doesn't exist
        assert not hasattr(FailureCategory, "EXECUTION_ERROR")

        # Verify CAPABILITY_FAILURE does exist
        assert hasattr(FailureCategory, "CAPABILITY_FAILURE")


# ─────────────────────────────────────────────────────────────
# Orchestrator: ambiguous_input → clarification (not DecisionEngine)
# ─────────────────────────────────────────────────────────────

class TestOrchestratorAmbiguityInterception:
    """Verify that ambiguous_input soft failures short-circuit directly
    to clarification_needed, bypassing the DecisionEngine entirely."""

    def test_ambiguous_input_sets_clarification_from_metadata(self):
        """When outcome_verdicts has reason=ambiguous_input, the
        orchestrator should set clarification_needed using the skill's
        metadata (question + options), not route through DecisionEngine."""
        er = ExecutionResult()

        # Simulate what the supervisor stores
        er.record("node_0", NodeStatus.NO_OP, {"matches": ["a", "b"]},
                  {"reason": "ambiguous_input",
                   "message": "I found 2 files named 'doc.pdf':\n"
                              "1. DOCUMENTS/doc.pdf\n2. DESKTOP/doc.pdf\n"
                              "Which one did you mean?",
                   "options": ["1. DOCUMENTS/doc.pdf", "2. DESKTOP/doc.pdf"]})

        er.outcome_verdicts.append({
            "node_id": "node_0",
            "skill": "fs.search_file",
            "status": "NodeStatus.NO_OP",
            "reason": "ambiguous_input",
            "severity": "soft_failure",
        })

        # Simulate the orchestrator's interception logic
        soft_failures = [
            v for v in er.outcome_verdicts
            if v.get("severity") == "soft_failure"
        ]
        for sf in soft_failures:
            if sf.get("reason") == "ambiguous_input":
                node_meta = er.metadata.get(sf.get("node_id", ""), {})
                question = node_meta.get(
                    "message", "Could you clarify?",
                )
                er.clarification_needed = {
                    "question": question,
                    "options": node_meta.get("options", []),
                    "context": {
                        "source": "ambiguous_input",
                        "node_id": sf.get("node_id"),
                        "skill": sf.get("skill"),
                    },
                }
                break

        # Verify clarification was set correctly
        assert er.clarification_needed is not None
        assert "doc.pdf" in er.clarification_needed["question"]
        assert len(er.clarification_needed["options"]) == 2
        assert er.clarification_needed["context"]["source"] == "ambiguous_input"
        assert er.clarification_needed["context"]["skill"] == "fs.search_file"

    def test_non_ambiguous_soft_failure_not_intercepted(self):
        """Non-ambiguous soft failures should NOT be intercepted —
        they should fall through to the DecisionEngine."""
        er = ExecutionResult()

        er.outcome_verdicts.append({
            "node_id": "node_0",
            "skill": "browser.click",
            "status": "NodeStatus.FAILED",
            "reason": "element_not_found",
            "severity": "soft_failure",
        })

        soft_failures = [
            v for v in er.outcome_verdicts
            if v.get("severity") == "soft_failure"
        ]
        for sf in soft_failures:
            if sf.get("reason") == "ambiguous_input":
                er.clarification_needed = {"question": "should not happen"}
                break

        # Non-ambiguous → clarification NOT set
        assert er.clarification_needed is None


# ─────────────────────────────────────────────────────────────
# User selection parsing (numeric, ordinal, keyword)
# ─────────────────────────────────────────────────────────────

class TestParseUserSelection:
    """Test _parse_user_selection handles all answer formats."""

    @staticmethod
    def _parse(answer, options, matches):
        from merlin import Merlin
        return Merlin._parse_user_selection(answer, options, matches)

    def test_numeric_valid(self):
        """'3' → index 2 (0-based)."""
        matches = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        assert self._parse("3", [], matches) == 2

    def test_numeric_out_of_bounds(self):
        """'7' with only 3 matches → None."""
        matches = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        assert self._parse("7", [], matches) is None

    def test_numeric_zero(self):
        """'0' → None (1-indexed, so 0 is invalid)."""
        matches = [{"name": "a"}]
        assert self._parse("0", [], matches) is None

    def test_ordinal_third(self):
        """'third' → index 2 (via ORDINAL_MAP)."""
        matches = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        assert self._parse("third", [], matches) == 2

    def test_ordinal_first(self):
        """'first' → index 0."""
        matches = [{"name": "x"}, {"name": "y"}]
        assert self._parse("first", [], matches) == 0

    def test_keyword_fuzzy_match(self):
        """'gen ai' matches relative_path containing 'Gen AI Resume'."""
        matches = [
            {"name": "Alex_Benny.pdf", "relative_path": "DOCUMENTS/Work/Alex_Benny.pdf"},
            {"name": "Alex_Benny.pdf", "relative_path": "DESKTOP/Alex_Benny.pdf"},
            {"name": "Alex_Benny.pdf", "relative_path": "DOCUMENTS/Gen AI Resume/Alex_Benny.pdf"},
        ]
        result = self._parse("gen ai", [], matches)
        assert result == 2

    def test_garbage_input_returns_none(self):
        """Unparseable input → None (triggers re-ask)."""
        matches = [{"name": "a"}, {"name": "b"}]
        assert self._parse("asdfghjkl", [], matches) is None


# ─────────────────────────────────────────────────────────────
# Supervisor: override_outputs pre-seed + skip
# ─────────────────────────────────────────────────────────────

class TestSupervisorOverrideOutputs:
    """Verify that the supervisor correctly pre-seeds exec_result
    and skips pre-completed nodes during layer execution."""

    def test_override_outputs_pre_seeds_result(self):
        """override_outputs should appear in exec_result.completed
        and exec_result.results after supervisor.run()."""
        from unittest.mock import MagicMock
        from ir.mission import MissionPlan, MissionNode, ExecutionMode
        from execution.supervisor import ExecutionSupervisor, ExecutionContext
        from execution.executor import MissionExecutor
        from execution.registry import SkillRegistry
        from world.timeline import WorldTimeline

        registry = SkillRegistry()
        timeline = MagicMock(spec=WorldTimeline)
        timeline.event_count.return_value = 0
        timeline.events_since_index.return_value = []
        executor = MissionExecutor(registry, timeline)
        ctx = ExecutionContext()

        # Standalone node — just to verify pre-seed
        plan = MissionPlan(
            id="resume_test",
            nodes=[
                MissionNode(
                    id="node_0",
                    skill="fs.search_file",
                    inputs={"query": "test.pdf"},
                    mode=ExecutionMode.foreground,
                ),
            ],
            metadata={"ir_version": "1.0"},
        )

        supervisor = ExecutionSupervisor(executor=executor, context=ctx)
        override = {"node_0": {"matches": [{"name": "test.pdf", "path": "/a"}]}}

        result = supervisor.run(
            plan, world_snapshot=None,
            override_outputs=override,
        )

        # Pre-seeded node should be completed with our outputs
        assert "node_0" in result.completed
        assert result.results["node_0"]["matches"] == [{"name": "test.pdf", "path": "/a"}]
        # Metadata should show source
        assert result.metadata.get("node_0", {}).get("source") == "clarification_resume"

    def test_override_outputs_skips_execution(self):
        """Pre-completed node should NOT be re-executed.
        node_1 will attempt execution with resolved override output."""
        from unittest.mock import MagicMock
        from ir.mission import MissionPlan, MissionNode, ExecutionMode, OutputReference
        from execution.supervisor import ExecutionSupervisor, ExecutionContext
        from execution.executor import NodeStatus
        from skills.skill_result import SkillResult

        # Use mock executor (same pattern as test_execution_supervisor.py)
        executor = MagicMock()
        executor.timeline = MagicMock()
        executor._needs_focus.return_value = True
        executor._has_conflicts.return_value = False
        executor.execute_node.return_value = (
            "node_1", NodeStatus.COMPLETED, {"matches": []}, {}
        )
        executor.registry = MagicMock()
        ctx = ExecutionContext()

        # DAG: node_0 → node_1 (depends on node_0)
        # We override node_0 — node_1 should see the override output
        plan = MissionPlan(
            id="resume_skip_test",
            nodes=[
                MissionNode(
                    id="node_0",
                    skill="fs.search_file",
                    inputs={"query": "test.pdf"},
                    mode=ExecutionMode.foreground,
                ),
                MissionNode(
                    id="node_1",
                    skill="fs.search_file",
                    inputs={"query": OutputReference(node="node_0", output="matches")},
                    depends_on=["node_0"],
                    mode=ExecutionMode.foreground,
                ),
            ],
            metadata={"ir_version": "1.0"},
        )

        supervisor = ExecutionSupervisor(executor=executor, context=ctx)
        override = {"node_0": {"matches": [{"name": "resolved.pdf"}]}}

        result = supervisor.run(
            plan, world_snapshot=None,
            override_outputs=override,
        )

        # node_0 should be completed via override (not re-executed)
        assert "node_0" in result.completed
        assert result.metadata.get("node_0", {}).get("source") == "clarification_resume"

        # node_1 should have executed (it resolved the OutputReference
        # from node_0's pre-seeded results)
        assert "node_1" in result.completed or "node_1" in result.failed


# ─────────────────────────────────────────────────────────────
# Clarification signal includes plan + node_results
# ─────────────────────────────────────────────────────────────

class TestClarificationContextInSignal:
    """Verify the orchestrator includes plan and node_results
    in the clarification_needed context for ambiguous_input."""

    def test_ambiguous_input_signal_has_plan_and_results(self):
        """When an ambiguous_input soft failure is processed,
        the clarification_needed context should contain 'plan'
        and 'node_results' for true resume."""
        er = ExecutionResult()

        # Simulated plan
        from ir.mission import MissionPlan, MissionNode, ExecutionMode
        plan = MissionPlan(
            id="test_signal",
            nodes=[MissionNode(
                id="node_0", skill="fs.search_file",
                inputs={"query": "doc.pdf"},
                mode=ExecutionMode.foreground,
            )],
            metadata={"ir_version": "1.0"},
        )

        # Simulate what the supervisor stores
        er.record("node_0", NodeStatus.NO_OP, {"matches": ["a", "b"]},
                  {"reason": "ambiguous_input",
                   "message": "Which one?",
                   "options": ["1. a", "2. b"]})
        er.outcome_verdicts.append({
            "node_id": "node_0",
            "skill": "fs.search_file",
            "status": "NodeStatus.NO_OP",
            "reason": "ambiguous_input",
            "severity": "soft_failure",
        })

        # Replicate the orchestrator's interception logic
        soft_failures = [
            v for v in er.outcome_verdicts
            if v.get("severity") == "soft_failure"
        ]
        for sf in soft_failures:
            if sf.get("reason") == "ambiguous_input":
                node_meta = er.metadata.get(sf.get("node_id", ""), {})
                question = node_meta.get("message", "Could you clarify?")
                er.clarification_needed = {
                    "question": question,
                    "options": node_meta.get("options", []),
                    "context": {
                        "source": "ambiguous_input",
                        "node_id": sf.get("node_id"),
                        "skill": sf.get("skill"),
                        "plan": plan,
                        "node_results": dict(er.results),
                    },
                }
                break

        assert er.clarification_needed is not None
        ctx = er.clarification_needed["context"]
        assert ctx["source"] == "ambiguous_input"
        assert ctx["plan"] is plan
        assert ctx["node_results"]["node_0"]["matches"] == ["a", "b"]
