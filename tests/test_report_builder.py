# tests/test_report_builder.py

"""
Tests for truthful dual-channel reporting.

Covers:
- StructuredReport construction from MissionPlan + ExecutionResult
- ActionRecord with inputs/outputs for all scenarios
- NO_OP status for semantic no-ops (changed=False + reason)
- Constrained LLM prompt includes per-action detail with status labels
- Deterministic fallback produces per-action descriptions
- Soft validation detects missing entities in LLM response
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from ir.mission import IR_VERSION, MissionNode, MissionPlan, OutputReference
from conversation.frame import ConversationFrame
from execution.executor import NodeStatus, ExecutionResult
from reporting.report_builder import (
    ActionRecord,
    ReportBuilder,
    ReportType,
    StructuredReport,
)
from world.snapshot import WorldSnapshot
from world.state import WorldState
from world.timeline import WorldTimeline


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

def _make_node(node_id: str, skill: str, inputs: dict = None, depends_on: list = None) -> MissionNode:
    return MissionNode(
        id=node_id,
        skill=skill,
        inputs=inputs or {},
        depends_on=depends_on or [],
    )


def _make_plan(*nodes: MissionNode, plan_id: str = "test-plan") -> MissionPlan:
    return MissionPlan(
        id=plan_id,
        nodes=list(nodes),
        metadata={"ir_version": IR_VERSION},
    )


def _make_conversation(user_text: str = "test input") -> ConversationFrame:
    conv = ConversationFrame()
    conv.append_turn("user", user_text)
    return conv


def _make_timeline() -> WorldTimeline:
    return WorldTimeline()


def _make_snapshot() -> WorldSnapshot:
    state = WorldState()
    return WorldSnapshot.build(state, [])


def _make_exec_result(
    results: dict = None,
    metadata: dict = None,
    statuses: dict = None,
) -> ExecutionResult:
    """Build an ExecutionResult from simple dicts.

    results:  node_id → outputs dict
    metadata: node_id → metadata dict
    statuses: node_id → NodeStatus

    If statuses not provided, all nodes in results get COMPLETED.
    """
    er = ExecutionResult()
    results = results or {}
    metadata = metadata or {}
    statuses = statuses or {}

    for nid, outputs in results.items():
        status = statuses.get(nid, NodeStatus.COMPLETED)
        meta = metadata.get(nid)
        er.record(nid, status, outputs, meta)

    # Handle nodes in statuses that are not in results (failed/skipped)
    for nid, status in statuses.items():
        if nid not in results:
            er.record(nid, status, {})

    return er


# ─────────────────────────────────────────────────────────────
# Step 1: StructuredReport Construction
# ─────────────────────────────────────────────────────────────

class TestStructuredReport:
    """Tests for _build_structured_report → StructuredReport."""

    def test_all_success(self):
        """All nodes completed → SUCCESS report with full action records."""
        builder = ReportBuilder()
        plan = _make_plan(
            _make_node("n1", "fs.create_folder", {"name": "X", "anchor": "WORKSPACE"}),
            _make_node("n2", "fs.create_folder", {"name": "Y", "anchor": "WORKSPACE"}),
        )
        exec_result = _make_exec_result(
            results={"n1": {"created": "D:\\X"}, "n2": {"created": "D:\\Y"}},
        )
        conv = _make_conversation("create folders X and Y")

        report = builder._build_structured_report(plan, exec_result, conv)

        assert report.report_type == ReportType.SUCCESS
        assert len(report.actions) == 2
        assert report.mission_id == "test-plan"
        assert report.user_query == "create folders X and Y"

        # Check action records carry inputs AND outputs
        a1 = report.actions[0]
        assert a1.node_id == "n1"
        assert a1.skill == "fs.create_folder"
        assert a1.status == NodeStatus.COMPLETED.value
        assert a1.inputs["name"] == "X"
        assert a1.outputs["created"] == "D:\\X"
        assert a1.error is None
        assert a1.reason is None

    def test_all_failed(self):
        """No nodes completed → FAILURE report."""
        builder = ReportBuilder()
        plan = _make_plan(
            _make_node("n1", "fs.create_folder", {"name": "X"}),
        )
        # Node has no status at all — not in results or statuses
        exec_result = ExecutionResult()

        report = builder._build_structured_report(plan, exec_result, _make_conversation())

        assert report.report_type == ReportType.FAILURE
        assert len(report.actions) == 1
        assert report.actions[0].status == NodeStatus.FAILED.value
        assert report.actions[0].error is not None

    def test_partial_success(self):
        """Some nodes completed, some failed → PARTIAL report."""
        builder = ReportBuilder()
        plan = _make_plan(
            _make_node("n1", "fs.create_folder", {"name": "X"}),
            _make_node("n2", "fs.create_folder", {"name": "Y"}),
        )
        exec_result = _make_exec_result(
            results={"n1": {"created": "D:\\X"}},
            statuses={"n2": NodeStatus.FAILED},
        )

        report = builder._build_structured_report(plan, exec_result, _make_conversation())

        assert report.report_type == ReportType.PARTIAL
        assert report.actions[0].status == NodeStatus.COMPLETED.value
        assert report.actions[1].status == NodeStatus.FAILED.value

    def test_output_reference_sanitized(self):
        """OutputReference inputs are sanitized to readable string."""
        builder = ReportBuilder()
        ref = OutputReference(node="n1", output="created")
        plan = _make_plan(
            _make_node("n1", "fs.create_folder", {"name": "X"}),
            _make_node("n2", "fs.create_folder", {"name": "Z", "parent": ref}, depends_on=["n1"]),
        )
        exec_result = _make_exec_result(
            results={"n1": {"created": "D:\\X"}, "n2": {"created": "D:\\X\\Z"}},
        )

        report = builder._build_structured_report(plan, exec_result, _make_conversation())

        # OutputReference should be sanitized
        assert report.actions[1].inputs["parent"] == "<ref:n1.created>"
        # Primitives should be preserved
        assert report.actions[1].inputs["name"] == "Z"

    def test_user_query_from_history(self):
        """User query is extracted from conversation history."""
        builder = ReportBuilder()
        plan = _make_plan(_make_node("n1", "fs.create_folder", {"name": "X"}))
        conv = ConversationFrame()
        conv.append_turn("user", "first question")
        conv.append_turn("assistant", "first answer")
        conv.append_turn("user", "create folder X")

        exec_result = _make_exec_result(results={"n1": {}})
        report = builder._build_structured_report(plan, exec_result, conv)
        assert report.user_query == "create folder X"

    def test_empty_conversation_history(self):
        """Empty conversation history → empty user query (no crash)."""
        builder = ReportBuilder()
        plan = _make_plan(_make_node("n1", "fs.create_folder", {"name": "X"}))
        conv = ConversationFrame()

        exec_result = _make_exec_result(results={"n1": {}})
        report = builder._build_structured_report(plan, exec_result, conv)
        assert report.user_query == ""


# ─────────────────────────────────────────────────────────────
# Step 1.5: NO_OP Status
# ─────────────────────────────────────────────────────────────

class TestNoOpStatus:
    """Tests for NO_OP semantic status in structured report and narration."""

    def test_no_op_status_from_metadata(self):
        """Node with NO_OP status carries reason in ActionRecord."""
        builder = ReportBuilder()
        plan = _make_plan(
            _make_node("n1", "system.media_play", {}),
        )
        exec_result = _make_exec_result(
            results={"n1": {"changed": False}},
            metadata={"n1": {"domain": "system", "reason": "no_media_session"}},
            statuses={"n1": NodeStatus.NO_OP},
        )

        report = builder._build_structured_report(plan, exec_result, _make_conversation())

        assert report.report_type == ReportType.SUCCESS  # NO_OP counts as completed
        a1 = report.actions[0]
        assert a1.status == NodeStatus.NO_OP.value
        assert a1.reason == "no_media_session"
        assert a1.outputs == {"changed": False}

    def test_no_op_in_multi_node_plan(self):
        """Mixed plan: 3 SUCCESS + 1 NO_OP → overall SUCCESS, no_op action carries reason."""
        builder = ReportBuilder()
        plan = _make_plan(
            _make_node("n0", "system.unmute", {}),
            _make_node("n1", "system.set_volume", {"level": "10"}),
            _make_node("n2", "system.set_brightness", {"level": "10"}),
            _make_node("n3", "system.media_play", {}),
        )
        exec_result = _make_exec_result(
            results={
                "n0": {"muted": False},
                "n1": {"level": 10},
                "n2": {"level": 10},
                "n3": {"changed": False},
            },
            metadata={
                "n0": {"domain": "system"},
                "n1": {"domain": "system"},
                "n2": {"domain": "system"},
                "n3": {"domain": "system", "reason": "no_media_session"},
            },
            statuses={
                "n0": NodeStatus.COMPLETED,
                "n1": NodeStatus.COMPLETED,
                "n2": NodeStatus.COMPLETED,
                "n3": NodeStatus.NO_OP,
            },
        )

        report = builder._build_structured_report(plan, exec_result, _make_conversation())

        assert report.report_type == ReportType.SUCCESS
        assert len(report.actions) == 4
        # First 3 are COMPLETED
        for i in range(3):
            assert report.actions[i].status == NodeStatus.COMPLETED.value
        # Last is NO_OP with reason
        assert report.actions[3].status == NodeStatus.NO_OP.value
        assert report.actions[3].reason == "no_media_session"

    def test_no_op_describe_action_shows_reason(self):
        """_describe_action includes reason for NO_OP actions."""
        action = ActionRecord(
            node_id="n1",
            skill="system.media_play",
            status=NodeStatus.NO_OP.value,
            inputs={},
            outputs={"changed": False},
            reason="no_media_session",
        )
        desc = ReportBuilder._describe_action(action)
        assert "no_media_session" in desc
        # Should NOT show raw changed=False when reason is present
        assert "changed" not in desc

    def test_no_op_llm_prompt_tags_correctly(self):
        """LLM prompt uses [NO_OP] tag, not [SUCCESS]."""
        builder = ReportBuilder()
        report = StructuredReport(
            report_type=ReportType.SUCCESS,
            actions=[
                ActionRecord(
                    node_id="n1",
                    skill="system.media_play",
                    status=NodeStatus.NO_OP.value,
                    inputs={},
                    outputs={"changed": False},
                    reason="no_media_session",
                ),
            ],
            user_query="play music",
            mission_id="m1",
        )
        prompt = builder._build_llm_prompt(report, _make_conversation())

        assert "[NO_OP]" in prompt
        assert "no_media_session" in prompt
        assert "[SUCCESS]" not in prompt  # Must NOT be labeled SUCCESS

    def test_no_op_fallback_text(self):
        """Fallback text for NO_OP says 'No action' with reason."""
        builder = ReportBuilder()
        plan = _make_plan(
            _make_node("n1", "system.media_play", {}),
        )
        exec_result = _make_exec_result(
            results={"n1": {"changed": False}},
            metadata={"n1": {"domain": "system", "reason": "no_media_session"}},
            statuses={"n1": NodeStatus.NO_OP},
        )

        text = builder.build(
            plan, exec_result,
            _make_timeline(), _make_snapshot(), _make_conversation(),
        )

        assert "No action" in text
        assert "no_media_session" in text

    def test_no_op_already_playing(self):
        """already_playing reason is preserved."""
        builder = ReportBuilder()
        plan = _make_plan(
            _make_node("n1", "system.media_play", {}),
        )
        exec_result = _make_exec_result(
            results={"n1": {"changed": False}},
            metadata={"n1": {"domain": "system", "reason": "already_playing"}},
            statuses={"n1": NodeStatus.NO_OP},
        )

        report = builder._build_structured_report(plan, exec_result, _make_conversation())

        assert report.actions[0].reason == "already_playing"


# ─────────────────────────────────────────────────────────────
# Step 2: Constrained LLM Prompt
# ─────────────────────────────────────────────────────────────

class TestConstrainedPrompt:
    """Tests that the LLM prompt contains explicit action detail."""

    def test_prompt_contains_action_detail(self):
        """LLM prompt must contain skill names, inputs, and outputs."""
        builder = ReportBuilder()
        report = StructuredReport(
            report_type=ReportType.SUCCESS,
            actions=[
                ActionRecord(
                    node_id="n1",
                    skill="fs.create_folder",
                    status=NodeStatus.COMPLETED.value,
                    inputs={"name": "HelloWorld", "anchor": "DESKTOP"},
                    outputs={"created": "C:\\Users\\Desktop\\HelloWorld"},
                ),
            ],
            user_query="create a folder called HelloWorld on desktop",
            mission_id="m1",
        )
        conv = _make_conversation()

        prompt = builder._build_llm_prompt(report, conv)

        # Must contain the folder name (the crucial fact!)
        assert "HelloWorld" in prompt
        # Must contain the output path
        assert "HelloWorld" in prompt
        # Must contain the skill
        assert "create_folder" in prompt
        # Must contain the status
        assert "SUCCESS" in prompt
        # Must contain guardrails
        assert "Do NOT add actions" in prompt
        assert "Do NOT claim actions that are not listed" in prompt

    def test_prompt_contains_failure_detail(self):
        """Failed actions must show error detail in prompt."""
        builder = ReportBuilder()
        report = StructuredReport(
            report_type=ReportType.FAILURE,
            actions=[
                ActionRecord(
                    node_id="n1",
                    skill="fs.create_folder",
                    status=NodeStatus.FAILED.value,
                    inputs={"name": "X"},
                    error="Permission denied",
                ),
            ],
            user_query="create folder X",
            mission_id="m1",
        )
        conv = _make_conversation()

        prompt = builder._build_llm_prompt(report, conv)

        assert "FAILED" in prompt
        assert "Permission denied" in prompt

    def test_prompt_does_not_contain_raw_user_query(self):
        """LLM prompt must NOT contain the raw user query (hallucination vector).

        Instead it should contain 'Context: The assistant executed the following plan.'
        and an 'Unsupported requests' section.
        """
        builder = ReportBuilder()
        report = StructuredReport(
            report_type=ReportType.SUCCESS,
            actions=[
                ActionRecord(
                    node_id="n1",
                    skill="fs.create_folder",
                    status=NodeStatus.COMPLETED.value,
                    inputs={"name": "X"},
                    outputs={"created": "D:\\X"},
                ),
            ],
            user_query="create folders X and Y",
            mission_id="m1",
        )
        conv = _make_conversation()

        prompt = builder._build_llm_prompt(report, conv)

        # Raw user query must NOT appear in prompt
        assert 'User request:' not in prompt
        # New context header must be present
        assert "Context: The assistant executed the following plan." in prompt
        # Unsupported requests section must exist
        assert "Unsupported requests" in prompt

    def test_prompt_multi_action_lists_each(self):
        """Multi-action missions list each action separately."""
        builder = ReportBuilder()
        report = StructuredReport(
            report_type=ReportType.SUCCESS,
            actions=[
                ActionRecord(
                    node_id="n1", skill="fs.create_folder",
                    status=NodeStatus.COMPLETED.value,
                    inputs={"name": "Alpha"},
                    outputs={"created": "D:\\Alpha"},
                ),
                ActionRecord(
                    node_id="n2", skill="fs.create_folder",
                    status=NodeStatus.COMPLETED.value,
                    inputs={"name": "Beta"},
                    outputs={"created": "D:\\Beta"},
                ),
                ActionRecord(
                    node_id="n3", skill="fs.create_folder",
                    status=NodeStatus.COMPLETED.value,
                    inputs={"name": "Gamma"},
                    outputs={"created": "D:\\Alpha\\Gamma"},
                ),
            ],
            user_query="create folders Alpha and Beta and inside Alpha create Gamma",
            mission_id="m1",
        )
        conv = _make_conversation()

        prompt = builder._build_llm_prompt(report, conv)

        # Each action must appear as a numbered item
        assert "1." in prompt
        assert "2." in prompt
        assert "3." in prompt
        # Each name must appear
        assert "Alpha" in prompt
        assert "Beta" in prompt
        assert "Gamma" in prompt

    def test_prompt_no_op_grounding_instructions(self):
        """LLM prompt includes explicit NO_OP grounding instructions."""
        builder = ReportBuilder()
        report = StructuredReport(
            report_type=ReportType.SUCCESS,
            actions=[
                ActionRecord(
                    node_id="n1",
                    skill="system.media_play",
                    status=NodeStatus.NO_OP.value,
                    inputs={},
                    reason="no_media_session",
                ),
            ],
            user_query="play music",
            mission_id="m1",
        )
        prompt = builder._build_llm_prompt(report, _make_conversation())

        # Must instruct LLM about NO_OP semantics
        assert "NO_OP" in prompt
        assert "Do NOT claim a NO_OP action succeeded" in prompt


# ─────────────────────────────────────────────────────────────
# Step 3: Deterministic Fallback
# ─────────────────────────────────────────────────────────────

class TestDeterministicFallback:
    """Tests that fallback text is useful and specific."""

    def test_fallback_includes_action_names(self):
        """Fallback text must name what was created, not just count."""
        builder = ReportBuilder()  # No LLM
        plan = _make_plan(
            _make_node("n1", "fs.create_folder", {"name": "X"}),
            _make_node("n2", "fs.create_folder", {"name": "Y"}),
        )
        exec_result = _make_exec_result(
            results={"n1": {"created": "D:\\X"}, "n2": {"created": "D:\\Y"}},
        )

        text = builder.build(
            plan, exec_result,
            _make_timeline(), _make_snapshot(), _make_conversation(),
        )

        # Must mention actual names — never just "Completed 2 action(s)"
        assert "create_folder" in text
        assert "X" in text or "D:\\X" in text.replace("/", "\\")
        assert "Y" in text or "D:\\Y" in text.replace("/", "\\")

    def test_fallback_shows_failures(self):
        """Fallback text must describe failed actions."""
        builder = ReportBuilder()
        plan = _make_plan(
            _make_node("n1", "fs.create_folder", {"name": "X"}),
        )
        exec_result = _make_exec_result(
            statuses={"n1": NodeStatus.FAILED},
        )

        text = builder.build(
            plan, exec_result,
            _make_timeline(), _make_snapshot(), _make_conversation(),
        )

        assert "Failed" in text

    def test_fallback_empty_plan(self):
        """Empty plan → simple 'Done.'"""
        builder = ReportBuilder()
        plan = _make_plan(plan_id="empty-plan")
        exec_result = ExecutionResult()

        text = builder.build(
            plan, exec_result,
            _make_timeline(), _make_snapshot(), _make_conversation(),
        )

        assert text == "Done."

    def test_fallback_llm_failure_graceful(self):
        """If LLM raises, fallback must produce useful text, not crash."""
        mock_llm = MagicMock()
        mock_llm.complete.side_effect = RuntimeError("LLM unavailable")
        builder = ReportBuilder(llm=mock_llm)
        plan = _make_plan(
            _make_node("n1", "fs.create_folder", {"name": "X"}),
        )
        exec_result = _make_exec_result(
            results={"n1": {"created": "D:\\X"}},
        )

        text = builder.build(
            plan, exec_result,
            _make_timeline(), _make_snapshot(), _make_conversation(),
        )

        # Must not crash, must produce useful text
        assert "create_folder" in text
        assert "X" in text or "D:\\X" in text


# ─────────────────────────────────────────────────────────────
# Step 4: Soft Validation
# ─────────────────────────────────────────────────────────────

class TestSoftValidation:
    """Tests for Phase 1 validation — log anomalies, no rejection."""

    def test_validation_logs_warning_when_no_entities_mentioned(self, caplog):
        """LLM response that mentions NO known entities triggers warning."""
        builder = ReportBuilder()
        report = StructuredReport(
            report_type=ReportType.SUCCESS,
            actions=[
                ActionRecord(
                    node_id="n1",
                    skill="fs.create_folder",
                    status=NodeStatus.COMPLETED.value,
                    inputs={"name": "ImportantFolder"},
                    outputs={"created": "D:\\ImportantFolder"},
                ),
            ],
            user_query="create folder ImportantFolder",
            mission_id="m1",
        )

        with caplog.at_level(logging.WARNING):
            builder._validate_response(
                "I've completed your request successfully.", report,
            )

        assert any("[VALIDATION]" in r.message for r in caplog.records)

    def test_validation_no_warning_when_entities_mentioned(self, caplog):
        """LLM response mentioning known entities → no warning."""
        builder = ReportBuilder()
        report = StructuredReport(
            report_type=ReportType.SUCCESS,
            actions=[
                ActionRecord(
                    node_id="n1",
                    skill="fs.create_folder",
                    status=NodeStatus.COMPLETED.value,
                    inputs={"name": "MyProject"},
                    outputs={"created": "D:\\MyProject"},
                ),
            ],
            user_query="create folder MyProject",
            mission_id="m1",
        )

        with caplog.at_level(logging.WARNING):
            builder._validate_response(
                "Created folder MyProject on your workspace.", report,
            )

        assert not any("[VALIDATION]" in r.message for r in caplog.records)

    def test_validation_no_crash_on_empty_actions(self, caplog):
        """Validation doesn't crash on empty actions list."""
        builder = ReportBuilder()
        report = StructuredReport(
            report_type=ReportType.FAILURE,
            actions=[],
            user_query="test",
            mission_id="m1",
        )

        # Must not raise
        builder._validate_response("Something happened.", report)


# ─────────────────────────────────────────────────────────────
# Integration: Full build() with mock LLM
# ─────────────────────────────────────────────────────────────

class TestBuildIntegration:
    """End-to-end tests for the full build() pipeline with mocked LLM."""

    def test_build_passes_structured_truth_to_llm(self):
        """The LLM receives a prompt containing per-action detail."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "Created folders X and Y."
        builder = ReportBuilder(llm=mock_llm)

        plan = _make_plan(
            _make_node("n1", "fs.create_folder", {"name": "X"}),
            _make_node("n2", "fs.create_folder", {"name": "Y"}),
        )
        exec_result = _make_exec_result(
            results={"n1": {"created": "D:\\X"}, "n2": {"created": "D:\\Y"}},
        )
        conv = _make_conversation("create folders X and Y")

        text = builder.build(
            plan, exec_result,
            _make_timeline(), _make_snapshot(), conv,
        )

        # LLM was called
        assert mock_llm.complete.called
        prompt = mock_llm.complete.call_args[0][0]

        # Prompt contains per-action truth
        assert "X" in prompt
        assert "Y" in prompt
        assert "create_folder" in prompt

        # LLM response is returned
        assert text == "Created folders X and Y."

    def test_build_returns_llm_response_on_success(self):
        """build() returns the LLM's response text."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "  Folder created.  "
        builder = ReportBuilder(llm=mock_llm)

        plan = _make_plan(
            _make_node("n1", "fs.create_folder", {"name": "X"}),
        )
        exec_result = _make_exec_result(
            results={"n1": {"created": "D:\\X"}},
        )

        text = builder.build(
            plan, exec_result,
            _make_timeline(), _make_snapshot(), _make_conversation(),
        )

        # Response is stripped
        assert text == "Folder created."


# ─────────────────────────────────────────────────────────────
# ActionRecord describe
# ─────────────────────────────────────────────────────────────

class TestDescribeAction:
    """Tests for _describe_action static method."""

    def test_describe_success_with_outputs(self):
        action = ActionRecord(
            node_id="n1",
            skill="fs.create_folder",
            status=NodeStatus.COMPLETED.value,
            inputs={"name": "Hello"},
            outputs={"created": "D:\\Hello"},
        )
        desc = ReportBuilder._describe_action(action)
        assert "create_folder" in desc
        assert "Hello" in desc
        assert "D:\\Hello" in desc

    def test_describe_failure_with_error(self):
        action = ActionRecord(
            node_id="n1",
            skill="fs.create_folder",
            status=NodeStatus.FAILED.value,
            inputs={"name": "X"},
            error="Permission denied",
        )
        desc = ReportBuilder._describe_action(action)
        assert "Permission denied" in desc

    def test_describe_strips_domain_prefix(self):
        action = ActionRecord(
            node_id="n1",
            skill="system.open_app",
            status=NodeStatus.COMPLETED.value,
            inputs={"app_name": "Chrome"},
            outputs={"app_name": "chrome.exe"},
        )
        desc = ReportBuilder._describe_action(action)
        assert "open_app" in desc
        assert "Chrome" in desc

    def test_describe_no_op_with_reason(self):
        """NO_OP action shows reason, not raw output values."""
        action = ActionRecord(
            node_id="n1",
            skill="system.media_play",
            status=NodeStatus.NO_OP.value,
            inputs={},
            outputs={"changed": False},
            reason="already_playing",
        )
        desc = ReportBuilder._describe_action(action)
        assert "already_playing" in desc
        assert "changed" not in desc


# ─────────────────────────────────────────────────────────────
# NodeStatus centralization
# ─────────────────────────────────────────────────────────────

class TestNodeStatusCentralization:
    """Ensure NodeStatus is a proper str enum and labels are exhaustive."""

    def test_node_status_is_str_enum(self):
        """NodeStatus values are comparable as strings."""
        assert NodeStatus.COMPLETED == "completed"
        assert NodeStatus.NO_OP == "no_op"
        assert NodeStatus.SKIPPED == "skipped"
        assert NodeStatus.FAILED == "failed"
        assert NodeStatus.TIMED_OUT == "timed_out"

    def test_all_statuses_have_labels(self):
        """Every NodeStatus value has a label in ReportBuilder._STATUS_LABELS."""
        for status in NodeStatus:
            assert status in ReportBuilder._STATUS_LABELS, (
                f"NodeStatus.{status.name} missing from _STATUS_LABELS"
            )

    def test_execution_result_no_op_treated_as_completed(self):
        """NO_OP nodes appear in .completed set and .results dict."""
        er = ExecutionResult()
        er.record("n1", NodeStatus.NO_OP, {"changed": False}, {"reason": "already_playing"})

        assert "n1" in er.completed
        assert "n1" in er.results
        assert er.results["n1"] == {"changed": False}
        assert er.metadata["n1"] == {"reason": "already_playing"}
        assert "n1" not in er.failed
        assert "n1" not in er.skipped
