# tests/test_mission_outcome.py

"""
Tests for Phase 1B + Pre-Phase 4: Mission Outcome Index.

Validates:
- MissionOutcome schema enforcement
- Outcome created for every mission
- artifacts vs visible_lists strict separation
- nodes_executed / nodes_skipped correctness
- nodes_failed / nodes_timed_out correctness
- _build_outcome helper with typed ExecutionResult
"""

import pytest
from pydantic import ValidationError

from conversation.outcome import MissionOutcome
from conversation.frame import ConversationFrame
from orchestrator.mission_orchestrator import MissionOrchestrator
from execution.executor import ExecutionResult, NodeStatus
from ir.mission import MissionPlan, MissionNode, IR_VERSION


def _make_exec_result(
    completed: dict | None = None,
    skipped: list | None = None,
    failed: list | None = None,
    timed_out: list | None = None,
) -> ExecutionResult:
    """Helper to build an ExecutionResult from test data."""
    er = ExecutionResult()
    for nid, outputs in (completed or {}).items():
        er.record(nid, NodeStatus.COMPLETED, outputs)
    for nid in (skipped or []):
        er.record(nid, NodeStatus.SKIPPED, {})
    for nid in (failed or []):
        er.record(nid, NodeStatus.FAILED, {})
    for nid in (timed_out or []):
        er.record(nid, NodeStatus.TIMED_OUT, {})
    return er


class TestMissionOutcomeSchema:
    """Validate MissionOutcome type enforcement."""

    def test_valid_outcome(self):
        outcome = MissionOutcome(
            mission_id="m_1",
            nodes_executed=["n_0", "n_1"],
            nodes_skipped=[],
        )
        assert outcome.mission_id == "m_1"
        assert outcome.nodes_executed == ["n_0", "n_1"]
        assert outcome.nodes_skipped == []
        assert outcome.artifacts == {}
        assert outcome.visible_lists == {}
        assert outcome.active_entity is None
        assert outcome.active_domain is None

    def test_empty_lists_allowed(self):
        outcome = MissionOutcome(
            mission_id="m_1",
            nodes_executed=[],
            nodes_skipped=[],
        )
        assert outcome.nodes_executed == []
        assert outcome.nodes_skipped == []

    def test_missing_required_fields_rejected(self):
        """nodes_executed and nodes_skipped are required."""
        with pytest.raises(ValidationError):
            MissionOutcome(mission_id="m_1")

    def test_missing_mission_id_rejected(self):
        with pytest.raises(ValidationError):
            MissionOutcome(
                nodes_executed=[],
                nodes_skipped=[],
            )

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            MissionOutcome(
                mission_id="m_1",
                nodes_executed=[],
                nodes_skipped=[],
                garbage="nope",
            )

    def test_artifacts_and_visible_lists_separate(self):
        """Artifacts hold scalars, visible_lists hold lists."""
        outcome = MissionOutcome(
            mission_id="m_1",
            nodes_executed=["n_0"],
            nodes_skipped=[],
            artifacts={"folder": "/Desktop/test"},
            visible_lists={"videos": [{"title": "v1"}, {"title": "v2"}]},
        )
        assert isinstance(outcome.artifacts["folder"], str)
        assert isinstance(outcome.visible_lists["videos"], list)
        assert len(outcome.visible_lists["videos"]) == 2

    def test_timestamp_auto_set(self):
        outcome = MissionOutcome(
            mission_id="m_1",
            nodes_executed=[],
            nodes_skipped=[],
        )
        assert outcome.timestamp > 0

    def test_nodes_failed_and_timed_out_defaults(self):
        """nodes_failed and nodes_timed_out default to empty lists."""
        outcome = MissionOutcome(
            mission_id="m_1",
            nodes_executed=[],
            nodes_skipped=[],
        )
        assert outcome.nodes_failed == []
        assert outcome.nodes_timed_out == []

    def test_nodes_failed_and_timed_out_set(self):
        outcome = MissionOutcome(
            mission_id="m_1",
            nodes_executed=["n_0"],
            nodes_skipped=[],
            nodes_failed=["n_1"],
            nodes_timed_out=["n_2"],
        )
        assert outcome.nodes_failed == ["n_1"]
        assert outcome.nodes_timed_out == ["n_2"]


class TestBuildOutcome:
    """Test MissionOrchestrator._build_outcome()."""

    def _make_plan(self, nodes):
        return MissionPlan(
            id="test_plan",
            nodes=nodes,
            metadata={"ir_version": IR_VERSION},
        )

    def test_all_nodes_executed(self):
        plan = self._make_plan([
            MissionNode(id="n_0", skill="fs.create_folder", inputs={"path": "/tmp"}),
            MissionNode(id="n_1", skill="fs.create_folder", inputs={"path": "/tmp2"}),
        ])
        er = _make_exec_result(completed={
            "n_0": {"path": "/tmp"},
            "n_1": {"path": "/tmp2"},
        })
        outcome = MissionOrchestrator._build_outcome(plan, er)
        assert sorted(outcome.nodes_executed) == ["n_0", "n_1"]
        assert outcome.nodes_skipped == []
        assert outcome.nodes_failed == []
        assert outcome.nodes_timed_out == []

    def test_some_nodes_skipped(self):
        plan = self._make_plan([
            MissionNode(id="n_0", skill="fs.create_folder", inputs={}),
            MissionNode(id="n_1", skill="browser.search", inputs={}),
        ])
        er = _make_exec_result(
            completed={"n_0": {"path": "/tmp"}},
            skipped=["n_1"],
        )
        outcome = MissionOrchestrator._build_outcome(plan, er)
        assert outcome.nodes_executed == ["n_0"]
        assert outcome.nodes_skipped == ["n_1"]

    def test_node_failed(self):
        """Failed nodes appear in nodes_failed, not nodes_skipped."""
        plan = self._make_plan([
            MissionNode(id="n_0", skill="fs.create_folder", inputs={}),
            MissionNode(id="n_1", skill="browser.search", inputs={}),
        ])
        er = _make_exec_result(
            completed={"n_0": {"path": "/tmp"}},
            failed=["n_1"],
        )
        outcome = MissionOrchestrator._build_outcome(plan, er)
        assert outcome.nodes_executed == ["n_0"]
        assert outcome.nodes_failed == ["n_1"]
        assert outcome.nodes_skipped == []

    def test_node_timed_out(self):
        """Timed-out nodes appear in nodes_timed_out."""
        plan = self._make_plan([
            MissionNode(id="n_0", skill="fs.create_folder", inputs={}),
        ])
        er = _make_exec_result(timed_out=["n_0"])
        outcome = MissionOrchestrator._build_outcome(plan, er)
        assert outcome.nodes_executed == []
        assert outcome.nodes_timed_out == ["n_0"]
        assert outcome.nodes_failed == []

    def test_scalar_outputs_become_artifacts(self):
        plan = self._make_plan([
            MissionNode(id="n_0", skill="fs.create_folder", inputs={}),
        ])
        er = _make_exec_result(completed={"n_0": {"path": "/Desktop/hello"}})
        outcome = MissionOrchestrator._build_outcome(plan, er)
        assert "n_0.path" in outcome.artifacts
        assert outcome.artifacts["n_0.path"] == "/Desktop/hello"

    def test_list_outputs_become_visible_lists(self):
        plan = self._make_plan([
            MissionNode(id="n_0", skill="browser.search", inputs={}),
        ])
        er = _make_exec_result(completed={"n_0": {"results": [{"title": "a"}, {"title": "b"}]}})
        outcome = MissionOrchestrator._build_outcome(plan, er)
        assert "n_0.results" in outcome.visible_lists
        assert len(outcome.visible_lists["n_0.results"]) == 2
        # Artifacts should NOT contain the list
        assert "n_0.results" not in outcome.artifacts

    def test_domain_inferred_from_skill_name(self):
        plan = self._make_plan([
            MissionNode(id="n_0", skill="fs.create_folder", inputs={}),
        ])
        er = _make_exec_result(completed={"n_0": {"path": "/tmp"}})
        outcome = MissionOrchestrator._build_outcome(plan, er)
        assert outcome.active_domain == "fs"

    def test_domain_from_explicit_output(self):
        plan = self._make_plan([
            MissionNode(id="n_0", skill="fs.create_folder", inputs={}),
        ])
        er = _make_exec_result(completed={"n_0": {"path": "/tmp"}})
        # Domain now comes through metadata channel (SkillResult.metadata)
        er.metadata["n_0"] = {"domain": "filesystem"}
        outcome = MissionOrchestrator._build_outcome(plan, er)
        assert outcome.active_domain == "filesystem"

    def test_entity_from_explicit_output(self):
        plan = self._make_plan([
            MissionNode(id="n_0", skill="fs.create_folder", inputs={}),
        ])
        er = _make_exec_result(completed={"n_0": {"path": "/tmp"}})
        # Entity now comes through metadata channel (SkillResult.metadata)
        er.metadata["n_0"] = {"entity": "folder 'hello'"}
        outcome = MissionOrchestrator._build_outcome(plan, er)
        assert outcome.active_entity == "folder 'hello'"

    def test_empty_results(self):
        plan = self._make_plan([
            MissionNode(id="n_0", skill="fs.create_folder", inputs={}),
        ])
        er = _make_exec_result()
        outcome = MissionOrchestrator._build_outcome(plan, er)
        assert outcome.nodes_executed == []
        # n_0 has no status at all → it's not in any set
        # This is a "no nodes were scheduled" case
        assert outcome.artifacts == {}
        assert outcome.visible_lists == {}


class TestOutcomeOnConversationFrame:
    """Verify outcomes are appended to ConversationFrame."""

    def test_outcomes_starts_empty(self):
        frame = ConversationFrame()
        assert frame.outcomes == []

    def test_outcome_appended(self):
        frame = ConversationFrame()
        outcome = MissionOutcome(
            mission_id="m_1",
            nodes_executed=["n_0"],
            nodes_skipped=[],
        )
        frame.outcomes.append(outcome)
        assert len(frame.outcomes) == 1
        assert frame.outcomes[0].mission_id == "m_1"

    def test_multiple_outcomes_ordered(self):
        frame = ConversationFrame()
        for i in range(3):
            frame.outcomes.append(MissionOutcome(
                mission_id=f"m_{i}",
                nodes_executed=[],
                nodes_skipped=[],
            ))
        assert [o.mission_id for o in frame.outcomes] == ["m_0", "m_1", "m_2"]
