# tests/test_integration_steps_3_4.py

"""
Tests for integration Steps 3 and 4:
- Step 3: context_provider.build_from_cognitive()
- Step 4: frame.GoalState version + refine()
"""

import pytest
import time

from cortex.context_provider import (
    ContextProvider,
    SimpleContextProvider,
    RetrievalContextProvider,
)
from execution.cognitive_context import (
    CognitiveContext,
    GoalState as CognitiveGoalState,
    ExecutionState,
    Commitment,
)
from conversation.frame import GoalState as FrameGoalState


# ─────────────────────────────────────────────────────────────
# Step 3: build_from_cognitive
# ─────────────────────────────────────────────────────────────

class TestBuildFromCognitive:

    def _make_ctx(
        self,
        required=None,
        achieved=None,
        attempts=None,
        commitments=None,
    ):
        gs = CognitiveGoalState(
            original_query="test query",
            required_outcomes=["send", "read"] if required is None else required,
            achieved_outcomes=[] if achieved is None else achieved,
        )
        es = ExecutionState()
        if attempts:
            es.attempt_history = attempts
        if commitments:
            es.commitments = commitments
        return CognitiveContext(goal=gs, execution=es)

    def test_abc_default_returns_empty(self):
        """ContextProvider ABC default returns empty string."""
        class TestProvider(ContextProvider):
            def build_context(self, query, conversation, world_state):
                return ""
        provider = TestProvider()
        ctx = self._make_ctx()
        assert provider.build_from_cognitive(ctx) == ""

    def test_simple_provider_returns_empty(self):
        """SimpleContextProvider inherits no-op from ABC."""
        provider = SimpleContextProvider()
        ctx = self._make_ctx()
        assert provider.build_from_cognitive(ctx) == ""

    def test_retrieval_returns_pending_outcomes(self):
        provider = RetrievalContextProvider()
        ctx = self._make_ctx(
            required=["send", "read"],
            achieved=["read"],
        )
        result = provider.build_from_cognitive(ctx)
        assert "Pending" in result
        assert "send" in result
        assert "read" not in result  # already achieved

    def test_retrieval_returns_recent_attempts(self):
        provider = RetrievalContextProvider()
        ctx = self._make_ctx(attempts=[
            {"skill": "fs.search_file", "result": "not_found"},
            {"skill": "fs.list_directory", "result": "success"},
        ])
        result = provider.build_from_cognitive(ctx)
        assert "Recent attempts" in result
        assert "search_file" in result
        assert "list_directory" in result

    def test_retrieval_returns_commitments(self):
        provider = RetrievalContextProvider()
        ctx = self._make_ctx(commitments={
            "target_file": Commitment(
                key="target_file",
                value="report_v2.txt",
            ),
        })
        result = provider.build_from_cognitive(ctx)
        assert "Commitments" in result
        assert "report_v2.txt" in result

    def test_retrieval_empty_ctx_returns_empty(self):
        provider = RetrievalContextProvider()
        ctx = self._make_ctx(required=[], achieved=[])
        result = provider.build_from_cognitive(ctx)
        assert result == ""

    def test_retrieval_bounded_output(self):
        """Output must be <= 800 chars."""
        provider = RetrievalContextProvider()
        # Create many attempts to push over budget
        big_attempts = [
            {"skill": f"fs.some_long_skill_name_{i}", "result": "fail" * 20}
            for i in range(50)
        ]
        ctx = self._make_ctx(
            required=[f"outcome_{i}" for i in range(20)],
            attempts=big_attempts,
        )
        result = provider.build_from_cognitive(ctx)
        assert len(result) <= 800

    def test_attempts_limited_to_last_3(self):
        provider = RetrievalContextProvider()
        ctx = self._make_ctx(attempts=[
            {"skill": f"skill_{i}", "result": "ok"} for i in range(10)
        ])
        result = provider.build_from_cognitive(ctx)
        # Only last 3 should appear
        assert "skill_7" in result
        assert "skill_8" in result
        assert "skill_9" in result
        assert "skill_0" not in result


# ─────────────────────────────────────────────────────────────
# Step 4: frame.GoalState versioning
# ─────────────────────────────────────────────────────────────

class TestFrameGoalStateVersioning:

    def _make_goal(self, desc="Send the report"):
        return FrameGoalState(id="g1", description=desc)

    def test_default_version_is_1(self):
        goal = self._make_goal()
        assert goal.version == 1
        assert goal.refinement_history == []

    def test_refine_bumps_version(self):
        goal = self._make_goal()
        goal.refine("Send the CORRECT report")
        assert goal.version == 2
        assert goal.description == "Send the CORRECT report"

    def test_refine_records_history(self):
        goal = self._make_goal("Send the report")
        goal.refine("Send the CORRECT report")
        assert len(goal.refinement_history) == 1
        entry = goal.refinement_history[0]
        assert entry["version"] == 1
        assert entry["old_description"] == "Send the report"
        assert entry["new_description"] == "Send the CORRECT report"
        assert "refined_at" in entry

    def test_refine_preserves_status(self):
        goal = self._make_goal()
        goal.status = "active"
        goal.progress = 0.5
        goal.refine("Updated description")
        assert goal.status == "active"
        assert goal.progress == 0.5

    def test_multiple_refinements(self):
        goal = self._make_goal("v1")
        goal.refine("v2")
        goal.refine("v3")
        goal.refine("v4")
        assert goal.version == 4
        assert goal.description == "v4"
        assert len(goal.refinement_history) == 3
        # History should be v1→v2, v2→v3, v3→v4
        assert goal.refinement_history[0]["old_description"] == "v1"
        assert goal.refinement_history[2]["old_description"] == "v3"

    def test_refine_updates_timestamp(self):
        goal = self._make_goal()
        old_ts = goal.updated_at
        time.sleep(0.01)
        goal.refine("new")
        assert goal.updated_at > old_ts

    def test_backward_compat_no_version(self):
        """Goals created without version field default to 1."""
        goal = FrameGoalState(id="g2", description="test")
        assert goal.version == 1
        assert goal.refinement_history == []
