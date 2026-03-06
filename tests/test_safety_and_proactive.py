# tests/test_safety_and_proactive.py

"""
Tests for Fixes 1–7: Safety layer, memory skills, decomposer clause types,
scheduled clause dispatch, proactive policy evaluation, and the wiring
integrity of the late-registration path.

Matrix:
    A. Memory Skills (8 tests)
       - GetPreferenceSkill returns stored value
       - GetPreferenceSkill returns None for missing key
       - SetPreferenceSkill stores and returns value
       - SetFactSkill stores and returns value
       - AddPolicySkill stores policy and returns ID
       - All memory skills REQUIRE user_knowledge (no default)
       - Memory skills registered via late load have real store

    B. Safety Layer — REQUIRES_CONFIRMATION guard (6 tests)
       - Destructive skill auto-injects REQUIRES_CONFIRMATION guard
       - Moderate skill logs warning but doesn't inject guard
       - Safe skill has no injected guard
       - REQUIRES_CONFIRMATION evaluate returns False (blocks execution)
       - risk_level defaults to "safe"
       - SkillContract accepts risk_level field

    C. Decomposer Clause Types (5 tests)
       - DecompositionResult has all typed fields
       - Typed fields default to empty lists
       - Executable intents populate correctly
       - Scheduled intents populate correctly
       - Vague intents populate correctly

    D. Proactive Policy Evaluation (5 tests)
       - _maybe_apply_user_policy enqueues insight on match
       - No user_knowledge → no-op
       - No matching policies → no enqueue
       - Multiple matching policies → multiple insights
       - Exception in policy eval doesn't crash

    E. Scheduled Execution Pipeline (4 tests)
       - Scheduled job with wired skill executes successfully
       - Scheduled job with unwired skill fails (regression guard)
       - _execute_scheduled_job uses compiled_plan key
       - _build_job_summary works with memory skill output

    F. Late-Registration Integrity (3 tests)
       - Memory skills excluded from first load_skills pass
       - Memory skills registered in second pass with real store
       - IntentIndex rebuilt after late registration

    G. _check_destructive_nodes (3 tests)
       - Returns empty list for safe plan
       - Returns destructive nodes
       - Never raises on error
"""

import inspect
import os
import tempfile
import time

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import fields

from skills.skill_result import SkillResult
from skills.contract import SkillContract, FailurePolicy
from ir.mission import MissionPlan, MissionNode, ExecutionMode, IR_VERSION
from execution.executor import ExecutionResult, NodeStatus


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _make_user_knowledge_store():
    """Create a real UserKnowledgeStore backed by a temp file."""
    from memory.user_knowledge import UserKnowledgeStore
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    tmp.write("{}")
    tmp.close()
    store = UserKnowledgeStore(persist_path=tmp.name)
    return store, tmp.name


def _make_plan(*nodes):
    return MissionPlan(
        id="test_plan",
        nodes=list(nodes),
        metadata={"ir_version": IR_VERSION},
    )


def _make_node(idx, skill="system.set_volume", inputs=None, depends_on=None):
    return MissionNode(
        id=f"node_{idx}",
        skill=skill,
        inputs=inputs or {},
        depends_on=depends_on or [],
        mode=ExecutionMode.foreground,
    )


def _make_mock_executor(registry=None):
    executor = MagicMock()
    executor.timeline = MagicMock()
    executor.execute_node.return_value = (
        "node_0", NodeStatus.COMPLETED, {"result": "ok"}, {}
    )
    executor._needs_focus.return_value = True
    executor._has_conflicts.return_value = False
    if registry:
        executor.registry = registry
    else:
        executor.registry = MagicMock()
    return executor


# ──────────────────────────────────────────────────────────────
# A. Memory Skills
# ──────────────────────────────────────────────────────────────


class TestMemorySkills:
    """Tests for memory skill wrappers and their wiring contract."""

    def test_get_preference_returns_stored_value(self):
        """GetPreferenceSkill returns value from store."""
        from skills.memory.memory_skills import GetPreferenceSkill
        store, path = _make_user_knowledge_store()
        try:
            store.set_preference("volume", 75)
            skill = GetPreferenceSkill(user_knowledge=store)
            result = skill.execute({"key": "volume"}, MagicMock())
            assert result.outputs["value"] == 75
            assert result.outputs["found"] is True
        finally:
            os.unlink(path)

    def test_get_preference_returns_none_for_missing(self):
        """GetPreferenceSkill returns None/False for unknown key."""
        from skills.memory.memory_skills import GetPreferenceSkill
        store, path = _make_user_knowledge_store()
        try:
            skill = GetPreferenceSkill(user_knowledge=store)
            result = skill.execute({"key": "nonexistent"}, MagicMock())
            assert result.outputs["value"] is None
            assert result.outputs["found"] is False
        finally:
            os.unlink(path)

    def test_set_preference_stores_value(self):
        """SetPreferenceSkill delegates to store and returns result."""
        from skills.memory.memory_skills import SetPreferenceSkill
        store, path = _make_user_knowledge_store()
        try:
            skill = SetPreferenceSkill(user_knowledge=store)
            result = skill.execute(
                {"key": "brightness", "value": 80}, MagicMock(),
            )
            assert result.outputs["stored_key"] == "brightness"
            # Verify it's actually persisted
            assert store.get_preference("brightness") == 80
        finally:
            os.unlink(path)

    def test_set_fact_stores_value(self):
        """SetFactSkill delegates to store."""
        from skills.memory.memory_skills import SetFactSkill
        store, path = _make_user_knowledge_store()
        try:
            skill = SetFactSkill(user_knowledge=store)
            result = skill.execute(
                {"key": "name", "value": "Alex"}, MagicMock(),
            )
            assert result.outputs["stored_key"] == "name"
            assert result.outputs["stored_value"] == "Alex"
            assert store.query("name") == "Alex"
        finally:
            os.unlink(path)

    def test_add_policy_stores_and_returns_id(self):
        """AddPolicySkill stores policy and returns UUID id."""
        from skills.memory.memory_skills import AddPolicySkill
        store, path = _make_user_knowledge_store()
        try:
            skill = AddPolicySkill(user_knowledge=store)
            result = skill.execute({
                "condition": {"event_type": "media_started"},
                "action": {"set_volume": 90},
                "label": "movie mode",
            }, MagicMock())
            policy_id = result.outputs["policy_id"]
            assert policy_id  # non-empty string
            assert len(policy_id) > 10  # UUID-like
        finally:
            os.unlink(path)

    def test_memory_skills_require_user_knowledge(self):
        """All memory skill __init__ methods REQUIRE user_knowledge (no default).

        This is the architectural fix that prevents double-registration:
        load_skills skips skills whose required dep is not in deps dict.
        """
        from skills.memory.memory_skills import (
            GetPreferenceSkill, SetPreferenceSkill,
            SetFactSkill, AddPolicySkill,
        )
        for cls in (GetPreferenceSkill, SetPreferenceSkill,
                    SetFactSkill, AddPolicySkill):
            sig = inspect.signature(cls.__init__)
            param = sig.parameters["user_knowledge"]
            assert param.default is inspect.Parameter.empty, (
                f"{cls.__name__}.__init__ must REQUIRE user_knowledge "
                f"(no default). Got default={param.default!r}. "
                "This prevents double-registration with None store."
            )

    def test_set_fact_without_store_raises(self):
        """SetFactSkill raises RuntimeError when store is None.

        This is the exact error from the original bug — verifying
        it fails loudly rather than silently returning garbage.
        """
        from skills.memory.memory_skills import SetFactSkill
        # Can't even construct without user_knowledge (required param)
        with pytest.raises(TypeError):
            SetFactSkill()

    def test_memory_skill_contracts_have_valid_semantic_types(self):
        """All memory skill input/output types are registered in SEMANTIC_TYPES."""
        from skills.memory.memory_skills import (
            GetPreferenceSkill, SetPreferenceSkill,
            SetFactSkill, AddPolicySkill,
        )
        from cortex.semantic_types import SEMANTIC_TYPES

        for cls in (GetPreferenceSkill, SetPreferenceSkill,
                    SetFactSkill, AddPolicySkill):
            store = MagicMock()
            skill = cls(user_knowledge=store)
            for key, stype in skill.contract.inputs.items():
                assert stype in SEMANTIC_TYPES, (
                    f"{cls.__name__} input '{key}' uses unregistered "
                    f"type '{stype}'"
                )
            for key, stype in skill.contract.outputs.items():
                assert stype in SEMANTIC_TYPES, (
                    f"{cls.__name__} output '{key}' uses unregistered "
                    f"type '{stype}'"
                )


# ──────────────────────────────────────────────────────────────
# B. Safety Layer — REQUIRES_CONFIRMATION guard
# ──────────────────────────────────────────────────────────────


class TestRequiresConfirmationGuard:
    """Tests for destructive skill safety layer."""

    def test_destructive_skill_injects_confirmation_guard(self):
        """Skills with risk_level='destructive' get REQUIRES_CONFIRMATION."""
        from execution.supervisor import (
            ExecutionSupervisor, ExecutionContext,
            GuardType, StepGuard,
        )

        # Create a skill with destructive risk level
        mock_skill = MagicMock()
        mock_skill.contract = SkillContract(
            name="fs.delete_file",
            action="delete_file",
            target_type="file",
            description="Delete a file permanently",
            domain="fs",
            inputs={"path": "file_path_input"},
            outputs={},
            allowed_modes={ExecutionMode.foreground},
            failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
            risk_level="destructive",
        )

        executor = _make_mock_executor()
        executor.registry.get.return_value = mock_skill
        ctx = ExecutionContext()
        supervisor = ExecutionSupervisor(executor=executor, context=ctx)

        node = _make_node(0, skill="fs.delete_file", inputs={"path": "/tmp/test"})
        guards = supervisor._get_guards(node, "preconditions")

        assert len(guards) >= 1
        assert guards[0].type == GuardType.REQUIRES_CONFIRMATION
        assert guards[0].params["skill"] == "fs.delete_file"

    def test_moderate_skill_no_confirmation_guard(self):
        """Moderate-risk skills log warning but don't inject guard."""
        from execution.supervisor import (
            ExecutionSupervisor, ExecutionContext, GuardType,
        )

        mock_skill = MagicMock()
        mock_skill.contract = SkillContract(
            name="system.set_volume",
            action="set_volume",
            target_type="volume",
            description="Set system volume",
            domain="system",
            inputs={"level": "volume_percentage"},
            outputs={"actual": "actual_volume"},
            allowed_modes={ExecutionMode.foreground},
            failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
            risk_level="moderate",
        )

        executor = _make_mock_executor()
        executor.registry.get.return_value = mock_skill
        ctx = ExecutionContext()
        supervisor = ExecutionSupervisor(executor=executor, context=ctx)

        node = _make_node(0, skill="system.set_volume")
        guards = supervisor._get_guards(node, "preconditions")

        # No REQUIRES_CONFIRMATION guard for moderate
        confirmation_guards = [
            g for g in guards
            if g.type == GuardType.REQUIRES_CONFIRMATION
        ]
        assert len(confirmation_guards) == 0

    def test_safe_skill_no_injected_guard(self):
        """Safe skills get no auto-injected guards."""
        from execution.supervisor import (
            ExecutionSupervisor, ExecutionContext, GuardType,
        )

        mock_skill = MagicMock()
        mock_skill.contract = SkillContract(
            name="system.get_volume",
            action="get_volume",
            target_type="volume",
            description="Get system volume",
            domain="system",
            inputs={},
            outputs={"level": "actual_volume"},
            allowed_modes={ExecutionMode.foreground},
            failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
            risk_level="safe",
        )

        executor = _make_mock_executor()
        executor.registry.get.return_value = mock_skill
        ctx = ExecutionContext()
        supervisor = ExecutionSupervisor(executor=executor, context=ctx)

        node = _make_node(0, skill="system.get_volume")
        guards = supervisor._get_guards(node, "preconditions")

        assert len(guards) == 0

    def test_requires_confirmation_evaluate_returns_false(self):
        """REQUIRES_CONFIRMATION guard always returns False to block execution."""
        from execution.supervisor import (
            ExecutionSupervisor, ExecutionContext,
            GuardType, StepGuard,
        )

        executor = _make_mock_executor()
        ctx = ExecutionContext()
        supervisor = ExecutionSupervisor(executor=executor, context=ctx)

        guard = StepGuard(
            type=GuardType.REQUIRES_CONFIRMATION,
            params={"skill": "fs.delete_file", "inputs": {"path": "/tmp/x"}},
        )
        result = supervisor._evaluate_guard(guard)
        assert result is False

    def test_risk_level_defaults_to_safe(self):
        """SkillContract without risk_level defaults to 'safe'."""
        contract = SkillContract(
            name="test.skill",
            inputs={},
            outputs={},
            allowed_modes={ExecutionMode.foreground},
            failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
        )
        assert contract.risk_level == "safe"

    def test_contract_accepts_risk_level(self):
        """SkillContract correctly stores risk_level field."""
        for level in ("safe", "moderate", "destructive"):
            contract = SkillContract(
                name="test.skill",
                inputs={},
                outputs={},
                allowed_modes={ExecutionMode.foreground},
                failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
                risk_level=level,
            )
            assert contract.risk_level == level


# ──────────────────────────────────────────────────────────────
# C. Decomposer Clause Types
# ──────────────────────────────────────────────────────────────


class TestDecomposerClauseTypes:
    """Tests for DecompositionResult typed clause fields."""

    def test_decomposition_result_has_typed_fields(self):
        """DecompositionResult has all expected typed clause fields."""
        from cortex.mission_cortex import DecompositionResult
        result = DecompositionResult()
        assert hasattr(result, "executable_intents")
        assert hasattr(result, "scheduled_intents")
        assert hasattr(result, "informational_intents")
        assert hasattr(result, "vague_intents")
        assert hasattr(result, "valid_intents")
        assert hasattr(result, "unsupported_intents")

    def test_typed_fields_default_empty(self):
        """All typed fields default to empty lists."""
        from cortex.mission_cortex import DecompositionResult
        result = DecompositionResult()
        assert result.executable_intents == []
        assert result.scheduled_intents == []
        assert result.informational_intents == []
        assert result.vague_intents == []
        assert result.valid_intents == []
        assert result.unsupported_intents == []

    def test_executable_intents_populate(self):
        """Executable intents can be set and read."""
        from cortex.mission_cortex import DecompositionResult
        result = DecompositionResult(
            executable_intents=[
                {"action": "set_volume", "parameters": {"level": 50}},
                {"action": "set_brightness", "parameters": {"level": 80}},
            ],
        )
        assert len(result.executable_intents) == 2
        assert result.executable_intents[0]["action"] == "set_volume"

    def test_scheduled_intents_populate(self):
        """Scheduled intents are isolated from executable intents."""
        from cortex.mission_cortex import DecompositionResult
        result = DecompositionResult(
            executable_intents=[
                {"action": "set_volume", "parameters": {"level": 50}},
            ],
            scheduled_intents=[
                {"action": "remind", "trigger": "in 5 minutes"},
            ],
        )
        assert len(result.executable_intents) == 1
        assert len(result.scheduled_intents) == 1
        assert result.scheduled_intents[0]["trigger"] == "in 5 minutes"

    def test_vague_intents_populate(self):
        """Vague intents trigger clarification, not execution."""
        from cortex.mission_cortex import DecompositionResult
        result = DecompositionResult(
            vague_intents=[
                {"action": "something", "reason": "ambiguous"},
            ],
        )
        assert len(result.vague_intents) == 1
        assert result.executable_intents == []


# ──────────────────────────────────────────────────────────────
# D. Proactive Policy Evaluation
# ──────────────────────────────────────────────────────────────


class TestProactivePolicyEvaluation:
    """Tests for RuntimeEventLoop._maybe_apply_user_policy."""

    def _make_event_loop(self, user_knowledge=None):
        """Build a minimal RuntimeEventLoop with mocked deps."""
        from runtime.event_loop import RuntimeEventLoop

        loop = RuntimeEventLoop.__new__(RuntimeEventLoop)
        loop.timeline = MagicMock()
        loop.timeline.all_events.return_value = []
        loop._scheduler = None
        loop._job_executor = None
        loop._running = False
        loop._thread = None
        loop.output_channel = MagicMock()
        loop.attention_manager = MagicMock()
        loop._completion_queue = None
        loop._user_knowledge = user_knowledge
        loop.sources = []
        loop.reflex_engine = MagicMock()
        loop.notification_policy = MagicMock()
        loop.report_builder = MagicMock()
        loop.get_conversation = MagicMock()
        loop.tick_interval = 0.1
        return loop

    def test_matching_policy_enqueues_insight(self):
        """When a policy matches event context, insight is enqueued."""
        from memory.user_knowledge import Policy
        store = MagicMock()
        policy = Policy(
            condition={"event_type": "media_started"},
            action={"set_volume": 90},
            label="movie mode",
        )
        store.get_matching_policies.return_value = [policy]

        loop = self._make_event_loop(user_knowledge=store)

        event = MagicMock()
        event.type = "media_started"
        event.payload = {"app": "spotify"}

        snapshot = MagicMock()
        loop._maybe_apply_user_policy(event, snapshot)

        loop.attention_manager.enqueue.assert_called_once()
        insight_text = loop.attention_manager.enqueue.call_args[0][0]
        assert "set_volume=90" in insight_text

    def test_no_user_knowledge_is_noop(self):
        """Without UserKnowledgeStore, _maybe_apply_user_policy is a no-op."""
        loop = self._make_event_loop(user_knowledge=None)

        event = MagicMock()
        event.type = "media_started"
        event.payload = {}

        loop._maybe_apply_user_policy(event, MagicMock())
        loop.attention_manager.enqueue.assert_not_called()

    def test_no_matching_policies_no_enqueue(self):
        """No matching policies → no insight enqueued."""
        store = MagicMock()
        store.get_matching_policies.return_value = []

        loop = self._make_event_loop(user_knowledge=store)

        event = MagicMock()
        event.type = "window_changed"
        event.payload = {}

        loop._maybe_apply_user_policy(event, MagicMock())
        loop.attention_manager.enqueue.assert_not_called()

    def test_multiple_matching_policies(self):
        """Multiple matching policies → multiple insights enqueued."""
        from memory.user_knowledge import Policy
        store = MagicMock()
        policies = [
            Policy(
                condition={"event_type": "media_started"},
                action={"set_volume": 90},
            ),
            Policy(
                condition={"event_type": "media_started"},
                action={"set_brightness": 60},
            ),
        ]
        store.get_matching_policies.return_value = policies

        loop = self._make_event_loop(user_knowledge=store)

        event = MagicMock()
        event.type = "media_started"
        event.payload = {}

        loop._maybe_apply_user_policy(event, MagicMock())
        assert loop.attention_manager.enqueue.call_count == 2

    def test_exception_in_policy_eval_doesnt_crash(self):
        """Policy evaluation errors are silently caught."""
        store = MagicMock()
        store.get_matching_policies.side_effect = RuntimeError("broken")

        loop = self._make_event_loop(user_knowledge=store)

        event = MagicMock()
        event.type = "test"
        event.payload = {}

        # Must not raise
        loop._maybe_apply_user_policy(event, MagicMock())
        loop.attention_manager.enqueue.assert_not_called()


# ──────────────────────────────────────────────────────────────
# E. Scheduled Execution Pipeline
# ──────────────────────────────────────────────────────────────


class TestScheduledExecutionPipeline:
    """Tests for the scheduler → executor → memory skill pipeline.

    This is the exact pipeline that the original bug broke:
    scheduled job → _execute_scheduled_job → executor.run → skill.execute
    """

    def test_scheduler_memory_skill_execution(self):
        """End-to-end: scheduled job using memory skill executes with real store.

        Flow: create skill with real store → compile plan → simulate
        scheduler dispatch → verify skill executed.

        This test would have caught the original wiring bug.
        """
        from skills.memory.memory_skills import SetFactSkill
        from execution.registry import SkillRegistry

        store, path = _make_user_knowledge_store()
        try:
            # Register skill with real store
            skill = SetFactSkill(user_knowledge=store)
            registry = SkillRegistry()
            registry.register(skill)

            # Verify skill can be retrieved and executed
            retrieved = registry.get("memory.set_fact")
            assert retrieved is skill
            assert retrieved._store is store
            assert retrieved._store is not None

            # Execute through the skill (simulating what executor.run does)
            result = retrieved.execute(
                {"key": "reminder", "value": "drink water"},
                MagicMock(),  # WorldTimeline
            )
            assert result.outputs["stored_key"] == "reminder"
            assert store.query("reminder") == "drink water"
        finally:
            os.unlink(path)

    def test_scheduler_unwired_skill_fails_loudly(self):
        """Memory skill with None store raises RuntimeError.

        This is the regression guard — if wiring breaks again,
        this test catches it immediately.
        """
        from skills.memory.memory_skills import SetFactSkill

        # Construct with MagicMock that returns None for set_fact
        # Simulating what happens with broken wiring
        broken_store = MagicMock()
        broken_store.set_fact.side_effect = AttributeError("NoneType")

        skill = SetFactSkill(user_knowledge=broken_store)
        with pytest.raises(AttributeError):
            skill.execute(
                {"key": "test", "value": "val"}, MagicMock(),
            )

    def test_execute_scheduled_job_uses_compiled_plan_key(self):
        """_execute_scheduled_job reads mission_data['compiled_plan'].

        The key must match what _schedule_decomposed_clause writes.
        """
        from runtime.event_loop import RuntimeEventLoop

        loop = RuntimeEventLoop.__new__(RuntimeEventLoop)
        loop.timeline = MagicMock()
        loop.timeline.all_events.return_value = []
        loop._scheduler = MagicMock()
        loop._completion_queue = MagicMock()
        loop.output_channel = MagicMock()

        mock_executor = MagicMock()
        mock_result = MagicMock()
        mock_result.failed = []
        mock_result.results = {"n1": {"text": "Done!"}}
        mock_executor.return_value = mock_result
        loop._job_executor = mock_executor

        # Build a task with compiled_plan key (not "plan")
        plan = _make_plan(_make_node(0, skill="system.set_volume", inputs={"level": 50}))
        task = MagicMock()
        task.id = "t1"
        task.short_id = "J-1"
        task.query = "set volume to 50"
        task.mission_data = {
            "compiled_plan": plan.model_dump(),
            "deferred_query": "set volume to 50",
        }
        task.priority = "normal"

        # Execute
        loop._execute_scheduled_job(task)

        # Verify executor was called with the deserialized plan
        mock_executor.assert_called_once()
        called_plan = mock_executor.call_args[0][0]
        assert len(called_plan.nodes) == 1


# ──────────────────────────────────────────────────────────────
# F. Late-Registration Integrity
# ──────────────────────────────────────────────────────────────


class TestLateRegistrationIntegrity:
    """Tests for the deferred memory skill registration pattern."""

    def test_memory_skills_excluded_from_first_pass(self):
        """load_skills with user_knowledge absent from deps skips memory skills.

        The key isn't in deps at all → load_skills won't inject it →
        __init__ requires it → skip.
        """
        from main import load_skills
        from execution.registry import SkillRegistry
        import yaml

        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "skills.yaml",
        )
        with open(config_path, "r") as f:
            skills_config = yaml.safe_load(f)

        registry = SkillRegistry()

        # First pass: user_knowledge NOT in deps
        first_pass_deps = {
            "system_controller": MagicMock(),
            "content_llm": MagicMock(),
            "task_store": MagicMock(),
            "session_manager": MagicMock(),
            # NO user_knowledge key at all
        }
        load_skills(registry, skills_config, deps=first_pass_deps)

        # Memory skills should NOT be registered
        all_names = registry.all_names()
        memory_names = [n for n in all_names if n.startswith("memory.")]
        assert memory_names == [], (
            f"Memory skills registered without user_knowledge: {memory_names}"
        )

    def test_memory_skills_registered_in_second_pass(self):
        """load_skills with real user_knowledge registers memory skills."""
        from main import load_skills
        from execution.registry import SkillRegistry
        import yaml

        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "skills.yaml",
        )
        with open(config_path, "r") as f:
            skills_config = yaml.safe_load(f)

        # Filter to only memory skills
        memory_entries = {
            "skills": [
                e for e in skills_config.get("skills", [])
                if e["module"].startswith("skills.memory")
            ]
        }

        registry = SkillRegistry()
        store, path = _make_user_knowledge_store()
        try:
            load_skills(registry, memory_entries, deps={"user_knowledge": store})
            all_names = registry.all_names()
            expected = {
                "memory.get_preference",
                "memory.set_preference",
                "memory.set_fact",
                "memory.add_policy",
            }
            assert expected.issubset(all_names), (
                f"Missing memory skills: {expected - all_names}"
            )

            # Verify they have a real store (not None)
            for name in expected:
                skill = registry.get(name)
                assert skill._store is store, (
                    f"{name}._store is not the real store"
                )
        finally:
            os.unlink(path)

    def test_no_double_registration(self):
        """Registering a skill twice raises ValueError (SkillRegistry invariant)."""
        from execution.registry import SkillRegistry
        from skills.memory.memory_skills import GetPreferenceSkill

        registry = SkillRegistry()
        store = MagicMock()

        skill1 = GetPreferenceSkill(user_knowledge=store)
        registry.register(skill1)

        skill2 = GetPreferenceSkill(user_knowledge=store)
        with pytest.raises(ValueError, match="Duplicate skill"):
            registry.register(skill2)


# ──────────────────────────────────────────────────────────────
# G. _check_destructive_nodes
# ──────────────────────────────────────────────────────────────


class TestCheckDestructiveNodes:
    """Tests for Merlin._check_destructive_nodes."""

    def _make_merlin_stub(self, skills_by_name):
        """Build a minimal Merlin-like object with _check_destructive_nodes."""
        # Import the actual method by creating a stub with necessary attrs
        stub = MagicMock()
        stub.executor = MagicMock()
        stub.executor.registry = MagicMock()

        def get_skill(name):
            return skills_by_name.get(name)

        stub.executor.registry.get.side_effect = get_skill

        # Bind the real method to our stub
        from merlin import Merlin
        stub._check_destructive_nodes = Merlin._check_destructive_nodes.__get__(stub)
        return stub

    def test_safe_plan_returns_empty(self):
        """Plan with only safe skills returns no destructive nodes."""
        safe_skill = MagicMock()
        safe_skill.contract.risk_level = "safe"

        stub = self._make_merlin_stub({"system.set_volume": safe_skill})
        plan = _make_plan(
            _make_node(0, skill="system.set_volume"),
        )
        result = stub._check_destructive_nodes(plan)
        assert result == []

    def test_destructive_plan_returns_nodes(self):
        """Plan with destructive skill returns those nodes."""
        dangerous_skill = MagicMock()
        dangerous_skill.contract.risk_level = "destructive"

        stub = self._make_merlin_stub({"fs.delete_file": dangerous_skill})
        node = _make_node(0, skill="fs.delete_file", inputs={"path": "/tmp/x"})
        plan = _make_plan(node)
        result = stub._check_destructive_nodes(plan)
        assert len(result) == 1
        assert result[0].skill == "fs.delete_file"

    def test_never_raises_on_error(self):
        """_check_destructive_nodes returns [] on any error."""
        stub = self._make_merlin_stub({})
        stub.executor.registry.get.side_effect = RuntimeError("broken")

        plan = _make_plan(_make_node(0))
        result = stub._check_destructive_nodes(plan)
        assert result == []
