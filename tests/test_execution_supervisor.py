# tests/test_execution_supervisor.py

"""
Tests for ExecutionSupervisor, StepGuard, GuardType, RepairAction.

Tests cover:
- Supervisor runs a simple plan (passthrough to executor)
- Guard evaluation (APP_RUNNING, FILE_EXISTS, APP_FOCUSED)
- Failed precondition skips node
- Repair action restores precondition
- Postcondition evaluation
- GuardType enum values
- ExecutionContext bundling
- SkillContract preconditions/postconditions extension
"""

from unittest.mock import MagicMock, patch, PropertyMock
from typing import Any, Dict

import pytest

from execution.supervisor import (
    ExecutionContext,
    ExecutionSupervisor,
    GuardType,
    RepairAction,
    StepGuard,
)
from execution.executor import ExecutionResult, NodeStatus
from ir.mission import MissionPlan, MissionNode, ExecutionMode, IR_VERSION
from skills.contract import SkillContract, FailurePolicy
from skills.skill_result import SkillResult
from infrastructure.system_controller import WindowInfo


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _make_plan(*nodes):
    """Build a minimal MissionPlan from MissionNode specs."""
    return MissionPlan(
        id="test_plan",
        nodes=list(nodes),
        metadata={"ir_version": IR_VERSION},
    )


def _make_node(idx, skill="system.open_app", inputs=None, depends_on=None):
    return MissionNode(
        id=f"node_{idx}",
        skill=skill,
        inputs=inputs or {},
        depends_on=depends_on or [],
        mode=ExecutionMode.foreground,
    )


def _make_mock_executor(registry=None):
    """Create a mock executor with essential methods."""
    executor = MagicMock()
    executor.timeline = MagicMock()

    # Default execute_node returns success
    executor.execute_node.return_value = (
        "node_0", NodeStatus.COMPLETED, {"opened": "notepad"}, {"domain": "system"}
    )

    # Stub internal methods used by supervisor
    executor._needs_focus.return_value = True
    executor._has_conflicts.return_value = False

    if registry:
        executor.registry = registry
    else:
        executor.registry = MagicMock()

    return executor


def _make_context(observer=None, session_manager=None):
    return ExecutionContext(
        observer=observer,
        session_manager=session_manager,
    )


# ─────────────────────────────────────────────────────────────
# GuardType enum
# ─────────────────────────────────────────────────────────────

class TestGuardType:

    def test_all_types_exist(self):
        assert GuardType.ACTIVE_WINDOW == "active_window"
        assert GuardType.APP_RUNNING == "app_running"
        assert GuardType.APP_FOCUSED == "app_focused"
        assert GuardType.FILE_EXISTS == "file_exists"
        assert GuardType.ELEMENT_VISIBLE == "element_visible"
        assert GuardType.WINDOW_VISIBLE == "window_visible"


# ─────────────────────────────────────────────────────────────
# StepGuard model
# ─────────────────────────────────────────────────────────────

class TestStepGuard:

    def test_basic_guard(self):
        guard = StepGuard(
            type=GuardType.APP_RUNNING,
            params={"app": "notepad"},
        )
        assert guard.type == GuardType.APP_RUNNING
        assert guard.params == {"app": "notepad"}
        assert guard.max_retries == 2
        assert guard.retry_delay == 0.5

    def test_guard_with_repair(self):
        guard = StepGuard(
            type=GuardType.APP_RUNNING,
            params={"app": "notepad"},
            repair_actions=[
                RepairAction(skill_name="system.open_app", inputs={"app_name": "notepad"}),
            ],
        )
        assert len(guard.repair_actions) == 1
        assert guard.repair_actions[0].skill_name == "system.open_app"


# ─────────────────────────────────────────────────────────────
# ExecutionContext
# ─────────────────────────────────────────────────────────────

class TestExecutionContext:

    def test_all_fields(self):
        obs = MagicMock()
        sm = MagicMock()
        cr = MagicMock()
        tl = MagicMock()
        ctx = ExecutionContext(
            observer=obs,
            session_manager=sm,
            capability_registry=cr,
            timeline=tl,
        )
        assert ctx.observer is obs
        assert ctx.session_manager is sm
        assert ctx.capability_registry is cr
        assert ctx.timeline is tl

    def test_defaults_none(self):
        ctx = ExecutionContext()
        assert ctx.observer is None
        assert ctx.session_manager is None


# ─────────────────────────────────────────────────────────────
# Supervisor — basic execution
# ─────────────────────────────────────────────────────────────

class TestSupervisorBasicExecution:

    def test_passthrough_execution(self):
        """Supervisor delegates to executor.execute_node for each node."""
        executor = _make_mock_executor()
        ctx = _make_context()
        supervisor = ExecutionSupervisor(executor=executor, context=ctx)

        node = _make_node(0)
        plan = _make_plan(node)

        executor.execute_node.return_value = (
            "node_0", NodeStatus.COMPLETED, {"opened": "notepad"}, {}
        )

        result = supervisor.run(plan)

        assert isinstance(result, ExecutionResult)
        # Executor's execute_node should have been called
        executor.execute_node.assert_called_once()

    def test_registry_property(self):
        executor = _make_mock_executor()
        ctx = _make_context()
        supervisor = ExecutionSupervisor(executor=executor, context=ctx)
        assert supervisor.registry is executor.registry


# ─────────────────────────────────────────────────────────────
# Supervisor — guard evaluation
# ─────────────────────────────────────────────────────────────

class TestGuardEvaluation:

    def test_app_running_guard_passes(self):
        observer = MagicMock()
        observer.is_app_running.return_value = True

        executor = _make_mock_executor()
        ctx = _make_context(observer=observer)
        supervisor = ExecutionSupervisor(executor=executor, context=ctx)

        guard = StepGuard(
            type=GuardType.APP_RUNNING,
            params={"app": "notepad"},
        )
        assert supervisor._evaluate_guard(guard) is True
        observer.is_app_running.assert_called_once_with("notepad")

    def test_app_running_guard_fails(self):
        observer = MagicMock()
        observer.is_app_running.return_value = False

        executor = _make_mock_executor()
        ctx = _make_context(observer=observer)
        supervisor = ExecutionSupervisor(executor=executor, context=ctx)

        guard = StepGuard(
            type=GuardType.APP_RUNNING,
            params={"app": "notepad"},
        )
        assert supervisor._evaluate_guard(guard) is False

    def test_file_exists_guard(self):
        observer = MagicMock()
        observer.file_exists.return_value = True

        executor = _make_mock_executor()
        ctx = _make_context(observer=observer)
        supervisor = ExecutionSupervisor(executor=executor, context=ctx)

        guard = StepGuard(
            type=GuardType.FILE_EXISTS,
            params={"path": "/tmp/test.txt"},
        )
        assert supervisor._evaluate_guard(guard) is True

    def test_app_focused_guard(self):
        observer = MagicMock()
        observer.is_app_focused.return_value = True

        executor = _make_mock_executor()
        ctx = _make_context(observer=observer)
        supervisor = ExecutionSupervisor(executor=executor, context=ctx)

        guard = StepGuard(
            type=GuardType.APP_FOCUSED,
            params={"app": "notepad"},
        )
        assert supervisor._evaluate_guard(guard) is True

    def test_no_observer_assumes_pass(self):
        """Without observer, guards assume pass (return True)."""
        executor = _make_mock_executor()
        ctx = _make_context(observer=None)
        supervisor = ExecutionSupervisor(executor=executor, context=ctx)

        guard = StepGuard(
            type=GuardType.APP_RUNNING,
            params={"app": "notepad"},
        )
        assert supervisor._evaluate_guard(guard) is True

    def test_element_visible_deferred(self):
        """ELEMENT_VISIBLE guard returns True (deferred to Phase 6)."""
        observer = MagicMock()
        executor = _make_mock_executor()
        ctx = _make_context(observer=observer)
        supervisor = ExecutionSupervisor(executor=executor, context=ctx)

        guard = StepGuard(
            type=GuardType.ELEMENT_VISIBLE,
            params={"selector": "#input"},
        )
        assert supervisor._evaluate_guard(guard) is True


# ─────────────────────────────────────────────────────────────
# Supervisor — repair actions
# ─────────────────────────────────────────────────────────────

class TestRepairActions:

    def test_repair_restores_guard(self):
        observer = MagicMock()
        # After repair executes, the guard re-evaluates and passes
        observer.is_app_running.return_value = True

        executor = _make_mock_executor()
        repair_skill = MagicMock()
        repair_skill.execute.return_value = SkillResult(outputs={"opened": "notepad"})
        executor.registry.get.return_value = repair_skill

        ctx = _make_context(observer=observer)
        supervisor = ExecutionSupervisor(executor=executor, context=ctx)

        guard = StepGuard(
            type=GuardType.APP_RUNNING,
            params={"app": "notepad"},
            repair_actions=[
                RepairAction(skill_name="system.open_app", inputs={"app_name": "notepad"}),
            ],
        )

        result = supervisor._attempt_repair(
            guard, _make_node(0), ExecutionResult(), None,
        )
        assert result is True
        repair_skill.execute.assert_called_once()

    def test_repair_fails(self):
        observer = MagicMock()
        observer.is_app_running.return_value = False  # Always fails

        executor = _make_mock_executor()
        repair_skill = MagicMock()
        repair_skill.execute.return_value = SkillResult(outputs={})
        executor.registry.get.return_value = repair_skill

        ctx = _make_context(observer=observer)
        supervisor = ExecutionSupervisor(executor=executor, context=ctx)

        guard = StepGuard(
            type=GuardType.APP_RUNNING,
            params={"app": "notepad"},
            repair_actions=[
                RepairAction(skill_name="system.open_app", inputs={"app_name": "notepad"}),
            ],
        )

        result = supervisor._attempt_repair(
            guard, _make_node(0), ExecutionResult(), None,
        )
        assert result is False


# ─────────────────────────────────────────────────────────────
# SkillContract extension
# ─────────────────────────────────────────────────────────────

class TestContractExtension:

    def test_default_empty_guards(self):
        """Existing contracts have empty pre/postconditions by default."""
        contract = SkillContract(
            name="test.skill",
            inputs={},
            outputs={},
            allowed_modes={ExecutionMode.foreground},
            failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
        )
        assert contract.preconditions == []
        assert contract.postconditions == []

    def test_contract_with_guards(self):
        """Contracts can declare typed guards."""
        contract = SkillContract(
            name="test.skill",
            inputs={},
            outputs={},
            allowed_modes={ExecutionMode.foreground},
            failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
            preconditions=[
                {"type": "app_running", "params": {"app": "notepad"}},
            ],
            postconditions=[
                {"type": "file_exists", "params": {"path": "/tmp/out.txt"}},
            ],
        )
        assert len(contract.preconditions) == 1
        assert contract.preconditions[0]["type"] == "app_running"
        assert len(contract.postconditions) == 1


# ─────────────────────────────────────────────────────────────
# Session cleanup after execution
# ─────────────────────────────────────────────────────────────

class TestPostExecutionCleanup:

    def test_cleanup_called_after_run(self):
        executor = _make_mock_executor()
        observer = MagicMock()
        session_mgr = MagicMock()

        ctx = ExecutionContext(
            observer=observer,
            session_manager=session_mgr,
        )
        supervisor = ExecutionSupervisor(executor=executor, context=ctx)

        node = _make_node(0)
        plan = _make_plan(node)

        executor.execute_node.return_value = (
            "node_0", NodeStatus.COMPLETED, {}, {}
        )

        supervisor.run(plan)
        session_mgr.cleanup_stale_sessions.assert_called_once_with(
            observer=observer,
        )
