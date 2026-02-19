# tests/test_event_source_attribution.py

"""
Tests for event source attribution in MissionExecutor.

Validates the fix for the race condition where background event sources
(TimeSource, MediaSource, SystemSource) emit events to the shared
WorldTimeline during skill execution, causing false contract violations.

The executor must filter events by source == skill.contract.name
before enforcing contract rules. Background events remain in the
timeline for WorldState building, proactive logic, and goal evaluation.
"""

import pytest

from ir.mission import IR_VERSION, ExecutionMode, MissionNode, MissionPlan
from execution.executor import MissionExecutor, NodeStatus
from execution.registry import SkillRegistry
from skills.base import Skill
from skills.skill_result import SkillResult
from skills.contract import SkillContract, FailurePolicy
from world.timeline import WorldTimeline


# ------------------------------------------------------------------
# Test Skills
# ------------------------------------------------------------------

class BrightnessSkill(Skill):
    """Simulates system.set_brightness — mutates_world, declares emits_events."""
    contract = SkillContract(
        name="system.set_brightness",
        description="Set brightness",
        inputs={"level": "number"},
        outputs={"actual": "number"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
        emits_events=["brightness_changed"],
        mutates_world=True,
    )

    def execute(self, inputs, world, snapshot=None):
        # Skill does NOT emit events (production behavior matches this)
        return SkillResult(outputs={"actual": inputs["level"]})


class CleanSkill(Skill):
    """Skill that does not mutate world or emit events."""
    contract = SkillContract(
        name="test.clean",
        description="Clean skill",
        inputs={},
        outputs={"status": "text"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
        emits_events=[],
        mutates_world=False,
    )

    def execute(self, inputs, world, snapshot=None):
        return SkillResult(outputs={"status": "ok"})


class CorrectEmitterSkill(Skill):
    """Skill that correctly emits declared events with contract.name source."""
    contract = SkillContract(
        name="test.correct_emitter",
        description="Correctly emits events",
        inputs={},
        outputs={"status": "text"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
        emits_events=["task_done"],
        mutates_world=True,
    )

    def execute(self, inputs, world, snapshot=None):
        world.emit(self.contract.name, "task_done", {"detail": "finished"})
        return SkillResult(outputs={"status": "done"})


class WrongSourceEmitterSkill(Skill):
    """Skill that emits with a non-contract source — should NOT be caught."""
    contract = SkillContract(
        name="test.wrong_source",
        description="Emits with wrong source",
        inputs={},
        outputs={"status": "text"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
        emits_events=[],
        mutates_world=False,
    )

    def execute(self, inputs, world, snapshot=None):
        # Emitting with a non-contract source simulates background leakage
        world.emit("time", "time_tick", {"hour": 14})
        return SkillResult(outputs={"status": "ok"})


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_executor(*skills) -> tuple:
    reg = SkillRegistry()
    for s in skills:
        reg.register(s, validate_types=False)
    tl = WorldTimeline()
    return MissionExecutor(reg, tl), tl


def _make_plan(nodes) -> MissionPlan:
    return MissionPlan(
        id="test_plan",
        nodes=nodes,
        metadata={"ir_version": IR_VERSION},
    )


def _inject_background_event(timeline, source="time", event_type="time_tick"):
    """Simulate a background source event arriving in the timeline."""
    timeline.emit(source=source, event_type=event_type, payload={"severity": "background"})


# ==================================================================
# Core Attribution Tests
# ==================================================================


class TestBackgroundEventsIgnoredForEnforcement:
    """
    Background events in the timeline must NOT trigger contract violations.
    This is the core fix for the time_tick race condition.
    """

    def test_time_tick_during_brightness_does_not_crash(self):
        """
        Exact reproduction of the reported bug:
        TimeSource emits time_tick while set_brightness executes.
        """
        executor, tl = _make_executor(BrightnessSkill())

        # Pre-inject a background event (simulates TimeSource firing mid-execution)
        _inject_background_event(tl, "time", "time_tick")

        plan = _make_plan([
            MissionNode(
                id="n1",
                skill="system.set_brightness",
                inputs={"level": 50},
            ),
        ])

        # This must NOT raise RuntimeError
        er = executor.run(plan)
        assert er.results["n1"]["actual"] == 50

    def test_foreground_window_changed_during_execution(self):
        """SystemSource foreground_window_changed must not crash skills."""
        executor, tl = _make_executor(BrightnessSkill())

        _inject_background_event(tl, "system", "foreground_window_changed")

        plan = _make_plan([
            MissionNode(
                id="n1",
                skill="system.set_brightness",
                inputs={"level": 75},
            ),
        ])

        er = executor.run(plan)
        assert er.results["n1"]["actual"] == 75

    def test_media_track_changed_during_execution(self):
        """MediaSource events must not crash skills."""
        executor, tl = _make_executor(CleanSkill())

        _inject_background_event(tl, "media", "media_track_changed")

        plan = _make_plan([
            MissionNode(id="n1", skill="test.clean"),
        ])

        er = executor.run(plan)
        assert er.results["n1"]["status"] == "ok"

    def test_multiple_background_events_during_execution(self):
        """Multiple background events from different sources — all ignored."""
        executor, tl = _make_executor(BrightnessSkill())

        _inject_background_event(tl, "time", "time_tick")
        _inject_background_event(tl, "system", "cpu_high")
        _inject_background_event(tl, "media", "media_started")

        plan = _make_plan([
            MissionNode(
                id="n1",
                skill="system.set_brightness",
                inputs={"level": 100},
            ),
        ])

        er = executor.run(plan)
        assert er.results["n1"]["actual"] == 100

    def test_mutates_world_false_not_triggered_by_background(self):
        """
        CleanSkill has mutates_world=False.
        Background events must not trigger the 'skill emitted events
        but declares mutates_world=False' error.
        """
        executor, tl = _make_executor(CleanSkill())

        _inject_background_event(tl, "time", "time_tick")
        _inject_background_event(tl, "system", "battery_low")

        plan = _make_plan([
            MissionNode(id="n1", skill="test.clean"),
        ])

        # Must NOT raise "mutates_world=False" error
        er = executor.run(plan)
        assert er.results["n1"]["status"] == "ok"


class TestBackgroundEventsPreservedInTimeline:
    """
    Background events must remain in the timeline after filtering.
    They are needed for WorldState building, proactive logic, and goals.
    """

    def test_background_events_not_dropped(self):
        """Filtering is read-only — events stay in timeline."""
        executor, tl = _make_executor(CleanSkill())

        _inject_background_event(tl, "time", "time_tick")

        plan = _make_plan([
            MissionNode(id="n1", skill="test.clean"),
        ])

        executor.run(plan)

        all_events = tl.all_events()
        time_events = [e for e in all_events if e.source == "time"]
        assert len(time_events) == 1
        assert time_events[0].type == "time_tick"

    def test_mixed_events_all_preserved(self):
        """Both background and skill events stay in timeline."""
        executor, tl = _make_executor(CorrectEmitterSkill())

        _inject_background_event(tl, "time", "hour_changed")

        plan = _make_plan([
            MissionNode(id="n1", skill="test.correct_emitter"),
        ])

        executor.run(plan)

        all_events = tl.all_events()
        types = {e.type for e in all_events}
        assert "hour_changed" in types  # background preserved
        assert "task_done" in types      # skill event preserved


class TestGenuineContractViolationsStillCaught:
    """
    Source-based filtering must NOT create false negatives.
    Genuine violations must still be caught.
    """

    def test_undeclared_skill_event_still_caught(self):
        """Skill emitting undeclared event type with correct source → RuntimeError."""

        class BadEmitter(Skill):
            contract = SkillContract(
                name="test.bad_emitter",
                description="Emits wrong type",
                inputs={},
                outputs={"status": "text"},
                allowed_modes={ExecutionMode.foreground},
                failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
                emits_events=["expected_event"],
                mutates_world=True,
            )

            def execute(self, inputs, world, snapshot=None):
                # Uses contract.name as source but wrong event type
                world.emit(self.contract.name, "unexpected_event", {})
                return SkillResult(outputs={"status": "done"})

        executor, _ = _make_executor(BadEmitter())

        plan = _make_plan([
            MissionNode(id="n1", skill="test.bad_emitter"),
        ])

        with pytest.raises(RuntimeError, match="undeclared event type"):
            executor.run(plan)

    def test_mutates_world_false_with_skill_event_still_caught(self):
        """Skill claims mutates_world=False but emits with correct source → RuntimeError."""

        class LiarSkill(Skill):
            contract = SkillContract(
                name="test.liar_skill",
                description="Lies about mutation",
                inputs={},
                outputs={"status": "text"},
                allowed_modes={ExecutionMode.foreground},
                failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
                mutates_world=False,
            )

            def execute(self, inputs, world, snapshot=None):
                world.emit(self.contract.name, "sneaky_event", {})
                return SkillResult(outputs={"status": "done"})

        executor, _ = _make_executor(LiarSkill())

        plan = _make_plan([
            MissionNode(id="n1", skill="test.liar_skill"),
        ])

        with pytest.raises(RuntimeError, match="mutates_world=False"):
            executor.run(plan)

    def test_no_declared_events_but_skill_emits_still_caught(self):
        """Skill declares emits_events=[] but emits with correct source → RuntimeError."""

        class SneakySkill(Skill):
            contract = SkillContract(
                name="test.sneaky_skill",
                description="Sneaky",
                inputs={},
                outputs={"status": "text"},
                allowed_modes={ExecutionMode.foreground},
                failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
                emits_events=[],
                mutates_world=True,
            )

            def execute(self, inputs, world, snapshot=None):
                world.emit(self.contract.name, "rogue_event", {})
                return SkillResult(outputs={"status": "done"})

        executor, _ = _make_executor(SneakySkill())

        plan = _make_plan([
            MissionNode(id="n1", skill="test.sneaky_skill"),
        ])

        with pytest.raises(RuntimeError, match="emits_events="):
            executor.run(plan)


class TestSourceNamespaceCollisionSafety:
    """
    Verify that background source names never collide with skill contract names.
    Skill names always contain a dot. Background sources never do.
    """

    def test_background_source_without_dot_ignored(self):
        """Events with bare source names (no dot) are never skill events."""
        executor, tl = _make_executor(CleanSkill())

        # These are the actual background source names — no dots
        for source in ["time", "system", "media"]:
            tl.emit(source=source, event_type=f"{source}_event")

        plan = _make_plan([
            MissionNode(id="n1", skill="test.clean"),
        ])

        # None of these should trigger enforcement
        er = executor.run(plan)
        assert er.results["n1"]["status"] == "ok"

    def test_wrong_source_emitter_passes_enforcement(self):
        """
        Skill that emits with a background source name bypasses enforcement.
        This is a design invariant: we ONLY enforce skill-sourced events.
        """
        executor, tl = _make_executor(WrongSourceEmitterSkill())

        plan = _make_plan([
            MissionNode(id="n1", skill="test.wrong_source"),
        ])

        # The skill emits source="time" — executor won't see it as a skill event
        # This is by design: enforcement only validates skill-sourced events
        er = executor.run(plan)
        assert er.results["n1"]["status"] == "ok"

        # But the event IS in the timeline
        all_events = tl.all_events()
        assert any(e.type == "time_tick" for e in all_events)
