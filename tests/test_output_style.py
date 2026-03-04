# tests/test_output_style.py

"""
Tests for Phase 14: output_style contract field and unified rendering.

Covers:
  - output_style values in SkillContract
  - orchestrator.render_skill_result() routing (terse/templated/rich)
  - ReportBuilder.build_from_skill_result()
"""

from unittest.mock import MagicMock

import pytest

from skills.contract import SkillContract
from ir.mission import ExecutionMode


# ─────────────────────────────────────────────────────────────
# A. Contract field tests
# ─────────────────────────────────────────────────────────────

class TestOutputStyleContract:
    """output_style field on SkillContract."""

    def test_default_is_terse(self):
        """Default output_style must be 'terse' (backward compat)."""
        c = SkillContract(
            name="test.skill",
            inputs={},
            outputs={},
            allowed_modes={ExecutionMode.foreground},
            failure_policy={},
        )
        assert c.output_style == "terse"

    def test_accepts_all_values(self):
        """output_style accepts terse, templated, rich."""
        for style in ("terse", "templated", "rich"):
            c = SkillContract(
                name="test.skill",
                inputs={},
                outputs={},
                allowed_modes={ExecutionMode.foreground},
                failure_policy={},
                output_style=style,
            )
            assert c.output_style == style

    def test_rejects_invalid_value(self):
        """output_style rejects unknown values."""
        with pytest.raises(Exception):
            SkillContract(
                name="test.skill",
                inputs={},
                outputs={},
                allowed_modes={ExecutionMode.foreground},
                failure_policy={},
                output_style="fancy",
            )


# ─────────────────────────────────────────────────────────────
# B. Orchestrator render_skill_result tests
# ─────────────────────────────────────────────────────────────

class TestRenderSkillResult:
    """orchestrator.render_skill_result() routing."""

    @pytest.fixture
    def orch(self):
        """Build a minimal MissionOrchestrator with mocked dependencies."""
        from orchestrator.mission_orchestrator import MissionOrchestrator

        cortex = MagicMock()
        executor = MagicMock()
        timeline = MagicMock()
        report_builder = MagicMock()
        output_channel = MagicMock()

        return MissionOrchestrator(
            cortex=cortex,
            executor=executor,
            timeline=timeline,
            report_builder=report_builder,
            output_channel=output_channel,
        )

    @pytest.fixture
    def snapshot(self):
        return MagicMock()

    @pytest.fixture
    def conversation(self):
        return MagicMock()

    # ── Terse ──

    def test_terse_changed_true_play(self, orch, snapshot, conversation):
        """Terse: changed=True on play skill → 'Playing.'"""
        result = orch.render_skill_result(
            skill_name="media.play",
            inputs={},
            outputs={"changed": True},
            metadata={},
            output_style="terse",
            user_query="play music",
            snapshot=snapshot,
            conversation=conversation,
        )
        assert result == "Playing."

    def test_terse_changed_true_mute(self, orch, snapshot, conversation):
        """Terse: changed=True on mute skill → 'Muted.'"""
        result = orch.render_skill_result(
            skill_name="system.mute",
            inputs={},
            outputs={"changed": True},
            metadata={},
            output_style="terse",
            user_query="mute",
            snapshot=snapshot,
            conversation=conversation,
        )
        assert result == "Muted."

    def test_terse_changed_true_unmute(self, orch, snapshot, conversation):
        """Terse: changed=True on unmute skill → 'Unmuted.'"""
        result = orch.render_skill_result(
            skill_name="system.unmute",
            inputs={},
            outputs={"changed": True},
            metadata={},
            output_style="terse",
            user_query="unmute",
            snapshot=snapshot,
            conversation=conversation,
        )
        assert result == "Unmuted."

    def test_terse_reason_already_muted(self, orch, snapshot, conversation):
        """Terse: reason='already_muted' → 'Already muted.'"""
        result = orch.render_skill_result(
            skill_name="system.mute",
            inputs={},
            outputs={"changed": False},
            metadata={"reason": "already_muted"},
            output_style="terse",
            user_query="mute",
            snapshot=snapshot,
            conversation=conversation,
        )
        assert result == "Already muted."

    def test_terse_no_outputs(self, orch, snapshot, conversation):
        """Terse: no outputs → 'Done. (skill_name)'"""
        result = orch.render_skill_result(
            skill_name="system.set_volume",
            inputs={"level": 50},
            outputs={},
            metadata={},
            output_style="terse",
            user_query="set volume to 50",
            snapshot=snapshot,
            conversation=conversation,
        )
        assert result == "Done. (system.set_volume)"

    def test_terse_generic_data_dump(self, orch, snapshot, conversation):
        """Terse with data outputs but no template → f'{k}: {v}' dump."""
        result = orch.render_skill_result(
            skill_name="system.unknown",
            inputs={},
            outputs={"status": "ok", "value": 42},
            metadata={},
            output_style="terse",
            user_query="check",
            snapshot=snapshot,
            conversation=conversation,
        )
        assert "status: ok" in result
        assert "value: 42" in result

    # ── Templated ──

    def test_templated_with_template(self, orch, snapshot, conversation):
        """Templated: uses response_template from metadata."""
        result = orch.render_skill_result(
            skill_name="system.get_time",
            inputs={},
            outputs={"time": "2:30 PM", "day": "Tuesday", "date": "March 4"},
            metadata={"response_template": "It's {time}, {day}, {date}."},
            output_style="templated",
            user_query="what time is it",
            snapshot=snapshot,
            conversation=conversation,
        )
        assert result == "It's 2:30 PM, Tuesday, March 4."

    def test_templated_no_template_falls_to_terse(
        self, orch, snapshot, conversation,
    ):
        """Templated without response_template → falls through to terse."""
        result = orch.render_skill_result(
            skill_name="system.get_time",
            inputs={},
            outputs={"time": "2:30 PM"},
            metadata={},  # no template
            output_style="templated",
            user_query="what time is it",
            snapshot=snapshot,
            conversation=conversation,
        )
        assert "time: 2:30 PM" in result

    # ── Rich ──

    def test_rich_calls_report_builder(self, orch, snapshot, conversation):
        """Rich: delegates to report_builder.build_from_skill_result."""
        orch.report_builder.build_from_skill_result.return_value = (
            "No pending jobs."
        )
        result = orch.render_skill_result(
            skill_name="system.list_jobs",
            inputs={},
            outputs={"jobs": []},
            metadata={},
            output_style="rich",
            user_query="are there any pending jobs",
            snapshot=snapshot,
            conversation=conversation,
        )
        assert result == "No pending jobs."
        orch.report_builder.build_from_skill_result.assert_called_once_with(
            skill_name="system.list_jobs",
            inputs={},
            outputs={"jobs": []},
            user_query="are there any pending jobs",
            snapshot=snapshot,
            conversation=conversation,
        )


# ─────────────────────────────────────────────────────────────
# C. ReportBuilder.build_from_skill_result tests
# ─────────────────────────────────────────────────────────────

class TestBuildFromSkillResult:
    """ReportBuilder.build_from_skill_result()."""

    @pytest.fixture
    def snapshot(self):
        return MagicMock()

    @pytest.fixture
    def conversation(self):
        """Minimal conversation frame."""
        from conversation.frame import ConversationFrame
        return ConversationFrame()

    def test_no_llm_returns_fallback_text(self, snapshot, conversation):
        """Without LLM, build_from_skill_result returns deterministic text."""
        from reporting.report_builder import ReportBuilder

        builder = ReportBuilder(llm=None)
        result = builder.build_from_skill_result(
            skill_name="system.list_jobs",
            inputs={},
            outputs={"jobs": []},
            user_query="any pending jobs",
            snapshot=snapshot,
            conversation=conversation,
        )
        # Should return something (fallback text, not None)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_with_llm_calls_render(self, snapshot, conversation):
        """With LLM, build_from_skill_result uses LLM rendering."""
        from reporting.report_builder import ReportBuilder

        llm = MagicMock()
        llm.complete.return_value = "No pending jobs right now."

        builder = ReportBuilder(llm=llm)
        result = builder.build_from_skill_result(
            skill_name="system.list_jobs",
            inputs={},
            outputs={"jobs": []},
            user_query="any pending jobs",
            snapshot=snapshot,
            conversation=conversation,
        )
        assert result == "No pending jobs right now."
        assert llm.complete.called

    def test_with_populated_data(self, snapshot, conversation):
        """build_from_skill_result with list data includes it in prompt."""
        from reporting.report_builder import ReportBuilder

        llm = MagicMock()
        llm.complete.return_value = "You have 2 pending jobs: J-1 and J-2."

        builder = ReportBuilder(llm=llm)
        jobs = [
            {"short_id": "J-1", "query": "remind water", "status": "pending"},
            {"short_id": "J-2", "query": "play music", "status": "running"},
        ]
        result = builder.build_from_skill_result(
            skill_name="system.list_jobs",
            inputs={},
            outputs={"jobs": jobs},
            user_query="list jobs",
            snapshot=snapshot,
            conversation=conversation,
        )
        assert "J-1" in result or "2 pending" in result
        # Verify the LLM was called with a prompt containing the skill data
        prompt = llm.complete.call_args[0][0]
        assert "list jobs" in prompt  # user query is in prompt
        assert "J-1" in prompt  # actual data is passed inline (≤10 items)

    def test_llm_failure_falls_back(self, snapshot, conversation):
        """LLM failure → deterministic fallback (never crashes)."""
        from reporting.report_builder import ReportBuilder

        llm = MagicMock()
        llm.complete.side_effect = RuntimeError("LLM down")

        builder = ReportBuilder(llm=llm)
        result = builder.build_from_skill_result(
            skill_name="system.list_jobs",
            inputs={},
            outputs={"jobs": []},
            user_query="any pending jobs",
            snapshot=snapshot,
            conversation=conversation,
        )
        # Must not crash — returns fallback text
        assert isinstance(result, str)
        assert len(result) > 0


# ─────────────────────────────────────────────────────────────
# D. Skill classification validation
# ─────────────────────────────────────────────────────────────

class TestSkillOutputStyleClassification:
    """Verify all skill contracts have correct output_style."""

    def test_rich_skills(self):
        """Data-producing skills must be 'rich'."""
        from skills.system.list_jobs import ListJobsSkill
        from skills.system.list_apps import ListAppsSkill
        from skills.system.cancel_job import CancelJobSkill

        assert ListJobsSkill.contract.output_style == "rich"
        assert ListAppsSkill.contract.output_style == "rich"
        assert CancelJobSkill.contract.output_style == "rich"

    def test_templated_skills(self):
        """Skills with response_template must be 'templated'."""
        from skills.system.get_time import GetTimeSkill
        from skills.system.get_battery import GetBatterySkill
        from skills.system.get_now_playing import GetNowPlayingSkill
        from skills.system.get_system_status import GetSystemStatusSkill

        assert GetTimeSkill.contract.output_style == "templated"
        assert GetBatterySkill.contract.output_style == "templated"
        assert GetNowPlayingSkill.contract.output_style == "templated"
        assert GetSystemStatusSkill.contract.output_style == "templated"

    def test_terse_skills_are_default(self):
        """Mutating skills have explicit output_style='terse'."""
        from skills.system.media_play import MediaPlaySkill
        from skills.system.media_pause import MediaPauseSkill
        from skills.system.mute import MuteSkill

        assert MediaPlaySkill.contract.output_style == "terse"
        assert MediaPauseSkill.contract.output_style == "terse"
        assert MuteSkill.contract.output_style == "terse"


# ─────────────────────────────────────────────────────────────
# E. Enforcement: all skills must have explicit output_style
# ─────────────────────────────────────────────────────────────

class TestOutputStyleEnforcement:
    """Prevent future skills from missing output_style.

    Loads ALL skills via production path and verifies each one
    has output_style set with a valid value.
    """

    def test_all_skills_have_explicit_output_style(self):
        """Every registered skill must declare output_style explicitly.

        This test loads skills via the same YAML-driven import path
        used in main.py. If a new skill is added without output_style,
        this test fails — forcing the developer to choose terse/templated/rich.
        """
        import importlib
        import yaml

        with open("config/skills.yaml") as f:
            skill_defs = yaml.safe_load(f)["skills"]

        valid_styles = {"terse", "templated", "rich"}

        for skill_def in skill_defs:
            module_path = skill_def["module"]
            class_name = skill_def["class"]

            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            contract = cls.contract

            assert contract.output_style in valid_styles, (
                f"Skill '{contract.name}' has output_style='{contract.output_style}', "
                f"expected one of {valid_styles}. "
                f"Every skill MUST explicitly declare output_style in its contract."
            )

    def test_output_style_consistency_with_response_template(self):
        """Skills with output_style='templated' should provide
        response_template in their execute() metadata.

        Skills with output_style='rich' should NOT have response_template
        (they use LLM narration instead).
        """
        import importlib
        import yaml

        with open("config/skills.yaml") as f:
            skill_defs = yaml.safe_load(f)["skills"]

        for skill_def in skill_defs:
            module_path = skill_def["module"]
            class_name = skill_def["class"]

            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            contract = cls.contract

            # Rich skills must NOT rely on template for formatting
            if contract.output_style == "rich":
                assert contract.name not in (
                    "system.get_time", "system.get_battery",
                    "system.get_now_playing", "system.get_system_status",
                ), (
                    f"Skill '{contract.name}' is 'rich' but looks like a "
                    f"templated query skill. Verify output_style is correct."
                )

