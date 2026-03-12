# tests/test_browser_skill_outputs.py

"""
Tests for browser skill output enrichment — Phase 3.

Covers:
1. All browser skill contracts declare page_title output
2. Skill execute() methods return page_title in outputs
3. OutputReference chaining compatibility
"""

import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass
from typing import Optional

from skills.skill_result import SkillResult


# ─────────────────────────────────────────────────────────────
# Mock infrastructure
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MockDOMEntity:
    index: int
    backend_node_id: int
    entity_type: str
    text: str
    url: Optional[str] = None
    ax_role: Optional[str] = None


@dataclass(frozen=True)
class MockPageSnapshot:
    url: str
    title: str
    entities: tuple = ()
    snapshot_id: str = "test"
    entity_count: int = 0
    tab_count: int = 1
    tabs: tuple = ()


@dataclass(frozen=True)
class MockBrowserResult:
    success: bool
    snapshot: Optional[MockPageSnapshot] = None
    error: Optional[str] = None
    navigated: bool = False
    new_tab_opened: bool = False


def _mock_controller(url="https://test.com", title="Test Page"):
    """Create a mock BrowserController."""
    ctrl = MagicMock()
    snapshot = MockPageSnapshot(
        url=url, title=title,
        entities=(
            MockDOMEntity(1, 100, "link", "Link 1"),
            MockDOMEntity(2, 101, "button", "Button 1"),
        ),
        entity_count=2,
    )
    ctrl.get_snapshot.return_value = snapshot
    ctrl.click.return_value = MockBrowserResult(
        success=True, snapshot=snapshot,
    )
    ctrl.navigate.return_value = MockBrowserResult(
        success=True, snapshot=snapshot,
    )
    ctrl.go_back.return_value = MockBrowserResult(
        success=True, snapshot=snapshot,
    )
    ctrl.go_forward.return_value = MockBrowserResult(
        success=True, snapshot=snapshot,
    )
    return ctrl


def _mock_world():
    """Create a mock WorldTimeline."""
    world = MagicMock()
    world.emit = MagicMock()
    return world


# ─────────────────────────────────────────────────────────────
# 1. Contract tests — page_title in output declaration
# ─────────────────────────────────────────────────────────────

class TestContractOutputDeclarations:
    """Every browser skill must declare page_title in its contract."""

    def test_click_has_page_title_output(self):
        from skills.browser.browser_click import BrowserClickSkill
        assert "page_title" in BrowserClickSkill.contract.outputs

    def test_navigate_has_page_title_output(self):
        from skills.browser.browser_navigate import BrowserNavigateSkill
        assert "page_title" in BrowserNavigateSkill.contract.outputs

    def test_go_back_has_page_title_output(self):
        from skills.browser.browser_go_back import BrowserGoBackSkill
        assert "page_title" in BrowserGoBackSkill.contract.outputs

    def test_go_forward_has_page_title_output(self):
        from skills.browser.browser_go_forward import BrowserGoForwardSkill
        assert "page_title" in BrowserGoForwardSkill.contract.outputs

    def test_autonomous_task_has_page_title_output(self):
        from skills.browser.autonomous_task import BrowserAutonomousTaskSkill
        assert "page_title" in BrowserAutonomousTaskSkill.contract.outputs

    def test_page_title_type_is_info_string(self):
        """All page_title outputs use info_string semantic type."""
        from skills.browser.browser_click import BrowserClickSkill
        from skills.browser.browser_navigate import BrowserNavigateSkill
        from skills.browser.browser_go_back import BrowserGoBackSkill
        from skills.browser.browser_go_forward import BrowserGoForwardSkill
        from skills.browser.autonomous_task import BrowserAutonomousTaskSkill

        for skill_cls in [
            BrowserClickSkill, BrowserNavigateSkill,
            BrowserGoBackSkill, BrowserGoForwardSkill,
            BrowserAutonomousTaskSkill,
        ]:
            assert skill_cls.contract.outputs["page_title"] == "info_string", (
                f"{skill_cls.__name__}.contract.outputs['page_title'] "
                f"should be 'info_string'"
            )


# ─────────────────────────────────────────────────────────────
# 2. Execute() returns page_title
# ─────────────────────────────────────────────────────────────

class TestExecuteReturnsPageTitle:
    """Verify execute() includes page_title in SkillResult.outputs."""

    def test_click_returns_page_title(self):
        from skills.browser.browser_click import BrowserClickSkill
        ctrl = _mock_controller(title="Amazon Product Page")
        skill = BrowserClickSkill(ctrl)
        result = skill.execute(
            {"entity_index": 1}, _mock_world(),
        )
        assert result.outputs["page_title"] == "Amazon Product Page"

    def test_navigate_returns_page_title(self):
        from skills.browser.browser_navigate import BrowserNavigateSkill
        ctrl = _mock_controller(title="YouTube Home")
        skill = BrowserNavigateSkill(ctrl)
        result = skill.execute(
            {"url": "https://youtube.com"}, _mock_world(),
        )
        assert result.outputs["page_title"] == "YouTube Home"

    def test_go_back_returns_page_title(self):
        from skills.browser.browser_go_back import BrowserGoBackSkill
        ctrl = _mock_controller(title="Previous Page")
        skill = BrowserGoBackSkill(ctrl)
        result = skill.execute({}, _mock_world())
        assert result.outputs["page_title"] == "Previous Page"

    def test_go_forward_returns_page_title(self):
        from skills.browser.browser_go_forward import BrowserGoForwardSkill
        ctrl = _mock_controller(title="Next Page")
        skill = BrowserGoForwardSkill(ctrl)
        result = skill.execute({}, _mock_world())
        assert result.outputs["page_title"] == "Next Page"


# ─────────────────────────────────────────────────────────────
# 3. OutputReference compatibility
# ─────────────────────────────────────────────────────────────

class TestOutputReferenceChaining:
    """Verify outputs are referenceable by downstream nodes."""

    def test_click_outputs_chainable(self):
        """Click outputs both url and page_title — both can be referenced."""
        from skills.browser.browser_click import BrowserClickSkill
        ctrl = _mock_controller(
            url="https://amazon.com/product",
            title="iPhone 14 Pro",
        )
        skill = BrowserClickSkill(ctrl)
        result = skill.execute(
            {"entity_index": 1}, _mock_world(),
        )

        # Both fields available for OutputReference resolution
        assert "url" in result.outputs
        assert "page_title" in result.outputs
        assert result.outputs["url"] == "https://amazon.com/product"
        assert result.outputs["page_title"] == "iPhone 14 Pro"

    def test_navigate_outputs_chainable(self):
        """Navigate outputs both final_url and page_title."""
        from skills.browser.browser_navigate import BrowserNavigateSkill
        ctrl = _mock_controller(
            url="https://youtube.com",
            title="YouTube",
        )
        skill = BrowserNavigateSkill(ctrl)
        result = skill.execute(
            {"url": "https://youtube.com"}, _mock_world(),
        )

        assert "final_url" in result.outputs
        assert "page_title" in result.outputs
        assert result.outputs["final_url"] == "https://youtube.com"
        assert result.outputs["page_title"] == "YouTube"

    def test_no_snapshot_returns_empty_strings(self):
        """When snapshot is None, page_title should be empty string."""
        from skills.browser.browser_click import BrowserClickSkill
        ctrl = _mock_controller()
        ctrl.click.return_value = MockBrowserResult(
            success=True, snapshot=None,
        )
        skill = BrowserClickSkill(ctrl)
        result = skill.execute(
            {"entity_index": 1}, _mock_world(),
        )

        assert result.outputs["url"] == ""
        assert result.outputs["page_title"] == ""
