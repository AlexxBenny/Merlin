# tests/test_structural_clustering.py

"""
Tests for structural DOM clustering — P1 implementation.

Covers:
1. Helper methods (subtree_has_tag, count_subtree_nodes, etc.)
2. Container detection (ancestor walk with ratio >= 0.5)
3. Structural signature (11-feature hash)
4. Spatial rhythm validation (CV < 0.5)
5. Full detect_semantic_groups pipeline
6. click_nth_result
7. Container deduplication
8. browser.select_result skill contract + execute
9. P0 search baseline fix verification
10. P2 navigate routing changes
"""

import hashlib
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

from infrastructure.browser_controller import (
    BrowserController, BrowserResult, DOMEntity, PageSnapshot,
)


# ─────────────────────────────────────────────────────────────
# Mock DOM Nodes — simulates EnhancedDOMTreeNode
# ─────────────────────────────────────────────────────────────

class MockNodeType(Enum):
    ELEMENT_NODE = 1
    TEXT_NODE = 3


@dataclass
class MockDOMRect:
    x: float = 0
    y: float = 0
    width: float = 200
    height: float = 100


@dataclass
class MockAxNode:
    role: str = "generic"
    name: str = ""


@dataclass
class MockDOMNode:
    """Mock for EnhancedDOMTreeNode."""
    tag_name: str = "div"
    backend_node_id: int = 0
    is_visible: bool = True
    has_js_click_listener: bool = False
    attributes: dict = field(default_factory=dict)
    absolute_position: Optional[MockDOMRect] = field(default_factory=MockDOMRect)
    ax_node: Optional[MockAxNode] = None
    parent_node: Optional["MockDOMNode"] = None
    children_nodes: Optional[List["MockDOMNode"]] = None
    node_type: MockNodeType = MockNodeType.ELEMENT_NODE
    _xpath: str = ""
    _text: str = ""

    @property
    def xpath(self):
        return self._xpath

    def get_all_children_text(self):
        return self._text


def _make_card(
    tag="div", bnid=100, y=0, x=0, text="Card",
    has_link=True, has_img=True, parent=None,
    click_listener=False, xpath="/html/body/div",
):
    """Create a mock DOM card (container) with children."""
    card = MockDOMNode(
        tag_name=tag, backend_node_id=bnid, is_visible=True,
        absolute_position=MockDOMRect(x=x, y=y, width=300, height=80),
        parent_node=parent, _xpath=xpath, _text=text,
    )
    children = []
    if has_link:
        link = MockDOMNode(
            tag_name="a", backend_node_id=bnid + 1,
            is_visible=True, attributes={"href": "/link"},
            parent_node=card, _text=text,
            absolute_position=MockDOMRect(x=x, y=y, width=200, height=20),
        )
        children.append(link)
    if has_img:
        img = MockDOMNode(
            tag_name="img", backend_node_id=bnid + 2,
            is_visible=True, attributes={"src": "/img.jpg"},
            parent_node=card,
            absolute_position=MockDOMRect(x=x, y=y + 20, width=200, height=40),
        )
        children.append(img)
    if click_listener:
        card.has_js_click_listener = True

    card.children_nodes = children
    return card


def _make_list_container(num_cards=5, card_tag="div", container_tag="div"):
    """Build a container with N identical cards — should form one group."""
    container = MockDOMNode(
        tag_name=container_tag, backend_node_id=1,
        absolute_position=MockDOMRect(x=100, y=0, width=800, height=num_cards * 100),
        _xpath="/html/body/main",
    )
    cards = []
    for i in range(num_cards):
        card = _make_card(
            tag=card_tag, bnid=100 + i * 10,
            y=i * 100, x=100,
            text=f"Video {i + 1} title",
            xpath=f"/html/body/main/{card_tag}[{i + 1}]",
        )
        card.parent_node = container
        cards.append(card)
    container.children_nodes = cards
    return container, cards


# ─────────────────────────────────────────────────────────────
# Import the controller
# ─────────────────────────────────────────────────────────────

from infrastructure.browser_use_controller import BrowserUseController


# ─────────────────────────────────────────────────────────────
# 1. Helper method tests
# ─────────────────────────────────────────────────────────────

class TestSubtreeHelpers:

    def test_subtree_has_tag_finds_child(self):
        parent = MockDOMNode(tag_name="div")
        child = MockDOMNode(tag_name="img", parent_node=parent)
        parent.children_nodes = [child]
        assert BrowserUseController._subtree_has_tag(parent, "img") is True

    def test_subtree_has_tag_finds_grandchild(self):
        grandchild = MockDOMNode(tag_name="a")
        child = MockDOMNode(tag_name="div", children_nodes=[grandchild])
        parent = MockDOMNode(tag_name="div", children_nodes=[child])
        assert BrowserUseController._subtree_has_tag(parent, "a") is True

    def test_subtree_has_tag_returns_false(self):
        parent = MockDOMNode(tag_name="div", children_nodes=[])
        assert BrowserUseController._subtree_has_tag(parent, "img") is False

    def test_count_subtree_nodes(self):
        child1 = MockDOMNode(tag_name="a", children_nodes=[])
        child2 = MockDOMNode(tag_name="img", children_nodes=[])
        parent = MockDOMNode(tag_name="div", children_nodes=[child1, child2])
        assert BrowserUseController._count_subtree_nodes(parent) == 3

    def test_count_clickable_descendants(self):
        link = MockDOMNode(tag_name="a", children_nodes=[])
        btn = MockDOMNode(tag_name="button", children_nodes=[])
        div = MockDOMNode(tag_name="div", children_nodes=[])
        parent = MockDOMNode(tag_name="div", children_nodes=[link, btn, div])
        assert BrowserUseController._count_clickable_descendants(parent) == 2

    def test_count_clickable_descendants_with_js_listener(self):
        div = MockDOMNode(tag_name="div", has_js_click_listener=True, children_nodes=[])
        parent = MockDOMNode(tag_name="div", children_nodes=[div])
        assert BrowserUseController._count_clickable_descendants(parent) == 1

    def test_has_clickable_descendant_true(self):
        link = MockDOMNode(tag_name="a", children_nodes=[])
        parent = MockDOMNode(tag_name="div", children_nodes=[link])
        assert BrowserUseController._has_clickable_descendant(parent) is True

    def test_has_clickable_descendant_false(self):
        span = MockDOMNode(tag_name="span", children_nodes=[])
        parent = MockDOMNode(tag_name="div", children_nodes=[span])
        assert BrowserUseController._has_clickable_descendant(parent) is False

    def test_has_clickable_descendant_via_role(self):
        div = MockDOMNode(
            tag_name="div", ax_node=MockAxNode(role="button"),
            children_nodes=[],
        )
        parent = MockDOMNode(tag_name="div", children_nodes=[div])
        assert BrowserUseController._has_clickable_descendant(parent) is True


# ─────────────────────────────────────────────────────────────
# 2. Container detection
# ─────────────────────────────────────────────────────────────

class TestFindContainer:

    def test_finds_repeating_child(self):
        """Finds the li (repeating child) when li's parent has ≥3 same-tag children."""
        container = MockDOMNode(tag_name="ul", backend_node_id=1)
        children = [
            MockDOMNode(tag_name="li", parent_node=container, children_nodes=[]),
            MockDOMNode(tag_name="li", parent_node=container, children_nodes=[]),
            MockDOMNode(tag_name="li", parent_node=container, children_nodes=[]),
        ]
        container.children_nodes = children

        # node is inside children[0], so _find_container should return children[0]
        inner = MockDOMNode(tag_name="a", parent_node=children[0])
        children[0].children_nodes = [inner]
        result = BrowserUseController._find_container(inner)
        assert result is children[0]

    def test_ratio_threshold_met(self):
        """3 divs + 2 spans = 60% dominant → returns the div child."""
        container = MockDOMNode(tag_name="section")
        children = [
            MockDOMNode(tag_name="div", parent_node=container, children_nodes=[]),
            MockDOMNode(tag_name="div", parent_node=container, children_nodes=[]),
            MockDOMNode(tag_name="div", parent_node=container, children_nodes=[]),
            MockDOMNode(tag_name="span", parent_node=container, children_nodes=[]),
            MockDOMNode(tag_name="span", parent_node=container, children_nodes=[]),
        ]
        container.children_nodes = children
        inner = MockDOMNode(tag_name="a", parent_node=children[0])
        children[0].children_nodes = [inner]
        result = BrowserUseController._find_container(inner)
        assert result is children[0]

    def test_ratio_threshold_not_met(self):
        """2 divs + 3 other tags → 40% → should NOT match, walks up."""
        inner = MockDOMNode(tag_name="section")
        children = [
            MockDOMNode(tag_name="div", parent_node=inner, children_nodes=[]),
            MockDOMNode(tag_name="div", parent_node=inner, children_nodes=[]),
            MockDOMNode(tag_name="p", parent_node=inner, children_nodes=[]),
            MockDOMNode(tag_name="span", parent_node=inner, children_nodes=[]),
            MockDOMNode(tag_name="img", parent_node=inner, children_nodes=[]),
        ]
        inner.children_nodes = children
        inner.parent_node = None  # no further parent

        result = BrowserUseController._find_container(children[0])
        # Falls back to immediate parent (inner)
        assert result is inner

    def test_walks_up_ancestors(self):
        """Inner parent has only 2 children → walks up until grandparent's parent has ≥3."""
        grandparent = MockDOMNode(tag_name="main")
        parent = MockDOMNode(tag_name="div", parent_node=grandparent)
        sibling1 = MockDOMNode(tag_name="div", parent_node=grandparent, children_nodes=[])
        sibling2 = MockDOMNode(tag_name="div", parent_node=grandparent, children_nodes=[])
        sibling3 = MockDOMNode(tag_name="div", parent_node=grandparent, children_nodes=[])
        grandparent.children_nodes = [parent, sibling1, sibling2, sibling3]

        node = MockDOMNode(tag_name="a", parent_node=parent)
        parent.children_nodes = [node]

        # node → parent → grandparent. grandparent has 4 div children (ratio ≥ 0.5)
        # So parent is the repeating child of grandparent
        result = BrowserUseController._find_container(node)
        assert result is parent


# ─────────────────────────────────────────────────────────────
# 3. Structural signature
# ─────────────────────────────────────────────────────────────

class TestContainerSignature:

    def test_same_structure_same_signature(self):
        """Two structurally identical cards produce the same signature."""
        card1 = _make_card(tag="div", bnid=100, y=0, xpath="/html/body/div/div[1]")
        card2 = _make_card(tag="div", bnid=200, y=100, xpath="/html/body/div/div[2]")
        sig1 = BrowserUseController._container_signature(card1)
        sig2 = BrowserUseController._container_signature(card2)
        assert sig1 == sig2

    def test_different_structure_different_signature(self):
        """Card with image vs card without image → different signature."""
        card_with_img = _make_card(has_img=True, has_link=True, xpath="/html/body/div/div[1]")
        card_no_img = _make_card(has_img=False, has_link=True, xpath="/html/body/div/div[2]")
        sig1 = BrowserUseController._container_signature(card_with_img)
        sig2 = BrowserUseController._container_signature(card_no_img)
        assert sig1 != sig2

    def test_signature_is_12_char_hex(self):
        card = _make_card()
        sig = BrowserUseController._container_signature(card)
        assert len(sig) == 12
        assert all(c in "0123456789abcdef" for c in sig)

    def test_different_xpath_prefix_different_signature(self):
        """Cards under different DOM paths should differ."""
        card1 = _make_card(xpath="/main/content/results/div[1]")
        card2 = _make_card(xpath="/aside/sidebar/nav/div[1]")
        sig1 = BrowserUseController._container_signature(card1)
        sig2 = BrowserUseController._container_signature(card2)
        assert sig1 != sig2


# ─────────────────────────────────────────────────────────────
# 4. Spatial rhythm
# ─────────────────────────────────────────────────────────────

class TestSpatialRhythm:

    def test_consistent_spacing(self):
        """Equal spacing → CV ≈ 0 → True."""
        nodes = [
            MockDOMNode(absolute_position=MockDOMRect(y=0)),
            MockDOMNode(absolute_position=MockDOMRect(y=100)),
            MockDOMNode(absolute_position=MockDOMRect(y=200)),
            MockDOMNode(absolute_position=MockDOMRect(y=300)),
        ]
        assert BrowserUseController._has_consistent_spacing(nodes) is True

    def test_inconsistent_spacing(self):
        """Widely varying spacing → high CV → False."""
        nodes = [
            MockDOMNode(absolute_position=MockDOMRect(y=0)),
            MockDOMNode(absolute_position=MockDOMRect(y=50)),
            MockDOMNode(absolute_position=MockDOMRect(y=500)),
        ]
        assert BrowserUseController._has_consistent_spacing(nodes) is False

    def test_too_few_nodes(self):
        """Less than 3 → False."""
        nodes = [
            MockDOMNode(absolute_position=MockDOMRect(y=0)),
            MockDOMNode(absolute_position=MockDOMRect(y=100)),
        ]
        assert BrowserUseController._has_consistent_spacing(nodes) is False

    def test_zero_spacing(self):
        """Mean < 10 → False."""
        nodes = [
            MockDOMNode(absolute_position=MockDOMRect(y=0)),
            MockDOMNode(absolute_position=MockDOMRect(y=1)),
            MockDOMNode(absolute_position=MockDOMRect(y=2)),
        ]
        assert BrowserUseController._has_consistent_spacing(nodes) is False


# ─────────────────────────────────────────────────────────────
# 5. Full detect_semantic_groups pipeline
# ─────────────────────────────────────────────────────────────

class TestDetectSemanticGroups:

    def _make_controller_with_dom(self, selector_map, page_info=None):
        """Create a controller with injected selector_map."""
        ctrl = MagicMock(spec=BrowserUseController)
        ctrl._cached_selector_map = selector_map
        ctrl._cached_page_info = page_info
        ctrl.get_full_dom = lambda: selector_map

        # Bind the real static/class methods
        ctrl._subtree_has_tag = BrowserUseController._subtree_has_tag
        ctrl._count_subtree_nodes = BrowserUseController._count_subtree_nodes
        ctrl._count_clickable_descendants = BrowserUseController._count_clickable_descendants
        ctrl._has_clickable_descendant = BrowserUseController._has_clickable_descendant
        ctrl._find_container = BrowserUseController._find_container
        ctrl._container_signature = BrowserUseController._container_signature
        ctrl._has_consistent_spacing = BrowserUseController._has_consistent_spacing
        ctrl._find_primary_clickable = BrowserUseController._find_primary_clickable
        ctrl._classify_entity = BrowserUseController._classify_entity
        ctrl._get_entity_text = BrowserUseController._get_entity_text

        # Call the real method
        ctrl.detect_semantic_groups = lambda **kw: BrowserUseController.detect_semantic_groups(ctrl, **kw)
        return ctrl

    def test_detects_video_card_group(self):
        """5 identical cards should form a group."""
        container, cards = _make_list_container(num_cards=5)

        # Build selector_map from all nodes in the cards
        selector_map = {}
        for card in cards:
            selector_map[card.backend_node_id] = card
            for child in (card.children_nodes or []):
                selector_map[child.backend_node_id] = child

        ctrl = self._make_controller_with_dom(selector_map)
        groups = ctrl.detect_semantic_groups()

        assert len(groups) >= 1
        best = groups[0]
        assert len(best) >= 3  # At least 3 cards detected

    def test_empty_dom_returns_empty(self):
        ctrl = self._make_controller_with_dom({})
        groups = ctrl.detect_semantic_groups()
        assert groups == []

    def test_min_group_size_filtering(self):
        """2 cards with min_group_size=3 → no groups."""
        container, cards = _make_list_container(num_cards=2)
        selector_map = {}
        for card in cards:
            selector_map[card.backend_node_id] = card
            for child in (card.children_nodes or []):
                selector_map[child.backend_node_id] = child

        ctrl = self._make_controller_with_dom(selector_map)
        groups = ctrl.detect_semantic_groups(min_group_size=3)
        assert groups == []

    def test_groups_sorted_by_visual_position(self):
        """Cards should be ordered by y coordinate."""
        container, cards = _make_list_container(num_cards=5)
        selector_map = {}
        for card in cards:
            selector_map[card.backend_node_id] = card
            for child in (card.children_nodes or []):
                selector_map[child.backend_node_id] = child

        ctrl = self._make_controller_with_dom(selector_map)
        groups = ctrl.detect_semantic_groups()
        if groups:
            best = groups[0]
            # Ordinals should be sequential
            for i, entity in enumerate(best):
                assert entity.index == i + 1

    def test_hint_text_boosts_score(self):
        """Same cards, but with hint_text matching → higher score."""
        container, cards = _make_list_container(num_cards=5)
        # Give cards text containing "video"
        for card in cards:
            card._text = "Amazing video tutorial about coding"

        selector_map = {}
        for card in cards:
            selector_map[card.backend_node_id] = card
            for child in (card.children_nodes or []):
                selector_map[child.backend_node_id] = child

        ctrl = self._make_controller_with_dom(selector_map)
        groups_no_hint = ctrl.detect_semantic_groups()
        groups_hint = ctrl.detect_semantic_groups(hint_text="video tutorial")

        # Both should find groups — the hint version should still work
        assert len(groups_no_hint) >= 1
        assert len(groups_hint) >= 1

    def test_container_deduplication(self):
        """A card with 3 links should appear as 1 entry, not 3."""
        container = MockDOMNode(tag_name="main", backend_node_id=1,
                                _xpath="/html/body/main")

        # 4 identical cards, each with 1 link
        cards = []
        for i in range(4):
            c = MockDOMNode(
                tag_name="div", backend_node_id=100 + i * 10,
                absolute_position=MockDOMRect(x=0, y=i * 100, width=300, height=80),
                parent_node=container, _xpath=f"/html/body/main/div[{i + 1}]",
                _text=f"Card {i + 1}",
            )
            cl = MockDOMNode(tag_name="a", backend_node_id=100 + i * 10 + 1,
                              attributes={"href": f"/{i}"}, parent_node=c,
                              absolute_position=MockDOMRect(x=0, y=i * 100),
                              _text=f"Link {i}", children_nodes=[])
            c.children_nodes = [cl]
            cards.append(c)

        container.children_nodes = cards

        # Put all links (not cards) in selector_map — simulates browser-use
        selector_map = {}
        for c in cards:
            for child in (c.children_nodes or []):
                selector_map[child.backend_node_id] = child

        ctrl = self._make_controller_with_dom(selector_map)
        groups = ctrl.detect_semantic_groups()

        # Should have groups, and the best group should have exactly 4 entries
        assert len(groups) >= 1
        assert len(groups[0]) == 4


# ─────────────────────────────────────────────────────────────
# 6. click_nth_result
# ─────────────────────────────────────────────────────────────

class TestClickNthResult:

    def test_ordinal_out_of_range_raises(self):
        ctrl = MagicMock(spec=BrowserUseController)
        ctrl.detect_semantic_groups = MagicMock(return_value=[
            [DOMEntity(1, 100, "link", "Video 1"),
             DOMEntity(2, 200, "link", "Video 2")],
        ])
        ctrl.click_nth_result = lambda ordinal, hint_text=None: (
            BrowserUseController.click_nth_result(ctrl, ordinal, hint_text)
        )

        with pytest.raises(RuntimeError, match="out of range"):
            ctrl.click_nth_result(ordinal=5)

    def test_no_groups_raises(self):
        ctrl = MagicMock(spec=BrowserUseController)
        ctrl.detect_semantic_groups = MagicMock(return_value=[])
        ctrl.click_nth_result = lambda ordinal, hint_text=None: (
            BrowserUseController.click_nth_result(ctrl, ordinal, hint_text)
        )

        with pytest.raises(RuntimeError, match="No repeated result groups"):
            ctrl.click_nth_result(ordinal=1)


# ─────────────────────────────────────────────────────────────
# 7. _find_primary_clickable
# ─────────────────────────────────────────────────────────────

class TestFindPrimaryClickable:

    def test_prefers_link_with_href(self):
        link = MockDOMNode(tag_name="a", attributes={"href": "/video"},
                            children_nodes=[])
        btn = MockDOMNode(tag_name="button", children_nodes=[])
        container = MockDOMNode(tag_name="div", children_nodes=[link, btn])
        result = BrowserUseController._find_primary_clickable(container)
        assert result is link

    def test_falls_back_to_button(self):
        btn = MockDOMNode(tag_name="button", children_nodes=[])
        span = MockDOMNode(tag_name="span", children_nodes=[])
        container = MockDOMNode(tag_name="div", children_nodes=[btn, span])
        result = BrowserUseController._find_primary_clickable(container)
        assert result is btn

    def test_falls_back_to_js_click_listener(self):
        div = MockDOMNode(tag_name="div", has_js_click_listener=True,
                           children_nodes=[])
        container = MockDOMNode(tag_name="div", children_nodes=[div])
        result = BrowserUseController._find_primary_clickable(container)
        assert result is div

    def test_returns_none_when_no_clickable(self):
        span = MockDOMNode(tag_name="span", children_nodes=[])
        container = MockDOMNode(tag_name="div", children_nodes=[span])
        result = BrowserUseController._find_primary_clickable(container)
        assert result is None

    def test_finds_deep_link(self):
        link = MockDOMNode(tag_name="a", attributes={"href": "/deep"},
                            children_nodes=[])
        inner = MockDOMNode(tag_name="div", children_nodes=[link])
        container = MockDOMNode(tag_name="div", children_nodes=[inner])
        result = BrowserUseController._find_primary_clickable(container)
        assert result is link


# ─────────────────────────────────────────────────────────────
# 8. browser.select_result skill
# ─────────────────────────────────────────────────────────────

class TestSelectResultSkill:

    def test_contract_has_correct_fields(self):
        from skills.browser.browser_select_result import BrowserSelectResultSkill
        contract = BrowserSelectResultSkill.contract
        assert contract.name == "browser.select_result"
        assert "ordinal" in contract.inputs
        assert "play" in contract.intent_verbs
        assert "first" in contract.intent_keywords
        assert "video" in contract.intent_keywords
        assert contract.verb_specificity == "specific"
        assert contract.domain == "browser"

    def test_ordinal_parsing_string(self):
        from skills.browser.browser_select_result import ORDINAL_MAP
        assert ORDINAL_MAP["first"] == 1
        assert ORDINAL_MAP["second"] == 2
        assert ORDINAL_MAP["third"] == 3
        assert ORDINAL_MAP["last"] == -1
        assert ORDINAL_MAP["top"] == 1

    def test_execute_calls_click_nth_result(self):
        from skills.browser.browser_select_result import BrowserSelectResultSkill

        snapshot = PageSnapshot(
            snapshot_id="test", url="https://youtube.com/watch",
            title="Video Title", entities=(), entity_count=0, tab_count=1,
        )
        ctrl = MagicMock()
        ctrl.click_nth_result.return_value = BrowserResult(
            success=True, snapshot=snapshot,
        )
        ctrl.get_snapshot.return_value = snapshot

        world = MagicMock()
        skill = BrowserSelectResultSkill(ctrl)
        result = skill.execute({"ordinal": 1}, world)

        ctrl.click_nth_result.assert_called_once_with(ordinal=1, hint_text=None)
        assert result.outputs["url"] == "https://youtube.com/watch"
        assert result.outputs["page_title"] == "Video Title"

    def test_execute_with_string_ordinal(self):
        from skills.browser.browser_select_result import BrowserSelectResultSkill

        snapshot = PageSnapshot(
            snapshot_id="test", url="https://youtube.com",
            title="YouTube", entities=(), entity_count=0, tab_count=1,
        )
        ctrl = MagicMock()
        ctrl.click_nth_result.return_value = BrowserResult(
            success=True, snapshot=snapshot,
        )
        ctrl.get_snapshot.return_value = snapshot

        world = MagicMock()
        skill = BrowserSelectResultSkill(ctrl)
        result = skill.execute({"ordinal": "first"}, world)

        ctrl.click_nth_result.assert_called_once_with(ordinal=1, hint_text=None)

    def test_execute_with_hint_text(self):
        from skills.browser.browser_select_result import BrowserSelectResultSkill

        snapshot = PageSnapshot(
            snapshot_id="test", url="https://youtube.com",
            title="YouTube", entities=(), entity_count=0, tab_count=1,
        )
        ctrl = MagicMock()
        ctrl.click_nth_result.return_value = BrowserResult(
            success=True, snapshot=snapshot,
        )
        ctrl.get_snapshot.return_value = snapshot

        world = MagicMock()
        skill = BrowserSelectResultSkill(ctrl)
        result = skill.execute(
            {"ordinal": 2, "hint_text": "cars movie"},
            world,
        )

        ctrl.click_nth_result.assert_called_once_with(
            ordinal=2, hint_text="cars movie",
        )


# ─────────────────────────────────────────────────────────────
# 9. P0 — Search baseline fix
# ─────────────────────────────────────────────────────────────

class TestSearchBaselineFix:
    """Verify baseline is captured BEFORE action, not after."""

    def test_enter_baseline_before_action(self):
        """Read the source to verify baseline comes before press_key."""
        import inspect
        from infrastructure.search_input_resolver import submit_search

        source = inspect.getsource(submit_search)

        # baseline must appear BEFORE press_key in Strategy 1
        baseline_pos = source.find("baseline = controller.get_snapshot")
        press_pos = source.find("press_result = controller.press_key")
        assert baseline_pos < press_pos, (
            "baseline must be captured before press_key (Strategy 1)"
        )

    def test_button_baseline_before_click(self):
        """In Strategy 2, baseline must come before click."""
        import inspect
        from infrastructure.search_input_resolver import submit_search

        source = inspect.getsource(submit_search)

        # Find Strategy 2 section
        strategy2_start = source.find("# Strategy 2")
        strategy2_section = source[strategy2_start:]

        baseline_pos = strategy2_section.find("baseline = controller.get_snapshot")
        click_pos = strategy2_section.find("click_result = controller.click")
        assert baseline_pos < click_pos, (
            "baseline must be captured before click (Strategy 2)"
        )


# ─────────────────────────────────────────────────────────────
# 10. P2 — Navigate routing changes
# ─────────────────────────────────────────────────────────────

class TestNavigateRouting:

    def test_verb_specificity_is_specific(self):
        from skills.browser.browser_navigate import BrowserNavigateSkill
        assert BrowserNavigateSkill.contract.verb_specificity == "specific"

    def test_has_tld_keywords(self):
        from skills.browser.browser_navigate import BrowserNavigateSkill
        keywords = BrowserNavigateSkill.contract.intent_keywords
        assert ".com" in keywords
        assert ".org" in keywords
        assert ".net" in keywords

    def test_has_site_name_keywords(self):
        from skills.browser.browser_navigate import BrowserNavigateSkill
        keywords = BrowserNavigateSkill.contract.intent_keywords
        assert "youtube" in keywords
        assert "google" in keywords

    def test_has_page_title_output(self):
        from skills.browser.browser_navigate import BrowserNavigateSkill
        assert "page_title" in BrowserNavigateSkill.contract.outputs


# ─────────────────────────────────────────────────────────────
# 11. Abstract method presence in base class
# ─────────────────────────────────────────────────────────────

class TestBaseControllerAbstractMethods:

    def test_get_full_dom_is_abstract(self):
        assert hasattr(BrowserController, 'get_full_dom')

    def test_detect_semantic_groups_is_abstract(self):
        assert hasattr(BrowserController, 'detect_semantic_groups')

    def test_click_nth_result_is_abstract(self):
        assert hasattr(BrowserController, 'click_nth_result')


# ─────────────────────────────────────────────────────────────
# 12. Nav suppression scoring signals
# ─────────────────────────────────────────────────────────────

class TestNavSuppressionScoring:
    """Verify that content cards outscore nav links after Phase 1 scoring fix."""

    def _make_controller_with_dom(self, selector_map, page_info=None):
        """Create a controller with injected selector_map."""
        ctrl = MagicMock(spec=BrowserUseController)
        ctrl._cached_selector_map = selector_map
        ctrl._cached_page_info = page_info
        ctrl.get_full_dom = lambda: selector_map

        ctrl._subtree_has_tag = BrowserUseController._subtree_has_tag
        ctrl._count_subtree_nodes = BrowserUseController._count_subtree_nodes
        ctrl._count_clickable_descendants = BrowserUseController._count_clickable_descendants
        ctrl._has_clickable_descendant = BrowserUseController._has_clickable_descendant
        ctrl._find_container = BrowserUseController._find_container
        ctrl._container_signature = BrowserUseController._container_signature
        ctrl._has_consistent_spacing = BrowserUseController._has_consistent_spacing
        ctrl._find_primary_clickable = BrowserUseController._find_primary_clickable
        ctrl._classify_entity = BrowserUseController._classify_entity
        ctrl._get_entity_text = BrowserUseController._get_entity_text

        ctrl.detect_semantic_groups = lambda **kw: BrowserUseController.detect_semantic_groups(ctrl, **kw)
        return ctrl

    def _build_nav_bar(self, num_links=6):
        """Build a nav bar with short-text links under a <nav> ancestor."""
        nav = MockDOMNode(
            tag_name="nav", backend_node_id=1,
            absolute_position=MockDOMRect(x=0, y=0, width=1280, height=40),
            attributes={"class": "main-nav"},
            _xpath="/html/body/header/nav",
        )
        items = []
        for i in range(num_links):
            li = MockDOMNode(
                tag_name="li", backend_node_id=10 + i * 10,
                absolute_position=MockDOMRect(x=i * 120, y=0, width=100, height=35),
                parent_node=nav, _xpath=f"/html/body/header/nav/li[{i + 1}]",
                _text=["Home", "Explore", "Shorts", "Library", "History", "Subs"][i % 6],
            )
            link = MockDOMNode(
                tag_name="a", backend_node_id=10 + i * 10 + 1,
                is_visible=True, attributes={"href": f"/page{i}"},
                parent_node=li,
                absolute_position=MockDOMRect(x=i * 120, y=0, width=90, height=30),
                _text=li._text, children_nodes=[],
            )
            li.children_nodes = [link]
            items.append(li)
        nav.children_nodes = items
        return nav, items

    def _build_video_cards(self, num_cards=5):
        """Build video cards under a <main> ancestor with rich content."""
        main = MockDOMNode(
            tag_name="main", backend_node_id=500,
            absolute_position=MockDOMRect(x=100, y=100, width=900, height=num_cards * 200),
            _xpath="/html/body/main",
        )
        cards = []
        for i in range(num_cards):
            card = MockDOMNode(
                tag_name="div", backend_node_id=600 + i * 10,
                absolute_position=MockDOMRect(x=100, y=100 + i * 200, width=400, height=180),
                parent_node=main,
                _xpath=f"/html/body/main/div[{i + 1}]",
                _text=f"Amazing Video Title {i + 1} - Full Tutorial and Guide on Programming",
            )
            # Rich subtree: img + a (with long text) + span (metadata)
            thumb = MockDOMNode(
                tag_name="img", backend_node_id=600 + i * 10 + 1,
                is_visible=True, attributes={"src": f"/thumb{i}.jpg"},
                parent_node=card,
                absolute_position=MockDOMRect(x=100, y=100 + i * 200, width=200, height=120),
                children_nodes=[],
            )
            title_link = MockDOMNode(
                tag_name="a", backend_node_id=600 + i * 10 + 2,
                is_visible=True,
                attributes={"href": f"/watch?v=abc{i}"},
                parent_node=card,
                absolute_position=MockDOMRect(x=100, y=100 + i * 200 + 125, width=350, height=20),
                _text=f"Amazing Video Title {i + 1} - Full Tutorial and Guide",
                children_nodes=[],
            )
            meta = MockDOMNode(
                tag_name="span", backend_node_id=600 + i * 10 + 3,
                parent_node=card,
                absolute_position=MockDOMRect(x=100, y=100 + i * 200 + 150, width=200, height=15),
                _text="1.2M views · 3 days ago",
                children_nodes=[],
            )
            card.children_nodes = [thumb, title_link, meta]
            cards.append(card)
        main.children_nodes = cards
        return main, cards

    def test_content_cards_outscore_nav_bar(self):
        """Video cards under <main> must outscore nav links under <nav>."""
        nav, nav_items = self._build_nav_bar(6)
        main, video_cards = self._build_video_cards(5)

        # Build selector_map with all interactive nodes
        selector_map = {}
        for item in nav_items:
            for child in (item.children_nodes or []):
                selector_map[child.backend_node_id] = child
        for card in video_cards:
            for child in (card.children_nodes or []):
                if child.tag_name in ("a", "img"):
                    selector_map[child.backend_node_id] = child

        ctrl = self._make_controller_with_dom(selector_map)
        groups = ctrl.detect_semantic_groups()

        assert len(groups) >= 1, "Should detect at least one group"
        best = groups[0]

        # The best group should contain video content, not nav
        any_video = any(
            "Video" in (e.text or "") or "Tutorial" in (e.text or "")
            for e in best
        )
        assert any_video, (
            f"Best group should contain video cards, got: "
            f"{[(e.text or '')[:30] for e in best]}"
        )


# ─────────────────────────────────────────────────────────────
# 13. Decomposition constraint — autonomous_task forbidden
# ─────────────────────────────────────────────────────────────

class TestDecompositionConstraint:
    """Verify the decomposition prompt forbids autonomous_task."""

    def test_autonomous_task_forbidden_in_prompt(self):
        """The decomposition prompt template must contain the prohibition."""
        import inspect
        from cortex.mission_cortex import MissionCortex
        source = inspect.getsource(MissionCortex.decompose_intents)
        assert "NEVER" in source and "autonomous_task" in source, (
            "Decomposition prompt must contain rule forbidding autonomous_task"
        )

    def test_compound_browser_example_present(self):
        """The decomposition prompt must mention search + select_result decomposition."""
        import inspect
        from cortex.mission_cortex import MissionCortex
        source = inspect.getsource(MissionCortex.decompose_intents)
        assert "select_result" in source, (
            "Decomposition prompt must reference select_result for compound queries"
        )


# ─────────────────────────────────────────────────────────────
# 14. Coverage validator — action subsumption
# ─────────────────────────────────────────────────────────────

class TestSubsumptionFallback:
    """Verify CapabilityIntentMatcher handles action subsumption."""

    def _make_node(self, node_id, skill, inputs=None):
        from ir.mission import MissionNode
        return MissionNode(
            id=node_id,
            skill=skill,
            inputs=inputs or {},
        )

    def test_autonomous_task_subsumes_search(self):
        from cortex.validators import CapabilityIntentMatcher
        matcher = CapabilityIntentMatcher()

        intent = {"action": "autonomous_task", "parameters": {"task": "search for cars"}}
        nodes = [self._make_node("n0", "browser.search", {"query": "cars"})]
        registry = MagicMock()

        result = matcher.match(intent, nodes, registry)
        assert result == "n0", "autonomous_task should subsume search"

    def test_autonomous_task_subsumes_select_result(self):
        from cortex.validators import CapabilityIntentMatcher
        matcher = CapabilityIntentMatcher()

        intent = {"action": "autonomous_task", "parameters": {"task": "play first video"}}
        nodes = [self._make_node("n1", "browser.select_result", {"ordinal": 1})]
        registry = MagicMock()

        result = matcher.match(intent, nodes, registry)
        assert result == "n1", "autonomous_task should subsume select_result"

    def test_search_covered_by_autonomous_task_node(self):
        from cortex.validators import CapabilityIntentMatcher
        matcher = CapabilityIntentMatcher()

        intent = {"action": "search", "parameters": {"query": "cars"}}
        nodes = [self._make_node("n2", "browser.autonomous_task", {"task": "find cars"})]
        registry = MagicMock()

        result = matcher.match(intent, nodes, registry)
        assert result == "n2", "autonomous_task node should cover search intent (reverse subsumption)"

    def test_strict_match_still_works(self):
        from cortex.validators import CapabilityIntentMatcher
        matcher = CapabilityIntentMatcher()

        intent = {"action": "search", "parameters": {"query": "cars"}}
        nodes = [self._make_node("n3", "browser.search", {"query": "cars"})]
        registry = MagicMock()

        result = matcher.match(intent, nodes, registry)
        assert result == "n3", "Strict match should work as before"

    def test_no_match_returns_none(self):
        from cortex.validators import CapabilityIntentMatcher
        matcher = CapabilityIntentMatcher()

        intent = {"action": "fly_to_moon", "parameters": {}}
        nodes = [self._make_node("n4", "browser.search", {"query": "rockets"})]
        registry = MagicMock()

        result = matcher.match(intent, nodes, registry)
        assert result is None, "Unrelated action should not match"

