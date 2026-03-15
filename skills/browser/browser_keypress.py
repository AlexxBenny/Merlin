# skills/browser/browser_keypress.py

"""
BrowserKeypressSkill — Press a keyboard key in the browser.

Delegates to BrowserController.press_key().
Enables form submission (Enter), modal dismissal (Escape),
and keyboard navigation (Tab, Arrow keys).
"""

from typing import Any, Dict

from runtime.sources.browser import BrowserSource
from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from skills.skill_result import SkillResult
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline


class BrowserKeypressSkill(Skill):
    """Press a keyboard key in the browser."""

    contract = SkillContract(
        name="browser.keypress",
        action="keypress",
        target_type="browser_page",
        description="Press a key in browser",
        narration_template="press {key}",
        intent_verbs=["press", "hit", "submit", "enter"],
        intent_keywords=["key", "enter", "escape", "tab"],
        verb_specificity="specific",
        domain="browser",
        requires_focus=True,
        inputs={
            "key": "key_name",
        },
        outputs={
            "pressed": "boolean",
        },
        allowed_modes={ExecutionMode.foreground},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
        },
        emits_events=["browser_entities_refreshed"],
        mutates_world=True,
        output_style="terse",
    )

    def __init__(self, browser_controller):
        self._controller = browser_controller

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        key = str(inputs.get("key", "Enter")).strip()

        # Normalize common natural language → Playwright key names
        KEY_ALIASES = {
            "enter": "Enter",
            "return": "Enter",
            "escape": "Escape",
            "esc": "Escape",
            "tab": "Tab",
            "space": "Space",
            "backspace": "Backspace",
            "delete": "Delete",
            "arrowdown": "ArrowDown",
            "arrowup": "ArrowUp",
            "arrowleft": "ArrowLeft",
            "arrowright": "ArrowRight",
            "down": "ArrowDown",
            "up": "ArrowUp",
            "left": "ArrowLeft",
            "right": "ArrowRight",
        }
        key = KEY_ALIASES.get(key.lower(), key)

        result = self._controller.press_key(key)

        if not result.success:
            raise RuntimeError(f"Keypress failed: {result.error}")

        # Keypress may cause DOM change (Enter → search results, etc)
        post_snap = self._controller.get_snapshot(cached=False)
        world.emit("skill.browser", "browser_entities_refreshed", {
            "url": post_snap.url if post_snap else "",
            "title": post_snap.title if post_snap else "",
            "entity_count": len(post_snap.entities) if post_snap else 0,
            "tab_count": post_snap.tab_count if post_snap else 0,
            "top_entities": (
                BrowserSource._extract_top_entities(post_snap)
                if post_snap else []
            ),
        })

        return SkillResult(
            outputs={"pressed": True},
            metadata={"domain": "browser", "entity": f"keypress {key}"},
        )
