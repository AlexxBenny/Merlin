# skills/browser/autonomous_task.py

"""
BrowserAutonomousTaskSkill — Browser automation via browser-use Agent.

Architecture:
    1. Safety gate checks task text
    2. Enrich task with current page context (URL, title)
    3. browser-use Agent executes the task
    4. Extract structured result (success, URL, title, data, agent answer)
    5. Update BrowserSession for follow-up context

Design rules:
    - Skill is sync; adapter handles async bridge
    - Safety gate is mandatory — checked before dispatch
    - Agent's extracted answer is passed through verbatim (no re-generation)
    - Controller snapshot enriches context to prevent redundant navigation
    - Hard timeout via adapter (max_steps)
"""

import logging
from typing import Any, Dict, Optional

from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from skills.skill_result import SkillResult
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline

logger = logging.getLogger(__name__)


class BrowserAutonomousTaskSkill(Skill):
    """Execute a browser task via browser-use Agent.

    Inputs:
        task: Natural language description of the browser task
        max_steps: (optional) Override default max agent steps

    Outputs:
        success: Whether the task completed
        final_url: The URL of the page after task completion
        page_title: Title of the final page
        extracted_data: List of links/entities from the page
        extracted_text: Agent's answer text (passed through verbatim)
    """

    contract = SkillContract(
        name="browser.autonomous_task",
        action="autonomous_task",
        target_type="browser",
        description="Perform an autonomous browser task",
        narration_template="browsing: {task}",
        intent_verbs=["search", "browse", "find", "look up", "go to",
                       "navigate", "visit"],
        intent_keywords=["website", "browser", "web", "online", "internet",
                         "site", "page", "google", "amazon", "youtube",
                         "on the web", "in browser"],
        verb_specificity="generic",
        domain="browser",
        requires_focus=True,
        risk_level="moderate",
        data_freshness="live",
        inputs={
            "task": "browser_task_description",
        },
        optional_inputs={
            "max_steps": "step_limit",
        },
        outputs={
            "success": "boolean",
            "final_url": "url_string",
            "page_title": "info_string",
            "extracted_data": "any",
            "extracted_text": "info_string",
        },
        allowed_modes={ExecutionMode.foreground},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
        },
        emits_events=["browser_task_completed"],
        mutates_world=True,
        output_style="rich",
    )

    def __init__(
        self,
        browser_adapter,
        session_manager=None,
        browser_controller=None,
    ):
        """
        Args:
            browser_adapter: BrowserUseAdapter instance
            session_manager: SessionManager instance (for session tracking)
            browser_controller: BrowserUseController instance (for context)
        """
        self._adapter = browser_adapter
        self._session_mgr = session_manager
        self._controller = browser_controller
        self._safety = None

        # Late-init safety gate if adapter has config
        if hasattr(browser_adapter, '_config'):
            try:
                from infrastructure.browser_safety import BrowserSafetyGate
                self._safety = BrowserSafetyGate(browser_adapter._config)
            except Exception:
                pass

    def execute(
        self,
        inputs: Dict[str, Any],
        world: WorldTimeline,
        snapshot=None,
    ) -> SkillResult:
        """Execute browser task via browser-use Agent.

        Steps:
            1. Validate adapter availability
            2. Safety gate check
            3. Enrich task with current page context
            4. Run browser-use Agent
            5. Update BrowserSession
            6. Return structured result with agent answer

        Failure semantics:
            Raises RuntimeError on failure (same as all MERLIN skills).
            The executor catches this and reports it properly.
        """
        task = inputs["task"]
        max_steps = inputs.get("max_steps")

        # ── 1. Availability check ──
        if self._adapter is None:
            raise RuntimeError("Browser adapter not available")

        if not self._adapter.is_available():
            raise RuntimeError(
                "browser-use dependencies not installed "
                "(pip install browser-use langchain-google-genai)"
            )

        # ── 2. Safety gate ──
        if self._safety:
            from infrastructure.browser_safety import SafetyVerdict
            verdict = self._safety.check_task(task)
            if verdict.verdict == SafetyVerdict.BLOCK:
                raise RuntimeError(
                    f"Safety gate blocked: {verdict.reason}"
                )

        # ── 3. Enrich task with current page context ──
        enriched_task = task
        if self._controller:
            try:
                if self._controller.is_alive():
                    current = self._controller.get_snapshot(cached=True)
                    if current.url and current.url != "about:blank":
                        enriched_task = (
                            f"Context: Browser is currently on {current.url}"
                            f' ("{current.title or ""}"). '
                            f"Continue from this page if relevant. "
                            f"Task: {task}"
                        )
            except Exception:
                pass  # Graceful degradation — just use raw task

        # ── 4. Run browser-use Agent ──
        result = self._adapter.run_task(enriched_task, max_steps=max_steps)

        # ── 5. ALWAYS update BrowserSession ──
        if self._session_mgr:
            self._update_browser_session(result, task)

        # ── 6. Check for failure ──
        if not result.get("success"):
            error_msg = result.get("error", "Browser task failed")
            raise RuntimeError(f"Browser task failed: {error_msg}")

        # ── 7. Emit event ──
        try:
            world.emit("skill.browser", "browser_task_completed", {
                "task": task,
                "url": result.get("final_url", ""),
                "success": True,
            })
        except Exception:
            pass

        # ── 8. Extract structured result ──
        # Agent's answer text is passed through verbatim.
        # MERLIN must NOT regenerate this with LLM — it's the ground truth.
        agent_text = result.get("extracted_text", "") or ""

        # Post-agent state via controller (more accurate than adapter data)
        controller_entities = []
        final_url = result.get("final_url", "")
        page_title = result.get("page_title", "")
        if self._controller:
            try:
                snap = self._controller.get_snapshot(cached=False)
                final_url = snap.url or final_url
                page_title = snap.title or page_title
                controller_entities = [
                    {
                        "index": e.index,
                        "type": e.entity_type,
                        "title": e.text[:80],
                        "url": e.url or "",
                        "backend_node_id": e.backend_node_id,
                    }
                    for e in snap.entities[:15]
                ]
            except Exception:
                pass

        extracted_data = controller_entities or result.get("links", [])

        return SkillResult(
            outputs={
                "success": True,
                "final_url": final_url,
                "page_title": page_title,
                "extracted_data": extracted_data,
                "extracted_text": agent_text,
            },
            metadata={
                "domain": "browser",
                "entity": f"browser: {task[:60]}",
                "steps_taken": result.get("steps_taken", 0),
            },
        )

    # ── Helper: update browser session ──

    def _update_browser_session(
        self, result: Dict[str, Any], task: str,
    ) -> None:
        """Create or update BrowserSession in SessionManager.

        Rule: Browser exists → session must exist.
        Updates even on task failure (URL may still be valid).
        """
        try:
            from infrastructure.session import (
                BrowserSession, SessionType,
            )

            url = result.get("final_url", "")
            title = result.get("page_title", "")
            links = result.get("links", [])

            entity_summary: Dict[str, Any] = {
                "links_count": len(links),
            }
            if links:
                entity_summary["top_links"] = [
                    {
                        "index": l["index"],
                        "title": l.get("title", "")[:80],
                        "url": l.get("url", ""),
                    }
                    for l in links[:10]
                ]

            existing_sessions = self._session_mgr.get_sessions_by_type(
                SessionType.BROWSER,
            )
            existing = existing_sessions[0] if existing_sessions else None

            if existing:
                self._session_mgr.update_session(
                    existing.id,
                    current_url=url or existing.current_url,
                    page_title=title or existing.page_title,
                    last_action_summary=task[:120],
                    extracted_entities=entity_summary,
                )
            else:
                session = BrowserSession(
                    current_url=url,
                    page_title=title,
                    tab_count=1,
                    last_action_summary=task[:120],
                    extracted_entities=entity_summary,
                )
                self._session_mgr.create_session(session)

        except Exception as e:
            logger.debug(
                "BrowserSession update failed: %s", e,
            )
