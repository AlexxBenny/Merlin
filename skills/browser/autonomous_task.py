# skills/browser/autonomous_task.py

"""
BrowserAutonomousTaskSkill — AI-driven browser automation via browser-use.

This is the primary (and Phase 1 only) browser skill. It delegates to
BrowserUseAdapter which wraps the browser-use library.

Flow:
    1. Safety gate checks task text
    2. Adapter runs agent on persistent browser
    3. DOM extraction captures page state
    4. SkillResult contains structured links for entity resolution
    5. BrowserSession updated for follow-up context

Design rules:
    - Skill is sync; adapter handles async bridge
    - Safety gate is mandatory — checked before dispatch
    - Extracted links go into result outputs for entity_registry
    - MERLIN resolves follow-up references (URLs), not browser-use
"""

from typing import Any, Dict, Optional

from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from skills.skill_result import SkillResult
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline


class BrowserAutonomousTaskSkill(Skill):
    """Execute a browser task autonomously via browser-use.

    Inputs:
        task: Natural language description of the browser task
        max_steps: (optional) Override default max agent steps

    Outputs:
        success: Whether the task completed
        final_url: The URL of the page after task completion
        page_title: Title of the final page
        extracted_data: List of links ({index, title, url}) from the page

    Delegates to BrowserUseAdapter.run_task().
    Updates BrowserSession via SessionManager on success.
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
        },
        allowed_modes={ExecutionMode.foreground},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
        },
        emits_events=["browser_task_completed"],
        mutates_world=True,
        output_style="rich",
    )

    def __init__(self, browser_adapter, session_manager=None, browser_controller=None):
        """
        Args:
            browser_adapter: BrowserUseAdapter instance (injected by main.py)
            session_manager: SessionManager instance (optional, for session tracking)
            browser_controller: BrowserUseController instance (optional, for post-agent state)
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
        """Execute browser task.

        Steps:
            1. Validate adapter availability
            2. Safety gate check
            3. Run task via adapter
            4. Update BrowserSession
            5. Return structured result

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

        # ── 3. Run task via adapter ──
        result = self._adapter.run_task(task, max_steps=max_steps)

        # ── 4. ALWAYS update BrowserSession ──
        # Browser exists → session must exist, even if task failed.
        # This ensures MERLIN knows a browser window is active.
        if self._session_mgr:
            self._update_browser_session(result, task)

        # ── 5. Check for failure — do NOT lie to the user ──
        if not result.get("success"):
            error_msg = result.get("error", "Browser task failed")
            raise RuntimeError(f"Browser task failed: {error_msg}")

        # ── 6. Emit event ──
        try:
            world.emit("skill.browser", "browser_task_completed", {
                "task": task,
                "url": result.get("final_url", ""),
                "success": True,
            })
        except Exception:
            pass  # Event emission is best-effort

        # ── 7. Post-agent state via controller (if available) ──
        # The controller rebuilds a fresh snapshot from actual DOM,
        # which is far more accurate than the agent's internal state.
        controller_entities = []
        final_url = result.get("final_url", "")
        page_title = result.get("page_title", "")
        if self._controller:
            try:
                snapshot = self._controller.get_snapshot(cached=False)
                final_url = snapshot.url or final_url
                page_title = snapshot.title or page_title
                controller_entities = [
                    {
                        "index": e.index,
                        "type": e.entity_type,
                        "title": e.text[:80],
                        "url": e.url or "",
                        "backend_node_id": e.backend_node_id,
                    }
                    for e in snapshot.entities[:15]
                ]
            except Exception:
                pass  # Fallback to adapter result below

        # Use controller entities if available, else adapter links
        extracted_data = controller_entities or result.get("links", [])

        # ── 8. Return structured result (only on success) ──
        return SkillResult(
            outputs={
                "success": True,
                "final_url": final_url,
                "page_title": page_title,
                "extracted_data": extracted_data,
            },
            metadata={
                "domain": "browser",
                "entity": f"browser: {task[:60]}",
                "steps_taken": result.get("steps_taken", 0),
            },
        )

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

            # Build entity summary (counts, not raw arrays)
            entity_summary: Dict[str, Any] = {
                "links_count": len(links),
            }
            # Store top link details for referential resolution
            # Include URLs so MERLIN can resolve "the second one" → URL
            if links:
                entity_summary["top_links"] = [
                    {
                        "index": l["index"],
                        "title": l.get("title", "")[:80],
                        "url": l.get("url", ""),
                    }
                    for l in links[:10]
                ]

            # Check for existing browser session
            existing_sessions = self._session_mgr.get_sessions_by_type(
                SessionType.BROWSER,
            )
            existing = existing_sessions[0] if existing_sessions else None

            if existing:
                # Update existing session via SessionManager API
                self._session_mgr.update_session(
                    existing.id,
                    current_url=url or existing.current_url,
                    page_title=title or existing.page_title,
                    last_action_summary=task[:120],
                    extracted_entities=entity_summary,
                )
            else:
                # Create new browser session
                session = BrowserSession(
                    current_url=url,
                    page_title=title,
                    tab_count=1,
                    last_action_summary=task[:120],
                    extracted_entities=entity_summary,
                )
                self._session_mgr.create_session(session)

        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(
                "BrowserSession update failed: %s", e,
            )
