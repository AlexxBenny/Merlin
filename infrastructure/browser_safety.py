# infrastructure/browser_safety.py

"""
BrowserSafetyGate — Pre-execution safety checks for browser tasks.

MANDATORY for browser automation. Without this, browser agents will:
- Click "Buy Now" and complete purchases
- Accept cookie/privacy dialogs blindly
- Fill login forms with hallucinated credentials
- Download files without consent

Design rules:
- Check BEFORE dispatching to browser-use, not after
- Domain checks match substrings (catches subdomains)
- Action checks match against task description text
- Never raises — returns SafetyVerdict
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)


class SafetyVerdict(Enum):
    """Result of a safety check."""
    ALLOW = "allow"
    BLOCK = "block"
    CONFIRM_REQUIRED = "confirm_required"


@dataclass(frozen=True)
class SafetyResult:
    """Detailed safety check result."""
    verdict: SafetyVerdict
    reason: str = ""
    blocked_by: str = ""  # Which rule triggered the block


class BrowserSafetyGate:
    """Pre-execution safety checks for browser automation tasks.

    Enforces three layers:
    1. Domain blocklist — blocks tasks mentioning financial domains
    2. Action blocklist — blocks purchase/checkout language in task text
    3. Confirmation list — flags login/download for user confirmation

    Loaded from config/browser.yaml safety section.
    """

    def __init__(self, config: Dict[str, Any]):
        safety = config.get("safety", {})
        self._blocked_domains: Set[str] = {
            d.lower() for d in safety.get("blocked_domains", [])
        }
        self._blocked_actions: Set[str] = {
            a.lower() for a in safety.get("blocked_actions", [])
        }
        self._confirm_actions: Set[str] = {
            a.lower() for a in safety.get("require_confirmation_for", [])
        }
        self._max_steps: int = safety.get("max_steps_hard_limit", 50)

        logger.info(
            "[BrowserSafety] Initialized: %d blocked domains, "
            "%d blocked actions, %d confirm actions",
            len(self._blocked_domains),
            len(self._blocked_actions),
            len(self._confirm_actions),
        )

    def check_task(self, task: str) -> SafetyResult:
        """Check if a browser task is safe to execute.

        Checks task description text against domain and action blocklists.

        Args:
            task: Natural language task description

        Returns:
            SafetyResult with verdict and reason.
        """
        task_lower = task.lower()

        # ── Domain blocklist ──
        for domain in self._blocked_domains:
            if domain in task_lower:
                logger.warning(
                    "[BrowserSafety] BLOCKED: domain '%s' in task '%s'",
                    domain, task[:80],
                )
                return SafetyResult(
                    verdict=SafetyVerdict.BLOCK,
                    reason=f"Task involves a blocked domain: {domain}",
                    blocked_by=f"domain:{domain}",
                )

        # ── Action blocklist ──
        for action in self._blocked_actions:
            if action in task_lower:
                logger.warning(
                    "[BrowserSafety] BLOCKED: action '%s' in task '%s'",
                    action, task[:80],
                )
                return SafetyResult(
                    verdict=SafetyVerdict.BLOCK,
                    reason=f"Task involves a blocked action: {action}",
                    blocked_by=f"action:{action}",
                )

        # ── Confirmation list ──
        for action in self._confirm_actions:
            if action in task_lower:
                logger.info(
                    "[BrowserSafety] CONFIRM: action '%s' in task '%s'",
                    action, task[:80],
                )
                return SafetyResult(
                    verdict=SafetyVerdict.CONFIRM_REQUIRED,
                    reason=f"Task involves an action requiring confirmation: {action}",
                    blocked_by=f"confirm:{action}",
                )

        return SafetyResult(verdict=SafetyVerdict.ALLOW)

    def check_url(self, url: str) -> SafetyResult:
        """Check if a URL's domain is not blocklisted.

        Args:
            url: URL string to check

        Returns:
            SafetyResult with verdict.
        """
        url_lower = url.lower()
        for domain in self._blocked_domains:
            if domain in url_lower:
                return SafetyResult(
                    verdict=SafetyVerdict.BLOCK,
                    reason=f"URL involves a blocked domain: {domain}",
                    blocked_by=f"domain:{domain}",
                )
        return SafetyResult(verdict=SafetyVerdict.ALLOW)

    def clamp_steps(self, requested: int) -> int:
        """Enforce hard step limit."""
        return min(requested, self._max_steps)
