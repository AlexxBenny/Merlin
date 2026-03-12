# infrastructure/browser_use_adapter.py

"""
BrowserUseAdapter — Persistent browser-use wrapper for MERLIN.

This is the translation boundary between MERLIN's synchronous skill
executor and browser-use's async Playwright-based automation.

Architecture:
    - Dedicated daemon thread runs a ProactorEventLoop forever
    - run_task() submits coroutines via run_coroutine_threadsafe()
    - Browser instance persists across tasks (keep_alive=True)
    - Agent is created per task (disposable)
    - LLM client (ChatGoogle) is created once

Event loop lifecycle (CRITICAL):
    MERLIN start → BrowserUseAdapter.__init__()
        ↓
    First run_task() → _ensure_loop() [ONCE]
        ↓
    Creates dedicated thread
    Thread sets ProactorEventLoop policy
    Thread creates new event loop
    Thread calls loop.run_forever()
        ↓
    All subsequent run_task() calls → run_coroutine_threadsafe()
    The loop is NEVER recreated. It runs forever until shutdown().

Threading model:
    MERLIN Executor Thread          Adapter Async Thread
    ─────────────────────          ────────────────────
    skill.execute()
      → adapter.run_task()
        → run_coroutine_threadsafe()  → _run_task_async()
        → future.result(timeout)        → Agent(task,llm,browser)
                                        → agent.run()
                                        → _extract_page_data()
        ← structured result           ←

API key resolution:
    Uses MERLIN's key_pool pattern for round-robin rotation.
    Env vars (in priority order):
      1. GOOGLE_BROWSER_AGENT_API_KEYS  (plural, comma-separated)
      2. GOOGLE_BROWSER_AGENT_API_KEY   (singular)
    Each task call rotates to the next key via itertools.cycle.

Design rules:
    - Browser persists; Agent is per-task
    - LLM is created once at init
    - _ensure_browser() detects crashes and recreates
    - DOM extraction via page.evaluate(), not history parsing
    - All errors caught and returned as structured failure dicts
    - Extensive debug logging for every lifecycle/agent/extraction event
"""

import asyncio
import glob
import logging
import os
import sys
import threading
import time
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

# ── Availability checks ──
BROWSER_USE_AVAILABLE = False
CHAT_GOOGLE_AVAILABLE = False

try:
    from browser_use import Agent
    from browser_use import BrowserSession as BrowserUseBrowser
    from browser_use import BrowserProfile
    BROWSER_USE_AVAILABLE = True
    logger.debug("[BrowserAdapter] browser-use library available")
except ImportError:
    Agent = None
    BrowserUseBrowser = None
    BrowserProfile = None
    logger.debug("[BrowserAdapter] browser-use library NOT available")

try:
    from browser_use import ChatGoogle
    CHAT_GOOGLE_AVAILABLE = True
    logger.debug("[BrowserAdapter] browser-use ChatGoogle available")
except ImportError:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI as ChatGoogle
        CHAT_GOOGLE_AVAILABLE = True
        logger.debug("[BrowserAdapter] langchain ChatGoogle available (fallback)")
    except ImportError:
        ChatGoogle = None
        logger.debug("[BrowserAdapter] ChatGoogle NOT available")


# ── Default Chrome paths ──
_DEFAULT_CHROME_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

_DEFAULT_USER_DATA_DIR = os.path.expanduser(
    r"~\AppData\Local\Google\Chrome\User Data"
)


def _detect_chrome_executable(configured_path: str = "") -> Optional[str]:
    """Find Chrome executable. Config path > default locations."""
    if configured_path and os.path.isfile(configured_path):
        logger.debug("[BrowserAdapter] Using configured Chrome: %s", configured_path)
        return configured_path
    for path in _DEFAULT_CHROME_PATHS:
        if os.path.isfile(path):
            logger.debug("[BrowserAdapter] Found Chrome at: %s", path)
            return path
    logger.warning("[BrowserAdapter] Chrome executable not found in default locations")
    return None


def _detect_chrome_profile(user_data_dir: str = "") -> str:
    """Detect the most recently used Chrome profile.

    Checks modification time of Preferences files across all profiles.
    Falls back to 'Default' if detection fails.

    Same pattern as the reference browser_agent.py.
    """
    data_dir = user_data_dir or _DEFAULT_USER_DATA_DIR
    profile_dir = "Default"

    try:
        profile_paths = glob.glob(os.path.join(data_dir, "Profile *"))
        profile_paths.append(os.path.join(data_dir, "Default"))

        latest_time = 0
        for profile_path in profile_paths:
            pref_file = os.path.join(profile_path, "Preferences")
            if os.path.exists(pref_file):
                mtime = os.path.getmtime(pref_file)
                if mtime > latest_time:
                    latest_time = mtime
                    profile_dir = os.path.basename(profile_path)

        logger.info("[BrowserAdapter] Detected Chrome profile: %s", profile_dir)
    except Exception as e:
        logger.warning(
            "[BrowserAdapter] Chrome profile detection failed, using Default: %s", e
        )

    return profile_dir


# ─────────────────────────────────────────────────────────────
# API Key Pool (round-robin, same pattern as models/key_pool.py)
# ─────────────────────────────────────────────────────────────

def _parse_keys(value: Optional[str]) -> List[str]:
    """Parse comma-separated key string into list."""
    if not value:
        return []
    return [k.strip() for k in value.split(",") if k.strip()]


class _BrowserKeyPool:
    """Round-robin API key rotation for the browser agent.

    Resolution order (most-specific → least-specific):
      1. GOOGLE_BROWSER_AGENT_API_KEYS  (plural, comma-separated)
      2. GOOGLE_BROWSER_AGENT_API_KEY   (singular)
      3. GOOGLE_API_KEY                 (global fallback)

    Thread-safe enough for single-user (GIL protects next()).
    """

    def __init__(self):
        self._cycle: Optional[Iterator[str]] = None
        self._keys: List[str] = []
        self._initialized = False

    def _init_pool(self) -> None:
        """Load keys from environment."""
        # 1. Plural role-specific
        keys = _parse_keys(os.environ.get("GOOGLE_BROWSER_AGENT_API_KEYS"))
        if keys:
            logger.info(
                "[BrowserKeyPool] Loaded %d keys from GOOGLE_BROWSER_AGENT_API_KEYS",
                len(keys),
            )
            self._keys = keys
            self._cycle = cycle(keys)
            self._initialized = True
            return

        # 2. Singular role-specific
        single = os.environ.get("GOOGLE_BROWSER_AGENT_API_KEY", "").strip()
        if single:
            logger.info("[BrowserKeyPool] Loaded 1 key from GOOGLE_BROWSER_AGENT_API_KEY")
            self._keys = [single]
            self._cycle = cycle([single])
            self._initialized = True
            return

        # 3. Global fallback
        global_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if global_key:
            logger.info("[BrowserKeyPool] Loaded 1 key from GOOGLE_API_KEY (fallback)")
            self._keys = [global_key]
            self._cycle = cycle([global_key])
            self._initialized = True
            return

        logger.warning("[BrowserKeyPool] No API keys found for browser agent")
        self._initialized = True

    def next_key(self) -> Optional[str]:
        """Get next API key via round-robin rotation."""
        if not self._initialized:
            self._init_pool()
        if self._cycle is None:
            return None
        key = next(self._cycle)
        logger.debug(
            "[BrowserKeyPool] Rotated to key: %s...%s (pool size: %d)",
            key[:8], key[-4:] if len(key) > 8 else "****", len(self._keys),
        )
        return key

    @property
    def pool_size(self) -> int:
        if not self._initialized:
            self._init_pool()
        return len(self._keys)

    @property
    def has_keys(self) -> bool:
        if not self._initialized:
            self._init_pool()
        return len(self._keys) > 0


# Module-level singleton
_browser_key_pool = _BrowserKeyPool()


class BrowserUseAdapter:
    """Persistent browser-use wrapper. Browser lives across tasks.

    Thread-safe: run_task() can be called from any thread (typically
    the MissionExecutor's ThreadPoolExecutor worker).

    CRITICAL: The event loop is created ONCE in _ensure_loop() and runs
    forever on a dedicated daemon thread. It is NEVER recreated per task.
    Every run_task() submits to the same loop via run_coroutine_threadsafe().

    Lifecycle:
        adapter = BrowserUseAdapter(config, api_key, model_name)
        result = adapter.run_task("search gaming laptops on amazon")
        result = adapter.run_task("Open https://amazon.com/product/...")
        adapter.shutdown()
    """

    def __init__(
        self,
        config: Dict[str, Any],
        api_key: str = "",
        model_name: str = "gemini-2.5-flash",
    ):
        self._config = config
        self._model_name = model_name
        self._task_counter = 0  # monotonic task ID for debug logs

        # Legacy single key — if provided, seed it into env as fallback
        if api_key:
            existing = os.environ.get("GOOGLE_BROWSER_AGENT_API_KEY", "")
            if not existing:
                os.environ["GOOGLE_BROWSER_AGENT_API_KEY"] = api_key
                logger.debug(
                    "[BrowserAdapter] Seeded GOOGLE_BROWSER_AGENT_API_KEY from "
                    "constructor arg"
                )

        # ── Persistent state ──
        self._browser = None                # BrowserUseBrowser (persistent)
        self._llm = None                    # ChatGoogle (created once)

        # ── Settings ──
        self._max_steps = config.get("max_steps", 20)
        self._timeout = config.get("agent_timeout_seconds", 120)
        self._headless = config.get("headless", False)
        self._keep_alive = config.get("keep_alive", True)
        self._chrome_exe = config.get("chrome_executable", "")
        self._chrome_profile = config.get("chrome_profile", "Default")
        self._user_data_dir = config.get("user_data_dir", "")
        self._max_links = config.get("max_extracted_links", 15)

        # ── Async event loop (dedicated thread — created ONCE) ──
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._lock = threading.Lock()

        logger.info(
            "[BrowserAdapter] Initialized: model=%s, max_steps=%d, "
            "timeout=%ds, headless=%s, keep_alive=%s, key_pool_size=%d",
            self._model_name, self._max_steps, self._timeout,
            self._headless, self._keep_alive, _browser_key_pool.pool_size,
        )

    # ─────────────────────────────────────────────────────────
    # Event Loop Lifecycle (CREATED ONCE, NEVER RECREATED)
    # ─────────────────────────────────────────────────────────

    def _ensure_loop(self) -> None:
        """Start the dedicated async event loop thread if not running.

        CRITICAL: This is called at most ONCE. The loop runs forever
        until shutdown(). Every task reuses the same loop via
        run_coroutine_threadsafe(). If this method is called after the
        loop is already running, it returns immediately.

        The ProactorEventLoop policy is set ON THE LOOP THREAD, not
        the calling thread, ensuring the correct event loop type.
        """
        if self._started:
            return

        with self._lock:
            if self._started:
                return

            logger.info("[BrowserAdapter] Starting dedicated async event loop thread...")

            self._thread = threading.Thread(
                target=self._run_loop,
                name="browser-use-loop",
                daemon=True,
            )
            self._thread.start()

            # Wait for the loop to be set by the thread
            deadline = time.monotonic() + 5.0
            while self._loop is None and time.monotonic() < deadline:
                time.sleep(0.01)

            if self._loop is None:
                raise RuntimeError(
                    "[BrowserAdapter] Event loop thread failed to start within 5s"
                )

            self._started = True
            logger.info(
                "[BrowserAdapter] Event loop thread started (thread=%s, loop=%s). "
                "This loop will be reused for ALL browser tasks.",
                self._thread.name, type(self._loop).__name__,
            )

    def _run_loop(self) -> None:
        """Run the event loop forever on the dedicated thread.

        CRITICAL: The ProactorEventLoop policy is set HERE (on the loop
        thread), not on the calling thread. This ensures Playwright
        subprocesses work correctly on Windows.

        This method runs until shutdown() calls loop.stop().
        """
        # Set policy on THIS thread (the loop thread)
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(
                asyncio.WindowsProactorEventLoopPolicy()
            )
            logger.debug("[BrowserAdapter] Set WindowsProactorEventLoopPolicy on loop thread")

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        logger.debug(
            "[BrowserAdapter] Created event loop: %s (thread=%s)",
            type(self._loop).__name__, threading.current_thread().name,
        )

        # run_forever() blocks until loop.stop() is called from shutdown()
        self._loop.run_forever()
        logger.debug("[BrowserAdapter] Event loop stopped")

    def shutdown(self) -> None:
        """Close browser and stop event loop. Called on MERLIN shutdown."""
        if not self._started:
            return

        logger.info("[BrowserAdapter] Shutting down...")

        # Close browser
        if self._browser is not None:
            try:
                logger.debug("[BrowserAdapter] Closing browser...")
                future = asyncio.run_coroutine_threadsafe(
                    self._close_browser(), self._loop
                )
                future.result(timeout=10)
                logger.debug("[BrowserAdapter] Browser closed")
            except Exception as e:
                logger.warning("[BrowserAdapter] Browser close failed: %s", e)

        # Stop loop
        if self._loop and self._loop.is_running():
            logger.debug("[BrowserAdapter] Stopping event loop...")
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
            logger.debug("[BrowserAdapter] Loop thread joined")

        self._started = False
        logger.info("[BrowserAdapter] Shutdown complete")

    async def _close_browser(self) -> None:
        """Close the persistent browser instance."""
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception as e:
                logger.debug("[BrowserAdapter] Browser close error: %s", e)
            self._browser = None

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def run_task(self, task: str, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """Execute a browser task synchronously.

        Thread-safe entry point called from MERLIN skill executor.
        Submits async work to the dedicated loop thread and blocks.

        INVARIANT: self._loop is created ONCE in _ensure_loop() and
        reused for every call. No new loops or threads are created here.

        Args:
            task: Natural language task description
            max_steps: Override default max steps (capped by hard limit)

        Returns:
            Structured result dict:
                success: bool
                final_url: str
                page_title: str
                links: list[{title, url, index}]
                action_history: list[str]
                steps_taken: int
                error: str (empty if success)
        """
        self._task_counter += 1
        task_id = self._task_counter

        logger.info(
            "[BrowserAdapter] ═══ TASK #%d START ═══\n"
            "  Task: %s\n"
            "  Max steps: %s\n"
            "  Loop reuse: %s (thread=%s)",
            task_id, task[:120],
            max_steps or f"{self._max_steps} (default)",
            "yes" if self._started else "first-time init",
            self._thread.name if self._thread else "not-started",
        )

        if not BROWSER_USE_AVAILABLE:
            logger.error("[BrowserAdapter] Task #%d: browser-use not installed", task_id)
            return self._error_result("browser-use library not installed")
        if not CHAT_GOOGLE_AVAILABLE:
            logger.error("[BrowserAdapter] Task #%d: ChatGoogle not installed", task_id)
            return self._error_result("ChatGoogle (langchain) not installed")

        self._ensure_loop()

        steps = min(
            max_steps or self._max_steps,
            self._config.get("safety", {}).get("max_steps_hard_limit", 50),
        )

        start_time = time.monotonic()

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._run_task_async(task, steps, task_id),
                self._loop,
            )
            result = future.result(timeout=self._timeout)

            elapsed = time.monotonic() - start_time
            logger.info(
                "[BrowserAdapter] ═══ TASK #%d %s ═══ (%.1fs)\n"
                "  URL: %s\n"
                "  Title: %s\n"
                "  Links extracted: %d\n"
                "  Steps taken: %d",
                task_id,
                "COMPLETE" if result.get("success") else "FAILED",
                elapsed,
                result.get("final_url", "")[:100],
                result.get("page_title", "")[:80],
                len(result.get("links", [])),
                result.get("steps_taken", 0),
            )
            return result

        except TimeoutError:
            elapsed = time.monotonic() - start_time
            logger.error(
                "[BrowserAdapter] ═══ TASK #%d TIMEOUT ═══ (%.1fs / %ds limit)\n"
                "  Task: %s",
                task_id, elapsed, self._timeout, task[:80],
            )
            future.cancel()
            return self._error_result(
                f"Browser task timed out after {self._timeout}s"
            )
        except Exception as e:
            elapsed = time.monotonic() - start_time
            logger.error(
                "[BrowserAdapter] ═══ TASK #%d ERROR ═══ (%.1fs)\n"
                "  Error: %s",
                task_id, elapsed, e, exc_info=True,
            )
            return self._error_result(str(e))

    def is_available(self) -> bool:
        """Check if browser-use dependencies are installed."""
        return BROWSER_USE_AVAILABLE and CHAT_GOOGLE_AVAILABLE

    # ─────────────────────────────────────────────────────────
    # Async implementation
    # ─────────────────────────────────────────────────────────

    async def _run_task_async(
        self, task: str, max_steps: int, task_id: int,
    ) -> Dict[str, Any]:
        """Core async implementation. Runs on the dedicated loop thread.

        1. Ensure browser is alive (lazy-create or reconnect)
        2. Ensure LLM is created (with round-robin key)
        3. Create Agent per task
        4. Run agent
        5. Extract structured data from live page DOM
        6. Return structured result
        """
        try:
            # ── 1. Browser lifecycle ──
            logger.debug("[BrowserAdapter] Task #%d: ensuring browser...", task_id)
            await self._ensure_browser(task_id)

            # ── 2. LLM lifecycle ──
            logger.debug("[BrowserAdapter] Task #%d: ensuring LLM...", task_id)
            self._ensure_llm(task_id)

            # ── 3. Create agent (disposable, per-task) ──
            logger.debug(
                "[BrowserAdapter] Task #%d: creating Agent(max_steps=%d)...",
                task_id, max_steps,
            )
            agent = Agent(
                task=task,
                llm=self._llm,
                browser=self._browser,
                max_steps=max_steps,
            )

            logger.info(
                "[BrowserAdapter] Task #%d: agent created, running...\n"
                "  Model: %s\n"
                "  Max steps: %d\n"
                "  Browser alive: %s",
                task_id, self._model_name, max_steps,
                "yes" if self._browser else "no",
            )

            # ── 4. Run agent ──
            step_start = time.monotonic()
            history = await agent.run()
            step_elapsed = time.monotonic() - step_start

            # ── 5. Extract action history ──
            action_summary = self._extract_action_history(history, task_id)
            steps_taken = len(action_summary)

            logger.info(
                "[BrowserAdapter] Task #%d: agent finished in %.1fs (%d steps)\n"
                "  Actions:\n%s",
                task_id, step_elapsed, steps_taken,
                self._format_action_log(action_summary),
            )

            # ── 6. Detect task failure from agent history ──
            # browser-use agent.run() does NOT throw on task failure.
            # It returns a history with errors. We must inspect it.
            task_failed = False
            failure_reason = ""
            task_failed, failure_reason = self._detect_task_failure(
                history, steps_taken, task_id,
            )

            # ── 7. Extract DOM data ──
            logger.debug("[BrowserAdapter] Task #%d: extracting page data...", task_id)
            page_data = await self._extract_page_data(task_id)

            logger.info(
                "[BrowserAdapter] Task #%d: page data extracted\n"
                "  URL: %s\n"
                "  Title: %s\n"
                "  Links: %d\n"
                "  Task success: %s",
                task_id,
                page_data.get("url", "")[:100],
                page_data.get("title", "")[:80],
                len(page_data.get("links", [])),
                "no — " + failure_reason if task_failed else "yes",
            )

            return {
                "success": not task_failed,
                "final_url": page_data.get("url", ""),
                "page_title": page_data.get("title", ""),
                "links": page_data.get("links", []),
                "action_history": action_summary,
                "steps_taken": steps_taken,
                "error": failure_reason,
            }

        except Exception as e:
            logger.error(
                "[BrowserAdapter] Task #%d: async execution failed: %s",
                task_id, e, exc_info=True,
            )
            # Try to get current page state even on failure
            page_data = {}
            try:
                page_data = await self._extract_page_data(task_id)
            except Exception:
                pass

            return {
                "success": False,
                "final_url": page_data.get("url", ""),
                "page_title": page_data.get("title", ""),
                "links": page_data.get("links", []),
                "action_history": [],
                "steps_taken": 0,
                "error": str(e),
            }

    # ─────────────────────────────────────────────────────────
    # Browser lifecycle (crash detection + lazy creation)
    # ─────────────────────────────────────────────────────────

    async def _ensure_browser(self, task_id: int = 0) -> None:
        """Ensure browser is alive. Recreate on crash or first call.

        Checks browser connectivity before reuse. If Chrome was closed
        by the user or crashed, creates a fresh instance.

        The browser instance is PERSISTENT across tasks — only recreated
        on crash or first call.
        """
        if self._browser is not None:
            # Check if browser is still connected via CDP-level signals.
            # DO NOT use Playwright internals (browser.contexts[0].pages) —
            # browser-use manages targets independently via CDP, so
            # Playwright's page list can be empty even when CDP is alive.
            try:
                is_cdp_alive = getattr(
                    self._browser, 'is_cdp_connected', False,
                )
                conn = getattr(self._browser, 'connection', None)
                is_ws_open = (
                    getattr(conn, 'is_open', True) if conn else True
                )
                if is_cdp_alive and is_ws_open:
                    logger.debug(
                        "[BrowserAdapter] Task #%d: browser alive "
                        "(CDP connected, WebSocket open)",
                        task_id,
                    )
                    return
                # CDP or WebSocket dead — browser needs recreation
                logger.info(
                    "[BrowserAdapter] Task #%d: browser disconnected "
                    "(cdp=%s, ws=%s), recreating...",
                    task_id, is_cdp_alive, is_ws_open,
                )
            except Exception as e:
                logger.info(
                    "[BrowserAdapter] Task #%d: browser check failed (%s), "
                    "recreating...",
                    task_id, e,
                )
            # Close the dead instance
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None

        # Create fresh browser
        chrome_exe = _detect_chrome_executable(self._chrome_exe)
        if not chrome_exe:
            raise RuntimeError(
                "Chrome executable not found. Install Chrome or set "
                "chrome_executable in config/browser.yaml"
            )

        # Use dedicated browser profile — NOT the user's personal Chrome
        user_data_dir = self._user_data_dir or _DEFAULT_USER_DATA_DIR
        profile_dir = self._chrome_profile or "Default"

        # Ensure user_data_dir exists
        os.makedirs(os.path.join(user_data_dir, profile_dir), exist_ok=True)

        logger.info(
            "[BrowserAdapter] Task #%d: creating browser\n"
            "  Chrome: %s\n"
            "  Profile: %s\n"
            "  User data: %s\n"
            "  Headless: %s\n"
            "  Keep alive: %s",
            task_id, chrome_exe, profile_dir,
            user_data_dir, self._headless, self._keep_alive,
        )

        self._browser = BrowserUseBrowser(
            executable_path=chrome_exe,
            user_data_dir=user_data_dir,
            profile_directory=profile_dir,
            headless=self._headless,
            keep_alive=self._keep_alive,
        )

        logger.info(
            "[BrowserAdapter] Task #%d: browser created successfully "
            "(using dedicated profile at %s/%s)",
            task_id, user_data_dir, profile_dir,
        )

    def _ensure_llm(self, task_id: int = 0) -> None:
        """Create LLM client once. Reused across all tasks.

        Uses round-robin key rotation from _BrowserKeyPool.
        The key is set as GOOGLE_API_KEY env var (required by
        browser-use/langchain).
        """
        # Rotate API key on every task for round-robin
        key = _browser_key_pool.next_key()
        if key:
            os.environ["GOOGLE_API_KEY"] = key
            logger.debug(
                "[BrowserAdapter] Task #%d: set GOOGLE_API_KEY = %s...%s "
                "(pool size: %d)",
                task_id, key[:8],
                key[-4:] if len(key) > 8 else "****",
                _browser_key_pool.pool_size,
            )
        elif not os.environ.get("GOOGLE_API_KEY"):
            logger.warning(
                "[BrowserAdapter] Task #%d: no API key available! "
                "Set GOOGLE_BROWSER_AGENT_API_KEY or GOOGLE_BROWSER_AGENT_API_KEYS "
                "in .env",
                task_id,
            )

        if self._llm is not None:
            return

        self._llm = ChatGoogle(model=self._model_name)
        logger.info(
            "[BrowserAdapter] Task #%d: LLM created (once): "
            "ChatGoogle(model=%s)",
            task_id, self._model_name,
        )

    # ─────────────────────────────────────────────────────────
    # DOM extraction (query live page, not history)
    # ─────────────────────────────────────────────────────────

    async def _extract_page_data(self, task_id: int = 0) -> Dict[str, Any]:
        """Extract structured data from the current page via DOM queries.

        Uses browser-use's native BrowserSession API (CDP-based):
        - BrowserSession.get_current_page_url()
        - BrowserSession.get_current_page_title()
        - Page.evaluate() for link extraction

        Returns:
            url: current page URL
            title: page title
            links: list of {index, title, url} (capped at max_links)
        """
        result: Dict[str, Any] = {"url": "", "title": "", "links": []}

        if self._browser is None:
            logger.debug(
                "[BrowserAdapter] Task #%d: no browser for extraction",
                task_id,
            )
            return result

        # ── Extract URL ──
        try:
            url = await self._browser.get_current_page_url()
            result["url"] = url or ""
            logger.debug(
                "[BrowserAdapter] Task #%d: got URL via get_current_page_url(): %s",
                task_id, result["url"][:100],
            )
        except Exception as e:
            logger.debug(
                "[BrowserAdapter] Task #%d: get_current_page_url() failed: %s",
                task_id, e,
            )

        # ── Extract title ──
        try:
            title = await self._browser.get_current_page_title()
            result["title"] = title or ""
            logger.debug(
                "[BrowserAdapter] Task #%d: got title via get_current_page_title(): %s",
                task_id, result["title"][:80],
            )
        except Exception as e:
            logger.debug(
                "[BrowserAdapter] Task #%d: get_current_page_title() failed: %s",
                task_id, e,
            )

        # ── Extract links via DOM evaluation ──
        try:
            page = await self._get_active_page(task_id)
            if page is None:
                logger.debug(
                    "[BrowserAdapter] Task #%d: no active page for link extraction",
                    task_id,
                )
                return result

            logger.debug(
                "[BrowserAdapter] Task #%d: extracting links from DOM...",
                task_id,
            )

            links_raw = await page.evaluate("""
                () => {
                    const links = [];
                    const anchors = document.querySelectorAll('a[href]');
                    const seen = new Set();
                    for (const a of anchors) {
                        const href = a.href;
                        const text = (a.innerText || a.textContent || '').trim();
                        if (!href || href.startsWith('javascript:') ||
                            href.startsWith('mailto:') || href === '#' ||
                            !text || text.length < 3) {
                            continue;
                        }
                        if (seen.has(href)) continue;
                        seen.add(href);
                        links.push({title: text.substring(0, 120), url: href});
                    }
                    return links;
                }
            """)

            # Page.evaluate() returns str, not parsed objects
            import json
            links_data = json.loads(links_raw) if isinstance(links_raw, str) else (links_raw or [])

            total_raw = len(links_data)

            # Cap at configured limit
            max_links = self._max_links
            capped_links = []
            for i, link in enumerate(links_data[:max_links]):
                capped_links.append({
                    "index": i + 1,
                    "title": link.get("title", ""),
                    "url": link.get("url", ""),
                })
            result["links"] = capped_links

            logger.info(
                "[BrowserAdapter] Task #%d: extracted %d links "
                "(capped from %d total) from %s",
                task_id, len(capped_links), total_raw, result["url"][:80],
            )

            # Log first few links for debugging
            for link in capped_links[:5]:
                logger.debug(
                    "[BrowserAdapter] Task #%d:   [%d] %s → %s",
                    task_id, link["index"],
                    link["title"][:60], link["url"][:80],
                )
            if len(capped_links) > 5:
                logger.debug(
                    "[BrowserAdapter] Task #%d:   ... and %d more links",
                    task_id, len(capped_links) - 5,
                )

        except Exception as e:
            logger.warning(
                "[BrowserAdapter] Task #%d: link extraction failed: %s",
                task_id, e,
            )

        return result

    async def _get_active_page(self, task_id: int = 0):
        """Get the active Page from the browser-use BrowserSession.

        browser-use uses CDP (Chrome DevTools Protocol), not Playwright.
        The correct API is BrowserSession.get_current_page() which returns
        a browser_use.actor.Page (NOT a Playwright Page).

        Probe order:
          1. BrowserSession.get_current_page() — native API (preferred)
          2. Fallback: None (page not available)
        """
        try:
            if self._browser is None:
                return None

            # Path 1: browser-use's native API (CDP-based)
            if hasattr(self._browser, 'get_current_page'):
                page = await self._browser.get_current_page()
                if page is not None:
                    logger.debug(
                        "[BrowserAdapter] Task #%d: got page via "
                        "BrowserSession.get_current_page()",
                        task_id,
                    )
                    return page

            logger.debug(
                "[BrowserAdapter] Task #%d: get_current_page() returned None",
                task_id,
            )

        except Exception as e:
            logger.debug(
                "[BrowserAdapter] Task #%d: could not get active page: %s",
                task_id, e,
            )

        return None

    # ─────────────────────────────────────────────────────────
    # History extraction (supplementary debug logging)
    # ─────────────────────────────────────────────────────────

    def _extract_action_history(
        self, history, task_id: int = 0,
    ) -> List[str]:
        """Extract human-readable action summaries from AgentHistoryList.

        Used for logging and metadata, NOT for structured data extraction.
        DOM extraction via _extract_page_data() is the primary data source.
        """
        summaries = []
        try:
            if hasattr(history, 'history') and history.history:
                for i, step in enumerate(history.history):
                    summary = ""
                    if hasattr(step, 'action') and step.action:
                        summary = str(step.action)[:200]
                    elif hasattr(step, 'result') and step.result:
                        summary = str(step.result)[:200]

                    if summary:
                        summaries.append(summary)
                        logger.debug(
                            "[BrowserAdapter] Task #%d: step %d → %s",
                            task_id, i + 1, summary[:120],
                        )

            elif hasattr(history, 'final_result') and history.final_result:
                final = str(history.final_result())[:500]
                summaries.append(final)
                logger.debug(
                    "[BrowserAdapter] Task #%d: final_result → %s",
                    task_id, final[:200],
                )

        except Exception as e:
            logger.debug(
                "[BrowserAdapter] Task #%d: history extraction failed: %s",
                task_id, e,
            )

        return summaries

    @staticmethod
    def _format_action_log(actions: List[str]) -> str:
        """Format actions for multi-line log output."""
        if not actions:
            return "    (no actions recorded)"
        lines = []
        for i, action in enumerate(actions):
            lines.append(f"    [{i + 1}] {action[:120]}")
        return "\n".join(lines)

    def _detect_task_failure(
        self, history, steps_taken: int, task_id: int = 0,
    ) -> tuple:
        """Detect whether the browser agent task actually failed.

        browser-use's agent.run() does NOT throw exceptions on task failure.
        It returns a history that may contain errors. If we don't check,
        MERLIN reports "success" to the user — which is a lie.

        Checks (in order):
            1. final_result() contains error keywords
            2. History steps contain error entries
            3. Zero steps taken (agent couldn't start)
            4. Agent hit max_steps (likely stuck/failed)

        Returns:
            (failed: bool, reason: str)
        """
        # ── 1. Check final_result for error indicators ──
        try:
            if hasattr(history, 'final_result') and callable(history.final_result):
                final = str(history.final_result() or "")
                final_lower = final.lower()
                error_keywords = [
                    "error", "failed", "could not", "unable to",
                    "not found", "timed out", "timeout",
                    "exception", "crash", "profile copy failed",
                    "cannot", "refused", "blocked",
                ]
                for keyword in error_keywords:
                    if keyword in final_lower:
                        logger.warning(
                            "[BrowserAdapter] Task #%d: FAILURE detected "
                            "in final_result: '%s' (keyword: '%s')",
                            task_id, final[:200], keyword,
                        )
                        return (True, final[:300])
        except Exception as e:
            logger.debug(
                "[BrowserAdapter] Task #%d: final_result check error: %s",
                task_id, e,
            )

        # ── 2. Check history steps for errors ──
        try:
            if hasattr(history, 'history') and history.history:
                for step in history.history:
                    # Check for error fields
                    if hasattr(step, 'error') and step.error:
                        error_msg = str(step.error)[:300]
                        logger.warning(
                            "[BrowserAdapter] Task #%d: FAILURE detected "
                            "in history step error: %s",
                            task_id, error_msg,
                        )
                        return (True, error_msg)
                    # Check for is_error flag
                    if hasattr(step, 'is_error') and step.is_error:
                        result_msg = str(getattr(step, 'result', 'unknown error'))[:300]
                        logger.warning(
                            "[BrowserAdapter] Task #%d: FAILURE detected "
                            "(step.is_error=True): %s",
                            task_id, result_msg,
                        )
                        return (True, result_msg)
        except Exception as e:
            logger.debug(
                "[BrowserAdapter] Task #%d: history error check failed: %s",
                task_id, e,
            )

        # ── 3. Zero steps = agent couldn't even start ──
        if steps_taken == 0:
            logger.warning(
                "[BrowserAdapter] Task #%d: FAILURE — zero steps taken "
                "(agent couldn't start)",
                task_id,
            )
            return (True, "Browser agent completed zero steps")

        # ── 4. Check if agent was stuck (no success signal) ──
        # If agent hit max_steps, it likely failed to complete
        try:
            if hasattr(history, 'history') and history.history:
                max_steps_cfg = self._max_steps
                if len(history.history) >= max_steps_cfg:
                    # Check if the last step looks like completion or exhaustion
                    last_step = history.history[-1]
                    last_action = str(getattr(last_step, 'action', ''))
                    if 'done' not in last_action.lower():
                        logger.warning(
                            "[BrowserAdapter] Task #%d: possible failure — "
                            "hit max_steps (%d) without 'done' action",
                            task_id, max_steps_cfg,
                        )
                        # Not a hard failure — just a warning
                        # Some tasks legitimately use all steps
        except Exception:
            pass

        return (False, "")

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _error_result(message: str) -> Dict[str, Any]:
        """Build a structured error result."""
        return {
            "success": False,
            "final_url": "",
            "page_title": "",
            "links": [],
            "action_history": [],
            "steps_taken": 0,
            "error": message,
        }
