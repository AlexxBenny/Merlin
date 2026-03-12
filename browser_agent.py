"""
AURA v3 - Browser Agent Module
Integrates browser-use library for AI-driven browser automation.
Uses Gemini (ChatGoogle) as the LLM and Playwright for browser control.

ARCHITECTURAL NOTE:
This module uses a dedicated QThread + ProactorEventLoop pattern to support 
asyncio subprocesses (Playwright) correctly on Windows within a PyQt5 application.
"""

import sys
import os
import asyncio
import logging
from typing import Optional

# 1. SET POLICY IMMEDIATELY (Critical for Windows + Playwright)
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception as e:
        logging.warning(f"Could not set WindowsProactorEventLoopPolicy: {e}")

# Check availability
BROWSER_USE_AVAILABLE = False
GEMINI_AVAILABLE = False
PYQT_AVAILABLE = False

try:
    from browser_use import Agent
    from browser_use import BrowserSession as Browser
    from browser_use import BrowserProfile
    BROWSER_USE_AVAILABLE = True
except ImportError:
    pass

try:
    from browser_use import ChatGoogle
    GEMINI_AVAILABLE = True
except ImportError:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI as ChatGoogle
        GEMINI_AVAILABLE = True
    except ImportError:
        pass

try:
    from PyQt5.QtCore import QThread, pyqtSignal, QCoreApplication
    PYQT_AVAILABLE = True
except ImportError:
    pass


def _ensure_google_api_key():
    """Ensure GOOGLE_API_KEY is set for the browser-use library.

    Priority:
      1. GEMINI_API_KEY_BROWSER  (dedicated browser key)
      2. GOOGLE_API_KEY          (already in env)
      3. GEMINI_API_KEY          (shared fallback)
    """
    # Use the dedicated browser key if available
    from config import config
    browser_key = config.api_key_browser  # reads GEMINI_API_KEY_BROWSER or falls back

    if browser_key:
        # browser-use reads GOOGLE_API_KEY; set it to our chosen key
        first_key = browser_key.split(",")[0].strip()
        os.environ["GOOGLE_API_KEY"] = first_key
        logging.info(
            f"[BrowserAgent] Using key: {first_key[:10]}...{first_key[-4:]} "
            f"(GEMINI_API_KEY_BROWSER → GOOGLE_API_KEY)"
        )
        return True

    # Final fallback: already in env?
    return bool(os.environ.get("GOOGLE_API_KEY"))


# 2. DEFINE WORKER THREAD
if PYQT_AVAILABLE:
    class BrowserWorker(QThread):
        """
        Dedicated worker thread for running the async browser agent.
        Create a fresh Proactor event loop for every task to ensure clean state.
        """
        finished = pyqtSignal(str)
        
        def __init__(self, task: str, url: str = None, max_steps: int = 20):
            super().__init__()
            self.task = task
            self.url = url
            self.max_steps = max_steps
            self.result = ""
            
        def run(self):
            # Create a NEW event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                self.result = loop.run_until_complete(
                    self._run_async()
                )
            except Exception as e:
                self.result = f"Browser Error: {str(e)}"
            finally:
                # Cleanup
                try:
                    pending = asyncio.all_tasks(loop)
                    for t in pending:
                        t.cancel()
                    if pending:
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    loop.close()
                except:
                    pass
            
            self.finished.emit(self.result)

        async def _run_async(self):
            if not BROWSER_USE_AVAILABLE:
                return "Error: browser-use not installed."
            if not GEMINI_AVAILABLE:
                return "Error: ChatGoogle not installed."
                
            _ensure_google_api_key()
            
            # Initialize
            # Note: We create new instances here to be thread-safe
            llm = ChatGoogle(model="gemini-2.5-flash")
            
            # Detect which Chrome profile is actively being used
            import os
            import glob
            from pathlib import Path
            
            user_data_dir = os.path.expanduser(r'~\AppData\Local\Google\Chrome\User Data')
            
            # Find the most recently used profile by checking Preferences file modification time
            profile_dir = 'Default'  # Fallback
            try:
                profile_paths = glob.glob(os.path.join(user_data_dir, 'Profile *'))
                profile_paths.append(os.path.join(user_data_dir, 'Default'))
                
                latest_time = 0
                for profile_path in profile_paths:
                    pref_file = os.path.join(profile_path, 'Preferences')
                    if os.path.exists(pref_file):
                        mtime = os.path.getmtime(pref_file)
                        if mtime > latest_time:
                            latest_time = mtime
                            profile_dir = os.path.basename(profile_path)
                
                print(f"Using Chrome profile: {profile_dir}")
            except Exception as e:
                print(f"Could not detect active profile, using Default: {e}")
            
            browser = Browser(
                executable_path='C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
                user_data_dir=user_data_dir,
                profile_directory=profile_dir,
                headless=False,  # Show the browser
                # Keep browser alive after task completes so you can see results
                keep_alive=True
            )
            
            full_task = self.task
            if self.url:
                full_task = f"Navigate to {self.url} and then {self.task}"
                
            agent = Agent(
                task=full_task,
                llm=llm,
                browser=browser,
                max_steps=self.max_steps,
            )
            
            try:
                history = await agent.run()
                
                # Extract result
                if hasattr(history, 'final_result') and history.final_result:
                    return str(history.final_result())
                elif hasattr(history, 'history'):
                     return f"Task completed in {len(history.history)} steps."
                else:
                    return str(history)
            finally:
                await browser.close()


def run_browser_task(task: str, url: str = None, max_steps: int = 20, timeout: int = 120) -> str:
    """
    Main entry point.
    Runs the BrowserWorker and waits for it to finish.
    Safe to call from a BACKGROUND thread (like FunctionExecutor).
    """
    if not PYQT_AVAILABLE:
        return "Error: PyQt5 not available for browser execution."
        
    if not is_available():
        return "Error: browser-use dependencies missing."

    # Create worker
    worker = BrowserWorker(task, url, max_steps)
    
    # Run the thread
    worker.start()
    
    # Wait for execution to finish
    # Since we are likely already in a QThread (ProcessingThread), 
    # blocking here via wait() is safe and won't freeze the GUI.
    if worker.wait(timeout * 1000):
        return worker.result
    else:
        # Timeout
        worker.terminate()
        worker.wait()
        return "Browser task timed out."


def is_available() -> bool:
    return BROWSER_USE_AVAILABLE and GEMINI_AVAILABLE and PYQT_AVAILABLE

# Self-test logic
if __name__ == "__main__":
    if is_available():
        print("Available. To test, run within a PyQt application context.")
    else:
        print(f"Not available. BrowserUse: {BROWSER_USE_AVAILABLE}, Gemini: {GEMINI_AVAILABLE}, PyQt: {PYQT_AVAILABLE}")
