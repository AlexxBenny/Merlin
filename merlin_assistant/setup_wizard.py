# merlin_assistant/setup_wizard.py

"""
MERLIN Interactive Setup Wizard.

Runs on `merlin init`. Guides user through:
  1. Preflight checks (Python version, writable config dir)
  2. Config directory creation
  3. LLM provider selection + API key validation
  4. Config file generation
  5. Connectivity validation

Design rules:
  - NEVER import main.py (module-level side effects: load_dotenv, logging)
  - Validate using direct LLM client instantiation, NOT ModelRouter
  - Write .env manually (no load_dotenv dependency)
  - Fail fast with actionable error messages
"""

import getpass
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────
# ANSI helpers (best-effort, degrades gracefully on dumb terms)
# ─────────────────────────────────────────────────────────────

def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if sys.platform == "win32":
        return os.environ.get("ANSICON") or "WT_SESSION" in os.environ
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

_COLOR = _supports_color()

def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m" if _COLOR else s

def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m" if _COLOR else s

def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m" if _COLOR else s

def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m" if _COLOR else s

def _cyan(s: str) -> str:
    return f"\033[36m{s}\033[0m" if _COLOR else s


# ─────────────────────────────────────────────────────────────
# Preflight
# ─────────────────────────────────────────────────────────────

def _preflight() -> None:
    """Check minimum requirements before starting setup."""
    print(f"\n{_bold('🔍 Preflight checks...')}")

    # Python version
    if sys.version_info < (3, 10):
        print(_red(f"  ✗ Python {sys.version_info.major}.{sys.version_info.minor} "
                    f"detected — MERLIN requires Python 3.10+"))
        sys.exit(1)
    print(f"  {_green('✓')} Python {sys.version_info.major}.{sys.version_info.minor}")

    # Config dir writable
    from merlin_assistant.config_discovery import get_user_config_root
    config_root = get_user_config_root()
    try:
        config_root.mkdir(parents=True, exist_ok=True)
        test_file = config_root / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        print(f"  {_green('✓')} Config directory writable: {config_root}")
    except (PermissionError, OSError) as e:
        print(_red(f"  ✗ Cannot write to {config_root}: {e}"))
        sys.exit(1)


# ─────────────────────────────────────────────────────────────
# Config directory setup
# ─────────────────────────────────────────────────────────────

def _prepare_directory(force: bool) -> Path:
    """Create the user config directory. Prompt before overwriting."""
    from merlin_assistant.config_discovery import get_user_config_root
    config_root = get_user_config_root()
    config_dir = config_root / "config"

    if config_dir.exists() and not force:
        print(f"\n{_yellow('⚠')} Existing configuration found at:")
        print(f"  {config_dir}")
        answer = input("\n  Overwrite? [y/N]: ").strip().lower()
        if answer != "y":
            print("  Setup cancelled.")
            sys.exit(0)
        shutil.rmtree(config_dir)

    config_dir.mkdir(parents=True, exist_ok=True)
    return config_root


# ─────────────────────────────────────────────────────────────
# Provider selection
# ─────────────────────────────────────────────────────────────

def _choose_provider() -> str:
    """Ask user to choose an LLM provider. Returns 'cloud' or 'ollama'."""
    print(f"\n{_bold('🧠 Choose how MERLIN should think:')}\n")
    print(f"  {_cyan('1)')} Cloud {_green('(recommended)')} — fastest setup, requires API key")
    print(f"  {_cyan('2)')} Local (Ollama) — runs on your machine, no API key needed")

    while True:
        choice = input(f"\n  Select [1/2]: ").strip()
        if choice == "1":
            return "cloud"
        elif choice == "2":
            return "ollama"
        print(_yellow("  Please enter 1 or 2"))


def _choose_cloud_provider() -> str:
    """Ask which cloud provider to use. Returns 'openrouter' or 'gemini'."""
    print(f"\n{_bold('☁ Choose cloud provider:')}\n")
    print(f"  {_cyan('1)')} OpenRouter {_green('(recommended)')} — access to 200+ models")
    print(f"  {_cyan('2)')} Google Gemini — Google's models")

    while True:
        choice = input(f"\n  Select [1/2]: ").strip()
        if choice == "1":
            return "openrouter"
        elif choice == "2":
            return "gemini"
        print(_yellow("  Please enter 1 or 2"))


# ─────────────────────────────────────────────────────────────
# Cloud setup (OpenRouter / Gemini)
# ─────────────────────────────────────────────────────────────

def _validate_openrouter_key(api_key: str) -> bool:
    """Validate an OpenRouter API key using the existing client."""
    try:
        from models.openrouter_client import OpenRouterClient
        client = OpenRouterClient(
            model="openai/gpt-4o-mini",  # doesn't matter for is_available
            api_key=api_key,
        )
        return client.is_available()
    except Exception:
        return False


def _validate_gemini_key(api_key: str) -> bool:
    """Validate a Gemini API key using the existing client."""
    try:
        from models.gemini_client import GeminiClient
        client = GeminiClient(
            model="gemini-2.0-flash",
            api_key=api_key,
        )
        return client.is_available()
    except Exception:
        return False


def _setup_cloud() -> Tuple[str, str, str]:
    """Guide cloud provider setup. Returns (provider, api_key, model).

    Loops until valid key, or user switches/exits.
    """
    provider = _choose_cloud_provider()

    if provider == "openrouter":
        env_var = "OPENROUTER_API_KEY"
        default_model = "openai/gpt-4o-mini"
        validate_fn = _validate_openrouter_key
        key_hint = "sk-or-..."
    else:
        env_var = "GEMINI_API_KEY"
        default_model = "gemini-2.0-flash"
        validate_fn = _validate_gemini_key
        key_hint = "AIza..."

    while True:
        print(f"\n{_bold(f'🔑 Enter your {provider.title()} API key')}")
        print(f"  (starts with {_cyan(key_hint)})")
        api_key = getpass.getpass("  API key (hidden): ").strip()

        if not api_key:
            print(_yellow("  No key entered."))
            continue

        print(f"  Validating...", end="", flush=True)
        if validate_fn(api_key):
            print(f" {_green('✓ Valid!')}")
            return provider, api_key, default_model
        else:
            print(f" {_red('✗ Rejected')}")
            print(f"\n  {_red('API key rejected by ' + provider.title())}")
            print(f"\n  Possible reasons:")
            print(f"  - Key is incorrect or expired")
            print(f"  - Account has no credits")
            print(f"  - Network connectivity issue")
            print(f"\n  Options:")
            print(f"    {_cyan('1)')} Retry")
            print(f"    {_cyan('2)')} Switch to Ollama (local)")
            print(f"    {_cyan('3)')} Exit")

            choice = input(f"\n  Select [1/2/3]: ").strip()
            if choice == "2":
                return _setup_ollama_flow()
            elif choice == "3":
                print("  Setup cancelled.")
                sys.exit(0)
            # else: retry loop


# ─────────────────────────────────────────────────────────────
# Ollama setup
# ─────────────────────────────────────────────────────────────

def _check_ollama_running() -> bool:
    """Check if Ollama server is reachable."""
    try:
        from models.ollama_client import OllamaClient
        client = OllamaClient()
        return client.is_available()
    except Exception:
        return False


def _list_ollama_models() -> list:
    """List locally available Ollama models via /api/tags."""
    try:
        from urllib import request
        req = request.Request(
            "http://localhost:11434/api/tags",
            method="GET",
        )
        with request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("models", [])
            return [m.get("name", "unknown") for m in models]
    except Exception:
        return []


def _setup_ollama_flow() -> Tuple[str, str, str]:
    """Guide Ollama setup. Returns ('ollama', '', model_name)."""
    print(f"\n{_bold('🦙 Ollama Setup')}")

    while True:
        print(f"\n  Checking Ollama...", end="", flush=True)

        if not _check_ollama_running():
            print(f" {_red('✗ Not running')}")
            print(f"\n  {_red('Ollama is not installed or not running.')}")
            print(f"\n  Install from: {_cyan('https://ollama.com')}")
            print(f"  Then run:     {_cyan('ollama serve')}")
            print(f"\n  Options:")
            print(f"    {_cyan('1)')} Retry")
            print(f"    {_cyan('2)')} Switch to cloud")
            print(f"    {_cyan('3)')} Exit")

            choice = input(f"\n  Select [1/2/3]: ").strip()
            if choice == "2":
                return _setup_cloud()
            elif choice == "3":
                print("  Setup cancelled.")
                sys.exit(0)
            continue

        print(f" {_green('✓ Running')}")

        # List models
        models = _list_ollama_models()
        if not models:
            print(f"\n  {_yellow('⚠ No models found.')}")
            print(f"\n  To download a model, run:")
            print(f"    {_cyan('ollama pull llama3')}")
            print(f"\n  Press Enter after downloading...")
            input()
            continue

        # Show available models
        print(f"\n  {_bold('Available models:')}")
        for i, m in enumerate(models[:10], 1):
            print(f"    {_cyan(f'{i})')} {m}")

        while True:
            choice = input(f"\n  Select model [1-{min(len(models), 10)}]: ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < min(len(models), 10):
                    selected = models[idx]
                    print(f"  Selected: {_green(selected)}")
                    return "ollama", "", selected
            except ValueError:
                pass
            print(_yellow(f"  Please enter a number 1-{min(len(models), 10)}"))


# ─────────────────────────────────────────────────────────────
# Config writers
# ─────────────────────────────────────────────────────────────

def _write_env(config_root: Path, provider: str, api_key: str) -> None:
    """Write .env file with the API key. Manual write — no load_dotenv."""
    env_path = config_root / ".env"

    lines = []
    if provider == "openrouter":
        lines.append(f"OPENROUTER_API_KEY={api_key}")
    elif provider == "gemini":
        lines.append(f"GEMINI_API_KEY={api_key}")

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  {_green('✓')} API key saved to {env_path}")


def _write_models_yaml(config_dir: Path, provider: str, model: str) -> None:
    """Write a minimal models.yaml with all roles pointing to the chosen provider."""
    roles = [
        "mission_compiler",
        "report_generator",
        "clarifier",
        "relational_responder",
        "speech_act_classifier",
        "proactive_scorer",
        "safety_evaluator",
        "narrator",
        "content_generator",
        "cognitive_coordinator",
    ]

    lines = [
        "# models.yaml — Generated by merlin init",
        "# All roles use the same provider. Customize later as needed.",
        "",
    ]

    for role in roles:
        lines.append(f"{role}:")
        lines.append(f"  provider: {provider}")
        lines.append(f"  model: {model}")
        if provider == "openrouter":
            lines.append("  temperature: 0.2")
            lines.append("  max_tokens: 4096")
            lines.append("  timeout: 10.0")
        elif provider == "gemini":
            lines.append("  temperature: 0.2")
            lines.append("  max_tokens: 4096")
            lines.append("  timeout: 10.0")
        elif provider == "ollama":
            lines.append("  temperature: 0.2")
            lines.append("  timeout: 120.0")
        lines.append("")

    config_path = config_dir / "models.yaml"
    config_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  {_green('✓')} Models config written to {config_path}")


def _copy_default_configs(config_dir: Path) -> None:
    """Copy bundled default configs to user config directory."""
    from merlin_assistant.config_discovery import _get_package_default_dir
    defaults = _get_package_default_dir()

    if not defaults.is_dir():
        print(_yellow(f"  ⚠ No bundled defaults found at {defaults}"))
        return

    copied = 0
    for src in defaults.iterdir():
        if src.name == "models.yaml":
            # Skip — we wrote this already with the chosen provider
            continue
        dst = config_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

    if copied:
        print(f"  {_green('✓')} Copied {copied} default config files")


# ─────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────

def _validate_setup(provider: str, api_key: str, model: str) -> bool:
    """Validate that the chosen provider is reachable.

    Uses direct client instantiation — NOT ModelRouter.
    This avoids importing main.py and its module-level side effects.
    """
    print(f"\n{_bold('🔍 Validating setup...')}")

    try:
        if provider == "openrouter":
            from models.openrouter_client import OpenRouterClient
            client = OpenRouterClient(model=model, api_key=api_key)
        elif provider == "gemini":
            from models.gemini_client import GeminiClient
            client = GeminiClient(model=model, api_key=api_key)
        elif provider == "ollama":
            from models.ollama_client import OllamaClient
            client = OllamaClient(model=model)
        else:
            print(f"  {_yellow('⚠ Unknown provider, skipping validation')}")
            return True

        if client.is_available():
            print(f"  {_green('✓')} {provider.title()} is reachable")
            return True
        else:
            print(f"  {_red('✗')} {provider.title()} is not reachable")
            return False

    except Exception as e:
        print(f"  {_red('✗')} Validation error: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────

def run_init(force: bool = False) -> None:
    """Run the interactive setup wizard."""
    print(f"\n{_bold('🧙 Welcome to MERLIN Setup!')}")
    print(f"   Deterministic Autonomous AI Assistant\n")

    # Step 0: Preflight
    _preflight()

    # Step 1: Prepare config directory
    config_root = _prepare_directory(force)
    config_dir = config_root / "config"

    # Step 2: Choose provider
    provider_type = _choose_provider()

    # Step 3: Setup chosen provider
    if provider_type == "cloud":
        provider, api_key, model = _setup_cloud()
    else:
        provider, api_key, model = _setup_ollama_flow()

    # Step 4: Write configs
    print(f"\n{_bold('📝 Writing configuration...')}")

    # Write .env (only for cloud providers)
    if api_key:
        _write_env(config_root, provider, api_key)

    # Write models.yaml
    _write_models_yaml(config_dir, provider, model)

    # Copy other default configs
    _copy_default_configs(config_dir)

    # Step 5: Validate
    if not _validate_setup(provider, api_key, model):
        print(f"\n{_yellow('⚠ Validation failed, but config was saved.')}")
        print(f"  You can fix the issue and run {_cyan('merlin')} to try again.\n")
        return

    # Step 6: Success
    print(f"\n{_green('═' * 50)}")
    print(f"{_green('  ✅ MERLIN is ready!')}")
    print(f"{_green('═' * 50)}")
    print(f"\n  Config saved to: {_cyan(str(config_root))}")
    print(f"\n  Run {_bold('merlin')} to start.")
    print(f"  Try: {_cyan('open notepad')}\n")
