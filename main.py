# main.py

"""
MERLIN Entry Point.

Constructs all components from config, registers skills,
creates the Merlin conductor, and runs the interactive loop.
"""

import importlib
import inspect
import logging
import sys
from pathlib import Path

import yaml

from brain.core import BrainCore
from brain.escalation_policy import EscalationPolicy
from cortex.mission_cortex import MissionCortex
from execution.registry import SkillRegistry
from infrastructure.location_config import LocationConfig
from world.timeline import WorldTimeline
from runtime.reflex_engine import ReflexEngine
from runtime.sources.base import EventSource
from runtime.sources.system import SystemSource
from runtime.sources.time import TimeSource
from runtime.sources.media import MediaSource
from reporting.report_builder import ReportBuilder
from reporting.output import ConsoleOutputChannel
from reporting.notification_policy import NotificationPolicy
from models.router import ModelRouter
from perception.text import TextPerception
from cortex.world_state_provider import WorldStateProvider, SimpleWorldStateProvider
from cortex.filtered_world_state_provider import FilteredWorldStateProvider
from cortex.context_provider import SimpleContextProvider, RetrievalContextProvider
from execution.executor import MissionExecutor
from memory.store import ListMemoryStore
from merlin import Merlin


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────

CONFIG_DIR = Path(__file__).parent / "config"


def load_yaml(filename: str) -> dict:
    """Load a YAML config file, return empty dict if missing."""
    path = CONFIG_DIR / filename
    if not path.exists():
        logger.warning("Config file not found: %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ─────────────────────────────────────────────────────────────
# Skill auto-loading from config
# ─────────────────────────────────────────────────────────────

def load_skills(
    registry: SkillRegistry,
    skills_config: dict,
    deps: dict,
) -> None:
    """
    Load skills from config/skills.yaml into the registry.

    Each entry: {name, module, class}

    Skills that require infrastructure (e.g., LocationConfig) receive
    it via constructor injection. The deps dict provides injectable
    objects keyed by type — the loader inspects __init__ signatures
    and injects matching parameters.
    """
    entries = skills_config.get("skills", [])
    for entry in entries:
        name = entry["name"]
        module_path = entry["module"]
        class_name = entry["class"]

        try:
            module = importlib.import_module(module_path)
            skill_class = getattr(module, class_name)

            # Inspect __init__ to inject dependencies
            sig = inspect.signature(skill_class.__init__)
            kwargs = {}
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                if param_name in deps:
                    kwargs[param_name] = deps[param_name]

            skill_instance = skill_class(**kwargs)
            registry.register(skill_instance)
            logger.info("Skill loaded: %s", name)
        except Exception as e:
            logger.error("Failed to load skill '%s': %s", name, e)


# ─────────────────────────────────────────────────────────────
# Event source construction from config
# ─────────────────────────────────────────────────────────────

def build_event_sources(execution_config: dict, system_controller=None) -> list:
    """
    Construct enabled event sources from execution.yaml.
    Sources that fail to construct are silently skipped.
    """
    sources_cfg = execution_config.get("sources", {})
    thresholds_cfg = execution_config.get("system_thresholds", {})
    sources: list = []

    # SystemSource
    sys_cfg = sources_cfg.get("system", {})
    if sys_cfg.get("enabled", False):
        try:
            sources.append(SystemSource(
                thresholds=thresholds_cfg,
                intervals={
                    "resource_poll_interval": sys_cfg.get("resource_poll_interval", 2.0),
                    "window_poll_interval": sys_cfg.get("window_poll_interval", 0.5),
                    "idle_poll_interval": sys_cfg.get("idle_poll_interval", 5.0),
                },
                system_controller=system_controller,
            ))
            logger.info("SystemSource enabled")
        except Exception as e:
            logger.warning("SystemSource construction failed: %s", e)

    # TimeSource
    time_cfg = sources_cfg.get("time", {})
    if time_cfg.get("enabled", False):
        try:
            sources.append(TimeSource(
                tick_interval=time_cfg.get("tick_interval", 60),
            ))
            logger.info("TimeSource enabled")
        except Exception as e:
            logger.warning("TimeSource construction failed: %s", e)

    # MediaSource
    media_cfg = sources_cfg.get("media", {})
    if media_cfg.get("enabled", False):
        try:
            sources.append(MediaSource(
                poll_interval=media_cfg.get("poll_interval", 1.0),
            ))
            logger.info("MediaSource enabled")
        except Exception as e:
            logger.warning("MediaSource construction failed: %s", e)

    return sources


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    """Build all components and run the interactive loop."""

    # ── Load configs ──
    models_config = load_yaml("models.yaml")
    skills_config = load_yaml("skills.yaml")
    execution_config = load_yaml("execution.yaml")
    routing_config = load_yaml("routing.yaml")

    # ── Model Router (role-based LLM clients) ──
    router = ModelRouter(models_config)
    compiler_client = None
    reporter_client = None

    try:
        compiler_client = router.get_client("mission_compiler")
        if not compiler_client.is_available():
            logger.warning("mission_compiler LLM not available")
            compiler_client = None
    except Exception as e:
        logger.warning("mission_compiler init failed: %s", e)

    try:
        reporter_client = router.get_client("report_generator")
        if not reporter_client.is_available():
            logger.warning("report_generator LLM not available")
            reporter_client = None
    except Exception as e:
        logger.warning("report_generator init failed: %s", e)

    clarifier_client = None
    try:
        clarifier_client = router.get_client("clarifier")
        if not clarifier_client.is_available():
            logger.warning("clarifier LLM not available")
            clarifier_client = None
    except Exception as e:
        logger.warning("clarifier init failed: %s", e)

    # ── Core components ──
    timeline = WorldTimeline()
    registry = SkillRegistry()

    # LocationConfig — infrastructure-only path resolution
    paths_yaml = CONFIG_DIR / "paths.yaml"
    location_config = LocationConfig.from_yaml(paths_yaml)

    # BrainCore — config-driven circuit breaker
    # ReflexEngine must be built first so BrainCore can delegate try_match()
    reflex = ReflexEngine(timeline, registry=registry)
    template_entries = routing_config.get("reflex_templates", [])
    if template_entries:
        templates = ReflexEngine.load_templates(template_entries)
        for t in templates:
            reflex.register_template(t)
        logger.info("Loaded %d reflex templates", len(templates))

    brain = BrainCore(
        mission_indicators=routing_config.get("mission_indicators", []),
        refuse_indicators=routing_config.get("refuse_indicators", []),
        reflex_engine=reflex,
    )

    # EscalationPolicy — context-aware triage
    escalation = EscalationPolicy(
        referential_markers=routing_config.get("referential_markers", []),
    )

    # ── Load skills (inject infrastructure deps) ──
    from infrastructure.system_controller import SystemController
    system_controller = SystemController()

    skill_deps = {
        "location_config": location_config,
        "system_controller": system_controller,
    }
    load_skills(registry, skills_config, deps=skill_deps)

    # ── Reporting ──
    output_channel = ConsoleOutputChannel(prefix="MERLIN")
    event_templates = execution_config.get("event_templates", {})
    report_builder = ReportBuilder(
        llm=reporter_client,
        draft_templates=event_templates,
    )

    # ── Notification policy ──
    notif_entries = execution_config.get("notifications", [])
    if notif_entries:
        notification_policy = NotificationPolicy.from_config(notif_entries)
    else:
        notification_policy = NotificationPolicy.default()

    # ── Event sources (from config) ──
    event_sources = build_event_sources(execution_config, system_controller=system_controller)

    # ── Memory ──
    memory_store = ListMemoryStore()

    # ── World State Provider (config-driven view projection) ──
    wsp_strategy = skills_config.get("world_state_provider", "simple")
    if wsp_strategy == "filtered":
        domain_map_cfg = skills_config.get("domain_state_mapping")
        world_state_provider = FilteredWorldStateProvider(
            domain_state_map=domain_map_cfg,
        )
        logger.info("WorldStateProvider: FilteredWorldStateProvider")
    else:
        world_state_provider = SimpleWorldStateProvider()
        logger.info("WorldStateProvider: SimpleWorldStateProvider")

    # ── Context Provider (config-driven temporal bounding) ──
    cp_strategy = skills_config.get("context_provider", "simple")
    if cp_strategy == "retrieval":
        context_provider = RetrievalContextProvider(
            memory=memory_store,
            token_budget=skills_config.get("context_token_budget", 800),
        )
        logger.info("ContextProvider: RetrievalContextProvider (budget=%d)",
                    skills_config.get("context_token_budget", 800))
    else:
        context_provider = SimpleContextProvider()
        logger.info("ContextProvider: SimpleContextProvider")

    # ── Cortex (after all providers are initialized) ──
    cortex = MissionCortex(
        llm_client=compiler_client,
        registry=registry,
        location_config=location_config,
        context_provider=context_provider,
    )

    # ── Build Merlin ──
    merlin = Merlin(
        brain=brain,
        escalation_policy=escalation,
        cortex=cortex,
        registry=registry,
        timeline=timeline,
        reflex_engine=reflex,
        report_builder=report_builder,
        output_channel=output_channel,
        notification_policy=notification_policy,
        event_sources=event_sources,
        max_workers=execution_config.get("executor", {}).get("max_workers", 4),
        node_timeout=execution_config.get("executor", {}).get("node_timeout_seconds"),
        clarifier_llm=clarifier_client,
        world_state_provider=world_state_provider,
        memory=memory_store,
    )


    # ── Perception ──
    text_perception = TextPerception()

    # ── Start runtime ──
    merlin.start()

    # ── Interactive loop ──
    print("=" * 60)
    print("  MERLIN — Online")
    print("  Type a command, or 'exit' to quit.")
    if compiler_client:
        print(f"  Compiler: {compiler_client.model} (temp={compiler_client.default_temperature})")
    else:
        print("  Compiler: not connected")
    if reporter_client:
        print(f"  Reporter: {reporter_client.model} (temp={reporter_client.default_temperature})")
    else:
        print("  Reporter: not connected")
    print("=" * 60)
    print()

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit", "q"}:
                break

            # Perceive → Route → Execute → Report
            percept = text_perception.perceive(user_input)
            merlin.handle_percept(percept)
            print()  # visual spacing

    except KeyboardInterrupt:
        print("\n\nInterrupted.")

    finally:
        merlin.stop()
        print("MERLIN — Offline.")


if __name__ == "__main__":
    main()
