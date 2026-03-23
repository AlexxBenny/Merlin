# main.py

"""
MERLIN Entry Point.

Constructs all components from config, registers skills,
creates the Merlin conductor, and runs the interactive loop.

Voice modes:
    python main.py               # text-only (default, unchanged)
    python main.py --voice       # voice-only (mic input, TTS output)
    python main.py --hybrid      # both text and voice (first wins per cycle)
"""

import argparse
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
from reporting.output import (
    ConsoleOutputChannel, SpeechOutputChannel, CompositeOutputChannel,
)
from reporting.notification_policy import NotificationPolicy
from models.router import ModelRouter
from perception.text import TextPerception
from perception.perception_orchestrator import PerceptionOrchestrator
from cortex.world_state_provider import WorldStateProvider, SimpleWorldStateProvider
from cortex.filtered_world_state_provider import FilteredWorldStateProvider
from cortex.context_provider import SimpleContextProvider, RetrievalContextProvider
from execution.executor import MissionExecutor
from memory.store import ListMemoryStore
from infrastructure.voice_factory import VoiceEngineFactory
from merlin import Merlin


# ── ANSI colored logging ──────────────────────────────────

class _ColoredFormatter(logging.Formatter):
    """Log formatter with ANSI colors per level. Keeps user output clean."""

    _COLORS = {
        logging.DEBUG:    "\033[90m",   # dim gray
        logging.INFO:     "\033[36m",   # cyan
        logging.WARNING:  "\033[33m",   # yellow
        logging.ERROR:    "\033[31m",   # red
        logging.CRITICAL: "\033[1;31m", # bold red
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelno, "")
        msg = super().format(record)
        return f"{color}{msg}{self._RESET}" if color else msg


class _DynamicStdoutHandler(logging.StreamHandler):
    """
    Handler that resolves sys.stdout at emit-time, not construction-time.

    Why: StreamHandler(sys.stdout) captures the original stdout object.
    When patch_stdout() later replaces sys.stdout with a proxy, the
    handler still writes to the old stream → prompt corruption.
    This handler always reads the CURRENT sys.stdout, so it automatically
    writes through the proxy after patch_stdout() activates.
    """

    def __init__(self):
        super().__init__()

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, _value):
        pass  # ignore — always use current sys.stdout


_handler = _DynamicStdoutHandler()
_handler.setFormatter(_ColoredFormatter(
    fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
))
logging.basicConfig(level=logging.INFO, handlers=[_handler])
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────

CONFIG_DIR = Path(__file__).parent / "config"

# Load .env into os.environ before any config reads
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")


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
            skip = False
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                if param_name in deps:
                    if deps[param_name] is None and param.default is inspect.Parameter.empty:
                        # Required dep is None — skip this skill entirely
                        logger.warning(
                            "Skill '%s' skipped: required dep '%s' is unavailable",
                            name, param_name,
                        )
                        skip = True
                        break
                    kwargs[param_name] = deps[param_name]

            if skip:
                continue

            skill_instance = skill_class(**kwargs)
            registry.register(skill_instance)
            logger.info("Skill loaded: %s", name)
        except Exception as e:
            logger.error("Failed to load skill '%s': %s", name, e)


# ─────────────────────────────────────────────────────────────
# Event source construction from config
# ─────────────────────────────────────────────────────────────

def build_event_sources(
    execution_config: dict,
    system_controller=None,
    timeline=None,
    app_registry=None,
    browser_controller=None,
) -> list:
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
                    "process_poll_interval": sys_cfg.get("process_poll_interval", 5.0),
                    "refresh_interval": sys_cfg.get("refresh_interval", 30.0),
                },
                system_controller=system_controller,
                timeline=timeline,
                app_registry=app_registry,
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

    # BrowserSource
    browser_cfg = sources_cfg.get("browser", {})
    if browser_cfg.get("enabled", False):
        try:
            from runtime.sources.browser import BrowserSource
            sources.append(BrowserSource(
                browser_controller=browser_controller,
                poll_interval=browser_cfg.get("poll_interval", 30.0),
            ))
            logger.info("BrowserSource enabled (poll_interval=%.0fs)",
                        browser_cfg.get("poll_interval", 30.0))
        except Exception as e:
            logger.warning("BrowserSource construction failed: %s", e)

    return sources


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main(args=None):
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

    # ── Structural Analyzer (with optional LLM speech-act classifier) ──
    from brain.structural_classifier import StructuralAnalyzer
    speech_act_llm = None
    try:
        speech_act_llm = router.get_client("speech_act_classifier")
        if not speech_act_llm.is_available():
            logger.warning("speech_act_classifier LLM not available — regex fallback")
            speech_act_llm = None
    except KeyError:
        logger.info("speech_act_classifier not configured — regex fallback")
    except Exception as e:
        logger.warning("speech_act_classifier init failed: %s — regex fallback", e)

    structural_analyzer = StructuralAnalyzer(llm=speech_act_llm)
    logger.info(
        "StructuralAnalyzer enabled (%s)",
        "LLM speech-act classifier" if speech_act_llm else "regex fallback",
    )

    # ── Core components ──
    timeline = WorldTimeline()
    registry = SkillRegistry()

    # LocationConfig — infrastructure-only path resolution
    paths_yaml = CONFIG_DIR / "paths.yaml"
    location_config = LocationConfig.from_yaml(paths_yaml)

    # BrainCore — three-stage hybrid routing
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
        relational_indicators=routing_config.get("relational_indicators", []),
        reflex_engine=reflex,
        analyzer=structural_analyzer,
    )

    # EscalationPolicy — context-aware triage
    escalation = EscalationPolicy(
        referential_markers=routing_config.get("referential_markers", []),
    )

    # ── Load skills (inject infrastructure deps) ──
    from infrastructure.system_controller import SystemController
    system_controller = SystemController()

    # ── Session Management (Phase 1: Interactive Execution Layer) ──
    from infrastructure.app_capabilities import AppCapabilityRegistry
    from infrastructure.session import SessionManager

    app_capabilities_path = CONFIG_DIR / "app_capabilities.yaml"
    capability_registry = AppCapabilityRegistry.from_yaml(str(app_capabilities_path))

    # ── Application Registry (boot-time app discovery) ──
    from infrastructure.app_discovery import ApplicationDiscoveryService
    from infrastructure.application_registry import ApplicationRegistry

    app_discovery = ApplicationDiscoveryService(capability_registry=capability_registry)
    app_registry = ApplicationRegistry()
    try:
        discovered_entities = app_discovery.discover_all()
        for entity in discovered_entities:
            app_registry.register(entity)
        summary = app_registry.summary()
        logger.info(
            "ApplicationRegistry: %d entities (%d desktop, %d UWP), "
            "%d names indexed, %d process names indexed",
            summary["total_entities"],
            summary["by_type"].get("desktop", 0),
            summary["by_type"].get("uwp", 0),
            summary["total_names_indexed"],
            summary["total_process_names_indexed"],
        )
    except Exception as e:
        logger.warning("Application discovery failed — registry empty: %s", e)

    # ── Entity Resolver (post-compilation transform — Phase 9C) ──
    from cortex.entity_resolver import EntityResolver
    import yaml as _yaml

    _alias_path = CONFIG_DIR / "app_aliases.yaml"
    _alias_map = {}
    if _alias_path.exists():
        try:
            with open(_alias_path, "r", encoding="utf-8") as f:
                _alias_map = _yaml.safe_load(f) or {}
            logger.info("Loaded %d app aliases from %s", len(_alias_map), _alias_path)
        except Exception as e:
            logger.warning("Failed to load app aliases: %s", e)

    # NOTE: EntityResolver needs skill_registry which isn't created yet.
    # We create it here with registry=None and wire skill_registry post-hoc
    # (same pattern as PreferenceResolver).
    entity_resolver = EntityResolver(
        registry=app_registry,
        skill_registry=None,  # wired after Merlin construction
        alias_map=_alias_map,
    )

    session_manager = SessionManager(
        capability_registry=capability_registry,
        timeline=timeline,
    )
    logger.info("SessionManager initialized with %d known app types",
                len(capability_registry.known_apps))

    # ── Environment Observer (Phase 2: Interactive Execution Layer) ──
    from infrastructure.observer import SystemEnvironmentObserver
    environment_observer = SystemEnvironmentObserver(controller=system_controller)
    logger.info("EnvironmentObserver initialized (SystemEnvironmentObserver)")

    # Content generation LLM (for reasoning.generate_text skill)
    content_llm = None
    try:
        content_llm = router.get_client("content_generator")
        if not content_llm.is_available():
            logger.warning("content_generator LLM not available")
            content_llm = None
    except Exception as e:
        logger.warning("content_generator init failed: %s", e)

    # ── Task store (created early so job skills can use it) ──
    task_store = None
    scheduler_config = execution_config.get("scheduler", {})
    if scheduler_config.get("enabled", False):
        try:
            import os
            from runtime.json_task_store import JsonTaskStore

            job_store_path = scheduler_config.get(
                "job_store_path", "state/jobs/jobs.json"
            )
            job_dir = os.path.dirname(job_store_path)
            if job_dir:
                os.makedirs(job_dir, exist_ok=True)

            task_store = JsonTaskStore(path=job_store_path)
        except Exception as e:
            logger.warning("Task store init failed: %s", e)

    # ── Browser-Use Adapter (browser AI automation) ──
    browser_config = load_yaml("browser.yaml")
    browser_adapter = None
    if browser_config.get("browser_use", {}).get("enabled", False):
        try:
            import os as _os
            from infrastructure.browser_use_adapter import BrowserUseAdapter

            browser_agent_key = _os.environ.get("GOOGLE_BROWSER_AGENT_API_KEY", "")
            browser_model = models_config.get("browser_agent", {}).get(
                "model", "gemini-2.5-flash"
            )

            browser_adapter = BrowserUseAdapter(
                config=browser_config.get("browser_use", {}),
                api_key=browser_agent_key,
                model_name=browser_model,
            )
            logger.info(
                "BrowserUseAdapter initialized (model=%s, max_steps=%d)",
                browser_model,
                browser_config.get("browser_use", {}).get("max_steps", 20),
            )
        except Exception as e:
            logger.warning("BrowserUseAdapter init failed — browser skills disabled: %s", e)
    else:
        logger.info("Browser-use disabled in config")

    # ── Browser Controller (deterministic browser control layer) ──
    browser_controller = None
    if browser_adapter:
        try:
            from infrastructure.browser_use_controller import BrowserUseController
            browser_controller = BrowserUseController(browser_adapter)
            logger.info("BrowserUseController initialized")
        except Exception as e:
            logger.warning("BrowserUseController init failed: %s", e)

    # ── Email Provider (pluggable — v1: SMTP) ──
    email_config = load_yaml("email.yaml")
    email_client = None
    if email_config.get("email", {}).get("enabled", False):
        try:
            from providers.email.smtp_provider import SMTPProvider
            from providers.email.client import EmailClient

            provider = SMTPProvider(email_config["email"])
            email_client = EmailClient(
                provider=provider,
                drafts_dir="state/email/drafts",
                from_address=email_config["email"].get("defaults", {}).get(
                    "from_address", ""
                ),
                signature=email_config["email"].get("defaults", {}).get(
                    "signature", ""
                ),
            )
            logger.info(
                "EmailClient initialized (provider=%s, configured=%s)",
                email_config["email"].get("provider", "smtp"),
                provider.is_configured(),
            )
        except Exception as e:
            logger.warning("EmailClient init failed — email skills disabled: %s", e)
    else:
        logger.info("Email disabled in config")

    # ── User Knowledge Store (moved up — zero dependencies, safe to create early) ──
    # Created before skill_deps so ALL skills (email, reasoning, etc.) can
    # receive user_knowledge via DI. Previously only memory skills had access.
    from memory.user_knowledge import UserKnowledgeStore
    user_knowledge = UserKnowledgeStore(persist_path="state/user_knowledge.json")
    logger.info("UserKnowledgeStore initialized (early — for skill DI)")

    # ── WhatsApp Provider (pluggable — v1: Neonize/Whatsmeow) ──
    wa_config = load_yaml("whatsapp.yaml")
    whatsapp_client = None
    if wa_config.get("whatsapp", {}).get("enabled", False):
        try:
            from providers.whatsapp.connection_manager import WhatsAppConnectionManager
            from providers.whatsapp.rate_limiter import WhatsAppRateLimiter
            from providers.whatsapp.neonize_provider import NeonizeProvider
            from providers.whatsapp.contact_resolver import ContactResolver
            from providers.whatsapp.client import WhatsAppClient

            wa_settings = wa_config["whatsapp"]
            rate_cfg = wa_settings.get("rate_limit", {})

            wa_conn_manager = WhatsAppConnectionManager(
                session_name=wa_settings.get("session_name", "merlin_whatsapp"),
                database_path=wa_settings.get(
                    "database_path", "state/whatsapp/neonize.db",
                ),
            )
            wa_conn_manager.start()

            wa_rate_limiter = WhatsAppRateLimiter(
                max_messages=rate_cfg.get("max_messages", 10),
                window_seconds=rate_cfg.get("window_seconds", 60),
            )

            wa_provider = NeonizeProvider(
                connection_manager=wa_conn_manager,
                rate_limiter=wa_rate_limiter,
            )

            wa_contact_resolver = ContactResolver(
                user_knowledge,
                connection_manager=wa_conn_manager,
            )

            whatsapp_client = WhatsAppClient(
                provider=wa_provider,
                contact_resolver=wa_contact_resolver,
                messages_dir=wa_settings.get(
                    "messages_dir", "state/whatsapp/messages",
                ),
            )
            logger.info(
                "WhatsAppClient initialized (session=%s)",
                wa_settings.get("session_name", "merlin_whatsapp"),
            )
        except Exception as e:
            logger.warning(
                "WhatsAppClient init failed — WhatsApp skills disabled: %s", e,
            )
    else:
        logger.info("WhatsApp disabled in config")

    # FileIndex — lazy-built file search index across anchors
    from world.file_index import FileIndex
    file_index = FileIndex()
    logger.info("FileIndex initialized (lazy — builds on first search)")

    skill_deps = {
        "location_config": location_config,
        "system_controller": system_controller,
        "content_llm": content_llm,
        "task_store": task_store,
        "session_manager": session_manager,
        "app_registry": app_registry,
        "browser_adapter": browser_adapter,
        "browser_controller": browser_controller,
        "email_client": email_client,
        "whatsapp_client": whatsapp_client,
        "user_knowledge": user_knowledge,
        "file_index": file_index,
    }
    load_skills(registry, skills_config, deps=skill_deps)

    # Action namespace audit is deferred to AFTER late-registration
    # so memory skills are included in the audit.

    # ── Phase 10: Build IntentIndex + IntentMatcher (after skills loaded) ──
    from cortex.intent_engine import IntentIndex, IntentMatcher
    intent_index = IntentIndex()
    intent_index.build(registry)
    intent_matcher = IntentMatcher(intent_index, registry)
    reflex._intent_matcher = intent_matcher
    logger.info("IntentMatcher built and injected into ReflexEngine")

    # ── Output channel (TTS always-on, independent of input mode) ──
    console_channel = ConsoleOutputChannel(prefix="MERLIN")
    voice_config = execution_config.get("voice", {})
    tts_config = voice_config.get("tts", {})

    # TTS output is independent of input mode — MERLIN always speaks,
    # regardless of whether input is keyboard, mic, or hybrid.
    if tts_config.get("enabled", True):
        tts_engine = VoiceEngineFactory.create_tts(voice_config)
        if tts_engine:
            # Eager init: start TTS worker NOW, not on first speak.
            # Fail-fast: COM/audio errors surface here, not after first user command.
            if hasattr(tts_engine, 'start'):
                tts_engine.start(timeout=5.0)

            speech_channel = SpeechOutputChannel(tts_engine)
            output_channel = CompositeOutputChannel([console_channel, speech_channel])
            logger.info("Output: CompositeOutputChannel (console + speech)")
        else:
            logger.warning(
                "TTS engine unavailable — falling back to console-only output."
            )
            output_channel = console_channel
    else:
        output_channel = console_channel
        logger.info("Output: ConsoleOutputChannel (TTS disabled in config)")

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
    event_sources = build_event_sources(
        execution_config,
        system_controller=system_controller,
        timeline=timeline,
        app_registry=app_registry,
        browser_controller=browser_controller,
    )

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
    from cortex.scored_discovery import DomainScoredDiscovery
    cortex = MissionCortex(
        llm_client=compiler_client,
        registry=registry,
        location_config=location_config,
        context_provider=context_provider,
        skill_discovery=DomainScoredDiscovery(),
        session_manager=session_manager,
    )

    # ── Attention Arbitration ──
    from runtime.attention import AttentionManager
    attention_manager = AttentionManager.from_config(
        execution_config,
        deliver_fn=output_channel.send,
    )
    logger.info(
        "AttentionManager: cooldown=%ss, max_queue=%d",
        attention_manager._config.cooldown_seconds,
        attention_manager._config.max_queue_size,
    )

    # ── Narration Policy (Phase 8) ──
    from reporting.narration import NarrationPolicy
    narration_config = execution_config.get("narration", {})
    narration_policy = None
    if narration_config.get("enabled", True):
        narration_policy = NarrationPolicy.from_config(narration_config)
        logger.info(
            "NarrationPolicy: single_node_silent=%s, compression=%d, heartbeat=%.1fs",
            narration_policy._single_node_silent,
            narration_policy._compression_threshold,
            narration_policy._heartbeat_threshold,
        )
    else:
        logger.info("Narration disabled in config")

    # ── Cognitive Coordinator (bounded reasoning pre-phase) ──
    from cortex.cognitive_coordinator import LLMCognitiveCoordinator
    coordinator = None
    try:
        coordinator_client = router.get_client("cognitive_coordinator")
        if coordinator_client.is_available():
            coordinator = LLMCognitiveCoordinator(llm=coordinator_client)
            logger.info("CognitiveCoordinator: LLMCognitiveCoordinator (Phase 1)")
        else:
            logger.warning("cognitive_coordinator LLM not available")
    except Exception as e:
        logger.warning("cognitive_coordinator init failed: %s", e)

    # ── Job Scheduler (persistent, tick-based) ──
    scheduler = None
    completion_queue = None

    if scheduler_config.get("enabled", False) and task_store is not None:
        try:
            from runtime.tick_scheduler import TickSchedulerManager
            from runtime.completion_event import CompletionQueue

            max_concurrent = scheduler_config.get("max_concurrent_jobs", 2)
            max_retries = scheduler_config.get("max_retry_attempts", 3)

            scheduler = TickSchedulerManager(
                store=task_store,
                max_concurrent_jobs=max_concurrent,
            )
            completion_queue = CompletionQueue()

            logger.info(
                "Scheduler: TickSchedulerManager (max_concurrent=%d, "
                "max_retries=%d)",
                max_concurrent, max_retries,
            )
        except Exception as e:
            logger.warning("Scheduler init failed — running without: %s", e)
            scheduler = None
            completion_queue = None
    else:
        logger.info("Scheduler disabled in config")

    # ── Execution Supervisor (Phase 3: Interactive Execution Layer) ──
    from execution.supervisor import ExecutionSupervisor, ExecutionContext
    execution_context = ExecutionContext(
        observer=environment_observer,
        session_manager=session_manager,
        capability_registry=capability_registry,
        timeline=timeline,
    )
    # Supervisor needs executor — but executor is constructed inside Merlin.
    # We construct a temporary executor here for the supervisor, then
    # Merlin will use its internal executor for everything else.
    # NOTE: The supervisor is constructed with a placeholder executor;
    # Merlin._init_ builds the real executor and passes the supervisor
    # to the orchestrator. The supervisor's executor will be set after
    # Merlin construction.
    logger.info("ExecutionSupervisor initialized (Phase 3)")

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
        attention_manager=attention_manager,
        narration_policy=narration_policy,
        coordinator=coordinator,
        scheduler=scheduler,
        completion_queue=completion_queue,
    )

    # ── Wire supervisor with the real executor (post-construction) ──
    supervisor = ExecutionSupervisor(
        executor=merlin.executor,
        context=execution_context,
    )
    merlin.orchestrator._supervisor = supervisor
    logger.info("ExecutionSupervisor wired to MissionOrchestrator")

    # ── User Knowledge Store — already created above (before skill_deps) ──
    # Reuse the same instance for PreferenceResolver, event loop, etc.

    # ── Preference Resolver (delegates to UserKnowledgeStore) ──
    from cortex.preference_resolver import PreferenceResolver, PreferenceMemory
    pref_memory = PreferenceMemory(
        memory_store=memory_store,
        user_knowledge=user_knowledge,
    )
    pref_resolver = PreferenceResolver(memory=pref_memory)
    merlin.orchestrator._pref_resolver = pref_resolver
    merlin._user_knowledge = user_knowledge
    merlin.orchestrator._user_knowledge = user_knowledge  # for per-mission SkillContext
    logger.info("PreferenceResolver + UserKnowledgeStore wired")

    # ── Entity Resolver post-wiring (needs SkillRegistry from the executor) ──
    entity_resolver._skill_registry = merlin.executor.registry
    entity_resolver._file_index = file_index
    entity_resolver._location_config = location_config
    merlin.orchestrator._entity_resolver = entity_resolver
    logger.info("EntityResolver wired to MissionOrchestrator (Phase 9C/9D/9E)")

    # ── Late-register memory skills (require UserKnowledgeStore) ──
    # Memory skills (memory.get_preference, memory.set_preference, etc.) require
    # a live UserKnowledgeStore instance. They were NOT registered in the first
    # load_skills() pass because user_knowledge was not in the deps dict.
    # load_skills skips any skill whose required dep is missing.
    memory_skill_entries = {
        "skills": [
            e for e in skills_config.get("skills", [])
            if e["module"].startswith("skills.memory")
        ]
    }
    if memory_skill_entries["skills"]:
        load_skills(registry, memory_skill_entries, deps={"user_knowledge": user_knowledge})
        logger.info("Memory skills registered with UserKnowledgeStore")
        # Rebuild IntentIndex so memory skills are available to reflex path too
        intent_index.build(registry)
        reflex._intent_matcher = IntentMatcher(intent_index, registry)
        logger.info("IntentIndex rebuilt with memory skills")

    # ── Action namespace governance audit (after ALL skills registered) ──
    violations = registry.audit_action_namespace()
    if violations:
        raise RuntimeError(
            f"Action namespace audit failed with {len(violations)} violation(s). "
            "Fix skill contracts before starting."
        )

    # ── Wire UserKnowledgeStore into event loop for proactive policy eval ──
    # Event loop is constructed without UKS (doesn't exist at that point).
    # Late-inject it now so _maybe_apply_user_policy can match policies
    # against world events (media_started, foreground_changed, etc.).
    merlin.event_loop._user_knowledge = user_knowledge
    logger.info("UserKnowledgeStore wired into RuntimeEventLoop for proactive policy eval")


    # ── Perception (voice-aware via PerceptionOrchestrator) ──
    text_perception = TextPerception()
    speech_perception = None

    if getattr(args, 'voice', False) or getattr(args, 'hybrid', False):
        stt_engine = VoiceEngineFactory.create_stt(voice_config)
        if stt_engine:
            from perception.audio_recorder import AudioRecorder
            from perception.speech import SpeechPerception

            audio_cfg = voice_config.get("audio", {})
            recorder = AudioRecorder(
                sample_rate=audio_cfg.get("sample_rate", 16000),
                silence_duration=audio_cfg.get("silence_duration", 1.5),
                max_record_seconds=audio_cfg.get("max_record_seconds", 30),
                vad_mode=audio_cfg.get("vad_mode", 2),
            )
            speech_perception = SpeechPerception(stt_engine, recorder)
            logger.info("SpeechPerception initialized.")
        else:
            logger.warning(
                "STT engine unavailable — voice input disabled, "
                "falling back to text-only."
            )

    # Build orchestrator based on mode
    if getattr(args, 'voice', False) and speech_perception:
        # Voice-only: no text channel
        orchestrator = PerceptionOrchestrator(text=None, speech=speech_perception)
        input_mode = "voice"
    elif getattr(args, 'hybrid', False) and speech_perception:
        # Hybrid: both channels
        orchestrator = PerceptionOrchestrator(text=text_perception, speech=speech_perception)
        input_mode = "hybrid"
    else:
        # Text-only (default, or fallback if voice deps missing)
        orchestrator = PerceptionOrchestrator(text=text_perception)
        input_mode = "text"

    # ── PromptSession for safe concurrent terminal output ──
    from prompt_toolkit import PromptSession
    prompt_session = PromptSession()
    orchestrator._session = prompt_session  # inject after construction

    # ── Start runtime ──
    merlin.start()

    # ── UI Mode: Start bridge, API server, widget ──
    ui_mode = getattr(args, 'ui', False)
    telegram_mode = getattr(args, 'telegram', False)
    bridge = None
    api_process = None
    widget_process = None
    telegram_process = None

    # Bridge is needed for both UI and Telegram (file-based IPC)
    needs_bridge = ui_mode or telegram_mode

    if needs_bridge:
        import subprocess
        import sys
        project_root = Path(__file__).resolve().parent

        # 1. Install log buffer (inside MERLIN process)
        from interface.log_buffer import install_log_buffer
        log_buffer = install_log_buffer(maxlen=500)

        # 2. Start bridge daemon thread
        from interface.bridge import MerlinBridge
        bridge = MerlinBridge(
            merlin=merlin,
            base_path=str(project_root),
            log_buffer=log_buffer,
        )
        bridge.start()

    if ui_mode:
        if not needs_bridge:  # should not happen, but guard
            import subprocess
            import sys
            project_root = Path(__file__).resolve().parent

        # 3. Start API server subprocess
        api_process = subprocess.Popen(
            [sys.executable, "-m", "interface.api_server"],
            cwd=str(project_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # 4. Start widget subprocess (if PySide6 available)
        try:
            widget_process = subprocess.Popen(
                [sys.executable, "-m", "ui.widget.widget"],
                cwd=str(project_root),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.warning("Widget launch failed (PySide6 may not be installed): %s", e)
            widget_process = None

    # ── Telegram Mode: Start Telegram bot subprocess ──
    if telegram_mode:
        import os as _os
        import subprocess
        import sys
        project_root = Path(__file__).resolve().parent

        # Validate token
        tg_token = _os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if not tg_token:
            logger.error(
                "TELEGRAM_BOT_TOKEN not set in .env — cannot start Telegram bot"
            )
        else:
            # Validate config
            tg_config = load_yaml("telegram.yaml").get("telegram", {})
            tg_enabled = tg_config.get("enabled", False)
            tg_users = tg_config.get("allowed_user_ids", [])

            if not tg_enabled:
                logger.error(
                    "Telegram is disabled in config/telegram.yaml — "
                    "set 'enabled: true' to activate"
                )
            elif not tg_users:
                logger.error(
                    "allowed_user_ids is empty in config/telegram.yaml — "
                    "add your Telegram user ID for security"
                )
            else:
                telegram_process = subprocess.Popen(
                    [sys.executable, "-m", "interface.telegram_bot"],
                    cwd=str(project_root),
                )
                logger.info(
                    "Telegram bot started (allowed_users=%s)", tg_users
                )

    # ── Interactive loop ──
    print("=" * 60)
    print("  MERLIN — Online")
    print(f"  Input mode: {input_mode}")
    if input_mode == "text":
        print("  Type a command, or 'exit' to quit.")
    elif input_mode == "voice":
        print("  Speak a command. Say 'exit' to quit.")
    else:
        print("  Type or speak a command. 'exit' to quit.")
    if compiler_client:
        print(f"  Compiler: {compiler_client.model} (temp={compiler_client.default_temperature})")
    else:
        print("  Compiler: not connected")
    if reporter_client:
        print(f"  Reporter: {reporter_client.model} (temp={reporter_client.default_temperature})")
    else:
        print("  Reporter: not connected")
    if ui_mode:
        print(f"  Dashboard: http://localhost:8420")
        print(f"  API: http://localhost:8420/api/v1/")
        print(f"  API Docs: http://localhost:8420/docs")
        # STT status
        stt_mode = voice_config.get("ui_stt_mode", "controlled") if voice_config else "controlled"
        stt_engine_name = voice_config.get("stt", {}).get("engine", "?") if voice_config else "?"
        stt_model = voice_config.get("stt", {}).get("model", "?") if voice_config else "?"
        if voice_config and voice_config.get("enabled", True):
            print(f"  STT: {stt_engine_name}/{stt_model} (mode={stt_mode})")
        else:
            print(f"  STT: disabled")
    if telegram_mode and telegram_process is not None:
        print(f"  Telegram: Bot active")
    print("=" * 60)
    print()

    # ── Boot self-test: speak "Online" to validate audio pipeline ──
    # If the user hears this, COM + SAPI5 + audio device + volume are all working.
    output_channel.send("Online.")

    from prompt_toolkit.patch_stdout import patch_stdout

    try:
        with patch_stdout(raw=True):
            while True:
                try:
                    percept = orchestrator.next_percept()
                except EOFError:
                    break

                if not percept.payload:
                    continue

                # Normalized exit detection (handles voice transcriptions like "Exit.")
                from perception.normalize import normalize_for_matching
                normalized = normalize_for_matching(percept.payload)
                if normalized in {"exit", "quit", "q"}:
                    break

                # Route → Execute → Report
                merlin.handle_percept(percept)
                print()  # visual spacing

    except KeyboardInterrupt:
        print("\n\nInterrupted.")

    finally:
        # Shutdown in reverse order: telegram → widget → API → bridge → MERLIN core
        if telegram_process is not None:
            try:
                telegram_process.terminate()
                telegram_process.wait(timeout=5)
            except Exception as e:
                logger.warning("Telegram bot shutdown error: %s", e)
        if widget_process is not None:
            try:
                widget_process.terminate()
                widget_process.wait(timeout=5)
            except Exception as e:
                logger.warning("Widget shutdown error: %s", e)
        if api_process is not None:
            try:
                api_process.terminate()
                api_process.wait(timeout=5)
            except Exception as e:
                logger.warning("API server shutdown error: %s", e)
        if bridge is not None:
            bridge.stop()
        merlin.stop()
        # Shutdown browser adapter (close Chrome, stop event loop)
        if browser_adapter is not None:
            try:
                browser_adapter.shutdown()
            except Exception as e:
                logger.warning("Browser adapter shutdown error: %s", e)
        print("MERLIN — Offline.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MERLIN — Personal AI Assistant")
    parser.add_argument(
        "--voice", action="store_true",
        help="Voice-only mode: mic input + TTS output",
    )
    parser.add_argument(
        "--hybrid", action="store_true",
        help="Hybrid mode: both text and voice input (first wins per cycle)",
    )
    parser.add_argument(
        "--ui", action="store_true",
        help="Launch dashboard UI, API server, and desktop widget",
    )
    parser.add_argument(
        "--telegram", action="store_true",
        help="Launch Telegram bot adapter (requires TELEGRAM_BOT_TOKEN in .env)",
    )
    args = parser.parse_args()
    main(args)
