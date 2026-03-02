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
                    "refresh_interval": sys_cfg.get("refresh_interval", 30.0),
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

    # ── Structural Analyzer (deterministic, no LLM) ──
    from brain.structural_classifier import StructuralAnalyzer
    structural_analyzer = StructuralAnalyzer()
    logger.info("StructuralAnalyzer enabled (deterministic feature analysis)")

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

    skill_deps = {
        "location_config": location_config,
        "system_controller": system_controller,
    }
    load_skills(registry, skills_config, deps=skill_deps)

    # ── Action namespace governance audit (fail-fast on violations) ──
    violations = registry.audit_action_namespace()
    if violations:
        raise RuntimeError(
            f"Action namespace audit failed with {len(violations)} violation(s). "
            "Fix skill contracts before starting."
        )

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
    )


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
        merlin.stop()
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
    args = parser.parse_args()
    main(args)
