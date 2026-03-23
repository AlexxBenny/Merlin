# interface/config_schema.py

"""
Config Schema — Pydantic models for validated config editing.

Defines the editable fields for each MERLIN config section.
The API validates incoming PATCH requests through these schemas
before writing to YAML. This prevents invalid values from
corrupting config files.

Uses ruamel.yaml for comment-preserving round-trip editing.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# Editable config sections (from config/execution.yaml)
# ─────────────────────────────────────────────────────────────

class EditableExecutorConfig(BaseModel):
    """Executor settings."""
    max_workers: Optional[int] = Field(None, ge=1, le=16)
    node_timeout_seconds: Optional[int] = Field(None, ge=5, le=300)


class EditableEventLoopConfig(BaseModel):
    """Event loop settings."""
    tick_interval: Optional[float] = Field(None, ge=0.05, le=1.0)


class EditableSchedulerConfig(BaseModel):
    """Scheduler settings."""
    enabled: Optional[bool] = None
    max_retry_attempts: Optional[int] = Field(None, ge=1, le=10)
    missed_job_staleness_seconds: Optional[int] = Field(None, ge=60, le=3600)
    max_concurrent_jobs: Optional[int] = Field(None, ge=1, le=10)


class EditableAttentionConfig(BaseModel):
    """Attention arbitration settings."""
    cooldown_seconds: Optional[float] = Field(None, ge=1, le=60)
    max_queue_size: Optional[int] = Field(None, ge=1, le=50)
    merge_duplicates: Optional[bool] = None


class EditableNarrationConfig(BaseModel):
    """Narration policy settings."""
    enabled: Optional[bool] = None
    single_node_silent: Optional[bool] = None
    compression_threshold: Optional[int] = Field(None, ge=1, le=10)
    heartbeat_threshold_seconds: Optional[float] = Field(None, ge=1.0, le=30.0)
    merge_proactive_insights: Optional[bool] = None


class EditableVoiceConfig(BaseModel):
    """Voice subsystem settings."""
    enabled: Optional[bool] = None
    ui_stt_mode: Optional[str] = Field(None, pattern="^(controlled|fast)$")
    tts_enabled: Optional[bool] = None
    tts_rate: Optional[int] = Field(None, ge=100, le=300)


class EditableBrowserConfig(BaseModel):
    """Browser settings."""
    headless: Optional[bool] = None


class EditableEmailConfig(BaseModel):
    """Email settings (mirrors email.yaml → email section)."""
    enabled: Optional[bool] = None
    provider: Optional[str] = None


class EditableEmailSmtpConfig(BaseModel):
    """Email SMTP settings."""
    host: Optional[str] = None
    port: Optional[int] = Field(None, ge=1, le=65535)
    use_tls: Optional[bool] = None


class EditableEmailImapConfig(BaseModel):
    """Email IMAP settings."""
    host: Optional[str] = None
    port: Optional[int] = Field(None, ge=1, le=65535)
    use_ssl: Optional[bool] = None


class EditableEmailDefaultsConfig(BaseModel):
    """Email defaults settings."""
    from_address: Optional[str] = None
    signature: Optional[str] = None
    max_inbox_fetch: Optional[int] = Field(None, ge=1, le=100)


class EditableWhatsAppConfig(BaseModel):
    """WhatsApp settings (mirrors whatsapp.yaml → whatsapp section)."""
    enabled: Optional[bool] = None
    session_name: Optional[str] = None


class EditableWhatsAppRateLimitConfig(BaseModel):
    """WhatsApp rate limit settings."""
    max_messages: Optional[int] = Field(None, ge=1, le=100)
    window_seconds: Optional[int] = Field(None, ge=10, le=3600)


class EditableTelegramConfig(BaseModel):
    """Telegram settings (mirrors telegram.yaml → telegram section)."""
    enabled: Optional[bool] = None
    allowed_user_ids: Optional[str] = Field(
        None,
        description="Comma-separated Telegram user IDs (e.g. 123456,789012)",
    )
    max_queue_depth: Optional[int] = Field(None, ge=1, le=20)
    response_timeout: Optional[int] = Field(None, ge=10, le=600)


# ─────────────────────────────────────────────────────────────
# Master config update model
# ─────────────────────────────────────────────────────────────

class ConfigUpdateRequest(BaseModel):
    """Top-level config update request.

    Each field corresponds to a section in config/execution.yaml.
    Only non-None fields are applied.
    """
    executor: Optional[EditableExecutorConfig] = None
    event_loop: Optional[EditableEventLoopConfig] = None
    scheduler: Optional[EditableSchedulerConfig] = None
    attention: Optional[EditableAttentionConfig] = None
    narration: Optional[EditableNarrationConfig] = None
    voice: Optional[EditableVoiceConfig] = None
    browser: Optional[EditableBrowserConfig] = None
    email: Optional[EditableEmailConfig] = None
    email_smtp: Optional[EditableEmailSmtpConfig] = None
    email_imap: Optional[EditableEmailImapConfig] = None
    email_defaults: Optional[EditableEmailDefaultsConfig] = None
    whatsapp: Optional[EditableWhatsAppConfig] = None
    whatsapp_rate_limit: Optional[EditableWhatsAppRateLimitConfig] = None
    telegram: Optional[EditableTelegramConfig] = None


# ─────────────────────────────────────────────────────────────
# Config field metadata (for dashboard display)
# ─────────────────────────────────────────────────────────────

CONFIG_FIELD_METADATA: Dict[str, Dict[str, Any]] = {
    "executor.max_workers": {
        "label": "Max Workers",
        "description": "Maximum parallel executor threads",
        "type": "int", "min": 1, "max": 16,
    },
    "executor.node_timeout_seconds": {
        "label": "Node Timeout (seconds)",
        "description": "Max seconds per node before timeout",
        "type": "int", "min": 5, "max": 300,
    },
    "event_loop.tick_interval": {
        "label": "Tick Interval (seconds)",
        "description": "RuntimeEventLoop tick frequency",
        "type": "float", "min": 0.05, "max": 1.0,
    },
    "scheduler.enabled": {
        "label": "Scheduler Enabled",
        "description": "Enable/disable the job scheduler",
        "type": "bool",
    },
    "scheduler.max_retry_attempts": {
        "label": "Max Retry Attempts",
        "description": "Maximum retries for failed jobs",
        "type": "int", "min": 1, "max": 10,
    },
    "scheduler.max_concurrent_jobs": {
        "label": "Max Concurrent Jobs",
        "description": "Maximum jobs running simultaneously",
        "type": "int", "min": 1, "max": 10,
    },
    "attention.cooldown_seconds": {
        "label": "Notification Cooldown (seconds)",
        "description": "Minimum gap between delivered notifications",
        "type": "float", "min": 1, "max": 60,
    },
    "attention.max_queue_size": {
        "label": "Max Notification Queue",
        "description": "Maximum deferred notifications before oldest dropped",
        "type": "int", "min": 1, "max": 50,
    },
    "narration.enabled": {
        "label": "Narration Enabled",
        "description": "Enable/disable mission narration",
        "type": "bool",
    },
    "narration.single_node_silent": {
        "label": "Single Node Silent",
        "description": "Suppress narration for 1-node missions",
        "type": "bool",
    },
    "narration.compression_threshold": {
        "label": "Compression Threshold",
        "description": "Max foreground nodes for compressed intent narration",
        "type": "int", "min": 1, "max": 10,
    },
    "voice.enabled": {
        "label": "Voice Enabled",
        "description": "Master switch for voice subsystem",
        "type": "bool",
    },
    "voice.tts_enabled": {
        "label": "TTS Enabled",
        "description": "Enable/disable text-to-speech output",
        "type": "bool",
    },
    "voice.tts_rate": {
        "label": "TTS Rate (WPM)",
        "description": "Text-to-speech words per minute",
        "type": "int", "min": 100, "max": 300,
    },
    "voice.ui_stt_mode": {
        "label": "UI STT Mode",
        "description": "controlled (server Whisper) or fast (browser Web Speech API)",
        "type": "str",
    },
    "browser.headless": {
        "label": "Headless Browser",
        "description": "Run browser in headless mode",
        "type": "bool",
    },
    # Email config fields
    "email.enabled": {
        "label": "Email Enabled",
        "description": "Master switch for email integration",
        "type": "bool",
    },
    "email.provider": {
        "label": "Email Provider",
        "description": "Provider type: smtp, gmail_api, microsoft_graph",
        "type": "str",
    },
    "email.smtp.host": {
        "label": "SMTP Host",
        "description": "SMTP server hostname (e.g., smtp.gmail.com)",
        "type": "str",
    },
    "email.smtp.port": {
        "label": "SMTP Port",
        "description": "SMTP server port (587 for TLS, 465 for SSL)",
        "type": "int", "min": 1, "max": 65535,
    },
    "email.smtp.use_tls": {
        "label": "SMTP Use TLS",
        "description": "Enable STARTTLS for SMTP connection",
        "type": "bool",
    },
    "email.imap.host": {
        "label": "IMAP Host",
        "description": "IMAP server hostname (e.g., imap.gmail.com)",
        "type": "str",
    },
    "email.imap.port": {
        "label": "IMAP Port",
        "description": "IMAP server port (993 for SSL)",
        "type": "int", "min": 1, "max": 65535,
    },
    "email.imap.use_ssl": {
        "label": "IMAP Use SSL",
        "description": "Enable SSL for IMAP connection",
        "type": "bool",
    },
    "email.defaults.from_address": {
        "label": "From Address",
        "description": "Default sender email address",
        "type": "str",
    },
    "email.defaults.signature": {
        "label": "Email Signature",
        "description": "Signature appended to all outgoing emails",
        "type": "str",
    },
    "email.defaults.max_inbox_fetch": {
        "label": "Max Inbox Fetch",
        "description": "Maximum emails to fetch per inbox read",
        "type": "int", "min": 1, "max": 100,
    },
    # WhatsApp config fields
    "whatsapp.enabled": {
        "label": "WhatsApp Enabled",
        "description": "Master switch for WhatsApp integration",
        "type": "bool",
    },
    "whatsapp.session_name": {
        "label": "Session Name",
        "description": "WhatsApp session identifier",
        "type": "str",
    },
    "whatsapp.rate_limit.max_messages": {
        "label": "Rate Limit — Max Messages",
        "description": "Maximum messages allowed per window",
        "type": "int", "min": 1, "max": 100,
    },
    "whatsapp.rate_limit.window_seconds": {
        "label": "Rate Limit — Window (seconds)",
        "description": "Time window for rate limiting",
        "type": "int", "min": 10, "max": 3600,
    },
    # Telegram config fields
    "telegram.enabled": {
        "label": "Telegram Enabled",
        "description": "Master switch for Telegram bot integration",
        "type": "bool",
    },
    "telegram.allowed_user_ids": {
        "label": "Allowed User IDs",
        "description": "Comma-separated Telegram user IDs (e.g. 123456,789012)",
        "type": "str",
    },
    "telegram.max_queue_depth": {
        "label": "Max Queue Depth",
        "description": "Max pending commands before rejecting new messages",
        "type": "int", "min": 1, "max": 20,
    },
    "telegram.response_timeout": {
        "label": "Response Timeout (seconds)",
        "description": "Max wait time for a bridge response",
        "type": "int", "min": 10, "max": 600,
    },
}


# ─────────────────────────────────────────────────────────────
# YAML round-trip editor
# ─────────────────────────────────────────────────────────────

def apply_config_update(
    yaml_path: str,
    update: ConfigUpdateRequest,
) -> Dict[str, Any]:
    """Apply validated config updates to a YAML file.

    Uses ruamel.yaml for comment-preserving round-trip editing.

    Args:
        yaml_path: Path to the YAML config file.
        update: Validated update request.

    Returns:
        Dict of actually changed fields: {"section.key": new_value}
    """
    try:
        from ruamel.yaml import YAML
    except ImportError:
        # Fallback to standard yaml if ruamel not available
        import yaml
        return _apply_with_standard_yaml(yaml_path, update)

    yml = YAML()
    yml.preserve_quotes = True

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yml.load(f)

    if data is None:
        data = {}

    changes = {}

    for section_name, section_model in [
        ("executor", update.executor),
        ("event_loop", update.event_loop),
        ("scheduler", update.scheduler),
        ("attention", update.attention),
        ("narration", update.narration),
        ("voice", update.voice),
        ("browser", update.browser),
    ]:
        if section_model is None:
            continue

        if section_name not in data:
            data[section_name] = {}

        section_data = data[section_name]
        update_dict = section_model.model_dump(exclude_none=True)

        for key, value in update_dict.items():
            # Handle nested voice config
            if section_name == "voice" and key == "tts_enabled":
                if "tts" not in section_data:
                    section_data["tts"] = {}
                section_data["tts"]["enabled"] = value
                changes[f"voice.tts.enabled"] = value
            elif section_name == "voice" and key == "tts_rate":
                if "tts" not in section_data:
                    section_data["tts"] = {}
                section_data["tts"]["rate"] = value
                changes[f"voice.tts.rate"] = value
            else:
                section_data[key] = value
                changes[f"{section_name}.{key}"] = value

    if changes:
        with open(yaml_path, "w", encoding="utf-8") as f:
            yml.dump(data, f)

    return changes


def _apply_with_standard_yaml(
    yaml_path: str,
    update: ConfigUpdateRequest,
) -> Dict[str, Any]:
    """Fallback: apply updates using standard yaml (loses comments)."""
    import yaml

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    changes = {}

    for section_name, section_model in [
        ("executor", update.executor),
        ("event_loop", update.event_loop),
        ("scheduler", update.scheduler),
        ("attention", update.attention),
        ("narration", update.narration),
        ("voice", update.voice),
        ("browser", update.browser),
    ]:
        if section_model is None:
            continue

        if section_name not in data:
            data[section_name] = {}

        section_data = data[section_name]
        update_dict = section_model.model_dump(exclude_none=True)

        for key, value in update_dict.items():
            if section_name == "voice" and key == "tts_enabled":
                if "tts" not in section_data:
                    section_data["tts"] = {}
                section_data["tts"]["enabled"] = value
                changes[f"voice.tts.enabled"] = value
            elif section_name == "voice" and key == "tts_rate":
                if "tts" not in section_data:
                    section_data["tts"] = {}
                section_data["tts"]["rate"] = value
                changes[f"voice.tts.rate"] = value
            else:
                section_data[key] = value
                changes[f"{section_name}.{key}"] = value

    if changes:
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return changes
