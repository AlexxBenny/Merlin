# Configuration Files

MERLIN resolves configuration in the following order:

1. **Environment override** — `MERLIN_CONFIG_DIR` environment variable
2. **User config directory** — `~/.config/merlin/config/` or `%APPDATA%\Local\merlin\config\`
3. **Current working directory** — `./config/` (development mode only, requires repo fingerprint)
4. **Package defaults** — bundled templates inside `merlin_assistant/default_config/`

For pip-installed users, `merlin init` populates the user config directory automatically.

For developers working from a cloned repo, MERLIN detects the repo root and uses `./config/` directly.

## `models.yaml`

LLM model assignments by cognitive role.

```yaml
roles:
  cognitive_coordinator:
    provider: openrouter
    model: google/gemini-2.5-flash
    temperature: 0.3
  mission_compiler:
    provider: openrouter
    model: google/gemini-2.5-flash
    temperature: 0.2
  report_builder:
    provider: openrouter
    model: google/gemini-2.5-flash
    temperature: 0.4
  recovery_reasoner:
    provider: openrouter
    model: google/gemini-2.5-flash
    temperature: 0.2
```

## `skills.yaml`

Skill registry metadata and configuration.

## `routing.yaml`

Cognitive routing rules for BrainCore and EscalationPolicy:
- Pattern-based routing rules
- Tier classification thresholds
- Domain routing priorities

## `execution.yaml`

Execution engine configuration:
- `max_workers` — parallel execution threads
- `node_timeout` — per-node timeout
- `max_retries` — retry limits
- Failure policy defaults
- Supervisor settings

## `paths.yaml`

OS path aliases:
```yaml
downloads: "C:/Users/{user}/Downloads"
desktop: "C:/Users/{user}/Desktop"
documents: "C:/Users/{user}/Documents"
```

## `browser.yaml`

Browser configuration:
- Headless mode
- Viewport dimensions
- Navigation timeout
- CDP connection settings
- Security settings

## `app_aliases.yaml`

Application name aliases:
```yaml
chrome: google chrome
firefox: mozilla firefox
vscode: visual studio code
notepad: notepad
```

## `app_capabilities.yaml`

Per-app media capabilities:
```yaml
spotify:
  can_play: true
  can_pause: true
  can_next: true
  can_previous: true
vlc:
  can_play: true
  can_pause: true
```

## `email.yaml`

Email provider configuration:
- Provider type (`smtp`, `gmail_api`, `microsoft_graph`)
- SMTP/IMAP host and port settings
- Default sender address and signature
- Credentials via `.env` (never in YAML)

## `whatsapp.yaml`

WhatsApp integration settings:
- Provider configuration (neonize bridge)
- Connection parameters
- Rate limiting settings

## `telegram.yaml`

Telegram bot adapter configuration:
- `enabled` — toggle adapter on/off
- `allowed_user_ids` — whitelist of authorized Telegram user IDs
- `max_queue_depth` — queue pressure guard threshold
- `response_timeout` — response wait timeout in seconds
- Bot token stored in `.env` (`TELEGRAM_BOT_TOKEN`), never in this file
