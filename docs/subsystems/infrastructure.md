# Infrastructure

**Location**: `infrastructure/`

Physiology — not intelligence. OS interaction, browser management, app discovery.

## Browser Stack

### BrowserController (`infrastructure/browser_controller.py`)

Abstract interface defining the browser contract:

```python
DOMEntity:     index, backend_node_id, entity_type, text, url, ax_role
PageSnapshot:  snapshot_id, url, title, entities, entity_count, tabs, scroll_pct
TabInfo:       tab_id, url, title
BrowserResult: success, error, snapshot, navigated, new_tab_opened
```

**Key distinction**: `backend_node_id` is the TRUE identity (stable within a snapshot). `index` is ephemeral display order for LLM readability.

### BrowserUseController (`infrastructure/browser_use_controller.py`)

Concrete implementation using the `browser-use` library (Playwright + CDP).

**Operations**:
| Method | Purpose |
|--------|---------|
| `navigate(url)` | Navigate to URL |
| `click(backend_node_id)` | Click element by CDP node ID |
| `fill(backend_node_id, text)` | Type text into input |
| `scroll_page(direction)` | Scroll up/down |
| `go_back()` / `go_forward()` | Browser history navigation |
| `get_snapshot(cached)` | Get current page state |
| `find_entities(text)` | Search entities by text |

### BrowserUseAdapter (`infrastructure/browser_use_adapter.py`)

Manages browser lifecycle via Playwright/CDP:
- Browser creation and connection
- CDP session management
- Health checks using `is_cdp_connected` + `connection.is_open`
- Session reuse across tasks
- Popup watchdog attachment

### BrowserSafety (`infrastructure/browser_safety.py`)

URL allowlisting and download blocking for safe browsing.

### Session (`infrastructure/session.py`)

Browser session management:
- Session creation with unique IDs
- Popup watchdog (auto-dismiss JS dialogs)
- Session lifecycle tracking

## System Controller

### SystemController (`infrastructure/system_controller.py`)

Windows OS interaction:

| Category | Operations |
|----------|-----------|
| **Audio** | Get/set volume, mute/unmute |
| **Display** | Get/set brightness, toggle night light |
| **Battery** | Get battery status and percentage |
| **Media** | Play, pause, next, previous (via Windows API) |
| **Apps** | Launch, close, focus, list running apps |
| **System** | CPU/memory usage, idle detection |

## Application Management

### ApplicationRegistry (`infrastructure/application_registry.py`)

Application name → launch path resolution.

**Resolution pipeline**:
1. Direct registry lookup by `app_id`
2. Name index lookup (display names)
3. Alias table (`config/app_aliases.yaml`)
4. Fuzzy substring match

### AppDiscovery (`infrastructure/app_discovery.py`)

Windows application enumeration:
- Start Menu shortcuts (`.lnk` files)
- PATH executables
- UWP/Microsoft Store apps
- Registry-installed programs

### AppCapabilities (`infrastructure/app_capabilities.py`)

Per-app media capability detection from `config/app_capabilities.yaml`.

## Other

### LocationConfig (`infrastructure/location_config.py`)

User-configurable paths from `config/paths.yaml` for downloads, documents, etc.

### Observer (`infrastructure/observer.py`)

File system observer for download detection and monitoring.

### VoiceFactory (`infrastructure/voice_factory.py`)

TTS voice selection and configuration.
