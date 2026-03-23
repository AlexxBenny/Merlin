# UI Layer (`ui/`)

> React dashboard and PySide6 desktop widget for the MERLIN frontend.

---

## Overview

The UI layer provides two visual frontends for MERLIN:

1. **React Dashboard** — Full-featured web application at `http://localhost:8420`
2. **Desktop Widget** — Floating PySide6 orb with quick-chat capability

Both communicate exclusively through the API server (`interface/api_server.py`). Neither imports MERLIN core modules.

---

## React Dashboard (`ui/dashboard/`)

### Tech Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| React | 19 | Component framework |
| Vite | 8 | Build tool + dev server |
| TypeScript | 5 | Type safety |
| Tailwind CSS | 4 | Utility-first styling |
| Lucide React | — | Icon library |
| React Router | 7 | Client-side routing |

### Design System

Dark theme with cyan accent (`#00c8ff`). Custom CSS tokens defined in `index.css`:

| Token | Value | Usage |
|-------|-------|-------|
| `--color-bg-primary` | `#0a0a0f` | Page background |
| `--color-bg-card` | `#16161f` | Card backgrounds |
| `--color-accent` | `#00c8ff` | Primary accent (links, active states) |
| `--color-accent-glow` | `rgba(0,200,255,0.15)` | Hover/active glow effect |

Glass-morphism cards (`.glass-card`), animated SVG gauges, and `fadeInUp` page transitions.

### Pages

| Route | Page | Features |
|-------|------|----------|
| `/` | **Overview** | CPU/RAM/Disk SVG gauges, uptime counter, mission state badge, battery |
| `/chat` | **Chat** | SSE streaming with cursor animation, typing indicator, session management |
| `/mail` | **Mail** | Email drafts, inbox, compose with approve/send workflow |
| `/whatsapp` | **WhatsApp** | WhatsApp message history and composition |
| `/scheduler` | **Scheduler** | Jobs table with status badges, pause/resume/cancel actions |
| `/memory` | **Memory** | 5 knowledge domain cards (preferences, facts, traits, policies, relationships) |
| `/logs` | **Logs** | WebSocket real-time stream, level filter chips, text search, auto-scroll |
| `/config` | **Config** | Dynamic form fields with inline save, secret masking, field descriptions |
| `/missions` | **Missions** | Mission list + DAG node inspector with skill/inputs/dependencies/status |
| `/world` | **World State** | Recursive tree view with type-colored values (bool, number, null, string) |

### API Client (`src/lib/api.ts`)

Typed client with:
- `api.*` — Fetch wrappers for all REST endpoints
- `createWebSocket()` — WebSocket connection factory
- `streamChat()` — SSE reader for progressive chat responses

### Development

```bash
cd ui/dashboard
npm run dev     # Dev server at http://localhost:5173 (API proxied to :8420)
npm run build   # Production build to dist/
```

The Vite dev server proxies `/api` and `/ws` requests to `http://localhost:8420`.

---

## Desktop Widget (`ui/widget/`)

### Tech Stack

| Technology | Purpose |
|------------|---------|
| PySide6 | Qt6 for Python — frameless, always-on-top widget |
| `requests` | HTTP client for API calls |

### Widget Behavior

| State | Visual |
|-------|--------|
| **Idle** | 60px dark circle with cyan border, "M" text |
| **Processing** | Cyan glow pulse animation |
| **Disconnected** | Grey appearance (no cyan) |
| **Expanded** | 320×440 chat panel |

**Interaction:**
- Click orb → expand to chat panel
- Escape / close button → collapse back to orb
- Send message → background thread API call → response bubble

**Health heartbeat:** Polls `/api/v1/health` every 5 seconds. If unreachable, orb turns grey and status indicator updates.

### Chat

Messages are sent via `POST /api/v1/chat` on a background `QThread` to avoid blocking the UI event loop. Responses appear as styled chat bubbles (user = right-aligned, assistant = left-aligned).

---

## Startup Flow

When `python main.py --ui` is used:

```
1. MERLIN core starts (merlin.start())
2. LogBufferHandler installed on root logger
3. MerlinBridge daemon thread starts (export + command loops)
4. API server subprocess launched (python -m interface.api_server)
5. Widget subprocess launched (python -m ui.widget.widget)
6. Telegram subprocess launched (if --telegram flag, python -m interface.telegram_bot)
7. Interactive terminal loop runs normally
```

### Shutdown Order (reverse)

```
1. Telegram subprocess terminated (if running)
2. Widget subprocess terminated
3. API server subprocess terminated
4. Bridge daemon thread stopped
5. MERLIN core stopped (merlin.stop())
6. Browser adapter shutdown
```
