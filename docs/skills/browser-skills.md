# Browser Skills

**Location**: `skills/browser/`

12 skills for web browser interaction. Follow the three-tier execution model.

## Skills

### `browser.click` (`browser_click.py`)

Click a browser entity by index or text reference.

| Input | Type | Description |
|-------|------|-----------|
| `entity_index` | `entity_index` | Display index of the entity |
| `entity_ref` | `entity_ref` | Text reference (resolved by EntityResolver) |

**Index drift protection**: Uses fresh snapshot (`cached=False`), verifies entity text against `_resolved_entity_text`, falls back to text search on drift.

### `browser.fill` (`browser_fill.py`)

Type text into an input field.

| Input | Type | Description |
|-------|------|-----------|
| `entity_index` | `entity_index` | Input field entity index |
| `text` | `text_content` | Text to type |

### `browser.scroll` (`browser_scroll.py`)

Scroll the page up or down.

| Input | Type | Description |
|-------|------|-----------|
| `direction` | `scroll_direction` | "up" or "down" |

**Normalization**: "above"/"top"/"upward" → "up". Everything else → "down".

### `browser.navigate` (`browser_navigate.py`)

Navigate to a URL.

| Input | Type | Description |
|-------|------|-----------|
| `url` | `url_string` | Destination URL |

### `browser.go_back` (`browser_go_back.py`)

Navigate back in browser history. No inputs.

### `browser.go_forward` (`browser_go_forward.py`)

Navigate forward in browser history. No inputs.

### `browser.find_element` (`browser_find_element.py`)

Find a specific element on the page by text or description.

### `browser.keypress` (`browser_keypress.py`)

Send keyboard events (key presses, shortcuts) to the browser.

### `browser.search` (`browser_search.py`)

Perform a web search query.

### `browser.select_result` (`browser_select_result.py`)

Select a specific result from search results or a list.

### `browser.wait_for` (`browser_wait_for.py`)

Wait for a page element or condition before proceeding.

### `browser.autonomous_task` (`autonomous_task.py`)

Tier 3 — LLM-driven autonomous page interaction for complex tasks.

| Input | Type | Description |
|-------|------|-----------|
| `task` | `task_description` | Natural language task description |

Uses the `browser-use` agent. Last resort after deterministic skills and entity resolution fail.

## Common Patterns

All browser skills:
- Use `BrowserController` for DOM interaction
- Emit `browser_action_completed` events to WorldTimeline
- Return `SkillResult` with `url` and `page_title` outputs
- Are `foreground` execution mode only
- Have `FailurePolicy.FAIL`
