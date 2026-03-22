# File System Skills

**Location**: `skills/fs/`

5 skills for file and folder operations.

## Skills

### `fs.read_file` (`read_file.py`)

Read contents of a file.

| Input | Type | Description |
|-------|------|-------------|
| `file_path` | `file_path_input` | Path to the file |
| `anchor` | `str` | Location anchor (default: `WORKSPACE`) |

| Output | Type | Description |
|--------|------|-------------|
| `content` | `text_content` | File contents |
| `size` | `file_size` | File size in bytes |

**Contract**: `requires=["file_exists"]`, `effect_type="maintain"`

### `fs.write_file` (`write_file.py`)

Write content to a file. Creates the file if it doesn't exist.

| Input | Type | Description |
|-------|------|-------------|
| `file_path` | `file_path` | Destination path |
| `content` | `text_content` | Content to write |

**Contract**: `produces=["file_exists"]`, `effect_type="create"`

### `fs.create_folder` (`create_folder.py`)

Create a directory. Creates parent directories if needed.

| Input | Type | Description |
|-------|------|-------------|
| `folder_path` | `folder_path` | Path to create |

**Contract**: `produces=["file_exists"]`, `effect_type="create"`

### `fs.search_file` (`search_file.py`)

Search for files by name using `FileIndex`.

| Input | Type | Description |
|-------|------|-------------|
| `query` | `str` | Search query (filename or pattern) |

| Output | Type | Description |
|--------|------|-------------|
| `matches` | `List[FileRef]` | Found file references |

**Contract**: `produces=["file_reference"]`, `effect_type="reveal"`

### `fs.list_directory` (`list_directory.py`)

List contents of a directory.

| Input | Type | Description |
|-------|------|-------------|
| `path` | `folder_path` | Directory to list |

| Output | Type | Description |
|--------|------|-------------|
| `entries` | `List[Dict]` | Directory entries |

**Contract**: `produces=["file_reference"]`, `effect_type="reveal"`

## Path Resolution

File paths are resolved via `LocationConfig` and `config/paths.yaml`. Users can use aliases:
- `downloads` → user's Downloads folder
- `desktop` → user's Desktop
- `documents` → user's Documents folder

## File Entity Resolution (Phase 9E)

At compile-time, `EntityResolver._resolve_file_entities()` resolves bare filenames:

```
"resume" → "documents/resume.pdf" + anchor="DESKTOP"
```

- Uses `FileIndex.search()` for lookup
- 0 matches → `not_found_file` violation
- 1 match → replaces path + sets anchor
- N matches → `ambiguous_file` violation with structured options
- Skips explicit paths (containing `/` or `\`) and `IRReference` values
