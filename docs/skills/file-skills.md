# File System Skills

**Location**: `skills/fs/`

3 skills for file and folder operations.

## Skills

### `fs.read_file` (`read_file.py`)

Read contents of a file.

| Input | Type | Description |
|-------|------|-----------|
| `file_path` | `file_path` | Path to the file |

| Output | Type | Description |
|--------|------|-----------|
| `content` | `text_content` | File contents |
| `size` | `file_size` | File size in bytes |

### `fs.write_file` (`write_file.py`)

Write content to a file. Creates the file if it doesn't exist.

| Input | Type | Description |
|-------|------|-----------|
| `file_path` | `file_path` | Destination path |
| `content` | `text_content` | Content to write |

### `fs.create_folder` (`create_folder.py`)

Create a directory. Creates parent directories if needed.

| Input | Type | Description |
|-------|------|-----------|
| `folder_path` | `folder_path` | Path to create |

## Path Resolution

File paths are resolved via `LocationConfig` and `config/paths.yaml`. Users can use aliases:
- `downloads` → user's Downloads folder
- `desktop` → user's Desktop
- `documents` → user's Documents folder
