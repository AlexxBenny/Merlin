# System Skills

**Location**: `skills/system/`

19 skills for OS interaction, media control, and job management.

## Media Control

| Skill | File | Description | Key Inputs |
|-------|------|-----------|-----------|
| `system.media_play` | `media_play.py` | Resume/start playback | `app_target` (optional) |
| `system.media_pause` | `media_pause.py` | Pause playback | `app_target` (optional) |
| `system.media_next` | `media_next.py` | Next track | — |
| `system.media_previous` | `media_previous.py` | Previous track | — |

## Volume & Audio

| Skill | File | Description | Key Inputs |
|-------|------|-----------|-----------|
| `system.set_volume` | `set_volume.py` | Set volume level | `level` (0-100) |
| `system.mute` | `mute.py` | Mute audio | — |
| `system.unmute` | `unmute.py` | Unmute audio | — |

## Display

| Skill | File | Description | Key Inputs |
|-------|------|-----------|-----------|
| `system.set_brightness` | `set_brightness.py` | Set screen brightness | `level` (0-100) |
| `system.toggle_nightlight` | `toggle_nightlight.py` | Toggle night light | `state` (on/off) |

## Application Management

| Skill | File | Description | Key Inputs |
|-------|------|-----------|-----------|
| `system.open_app` | `open_app.py` | Launch application | `app_target` |
| `system.close_app` | `close_app.py` | Close application | `app_target` |
| `system.focus_app` | `focus_app.py` | Bring app to foreground | `app_target` |
| `system.list_apps` | `list_apps.py` | List running applications | — |

## Information & Status

| Skill | File | Description | Key Inputs |
|-------|------|-----------|-----------|
| `system.get_time` | `get_time.py` | Get current time | — |
| `system.get_battery` | `get_battery.py` | Get battery status | — |
| `system.get_now_playing` | `get_now_playing.py` | Get current media info | — |
| `system.get_system_status` | `get_system_status.py` | CPU, memory, battery overview | — |

## Job Management

| Skill | File | Description | Key Inputs |
|-------|------|-----------|-----------|
| `system.list_jobs` | `list_jobs.py` | List scheduled jobs | `status` (optional filter) |
| `system.cancel_job` | `cancel_job.py` | Cancel a scheduled job | `job_id` |
