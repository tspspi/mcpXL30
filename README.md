# mcpXL30

`mcpXL30` is an MCP (Model Context Protocol) server that exposes a curated
subset of the [pyxl30](https://github.com/tspspi/pyxl30) control library to LLM
agents.

## Features

- Async MCP tools for common Philips XL30 operations (identify, read/set high
  tension, change scan modes, trigger pump/vent cycles, capture TIFF images).
- Live MCP resources that describe the connected instrument and active config.
- Safety envelope that limits accelerating voltage and sensitive operations.
- Optional remote HTTP/UDS transport with the same authentication workflow as
  `mcpMQTT`.

## Installation

```bash
pip install .
# or
pip install ".[remote]"  # adds FastAPI/uvicorn/argon2 for remote mode
```

## Configuration

The server loads JSON configuration from `~/.config/mcpxl30/config.json` by
default (override via `--config`). A minimal example:

```json
{
  "instrument": {
    "port": "/dev/ttyUSB0",
    "log_level": "INFO",
    "retry_count": 3,
    "reconnect_count": 3
  },
  "image_capture": {
    "remote_directory": "C:\\\\TEMP",
    "filename_prefix": "MCPIMG_"
  },
  "safety": {
    "max_high_tension_kv": 15.0,
    "allow_venting": false,
    "allow_pumping": true,
    "allow_scan_mode_changes": true,
    "allow_stage_motion": false,
    "allow_detector_switching": true,
    "allow_beam_shift": true,
    "allow_scan_rotation_changes": true,
    "allow_image_filter_changes": true,
    "allow_specimen_current_mode_changes": false,
    "allow_beam_blank_control": true,
    "allow_oplock_control": false
  },
  "logging": {
    "level": "INFO",
    "logfile": null
  },
  "remote_server": {
    "uds": "/var/run/mcpxl30.sock",
    "api_key_kdf": {
      "algorithm": "argon2id",
      "salt": "<base64>",
      "time_cost": 3,
      "memory_cost": 65536,
      "parallelism": 1,
      "hash_len": 32,
      "hash": "<base64>"
    }
  }
}
```

Use `mcpxl30-genkey --config /path/to/config.json` (or `mcpxl30 --genkey`) to
generate a new API key and populate the Argon2 hash inside the `remote_server`
block. The plain token prints once to stdout.

The `safety` block gates riskier capabilities. Keep `allow_stage_motion` and
`allow_oplock_control` disabled unless you trust the calling agent. Imaging and
detector-related fields default to safe-but-capable settings, but you can toggle
them per deployment.

## Running the server

### stdio transport (default)

```bash
mcpxl30 --config ~/.config/mcpxl30/config.json
```

The process reads/writes MCP messages on stdio for integration into compatible
LLM orchestrators.

### Remote FastAPI/uvicorn transport

```bash
pip install "mcpXL30[remote]"
mcpxl30 --transport remotehttp --config ~/.config/mcpxl30/config.json
```

- The FastAPI app exposes `/mcp` (MCP streaming API) and `/status` (unauthenticated health check).
- Authentication expects the API key in `Authorization: Bearer`, `X-API-Key`,
  or the `?api_key=` query parameter.
- Binding uses a Unix domain socket (`remote_server.uds`) unless you specify a
  TCP `port`, in which case `host` (default `0.0.0.0`) applies.

## MCP Surface

### Tools

**Instrument Basics**

| Tool | Purpose |
| --- | --- |
| `instrument_identify` | Return instrument type/serial, scan mode, high tension. |
| `read_high_tension` / `set_high_tension` | Inspect or change accelerating voltage (safety-capped). |
| `get_scan_mode` / `set_scan_mode` | Read or change scan mode (`allow_scan_mode_changes`). |
| `capture_image` / `trigger_photo_capture` | Store images (TIFF or console photo). |
| `control_vent`, `pump_chamber` | Chamber vent/pump control (safety gated). |

**Beam & Detector Controls**

| Tool | Purpose |
| --- | --- |
| `get_spot_size` / `set_spot_size` | Read or set probe current (1–10). |
| `get_magnification` / `set_magnification` | Read or set magnification (20–400 000). |
| `get_stigmator` / `set_stigmator` | Inspect/update stigmator X/Y. |
| `get_detector` / `set_detector` | Inspect or switch active detector (`allow_detector_switching`). |
| `read_high_tension` | Included above but relevant for beam tuning. |

**Scan Timing & Geometry**

| Tool | Purpose |
| --- | --- |
| `get_line_time` / `set_line_time` | Read or set line time (ms or `TV`). |
| `get_lines_per_frame` / `set_lines_per_frame` | Inspect or adjust lines per frame. |
| `get_scan_rotation` / `set_scan_rotation` | Read or set scan rotation (`allow_scan_rotation_changes`). |
| `get_area_dot_shift` / `set_area_dot_shift` | Manage area/dot shift percentages (`allow_beam_shift`). |
| `get_selected_area_size` / `set_selected_area_size` | Control selected area dimensions. |

**Imaging Utilities**

| Tool | Purpose |
| --- | --- |
| `get_contrast` / `set_contrast` | Read or set contrast (0–100). |
| `get_brightness` / `set_brightness` | Read or set brightness (0–100). |
| `auto_contrast_brightness`, `auto_focus` | Run built-in adjustment routines. |
| `get_databar_text` / `set_databar_text` | Inspect or update the image databar text. |

**Stage & Alignment**

| Tool | Purpose |
| --- | --- |
| `stage_home` | Home the stage (`allow_stage_motion`). |
| `get_stage_position` / `set_stage_position` | Read or move X/Y/Z/tilt/rotation (`allow_stage_motion`). |
| `get_beam_shift` / `set_beam_shift` | Inspect or adjust beam shift (`allow_beam_shift`). |

**Image Filtering & Specimen Current**

| Tool | Purpose |
| --- | --- |
| `get_image_filter_mode` / `set_image_filter_mode` | Manage FastMCP image filter + frame count (`allow_image_filter_changes`). |
| `get_specimen_current_detector_mode` / `set_specimen_current_detector_mode` | Inspect or change detector mode (`allow_specimen_current_mode_changes`). |
| `get_specimen_current` | Read specimen current (requires measure mode). |

**Beam Safety & Locks**

| Tool | Purpose |
| --- | --- |
| `is_beam_blanked`, `blank_beam`, `unblank_beam` | Inspect or control beam blank state (`allow_beam_blank_control`). |
| `get_oplock_state`, `set_oplock_state` | Inspect or control the operator lock (`allow_oplock_control`). |

Every setter uses blocking pyxl30 calls inside `asyncio.to_thread`, preserving the FastMCP event loop responsiveness. Review the safety settings to enable only the tools you trust agents with.

All setters perform blocking pyxl30 calls inside `asyncio.to_thread` so the MCP
event loop stays responsive.

### Resources

- `mcpxl30://instrument/capabilities` – supported scan modes, image filter
  names, and the configured safety envelope.
- `mcpxl30://instrument/config` – sanitized live configuration (excludes API
  secrets).

## Examples

An `examples/example_config.json` file is included to bootstrap deployments.
The repository mirrors `mcpMQTT`'s project layout so existing FastMCP
infrastructure (supervisors, packaging, docs) can be reused with minimal
changes.
