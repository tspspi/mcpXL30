"""FastMCP server exposing Philips XL30 microscope controls."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import stat
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Optional, Tuple

from mcp.server.fastmcp import Context, FastMCP

from xl30serial.xl30serial import XL30Serial
from xl30serial.scanningelectronmicroscope import (
    ScanningElectronMicroscope_ImageFilterMode,
    ScanningElectronMicroscope_ScanMode,
    ScanningElectronMicroscope_SpecimenCurrentDetectorMode,
)

from mcpXL30.config.api_keys import verify_api_key
from mcpXL30.config.config_manager import (
    generate_and_store_api_key,
    get_config,
    parse_arguments,
)
from mcpXL30.config.schema import Config


logger = logging.getLogger(__name__)


@dataclass
class XL30AppContext:
    """Application state shared with FastMCP handlers."""

    instrument: XL30Serial
    config: Config


_current_app_ctx: Optional[XL30AppContext] = None


LINE_TIME_VALUES = [1.25, 1.87, 3.43, 6.86, 20.0, 40.0, 60.0, 120.0, 240.0, 360.0, 1020.0]
LINE_TIME_SPECIAL = {"TV"}
LINES_PER_FRAME_VALUES = [121, 242, 484, 968, 1452, 1936, 2420, 2904, 3388, 3872, 180, 360, 720]
LINES_PER_FRAME_SPECIAL = {"TV"}


async def _ctx_info(ctx: Optional[Context], message: str) -> None:
    if ctx is not None:
        await ctx.info(message)


def _create_instrument(config: Config) -> XL30Serial:
    instr = config.instrument
    instrument_logger = logging.getLogger("mcpXL30.instrument")
    instrument = XL30Serial(
        port=instr.port,
        logger=instrument_logger,
        debug=instr.debug,
        loglevel=instr.log_level,
        detectorsAutodetect=instr.detectors_autodetect,
        retryCount=instr.retry_count,
        reconnectCount=instr.reconnect_count,
        retryDelay=instr.retry_delay,
        reconnectDelay=instr.reconnect_delay,
    )
    return instrument


async def _enter_instrument(instrument: XL30Serial) -> None:
    await asyncio.to_thread(instrument.__enter__)


async def _exit_instrument(instrument: XL30Serial) -> None:
    await asyncio.to_thread(instrument.__exit__, None, None, None)


def _get_app_context(ctx: Optional[Context]) -> XL30AppContext:
    if ctx is None or ctx.request_context is None:
        raise RuntimeError("FastMCP context was not provided")
    return ctx.request_context.lifespan_context


async def _call_instrument(method, *args, **kwargs):
    return await asyncio.to_thread(method, *args, **kwargs)


def _normalize_line_time(value: Any) -> Any:
    if isinstance(value, str):
        candidate = value.strip().upper()
        if candidate in LINE_TIME_SPECIAL:
            return candidate
        try:
            value = float(value)
        except ValueError as exc:
            raise ValueError("Line time must be a numeric value or 'TV'") from exc
    if isinstance(value, (int, float)):
        for option in LINE_TIME_VALUES:
            if math.isclose(float(value), option, abs_tol=1e-2):
                return option
    raise ValueError(f"Unsupported line time '{value}'. Allowed: {LINE_TIME_VALUES + list(LINE_TIME_SPECIAL)}")


def _normalize_lines_per_frame(value: Any) -> Any:
    if isinstance(value, str):
        candidate = value.strip().upper()
        if candidate in LINES_PER_FRAME_SPECIAL:
            return candidate
        try:
            value = int(value)
        except ValueError as exc:
            raise ValueError("Lines per frame must be an integer or 'TV'") from exc
    if isinstance(value, (int, float)):
        for option in LINES_PER_FRAME_VALUES:
            if int(value) == option:
                return option
    raise ValueError(
        f"Unsupported lines per frame '{value}'. Allowed: {LINES_PER_FRAME_VALUES + list(LINES_PER_FRAME_SPECIAL)}"
    )


def _lookup_image_filter_mode(mode: str) -> ScanningElectronMicroscope_ImageFilterMode:
    normalized = mode.strip().upper()
    try:
        return ScanningElectronMicroscope_ImageFilterMode[normalized]
    except KeyError as exc:
        valid = ", ".join(m.name for m in ScanningElectronMicroscope_ImageFilterMode)
        raise ValueError(f"Unknown image filter mode '{mode}'. Valid modes: {valid}") from exc


def _lookup_specimen_current_mode(mode: str) -> ScanningElectronMicroscope_SpecimenCurrentDetectorMode:
    normalized = mode.strip().upper()
    try:
        return ScanningElectronMicroscope_SpecimenCurrentDetectorMode[normalized]
    except KeyError as exc:
        valid = ", ".join(m.name for m in ScanningElectronMicroscope_SpecimenCurrentDetectorMode)
        raise ValueError(f"Unknown specimen current detector mode '{mode}'. Valid modes: {valid}") from exc


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


@asynccontextmanager
async def xl30_lifespan(server: FastMCP) -> AsyncIterator[XL30AppContext]:
    """Initialize the XL30 driver when FastMCP starts."""
    global _current_app_ctx
    config = get_config()
    instrument = _create_instrument(config)
    logger.info("Connecting to XL30 on %s", config.instrument.port)
    await _enter_instrument(instrument)
    app_ctx = XL30AppContext(instrument=instrument, config=config)
    _current_app_ctx = app_ctx

    try:
        yield app_ctx
    finally:
        logger.info("Disconnecting from XL30")
        _current_app_ctx = None
        await _exit_instrument(instrument)


mcp = FastMCP("mcpxl30", lifespan=xl30_lifespan)


def _safe_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, indent=2, default=str)


@mcp.tool(
    annotations={
        "title": "Identify the connected XL30 instrument and summarize state.",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def instrument_identify(ctx: Context = None) -> str:
    """Return the microscope type, serial number, and selected telemetry."""
    try:
        app_ctx = _get_app_context(ctx)
        ident = await _call_instrument(app_ctx.instrument._get_id)
        hv = await _call_instrument(app_ctx.instrument._get_hightension)
        scan = await _call_instrument(app_ctx.instrument._get_scanmode)
        status = {
            "instrument": ident,
            "high_tension_volts": hv if hv else 0,
            "high_tension_enabled": bool(hv),
            "scan_mode": scan["name"] if scan else None,
            "port": app_ctx.config.instrument.port,
        }
        return _safe_json(status)
    except Exception as exc:  # pragma: no cover - operational guard
        logger.exception("instrument_identify failed")
        return f"Error identifying instrument: {exc}"


@mcp.tool(
    annotations={
        "title": "Set the accelerating voltage (kV) or disable high tension.",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def set_high_tension(target_kv: float, ctx: Context = None) -> str:
    """Set the XL30 high tension in kilovolts (0 disables)."""
    try:
        app_ctx = _get_app_context(ctx)
        if target_kv < 0:
            return "Error: target_kv must be >= 0"
        safety_limit = app_ctx.config.safety.max_high_tension_kv
        if target_kv > safety_limit:
            return f"Error: target_kv exceeds configured safety limit of {safety_limit} kV"

        volts = 0 if target_kv == 0 else int(target_kv * 1000)
        await _ctx_info(ctx, f"Setting high tension to {volts} V")
        success = await _call_instrument(app_ctx.instrument._set_hightension, volts)
        result = {
            "requested_kv": target_kv,
            "requested_volts": volts,
            "result": bool(success),
        }
        return _safe_json(result)
    except Exception as exc:  # pragma: no cover - operational guard
        logger.exception("set_high_tension failed")
        return f"Error setting high tension: {exc}"


@mcp.tool(
    annotations={
        "title": "Read the current high tension voltage.",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def read_high_tension(ctx: Context = None) -> str:
    """Fetch the latest measured high tension voltage."""
    try:
        app_ctx = _get_app_context(ctx)
        volts = await _call_instrument(app_ctx.instrument._get_hightension)
        result = {
            "volts": volts if volts else 0,
            "enabled": bool(volts),
        }
        return _safe_json(result)
    except Exception as exc:  # pragma: no cover - operational guard
        logger.exception("read_high_tension failed")
        return f"Error reading high tension: {exc}"


def _lookup_scan_mode(mode: str) -> ScanningElectronMicroscope_ScanMode:
    normalized = mode.strip().upper()
    try:
        return ScanningElectronMicroscope_ScanMode[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown scan mode '{mode}'. Valid modes: {', '.join(e.name for e in ScanningElectronMicroscope_ScanMode)}"
        ) from exc


@mcp.tool(
    annotations={
        "title": "Change the scan mode (e.g., FULL_FRAME, LINE_X).",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def set_scan_mode(mode: str, ctx: Context = None) -> str:
    """Switch the XL30 scan mode."""
    try:
        app_ctx = _get_app_context(ctx)
        if not app_ctx.config.safety.allow_scan_mode_changes:
            return "Error: Scan mode changes disabled by safety configuration"
        try:
            target_mode = _lookup_scan_mode(mode)
        except ValueError as exc:
            return str(exc)

        await _ctx_info(ctx, f"Setting scan mode to {target_mode.name}")
        await _call_instrument(app_ctx.instrument._set_scanmode, target_mode)
        return _safe_json({"scan_mode": target_mode.name})
    except Exception as exc:  # pragma: no cover - operational guard
        logger.exception("set_scan_mode failed")
        return f"Error setting scan mode: {exc}"


def _build_remote_path(directory: str, filename: str) -> str:
    if directory.endswith(("\\", "/")):
        return f"{directory}{filename}"
    if "\\" in directory and "/" not in directory:
        return f"{directory}\\{filename}"
    return f"{directory}/{filename}"


@mcp.tool(
    annotations={
        "title": "Capture a TIFF image to the console PC.",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def capture_image(
    remote_path: Optional[str] = None,
    include_databar: Optional[bool] = None,
    include_magnification: Optional[bool] = None,
    include_graphics_plane: Optional[bool] = None,
    overwrite: Optional[bool] = None,
    ctx: Context = None,
) -> str:
    """Write a TIFF image to a path visible to the XL30 console host."""
    try:
        app_ctx = _get_app_context(ctx)
        capture_cfg = app_ctx.config.image_capture

        if not remote_path:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"{capture_cfg.filename_prefix}{timestamp}.TIF"
            remote_path = _build_remote_path(capture_cfg.remote_directory, filename)

        databar = capture_cfg.include_databar if include_databar is None else include_databar
        magnification = (
            capture_cfg.include_magnification if include_magnification is None else include_magnification
        )
        graphics = (
            capture_cfg.include_graphics_plane if include_graphics_plane is None else include_graphics_plane
        )
        overwrite_flag = capture_cfg.allow_overwrite if overwrite is None else overwrite

        await _ctx_info(ctx, f"Writing TIFF image to {remote_path}")
        resp = await _call_instrument(
            app_ctx.instrument._write_tiff_image,
            remote_path,
            printmagnification=magnification,
            graphicsbitplane=graphics,
            databar=databar,
            overwrite=overwrite_flag,
        )
        return _safe_json({"remote_path": remote_path, "response": resp})
    except Exception as exc:  # pragma: no cover - operational guard
        logger.exception("capture_image failed")
        return f"Error capturing image: {exc}"


@mcp.tool(
    annotations={
        "title": "Vent or stop venting the specimen chamber.",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def control_vent(start: bool = True, ctx: Context = None) -> str:
    """Start or stop the venting cycle."""
    try:
        app_ctx = _get_app_context(ctx)
        if not app_ctx.config.safety.allow_venting and start:
            return "Error: Venting disabled by safety configuration"

        result = await _call_instrument(app_ctx.instrument._vent, stop=not start)
        return _safe_json({"venting": start, "result": bool(result)})
    except Exception as exc:  # pragma: no cover - operational guard
        logger.exception("control_vent failed")
        return f"Error controlling vent: {exc}"


@mcp.tool(
    annotations={
        "title": "Start a pump cycle on the chamber.",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def pump_chamber(ctx: Context = None) -> str:
    """Trigger the XL30 pump cycle."""
    try:
        app_ctx = _get_app_context(ctx)
        if not app_ctx.config.safety.allow_pumping:
            return "Error: Pumping disabled by safety configuration"
        result = await _call_instrument(app_ctx.instrument._pump)
        return _safe_json({"pumping": True, "result": bool(result)})
    except Exception as exc:  # pragma: no cover - operational guard
        logger.exception("pump_chamber failed")
        return f"Error pumping chamber: {exc}"


# === Beam parameter controls ===


@mcp.tool(
    annotations={
        "title": "Read the current spot size/probe current setting.",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def get_spot_size(ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        value = await _call_instrument(app_ctx.instrument._get_spotsize)
        return _safe_json({"spot_size": value})
    except Exception as exc:
        logger.exception("get_spot_size failed")
        return f"Error reading spot size: {exc}"


@mcp.tool(
    annotations={
        "title": "Set the spot size/probe current (1.0-10.0).",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def set_spot_size(value: float, ctx: Context = None) -> str:
    try:
        if value < 1.0 or value > 10.0:
            return "Error: spot size must be between 1.0 and 10.0"
        app_ctx = _get_app_context(ctx)
        await _ctx_info(ctx, f"Setting spot size to {value}")
        result = await _call_instrument(app_ctx.instrument._set_spotsize, value)
        return _safe_json({"spot_size": value, "result": bool(result)})
    except Exception as exc:
        logger.exception("set_spot_size failed")
        return f"Error setting spot size: {exc}"


@mcp.tool(
    annotations={
        "title": "Read the current magnification.",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def get_magnification(ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        mag = await _call_instrument(app_ctx.instrument._get_magnification)
        return _safe_json({"magnification": mag})
    except Exception as exc:
        logger.exception("get_magnification failed")
        return f"Error reading magnification: {exc}"


@mcp.tool(
    annotations={
        "title": "Set magnification (20-400000).",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def set_magnification(value: float, ctx: Context = None) -> str:
    try:
        if value < 20 or value > 400000:
            return "Error: magnification must be between 20 and 400000"
        app_ctx = _get_app_context(ctx)
        await _ctx_info(ctx, f"Setting magnification to {value}")
        result = await _call_instrument(app_ctx.instrument._set_magnification, value)
        return _safe_json({"magnification": value, "result": bool(result)})
    except Exception as exc:
        logger.exception("set_magnification failed")
        return f"Error setting magnification: {exc}"


@mcp.tool(
    annotations={
        "title": "Read the stigmator X/Y values.",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def get_stigmator(ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        x, y = await _call_instrument(app_ctx.instrument._get_stigmator)
        return _safe_json({"x": x, "y": y})
    except Exception as exc:
        logger.exception("get_stigmator failed")
        return f"Error reading stigmator: {exc}"


@mcp.tool(
    annotations={
        "title": "Set stigmator X/Y values.",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def set_stigmator(x: Optional[float] = None, y: Optional[float] = None, ctx: Context = None) -> str:
    try:
        if x is None and y is None:
            return "Error: provide at least one of x or y"
        app_ctx = _get_app_context(ctx)
        await _ctx_info(ctx, "Updating stigmator")
        result = await _call_instrument(app_ctx.instrument._set_stigmator, x, y)
        return _safe_json({"x": x, "y": y, "result": bool(result)})
    except Exception as exc:
        logger.exception("set_stigmator failed")
        return f"Error setting stigmator: {exc}"


@mcp.tool(
    annotations={
        "title": "Read the currently selected detector.",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def get_detector(ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        detector = await _call_instrument(app_ctx.instrument._get_detector)
        return _safe_json({"detector": detector})
    except Exception as exc:
        logger.exception("get_detector failed")
        return f"Error reading detector: {exc}"


@mcp.tool(
    annotations={
        "title": "Select a detector by numeric ID.",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def set_detector(detector_id: int, ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        if not app_ctx.config.safety.allow_detector_switching:
            return "Error: Detector switching disabled by safety configuration"
        await _ctx_info(ctx, f"Switching detector to ID {detector_id}")
        result = await _call_instrument(app_ctx.instrument._set_detector, int(detector_id))
        return _safe_json({"detector_id": detector_id, "result": bool(result)})
    except Exception as exc:
        logger.exception("set_detector failed")
        return f"Error setting detector: {exc}"


# === Scan timing ===


@mcp.tool(
    annotations={
        "title": "Read the configured line time (ms or TV).",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def get_line_time(ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        value = await _call_instrument(app_ctx.instrument._get_linetime)
        return _safe_json({"line_time": value})
    except Exception as exc:
        logger.exception("get_line_time failed")
        return f"Error reading line time: {exc}"


@mcp.tool(
    annotations={
        "title": "Set the line time (ms) or 'TV'.",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def set_line_time(value: str, ctx: Context = None) -> str:
    try:
        normalized = _normalize_line_time(value)
        app_ctx = _get_app_context(ctx)
        await _ctx_info(ctx, f"Setting line time to {normalized}")
        result = await _call_instrument(app_ctx.instrument._set_linetime, normalized)
        return _safe_json({"line_time": normalized, "result": bool(result)})
    except Exception as exc:
        logger.exception("set_line_time failed")
        return f"Error setting line time: {exc}"


@mcp.tool(
    annotations={
        "title": "Read the number of lines per frame.",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def get_lines_per_frame(ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        value = await _call_instrument(app_ctx.instrument._get_linesperframe)
        return _safe_json({"lines_per_frame": value})
    except Exception as exc:
        logger.exception("get_lines_per_frame failed")
        return f"Error reading lines per frame: {exc}"


@mcp.tool(
    annotations={
        "title": "Set lines per frame (value or 'TV').",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def set_lines_per_frame(value: str, ctx: Context = None) -> str:
    try:
        normalized = _normalize_lines_per_frame(value)
        app_ctx = _get_app_context(ctx)
        await _ctx_info(ctx, f"Setting lines per frame to {normalized}")
        result = await _call_instrument(app_ctx.instrument._set_linesperframe, normalized)
        return _safe_json({"lines_per_frame": normalized, "result": bool(result)})
    except Exception as exc:
        logger.exception("set_lines_per_frame failed")
        return f"Error setting lines per frame: {exc}"


@mcp.tool(
    annotations={
        "title": "Read the current scan mode.",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def get_scan_mode(ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        mode = await _call_instrument(app_ctx.instrument._get_scanmode)
        return _safe_json({"scan_mode": mode})
    except Exception as exc:
        logger.exception("get_scan_mode failed")
        return f"Error reading scan mode: {exc}"


# === Imaging utilities ===


@mcp.tool(
    annotations={
        "title": "Trigger the console PC to store a photo.",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def trigger_photo_capture(ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        await _ctx_info(ctx, "Requesting photo capture on console PC")
        result = await _call_instrument(app_ctx.instrument._make_photo)
        return _safe_json({"result": bool(result)})
    except Exception as exc:
        logger.exception("trigger_photo_capture failed")
        return f"Error triggering photo capture: {exc}"


@mcp.tool(
    annotations={
        "title": "Read contrast (0-100).",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def get_contrast(ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        value = await _call_instrument(app_ctx.instrument._get_contrast)
        return _safe_json({"contrast": value})
    except Exception as exc:
        logger.exception("get_contrast failed")
        return f"Error reading contrast: {exc}"


@mcp.tool(
    annotations={
        "title": "Set contrast (0-100).",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def set_contrast(value: float, ctx: Context = None) -> str:
    try:
        if value < 0 or value > 100:
            return "Error: contrast must be between 0 and 100"
        app_ctx = _get_app_context(ctx)
        await _ctx_info(ctx, f"Setting contrast to {value}")
        result = await _call_instrument(app_ctx.instrument._set_contrast, value)
        return _safe_json({"contrast": value, "result": bool(result)})
    except Exception as exc:
        logger.exception("set_contrast failed")
        return f"Error setting contrast: {exc}"


@mcp.tool(
    annotations={
        "title": "Read brightness (0-100).",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def get_brightness(ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        value = await _call_instrument(app_ctx.instrument._get_brightness)
        return _safe_json({"brightness": value})
    except Exception as exc:
        logger.exception("get_brightness failed")
        return f"Error reading brightness: {exc}"


@mcp.tool(
    annotations={
        "title": "Set brightness (0-100).",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def set_brightness(value: float, ctx: Context = None) -> str:
    try:
        if value < 0 or value > 100:
            return "Error: brightness must be between 0 and 100"
        app_ctx = _get_app_context(ctx)
        await _ctx_info(ctx, f"Setting brightness to {value}")
        result = await _call_instrument(app_ctx.instrument._set_brightness, value)
        return _safe_json({"brightness": value, "result": bool(result)})
    except Exception as exc:
        logger.exception("set_brightness failed")
        return f"Error setting brightness: {exc}"


@mcp.tool(
    annotations={
        "title": "Run auto contrast/brightness.",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def auto_contrast_brightness(ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        await _ctx_info(ctx, "Starting auto contrast/brightness (takes ~30s)")
        result = await _call_instrument(app_ctx.instrument._auto_contrastbrightness)
        return _safe_json({"result": bool(result)})
    except Exception as exc:
        logger.exception("auto_contrast_brightness failed")
        return f"Error running auto contrast/brightness: {exc}"


@mcp.tool(
    annotations={
        "title": "Run auto focus routine.",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def auto_focus(ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        await _ctx_info(ctx, "Starting auto focus (blocking until finished)")
        result = await _call_instrument(app_ctx.instrument._auto_focus)
        return _safe_json({"result": bool(result)})
    except Exception as exc:
        logger.exception("auto_focus failed")
        return f"Error running auto focus: {exc}"


@mcp.tool(
    annotations={
        "title": "Read current databar text.",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def get_databar_text(ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        text = await _call_instrument(app_ctx.instrument._get_databar_text)
        return _safe_json({"databar_text": text})
    except Exception as exc:
        logger.exception("get_databar_text failed")
        return f"Error reading databar text: {exc}"


@mcp.tool(
    annotations={
        "title": "Set databar text (<=40 ASCII chars).",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def set_databar_text(text: str, ctx: Context = None) -> str:
    try:
        if len(text) > 40:
            return "Error: databar text limited to 40 characters"
        text.encode("ascii")
        app_ctx = _get_app_context(ctx)
        await _ctx_info(ctx, "Updating databar text")
        result = await _call_instrument(app_ctx.instrument._set_databar_text, text)
        return _safe_json({"databar_text": text, "result": bool(result)})
    except UnicodeEncodeError:
        return "Error: databar text must be ASCII"
    except Exception as exc:
        logger.exception("set_databar_text failed")
        return f"Error setting databar text: {exc}"


# === Stage and alignment ===


@mcp.tool(
    annotations={
        "title": "Home the stage (prompts on console).",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def stage_home(ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        if not app_ctx.config.safety.allow_stage_motion:
            return "Error: Stage motion disabled by safety configuration"
        await _ctx_info(ctx, "Homing stage (requires console confirmation)")
        result = await _call_instrument(app_ctx.instrument._stage_home)
        return _safe_json({"result": bool(result)})
    except Exception as exc:
        logger.exception("stage_home failed")
        return f"Error homing stage: {exc}"


@mcp.tool(
    annotations={
        "title": "Read the current stage position.",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def get_stage_position(ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        pos = await _call_instrument(app_ctx.instrument._get_stage_position)
        return _safe_json({"position": pos})
    except Exception as exc:
        logger.exception("get_stage_position failed")
        return f"Error reading stage position: {exc}"


@mcp.tool(
    annotations={
        "title": "Move the stage (supply any subset of axes).",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def set_stage_position(
    x: Optional[float] = None,
    y: Optional[float] = None,
    z: Optional[float] = None,
    tilt: Optional[float] = None,
    rot: Optional[float] = None,
    ctx: Context = None,
) -> str:
    try:
        if x is None and y is None and z is None and tilt is None and rot is None:
            return "Error: provide at least one axis to move"
        app_ctx = _get_app_context(ctx)
        if not app_ctx.config.safety.allow_stage_motion:
            return "Error: Stage motion disabled by safety configuration"
        await _ctx_info(ctx, "Moving stage")
        result = await _call_instrument(app_ctx.instrument._set_stage_position, x, y, z, tilt, rot)
        return _safe_json({"x": x, "y": y, "z": z, "tilt": tilt, "rot": rot, "result": bool(result)})
    except Exception as exc:
        logger.exception("set_stage_position failed")
        return f"Error moving stage: {exc}"


@mcp.tool(
    annotations={
        "title": "Read beam shift (mm).",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def get_beam_shift(ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        shift = await _call_instrument(app_ctx.instrument._get_beamshift)
        return _safe_json({"beam_shift": shift})
    except Exception as exc:
        logger.exception("get_beam_shift failed")
        return f"Error reading beam shift: {exc}"


@mcp.tool(
    annotations={
        "title": "Set beam shift (mm).",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def set_beam_shift(x: Optional[float] = None, y: Optional[float] = None, ctx: Context = None) -> str:
    try:
        if x is None and y is None:
            return "Error: provide at least x or y"
        app_ctx = _get_app_context(ctx)
        if not app_ctx.config.safety.allow_beam_shift:
            return "Error: Beam shift adjustments disabled by safety configuration"
        await _ctx_info(ctx, "Setting beam shift")
        result = await _call_instrument(app_ctx.instrument._set_beamshift, x, y)
        return _safe_json({"x": x, "y": y, "result": bool(result)})
    except Exception as exc:
        logger.exception("set_beam_shift failed")
        return f"Error setting beam shift: {exc}"


@mcp.tool(
    annotations={
        "title": "Read scan rotation (deg).",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def get_scan_rotation(ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        value = await _call_instrument(app_ctx.instrument._get_scanrotation)
        return _safe_json({"scan_rotation_deg": value})
    except Exception as exc:
        logger.exception("get_scan_rotation failed")
        return f"Error reading scan rotation: {exc}"


@mcp.tool(
    annotations={
        "title": "Set scan rotation (-90 to 90 deg).",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def set_scan_rotation(value: float, ctx: Context = None) -> str:
    try:
        if value < -90 or value > 90:
            return "Error: scan rotation must be between -90 and 90 degrees"
        app_ctx = _get_app_context(ctx)
        if not app_ctx.config.safety.allow_scan_rotation_changes:
            return "Error: Scan rotation changes disabled by safety configuration"
        await _ctx_info(ctx, f"Setting scan rotation to {value}")
        result = await _call_instrument(app_ctx.instrument._set_scanrotation, value)
        return _safe_json({"scan_rotation_deg": value, "result": bool(result)})
    except Exception as exc:
        logger.exception("set_scan_rotation failed")
        return f"Error setting scan rotation: {exc}"


@mcp.tool(
    annotations={
        "title": "Read area/dot shift (%).",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def get_area_dot_shift(ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        xshift, yshift = await _call_instrument(app_ctx.instrument._get_area_or_dot_shift)
        return _safe_json({"x_shift_percent": xshift, "y_shift_percent": yshift})
    except Exception as exc:
        logger.exception("get_area_dot_shift failed")
        return f"Error reading area/dot shift: {exc}"


@mcp.tool(
    annotations={
        "title": "Set area/dot shift (%).",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def set_area_dot_shift(x: Optional[float] = None, y: Optional[float] = None, ctx: Context = None) -> str:
    try:
        if x is None and y is None:
            return "Error: provide at least x or y shift"
        app_ctx = _get_app_context(ctx)
        if not app_ctx.config.safety.allow_beam_shift:
            return "Error: Area/dot shift disabled by safety configuration"
        if x is not None and (x < -100 or x > 100):
            return "Error: x shift must be between -100 and 100 percent"
        if y is not None and (y < -100 or y > 100):
            return "Error: y shift must be between -100 and 100 percent"
        await _ctx_info(ctx, "Setting area/dot shift")
        result = await _call_instrument(app_ctx.instrument._set_area_or_dot_shift, x, y)
        return _safe_json({"x_shift_percent": x, "y_shift_percent": y, "result": bool(result)})
    except Exception as exc:
        logger.exception("set_area_dot_shift failed")
        return f"Error setting area/dot shift: {exc}"


@mcp.tool(
    annotations={
        "title": "Read selected area size (%).",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def get_selected_area_size(ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        size = await _call_instrument(app_ctx.instrument._get_selected_area_size)
        return _safe_json({"selected_area_size": size})
    except Exception as exc:
        logger.exception("get_selected_area_size failed")
        return f"Error reading selected area size: {exc}"


@mcp.tool(
    annotations={
        "title": "Set selected area size (%).",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def set_selected_area_size(
    sizex: Optional[float] = None, sizey: Optional[float] = None, ctx: Context = None
) -> str:
    try:
        if sizex is None and sizey is None:
            return "Error: provide at least sizex or sizey"
        if sizex is not None and (sizex < 0 or sizex > 100):
            return "Error: sizex must be between 0 and 100 percent"
        if sizey is not None and (sizey < 0 or sizey > 100):
            return "Error: sizey must be between 0 and 100 percent"
        app_ctx = _get_app_context(ctx)
        await _ctx_info(ctx, "Setting selected area size")
        result = await _call_instrument(app_ctx.instrument._set_selected_area_size, sizex, sizey)
        return _safe_json({"sizex_percent": sizex, "sizey_percent": sizey, "result": bool(result)})
    except Exception as exc:
        logger.exception("set_selected_area_size failed")
        return f"Error setting selected area size: {exc}"


# === Image filtering and specimen current ===


@mcp.tool(
    annotations={
        "title": "Read the current image filter mode.",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def get_image_filter_mode(ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        value = await _call_instrument(app_ctx.instrument._get_imagefilter_mode)
        if value and value.get("mode"):
            value["mode"] = value["mode"].name
        return _safe_json({"image_filter": value})
    except Exception as exc:
        logger.exception("get_image_filter_mode failed")
        return f"Error reading image filter mode: {exc}"


@mcp.tool(
    annotations={
        "title": "Set the image filter mode and frame count (power of two).",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def set_image_filter_mode(mode: str, frames: int, ctx: Context = None) -> str:
    try:
        if frames < 1 or not _is_power_of_two(int(frames)):
            return "Error: frames must be a positive power of two"
        app_ctx = _get_app_context(ctx)
        if not app_ctx.config.safety.allow_image_filter_changes:
            return "Error: Image filter changes disabled by safety configuration"
        filter_mode = _lookup_image_filter_mode(mode)
        await _ctx_info(ctx, f"Setting image filter to {filter_mode.name} ({frames} frames)")
        result = await _call_instrument(app_ctx.instrument._set_imagefilter_mode, filter_mode, int(frames))
        return _safe_json({"mode": filter_mode.name, "frames": int(frames), "result": bool(result)})
    except Exception as exc:
        logger.exception("set_image_filter_mode failed")
        return f"Error setting image filter mode: {exc}"


@mcp.tool(
    annotations={
        "title": "Read the specimen current detector mode.",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def get_specimen_current_detector_mode(ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        mode = await _call_instrument(app_ctx.instrument._get_specimen_current_detector_mode)
        if mode:
            mode = mode.name
        return _safe_json({"specimen_current_detector_mode": mode})
    except Exception as exc:
        logger.exception("get_specimen_current_detector_mode failed")
        return f"Error reading specimen current detector mode: {exc}"


@mcp.tool(
    annotations={
        "title": "Set the specimen current detector mode.",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def set_specimen_current_detector_mode(mode: str, ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        if not app_ctx.config.safety.allow_specimen_current_mode_changes:
            return "Error: Specimen current detector changes disabled by safety configuration"
        target = _lookup_specimen_current_mode(mode)
        await _ctx_info(ctx, f"Setting specimen current detector mode to {target.name}")
        result = await _call_instrument(app_ctx.instrument._set_specimen_current_detector_mode, target)
        return _safe_json({"mode": target.name, "result": bool(result)})
    except Exception as exc:
        logger.exception("set_specimen_current_detector_mode failed")
        return f"Error setting specimen current detector mode: {exc}"


@mcp.tool(
    annotations={
        "title": "Read the specimen current (requires measure mode).",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def get_specimen_current(ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        value = await _call_instrument(app_ctx.instrument._get_specimen_current)
        return _safe_json({"specimen_current": value})
    except Exception as exc:
        logger.exception("get_specimen_current failed")
        return f"Error reading specimen current: {exc}"


# === Beam blanking and operator lock ===


@mcp.tool(
    annotations={
        "title": "Check whether the beam is blanked.",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def is_beam_blanked(ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        value = await _call_instrument(app_ctx.instrument._is_beam_blanked)
        return _safe_json({"beam_blanked": bool(value)})
    except Exception as exc:
        logger.exception("is_beam_blanked failed")
        return f"Error querying beam blank state: {exc}"


@mcp.tool(
    annotations={
        "title": "Blank the electron beam.",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def blank_beam(ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        if not app_ctx.config.safety.allow_beam_blank_control:
            return "Error: Beam blank control disabled by safety configuration"
        await _ctx_info(ctx, "Blanking beam")
        result = await _call_instrument(app_ctx.instrument._blank)
        return _safe_json({"beam_blanked": True, "result": bool(result)})
    except Exception as exc:
        logger.exception("blank_beam failed")
        return f"Error blanking beam: {exc}"


@mcp.tool(
    annotations={
        "title": "Unblank the electron beam.",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def unblank_beam(ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        if not app_ctx.config.safety.allow_beam_blank_control:
            return "Error: Beam blank control disabled by safety configuration"
        await _ctx_info(ctx, "Unblanking beam")
        result = await _call_instrument(app_ctx.instrument._unblank)
        return _safe_json({"beam_blanked": False, "result": bool(result)})
    except Exception as exc:
        logger.exception("unblank_beam failed")
        return f"Error unblanking beam: {exc}"


@mcp.tool(
    annotations={
        "title": "Read operator lock status.",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def get_oplock_state(ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        value = await _call_instrument(app_ctx.instrument._isOplocked)
        return _safe_json({"operator_locked": bool(value)})
    except Exception as exc:
        logger.exception("get_oplock_state failed")
        return f"Error reading operator lock state: {exc}"


@mcp.tool(
    annotations={
        "title": "Set the operator lock state.",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def set_oplock_state(lock: bool, ctx: Context = None) -> str:
    try:
        app_ctx = _get_app_context(ctx)
        if not app_ctx.config.safety.allow_oplock_control:
            return "Error: Operator lock control disabled by safety configuration"
        await _ctx_info(ctx, "Updating operator lock")
        result = await _call_instrument(app_ctx.instrument._oplock, bool(lock))
        return _safe_json({"operator_locked": bool(lock), "result": bool(result)})
    except Exception as exc:
        logger.exception("set_oplock_state failed")
        return f"Error setting operator lock: {exc}"


@mcp.resource("mcpxl30://instrument/capabilities")
def instrument_capabilities() -> str:
    """Expose instrument ranges and supported scan modes."""
    config = _current_app_ctx.config if _current_app_ctx else None
    safety = (
        {
            "max_high_tension_kv": config.safety.max_high_tension_kv,
            "allow_venting": config.safety.allow_venting,
            "allow_pumping": config.safety.allow_pumping,
            "allow_scan_mode_changes": config.safety.allow_scan_mode_changes,
        }
        if config
        else {}
    )
    capabilities = {
        "connected": bool(_current_app_ctx),
        "scan_modes": [mode.name for mode in ScanningElectronMicroscope_ScanMode],
        "image_filters": [mode.name for mode in ScanningElectronMicroscope_ImageFilterMode],
        "safety": safety,
    }
    return _safe_json(capabilities)


@mcp.resource("mcpxl30://instrument/config")
def instrument_config_resource() -> str:
    """Expose a sanitized version of the live configuration."""
    if _current_app_ctx is None:
        return _safe_json({"error": "Configuration not available"})
    config = _current_app_ctx.config
    redacted = {
        "instrument": config.instrument.model_dump(),
        "image_capture": config.image_capture.model_dump(),
        "safety": config.safety.model_dump(),
        "logging": config.logging.model_dump(),
    }
    return _safe_json(redacted)


def _build_remote_fastapi_app(config: Config):
    """Create FastAPI application that wraps the MCP Starlette app."""
    remote_config = config.remote_server
    if remote_config is None:
        raise RuntimeError("Remote transport selected but remote_server config is missing")

    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse
        from starlette.middleware.base import BaseHTTPMiddleware
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Remote HTTP transport requires FastAPI. Install optional extras: 'pip install \"mcpXL30[remote]\"'"
        ) from exc

    API_KEY_QUERY_PARAM = "api_key"
    STATUS_PATH = "/status"
    MCP_MOUNT_PATH = "/mcp"

    mcp_http_app = mcp.streamable_http_app()

    @asynccontextmanager
    async def fastapi_lifespan(app):
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(mcp_http_app.router.lifespan_context(mcp_http_app))
            yield

    fastapi_app = FastAPI(title="mcpXL30 Remote Server", version="1.0", lifespan=fastapi_lifespan)

    class APIKeyMiddleware(BaseHTTPMiddleware):
        def __init__(self, app):
            super().__init__(app)
            self._remote_config = remote_config

        async def dispatch(self, request: Request, call_next):
            if request.url.path == STATUS_PATH:
                return await call_next(request)

            token = None
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ", 1)[1].strip()
            if not token:
                token = request.headers.get("x-api-key")
            if not token:
                token = request.query_params.get(API_KEY_QUERY_PARAM)

            if not verify_api_key(token, self._remote_config):
                raise HTTPException(status_code=401, detail="Invalid or missing API key")
            return await call_next(request)

    fastapi_app.add_middleware(APIKeyMiddleware)

    @fastapi_app.get(STATUS_PATH)
    async def status():
        """Unprotected health endpoint."""
        return JSONResponse(
            {
                "running": True,
                "instrument_connected": bool(_current_app_ctx),
                "port": _current_app_ctx.config.instrument.port if _current_app_ctx else None,
            }
        )

    fastapi_app.mount(MCP_MOUNT_PATH, mcp_http_app)
    return fastapi_app


def _prepare_uds_socket(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    if os.path.exists(path):
        current_stat = os.stat(path)
        if stat.S_ISSOCK(current_stat.st_mode):
            os.remove(path)
        else:
            raise RuntimeError(f"UDS path {path} exists and is not a socket")


def _run_remote_transport(config: Config) -> None:
    remote_config = config.remote_server
    if remote_config is None:
        raise RuntimeError("remote_server configuration is required for remote transport")

    app = _build_remote_fastapi_app(config)

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Remote HTTP transport requires uvicorn. Install optional extras: 'pip install \"mcpXL30[remote]\"'"
        ) from exc

    uvicorn_kwargs: Dict[str, Any] = {"app": app, "lifespan": "on", "log_config": None}

    if remote_config.port is not None:
        uvicorn_kwargs["host"] = remote_config.host or "0.0.0.0"
        uvicorn_kwargs["port"] = remote_config.port
        logger.info(
            "Starting remote MCP server over TCP at %s:%s",
            uvicorn_kwargs["host"],
            uvicorn_kwargs["port"],
        )
    else:
        uds_path = remote_config.uds
        _prepare_uds_socket(uds_path)
        uvicorn_kwargs["uds"] = uds_path
        logger.info("Starting remote MCP server over UDS at %s", uds_path)

    uvicorn.run(**uvicorn_kwargs)


def run_mcp_server():
    """Entry point shared by console scripts."""
    args = parse_arguments()

    if getattr(args, "genkey", False):
        new_key = generate_and_store_api_key(args.config)
        print(new_key)
        return

    config = get_config(args)

    try:
        if args.transport == "remotehttp":
            if not config.remote_server:
                raise RuntimeError(
                    "remotehttp transport selected but remote_server configuration block is missing"
                )
            _run_remote_transport(config)
        else:
            mcp.run()
    except KeyboardInterrupt:
        logger.info("MCP server shutting down...")


if __name__ == "__main__":
    run_mcp_server()
