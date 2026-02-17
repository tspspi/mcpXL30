"""Pydantic schemas for mcpXL30 configuration."""

from __future__ import annotations

import base64
import binascii
from typing import Literal, Optional

from pydantic import BaseModel, Field, validator, root_validator


class InstrumentConfig(BaseModel):
    """Serial connection and behavior settings for the XL30."""

    port: str = Field(
        default="/dev/ttyUSB0",
        description="Serial device path or pyserial URL connected to the XL30 console server",
    )
    debug: bool = Field(
        default=False,
        description="Enable verbose debug logging inside the pyxl30 driver",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="ERROR",
        description="Log level forwarded to the pyxl30 logger when one is not provided",
    )
    retry_count: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of retries performed by pyxl30 before reconnecting",
    )
    reconnect_count: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of reconnect attempts before surfacing an error",
    )
    retry_delay: float = Field(
        default=5.0,
        ge=0.1,
        description="Seconds to wait between driver retries",
    )
    reconnect_delay: float = Field(
        default=5.0,
        ge=0.1,
        description="Seconds to wait between reconnect attempts",
    )
    detectors_autodetect: bool = Field(
        default=False,
        description="Enable automatic detector probing during startup",
    )

    @validator("port")
    def validate_port(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Instrument port must be provided")
        return value.strip()


class ImageCaptureConfig(BaseModel):
    """Image capture defaults for TIFF exports."""

    remote_directory: str = Field(
        default=r"C:\TEMP",
        description="Directory path on the XL30 console PC that will receive TIFF files",
    )
    filename_prefix: str = Field(
        default="MCPIMG_",
        description="Prefix used when auto-generating filenames",
    )
    include_databar: bool = Field(
        default=True,
        description="Include instrument data bar overlay in captured images",
    )
    include_magnification: bool = Field(
        default=False,
        description="Overlay magnification text into captured images",
    )
    include_graphics_plane: bool = Field(
        default=False,
        description="Include graphical bit plane from the XL30 UI",
    )
    allow_overwrite: bool = Field(
        default=False,
        description="Permit overwriting files without prompting",
    )

    @validator("remote_directory")
    def validate_remote_directory(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("remote_directory cannot be empty")
        return value.strip()


class SafetyConfig(BaseModel):
    """Safety envelope to protect the instrument when driven by LLM agents."""

    max_high_tension_kv: float = Field(
        default=20.0,
        ge=0.2,
        le=30.0,
        description="Absolute maximum accelerating voltage (in kilovolts) the agent may request",
    )
    allow_venting: bool = Field(
        default=False,
        description="Permit the agent to trigger vent cycles",
    )
    allow_pumping: bool = Field(
        default=True,
        description="Permit the agent to trigger pump cycles",
    )
    allow_scan_mode_changes: bool = Field(
        default=True,
        description="Permit the agent to switch scan modes",
    )
    allow_stage_motion: bool = Field(
        default=False,
        description="Permit the agent to home or move the stage",
    )
    allow_detector_switching: bool = Field(
        default=True,
        description="Permit changing the active detector",
    )
    allow_beam_shift: bool = Field(
        default=True,
        description="Permit beam shift and area/dot shift adjustments",
    )
    allow_scan_rotation_changes: bool = Field(
        default=True,
        description="Permit modifying the scan rotation angle",
    )
    allow_image_filter_changes: bool = Field(
        default=True,
        description="Permit changing the image filter mode",
    )
    allow_specimen_current_mode_changes: bool = Field(
        default=False,
        description="Permit switching specimen current detector modes",
    )
    allow_beam_blank_control: bool = Field(
        default=True,
        description="Permit blanking/unblanking the beam via MCP",
    )
    allow_oplock_control: bool = Field(
        default=False,
        description="Permit enabling or disabling the operator lock remotely",
    )


class LoggingConfig(BaseModel):
    """Logging options for the server."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging verbosity for the MCP service",
    )
    logfile: Optional[str] = Field(
        default=None,
        description="Optional log file path (appends). None logs to stdout only.",
    )


class APIKeyKDFConfig(BaseModel):
    """Argon2id KDF configuration for remote API keys."""

    algorithm: Literal["argon2id"] = Field(
        default="argon2id",
        description="KDF algorithm identifier",
    )
    salt: str = Field(description="Base64 encoded salt")
    time_cost: int = Field(default=3, ge=1, description="Argon2 time cost")
    memory_cost: int = Field(default=65536, ge=8, description="Argon2 memory cost (KiB)")
    parallelism: int = Field(default=1, ge=1, description="Argon2 parallelism")
    hash_len: int = Field(default=32, ge=16, description="Derived hash length")
    hash: str = Field(description="Base64 encoded Argon2id hash output")

    @validator("salt", "hash")
    def validate_base64(cls, value: str) -> str:
        try:
            base64.b64decode(value, validate=True)
        except binascii.Error as exc:  # pragma: no cover - guard clause
            raise ValueError("Value must be base64 encoded") from exc
        return value


class RemoteServerConfig(BaseModel):
    """FastAPI/uvicorn remote MCP transport configuration."""

    api_key_kdf: Optional[APIKeyKDFConfig] = Field(
        default=None,
        description="Argon2id protected API key definition",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Deprecated plaintext API key (prefer api_key_kdf)",
    )
    host: str = Field(
        default="0.0.0.0",
        description="Bind host when TCP is enabled (ignored for UDS)",
    )
    port: Optional[int] = Field(
        default=None,
        ge=1,
        le=65535,
        description="TCP port for remote access. Use None to enable UDS mode.",
    )
    uds: str = Field(
        default="/var/run/mcpxl30.sock",
        description="Unix domain socket path when port is not set",
    )

    @validator("api_key")
    def validate_plaintext_key(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if not value:
            raise ValueError("API key cannot be empty")
        return value

    @validator("uds")
    def validate_uds(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("UDS path cannot be empty")
        return value.strip()

    @root_validator(skip_on_failure=True)
    def ensure_secret_present(cls, values):
        if not values.get("api_key") and not values.get("api_key_kdf"):
            raise ValueError("remote_server requires api_key_kdf or api_key")
        return values

    @root_validator(skip_on_failure=True)
    def default_host_for_tcp(cls, values):
        if values.get("port") is not None and not values.get("host"):
            values["host"] = "0.0.0.0"
        return values


class Config(BaseModel):
    """Root configuration document."""

    instrument: InstrumentConfig = Field(default_factory=InstrumentConfig)
    image_capture: ImageCaptureConfig = Field(default_factory=ImageCaptureConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    remote_server: Optional[RemoteServerConfig] = Field(
        default=None,
        description="Remote MCP transport configuration",
    )
