from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

from voicepad_core.deployments import PARAKEET_V3_CUDA

from voicepad.tui.theme import DEFAULT_THEME, THEMES

CONFIG_SCHEMA = 1


class ConfigurationError(ValueError):
    """Raised when VoicePad configuration is obsolete or invalid."""


@dataclass(frozen=True, slots=True)
class AppConfig:
    deployment_id: str = PARAKEET_V3_CUDA.id
    recordings_path: Path = field(default_factory=lambda: Path.home() / ".config/voicepad/data/recordings")
    markdown_path: Path = field(default_factory=lambda: Path.home() / ".config/voicepad/data/markdown")
    artifact_cache_path: Path = field(default_factory=lambda: Path.home() / ".cache/voicepad-v2/artifacts")
    recording_prefix: str = "recording"
    input_device_index: int | None = None
    copy_complete_text: bool = True
    theme: str = DEFAULT_THEME

    def __post_init__(self) -> None:
        if self.deployment_id != PARAKEET_V3_CUDA.id:
            raise ConfigurationError(f"Unsupported deployment_id: {self.deployment_id}")
        if not self.recording_prefix or any(separator in self.recording_prefix for separator in ("/", "\\")):
            raise ConfigurationError("recording_prefix must be a nonempty filename prefix")
        if self.theme not in THEMES:
            raise ConfigurationError(f"Unsupported Textual theme: {self.theme}")


def config_path() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "voicepad" / "config-v2.json"


def load_config(path: Path | None = None) -> AppConfig:
    selected = path or config_path()
    if not selected.exists():
        return AppConfig()
    try:
        loaded = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"Could not read configuration: {selected}") from error
    if not isinstance(loaded, dict):
        raise ConfigurationError("VoicePad configuration must be a JSON object")
    raw = cast(dict[str, object], loaded)
    raw.pop("proper_nouns", None)
    expected = {
        "schema",
        "deployment_id",
        "recordings_path",
        "markdown_path",
        "artifact_cache_path",
        "recording_prefix",
        "input_device_index",
        "copy_complete_text",
        "theme",
    }
    unknown = set(raw) - expected
    if unknown:
        raise ConfigurationError(
            f"Obsolete or unknown configuration fields: {', '.join(sorted(unknown))}. "
            "Rewrite the file for VoicePad configuration schema 1."
        )
    if raw.get("schema") != CONFIG_SCHEMA:
        raise ConfigurationError(f"Unsupported configuration schema: {raw.get('schema')!r}")
    try:
        input_device = raw.get("input_device_index")
        if input_device is not None and (not isinstance(input_device, int) or isinstance(input_device, bool)):
            raise TypeError("input_device_index must be an integer or null")
        copy_complete = raw["copy_complete_text"]
        if not isinstance(copy_complete, bool):
            raise TypeError("copy_complete_text must be a boolean")
        theme = raw.get("theme", DEFAULT_THEME)
        if not isinstance(theme, str):
            raise TypeError("theme must be a string")
        return AppConfig(
            deployment_id=_required_string(raw, "deployment_id"),
            recordings_path=Path(_required_string(raw, "recordings_path")).expanduser(),
            markdown_path=Path(_required_string(raw, "markdown_path")).expanduser(),
            artifact_cache_path=Path(_required_string(raw, "artifact_cache_path")).expanduser(),
            recording_prefix=_required_string(raw, "recording_prefix"),
            input_device_index=input_device,
            copy_complete_text=copy_complete,
            theme=theme,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigurationError("VoicePad configuration schema 1 is invalid") from error


def _required_string(raw: dict[str, object], key: str) -> str:
    value = raw[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def save_config(config: AppConfig, path: Path | None = None) -> Path:
    selected = path or config_path()
    selected.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    payload["schema"] = CONFIG_SCHEMA
    for key in ("recordings_path", "markdown_path", "artifact_cache_path"):
        payload[key] = str(payload[key])
    descriptor, temporary_name = tempfile.mkstemp(dir=selected.parent, prefix=".config-v2-", suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, selected)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return selected
