"""Configuration management commands."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import TypedDict, cast

import sounddevice as sd
import typer
from voicepad_core import get_config, get_config_with_metadata
from voicepad_core.config import Config

logger = logging.getLogger(__name__)

config_app = typer.Typer(help="Configuration management commands")


# ---------------------------------------------------------------------------
# config show
# ---------------------------------------------------------------------------


@config_app.command("show")
def show_config() -> None:
    """Show current configuration and where each value is loaded from."""
    from rich.console import Console
    from rich.table import Table

    config, metadata = get_config_with_metadata()

    console = Console()
    table = Table(title="Voicepad Configuration")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Source", style="yellow")

    for field_name in Config.model_fields:
        value = getattr(config, field_name)
        source_desc = "default"

        if metadata.per_field and field_name in metadata.per_field:
            src = metadata.per_field[field_name]
            if src.source == "defaults":
                source_desc = "default"
            elif src.source == "env":
                source_desc = "env var"
            else:
                path = src.source_path or ""
                source_desc = f"{src.source} ({path})" if path else src.source

        value_str = str(value)
        if len(value_str) > 50:
            value_str = value_str[:47] + "..."

        table.add_row(field_name, value_str, source_desc)

    console.print(table)
    _config_file_hint()


# ---------------------------------------------------------------------------
# config input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    channels: int
    sample_rate: int

    def __str__(self) -> str:
        return f"[{self.index}] {self.name}  ({self.channels}ch, {self.sample_rate}Hz)"


def _get_input_devices() -> list[AudioDevice]:
    """Return deduplicated input devices, preferring WASAPI on Windows.

    sounddevice exposes the same physical microphone multiple times — once per
    host API (MME, DirectSound, WASAPI, WDM-KS). We keep only one entry per
    device by preferring WASAPI (best quality/latency on Windows), then
    DirectSound, then MME. WDM-KS entries are kernel-level and not useful for
    normal recording, so they are always excluded.
    """

    class _DeviceInfo(TypedDict, total=False):
        name: str
        max_input_channels: int
        default_samplerate: float
        hostapi: int

    # Host API preference order: higher = preferred
    api_priority: dict[str, int] = {
        "windows wasapi": 3,
        "windows directsound": 2,
        "mme": 1,
    }

    all_devices = cast(list[_DeviceInfo], sd.query_devices())
    try:
        host_apis = sd.query_hostapis()
    except Exception:
        host_apis = []

    def _api_name(hostapi_idx: int) -> str:
        try:
            return host_apis[hostapi_idx]["name"].lower()
        except (IndexError, KeyError):
            return ""

    def _normalise_name(name: str) -> str:
        """Normalise device name for deduplication.

        MME truncates names at 31 characters, which can leave unclosed
        parentheses (e.g. 'CABLE Output (VB-Audio Virtual '). We strip
        everything from the first '(' onward so truncated and full names
        both reduce to the same root (e.g. 'cable output').
        """
        if "(" in name:
            name = name[: name.index("(")]
        return name.strip().lower()

    # Windows virtual routing devices — not real microphones, confuse users
    virtual_device_names = frozenset({
        "microsoft sound mapper - input",
        "primary sound capture driver",
    })

    # Collect all valid input devices with their API priority
    candidates: list[tuple[int, AudioDevice, int]] = []  # (priority, device, original_idx)
    for idx, dev in enumerate(all_devices):
        ch = dev.get("max_input_channels") or 0
        if ch <= 0:
            continue
        api = _api_name(dev.get("hostapi", -1))
        # Skip WDM-KS entirely — kernel-level, confusing names, not useful for dictation
        if "wdm" in api or "wdm-ks" in api:
            continue
        # Skip Windows virtual routing devices
        name = dev.get("name", f"Device {idx}")
        if name.lower().strip() in virtual_device_names:
            continue
        priority = api_priority.get(api, 0)
        rate = int(dev.get("default_samplerate") or 44100)
        candidates.append((priority, AudioDevice(index=idx, name=name, channels=ch, sample_rate=rate), idx))

    # Deduplicate: for each normalised name, keep the highest-priority API entry
    best: dict[str, tuple[int, AudioDevice]] = {}
    for priority, device, _ in candidates:
        key = _normalise_name(device.name)
        if key not in best or priority > best[key][0]:
            best[key] = (priority, device)

    # Return in original index order
    result = [device for _, device in best.values()]
    result.sort(key=lambda d: d.index)
    return result


def _get_input_device_options() -> list[tuple[str, int]]:
    if sys.platform == "linux":
        return [("System default (PipeWire / PulseAudio)", -1)]
    return [("System default", -1), *((device.name, device.index) for device in _get_input_devices())]


@config_app.command("input")
def list_input_devices() -> None:
    """List available audio input devices."""
    if sys.platform == "linux":
        configured = get_config().input_device_index
        typer.echo("Input device: system default (managed by PipeWire / PulseAudio)")
        if configured is not None:
            typer.echo(f"Configured device index {configured} is ignored on Linux.")
        typer.echo("Choose the microphone in your desktop's Sound settings.")
        _config_file_hint()
        return

    devices = _get_input_devices()
    if not devices:
        typer.secho("No audio input devices found.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    config = get_config()
    configured = config.input_device_index

    typer.echo(f"Configured device: {configured if configured is not None else 'system default'}")
    typer.echo()
    typer.echo("Available input devices:")
    typer.echo("-" * 60)
    for dev in devices:
        marker = "  ← configured" if dev.index == configured else ""
        typer.echo(f"  {dev}{marker}")
    typer.echo("-" * 60)
    _config_file_hint()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config_file_hint() -> None:
    _, metadata = get_config_with_metadata()
    if metadata.per_field:
        src = next(iter(metadata.per_field.values()))
        if src.source != "defaults" and src.source_path:
            typer.echo(f"\nConfig file: {src.source_path}")
            return
    typer.echo("\nNo config file found — using defaults.")
    typer.echo("Create voicepad.yaml in the current directory to customise settings.")
