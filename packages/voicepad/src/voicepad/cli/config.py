"""Configuration management commands."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TypedDict, cast

import sounddevice as sd
import typer
from voicepad_core import get_config, get_config_with_metadata

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

    for field_name in config.model_fields:
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
    class _DeviceInfo(TypedDict, total=False):
        name: str
        max_input_channels: int
        default_samplerate: float

    devices: list[AudioDevice] = []
    for idx, dev in enumerate(cast(list[_DeviceInfo], sd.query_devices())):
        ch = dev.get("max_input_channels") or 0
        if ch <= 0:
            continue
        name = dev.get("name", f"Device {idx}")
        rate = int(dev.get("default_samplerate") or 44100)
        devices.append(AudioDevice(index=idx, name=name, channels=ch, sample_rate=rate))
    return devices


@config_app.command("input")
def list_input_devices() -> None:
    """List available audio input devices."""
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
