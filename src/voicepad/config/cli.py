"""CLI commands for configuration management."""

import logging
from pathlib import Path
from typing import Annotated

import typer

from voicepad.config.settings import get_config
from voicepad.system_utils import check_gpu_capabilities, print_system_status, recommend_faster_whisper_model

logger = logging.getLogger(__name__)

config_app = typer.Typer(help="Configuration management commands")


@config_app.command()
def init() -> None:
    """Initialize and verify system capabilities for first-time setup."""
    print_system_status()


@config_app.command()
def show() -> None:
    """Display current configuration."""
    config = get_config()

    typer.echo("Current configuration:")
    typer.echo(f"  Recordings path: {config.recordings_path}")
    typer.echo(f"  Markdown path: {config.markdown_path}")
    typer.echo("\nTranscription settings:")
    typer.echo(f"  Model: {config.transcription.model}")
    typer.echo(f"  Device: {config.transcription.device}")
    typer.echo(f"  Compute type: {config.transcription.compute_type}")
    typer.echo(f"  Language: {config.transcription.language or 'auto-detect'}")


@config_app.command()
def verify() -> None:
    """Verify configuration and directories."""
    config = get_config()

    typer.echo("Verifying configuration...")

    # Check recordings path
    if config.recordings_path.exists():
        typer.echo(f"✓ Recordings path exists: {config.recordings_path}")
    else:
        typer.echo(f"✗ Recordings path not found: {config.recordings_path}")
        try:
            config.recordings_path.mkdir(parents=True, exist_ok=True)
            typer.echo(f"✓ Created recordings path: {config.recordings_path}")
        except Exception as e:
            typer.echo(f"✗ Failed to create recordings path: {e}", err=True)

    # Check markdown path
    if config.markdown_path.exists():
        typer.echo(f"✓ Markdown path exists: {config.markdown_path}")
    else:
        typer.echo(f"✗ Markdown path not found: {config.markdown_path}")
        try:
            config.markdown_path.mkdir(parents=True, exist_ok=True)
            typer.echo(f"✓ Created markdown path: {config.markdown_path}")
        except Exception as e:
            typer.echo(f"✗ Failed to create markdown path: {e}", err=True)

    typer.echo("✓ Configuration verified!")


@config_app.command()
def set_model(
    model: Annotated[
        str,
        typer.Argument(
            help="Model to use (tiny, base, small, medium, large-v2, large-v3, turbo, distil-large-v3, auto)",
        ),
    ],
) -> None:
    """Set the default transcription model.

    Use 'auto' to automatically detect the best model based on your GPU.
    """
    import yaml

    config_file = Path.home() / ".config" / "voicepad" / "voicepad.yaml"
    config_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config or create new
    if config_file.exists():
        with open(config_file) as f:
            config_data = yaml.safe_load(f) or {}
    else:
        config_data = {}

    # Update transcription model
    if "transcription" not in config_data:
        config_data["transcription"] = {}
    config_data["transcription"]["model"] = model

    # Save config
    with open(config_file, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False)

    typer.secho(f"✓ Set default model to: {model}", fg=typer.colors.GREEN)

    if model == "auto":
        gpu_info = check_gpu_capabilities()
        rec = recommend_faster_whisper_model(gpu_info.device_type, gpu_info.total_memory_gb)
        typer.echo(f"  (Auto-detection will use: {rec.model_size} based on your {gpu_info.device_type})")


@config_app.command()
def set_device(
    device: Annotated[
        str,
        typer.Argument(help="Device to use (cuda, cpu, auto)"),
    ],
) -> None:
    """Set the default transcription device."""
    import yaml

    config_file = Path.home() / ".config" / "voicepad" / "voicepad.yaml"
    config_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config or create new
    if config_file.exists():
        with open(config_file) as f:
            config_data = yaml.safe_load(f) or {}
    else:
        config_data = {}

    # Update transcription device
    if "transcription" not in config_data:
        config_data["transcription"] = {}
    config_data["transcription"]["device"] = device

    # Save config
    with open(config_file, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False)

    typer.secho(f"✓ Set default device to: {device}", fg=typer.colors.GREEN)
