from __future__ import annotations

import json

import typer

from voicepad.config import AppConfig, config_path, load_config, save_config

config_app = typer.Typer(help="Inspect or initialize strict VoicePad configuration.")


@config_app.command("show")
def show_config() -> None:
    """Print the effective application configuration."""
    config = load_config()
    payload = config.model_dump(mode="json")
    payload.setdefault("input_device_index", None)
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@config_app.command("path")
def show_path() -> None:
    """Print the application configuration path."""
    typer.echo(config_path())


@config_app.command("init")
def initialize_config() -> None:
    """Create the configuration, refusing to overwrite an existing file."""
    path = config_path()
    if path.exists():
        typer.secho(f"Configuration already exists: {path}", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(1)
    saved = save_config(AppConfig(), path)
    typer.echo(saved)
