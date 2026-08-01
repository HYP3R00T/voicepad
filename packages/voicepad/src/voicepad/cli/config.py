from __future__ import annotations

import json
from dataclasses import asdict

import typer

from voicepad.config import config_path, load_config, save_config

config_app = typer.Typer(help="Inspect or initialize strict VoicePad configuration.")


@config_app.command("show")
def show_config() -> None:
    """Print the effective schema-1 application configuration."""
    config = load_config()
    payload = asdict(config)
    for key in ("recordings_path", "markdown_path", "artifact_cache_path"):
        payload[key] = str(payload[key])
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@config_app.command("path")
def show_path() -> None:
    """Print the application configuration path."""
    typer.echo(config_path())


@config_app.command("init")
def initialize_config() -> None:
    """Create schema-1 configuration, refusing to overwrite an existing file."""
    path = config_path()
    if path.exists():
        typer.secho(f"Configuration already exists: {path}", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(1)
    saved = save_config(load_config(), path)
    typer.echo(saved)
