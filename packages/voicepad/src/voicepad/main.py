from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from voicepad.cli import config_app, record_app
from voicepad.config import load_config
from voicepad.output import persist_markdown
from voicepad.runtime import ApplicationRuntime
from voicepad.tui.utils.clipboard import copy_to_clipboard

app = typer.Typer(
    help="VoicePad — private local NVIDIA-accelerated dictation.",
    invoke_without_command=True,
    no_args_is_help=False,
)
app.add_typer(config_app, name="config")
app.add_typer(record_app, name="record")


@app.command("prepare")
def prepare() -> None:
    """Download, verify, load, and warm the selected deployment."""
    runtime = ApplicationRuntime(load_config())
    try:
        active = runtime.activate()
        typer.echo(f"Ready: {active.definition.id} on {active.device_name}")
    except Exception as error:
        typer.secho(f"Preparation failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from error
    finally:
        runtime.close()


@app.command("transcribe")
def transcribe_file(
    audio: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    no_copy: Annotated[bool, typer.Option("--no-copy", help="Do not copy a complete result.")] = False,
) -> None:
    """Transcribe an existing immutable audio file."""
    config = load_config()
    runtime = ApplicationRuntime(config)
    try:
        runtime.activate()
        result = runtime.transcribe_file(audio)
        markdown = persist_markdown(audio, result, config.markdown_path)
        typer.echo(result.text)
        typer.echo(f"Markdown: {markdown}")
        if result.complete and result.text and config.copy_complete_text and not no_copy:
            copy_to_clipboard(result.text)
        if not result.complete:
            typer.secho("Result is incomplete; inspect its Markdown metadata.", fg=typer.colors.YELLOW, err=True)
            raise typer.Exit(2)
    except typer.Exit:
        raise
    except Exception as error:
        typer.secho(f"Transcription failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from error
    finally:
        runtime.close()


@app.command("toggle")
def toggle_recording() -> None:
    """Toggle recording in a running VoicePad TUI."""
    from voicepad.tui.control import run_toggle_command

    exit_code = run_toggle_command()
    if exit_code:
        raise typer.Exit(code=exit_code)


@app.callback()
def main(ctx: typer.Context) -> None:
    """Launch the TUI when no command is supplied."""
    if ctx.invoked_subcommand is None:
        from voicepad.tui.app import run

        run()
