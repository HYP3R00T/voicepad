"""VoicePad entry point.

Running `voicepad` with no arguments launches the TUI.
Subcommands (`voicepad record ...`, `voicepad config ...`) use the CLI.
"""

from __future__ import annotations

import logging

import typer

from voicepad.cli import config_app, record_app

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="VoicePad — local dictation with Whisper.\n\nRun without arguments to open the TUI.",
    invoke_without_command=True,  # allows the callback to run when no subcommand given
    no_args_is_help=False,
)

app.add_typer(config_app, name="config", help="Configuration management")
app.add_typer(record_app, name="record", help="Recording commands (CLI mode)")


@app.command("toggle")
def toggle_recording() -> None:
    """Toggle recording in a running VoicePad TUI."""
    from voicepad.tui.control import run_toggle_command

    exit_code = run_toggle_command()
    if exit_code:
        raise typer.Exit(code=exit_code)


@app.callback()
def main(ctx: typer.Context) -> None:
    """Launch the TUI when called with no subcommand."""
    if ctx.invoked_subcommand is None:
        from voicepad.tui.app import run

        run()
