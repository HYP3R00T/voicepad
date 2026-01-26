import typer

from voicepad.config.cli import config_app
from voicepad.ui.cli import start_ui
from voicepad.voice.cli import voice_app

app = typer.Typer(invoke_without_command=True)

# Register voice sub-commands
app.add_typer(voice_app, name="voice", help="Voice recording and transcription commands")

# Register config sub-commands
app.add_typer(config_app, name="config", help="Configuration management commands")


@app.callback()
def _default(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        # Run UI by default when no subcommand is provided
        start_ui()


def main() -> None:
    """Entry point for the CLI application."""
    app()
