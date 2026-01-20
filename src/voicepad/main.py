import typer

from voicepad.audio.cli import audio_app
from voicepad.ui.voicepad_ui import VoicepadUI

app = typer.Typer(invoke_without_command=True)

# Register audio sub-commands under 'cli'
app.add_typer(audio_app, name="cli", help="CLI commands for testing audio features")


def ui() -> None:
    """Create and run the Voicepad UI application."""
    voicepad_app = VoicepadUI()
    voicepad_app.run()


@app.callback()
def _default(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        ui()


def main() -> None:
    """Entry point for the CLI application."""
    app()
