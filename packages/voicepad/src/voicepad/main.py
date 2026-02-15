import logging

import typer

from voicepad.cli import voice_app

logger = logging.getLogger(__name__)

app = typer.Typer(
    invoke_without_command=True,
    help="Voicepad - Voice recording and audio processing application",
)

app.add_typer(voice_app, name="voice", help="Voice recording and audio device commands")


def main() -> None:
    app()
