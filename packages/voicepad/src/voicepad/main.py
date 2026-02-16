import logging

import typer

from voicepad.cli import doctor_app, voice_app

logger = logging.getLogger(__name__)

app = typer.Typer(
    invoke_without_command=True,
    help="Voicepad - Voice recording and audio processing application",
)

app.add_typer(voice_app, name="voice", help="Voice recording and audio device commands")
app.add_typer(doctor_app, name="doctor", help="System health and configuration checks")


def main() -> None:
    app()
