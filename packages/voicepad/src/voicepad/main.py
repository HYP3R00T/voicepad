import logging

import typer

from voicepad.cli import config_app, record_app

logger = logging.getLogger(__name__)

app = typer.Typer(
    invoke_without_command=True,
    help="Voicepad - Voice recording and audio processing application",
)

app.add_typer(config_app, name="config", help="Configuration management commands")
app.add_typer(record_app, name="record", help="Audio recording commands")


def main() -> None:
    app()
