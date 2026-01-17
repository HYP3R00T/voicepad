import typer
from utilityhub_config import load_settings

from voicepad.audio.scanner import get_device_by_index, record_voice_continuous
from voicepad.config import Config
from voicepad.ui.voicepad_ui import VoicepadUI

app = typer.Typer(invoke_without_command=True)


def ui():
    app = VoicepadUI()
    app.run()


@app.command()
def cli():
    settings, metadata = load_settings(Config, app_name="voicepad", env_prefix="VP")
    print(settings.timeout)

    print(get_device_by_index(1))
    record_voice_continuous(1)


@app.callback()
def _default(ctx: typer.Context):
    # When invoked without a subcommand, run the `ui` command.
    if ctx.invoked_subcommand is None:
        ui()


def main():
    app()
