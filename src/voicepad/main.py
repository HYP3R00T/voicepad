import typer

from voicepad.ui.voicepad_ui import VoicepadUI

app = typer.Typer(invoke_without_command=True)


def ui():
    app = VoicepadUI()
    app.run()


@app.command()
def hello(name: str):
    print(f"Hello {name}")


@app.callback()
def _default(ctx: typer.Context):
    # When invoked without a subcommand, run the `ui` command.
    if ctx.invoked_subcommand is None:
        ui()


def main():
    app()
