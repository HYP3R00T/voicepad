"""VoicePad — local dictation with Whisper."""

from voicepad.main import app


def main() -> None:
    """Entry point — delegates to the Typer app."""
    app()


__all__ = ["main"]
