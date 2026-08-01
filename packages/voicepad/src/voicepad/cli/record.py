from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path
from typing import Annotated

import typer
from voicepad_core.audio import MicrophoneStream

from voicepad.config import load_config
from voicepad.output import persist_markdown
from voicepad.runtime import ApplicationRuntime
from voicepad.tui.utils.clipboard import copy_to_clipboard

record_app = typer.Typer(help="Record canonical audio and transcribe it with the resident NVIDIA deployment.")


@record_app.command("start")
def start_recording(
    duration: Annotated[
        float | None,
        typer.Option("--duration", "-d", min=0.1, help="Stop automatically after this many seconds."),
    ] = None,
    no_transcribe: Annotated[
        bool,
        typer.Option("--no-transcribe", help="Persist the WAV without loading the transcription deployment."),
    ] = False,
) -> None:
    """Record from the shared Linux microphone and optionally transcribe."""
    config = load_config()
    runtime = ApplicationRuntime(config)
    microphone: MicrophoneStream | None = None
    job = None
    try:
        if no_transcribe:
            config.recordings_path.mkdir(parents=True, exist_ok=True)
            path = _recording_path(config.recordings_path, config.recording_prefix)
            microphone = MicrophoneStream(path, device_index=config.input_device_index)
            microphone.start()
        else:
            typer.echo("Preparing and warming the NVIDIA deployment…")
            active = runtime.activate()
            typer.echo(f"Ready: {active.device_name} / {active.definition.precision.value}")
            microphone, job = runtime.start_recording()

        typer.secho("Recording…", fg=typer.colors.RED, bold=True)
        _wait_for_stop(duration)
        if no_transcribe:
            assert microphone is not None
            artifact = microphone.stop()
            typer.echo(f"Saved WAV: {artifact.path}")
            return

        assert microphone is not None and job is not None
        artifact, result = runtime.stop_recording(microphone, job)
        markdown = persist_markdown(artifact.path, result, config.markdown_path)
        typer.echo(result.text)
        typer.echo(f"WAV: {artifact.path}")
        typer.echo(f"Markdown: {markdown}")
        if result.complete and result.text and config.copy_complete_text:
            if copy_to_clipboard(result.text):
                typer.echo("Copied complete transcription to clipboard.")
        elif not result.complete:
            typer.secho("Transcription is incomplete; it was not copied.", fg=typer.colors.YELLOW, err=True)
            raise typer.Exit(2)
    except typer.Exit:
        raise
    except Exception as error:
        if microphone is not None and microphone.is_recording:
            with contextlib.suppress(Exception):
                microphone.stop()
        typer.secho(f"VoicePad failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from error
    finally:
        runtime.close()


def _wait_for_stop(duration: float | None) -> None:
    if duration is not None:
        time.sleep(duration)
        return
    typer.echo("Type q and press Enter to stop.")
    stopped = threading.Event()

    def wait() -> None:
        while input().strip().casefold() != "q":
            pass
        stopped.set()

    threading.Thread(target=wait, daemon=True).start()
    while not stopped.wait(0.05):
        pass


def _recording_path(directory: Path, prefix: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = directory / f"{prefix}_{stamp}.wav"
    if candidate.exists():
        raise FileExistsError(f"Recording destination already exists: {candidate}")
    return candidate
