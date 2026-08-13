from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Annotated

import typer
from voicepad_core.audio import MicrophoneStream

from voicepad.config import load_config
from voicepad.output import persist_markdown
from voicepad.runtime import ApplicationRuntime
from voicepad.tui.utils.clipboard import copy_to_clipboard

record_app = typer.Typer(help="Record canonical audio and transcribe it with the resident NVIDIA deployment.")
logger = logging.getLogger(__name__)


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
            microphone = runtime.start_capture()
        else:
            typer.echo("Preparing and warming the NVIDIA deployment…")
            active = runtime.activate()
            typer.echo(f"Ready: {active.device_name} / {active.definition.precision.value}")
            microphone, job = runtime.start_recording()

        typer.secho("Recording…", fg=typer.colors.RED, bold=True)
        try:
            assert microphone is not None
            _wait_for_stop(duration, microphone)
        except KeyboardInterrupt:
            typer.echo("\nStop requested; finalizing audio…")
        if no_transcribe:
            assert microphone is not None
            artifact = runtime.stop_capture(microphone)
            if microphone.capture_error is not None:
                runtime.end_recording(outcome="incomplete")
                typer.secho(f"Partial WAV preserved: {artifact.path}", fg=typer.colors.YELLOW, err=True)
                raise typer.Exit(2)
            runtime.end_recording(outcome="completed")
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
            runtime.end_recording(outcome="incomplete")
            typer.secho("Transcription is incomplete; it was not copied.", fg=typer.colors.YELLOW, err=True)
            raise typer.Exit(2)
        runtime.end_recording(outcome="completed")
    except typer.Exit:
        raise
    except Exception as error:
        if microphone is not None and microphone.is_recording:
            try:
                microphone.stop()
            except Exception as cleanup_error:
                logger.exception("Microphone cleanup failed after record command failure")
                error.add_note(f"Microphone cleanup also failed: {cleanup_error}")
        if job is not None:
            job.cancel()
            try:
                job.finish(timeout=30)
            except Exception as cleanup_error:
                logger.exception("Transcription cleanup failed after record command failure")
                error.add_note(f"Transcription cleanup also failed: {cleanup_error}")
        runtime.end_recording(outcome="failed", error=error)
        typer.secho(f"VoicePad failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from error
    finally:
        primary_error = sys.exception()
        try:
            runtime.close()
        except Exception as cleanup_error:
            logger.exception("Runtime cleanup failed after record command")
            if primary_error is None:
                raise
            (primary_error.__cause__ or primary_error).add_note(f"Runtime cleanup also failed: {cleanup_error}")


def _wait_for_stop(duration: float | None, microphone: MicrophoneStream) -> None:
    deadline = None if duration is None else time.monotonic() + duration
    if deadline is not None:
        while microphone.capture_error is None and time.monotonic() < deadline:
            time.sleep(0.05)
        return
    typer.echo("Type q and press Enter to stop.")
    stopped = threading.Event()

    def wait() -> None:
        while input().strip().casefold() != "q":
            pass
        stopped.set()

    threading.Thread(target=wait, daemon=True).start()
    while not stopped.wait(0.05):
        if microphone.capture_error is not None:
            logger.error("Capture failed while waiting for CLI stop")
            return
