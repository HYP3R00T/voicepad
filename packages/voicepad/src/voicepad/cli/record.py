from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Annotated, Protocol

import typer
from voicepad_core.audio import CaptureFailure, MicrophoneStream

from voicepad.config import load_config
from voicepad.output import persist_markdown
from voicepad.runtime import ApplicationRuntime
from voicepad.tui.utils.clipboard import copy_to_clipboard

record_app = typer.Typer(help="Record canonical audio and transcribe it with the resident NVIDIA deployment.")
logger = logging.getLogger(__name__)


class CaptureHealth(Protocol):
    @property
    def capture_failures(self) -> tuple[CaptureFailure, ...]: ...


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
        try:
            assert microphone is not None
            _wait_for_stop(duration, microphone)
        except KeyboardInterrupt:
            typer.echo("\nStop requested; finalizing audio…")
        if no_transcribe:
            assert microphone is not None
            artifact = microphone.stop()
            if microphone.capture_failures:
                typer.secho(
                    f"Partial WAV preserved after capture failure: {artifact.path}",
                    fg=typer.colors.YELLOW,
                    err=True,
                )
                raise typer.Exit(2)
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
        logger.exception("CLI recording failed")
        if microphone is not None and microphone.is_recording:
            try:
                microphone.stop()
            except Exception:
                logger.exception("CLI cleanup could not finalize microphone audio")
        if job is not None:
            job.cancel()
            try:
                job.finish(timeout=30)
            except Exception:
                logger.exception("CLI cleanup could not release the recording job")
        typer.secho(f"VoicePad failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from error
    finally:
        runtime.close()


def _wait_for_stop(duration: float | None, microphone: CaptureHealth) -> None:
    stopped = threading.Event()
    if duration is not None:
        deadline = time.monotonic() + duration
        while not microphone.capture_failures:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            stopped.wait(min(0.05, remaining))
        logger.error("CLI detected failed capture before its requested duration elapsed")
        return
    typer.echo("Type q and press Enter to stop.")

    def wait() -> None:
        while input().strip().casefold() != "q":
            pass
        stopped.set()

    threading.Thread(target=wait, daemon=True).start()
    while not stopped.wait(0.05):
        if microphone.capture_failures:
            logger.error("CLI detected failed capture while waiting for an interactive stop")
            return


def _recording_path(directory: Path, prefix: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = directory / f"{prefix}_{stamp}.wav"
    if candidate.exists():
        raise FileExistsError(f"Recording destination already exists: {candidate}")
    return candidate
