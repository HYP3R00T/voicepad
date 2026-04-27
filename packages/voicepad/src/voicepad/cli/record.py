"""Recording commands for the voicepad CLI."""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

import typer
from voicepad_core import (
    AudioRecorder,
    AudioRecorderError,
    AudioTooShortError,
    TranscriptionError,
    get_config,
    transcribe_file,
)
from voicepad_core.transcription import BEAM_SIZE, COMPUTE_TYPE, DEVICE, LANGUAGE

logger = logging.getLogger(__name__)

record_app = typer.Typer(help="Audio recording commands")


def _wait_for_quit(stop_event: threading.Event) -> None:
    """Block until the user types 'q' + Enter. Sets stop_event when triggered."""
    while not stop_event.is_set():
        try:
            line = sys.stdin.readline().strip().lower()
            if line == "q":
                stop_event.set()
                break
        except (EOFError, OSError):
            stop_event.set()
            break


@record_app.command("start")
def start_recording(
    prefix: str | None = typer.Option(
        None,
        "--prefix",
        "-p",
        help="Filename prefix (overrides config)",
    ),
    duration: float | None = typer.Option(
        None,
        "--duration",
        "-d",
        help="Fixed recording duration in seconds",
        min=0.1,
    ),
    transcribe: bool = typer.Option(
        True,
        "--transcribe/--no-transcribe",
        help="Transcribe after recording (default: on)",
    ),
    save: bool = typer.Option(
        True,
        "--save/--no-save",
        help="Save WAV file to disk (default: on)",
    ),
) -> None:
    """Record audio and optionally transcribe it.

    Press q + Enter to stop, or use --duration for a fixed-length recording.
    """
    config = get_config()
    recorder = AudioRecorder(config)

    # --- Start ---
    try:
        recorder.start()
    except AudioRecorderError as e:
        typer.secho(f"[ERROR] {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e

    wav_path = recorder.generate_wav_path(prefix)
    typer.secho("[REC] Recording...", fg=typer.colors.YELLOW)
    typer.echo(f"      device : {config.input_device_index or 'default'}")
    typer.echo(f"      output : {wav_path}")
    typer.echo()

    # --- Wait ---
    if duration is not None:
        typer.echo(f"      stopping in {duration:.1f}s")
        time.sleep(duration)
    else:
        typer.echo("      type q + Enter to stop")
        stop_event = threading.Event()
        t = threading.Thread(target=_wait_for_quit, args=(stop_event,), daemon=True)
        t.start()
        while recorder.is_recording() and not stop_event.is_set():
            time.sleep(0.05)

    # --- Stop ---
    typer.echo("\n[!] Stopping...")
    try:
        audio = recorder.stop()
    except AudioRecorderError as e:
        typer.secho(f"[ERROR] {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e

    duration_s = len(audio) / 16000
    typer.secho(f"[OK] Captured {duration_s:.2f}s of audio", fg=typer.colors.GREEN)

    # --- Save WAV ---
    if save:
        try:
            recorder.save_wav(audio, wav_path)
            typer.echo(f"     saved  : {wav_path}")
        except Exception as e:
            typer.secho(f"[WARN] Could not save WAV: {e}", fg=typer.colors.YELLOW)

    # --- Transcribe ---
    if not transcribe:
        return

    if not save:
        # Transcribe from buffer directly
        _transcribe_buffer(audio, config)
    else:
        # Transcribe from saved file (consistent path for markdown output)
        _transcribe_wav(wav_path, config)


def _transcribe_buffer(audio, config) -> None:
    """Transcribe audio from a numpy array and print the result."""
    from voicepad_core import transcribe_buffer

    typer.echo()
    typer.secho("[*] Transcribing...", fg=typer.colors.CYAN)
    try:
        result = transcribe_buffer(audio, config)
        _print_result(result)
    except AudioTooShortError as e:
        typer.secho(f"[SKIP] {e}", fg=typer.colors.YELLOW)
    except TranscriptionError as e:
        typer.secho(f"[ERROR] Transcription failed: {e}", fg=typer.colors.RED, err=True)


def _transcribe_wav(wav_path: Path, config) -> None:
    """Transcribe a WAV file and save markdown alongside it."""
    typer.echo()
    typer.secho("[*] Transcribing...", fg=typer.colors.CYAN)
    try:
        result = transcribe_file(wav_path, config)
        _print_result(result)

        # Save markdown
        md_path = config.markdown_path / f"{wav_path.stem}.md"
        config.markdown_path.mkdir(parents=True, exist_ok=True)
        md_path.write_text(_format_markdown(wav_path, result), encoding="utf-8")
        typer.echo(f"     markdown: {md_path}")

    except AudioTooShortError as e:
        typer.secho(f"[SKIP] {e}", fg=typer.colors.YELLOW)
    except TranscriptionError as e:
        typer.secho(f"[ERROR] Transcription failed: {e}", fg=typer.colors.RED, err=True)


def _print_result(result) -> None:
    """Print a TranscriptionResult to the terminal."""
    typer.secho("[OK] Transcription complete", fg=typer.colors.GREEN)
    typer.echo(f"     device  : {result.device} ({result.compute_type})")
    typer.echo(f"     language: {result.language} ({result.language_probability * 100:.0f}%)")
    typer.echo(f"     duration: {result.duration_s:.1f}s")
    typer.echo(f"     latency : {result.latency_ms:.0f}ms")
    if result.fallback_to_cpu:
        typer.secho("     [!] CUDA requested but fell back to CPU", fg=typer.colors.YELLOW)
    typer.echo()
    typer.secho("--- Transcription ---", fg=typer.colors.CYAN)
    typer.echo(result.text or "(no speech detected)")
    typer.secho("---------------------", fg=typer.colors.CYAN)


def _format_markdown(wav_path: Path, result) -> str:
    """Format a TranscriptionResult as a markdown document."""
    lines = [
        "# Transcription",
        "",
        f"**File:** {wav_path.name}",
        f"**Model:** {result.device} / {result.compute_type}",
        f"**Language:** {result.language} ({result.language_probability * 100:.1f}%)",
        f"**Duration:** {result.duration_s:.1f}s",
        f"**Latency:** {result.latency_ms:.0f}ms",
        "",
    ]
    if result.fallback_to_cpu:
        lines += ["> **Note:** CUDA requested but fell back to CPU.", ""]

    lines += ["---", "", "## Text", ""]
    for seg in result.segments:
        lines.append(f"[{seg.start:.1f}s → {seg.end:.1f}s] {seg.text}")

    return "\n".join(lines) + "\n"


@record_app.command("info")
def show_info() -> None:
    """Show current recording configuration."""
    config = get_config()
    typer.echo("Recording configuration")
    typer.echo("=" * 50)
    typer.echo(f"  input device : {config.input_device_index or 'system default'}")
    typer.echo(f"  recordings   : {config.recordings_path}")
    typer.echo(f"  markdown     : {config.markdown_path}")
    typer.echo(f"  prefix       : {config.recording_prefix}")
    typer.echo(f"  model        : {config.transcription_model}")
    typer.echo(f"  device       : {DEVICE}  (constant)")
    typer.echo(f"  compute type : {COMPUTE_TYPE}  (constant)")
    typer.echo(f"  beam size    : {BEAM_SIZE}  (constant)")
    typer.echo(f"  language     : {LANGUAGE or 'auto-detect'}  (constant)")
    typer.echo("=" * 50)
