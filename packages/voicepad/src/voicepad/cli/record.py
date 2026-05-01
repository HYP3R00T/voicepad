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
    ensure_model_downloaded,
    get_config,
    get_or_load_model,
    model_downloaded,
    transcribe_buffer,
    transcribe_file,
)
from voicepad_core.transcription import BEAM_SIZE, COMPUTE_TYPE, DEVICE, LANGUAGE

logger = logging.getLogger(__name__)

record_app = typer.Typer(help="Audio recording commands")


# ---------------------------------------------------------------------------
# record start
# ---------------------------------------------------------------------------


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
    no_transcribe: bool = typer.Option(
        False,
        "--no-transcribe",
        help="Skip transcription after recording",
    ),
    no_save: bool = typer.Option(
        False,
        "--no-save",
        help="Do not save WAV to disk (transcribe from memory only)",
    ),
) -> None:
    """Record audio from your microphone and transcribe it.

    \b
    Stop options:
      - Fixed duration:  --duration 10
      - Manual stop:     type  q  then press Enter

    \b
    Examples:
      voicepad record start
      voicepad record start --duration 5
      voicepad record start --no-save
    """
    config = get_config()

    # --- Step 1: Ensure model is downloaded ---
    # Check before opening the mic so the user knows what's happening.
    # On first run this downloads the model (~500MB–1.5GB depending on size).
    # On subsequent runs the cache check is instant.
    if not no_transcribe:
        model_name = config.transcription_model
        if not model_downloaded(model_name, config):
            typer.echo()
            typer.secho(
                f"[↓] Model '{model_name}' not found locally — downloading now.",
                fg=typer.colors.CYAN,
            )
            typer.echo("    This only happens once. Subsequent runs start immediately.")
            typer.echo()
            try:
                ensure_model_downloaded(model_name, config)
                typer.secho(f"    [OK] '{model_name}' downloaded.", fg=typer.colors.GREEN)
            except TranscriptionError as e:
                typer.secho(f"[ERROR] Download failed: {e}", fg=typer.colors.RED, err=True)
                raise typer.Exit(1) from e

        # --- Step 2: Load model into VRAM ---
        # Model is local — load it now so transcription is instant after recording.
        typer.echo(f"[~] Loading '{model_name}' into memory...")
        try:
            _, actual_device, actual_compute, fallback = get_or_load_model(config)
            if fallback:
                typer.secho(
                    f"    [!] CUDA not available — using CPU ({actual_compute})",
                    fg=typer.colors.YELLOW,
                )
            else:
                typer.secho(
                    f"    [OK] Ready on {actual_device} ({actual_compute})",
                    fg=typer.colors.GREEN,
                )
        except TranscriptionError as e:
            typer.secho(f"[ERROR] Could not load model: {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from e

    # --- Start recording ---
    recorder = AudioRecorder(config)
    try:
        recorder.start()
    except AudioRecorderError as e:
        typer.secho(f"[ERROR] {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e

    typer.echo()
    typer.secho("[REC] Recording...", fg=typer.colors.YELLOW, bold=True)
    typer.echo(
        f"      device : {config.input_device_index if config.input_device_index is not None else 'system default'}"
    )
    typer.echo()

    # --- Wait for stop ---
    if duration is not None:
        typer.echo(f"      recording for {duration:.1f}s...")
        time.sleep(duration)
    else:
        typer.echo("      speak now — type  q  then Enter to stop")
        stop_event = threading.Event()
        listener = threading.Thread(
            target=_wait_for_quit,
            args=(stop_event,),
            daemon=True,
        )
        listener.start()
        while not stop_event.is_set():
            time.sleep(0.05)

    # --- Stop and collect audio ---
    typer.echo()
    try:
        audio = recorder.stop()
    except AudioRecorderError as e:
        typer.secho(f"[ERROR] {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e

    captured_s = len(audio) / 16000
    typer.secho(f"[OK] Captured {captured_s:.2f}s", fg=typer.colors.GREEN)

    if captured_s < 0.5:
        typer.secho("[SKIP] Too short to transcribe (< 0.5s)", fg=typer.colors.YELLOW)
        return

    # --- Save WAV ---
    wav_path: Path | None = None
    if not no_save:
        wav_path = recorder.make_wav_path(prefix)
        try:
            recorder.save_wav(audio, wav_path)
            typer.echo(f"      saved  : {wav_path}")
        except Exception as e:
            typer.secho(f"[WARN] Could not save WAV: {e}", fg=typer.colors.YELLOW)
            wav_path = None

    # --- Transcribe ---
    if no_transcribe:
        return

    typer.echo()
    typer.secho("[*] Transcribing...", fg=typer.colors.CYAN)

    try:
        # Model is already in cache — this call returns immediately from cache
        if wav_path and wav_path.exists():
            result = transcribe_file(wav_path, config)
        else:
            result = transcribe_buffer(audio, config)
    except AudioTooShortError as e:
        typer.secho(f"[SKIP] {e}", fg=typer.colors.YELLOW)
        return
    except TranscriptionError as e:
        typer.secho(f"[ERROR] {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e

    _print_result(result)

    # Save markdown alongside the WAV
    if wav_path and wav_path.exists():
        md_path = config.markdown_path / f"{wav_path.stem}.md"
        try:
            config.markdown_path.mkdir(parents=True, exist_ok=True)
            md_path.write_text(_format_markdown(wav_path, result, config.transcription_model), encoding="utf-8")
            typer.echo(f"      markdown: {md_path}")
        except Exception as e:
            typer.secho(f"[WARN] Could not save markdown: {e}", fg=typer.colors.YELLOW)


# ---------------------------------------------------------------------------
# record info
# ---------------------------------------------------------------------------


@record_app.command("info")
def show_info() -> None:
    """Show current recording and transcription configuration."""
    config = get_config()
    typer.echo()
    typer.echo("  Recording")
    typer.echo("  " + "─" * 40)
    typer.echo(
        f"  input device : {config.input_device_index if config.input_device_index is not None else 'system default'}"
    )
    typer.echo(f"  recordings   : {config.recordings_path}")
    typer.echo(f"  markdown     : {config.markdown_path}")
    typer.echo(f"  prefix       : {config.recording_prefix}")
    typer.echo()
    typer.echo("  Transcription  (hardcoded constants)")
    typer.echo("  " + "─" * 40)
    typer.echo(f"  model        : {config.transcription_model}")
    typer.echo(f"  device       : {DEVICE}")
    typer.echo(f"  compute type : {COMPUTE_TYPE}")
    typer.echo(f"  beam size    : {BEAM_SIZE}")
    typer.echo(f"  language     : {LANGUAGE or 'auto-detect'}")
    typer.echo()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_for_quit(stop_event: threading.Event) -> None:
    """Block until the user types 'q' + Enter."""
    while not stop_event.is_set():
        try:
            line = sys.stdin.readline()
            if line == "":
                # EOF reached (e.g. stdin closed or redirected)
                stop_event.set()
                break
            if line.strip().lower() == "q":
                stop_event.set()
                break
        except (EOFError, OSError):
            stop_event.set()
            break


def _print_result(result) -> None:
    typer.echo()
    typer.secho("┌─ Transcription " + "─" * 34, fg=typer.colors.CYAN)
    typer.echo(result.text or "(no speech detected)")
    typer.secho("└" + "─" * 50, fg=typer.colors.CYAN)
    typer.echo()
    typer.echo(f"  device   : {result.device} ({result.compute_type})")
    typer.echo(f"  language : {result.language} ({result.language_probability * 100:.0f}% confidence)")
    typer.echo(f"  audio    : {result.duration_s:.1f}s")
    typer.echo(f"  latency  : {result.latency_ms:.0f}ms")
    if result.fallback_to_cpu:
        typer.secho("  [!] CUDA not available — ran on CPU", fg=typer.colors.YELLOW)


def _format_markdown(wav_path: Path, result, model_name: str = "") -> str:
    model_str = (
        f"{model_name} · {result.device} / {result.compute_type}"
        if model_name
        else f"{result.device} / {result.compute_type}"
    )
    lines = [
        "---",
        f"file: {wav_path.name}",
        "transcriptions:",
        "  - n: 1",
        f"    model: {model_str}",
        f"    language: {result.language} ({result.language_probability * 100:.1f}%)",
        f"    duration: {result.duration_s:.1f}s",
        f"    latency: {result.latency_ms:.0f}ms",
    ]
    if result.fallback_to_cpu:
        lines.append("    fallback: cpu")
    lines += [
        "---",
        "",
        "## Transcription 1",
        "",
        result.text or "*(no speech detected)*",
        "",
    ]
    return "\n".join(lines) + "\n"
