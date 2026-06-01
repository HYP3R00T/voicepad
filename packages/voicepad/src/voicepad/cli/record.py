"""Recording commands for the voicepad CLI."""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Annotated

import typer
from voicepad_core import (
    AudioPreProcessor,
    AudioTooShortError,
    MicrophoneStream,
    TranscriptionError,
    _model_cache,
    ensure_model_downloaded,
    get_config,
    load_model,
    model_downloaded,
    transcribe,
)
from voicepad_core.inference.constants import BEAM_SIZE, COMPUTE_TYPE, DEVICE, LANGUAGE

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
        if not model_downloaded(model_name):
            typer.echo()
            typer.secho(
                f"[↓] Model '{model_name}' not found locally — downloading now.",
                fg=typer.colors.CYAN,
            )
            typer.echo("    This only happens once. Subsequent runs start immediately.")
            typer.echo()
            try:
                ensure_model_downloaded(model_name)
                typer.secho(f"    [OK] '{model_name}' downloaded.", fg=typer.colors.GREEN)
            except TranscriptionError as e:
                typer.secho(f"[ERROR] Download failed: {e}", fg=typer.colors.RED, err=True)
                raise typer.Exit(1) from e

        # --- Step 2: Load model into VRAM ---
        # Model is local — load_model it now so transcription is instant after recording.
        typer.echo(f"[~] Loading '{model_name}' into memory...")
        try:
            model = load_model(
                config.transcription_model,
                config.transcription_device,
                config.transcription_compute_type,
            )
            actual_device = config.transcription_device
            actual_compute = config.transcription_compute_type
            fallback = False
            for (m, d, c), cached_model in _model_cache.items():
                if m == config.transcription_model and cached_model is model:
                    actual_device = d
                    actual_compute = c
                    fallback = actual_device == "cpu" and config.transcription_device != "cpu"
                    break

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
            typer.secho(f"[ERROR] Could not load_model model: {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from e

    # --- Start recording ---
    recorder = MicrophoneStream(config.input_device_index)
    try:
        recorder.start()
    except Exception as e:
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
        raw_audio = recorder.stop()
        processor = AudioPreProcessor(recorder)  # type: ignore
        audio = processor.process_array(raw_audio, recorder.sample_rate)
    except Exception as e:
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
        ts = time.strftime("%Y%m%d_%H%M%S")
        p = prefix or config.recording_prefix
        wav_path = config.recordings_path / f"{p}_{ts}.wav"
        try:
            recorder.save_wav(audio, wav_path, sample_rate=16000)
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
            import soundfile as sf

            file_audio, _ = sf.read(str(wav_path), dtype="float32", always_2d=False)
            if file_audio.ndim > 1:
                file_audio = file_audio.mean(axis=1)
            result = transcribe(
                file_audio,
                model_name=config.transcription_model,
                device=config.transcription_device,
                compute_type=config.transcription_compute_type,
                language=config.language,
                word_timestamps=False,
            )
        else:
            result = transcribe(
                audio,
                model_name=config.transcription_model,
                device=config.transcription_device,
                compute_type=config.transcription_compute_type,
                language=config.language,
                word_timestamps=False,
            )
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
    # Add segments if present
    if hasattr(result, "segments") and result.segments:
        lines.append("## Segments")
        lines.append("")
        for seg in result.segments:
            lines.append(f"**[{seg.start:.2f}s - {seg.end:.2f}s]** {seg.text}")
        lines.append("")
    return "\n".join(lines) + "\n"


@record_app.command("benchmark")
def benchmark(
    wav_path: Annotated[
        Path,
        typer.Option(
            "--wav",
            "-w",
            help="Path to the WAV file to benchmark stream chunking against",
            exists=True,
            dir_okay=False,
        ),
    ],
) -> None:
    """Run a chunking benchmark over a WAV file.

    Tests multiple combinations of silence_threshold_ms and min_chunk_s
    to empirically determine the best streaming latency and quality.
    """
    import numpy as np
    import soundfile as sf
    from rich.console import Console
    from rich.table import Table
    from voicepad_core import ensure_model_downloaded, load_model
    from voicepad_core.streaming import StreamingTranscriber

    config = get_config()

    if not model_downloaded(config.transcription_model):
        typer.secho(f"Downloading model '{config.transcription_model}'...", fg=typer.colors.CYAN)
        ensure_model_downloaded(config.transcription_model)

    typer.echo("Loading model...")
    load_model(config.transcription_model, config.transcription_device, config.transcription_compute_type)

    typer.echo(f"Loading {wav_path}...")
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != 16000:
        audio = np.interp(
            np.linspace(0, len(audio), int(len(audio) * 16000 / sr)),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
        sr = 16000

    thresholds = [500, 800, 1000, 1500]
    mins = [10.0, 15.0, 20.0, 29.0]

    console = Console()
    table = Table(title=f"Benchmark results for {wav_path.name}")
    table.add_column("Silence Threshold (ms)", justify="right")
    table.add_column("Min Chunk (s)", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Total Latency (ms)", justify="right")

    class MockRecorder:
        def __init__(self, audio_data):
            self.audio_data = audio_data
            self.sample_rate = 16000
            self.cursor = 0

        def get_snapshot(self):
            return self.audio_data[: self.cursor]

    typer.echo("Running permutations...")

    for thresh in thresholds:
        for m in mins:
            recorder = MockRecorder(audio)

            chunks = []

            def on_chunk(c, chunks=chunks):
                chunks.append(c)

            transcriber = StreamingTranscriber(
                recorder=recorder,
                on_chunk=on_chunk,
                on_error=lambda err: None,
                model_name=config.transcription_model,
                device=config.transcription_device,
                compute_type=config.transcription_compute_type,
                min_chunk_s=m,
                max_chunk_s=29.0,
                overlap_s=0.5,
                silence_threshold_ms=thresh,
            )

            # Setup offline run without threads
            transcriber._stop_event.clear()
            transcriber._consumed_samples = 0
            transcriber._chunk_index = 0
            transcriber._prev_context = ""
            transcriber._prev_chunk_text = ""
            from voicepad_core.vad import SileroVAD

            transcriber._vad = SileroVAD(
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=thresh,
                speech_pad_ms=30,
            )
            transcriber._vad.reset()

            # Iterate over the audio in small polling intervals
            # Real streaming polls every 0.3s
            poll_s = 0.3
            step_samples = int(poll_s * 16000)

            while recorder.cursor < len(audio):
                recorder.cursor += step_samples
                if recorder.cursor > len(audio):
                    recorder.cursor = len(audio)

                # Mock time inside _monitor_loop one iteration
                # We can't really call _monitor_loop() directly because it has a while loop over stop_event with sleep
                # We will just extract the core dispatch logic

                accumulated_s = (recorder.cursor - transcriber._consumed_samples) / 16000
                if accumulated_s < m:
                    if accumulated_s >= 29.0:
                        transcriber._dispatch_chunk(
                            recorder.audio_data[: recorder.cursor], is_final=False, capture_rate=16000
                        )
                        transcriber._vad.reset()
                    continue

                tail_duration_s = thresh / 1000.0
                tail_samples = int(tail_duration_s * 16000)
                tail_audio = recorder.audio_data[: recorder.cursor]
                tail = tail_audio[-tail_samples:] if len(tail_audio) >= tail_samples else tail_audio

                speech_segments = transcriber._vad.detect(tail, sample_rate=16000)

                if not speech_segments or accumulated_s >= 29.0:
                    transcriber._dispatch_chunk(tail_audio, is_final=False, capture_rate=16000)
                    transcriber._vad.reset()

            # End
            if transcriber._consumed_samples < len(audio):
                transcriber._dispatch_chunk(audio, is_final=True, capture_rate=16000)
            else:
                from voicepad_core.streaming import ChunkResult

                chunks.append(
                    ChunkResult(
                        index=transcriber._chunk_index + 1,
                        text="",
                        start_s=transcriber._consumed_samples / 16000,
                        end_s=len(audio) / 16000,
                        is_final=True,
                    )
                )

            total_latency = sum(c.latency_ms for c in chunks if c.latency_ms is not None)
            total_chunks = len([c for c in chunks if c.text])

            table.add_row(str(thresh), f"{m:.1f}", str(total_chunks), f"{total_latency:.0f}")

    console.print(table)
