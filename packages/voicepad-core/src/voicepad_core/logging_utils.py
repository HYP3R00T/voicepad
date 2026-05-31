"""Logging utilities for VoicePad transcription operations.

Provides per-transcription log files with automatic cleanup and graceful shutdown.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import logging.handlers
import queue as _queue
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Global registry of active log handlers and listeners for cleanup
_active_handlers: list[logging.Handler] = []
_active_listeners: list[logging.handlers.QueueListener] = []
_shutdown_registered = False


def setup_transcription_logger(
    logs_path: Path,
    log_level: str = "INFO",
    session_id: str | None = None,
    structured: bool = False,
    use_queue: bool = False,
    console: bool = False,
) -> tuple[logging.Logger, Path]:
    """Set up a per-transcription logger with file and console output.

    Creates a timestamped log file for each transcription session.
    Logs are written immediately (no buffering) to ensure they persist
    even if the program crashes or is interrupted.

    Args:
        logs_path: Directory where log files should be saved
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        session_id: Optional session identifier (timestamp used if None)

    Returns:
        Tuple of (logger, log_file_path)
    """
    global _shutdown_registered

    # Ensure logs directory exists
    logs_path.mkdir(parents=True, exist_ok=True)

    # Generate session ID if not provided
    if session_id is None:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create log file path
    log_file = logs_path / f"transcription_{session_id}.log"

    # Create logger
    logger = logging.getLogger(f"voicepad.transcription.{session_id}")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.propagate = False  # Don't propagate to root logger

    # Remove any existing handlers
    logger.handlers.clear()

    # Create file handler with immediate flush
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(getattr(logging, log_level.upper()))

    # Create console handler (optional)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console

    # Create human-readable formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Optionally use structured JSON formatting for file output
    if structured:
        json_formatter = JSONFormatter()
        file_handler.setFormatter(json_formatter)
        console_handler.setFormatter(json_formatter)
    else:
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

    # Optionally use a background queue to decouple logging I/O from main thread
    if use_queue:
        log_queue: _queue.Queue[logging.LogRecord] = _queue.Queue(-1)
        queue_handler = logging.handlers.QueueHandler(log_queue)
        logger.addHandler(queue_handler)
        # Start a listener that writes from the queue to the real handlers
        handlers: list[logging.Handler] = [file_handler]
        if console:
            handlers.append(console_handler)
        listener = logging.handlers.QueueListener(log_queue, *handlers)
        listener.start()
        _active_listeners.append(listener)
        _active_handlers.extend(handlers)
    else:
        # Add handlers directly
        logger.addHandler(file_handler)
        _active_handlers.append(file_handler)
        if console:
            logger.addHandler(console_handler)
            _active_handlers.append(console_handler)

    # Register shutdown handlers (only once)
    if not _shutdown_registered:
        atexit.register(_cleanup_handlers)
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
        _shutdown_registered = True

    # Log session start
    logger.info("=" * 80)
    logger.info(f"Transcription session started: {session_id}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)

    return logger, log_file


class JSONFormatter(logging.Formatter):
    """Simple JSON formatter for structured logging.

    It will include timestamp, level, logger name, message, event, and any
    structured payload attached via the `__structured_payload__` attribute
    on the LogRecord (set by `log_event`).
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        structured = getattr(record, "__structured_payload__", None)
        event = getattr(record, "event", None)
        if event:
            payload["event"] = event
        if isinstance(structured, dict):
            payload.update(structured)
        try:
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            return json.dumps({"message": record.getMessage()})


def log_event(logger: logging.Logger, event: str, level: str = "info", **data: Any) -> None:
    """Emit a structured event to the logger.

    The event is included both in the human-readable log message and as a
    structured payload available to the JSON formatter.
    """
    msg = f"EVENT {event}: {data}"
    level_name = level.lower()
    extra = {"__structured_payload__": data, "event": event}
    if level_name == "debug":
        logger.debug(msg, extra=extra)
    elif level_name == "warning" or level_name == "warn":
        logger.warning(msg, extra=extra)
    elif level_name == "error":
        logger.error(msg, extra=extra)
    else:
        logger.info(msg, extra=extra)


def _cleanup_handlers() -> None:
    """Flush and close all active log handlers.

    Called automatically on program exit or signal interrupt.
    """
    # Stop any active QueueListeners first
    for listener in _active_listeners:
        try:
            import contextlib

            with contextlib.suppress(Exception):
                listener.stop()
        except Exception:
            pass
    _active_listeners.clear()

    for handler in _active_handlers:
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass  # Ignore errors during cleanup
    _active_handlers.clear()


def _signal_handler(signum: int, frame: Any) -> None:
    """Handle interrupt signals (Ctrl+C, SIGTERM).

    Ensures logs are flushed before program exits.

    Args:
        signum: Signal number
        frame: Current stack frame
    """
    print(f"\n\nReceived signal {signum}, flushing logs and exiting...")
    _cleanup_handlers()
    sys.exit(0)


def log_transcription_start(
    logger: logging.Logger,
    audio_duration: float,
    model_name: str,
    device: str,
    compute_type: str,
) -> None:
    """Log the start of a transcription operation.

    Args:
        logger: Logger instance
        audio_duration: Duration of audio in seconds
        model_name: Whisper model name
        device: Device (cuda/cpu)
        compute_type: Compute type (int8/float16/etc)
    """
    logger.info("-" * 80)
    logger.info("TRANSCRIPTION START")
    logger.info(f"  Audio duration: {audio_duration:.2f}s")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Compute type: {compute_type}")
    logger.info("-" * 80)
    # Emit structured event for downstream processing
    with contextlib.suppress(Exception):
        log_event(
            logger,
            "transcription.start",
            level="info",
            audio_duration=audio_duration,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
        )


def log_transcription_end(
    logger: logging.Logger,
    success: bool,
    latency_ms: float | None = None,
    text_length: int | None = None,
    error: str | None = None,
) -> None:
    """Log the end of a transcription operation.

    Args:
        logger: Logger instance
        success: Whether transcription succeeded
        latency_ms: Transcription latency in milliseconds
        text_length: Length of transcribed text
        error: Error message if failed
    """
    logger.info("-" * 80)
    if success:
        logger.info("TRANSCRIPTION COMPLETE")
        if latency_ms is not None:
            logger.info(f"  Latency: {latency_ms:.0f}ms")
        if text_length is not None:
            logger.info(f"  Text length: {text_length} characters")
    else:
        logger.error("TRANSCRIPTION FAILED")
        if error:
            logger.error(f"  Error: {error}")
    logger.info("-" * 80)
    # Emit structured end event
    with contextlib.suppress(Exception):
        log_event(
            logger,
            "transcription.end",
            level=("info" if success else "error"),
            success=success,
            latency_ms=latency_ms,
            text_length=text_length,
            error=error,
        )


def log_chunk_processing(
    logger: logging.Logger,
    chunk_idx: int,
    chunk_duration: float,
    chunk_start: float,
    chunk_end: float,
) -> None:
    """Log processing of an audio chunk during streaming.

    Args:
        logger: Logger instance
        chunk_idx: Chunk index (0-based)
        chunk_duration: Duration of chunk in seconds
        chunk_start: Start time in seconds
        chunk_end: End time in seconds
    """
    logger.debug(
        f"Processing chunk {chunk_idx}: duration={chunk_duration:.2f}s, range=[{chunk_start:.2f}s - {chunk_end:.2f}s]"
    )
    with contextlib.suppress(Exception):
        log_event(
            logger,
            "chunk.processing",
            level="debug",
            chunk_idx=chunk_idx,
            chunk_duration=chunk_duration,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
        )


def log_model_load(
    logger: logging.Logger,
    model_name: str,
    device: str,
    compute_type: str,
    load_time_ms: float,
) -> None:
    """Log model loading operation.

    Args:
        logger: Logger instance
        model_name: Whisper model name
        device: Device (cuda/cpu)
        compute_type: Compute type
        load_time_ms: Time taken to load model in milliseconds
    """
    logger.info(f"Model loaded: {model_name} on {device} ({compute_type}) in {load_time_ms:.0f}ms")
    with contextlib.suppress(Exception):
        log_event(
            logger,
            "model.load",
            level="info",
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            load_time_ms=load_time_ms,
        )


def log_system_info(logger: logging.Logger) -> None:
    """Log system information for debugging.

    Args:
        logger: Logger instance
    """
    import platform
    import sys

    logger.info("=" * 80)
    logger.info("SYSTEM INFORMATION")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python: {sys.version.split()[0]}")
    logger.info(f"  Architecture: {platform.machine()}")
    logger.info(f"  Processor: {platform.processor()}")

    # Check for CUDA availability
    try:
        import importlib

        torch = importlib.import_module("torch")
        cuda_available = getattr(torch.cuda, "is_available", lambda: False)()
        logger.info(f"  PyTorch CUDA available: {cuda_available}")
        if cuda_available:
            logger.info(f"  CUDA version: {getattr(torch.version, 'cuda', 'unknown')}")
            try:
                gpu_count = getattr(torch.cuda, "device_count", lambda: 0)()
                logger.info(f"  GPU count: {gpu_count}")
                for i in range(gpu_count):
                    try:
                        name = getattr(torch.cuda, "get_device_name", lambda idx: "unknown")(i)
                        logger.info(f"  GPU {i}: {name}")
                    except Exception:
                        logger.info(f"  GPU {i}: unknown")
            except Exception:
                logger.warning("  Failed to inspect GPU details")
    except Exception:
        logger.info("  PyTorch: Not installed or unavailable")

    logger.info("=" * 80)
    with contextlib.suppress(Exception):
        log_event(logger, "system.info", level="info")


def log_config_info(logger: logging.Logger, config: Any) -> None:
    """Log configuration information.

    Args:
        logger: Logger instance
        config: Configuration object
    """
    logger.info("=" * 80)
    logger.info("CONFIGURATION")

    # Log all config fields
    if hasattr(config, "model_dump"):
        config_dict = config.model_dump()
    elif hasattr(config, "__dict__"):
        config_dict = config.__dict__
    else:
        config_dict = {}

    for key, value in sorted(config_dict.items()):
        # Convert Path objects to strings
        if hasattr(value, "__fspath__"):
            value = str(value)
        logger.info(f"  {key}: {value}")

    logger.info("=" * 80)
    with contextlib.suppress(Exception):
        log_event(logger, "config.info", level="info", config=config_dict)


def log_audio_info(
    logger: logging.Logger,
    audio: Any,
    sample_rate: int,
    source: str = "unknown",
) -> None:
    """Log detailed audio information.

    Args:
        logger: Logger instance
        audio: Audio array
        sample_rate: Sample rate in Hz
        source: Source of the audio (file, mic, buffer)
    """
    import numpy as np

    logger.info("-" * 80)
    logger.info(f"AUDIO INFO ({source})")
    logger.info(f"  Shape: {audio.shape}")
    logger.info(f"  Dtype: {audio.dtype}")
    logger.info(f"  Sample rate: {sample_rate} Hz")
    logger.info(f"  Duration: {len(audio) / sample_rate:.3f}s")
    logger.info(f"  Samples: {len(audio)}")

    # Audio statistics
    logger.info(f"  Min value: {np.min(audio):.6f}")
    logger.info(f"  Max value: {np.max(audio):.6f}")
    logger.info(f"  Mean: {np.mean(audio):.6f}")
    logger.info(f"  RMS: {np.sqrt(np.mean(audio**2)):.6f}")
    logger.info(f"  Peak amplitude: {np.max(np.abs(audio)):.6f}")

    # Check for silence
    rms = np.sqrt(np.mean(audio**2))
    if rms < 0.001:
        logger.warning(f"  ⚠️  Audio appears to be very quiet (RMS={rms:.6f})")

    # Check for clipping
    if np.max(np.abs(audio)) > 0.99:
        logger.warning(f"  ⚠️  Audio may be clipping (peak={np.max(np.abs(audio)):.6f})")

    logger.info("-" * 80)
    with contextlib.suppress(Exception):
        log_event(
            logger,
            "audio.info",
            level="info",
            source=source,
            shape=getattr(audio, "shape", None),
            dtype=getattr(audio, "dtype", None),
            sample_rate=sample_rate,
            duration=len(audio) / sample_rate,
        )


def log_vad_info(
    logger: logging.Logger,
    speech_segments: list,
    audio_duration: float,
) -> None:
    """Log VAD (Voice Activity Detection) results.

    Args:
        logger: Logger instance
        speech_segments: List of speech segments detected
        audio_duration: Total audio duration in seconds
    """
    logger.info("-" * 80)
    logger.info("VAD RESULTS")
    logger.info(f"  Total segments: {len(speech_segments)}")

    if speech_segments:
        total_speech = sum(seg.end - seg.start for seg in speech_segments)
        logger.info(f"  Total speech duration: {total_speech:.3f}s")
        logger.info(f"  Speech ratio: {total_speech / audio_duration * 100:.1f}%")

        for i, seg in enumerate(speech_segments):
            logger.info(f"  Segment {i + 1}: {seg.start:.3f}s - {seg.end:.3f}s ({seg.end - seg.start:.3f}s)")
    else:
        logger.warning("  ⚠️  No speech detected!")

    logger.info("-" * 80)
    with contextlib.suppress(Exception):
        log_event(
            logger,
            "vad.results",
            level="info",
            total_segments=len(speech_segments),
            audio_duration=audio_duration,
        )


def log_model_cache_info(logger: logging.Logger) -> None:
    """Log information about cached models.

    Args:
        logger: Logger instance
    """
    try:
        from voicepad_core.inference.model_manager import _model_cache

        logger.info("-" * 80)
        logger.info("MODEL CACHE")
        logger.info(f"  Cached models: {len(_model_cache)}")

        for (model_name, device, compute_type), model in _model_cache.items():
            logger.info(f"  - {model_name} on {device} ({compute_type})")
            logger.info(f"    Model object: {type(model).__name__}")

        if not _model_cache:
            logger.info("  (empty)")

        logger.info("-" * 80)
    except Exception as e:
        logger.warning(f"Failed to get model cache info: {e}")
        with contextlib.suppress(Exception):
            log_event(logger, "model.cache.error", level="warning", error=str(e))


def log_segments_info(
    logger: logging.Logger,
    segments: list,
    stage: str = "final",
) -> None:
    """Log detailed segment information.

    Args:
        logger: Logger instance
        segments: List of transcription segments
        stage: Stage of processing (raw, filtered, final)
    """
    logger.info("-" * 80)
    logger.info(f"SEGMENTS ({stage})")
    logger.info(f"  Total segments: {len(segments)}")

    if segments:
        total_duration = sum(seg.end - seg.start for seg in segments)
        logger.info(f"  Total duration: {total_duration:.3f}s")

        for i, seg in enumerate(segments):
            logger.info(
                f"  Segment {i + 1}: [{seg.start:.3f}s - {seg.end:.3f}s] "
                f"conf={seg.avg_logprob:.2f} no_speech={seg.no_speech_prob:.2f}"
            )
            logger.info(f"    Text: '{seg.text}'")
            if hasattr(seg, "words") and seg.words:
                logger.debug(f"    Words: {len(seg.words)}")

    logger.info("-" * 80)
    with contextlib.suppress(Exception):
        log_event(
            logger,
            "segments.info",
            level="info",
            stage=stage,
            total_segments=len(segments),
        )


def log_postprocessing_step(
    logger: logging.Logger,
    step_name: str,
    input_text: str,
    output_text: str,
) -> None:
    """Log a post-processing step.

    Args:
        logger: Logger instance
        step_name: Name of the processing step
        input_text: Text before processing
        output_text: Text after processing
    """
    changed = input_text != output_text
    logger.debug(f"Post-processing: {step_name}")
    logger.debug(f"  Input length: {len(input_text)} chars")
    logger.debug(f"  Output length: {len(output_text)} chars")
    logger.debug(f"  Changed: {changed}")

    if changed:
        logger.debug(f"  Before: '{input_text[:100]}{'...' if len(input_text) > 100 else ''}'")
        logger.debug(f"  After:  '{output_text[:100]}{'...' if len(output_text) > 100 else ''}'")
    with contextlib.suppress(Exception):
        log_event(
            logger,
            "postprocessing.step",
            level="debug",
            step_name=step_name,
            changed=changed,
            input_length=len(input_text),
            output_length=len(output_text),
        )


def configure_global_logging(log_level: str = "INFO", logs_path: Path | None = None, console: bool = False) -> None:
    """Configure application-wide logging (root logger).

    Creates a console + file handler writing to `logs_path/voicepad.log` and
    sets the root logger level. Safe to call multiple times; existing handlers
    are replaced by this call to ensure deterministic behavior.
    """
    root = logging.getLogger()
    level = getattr(logging, log_level.upper(), logging.INFO)
    root.setLevel(level)

    # Clear existing handlers to avoid duplicate logging
    for h in list(root.handlers):
        try:
            root.removeHandler(h)
            h.close()
        except Exception:
            pass

    if logs_path is None:
        logs_path = Path.cwd()
    else:
        logs_path.mkdir(parents=True, exist_ok=True)

    log_file = logs_path / "voicepad.log"

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    if console:
        root.addHandler(console_handler)

    logging.getLogger(__name__).info(f"Global logging configured: level={log_level}, file={log_file}")
