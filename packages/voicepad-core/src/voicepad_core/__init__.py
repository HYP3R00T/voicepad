"""voicepad-core — audio capture and transcription engine.

On Windows, the nvidia-cublas-cu12 and nvidia-cudnn-cu12 packages install their
DLLs under site-packages/nvidia/*/bin/.  Windows does not search Python package
directories for DLLs, so we register each bin directory with os.add_dll_directory()
before any ctranslate2 import can happen.  This is a no-op on non-Windows platforms.

Public API:

    Recording:
        AudioRecorder       — open mic → collect samples → return np.ndarray
        AudioRecorderError  — raised on device errors
        SAMPLE_RATE         — 16000 Hz (fixed for Whisper)

    Transcription:
        transcribe_buffer   — np.ndarray → TranscriptionResult  (primary)
        transcribe_file     — Path → TranscriptionResult         (convenience)
        get_or_load_model   — pre-warm the model cache
        TranscriptionResult — text, segments, latency, device info
        Segment             — (start, end, text)
        TranscriptionError  — base exception
        AudioTooShortError  — audio below MIN_AUDIO_DURATION_S
        AudioTooLongWarning — audio above MAX_AUDIO_DURATION_S

    Configuration:
        Config              — 5-field Pydantic model
        get_config          — load from YAML / env / defaults
        get_config_with_metadata — load + per-field source info
"""

import os
import sys

if sys.platform == "win32":
    import pathlib
    import site

    _site_pkgs = pathlib.Path(site.getsitepackages()[0])
    _nvidia_root = _site_pkgs / "nvidia"
    if _nvidia_root.is_dir():
        for _bin_dir in _nvidia_root.glob("*/bin"):
            if _bin_dir.is_dir():
                os.add_dll_directory(str(_bin_dir))

from voicepad_core.audio import SAMPLE_RATE, AudioRecorder, AudioRecorderError
from voicepad_core.config import Config, get_config, get_config_with_metadata
from voicepad_core.config.settings import VALID_TRANSCRIPTION_MODELS
from voicepad_core.streaming import ChunkResult, StreamingTranscriber
from voicepad_core.transcription import (
    AudioTooLongWarning,
    AudioTooShortError,
    Segment,
    TranscriptionError,
    TranscriptionResult,
    ensure_model_downloaded,
    get_or_load_model,
    model_downloaded,
    transcribe_buffer,
    transcribe_file,
)

__all__ = [
    # Streaming
    "StreamingTranscriber",
    "ChunkResult",
    # Recording
    "AudioRecorder",
    "AudioRecorderError",
    "SAMPLE_RATE",
    # Transcription
    "transcribe_buffer",
    "transcribe_file",
    "get_or_load_model",
    "model_downloaded",
    "ensure_model_downloaded",
    "TranscriptionResult",
    "Segment",
    "TranscriptionError",
    "AudioTooShortError",
    "AudioTooLongWarning",
    # Config
    "Config",
    "get_config",
    "get_config_with_metadata",
    "VALID_TRANSCRIPTION_MODELS",
]
