"""voicepad-core — audio capture and transcription engine.

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

from voicepad_core.audio import SAMPLE_RATE, AudioRecorder, AudioRecorderError
from voicepad_core.config import Config, get_config, get_config_with_metadata
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
]
