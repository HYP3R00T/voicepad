"""voicepad-core — audio recording and transcription engine.

Public API:

    Audio recording:
        AudioRecorder       — captures microphone input as numpy arrays
        AudioRecorderError  — raised on device or recording errors
        SAMPLE_RATE         — 16000 (fixed for Whisper compatibility)

    Transcription:
        transcribe_buffer   — primary function: np.ndarray → TranscriptionResult
        transcribe_file     — convenience: Path → TranscriptionResult
        get_or_load_model   — cached model loader (useful for pre-warming)
        TranscriptionResult — dataclass with text, segments, timing, device info
        Segment             — individual timed segment (start, end, text)
        TranscriptionError  — base transcription exception
        AudioTooShortError  — audio below min_audio_duration_s
        AudioTooLongWarning — audio above max_audio_duration_s

    VAD chunking (long-form recording):
        RealtimeChunker     — splits long audio at natural speech boundaries
        ChunkMetadata       — timing info for each chunk

    Configuration:
        Config              — Pydantic settings model
        get_config          — load config from YAML / env / defaults
        get_config_with_metadata — load config + source metadata per field
"""

from voicepad_core.audio import SAMPLE_RATE, AudioRecorder, AudioRecorderError
from voicepad_core.chunking import ChunkMetadata, RealtimeChunker
from voicepad_core.config import Config, get_config, get_config_with_metadata
from voicepad_core.transcription import (
    AudioTooLongWarning,
    AudioTooShortError,
    Segment,
    TranscriptionError,
    TranscriptionResult,
    get_or_load_model,
    transcribe_buffer,
    transcribe_file,
)

__all__ = [
    # Audio
    "AudioRecorder",
    "AudioRecorderError",
    "SAMPLE_RATE",
    # Transcription
    "transcribe_buffer",
    "transcribe_file",
    "get_or_load_model",
    "TranscriptionResult",
    "Segment",
    "TranscriptionError",
    "AudioTooShortError",
    "AudioTooLongWarning",
    # VAD chunking
    "RealtimeChunker",
    "ChunkMetadata",
    # Config
    "Config",
    "get_config",
    "get_config_with_metadata",
]
