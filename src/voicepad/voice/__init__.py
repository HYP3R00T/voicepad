"""Voice recording and transcription functionality."""

from voicepad.voice.recorder import (
    AudioDevice,
    capture_audio_background,
    get_device_by_index,
    get_input_devices,
    print_devices,
    record_voice,
)
from voicepad.voice.transcriber import (
    TranscriptionPoller,
    TranscriptionResult,
    WhisperTranscriber,
    transcribe_audio,
)
from voicepad.voice.utils import get_recording_path, get_timestamp

__all__ = [
    # Recorder
    "AudioDevice",
    "capture_audio_background",
    "get_device_by_index",
    "get_input_devices",
    "print_devices",
    "record_voice",
    # Transcriber
    "TranscriptionPoller",
    "TranscriptionResult",
    "WhisperTranscriber",
    "transcribe_audio",
    # Utils
    "get_recording_path",
    "get_timestamp",
]
