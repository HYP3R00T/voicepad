# inference/__init__.py

"""Inference package — Whisper transcription via CTranslate2 (faster-whisper).

Quick start:
    from inference import transcribe, load, unload_all
    from inference import TranscriptionResult, TranscriptionError

    load()                           # warm model at app startup
    result = transcribe(audio)       # audio: float32 np.ndarray at 16kHz
    unload_all()                     # free GPU on TUI exit
"""

from .download import ensure_model_downloaded, model_downloaded
from .engine import transcribe
from .exceptions import (
    AudioTooLongWarning,
    AudioTooShortError,
    ModelNotFoundError,
    TranscriptionError,
)
from .model_manager import get, is_loaded, load, unload, unload_all
from .types import Segment, TranscriptionResult, WordTimestamp

__all__ = [
    # Core
    "transcribe",
    # Model lifecycle
    "load",
    "unload",
    "unload_all",
    "is_loaded",
    "get",
    # Download
    "ensure_model_downloaded",
    "model_downloaded",
    # Types
    "TranscriptionResult",
    "Segment",
    "WordTimestamp",
    # Exceptions
    "TranscriptionError",
    "AudioTooShortError",
    "AudioTooLongWarning",
    "ModelNotFoundError",
]
