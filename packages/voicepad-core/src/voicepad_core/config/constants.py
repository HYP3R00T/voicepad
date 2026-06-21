from __future__ import annotations

from ..models import VALID_TRANSCRIPTION_MODELS

DEFAULT_INITIAL_PROMPT = "Hello. This is a transcription with proper punctuation, capitalization, and grammar."
DEFAULT_VAD_MODEL_FILENAME = "silero_vad_v6.onnx"
DEFAULT_VAD_MODEL_URL = (
    "https://raw.githubusercontent.com/SYSTRAN/faster-whisper/master/faster_whisper/assets/silero_vad_v6.onnx"
)

__all__ = [
    "VALID_TRANSCRIPTION_MODELS",
    "DEFAULT_INITIAL_PROMPT",
    "DEFAULT_VAD_MODEL_FILENAME",
    "DEFAULT_VAD_MODEL_URL",
]
