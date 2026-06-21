from __future__ import annotations


def _get_available_models() -> tuple[str, ...]:
    """Query available models from faster_whisper at import time."""
    try:
        from faster_whisper.utils import available_models

        return tuple(available_models())
    except Exception:
        return (
            "tiny.en",
            "tiny",
            "base.en",
            "base",
            "small.en",
            "small",
            "medium.en",
            "medium",
            "large-v1",
            "large-v2",
            "large-v3",
            "large",
            "distil-large-v2",
            "distil-medium.en",
            "distil-small.en",
            "distil-large-v3",
            "distil-large-v3.5",
            "large-v3-turbo",
            "turbo",
        )


VALID_TRANSCRIPTION_MODELS: tuple[str, ...] = _get_available_models()
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
