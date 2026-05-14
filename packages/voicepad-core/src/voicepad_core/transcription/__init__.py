"""Audio transcription using faster-whisper (CTranslate2 backend)."""

from .constants import (
    BEAM_SIZE,
    COMPUTE_TYPE,
    CUDA_ERROR_KEYWORDS,
    DEVICE,
    DISTIL_MODELS,
    HALLUCINATION_SILENCE_THRESHOLD,
    HF_REPO_PREFIX,
    INITIAL_PROMPT,
    LANGUAGE,
    MAX_AUDIO_DURATION_S,
    MIN_AUDIO_DURATION_S,
    NO_SPEECH_THRESHOLD,
)
from .core import transcribe_buffer, transcribe_file
from .download import _get_repo_id, ensure_model_downloaded, model_downloaded
from .exceptions import AudioTooLongWarning, AudioTooShortError, TranscriptionError
from .model_manager import _is_cuda_error, _load_cpu_fallback, _model_cache, get_or_load_model
from .types import Segment, TranscriptionResult
from .utils import _filter_segments, _get_vad_parameters, _trim_trailing_silence

__all__ = [
    # Types
    "Segment",
    "TranscriptionResult",
    # Exceptions
    "TranscriptionError",
    "AudioTooShortError",
    "AudioTooLongWarning",
    # Constants
    "DEVICE",
    "COMPUTE_TYPE",
    "BEAM_SIZE",
    "LANGUAGE",
    "HALLUCINATION_SILENCE_THRESHOLD",
    "NO_SPEECH_THRESHOLD",
    "INITIAL_PROMPT",
    "DISTIL_MODELS",
    "MIN_AUDIO_DURATION_S",
    "MAX_AUDIO_DURATION_S",
    "HF_REPO_PREFIX",
    "CUDA_ERROR_KEYWORDS",
    # Core functions
    "transcribe_buffer",
    "transcribe_file",
    # Model management
    "get_or_load_model",
    "_model_cache",
    "_is_cuda_error",
    "_load_cpu_fallback",
    "model_downloaded",
    "ensure_model_downloaded",
    "_get_repo_id",
    # Utilities (for internal use)
    "_trim_trailing_silence",
    "_filter_segments",
    "_get_vad_parameters",
]
