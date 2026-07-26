from .backend_manager import BackendRegistry, SessionManager
from .backends import (
    FasterWhisperDriver,
    FasterWhisperSession,
    ParakeetOnnxDriver,
    ParakeetOnnxSession,
)
from .composition import (
    InferenceCoordinator,
    activate_model,
    deactivate_model,
    get_default_coordinator,
    model_is_ready,
    prepare_model,
)
from .contracts import (
    BackendCapabilities,
    BackendDriver,
    PreparedModel,
    RuntimeInfo,
    RuntimeOptions,
    TranscriptionContext,
    TranscriptionRequest,
    TranscriptionSession,
)
from .engine import close_default_sessions, transcribe
from .errors import (
    AudioTooLongWarning,
    AudioTooShortError,
    BackendLookupError,
    BackendSessionError,
    BackendUnavailableError,
    ModelNotFoundError,
    TranscriptionError,
)
from .types import Segment, TranscriptionResult, WordTimestamp

__all__ = [
    "transcribe",
    "close_default_sessions",
    "BackendRegistry",
    "SessionManager",
    "InferenceCoordinator",
    "activate_model",
    "deactivate_model",
    "get_default_coordinator",
    "model_is_ready",
    "prepare_model",
    "BackendCapabilities",
    "BackendDriver",
    "PreparedModel",
    "RuntimeInfo",
    "RuntimeOptions",
    "TranscriptionContext",
    "TranscriptionRequest",
    "TranscriptionSession",
    "FasterWhisperDriver",
    "FasterWhisperSession",
    "ParakeetOnnxDriver",
    "ParakeetOnnxSession",
    "TranscriptionResult",
    "Segment",
    "WordTimestamp",
    "TranscriptionError",
    "AudioTooShortError",
    "AudioTooLongWarning",
    "ModelNotFoundError",
    "BackendLookupError",
    "BackendUnavailableError",
    "BackendSessionError",
]
