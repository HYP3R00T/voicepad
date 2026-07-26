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
    BackendContract,
    BackendDriver,
    OutputCapabilities,
    PreparedModel,
    RuntimeInfo,
    RuntimeOptions,
    TranscriptionContext,
    TranscriptionRequest,
    TranscriptionSession,
)
from .engine import transcribe
from .errors import (
    AudioTooShortError,
    BackendLookupError,
    BackendSessionError,
    BackendUnavailableError,
    TranscriptionError,
)
from .runtime import BackendRegistry, RuntimeManager
from .types import Segment, TranscriptionResult, WordTimestamp

__all__ = [
    "transcribe",
    "BackendRegistry",
    "RuntimeManager",
    "InferenceCoordinator",
    "activate_model",
    "deactivate_model",
    "get_default_coordinator",
    "model_is_ready",
    "prepare_model",
    "BackendCapabilities",
    "BackendContract",
    "BackendDriver",
    "OutputCapabilities",
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
    "BackendLookupError",
    "BackendUnavailableError",
    "BackendSessionError",
]
