from pathlib import Path

from .audio import FileSource, MicrophoneStream, WavArtifact, write_wav_atomic
from .config import VALID_TRANSCRIPTION_MODELS, Config, get_config, get_config_with_metadata
from .inference import (
    BackendCapabilities,
    BackendDriver,
    BackendRegistry,
    FasterWhisperDriver,
    InferenceCoordinator,
    ParakeetOnnxDriver,
    PreparedModel,
    RuntimeInfo,
    RuntimeOptions,
    SessionManager,
    TranscriptionContext,
    TranscriptionRequest,
    TranscriptionSession,
    activate_model,
    close_default_sessions,
    deactivate_model,
    get_default_coordinator,
    model_is_ready,
    prepare_model,
    transcribe,
)
from .inference.constants import COMPUTE_TYPE, DEFAULT_MODEL, DEVICE, LANGUAGE, SAMPLE_RATE
from .inference.errors import (
    AudioTooLongWarning,
    AudioTooShortError,
    ModelNotFoundError,
    TranscriptionError,
)
from .inference.types import Segment, TranscriptionResult, WordTimestamp
from .logging_utils import (
    begin_transcription_session,
    configure_global_logging,
    end_transcription_session,
    log_transcription_end,
    log_transcription_start,
    setup_transcription_logger,
)
from .models import (
    ModelCompatibilityError,
    ModelSpec,
    get_model_hint,
    get_model_label,
    list_basic_model_ids,
    list_basic_model_options,
    list_model_ids,
    list_model_specs,
    register_model,
    register_models,
)
from .postprocessing import (
    deduplicate_overlap,
    filter_segments,
    normalize,
    remove_hallucinations,
)
from .preprocessing import TARGET_SAMPLE_RATE, AudioPreProcessor
from .streaming import ChunkResult, StreamingTranscriber
from .vad import SileroVAD, SpeechSegment
from .vad import ensure_model_exists as ensure_vad_model


def transcribe_file(
    wav_path: str | Path,
    model_name: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    language: str | None = None,
    local_agreement: bool | None = None,
    config: Config | None = None,
) -> TranscriptionResult:
    """Transcribe a WAV file through its model-selected backend.

    Audio loading and normalization remain backend-neutral.

    Args:
        wav_path: Path to WAV file to transcribe
        model_name: Registered model to use
        device: Inference device ('cuda' or 'cpu')
        compute_type: Runtime precision string
        language: BCP-47 language code (default: 'en')
        local_agreement: Enable two-pass verification for higher accuracy

    Returns:
        TranscriptionResult with full transcription, segments, and metadata

    Raises:
        FileNotFoundError: If WAV file doesn't exist
        AudioTooShortError: If audio is below minimum duration
        TranscriptionError: If transcription fails
    """
    resolved_config = config or get_config()
    model_name = model_name if model_name is not None else resolved_config.transcription_model
    device = device if device is not None else resolved_config.transcription_device
    compute_type = compute_type if compute_type is not None else resolved_config.transcription_compute_type
    language = language if language is not None else resolved_config.language
    local_agreement = local_agreement if local_agreement is not None else resolved_config.local_agreement_file

    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    source = FileSource(wav_path)
    source.read()

    preprocessor = AudioPreProcessor(source)
    processed_audio = preprocessor.process()

    result = transcribe(
        processed_audio.samples,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        language=language,
        config=resolved_config,
    )

    if local_agreement:
        from .postprocessing.agreement import apply_local_agreement

        result = apply_local_agreement(processed_audio.samples, result, model_name, device, compute_type, language)

    return result


__all__ = [
    # Audio
    "MicrophoneStream",
    "FileSource",
    "WavArtifact",
    "write_wav_atomic",
    "AudioPreProcessor",
    "TARGET_SAMPLE_RATE",
    "SAMPLE_RATE",
    # Config
    "Config",
    "get_config",
    "get_config_with_metadata",
    "get_model_hint",
    "get_model_label",
    "list_basic_model_ids",
    "list_basic_model_options",
    "VALID_TRANSCRIPTION_MODELS",
    "ModelCompatibilityError",
    "ModelSpec",
    "list_model_ids",
    "list_model_specs",
    "register_model",
    "register_models",
    # Inference
    "transcribe",
    "transcribe_file",
    "close_default_sessions",
    "InferenceCoordinator",
    "activate_model",
    "deactivate_model",
    "get_default_coordinator",
    "model_is_ready",
    "prepare_model",
    "BackendRegistry",
    "SessionManager",
    "BackendCapabilities",
    "BackendDriver",
    "PreparedModel",
    "RuntimeInfo",
    "RuntimeOptions",
    "TranscriptionContext",
    "TranscriptionRequest",
    "TranscriptionSession",
    "FasterWhisperDriver",
    "ParakeetOnnxDriver",
    "DEVICE",
    "COMPUTE_TYPE",
    "DEFAULT_MODEL",
    "LANGUAGE",
    # Types
    "TranscriptionResult",
    "Segment",
    "WordTimestamp",
    # Exceptions
    "TranscriptionError",
    "AudioTooShortError",
    "AudioTooLongWarning",
    "ModelNotFoundError",
    # Postprocessing
    "filter_segments",
    "deduplicate_overlap",
    "remove_hallucinations",
    "normalize",
    # Streaming
    "ChunkResult",
    "StreamingTranscriber",
    # VAD
    "SileroVAD",
    "SpeechSegment",
    "ensure_vad_model",
    # Logging
    "begin_transcription_session",
    "configure_global_logging",
    "end_transcription_session",
    "setup_transcription_logger",
    "log_transcription_start",
    "log_transcription_end",
]
