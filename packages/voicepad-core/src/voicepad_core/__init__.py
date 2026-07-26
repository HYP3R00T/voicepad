from pathlib import Path

from .audio import FileSource, MicrophoneStream, RawAudio, WavArtifact, WaveformSpec, write_wav_atomic
from .config import Config, get_config, get_config_with_metadata
from .inference import (
    RuntimeInfo,
    activate_model,
    deactivate_model,
    model_is_ready,
    prepare_model,
    transcribe,
)
from .inference.errors import AudioTooShortError, TranscriptionError
from .inference.types import Segment, TranscriptionResult, WordTimestamp
from .logging_utils import (
    begin_transcription_session,
    configure_global_logging,
    end_transcription_session,
    log_transcription_end,
    log_transcription_start,
)
from .models import (
    MODELS,
    Model,
    get_model,
    model_options,
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
    """Load source audio and let the selected backend contract prepare it."""
    resolved_config = config or get_config()
    model_name = model_name if model_name is not None else resolved_config.transcription_model
    device = device if device is not None else resolved_config.transcription_device
    compute_type = compute_type if compute_type is not None else resolved_config.transcription_compute_type
    language = language if language is not None else resolved_config.language
    local_agreement = local_agreement if local_agreement is not None else resolved_config.local_agreement_file

    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    raw_audio = FileSource(wav_path).read_audio()

    result = transcribe(
        raw_audio,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        language=language,
        config=resolved_config,
    )

    if local_agreement:
        from .postprocessing.agreement import apply_local_agreement

        result = apply_local_agreement(raw_audio, result, model_name, device, compute_type, language)

    return result


__all__ = [
    "MicrophoneStream",
    "WaveformSpec",
    "RawAudio",
    "FileSource",
    "WavArtifact",
    "write_wav_atomic",
    "AudioPreProcessor",
    "TARGET_SAMPLE_RATE",
    "Config",
    "get_config",
    "get_config_with_metadata",
    "MODELS",
    "Model",
    "get_model",
    "model_options",
    "transcribe",
    "transcribe_file",
    "activate_model",
    "deactivate_model",
    "model_is_ready",
    "prepare_model",
    "RuntimeInfo",
    "TranscriptionResult",
    "Segment",
    "WordTimestamp",
    "TranscriptionError",
    "AudioTooShortError",
    "ChunkResult",
    "StreamingTranscriber",
    "SileroVAD",
    "SpeechSegment",
    "ensure_vad_model",
    "begin_transcription_session",
    "configure_global_logging",
    "end_transcription_session",
    "log_transcription_start",
    "log_transcription_end",
]
