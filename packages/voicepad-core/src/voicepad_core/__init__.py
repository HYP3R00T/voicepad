"""Audio capture and transcription engine.

On Windows, ctranslate2 cannot discover CUDA DLLs from site-packages.
We pre-load them via ctypes.WinDLL so they're found by name in memory.
"""

import sys
from pathlib import Path

# Pre-load CUDA DLLs on Windows so ctranslate2 can find them.
if sys.platform == "win32":
    import ctypes
    import os
    import pathlib

    _cuda_dll_dirs: set[str] = set()

    # Find nvidia DLL directories from sys.path
    for _path_entry in sys.path:
        _nvidia_dir = pathlib.Path(_path_entry) / "nvidia"
        if _nvidia_dir.is_dir():
            for _dll_file in _nvidia_dir.rglob("*.dll"):
                _cuda_dll_dirs.add(str(_dll_file.parent))
            break

    # Register directories and pre-load DLLs for ctranslate2 discovery
    _loaded = 0
    for _d in sorted(_cuda_dll_dirs):
        os.add_dll_directory(_d)
        os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")

    # Two passes to handle DLL dependency ordering (cudnn depends on cublas)
    _dll_files = []
    for _d in sorted(_cuda_dll_dirs):
        _dll_files.extend(sorted(pathlib.Path(_d).glob("*.dll")))
    _remaining = list(_dll_files)
    for _pass_num in range(2):
        _still_remaining = []
        for _dll in _remaining:
            try:
                ctypes.WinDLL(str(_dll))
                _loaded += 1
            except OSError:
                _still_remaining.append(_dll)
        _remaining = _still_remaining
        if not _remaining:
            break

# ---------------------------------------------------------------------------
# Public API — audio
# ---------------------------------------------------------------------------
from .audio import FileSource, MicrophoneStream

# ---------------------------------------------------------------------------
# Public API — config
# ---------------------------------------------------------------------------
from .config import Config, get_config, get_config_with_metadata
from .config.settings import VALID_TRANSCRIPTION_MODELS

# ---------------------------------------------------------------------------
# Public API — inference
# ---------------------------------------------------------------------------
from .inference import transcribe
from .inference.constants import COMPUTE_TYPE, DEFAULT_MODEL, DEVICE, LANGUAGE, SAMPLE_RATE
from .inference.download import ensure_model_downloaded, model_downloaded
from .inference.errors import (
    AudioTooLongWarning,
    AudioTooShortError,
    ModelNotFoundError,
    TranscriptionError,
)
from .inference.model_manager import _model_cache
from .inference.model_manager import get as get_model
from .inference.model_manager import is_loaded as is_model_loaded
from .inference.model_manager import load as load_model
from .inference.model_manager import unload as unload_model
from .inference.model_manager import unload_all as unload_all_models
from .inference.types import Segment, TranscriptionResult, WordTimestamp

# ---------------------------------------------------------------------------
# Public API — logging
# ---------------------------------------------------------------------------
from .logging_utils import (
    begin_transcription_session,
    configure_global_logging,
    end_transcription_session,
    log_transcription_end,
    log_transcription_start,
    setup_transcription_logger,
)

# ---------------------------------------------------------------------------
# Public API — postprocessing
# ---------------------------------------------------------------------------
from .postprocessing import (
    deduplicate_overlap,
    filter_segments,
    normalize,
    remove_hallucinations,
)
from .preprocessing import TARGET_SAMPLE_RATE, AudioPreProcessor

# ---------------------------------------------------------------------------
# Public API — streaming
# ---------------------------------------------------------------------------
from .streaming import ChunkResult, StreamingTranscriber

# ---------------------------------------------------------------------------
# Public API — VAD
# ---------------------------------------------------------------------------
from .vad import SileroVAD, SpeechSegment
from .vad import ensure_model_exists as ensure_vad_model

# ---------------------------------------------------------------------------
# High-level transcription functions
# ---------------------------------------------------------------------------


def transcribe_file(
    wav_path: str | Path,
    model_name: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    language: str | None = None,
    local_agreement: bool | None = None,
    config: Config | None = None,
) -> TranscriptionResult:
    """Transcribe a WAV file using the new architecture.

    Blueprint Part 8 implementation. Loads audio via FileSource,
    preprocesses to 16kHz mono, and transcribes using the inference engine.

    Args:
        wav_path: Path to WAV file to transcribe
        model_name: Whisper model to use (default: 'turbo')
        device: Inference device ('cuda' or 'cpu')
        compute_type: CTranslate2 precision string
        language: BCP-47 language code (default: 'en')
        local_agreement: Enable two-pass verification for higher accuracy

    Returns:
        TranscriptionResult with full transcription, segments, and metadata

    Raises:
        FileNotFoundError: If WAV file doesn't exist
        AudioTooShortError: If audio is below minimum duration
        TranscriptionError: If transcription fails
    """
    from pathlib import Path

    resolved_config = config or get_config()
    model_name = model_name if model_name is not None else resolved_config.transcription_model
    device = device if device is not None else resolved_config.transcription_device
    compute_type = compute_type if compute_type is not None else resolved_config.transcription_compute_type
    language = language if language is not None else resolved_config.language
    local_agreement = local_agreement if local_agreement is not None else resolved_config.local_agreement_file

    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    # 1. Load audio via FileSource
    source = FileSource(wav_path)
    source.read()

    # 2. Preprocess to 16kHz mono using the preprocessing contract
    preprocessor = AudioPreProcessor(source)
    processed_audio = preprocessor.process()

    # 3. Transcribe
    result = transcribe(
        processed_audio.samples,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        language=language,
        config=resolved_config,
    )

    # 4. Apply LocalAgreement if enabled
    if local_agreement:
        from .postprocessing.agreement import apply_local_agreement

        result = apply_local_agreement(processed_audio.samples, result, model_name, device, compute_type, language)

    return result


__all__ = [
    # Audio
    "MicrophoneStream",
    "FileSource",
    "AudioPreProcessor",
    "TARGET_SAMPLE_RATE",
    "SAMPLE_RATE",
    # Config
    "Config",
    "get_config",
    "get_config_with_metadata",
    "VALID_TRANSCRIPTION_MODELS",
    # Inference
    "transcribe",
    "transcribe_file",
    "DEVICE",
    "COMPUTE_TYPE",
    "DEFAULT_MODEL",
    "LANGUAGE",
    "load_model",
    "unload_model",
    "unload_all_models",
    "is_model_loaded",
    "get_model",
    "model_downloaded",
    "ensure_model_downloaded",
    "_model_cache",
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
