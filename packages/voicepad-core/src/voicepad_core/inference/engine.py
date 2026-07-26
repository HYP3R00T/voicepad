from __future__ import annotations

import logging
import time
from dataclasses import replace

import numpy as np

from .backend_manager import SessionManager
from .composition import deactivate_model, get_default_coordinator
from .constants import SAMPLE_RATE
from .contracts import RuntimeOptions, TranscriptionContext, TranscriptionRequest
from .errors import AudioTooShortError, TranscriptionError
from .types import TranscriptionResult
from ..config import Config, get_config
from ..models import resolve_model_spec
from ..postprocessing.hallucination import remove_hallucinations
from ..postprocessing.normalizer import normalize

logger = logging.getLogger(__name__)

_session_logger: logging.Logger | None = None


def set_session_logger(session_logger: logging.Logger | None) -> None:
    """Route detailed inference logs to the current transcription session."""
    global _session_logger
    _session_logger = session_logger


def close_default_sessions() -> None:
    """Close every session opened by the default inference engine."""
    deactivate_model()


def transcribe(
    audio: np.ndarray,
    model_name: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    word_timestamps: bool = False,
    language: str | None = None,
    initial_prompt: str | None = None,
    beam_size: int | None = None,
    vad_filter: bool | None = None,
    config: Config | None = None,
    session_manager: SessionManager | None = None,
) -> TranscriptionResult:
    """Transcribe canonical audio through the backend selected by the model."""
    resolved_config = config or get_config()
    resolved_model = model_name if model_name is not None else resolved_config.transcription_model
    resolved_device = device if device is not None else resolved_config.transcription_device
    resolved_precision = compute_type if compute_type is not None else resolved_config.transcription_compute_type
    resolved_language = language if language is not None else resolved_config.language
    resolved_beam_size = beam_size if beam_size is not None else resolved_config.beam_size
    resolved_vad_filter = vad_filter if vad_filter is not None else resolved_config.transcription_vad_filter
    context_text = resolved_config.initial_prompt if initial_prompt is None else initial_prompt
    started_at = time.perf_counter()

    _log_request(resolved_model, resolved_device, resolved_precision)
    if resolved_language != "en":
        message = (
            f"Language '{resolved_language}' selected. English is the primary supported language. "
            "Non-English results may have reduced accuracy."
        )
        logger.warning(message)
        if _session_logger:
            _session_logger.warning(message)

    canonical_audio = np.asarray(audio, dtype=np.float32)
    if canonical_audio.ndim > 1:
        canonical_audio = canonical_audio.flatten()

    canonical_audio = _trim_trailing_silence(
        canonical_audio,
        rms_threshold=resolved_config.trim_trailing_silence_rms_threshold,
        frame_ms=resolved_config.trim_trailing_silence_frame_ms,
    )
    duration_s = len(canonical_audio) / SAMPLE_RATE
    if duration_s < resolved_config.min_audio_duration_s:
        message = (
            f"Audio is {duration_s:.2f}s — below minimum {resolved_config.min_audio_duration_s}s. "
            f"Speak for at least {resolved_config.min_audio_duration_s:.1f} seconds."
        )
        if _session_logger:
            _session_logger.error(message)
        raise AudioTooShortError(message)

    model = resolve_model_spec(resolved_model)
    runtime_options = RuntimeOptions(device=resolved_device, precision=resolved_precision)
    if session_manager is not None:
        session = session_manager.open(model, runtime_options)
    else:
        session = get_default_coordinator(resolved_config.model_cache_path).activate(model, runtime_options)
    request = TranscriptionRequest(
        audio=canonical_audio,
        sample_rate=SAMPLE_RATE,
        language=resolved_language,
        word_timestamps=word_timestamps,
        beam_size=resolved_beam_size,
        context=TranscriptionContext(
            proper_nouns=resolved_config.proper_nouns,
            previous_text=context_text or None,
        ),
        vad_filter=resolved_vad_filter,
        no_speech_threshold=resolved_config.no_speech_threshold,
        hallucination_silence_threshold=resolved_config.hallucination_silence_threshold,
    )

    try:
        result = session.transcribe(request)
    except TranscriptionError:
        raise
    except Exception as exc:
        raise TranscriptionError(
            f"Backend '{model.backend_id}' failed to transcribe model '{model.id}': {exc}"
        ) from exc

    text = result.text
    if resolved_config.text_postprocessing_enabled:
        text = remove_hallucinations(
            text,
            max_repetitions=resolved_config.hallucination_max_repetitions,
        )
        text = normalize(text)
    result = replace(result, text=text)

    elapsed_ms = (time.perf_counter() - started_at) * 1000
    logger.info(
        "Transcribed %.1fs in %.0fms on %s (%s) with %s/%s",
        result.duration_s,
        elapsed_ms,
        result.device,
        result.compute_type,
        result.backend_id or model.backend_id,
        result.model_id or model.id,
    )
    if _session_logger:
        _session_logger.info(
            "Transcription completed: duration_s=%.1f latency_ms=%.0f backend=%s model=%s",
            result.duration_s,
            result.latency_ms,
            result.backend_id or model.backend_id,
            result.model_id or model.id,
        )
    return result


def _log_request(model_name: str, device: str, precision: str) -> None:
    if _session_logger:
        _session_logger.debug(
            "transcribe() called with model=%s, device=%s, compute=%s",
            model_name,
            device,
            precision,
        )


def _trim_trailing_silence(
    audio: np.ndarray,
    rms_threshold: float = 0.01,
    frame_ms: int = 20,
) -> np.ndarray:
    frame_size = int(SAMPLE_RATE * frame_ms / 1000)
    end = len(audio)

    while end > frame_size:
        frame = audio[end - frame_size : end]
        rms = float(np.sqrt(np.mean(frame**2)))
        if rms > rms_threshold:
            break
        end -= frame_size

    return audio[:end] if end < len(audio) else audio
