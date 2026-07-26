from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import cast

import numpy as np

from .contracts import BackendCapabilities, RuntimeOptions, TranscriptionContext, TranscriptionRequest
from .errors import AudioTooShortError, TranscriptionError
from .runtime import RuntimeManager, get_runtime_manager
from .types import TranscriptionResult
from ..audio import RawAudio
from ..config import Config, get_config
from ..models import get_model
from ..postprocessing.hallucination import remove_hallucinations
from ..postprocessing.normalizer import normalize
from ..preprocessing import AudioPreProcessor

logger = logging.getLogger(__name__)

_session_logger: logging.Logger | None = None


def set_session_logger(session_logger: logging.Logger | None) -> None:
    """Route detailed inference logs to the current transcription session."""
    global _session_logger
    _session_logger = session_logger


def transcribe(
    audio: RawAudio,
    model_name: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    word_timestamps: bool = False,
    language: str | None = None,
    initial_prompt: str | None = None,
    beam_size: int | None = None,
    vad_filter: bool | None = None,
    config: Config | None = None,
    runtime_manager: RuntimeManager | None = None,
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
    model = get_model(resolved_model)
    runtime_options = RuntimeOptions(device=resolved_device, precision=resolved_precision)
    runtime = runtime_manager or get_runtime_manager(resolved_config.model_cache_path)
    contract = runtime.contract(model)

    prepared_audio = AudioPreProcessor.prepare(audio, contract.audio)
    canonical_audio = _trim_trailing_silence(
        prepared_audio.samples,
        rms_threshold=resolved_config.trim_trailing_silence_rms_threshold,
        frame_ms=resolved_config.trim_trailing_silence_frame_ms,
        sample_rate=prepared_audio.sample_rate,
    )
    duration_s = len(canonical_audio) / prepared_audio.sample_rate
    if duration_s < resolved_config.min_audio_duration_s:
        message = (
            f"Audio is {duration_s:.2f}s — below minimum {resolved_config.min_audio_duration_s}s. "
            f"Speak for at least {resolved_config.min_audio_duration_s:.1f} seconds."
        )
        if _session_logger:
            _session_logger.error(message)
        raise AudioTooShortError(message)

    session = runtime.open(model, runtime_options)

    request, applied_options, ignored_options = _build_request(
        audio=canonical_audio,
        sample_rate=prepared_audio.sample_rate,
        language=resolved_language,
        word_timestamps=word_timestamps,
        beam_size=resolved_beam_size,
        proper_nouns=resolved_config.proper_nouns,
        previous_text=context_text or None,
        vad_filter=resolved_vad_filter,
        no_speech_threshold=resolved_config.no_speech_threshold,
        hallucination_silence_threshold=resolved_config.hallucination_silence_threshold,
        capabilities=contract.decoding,
    )

    try:
        result = session.transcribe(request)
    except TranscriptionError:
        raise
    except Exception as exc:
        raise TranscriptionError(f"Backend '{model.backend}' failed to transcribe model '{model.id}': {exc}") from exc

    text = result.text
    if resolved_config.text_postprocessing_enabled:
        text = remove_hallucinations(
            text,
            max_repetitions=resolved_config.hallucination_max_repetitions,
        )
        text = normalize(text)
    output = contract.output
    result = replace(
        result,
        text=text,
        audio_transformations=prepared_audio.transformations,
        applied_options=applied_options,
        ignored_options=ignored_options,
        language_source=output.language,
        word_timestamp_source=output.word_timestamps,
        word_confidence_source=output.word_confidence,
        segment_log_probability_source=output.segment_log_probability,
        segment_confidence_source=output.segment_confidence,
        no_speech_probability_source=output.no_speech_probability,
    )

    elapsed_ms = (time.perf_counter() - started_at) * 1000
    logger.info(
        "Transcribed %.1fs in %.0fms on %s (%s) with %s/%s",
        result.duration_s,
        elapsed_ms,
        result.device,
        result.compute_type,
        result.backend_id or model.backend,
        result.model_id or model.id,
    )
    if _session_logger:
        _session_logger.info(
            "Transcription completed: duration_s=%.1f latency_ms=%.0f backend=%s model=%s",
            result.duration_s,
            result.latency_ms,
            result.backend_id or model.backend,
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
    sample_rate: int,
    rms_threshold: float = 0.01,
    frame_ms: int = 20,
) -> np.ndarray:
    frame_size = int(sample_rate * frame_ms / 1000)
    end = len(audio)

    while end > frame_size:
        frame = audio[end - frame_size : end]
        rms = float(np.sqrt(np.mean(frame**2)))
        if rms > rms_threshold:
            break
        end -= frame_size

    return audio[:end] if end < len(audio) else audio


def _build_request(
    *,
    audio: np.ndarray,
    sample_rate: int,
    language: str | None,
    word_timestamps: bool,
    beam_size: int,
    proper_nouns: tuple[str, ...],
    previous_text: str | None,
    vad_filter: bool,
    no_speech_threshold: float,
    hallucination_silence_threshold: float,
    capabilities: BackendCapabilities,
) -> tuple[TranscriptionRequest, tuple[str, ...], tuple[str, ...]]:
    applied: list[str] = []
    ignored: list[str] = []

    def supported(name: str, value: object, available: bool, requested: bool = True) -> object | None:
        if not requested:
            return None
        (applied if available else ignored).append(name)
        return value if available else None

    selected_language = supported("language", language, capabilities.language_selection, language is not None)
    selected_beam = supported("beam_size", beam_size, capabilities.beam_search)
    selected_vad = supported("vad_filter", vad_filter, capabilities.vad_filter, vad_filter)
    selected_no_speech = supported("no_speech_threshold", no_speech_threshold, capabilities.no_speech_threshold)
    selected_hallucination = supported(
        "hallucination_silence_threshold",
        hallucination_silence_threshold,
        capabilities.hallucination_silence_threshold,
    )
    selected_nouns = supported(
        "proper_nouns",
        proper_nouns,
        capabilities.context_biasing,
        bool(proper_nouns),
    )
    selected_previous = supported(
        "previous_text",
        previous_text,
        capabilities.text_prompt,
        previous_text is not None,
    )
    selected_words = supported(
        "word_timestamps",
        word_timestamps,
        capabilities.word_timestamps,
        word_timestamps,
    )

    request = TranscriptionRequest(
        audio=audio,
        sample_rate=sample_rate,
        language=selected_language if isinstance(selected_language, str) else None,
        word_timestamps=bool(selected_words),
        beam_size=selected_beam if isinstance(selected_beam, int) else None,
        context=TranscriptionContext(
            proper_nouns=cast(tuple[str, ...], selected_nouns) if isinstance(selected_nouns, tuple) else (),
            previous_text=selected_previous if isinstance(selected_previous, str) else None,
        ),
        vad_filter=selected_vad if isinstance(selected_vad, bool) else None,
        no_speech_threshold=selected_no_speech if isinstance(selected_no_speech, float) else None,
        hallucination_silence_threshold=(selected_hallucination if isinstance(selected_hallucination, float) else None),
    )
    return request, tuple(applied), tuple(ignored)
