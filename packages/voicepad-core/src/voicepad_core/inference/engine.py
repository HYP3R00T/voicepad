# inference/engine.py

"""Core transcription engine.

Accepts a pre-chunked float32 audio array at 16kHz and returns a
TranscriptionResult. Chunking is the caller's responsibility — the engine
transcribes exactly what it receives.

Public API:
    transcribe(audio, model_name, ...)  -> TranscriptionResult
"""

from __future__ import annotations

import logging
import time

import numpy as np

from .constants import (
    BEAM_SIZE,
    COMPUTE_TYPE,
    DEFAULT_MODEL,
    DEVICE,
    DISTIL_MODELS,
    HALLUCINATION_SILENCE_THRESHOLD,
    INITIAL_PROMPT,
    LANGUAGE,
    MIN_AUDIO_DURATION_S,
    NO_SPEECH_THRESHOLD,
    SAMPLE_RATE,
)
from .exceptions import AudioTooShortError, TranscriptionError
from .model_manager import (
    _is_cuda_error,
    _load_cpu_fallback,
    _model_cache,
    load,
    set_model_manager_session_logger,
)
from .types import Segment, TranscriptionResult, WordTimestamp

# Post-processing is imported here so the engine applies the full pipeline.
# _trim_trailing_silence lives in this module because it is a pre-inference
# concern (prevents hallucinations on quiet tails), not a text post-process.
from ..postprocessing.hallucination import remove_hallucinations
from ..postprocessing.normalizer import normalize

logger = logging.getLogger(__name__)

# Session logger for detailed per-transcription logging
_session_logger: logging.Logger | None = None


def set_session_logger(session_logger: logging.Logger | None) -> None:
    """Set the session logger for detailed transcription logging.

    Args:
        session_logger: Logger instance for the current transcription session
    """
    global _session_logger
    _session_logger = session_logger

    # Also set it for the model manager
    set_model_manager_session_logger(session_logger)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def transcribe(
    audio: np.ndarray,
    model_name: str = DEFAULT_MODEL,
    device: str = DEVICE,
    compute_type: str = COMPUTE_TYPE,
    word_timestamps: bool = False,
    language: str = LANGUAGE,
    initial_prompt: str | None = None,
) -> TranscriptionResult:
    """Transcribe a pre-chunked audio buffer to text.

    The caller is responsible for chunking. This function transcribes
    exactly the audio it receives and returns a fully populated
    TranscriptionResult.

    Args:
        audio:           float32 mono numpy array at 16kHz.
        model_name:      Whisper model to use.
        device:          'cuda' or 'cpu'.
        compute_type:    CTranslate2 precision string.
        word_timestamps: If True, populate Segment.words with per-word timing.
        language:        BCP-47 language code. Defaults to 'en'.
                         NOTE: English is the primary supported language.
                         Non-English results may have reduced accuracy.

    Returns:
        TranscriptionResult with text, segments, timing, and metadata.

    Raises:
        AudioTooShortError: If audio is below MIN_AUDIO_DURATION_S.
        TranscriptionError: If transcription fails on all devices.
    """
    call_start = time.perf_counter()
    slog = _session_logger  # Use session logger if available

    if slog:
        slog.debug(f"transcribe() called with model={model_name}, device={device}, compute={compute_type}")

    # --- Language warning for non-English ---
    if language != "en":
        msg = (
            f"Language '{language}' selected. English is the primary supported language. "
            "Non-English results may have reduced accuracy."
        )
        logger.warning(msg)
        if slog:
            slog.warning(msg)

    # --- Normalise input ---
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        if slog:
            slog.debug(f"Audio has {audio.ndim} dimensions, flattening to mono")
        audio = audio.flatten()

    if slog:
        slog.debug(f"Audio shape before trim: {audio.shape}")

    audio = _trim_trailing_silence(audio)

    if slog:
        slog.debug(f"Audio shape after trim: {audio.shape}")

    duration_s = len(audio) / SAMPLE_RATE

    if slog:
        slog.info(f"Audio duration: {duration_s:.2f}s (min={MIN_AUDIO_DURATION_S}s)")

    if duration_s < MIN_AUDIO_DURATION_S:
        msg = f"Audio is {duration_s:.2f}s — below minimum {MIN_AUDIO_DURATION_S}s. Speak for at least 0.5 seconds."
        if slog:
            slog.error(msg)
        raise AudioTooShortError(msg)

    # No finite total-duration cap; chunking is handled by the streaming layer.

    # --- Load model (cached after first call) ---
    if slog:
        slog.info(f"Loading model: {model_name} on {device} ({compute_type})")

    model_load_start = time.perf_counter()
    model = load(model_name, device, compute_type)
    model_load_time = (time.perf_counter() - model_load_start) * 1000

    if slog:
        slog.info(f"Model loaded in {model_load_time:.0f}ms")

    # --- Build prompt (distil models don't support initial_prompt) ---
    is_distil = model_name in DISTIL_MODELS
    prompt = None if is_distil else (initial_prompt if initial_prompt is not None else INITIAL_PROMPT)

    if slog:
        slog.debug(f"Using prompt: {prompt if prompt else '(none - distil model)'}")

    fallback = False
    actual_device = device
    actual_compute = compute_type

    # --- Run inference ---
    if slog:
        slog.info("Starting inference...")

    inference_start = time.perf_counter()

    try:
        segments_raw, info = model.transcribe(
            audio,
            language=language,
            beam_size=BEAM_SIZE,
            vad_filter=True,
            vad_parameters=_vad_parameters(),
            hallucination_silence_threshold=HALLUCINATION_SILENCE_THRESHOLD,
            no_speech_threshold=NO_SPEECH_THRESHOLD,
            initial_prompt=prompt,
            condition_on_previous_text=False,
            word_timestamps=word_timestamps,
        )

        inference_time = (time.perf_counter() - inference_start) * 1000
        if slog:
            slog.info(f"Inference completed in {inference_time:.0f}ms")

        segments = _build_segments(segments_raw, duration_s, word_timestamps)

        if slog:
            slog.info(f"Built {len(segments)} segments")

    except RuntimeError as e:
        if _is_cuda_error(e):
            msg = f"CUDA inference error: {e} — retrying on CPU."
            logger.warning(msg)
            if slog:
                slog.warning(msg)

            # Evict the broken GPU entry from cache
            _model_cache.pop((model_name, device, compute_type), None)

            if slog:
                slog.info("Loading CPU fallback model...")

            cpu_model = _load_cpu_fallback(model_name)
            fallback = True
            actual_device = "cpu"
            actual_compute = "int8"

            if slog:
                slog.info("Retrying inference on CPU...")

            inference_start = time.perf_counter()

            segments_raw, info = cpu_model.transcribe(
                audio,
                language=language,
                beam_size=BEAM_SIZE,
                vad_filter=True,
                vad_parameters=_vad_parameters(),
                hallucination_silence_threshold=HALLUCINATION_SILENCE_THRESHOLD,
                no_speech_threshold=NO_SPEECH_THRESHOLD,
                initial_prompt=prompt,
                condition_on_previous_text=False,
                word_timestamps=word_timestamps,
            )

            inference_time = (time.perf_counter() - inference_start) * 1000
            if slog:
                slog.info(f"CPU inference completed in {inference_time:.0f}ms")

            segments = _build_segments(segments_raw, duration_s, word_timestamps)

            if slog:
                slog.info(f"Built {len(segments)} segments (CPU fallback)")

        else:
            msg = f"Transcription failed: {e}"
            if slog:
                slog.error(msg)
            raise TranscriptionError(msg) from e

    except Exception as e:
        msg = f"Transcription failed: {e}"
        if slog:
            slog.error(msg)
        raise TranscriptionError(msg) from e

    # --- Post-process text ---
    if slog:
        slog.debug("Starting post-processing...")

    text = " ".join(s.text for s in segments if s.text).strip()

    if slog:
        slog.debug(f"Raw text length: {len(text)} characters")

    text = remove_hallucinations(text)
    text = normalize(text)

    if slog:
        slog.debug(f"Post-processed text length: {len(text)} characters")

    # --- Compute quality metrics ---
    avg_confidence = sum(s.avg_logprob for s in segments) / len(segments) if segments else 0.0
    low_confidence_count = sum(1 for s in segments if s.avg_logprob < -1.0)

    latency_ms = (time.perf_counter() - call_start) * 1000

    msg = (
        f"Transcribed {duration_s:.1f}s in {latency_ms:.0f}ms "
        f"on {actual_device} ({actual_compute}) — "
        f"{len(segments)} segments, avg_conf={avg_confidence:.2f}, "
        f"low_conf={low_confidence_count}"
    )
    logger.info(msg)
    if slog:
        slog.info(msg)
        slog.info(f"Transcription result: '{text[:100]}{'...' if len(text) > 100 else ''}'")

    return TranscriptionResult(
        text=text,
        segments=segments,
        language=info.language,
        language_probability=info.language_probability,
        duration_s=duration_s,
        latency_ms=latency_ms,
        device=actual_device,
        compute_type=actual_compute,
        fallback_to_cpu=fallback,
        avg_confidence=avg_confidence,
        low_confidence_count=low_confidence_count,
    )


# ---------------------------------------------------------------------------
# Internal — segment building
# ---------------------------------------------------------------------------


def _build_segments(
    segments_iter,
    duration_s: float,
    word_timestamps: bool,
) -> list[Segment]:
    """Materialise the lazy segments iterator and filter bad segments.

    Args:
        segments_iter:   Lazy iterator from model.transcribe().
        duration_s:      Total audio duration for boundary checks.
        word_timestamps: Whether to populate Segment.words.

    Returns:
        List of clean Segment objects.
    """
    result: list[Segment] = []

    for s in segments_iter:
        # Drop segments that are entirely outside the audio duration
        if s.start >= duration_s:
            continue

        # Drop segments where Whisper is almost certain there's no speech
        if s.no_speech_prob > NO_SPEECH_THRESHOLD:
            continue

        words: list[WordTimestamp] = []
        if word_timestamps and s.words:
            words = [
                WordTimestamp(
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    probability=w.probability,
                )
                for w in s.words
            ]

        result.append(
            Segment(
                start=s.start,
                end=min(s.end, duration_s),
                text=s.text.strip(),
                avg_logprob=s.avg_logprob,
                no_speech_prob=s.no_speech_prob,
                words=words,
            )
        )

    return result


# ---------------------------------------------------------------------------
# Internal — audio utilities
# ---------------------------------------------------------------------------


def _trim_trailing_silence(
    audio: np.ndarray,
    rms_threshold: float = 0.01,
    frame_ms: int = 20,
) -> np.ndarray:
    """Trim silent frames from the end of an audio array.

    Scans backwards in 20ms frames. Stops at the first frame whose RMS
    energy is above rms_threshold and returns the array up to that point.

    Args:
        audio:         float32 mono audio at 16kHz.
        rms_threshold: Frames below this RMS are considered silent.
        frame_ms:      Frame size in milliseconds.

    Returns:
        Trimmed audio array. Original array returned if no silence found.
    """
    frame_size = int(SAMPLE_RATE * frame_ms / 1000)
    end = len(audio)

    while end > frame_size:
        frame = audio[end - frame_size : end]
        rms = float(np.sqrt(np.mean(frame**2)))
        if rms > rms_threshold:
            break
        end -= frame_size

    return audio[:end] if end < len(audio) else audio


# ---------------------------------------------------------------------------
# Internal — VAD parameters
# ---------------------------------------------------------------------------


def _vad_parameters() -> dict[str, float | int]:
    """Return standard VAD parameters for consistent transcription quality.

    `max_speech_duration_s` is kept near Whisper's context window so a long
    uninterrupted utterance is not artificially capped at 15 seconds.
    `speech_pad_ms` is kept at 500ms (reduced from 1000ms in the old codebase)
    to minimise overlap duplication in chunked transcription.

    Returns:
        Dictionary of VAD parameters accepted by faster-whisper.
    """
    return {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "max_speech_duration_s": 29.0,
        "min_silence_duration_ms": 1000,
        "speech_pad_ms": 500,
    }
