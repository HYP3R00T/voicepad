"""Core transcription functions."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf

from .constants import (
    BEAM_SIZE,
    DISTIL_MODELS,
    HALLUCINATION_SILENCE_THRESHOLD,
    INITIAL_PROMPT,
    LANGUAGE,
    MAX_AUDIO_DURATION_S,
    MIN_AUDIO_DURATION_S,
    NO_SPEECH_THRESHOLD,
)
from .exceptions import AudioTooShortError, TranscriptionError
from .model_manager import _is_cuda_error, _load_cpu_fallback, get_or_load_model
from .types import TranscriptionResult
from .utils import _filter_segments, _get_vad_parameters, _trim_trailing_silence

if TYPE_CHECKING:
    from voicepad_core.config import Config

logger = logging.getLogger(__name__)


def transcribe_buffer(audio: np.ndarray, config: Config) -> TranscriptionResult:
    """Transcribe audio from numpy array.

    Args:
        audio: float32 array at 16 kHz mono
        config: Configuration with model settings

    Returns:
        TranscriptionResult with text, segments, timing, and metadata

    Raises:
        AudioTooShortError: If audio below minimum duration
        TranscriptionError: If transcription fails
    """

    call_start = time.perf_counter()

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.flatten()

    audio = _trim_trailing_silence(audio)

    duration_s = len(audio) / 16000

    if duration_s < MIN_AUDIO_DURATION_S:
        raise AudioTooShortError(f"Audio is {duration_s:.2f}s — below minimum {MIN_AUDIO_DURATION_S}s")

    if duration_s > MAX_AUDIO_DURATION_S:
        logger.info(f"Long recording: {duration_s:.1f}s — transcription will take a moment")

    model, device, compute, fallback = get_or_load_model(config)

    is_distil = config.transcription_model in DISTIL_MODELS
    prompt = None if is_distil else INITIAL_PROMPT

    try:
        vad_params = _get_vad_parameters()

        segments_iter, info = model.transcribe(
            audio,
            language=LANGUAGE,
            beam_size=BEAM_SIZE,
            vad_filter=True,
            vad_parameters=vad_params,
            hallucination_silence_threshold=HALLUCINATION_SILENCE_THRESHOLD,
            no_speech_threshold=NO_SPEECH_THRESHOLD,
            initial_prompt=prompt,
            condition_on_previous_text=False,
        )
        segments = _filter_segments(segments_iter, duration_s)

    except RuntimeError as e:
        if _is_cuda_error(e):
            logger.warning(f"CUDA inference error: {e} — retrying on CPU")
            from .model_manager import _model_cache

            _model_cache.pop((config.transcription_model, device, compute), None)
            download_root_fb: str | None = None
            if hasattr(config, "model_cache_path"):
                download_root_fb = str(config.model_cache_path / "hub")
            cpu_model, device, compute = _load_cpu_fallback(config.transcription_model, download_root_fb)
            fallback = True
            segments_iter, info = cpu_model.transcribe(
                audio,
                language=LANGUAGE,
                beam_size=BEAM_SIZE,
                vad_filter=True,
                vad_parameters=vad_params,
                hallucination_silence_threshold=HALLUCINATION_SILENCE_THRESHOLD,
                no_speech_threshold=NO_SPEECH_THRESHOLD,
                initial_prompt=prompt,
                condition_on_previous_text=False,
            )
            segments = _filter_segments(segments_iter, duration_s)
        else:
            raise TranscriptionError(f"Transcription failed: {e}") from e
    except Exception as e:
        raise TranscriptionError(f"Transcription failed: {e}") from e

    text = " ".join(s.text for s in segments if s.text).strip()

    avg_confidence = sum(s.avg_logprob for s in segments) / len(segments) if segments else 0.0
    low_confidence_segments = sum(1 for s in segments if s.avg_logprob < -1.0)

    latency_ms = (time.perf_counter() - call_start) * 1000

    logger.info(
        f"Transcribed {duration_s:.1f}s in {latency_ms:.0f}ms on {device} ({compute}) — "
        f"{len(segments)} segments, avg confidence: {avg_confidence:.2f}, "
        f"low confidence: {low_confidence_segments}"
    )

    return TranscriptionResult(
        text=text,
        segments=segments,
        language=info.language,
        language_probability=info.language_probability,
        duration_s=duration_s,
        latency_ms=latency_ms,
        device=device,
        compute_type=compute,
        fallback_to_cpu=fallback,
        avg_confidence=avg_confidence,
        low_confidence_segments=low_confidence_segments,
    )


def transcribe_file(audio_path: Path, config: Config) -> TranscriptionResult:
    """Load audio file and transcribe with transcribe_buffer()."""
    if not audio_path.exists():
        raise TranscriptionError(f"Audio file not found: {audio_path}")
    if not audio_path.is_file():
        raise TranscriptionError(f"Path is not a file: {audio_path}")

    try:
        audio, file_sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    except Exception as e:
        raise TranscriptionError(f"Failed to read '{audio_path}': {e}") from e

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if file_sr != 16000:
        logger.warning(
            f"File sample rate is {file_sr} Hz, expected 16000 Hz — quality may be reduced (no resampling performed)"
        )

    return transcribe_buffer(audio, config)
