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
import warnings

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
    MAX_AUDIO_DURATION_S,
    MIN_AUDIO_DURATION_S,
    NO_SPEECH_THRESHOLD,
    SAMPLE_RATE,
)
from .exceptions import AudioTooLongWarning, AudioTooShortError, TranscriptionError
from .model_manager import _is_cuda_error, _load_cpu_fallback, _model_cache, load
from .types import Segment, TranscriptionResult, WordTimestamp

logger = logging.getLogger(__name__)


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

    # --- Normalise input ---
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.flatten()

    audio = _trim_trailing_silence(audio)

    duration_s = len(audio) / SAMPLE_RATE

    if duration_s < MIN_AUDIO_DURATION_S:
        raise AudioTooShortError(
            f"Audio is {duration_s:.2f}s — below minimum {MIN_AUDIO_DURATION_S}s. Speak for at least 0.5 seconds."
        )

    if duration_s > MAX_AUDIO_DURATION_S:
        warnings.warn(
            f"Audio is {duration_s:.1f}s — exceeds recommended maximum "
            f"{MAX_AUDIO_DURATION_S}s. Transcription may be slow.",
            AudioTooLongWarning,
            stacklevel=2,
        )
        logger.info(f"Long audio: {duration_s:.1f}s — transcription will take a moment.")

    # --- Load model (cached after first call) ---
    model = load(model_name, device, compute_type)

    # --- Build prompt (distil models don't support initial_prompt) ---
    is_distil = model_name in DISTIL_MODELS
    prompt = None if is_distil else INITIAL_PROMPT

    fallback = False
    actual_device = device
    actual_compute = compute_type

    # --- Run inference ---
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
        segments = _build_segments(segments_raw, duration_s, word_timestamps)

    except RuntimeError as e:
        if _is_cuda_error(e):
            logger.warning(f"CUDA inference error: {e} — retrying on CPU.")

            # Evict the broken GPU entry from cache
            _model_cache.pop((model_name, device, compute_type), None)

            cpu_model = _load_cpu_fallback(model_name)
            fallback = True
            actual_device = "cpu"
            actual_compute = "int8"

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
            segments = _build_segments(segments_raw, duration_s, word_timestamps)

        else:
            raise TranscriptionError(f"Transcription failed: {e}") from e

    except Exception as e:
        raise TranscriptionError(f"Transcription failed: {e}") from e

    # --- Post-process text ---
    text = " ".join(s.text for s in segments if s.text).strip()
    text = _remove_hallucinated_repetitions(text)

    # --- Compute quality metrics ---
    avg_confidence = sum(s.avg_logprob for s in segments) / len(segments) if segments else 0.0
    low_confidence_count = sum(1 for s in segments if s.avg_logprob < -1.0)

    latency_ms = (time.perf_counter() - call_start) * 1000

    logger.info(
        f"Transcribed {duration_s:.1f}s in {latency_ms:.0f}ms "
        f"on {actual_device} ({actual_compute}) — "
        f"{len(segments)} segments, avg_conf={avg_confidence:.2f}, "
        f"low_conf={low_confidence_count}"
    )

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

    speech_pad_ms is kept at 500ms (reduced from 1000ms in the old codebase)
    to minimise overlap duplication in chunked transcription.

    Returns:
        Dictionary of VAD parameters accepted by faster-whisper.
    """
    return {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "max_speech_duration_s": 15.0,
        "min_silence_duration_ms": 1000,
        "speech_pad_ms": 500,
    }


# ---------------------------------------------------------------------------
# Internal — hallucination removal
# ---------------------------------------------------------------------------


def _remove_hallucinated_repetitions(text: str, max_repetitions: int = 3) -> str:
    """Remove hallucinated repetitive patterns from transcription output.

    Whisper occasionally repeats the same word or short phrase many times,
    especially on silence or low-quality audio. This function:
      1. Detects single-word repetitions exceeding max_repetitions.
      2. Detects 2-word phrase repetitions (3+ consecutive occurrences).
      3. Keeps at most max_repetitions copies of any repeated token.

    Args:
        text:            Raw transcription text.
        max_repetitions: Maximum allowed consecutive repetitions.

    Returns:
        Cleaned text with excess repetitions removed.
    """
    if not text:
        return text

    words = text.split()
    if len(words) < max_repetitions + 1:
        return text

    # --- Pass 1: single-word repetitions ---
    cleaned: list[str] = []
    i = 0
    while i < len(words):
        word = words[i]
        count = 1
        while i + count < len(words) and words[i + count].lower() == word.lower():
            count += 1
        if count > max_repetitions:
            cleaned.extend([word] * max_repetitions)
            i += count
        else:
            cleaned.append(word)
            i += 1

    # --- Pass 2: 2-word phrase repetitions ---
    words = cleaned
    final: list[str] = []
    i = 0
    while i < len(words):
        if i + 5 < len(words):
            phrase = f"{words[i]} {words[i + 1]}"
            phrase2 = f"{words[i + 2]} {words[i + 3]}"
            phrase3 = f"{words[i + 4]} {words[i + 5]}"
            if phrase.lower() == phrase2.lower() == phrase3.lower():
                final.extend([words[i], words[i + 1]])
                j = i + 6
                while j + 1 < len(words):
                    if f"{words[j]} {words[j + 1]}".lower() == phrase.lower():
                        j += 2
                    else:
                        break
                i = j
                continue
        final.append(words[i])
        i += 1

    return " ".join(final)
