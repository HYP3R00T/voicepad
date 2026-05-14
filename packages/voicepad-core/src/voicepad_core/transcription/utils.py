"""Utility functions for audio processing and segment filtering."""

from __future__ import annotations

import numpy as np

from .types import Segment


def _trim_trailing_silence(
    audio: np.ndarray, sr: int = 16000, threshold: float = 0.003, window_s: float = 0.2
) -> np.ndarray:
    """Remove trailing silence to prevent hallucinations.

    Args:
        audio: Audio array to trim
        sr: Sample rate in Hz
        threshold: RMS threshold for silence detection (lower preserves quiet speech)
        window_s: Scan window size in seconds

    Returns:
        Trimmed audio array
    """
    window = int(window_s * sr)
    if len(audio) <= window:
        return audio
    end = len(audio)
    while end > window:
        rms = float(np.sqrt(np.mean(audio[end - window : end] ** 2)))
        if rms > threshold:
            end = min(len(audio), end + window // 2)
            break
        end -= window // 2
    return audio[:end]


def _filter_segments(segments_iter, duration_s: float) -> list[Segment]:
    """Filter and convert raw segments to Segment objects.

    Removes segments beyond audio duration and suspicious short segments at the end.

    Args:
        segments_iter: Iterator of raw segments from faster-whisper
        duration_s: Total audio duration in seconds

    Returns:
        List of filtered Segment objects
    """
    segments = []
    for s in segments_iter:
        if s.start >= duration_s:
            continue
        seg_duration = s.end - s.start
        if s.end > duration_s - 1.0 and seg_duration < 0.5:
            continue
        segments.append(
            Segment(
                start=s.start,
                end=s.end,
                text=s.text.strip(),
                avg_logprob=s.avg_logprob,
                no_speech_prob=s.no_speech_prob,
            )
        )
    return segments


def _get_vad_parameters() -> dict[str, float | int]:
    """Get standard VAD parameters for consistent transcription quality.

    Returns:
        Dictionary of VAD parameters for faster-whisper
    """
    return {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "max_speech_duration_s": 15.0,
        "min_silence_duration_ms": 1000,
        "speech_pad_ms": 1000,
    }
