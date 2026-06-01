# postprocessing/filters.py

"""Segment-level filtering.

Receives a raw segment iterator from faster-whisper and returns a clean
list of Segment objects, dropping anything that is out-of-bounds, too
short, or statistically unlikely to contain real speech.

Public API:
    filter_segments(segments_iter, duration_s) -> list[Segment]
"""

from __future__ import annotations

import logging

from ..inference.types import Segment

logger = logging.getLogger(__name__)


def filter_segments(segments_iter, duration_s: float) -> list[Segment]:
    """Filter and materialise a raw faster-whisper segment iterator.

    Drops segments that are:
      - Starting at or beyond the total audio duration (out-of-bounds).
      - Very short (< 0.5s) and near the tail of the audio (last 1s).
        These are almost always silence artefacts or hallucinations.
      - Above the no-speech probability threshold (0.9), meaning Whisper
        itself is confident there is no real speech in that segment.

    Args:
        segments_iter: Lazy iterator of raw segments from model.transcribe().
        duration_s:    Total audio duration in seconds. Used for boundary
                       checks and tail-segment detection.

    Returns:
        List of clean, filtered Segment objects with stripped text.
    """
    segments: list[Segment] = []
    dropped = 0

    for s in segments_iter:
        # Drop segments that start at or beyond the audio boundary
        if s.start >= duration_s:
            dropped += 1
            continue

        seg_duration = s.end - s.start

        # Drop suspiciously short segments near the tail
        if s.end > duration_s - 1.0 and seg_duration < 0.5:
            dropped += 1
            continue

        # Drop segments where Whisper is near-certain there's no speech
        if s.no_speech_prob > 0.9:
            dropped += 1
            continue

        segments.append(
            Segment(
                start=s.start,
                end=min(s.end, duration_s),
                text=s.text.strip(),
                avg_logprob=s.avg_logprob,
                no_speech_prob=s.no_speech_prob,
                words=s.words if hasattr(s, "words") and s.words else [],
            )
        )

    if dropped:
        logger.debug(f"filter_segments: dropped {dropped} segment(s) out of {len(segments) + dropped} total.")

    return segments
