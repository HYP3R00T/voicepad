# postprocessing/deduplication.py

"""Overlap deduplication for chunked transcription.

When audio is processed in overlapping chunks, Whisper often re-transcribes
text from the previous chunk's overlap window. This module detects and removes
those duplicates by comparing the overlap region against the tail of the
previous chunk's text.

Public API:
    deduplicate_overlap(segments, chunk_start_s, prev_text) -> list[Segment]
"""

from __future__ import annotations

import difflib
import logging

from ..inference.types import Segment

logger = logging.getLogger(__name__)

# How many words from the end of the previous chunk to compare against.
_PREV_TAIL_WORDS: int = 50

# Similarity ratio above which the overlap is considered a full duplicate.
_FULL_DUPLICATE_THRESHOLD: float = 0.8

# Minimum number of overlap words needed for partial word-level dedup.
_MIN_OVERLAP_WORDS_FOR_PARTIAL: int = 3

# Number of leading overlap words to check for partial match.
_PARTIAL_LEAD_WORDS: int = 5


def deduplicate_overlap(
    segments: list[Segment],
    chunk_start_s: float,
    prev_text: str,
    *,
    prev_tail_words: int = _PREV_TAIL_WORDS,
    full_duplicate_threshold: float = _FULL_DUPLICATE_THRESHOLD,
    min_overlap_words_for_partial: int = _MIN_OVERLAP_WORDS_FOR_PARTIAL,
    partial_lead_words: int = _PARTIAL_LEAD_WORDS,
) -> list[Segment]:
    """Remove duplicated segments from the overlap region of a chunk.

    Splits incoming segments into two groups:
      - overlap_segments : segments whose start time is before chunk_start_s
                           (i.e. inside the overlap window of the previous chunk)
      - non_overlap_segments : everything else

    Compares the overlap region text to the tail of prev_text using
    difflib.SequenceMatcher. Three outcomes are possible:

      1. Full duplicate (similarity >= 0.8)
         → Drop all overlap_segments, keep non_overlap_segments only.

      2. Partial duplicate (first N words of overlap found in prev_tail)
         → Drop only the overlap_segments whose text already appears in
           prev_tail; keep the rest plus non_overlap_segments.

      3. No duplicate detected
         → Return all segments unchanged.

    Args:
        segments:      All segments from the current chunk.
        chunk_start_s: The logical start time of this chunk (excluding overlap).
                       Segments before this time are in the overlap window.
        prev_text:     Full text from the previous chunk, used for comparison.

    Returns:
        Deduplicated list of Segment objects.
    """
    if not segments or not prev_text:
        return segments

    # Build comparison tail from the last N words of prev_text
    prev_words = prev_text.split()[-prev_tail_words:]
    prev_tail = " ".join(prev_words).lower()

    # Partition segments into overlap and non-overlap regions
    overlap_segments = [s for s in segments if s.start < chunk_start_s]
    non_overlap_segments = [s for s in segments if s.start >= chunk_start_s]

    if not overlap_segments:
        return segments

    overlap_text = " ".join(s.text for s in overlap_segments if s.text).strip().lower()

    matcher = difflib.SequenceMatcher(None, prev_tail, overlap_text)
    similarity = matcher.ratio()

    if similarity >= full_duplicate_threshold:
        logger.debug(
            f"deduplicate_overlap: full duplicate detected "
            f"(similarity={similarity:.2f}) — dropped {len(overlap_segments)} overlap segment(s)."
        )
        return non_overlap_segments

    overlap_words = overlap_text.split()
    if len(overlap_words) >= min_overlap_words_for_partial:
        lead = " ".join(overlap_words[:partial_lead_words])
        if lead in prev_tail:
            kept = [s for s in overlap_segments if s.text.strip().lower() not in prev_tail]
            removed = len(overlap_segments) - len(kept)
            if removed:
                logger.debug(
                    f"deduplicate_overlap: partial duplicate — removed {removed} overlap segment(s), kept {len(kept)}."
                )
            return kept + non_overlap_segments

    return segments
