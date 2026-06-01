# postprocessing/normalizer.py

"""Minimal text normalisation for transcription output.

Handles whitespace cleanup and known Whisper output artefacts that
appear in raw transcription text. Intentionally kept minimal — no
grammar correction, no punctuation enforcement.

Public API:
    normalize(text) -> str
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Whisper artefacts — bracketed tokens it emits on silence or noise.
# Kept as a tuple so it is easy to extend.
_ARTEFACT_PATTERNS: tuple[str, ...] = (
    r"\[BLANK_AUDIO\]",
    r"\[MUSIC\]",
    r"\[NOISE\]",
    r"\[INAUDIBLE\]",
    r"\(MUSIC\)",
    r"\(NOISE\)",
    r"\(INAUDIBLE\)",
    r"\*\*\*",  # *** sometimes emitted on silence
    r"\.{4,}",  # four or more consecutive dots
)

# Pre-compiled single pattern for efficiency
_ARTEFACT_RE = re.compile(
    "|".join(_ARTEFACT_PATTERNS),
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    """Clean raw transcription text.

    Steps (in order):
      1. Strip leading/trailing whitespace.
      2. Remove known Whisper artefact tokens.
      3. Collapse multiple consecutive spaces into one.
      4. Strip again after artefact removal.

    Args:
        text: Raw transcription text.

    Returns:
        Normalised text string. Returns an empty string if the input
        is empty or becomes empty after cleaning.
    """
    if not text:
        return text

    # Step 1 — initial strip
    result = text.strip()

    # Step 2 — remove artefact tokens
    result = _ARTEFACT_RE.sub(" ", result)

    # Step 3 — collapse multiple spaces
    result = re.sub(r" {2,}", " ", result)

    # Step 4 — final strip
    result = result.strip()

    if not result:
        logger.debug("normalize: text became empty after cleaning (was: %r)", text[:80])

    return result
