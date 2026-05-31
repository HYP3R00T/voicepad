# postprocessing/hallucination.py

"""Text-level hallucination removal.

Whisper occasionally hallucinates by repeating the same word or short phrase
many times in a row, especially on silence or low-quality audio. This module
detects and removes those excess repetitions at the text level.

Public API:
    remove_hallucinations(text, max_repetitions) -> str
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Maximum allowed consecutive repetitions of any word or phrase before
# the excess copies are removed.
_DEFAULT_MAX_REPETITIONS: int = 3


def remove_hallucinations(
    text: str,
    max_repetitions: int = _DEFAULT_MAX_REPETITIONS,
) -> str:
    """Remove hallucinated repetitive patterns from transcription text.

    Runs two passes:
      Pass 1 — Single-word repetitions
        e.g. "the the the the cat" → "the the the cat"

      Pass 2 — Two-word phrase repetitions (requires 3+ consecutive occurrences)
        e.g. "thank you thank you thank you for coming" → "thank you for coming"

    Each pass keeps at most max_repetitions copies of any repeated token.

    Args:
        text:            Raw transcription text to clean.
        max_repetitions: Maximum consecutive repetitions allowed before
                         excess copies are removed. Defaults to 3.

    Returns:
        Cleaned text with hallucinated repetitions removed.
        Returns the original string unchanged if it is empty or too short
        to contain hallucinations.
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
            logger.debug(f"remove_hallucinations: word '{word}' repeated {count}x — keeping {max_repetitions}.")
            cleaned.extend([word] * max_repetitions)
            i += count
        else:
            cleaned.append(word)
            i += 1

    # --- Pass 2: two-word phrase repetitions ---
    words = cleaned
    final: list[str] = []
    i = 0
    while i < len(words):
        # Need at least 3 consecutive occurrences of a 2-word phrase
        if i + 5 < len(words):
            phrase = f"{words[i]} {words[i + 1]}"
            phrase2 = f"{words[i + 2]} {words[i + 3]}"
            phrase3 = f"{words[i + 4]} {words[i + 5]}"

            if phrase.lower() == phrase2.lower() == phrase3.lower():
                logger.debug(f"remove_hallucinations: 2-word phrase '{phrase}' repeated 3+ times — keeping one copy.")
                # Keep exactly one copy
                final.extend([words[i], words[i + 1]])
                # Skip all further consecutive occurrences
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
