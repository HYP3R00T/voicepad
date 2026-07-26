"""LocalAgreement two-pass verification.

Runs transcription twice on the same audio and only emits tokens that
appear at the same position in both passes. Discards uncertain words.

Blueprint Part 11 implementation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..audio import RawAudio
    from ..inference.types import TranscriptionResult

logger = logging.getLogger(__name__)


def apply_local_agreement(
    audio: RawAudio,
    first_result: TranscriptionResult,
    model_name: str,
    device: str,
    compute_type: str,
    language: str,
) -> TranscriptionResult:
    """Apply two-pass LocalAgreement verification.

    Runs transcription a second time on the same audio and compares
    the results token-by-token. Only tokens that appear at the same
    position in both passes are kept. This filters out uncertain words
    that Whisper was not confident about.

    Args:
        audio: Audio array (already transcribed once)
        first_result: Result from first pass
        model_name: Model to use for second pass
        device: Device for second pass
        compute_type: Compute type for second pass
        language: Language for second pass

    Returns:
        TranscriptionResult with only agreed-upon tokens
    """
    from ..inference import transcribe

    logger.debug("LocalAgreement: Running second pass")

    # Second pass
    second_result = transcribe(
        audio,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        language=language,
    )

    # Token-level comparison
    agreed_text = _compare_tokens(first_result.text, second_result.text)

    logger.debug(
        "Local agreement: %s -> %s tokens",
        len(first_result.text.split()),
        len(agreed_text.split()),
    )

    # Return modified result with agreed text
    from dataclasses import replace

    return replace(
        first_result,
        text=agreed_text,
        latency_ms=first_result.latency_ms + second_result.latency_ms,
    )


def _compare_tokens(text1: str, text2: str) -> str:
    """Compare two transcriptions token-by-token.

    Only emit tokens that appear at the same position in both.
    This is the core LocalAgreement algorithm.

    Args:
        text1: First transcription
        text2: Second transcription

    Returns:
        Text with only agreed-upon tokens
    """
    tokens1 = text1.split()
    tokens2 = text2.split()

    agreed = []
    for i, (t1, t2) in enumerate(zip(tokens1, tokens2, strict=False)):
        if t1.lower() == t2.lower():
            agreed.append(t1)
        else:
            logger.debug("Token mismatch at position %s: %r vs %r", i, t1, t2)

    return " ".join(agreed)
