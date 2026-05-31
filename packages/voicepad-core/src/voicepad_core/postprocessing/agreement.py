"""LocalAgreement two-pass verification.

Runs transcription twice on the same audio and only emits tokens that
appear at the same position in both passes. Discards uncertain words.

Blueprint Part 11 implementation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..inference.types import TranscriptionResult

logger = logging.getLogger(__name__)


def apply_local_agreement(
    audio: np.ndarray,
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
    from ..inference.types import TranscriptionResult

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

    logger.debug(f"LocalAgreement: {len(first_result.text.split())} → {len(agreed_text.split())} tokens")

    # Return modified result with agreed text
    return TranscriptionResult(
        text=agreed_text,
        segments=first_result.segments,  # Keep first pass segments
        language=first_result.language,
        language_probability=first_result.language_probability,
        duration_s=first_result.duration_s,
        latency_ms=first_result.latency_ms + second_result.latency_ms,
        device=first_result.device,
        compute_type=first_result.compute_type,
        fallback_to_cpu=first_result.fallback_to_cpu,
        avg_confidence=first_result.avg_confidence,
        low_confidence_count=first_result.low_confidence_count,
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
            logger.debug(f"Token mismatch at position {i}: '{t1}' vs '{t2}'")

    return " ".join(agreed)
