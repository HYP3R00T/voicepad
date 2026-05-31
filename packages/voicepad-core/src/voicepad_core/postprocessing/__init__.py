# postprocessing/__init__.py

"""Post-processing pipeline for transcription output.

All functions are pure and stateless — they take segments or text in
and return segments or text out. No dependency on inference/ or audio/.

Typical call order:
    segments = filter_segments(raw_iter, duration_s)
    segments = deduplicate_overlap(segments, chunk_start_s, prev_text)
    text     = " ".join(s.text for s in segments).strip()
    text     = remove_hallucinations(text)
    text     = normalize(text)
"""

from .agreement import apply_local_agreement
from .deduplication import deduplicate_overlap
from .filters import filter_segments
from .hallucination import remove_hallucinations
from .normalizer import normalize

__all__ = [
    "apply_local_agreement",
    "deduplicate_overlap",
    "filter_segments",
    "remove_hallucinations",
    "normalize",
]
