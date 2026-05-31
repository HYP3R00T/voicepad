"""Text post-processing for transcription quality improvements."""

import re


class CapitalizationFixer:
    """Fixes 'i' → 'I' and sentence-start capitalization."""

    def __init__(self):
        # Compile regex patterns once
        self._standalone_i = re.compile(r"\bi\b")
        self._i_contraction = re.compile(r"\bi'")
        self._sentence_start = re.compile(r"([.!?]\s+)([a-z])")

    def process(self, text: str) -> str:
        """Apply capitalization fixes to text."""
        if not text:
            return text

        # Fix standalone 'i' → 'I'
        text = self._standalone_i.sub("I", text)

        # Fix "i'" → "I'"
        text = self._i_contraction.sub("I'", text)

        # Fix sentence starts
        text = self._sentence_start.sub(lambda m: m.group(1) + m.group(2).upper(), text)

        # Fix very first character
        if text[0].islower():
            text = text[0].upper() + text[1:]

        return text


class PunctuationNormalizer:
    """Normalize punctuation spacing."""

    def __init__(self):
        self._space_before_punct = re.compile(r"\s+([,.!?;:])")
        self._space_after_punct = re.compile(r"([,.!?;:])([^\s\d])")
        self._multiple_spaces = re.compile(r"\s+")

    def process(self, text: str) -> str:
        """Apply punctuation normalization to text."""
        if not text:
            return text

        # Remove spaces before punctuation
        text = self._space_before_punct.sub(r"\1", text)

        # Add space after punctuation
        text = self._space_after_punct.sub(r"\1 \2", text)

        # Collapse multiple spaces
        text = self._multiple_spaces.sub(" ", text)

        return text.strip()


class PostProcessorChain:
    """Chain multiple post-processors together."""

    def __init__(self, processors: list):
        """Initialize with list of processor instances."""
        self.processors = processors

    def process(self, text: str) -> str:
        """Apply all processors in sequence."""
        for processor in self.processors:
            text = processor.process(text)
        return text
