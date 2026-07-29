from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from utilityhub_config import expand_path

from ..models import MODELS

DEFAULT_INITIAL_PROMPT = "Hello. This is a transcription with proper punctuation, capitalization, and grammar."


class ConfigError(ValueError):
    """Base configuration error."""


class UnknownTranscriptionModelError(ConfigError):
    """Raised when a model identifier is not registered."""


class Config(BaseModel):
    """Voicepad configuration with hierarchical loading."""

    model_config = ConfigDict(frozen=True)

    recordings_path: Path = Field(default_factory=lambda: expand_path("~/.config/voicepad/data/recordings"))
    markdown_path: Path = Field(default_factory=lambda: expand_path("~/.config/voicepad/data/markdown"))
    model_cache_path: Path = Field(default_factory=lambda: expand_path("~/.config/voicepad/models"))
    vad_model_path: Path = Field(default_factory=lambda: expand_path("~/.config/voicepad/models/vad"))
    logs_path: Path = Field(default_factory=lambda: expand_path("~/.config/voicepad/logs"))
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    input_device_index: int | None = None
    recording_prefix: str = "recording"
    transcription_model: str = "turbo"
    transcription_device: Literal["auto", "cuda", "cpu"] = "auto"
    transcription_compute_type: Literal["auto", "float16", "int8", "float32", "int8_float16"] = "auto"
    global_hotkey: str = "ctrl+alt+v"
    language: str = "en"
    beam_size: int = 5
    transcription_vad_filter: bool = False
    silence_threshold_ms: int = 1000
    min_chunk_s: float = 15.0
    max_chunk_s: float = 29.0
    overlap_s: float = 0.5
    local_agreement_file: bool = True
    initial_prompt: str = DEFAULT_INITIAL_PROMPT
    proper_nouns: tuple[str, ...] = ()
    text_postprocessing_enabled: bool = False
    no_speech_threshold: float = 0.6
    hallucination_silence_threshold: float = 2.0
    hallucination_max_repetitions: int = 3
    min_audio_duration_s: float = 0.5
    min_fresh_speech_duration_s: float = 0.25
    trim_trailing_silence_rms_threshold: float = 0.01
    trim_trailing_silence_frame_ms: int = 20
    stream_poll_interval_s: float = 0.3
    stream_context_chars: int = 200
    dedup_prev_tail_words: int = 50
    dedup_full_duplicate_threshold: float = 0.8
    dedup_min_overlap_words_for_partial: int = 3
    dedup_partial_lead_words: int = 5
    vad_threshold: float = 0.5
    vad_min_speech_duration_ms: int = 250
    vad_speech_pad_ms: int = 30
    vad_download_chunk_size: int = 8192

    @field_validator("transcription_model")
    @classmethod
    def validate_transcription_model(cls, v: str) -> str:
        if v not in MODELS:
            valid = ", ".join(MODELS)
            raise UnknownTranscriptionModelError(f"Unknown transcription model '{v}'. Valid models: {valid}.")
        return v

    @field_validator("proper_nouns")
    @classmethod
    def validate_proper_nouns(cls, terms: tuple[str, ...]) -> tuple[str, ...]:
        if any(not term.strip() for term in terms):
            raise ValueError("proper_nouns must not contain empty terms")
        return terms

    @field_validator(
        "recordings_path", "markdown_path", "model_cache_path", "vad_model_path", "logs_path", mode="before"
    )
    @classmethod
    def expand_paths(cls, v: Path | str) -> Path:
        return expand_path(str(v) if isinstance(v, Path) else v)


__all__ = [
    "Config",
    "ConfigError",
    "DEFAULT_INITIAL_PROMPT",
    "UnknownTranscriptionModelError",
]
