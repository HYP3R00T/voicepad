"""Core configuration types for voicepad."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from utilityhub_config import expand_path

from .constants import (
    DEFAULT_INITIAL_PROMPT,
    DEFAULT_VAD_MODEL_FILENAME,
    DEFAULT_VAD_MODEL_URL,
    VALID_TRANSCRIPTION_MODELS,
)
from .errors import UnknownTranscriptionModelError
from ..models import list_model_ids


class Config(BaseModel):
    """Voicepad configuration with hierarchical loading."""

    model_config = ConfigDict(frozen=True)

    recordings_path: Path = Field(default_factory=lambda: expand_path("~/.config/voicepad/data/recordings"))
    markdown_path: Path = Field(default_factory=lambda: expand_path("~/.config/voicepad/data/markdown"))
    model_cache_path: Path = Field(default_factory=lambda: expand_path("~/.config/voicepad/models"))
    vad_model_path: Path = Field(default_factory=lambda: expand_path("~/.config/voicepad/models/vad"))
    logs_path: Path = Field(default_factory=lambda: expand_path("~/.config/voicepad/logs"))
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    input_device_index: int | None = Field(default=None)
    recording_prefix: str = Field(default="recording")
    transcription_model: str = Field(default="turbo")
    transcription_device: Literal["auto", "cuda", "cpu"] = Field(default="auto")
    transcription_compute_type: Literal["auto", "float16", "int8", "float32", "int8_float16"] = Field(default="auto")
    global_hotkey: str = Field(default="<ctrl>+<alt>+v")
    language: str = Field(default="en")
    beam_size: int = Field(default=5)
    transcription_vad_filter: bool = Field(default=False)
    silence_threshold_ms: int = Field(default=1000)
    min_chunk_s: float = Field(default=15.0)
    max_chunk_s: float = Field(default=29.0)
    overlap_s: float = Field(default=0.5)
    local_agreement_mic: bool = Field(default=False)
    local_agreement_file: bool = Field(default=True)
    initial_prompt: str = Field(default=DEFAULT_INITIAL_PROMPT)
    text_postprocessing_enabled: bool = Field(default=False)
    no_speech_threshold: float = Field(default=0.6)
    hallucination_silence_threshold: float = Field(default=2.0)
    hallucination_max_repetitions: int = Field(default=3)
    min_audio_duration_s: float = Field(default=0.5)
    min_fresh_speech_duration_s: float = Field(default=0.25)
    trim_trailing_silence_rms_threshold: float = Field(default=0.01)
    trim_trailing_silence_frame_ms: int = Field(default=20)
    stream_poll_interval_s: float = Field(default=0.3)
    stream_context_chars: int = Field(default=200)
    dedup_prev_tail_words: int = Field(default=50)
    dedup_full_duplicate_threshold: float = Field(default=0.8)
    dedup_min_overlap_words_for_partial: int = Field(default=3)
    dedup_partial_lead_words: int = Field(default=5)
    vad_threshold: float = Field(default=0.5)
    vad_min_speech_duration_ms: int = Field(default=250)
    vad_speech_pad_ms: int = Field(default=30)
    vad_model_filename: str = Field(default=DEFAULT_VAD_MODEL_FILENAME)
    vad_model_url: str = Field(default=DEFAULT_VAD_MODEL_URL)
    vad_download_chunk_size: int = Field(default=8192)
    model_warmup_enabled: bool = Field(default=True)
    model_warmup_duration_s: float = Field(default=0.5)
    model_warmup_language: str = Field(default="en")
    model_warmup_beam_size: int = Field(default=1)
    model_warmup_vad_filter: bool = Field(default=False)

    @field_validator("transcription_model")
    @classmethod
    def validate_transcription_model(cls, v: str) -> str:
        if v not in VALID_TRANSCRIPTION_MODELS:
            valid = ", ".join(list_model_ids())
            raise UnknownTranscriptionModelError(f"Unknown transcription model '{v}'. Valid models: {valid}.")
        return v

    @field_validator(
        "recordings_path", "markdown_path", "model_cache_path", "vad_model_path", "logs_path", mode="before"
    )
    @classmethod
    def expand_paths(cls, v: Path | str) -> Path:
        return expand_path(str(v) if isinstance(v, Path) else v)

    @model_validator(mode="after")
    def ensure_paths_expanded(self) -> Config:
        for path_field in ("recordings_path", "markdown_path", "model_cache_path", "vad_model_path", "logs_path"):
            current_path = getattr(self, path_field)
            expanded = expand_path(str(current_path))
            object.__setattr__(self, path_field, expanded)
        return self


__all__ = ["Config"]
