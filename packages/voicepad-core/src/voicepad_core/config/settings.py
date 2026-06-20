"""Core configuration for voicepad."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from utilityhub_config import expand_path, load_settings


def _get_available_models() -> tuple[str, ...]:
    """Query available models from faster_whisper at import time."""
    try:
        from faster_whisper.utils import available_models

        return tuple(available_models())
    except Exception:
        return (
            "tiny.en",
            "tiny",
            "base.en",
            "base",
            "small.en",
            "small",
            "medium.en",
            "medium",
            "large-v1",
            "large-v2",
            "large-v3",
            "large",
            "distil-large-v2",
            "distil-medium.en",
            "distil-small.en",
            "distil-large-v3",
            "distil-large-v3.5",
            "large-v3-turbo",
            "turbo",
        )


VALID_TRANSCRIPTION_MODELS: tuple[str, ...] = _get_available_models()

DEFAULT_INITIAL_PROMPT = "Hello. This is a transcription with proper punctuation, capitalization, and grammar."
DEFAULT_VAD_MODEL_FILENAME = "silero_vad_v6.onnx"
DEFAULT_VAD_MODEL_URL = (
    "https://raw.githubusercontent.com/SYSTRAN/faster-whisper/master/faster_whisper/assets/silero_vad_v6.onnx"
)


class Config(BaseModel):
    """Voicepad configuration with hierarchical loading.

    Configuration sources (highest to lowest precedence):
    1. Environment variables (VOICEPAD_*)
    2. ./voicepad.yaml (current directory)
    3. ~/.config/voicepad/voicepad.yaml (user config)
    4. Default values
    """

    model_config = ConfigDict(frozen=True)

    recordings_path: Path = Field(
        default_factory=lambda: expand_path("~/.config/voicepad/data/recordings"),
        description="Directory where WAV recordings are saved.",
    )
    markdown_path: Path = Field(
        default_factory=lambda: expand_path("~/.config/voicepad/data/markdown"),
        description="Directory where markdown transcriptions are saved.",
    )
    model_cache_path: Path = Field(
        default_factory=lambda: expand_path("~/.config/voicepad/models"),
        description="Directory where Whisper model weights are cached.",
    )
    vad_model_path: Path = Field(
        default_factory=lambda: expand_path("~/.config/voicepad/models/vad"),
        description="Directory where VAD (Voice Activity Detection) model files are stored.",
    )
    logs_path: Path = Field(
        default_factory=lambda: expand_path("~/.config/voicepad/logs"),
        description="Directory where transcription logs are saved.",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level for transcription operations.",
    )

    input_device_index: int | None = Field(
        default=None,
        description="Audio input device index. None = system default. "
        "Run 'voicepad config input list' to see available devices.",
    )
    recording_prefix: str = Field(
        default="recording",
        description="Filename prefix for recordings (timestamp is appended automatically).",
    )

    transcription_model: str = Field(
        default="turbo",
        description=(
            "Whisper model for transcription. Must be one of the supported faster-whisper models. "
            "Models are auto-downloaded from HuggingFace on first use. "
            f"Valid values: {', '.join(VALID_TRANSCRIPTION_MODELS)}."
        ),
    )

    @field_validator("transcription_model")
    @classmethod
    def validate_transcription_model(cls, v: str) -> str:
        """Validate model name against faster_whisper's available models."""
        if v not in VALID_TRANSCRIPTION_MODELS:
            valid = ", ".join(VALID_TRANSCRIPTION_MODELS)
            raise ValueError(f"Unknown transcription model '{v}'. Valid models: {valid}.")
        return v

    transcription_device: Literal["auto", "cuda", "cpu"] = Field(
        default="auto",
        description=(
            "Device for transcription. "
            "'auto' tries CUDA first and falls back to CPU. "
            "'cuda' forces GPU (fails if unavailable). "
            "'cpu' forces CPU."
        ),
    )
    transcription_compute_type: Literal["auto", "float16", "int8", "float32", "int8_float16"] = Field(
        default="auto",
        description=(
            "Compute precision for CTranslate2. "
            "'auto' uses int8 on CUDA, int8 on CPU. "
            "'float16' — higher accuracy, ~2x VRAM vs int8 (GPU only). "
            "'int8' — best speed/VRAM tradeoff (recommended). "
            "'float32' — full precision, slowest. "
            "'int8_float16' — mixed precision."
        ),
    )

    global_hotkey: str = Field(
        default="<ctrl>+<alt>+v",
        description=(
            "System-wide hotkey to start/stop recording from any application. "
            "Press once to start, press again to stop and copy transcription to clipboard. "
            "Uses pynput key syntax, e.g. '<ctrl>+<alt>+v' or '<ctrl>+<shift>+space'. "
            "Set to empty string to disable."
        ),
    )

    language: str = Field(
        default="en",
        description=(
            "Primary language for transcription. Non-English languages are supported "
            "but may have reduced accuracy. A warning is emitted for non-English use."
        ),
    )

    beam_size: int = Field(
        default=5,
        description=(
            "Beam search width for Whisper decoder. "
            "5 = quality-first default for general dictation. "
            "1 = greedy decoding when speed matters more than accuracy."
        ),
    )

    transcription_vad_filter: bool = Field(
        default=False,
        description=(
            "Enable Whisper's built-in VAD filter during inference. "
            "Redundant when using streaming mode (which already runs VAD for chunk splitting). "
            "Useful for single-shot file transcription with no prior VAD."
        ),
    )

    silence_threshold_ms: int = Field(
        default=1000,
        description=(
            "VAD silence duration (ms) to trigger a chunk split during streaming. "
            "Benchmarkable — test with 500, 800, 1000, 1500 to find optimal value."
        ),
    )

    min_chunk_s: float = Field(
        default=15.0,
        description=(
            "Minimum audio duration (seconds) before considering a silence-triggered split. "
            "Benchmarkable — test with 10, 15, 20, 29."
        ),
    )

    max_chunk_s: float = Field(
        default=29.0,
        description=(
            "Per-chunk safety limit in seconds. Long recording sessions remain unbounded; "
            "this only forces a split before Whisper's 30s context window is exceeded."
        ),
    )

    overlap_s: float = Field(
        default=0.5,
        description="Audio overlap (seconds) kept at chunk boundaries for acoustic continuity.",
    )

    local_agreement_mic: bool = Field(
        default=False,
        description=(
            "Enable two-pass LocalAgreement verification for mic streaming mode. "
            "Roughly doubles per-chunk latency but improves accuracy."
        ),
    )

    local_agreement_file: bool = Field(
        default=True,
        description=(
            "Enable two-pass LocalAgreement verification for file retranscription. "
            "No latency impact — user is already waiting for full-file processing."
        ),
    )

    initial_prompt: str = Field(
        default=DEFAULT_INITIAL_PROMPT,
        description="Default Whisper initial prompt for non-distil models.",
    )
    no_speech_threshold: float = Field(
        default=0.6,
        description="Segments above this no-speech probability are discarded.",
    )
    hallucination_silence_threshold: float = Field(
        default=2.0,
        description="Silence threshold passed to faster-whisper hallucination suppression.",
    )
    hallucination_max_repetitions: int = Field(
        default=3,
        description="Maximum consecutive repeated tokens allowed before text cleanup trims extras.",
    )
    min_audio_duration_s: float = Field(
        default=0.5,
        description="Minimum audio duration required before transcription runs.",
    )
    trim_trailing_silence_rms_threshold: float = Field(
        default=0.01,
        description="RMS threshold used when trimming trailing silence before inference.",
    )
    trim_trailing_silence_frame_ms: int = Field(
        default=20,
        description="Frame size in milliseconds for trailing-silence trimming.",
    )

    stream_poll_interval_s: float = Field(
        default=0.3,
        description="How often the streaming monitor thread polls the recorder buffer.",
    )
    stream_context_chars: int = Field(
        default=200,
        description="How many trailing characters from the previous chunk are carried into the next prompt.",
    )
    dedup_prev_tail_words: int = Field(
        default=50,
        description="How many words from the previous chunk are used for overlap deduplication.",
    )
    dedup_full_duplicate_threshold: float = Field(
        default=0.8,
        description="Similarity threshold above which overlap is treated as a full duplicate.",
    )
    dedup_min_overlap_words_for_partial: int = Field(
        default=3,
        description="Minimum overlap words required before partial deduplication runs.",
    )
    dedup_partial_lead_words: int = Field(
        default=5,
        description="How many leading overlap words are checked for partial deduplication.",
    )

    vad_threshold: float = Field(
        default=0.5,
        description="Speech probability threshold for Silero VAD.",
    )
    vad_min_speech_duration_ms: int = Field(
        default=250,
        description="Minimum duration for a VAD speech region to be kept.",
    )
    vad_speech_pad_ms: int = Field(
        default=30,
        description="Padding added to both sides of each VAD speech region.",
    )
    vad_model_filename: str = Field(
        default=DEFAULT_VAD_MODEL_FILENAME,
        description="Filename used for the downloaded Silero VAD model.",
    )
    vad_model_url: str = Field(
        default=DEFAULT_VAD_MODEL_URL,
        description="Download URL for the Silero VAD ONNX model.",
    )
    vad_download_chunk_size: int = Field(
        default=8192,
        description="Chunk size in bytes used while downloading the VAD model.",
    )

    model_warmup_enabled: bool = Field(
        default=True,
        description="Run a short warm-up inference after loading a model.",
    )
    model_warmup_duration_s: float = Field(
        default=0.5,
        description="Duration of the silent warm-up inference buffer.",
    )
    model_warmup_language: str = Field(
        default="en",
        description="Language used for model warm-up inference.",
    )
    model_warmup_beam_size: int = Field(
        default=1,
        description="Beam size used during model warm-up inference.",
    )
    model_warmup_vad_filter: bool = Field(
        default=False,
        description="Whether model warm-up should enable Whisper VAD filtering.",
    )

    @field_validator(
        "recordings_path", "markdown_path", "model_cache_path", "vad_model_path", "logs_path", mode="before"
    )
    @classmethod
    def expand_paths(cls, v: Path | str) -> Path:
        """Expand ~ and environment variables in path fields."""
        return expand_path(str(v) if isinstance(v, Path) else v)

    @model_validator(mode="after")
    def ensure_paths_expanded(self) -> Config:
        """Expand all path fields (safety check for default values)."""
        for path_field in ("recordings_path", "markdown_path", "model_cache_path", "vad_model_path", "logs_path"):
            current_path = getattr(self, path_field)
            expanded = expand_path(str(current_path))
            object.__setattr__(self, path_field, expanded)
        return self


def get_config(cwd: Path | None = None, app_name: str = "voicepad") -> Config:
    """Load configuration with hierarchical precedence.

    Args:
        cwd: Optional working directory for config search
        app_name: Application name for config file lookup

    Returns:
        Loaded configuration object
    """
    settings, _ = load_settings(Config, app_name=app_name, cwd=cwd)
    return settings


def get_config_with_metadata(
    cwd: Path | None = None,
    app_name: str = "voicepad",
) -> tuple[Config, Any]:
    """Load configuration with source metadata.

    Args:
        cwd: Optional working directory for config search
        app_name: Application name for config file lookup

    Returns:
        Tuple of (config, metadata) where metadata tracks source of each field
    """
    return load_settings(Config, app_name=app_name, cwd=cwd)
