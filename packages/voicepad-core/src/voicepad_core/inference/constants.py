# inference/constants.py

"""Constants for the inference package."""

# ---------------------------------------------------------------------------
# Device & precision
# ---------------------------------------------------------------------------

DEVICE: str = "cuda"
COMPUTE_TYPE: str = "int8_float16"  # GPU: weights int8, activations float16
CPU_COMPUTE_TYPE: str = "int8"  # CPU fallback: pure int8

# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL: str = "turbo"

DISTIL_MODELS: frozenset[str] = frozenset({
    "distil-small.en",
    "distil-medium.en",
    "distil-large-v2",
    "distil-large-v3",
    "distil-large-v3.5",
})

# ---------------------------------------------------------------------------
# Transcription quality
# ---------------------------------------------------------------------------

BEAM_SIZE: int = 5
LANGUAGE: str = "en"
NO_SPEECH_THRESHOLD: float = 0.6
HALLUCINATION_SILENCE_THRESHOLD: float = 2.0
INITIAL_PROMPT: str = "Hello. This is a transcription with proper punctuation, capitalization, and grammar."

# ---------------------------------------------------------------------------
# Audio duration guards
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 16_000
MIN_AUDIO_DURATION_S: float = 0.5
MAX_AUDIO_DURATION_S: float = 300.0  # 5-minute hard cap

# ---------------------------------------------------------------------------
# HuggingFace
# ---------------------------------------------------------------------------

HF_REPO_PREFIX: str = "Systran/faster-whisper-"

# ---------------------------------------------------------------------------
# CUDA error detection keywords
# ---------------------------------------------------------------------------

CUDA_ERROR_KEYWORDS: tuple[str, ...] = (
    "cublas",
    "cuda",
    "cudnn",
    "nvrtc",
    "cufft",
    "curand",
)
