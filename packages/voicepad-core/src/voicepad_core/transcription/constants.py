"""Constants for transcription configuration."""

DEVICE: str = "cuda"
COMPUTE_TYPE: str = "int8"
BEAM_SIZE: int = 3
LANGUAGE: str = "en"
HALLUCINATION_SILENCE_THRESHOLD: float = 0.5
NO_SPEECH_THRESHOLD: float = 0.8
INITIAL_PROMPT: str = "Hello. This is a transcription with proper punctuation, capitalization, and grammar."

DISTIL_MODELS: frozenset[str] = frozenset({
    "distil-small.en",
    "distil-medium.en",
    "distil-large-v2",
    "distil-large-v3",
    "distil-large-v3.5",
})

MIN_AUDIO_DURATION_S: float = 0.5
MAX_AUDIO_DURATION_S: float = 900.0
HF_REPO_PREFIX = "Systran/faster-whisper-"
CUDA_ERROR_KEYWORDS = ("cublas", "cuda", "cudnn", "nvrtc", "cufft", "curand")
