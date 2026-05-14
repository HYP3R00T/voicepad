"""Audio capture and transcription engine.

On Windows, ctranslate2 cannot discover CUDA DLLs from site-packages.
We pre-load them via ctypes.WinDLL so they're found by name in memory.
"""

import sys

# Pre-load CUDA DLLs on Windows so ctranslate2 can find them.
if sys.platform == "win32":
    import ctypes
    import os
    import pathlib

    _cuda_dll_dirs: set[str] = set()

    # Find nvidia DLL directories from sys.path
    for _path_entry in sys.path:
        _nvidia_dir = pathlib.Path(_path_entry) / "nvidia"
        if _nvidia_dir.is_dir():
            for _dll_file in _nvidia_dir.rglob("*.dll"):
                _cuda_dll_dirs.add(str(_dll_file.parent))
            break

    # Register directories and pre-load DLLs for ctranslate2 discovery
    _loaded = 0
    for _d in sorted(_cuda_dll_dirs):
        os.add_dll_directory(_d)
        os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")

    # Two passes to handle DLL dependency ordering (cudnn depends on cublas)
    _dll_files = []
    for _d in sorted(_cuda_dll_dirs):
        _dll_files.extend(sorted(pathlib.Path(_d).glob("*.dll")))
    _remaining = list(_dll_files)
    for _pass_num in range(2):
        _still_remaining = []
        for _dll in _remaining:
            try:
                ctypes.WinDLL(str(_dll))
                _loaded += 1
            except OSError:
                _still_remaining.append(_dll)
        _remaining = _still_remaining
        if not _remaining:
            break

from voicepad_core.audio import SAMPLE_RATE, AudioRecorder, AudioRecorderError
from voicepad_core.config import Config, get_config, get_config_with_metadata
from voicepad_core.config.settings import VALID_TRANSCRIPTION_MODELS
from voicepad_core.streaming import ChunkResult, StreamingTranscriber
from voicepad_core.transcription import (
    AudioTooLongWarning,
    AudioTooShortError,
    Segment,
    TranscriptionError,
    TranscriptionResult,
    ensure_model_downloaded,
    get_or_load_model,
    model_downloaded,
    transcribe_buffer,
    transcribe_file,
)

__all__ = [
    # Streaming
    "StreamingTranscriber",
    "ChunkResult",
    # Recording
    "AudioRecorder",
    "AudioRecorderError",
    "SAMPLE_RATE",
    # Transcription
    "transcribe_buffer",
    "transcribe_file",
    "get_or_load_model",
    "model_downloaded",
    "ensure_model_downloaded",
    "TranscriptionResult",
    "Segment",
    "TranscriptionError",
    "AudioTooShortError",
    "AudioTooLongWarning",
    # Config
    "Config",
    "get_config",
    "get_config_with_metadata",
    "VALID_TRANSCRIPTION_MODELS",
]
