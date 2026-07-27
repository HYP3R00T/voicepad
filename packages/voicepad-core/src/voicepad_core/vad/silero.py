from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sherpa_onnx

from .silero_download import ensure_model_exists
from ..config import Config, get_config

REQUIRED_SAMPLE_RATE = 16_000
WINDOW_SIZE = 512


class InvalidVADSampleRateError(ValueError):
    """Raised when Silero receives audio at an unsupported sample rate."""


@dataclass(frozen=True)
class SpeechSegment:
    """A detected speech interval measured in seconds."""

    start: float
    end: float


class SileroVAD:
    """Detect speech with Silero through VoicePad's shared Sherpa runtime."""

    def __init__(
        self,
        threshold: float | None = None,
        min_speech_duration_ms: int | None = None,
        min_silence_duration_ms: int | None = None,
        speech_pad_ms: int | None = None,
        vad_model_dir: Path | None = None,
        config: Config | None = None,
    ) -> None:
        self._config = config or get_config()
        threshold = threshold if threshold is not None else self._config.vad_threshold
        min_speech_duration_ms = (
            min_speech_duration_ms if min_speech_duration_ms is not None else self._config.vad_min_speech_duration_ms
        )
        min_silence_duration_ms = (
            min_silence_duration_ms if min_silence_duration_ms is not None else self._config.silence_threshold_ms
        )
        speech_pad_ms = speech_pad_ms if speech_pad_ms is not None else self._config.vad_speech_pad_ms

        self._speech_pad_samples = int(speech_pad_ms * REQUIRED_SAMPLE_RATE / 1000)
        self._vad_model_dir = vad_model_dir
        self._detector = self._load_detector(
            threshold,
            min_speech_duration_ms,
            min_silence_duration_ms,
        )

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int = REQUIRED_SAMPLE_RATE,
    ) -> list[SpeechSegment]:
        """Return detected speech intervals in chronological order."""
        if sample_rate != REQUIRED_SAMPLE_RATE:
            raise InvalidVADSampleRateError(
                f"SileroVAD requires audio at {REQUIRED_SAMPLE_RATE}Hz. "
                f"Got {sample_rate}Hz. Run AudioPreProcessor first."
            )

        if len(audio) < WINDOW_SIZE:
            return []

        samples = np.ascontiguousarray(audio, dtype=np.float32)
        if len(samples) % WINDOW_SIZE:
            samples = np.pad(samples, (0, WINDOW_SIZE - len(samples) % WINDOW_SIZE))

        for start in range(0, len(samples), WINDOW_SIZE):
            self._detector.accept_waveform(samples[start : start + WINDOW_SIZE])
        self._detector.flush()

        segments: list[SpeechSegment] = []
        while not self._detector.empty():
            segment = self._detector.front
            start = max(0, int(segment.start) - self._speech_pad_samples)
            end = min(len(audio), int(segment.start) + len(segment.samples) + self._speech_pad_samples)
            segments.append(SpeechSegment(start / REQUIRED_SAMPLE_RATE, end / REQUIRED_SAMPLE_RATE))
            self._detector.pop()
        return segments

    def reset(self) -> None:
        """Clear detector state between recordings."""
        self._detector.reset()

    def _load_detector(
        self,
        threshold: float,
        min_speech_duration_ms: int,
        min_silence_duration_ms: int,
    ) -> Any:
        model_path = ensure_model_exists(vad_model_dir=self._vad_model_dir, verbose=True, config=self._config)
        max_speech_duration = max(60.0, self._config.max_chunk_s)
        model = sherpa_onnx.SileroVadModelConfig(
            model=str(model_path),
            threshold=threshold,
            min_silence_duration=min_silence_duration_ms / 1000,
            min_speech_duration=min_speech_duration_ms / 1000,
            window_size=WINDOW_SIZE,
            max_speech_duration=max_speech_duration,
        )
        config = sherpa_onnx.VadModelConfig(
            silero_vad=model,
            sample_rate=REQUIRED_SAMPLE_RATE,
            num_threads=1,
            provider="cpu",
            debug=False,
        )
        return sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=max_speech_duration + 1)
