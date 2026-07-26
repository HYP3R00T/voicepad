from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .silero_download import ensure_model_exists
from ..config import Config, get_config

REQUIRED_SAMPLE_RATE = 16_000
WINDOW_SIZE = 512
STATE_SIZE = (1, 1, 128)


class InvalidVADSampleRateError(ValueError):
    """Raised when Silero receives audio at an unsupported sample rate."""


@dataclass(frozen=True)
class SpeechSegment:
    """A detected speech interval measured in seconds."""

    start: float
    end: float


class SileroVAD:
    """Detect speech in 16 kHz mono audio with Silero ONNX on CPU."""

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

        self._threshold = threshold
        self._min_speech_samples = int(min_speech_duration_ms * REQUIRED_SAMPLE_RATE / 1000)
        self._min_silence_samples = int(min_silence_duration_ms * REQUIRED_SAMPLE_RATE / 1000)
        self._speech_pad_samples = int(speech_pad_ms * REQUIRED_SAMPLE_RATE / 1000)
        self._vad_model_dir = vad_model_dir

        self._session: ort.InferenceSession = self._load_session()
        self._h: np.ndarray
        self._c: np.ndarray
        self._h, self._c = self._init_states()

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

        audio = _ensure_float32(audio)

        speech_probs = self._score_windows(audio)
        return self._build_segments(speech_probs, total_samples=len(audio))

    def reset(self) -> None:
        """Clear recurrent state between recordings."""
        self._h, self._c = self._init_states()

    def _score_windows(self, audio: np.ndarray) -> list[tuple[int, float]]:
        num_samples = len(audio)
        if num_samples % WINDOW_SIZE != 0:
            pad_size = WINDOW_SIZE - (num_samples % WINDOW_SIZE)
            audio = np.pad(audio, (0, pad_size), "constant")

        num_chunks = len(audio) // WINDOW_SIZE
        batched_audio = audio.reshape(num_chunks, WINDOW_SIZE)

        context_size = 64
        context = np.zeros((num_chunks, context_size), dtype=np.float32)
        if num_chunks > 1:
            context[1:] = batched_audio[:-1, -context_size:]

        inputs_audio = np.concatenate([context, batched_audio], axis=1)

        outputs = self._session.run(
            None,
            {
                "input": inputs_audio,
                "h": self._h,
                "c": self._c,
            },
        )

        probs = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        self._h = np.asarray(outputs[1], dtype=np.float32)
        self._c = np.asarray(outputs[2], dtype=np.float32)

        results = [(i * WINDOW_SIZE, float(probs[i])) for i in range(num_chunks)]
        return results

    def _build_segments(
        self,
        speech_probs: list[tuple[int, float]],
        total_samples: int,
    ) -> list[SpeechSegment]:
        if not speech_probs:
            return []

        raw: list[tuple[int, int]] = []  # (start_sample, end_sample)
        in_speech = False
        seg_start = 0

        for start, prob in speech_probs:
            end = start + WINDOW_SIZE
            if prob >= self._threshold:
                if not in_speech:
                    seg_start = start
                    in_speech = True
                seg_end = end
            else:
                if in_speech:
                    raw.append((seg_start, seg_end))  # type: ignore[possibly-undefined]
                    in_speech = False

        if in_speech:
            raw.append((seg_start, speech_probs[-1][0] + WINDOW_SIZE))

        if not raw:
            return []

        merged: list[tuple[int, int]] = [raw[0]]
        for curr_start, curr_end in raw[1:]:
            prev_start, prev_end = merged[-1]
            if curr_start - prev_end < self._min_silence_samples:
                merged[-1] = (prev_start, curr_end)
            else:
                merged.append((curr_start, curr_end))

        segments: list[SpeechSegment] = []

        for start_s, end_s in merged:
            if end_s - start_s < self._min_speech_samples:
                continue

            padded_start = max(0, start_s - self._speech_pad_samples)
            padded_end = min(total_samples, end_s + self._speech_pad_samples)

            segments.append(
                SpeechSegment(
                    start=padded_start / REQUIRED_SAMPLE_RATE,
                    end=padded_end / REQUIRED_SAMPLE_RATE,
                )
            )

        return segments

    def _load_session(self) -> ort.InferenceSession:
        model_path = ensure_model_exists(vad_model_dir=self._vad_model_dir, verbose=True, config=self._config)

        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3  # suppress onnxruntime INFO logs

        session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        return session

    @staticmethod
    def _init_states() -> tuple[np.ndarray, np.ndarray]:
        h = np.zeros(STATE_SIZE, dtype=np.float32)
        c = np.zeros(STATE_SIZE, dtype=np.float32)
        return h, c


def _ensure_float32(audio: np.ndarray) -> np.ndarray:
    if audio.dtype != np.float32:
        return audio.astype(np.float32)
    return audio
