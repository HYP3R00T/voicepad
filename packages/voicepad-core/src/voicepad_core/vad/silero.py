from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512
CONTEXT_SAMPLES = 64
STATE_SHAPE = (2, 1, 128)


class VadError(RuntimeError):
    """Raised when sequential Silero VAD processing cannot continue safely."""


class OnnxSession(Protocol):
    def run(self, output_names: None, input_feed: dict[str, np.ndarray]) -> Sequence[object]: ...


@dataclass(frozen=True, slots=True)
class VadFrame:
    start_sample: int
    end_sample: int
    speech_probability: float


class SileroVad:
    """Stateful official Silero v6.2.1 ONNX inference on CPU."""

    def __init__(self, model_path: Path, *, session: OnnxSession | None = None) -> None:
        self._session = session or _open_cpu_session(model_path)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)
        self._pending = np.empty(0, dtype=np.float32)
        self._next_sample = 0

    def accept(self, samples: np.ndarray, start_sample: int, *, final: bool = False) -> tuple[VadFrame, ...]:
        if samples.ndim != 1 or samples.dtype != np.float32:
            raise VadError("Silero VAD requires a one-dimensional float32 waveform.")
        if start_sample != self._next_sample:
            raise VadError(f"Silero VAD input must be sequential: expected={self._next_sample} actual={start_sample}.")
        self._next_sample += len(samples)
        combined = np.concatenate((self._pending, samples))
        combined_start = start_sample - len(self._pending)
        frames: list[VadFrame] = []
        offset = 0
        while len(combined) - offset >= FRAME_SAMPLES:
            frame = combined[offset : offset + FRAME_SAMPLES]
            frames.append(self._infer(frame, combined_start + offset, FRAME_SAMPLES))
            offset += FRAME_SAMPLES
        self._pending = np.ascontiguousarray(combined[offset:])
        if final and self._pending.size:
            valid = len(self._pending)
            padded = np.pad(self._pending, (0, FRAME_SAMPLES - valid)).astype(np.float32)
            frames.append(self._infer(padded, self._next_sample - valid, valid))
            self._pending = np.empty(0, dtype=np.float32)
        return tuple(frames)

    def _infer(self, frame: np.ndarray, start_sample: int, valid_samples: int) -> VadFrame:
        model_input = np.concatenate((self._context, frame.reshape(1, -1)), axis=1)
        outputs = self._session.run(
            None,
            {
                "input": model_input,
                "state": self._state,
                "sr": np.array(SAMPLE_RATE, dtype=np.int64),
            },
        )
        if len(outputs) != 2:
            raise VadError("Official Silero ONNX model returned an unexpected output contract.")
        probability = float(np.asarray(outputs[0]).reshape(-1)[0])
        state = np.asarray(outputs[1], dtype=np.float32)
        if state.shape != STATE_SHAPE or not 0.0 <= probability <= 1.0:
            raise VadError("Official Silero ONNX model returned invalid state or probability.")
        self._state = state
        self._context = model_input[:, -CONTEXT_SAMPLES:]
        return VadFrame(start_sample, start_sample + valid_samples, probability)


def _open_cpu_session(model_path: Path) -> OnnxSession:
    if not model_path.is_file():
        raise VadError("Verified official Silero ONNX model is missing.")
    try:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        return ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as error:
        raise VadError(f"Could not open official Silero VAD on CPU: {error}") from error
