# vad/silero.py

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from .base import SpeechSegment, VADBase
from .silero_download import MODEL_PATH, ensure_model_exists

# Silero VAD requires audio at exactly 16kHz.
_REQUIRED_SAMPLE_RATE = 16_000

# Silero processes audio in fixed 512-sample windows at 16kHz.
# 512 samples = 32ms per window. This is non-negotiable —
# the ONNX model weights were trained for exactly this window size.
_WINDOW_SIZE = 512

# Hidden and cell state dimensions for the Silero LSTM.
# These are fixed by the model architecture — do not change.
_H_SIZE = (2, 1, 64)
_C_SIZE = (2, 1, 64)


class SileroVAD(VADBase):
    """
    Voice Activity Detection using the Silero VAD ONNX model.

    Runs on CPU via onnxruntime. The model is ~2.5 MB and fast
    enough that GPU is unnecessary for VAD — save the GPU for
    Whisper inference.

    The ONNX file lives at vad/silero_vad.onnx and is downloaded
    automatically on first app startup via ensure_model_exists().

    Processing:
      - Audio is sliced into 512-sample (32ms) windows.
      - Each window is passed through the ONNX session with
        the LSTM hidden/cell state carried forward across windows
        for accurate sequential detection.
      - Windows above threshold are marked as speech.
      - Adjacent speech windows are merged into SpeechSegment objects
        with min_speech_duration, min_silence_duration, and padding
        applied during merging.

    Usage:
        vad = SileroVAD(threshold=0.5)
        segments = vad.detect(audio_float32_16khz)
        vad.reset()  # between recordings
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ) -> None:
        """
        Args:
            threshold:
                Speech probability threshold. Frames above this
                are classified as speech. Range [0.0, 1.0].
                0.5 is safe for most environments.

            min_speech_duration_ms:
                Minimum duration (ms) to keep a speech region.
                Shorter bursts are discarded as noise.

            min_silence_duration_ms:
                Minimum silence gap (ms) to split two speech regions.
                Shorter gaps merge adjacent segments into one.

            speech_pad_ms:
                Extra padding (ms) added to the start and end of each
                speech segment to avoid clipping first/last syllables.
        """
        self._threshold = threshold
        self._min_speech_samples = int(min_speech_duration_ms * _REQUIRED_SAMPLE_RATE / 1000)
        self._min_silence_samples = int(min_silence_duration_ms * _REQUIRED_SAMPLE_RATE / 1000)
        self._speech_pad_samples = int(speech_pad_ms * _REQUIRED_SAMPLE_RATE / 1000)

        self._session: ort.InferenceSession = self._load_session()
        self._h, self._c = self._init_states()

    # ------------------------------------------------------------------
    # VADBase interface
    # ------------------------------------------------------------------

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int = _REQUIRED_SAMPLE_RATE,
    ) -> list[SpeechSegment]:
        """
        Run Silero VAD on a full audio buffer.

        Args:
            audio:       float32 mono array at 16kHz.
                         AudioPreProcessor guarantees this upstream.
            sample_rate: Must be 16000. Raises clearly if not.

        Returns:
            List of SpeechSegment sorted by start time.
            Empty list if no speech detected or audio too short.

        Raises:
            ValueError: If sample_rate is not 16000.
        """
        if sample_rate != _REQUIRED_SAMPLE_RATE:
            raise ValueError(
                f"SileroVAD requires audio at {_REQUIRED_SAMPLE_RATE}Hz. "
                f"Got {sample_rate}Hz. Run AudioPreProcessor first."
            )

        if len(audio) < _WINDOW_SIZE:
            return []

        audio = _ensure_float32(audio)

        # Score each 512-sample window
        speech_probs = self._score_windows(audio)

        # Merge scored windows into SpeechSegment objects
        segments = self._build_segments(speech_probs, total_samples=len(audio))

        return segments

    def reset(self) -> None:
        """
        Reset the LSTM hidden and cell states.

        Call this between separate recordings so the hidden state
        from a previous session does not affect the next detection pass.
        """
        self._h, self._c = self._init_states()

    # ------------------------------------------------------------------
    # Internal — ONNX inference
    # ------------------------------------------------------------------

    def _score_windows(self, audio: np.ndarray) -> list[tuple[int, float]]:
        """
        Slide a 512-sample window over the audio and run the ONNX
        session on each window, carrying LSTM state forward.

        Returns:
            List of (window_start_sample, speech_probability) tuples.
        """
        results: list[tuple[int, float]] = []
        num_samples = len(audio)

        for start in range(0, num_samples - _WINDOW_SIZE + 1, _WINDOW_SIZE):
            window = audio[start : start + _WINDOW_SIZE]

            # ONNX input tensors — shapes must match the model exactly
            inputs = {
                "input": window[np.newaxis, :],  # (1, 512)
                "sr": np.array([_REQUIRED_SAMPLE_RATE], dtype=np.int64),
                "h": self._h,
                "c": self._c,
            }

            outputs = self._session.run(None, inputs)

            # outputs[0] : speech probability  shape (1, 1)
            # outputs[1] : new h state         shape (2, 1, 64)
            # outputs[2] : new c state         shape (2, 1, 64)
            # ONNX session outputs are untyped; cast to ndarray for squeeze()
            prob = float(np.asarray(outputs[0]).squeeze())  # type: ignore[arg-type]
            self._h = outputs[1]
            self._c = outputs[2]

            results.append((start, prob))

        return results

    # ------------------------------------------------------------------
    # Internal — segment building
    # ------------------------------------------------------------------

    def _build_segments(
        self,
        speech_probs: list[tuple[int, float]],
        total_samples: int,
    ) -> list[SpeechSegment]:
        """
        Convert per-window probabilities into merged SpeechSegment objects.

        Steps:
          1. Threshold each window into speech / silence boolean.
          2. Group consecutive speech windows into raw segments.
          3. Merge segments whose gap is below min_silence_samples.
          4. Drop segments shorter than min_speech_samples.
          5. Add speech_pad_samples padding, clamped to audio bounds.
          6. Convert sample indices to seconds.
        """
        if not speech_probs:
            return []

        # --- Step 1 & 2: group consecutive speech windows ---
        raw: list[tuple[int, int]] = []  # (start_sample, end_sample)
        in_speech = False
        seg_start = 0

        for start, prob in speech_probs:
            end = start + _WINDOW_SIZE
            if prob >= self._threshold:
                if not in_speech:
                    seg_start = start
                    in_speech = True
                seg_end = end
            else:
                if in_speech:
                    raw.append((seg_start, seg_end))  # type: ignore[possibly-undefined]
                    in_speech = False

        # Close a segment that runs to the end of audio
        if in_speech:
            raw.append((seg_start, speech_probs[-1][0] + _WINDOW_SIZE))

        if not raw:
            return []

        # --- Step 3: merge segments with short silence gaps ---
        merged: list[tuple[int, int]] = [raw[0]]
        for curr_start, curr_end in raw[1:]:
            prev_start, prev_end = merged[-1]
            if curr_start - prev_end < self._min_silence_samples:
                merged[-1] = (prev_start, curr_end)
            else:
                merged.append((curr_start, curr_end))

        # --- Step 4 & 5: filter short segments and add padding ---
        segments: list[SpeechSegment] = []

        for start_s, end_s in merged:
            if end_s - start_s < self._min_speech_samples:
                continue

            padded_start = max(0, start_s - self._speech_pad_samples)
            padded_end = min(total_samples, end_s + self._speech_pad_samples)

            segments.append(
                SpeechSegment(
                    start=padded_start / _REQUIRED_SAMPLE_RATE,
                    end=padded_end / _REQUIRED_SAMPLE_RATE,
                )
            )

        return segments

    # ------------------------------------------------------------------
    # Internal — setup
    # ------------------------------------------------------------------

    @staticmethod
    def _load_session() -> ort.InferenceSession:
        """
        Run the startup model check and load the ONNX session.

        ensure_model_exists() downloads the file if missing.
        The session is created with CPU execution provider only.

        Raises:
            RuntimeError: If download fails or ONNX session cannot load.
        """
        ensure_model_exists(verbose=True)

        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3  # suppress onnxruntime INFO logs

        session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        return session

    @staticmethod
    def _init_states() -> tuple[np.ndarray, np.ndarray]:
        """
        Initialise LSTM hidden and cell states to zeros.

        Shapes are fixed by the Silero ONNX model architecture:
          h : (2, 1, 64)
          c : (2, 1, 64)
        """
        h = np.zeros(_H_SIZE, dtype=np.float32)
        c = np.zeros(_C_SIZE, dtype=np.float32)
        return h, c


# ------------------------------------------------------------------
# Module-level helper
# ------------------------------------------------------------------


def _ensure_float32(audio: np.ndarray) -> np.ndarray:
    """Cast audio to float32 if it is not already."""
    if audio.dtype != np.float32:
        return audio.astype(np.float32)
    return audio
