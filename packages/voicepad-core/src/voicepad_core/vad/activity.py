from __future__ import annotations

from dataclasses import dataclass

from .silero import FRAME_SAMPLES, SAMPLE_RATE, VadFrame


@dataclass(frozen=True, slots=True)
class SpeechRegion:
    start_sample: int
    end_sample: int


@dataclass(frozen=True, slots=True)
class NaturalPause:
    start_sample: int
    end_sample: int

    @property
    def breakpoint_sample(self) -> int:
        return (self.start_sample + self.end_sample) // 2


class PauseTracker:
    """Convert sequential Silero probabilities into confirmed natural pauses."""

    def __init__(
        self,
        *,
        speech_threshold: float = 0.5,
        negative_threshold: float = 0.35,
        minimum_speech_ms: int = 250,
        minimum_silence_ms: int = 500,
    ) -> None:
        if not 0.0 <= negative_threshold < speech_threshold <= 1.0:
            raise ValueError("VAD thresholds must satisfy 0 <= negative < speech <= 1")
        self._speech_threshold = speech_threshold
        self._negative_threshold = negative_threshold
        self._minimum_speech_frames = _duration_frames(minimum_speech_ms)
        self._minimum_silence_frames = _duration_frames(minimum_silence_ms)
        self.reset()

    def reset(self) -> None:
        self._speech_candidate = 0
        self._speech_active = False
        self._silence_start: int | None = None
        self._silence_frames = 0
        self._last_end = 0

    def accept(self, frame: VadFrame) -> NaturalPause | None:
        if frame.start_sample != self._last_end:
            raise ValueError("Pause tracker frames must be contiguous.")
        self._last_end = frame.end_sample
        probability = frame.speech_probability

        if not self._speech_active:
            if probability >= self._speech_threshold:
                self._speech_candidate += 1
                if self._speech_candidate >= self._minimum_speech_frames:
                    self._speech_active = True
            else:
                self._speech_candidate = 0
            return None

        if probability < self._negative_threshold:
            if self._silence_start is None:
                self._silence_start = frame.start_sample
                self._silence_frames = 0
            self._silence_frames += 1
            if self._silence_frames >= self._minimum_silence_frames:
                pause = NaturalPause(self._silence_start, frame.end_sample)
                self._speech_active = False
                self._speech_candidate = 0
                self._silence_start = None
                self._silence_frames = 0
                return pause
            return None

        self._silence_start = None
        self._silence_frames = 0
        return None


def material_speech_regions(
    frames: tuple[VadFrame, ...],
    *,
    speech_threshold: float = 0.5,
    negative_threshold: float = 0.35,
    minimum_speech_ms: int = 250,
    minimum_silence_ms: int = 500,
) -> tuple[SpeechRegion, ...]:
    minimum_speech_frames = _duration_frames(minimum_speech_ms)
    minimum_silence_frames = _duration_frames(minimum_silence_ms)
    candidate_start: int | None = None
    candidate_frames = 0
    active_start: int | None = None
    silence_start: int | None = None
    silence_frames = 0
    regions: list[SpeechRegion] = []
    last_end = 0
    for frame in frames:
        last_end = frame.end_sample
        if active_start is None:
            if frame.speech_probability >= speech_threshold:
                if candidate_start is None:
                    candidate_start = frame.start_sample
                candidate_frames += 1
                if candidate_frames >= minimum_speech_frames:
                    active_start = candidate_start
            else:
                candidate_start = None
                candidate_frames = 0
            continue
        if frame.speech_probability < negative_threshold:
            if silence_start is None:
                silence_start = frame.start_sample
            silence_frames += 1
            if silence_frames >= minimum_silence_frames:
                regions.append(SpeechRegion(active_start, silence_start))
                active_start = None
                candidate_start = None
                candidate_frames = 0
                silence_start = None
                silence_frames = 0
        else:
            silence_start = None
            silence_frames = 0
    if active_start is not None:
        regions.append(SpeechRegion(active_start, silence_start or last_end))
    return tuple(regions)


def _duration_frames(milliseconds: int) -> int:
    if milliseconds <= 0:
        raise ValueError("VAD durations must be positive")
    samples = milliseconds * SAMPLE_RATE
    return max(1, (samples + 1000 * FRAME_SAMPLES - 1) // (1000 * FRAME_SAMPLES))
