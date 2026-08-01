from pathlib import Path

import numpy as np
import pytest
from voicepad_core.vad import FRAME_SAMPLES, PauseTracker, SileroVad, VadError, VadFrame


class FakeSession:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = iter(probabilities)
        self.inputs: list[dict[str, np.ndarray]] = []

    def run(self, output_names: None, input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.inputs.append({key: value.copy() for key, value in input_feed.items()})
        state = input_feed["state"] + 1
        return [np.array([[next(self.probabilities)]], dtype=np.float32), state]


def test_silero_retains_context_and_recurrent_state_across_calls() -> None:
    session = FakeSession([0.25, 0.75])
    vad = SileroVad(Path("unused"), session=session)
    first = np.ones(FRAME_SAMPLES, dtype=np.float32)
    second = np.zeros(FRAME_SAMPLES, dtype=np.float32)

    frames = vad.accept(first, 0) + vad.accept(second, FRAME_SAMPLES)

    assert [frame.speech_probability for frame in frames] == pytest.approx([0.25, 0.75])
    assert np.all(session.inputs[1]["input"][0, :64] == 1.0)
    assert np.all(session.inputs[1]["state"] == 1.0)


def test_silero_buffers_partial_frames_and_pads_only_final_tail() -> None:
    session = FakeSession([0.5, 0.1])
    vad = SileroVad(Path("unused"), session=session)

    assert vad.accept(np.ones(300, dtype=np.float32), 0) == ()
    full = vad.accept(np.ones(212, dtype=np.float32), 300)
    tail = vad.accept(np.ones(10, dtype=np.float32), 512, final=True)

    assert full[0].end_sample == 512
    assert (tail[0].start_sample, tail[0].end_sample) == (512, 522)


def test_silero_rejects_nonsequential_input() -> None:
    vad = SileroVad(Path("unused"), session=FakeSession([]))

    with pytest.raises(VadError, match="must be sequential"):
        vad.accept(np.empty(0, dtype=np.float32), 1)


def test_pause_tracker_confirms_speech_then_500ms_silence() -> None:
    tracker = PauseTracker()
    frames = [VadFrame(index * 512, (index + 1) * 512, 0.8) for index in range(8)]
    frames.extend(VadFrame(index * 512, (index + 1) * 512, 0.1) for index in range(8, 24))

    pauses = [pause for frame in frames if (pause := tracker.accept(frame)) is not None]

    assert len(pauses) == 1
    assert pauses[0].start_sample == 8 * 512
    assert pauses[0].end_sample == 24 * 512
