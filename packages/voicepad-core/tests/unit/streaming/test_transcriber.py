from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import numpy as np
import pytest
from voicepad_core.audio import AudioWindow, RawAudio
from voicepad_core.config import Config
from voicepad_core.inference.errors import AudioTooShortError, TranscriptionError
from voicepad_core.inference.types import WordTimestamp
from voicepad_core.streaming.transcriber import StreamingTranscriber, _build_prompt, _resample


class _FakeThread:
    def __init__(self, *args, **kwargs) -> None:
        self.target = kwargs["target"]
        self.name = kwargs["name"]
        self.daemon = kwargs["daemon"]
        self.started = False
        self.join_timeout = None
        self.alive = False

    def start(self) -> None:
        self.started = True

    def join(self, timeout: float | None = None) -> None:
        self.join_timeout = timeout

    def is_alive(self) -> bool:
        return self.alive


class _FakeRecorder:
    sample_rate = 16_000

    def read_window(self, start_sample: int, max_samples: int) -> AudioWindow:
        return AudioWindow(np.array([], dtype=np.float32), start_sample)


class _FailingRecorder:
    sample_rate = 16_000

    def read_window(self, start_sample: int, max_samples: int) -> AudioWindow:
        raise RuntimeError("window read failed")


class _WindowRecorder:
    sample_rate = 16_000

    def __init__(self) -> None:
        self.requested_start: int | None = None
        self.requested_max: int | None = None

    def read_window(self, start_sample: int, max_samples: int) -> AudioWindow:
        self.requested_start = start_sample
        self.requested_max = max_samples
        return AudioWindow(np.zeros(min(16_000, max_samples), dtype=np.float32), start_sample)


def test_streamer_reads_only_unconsumed_audio_and_overlap() -> None:
    recorder = _WindowRecorder()
    streamer = StreamingTranscriber(recorder, lambda chunk: None, lambda error: None, config=make_config())
    streamer._consumed_samples = 32_000

    window = streamer._read_window(16_000)

    assert recorder.requested_start == 20_000
    assert recorder.requested_max == 396_000
    assert (window.start_sample, window.end_sample) == (20_000, 36_000)


def test_prepare_chunk_caps_disk_backlog_to_maximum_chunk() -> None:
    config = make_config().model_copy(update={"min_chunk_s": 1.0, "max_chunk_s": 2.0, "overlap_s": 0.5})
    streamer = StreamingTranscriber(_FakeRecorder(), lambda chunk: None, lambda error: None, config=config)
    streamer._consumed_samples = 32_000
    window = AudioWindow(np.zeros(80_000, dtype=np.float32), start_sample=24_000)

    chunk = streamer._prepare_chunk(window, is_final=False, capture_rate=16_000)

    assert chunk is not None
    assert len(chunk.audio) == 40_000
    assert chunk.end_sample == 64_000
    assert (chunk.start_s, chunk.end_s) == (2.0, 4.0)


class _ResetRequiredVAD:
    """Stateful VAD fake that detects speech only once after each reset."""

    def __init__(self) -> None:
        self._ready = False

    def reset(self) -> None:
        self._ready = True

    def detect(self, audio: np.ndarray, sample_rate: int):
        if not self._ready:
            return []
        self._ready = False
        return [SimpleNamespace(start=0.0, end=0.5)]


def make_config() -> Config:
    return Config(
        recordings_path=Path("data/recordings"),
        markdown_path=Path("data/markdown"),
        transcription_model="base",
        transcription_device="cpu",
        transcription_compute_type="int8",
        beam_size=3,
        transcription_vad_filter=True,
        min_chunk_s=12.0,
        max_chunk_s=24.0,
        overlap_s=0.75,
        silence_threshold_ms=900,
        stream_poll_interval_s=0.15,
        stream_context_chars=5,
        hallucination_max_repetitions=2,
        dedup_prev_tail_words=20,
        dedup_full_duplicate_threshold=0.9,
        dedup_min_overlap_words_for_partial=4,
        dedup_partial_lead_words=6,
        vad_threshold=0.65,
        vad_min_speech_duration_ms=320,
        vad_speech_pad_ms=45,
        initial_prompt="Prompt seed",
        text_postprocessing_enabled=True,
    )


def test_build_prompt_with_context() -> None:
    assert _build_prompt("world", "hello") == "hello world"


def test_build_prompt_without_context() -> None:
    assert _build_prompt("", "hello") == "hello"


def test_resample_returns_same_audio_for_same_rate() -> None:
    audio = np.array([0.0, 1.0], dtype=np.float32)
    result = _resample(audio, 16_000, 16_000)
    assert result is audio


def test_resample_changes_length_for_new_rate() -> None:
    audio = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    result = _resample(audio, 8_000, 16_000)
    assert len(result) == 8
    assert result.dtype == np.float32


@patch("voicepad_core.streaming.transcriber.get_config")
def test_init_uses_config_defaults(mock_get_config: Mock) -> None:
    config = make_config()
    mock_get_config.return_value = config
    streamer = StreamingTranscriber(_FakeRecorder(), lambda chunk: None, lambda err: None)

    assert streamer._model_name == "base"
    assert streamer._device == "cpu"
    assert streamer._compute_type == "int8"
    assert streamer._min_chunk_s == 12.0
    assert streamer._max_chunk_s == 24.0
    assert streamer._overlap_s == 0.75
    assert streamer._silence_threshold_ms == 900
    assert streamer._beam_size == 3
    assert streamer._vad_filter is True
    assert streamer._poll_interval_s == 0.15
    assert streamer._stream_context_chars == 5


@patch("voicepad_core.streaming.transcriber.threading.Thread", new=_FakeThread)
@patch("voicepad_core.streaming.transcriber.SileroVAD")
def test_start_builds_vad_from_config(mock_vad: Mock) -> None:
    config = make_config()
    vad_instance = Mock()
    mock_vad.return_value = vad_instance
    streamer = StreamingTranscriber(_FakeRecorder(), lambda chunk: None, lambda err: None, config=config)

    streamer.start()

    mock_vad.assert_called_once_with(
        threshold=0.65,
        min_speech_duration_ms=320,
        min_silence_duration_ms=900,
        speech_pad_ms=45,
        config=config,
    )
    vad_instance.reset.assert_called_once_with()
    assert isinstance(streamer._monitor_thread, _FakeThread)
    assert streamer._monitor_thread.started is True


def test_stop_is_noop_without_thread() -> None:
    streamer = StreamingTranscriber(_FakeRecorder(), lambda chunk: None, lambda err: None, config=make_config())
    streamer.stop()
    assert streamer._monitor_thread is None


def test_stop_joins_thread_and_clears_it() -> None:
    streamer = StreamingTranscriber(_FakeRecorder(), lambda chunk: None, lambda err: None, config=make_config())
    fake_thread = _FakeThread(target=lambda: None, name="stream-vad", daemon=True)
    streamer._monitor_thread = cast(Any, fake_thread)

    streamer.stop()

    assert fake_thread.join_timeout == 60.0
    assert streamer._monitor_thread is None


def test_monitor_loop_reports_final_snapshot_failure_and_completes() -> None:
    received = []
    errors = []
    streamer = StreamingTranscriber(_FailingRecorder(), received.append, errors.append, config=make_config())
    streamer._stop_event.set()

    streamer._monitor_loop()

    assert errors == ["window read failed"]
    assert [(chunk.is_final, chunk.text) for chunk in received] == [(True, "")]


@patch("voicepad_core.streaming.transcriber.normalize", side_effect=lambda text: text.upper())
@patch(
    "voicepad_core.streaming.transcriber.remove_hallucinations", side_effect=lambda text, max_repetitions: text + "!"
)
@patch("voicepad_core.streaming.transcriber.deduplicate_overlap")
@patch("voicepad_core.inference.transcribe")
def test_dispatch_chunk_uses_configured_postprocessing(
    mock_transcribe: Mock,
    mock_dedup: Mock,
    mock_remove: Mock,
    mock_normalize: Mock,
) -> None:
    received = []
    config = make_config()
    streamer = StreamingTranscriber(_FakeRecorder(), received.append, lambda err: None, config=config)
    streamer._prev_chunk_text = "previous chunk text"
    streamer._consumed_samples = 16_000

    segment = SimpleNamespace(
        start=0.0,
        end=1.2,
        text="hello",
        avg_logprob=-0.1,
        no_speech_prob=0.1,
        words=[],
    )
    result = SimpleNamespace(
        segments=[segment],
        latency_ms=123.0,
        device="cpu",
        language="en",
        language_probability=0.99,
    )
    mock_transcribe.return_value = result
    mock_dedup.side_effect = lambda segments, *_args, **_kwargs: segments

    audio = np.zeros(32_000, dtype=np.float32)
    streamer._dispatch_chunk(audio, is_final=False, capture_rate=16_000)

    kwargs = mock_dedup.call_args.kwargs
    assert kwargs["prev_tail_words"] == 20
    assert kwargs["full_duplicate_threshold"] == 0.9
    assert kwargs["min_overlap_words_for_partial"] == 4
    assert kwargs["partial_lead_words"] == 6
    transcription_audio = mock_transcribe.call_args.args[0]
    assert isinstance(transcription_audio, RawAudio)
    assert transcription_audio.samples.dtype == np.float32
    assert len(transcription_audio.samples) == 28_000
    assert transcription_audio.sample_rate == 16_000
    mock_remove.assert_called_once_with("hello", max_repetitions=2)
    mock_normalize.assert_called_once_with("hello!")
    assert streamer._prev_context == "ELLO!"
    assert received[0].text == "HELLO!"


@patch("voicepad_core.inference.transcribe", side_effect=AudioTooShortError("too short"))
def test_dispatch_chunk_emits_empty_final_marker_for_short_audio(mock_transcribe: Mock) -> None:
    received = []
    streamer = StreamingTranscriber(_FakeRecorder(), received.append, lambda err: None, config=make_config())

    streamer._dispatch_chunk(np.zeros(100, dtype=np.float32), is_final=True, capture_rate=16_000)

    assert received[0].is_final is True
    assert received[0].text == ""


@patch("voicepad_core.inference.transcribe", side_effect=TranscriptionError("boom"))
def test_dispatch_chunk_reports_errors(mock_transcribe: Mock) -> None:
    errors = []
    streamer = StreamingTranscriber(_FakeRecorder(), lambda chunk: None, errors.append, config=make_config())

    streamer._dispatch_chunk(np.zeros(16_000, dtype=np.float32), is_final=False, capture_rate=16_000)

    assert errors == ["boom"]


@patch("voicepad_core.inference.transcribe")
def test_dispatch_chunk_shifts_word_timestamps_to_session_time(
    mock_transcribe: Mock,
) -> None:
    """Overlapped chunks expose both segment and word timestamps in absolute session time."""
    received = []
    config = make_config().model_copy(update={"text_postprocessing_enabled": False})
    streamer = StreamingTranscriber(_FakeRecorder(), received.append, lambda err: None, config=config)
    streamer._consumed_samples = 16_000
    mock_transcribe.return_value = SimpleNamespace(
        segments=[
            SimpleNamespace(
                start=0.8,
                end=1.2,
                text="hello",
                avg_logprob=-0.1,
                no_speech_prob=0.1,
                words=[WordTimestamp(word="hello", start=0.9, end=1.0, probability=0.95)],
            )
        ],
        latency_ms=10.0,
        device="cpu",
        language="en",
        language_probability=0.99,
    )

    streamer._dispatch_chunk(np.ones(32_000, dtype=np.float32), is_final=False, capture_rate=16_000)

    segment = received[0].segments[0]
    word = segment.words[0]
    assert (segment.start, segment.end, word.start, word.end) == pytest.approx((1.05, 1.45, 1.15, 1.25))


@patch("voicepad_core.inference.transcribe", side_effect=TranscriptionError("backend unavailable"))
def test_dispatch_chunk_emits_final_marker_when_backend_fails(
    mock_transcribe: Mock,
) -> None:
    """A final backend failure reports the error and still completes the result stream."""
    received = []
    errors = []
    config = make_config().model_copy(update={"text_postprocessing_enabled": False})
    streamer = StreamingTranscriber(_FakeRecorder(), received.append, errors.append, config=config)
    streamer._vad = cast(Any, _ResetRequiredVAD())

    streamer._dispatch_chunk(np.ones(16_000, dtype=np.float32), is_final=True, capture_rate=16_000)

    assert (errors, [(chunk.is_final, chunk.text) for chunk in received]) == (
        ["backend unavailable"],
        [(True, "")],
    )


def test_final_tail_detection_is_independent_across_repeated_checks() -> None:
    """Each final-tail check resets VAD state so prior detection cannot suppress speech."""
    streamer = StreamingTranscriber(_FakeRecorder(), lambda chunk: None, lambda err: None, config=make_config())
    streamer._vad = cast(Any, _ResetRequiredVAD())
    audio = np.ones(16_000, dtype=np.float32)

    first = streamer._trim_final_audio_to_speech(audio, capture_rate=16_000)
    second = streamer._trim_final_audio_to_speech(audio, capture_rate=16_000)

    assert (len(first), len(second)) == (8_000, 8_000)


@patch("voicepad_core.inference.transcribe")
def test_dispatch_chunk_skips_final_tail_without_fresh_speech(mock_transcribe: Mock) -> None:
    received = []
    config = make_config().model_copy(update={"text_postprocessing_enabled": False})
    streamer = StreamingTranscriber(_FakeRecorder(), received.append, lambda err: None, config=config)
    streamer._consumed_samples = 32_000
    streamer._vad = Mock()
    streamer._vad.detect.return_value = []

    full_audio = np.zeros(32_100, dtype=np.float32)
    streamer._dispatch_chunk(full_audio, is_final=True, capture_rate=16_000)

    streamer._vad.reset.assert_called_once_with()
    streamer._vad.detect.assert_called_once()
    mock_transcribe.assert_not_called()
    assert received[0].is_final is True
    assert received[0].text == ""


@patch("voicepad_core.inference.transcribe")
def test_dispatch_chunk_skips_final_tail_below_min_fresh_speech(mock_transcribe: Mock) -> None:
    received = []
    config = make_config().model_copy(update={"text_postprocessing_enabled": False, "min_fresh_speech_duration_s": 0.4})
    streamer = StreamingTranscriber(_FakeRecorder(), received.append, lambda err: None, config=config)
    streamer._consumed_samples = 32_000
    streamer._vad = Mock()
    streamer._vad.detect.return_value = [SimpleNamespace(start=0.0, end=0.2)]

    full_audio = np.ones(44_000, dtype=np.float32)
    streamer._dispatch_chunk(full_audio, is_final=True, capture_rate=16_000)

    mock_transcribe.assert_not_called()
    assert received[0].is_final is True
    assert received[0].text == ""


@patch("voicepad_core.streaming.transcriber.normalize", side_effect=lambda text: text.upper())
@patch(
    "voicepad_core.streaming.transcriber.remove_hallucinations", side_effect=lambda text, max_repetitions: text + "!"
)
@patch("voicepad_core.inference.transcribe")
def test_dispatch_chunk_bypasses_text_postprocessing_when_disabled(
    mock_transcribe: Mock,
    mock_remove: Mock,
    mock_normalize: Mock,
) -> None:
    received = []
    config = make_config().model_copy(update={"text_postprocessing_enabled": False})
    streamer = StreamingTranscriber(_FakeRecorder(), received.append, lambda err: None, config=config)

    segment = SimpleNamespace(
        start=0.0,
        end=1.2,
        text="hello",
        avg_logprob=-0.1,
        no_speech_prob=0.1,
        words=[],
    )
    result = SimpleNamespace(
        segments=[segment],
        latency_ms=123.0,
        device="cpu",
        language="en",
        language_probability=0.99,
    )
    mock_transcribe.return_value = result

    audio = np.ones(32_000, dtype=np.float32)
    streamer._dispatch_chunk(audio, is_final=False, capture_rate=16_000)

    mock_remove.assert_not_called()
    mock_normalize.assert_not_called()
    assert received[0].text == "hello"


@patch("voicepad_core.inference.transcribe")
def test_dispatch_chunk_trims_final_audio_to_last_vad_speech_boundary(
    mock_transcribe: Mock,
) -> None:
    received = []
    config = make_config().model_copy(update={"text_postprocessing_enabled": False})
    streamer = StreamingTranscriber(_FakeRecorder(), received.append, lambda err: None, config=config)
    streamer._consumed_samples = 32_000
    streamer._vad = Mock()
    streamer._vad.detect.return_value = [SimpleNamespace(start=0.0, end=0.35)]

    segment = SimpleNamespace(
        start=0.5,
        end=0.8,
        text="final words",
        avg_logprob=-0.2,
        no_speech_prob=0.1,
        words=[],
    )
    result = SimpleNamespace(
        segments=[segment],
        latency_ms=123.0,
        device="cpu",
        language="en",
        language_probability=0.99,
    )
    mock_transcribe.return_value = result

    full_audio = np.ones(44_000, dtype=np.float32)
    streamer._dispatch_chunk(full_audio, is_final=True, capture_rate=16_000)

    transcription_audio = mock_transcribe.call_args.args[0]
    assert isinstance(transcription_audio, RawAudio)
    assert len(transcription_audio.samples) == 12_000 + int(round(0.35 * 16_000))
    assert received[0].is_final is True
    assert received[0].text == "final words"
