from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import numpy as np
from voicepad_core.config import Config
from voicepad_core.inference.errors import AudioTooShortError, TranscriptionError
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

    def get_snapshot(self):
        return np.array([], dtype=np.float32)


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


@patch("voicepad_core.streaming.transcriber.normalize", side_effect=lambda text: text.upper())
@patch(
    "voicepad_core.streaming.transcriber.remove_hallucinations", side_effect=lambda text, max_repetitions: text + "!"
)
@patch("voicepad_core.streaming.transcriber.deduplicate_overlap")
@patch("voicepad_core.streaming.transcriber.AudioPreProcessor")
@patch("voicepad_core.inference.transcribe")
def test_dispatch_chunk_uses_configured_postprocessing(
    mock_transcribe: Mock,
    mock_preprocessor: Mock,
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
    mock_preprocessor.return_value.process_array.side_effect = lambda audio, sample_rate: audio

    audio = np.zeros(32_000, dtype=np.float32)
    streamer._dispatch_chunk(audio, is_final=False, capture_rate=16_000)

    kwargs = mock_dedup.call_args.kwargs
    assert kwargs["prev_tail_words"] == 20
    assert kwargs["full_duplicate_threshold"] == 0.9
    assert kwargs["min_overlap_words_for_partial"] == 4
    assert kwargs["partial_lead_words"] == 6
    mock_preprocessor.assert_called_once_with(streamer._recorder)
    mock_preprocessor.return_value.process_array.assert_called_once()
    process_args = mock_preprocessor.return_value.process_array.call_args.args
    assert process_args[0].dtype == np.float32
    assert len(process_args[0]) == 28_000
    assert mock_preprocessor.return_value.process_array.call_args.kwargs == {"sample_rate": 16_000}
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
