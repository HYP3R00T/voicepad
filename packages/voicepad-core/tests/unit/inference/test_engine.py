from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
from voicepad_core.audio import RawAudio, WaveformSpec
from voicepad_core.config import Config
from voicepad_core.inference.contracts import (
    BackendCapabilities,
    BackendContract,
    OutputCapabilities,
    RuntimeInfo,
    RuntimeOptions,
    TranscriptionRequest,
)
from voicepad_core.inference.engine import _trim_trailing_silence, transcribe
from voicepad_core.inference.errors import AudioTooShortError, TranscriptionError
from voicepad_core.inference.runtime import RuntimeManager
from voicepad_core.inference.types import Segment, TranscriptionResult
from voicepad_core.models import Model

_WHISPER_CONTRACT = BackendContract(
    WaveformSpec(16_000),
    BackendCapabilities(
        word_timestamps=True,
        language_detection=True,
        translation=True,
        context_biasing=True,
        language_selection=True,
        beam_search=True,
        text_prompt=True,
        vad_filter=True,
        no_speech_threshold=True,
        hallucination_silence_threshold=True,
    ),
    OutputCapabilities(
        language="native",
        word_timestamps="native",
        word_confidence="native",
        segment_log_probability="native",
        no_speech_probability="native",
    ),
)
_PARAKEET_CONTRACT = BackendContract(
    WaveformSpec(16_000),
    BackendCapabilities(word_timestamps=True, context_biasing=True),
    OutputCapabilities(word_timestamps="derived"),
)


def _config(**updates: object) -> Config:
    return Config(
        recordings_path=Path("data/recordings"),
        markdown_path=Path("data/markdown"),
        model_cache_path=Path("data/models"),
        transcription_model="small",
        transcription_device="cpu",
        transcription_compute_type="int8",
    ).model_copy(update=updates)


def _result(**updates: object) -> TranscriptionResult:
    base = TranscriptionResult(
        text="VoicePad",
        segments=[Segment(start=0.0, end=1.0, text="VoicePad", avg_logprob=-0.2)],
        language="en",
        language_probability=0.99,
        duration_s=1.0,
        latency_ms=10.0,
        device="cpu",
        compute_type="int8",
        backend_id="faster-whisper",
        model_id="tiny",
        artifact_format="ctranslate2",
    )
    return replace(base, **updates)


class _Session:
    def __init__(
        self,
        result: TranscriptionResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.info = RuntimeInfo("faster-whisper", "tiny", "cpu", "int8")
        self.result = result or _result()
        self.error = error
        self.requests: list[TranscriptionRequest] = []

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        self.requests.append(request)
        if self.error:
            raise self.error
        return self.result

    def close(self) -> None:
        return None


class _Manager:
    def __init__(self, session: _Session, contract: BackendContract = _WHISPER_CONTRACT) -> None:
        self.session = session
        self._contract = contract
        self.opens: list[tuple[Model, RuntimeOptions]] = []

    def contract(self, model: Model) -> BackendContract:
        return self._contract

    def open(self, model: Model, options: RuntimeOptions) -> _Session:
        self.opens.append((model, options))
        return self.session


def _manager(session: _Session) -> RuntimeManager:
    return cast(RuntimeManager, _Manager(session))


class TestTrailingSilence:
    def test_keeps_audio_when_tail_has_energy(self) -> None:
        """A voiced final frame prevents trimming."""
        audio = np.ones(640, dtype=np.float32)

        assert len(_trim_trailing_silence(audio, 16_000)) == len(audio)

    def test_removes_complete_silent_tail_frames(self) -> None:
        """Silence is removed in complete configured frames."""
        audio = np.concatenate([np.ones(320, dtype=np.float32), np.zeros(640, dtype=np.float32)])

        assert len(_trim_trailing_silence(audio, 16_000)) == 320

    def test_leaves_short_audio_unchanged(self) -> None:
        """Audio shorter than one analysis frame is unchanged."""
        audio = np.zeros(100, dtype=np.float32)

        assert _trim_trailing_silence(audio, 16_000) is audio


class TestTranscribe:
    def test_rejects_audio_below_configured_minimum(self) -> None:
        """Short audio fails before a backend session is opened."""
        with pytest.raises(AudioTooShortError):
            transcribe(RawAudio(np.ones(100, dtype=np.float32), 16_000, 1), config=_config())

    def test_prepares_raw_audio_for_backend_contract(self) -> None:
        """Raw device metadata drives conversion at the backend handover."""
        session = _Session()

        result = transcribe(
            RawAudio(np.ones(8_000, dtype=np.float64), sample_rate=8_000, channels=1),
            config=_config(),
            runtime_manager=_manager(session),
        )

        request = session.requests[0]
        assert (request.sample_rate, request.audio.dtype, request.audio.size) == (
            16_000,
            np.dtype(np.float32),
            16_000,
        )
        assert result.audio_transformations == ("float32", "resample:8000->16000")
        assert (
            result.language_source,
            result.word_timestamp_source,
            result.word_confidence_source,
            result.segment_log_probability_source,
            result.no_speech_probability_source,
        ) == ("native", "native", "native", "native", "native")

    def test_passes_model_runtime_and_decoding_context(self) -> None:
        """Config becomes neutral runtime options and semantic backend context."""
        session = _Session()
        manager = _Manager(session)
        config = _config(
            language="en",
            beam_size=3,
            transcription_vad_filter=True,
            proper_nouns=("VoicePad", "HYP3R00T"),
            initial_prompt="Previous sentence.",
        )

        transcribe(
            RawAudio(np.ones(16_000, dtype=np.float32), 16_000, 1),
            word_timestamps=True,
            config=config,
            runtime_manager=cast(RuntimeManager, manager),
        )

        model, options = manager.opens[0]
        request = session.requests[0]
        assert (
            model.id,
            model.backend,
            options.device,
            options.precision,
            request.beam_size,
            request.word_timestamps,
            request.vad_filter,
            request.context.proper_nouns,
            request.context.previous_text,
        ) == (
            "small",
            "faster-whisper",
            "cpu",
            "int8",
            3,
            True,
            True,
            ("VoicePad", "HYP3R00T"),
            "Previous sentence.",
        )

    def test_explicit_arguments_override_config(self) -> None:
        """Call-level runtime and decoding options take precedence over Config."""
        session = _Session()
        manager = _Manager(session)

        transcribe(
            RawAudio(np.ones(16_000, dtype=np.float32), 16_000, 1),
            device="cuda",
            compute_type="float16",
            language="fr",
            beam_size=1,
            initial_prompt="Override",
            config=_config(),
            runtime_manager=cast(RuntimeManager, manager),
        )

        _, options = manager.opens[0]
        request = session.requests[0]
        assert (
            options.device,
            options.precision,
            request.language,
            request.beam_size,
            request.context.previous_text,
        ) == ("cuda", "float16", "fr", 1, "Override")

    def test_reports_options_ignored_by_backend_contract(self) -> None:
        """Unsupported controls remain visible instead of disappearing silently."""
        result = transcribe(
            RawAudio(np.ones(16_000, dtype=np.float32), 16_000, 1),
            language="fr",
            config=_config(beam_size=3, transcription_vad_filter=True),
            runtime_manager=cast(RuntimeManager, _Manager(_Session(), _PARAKEET_CONTRACT)),
        )

        assert set(result.ignored_options) >= {
            "language",
            "beam_size",
            "vad_filter",
            "no_speech_threshold",
            "hallucination_silence_threshold",
        }

    def test_applies_voicepad_text_postprocessing_once(self) -> None:
        """Enabled text cleanup changes result text after backend inference."""
        with (
            patch(
                "voicepad_core.inference.engine.remove_hallucinations",
                return_value="cleaned",
            ) as remove,
            patch(
                "voicepad_core.inference.engine.normalize",
                return_value="normalized",
            ) as normalize_text,
        ):
            result = transcribe(
                RawAudio(np.ones(16_000, dtype=np.float32), 16_000, 1),
                config=_config(text_postprocessing_enabled=True),
                runtime_manager=_manager(_Session()),
            )

        assert (result.text, remove.call_count, normalize_text.call_count) == (
            "normalized",
            1,
            1,
        )

    def test_does_not_postprocess_when_disabled(self) -> None:
        """Disabled cleanup returns backend text unchanged."""
        with patch("voicepad_core.inference.engine.normalize") as normalize_text:
            result = transcribe(
                RawAudio(np.ones(16_000, dtype=np.float32), 16_000, 1),
                config=_config(text_postprocessing_enabled=False),
                runtime_manager=_manager(_Session()),
            )

        assert (result.text, normalize_text.call_count) == ("VoicePad", 0)

    def test_preserves_backend_transcription_errors(self) -> None:
        """Backend-domain failures cross the public boundary unchanged."""
        error = TranscriptionError("backend unavailable")

        with pytest.raises(TranscriptionError) as exc_info:
            transcribe(
                RawAudio(np.ones(16_000, dtype=np.float32), 16_000, 1),
                config=_config(),
                runtime_manager=_manager(_Session(error=error)),
            )

        assert exc_info.value is error

    def test_wraps_unexpected_backend_errors(self) -> None:
        """Unexpected backend exceptions become the public TranscriptionError."""
        with pytest.raises(TranscriptionError, match="faster-whisper") as exc_info:
            transcribe(
                RawAudio(np.ones(16_000, dtype=np.float32), 16_000, 1),
                config=_config(),
                runtime_manager=_manager(_Session(error=RuntimeError("boom"))),
            )

        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_default_runtime_selects_parakeet_from_model_catalogue(self) -> None:
        """Without an injected manager, the selected Parakeet model reaches its backend."""
        manager = _Manager(_Session(), _PARAKEET_CONTRACT)
        with patch(
            "voicepad_core.inference.engine.get_runtime_manager",
            return_value=manager,
        ):
            transcribe(
                RawAudio(np.ones(16_000, dtype=np.float32), 16_000, 1),
                config=_config(
                    transcription_model="parakeet-tdt-0.6b-v3",
                    transcription_compute_type="float16",
                ),
            )

        model, options = manager.opens[0]
        assert (model.backend, options.precision) == ("sherpa-onnx", "float16")
