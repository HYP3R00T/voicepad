from __future__ import annotations

import gc
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from voicepad_core.inference.backends import FasterWhisperDriver, FasterWhisperSession
from voicepad_core.inference.contracts import (
    PreparedModel,
    RuntimeInfo,
    RuntimeOptions,
    TranscriptionContext,
    TranscriptionRequest,
)
from voicepad_core.inference.errors import TranscriptionError
from voicepad_core.models import Model


@dataclass
class _Word:
    word: str = " VoicePad"
    start: float = 0.1
    end: float = 0.4
    probability: float = 0.9


@dataclass
class _Segment:
    start: float = 0.0
    end: float = 0.5
    text: str = " VoicePad "
    avg_logprob: float = -0.2
    no_speech_prob: float = 0.1
    words: list[_Word] | None = None


@dataclass
class _Info:
    language: str = "en"
    language_probability: float = 0.95


class _Model:
    def __init__(
        self,
        segments: list[_Segment] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.segments = segments if segments is not None else [_Segment(words=[_Word()])]
        self.error = error
        self.calls: list[tuple[object, dict[str, object]]] = []

    def transcribe(self, audio: object, **kwargs: object) -> tuple[list[_Segment], _Info]:
        self.calls.append((audio, kwargs))
        if self.error is not None:
            raise self.error
        return self.segments, _Info()


def _spec(*, distil: bool = False, backend_id: str = "faster-whisper") -> Model:
    return Model(
        id="tiny",
        repo="owner/tiny",
        backend=backend_id,
        files=("model.bin",),
        label="Tiny",
        hint="Test model",
        accepts_prompt=not distil,
    )


def _prepared(*, distil: bool = False, backend_id: str = "faster-whisper") -> PreparedModel:
    return PreparedModel(
        _spec(distil=distil, backend_id=backend_id),
        Path("snapshot"),
    )


def _runtime_info(
    *,
    device: str = "cuda",
    precision: str = "float16",
    fallback: bool = False,
) -> RuntimeInfo:
    return RuntimeInfo(
        backend_id="faster-whisper",
        model_id="tiny",
        device=device,
        precision=precision,
        fallback_to_cpu=fallback,
    )


def _session(
    model: _Model,
    *,
    prepared: PreparedModel | None = None,
    device: str = "cuda",
    precision: str = "float16",
    fallback: bool = False,
    allow_cpu_fallback: bool = True,
) -> FasterWhisperSession:
    return FasterWhisperSession(
        prepared or _prepared(),
        cast(Any, model),
        _runtime_info(
            device=device,
            precision=precision,
            fallback=fallback,
        ),
        allow_cpu_fallback=allow_cpu_fallback,
    )


def _request(**kwargs: Any) -> TranscriptionRequest:
    return TranscriptionRequest(np.ones(16_000, dtype=np.float32), sample_rate=16_000, **kwargs)


class TestFasterWhisperDriver:
    def test_capabilities_are_explicit(self) -> None:
        """The driver advertises every backend feature it implements."""
        capabilities = FasterWhisperDriver().contract.decoding

        assert (
            capabilities.streaming,
            capabilities.word_timestamps,
            capabilities.language_detection,
            capabilities.translation,
            capabilities.context_biasing,
        ) == (False, True, True, True, True)

    def test_open_resolves_auto_runtime_and_loads_artifact(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Automatic options resolve before the driver constructs its native model."""
        calls: list[tuple[Path, str, str]] = []

        def load(artifact: Path, device: str, precision: str) -> _Model:
            calls.append((artifact, device, precision))
            return _Model()

        monkeypatch.setattr(
            "voicepad_core.inference.backends.faster_whisper._load_model",
            load,
        )

        session = FasterWhisperDriver().open(_prepared(), RuntimeOptions())

        assert (calls, session.info.device, session.info.precision) == (
            [(Path("snapshot"), "cuda", "float16")],
            "cuda",
            "float16",
        )

    def test_open_falls_back_from_cuda_to_cpu_when_allowed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A CUDA construction error retries once on CPU when fallback is allowed."""
        calls: list[tuple[str, str]] = []

        def load(_: Path, device: str, precision: str) -> _Model:
            calls.append((device, precision))
            if device == "cuda":
                raise RuntimeError("CUDA initialization failed")
            return _Model()

        monkeypatch.setattr(
            "voicepad_core.inference.backends.faster_whisper._load_model",
            load,
        )

        session = FasterWhisperDriver().open(
            _prepared(),
            RuntimeOptions(allow_cpu_fallback=True),
        )

        assert (
            calls,
            session.info.device,
            session.info.precision,
            session.info.fallback_to_cpu,
        ) == (
            [("cuda", "float16"), ("cpu", "int8")],
            "cpu",
            "int8",
            True,
        )

    def test_open_does_not_fallback_when_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A CUDA construction error remains an error when fallback is disabled."""
        calls: list[str] = []

        def fail(_: Path, device: str, _precision: str) -> _Model:
            calls.append(device)
            raise RuntimeError("CUDA initialization failed")

        monkeypatch.setattr(
            "voicepad_core.inference.backends.faster_whisper._load_model",
            fail,
        )

        with pytest.raises(TranscriptionError, match="CUDA initialization failed"):
            FasterWhisperDriver().open(
                _prepared(),
                RuntimeOptions(allow_cpu_fallback=False),
            )

        assert calls == ["cuda"]

    def test_open_reports_cpu_fallback_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Failure of both CUDA and CPU construction reports both lifecycle errors."""

        def fail(_: Path, device: str, _precision: str) -> _Model:
            raise RuntimeError(f"{device} load failed")

        monkeypatch.setattr(
            "voicepad_core.inference.backends.faster_whisper._load_model",
            fail,
        )

        with pytest.raises(TranscriptionError, match="CPU fallback failed"):
            FasterWhisperDriver().open(
                _prepared(),
                RuntimeOptions(allow_cpu_fallback=True),
            )

    def test_open_rejects_prepared_model_for_other_backend(self) -> None:
        """An artifact prepared for another backend cannot be opened."""
        with pytest.raises(TranscriptionError, match="not 'faster-whisper'"):
            FasterWhisperDriver().open(
                _prepared(backend_id="other"),
                RuntimeOptions(),
            )


class TestFasterWhisperSession:
    def test_transcribe_passes_neutral_decoding_arguments(self) -> None:
        """Neutral request controls map to faster-whisper inference arguments."""
        model = _Model()
        session = _session(model)

        session.transcribe(
            _request(
                language="en",
                beam_size=3,
                intent="translate",
                vad_filter=True,
                no_speech_threshold=0.4,
                hallucination_silence_threshold=1.5,
                word_timestamps=True,
            )
        )

        assert model.calls[0][1] == {
            "language": "en",
            "beam_size": 3,
            "task": "translate",
            "vad_filter": True,
            "hallucination_silence_threshold": 1.5,
            "no_speech_threshold": 0.4,
            "initial_prompt": None,
            "hotwords": None,
            "condition_on_previous_text": False,
            "word_timestamps": True,
        }

    def test_transcribe_maps_context_to_prompt_and_hotwords(self) -> None:
        """Previous text and proper nouns map to native context-biasing controls."""
        model = _Model()
        session = _session(model)
        context = TranscriptionContext(
            proper_nouns=("VoicePad", "Parakeet"),
            previous_text="Earlier sentence.",
        )

        session.transcribe(_request(context=context))

        assert (
            model.calls[0][1]["initial_prompt"],
            model.calls[0][1]["hotwords"],
        ) == ("Earlier sentence.", "VoicePad Parakeet")

    def test_transcribe_disables_prompt_for_distil_model(self) -> None:
        """Distil models do not receive unsupported previous-text context."""
        model = _Model()
        session = _session(model, prepared=_prepared(distil=True))

        session.transcribe(
            _request(
                context=TranscriptionContext(previous_text="Earlier sentence."),
            )
        )

        assert model.calls[0][1]["initial_prompt"] is None

    def test_transcribe_adapts_segments_words_and_provenance(self) -> None:
        """Raw backend output becomes a timed result with runtime provenance."""
        model = _Model([_Segment(end=2.0, words=[_Word()])])
        session = _session(model)

        result = session.transcribe(_request(word_timestamps=True))

        assert (
            result.text,
            result.segments[0].end,
            result.segments[0].words[0].word,
            result.language,
            result.backend_id,
            result.model_id,
            result.artifact_format,
        ) == (
            "VoicePad",
            1.0,
            " VoicePad",
            "en",
            "faster-whisper",
            "tiny",
            "ctranslate2",
        )

    def test_transcribe_filters_silence_and_out_of_bounds_segments(self) -> None:
        """Silence and segments beginning beyond the audio duration are omitted."""
        model = _Model([
            _Segment(text="keep", no_speech_prob=0.1),
            _Segment(text="silence", no_speech_prob=0.8),
            _Segment(start=1.0, end=1.2, text="late"),
        ])
        session = _session(model)

        result = session.transcribe(_request(no_speech_threshold=0.6))

        assert [segment.text for segment in result.segments] == ["keep"]

    def test_transcribe_retries_cuda_failure_on_cpu(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A CUDA inference failure replaces the session model with a CPU model."""
        gpu_model = _Model(error=RuntimeError("CUDA execution failed"))
        cpu_model = _Model()
        session = _session(gpu_model)
        loads: list[tuple[Path, str, str]] = []

        def load(artifact: Path, device: str, precision: str) -> _Model:
            loads.append((artifact, device, precision))
            return cpu_model

        monkeypatch.setattr(
            "voicepad_core.inference.backends.faster_whisper._load_model",
            load,
        )

        result = session.transcribe(_request())

        assert (
            loads,
            result.device,
            result.compute_type,
            result.fallback_to_cpu,
            len(cpu_model.calls),
        ) == (
            [(Path("snapshot"), "cpu", "int8")],
            "cpu",
            "int8",
            True,
            1,
        )

    def test_transcribe_does_not_fallback_when_disabled(self) -> None:
        """A CUDA inference failure remains an error when CPU fallback is disabled."""
        session = _session(
            _Model(error=RuntimeError("CUDA execution failed")),
            allow_cpu_fallback=False,
        )

        with pytest.raises(TranscriptionError, match="CUDA execution failed"):
            session.transcribe(_request())

    def test_transcribe_wraps_cpu_retry_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed CPU model construction remains a backend-neutral error."""
        session = _session(_Model(error=RuntimeError("CUDA execution failed")))

        def fail(*_: object) -> _Model:
            raise RuntimeError("CPU load failed")

        monkeypatch.setattr(
            "voicepad_core.inference.backends.faster_whisper._load_model",
            fail,
        )

        with pytest.raises(TranscriptionError, match="CPU retry failed"):
            session.transcribe(_request())

    def test_transcribe_wraps_non_cuda_failure(self) -> None:
        """Non-CUDA inference failures cross the backend boundary unchanged."""
        session = _session(_Model(error=ValueError("bad tokens")))

        with pytest.raises(TranscriptionError, match="bad tokens"):
            session.transcribe(_request())

    def test_close_releases_native_model_reference(self) -> None:
        """Closing a session releases its only retained native model reference."""
        model = _Model()
        reference = weakref.ref(model)
        session = _session(model)
        del model

        session.close()
        gc.collect()

        assert reference() is None

    def test_closed_session_rejects_reuse(self) -> None:
        """Closing twice is safe and a closed session rejects further inference."""
        session = _session(_Model())

        session.close()
        session.close()

        with pytest.raises(TranscriptionError, match="closed"):
            session.transcribe(_request())
