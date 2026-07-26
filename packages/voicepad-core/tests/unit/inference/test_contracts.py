"""Tests for backend-neutral inference contracts."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from voicepad_core.audio import WaveformSpec
from voicepad_core.inference.contracts import (
    BackendCapabilities,
    BackendContract,
    BackendDriver,
    DecodingIntent,
    PreparedModel,
    RuntimeInfo,
    RuntimeOptions,
    TranscriptionContext,
    TranscriptionRequest,
    TranscriptionSession,
)
from voicepad_core.inference.types import TranscriptionResult
from voicepad_core.models import HuggingFaceArtifact, ModelSpec


def _result() -> TranscriptionResult:
    return TranscriptionResult(
        text="VoicePad",
        segments=[],
        language="en",
        language_probability=1.0,
        duration_s=1.0,
        latency_ms=10.0,
        device="cuda",
        compute_type="int8",
    )


class _Session:
    @property
    def info(self) -> RuntimeInfo:
        return RuntimeInfo("test", "tiny", "cuda", "int8")

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        return _result()

    def close(self) -> None:
        return None


class _Driver:
    @property
    def id(self) -> str:
        return "test"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(word_timestamps=True)

    @property
    def contract(self) -> BackendContract:
        return BackendContract(WaveformSpec(16_000), self.capabilities)

    def is_available(self) -> bool:
        return True

    def prepare(self, model: ModelSpec) -> PreparedModel:
        return PreparedModel(model, Path("model.bin"))

    def open(self, model: PreparedModel, options: RuntimeOptions) -> TranscriptionSession:
        return _Session()


class TestBackendCapabilities:
    def test_defaults_describe_no_optional_features(self) -> None:
        """A backend must opt in to every optional capability."""
        assert BackendCapabilities() == BackendCapabilities(
            streaming=False,
            word_timestamps=False,
            language_detection=False,
            translation=False,
            context_biasing=False,
        )

    def test_capabilities_are_immutable(self) -> None:
        """Capabilities cannot change after a driver publishes them."""
        dataclass_parameters = BackendCapabilities.__dataclass_params__

        assert dataclass_parameters.frozen is True


class TestRuntimeMetadata:
    @pytest.mark.parametrize("field", ["device", "precision"])
    def test_runtime_options_reject_empty_values(self, field: str) -> None:
        """Runtime requests reject blank device and precision values."""
        values = {"device": "cuda", "precision": "int8"}
        values[field] = " "

        with pytest.raises(ValueError, match=field):
            RuntimeOptions(device=values["device"], precision=values["precision"])

    @pytest.mark.parametrize("field", ["backend_id", "model_id", "device", "precision"])
    def test_runtime_info_rejects_empty_actual_values(self, field: str) -> None:
        """Runtime reports reject missing execution identity fields."""
        values = {
            "backend_id": "faster-whisper",
            "model_id": "tiny",
            "device": "cuda",
            "precision": "int8",
        }
        values[field] = ""

        with pytest.raises(ValueError, match=field):
            RuntimeInfo(
                backend_id=values["backend_id"],
                model_id=values["model_id"],
                device=values["device"],
                precision=values["precision"],
            )

    def test_runtime_info_records_cpu_fallback(self) -> None:
        """Runtime reports distinguish requested execution from actual CPU fallback."""
        info = RuntimeInfo("faster-whisper", "tiny", "cpu", "int8", fallback_to_cpu=True)

        assert info.fallback_to_cpu is True


class TestTranscriptionContext:
    def test_proper_nouns_are_stored_as_an_immutable_tuple(self) -> None:
        """Mutable proper-noun input is copied into immutable context storage."""
        terms = ["VoicePad", "Parakeet"]

        context = TranscriptionContext(proper_nouns=cast(tuple[str, ...], terms))
        terms.append("Whisper")

        assert context.proper_nouns == ("VoicePad", "Parakeet")

    def test_empty_proper_noun_is_rejected(self) -> None:
        """A blank semantic hint is rejected before it reaches a backend."""
        with pytest.raises(ValueError, match="proper_nouns"):
            TranscriptionContext(proper_nouns=("VoicePad", " "))


class TestTranscriptionRequest:
    def test_defaults_capture_decoding_intent(self) -> None:
        """A canonical request defaults to transcription with beam search."""
        request = TranscriptionRequest(np.zeros(16_000, dtype=np.float32), sample_rate=16_000)

        assert (
            request.intent,
            request.beam_size,
            request.word_timestamps,
            request.vad_filter,
            request.no_speech_threshold,
            request.hallucination_silence_threshold,
        ) == ("transcribe", 5, False, False, 0.6, 2.0)

    def test_non_array_audio_is_rejected(self) -> None:
        """Backends never receive audio outside the canonical ndarray boundary."""
        with pytest.raises(TypeError, match="numpy.ndarray"):
            TranscriptionRequest(cast(NDArray[np.float32], [0.0]), sample_rate=16_000)

    def test_multichannel_audio_is_rejected(self) -> None:
        """Backends receive mono audio after VoicePad preprocessing."""
        with pytest.raises(ValueError, match="mono"):
            TranscriptionRequest(np.zeros((2, 100), dtype=np.float32), sample_rate=16_000)

    def test_non_float32_audio_is_rejected(self) -> None:
        """Backends receive one consistent sample representation."""
        with pytest.raises(ValueError, match="float32"):
            TranscriptionRequest(
                cast(NDArray[np.float32], np.zeros(100, dtype=np.float64)),
                sample_rate=16_000,
            )

    def test_non_positive_sample_rate_is_rejected(self) -> None:
        """A request cannot describe audio with an invalid sample rate."""
        with pytest.raises(ValueError, match="sample_rate"):
            TranscriptionRequest(np.zeros(100, dtype=np.float32), sample_rate=0)

    def test_blank_language_is_rejected(self) -> None:
        """Automatic language detection uses None rather than an empty code."""
        with pytest.raises(ValueError, match="language"):
            TranscriptionRequest(np.zeros(100, dtype=np.float32), sample_rate=16_000, language=" ")

    def test_non_positive_beam_size_is_rejected(self) -> None:
        """A decoding request requires at least one beam."""
        with pytest.raises(ValueError, match="beam_size"):
            TranscriptionRequest(np.zeros(100, dtype=np.float32), sample_rate=16_000, beam_size=0)

    def test_unknown_decoding_intent_is_rejected(self) -> None:
        """Only transcription and translation intents cross the driver boundary."""
        with pytest.raises(ValueError, match="decoding intent"):
            TranscriptionRequest(
                np.zeros(100, dtype=np.float32),
                sample_rate=16_000,
                intent=cast(DecodingIntent, "summarize"),
            )

    @pytest.mark.parametrize("threshold", [-0.1, 1.1])
    def test_out_of_range_no_speech_threshold_is_rejected(self, threshold: float) -> None:
        """No-speech filtering uses a probability threshold within the unit interval."""
        with pytest.raises(ValueError, match="no_speech_threshold"):
            TranscriptionRequest(
                np.zeros(100, dtype=np.float32),
                sample_rate=16_000,
                no_speech_threshold=threshold,
            )

    def test_negative_hallucination_silence_threshold_is_rejected(self) -> None:
        """Hallucination silence duration cannot be negative."""
        with pytest.raises(ValueError, match="hallucination_silence_threshold"):
            TranscriptionRequest(
                np.zeros(100, dtype=np.float32),
                sample_rate=16_000,
                hallucination_silence_threshold=-0.1,
            )


class TestPreparedModel:
    def test_prepared_model_retains_registry_identity(self) -> None:
        """Prepared artifacts remain tied to their model registry entry."""
        spec = ModelSpec("tiny", HuggingFaceArtifact("owner/tiny"))

        prepared = PreparedModel(spec, Path("model.bin"))

        assert prepared.spec is spec


class TestRuntimeProtocols:
    def test_driver_is_runtime_checkable(self) -> None:
        """A structurally complete backend satisfies the driver contract at runtime."""
        assert isinstance(_Driver(), BackendDriver)

    def test_session_is_runtime_checkable(self) -> None:
        """A structurally complete session satisfies the session contract at runtime."""
        assert isinstance(_Session(), TranscriptionSession)

    def test_incomplete_driver_fails_runtime_check(self) -> None:
        """An object missing driver operations does not satisfy the contract."""
        assert not isinstance(object(), BackendDriver)
