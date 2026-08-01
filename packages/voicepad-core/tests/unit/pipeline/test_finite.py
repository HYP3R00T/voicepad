import numpy as np
from voicepad_core.audio import RawAudio
from voicepad_core.deployments import PARAKEET_V3_CUDA, PARAKEET_V3_MANIFEST, HuggingFaceSource
from voicepad_core.inference import (
    ActiveDeployment,
    BackendResult,
    CancellationToken,
    TimedWord,
    TokenTimestamp,
)
from voicepad_core.pipeline import FiniteFileTranscriber
from voicepad_core.vad import VadFrame

SR = 16_000


class SpeechVad:
    def reset(self) -> None:
        pass

    def accept(self, samples: np.ndarray, start_sample: int, *, final: bool = False) -> tuple[VadFrame, ...]:
        return tuple(VadFrame(index, min(index + 512, len(samples)), 0.9) for index in range(0, len(samples), 512))


class FakeEngine:
    def __init__(self, *, failure: Exception | None = None) -> None:
        source = PARAKEET_V3_MANIFEST.source
        assert isinstance(source, HuggingFaceSource)
        self.active_deployment = ActiveDeployment(
            PARAKEET_V3_CUDA,
            source.revision,
            "GPU-test",
            "NVIDIA Test GPU",
            4_000_000_000,
        )
        self.failure = failure

    def transcribe(self, audio, intent=None, cancellation=None):  # type: ignore[no-untyped-def]
        if self.failure is not None:
            raise self.failure
        words = (TimedWord("working", 0.5, 1.5),)
        tokens = (TokenTimestamp("▁working", 0.5, 1.5),)
        return BackendResult("working", tokens, words)


def raw_audio() -> RawAudio:
    return RawAudio(np.zeros(2 * SR, dtype=np.float32), SR, 1)


def test_finite_pipeline_returns_one_authoritative_complete_result() -> None:
    result = FiniteFileTranscriber(FakeEngine(), SpeechVad()).transcribe_audio(raw_audio())

    assert result.text == "working"
    assert result.complete is True
    assert len(result.chunks) == 1
    assert len(result.speech_regions) == 1
    assert result.coverage_gaps == ()


def test_finite_pipeline_returns_honest_partial_failure() -> None:
    result = FiniteFileTranscriber(FakeEngine(failure=RuntimeError("failed")), SpeechVad()).transcribe_audio(
        raw_audio()
    )

    assert result.text == ""
    assert result.complete is False
    assert result.chunks[0].error == "RuntimeError: failed"
    assert result.coverage_gaps


def test_finite_pipeline_honors_precancelled_job() -> None:
    cancellation = CancellationToken()
    cancellation.cancel()

    result = FiniteFileTranscriber(FakeEngine(), SpeechVad()).transcribe_audio(raw_audio(), cancellation=cancellation)

    assert result.complete is False
    assert result.chunks == ()
    assert "cancelled" in result.warnings[0]
