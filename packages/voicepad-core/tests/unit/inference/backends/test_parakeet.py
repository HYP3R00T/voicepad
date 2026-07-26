from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from voicepad_core.inference.backends.parakeet import ParakeetOnnxDriver
from voicepad_core.inference.contracts import (
    PreparedModel,
    RuntimeInfo,
    RuntimeOptions,
    TranscriptionRequest,
)
from voicepad_core.inference.errors import TranscriptionError
from voicepad_core.models import HuggingFaceArtifact, LocalArtifact, ModelSpec

_FILES = (
    "config.json",
    "decoder_joint-model.fp16.onnx",
    "encoder-model.fp16.onnx",
    "nemo128.onnx",
    "vocab.txt",
)


class _SessionOptions:
    def __init__(self) -> None:
        self.log_severity_level = 0
        self.entries: dict[str, str] = {}

    def add_session_config_entry(self, key: str, value: str) -> None:
        self.entries[key] = value


class _Runtime:
    SessionOptions = _SessionOptions

    def __init__(self, providers: list[str]) -> None:
        self._providers = providers
        self.preload_calls: list[dict[str, str]] = []

    def get_available_providers(self) -> list[str]:
        return self._providers

    def preload_dlls(self, **kwargs: str) -> None:
        self.preload_calls.append(kwargs)


class _Recognizer:
    def __init__(
        self,
        result: object = " VoicePad works. ",
        providers: list[str] | None = None,
    ) -> None:
        self.result = result
        self.calls: list[tuple[np.ndarray[Any, Any], int]] = []
        session = SimpleNamespace(get_providers=lambda: providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.asr = SimpleNamespace(_encoder=session, _decoder_joint=session)

    def recognize(self, audio: np.ndarray[Any, Any], *, sample_rate: int) -> object:
        self.calls.append((audio, sample_rate))
        return self.result


class _Loader:
    def __init__(self, recognizer: _Recognizer) -> None:
        self.recognizer = recognizer
        self.calls: list[tuple[str, Path, dict[str, object]]] = []

    def __call__(self, model_type: str, path: Path, **kwargs: object) -> _Recognizer:
        self.calls.append((model_type, path, kwargs))
        return self.recognizer


def _spec(
    *,
    backend_id: str = "parakeet-onnx",
    artifact_source: HuggingFaceArtifact | LocalArtifact | None = None,
) -> ModelSpec:
    return ModelSpec(
        id="parakeet-tdt-0.6b-v3",
        family="parakeet",
        backend_id=backend_id,
        artifact_format="onnx",
        quantization="fp16",
        artifact_source=artifact_source or HuggingFaceArtifact("ysdede/parakeet-tdt-0.6b-v3-onnx"),
        required_files=_FILES,
    )


def _model_dir(path: Path) -> Path:
    for filename in _FILES:
        (path / filename).write_bytes(b"model")
    return path


def _prepared(path: Path) -> PreparedModel:
    model_dir = _model_dir(path)
    return PreparedModel(_spec(artifact_source=LocalArtifact(model_dir)), model_dir)


def _request(**kwargs: Any) -> TranscriptionRequest:
    sample_rate = kwargs.pop("sample_rate", 16_000)
    return TranscriptionRequest(np.ones(16_000, dtype=np.float32), sample_rate=sample_rate, **kwargs)


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    providers: list[str] | None = None,
    session_providers: list[str] | None = None,
    result: object = " VoicePad works. ",
) -> tuple[_Runtime, _Loader]:
    runtime = _Runtime(providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])
    loader = _Loader(_Recognizer(result, session_providers))
    monkeypatch.setattr(
        "voicepad_core.inference.backends.parakeet._load_runtime",
        lambda: (runtime, loader),
    )
    return runtime, loader


class TestParakeetOnnxDriver:
    def test_capabilities_do_not_claim_unimplemented_features(self) -> None:
        driver = ParakeetOnnxDriver()

        assert driver.capabilities.context_biasing is False
        assert driver.capabilities.word_timestamps is False
        assert driver.capabilities.translation is False
        assert driver.contract.audio.peak_normalize is True

    def test_prepare_accepts_declared_onnx_layout(self, tmp_path: Path) -> None:
        model_dir = _model_dir(tmp_path)

        prepared = ParakeetOnnxDriver(cache_dir=tmp_path / "cache").prepare(
            _spec(artifact_source=LocalArtifact(model_dir))
        )

        assert prepared.artifact_path == model_dir

    def test_prepare_rejects_missing_files(self, tmp_path: Path) -> None:
        with pytest.raises(TranscriptionError, match="missing required files"):
            ParakeetOnnxDriver(cache_dir=tmp_path / "cache").prepare(_spec(artifact_source=LocalArtifact(tmp_path)))

    def test_prepare_rejects_backend_mismatch(self) -> None:
        with pytest.raises(TranscriptionError, match="not 'parakeet-onnx'"):
            ParakeetOnnxDriver().prepare(_spec(backend_id="other"))

    def test_open_pins_cuda_and_verifies_cuda_sessions(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime, loader = _patch_runtime(monkeypatch)

        session = ParakeetOnnxDriver().open(
            _prepared(tmp_path),
            RuntimeOptions(device="cuda", precision="auto"),
        )

        assert session.info == RuntimeInfo(
            "parakeet-onnx",
            "parakeet-tdt-0.6b-v3",
            "cuda",
            "fp16",
        )
        assert runtime.preload_calls == [{"directory": ""}]
        _, path, options = loader.calls[0]
        session_options = options["sess_options"]
        assert path == tmp_path
        assert options["providers"] == ["CUDAExecutionProvider"]
        assert options["quantization"] == "fp16"
        assert isinstance(session_options, _SessionOptions)
        assert session_options.log_severity_level == 3
        assert session_options.entries == {}

    def test_open_fails_when_cuda_provider_is_unavailable(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_runtime(monkeypatch, providers=["CPUExecutionProvider"])

        with pytest.raises(TranscriptionError, match="CUDAExecutionProvider"):
            ParakeetOnnxDriver().open(_prepared(tmp_path), RuntimeOptions())

    def test_open_rejects_silent_session_cpu_fallback(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_runtime(monkeypatch, session_providers=["CPUExecutionProvider"])

        with pytest.raises(TranscriptionError, match="CPU inference fallback is disabled"):
            ParakeetOnnxDriver().open(_prepared(tmp_path), RuntimeOptions())

    @pytest.mark.parametrize(
        ("options", "message"),
        (
            (RuntimeOptions(device="cpu"), "CUDA-only"),
            (RuntimeOptions(precision="int8"), "supports"),
        ),
    )
    def test_open_rejects_incompatible_runtime_options(
        self,
        tmp_path: Path,
        options: RuntimeOptions,
        message: str,
    ) -> None:
        with pytest.raises(TranscriptionError, match=message):
            ParakeetOnnxDriver().open(_prepared(tmp_path), options)

    def test_open_accepts_voicepad_float16_alias(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_runtime(monkeypatch)

        session = ParakeetOnnxDriver().open(
            _prepared(tmp_path),
            RuntimeOptions(precision="float16"),
        )

        assert session.info.precision == "fp16"


class TestParakeetOnnxSession:
    def test_transcribe_adapts_onnx_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, loader = _patch_runtime(monkeypatch)
        session = ParakeetOnnxDriver().open(_prepared(tmp_path), RuntimeOptions())

        result = session.transcribe(_request())

        assert result.text == "VoicePad works."
        assert len(result.segments) == 1
        assert (result.segments[0].start, result.segments[0].end) == (0.0, 1.0)
        assert (result.backend_id, result.model_id, result.artifact_format) == (
            "parakeet-onnx",
            "parakeet-tdt-0.6b-v3",
            "onnx",
        )
        recognizer = loader.recognizer
        assert recognizer.calls[0][1] == 16_000
        assert recognizer.calls[0][0].dtype == np.float32

    def test_transcribe_rejects_translation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_runtime(monkeypatch)
        session = ParakeetOnnxDriver().open(_prepared(tmp_path), RuntimeOptions())

        with pytest.raises(TranscriptionError, match="does not support speech translation"):
            session.transcribe(_request(intent="translate"))

    def test_transcribe_rejects_wrong_sample_rate(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_runtime(monkeypatch)
        session = ParakeetOnnxDriver().open(_prepared(tmp_path), RuntimeOptions())

        with pytest.raises(TranscriptionError, match="16000 Hz"):
            session.transcribe(_request(sample_rate=8_000))

    def test_close_releases_model(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_runtime(monkeypatch)
        session = ParakeetOnnxDriver().open(_prepared(tmp_path), RuntimeOptions())

        session.close()
        session.close()

        with pytest.raises(TranscriptionError, match="closed"):
            session.transcribe(_request())
