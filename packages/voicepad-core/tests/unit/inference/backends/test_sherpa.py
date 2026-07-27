from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from voicepad_core.inference.backends.sherpa import SherpaOnnxDriver
from voicepad_core.inference.contracts import (
    PreparedModel,
    RuntimeInfo,
    RuntimeOptions,
    TranscriptionRequest,
)
from voicepad_core.inference.errors import TranscriptionError
from voicepad_core.models import Model

_FILES = (
    "decoder.int8.onnx",
    "encoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
)


class _Stream:
    def __init__(self, text: object) -> None:
        self.result = SimpleNamespace(text=text)
        self.calls: list[tuple[int, np.ndarray[Any, Any]]] = []

    def accept_waveform(self, sample_rate: int, audio: np.ndarray[Any, Any]) -> None:
        self.calls.append((sample_rate, audio))


class _Recognizer:
    def __init__(self, text: object = " VoicePad works. ") -> None:
        self.text = text
        self.streams: list[_Stream] = []
        self.decoded: list[_Stream] = []

    def create_stream(self) -> _Stream:
        stream = _Stream(self.text)
        self.streams.append(stream)
        return stream

    def decode_stream(self, stream: _Stream) -> None:
        self.decoded.append(stream)


class _OfflineRecognizer:
    def __init__(self, recognizer: _Recognizer) -> None:
        self.recognizer = recognizer
        self.calls: list[dict[str, object]] = []

    def from_transducer(self, **kwargs: object) -> _Recognizer:
        self.calls.append(kwargs)
        return self.recognizer


class _Sherpa:
    def __init__(self, recognizer: _Recognizer, version: str = "1.13.3+cuda12.cudnn9") -> None:
        self.__version__ = version
        self.OfflineRecognizer = _OfflineRecognizer(recognizer)


def _spec(*, backend_id: str = "sherpa-onnx") -> Model:
    return Model(
        id="parakeet-tdt-0.6b-v3",
        repo="csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
        backend=backend_id,
        files=_FILES,
        label="Parakeet",
        hint="Test model",
        precision="int8",
    )


def _prepared(path: Path) -> PreparedModel:
    for filename in _FILES:
        target = path / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"model")
    return PreparedModel(_spec(), path)


def _request(**kwargs: Any) -> TranscriptionRequest:
    sample_rate = kwargs.pop("sample_rate", 16_000)
    return TranscriptionRequest(np.ones(16_000, dtype=np.float32), sample_rate=sample_rate, **kwargs)


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    version: str = "1.13.3+cuda12.cudnn9",
    result: object = " VoicePad works. ",
) -> tuple[_Sherpa, _Recognizer]:
    recognizer = _Recognizer(result)
    sherpa = _Sherpa(recognizer, version)
    monkeypatch.setattr(
        "voicepad_core.inference.backends.sherpa._load_runtime",
        lambda: sherpa,
    )
    return sherpa, recognizer


class TestSherpaOnnxDriver:
    def test_contract_claims_only_implemented_features(self) -> None:
        """The driver advertises implemented decoding features and its waveform requirement."""
        contract = SherpaOnnxDriver().contract

        assert contract.decoding.context_biasing is False
        assert contract.decoding.word_timestamps is False
        assert contract.decoding.translation is False
        assert contract.audio.peak_normalize is True

    def test_open_configures_split_transducer_on_cuda(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A complete artifact opens with Sherpa's reliable CUDA greedy decoder."""
        sherpa, _ = _patch_runtime(monkeypatch)

        session = SherpaOnnxDriver().open(
            _prepared(tmp_path),
            RuntimeOptions(device="cuda", precision="auto"),
        )

        assert session.info == RuntimeInfo("sherpa-onnx", "parakeet-tdt-0.6b-v3", "cuda", "int8")
        options = sherpa.OfflineRecognizer.calls[0]
        assert options == {
            "encoder": str(tmp_path / "encoder.int8.onnx"),
            "decoder": str(tmp_path / "decoder.int8.onnx"),
            "joiner": str(tmp_path / "joiner.int8.onnx"),
            "tokens": str(tmp_path / "tokens.txt"),
            "decoding_method": "greedy_search",
            "provider": "cuda",
            "model_type": "nemo_transducer",
        }

    def test_open_rejects_cpu_only_sherpa_build(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The regular CPU-only Sherpa wheel cannot satisfy the CUDA-only backend."""
        _patch_runtime(monkeypatch, version="1.13.3")

        with pytest.raises(TranscriptionError, match="CUDA 12/cuDNN 9"):
            SherpaOnnxDriver().open(_prepared(tmp_path), RuntimeOptions())

    @pytest.mark.parametrize(
        ("options", "message"),
        (
            (RuntimeOptions(device="cpu"), "CUDA-only"),
            (RuntimeOptions(precision="float16"), "supports"),
        ),
    )
    def test_open_rejects_incompatible_runtime_options(
        self,
        tmp_path: Path,
        options: RuntimeOptions,
        message: str,
    ) -> None:
        """Device and precision requests must match the CUDA int8 artifact."""
        with pytest.raises(TranscriptionError, match=message):
            SherpaOnnxDriver().open(_prepared(tmp_path), options)

    def test_open_accepts_int8_precision(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """VoicePad's int8 setting matches the artifact precision."""
        _patch_runtime(monkeypatch)

        session = SherpaOnnxDriver().open(
            _prepared(tmp_path),
            RuntimeOptions(precision="int8"),
        )

        assert session.info.precision == "int8"


class TestSherpaOnnxSession:
    def test_transcribe_uses_unbiased_greedy_stream(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Parakeet decodes through the reliable stream without contextual biasing."""
        _, recognizer = _patch_runtime(monkeypatch)
        session = SherpaOnnxDriver().open(_prepared(tmp_path), RuntimeOptions())

        result = session.transcribe(_request())

        assert result.text == "VoicePad works."
        assert len(result.segments) == 1
        assert (result.segments[0].start, result.segments[0].end) == (0.0, 1.0)
        assert (result.backend_id, result.model_id, result.artifact_format) == (
            "sherpa-onnx",
            "parakeet-tdt-0.6b-v3",
            "onnx",
        )
        stream = recognizer.streams[0]
        assert stream.calls[0][0] == 16_000
        assert stream.calls[0][1].dtype == np.float32
        assert recognizer.decoded == [stream]

    def test_transcribe_rejects_translation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Parakeet cannot execute a translation intent."""
        _patch_runtime(monkeypatch)
        session = SherpaOnnxDriver().open(_prepared(tmp_path), RuntimeOptions())

        with pytest.raises(TranscriptionError, match="does not support speech translation"):
            session.transcribe(_request(intent="translate"))

    def test_transcribe_rejects_wrong_sample_rate(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The session refuses audio that bypassed the 16 kHz preprocessing contract."""
        _patch_runtime(monkeypatch)
        session = SherpaOnnxDriver().open(_prepared(tmp_path), RuntimeOptions())

        with pytest.raises(TranscriptionError, match="16000 Hz"):
            session.transcribe(_request(sample_rate=8_000))

    def test_close_releases_recognizer(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Closing repeatedly releases the recognizer and prevents later inference."""
        _patch_runtime(monkeypatch)
        session = SherpaOnnxDriver().open(_prepared(tmp_path), RuntimeOptions())

        session.close()
        session.close()

        with pytest.raises(TranscriptionError, match="closed"):
            session.transcribe(_request())
