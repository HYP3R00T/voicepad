from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from voicepad_core.inference.backends.parakeet import (
    REQUIRED_ARTIFACTS,
    ParakeetOnnxDriver,
    ParakeetOnnxSession,
    _decode_text,
    _load_vocab,
)
from voicepad_core.inference.contracts import (
    PreparedModel,
    RuntimeInfo,
    RuntimeOptions,
    TranscriptionContext,
    TranscriptionRequest,
)
from voicepad_core.inference.errors import TranscriptionError
from voicepad_core.models import HuggingFaceArtifact, LocalArtifact, ModelSpec


@dataclass
class _Node:
    name: str
    shape: list[int | str | None]


class _Session:
    def __init__(self, kind: str, providers: list[str] | None = None) -> None:
        self.kind = kind
        self.providers = providers or ["CPUExecutionProvider"]
        self.decoder_tokens = iter((2, 2, 2, 0, 2, 1, 2, 2))
        self.received_audio_samples = 0

    def get_providers(self) -> list[str]:
        return self.providers

    def get_inputs(self) -> list[_Node]:
        if self.kind != "decoder":
            return []
        return [
            _Node("input_states_1", [2, "batch", 4]),
            _Node("input_states_2", [2, "batch", 4]),
        ]

    def run(
        self,
        output_names: list[str],
        input_feed: dict[str, np.ndarray[Any, Any]],
    ) -> list[np.ndarray[Any, Any]]:
        if self.kind == "preprocessor":
            self.received_audio_samples = input_feed["waveforms"].shape[1]
            return [
                np.ones((1, 128, 6), dtype=np.float32),
                np.asarray([6], dtype=np.int64),
            ]
        if self.kind == "encoder":
            return [
                np.ones((1, 4, 6), dtype=np.float32),
                np.asarray([6], dtype=np.int64),
            ]

        token = next(self.decoder_tokens)
        logits = np.full((1, 1, 3), -10.0, dtype=np.float32)
        logits[..., token] = 10.0
        state = np.ones((2, 1, 4), dtype=np.float32)
        return [logits, state, state]


class _BiasDecoder:
    def __init__(self) -> None:
        self._scores = iter((
            (-10.0, -10.0, -10.0, 10.0),
            (-10.0, -10.0, -10.0, 10.0),
            (-10.0, -10.0, -10.0, 10.0),
            (1.0, -10.0, 1.5, -10.0),
            (-10.0, 1.0, 1.5, -10.0),
            (-10.0, -10.0, -10.0, 10.0),
            (-10.0, -10.0, -10.0, 10.0),
            (-10.0, -10.0, -10.0, 10.0),
        ))

    def get_inputs(self) -> list[_Node]:
        return [
            _Node("input_states_1", [2, "batch", 4]),
            _Node("input_states_2", [2, "batch", 4]),
        ]

    def run(
        self,
        output_names: list[str],
        input_feed: dict[str, np.ndarray[Any, Any]],
    ) -> list[np.ndarray[Any, Any]]:
        del output_names, input_feed
        logits = np.asarray(next(self._scores), dtype=np.float32).reshape(1, 1, 4)
        state = np.ones((2, 1, 4), dtype=np.float32)
        return [logits, state, state]


class _SessionOptions:
    graph_optimization_level: object | None = None
    log_severity_level: int | None = None


class _GraphOptimizationLevel:
    ORT_ENABLE_ALL = object()


class _Ort:
    SessionOptions = _SessionOptions
    GraphOptimizationLevel = _GraphOptimizationLevel

    def __init__(self, providers: list[str]) -> None:
        self._providers = providers
        self.calls: list[tuple[Path, list[str]]] = []
        self.session_options: list[_SessionOptions] = []
        self.sessions: dict[str, _Session] = {}
        self.preloaded = False

    def get_available_providers(self) -> list[str]:
        return self._providers

    def preload_dlls(self, *, directory: str) -> None:
        assert directory == ""
        self.preloaded = True

    def __getattr__(self, name: str) -> object:
        if name == "InferenceSession":
            return self._create_session
        raise AttributeError(name)

    def _create_session(
        self,
        path: str,
        *,
        sess_options: _SessionOptions,
        providers: list[str],
    ) -> _Session:
        artifact = Path(path)
        if artifact.name.startswith("encoder"):
            kind = "encoder"
        elif artifact.name.startswith("decoder"):
            kind = "decoder"
        else:
            kind = "preprocessor"
        session = _Session(kind, providers)
        self.calls.append((artifact, providers))
        self.session_options.append(sess_options)
        self.sessions[kind] = session
        return session


def _spec(
    *,
    backend_id: str = "parakeet-onnx",
    artifact_source: HuggingFaceArtifact | LocalArtifact | None = None,
) -> ModelSpec:
    return ModelSpec(
        id="parakeet-v3-int8",
        family="parakeet",
        backend_id=backend_id,
        artifact_format="onnx",
        quantization="int8",
        artifact_source=artifact_source or HuggingFaceArtifact("owner/parakeet"),
        required_files=REQUIRED_ARTIFACTS,
    )


def _model_dir(tmp_path: Path) -> Path:
    for artifact in REQUIRED_ARTIFACTS[:-1]:
        (tmp_path / artifact).write_bytes(b"model")
    (tmp_path / "vocab.txt").write_text("▁hello 0\n▁world. 1\n<blk> 2\n", encoding="utf-8")
    return tmp_path


def _prepared(tmp_path: Path) -> PreparedModel:
    return PreparedModel(_spec(artifact_source=LocalArtifact(tmp_path)), _model_dir(tmp_path))


def _request(**kwargs: Any) -> TranscriptionRequest:
    sample_rate = kwargs.pop("sample_rate", 16_000)
    return TranscriptionRequest(np.ones(16_000, dtype=np.float32), sample_rate=sample_rate, **kwargs)


class TestParakeetOnnxDriver:
    def test_capabilities_are_honest(self) -> None:
        """Capabilities include decoder bias but exclude unsupported streaming, detection, and translation."""
        driver = ParakeetOnnxDriver()
        capabilities = driver.capabilities

        assert (
            capabilities.streaming,
            capabilities.word_timestamps,
            capabilities.language_detection,
            capabilities.translation,
            capabilities.context_biasing,
        ) == (False, True, False, False, True)
        assert driver.contract.audio.peak_normalize is True

    def test_prepare_accepts_handy_artifact_layout(self, tmp_path: Path) -> None:
        """A local directory containing all four Handy artifacts is accepted unchanged."""
        model_dir = _model_dir(tmp_path)

        prepared = ParakeetOnnxDriver(cache_dir=tmp_path / "cache").prepare(
            _spec(artifact_source=LocalArtifact(model_dir))
        )

        assert prepared.artifact_path == model_dir

    def test_prepare_delegates_artifact_acquisition(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The driver delegates acquisition and validation to the shared artifact layer."""
        model_dir = _model_dir(tmp_path)
        calls: list[tuple[ModelSpec, Path]] = []

        def prepare(model: ModelSpec, cache_dir: Path) -> Path:
            calls.append((model, cache_dir))
            return model_dir

        monkeypatch.setattr("voicepad_core.inference.backends.parakeet.prepare_artifact", prepare)

        cache_dir = tmp_path / "cache"
        spec = _spec()
        prepared = ParakeetOnnxDriver(cache_dir=cache_dir).prepare(spec)

        assert (prepared.artifact_path, calls) == (model_dir, [(spec, cache_dir)])

    def test_prepare_rejects_missing_artifacts(self, tmp_path: Path) -> None:
        """An incomplete directory fails before ONNX Runtime is opened."""
        with pytest.raises(TranscriptionError, match="missing required files"):
            ParakeetOnnxDriver(cache_dir=tmp_path / "cache").prepare(_spec(artifact_source=LocalArtifact(tmp_path)))

    def test_prepare_rejects_backend_mismatch(self) -> None:
        """A model assigned to another backend cannot be prepared here."""
        with pytest.raises(TranscriptionError, match="not 'parakeet-onnx'"):
            ParakeetOnnxDriver().prepare(_spec(backend_id="other"))

    def test_open_uses_cuda_for_all_three_sessions(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """CUDA-capable ONNX Runtime places every Parakeet graph on CUDA with CPU fallback."""
        ort = _Ort(["CUDAExecutionProvider", "CPUExecutionProvider"])
        monkeypatch.setattr("voicepad_core.inference.backends.parakeet._load_onnxruntime", lambda: ort)

        session = ParakeetOnnxDriver().open(_prepared(tmp_path), RuntimeOptions(device="cuda", precision="int8"))

        assert (
            session.info.device,
            session.info.precision,
            ort.preloaded,
            [providers for _, providers in ort.calls],
            {options.log_severity_level for options in ort.session_options},
        ) == (
            "cuda",
            "int8",
            True,
            [["CUDAExecutionProvider", "CPUExecutionProvider"]] * 3,
            {3},
        )

    def test_open_rejects_silent_cuda_provider_fallback(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A strict CUDA request fails when ORT advertises CUDA but activates only CPU."""
        ort = _Ort(["CUDAExecutionProvider", "CPUExecutionProvider"])

        def create_cpu_session(
            path: str,
            *,
            sess_options: _SessionOptions,
            providers: list[str],
        ) -> _Session:
            del path, sess_options, providers
            return _Session("encoder", ["CPUExecutionProvider"])

        monkeypatch.setattr(ort, "_create_session", create_cpu_session)
        monkeypatch.setattr("voicepad_core.inference.backends.parakeet._load_onnxruntime", lambda: ort)

        with pytest.raises(TranscriptionError, match="failed to activate"):
            ParakeetOnnxDriver().open(
                _prepared(tmp_path),
                RuntimeOptions(device="cuda", allow_cpu_fallback=False),
            )

    def test_open_reports_cpu_fallback(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A CUDA request falls back explicitly when only the CPU provider is installed."""
        ort = _Ort(["CPUExecutionProvider"])
        monkeypatch.setattr("voicepad_core.inference.backends.parakeet._load_onnxruntime", lambda: ort)

        session = ParakeetOnnxDriver().open(_prepared(tmp_path), RuntimeOptions(device="cuda"))

        assert (session.info.device, session.info.fallback_to_cpu) == ("cpu", True)

    def test_open_rejects_disallowed_cpu_fallback(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A strict CUDA request fails when the CUDA execution provider is absent."""
        ort = _Ort(["CPUExecutionProvider"])
        monkeypatch.setattr("voicepad_core.inference.backends.parakeet._load_onnxruntime", lambda: ort)

        with pytest.raises(TranscriptionError, match="CPU fallback is disabled"):
            ParakeetOnnxDriver().open(
                _prepared(tmp_path),
                RuntimeOptions(device="cuda", allow_cpu_fallback=False),
            )

    def test_open_rejects_non_int8_precision(self, tmp_path: Path) -> None:
        """The int8 artifact set cannot be opened under a false float16 precision label."""
        with pytest.raises(TranscriptionError, match="int8"):
            ParakeetOnnxDriver().open(_prepared(tmp_path), RuntimeOptions(precision="float16"))


class TestParakeetOnnxSession:
    def test_transcribe_runs_tdt_decode_and_adapts_result(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Three ONNX graphs produce timed segments, words, and backend provenance."""
        ort = _Ort(["CUDAExecutionProvider", "CPUExecutionProvider"])
        monkeypatch.setattr("voicepad_core.inference.backends.parakeet._load_onnxruntime", lambda: ort)
        session = ParakeetOnnxDriver().open(_prepared(tmp_path), RuntimeOptions(device="cuda"))

        result = session.transcribe(
            _request(
                language="en",
                word_timestamps=True,
                context=TranscriptionContext(proper_nouns=("VoicePad",)),
            )
        )

        assert (
            result.text,
            result.segments[0].text,
            [word.word for word in result.segments[0].words],
            result.language_probability,
            result.backend_id,
            result.model_id,
            result.artifact_format,
            ort.sessions["preprocessor"].received_audio_samples,
        ) == (
            "hello world.",
            "hello world.",
            ["hello", "world."],
            None,
            "parakeet-onnx",
            "parakeet-v3-int8",
            "onnx",
            20_000,
        )

    def test_transcribe_biases_decoding_toward_tokenized_proper_noun(self, tmp_path: Path) -> None:
        """A tokenizable proper noun can win close decoder logits without rewriting output text."""
        session = ParakeetOnnxSession(
            _prepared(tmp_path),
            cast(Any, (_Session("encoder"), _BiasDecoder(), _Session("preprocessor"))),
            [" Voice", "Pad", " other", "<blk>"],
            3,
            RuntimeInfo(
                backend_id="parakeet-onnx",
                model_id="parakeet-v3-int8",
                device="cpu",
                precision="int8",
            ),
        )

        result = session.transcribe(_request(context=TranscriptionContext(proper_nouns=("VoicePad",))))

        assert result.text == "VoicePad"

    def test_transcribe_discards_completion_beyond_audio_boundary(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Late whole-word tokens cannot turn a clipped utterance into invented prose."""
        vocab = [" This", " is", " just", " a", " little", " bit", " more.", "<blk>"]
        session = ParakeetOnnxSession(
            _prepared(tmp_path),
            cast(Any, (_Session("encoder"), _Session("decoder"), _Session("preprocessor"))),
            vocab,
            7,
            RuntimeInfo("parakeet-onnx", "parakeet-v3-int8", "cpu", "int8"),
        )
        monkeypatch.setattr(
            session,
            "_infer",
            lambda *_args: (list(range(7)), [10, 13, 16, 21, 22, 22, 22]),
        )
        request = TranscriptionRequest(
            np.ones(round(1.39 * 16_000), dtype=np.float32),
            sample_rate=16_000,
            word_timestamps=True,
        )

        result = session.transcribe(request)

        assert result.text == "This is just a"
        assert [word.word for word in result.segments[0].words] == ["This", "is", "just", "a"]

    def test_transcribe_rejects_translation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The session rejects translation instead of silently treating it as transcription."""
        ort = _Ort(["CPUExecutionProvider"])
        monkeypatch.setattr("voicepad_core.inference.backends.parakeet._load_onnxruntime", lambda: ort)
        session = ParakeetOnnxDriver().open(_prepared(tmp_path), RuntimeOptions(device="cpu"))

        with pytest.raises(TranscriptionError, match="does not support speech translation"):
            session.transcribe(_request(intent="translate"))

    def test_transcribe_rejects_wrong_sample_rate(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The backend enforces the canonical 16 kHz boundary required by its ONNX graphs."""
        ort = _Ort(["CPUExecutionProvider"])
        monkeypatch.setattr("voicepad_core.inference.backends.parakeet._load_onnxruntime", lambda: ort)
        session = ParakeetOnnxDriver().open(_prepared(tmp_path), RuntimeOptions(device="cpu"))

        with pytest.raises(TranscriptionError, match="16000 Hz"):
            session.transcribe(_request(sample_rate=8_000))

    def test_close_is_idempotent_and_prevents_reuse(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Closing releases graph references and subsequent inference fails clearly."""
        ort = _Ort(["CPUExecutionProvider"])
        monkeypatch.setattr("voicepad_core.inference.backends.parakeet._load_onnxruntime", lambda: ort)
        session = ParakeetOnnxDriver().open(_prepared(tmp_path), RuntimeOptions(device="cpu"))

        session.close()
        session.close()

        with pytest.raises(TranscriptionError, match="closed"):
            session.transcribe(_request())


def test_load_vocab_requires_blank_token(tmp_path: Path) -> None:
    """A vocabulary without the TDT blank token is rejected during model opening."""
    path = tmp_path / "vocab.txt"
    path.write_text("▁hello 0\n", encoding="utf-8")

    with pytest.raises(TranscriptionError, match="<blk>"):
        _load_vocab(path)


def test_load_vocab_supports_one_token_per_line(tmp_path: Path) -> None:
    """NeMo vocabularies without explicit ids use stable zero-based line numbers."""
    path = tmp_path / "vocab.txt"
    path.write_text("▁Voice\nPad\n<blk>\n", encoding="utf-8")

    assert _load_vocab(path) == ([" Voice", "Pad", "<blk>"], 2)


def test_decode_text_matches_transcribe_rs_whitespace_cleanup() -> None:
    """Leading and token-boundary spaces follow transcribe-rs decoding semantics."""
    assert _decode_text([" ", "Voice", " ", "Pad"]) == "Voice Pad"
