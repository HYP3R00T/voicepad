from __future__ import annotations

import importlib.util
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from .linux_cuda import configure_linux_cuda_libraries
from .windows_cuda import configure_windows_cuda_dlls
from ..contracts import (
    BackendCapabilities,
    BackendContract,
    OutputCapabilities,
    PreparedModel,
    RuntimeInfo,
    RuntimeOptions,
    TranscriptionRequest,
)
from ..errors import TranscriptionError
from ..types import Segment, TranscriptionResult
from ...audio.types import WaveformSpec
from ...models import ModelCompatibilityError, validate_model

logger = logging.getLogger(__name__)

BACKEND_ID = "sherpa-onnx"
SAMPLE_RATE = 16_000
HOTWORDS_SCORE = 4.0
MAX_ACTIVE_PATHS = 16
_CONTRACT = BackendContract(
    audio=WaveformSpec(sample_rate=SAMPLE_RATE, peak_normalize=True),
    decoding=BackendCapabilities(context_biasing=True),
    output=OutputCapabilities(),
)


class SherpaOnnxDriver:
    """Run a Sherpa-compatible Parakeet transducer on NVIDIA CUDA."""

    @property
    def id(self) -> str:
        return BACKEND_ID

    @property
    def contract(self) -> BackendContract:
        return _CONTRACT

    def is_available(self) -> bool:
        try:
            if importlib.util.find_spec("sherpa_onnx") is None:
                return False
            return _is_cuda_build(_load_runtime())
        except Exception:
            return False

    def open(self, model: PreparedModel, options: RuntimeOptions) -> SherpaOnnxSession:
        if model.spec.backend != self.id:
            raise TranscriptionError(
                f"Model '{model.spec.id}' targets backend '{model.spec.backend}', not '{self.id}'."
            )
        try:
            validate_model(model.artifact_path, model.spec)
        except ModelCompatibilityError as exc:
            raise TranscriptionError(f"Could not open Parakeet model '{model.spec.id}': {exc}") from exc

        _validate_options(options, model.spec.precision)
        try:
            configure_windows_cuda_dlls()
            configure_linux_cuda_libraries()
            sherpa = _load_runtime()
            if not _is_cuda_build(sherpa):
                raise TranscriptionError("Parakeet requires the CUDA 12/cuDNN 9 build of Sherpa-ONNX.")

            recognizer = sherpa.OfflineRecognizer.from_transducer(
                encoder=str(model.artifact_path / "encoder.int8.onnx"),
                decoder=str(model.artifact_path / "decoder.int8.onnx"),
                joiner=str(model.artifact_path / "joiner.int8.onnx"),
                tokens=str(model.artifact_path / "tokens.txt"),
                decoding_method="modified_beam_search",
                max_active_paths=MAX_ACTIVE_PATHS,
                hotwords_score=HOTWORDS_SCORE,
                provider="cuda",
                model_type="nemo_transducer",
            )
            tokenizer = _load_tokenizer(model.artifact_path / "tokenizer.json")
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"Could not open Parakeet with Sherpa-ONNX on CUDA: {exc}") from exc

        info = RuntimeInfo(
            backend_id=self.id,
            model_id=model.spec.id,
            device="cuda",
            precision=model.spec.precision or "float32",
        )
        logger.info(
            "Opened Sherpa-ONNX runtime: model=%s provider=%s precision=%s decoder=modified_beam_search",
            model.spec.id,
            "cuda",
            info.precision,
        )
        return SherpaOnnxSession(recognizer, tokenizer, info)


class SherpaOnnxSession:
    """Adapt Sherpa-ONNX recognition to VoicePad's result contract."""

    def __init__(self, recognizer: Any, tokenizer: Any, info: RuntimeInfo) -> None:
        self._recognizer: Any | None = recognizer
        self._tokenizer = tokenizer
        self._info = info

    @property
    def info(self) -> RuntimeInfo:
        return self._info

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        recognizer = self._recognizer
        if recognizer is None:
            raise TranscriptionError("Cannot transcribe with a closed Sherpa-ONNX session.")
        if request.sample_rate != SAMPLE_RATE:
            raise TranscriptionError(f"Parakeet requires {SAMPLE_RATE} Hz audio, got {request.sample_rate} Hz.")
        if request.intent != "transcribe":
            raise TranscriptionError("Parakeet TDT does not support speech translation.")

        started_at = time.perf_counter()
        try:
            hotwords = _encode_hotwords(self._tokenizer, request.context.proper_nouns)
            stream = recognizer.create_stream(hotwords=hotwords or None)
            stream.accept_waveform(
                request.sample_rate,
                np.ascontiguousarray(request.audio, dtype=np.float32),
            )
            recognizer.decode_stream(stream)
            text = _recognized_text(stream.result)
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"Sherpa-ONNX transcription failed: {exc}") from exc

        duration_s = request.audio.size / request.sample_rate
        segments = [Segment(start=0.0, end=duration_s, text=text)] if text else []
        return TranscriptionResult(
            text=text,
            segments=segments,
            language=None,
            language_probability=None,
            duration_s=duration_s,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            device=self._info.device,
            compute_type=self._info.precision,
            fallback_to_cpu=False,
            backend_id=self._info.backend_id,
            model_id=self._info.model_id,
            artifact_format="onnx",
        )

    def close(self) -> None:
        self._recognizer = None


def _load_runtime() -> Any:
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise TranscriptionError("Parakeet requires the CUDA 12/cuDNN 9 build of 'sherpa-onnx'.") from exc
    return sherpa_onnx


def _load_tokenizer(path: Path) -> Any:
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise TranscriptionError("Parakeet vocabulary hints require the 'tokenizers' package.") from exc
    try:
        return Tokenizer.from_file(str(path))
    except Exception as exc:
        raise TranscriptionError(f"Could not load Parakeet tokenizer '{path}': {exc}") from exc


def _encode_hotwords(tokenizer: Any, proper_nouns: tuple[str, ...]) -> str:
    phrases: list[str] = []
    for term in proper_nouns:
        encoding = tokenizer.encode(term, add_special_tokens=False)
        tokens = tuple(str(token) for token in encoding.tokens)
        if not tokens or "<unk>" in tokens:
            raise TranscriptionError(f"Parakeet tokenizer cannot encode vocabulary hint '{term}'.")
        phrases.append(" ".join(tokens))
    return "/".join(phrases)


def _is_cuda_build(sherpa: Any) -> bool:
    return "+cuda12.cudnn9" in str(getattr(sherpa, "__version__", ""))


def _validate_options(options: RuntimeOptions, precision: str | None) -> None:
    if options.device.lower() not in {"auto", "cuda"}:
        raise TranscriptionError("Sherpa-ONNX Parakeet is CUDA-only; use device 'auto' or 'cuda'.")

    supported = {"auto", precision or "float32"}
    if precision == "fp16":
        supported.add("float16")
    if options.precision.lower() not in supported:
        choices = "', '".join(sorted(supported))
        raise TranscriptionError(f"This Parakeet artifact supports '{choices}', not '{options.precision}'.")


def _recognized_text(result: Any) -> str:
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text.strip()
    raise TranscriptionError(f"Sherpa-ONNX returned an unexpected result: {type(result).__name__}.")


__all__ = ["SherpaOnnxDriver", "SherpaOnnxSession"]
