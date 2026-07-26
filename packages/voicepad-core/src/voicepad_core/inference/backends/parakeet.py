from __future__ import annotations

import importlib.util
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .windows_cuda import configure_windows_cuda_dlls
from ..artifacts import prepare_artifact
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
from ...config import get_config
from ...models import ModelCompatibilityError, validate_model_artifact

if TYPE_CHECKING:
    from ...models import ModelSpec

logger = logging.getLogger(__name__)

BACKEND_ID = "parakeet-onnx"
CUDA_PROVIDER = "CUDAExecutionProvider"
MODEL_TYPE = "nemo-parakeet-tdt-0.6b-v3"
SAMPLE_RATE = 16_000
_CONTRACT = BackendContract(
    audio=WaveformSpec(sample_rate=SAMPLE_RATE, peak_normalize=True),
    decoding=BackendCapabilities(),
    output=OutputCapabilities(),
)


class ParakeetOnnxDriver:
    """Run a Parakeet ONNX export exclusively through NVIDIA CUDA."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir

    @property
    def id(self) -> str:
        return BACKEND_ID

    @property
    def capabilities(self) -> BackendCapabilities:
        return self.contract.decoding

    @property
    def contract(self) -> BackendContract:
        return _CONTRACT

    def is_available(self) -> bool:
        try:
            if importlib.util.find_spec("onnx_asr") is None or importlib.util.find_spec("onnxruntime") is None:
                return False
            runtime, _ = _load_runtime()
            return CUDA_PROVIDER in runtime.get_available_providers()
        except Exception:
            return False

    def prepare(self, model: ModelSpec) -> PreparedModel:
        if model.backend_id != self.id:
            raise TranscriptionError(f"Model '{model.id}' targets backend '{model.backend_id}', not '{self.id}'.")
        try:
            artifact_path = prepare_artifact(model, self._cache_dir or get_config().model_cache_path)
        except Exception as exc:
            raise TranscriptionError(f"Could not prepare Parakeet model '{model.id}': {exc}") from exc
        return PreparedModel(spec=model, artifact_path=artifact_path)

    def open(self, model: PreparedModel, options: RuntimeOptions) -> ParakeetOnnxSession:
        if model.spec.backend_id != self.id:
            raise TranscriptionError(
                f"Model '{model.spec.id}' targets backend '{model.spec.backend_id}', not '{self.id}'."
            )
        try:
            validate_model_artifact(model.artifact_path, model.spec)
        except ModelCompatibilityError as exc:
            raise TranscriptionError(f"Could not open Parakeet model '{model.spec.id}': {exc}") from exc

        _validate_options(options, model.spec.quantization)
        try:
            configure_windows_cuda_dlls()
            runtime, load_model = _load_runtime()
            preload_dlls = getattr(runtime, "preload_dlls", None)
            if callable(preload_dlls):
                preload_dlls(directory="")
            providers = runtime.get_available_providers()
            if CUDA_PROVIDER not in providers:
                available = ", ".join(providers) or "none"
                raise TranscriptionError(
                    f"Parakeet requires ONNX Runtime's CUDAExecutionProvider; available providers: {available}."
                )

            session_options = runtime.SessionOptions()
            session_options.log_severity_level = 3
            runtime_model = load_model(
                MODEL_TYPE,
                model.artifact_path,
                quantization=model.spec.quantization,
                sess_options=session_options,
                providers=[CUDA_PROVIDER],
            )
            _require_cuda_sessions(runtime_model)
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"Could not open Parakeet ONNX model '{model.spec.id}' on CUDA: {exc}") from exc

        info = RuntimeInfo(
            backend_id=self.id,
            model_id=model.spec.id,
            device="cuda",
            precision=model.spec.quantization or "float32",
        )
        logger.info(
            "Opened Parakeet ONNX runtime: model=%s provider=%s precision=%s",
            model.spec.id,
            CUDA_PROVIDER,
            info.precision,
        )
        return ParakeetOnnxSession(model, runtime_model, info)


class ParakeetOnnxSession:
    """Adapt ONNX Parakeet recognition to VoicePad's result contract."""

    def __init__(self, prepared: PreparedModel, model: Any, info: RuntimeInfo) -> None:
        self._prepared = prepared
        self._model: Any | None = model
        self._info = info

    @property
    def info(self) -> RuntimeInfo:
        return self._info

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        model = self._model
        if model is None:
            raise TranscriptionError("Cannot transcribe with a closed Parakeet session.")
        if request.sample_rate != SAMPLE_RATE:
            raise TranscriptionError(f"Parakeet requires {SAMPLE_RATE} Hz audio, got {request.sample_rate} Hz.")
        if request.intent != "transcribe":
            raise TranscriptionError("Parakeet TDT does not support speech translation.")

        started_at = time.perf_counter()
        try:
            recognized = model.recognize(
                np.ascontiguousarray(request.audio, dtype=np.float32),
                sample_rate=request.sample_rate,
            )
            text = _recognized_text(recognized)
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"Parakeet ONNX transcription failed: {exc}") from exc

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
            artifact_format=self._prepared.spec.artifact_format,
        )

    def close(self) -> None:
        self._model = None


def _load_runtime() -> tuple[Any, Any]:
    try:
        import onnxruntime
        from onnx_asr import load_model
    except ImportError as exc:
        raise TranscriptionError(
            "Parakeet requires the 'onnx-asr' and CUDA-enabled 'onnxruntime-gpu' packages."
        ) from exc
    return onnxruntime, load_model


def _validate_options(options: RuntimeOptions, quantization: str | None) -> None:
    if options.device.lower() not in {"auto", "cuda"}:
        raise TranscriptionError("Parakeet ONNX is CUDA-only; set the transcription device to 'auto' or 'cuda'.")

    supported = {"auto", quantization or "float32"}
    if quantization == "fp16":
        supported.add("float16")
    if options.precision.lower() not in supported:
        choices = "', '".join(sorted(supported))
        raise TranscriptionError(f"This Parakeet ONNX artifact supports '{choices}', not '{options.precision}'.")


def _require_cuda_sessions(model: Any) -> None:
    """Reject ONNX Runtime's silent whole-session CPU fallback."""
    asr = getattr(model, "asr", None)
    sessions = (
        getattr(asr, "_encoder", None),
        getattr(asr, "_decoder_joint", None),
    )
    if any(not hasattr(session, "get_providers") for session in sessions):
        raise TranscriptionError("Could not verify Parakeet ONNX CUDA sessions.")
    for session in sessions:
        providers = cast(Any, session).get_providers()
        if not providers or providers[0] != CUDA_PROVIDER:
            raise TranscriptionError("Parakeet ONNX could not activate CUDA; CPU inference fallback is disabled.")


def _recognized_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    raise TranscriptionError(f"Parakeet ONNX returned an unexpected value: {type(value).__name__}.")


__all__ = ["ParakeetOnnxDriver", "ParakeetOnnxSession"]
