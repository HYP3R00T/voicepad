from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from .windows_cuda import configure_windows_cuda_dlls
from ..artifacts import prepare_artifact
from ..constants import COMPUTE_TYPE, CPU_COMPUTE_TYPE, CUDA_ERROR_KEYWORDS, DEVICE
from ..contracts import (
    BackendCapabilities,
    PreparedModel,
    RuntimeInfo,
    RuntimeOptions,
    TranscriptionRequest,
)
from ..errors import TranscriptionError
from ..types import Segment, TranscriptionResult, WordTimestamp
from ...config import get_config

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ...models import ModelSpec


class _RawWord(Protocol):
    word: str
    start: float
    end: float
    probability: float


class _RawSegment(Protocol):
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    words: Iterable[_RawWord] | None


class _RawTranscriptionInfo(Protocol):
    language: str
    language_probability: float


class _RawModel(Protocol):
    def transcribe(
        self,
        audio: object,
        **kwargs: object,
    ) -> tuple[Iterable[_RawSegment], _RawTranscriptionInfo]: ...


class FasterWhisperDriver:
    """Prepare and open CTranslate2 Whisper models through faster-whisper."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir

    @property
    def id(self) -> str:
        return "faster-whisper"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            streaming=False,
            word_timestamps=True,
            language_detection=True,
            translation=True,
            context_biasing=True,
        )

    def is_available(self) -> bool:
        try:
            return importlib.util.find_spec("faster_whisper") is not None
        except Exception:
            return False

    def prepare(self, model: ModelSpec) -> PreparedModel:
        if model.backend_id != self.id:
            raise TranscriptionError(f"Model '{model.id}' targets backend '{model.backend_id}', not '{self.id}'.")

        try:
            artifact_path = prepare_artifact(
                model,
                self._cache_dir or get_config().model_cache_path,
            )
        except Exception as exc:
            raise TranscriptionError(f"Could not prepare faster-whisper model '{model.id}': {exc}") from exc

        return PreparedModel(spec=model, artifact_path=artifact_path)

    def open(self, model: PreparedModel, options: RuntimeOptions) -> FasterWhisperSession:
        if model.spec.backend_id != self.id:
            raise TranscriptionError(
                f"Model '{model.spec.id}' targets backend '{model.spec.backend_id}', not '{self.id}'."
            )

        device, precision = _resolve_runtime(options)
        try:
            raw_model = _load_model(model.artifact_path, device, precision)
            info = RuntimeInfo(
                backend_id=self.id,
                model_id=model.spec.id,
                device=device,
                precision=precision,
            )
        except Exception as exc:
            if not options.allow_cpu_fallback or device != "cuda" or not _is_cuda_error(exc):
                raise TranscriptionError(f"Could not open faster-whisper model '{model.spec.id}': {exc}") from exc
            try:
                raw_model = _load_model(model.artifact_path, "cpu", CPU_COMPUTE_TYPE)
            except Exception as fallback_error:
                raise TranscriptionError(
                    f"Could not open faster-whisper model '{model.spec.id}' on CUDA ({exc}); "
                    f"CPU fallback failed: {fallback_error}"
                ) from fallback_error
            info = RuntimeInfo(
                backend_id=self.id,
                model_id=model.spec.id,
                device="cpu",
                precision=CPU_COMPUTE_TYPE,
                fallback_to_cpu=True,
            )

        return FasterWhisperSession(
            model,
            raw_model,
            info,
            allow_cpu_fallback=options.allow_cpu_fallback,
        )


class FasterWhisperSession:
    """Open faster-whisper model session with backend-neutral results."""

    def __init__(
        self,
        prepared: PreparedModel,
        model: _RawModel,
        info: RuntimeInfo,
        *,
        allow_cpu_fallback: bool,
    ) -> None:
        self._prepared = prepared
        self._model: _RawModel | None = model
        self._info = info
        self._allow_cpu_fallback = allow_cpu_fallback

    @property
    def info(self) -> RuntimeInfo:
        return self._info

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        model = self._model
        if model is None:
            raise TranscriptionError("Cannot transcribe with a closed faster-whisper session.")

        started_at = time.perf_counter()
        arguments = self._build_arguments(request)

        try:
            raw_segments, raw_info = model.transcribe(request.audio, **arguments)
            segments = _adapt_segments(
                raw_segments,
                duration_s=request.audio.size / request.sample_rate,
                include_words=request.word_timestamps,
                no_speech_threshold=request.no_speech_threshold,
            )
        except Exception as exc:
            if not self._can_fallback(exc):
                raise TranscriptionError(f"faster-whisper transcription failed: {exc}") from exc
            raw_info, segments = self._retry_on_cpu(request, arguments, exc)

        duration_s = request.audio.size / request.sample_rate
        latency_ms = (time.perf_counter() - started_at) * 1000
        text = " ".join(segment.text for segment in segments if segment.text).strip()
        avg_confidence = sum(segment.avg_logprob for segment in segments) / len(segments) if segments else 0.0

        return TranscriptionResult(
            text=text,
            segments=segments,
            language=raw_info.language,
            language_probability=raw_info.language_probability,
            duration_s=duration_s,
            latency_ms=latency_ms,
            device=self._info.device,
            compute_type=self._info.precision,
            fallback_to_cpu=self._info.fallback_to_cpu,
            avg_confidence=avg_confidence,
            low_confidence_count=sum(segment.avg_logprob < -1.0 for segment in segments),
            backend_id=self._info.backend_id,
            model_id=self._info.model_id,
            artifact_format=self._prepared.spec.artifact_format,
        )

    def close(self) -> None:
        self._model = None

    def _build_arguments(self, request: TranscriptionRequest) -> dict[str, object]:
        previous_text = None if self._prepared.spec.distil else request.context.previous_text
        hotwords = " ".join(request.context.proper_nouns) or None
        return {
            "language": request.language,
            "beam_size": request.beam_size,
            "task": request.intent,
            "vad_filter": request.vad_filter,
            "hallucination_silence_threshold": request.hallucination_silence_threshold,
            "no_speech_threshold": request.no_speech_threshold,
            "initial_prompt": previous_text,
            "hotwords": hotwords,
            "condition_on_previous_text": False,
            "word_timestamps": request.word_timestamps,
        }

    def _can_fallback(self, exc: Exception) -> bool:
        return (
            self._allow_cpu_fallback
            and self._info.device == "cuda"
            and any(keyword in str(exc).lower() for keyword in CUDA_ERROR_KEYWORDS)
        )

    def _retry_on_cpu(
        self,
        request: TranscriptionRequest,
        arguments: dict[str, object],
        original_error: Exception,
    ) -> tuple[_RawTranscriptionInfo, list[Segment]]:
        try:
            model = _load_model(
                self._prepared.artifact_path,
                "cpu",
                CPU_COMPUTE_TYPE,
            )
            self._model = model
            self._info = RuntimeInfo(
                backend_id=self.id,
                model_id=self._prepared.spec.id,
                device="cpu",
                precision=CPU_COMPUTE_TYPE,
                fallback_to_cpu=True,
            )
            raw_segments, raw_info = model.transcribe(request.audio, **arguments)
            segments = _adapt_segments(
                raw_segments,
                duration_s=request.audio.size / request.sample_rate,
                include_words=request.word_timestamps,
                no_speech_threshold=request.no_speech_threshold,
            )
        except Exception as retry_error:
            raise TranscriptionError(
                f"faster-whisper CUDA inference failed ({original_error}); CPU retry failed: {retry_error}"
            ) from retry_error

        return raw_info, segments

    @property
    def id(self) -> str:
        return "faster-whisper"


def _resolve_runtime(options: RuntimeOptions) -> tuple[str, str]:
    device = DEVICE if options.device == "auto" else options.device
    if options.precision != "auto":
        return device, options.precision
    return device, CPU_COMPUTE_TYPE if device == "cpu" else COMPUTE_TYPE


def _load_model(
    artifact_path: Path,
    device: str,
    precision: str,
) -> _RawModel:
    if device == "cuda":
        configure_windows_cuda_dlls()
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError("The Faster-Whisper backend requires the 'faster-whisper' package.") from exc
    return cast(
        _RawModel,
        WhisperModel(
            str(artifact_path),
            device=device,
            compute_type=precision,
        ),
    )


def _is_cuda_error(exc: Exception) -> bool:
    return any(keyword in str(exc).lower() for keyword in CUDA_ERROR_KEYWORDS)


def _adapt_segments(
    raw_segments: Iterable[_RawSegment],
    *,
    duration_s: float,
    include_words: bool,
    no_speech_threshold: float,
) -> list[Segment]:
    segments: list[Segment] = []
    for raw in raw_segments:
        if raw.start >= duration_s or raw.no_speech_prob > no_speech_threshold:
            continue

        words = (
            [
                WordTimestamp(
                    word=word.word,
                    start=word.start,
                    end=word.end,
                    probability=word.probability,
                )
                for word in raw.words or ()
            ]
            if include_words
            else []
        )
        segments.append(
            Segment(
                start=raw.start,
                end=min(raw.end, duration_s),
                text=raw.text.strip(),
                avg_logprob=raw.avg_logprob,
                no_speech_prob=raw.no_speech_prob,
                words=words,
            )
        )
    return segments
