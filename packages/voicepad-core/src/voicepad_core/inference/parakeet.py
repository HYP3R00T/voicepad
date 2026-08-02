from __future__ import annotations

import gc
import math
from collections.abc import Iterable
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForTDT, AutoProcessor, StoppingCriteria, StoppingCriteriaList

from voicepad_core.deployments import (
    ArtifactManifest,
    DeclaredCapabilities,
    DeploymentDefinition,
    HuggingFaceSource,
)
from voicepad_core.preprocessing import PreprocessedAudio

from .cuda import CudaDevice
from .types import (
    ActiveDeployment,
    BackendResult,
    CancellationToken,
    InferenceError,
    InvalidTranscriptionInputError,
    SessionClosedError,
    TimedWord,
    TokenTimestamp,
    TranscriptionIntent,
    UnsupportedIntentError,
)

MAX_GENERATED_TOKENS = 4096


class _CancellationCriteria(StoppingCriteria):
    def __init__(self, token: CancellationToken) -> None:
        self._token = token

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs: object) -> torch.Tensor:
        del scores, kwargs
        return torch.full(
            (input_ids.shape[0],),
            self._token.is_cancelled,
            dtype=torch.bool,
            device=input_ids.device,
        )


class TransformersParakeetTDTSession:
    """Resident official Parakeet TDT session for one admitted CUDA device."""

    def __init__(
        self,
        definition: DeploymentDefinition,
        manifest: ArtifactManifest,
        snapshot: Path,
        device: CudaDevice,
    ) -> None:
        self._definition = definition
        self._manifest = manifest
        self._snapshot = snapshot.resolve()
        self._device = device
        self._lock = Lock()
        self._processor: Any | None = None
        self._model: Any | None = None
        if not isinstance(manifest.source, HuggingFaceSource):
            raise InferenceError("Parakeet requires a pinned Hugging Face artifact source.")
        self._active = ActiveDeployment(
            definition=definition,
            snapshot_revision=manifest.source.revision,
            device_id=device.stable_id,
            device_name=device.name,
            total_gpu_memory_bytes=device.total_memory_bytes,
        )
        self._load()

    @property
    def deployment(self) -> ActiveDeployment:
        return self._active

    @property
    def capabilities(self) -> DeclaredCapabilities:
        return self._definition.capabilities

    def warm(self) -> None:
        seconds = self._definition.processing.warmup_seconds
        samples = np.zeros(seconds * self.capabilities.native_sample_rate, dtype=np.float32)
        audio = PreprocessedAudio(samples, self.capabilities.native_sample_rate, channels=1)
        self.transcribe(audio, TranscriptionIntent(), CancellationToken())

    def transcribe(
        self,
        audio: PreprocessedAudio,
        intent: TranscriptionIntent,
        cancellation: CancellationToken,
    ) -> BackendResult:
        self._validate_request(audio, intent)
        if cancellation.is_cancelled:
            return BackendResult("", (), (), cancelled=True)

        with self._lock, torch.inference_mode():
            processor, model = self._require_open()
            inputs = processor(
                np.ascontiguousarray(audio.samples),
                sampling_rate=audio.sample_rate,
                return_tensors="pt",
            ).to(device=self._device.torch_device, dtype=torch.float16)
            output = model.generate(
                **inputs,
                return_dict_in_generate=True,
                max_new_tokens=MAX_GENERATED_TOKENS,
                stopping_criteria=StoppingCriteriaList([_CancellationCriteria(cancellation)]),
            )
            torch.cuda.synchronize(self._device.index)
            if output.sequences.shape[-1] >= MAX_GENERATED_TOKENS:
                raise InferenceError("Parakeet reached the native generation limit before a terminal token.")
            decoded = processor.decode(
                output.sequences,
                durations=output.durations,
                skip_special_tokens=True,
            )

        text, records = _split_decoded(decoded)
        tokens = _timestamp_tokens(records)
        return BackendResult(
            text=text,
            tokens=tokens,
            words=_tokens_to_words(tokens),
            cancelled=cancellation.is_cancelled,
        )

    def close(self) -> None:
        with self._lock:
            self._model = None
            self._processor = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self._device.index)

    def _load(self) -> None:
        if not self._snapshot.is_dir():
            raise InferenceError(f"Verified model snapshot does not exist: {self._manifest.id}")
        try:
            processor = AutoProcessor.from_pretrained(
                self._snapshot,
                local_files_only=True,
            )
            model = AutoModelForTDT.from_pretrained(
                self._snapshot,
                dtype=torch.float16,
                device_map=self._device.torch_device,
                low_cpu_mem_usage=True,
                local_files_only=True,
            ).eval()
            _validate_model_placement(model, self._device.index)
        except Exception as error:
            self._model = None
            self._processor = None
            gc.collect()
            torch.cuda.empty_cache()
            raise InferenceError(f"Could not load deployment '{self._definition.id}' on CUDA: {error}") from error
        self._processor = processor
        self._model = model

    def _require_open(self) -> tuple[Any, Any]:
        if self._processor is None or self._model is None:
            raise SessionClosedError("The Parakeet session is closed.")
        return self._processor, self._model

    def _validate_request(self, audio: PreprocessedAudio, intent: TranscriptionIntent) -> None:
        if intent.language is not None:
            raise UnsupportedIntentError("This deployment does not accept a language hint.")
        if intent.vocabulary:
            raise UnsupportedIntentError("This deployment does not expose native vocabulary biasing.")
        if audio.sample_rate != self.capabilities.native_sample_rate or audio.channels != 1:
            raise InvalidTranscriptionInputError(
                f"Parakeet requires mono {self.capabilities.native_sample_rate} Hz canonical audio."
            )
        if audio.samples.ndim != 1 or audio.samples.dtype != np.float32 or not audio.samples.flags.c_contiguous:
            raise InvalidTranscriptionInputError("Parakeet requires a contiguous one-dimensional float32 waveform.")
        if audio.samples.size == 0:
            raise InvalidTranscriptionInputError("Parakeet cannot transcribe an empty waveform.")
        duration = audio.duration()
        if duration > self._definition.processing.maximum_input_seconds:
            raise InvalidTranscriptionInputError(
                f"Audio exceeds the deployment limit: duration={duration:.3f}s "
                f"maximum={self._definition.processing.maximum_input_seconds}s."
            )


def _validate_model_placement(model: Any, device_index: int) -> None:
    for parameter in model.parameters():
        if parameter.device.type != "cuda" or parameter.device.index != device_index:
            raise InferenceError("Model parameters were not placed entirely on the selected CUDA device.")
        if parameter.is_floating_point() and parameter.dtype != torch.float16:
            raise InferenceError(f"Model parameter has unexpected precision: {parameter.dtype}.")


def _split_decoded(decoded: object) -> tuple[str, object]:
    if not isinstance(decoded, tuple) or len(decoded) != 2:
        raise InferenceError("Parakeet processor did not return text and token timestamps.")
    text, records = decoded
    if isinstance(text, list):
        if len(text) != 1:
            raise InferenceError("Parakeet returned an unexpected transcription batch.")
        text = text[0]
    if not isinstance(text, str):
        raise InferenceError("Parakeet returned non-text output.")
    return text, records


def _flatten_records(value: object) -> Iterable[dict[str, object]]:
    if isinstance(value, dict):
        yield value
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _flatten_records(nested)


def _timestamp_tokens(records: object) -> tuple[TokenTimestamp, ...]:
    tokens: list[TokenTimestamp] = []
    for record in _flatten_records(records):
        text = record.get("token")
        start = record.get("start")
        end = record.get("end")
        if not isinstance(text, str) or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise InferenceError("Parakeet returned an invalid token timestamp record.")
        start_seconds = float(start)
        end_seconds = float(end)
        if (
            not math.isfinite(start_seconds)
            or not math.isfinite(end_seconds)
            or start_seconds < 0
            or end_seconds < start_seconds
        ):
            raise InferenceError("Parakeet returned a non-monotonic token duration.")
        tokens.append(TokenTimestamp(text, start_seconds, end_seconds))
    return tuple(tokens)


def _tokens_to_words(tokens: tuple[TokenTimestamp, ...]) -> tuple[TimedWord, ...]:
    words: list[TimedWord] = []
    pieces: list[str] = []
    start = 0.0
    end = 0.0
    for token in tokens:
        begins_word = token.text.startswith(("▁", " "))
        if begins_word and pieces:
            words.append(TimedWord("".join(pieces), start, end))
            pieces = []
        if not pieces:
            start = token.start_seconds
        pieces.append(token.text.lstrip("▁ ") if begins_word else token.text)
        end = token.end_seconds
    if pieces:
        words.append(TimedWord("".join(pieces), start, end))
    return tuple(words)
