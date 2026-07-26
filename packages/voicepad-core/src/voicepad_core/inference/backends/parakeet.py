from __future__ import annotations

import importlib.util
import logging
import re
import time
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

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
from ..types import Segment, TranscriptionResult, WordTimestamp
from ...audio.types import WaveformSpec
from ...config import get_config
from ...models import ModelCompatibilityError, validate_model_artifact

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...models import ModelSpec

logger = logging.getLogger(__name__)

BACKEND_ID = "parakeet-onnx"
SAMPLE_RATE = 16_000
LEADING_SILENCE_S = 0.25
TIMESTAMP_STEP_S = 0.08
MAX_TOKENS_PER_STEP = 10
CONTEXT_LOGIT_BIAS = 2.0
REQUIRED_ARTIFACTS = (
    "encoder-model.int8.onnx",
    "decoder_joint-model.int8.onnx",
    "nemo128.onnx",
    "vocab.txt",
)
_DECODE_SPACE_RE = re.compile(r"\A\s|\s\B|(\s)\b")
_CONTRACT = BackendContract(
    audio=WaveformSpec(sample_rate=SAMPLE_RATE, peak_normalize=True),
    decoding=BackendCapabilities(
        streaming=False,
        word_timestamps=True,
        language_detection=False,
        translation=False,
        context_biasing=True,
    ),
    output=OutputCapabilities(
        language="unavailable",
        word_timestamps="derived",
        word_confidence="unavailable",
        segment_confidence="unavailable",
        no_speech_probability="unavailable",
    ),
)


class _OrtNode(Protocol):
    name: str
    shape: list[int | str | None]


class _OrtSession(Protocol):
    def get_inputs(self) -> list[_OrtNode]: ...

    def get_providers(self) -> list[str]: ...

    def run(
        self,
        output_names: list[str],
        input_feed: dict[str, NDArray[Any]],
    ) -> list[NDArray[Any]]: ...


@dataclass(frozen=True, slots=True)
class _TimedToken:
    text: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class _TimedWord:
    text: str
    start: float
    end: float


class ParakeetOnnxDriver:
    """Run Handy-compatible Parakeet TDT int8 artifacts with ONNX Runtime."""

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
            return importlib.util.find_spec("onnxruntime") is not None
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
        if options.precision not in {"auto", "int8"}:
            raise TranscriptionError(
                f"Parakeet artifact '{model.spec.id}' is int8; precision '{options.precision}' is unsupported."
            )

        try:
            validate_model_artifact(model.artifact_path, model.spec)
        except ModelCompatibilityError as exc:
            raise TranscriptionError(f"Could not open Parakeet model '{model.spec.id}': {exc}") from exc
        try:
            ort = _load_onnxruntime()
            device, fallback_to_cpu, providers = _resolve_providers(ort, options)
            if "CUDAExecutionProvider" in providers:
                _preload_cuda_libraries(ort)
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # ORT writes graph-partition warnings directly to stderr, which corrupts
            # Textual's display during normal CUDA session initialization.
            session_options.log_severity_level = 3
            sessions = (
                _create_ort_session(ort, model.artifact_path / REQUIRED_ARTIFACTS[0], session_options, providers),
                _create_ort_session(ort, model.artifact_path / REQUIRED_ARTIFACTS[1], session_options, providers),
                _create_ort_session(ort, model.artifact_path / REQUIRED_ARTIFACTS[2], session_options, providers),
            )
            device, fallback_to_cpu = _verify_active_providers(
                sessions,
                requested_device=device,
                fallback_to_cpu=fallback_to_cpu,
                allow_cpu_fallback=options.allow_cpu_fallback,
            )
            vocab, blank_id = _load_vocab(model.artifact_path / "vocab.txt")
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"Could not open Parakeet model '{model.spec.id}': {exc}") from exc

        info = RuntimeInfo(
            backend_id=self.id,
            model_id=model.spec.id,
            device=device,
            precision="int8",
            fallback_to_cpu=fallback_to_cpu,
        )
        return ParakeetOnnxSession(model, sessions, vocab, blank_id, info)


class ParakeetOnnxSession:
    """Open Parakeet TDT session using the three-model Handy ONNX layout."""

    def __init__(
        self,
        prepared: PreparedModel,
        sessions: tuple[_OrtSession, _OrtSession, _OrtSession],
        vocab: list[str],
        blank_id: int,
        info: RuntimeInfo,
    ) -> None:
        self._prepared = prepared
        self._sessions: tuple[_OrtSession, _OrtSession, _OrtSession] | None = sessions
        self._vocab = vocab
        self._blank_id = blank_id
        self._info = info

    @property
    def info(self) -> RuntimeInfo:
        return self._info

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        sessions = self._sessions
        if sessions is None:
            raise TranscriptionError("Cannot transcribe with a closed Parakeet session.")
        if request.sample_rate != SAMPLE_RATE:
            raise TranscriptionError(f"Parakeet requires {SAMPLE_RATE} Hz audio, got {request.sample_rate} Hz.")
        if request.intent != "transcribe":
            raise TranscriptionError("Parakeet TDT does not support speech translation.")

        started_at = time.perf_counter()
        try:
            padded = np.concatenate((
                np.zeros(round(LEADING_SILENCE_S * SAMPLE_RATE), dtype=np.float32),
                np.ascontiguousarray(request.audio, dtype=np.float32),
            ))
            bias_sequences = _tokenize_proper_nouns(
                request.context.proper_nouns,
                self._vocab,
                self._blank_id,
            )
            token_ids, frame_indices = self._infer(sessions, padded, bias_sequences)
            token_text = [self._vocab[token_id] for token_id in token_ids]
            timestamps = [max(0.0, frame_index * TIMESTAMP_STEP_S - LEADING_SILENCE_S) for frame_index in frame_indices]
            duration_s = request.audio.size / request.sample_rate
            token_text, timestamps = _clip_tokens_to_audio(token_text, timestamps, duration_s)
            text = _decode_text(token_text)
            segments = _build_segments(
                token_text,
                timestamps,
                duration_s=duration_s,
                include_words=request.word_timestamps,
            )
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"Parakeet ONNX transcription failed: {exc}") from exc

        duration_s = request.audio.size / request.sample_rate
        return TranscriptionResult(
            text=text,
            segments=segments,
            language=None,
            language_probability=None,
            duration_s=duration_s,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            device=self._info.device,
            compute_type=self._info.precision,
            fallback_to_cpu=self._info.fallback_to_cpu,
            backend_id=self._info.backend_id,
            model_id=self._info.model_id,
            artifact_format=self._prepared.spec.artifact_format,
        )

    def close(self) -> None:
        self._sessions = None

    def _infer(
        self,
        sessions: tuple[_OrtSession, _OrtSession, _OrtSession],
        audio: NDArray[np.float32],
        bias_sequences: tuple[tuple[int, ...], ...],
    ) -> tuple[list[int], list[int]]:
        encoder, decoder, preprocessor = sessions
        features, feature_lengths = preprocessor.run(
            ["features", "features_lens"],
            {
                "waveforms": audio[np.newaxis, :],
                "waveforms_lens": np.asarray([audio.size], dtype=np.int64),
            },
        )
        encoded, encoded_lengths = encoder.run(
            ["outputs", "encoded_lengths"],
            {
                "audio_signal": features,
                "length": feature_lengths,
            },
        )
        encoded = np.transpose(encoded, (0, 2, 1))
        encoded_length = int(np.asarray(encoded_lengths).reshape(-1)[0])
        return self._decode(decoder, encoded[0], encoded_length, bias_sequences)

    def _decode(
        self,
        decoder: _OrtSession,
        encoded: NDArray[np.float32],
        encoded_length: int,
        bias_sequences: tuple[tuple[int, ...], ...],
    ) -> tuple[list[int], list[int]]:
        state_1, state_2 = _initial_decoder_state(decoder)
        tokens: list[int] = []
        timestamps: list[int] = []
        frame = 0
        emitted_at_frame = 0

        while frame < encoded_length:
            target = tokens[-1] if tokens else self._blank_id
            logits, next_state_1, next_state_2 = decoder.run(
                ["outputs", "output_states_1", "output_states_2"],
                {
                    "encoder_outputs": encoded[frame][np.newaxis, :, np.newaxis],
                    "targets": np.asarray([[target]], dtype=np.int32),
                    "target_length": np.asarray([1], dtype=np.int32),
                    "input_states_1": state_1,
                    "input_states_2": state_2,
                },
            )
            token_logits = np.asarray(logits).reshape(-1)[: len(self._vocab)].copy()
            for candidate in _bias_candidates(tokens, bias_sequences):
                token_logits[candidate] += CONTEXT_LOGIT_BIAS
            token = int(np.argmax(token_logits))

            if token != self._blank_id:
                state_1 = np.asarray(next_state_1, dtype=np.float32)
                state_2 = np.asarray(next_state_2, dtype=np.float32)
                tokens.append(token)
                timestamps.append(frame)
                emitted_at_frame += 1

            if token == self._blank_id or emitted_at_frame == MAX_TOKENS_PER_STEP:
                frame += 1
                emitted_at_frame = 0

        return tokens, timestamps


def _load_onnxruntime() -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise TranscriptionError("Parakeet requires the optional 'onnxruntime' package.") from exc
    return ort


def _create_ort_session(
    ort: Any,
    artifact_path: Path,
    session_options: Any,
    providers: list[str],
) -> _OrtSession:
    return cast(
        _OrtSession,
        ort.InferenceSession(
            str(artifact_path),
            sess_options=session_options,
            providers=providers,
        ),
    )


def _preload_cuda_libraries(ort: Any) -> None:
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    try:
        preload(directory="")
    except Exception as exc:
        logger.warning("Could not preload ONNX Runtime CUDA libraries: %s", exc)


def _verify_active_providers(
    sessions: tuple[_OrtSession, _OrtSession, _OrtSession],
    *,
    requested_device: str,
    fallback_to_cpu: bool,
    allow_cpu_fallback: bool,
) -> tuple[str, bool]:
    if requested_device != "cuda":
        return requested_device, fallback_to_cpu

    active_cuda = all("CUDAExecutionProvider" in session.get_providers() for session in sessions)
    if active_cuda:
        return "cuda", fallback_to_cpu
    if not allow_cpu_fallback:
        raise TranscriptionError(
            "CUDAExecutionProvider was available but failed to activate for every Parakeet graph; "
            "CPU fallback is disabled."
        )

    logger.warning("CUDAExecutionProvider failed to activate; Parakeet is falling back to CPU.")
    return "cpu", True


def _resolve_providers(ort: Any, options: RuntimeOptions) -> tuple[str, bool, list[str]]:
    if options.device not in {"auto", "cuda", "cpu"}:
        raise TranscriptionError(f"Parakeet does not support device '{options.device}'.")

    available = set(ort.get_available_providers())
    cuda_available = "CUDAExecutionProvider" in available
    if options.device in {"auto", "cuda"} and cuda_available:
        return "cuda", False, ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if options.device == "cuda" and not options.allow_cpu_fallback:
        raise TranscriptionError("CUDAExecutionProvider is unavailable and CPU fallback is disabled.")

    fallback = options.device == "cuda"
    if fallback:
        logger.warning("CUDAExecutionProvider is unavailable; Parakeet is falling back to CPU.")
    return "cpu", fallback, ["CPUExecutionProvider"]


def _load_vocab(path: Path) -> tuple[list[str], int]:
    entries: list[tuple[int, str]] = []
    blank_id: int | None = None
    for line_index, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = raw_line.rstrip()
        if not line:
            continue

        token_and_id = line.rsplit(maxsplit=1)
        token = line
        token_id = line_index
        if len(token_and_id) == 2:
            try:
                token_id = int(token_and_id[1])
                token = token_and_id[0]
            except ValueError:
                pass

        normalized = token.replace("▁", " ")
        entries.append((token_id, normalized))
        if token == "<blk>":
            blank_id = token_id

    if not entries or blank_id is None:
        raise TranscriptionError(f"Vocabulary '{path}' does not define a valid <blk> token.")

    vocab = [""] * (max(token_id for token_id, _ in entries) + 1)
    for token_id, token in entries:
        vocab[token_id] = token
    return vocab, blank_id


def _decode_text(pieces: list[str]) -> str:
    joined = "".join(pieces)
    return _DECODE_SPACE_RE.sub(lambda match: " " if match.group(1) is not None else "", joined)


def _clip_tokens_to_audio(
    pieces: list[str],
    timestamps: list[float],
    duration_s: float,
) -> tuple[list[str], list[float]]:
    """Discard decoder completions timestamped beyond the supplied waveform."""
    keep = bisect_right(timestamps, duration_s + TIMESTAMP_STEP_S)
    if keep < len(pieces):
        logger.warning("Discarded %s Parakeet token(s) beyond the audio boundary.", len(pieces) - keep)
    return pieces[:keep], timestamps[:keep]


def _tokenize_proper_nouns(
    proper_nouns: tuple[str, ...],
    vocab: list[str],
    blank_id: int,
) -> tuple[tuple[int, ...], ...]:
    sequences: list[tuple[int, ...]] = []
    for proper_noun in proper_nouns:
        for text in (f" {proper_noun}", proper_noun):
            sequence = _tokenize_exact(text, vocab, blank_id)
            if sequence is not None:
                sequences.append(sequence)
                break
    return tuple(dict.fromkeys(sequences))


def _tokenize_exact(text: str, vocab: list[str], blank_id: int) -> tuple[int, ...] | None:
    best: list[tuple[int, ...] | None] = [None] * (len(text) + 1)
    best[0] = ()
    for offset in range(len(text)):
        prefix = best[offset]
        if prefix is None:
            continue
        for token_id, piece in enumerate(vocab):
            if token_id == blank_id or not piece or not text.startswith(piece, offset):
                continue
            end = offset + len(piece)
            candidate = (*prefix, token_id)
            existing = best[end]
            if existing is None or len(candidate) < len(existing):
                best[end] = candidate
    return best[-1]


def _bias_candidates(
    emitted: list[int],
    sequences: tuple[tuple[int, ...], ...],
) -> set[int]:
    candidates = {sequence[0] for sequence in sequences if sequence}
    for sequence in sequences:
        max_prefix = min(len(emitted), len(sequence) - 1)
        for prefix_length in range(max_prefix, 0, -1):
            if tuple(emitted[-prefix_length:]) == sequence[:prefix_length]:
                candidates.add(sequence[prefix_length])
                break
    return candidates


def _initial_decoder_state(decoder: _OrtSession) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    inputs = {node.name: node for node in decoder.get_inputs()}
    try:
        shape_1 = _concrete_state_shape(inputs["input_states_1"].shape)
        shape_2 = _concrete_state_shape(inputs["input_states_2"].shape)
    except KeyError as exc:
        raise TranscriptionError(f"Parakeet decoder is missing state input '{exc.args[0]}'.") from exc
    return np.zeros(shape_1, dtype=np.float32), np.zeros(shape_2, dtype=np.float32)


def _concrete_state_shape(shape: list[int | str | None]) -> tuple[int, int, int]:
    if len(shape) != 3 or not isinstance(shape[0], int) or not isinstance(shape[2], int):
        raise TranscriptionError(f"Unsupported Parakeet decoder state shape: {shape}.")
    return shape[0], 1, shape[2]


def _build_segments(
    pieces: list[str],
    timestamps: list[float],
    *,
    duration_s: float,
    include_words: bool,
) -> list[Segment]:
    tokens = [
        _TimedToken(
            text=piece,
            start=min(timestamp, duration_s),
            end=min(timestamps[index + 1] if index + 1 < len(timestamps) else timestamp + 0.05, duration_s),
        )
        for index, (piece, timestamp) in enumerate(zip(pieces, timestamps, strict=True))
        if piece.strip()
    ]
    words = _group_words(tokens)
    return _group_segments(words, include_words=include_words)


def _group_words(tokens: list[_TimedToken]) -> list[_TimedWord]:
    words: list[_TimedWord] = []
    current: list[_TimedToken] = []
    for token in tokens:
        if token.text.startswith((" ", "▁")) and current:
            words.append(_make_word(current))
            current = []
        current.append(token)
    if current:
        words.append(_make_word(current))
    return [word for word in words if word.text]


def _make_word(tokens: list[_TimedToken]) -> _TimedWord:
    text = "".join(token.text.removeprefix("▁").removeprefix(" ") for token in tokens).strip()
    return _TimedWord(text=text, start=tokens[0].start, end=tokens[-1].end)


def _group_segments(words: list[_TimedWord], *, include_words: bool) -> list[Segment]:
    segments: list[Segment] = []
    current: list[_TimedWord] = []
    for index, word in enumerate(words):
        current.append(word)
        if not any(separator in word.text for separator in ".?!") and index != len(words) - 1:
            continue
        segment_words = (
            [WordTimestamp(word=item.text, start=item.start, end=item.end) for item in current] if include_words else []
        )
        segments.append(
            Segment(
                start=current[0].start,
                end=current[-1].end,
                text=" ".join(item.text for item in current),
                words=segment_words,
            )
        )
        current = []
    return segments


__all__ = ["ParakeetOnnxDriver", "ParakeetOnnxSession"]
