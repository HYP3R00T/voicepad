from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from voicepad_core.artifacts import ArtifactError, ArtifactStore, ProgressCallback, WheelExtractor
from voicepad_core.audio import MicrophoneStream, WavArtifact
from voicepad_core.deployments import (
    PARAKEET_V3_CUDA,
    PARAKEET_V3_MANIFEST,
    SILERO_VAD_ONNX_EXTRACTION,
    get_manifest,
)
from voicepad_core.inference import ActiveDeployment, ResidentTranscriptionEngine
from voicepad_core.pipeline import (
    IncrementalTranscriptionJob,
    TranscriptionProgress,
    TranscriptionResult,
    build_batch_transcriber,
    build_incremental_job,
)

from .config import AppConfig
from .observability import RecordingLogScope

logger = logging.getLogger(__name__)


class ApplicationRuntime:
    """Own the selected resident deployment and every application transcription path."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.artifacts = ArtifactStore(config.artifact_cache_path)
        self.engine = ResidentTranscriptionEngine(config.artifact_cache_path, artifact_store=self.artifacts)
        self._silero_model: Path | None = None
        self._recording_scope: RecordingLogScope | None = None

    @property
    def active_deployment(self) -> ActiveDeployment | None:
        return self.engine.active_deployment

    def artifacts_ready(self) -> bool:
        """Return whether every required model artifact is present and valid."""
        try:
            if self.artifacts.locate(PARAKEET_V3_MANIFEST) is None:
                return False
            WheelExtractor(self.artifacts).verify(SILERO_VAD_ONNX_EXTRACTION)
        except (ArtifactError, OSError):
            return False
        return True

    def activate(self, on_progress: ProgressCallback | None = None) -> ActiveDeployment:
        parakeet_size = PARAKEET_V3_MANIFEST.total_size
        silero_manifest = get_manifest(SILERO_VAD_ONNX_EXTRACTION.wheel_manifest_id)
        total_size = parakeet_size + silero_manifest.total_size

        def parakeet_progress(completed: int, _total: int) -> None:
            if on_progress is not None:
                on_progress(completed, total_size)

        def silero_progress(completed: int, _total: int) -> None:
            if on_progress is not None:
                on_progress(parakeet_size + completed, total_size)

        active = self.engine.activate(self.config.deployment_id, on_progress=parakeet_progress)
        self._silero_model = WheelExtractor(self.artifacts).prepare(
            SILERO_VAD_ONNX_EXTRACTION,
            silero_progress,
        )
        return active

    def transcribe_file(self, path: Path) -> TranscriptionResult:
        model = self._require_silero()
        return build_batch_transcriber(self.engine, model).transcribe_file(path)

    def start_recording(
        self,
        *,
        on_update: Callable[[TranscriptionProgress], None] | None = None,
    ) -> tuple[MicrophoneStream, IncrementalTranscriptionJob]:
        model = self._require_silero()
        microphone = self._start_microphone()
        assert self._recording_scope is not None
        try:
            job = build_incremental_job(
                self.engine,
                model,
                microphone.incremental_source,
                on_update=on_update,
                logger=self._recording_scope.logger_for("voicepad_core.pipeline.incremental"),
                log_context=self._recording_scope.context,
            )
            job.start()
        except Exception as error:
            try:
                microphone.stop()
            except Exception as cleanup_error:
                logger.exception("Microphone cleanup failed after recording startup failure")
                error.add_note(f"Microphone cleanup also failed: {cleanup_error}")
            self._end_recording_scope(outcome="failed", error=error)
            raise
        return microphone, job

    def start_capture(self) -> MicrophoneStream:
        """Start a recording scope and microphone without transcription."""
        return self._start_microphone()

    def stop_recording(
        self,
        microphone: MicrophoneStream,
        job: IncrementalTranscriptionJob,
    ) -> tuple[WavArtifact, TranscriptionResult]:
        operation_logger = self._recording_logger()
        operation_logger.info("Recording stop requested")
        try:
            artifact = microphone.stop()
            result = job.finish()
            if microphone.capture_error is not None:
                result = replace(
                    result,
                    complete=False,
                    warnings=(*result.warnings, f"audio capture failed: {microphone.capture_error}"),
                )
            operation_logger.info(
                "Recording finalized: path=%s duration_s=%.3f complete=%s",
                artifact.path,
                artifact.duration_s,
                result.complete,
            )
        except Exception as error:
            self._end_recording_scope(outcome="failed", error=error)
            raise
        return artifact, result

    def stop_capture(self, microphone: MicrophoneStream) -> WavArtifact:
        """Finalize a capture-only recording and close its log scope."""
        try:
            artifact = microphone.stop()
        except Exception as error:
            self._end_recording_scope(outcome="failed", error=error)
            raise
        return artifact

    def end_recording(self, *, outcome: str, error: BaseException | None = None) -> None:
        """Close the active recording log after all host-side persistence finishes."""
        self._end_recording_scope(outcome=outcome, error=error)

    def close(self) -> None:
        try:
            self.engine.unload()
        finally:
            self._end_recording_scope(outcome="abandoned")

    def _require_silero(self) -> Path:
        if self.engine.active_deployment is None or self._silero_model is None:
            raise RuntimeError(f"Deployment '{PARAKEET_V3_CUDA.id}' is not activated.")
        return self._silero_model

    def _start_microphone(self) -> MicrophoneStream:
        if self._recording_scope is not None:
            raise RuntimeError("A recording scope is already active.")
        self.config.recordings_path.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.config.recordings_path / f"{self.config.recording_prefix}_{stamp}_{uuid4().hex[:8]}.wav"
        scope = RecordingLogScope.start(path.stem)
        self._recording_scope = scope
        scoped_logger = scope.logger_for("voicepad.runtime")
        scoped_logger.info("Recording start requested: path=%s", path)
        microphone = MicrophoneStream(
            path,
            device_index=self.config.input_device_index,
            logger=scope.logger_for("voicepad_core.audio.microphone"),
            log_context=scope.context,
        )
        try:
            microphone.start()
        except Exception as error:
            self._end_recording_scope(outcome="failed", error=error)
            raise
        return microphone

    def _end_recording_scope(self, *, outcome: str, error: BaseException | None = None) -> None:
        scope = self._recording_scope
        self._recording_scope = None
        if scope is not None:
            scope.close(outcome=outcome, error=error)

    def _recording_logger(self) -> logging.Logger | logging.LoggerAdapter[logging.Logger]:
        if self._recording_scope is None:
            return logger
        return self._recording_scope.logger_for("voicepad.runtime")
