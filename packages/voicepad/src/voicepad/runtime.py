from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from voicepad_core.artifacts import ArtifactStore, WheelExtractor
from voicepad_core.audio import MicrophoneStream, WavArtifact
from voicepad_core.deployments import PARAKEET_V3_CUDA, SILERO_VAD_ONNX_EXTRACTION
from voicepad_core.inference import ActiveDeployment, ResidentTranscriptionEngine
from voicepad_core.pipeline import (
    IncrementalTranscriptionJob,
    TranscriptionProgress,
    TranscriptionResult,
    build_batch_transcriber,
    build_incremental_job,
)

from .config import AppConfig

logger = logging.getLogger(__name__)


class ApplicationRuntime:
    """Own the selected resident deployment and every application transcription path."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.artifacts = ArtifactStore(config.artifact_cache_path)
        self.engine = ResidentTranscriptionEngine(config.artifact_cache_path, artifact_store=self.artifacts)
        self._silero_model: Path | None = None

    @property
    def active_deployment(self) -> ActiveDeployment | None:
        return self.engine.active_deployment

    def activate(self) -> ActiveDeployment:
        active = self.engine.activate(self.config.deployment_id)
        self._silero_model = WheelExtractor(self.artifacts).prepare(SILERO_VAD_ONNX_EXTRACTION)
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
        self.config.recordings_path.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.config.recordings_path / f"{self.config.recording_prefix}_{stamp}_{uuid4().hex[:8]}.wav"
        logger.info("Recording start requested: path=%s", path)
        microphone = MicrophoneStream(path, device_index=self.config.input_device_index)
        microphone.start()
        try:
            job = build_incremental_job(
                self.engine,
                model,
                microphone.incremental_source,
                on_update=on_update,
            )
            job.start()
        except Exception as error:
            try:
                microphone.stop()
            except Exception as cleanup_error:
                logger.exception("Microphone cleanup failed after recording startup failure")
                error.add_note(f"Microphone cleanup also failed: {cleanup_error}")
            raise
        return microphone, job

    @staticmethod
    def stop_recording(
        microphone: MicrophoneStream,
        job: IncrementalTranscriptionJob,
    ) -> tuple[WavArtifact, TranscriptionResult]:
        logger.info("Recording stop requested")
        artifact = microphone.stop()
        result = job.finish()
        if microphone.capture_error is not None:
            result = replace(
                result,
                complete=False,
                warnings=(*result.warnings, f"audio capture failed: {microphone.capture_error}"),
            )
        logger.info(
            "Recording finalized: path=%s duration_s=%.3f complete=%s",
            artifact.path,
            artifact.duration_s,
            result.complete,
        )
        return artifact, result

    def close(self) -> None:
        self.engine.unload()

    def _require_silero(self) -> Path:
        if self.engine.active_deployment is None or self._silero_model is None:
            raise RuntimeError(f"Deployment '{PARAKEET_V3_CUDA.id}' is not activated.")
        return self._silero_model
