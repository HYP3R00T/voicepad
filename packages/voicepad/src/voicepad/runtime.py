from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from voicepad_core.artifacts import ArtifactStore, WheelExtractor
from voicepad_core.audio import MicrophoneStream, WavArtifact
from voicepad_core.deployments import PARAKEET_V3_CUDA, SILERO_VAD_ONNX_EXTRACTION
from voicepad_core.inference import ActiveDeployment, ResidentTranscriptionEngine
from voicepad_core.pipeline import (
    FileTranscriptionResult,
    GrowingTranscriptionJob,
    build_finite_file_transcriber,
    build_growing_job,
)

from .config import AppConfig


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

    def transcribe_file(self, path: Path) -> FileTranscriptionResult:
        model = self._require_silero()
        return build_finite_file_transcriber(
            self.engine,
            model,
            aliases=self.config.alias_rules,
            terminal_punctuation=self.config.terminal_punctuation,
        ).transcribe_file(path)

    def start_recording(self) -> tuple[MicrophoneStream, GrowingTranscriptionJob]:
        model = self._require_silero()
        self.config.recordings_path.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.config.recordings_path / f"{self.config.recording_prefix}_{stamp}_{uuid4().hex[:8]}.wav"
        microphone = MicrophoneStream(path, device_index=self.config.input_device_index)
        microphone.start()
        try:
            job = build_growing_job(
                self.engine,
                model,
                microphone.growing_source,
                aliases=self.config.alias_rules,
                terminal_punctuation=self.config.terminal_punctuation,
            )
            job.start()
        except Exception:
            microphone.stop()
            raise
        return microphone, job

    @staticmethod
    def stop_recording(
        microphone: MicrophoneStream,
        job: GrowingTranscriptionJob,
    ) -> tuple[WavArtifact, FileTranscriptionResult]:
        artifact = microphone.stop()
        return artifact, job.finish()

    def close(self) -> None:
        self.engine.unload()

    def _require_silero(self) -> Path:
        if self.engine.active_deployment is None or self._silero_model is None:
            raise RuntimeError(f"Deployment '{PARAKEET_V3_CUDA.id}' is not activated.")
        return self._silero_model
